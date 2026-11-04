"""
residual.py - Hypothesis 3: policy-hedged residual RV.

Step 4 showed that fading raw 2s10s and AU-NZ 10y z-scores loses money, because
those spreads trend with the policy cycle. A desk-standard next try is to hedge
the policy (short-rate) leg out of the long-rate spread and fade what is left:

  residual_t = y_long,t − beta_t × y_short,t

beta_t is the slope from a 250-trading-day regression of daily CHANGES of the long
leg on the short leg, using only data dated t or earlier. Filled days and NZ-bond
roll days are left out of the fit. At date t the residual's z-score window is
REBUILT with beta_t (not stitched from yesterday's beta), same idea as the PCA fly.

Three residuals (all in bp):
  au_nz_box            (AU10 − NZ10) − beta × (AU2 − NZ2)
  au_gov_10s2s_resid   AU10 − beta × AU2
  nz_swap_10s2s_resid  NZ10 swap − beta × NZ2 swap

Entry adds a carry filter that was also fixed in advance: fade only when
|z_250| > 2 AND the expected 3-month carry + rolldown of that trade direction
is ≥ 0. Other rules (exit, stop, max hold, costs, next-day fill, roll blocks)
are the Step 4 rules. The same backtest is also run WITHOUT the carry filter so
the filter's effect is visible.

This hypothesis was chosen after seeing Step 4, so 2021- is not a pristine hold-out.

Run from the project root:   python -m src.residual
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.backtest import par_mod_duration, simulate, summarise_trades, tenor_years
from src.fetch import forward_fill_interior
from src.pca import endpoint_z
from src.rolls import load_roll_dates, pending_series
from src.scanner import gap_mask
from src.spreads import roll_flags, to_bp
from src.utils import get_logger, load_config, project_path, read_rates, with_gap_breaks

log = get_logger("residual")


# =============================================================================
# Beta
# =============================================================================

def combo_level(clean: pd.DataFrame, legs: dict[str, float]) -> pd.Series:
    """Yield combination in bp. NaN if any leg is missing."""
    return to_bp(sum(w * clean[s] for s, w in legs.items()))


def combo_changes(clean: pd.DataFrame, raw: pd.DataFrame, legs: dict[str, float],
                  roll_days: pd.Index | None) -> pd.Series:
    """Daily changes of a yield combination, in bp. NaN on filled or roll days."""
    series = list(legs)
    available = clean[series].dropna(how="all")
    cal = pd.bdate_range(available.index.min(), available.index.max())
    y = clean[series].reindex(cal)
    filled = y.notna() & raw.reindex(cal)[series].isna()
    lvl = combo_level(y, legs)
    chg = lvl.diff()
    chg = chg.mask(y.isna().any(axis=1) | filled.any(axis=1))
    if roll_days is not None and len(roll_days):
        chg.loc[chg.index.isin(pd.DatetimeIndex(roll_days))] = np.nan
    return chg


def ols_beta(x: np.ndarray, y: np.ndarray) -> float:
    """Slope of y on x with intercept (cov/var). NaN if x has no variation."""
    if len(x) < 3:
        return float("nan")
    vx = float(np.var(x, ddof=1))
    if vx < 1e-12:
        return float("nan")
    return float(np.cov(x, y, ddof=1)[0, 1] / vx)


def rolling_beta(d_long: pd.Series, d_short: pd.Series, window: int, min_real_fraction: float,
                 max_gap: int) -> pd.Series:
    """Trailing-window OLS beta of d_long on d_short, no look-ahead."""
    idx = d_long.index.union(d_short.index)
    d_long, d_short = d_long.reindex(idx), d_short.reindex(idx)
    need = math.ceil(min_real_fraction * window)
    out = pd.Series(np.nan, index=idx)
    dates = d_long.dropna().index.union(d_short.dropna().index)
    for end in dates:
        cal = pd.bdate_range(end - pd.offsets.BDay(window - 1), end)
        x = d_short.reindex(cal)
        y = d_long.reindex(cal)
        real = x.notna() & y.notna()
        if int(real.sum()) < need or bool(gap_mask(real, window, max_gap).iloc[-1]):
            continue
        out.at[end] = ols_beta(x[real].to_numpy(), y[real].to_numpy())
    return out.dropna()


def frozen_residual_z(long_bp: pd.Series, short_bp: pd.Series, beta: pd.Series, filled: pd.Series,
                      scanner_cfg: dict[str, Any]) -> pd.DataFrame:
    """At each date t, residual history = long − beta_t × short, then z-score the last point."""
    idx = long_bp.index.intersection(short_bp.index)
    long_bp, short_bp = long_bp.reindex(idx), short_bp.reindex(idx)
    fill = filled.reindex(idx).fillna(False).astype(bool)
    lv, sv = long_bp.to_numpy(), short_bp.to_numpy()
    loc = {d: i for i, d in enumerate(idx)}
    args = (scanner_cfg["windows"], scanner_cfg["min_real_fraction"],
            scanner_cfg["max_gap_days"], scanner_cfg["percentile_min_obs"])
    rows = []
    for t, b in beta.items():
        i = loc.get(t)
        if i is None or not np.isfinite(b):
            continue
        resid = pd.Series(lv[: i + 1] - float(b) * sv[: i + 1], index=idx[: i + 1])
        rows.append(endpoint_z(resid, fill.iloc[: i + 1], *args))
    z = pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()
    if not z.empty:
        z["beta"] = beta.reindex(z.index)
    return z


# =============================================================================
# Carry and rolldown
# =============================================================================

def interpolate_yield(tenors: np.ndarray, yields: np.ndarray, target: float) -> float:
    """Linear interpolation of yield vs tenor. Clamp to the end-points (no extrapolation)."""
    order = np.argsort(tenors)
    t, y = tenors[order], yields[order]
    ok = np.isfinite(t) & np.isfinite(y)
    t, y = t[ok], y[ok]
    if len(t) == 0:
        return float("nan")
    if target <= t[0]:
        return float(y[0])
    if target >= t[-1]:
        return float(y[-1])
    return float(np.interp(target, t, y))


def curve_points(clean_row: pd.Series, curve_series: list[str]) -> tuple[np.ndarray, np.ndarray]:
    tenors, ylds = [], []
    for s in curve_series:
        if s in clean_row.index and np.isfinite(clean_row[s]):
            tenors.append(tenor_years(s))
            ylds.append(float(clean_row[s]))
    return np.array(tenors, dtype=float), np.array(ylds, dtype=float)


def series_market(series: str) -> str:
    if series.startswith("au_"):
        return "au"
    if series.startswith("nz_swap"):
        return "nz_swap"
    return "nz_gov"


def rolldown_bp(series: str, clean_row: pd.Series, curve_series: list[str], horizon: float) -> float:
    """bp of yield a LONG position gains over `horizon` years if the curve is unchanged."""
    y = clean_row.get(series, np.nan)
    if not np.isfinite(y):
        return float("nan")
    tenors, ylds = curve_points(clean_row, curve_series)
    y_later = interpolate_yield(tenors, ylds, tenor_years(series) - horizon)
    if not np.isfinite(y_later):
        return float("nan")
    return float((y - y_later) * 100.0)  # percent -> bp


def carry_bp(series: str, clean_row: pd.Series, funding_y: float, horizon: float) -> float:
    """3-month (or `horizon`) carry of a LONG yield position, in bp of that yield.

    (y - funding) * h * 100 / modified_duration, y and funding in percent,
    duration from the par formula at the current yield.
    """
    y = clean_row.get(series, np.nan)
    if not (np.isfinite(y) and np.isfinite(funding_y)):
        return float("nan")
    dur = par_mod_duration(tenor_years(series), y / 100.0)
    if dur <= 0:
        return float("nan")
    return float((y - funding_y) * horizon * 100.0 / dur)


def combo_carry_rolldown(clean_row: pd.Series, legs: dict[str, float], cfg_res: dict[str, Any]) -> float:
    """3-month carry+rolldown of a LONG combination, in bp of the combination."""
    h = cfg_res["carry_horizon_years"]
    total = 0.0
    for series, w in legs.items():
        mkt = series_market(series)
        fund_name = cfg_res["funding"][mkt]
        fund_y = clean_row.get(fund_name, np.nan)
        if mkt == "nz_swap" and not np.isfinite(fund_y):
            fund_y = clean_row.get("nz_swap_2y", np.nan)
        rd = rolldown_bp(series, clean_row, cfg_res["curves"][mkt], h)
        cy = carry_bp(series, clean_row, fund_y, h)
        if not (np.isfinite(rd) and np.isfinite(cy)):
            return float("nan")
        total += w * (cy + rd)
    return total


def carry_of_long_residual(clean: pd.DataFrame, long_legs: dict[str, float], short_legs: dict[str, float],
                           beta: pd.Series, cfg_res: dict[str, Any]) -> pd.Series:
    """3-month carry+rd of +1 residual (long the long-leg combo, short beta of the short combo)."""
    out = pd.Series(np.nan, index=beta.index)
    for dt, b in beta.items():
        if dt not in clean.index or not np.isfinite(b):
            continue
        row = clean.loc[dt]
        c_long = combo_carry_rolldown(row, long_legs, cfg_res)
        c_short = combo_carry_rolldown(row, short_legs, cfg_res)
        if np.isfinite(c_long) and np.isfinite(c_short):
            out.at[dt] = c_long - float(b) * c_short
    return out


# =============================================================================
# Half-life
# =============================================================================

def half_life(x: pd.Series) -> float:
    """AR(1) half-life in observations: x_t = a + phi x_{t-1}. NaN if phi not in (0, 1)."""
    v = x.dropna()
    if len(v) < 50:
        return float("nan")
    lag, y = v.iloc[:-1].to_numpy(), v.iloc[1:].to_numpy()
    a = np.column_stack([np.ones(len(lag)), lag])
    coef, *_ = np.linalg.lstsq(a, y, rcond=None)
    phi = float(coef[1])
    if not (0.0 < phi < 1.0):
        return float("nan")
    return float(math.log(2.0) / -math.log(phi))


# =============================================================================
# Charts
# =============================================================================

def raw_vs_residual_chart(raw: dict[str, pd.Series], resid: dict[str, pd.Series],
                          titles: list[str]) -> go.Figure:
    fig = make_subplots(rows=len(titles), cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=titles)
    for i, name in enumerate(resid, start=1):
        r = with_gap_breaks(resid[name])
        w = with_gap_breaks(raw[name].reindex(resid[name].index))
        fig.add_trace(go.Scatter(x=w.index, y=w, mode="lines", line=dict(width=1, color="grey"),
                                 name="raw", showlegend=i == 1, legendgroup="raw"), row=i, col=1)
        fig.add_trace(go.Scatter(x=r.index, y=r, mode="lines", line=dict(width=1.2, color="#1f77b4"),
                                 name="residual (today's beta)", showlegend=i == 1, legendgroup="resid"),
                      row=i, col=1)
        fig.update_yaxes(title_text="bp", row=i, col=1)
    fig.update_layout(height=280 * len(titles), width=1100, template="plotly_white",
                      legend=dict(orientation="h", y=1.04, x=0),
                      title="Raw 10y spread vs policy-hedged residual (less trendy if the hedge is working)",
                      margin=dict(t=80, l=50, r=20, b=30))
    return fig


def beta_chart(betas: dict[str, pd.Series]) -> go.Figure:
    fig = go.Figure()
    colours = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for (name, s), c in zip(betas.items(), colours):
        d = with_gap_breaks(s)
        fig.add_trace(go.Scatter(x=d.index, y=d, mode="lines", line=dict(width=1.2, color=c), name=name))
    fig.update_layout(title="Rolling 250-day beta (long-leg changes on short-leg changes)",
                      yaxis_title="beta", template="plotly_white", height=420, width=1100,
                      legend=dict(orientation="h", y=1.08, x=0), margin=dict(t=60, l=50, r=20, b=30))
    return fig


def equity_compare_chart(daily_c: pd.DataFrame, daily_n: pd.DataFrame, is_end: pd.Timestamp) -> go.Figure:
    fig = go.Figure()
    end_x = None
    for df, name, colour in ((daily_n, "residual, no carry filter", "grey"),
                             (daily_c, "residual, carry filter", "#1f77b4")):
        if df.empty:
            continue
        s = with_gap_breaks(df.groupby("date")["pnl_usd"].sum().sort_index().cumsum())
        fig.add_trace(go.Scatter(x=s.index, y=s, mode="lines", line=dict(width=1.3, color=colour), name=name))
        end_x = df["date"].max()
    if end_x is not None:
        fig.add_vrect(x0=is_end, x1=end_x, fillcolor="rgba(255,127,14,0.08)", line_width=0)
    fig.update_layout(title="Residual strategy equity ($ per $10k on the 10y leg; shaded = OOS)",
                      yaxis_title="USD", template="plotly_white", height=420, width=1100,
                      legend=dict(orientation="h", y=1.08, x=0), margin=dict(t=60, l=50, r=20, b=30))
    return fig


def _write_fig(fig: go.Figure, path, stem: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    fig.write_html(path / f"{stem}.html", include_plotlyjs="cdn")
    try:
        fig.write_image(path / f"{stem}.png", scale=1)
    except Exception as e:
        log.warning("PNG export skipped for %s (%s)", stem, e)


# =============================================================================
# One residual
# =============================================================================

def build_one(name: str, spec: dict[str, Any], clean: pd.DataFrame, raw: pd.DataFrame,
              roll_days: pd.Index, roll_dates: pd.DataFrame, cfg: dict[str, Any]
              ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """History (scanner-like), beta series, carry of the long residual."""
    rcfg, scfg = cfg["residual"], cfg["scanner"]
    start = pd.Timestamp(spec["start"]) if spec.get("start") else clean.index.min()
    long_legs, short_legs = spec["long"], spec["short"]
    all_legs = list(dict.fromkeys([*long_legs, *short_legs]))
    nz_bonds = [s for s in all_legs if s.startswith("nz_gov_")]

    long_bp = combo_level(clean, long_legs)
    short_bp = combo_level(clean, short_legs)
    d_long = combo_changes(clean, raw, long_legs, roll_days)
    d_short = combo_changes(clean, raw, short_legs, roll_days)
    # restrict the fit start
    d_long = d_long.loc[d_long.index >= start]
    d_short = d_short.loc[d_short.index >= start]
    log.info("%s: rolling beta on %d change-dates", name, d_long.dropna().shape[0])
    beta = rolling_beta(d_long, d_short, rcfg["window"], rcfg["min_real_fraction"], rcfg["max_gap_days"])
    if beta.empty:
        raise RuntimeError(f"No beta for {name}")

    filled = (clean[all_legs].notna() & raw[all_legs].isna()).any(axis=1)
    common = long_bp.dropna().index.intersection(short_bp.dropna().index)
    common = common[common >= start]
    z = frozen_residual_z(long_bp.loc[common], short_bp.loc[common], beta, filled, scfg)
    flags = roll_flags(z.index, nz_bonds, roll_dates, cfg["instruments"]["post_roll_block_days"])
    hist = z.join(flags)
    hist["instrument"] = name
    hist["inputs_filled"] = filled.reindex(hist.index).fillna(False).astype(bool)
    pending = pending_series(roll_dates)
    pend_dates = [dt for s in all_legs for dt in pending.get(s, [])]
    hist["pending_roll"] = hist.index.isin(pend_dates)
    hist.index.name = "date"

    carry = carry_of_long_residual(clean, long_legs, short_legs, beta, rcfg)
    hist["carry_3m_bp"] = carry.reindex(hist.index)
    return hist.reset_index(), beta, carry


def residual_bundle(hist: pd.DataFrame, spec: dict[str, Any], beta: pd.Series, carry: pd.Series,
                    clean: pd.DataFrame) -> dict[str, Any]:
    long_bp = combo_level(clean, spec["long"])
    short_bp = combo_level(clean, spec["short"])
    return {
        "beta": beta, "long_bp": long_bp, "short_bp": short_bp,
        "long_legs": spec["long"], "short_legs": spec["short"], "carry_long": carry,
    }


def backtest_residuals(hist: pd.DataFrame, specs: dict[str, dict], bundles: dict[str, dict],
                       cfg: dict[str, Any], require_carry: bool, carry_only: bool = False
                       ) -> tuple[pd.DataFrame, pd.DataFrame]:
    bt = cfg["backtest"]
    rules = dict(entry_z=bt["entry_z"], exit_z=bt["exit_z"], stop_z=bt["stop_z"],
                 max_hold_days=bt["max_hold_days"], is_end=pd.Timestamp(bt["in_sample"][1]))
    dummy_spec = {"group": "", "pca": False, "legs": {}, "belly": None, "wings": None}
    trades, daily = [], []
    for name, spec in specs.items():
        d = hist[hist["instrument"] == name].set_index("date").sort_index()
        sp = {**dummy_spec, "group": spec["group"]}
        t, day = simulate(d, sp, name, rules, bt["costs"], bt["assume_yield"], bt["book_dv01_usd"],
                          {}, clean=None, residual=bundles[name], require_carry=require_carry,
                          carry_only=carry_only)
        trades.append(t)
        daily.append(day)
    return pd.concat(trades, ignore_index=True), pd.concat(daily, ignore_index=True)


def metrics_block(trades: pd.DataFrame, daily: pd.DataFrame, is_end: pd.Timestamp, label: str) -> pd.DataFrame:
    rows = []
    grouped = list(trades.groupby("instrument")) if len(trades) else []
    for sample in ("IS", "OOS", None):
        for name, g in grouped:
            m = summarise_trades(g, daily[daily["instrument"] == name], sample, is_end)
            m["instrument"] = name
            m["rule"] = label
            rows.append(m)
        m = summarise_trades(trades, daily, sample, is_end)
        m["instrument"] = "ALL"
        m["rule"] = label
        rows.append(m)
    return pd.DataFrame(rows)


# =============================================================================
# Main
# =============================================================================

def run(cfg: dict[str, Any] | None = None) -> dict[str, pd.DataFrame]:
    cfg = cfg or load_config()
    paths = {k: project_path(v) for k, v in cfg["paths"].items()}
    raw = read_rates(paths["rates_file"])
    if raw is None:
        raise FileNotFoundError("rates.csv not found - run `python -m src.fetch` first")
    clean, _ = forward_fill_interior(raw, cfg["update"]["ffill_max_days"])
    rolls = load_roll_dates(paths["roll_dates_file"])
    roll_days = pd.DatetimeIndex(rolls.loc[rolls["status"].isin(["confirmed", "unclear", "pending"]), "date"].unique())
    is_end = pd.Timestamp(cfg["backtest"]["in_sample"][1])
    is_start = pd.Timestamp(cfg["backtest"]["in_sample"][0])

    specs = cfg["residual"]["instruments"]
    frames, betas, carries, bundles = [], {}, {}, {}
    for name, spec in specs.items():
        hist, beta, carry = build_one(name, spec, clean, raw, roll_days, rolls, cfg)
        frames.append(hist)
        betas[name], carries[name] = beta, carry
        bundles[name] = residual_bundle(hist, spec, beta, carry, clean)
        last = beta.iloc[-1]
        print(f"\n{name} as of {beta.index[-1].date()}: beta = {last:.2f}  "
              f"(long 1 of the 10y combo vs {last:.2f} of the 2y combo)")
        c = carry.dropna()
        if len(c):
            print(f"  3-month carry+rolldown of a LONG residual, latest {c.iloc[-1]:+.1f}bp  "
                  f"(median {c.median():+.1f}bp)")

    history = pd.concat(frames, ignore_index=True)
    history = history[history["date"] >= is_start]
    hist_out = history.copy()
    hist_out["date"] = pd.to_datetime(hist_out["date"]).dt.strftime("%Y-%m-%d")
    paths["residual_history_file"].parent.mkdir(parents=True, exist_ok=True)
    hist_out.round(4).to_csv(paths["residual_history_file"], index=False)

    beta_out = pd.concat({k: v.rename("beta") for k, v in betas.items()}, names=["instrument", "date"]).reset_index()
    beta_out["date"] = pd.to_datetime(beta_out["date"]).dt.strftime("%Y-%m-%d")
    beta_out.to_csv(paths["residual_beta_file"], index=False, float_format="%.6f")

    log.info("Backtesting residuals with and without the carry filter")
    tr_c, dy_c = backtest_residuals(history, specs, bundles, cfg, require_carry=True)
    tr_n, dy_n = backtest_residuals(history, specs, bundles, cfg, require_carry=False)
    met = pd.concat([
        metrics_block(tr_c, dy_c, is_end, "residual_carry"),
        metrics_block(tr_n, dy_n, is_end, "residual_nocarry"),
    ], ignore_index=True)

    # Step 4 baseline (read, do not recompute)
    base_path = paths["backtest_metrics_file"]
    if base_path.exists():
        base = pd.read_csv(base_path)
        base["rule"] = "step4_baseline"
        counterparts = ["ALL"] + [s.get("raw_compare") for s in specs.values() if s.get("raw_compare")]
        met = pd.concat([base[base["instrument"].isin(counterparts)].assign(rule="step4_baseline"), met],
                        ignore_index=True)

    # Half-life, IS only
    hl_rows = []
    inst = pd.read_csv(paths["instruments_file"], parse_dates=["date"])
    for name, spec in specs.items():
        raw_name = spec.get("raw_compare")
        resid = history[(history["instrument"] == name) & (history["date"] <= is_end)].set_index("date")["value_bp"]
        hl_r = half_life(resid)
        hl_raw = float("nan")
        if raw_name:
            raw_s = inst[(inst["instrument"] == raw_name) & (inst["date"] <= is_end)].set_index("date")["value_bp"]
            hl_raw = half_life(raw_s)
        hl_rows.append({"instrument": name, "raw_compare": raw_name or "",
                        "halflife_raw_days": hl_raw, "halflife_resid_days": hl_r})
        def _hl(x: float) -> str:
            return f"{x:.0f}d" if np.isfinite(x) else "n/a (not mean-reverting)"
        print(f"  half-life (IS, AR(1) on levels): {name} residual {_hl(hl_r)}"
              + (f" vs raw {raw_name} {_hl(hl_raw)}" if raw_name else ""))
    hl = pd.DataFrame(hl_rows)

    for p in (paths["residual_metrics_file"], paths["residual_trades_file"],
              paths["residual_trades_nocarry_file"], paths["residual_halflife_file"]):
        p.parent.mkdir(parents=True, exist_ok=True)
    met.to_csv(paths["residual_metrics_file"], index=False, float_format="%.4f")
    hl.to_csv(paths["residual_halflife_file"], index=False, float_format="%.2f")
    for df, path in ((tr_c, paths["residual_trades_file"]), (tr_n, paths["residual_trades_nocarry_file"])):
        out = df.copy()
        for c in ("entry_date", "exit_date"):
            if c in out.columns:
                out[c] = pd.to_datetime(out[c]).dt.strftime("%Y-%m-%d")
        out.to_csv(path, index=False)

    # Charts
    charts = paths["charts_dir"] / "residual"
    raw_plot, resid_plot, titles = {}, {}, []
    for name, spec in specs.items():
        h = history[history["instrument"] == name].set_index("date").sort_index()
        resid_plot[name] = h["value_bp"]
        raw_name = spec.get("raw_compare")
        if raw_name:
            raw_plot[name] = inst[inst["instrument"] == raw_name].set_index("date")["value_bp"]
        else:
            raw_plot[name] = h["value_bp"]
        titles.append(f"{name}: raw {raw_name or ''} vs residual (bp)")
    _write_fig(raw_vs_residual_chart(raw_plot, resid_plot, titles), charts, "raw_vs_residual")
    _write_fig(beta_chart(betas), charts, "beta")
    _write_fig(equity_compare_chart(dy_c, dy_n, is_end), charts, "equity")

    cols = ["rule", "instrument", "sample", "trades", "hit_rate", "avg_bp", "total_usd",
            "sharpe", "max_dd_usd", "avg_hold_days", "pct_stopped"]
    show = met[met["sample"].isin(["IS", "OOS"]) & met["instrument"].eq("ALL")]
    extra = met[met["sample"].isin(["IS", "OOS"]) & met["rule"].str.startswith("residual")
                & ~met["instrument"].eq("ALL")]
    with pd.option_context("display.width", 220, "display.float_format", lambda x: f"{x: .2f}"):
        print("\nHypothesis 3 vs Step 4 baseline (ALL books). Residual is also shown per instrument.\n")
        print(pd.concat([show, extra])[cols].to_string(index=False))
        print("\nCarry filter: fade only when the 3-month carry+rolldown of the trade direction is >= 0.")
        print("OOS is not a pristine hold-out: this hypothesis was chosen after seeing Step 4.\n")
    log.info("Saved residual history, metrics, trades and charts")
    return {"metrics": met, "trades_carry": tr_c, "trades_nocarry": tr_n, "halflife": hl}


if __name__ == "__main__":
    run()
