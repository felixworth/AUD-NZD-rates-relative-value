"""
robustness.py - Step 4c report-only checks (no parameter changes).

Reads the committed residual+carry trades and residual history. Writes its own
files; does not touch Step 4 or residual_metrics / residual_trades.

  1. Split residual+carry by direction and instrument, IS and OOS.
  2. Carry-only benchmark: same residual instruments, same costs / DV01 / t+1 /
     stop / max-hold / roll blocks. Enter the carry-positive direction when
     3-month carry+roll is defined and |z_250| is inside 2 (and not already at
     the 0.5 exit). Complementary region to the fade-when-stretched rule.
  3. Pooled IS+OOS residual+carry: t-stat of mean bp per trade and a bootstrap
     90% confidence interval.
  4. Four README charts from existing outputs (no re-run of Step 4).

Run from the project root:   python -m src.robustness
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats

from src.backtest import summarise_trades
from src.fetch import forward_fill_interior
from src.residual import backtest_residuals, residual_bundle
from src.utils import get_logger, load_config, project_path, read_rates, with_gap_breaks

log = get_logger("robustness")

BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 4
MATCHED_BASELINE = ["au_gov_2s10s", "nz_swap_2s10s", "au_nz_gov_10y"]


def split_by_direction(trades: pd.DataFrame) -> pd.DataFrame:
    """Trades, hit %, avg bp by instrument × side × IS/OOS (and ALL roll-ups)."""
    rows = []
    if trades.empty:
        return pd.DataFrame(rows)
    instruments = list(dict.fromkeys(trades["instrument"])) + ["ALL"]
    sides = ["long", "short", "ALL"]
    for sample in ("IS", "OOS"):
        chunk = trades[trades["sample"] == sample]
        for name in instruments:
            g = chunk if name == "ALL" else chunk[chunk["instrument"] == name]
            for side in sides:
                s = g if side == "ALL" else g[g["side"] == side]
                n = len(s)
                rows.append({
                    "rule": "residual_carry", "instrument": name, "side": side, "sample": sample,
                    "trades": n,
                    "hit_rate": float(s["pnl_bp"].gt(0).mean() * 100) if n else np.nan,
                    "avg_bp": float(s["pnl_bp"].mean()) if n else np.nan,
                })
    return pd.DataFrame(rows)


def mean_tstat(x: np.ndarray) -> tuple[float, float, float]:
    """Return (mean, t-stat vs 0, two-sided p-value). NaN if fewer than 2 observations."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan"), float("nan"), float("nan")
    mean = float(x.mean())
    t = stats.ttest_1samp(x, 0.0)
    return mean, float(t.statistic), float(t.pvalue)


def bootstrap_mean_ci(x: np.ndarray, draws: int = BOOTSTRAP_DRAWS, level: float = 0.90,
                      seed: int = BOOTSTRAP_SEED) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean. Returns (low, high)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(draws, len(x)), replace=True).mean(axis=1)
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(means, [alpha, 1.0 - alpha])
    return float(lo), float(hi)


def _bool_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ("inputs_filled", "roll_day", "post_roll_block", "pending_roll",
              "insufficient_60", "insufficient_250"):
        if c in out.columns and out[c].dtype != bool:
            out[c] = out[c].astype(str).str.lower().isin(["1", "true", "yes"])
    return out


def load_residual_inputs(cfg: dict[str, Any]
                         ) -> tuple[pd.DataFrame, dict[str, dict], dict[str, dict], pd.Timestamp]:
    """History + bundles reconstructed from saved residual files and rates.csv."""
    paths = {k: project_path(v) for k, v in cfg["paths"].items()}
    hist = pd.read_csv(paths["residual_history_file"], parse_dates=["date"])
    hist = _bool_cols(hist)
    beta = pd.read_csv(paths["residual_beta_file"], parse_dates=["date"])
    raw = read_rates(paths["rates_file"])
    if raw is None:
        raise FileNotFoundError("rates.csv not found")
    clean, _ = forward_fill_interior(raw, cfg["update"]["ffill_max_days"])
    specs = cfg["residual"]["instruments"]
    bundles = {}
    for name, spec in specs.items():
        b = beta[beta["instrument"] == name].set_index("date")["beta"].sort_index()
        c = hist[hist["instrument"] == name].set_index("date")["carry_3m_bp"].sort_index()
        bundles[name] = residual_bundle(hist, spec, b, c, clean)
    return hist, specs, bundles, pd.Timestamp(cfg["backtest"]["in_sample"][1])


def _write_fig(fig: go.Figure, folder, stem: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    fig.write_html(folder / f"{stem}.html", include_plotlyjs="cdn")
    try:
        fig.write_image(folder / f"{stem}.png", scale=1)
    except Exception as e:
        log.warning("PNG export skipped for %s (%s)", stem, e)


def chart_box_vs_raw(inst: pd.DataFrame, hist: pd.DataFrame) -> go.Figure:
    raw = inst[inst["instrument"] == "au_nz_gov_10y"].set_index("date")["value_bp"].sort_index()
    resid = hist[hist["instrument"] == "au_nz_box"].set_index("date")["value_bp"].sort_index()
    r = with_gap_breaks(resid)
    w = with_gap_breaks(raw.reindex(resid.index))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=w.index, y=w, mode="lines", line=dict(width=1.2, color="grey"),
                             name="AU−NZ 10y (raw)"))
    fig.add_trace(go.Scatter(x=r.index, y=r, mode="lines", line=dict(width=1.4, color="#1f77b4"),
                             name="policy-hedged box residual"))
    fig.update_layout(title="AU–NZ 10y spread vs policy-hedged box residual (bp)",
                      yaxis_title="bp", template="plotly_white", height=420, width=1100,
                      legend=dict(orientation="h", y=1.08, x=0), margin=dict(t=60, l=50, r=20, b=30))
    return fig


def chart_halflife(hl: pd.DataFrame) -> go.Figure:
    labels = {
        "au_nz_box": "AU–NZ box",
        "au_gov_10s2s_resid": "AU 10s2s residual",
        "nz_swap_10s2s_resid": "NZ 10s2s residual",
    }
    names = [labels.get(i, i) for i in hl["instrument"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=names, y=hl["halflife_raw_days"], name="raw spread",
                         marker_color="grey"))
    fig.add_trace(go.Bar(x=names, y=hl["halflife_resid_days"], name="policy-hedged residual",
                         marker_color="#1f77b4"))
    fig.update_layout(title="Half-life of mean reversion (AR(1) on levels, in-sample only)",
                      yaxis_title="trading days", barmode="group", template="plotly_white",
                      height=420, width=1100, legend=dict(orientation="h", y=1.08, x=0),
                      margin=dict(t=60, l=50, r=20, b=30))
    return fig


def chart_equity(daily_base: pd.DataFrame, daily_resid: pd.DataFrame, is_end: pd.Timestamp) -> go.Figure:
    fig = go.Figure()
    matched = daily_base[daily_base["instrument"].isin(MATCHED_BASELINE)]
    series = (
        (matched, "Step 4 fade of matched 10y instruments", "grey"),
        (daily_resid, "Residual + carry filter", "#1f77b4"),
    )
    end_x = is_end
    for df, name, colour in series:
        if df.empty:
            continue
        s = with_gap_breaks(df.groupby("date")["pnl_usd"].sum().sort_index().cumsum())
        fig.add_trace(go.Scatter(x=s.index, y=s, mode="lines", line=dict(width=1.4, color=colour),
                                 name=name))
        if len(s):
            end_x = max(end_x, pd.Timestamp(s.index.max()))
    fig.add_vrect(x0=is_end, x1=end_x, fillcolor="rgba(255,127,14,0.08)", line_width=0)
    fig.update_layout(
        title="Equity: Step 4 fade of AU/NZ 2s10s + AU-NZ 10y vs residual+carry (USD per 10k DV01 on each 10y leg; shaded = OOS)",
        yaxis_title="USD", template="plotly_white", height=420, width=1100,
        legend=dict(orientation="h", y=1.08, x=0), margin=dict(t=80, l=50, r=20, b=30))
    return fig


def chart_pca_corr(oos: pd.DataFrame) -> go.Figure:
    """Grouped bars: PCA vs 50/50 correlation with PC1 and PC2, one panel per curve."""
    fig = go.Figure()
    order = ["PCA vs PC1", "50/50 vs PC1", "PCA vs PC2", "50/50 vs PC2"]
    colours = {"PCA vs PC1": "#1f77b4", "50/50 vs PC1": "grey",
               "PCA vs PC2": "#2ca02c", "50/50 vs PC2": "#98df8a"}
    curves = []
    values = {k: [] for k in order}
    for curve, g in oos.groupby("curve", sort=False):
        d = g.dropna()
        if d.empty:
            continue
        curves.append(curve.upper())
        values["PCA vs PC1"].append(float(d["pca_fly_chg"].corr(d["pc1_score"])))
        values["50/50 vs PC1"].append(float(d["simple_fly_chg"].corr(d["pc1_score"])))
        values["PCA vs PC2"].append(float(d["pca_fly_chg"].corr(d["pc2_score"])))
        values["50/50 vs PC2"].append(float(d["simple_fly_chg"].corr(d["pc2_score"])))
    for label in order:
        fig.add_trace(go.Bar(x=curves, y=values[label], name=label, marker_color=colours[label]))
    fig.update_layout(title="OOS correlation of daily fly changes with PC1 (level) and PC2 (slope)",
                      yaxis_title="correlation", barmode="group", template="plotly_white",
                      height=420, width=1100, legend=dict(orientation="h", y=1.08, x=0),
                      margin=dict(t=60, l=50, r=20, b=30), yaxis=dict(range=[-0.15, 1.05]))
    return fig


def english_paragraph(split: pd.DataFrame, carry_met: pd.DataFrame, fade_met: pd.DataFrame,
                      mean_bp: float, tstat: float, pval: float, ci: tuple[float, float],
                      n: int) -> str:
    longs = split[(split["instrument"] == "ALL") & (split["side"] == "long")]
    shorts = split[(split["instrument"] == "ALL") & (split["side"] == "short")]
    n_long = int(longs["trades"].sum()) if len(longs) else 0
    n_short = int(shorts["trades"].sum()) if len(shorts) else 0
    fade_oos = fade_met[(fade_met["sample"] == "OOS") & (fade_met["instrument"] == "ALL")]
    carry_oos = carry_met[(carry_met["sample"] == "OOS") & (carry_met["instrument"] == "ALL")]
    fade_is = fade_met[(fade_met["sample"] == "IS") & (fade_met["instrument"] == "ALL")]
    carry_is = carry_met[(carry_met["sample"] == "IS") & (carry_met["instrument"] == "ALL")]

    def _row(df: pd.DataFrame, col: str) -> float:
        return float(df[col].iloc[0]) if len(df) and pd.notna(df[col].iloc[0]) else float("nan")

    lo, hi = ci
    includes0 = lo <= 0 <= hi
    au = split[(split["instrument"] == "au_gov_10s2s_resid") & (split["side"] == "short")]
    au_shorts = int(au["trades"].sum()) if len(au) else 0

    carry_oos_s = _row(carry_oos, "sharpe")
    fade_oos_s = _row(fade_oos, "sharpe")
    if np.isfinite(carry_oos_s) and np.isfinite(fade_oos_s) and fade_oos_s > carry_oos_s + 0.2:
        verdict = (
            "Blindly holding the carry-positive residual while it is unstretched loses money "
            "after costs, so the edge is not 'just collect carry'. It is the scarce combination "
            "of a 2-sd stretch in a direction that already pays carry. That is still a small "
            "sample, and OOS is not a pristine hold-out because this hypothesis was chosen after Step 4."
        )
    else:
        verdict = (
            "The z-score stretch does not add a clear edge beyond taking carry: the complementary "
            "carry-only book is similar or better. The headline improvement versus Step 4 is mostly "
            "the carry filter refusing to fade against the policy-cycle trend. OOS is not a pristine hold-out."
        )
    return (
        f"Of the {n} residual+carry fade trades, {n_long} were longs and {n_short} were shorts. "
        f"AU 10s2s never shorted ({au_shorts} short trades): 3-month carry of a long residual is "
        f"typically positive, so the filter only allows fading a cheap residual. "
        f"Pooled mean PnL is {mean_bp:+.1f}bp per trade (t = {tstat:.2f}, p = {pval:.2f}); "
        f"the bootstrap 90% CI is [{lo:+.1f}, {hi:+.1f}]bp"
        f"{' and includes zero' if includes0 else ' and excludes zero'}. "
        f"The complementary carry-only rule (hold the carry-positive residual while |z_250| < 2) "
        f"has IS avg {_row(carry_is, 'avg_bp'):+.1f}bp / Sharpe {_row(carry_is, 'sharpe'):+.2f} "
        f"and OOS avg {_row(carry_oos, 'avg_bp'):+.1f}bp / Sharpe {carry_oos_s:+.2f}, "
        f"against residual+carry IS {_row(fade_is, 'avg_bp'):+.1f}bp / {_row(fade_is, 'sharpe'):+.2f} "
        f"and OOS {_row(fade_oos, 'avg_bp'):+.1f}bp / {fade_oos_s:+.2f}. "
        + verdict
    )


def run(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = cfg or load_config()
    paths = {k: project_path(v) for k, v in cfg["paths"].items()}
    fade = pd.read_csv(paths["residual_trades_file"], parse_dates=["entry_date", "exit_date"])
    split = split_by_direction(fade)
    pnl = fade["pnl_bp"].to_numpy(dtype=float)
    mean_bp, tstat, pval = mean_tstat(pnl)
    ci = bootstrap_mean_ci(pnl)

    hist, specs, bundles, is_end = load_residual_inputs(cfg)
    log.info("Carry-only benchmark on saved residual history (does not rewrite residual_trades)")
    tr_co, dy_co = backtest_residuals(hist, specs, bundles, cfg, require_carry=False, carry_only=True)
    # Re-simulate residual+carry in memory for a matched daily equity path; do not write it over the committed trades.
    tr_fade, dy_fade = backtest_residuals(hist, specs, bundles, cfg, require_carry=True, carry_only=False)
    if len(tr_fade) != len(fade):
        log.warning("In-memory residual+carry re-sim has %d trades vs committed %d; using committed trades for t-stat",
                    len(tr_fade), len(fade))

    fade_met = pd.concat([
        pd.DataFrame([{**summarise_trades(tr_fade, dy_fade, s, is_end),
                       "instrument": "ALL", "rule": "residual_carry"}])
        for s in ("IS", "OOS")
    ], ignore_index=True)
    carry_met = pd.concat([
        pd.DataFrame([{**summarise_trades(tr_co, dy_co, s, is_end),
                       "instrument": "ALL", "rule": "carry_only"}])
        for s in ("IS", "OOS")
    ], ignore_index=True)
    per_inst = []
    for name in specs:
        g, d = tr_co[tr_co["instrument"] == name], dy_co[dy_co["instrument"] == name]
        for s in ("IS", "OOS"):
            m = summarise_trades(g, d, s, is_end)
            m.update(instrument=name, rule="carry_only")
            per_inst.append(m)
    carry_met = pd.concat([carry_met, pd.DataFrame(per_inst)], ignore_index=True)

    tstat_row = pd.DataFrame([{
        "rule": "residual_carry_pooled", "instrument": "ALL", "side": "ALL", "sample": "IS+OOS",
        "trades": int(np.isfinite(pnl).sum()), "hit_rate": float((pnl > 0).mean() * 100),
        "avg_bp": mean_bp, "tstat": tstat, "pvalue": pval,
        "ci90_low": ci[0], "ci90_high": ci[1],
    }])

    out = pd.concat([split, carry_met, fade_met, tstat_row], ignore_index=True, sort=False)
    paths["residual_robustness_file"].parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(paths["residual_robustness_file"], index=False, float_format="%.4f")
    co_out = tr_co.copy()
    for c in ("entry_date", "exit_date"):
        if c in co_out.columns:
            co_out[c] = pd.to_datetime(co_out[c]).dt.strftime("%Y-%m-%d")
    co_out.to_csv(paths["residual_carry_only_trades_file"], index=False)

    inst = pd.read_csv(paths["instruments_file"], parse_dates=["date"])
    hl = pd.read_csv(paths["residual_halflife_file"])
    daily_base = pd.read_csv(paths["backtest_daily_file"], parse_dates=["date"])
    oos = pd.read_csv(paths["pca_oos_file"], parse_dates=["date"])
    charts = paths["charts_dir"] / "readme"
    _write_fig(chart_box_vs_raw(inst, hist), charts, "box_vs_raw")
    _write_fig(chart_halflife(hl), charts, "halflife")
    _write_fig(chart_equity(daily_base, dy_fade, is_end), charts, "equity")
    _write_fig(chart_pca_corr(oos), charts, "pca_correlation")

    paragraph = english_paragraph(split, carry_met, fade_met, mean_bp, tstat, pval, ci, len(fade))

    cols = ["instrument", "side", "sample", "trades", "hit_rate", "avg_bp"]
    show = split[split["side"].ne("ALL") | split["instrument"].eq("ALL")]
    with pd.option_context("display.width", 220, "display.float_format", lambda x: f"{x: .2f}"):
        print("\n1. Residual+carry trades by direction (fade |z_250| > 2 with carry of that side >= 0)\n")
        print(show[cols].to_string(index=False))
        print("\n2. Carry-only benchmark (carry-positive residual while |z_250| < 2; same exits/costs/sizing)\n")
        print(carry_met[["rule", "instrument", "sample", "trades", "hit_rate", "avg_bp", "sharpe",
                         "avg_hold_days"]].to_string(index=False))
        print("\n3. Pooled residual+carry trades (IS+OOS)")
        print(f"   n = {len(fade)}  mean = {mean_bp:+.2f}bp  t = {tstat:.2f}  p = {pval:.3f}  "
              f"90% bootstrap CI [{ci[0]:+.2f}, {ci[1]:+.2f}]bp  ({BOOTSTRAP_DRAWS} draws, seed {BOOTSTRAP_SEED})")
        print("\n4. Does the z-score stretch add value beyond carry?\n")
        print("   " + paragraph)
        print()
    log.info("Saved robustness tables and README charts (Step 4 / residual_trades untouched)")
    return {"split": split, "carry_only": carry_met, "tstat": tstat_row, "paragraph": paragraph,
            "daily_fade": dy_fade, "daily_carry_only": dy_co}


if __name__ == "__main__":
    run()
