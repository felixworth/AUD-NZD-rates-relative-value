"""
pca.py - PCA-weighted 2s5s10s butterflies, one for each curve.

A simple butterfly (5y − 0.5 x 2y − 0.5 x 10y) is roughly "duration-neutral" but
still makes or loses money when the whole curve shifts in parallel (PC1, "level")
or pivots (PC2, "slope"), because the three legs don't have equal sensitivity to
those moves. PCA weights are chosen so that, on the recent history of yield CHANGES,
a 1bp 5y position hedged with w2 of 2y and w10 of 10y has (approximately) zero
exposure to PC1 and PC2. The leftover risk is mostly PC3 — genuine curvature.

Fit (each date t, no look-ahead):
  rolling PCA on daily yield changes (bp) over the trailing 250 trading days,
  using only data dated t or earlier. Filled days and NZ-bond roll days are left out.
  NZ: all B2 swap tenors (1y, 2y, 3y, 4y, 5y, 7y, 10y, 15y).
  AU: government bonds 2y, 3y, 5y, 10y, from 2 Sep 2013.

Eigenvector signs are fixed so PC1 loadings sum positive (a parallel sell-off) and
PC2 is positive at the long end (a steepener). That stops the 2y/10y weights flipping
sign from one day to the next for no economic reason.

The 5y weight is always 1. w2 and w10 solve:
  w · PC1_loadings = 0  and  w · PC2_loadings = 0  on the 2y, 5y, 10y legs.

Z-scores: at date t the fly's last 60/250 days are REBUILT with date t's weights
(not stitched from yesterday's weights), then z-scored. A change in the weights
cannot by itself create a signal.

Outputs:
  outputs/pca_weights.csv              daily w2, w5=1, w10 and variance explained
  outputs/pca_oos.csv                  walk-forward fly changes vs PC1/PC2 scores
  data/processed/pca_flies.csv         PCA fly levels + frozen-weight z-scores (scanner)
  outputs/charts/pca_*.html/.png       variance, fly vs 50/50, OOS correlations

Run from the project root:   python -m src.pca
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA

from src.fetch import forward_fill_interior
from src.rolls import load_roll_dates
from src.scanner import gap_mask
from src.spreads import to_bp
from src.utils import get_logger, load_config, project_path, read_rates, with_gap_breaks

log = get_logger("pca")


# =============================================================================
# PCA fit and sign convention
# =============================================================================

def fix_signs(loadings: np.ndarray) -> np.ndarray:
    """Flip each eigenvector so the signs are economically stable.

    PC1: sum of loadings > 0  (a parallel sell-off has a positive PC1 score).
    PC2: loading at the longest tenor > 0  (a steepener has a positive PC2 score).
    PC3: loading at the longest tenor > 0  (only to stop day-to-day flips).
    """
    out = loadings.copy()
    if out[:, 0].sum() < 0:
        out[:, 0] *= -1
    if out.shape[1] > 1 and out[-1, 1] < 0:
        out[:, 1] *= -1
    if out.shape[1] > 2 and out[-1, 2] < 0:
        out[:, 2] *= -1
    return out


def fit_pca(changes: np.ndarray, n_components: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """PCA on a window of daily changes (obs x tenors), in bp.

    Returns (loadings of shape n_tenors x n_components, explained-variance ratios).
    sklearn centres each tenor (subtracts its mean change) but does not scale, so this
    is covariance PCA: tenors that move more — usually the long end — get more weight.
    """
    n_comp = int(min(n_components, changes.shape[1], max(1, changes.shape[0] - 1)))
    model = PCA(n_components=n_comp, svd_solver="full")
    model.fit(changes)
    loadings = fix_signs(model.components_.T)
    var = model.explained_variance_ratio_
    if len(var) < n_components:
        var = np.pad(var, (0, n_components - len(var)), constant_values=np.nan)
        loadings = np.hstack([loadings, np.full((loadings.shape[0], n_components - loadings.shape[1]), np.nan)])
    return loadings, var


def fly_weights(loadings: np.ndarray, tenors: list[str], belly: str, wings: list[str]
                ) -> tuple[float, float]:
    """Solve w2, w10 so [w2, 1, w10] is orthogonal to PC1 and PC2 on the fly legs."""
    idx = {n: i for i, n in enumerate(tenors)}
    try:
        i2, i5, i10 = idx[wings[0]], idx[belly], idx[wings[1]]
    except KeyError:
        return np.nan, np.nan
    if loadings.shape[1] < 2 or np.isnan(loadings[[i2, i5, i10], :2]).any():
        return np.nan, np.nan
    a = np.array([[loadings[i2, 0], loadings[i10, 0]],
                  [loadings[i2, 1], loadings[i10, 1]]], dtype=float)
    b = np.array([-loadings[i5, 0], -loadings[i5, 1]], dtype=float)
    if abs(np.linalg.det(a)) < 1e-12:
        return np.nan, np.nan
    w2, w10 = np.linalg.solve(a, b)
    return float(w2), float(w10)


def weight_sentence(w2: float, w10: float) -> str:
    """Plain English: 'for every 1 of 5y, sell 0.60 of 2y and 0.50 of 10y'."""
    if not (np.isfinite(w2) and np.isfinite(w10)):
        return "weights not available"

    def leg(w: float, name: str) -> str:
        return f"{'sell' if w < 0 else 'buy'} {abs(w):.2f} of {name}"

    return f"for every 1 of 5y, {leg(w2, '2y')} and {leg(w10, '10y')}"


# =============================================================================
# Daily changes used in the fit
# =============================================================================

def yield_changes(clean: pd.DataFrame, raw: pd.DataFrame, tenors: list[str],
                  roll_days: pd.Index | None = None) -> pd.DataFrame:
    """Daily changes in bp on a weekday calendar.

    A tenor-day is NaN if that tenor is missing or was forward-filled. Whole rows are
    set to NaN on NZ-bond roll days so those changes never enter the PCA fit.
    """
    available = clean[tenors].dropna(how="all")
    cal = pd.bdate_range(available.index.min(), available.index.max())
    y = clean[tenors].reindex(cal)
    filled = y.notna() & raw.reindex(cal)[tenors].isna()
    chg = to_bp(y.diff()).mask(y.isna())
    chg.loc[filled.any(axis=1)] = np.nan
    if roll_days is not None and len(roll_days):
        chg.loc[chg.index.isin(pd.DatetimeIndex(roll_days))] = np.nan
    return chg


def window_ok(chg: pd.DataFrame, end: pd.Timestamp, window: int, min_real_fraction: float,
              max_gap: int, required: list[str]) -> pd.DataFrame | None:
    """Valid change-rows in the trailing `window` weekdays ending at `end`.

    Tenors missing for more than (1 - min_real_fraction) of the window are dropped
    (so NZ 15y can join later without blocking earlier dates). `required` tenors
    (the fly legs) must remain. Returns None if the window is too thin or too gappy.
    """
    cal = pd.bdate_range(end - pd.offsets.BDay(window - 1), end)
    block = chg.reindex(cal)
    real_any = block.notna().any(axis=1)
    need = math.ceil(min_real_fraction * window)
    if int(real_any.sum()) < need or bool(gap_mask(real_any, window, max_gap).iloc[-1]):
        return None
    keep = [c for c in block.columns if float(block[c].notna().mean()) >= min_real_fraction]
    if not set(required).issubset(keep):
        return None
    block = block[keep].dropna(how="any")
    return block if len(block) >= need else None


# =============================================================================
# Rolling fit
# =============================================================================

def rolling_pca(chg: pd.DataFrame, spec: dict[str, Any], cfg_pca: dict[str, Any]
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rolling fit ending at each date t (no look-ahead).

    Returns (weights table, walk-forward OOS table). OOS applies t-1's loadings and
    weights to the change on day t, so it does not refit.
    """
    window, belly, wings = cfg_pca["window"], spec["belly"], spec["wings"]
    simple, required = cfg_pca["simple_wings"], [belly, *wings]
    rows, oos_rows = [], []
    prev_loadings = prev_tenors = prev_w = None
    for end in chg.dropna(how="all").index:
        if prev_loadings is not None and prev_w is not None:
            dy = chg.loc[end]
            if dy.reindex(required).notna().all() and dy.reindex(prev_tenors).notna().all():
                score = dy.reindex(prev_tenors).to_numpy() @ prev_loadings[:, :2]
                w2, w10 = prev_w
                oos_rows.append({
                    "date": end,
                    "pca_fly_chg": w2 * dy[wings[0]] + dy[belly] + w10 * dy[wings[1]],
                    "simple_fly_chg": dy[belly] - simple * dy[wings[0]] - simple * dy[wings[1]],
                    "pc1_score": score[0], "pc2_score": score[1],
                })
        block = window_ok(chg, end, window, cfg_pca["min_real_fraction"], cfg_pca["max_gap_days"], required)
        if block is None:
            prev_loadings = prev_tenors = prev_w = None
            continue
        tenors = list(block.columns)
        loadings, var = fit_pca(block.to_numpy(), cfg_pca["n_components"])
        w2, w10 = fly_weights(loadings, tenors, belly, wings)
        idx = {n: i for i, n in enumerate(tenors)}
        rows.append({
            "date": end, "w_2y": w2, "w_5y": 1.0, "w_10y": w10,
            "n_obs": len(block), "n_tenors": len(tenors),
            "var_pc1": var[0], "var_pc2": var[1], "var_pc3": var[2],
            "pc1_2y": loadings[idx[wings[0]], 0], "pc1_5y": loadings[idx[belly], 0],
            "pc1_10y": loadings[idx[wings[1]], 0],
            "pc2_2y": loadings[idx[wings[0]], 1], "pc2_5y": loadings[idx[belly], 1],
            "pc2_10y": loadings[idx[wings[1]], 1],
        })
        prev_loadings, prev_tenors, prev_w = loadings, tenors, (w2, w10)
    weights = pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()
    oos = pd.DataFrame(oos_rows).set_index("date") if oos_rows else pd.DataFrame()
    return weights, oos


# =============================================================================
# Frozen-weight fly levels and z-scores
# =============================================================================

def endpoint_z(fly_bp: pd.Series, filled: pd.Series, windows: list[int], min_real_fraction: float,
               max_gap: int, percentile_min_obs: int) -> dict[str, Any]:
    """z-scores, mean/std and percentile of `fly_bp` at its LAST date only (no look-ahead)."""
    fill = filled.reindex(fly_bp.index).fillna(False).astype(bool)
    last = fly_bp.index[-1]
    out: dict[str, Any] = {"date": last, "value_bp": fly_bp.iloc[-1]}
    real_all = fly_bp.where(~fill).dropna()
    last_real = bool(np.isfinite(fly_bp.iloc[-1])) and not bool(fill.iloc[-1])
    out["percentile"] = ((real_all <= fly_bp.iloc[-1]).mean() * 100
                         if last_real and len(real_all) >= percentile_min_obs else np.nan)
    for n in windows:
        cal = pd.bdate_range(last - pd.offsets.BDay(n - 1), last)
        x = fly_bp.where(~fill).reindex(cal)
        real = x.notna()
        need = math.ceil(min_real_fraction * n)
        gappy = bool(gap_mask(real, n, max_gap).iloc[-1]) if len(real) else True
        std = float(x.std()) if real.sum() else np.nan
        ok = real.sum() >= need and not gappy and std > 0 and np.isfinite(x.iloc[-1])
        out[f"insufficient_{n}"] = not ok
        if ok:
            mean = float(x.mean())
            out[f"z_{n}"] = (float(x.iloc[-1]) - mean) / std
            out[f"mean_{n}"] = mean
            out[f"std_{n}"] = std
        else:
            out[f"z_{n}"] = out[f"mean_{n}"] = out[f"std_{n}"] = np.nan
    return out


def frozen_zscores(y2: pd.Series, y5: pd.Series, y10: pd.Series, filled: pd.Series,
                   weights: pd.DataFrame, scanner_cfg: dict[str, Any]) -> pd.DataFrame:
    """At each date t, rebuild the fly with weights_t and z-score the last point.

    y2/y5/y10 are yields in percent. value_bp equals the stitched fly at t (today's
    weights on today's yields); the window mean/std use the same weights on past days.
    """
    idx = y2.index.intersection(y5.index).intersection(y10.index)
    y2, y5, y10 = y2.reindex(idx), y5.reindex(idx), y10.reindex(idx)
    fill = filled.reindex(idx).fillna(False).astype(bool)
    y2v, y5v, y10v = y2.to_numpy(), y5.to_numpy(), y10.to_numpy()
    loc = {d: i for i, d in enumerate(idx)}
    rows = []
    args = (scanner_cfg["windows"], scanner_cfg["min_real_fraction"],
            scanner_cfg["max_gap_days"], scanner_cfg["percentile_min_obs"])
    for t, w in weights.iterrows():
        i = loc.get(t)
        if i is None or not np.isfinite(w["w_2y"]):
            continue
        fly_bp = to_bp(pd.Series(w["w_2y"] * y2v[: i + 1] + y5v[: i + 1] + w["w_10y"] * y10v[: i + 1],
                                 index=idx[: i + 1]))
        rows.append(endpoint_z(fly_bp, fill.iloc[: i + 1], *args))
    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()


# =============================================================================
# Charts
# =============================================================================

def variance_chart(weights: dict[str, pd.DataFrame]) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=["NZ swap curve: share of daily-change variance",
                                        "AU government curve: share of daily-change variance"])
    colours = {"var_pc1": "#1f77b4", "var_pc2": "#ff7f0e", "var_pc3": "#2ca02c"}
    names = {"var_pc1": "PC1 (level)", "var_pc2": "PC2 (slope)", "var_pc3": "PC3 (curvature)"}
    for row, key in enumerate(("nz", "au"), start=1):
        d = with_gap_breaks(weights[key][list(colours)])
        for col, colour in colours.items():
            fig.add_trace(go.Scatter(x=d.index, y=100 * d[col], mode="lines",
                                     line=dict(width=1.2, color=colour), name=names[col],
                                     showlegend=row == 1, legendgroup=col), row=row, col=1)
        fig.update_yaxes(title_text="% of variance", range=[0, 100], row=row, col=1)
    fig.update_layout(height=700, width=1100, template="plotly_white",
                      legend=dict(orientation="h", y=1.06, x=0), margin=dict(t=80, l=50, r=20, b=30))
    return fig


def fly_chart(pca_level: dict[str, pd.Series], simple_level: dict[str, pd.Series]) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=["NZ swap 2s5s10s: PCA-weighted vs 50/50 (bp, 5y weight = 1)",
                                        "AU bond 2s5s10s: PCA-weighted vs 50/50 (bp, 5y weight = 1)"])
    for row, key in enumerate(("nz", "au"), start=1):
        pca = with_gap_breaks(pca_level[key])
        smp = with_gap_breaks(simple_level[key].reindex(pca_level[key].index))
        fig.add_trace(go.Scatter(x=pca.index, y=pca, mode="lines", line=dict(width=1.2, color="#1f77b4"),
                                 name="PCA-weighted fly", showlegend=row == 1, legendgroup="pca"), row=row, col=1)
        fig.add_trace(go.Scatter(x=smp.index, y=smp, mode="lines", line=dict(width=1, color="grey"),
                                 name="50/50 fly (5y − 0.5×2y − 0.5×10y)", showlegend=row == 1,
                                 legendgroup="simple"), row=row, col=1)
        fig.update_yaxes(title_text="bp", row=row, col=1)
    fig.update_layout(height=700, width=1100, template="plotly_white",
                      legend=dict(orientation="h", y=1.06, x=0), margin=dict(t=80, l=50, r=20, b=30))
    return fig


def oos_chart(oos: dict[str, pd.DataFrame]) -> go.Figure:
    fig = make_subplots(rows=2, cols=2, shared_xaxes=False, vertical_spacing=0.12, horizontal_spacing=0.08,
                        subplot_titles=["NZ: OOS fly change vs PC1 (level)", "NZ: OOS fly change vs PC2 (slope)",
                                        "AU: OOS fly change vs PC1 (level)", "AU: OOS fly change vs PC2 (slope)"])
    for r, key in enumerate(("nz", "au")):
        d = oos[key].dropna()
        for c, factor in enumerate(("pc1_score", "pc2_score")):
            fig.add_trace(go.Scatter(x=d[factor], y=d["pca_fly_chg"], mode="markers",
                                     marker=dict(size=4, color="#1f77b4", opacity=0.35),
                                     name="PCA fly", showlegend=r == 0 and c == 0, legendgroup="pca"),
                          row=r + 1, col=c + 1)
            fig.add_trace(go.Scatter(x=d[factor], y=d["simple_fly_chg"], mode="markers",
                                     marker=dict(size=4, color="grey", opacity=0.25),
                                     name="50/50 fly", showlegend=r == 0 and c == 0, legendgroup="simple"),
                          row=r + 1, col=c + 1)
            fig.update_xaxes(title_text=f"{factor.split('_')[0].upper()} score (bp)", row=r + 1, col=c + 1)
            fig.update_yaxes(title_text="fly change (bp)", row=r + 1, col=c + 1)
    fig.update_layout(height=750, width=1100, template="plotly_white",
                      legend=dict(orientation="h", y=1.06, x=0), margin=dict(t=80, l=50, r=20, b=40))
    return fig


def _write_fig(fig: go.Figure, charts, stem: str) -> None:
    fig.write_html(charts / f"{stem}.html", include_plotlyjs="cdn")
    try:
        fig.write_image(charts / f"{stem}.png", scale=1)
    except Exception as e:
        log.warning("PNG export skipped for %s (%s)", stem, e)


# =============================================================================
# One curve
# =============================================================================

def run_curve(name: str, spec: dict[str, Any], clean: pd.DataFrame, raw: pd.DataFrame,
              roll_days: pd.Index, cfg: dict[str, Any]
              ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """Fit, z-score and OOS-check one curve. Returns weights, oos, pca_level, simple_level, fly_z."""
    pca_cfg, scan_cfg = cfg["pca"], cfg["scanner"]
    tenors, belly, wings = spec["tenors"], spec["belly"], spec["wings"]
    start = pd.Timestamp(spec["start"]) if spec.get("start") else clean[tenors].dropna(how="all").index.min()
    chg = yield_changes(clean.loc[start:], raw.loc[start:], tenors, roll_days)
    log.info("%s: fitting rolling PCA on %d change-dates, %d tenors", name, chg.dropna(how="all").shape[0], len(tenors))
    weights, oos = rolling_pca(chg, spec, pca_cfg)
    if weights.empty:
        raise RuntimeError(f"PCA produced no weights for {name} — not enough history?")

    y2, y5, y10 = clean[wings[0]], clean[belly], clean[wings[1]]
    filled = (clean[[wings[0], belly, wings[1]]].notna() & raw[[wings[0], belly, wings[1]]].isna()).any(axis=1)
    full = y2.dropna().index.intersection(y5.dropna().index).intersection(y10.dropna().index)
    full = full[full >= start]
    w_idx = weights.index.intersection(full)
    pca_level = to_bp(weights.loc[w_idx, "w_2y"] * y2.loc[w_idx]
                      + y5.loc[w_idx] + weights.loc[w_idx, "w_10y"] * y10.loc[w_idx])
    simple = pca_cfg["simple_wings"]
    simple_level = to_bp(y5.loc[w_idx] - simple * y2.loc[w_idx] - simple * y10.loc[w_idx])
    fly_z = frozen_zscores(y2.loc[full], y5.loc[full], y10.loc[full],
                           filled.reindex(full).fillna(False).astype(bool),
                           weights.loc[w_idx], scan_cfg)
    if not fly_z.empty:
        fly_z["instrument"] = spec["fly_name"]
        fly_z["inputs_filled"] = filled.reindex(fly_z.index).fillna(False).astype(bool)
        fly_z["roll_day"] = False
        fly_z["post_roll_block"] = False
        fly_z["roll_info"] = ""
    return weights, oos, pca_level, simple_level, fly_z


def _corr_table(oos: pd.DataFrame) -> pd.DataFrame:
    d = oos.dropna()
    if d.empty:
        return pd.DataFrame()
    return pd.DataFrame({
        "pca_vs_pc1": [d["pca_fly_chg"].corr(d["pc1_score"])],
        "pca_vs_pc2": [d["pca_fly_chg"].corr(d["pc2_score"])],
        "simple_vs_pc1": [d["simple_fly_chg"].corr(d["pc1_score"])],
        "simple_vs_pc2": [d["simple_fly_chg"].corr(d["pc2_score"])],
        "n_oos": [len(d)],
    })


# =============================================================================
# Main run
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

    weights, oos, pca_level, simple_level, flies = {}, {}, {}, {}, []
    for name, spec in cfg["pca"]["curves"].items():
        w, o, lvl, smp, z = run_curve(name, spec, clean, raw, roll_days, cfg)
        w.insert(0, "curve", name)
        w.insert(1, "fly", spec["fly_name"])
        weights[name], oos[name], pca_level[name], simple_level[name] = w, o, lvl, smp
        if not z.empty:
            flies.append(z.reset_index())
        last = w.iloc[-1]
        print(f"\n{name.upper()} PCA fly as of {w.index[-1].date()}: {weight_sentence(last['w_2y'], last['w_10y'])}")
        print(f"  variance explained: PC1 {100*last['var_pc1']:.1f}%  PC2 {100*last['var_pc2']:.1f}%  "
              f"PC3 {100*last['var_pc3']:.1f}%  (PC1+PC2 {100*(last['var_pc1']+last['var_pc2']):.1f}%)")
        corr = _corr_table(o)
        if len(corr):
            r = corr.iloc[0]
            print(f"  OOS correlation of daily fly changes with PC1/PC2 "
                  f"(n={int(r.n_oos)}): PCA {r.pca_vs_pc1:+.3f}/{r.pca_vs_pc2:+.3f}  "
                  f"vs 50/50 {r.simple_vs_pc1:+.3f}/{r.simple_vs_pc2:+.3f}")

    w_all = pd.concat(weights.values()).reset_index()
    w_all["date"] = pd.to_datetime(w_all["date"]).dt.strftime("%Y-%m-%d")
    paths["pca_weights_file"].parent.mkdir(parents=True, exist_ok=True)
    w_all.to_csv(paths["pca_weights_file"], index=False, float_format="%.6f")

    o_frames = [v.assign(curve=k) for k, v in oos.items() if not v.empty]
    if o_frames:
        o_out = pd.concat(o_frames).reset_index()
        o_out["date"] = pd.to_datetime(o_out["date"]).dt.strftime("%Y-%m-%d")
        o_out.to_csv(paths["pca_oos_file"], index=False, float_format="%.6f")

    if flies:
        cols = ["date", "instrument", "value_bp", "inputs_filled", "roll_day", "post_roll_block",
                "z_60", "mean_60", "std_60", "insufficient_60", "z_250", "mean_250", "std_250",
                "insufficient_250", "percentile", "roll_info"]
        fly_out = pd.concat(flies, ignore_index=True)
        fly_out["date"] = pd.to_datetime(fly_out["date"]).dt.strftime("%Y-%m-%d")
        fly_out = fly_out[[c for c in cols if c in fly_out.columns]]
        fly_out.round(4).to_csv(paths["pca_flies_file"], index=False)

    charts = paths["charts_dir"]
    charts.mkdir(parents=True, exist_ok=True)
    _write_fig(variance_chart(weights), charts, "pca_variance")
    _write_fig(fly_chart(pca_level, simple_level), charts, "pca_fly_vs_simple")
    _write_fig(oos_chart(oos), charts, "pca_oos")
    log.info("Saved %s, %s, %s and charts", paths["pca_weights_file"].name,
             paths["pca_oos_file"].name, paths["pca_flies_file"].name)
    return weights


if __name__ == "__main__":
    run()
