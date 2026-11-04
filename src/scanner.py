"""
scanner.py - rolling z-scores, percentiles and the daily ranked RV table.

For every instrument (from data/processed/instruments.csv, values in bp):

  z_N = (today's value - mean of the last N trading days) / std of the last N trading days
        for N = 60 and 250. The window ends TODAY and uses nothing after it (no look-ahead).

  Trading days are counted on a weekday calendar starting at the instrument's first date.
  Only REAL values count: days where an input was forward-filled over a holiday
  (inputs_filled) and days with no data are left out of the window statistics.
  A z-score is NaN and flagged "insufficient history" if
    - fewer than 80% of the N trading days in the window have a real value, or
    - the window contains a run of more than 10 trading days without a real value.
  Days with forward-filled inputs never get a z-score, so they never produce a signal.

  percentile = share of all real values up to and including today that are <= today's
               value (an expanding window - again no look-ahead), in %.

The daily table ranks instruments by |z_250| ("signal_strength") and marks a "strong"
signal where z_60 and z_250 have the same sign and are both beyond +/-2. Instruments in
a post-roll block, with a pending roll, with insufficient history, a filled latest value
or stale data are shown but marked "no new trades".

Outputs:
  data/processed/scanner_history.csv      z-scores and percentile for every date (backtest input)
  outputs/scanner_<latest data date>.csv  today's ranked table (+ scanner_latest.csv)
  outputs/charts/scanner/<instrument>.html/.png  value with 250d +/-1 and +/-2 sd bands, and z-scores

Run from the project root:   python -m src.scanner
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.rolls import load_roll_dates, pending_series
from src.utils import get_logger, load_config, project_path, with_gap_breaks

log = get_logger("scanner")


# =============================================================================
# Rolling statistics (no look-ahead)
# =============================================================================

def gap_mask(real: pd.Series, window: int, max_gap: int) -> pd.Series:
    """True where the trailing `window` contains more than `max_gap` consecutive non-real days.

    `real` is a boolean Series on the trading-day calendar. For a run of missing days at
    positions a..b, the window ending at t overlaps it by min(b, t) - max(a, t-window+1) + 1
    days; that exceeds max_gap exactly for t in [a + max_gap, b + window - 1 - max_gap].
    """
    bad = np.zeros(len(real), dtype=bool)
    missing = ~real.to_numpy()
    i, n = 0, len(missing)
    while i < n:
        if missing[i]:
            j = i
            while j + 1 < n and missing[j + 1]:
                j += 1
            if j - i + 1 > max_gap:
                bad[i + max_gap: min(n, j + window - max_gap)] = True
            i = j + 1
        else:
            i += 1
    return pd.Series(bad, index=real.index)


def zscore_history(values: pd.Series, filled: pd.Series, windows: list[int], min_real_fraction: float,
                   max_gap: int, percentile_min_obs: int) -> pd.DataFrame:
    """Rolling z-scores, window means/stds and expanding percentile for ONE instrument.

    `values` (bp) and `filled` are indexed by the instrument's own dates. Every statistic
    at date t uses only values dated t or earlier. Returns a frame on the same dates.
    """
    calendar = pd.bdate_range(values.index.min(), values.index.max())
    x = values.where(~filled.astype(bool)).reindex(calendar)       # real values only
    real = x.notna()

    out = pd.DataFrame(index=calendar)
    for n in windows:
        need = math.ceil(min_real_fraction * n)
        mean = x.rolling(n, min_periods=need).mean()
        std = x.rolling(n, min_periods=need).std()
        enough = real.rolling(n, min_periods=1).sum() >= need
        ok = enough & ~gap_mask(real, n, max_gap) & real & (std > 0)
        out[f"z_{n}"] = ((x - mean) / std).where(ok)
        out[f"mean_{n}"] = mean.where(enough & ~gap_mask(real, n, max_gap))
        out[f"std_{n}"] = std.where(enough & ~gap_mask(real, n, max_gap))
        out[f"insufficient_{n}"] = ~(enough & ~gap_mask(real, n, max_gap))

    real_values = x.dropna()
    pct = real_values.expanding().rank(pct=True) * 100
    pct[np.arange(1, len(real_values) + 1) < percentile_min_obs] = np.nan
    out["percentile"] = pct.reindex(calendar)
    return out.reindex(values.index)


def build_history(instruments: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """z-scores and percentiles for every instrument and date (long format)."""
    s = cfg["scanner"]
    frames = []
    for name, d in instruments.groupby("instrument", sort=False):
        d = d.set_index("date").sort_index()
        z = zscore_history(d["value_bp"], d["inputs_filled"], s["windows"], s["min_real_fraction"],
                           s["max_gap_days"], s["percentile_min_obs"])
        frames.append(d.join(z).reset_index())
    return pd.concat(frames, ignore_index=True)


# =============================================================================
# Today's table
# =============================================================================

def change_over(d: pd.DataFrame, days: int) -> tuple[float, bool]:
    """Change in value_bp from the last date on or before (as_of - days calendar days) to as_of.

    Returns (change, spans_roll). A change that spans a roll day is set to NaN, because
    it contains an artificial jump.
    """
    last = d.index[-1]
    earlier = d.index[d.index <= last - pd.Timedelta(days=days)]
    if earlier.empty:
        return np.nan, False
    start = earlier[-1]
    spans_roll = bool(d.loc[(d.index > start) & (d.index <= last), "roll_day"].any())
    change = np.nan if spans_roll else d["value_bp"].iloc[-1] - d.loc[start, "value_bp"]
    return change, spans_roll


def describe(row: pd.Series, start: pd.Timestamp) -> str:
    """One-line plain-English description of an instrument's current reading."""
    name, v, z, pct = row["instrument"], row["value_bp"], row["z_250"], row["percentile"]
    high = z > 0 if pd.notna(z) else v > 0
    if name.startswith("au_nz_gov_"):
        t = name.split("_")[-1]
        what = f"AU {t} govt yield is {abs(v):.0f}bp {'above' if v >= 0 else 'below'} NZ {t}"
    elif name.startswith("nz_swap_spread_"):
        t = name.split("_")[-1]
        what = f"NZ {t} swap is {abs(v):.0f}bp {'above' if v >= 0 else 'below'} the NZ {t} govt bond"
    elif name.endswith("2s5s10s"):
        mkt = "NZ swap" if name.startswith("nz") else "AU bond"
        what = f"{mkt} 2s5s10s butterfly is {v:+.0f}bp (5y {'cheap' if high else 'rich'} vs 2y and 10y)"
    else:
        mkt = "NZ swap" if name.startswith("nz") else "AU bond"
        curve = name.split("_")[-1]
        what = f"{mkt} {curve} curve is {v:+.0f}bp ({'steeper' if high else 'flatter'} than usual)"
    z_txt = f"{z:+.1f} sd vs its 250-day average" if pd.notna(z) else "no 250-day z-score"
    pct_txt = f", higher than {pct:.0f}% of readings since {start:%b %Y}" if pd.notna(pct) else ""
    extra = f" [{row['trade_status']}]" if row["trade_status"] != "tradeable" else ""
    return f"{what}: {z_txt}{pct_txt}.{extra}"


def build_table(history: pd.DataFrame, defs: dict[str, dict], roll_dates: pd.DataFrame,
                cfg: dict[str, Any], today: pd.Timestamp) -> pd.DataFrame:
    """Ranked table of today's reading for every instrument."""
    s, max_age = cfg["scanner"], cfg["validation"]["max_age_days"]
    pending = pending_series(roll_dates)
    rows = []
    for name, d in history.groupby("instrument", sort=False):
        d = d.set_index("date").sort_index()
        last = d.iloc[-1]
        legs = list(defs[name]["legs"])
        market = "au" if any(l.startswith("au_") for l in legs) else "nz"
        chg_1d, roll_1d = change_over(d, 1)
        chg_1w, roll_1w = change_over(d, 7)
        age = (today - d.index[-1]).days
        pend = [f"{l} {dt.date()}" for l in legs for dt in pending.get(l, [])]

        flags = []
        if last["post_roll_block"]:
            flags.append("post-roll block")
        if pend:
            flags.append("pending roll review (" + ", ".join(pend) + ")")
        if last["insufficient_60"] or last["insufficient_250"]:
            flags.append("insufficient history (" + ", ".join(
                f"{n}d" for n in s["windows"] if last[f"insufficient_{n}"]) + ")")
        if age > max_age[market]:
            flags.append(f"stale data ({age} days old)")
        if last["inputs_filled"]:
            flags.append("filled input - no signal")
        if roll_1d or roll_1w:
            flags.append("change spans a roll day (not shown)")

        z60, z250 = last["z_60"], last["z_250"]
        strong = (pd.notna(z60) and pd.notna(z250) and np.sign(z60) == np.sign(z250)
                  and min(abs(z60), abs(z250)) > s["signal_z"])
        signal = ("strong high" if z250 > 0 else "strong low") if strong else ""
        blockers = [f for f in flags if not f.startswith("change spans")]
        trade_status = "no new trades (" + "; ".join(f.split(" (")[0] for f in blockers) + ")" if blockers else "tradeable"

        rows.append({
            "instrument": name, "group": defs[name]["group"], "as_of": d.index[-1].date(),
            "value_bp": last["value_bp"], "chg_1d_bp": chg_1d, "chg_1w_bp": chg_1w,
            "z_60": z60, "z_250": z250, "percentile": last["percentile"],
            "signal_strength": abs(z250) if pd.notna(z250) else np.nan, "signal": signal,
            "trade_status": trade_status, "flags": "; ".join(flags),
            "post_roll_block": bool(last["post_roll_block"]), "pending_roll": bool(pend),
            "insufficient_history": bool(last["insufficient_60"] or last["insufficient_250"]),
            "stale_data": age > max_age[market], "history_start": d.index[0],
        })
    table = pd.DataFrame(rows).sort_values("signal_strength", ascending=False, na_position="last")
    table.insert(0, "rank", range(1, len(table) + 1))
    table["description"] = [describe(r, r["history_start"]) for _, r in table.iterrows()]
    for c in ("value_bp", "chg_1d_bp", "chg_1w_bp"):
        table[c] = table[c].round(1)
    for c in ("z_60", "z_250", "signal_strength"):
        table[c] = table[c].round(2)
    table["percentile"] = table["percentile"].round(1)
    return table.drop(columns="history_start").reset_index(drop=True)


# =============================================================================
# Charts
# =============================================================================

def instrument_chart(d: pd.DataFrame, name: str) -> go.Figure:
    """Top: value with 250d mean and +/-1, +/-2 sd bands. Bottom: z_60 and z_250."""
    d = d.set_index("date").sort_index()
    rolls = d[d["roll_day"].astype(bool)]
    d = with_gap_breaks(d)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.62, 0.38], vertical_spacing=0.06,
                        subplot_titles=[f"{name}: value with 250-day mean and +/-1, +/-2 sd bands (bp)",
                                        "z-scores (60-day and 250-day)"])
    m, sd = d["mean_250"], d["std_250"]
    for k, colour in ((2, "rgba(214,39,40,0.6)"), (1, "rgba(255,127,14,0.6)")):
        for sign in (1, -1):
            fig.add_trace(go.Scatter(x=d.index, y=m + sign * k * sd, mode="lines",
                                     line=dict(width=1, dash="dot", color=colour),
                                     name=f"mean +/-{k} sd", showlegend=sign == 1, legendgroup=f"b{k}"), row=1, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=m, mode="lines", line=dict(width=1, color="grey"),
                             name="250-day mean"), row=1, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["value_bp"], mode="lines", line=dict(width=1.2, color="#1f77b4"),
                             name="value (bp)"), row=1, col=1)
    if len(rolls):
        fig.add_trace(go.Scatter(x=rolls.index, y=rolls["value_bp"], mode="markers", name="NZ bond roll day",
                                 marker=dict(color="red", size=6, symbol="triangle-down"),
                                 text=rolls["roll_info"], hoverinfo="x+text"), row=1, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["z_250"], mode="lines", line=dict(width=1.2, color="#1f77b4"),
                             name="z 250-day"), row=2, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["z_60"], mode="lines", line=dict(width=0.8, color="#9467bd"),
                             name="z 60-day", opacity=0.7), row=2, col=1)
    for level in (-2, 2):
        fig.add_hline(y=level, line=dict(color="red", dash="dash", width=1), row=2, col=1)
    fig.update_yaxes(title_text="bp", row=1, col=1)
    fig.update_yaxes(title_text="z", row=2, col=1)
    fig.update_layout(height=650, width=1100, template="plotly_white",
                      legend=dict(orientation="h", y=1.08, x=0), margin=dict(t=90, l=50, r=20, b=30))
    return fig


# =============================================================================
# Main run
# =============================================================================

def run(cfg: dict[str, Any] | None = None, today: pd.Timestamp | None = None) -> pd.DataFrame:
    """Compute z-score history, write today's ranked table and charts, print the top instruments."""
    cfg = cfg or load_config()
    today = (today or pd.Timestamp.today()).normalize()
    paths = {k: project_path(v) for k, v in cfg["paths"].items()}
    instruments = pd.read_csv(paths["instruments_file"], parse_dates=["date"], keep_default_na=False,
                              na_values=[""], dtype={"roll_info": str})
    instruments["roll_info"] = instruments["roll_info"].fillna("")
    defs = cfg["instruments"]["definitions"]

    history = build_history(instruments, cfg)
    hist_out = history.drop(columns=["roll_info"]).copy()
    hist_out["date"] = hist_out["date"].dt.strftime("%Y-%m-%d")
    hist_out.round(4).to_csv(paths["scanner_history_file"], index=False)

    table = build_table(history, defs, load_roll_dates(paths["roll_dates_file"]), cfg, today)
    data_date = pd.to_datetime(table["as_of"]).max().date()
    out_dir = paths["outputs_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / f"scanner_{data_date}.csv", index=False)
    table.to_csv(out_dir / "scanner_latest.csv", index=False)

    chart_dir = paths["charts_dir"] / "scanner"
    chart_dir.mkdir(parents=True, exist_ok=True)
    for name, d in history.groupby("instrument", sort=False):
        fig = instrument_chart(d, name)
        fig.write_html(chart_dir / f"{name}.html", include_plotlyjs="cdn")
        try:
            fig.write_image(chart_dir / f"{name}.png", scale=1)
        except Exception as e:  # static export needs kaleido + Chrome; HTML is still saved
            log.warning("PNG export skipped for %s (%s)", name, e)

    cols = ["rank", "instrument", "as_of", "value_bp", "chg_1d_bp", "chg_1w_bp", "z_60", "z_250",
            "percentile", "signal", "trade_status"]
    with pd.option_context("display.width", 250, "display.max_colwidth", 70):
        print(f"\nScanner - data to {data_date} (values and changes in bp)\n")
        print(table[cols].to_string(index=False))
        print(f"\nTop {cfg['scanner']['top_n']} by signal strength (|z_250|):")
        for r in table.head(cfg["scanner"]["top_n"]).itertuples():
            print(f"  {r.rank}. {r.instrument}{' [STRONG]' if r.signal else ''}: {r.description}")
        print()
    log.info("Saved scanner_%s.csv, scanner_latest.csv, %s and %d charts", data_date,
             paths["scanner_history_file"].name, len(defs))
    return table


if __name__ == "__main__":
    run()
