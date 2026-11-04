"""
spreads.py - build the relative-value instruments, in basis points (bp).

This is the ONLY place where yields (stored in percent) are converted to bp (x100).

Instruments (defined in config.yaml -> instruments.definitions):
  NZ swap curve      2s5s, 2s10s, 2s5s10s butterfly   from NZ swaps
  AU bond curve      2s5s, 2s10s, 2s5s10s butterfly   from AU govt bonds (from au_start)
  AU-NZ bond spread  2y, 5y, 10y: AU bond minus RAW NZ benchmark bond (from au_start)
  NZ swap spread     2y, 5y, 10y: NZ swap minus RAW NZ benchmark bond
Each instrument value = 100 x sum(weight x yield). Back-adjusted "*_adj" yields are
never used (they are diagnostic only).

Inputs are the stored yields with the usual interior forward-fill (at most 2 days,
never past a series' last published value). So every instrument automatically stops
at the latest date on which ALL its inputs have data; that date is its "as_of".

Output: data/processed/instruments.csv, one row per (date, instrument):
  value_bp         instrument level in bp
  inputs_filled    True if any input was forward-filled that day (not freshly published)
  roll_day         True if an NZ bond leg had a confirmed / unclear / pending roll that
                   day: the day's CHANGE must be excluded from backtest and PnL
  post_roll_block  True on the roll day and the next `post_roll_block_days` trading days
                   (confirmed / unclear rolls), or from a pending roll onwards until it is
                   reviewed: no new trades may be opened
  roll_info        which bond rolled and its status (on roll days)
Roll flags only apply to bond-based instruments (those with an NZ government bond leg):
NZ benchmark bonds roll, swaps and RBA's interpolated AU yields don't.

Run from the project root:   python -m src.spreads
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.fetch import forward_fill_interior
from src.rolls import load_roll_dates
from src.utils import get_logger, load_config, project_path, read_rates

log = get_logger("spreads")

BP_PER_PERCENT = 100.0
ROLL_STATUSES_FLAGGED = ("confirmed", "unclear", "pending")


def to_bp(percent: pd.Series | pd.DataFrame | float) -> pd.Series | pd.DataFrame | float:
    """Convert yields in percent to basis points (1% = 100bp)."""
    return percent * BP_PER_PERCENT


def check_definitions(defs: dict[str, dict]) -> None:
    """Refuse any instrument that uses a back-adjusted (*_adj) series."""
    for name, spec in defs.items():
        bad = [s for s in spec["legs"] if s.endswith("_adj")]
        if bad:
            raise ValueError(f"Instrument {name} uses back-adjusted series {bad}; *_adj series are diagnostic only")


def formula(legs: dict[str, float]) -> str:
    """Human-readable formula, e.g. '2 x nz_swap_5y - nz_swap_2y - nz_swap_10y'."""
    parts = []
    for i, (series, w) in enumerate(legs.items()):
        sign = "-" if w < 0 else ("+" if i else "")
        mag = "" if abs(w) == 1 else f"{abs(w):g} x "
        parts.append(f"{sign} {mag}{series}".strip())
    return " ".join(parts)


def instrument_level(clean: pd.DataFrame, legs: dict[str, float]) -> pd.Series:
    """Instrument level in bp: 100 x sum(weight x yield). NaN whenever any leg is missing."""
    level = sum(w * clean[s] for s, w in legs.items())
    return to_bp(level)


def roll_flags(index: pd.DatetimeIndex, nz_bonds: list[str], roll_dates: pd.DataFrame,
               block_days: int) -> pd.DataFrame:
    """Roll-day and post-roll-block flags on an instrument's own dates.

    A roll on a date with no instrument value is attached to the next instrument date
    (the change into that date contains the jump).
    """
    flags = pd.DataFrame({"roll_day": False, "post_roll_block": False, "roll_info": ""}, index=index)
    if len(index) == 0 or not nz_bonds:
        return flags
    rows = roll_dates[roll_dates["series"].isin(nz_bonds) & roll_dates["status"].isin(ROLL_STATUSES_FLAGGED)]
    for r in rows.itertuples():
        i = int(index.searchsorted(r.date))
        if i >= len(index):
            continue                       # roll after the instrument's latest date
        if i == 0 and r.date < index[0]:
            continue                       # roll before the instrument starts
        day = index[i]
        flags.at[day, "roll_day"] = True
        info = f"{r.series} {r.status} roll {r.date.date()}"
        flags.at[day, "roll_info"] = "; ".join(x for x in (flags.at[day, "roll_info"], info) if x)
        end = len(index) if r.status == "pending" else min(len(index), i + block_days + 1)
        flags.iloc[i:end, flags.columns.get_loc("post_roll_block")] = True
    return flags


def build_instruments(rates: pd.DataFrame, defs: dict[str, dict], au_start: str,
                      roll_dates: pd.DataFrame, block_days: int, ffill_limit: int
                      ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build every instrument. Returns (long table of values + flags, summary table)."""
    check_definitions(defs)
    clean, _ = forward_fill_interior(rates, ffill_limit)
    filled = clean.notna() & rates.isna()
    start_au = pd.Timestamp(au_start)

    frames, summary = [], []
    for name, spec in defs.items():
        legs = spec["legs"]
        level = instrument_level(clean, legs)
        start = start_au if any(s.startswith("au_") for s in legs) else level.first_valid_index()
        level = level[level.index >= start].dropna()
        nz_bonds = [s for s in legs if s.startswith("nz_gov_")]

        out = pd.DataFrame({"instrument": name, "value_bp": level.round(2),
                            "inputs_filled": filled.loc[level.index, list(legs)].any(axis=1)}, index=level.index)
        out = out.join(roll_flags(level.index, nz_bonds, roll_dates, block_days))
        frames.append(out)

        summary.append({
            "instrument": name, "group": spec["group"], "formula_bp": f"100 x ({formula(legs)})",
            "inputs": ", ".join(legs), "bond_based": bool(nz_bonds),
            "start": level.index.min().date(), "as_of": level.index.max().date(), "n_obs": len(level),
            "latest_bp": round(level.iloc[-1], 1), "min_bp": round(level.min(), 1),
            "min_date": level.idxmin().date(), "max_bp": round(level.max(), 1),
            "max_date": level.idxmax().date(), "mean_bp": round(level.mean(), 1),
            "roll_days": int(out["roll_day"].sum()), "blocked_days": int(out["post_roll_block"].sum()),
        })

    long = pd.concat(frames).rename_axis("date").reset_index()
    return long, pd.DataFrame(summary)


def instrument_chart(long: pd.DataFrame, summary: pd.DataFrame) -> go.Figure:
    """One panel per instrument (bp), with roll days marked on bond-based instruments."""
    names = summary["instrument"].tolist()
    fig = make_subplots(rows=4, cols=3, shared_xaxes=False, vertical_spacing=0.07, horizontal_spacing=0.05,
                        subplot_titles=[f"{n} (bp)" for n in names])
    for k, name in enumerate(names):
        row, col = k // 3 + 1, k % 3 + 1
        d = long[long["instrument"] == name]
        line = d.set_index("date")["value_bp"]
        gap_starts = line.index[1:][np.diff(line.index.values) > np.timedelta64(7, "D")]
        line = pd.concat([line, pd.Series(np.nan, index=gap_starts - pd.Timedelta(days=1))]).sort_index()
        fig.add_trace(go.Scatter(x=line.index, y=line.values, mode="lines", line=dict(width=1, color="#1f77b4"),
                                 name="value (bp)", showlegend=k == 0, legendgroup="v"), row=row, col=col)
        rolls = d[d["roll_day"]]
        if len(rolls):
            fig.add_trace(go.Scatter(x=rolls["date"], y=rolls["value_bp"], mode="markers",
                                     marker=dict(color="red", size=6, symbol="triangle-down"),
                                     name="NZ bond roll day (change excluded)", text=rolls["roll_info"],
                                     hoverinfo="x+text", showlegend="roll" not in {t.legendgroup for t in fig.data},
                                     legendgroup="roll"), row=row, col=col)
        fig.update_yaxes(title_text="bp" if col == 1 else None, row=row, col=col)
    fig.update_layout(height=1100, width=1400, template="plotly_white",
                      title="RV instruments in basis points (AU from 2 Sep 2013; NZ bonds raw, swaps constant maturity)",
                      legend=dict(orientation="h", y=1.06, x=0), margin=dict(t=110, l=50, r=20, b=30))
    return fig


def run(cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    """Build instruments from rates.csv, save the table, summary and chart; print the summary."""
    cfg = cfg or load_config()
    paths = {k: project_path(v) for k, v in cfg["paths"].items()}
    rates = read_rates(paths["rates_file"])
    if rates is None:
        raise FileNotFoundError("rates.csv not found - run `python -m src.fetch` first")
    inst = cfg["instruments"]
    long, summary = build_instruments(rates, inst["definitions"], inst["au_start"],
                                      load_roll_dates(paths["roll_dates_file"]),
                                      inst["post_roll_block_days"], cfg["update"]["ffill_max_days"])

    out = long.copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out.to_csv(paths["instruments_file"], index=False)
    summary.to_csv(paths["instrument_summary_file"], index=False)

    charts = paths["charts_dir"]
    charts.mkdir(parents=True, exist_ok=True)
    fig = instrument_chart(long, summary)
    fig.write_html(charts / "instruments.html", include_plotlyjs="cdn")
    try:
        fig.write_image(charts / "instruments.png", scale=1)
    except Exception as e:  # static export needs kaleido + Chrome; the HTML chart is still saved
        log.warning("PNG export skipped (%s)", e)

    cols = ["instrument", "inputs", "start", "as_of", "n_obs", "latest_bp", "min_bp", "min_date",
            "max_bp", "max_date", "mean_bp", "roll_days", "blocked_days"]
    with pd.option_context("display.width", 250, "display.max_colwidth", 40):
        print("\nInstrument summary (all values in bp):\n")
        print(summary[cols].to_string(index=False))
        print()
    log.info("Saved %s (%d rows), %s and charts", paths["instruments_file"].name, len(long),
             paths["instrument_summary_file"].name)
    return summary


if __name__ == "__main__":
    run()
