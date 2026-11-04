"""
rolls.py - benchmark-bond "rolls" (artificial jumps) in government bond yields.

Why: RBNZ's "2/5/10 year" bond yields are specific benchmark bonds. Roughly once a
year RBNZ switches each benchmark to a newer bond, so the series jumps even though
the market didn't move. Swap rates are constant-maturity and don't roll, so they
are a clean reference.

The reviewed list lives in data/processed/roll_dates.csv (one row per date+series):

  status     include  effect
  ---------  -------  ------------------------------------------------------------
  confirmed  Y        history before the date is back-adjusted by the jump, AND the
                      roll-day change is excluded from PCA / backtest / PnL
  unclear    N        roll-day change excluded as a precaution; NO back-adjustment
  pending    N        new automatic candidate awaiting review: change excluded from
                      paper-book PnL, instruments marked "pending roll review" in the
                      scanner, no new paper trades on them
  rejected   N        ignored (it was a real market move)

Back-adjustment works like a continuous futures contract: on a confirmed roll date
with an artificial jump J, every earlier value is shifted by J, so the series has no
jump. Shifting the whole past by a constant does not change any rolling z-score
computed at a date before the roll, so it does not leak future information into
z-scores - but adjusted LEVELS before a roll are expressed in "today's bond" terms.

Detection of NEW candidates (dates after `rolls.reviewed_through` in config.yaml):
  candidate if (|jump vs reference| > spread_jump_bp and |reference move| < reference_move_bp)
            or |jump vs reference| > new_candidate_jump_bp   (catches rolls on busy days)
            or the series restarts after a gap of gap_resume_weekdays or more
  where jump vs reference = change in bond - change in the same-tenor NZ swap.

Commands (run from the project root):
  python -m src.rolls                                  detect new candidates, adjust, chart
  python -m src.rolls --confirm 2026-11-20 nz_gov_2y   confirm a pending/unclear roll
  python -m src.rolls --reject  2026-11-20 nz_gov_2y   reject it
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.utils import get_logger, load_config, project_path, read_rates, write_rates

log = get_logger("rolls")

ROLL_COLUMNS = ["date", "series", "include", "status", "artificial_jump_bp",
                "persistent_shift_bp", "documented_change", "note"]
STATUSES = ("confirmed", "unclear", "pending", "rejected")


# =============================================================================
# Detection
# =============================================================================

def detect_rolls(rates: pd.DataFrame, reference: dict[str, str], spread_jump_bp: float,
                 reference_move_bp: float, gap_resume_weekdays: int, persistence_obs: int = 5) -> pd.DataFrame:
    """Return one row per candidate roll date.

    Changes are measured between consecutive dates on which BOTH the bond and its
    reference were published (raw values, nothing forward-filled).
    `artificial_jump_bp` is the change in the bond minus the change in the reference,
    i.e. the part of the move the market doesn't explain.
    `persistent_shift_bp` (review aid) is the average bond-reference spread over the
    `persistence_obs` observations from the date onwards minus the average over the
    same number before it. A genuine roll shifts the spread permanently; noise reverses.
    """
    rows = []
    for bond, ref in reference.items():
        pair = rates[[bond, ref]].dropna()
        if len(pair) < 2:
            continue
        d_bond = pair[bond].diff() * 100
        d_ref = pair[ref].diff() * 100
        jump = d_bond - d_ref
        prev_date = pd.Series(pair.index, index=pair.index).shift()
        gap = pd.Series(
            np.busday_count(prev_date.iloc[1:].values.astype("datetime64[D]"),
                            pair.index[1:].values.astype("datetime64[D]")) - 1,
            index=pair.index[1:],
        ).reindex(pair.index)

        is_jump = (jump.abs() > spread_jump_bp) & (d_ref.abs() < reference_move_bp)
        is_resume = gap >= gap_resume_weekdays
        spread_bp = (pair[bond] - pair[ref]) * 100
        for d in pair.index[is_jump | is_resume]:
            i = pair.index.get_loc(d)
            shift = spread_bp.iloc[i:i + persistence_obs].mean() - spread_bp.iloc[max(0, i - persistence_obs):i].mean()
            reason = " + ".join(r for r, flag in (("jump", is_jump[d]), ("gap_resume", is_resume[d])) if flag)
            rows.append({
                "date": d, "series": bond, "reference": ref, "reason": reason,
                "bond_change_bp": round(d_bond[d], 1), "reference_change_bp": round(d_ref[d], 1),
                "artificial_jump_bp": round(jump[d], 1), "persistent_shift_bp": round(shift, 1),
                "previous_date": prev_date[d], "gap_weekdays": int(gap[d]),
            })
    cols = ["date", "series", "reference", "reason", "bond_change_bp", "reference_change_bp",
            "artificial_jump_bp", "persistent_shift_bp", "previous_date", "gap_weekdays"]
    return pd.DataFrame(rows, columns=cols).sort_values(["series", "date"]).reset_index(drop=True)


def measure_jump(rates: pd.DataFrame, bond: str, ref: str, date: pd.Timestamp,
                 persistence_obs: int = 5) -> dict[str, Any]:
    """Measure the artificial jump on a given date (or the next date both series exist)."""
    pair = rates[[bond, ref]].dropna()
    i = int(pair.index.searchsorted(date))
    d = pair.index[i]
    spread_bp = (pair[bond] - pair[ref]) * 100
    jump = spread_bp.iloc[i] - spread_bp.iloc[i - 1]
    shift = spread_bp.iloc[i:i + persistence_obs].mean() - spread_bp.iloc[max(0, i - persistence_obs):i].mean()
    return {"date": d, "artificial_jump_bp": round(jump, 1), "persistent_shift_bp": round(shift, 1)}


def detect_new_candidates(rates: pd.DataFrame, roll_dates: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Candidates after `reviewed_through` that aren't in roll_dates.csv yet, as 'pending' rows."""
    r = cfg["rolls"]
    common = (r["gap_resume_weekdays"], r["persistence_obs"])
    found = pd.concat([
        detect_rolls(rates, r["nz_reference"], r["spread_jump_bp"], r["reference_move_bp"], *common),
        detect_rolls(rates, r["nz_reference"], r["new_candidate_jump_bp"], np.inf, *common),
    ]).drop_duplicates(["date", "series"])
    found = found[found["date"] > pd.Timestamp(r["reviewed_through"])]
    known = set(zip(roll_dates["date"], roll_dates["series"]))
    is_new = np.array([(d, s) not in known for d, s in zip(found["date"], found["series"])], dtype=bool)
    found = found[is_new]
    return pd.DataFrame({
        "date": found["date"], "series": found["series"], "include": "N", "status": "pending",
        "artificial_jump_bp": found["artificial_jump_bp"], "persistent_shift_bp": found["persistent_shift_bp"],
        "documented_change": "", "note": "auto-detected (" + found["reason"] + "), awaiting review",
    }, columns=ROLL_COLUMNS)


# =============================================================================
# The reviewed list and what it implies
# =============================================================================

def load_roll_dates(path: Path) -> pd.DataFrame:
    """Read roll_dates.csv (empty table if it doesn't exist yet)."""
    if not path.exists():
        return pd.DataFrame(columns=ROLL_COLUMNS)
    df = pd.read_csv(path, parse_dates=["date"], dtype={"include": str, "status": str,
                                                         "documented_change": str, "note": str})
    df[["documented_change", "note"]] = df[["documented_change", "note"]].fillna("")
    bad = set(df["status"]) - set(STATUSES)
    if bad:
        raise ValueError(f"roll_dates.csv has unknown status values: {bad}")
    return df[ROLL_COLUMNS]


def save_roll_dates(df: pd.DataFrame, path: Path) -> None:
    out = df.sort_values(["series", "date"]).copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out.to_csv(path, index=False)


def back_adjust(rates: pd.DataFrame, roll_dates: pd.DataFrame) -> pd.DataFrame:
    """Back-adjusted copies (`<series>_adj`) of every series with confirmed rolls.

    For each confirmed roll (include = Y) on date r with jump J bp, all values
    strictly before r are shifted by +J/100 percent. Unclear, pending and rejected
    rows never change history.
    """
    confirmed = roll_dates[roll_dates["include"] == "Y"]
    out = {}
    for series in sorted(roll_dates["series"].unique()):
        adj = rates[series].copy()
        for row in confirmed[confirmed["series"] == series].itertuples():
            adj[adj.index < row.date] += row.artificial_jump_bp / 100
        out[f"{series}_adj"] = adj
    return pd.DataFrame(out, index=rates.index)


def change_exclusions(index: pd.DatetimeIndex, roll_dates: pd.DataFrame,
                      statuses: tuple[str, ...] = ("confirmed", "unclear", "pending")) -> pd.DataFrame:
    """Boolean table (dates x series): True where that day's change must be excluded.

    Later steps set the daily change of a series - and of every instrument built
    from it - to NaN on these days. Default covers confirmed, unclear and pending.
    """
    rows = roll_dates[roll_dates["status"].isin(statuses)]
    mask = pd.DataFrame(False, index=index, columns=sorted(roll_dates["series"].unique()))
    for row in rows.itertuples():
        if row.date in mask.index:
            mask.loc[row.date, row.series] = True
    return mask


def pending_series(roll_dates: pd.DataFrame) -> dict[str, list[pd.Timestamp]]:
    """Series with roll candidates awaiting review -> their pending dates.

    The scanner marks instruments using these series as "pending roll review" and
    the paper book opens no new trades on them until the review is done.
    """
    pending = roll_dates[roll_dates["status"] == "pending"]
    return {s: sorted(g["date"]) for s, g in pending.groupby("series")}


def set_status(path: Path, date: str, series: str, decision: str) -> None:
    """Confirm or reject a roll in roll_dates.csv (used by the command-line options)."""
    df = load_roll_dates(path)
    sel = (df["date"] == pd.Timestamp(date)) & (df["series"] == series)
    if not sel.any():
        raise SystemExit(f"No roll_dates.csv row for {date} {series}")
    df.loc[sel, "status"] = "confirmed" if decision == "confirm" else "rejected"
    df.loc[sel, "include"] = "Y" if decision == "confirm" else "N"
    df.loc[sel, "note"] = df.loc[sel, "note"] + f" | {decision}ed by reviewer {pd.Timestamp.today().date()}"
    save_roll_dates(df, path)
    log.info("%s %s -> %s", date, series, df.loc[sel, "status"].iloc[0])


# =============================================================================
# Chart
# =============================================================================

def plot_adjustment(rates: pd.DataFrame, adjusted: pd.DataFrame, roll_dates: pd.DataFrame,
                    start: str = "2011-01-01") -> go.Figure:
    """One panel per NZ bond: raw vs back-adjusted yield, with roll dates marked."""
    series = sorted(roll_dates["series"].unique(), key=lambda s: int(s.split("_")[-1][:-1]))
    fig = make_subplots(rows=len(series), cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        subplot_titles=[f"{s}: raw vs back-adjusted (%)" for s in series])
    colours = {"confirmed": "green", "unclear": "orange", "pending": "red"}
    for i, s in enumerate(series, start=1):
        raw = rates.loc[start:, s]
        adj = adjusted.loc[start:, f"{s}_adj"]
        fig.add_trace(go.Scatter(x=raw.index, y=raw, name="raw", line=dict(color="grey", width=1),
                                 showlegend=i == 1), row=i, col=1)
        fig.add_trace(go.Scatter(x=adj.index, y=adj, name="back-adjusted", line=dict(color="royalblue", width=1),
                                 showlegend=i == 1), row=i, col=1)
        for status, colour in colours.items():
            rows = roll_dates[(roll_dates["series"] == s) & (roll_dates["status"] == status)]
            rows = rows[rows["date"] >= pd.Timestamp(start)]
            if rows.empty:
                continue
            y = rates[s].reindex(rows["date"]).ffill()
            fig.add_trace(go.Scatter(
                x=rows["date"], y=y, mode="markers", name=f"roll: {status}",
                marker=dict(color=colour, size=8, symbol="triangle-down"), showlegend=i == 1,
                text=[f"{d.date()} {status}: {j:+.1f}bp" for d, j in zip(rows["date"], rows["artificial_jump_bp"])],
                hoverinfo="text"), row=i, col=1)
    fig.update_layout(height=300 * len(series), width=1100, template="plotly_white",
                      legend=dict(orientation="h", y=1.04), margin=dict(t=70, l=50, r=20, b=30))
    return fig


# =============================================================================
# Main run
# =============================================================================

def run(cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    """Add new pending candidates, print items needing review, save adjusted series and chart."""
    cfg = cfg or load_config()
    paths = {k: project_path(v) for k, v in cfg["paths"].items()}
    rates = read_rates(paths["rates_file"])
    if rates is None:
        raise FileNotFoundError("rates.csv not found - run `python -m src.fetch` first")
    roll_dates = load_roll_dates(paths["roll_dates_file"])
    if roll_dates.empty:
        raise FileNotFoundError("roll_dates.csv not found - it is created from the reviewed candidate list")

    new = detect_new_candidates(rates, roll_dates, cfg)
    if len(new):
        roll_dates = pd.concat([roll_dates, new], ignore_index=True)
        save_roll_dates(roll_dates, paths["roll_dates_file"])
        for row in new.itertuples():
            log.warning("NEW roll candidate pending review: %s %s jump %+.1fbp", row.date.date(), row.series,
                        row.artificial_jump_bp)

    with pd.option_context("display.width", 250, "display.max_colwidth", 90):
        for status in ("pending", "unclear"):
            rows = roll_dates[roll_dates["status"] == status]
            print(f"\n=== {status.upper()} rolls ({len(rows)}) ===")
            if len(rows):
                print(rows.assign(date=rows["date"].dt.date).drop(columns="status").to_string(index=False))
        counts = roll_dates["status"].value_counts().to_dict()
        print(f"\nroll_dates.csv status counts: {counts}\n")

    adjusted = back_adjust(rates, roll_dates)
    write_rates(adjusted, paths["rates_adjusted_file"])

    charts = paths["charts_dir"]
    charts.mkdir(parents=True, exist_ok=True)
    fig = plot_adjustment(rates, adjusted, roll_dates)
    fig.write_html(charts / "nz_gov_roll_adjustment.html", include_plotlyjs="cdn")
    try:
        fig.write_image(charts / "nz_gov_roll_adjustment.png", scale=1.5)
    except Exception as e:  # static export needs kaleido + Chrome; the HTML chart is still saved
        log.warning("PNG export skipped (%s)", e)
    log.info("Saved %s and charts in %s", paths["rates_adjusted_file"].name, charts)
    return roll_dates


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark-roll review tools")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--confirm", nargs=2, metavar=("DATE", "SERIES"))
    group.add_argument("--reject", nargs=2, metavar=("DATE", "SERIES"))
    args = parser.parse_args()
    cfg = load_config()
    path = project_path(cfg["paths"]["roll_dates_file"])
    if args.confirm:
        set_status(path, *args.confirm, "confirm")
    elif args.reject:
        set_status(path, *args.reject, "reject")
    run(cfg)


if __name__ == "__main__":
    main()
