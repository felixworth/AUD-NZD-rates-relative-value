"""
validate.py - data-quality checks run on every update of rates.csv.

Each issue becomes one row of a report with a severity:
  error   - the data can't be trusted (e.g. a 2,500% yield = parsing problem).
            fetch.py will then refuse to save anything.
  warning - worth a human look (big move, stale value, missing dates, history
            revision, out-of-date source) - logged, but the run continues.
  info    - the same kinds of issue, but older than the "recent" window, so they
            go to the report file without cluttering the log on every run.

All checks look at the RAW stored values (no forward-filling).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.utils import get_logger

log = get_logger("validate")

COLUMNS = ["severity", "check", "series", "date", "detail"]


def _market(series: str) -> str:
    return series.split("_")[0]          # "au_gov_10y" -> "au"


def check_range(rates: pd.DataFrame, low: float, high: float) -> list[dict]:
    """Values outside [low, high] percent are almost certainly parsing errors."""
    issues = []
    for col in rates.columns:
        bad = rates[col][(rates[col] < low) | (rates[col] > high)]
        for d, v in bad.items():
            issues.append({"severity": "error", "check": "implausible_value", "series": col,
                           "date": d, "detail": f"{v:.4f}% outside [{low}, {high}]"})
    return issues


def check_big_moves(rates: pd.DataFrame, max_bp: float) -> list[dict]:
    """Day-on-day moves larger than max_bp (between consecutive published values)."""
    issues = []
    for col in rates.columns:
        s = rates[col].dropna()
        move_bp = s.diff() * 100
        for d, m in move_bp[move_bp.abs() > max_bp].items():
            issues.append({"severity": "warning", "check": "big_move", "series": col, "date": d,
                           "detail": f"{m:+.1f}bp vs previous value"})
    return issues


def check_stale(rates: pd.DataFrame, min_obs: int) -> list[dict]:
    """Runs of at least `min_obs` identical consecutive published values."""
    issues = []
    for col in rates.columns:
        s = rates[col].dropna().round(6)
        run_id = (s != s.shift()).cumsum()
        for _, run in s.groupby(run_id):
            if len(run) >= min_obs:
                issues.append({"severity": "warning", "check": "stale_value", "series": col,
                               "date": run.index[-1],
                               "detail": f"{run.iloc[0]:.4f} unchanged for {len(run)} obs since {run.index[0].date()}"})
    return issues


def check_gaps(rates: pd.DataFrame, max_gap: int) -> list[dict]:
    """Runs of more than `max_gap` consecutive missing weekdays inside each series' history.

    Short gaps (public holidays) are bridged later by the 2-day forward-fill, so
    only the longer gaps that will remain empty are reported.
    """
    issues = []
    for col in rates.columns:
        s = rates[col].dropna()
        if s.empty:
            continue
        weekdays = pd.bdate_range(s.index[0], s.index[-1])
        missing = weekdays.difference(s.index)
        if missing.empty:
            continue
        # group consecutive missing weekdays into runs
        pos = weekdays.get_indexer(missing)
        run_id = np.cumsum(np.diff(pos, prepend=pos[0] - 1) != 1)
        for rid in np.unique(run_id):
            run = missing[run_id == rid]
            if len(run) > max_gap:
                issues.append({"severity": "warning", "check": "missing_dates", "series": col,
                               "date": run[-1],
                               "detail": f"{len(run)} weekdays missing {run[0].date()} to {run[-1].date()}"})
    return issues


def check_freshness(rates: pd.DataFrame, today: pd.Timestamp, max_age: dict[str, int]) -> list[dict]:
    """Warn if a market's newest published value is older than allowed."""
    issues = []
    for market, limit in max_age.items():
        cols = [c for c in rates.columns if _market(c) == market]
        lasts = [rates[c].last_valid_index() for c in cols if rates[c].last_valid_index() is not None]
        if not lasts:
            continue
        latest = max(lasts)
        age = (today - latest).days
        if age > limit:
            issues.append({"severity": "warning", "check": "stale_source", "series": market,
                           "date": latest, "detail": f"latest {market.upper()} data is {age} days old (limit {limit})"})
    return issues


def check_revisions(events: pd.DataFrame | None) -> list[dict]:
    """Older history that differs from the source (kept unchanged, but reported)."""
    if events is None or events.empty:
        return []
    conflicts = events[events["action"] == "conflict_kept_old"]
    return [{"severity": "warning", "check": "history_revision", "series": r.series, "date": r.date,
             "detail": f"stored {r.old:.4f}, source now {r.new:.4f} (kept stored value)"}
            for r in conflicts.itertuples()]


def validate(rates: pd.DataFrame, cfg: dict[str, Any], today: pd.Timestamp,
             events: pd.DataFrame | None = None) -> pd.DataFrame:
    """Run every check and return the report (also logs recent warnings)."""
    v = cfg["validation"]
    issues = (
        check_range(rates, *v["plausible_range_pct"])
        + check_big_moves(rates, v["max_daily_move_bp"])
        + check_stale(rates, v["stale_obs"])
        + check_gaps(rates, v["max_gap_weekdays"])
        + check_freshness(rates, today, v["max_age_days"])
        + check_revisions(events)
    )
    report = pd.DataFrame(issues, columns=COLUMNS)
    if report.empty:
        log.info("Validation: no issues")
        return report

    report["date"] = pd.to_datetime(report["date"])
    old = (report["severity"] == "warning") & (report["date"] < today - pd.Timedelta(days=v["recent_days"]))
    old &= report["check"] != "history_revision"   # revisions are always worth a warning
    report.loc[old, "severity"] = "info"
    report = report.sort_values(["severity", "date"], ascending=[True, False]).reset_index(drop=True)

    for r in report[report["severity"] != "info"].itertuples():
        level = log.error if r.severity == "error" else log.warning
        level("%s | %s | %s | %s", r.check, r.series, r.date.date(), r.detail)
    summary = report.groupby(["severity", "check"]).size().to_dict()
    log.info("Validation summary: %s", summary)
    report["date"] = report["date"].dt.strftime("%Y-%m-%d")
    return report
