"""Tests for src/fetch.py and src/validate.py - parsing, upsert rules, forward-fill, validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from openpyxl import Workbook

from src.fetch import (SourceError, forward_fill_interior, parse_rba_csv, parse_rbnz_file, upsert)
from src.utils import load_config
from src.validate import check_big_moves, check_gaps, check_stale, validate

CFG = load_config()


# ---------------------------------------------------------------------------
# Small fake source files with the same layout as the real ones
# ---------------------------------------------------------------------------

def write_rba_csv(path: Path, ten_year_title: str = "Australian Government 10 year bond") -> Path:
    titles = ["Australian Government 2 year bond", "Australian Government 3 year bond",
              "Australian Government 5 year bond", ten_year_title, "Australian Government Indexed Bond"]
    lines = [
        "F2 CAPITAL MARKET YIELDS – GOVERNMENT BONDS",
        "Title," + ",".join(titles),
        "Units," + ",".join(["Per cent per annum"] * 5),
        "", "",
        "Publication date," + ",".join(["30-Oct-2026"] * 5),
        "Series ID,FCMYGBAG2D,FCMYGBAG3D,FCMYGBAG5D,FCMYGBAG10D,FCMYGBAGID",
        "27-Oct-2026,4.883,4.852,4.882,5.152,2.806",
        "28-Oct-2026,4.951,,4.946,5.208,2.841",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_rbnz_xlsx(path: Path, swap_group: str = "Swap rates close") -> Path:
    series = [
        ("Secondary market government bond closing yields", "2 year", "INM.DG102.NZZCF"),
        ("Secondary market government bond closing yields", "5 year", "INM.DG105.NZZCF"),
        ("Secondary market government bond closing yields", "10 year", "INM.DG110.NZZCF"),
        (swap_group, "2 year", "INM.DS02.NZZC"),
        (swap_group, "5 year", "INM.DS05.NZZC"),
        (swap_group, "10 year", "INM.DS10.NZZC"),
    ]
    wb = Workbook()
    ws = wb.active
    ws.title = "Data"
    ws.append([None] + [g for g, _, _ in series])
    ws.append([None] + [n for _, n, _ in series])
    ws.append(["Notes"] + [None] * 6)
    ws.append(["Unit"] + ["%pa"] * 6)
    ws.append(["Series Id"] + [i for _, _, i in series])
    ws.append([pd.Timestamp("2026-10-30").to_pydatetime(), 3.72, 4.28, 4.90, 3.61, 4.02, 4.51])
    ws.append([pd.Timestamp("2026-11-02").to_pydatetime(), 3.73, 4.29, 4.91, 3.61, 4.02, 4.52])
    desc = wb.create_sheet("Table Description")
    desc.append(["Published Date", pd.Timestamp("2026-11-03").to_pydatetime()])
    wb.save(path)
    return path


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def test_parse_rba_csv_reads_nominal_series(tmp_path):
    data, meta, published = parse_rba_csv(write_rba_csv(tmp_path / "f2.csv"), CFG["rba"]["series"])
    assert list(data.columns) == ["au_gov_2y", "au_gov_3y", "au_gov_5y", "au_gov_10y"]
    assert data.loc["2026-10-28", "au_gov_10y"] == pytest.approx(5.208)
    assert np.isnan(data.loc["2026-10-28", "au_gov_3y"])          # blank stays blank
    assert published == "2026-10-30"
    assert all("Indexed" not in m["original_name"] for m in meta)


def test_parse_rba_csv_rejects_wrong_series(tmp_path):
    path = write_rba_csv(tmp_path / "f2.csv", ten_year_title="Australian Government Indexed Bond")
    with pytest.raises(SourceError):
        parse_rba_csv(path, CFG["rba"]["series"])


def test_parse_rbnz_xlsx(tmp_path):
    data, meta, published = parse_rbnz_file(write_rbnz_xlsx(tmp_path / "hb2.xlsx"), CFG["rbnz"]["series"])
    assert data.loc["2026-11-02", "nz_swap_10y"] == pytest.approx(4.52)
    assert data.loc["2026-11-02", "nz_gov_2y"] == pytest.approx(3.73)
    assert published == "2026-11-03"


def test_parse_rbnz_rejects_swap_column_that_is_not_a_swap(tmp_path):
    path = write_rbnz_xlsx(tmp_path / "hb2.xlsx", swap_group="Bank bill yields")
    with pytest.raises(SourceError):
        parse_rbnz_file(path, CFG["rbnz"]["series"])


# ---------------------------------------------------------------------------
# Upsert rules
# ---------------------------------------------------------------------------

def frame(values: dict[str, list], dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame(values, index=pd.DatetimeIndex(pd.to_datetime(dates), name="date"))


def test_upsert_rules():
    today = pd.Timestamp("2026-11-03")
    existing = frame({"x": [1.00, 2.00, np.nan]}, ["2025-01-02", "2026-10-28", "2026-10-29"])
    incoming = frame({"x": [1.50, 2.10, 3.00, 4.00]}, ["2025-01-02", "2026-10-28", "2026-10-29", "2026-10-30"])
    merged, events = upsert(existing, incoming, today, window_days=30)

    assert merged.loc["2025-01-02", "x"] == 1.00    # old history never changed ...
    assert merged.loc["2026-10-28", "x"] == 2.10    # ... recent values can be revised
    assert merged.loc["2026-10-29", "x"] == 3.00    # missing value filled
    assert merged.loc["2026-10-30", "x"] == 4.00    # new date added
    assert set(events["action"]) == {"conflict_kept_old", "revised_recent", "filled"}


def test_upsert_keeps_values_the_source_dropped():
    today = pd.Timestamp("2026-11-03")
    existing = frame({"x": [2.0]}, ["2026-10-28"])
    incoming = frame({"x": [np.nan]}, ["2026-10-28"])
    merged, _ = upsert(existing, incoming, today, window_days=30)
    assert merged.loc["2026-10-28", "x"] == 2.0


def test_upsert_is_idempotent():
    today = pd.Timestamp("2026-11-03")
    data = frame({"x": [1.0, 2.0]}, ["2026-10-27", "2026-10-28"])
    merged, events = upsert(data, data, today, window_days=30)
    pd.testing.assert_frame_equal(merged, data)
    assert events.empty


# ---------------------------------------------------------------------------
# Forward-fill: interior only, never past a series' last real observation
# ---------------------------------------------------------------------------

def test_ffill_never_extends_past_last_observation():
    dates = pd.bdate_range("2026-10-26", periods=6)
    df = pd.DataFrame({"au": [1.0, 1.1, 1.2, np.nan, np.nan, np.nan],   # RBA not yet published
                       "nz": [2.0, np.nan, 2.2, 2.3, 2.4, 2.5]}, index=dates)
    filled, gaps = forward_fill_interior(df, limit=2)
    assert filled["au"].iloc[3:].isna().all()                # not extended
    assert filled["nz"].iloc[1] == 2.0                        # interior gap filled
    assert not gaps.any().any()


def test_ffill_limit_and_remaining_gaps():
    dates = pd.bdate_range("2026-10-01", periods=6)
    df = pd.DataFrame({"x": [1.0, np.nan, np.nan, np.nan, np.nan, 2.0]}, index=dates)
    filled, gaps = forward_fill_interior(df, limit=2)
    assert filled["x"].iloc[1:3].tolist() == [1.0, 1.0]      # at most 2 filled
    assert filled["x"].iloc[3:5].isna().all()
    assert gaps["x"].sum() == 2                               # flagged


def test_ffill_uses_no_future_data():
    """Filled values must come from the past: changing the future can't change the present."""
    dates = pd.bdate_range("2026-10-01", periods=8)
    base = pd.DataFrame({"x": [1.0, np.nan, 1.2, np.nan, np.nan, 1.5, np.nan, 1.7]}, index=dates)
    altered = base.copy()
    altered.iloc[5:, 0] = [9.0, np.nan, 9.9]
    f1, _ = forward_fill_interior(base, 2)
    f2, _ = forward_fill_interior(altered, 2)
    pd.testing.assert_frame_equal(f1.iloc[:5], f2.iloc[:5])


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_validation_checks():
    dates = pd.bdate_range("2026-09-01", periods=12)
    values = [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.30, 4.31, np.nan, np.nan, np.nan, 4.32]
    df = pd.DataFrame({"nz_swap_2y": values}, index=dates)
    assert len(check_big_moves(df, 25)) == 1        # +30bp
    assert len(check_stale(df, 5)) == 1             # 6 identical values
    assert len(check_gaps(df, 2)) == 1              # 3 missing weekdays


def test_implausible_value_is_an_error():
    df = pd.DataFrame({"au_gov_10y": [4.0, 420.0]}, index=pd.bdate_range("2026-10-28", periods=2))
    report = validate(df, CFG, pd.Timestamp("2026-11-03"))
    assert (report["severity"] == "error").any()
