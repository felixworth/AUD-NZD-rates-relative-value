"""Tests for src/rolls.py - benchmark-roll detection on synthetic data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.rolls import (ROLL_COLUMNS, back_adjust, change_exclusions, detect_new_candidates, detect_rolls,
                       pending_series)
from src.utils import load_config


def synthetic(roll_at: int = 30, jump: float = 0.20, n: int = 60) -> pd.DataFrame:
    """Swap follows a gentle random walk; the bond = swap + 10bp, plus a permanent jump at `roll_at`."""
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2024-01-01", periods=n)
    swap = 4 + np.cumsum(rng.choice([-0.01, 0.0, 0.01], size=n))
    bond = swap + 0.10
    bond[roll_at:] += jump
    return pd.DataFrame({"nz_gov_2y": bond.round(2), "nz_swap_2y": swap.round(2)}, index=dates)


def test_detects_a_roll():
    df = synthetic()
    c = detect_rolls(df, {"nz_gov_2y": "nz_swap_2y"}, 5, 2, 5)
    assert list(c["date"]) == [df.index[30]]
    assert c["artificial_jump_bp"].iloc[0] == 20.0
    assert c["persistent_shift_bp"].iloc[0] == 20.0


def test_no_roll_no_candidate():
    df = synthetic(jump=0.0)
    assert detect_rolls(df, {"nz_gov_2y": "nz_swap_2y"}, 5, 2, 5).empty


def test_gap_resume_is_a_candidate():
    df = synthetic(jump=0.0)
    df.iloc[10:20, 0] = np.nan            # bond missing for 10 weekdays
    c = detect_rolls(df, {"nz_gov_2y": "nz_swap_2y"}, 5, 2, 5)
    assert "gap_resume" in c["reason"].iloc[0]


def roll_table(rows: list[tuple[str, str, str, float]]) -> pd.DataFrame:
    """rows: (date, status, include, jump_bp) for nz_gov_2y."""
    return pd.DataFrame([{"date": pd.Timestamp(d), "series": "nz_gov_2y", "include": inc, "status": st,
                          "artificial_jump_bp": j, "persistent_shift_bp": j, "documented_change": "", "note": ""}
                         for d, st, inc, j in rows], columns=ROLL_COLUMNS)


def test_back_adjust_only_confirmed_rolls():
    df = synthetic(roll_at=30, jump=0.20)
    d_roll, d_other = df.index[30], df.index[45]
    rolls = roll_table([(str(d_roll.date()), "confirmed", "Y", 20.0),
                        (str(d_other.date()), "unclear", "N", 15.0)])
    adj = back_adjust(df, rolls)["nz_gov_2y_adj"]
    spread = (adj - df["nz_swap_2y"]).round(6)
    assert spread.nunique() == 1                        # confirmed jump removed, unclear one not applied
    assert (adj.iloc[30:] == df["nz_gov_2y"].iloc[30:]).all()   # values from the roll onwards unchanged


def test_change_exclusions_cover_unclear_and_pending_not_rejected():
    df = synthetic()
    idx = df.index
    rolls = roll_table([(str(idx[10].date()), "confirmed", "Y", 5), (str(idx[20].date()), "unclear", "N", 5),
                        (str(idx[30].date()), "pending", "N", 5), (str(idx[40].date()), "rejected", "N", 5)])
    mask = change_exclusions(idx, rolls)["nz_gov_2y"]
    assert list(mask[mask].index) == [idx[10], idx[20], idx[30]]
    assert pending_series(rolls) == {"nz_gov_2y": [idx[30]]}


def test_new_candidates_only_after_review_date_and_not_duplicated():
    cfg = load_config()
    df = synthetic(roll_at=30, jump=0.20)
    df["nz_gov_5y"] = df["nz_swap_5y"] = df["nz_gov_10y"] = df["nz_swap_10y"] = 4.0
    cfg["rolls"]["reviewed_through"] = str(df.index[20].date())
    new = detect_new_candidates(df, roll_table([]), cfg)
    assert list(new["date"]) == [df.index[30]] and new["status"].iloc[0] == "pending"
    assert detect_new_candidates(df, new, cfg).empty            # already listed -> not added again
    cfg["rolls"]["reviewed_through"] = str(df.index[40].date())
    assert detect_new_candidates(df, roll_table([]), cfg).empty  # reviewed period -> ignored


def test_back_adjustment_does_not_change_past_zscores():
    """Shifting all history before a roll by a constant leaves rolling z-scores before the roll unchanged."""
    df = synthetic(roll_at=40, jump=0.25)
    rolls = roll_table([(str(df.index[40].date()), "confirmed", "Y", 25.0)])
    adj = back_adjust(df, rolls)["nz_gov_2y_adj"]

    def z(s: pd.Series) -> pd.Series:
        return (s - s.rolling(10).mean()) / s.rolling(10).std()

    pd.testing.assert_series_equal(z(adj).iloc[:40], z(df["nz_gov_2y"]).iloc[:40], check_names=False)
