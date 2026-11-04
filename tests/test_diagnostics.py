"""Tests for src/diagnostics.py - the event-window check around RBA methodology changes."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.diagnostics import event_window_check


def synthetic(jump_on: str | None = None, jump: float = 0.10) -> pd.DataFrame:
    """AU yields move exactly with NZ swaps, plus an optional unexplained jump on one date."""
    dates = pd.bdate_range("2023-03-01", periods=40)
    rng = np.random.default_rng(1)
    swap = 4 + np.cumsum(rng.choice([-0.02, 0.0, 0.02], size=len(dates)))
    df = pd.DataFrame({c: swap for c in ["nz_swap_2y", "nz_swap_5y", "nz_swap_10y"]}, index=dates)
    for tenor in ["2y", "3y", "5y", "10y"]:
        df[f"au_gov_{tenor}"] = swap - 1
    if jump_on:
        df.loc[pd.Timestamp(jump_on):, ["au_gov_2y", "au_gov_3y", "au_gov_5y", "au_gov_10y"]] += jump
    return df


def test_no_unexplained_moves_no_flags():
    t = event_window_check(synthetic(), "2023-03-31")
    assert (t["flag"] == "").all()
    assert len(t[t["tenor"] == "10y"]) == 20          # 10 before + 10 from the event


def test_unexplained_jump_on_event_day_is_flagged():
    t = event_window_check(synthetic(jump_on="2023-03-31"), "2023-03-31")
    flagged = t[t["flag"] == "FLAG"]
    assert set(flagged.index) == {pd.Timestamp("2023-03-31")}
    assert set(flagged["tenor"]) == {"2y", "3y", "5y", "10y"}
