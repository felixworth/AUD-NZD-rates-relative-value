"""Tests for robustness helpers and the carry-only entry region."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest import simulate
from src.robustness import bootstrap_mean_ci, mean_tstat, split_by_direction


def test_split_by_direction_counts():
    trades = pd.DataFrame({
        "instrument": ["a", "a", "b"],
        "side": ["long", "short", "long"],
        "sample": ["IS", "IS", "OOS"],
        "pnl_bp": [2.0, -1.0, 4.0],
    })
    s = split_by_direction(trades)
    is_a_long = s[(s["instrument"] == "a") & (s["side"] == "long") & (s["sample"] == "IS")].iloc[0]
    assert is_a_long["trades"] == 1 and is_a_long["avg_bp"] == pytest.approx(2.0)
    is_all = s[(s["instrument"] == "ALL") & (s["side"] == "ALL") & (s["sample"] == "IS")].iloc[0]
    assert is_all["trades"] == 2 and is_all["hit_rate"] == pytest.approx(50.0)


def test_mean_tstat_and_bootstrap_ci_on_known_sample():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mean, t, p = mean_tstat(x)
    assert mean == pytest.approx(3.0)
    assert t > 0 and p < 0.05
    lo, hi = bootstrap_mean_ci(x, draws=2000, seed=4)
    assert lo < 3.0 < hi


def _frame(z, carry):
    n = len(z)
    dates = pd.bdate_range("2019-01-02", periods=n)
    d = pd.DataFrame({
        "value_bp": np.arange(n, dtype=float), "z_250": z, "inputs_filled": False, "roll_day": False,
        "post_roll_block": False, "pending_roll": False, "insufficient_250": False,
    }, index=dates)
    residual = {
        "beta": pd.Series(0.5, index=dates),
        "long_bp": pd.Series(np.arange(n, dtype=float) + 50, index=dates),
        "short_bp": pd.Series(np.arange(n, dtype=float), index=dates),
        "long_legs": {"au_gov_10y": 1.0}, "short_legs": {"au_gov_2y": 1.0},
        "carry_long": pd.Series(carry, index=dates, dtype=float),
    }
    spec = {"group": "AU bond residual", "pca": False, "legs": {}, "belly": None, "wings": None}
    rules = {"entry_z": 2.0, "exit_z": 0.5, "stop_z": 3.5, "max_hold_days": 60,
             "is_end": pd.Timestamp("2020-12-31")}
    return d, spec, residual, rules


def test_carry_only_enters_unstretched_carry_positive_direction():
    n = 30
    z, carry = np.zeros(n), np.full(n, 5.0)
    z[5] = 1.5
    z[6:20] = 1.2
    d, spec, residual, rules = _frame(z, carry)
    trades, _ = simulate(d, spec, "au_gov_10s2s_resid", rules, {"swap_bp": 0, "bond_bp": 0},
                         0.04, 10_000, {}, residual=residual, carry_only=True)
    assert len(trades) == 1 and trades.iloc[0]["side"] == "long"


def test_carry_only_does_not_enter_when_stretched():
    n = 30
    z, carry = np.zeros(n), np.full(n, 5.0)
    z[5] = 2.5
    z[6:20] = 2.2
    d, spec, residual, rules = _frame(z, carry)
    trades, _ = simulate(d, spec, "au_gov_10s2s_resid", rules, {"swap_bp": 0, "bond_bp": 0},
                         0.04, 10_000, {}, residual=residual, carry_only=True)
    assert trades.empty


def test_carry_only_shorts_when_long_carry_is_negative():
    n = 30
    z, carry = np.zeros(n), np.full(n, -5.0)
    z[5] = 1.5
    z[6:20] = 1.2
    d, spec, residual, rules = _frame(z, carry)
    trades, _ = simulate(d, spec, "au_gov_10s2s_resid", rules, {"swap_bp": 0, "bond_bp": 0},
                         0.04, 10_000, {}, residual=residual, carry_only=True)
    assert len(trades) == 1 and trades.iloc[0]["side"] == "short"
