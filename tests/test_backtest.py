"""Tests for src/backtest.py - next-day fill, blocked entries, DV01 neutrality."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest import (book_scale, net_dv01, one_way_cost_bp, par_mod_duration, simulate,
                          tenor_years)

SPEC = {"group": "AU bond curve", "legs": {"au_gov_10y": 1, "au_gov_2y": -1},
        "pca": False, "belly": None, "wings": None}
RULES = {"entry_z": 2.0, "exit_z": 0.5, "stop_z": 3.5, "max_hold_days": 60,
         "is_end": pd.Timestamp("2020-12-31")}
COSTS = {"swap_bp": 0.5, "bond_bp": 1.0}


def frame(n: int = 40, z=None, value=None, **flags) -> pd.DataFrame:
    dates = pd.bdate_range("2019-01-02", periods=n)
    z = np.zeros(n) if z is None else np.asarray(z, dtype=float)
    value = np.arange(n, dtype=float) if value is None else np.asarray(value, dtype=float)
    d = pd.DataFrame({
        "value_bp": value, "z_250": z, "inputs_filled": False, "roll_day": False,
        "post_roll_block": False, "pending_roll": False, "insufficient_250": False,
    }, index=dates)
    for k, v in flags.items():
        d[k] = v
    return d


def run(d: pd.DataFrame, spec=SPEC, costs=None):
    costs = costs or {"swap_bp": 0.0, "bond_bp": 0.0}   # zero costs unless the test is about costs
    return simulate(d, spec, "au_gov_2s10s", RULES, costs, 0.04, 10_000, {}, clean=None)


def test_par_duration_and_tenor():
    assert tenor_years("nz_swap_10y") == 10
    assert par_mod_duration(10, 0.04) == pytest.approx((1 - 1.04 ** -10) / 0.04)


def test_dv01_neutral_curve_has_zero_net_and_10k_per_bp_of_spread():
    legs = {"au_gov_10y": 1.0, "au_gov_2y": -1.0}
    k = book_scale(legs, 10_000)
    assert net_dv01(legs, k) == pytest.approx(0.0)
    assert k == pytest.approx(10_000)          # long weight = 1, so $10k per bp of 10y-2y
    # a 1bp parallel move of both legs is worth k * (1 - 1) = $0
    assert k * (1 * 1 + (-1) * 1) == pytest.approx(0.0)
    # a 1bp steepener (10y +1bp, 2y unchanged) is worth $10k
    assert k * (1 * 1 + (-1) * 0) == pytest.approx(10_000)


def test_dv01_neutral_fly_weights_sum_to_zero():
    legs = {"au_gov_5y": 2.0, "au_gov_2y": -1.0, "au_gov_10y": -1.0}
    k = book_scale(legs, 10_000)
    assert net_dv01(legs, k) == pytest.approx(0.0)
    assert k == pytest.approx(5_000)           # long weight = 2, $5k per bp of the 2x5y fly


def test_swap_vs_bond_costs():
    curve = {"nz_swap_10y": 1.0, "nz_swap_2y": -1.0}
    xccy = {"au_gov_10y": 1.0, "nz_gov_10y": -1.0}
    assert one_way_cost_bp(curve, COSTS) == pytest.approx(1.0)    # 0.5 + 0.5
    assert one_way_cost_bp(xccy, COSTS) == pytest.approx(2.0)     # 1 + 1


def test_signal_at_t_executes_at_t_plus_1():
    z = np.zeros(30)
    z[5] = 2.5                          # signal short at close t
    z[6:12] = 1.2
    z[12] = 0.2                         # target at close of day 12 -> exit next day
    d = frame(30, z=z, value=np.arange(30, dtype=float) * 1.0)
    trades, daily = run(d)
    assert len(trades) == 1
    t = trades.iloc[0]
    assert t["side"] == "short"
    assert t["entry_date"] == d.index[6]           # t+1
    assert t["exit_date"] == d.index[13]           # signal on 12, trade on 13
    # first PnL day is the close after entry (day 7 vs entry at 6)
    assert daily["pnl_bp"].iloc[6] == pytest.approx(0.0)
    assert daily["pnl_bp"].iloc[7] == pytest.approx(-1.0)   # short, value rose 1


def test_no_entry_on_filled_insufficient_post_roll_or_pending():
    z = np.zeros(20)
    z[:] = 2.5                          # always a signal if it were allowed
    for flag in ("inputs_filled", "insufficient_250", "post_roll_block", "pending_roll"):
        d = frame(20, z=z, **{flag: True})
        trades, _ = run(d)
        assert trades.empty, flag


def test_roll_day_change_is_excluded_and_flattens():
    z = np.zeros(20)
    z[2] = 2.5
    z[3:8] = 1.0
    value = np.array([0, 0, 0, 10, 11, 50, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12], dtype=float)
    roll = [False] * 20
    roll[5] = True                      # jump +39 is a roll, must not be in PnL
    d = frame(20, z=z, value=value, roll_day=roll)
    trades, daily = run(d)
    assert len(trades) == 1
    assert trades.iloc[0]["reason"] == "roll"
    assert trades.iloc[0]["exit_date"] == d.index[5]
    # PnL is short from entry at day 3 (value 10) flattened at day 4's close (11), not 50
    assert trades.iloc[0]["pnl_bp"] == pytest.approx(-1.0)   # 10 -> 11 against a short
    assert daily["pnl_bp"].iloc[5] == pytest.approx(0.0)     # roll-day change not booked
