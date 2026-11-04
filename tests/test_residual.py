"""Tests for src/residual.py - no look-ahead beta, frozen-beta z-scores, carry filter."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest import net_dv01, simulate
from src.residual import (carry_bp, combo_changes, frozen_residual_z, half_life,
                          interpolate_yield, ols_beta, rolldown_bp, rolling_beta)

SCAN = {"windows": [60, 250], "min_real_fraction": 0.8, "max_gap_days": 10, "percentile_min_obs": 250}


def test_ols_beta_recovers_known_slope():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 400)
    y = 0.7 * x + rng.normal(0, 0.05, 400)
    assert ols_beta(x, y) == pytest.approx(0.7, abs=0.02)


def test_rolling_beta_has_no_lookahead():
    dates = pd.bdate_range("2014-01-02", periods=400)
    rng = np.random.default_rng(1)
    short = pd.Series(rng.normal(0, 1, 400), index=dates)
    long = 0.6 * short + rng.normal(0, 0.1, 400)
    b_full = rolling_beta(long, short, 250, 0.8, 10)
    b_cut = rolling_beta(long.iloc[:-40], short.iloc[:-40], 250, 0.8, 10)
    common = b_cut.index
    pd.testing.assert_series_equal(b_full.loc[common], b_cut, check_names=False, atol=1e-10)
    short_alt, long_alt = short.copy(), long.copy()
    short_alt.iloc[-40:] = 9.0
    long_alt.iloc[-40:] = 9.0
    b_alt = rolling_beta(long_alt, short_alt, 250, 0.8, 10)
    pd.testing.assert_series_equal(b_alt.loc[common], b_cut, check_names=False, atol=1e-10)


def test_filled_and_roll_days_are_dropped_from_the_fit():
    dates = pd.bdate_range("2020-01-01", periods=20)
    clean = pd.DataFrame({"au_gov_10y": np.linspace(4, 5, 20), "au_gov_2y": np.linspace(3, 3.5, 20)}, index=dates)
    raw = clean.copy()
    raw.iloc[5, 0] = np.nan
    chg = combo_changes(clean, raw, {"au_gov_10y": 1}, roll_days=pd.DatetimeIndex([dates[8]]))
    assert np.isnan(chg.iloc[5]) and np.isnan(chg.iloc[8])
    assert np.isfinite(chg.iloc[6])


def test_frozen_zscore_uses_date_t_beta_on_the_whole_window():
    n = 300
    dates = pd.bdate_range("2020-01-01", periods=n)
    rng = np.random.default_rng(2)
    long_bp = pd.Series(100 + rng.normal(0, 2, n).cumsum(), index=dates)
    short_bp = pd.Series(40 + rng.normal(0, 2, n).cumsum(), index=dates)
    fill = pd.Series(False, index=dates)
    beta_t = 0.55
    beta = pd.Series(np.linspace(0.2, beta_t, n), index=dates)
    z = frozen_residual_z(long_bp, short_bp, beta.iloc[[-1]], fill, SCAN)
    rebuilt = long_bp - beta_t * short_bp
    window = rebuilt.iloc[-250:]
    assert z["z_250"].iloc[-1] == pytest.approx((window.iloc[-1] - window.mean()) / window.std())
    stitched = long_bp - beta * short_bp
    assert z["mean_250"].iloc[-1] != pytest.approx(stitched.iloc[-250:].mean())


def test_interpolate_clamps_and_rolls_down():
    tenors, ylds = np.array([2.0, 5.0, 10.0]), np.array([3.0, 4.0, 5.0])
    assert interpolate_yield(tenors, ylds, 9.75) == pytest.approx(4.95)
    assert interpolate_yield(tenors, ylds, 1.75) == pytest.approx(3.0)   # clamp to shortest
    row = pd.Series({"au_gov_2y": 3.0, "au_gov_5y": 4.0, "au_gov_10y": 5.0})
    rd10 = rolldown_bp("au_gov_10y", row, ["au_gov_2y", "au_gov_5y", "au_gov_10y"], 0.25)
    assert rd10 == pytest.approx((5.0 - 4.95) * 100)   # long 10y gains as it rolls to 9.75y


def test_carry_positive_when_yield_is_above_funding():
    row = pd.Series({"au_gov_10y": 5.0, "au_gov_2y": 3.0})
    c = carry_bp("au_gov_10y", row, funding_y=3.0, horizon=0.25)
    assert c > 0


def test_half_life_of_ar1():
    phi, n = 0.9, 2000
    rng = np.random.default_rng(0)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.normal(0, 1)
    hl = half_life(pd.Series(x))
    expected = np.log(2) / -np.log(phi)
    assert hl == pytest.approx(expected, rel=0.15)


def test_carry_filter_blocks_fade_against_carry():
    """z high (fade = short) but long-residual carry is positive => short has negative carry => no entry."""
    n = 40
    dates = pd.bdate_range("2019-01-02", periods=n)
    z = np.zeros(n)
    z[5] = 2.5
    z[6:20] = 1.0
    d = pd.DataFrame({
        "value_bp": np.arange(n, dtype=float), "z_250": z, "inputs_filled": False, "roll_day": False,
        "post_roll_block": False, "pending_roll": False, "insufficient_250": False,
    }, index=dates)
    long_bp = pd.Series(np.arange(n, dtype=float) + 50, index=dates)
    short_bp = pd.Series(np.arange(n, dtype=float), index=dates)
    beta = pd.Series(0.5, index=dates)
    carry = pd.Series(5.0, index=dates)   # long residual earns +5bp of carry
    residual = {"beta": beta, "long_bp": long_bp, "short_bp": short_bp,
                "long_legs": {"au_gov_10y": 1.0}, "short_legs": {"au_gov_2y": 1.0}, "carry_long": carry}
    spec = {"group": "AU bond residual", "pca": False, "legs": {}, "belly": None, "wings": None}
    rules = {"entry_z": 2.0, "exit_z": 0.5, "stop_z": 3.5, "max_hold_days": 60,
             "is_end": pd.Timestamp("2020-12-31")}
    costs = {"swap_bp": 0.0, "bond_bp": 0.0}
    blocked, _ = simulate(d, spec, "au_gov_10s2s_resid", rules, costs, 0.04, 10_000, {},
                          residual=residual, require_carry=True)
    allowed, _ = simulate(d, spec, "au_gov_10s2s_resid", rules, costs, 0.04, 10_000, {},
                          residual=residual, require_carry=False)
    assert blocked.empty
    assert len(allowed) == 1 and allowed.iloc[0]["side"] == "short"


def test_carry_filter_allows_fade_with_carry():
    """z low (fade = long) and long-residual carry is positive => entry is allowed."""
    n = 40
    dates = pd.bdate_range("2019-01-02", periods=n)
    z = np.zeros(n)
    z[5] = -2.5
    z[6:20] = -1.0
    d = pd.DataFrame({
        "value_bp": np.arange(n, dtype=float), "z_250": z, "inputs_filled": False, "roll_day": False,
        "post_roll_block": False, "pending_roll": False, "insufficient_250": False,
    }, index=dates)
    residual = {
        "beta": pd.Series(0.5, index=dates),
        "long_bp": pd.Series(np.arange(n, dtype=float) + 50, index=dates),
        "short_bp": pd.Series(np.arange(n, dtype=float), index=dates),
        "long_legs": {"au_gov_10y": 1.0}, "short_legs": {"au_gov_2y": 1.0},
        "carry_long": pd.Series(5.0, index=dates),
    }
    spec = {"group": "AU bond residual", "pca": False, "legs": {}, "belly": None, "wings": None}
    rules = {"entry_z": 2.0, "exit_z": 0.5, "stop_z": 3.5, "max_hold_days": 60,
             "is_end": pd.Timestamp("2020-12-31")}
    trades, _ = simulate(d, spec, "au_gov_10s2s_resid", rules, {"swap_bp": 0.0, "bond_bp": 0.0},
                         0.04, 10_000, {}, residual=residual, require_carry=True)
    assert len(trades) == 1 and trades.iloc[0]["side"] == "long"


def test_residual_hedge_is_beta_times_10y_dv01():
    """10y combo is $10k DV01; the 2y hedge is beta × that, not a duration-neutral book."""
    n = 20
    dates = pd.bdate_range("2019-01-02", periods=n)
    z = np.zeros(n)
    z[5] = 2.5
    z[6:] = 1.0
    d = pd.DataFrame({
        "value_bp": np.arange(n, dtype=float), "z_250": z, "inputs_filled": False, "roll_day": False,
        "post_roll_block": False, "pending_roll": False, "insufficient_250": False,
    }, index=dates)
    beta_t = 0.7
    residual = {
        "beta": pd.Series(beta_t, index=dates),
        "long_bp": pd.Series(100.0, index=dates),
        "short_bp": pd.Series(40.0, index=dates),
        "long_legs": {"au_gov_10y": 1.0}, "short_legs": {"au_gov_2y": 1.0},
        "carry_long": pd.Series(0.0, index=dates),
    }
    spec = {"group": "AU bond residual", "pca": False, "legs": {}, "belly": None, "wings": None}
    rules = {"entry_z": 2.0, "exit_z": 0.5, "stop_z": 3.5, "max_hold_days": 60,
             "is_end": pd.Timestamp("2020-12-31")}
    trades, _ = simulate(d, spec, "au_gov_10s2s_resid", rules, {"swap_bp": 0.0, "bond_bp": 1.0},
                         0.04, 10_000, {}, residual=residual, require_carry=False)
    assert len(trades) == 1
    k = trades.iloc[0]["k_usd_per_bp"]
    assert k == pytest.approx(10_000)
    legs = {"au_gov_10y": 1.0, "au_gov_2y": -beta_t}
    assert net_dv01(legs, k) == pytest.approx(k * (1.0 - beta_t))
    assert trades.iloc[0]["cost_bp_roundtrip"] == pytest.approx(2.0 * (1.0 + beta_t))


def test_box_keeps_10y_combo_at_10k_dv01():
    """Box hedge must not be rescaled by the 2y long weight (that would shrink the 10y DV01)."""
    n = 20
    dates = pd.bdate_range("2019-01-02", periods=n)
    z = np.zeros(n)
    z[5] = 2.5
    z[6:] = 1.0
    d = pd.DataFrame({
        "value_bp": np.arange(n, dtype=float), "z_250": z, "inputs_filled": False, "roll_day": False,
        "post_roll_block": False, "pending_roll": False, "insufficient_250": False,
    }, index=dates)
    beta_t = 0.6
    residual = {
        "beta": pd.Series(beta_t, index=dates),
        "long_bp": pd.Series(80.0, index=dates),
        "short_bp": pd.Series(20.0, index=dates),
        "long_legs": {"au_gov_10y": 1.0, "nz_gov_10y": -1.0},
        "short_legs": {"au_gov_2y": 1.0, "nz_gov_2y": -1.0},
        "carry_long": pd.Series(0.0, index=dates),
    }
    spec = {"group": "AU-NZ residual", "pca": False, "legs": {}, "belly": None, "wings": None}
    rules = {"entry_z": 2.0, "exit_z": 0.5, "stop_z": 3.5, "max_hold_days": 60,
             "is_end": pd.Timestamp("2020-12-31")}
    trades, _ = simulate(d, spec, "au_nz_box", rules, {"swap_bp": 0.0, "bond_bp": 0.0},
                         0.04, 10_000, {}, residual=residual, require_carry=False)
    k = trades.iloc[0]["k_usd_per_bp"]
    assert k == pytest.approx(10_000)          # not 10000 / (1 + beta)
    legs = {"au_gov_10y": 1.0, "nz_gov_10y": -1.0, "au_gov_2y": -beta_t, "nz_gov_2y": beta_t}
    assert net_dv01(legs, k) == pytest.approx(0.0)
