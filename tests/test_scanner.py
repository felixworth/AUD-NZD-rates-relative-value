"""Tests for src/scanner.py - no look-ahead, filled days, insufficient history, signals."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.scanner import build_table, gap_mask, zscore_history
from src.utils import load_config

CFG = load_config()
ARGS = dict(windows=[60, 250], min_real_fraction=0.8, max_gap=10, percentile_min_obs=250)


def random_walk(n: int = 600, seed: int = 0) -> pd.Series:
    dates = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(np.random.default_rng(seed).normal(0, 2, n).cumsum(), index=dates)


def no_fill(s: pd.Series) -> pd.Series:
    return pd.Series(False, index=s.index)


def test_zscores_unchanged_when_future_data_is_appended():
    full = random_walk(600)
    cut = full.iloc[:400]
    z_full = zscore_history(full, no_fill(full), **ARGS)
    z_cut = zscore_history(cut, no_fill(cut), **ARGS)
    cols = ["z_60", "z_250", "mean_250", "std_250", "percentile"]
    pd.testing.assert_frame_equal(z_full.loc[cut.index, cols], z_cut[cols])
    # and changing the future must not change the past either
    altered = full.copy()
    altered.iloc[400:] += 500
    pd.testing.assert_frame_equal(zscore_history(altered, no_fill(altered), **ARGS).loc[cut.index, cols],
                                  z_cut[cols])


def test_zscore_matches_manual_calculation():
    s = random_walk(300)
    z = zscore_history(s, no_fill(s), **ARGS)
    window = s.iloc[-60:]
    assert z["z_60"].iloc[-1] == pytest.approx((s.iloc[-1] - window.mean()) / window.std())


def test_filled_days_never_get_a_zscore_or_signal():
    s = random_walk(400)
    filled = no_fill(s)
    filled.iloc[[100, 250, -1]] = True
    z = zscore_history(s, filled, **ARGS)
    assert z.loc[filled, ["z_60", "z_250"]].isna().all().all()
    # the filled value is also left out of the window statistics
    expected_mean = s.where(~filled).iloc[-60:].dropna().mean()    # 59 real values
    assert z["mean_60"].iloc[-1] == pytest.approx(expected_mean)
    assert z["mean_60"].iloc[-1] != pytest.approx(s.iloc[-60:].mean())


def test_insufficient_history_when_too_few_real_values():
    s = random_walk(400)
    keep = np.ones(len(s), dtype=bool)
    keep[-250:][np.arange(250) % 4 == 0] = False          # only 75% real in the last 250 days
    sparse = s[keep]
    z = zscore_history(sparse, no_fill(sparse), **ARGS)
    assert np.isnan(z["z_250"].iloc[-1]) and z["insufficient_250"].iloc[-1]
    assert not z["insufficient_60"].iloc[-1] or np.isnan(z["z_60"].iloc[-1])


def test_insufficient_history_when_window_spans_a_long_gap():
    s = random_walk(400)
    gap_11 = s.drop(s.index[300:311])                      # 11 missing trading days
    gap_10 = s.drop(s.index[300:310])                      # 10 missing trading days
    z11 = zscore_history(gap_11, no_fill(gap_11), **ARGS)
    z10 = zscore_history(gap_10, no_fill(gap_10), **ARGS)
    assert z11["insufficient_250"].iloc[-1] and np.isnan(z11["z_250"].iloc[-1])
    assert not z10["insufficient_250"].iloc[-1] and np.isfinite(z10["z_250"].iloc[-1])


def test_gap_mask_window_edges():
    real = pd.Series(True, index=range(100))
    real[20:35] = False                                    # 15-day gap at positions 20..34
    bad = gap_mask(real, window=30, max_gap=10)
    assert not bad[29] and bad[30]                         # overlap reaches 11 days at t = 30
    assert bad[53] and not bad[54]                         # last t with > 10 days overlap: 34 + 30 - 1 - 10


def test_percentile_needs_minimum_history_and_uses_past_only():
    s = random_walk(400)
    z = zscore_history(s, no_fill(s), **ARGS)
    assert z["percentile"].iloc[:249].isna().all()
    t = 300
    expected = (s.iloc[: t + 1] <= s.iloc[t]).mean() * 100
    assert z["percentile"].iloc[t] == pytest.approx(expected)


def history_row(name: str, z60: float, z250: float, **extra) -> pd.DataFrame:
    dates = pd.bdate_range("2026-10-01", periods=3)
    base = {"instrument": name, "value_bp": 10.0, "inputs_filled": False, "roll_day": False,
            "post_roll_block": False, "roll_info": "", "z_60": z60, "z_250": z250, "percentile": 90.0,
            "insufficient_60": False, "insufficient_250": False}
    base.update(extra)
    return pd.DataFrame([{**base, "date": d} for d in dates])


def test_signal_strength_and_trade_status():
    hist = pd.concat([
        history_row("nz_swap_2s10s", 2.5, 2.2),
        history_row("au_gov_2s10s", -1.0, -3.0),
        history_row("nz_swap_spread_5y", -2.6, -2.4, post_roll_block=True),
    ])
    rolls = pd.DataFrame(columns=["date", "series", "status"])
    table = build_table(hist, CFG["instruments"]["definitions"], rolls, CFG, pd.Timestamp("2026-10-05"))
    t = table.set_index("instrument")
    assert list(table["instrument"]) == ["au_gov_2s10s", "nz_swap_spread_5y", "nz_swap_2s10s"]   # by |z_250|
    assert t.loc["nz_swap_2s10s", "signal"] == "strong high"
    assert t.loc["au_gov_2s10s", "signal"] == ""                   # z_60 doesn't agree
    assert t.loc["nz_swap_spread_5y", "signal"] == "strong low"
    assert t.loc["nz_swap_spread_5y", "trade_status"].startswith("no new trades")
    assert t.loc["nz_swap_2s10s", "trade_status"] == "tradeable"


def test_filled_latest_value_is_no_new_trades():
    hist = history_row("nz_swap_2s10s", np.nan, np.nan, inputs_filled=True)
    rolls = pd.DataFrame(columns=["date", "series", "status"])
    t = build_table(hist, CFG["instruments"]["definitions"], rolls, CFG, pd.Timestamp("2026-10-05")).iloc[0]
    assert t["signal"] == "" and "filled input" in t["flags"] and t["trade_status"].startswith("no new trades")
