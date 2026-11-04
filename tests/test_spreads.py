"""Tests for src/spreads.py - bp conversion, formulas, date limits and roll flags."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.spreads import build_instruments, check_definitions, instrument_level, roll_flags, to_bp
from src.utils import load_config

CFG = load_config()
DEFS = CFG["instruments"]["definitions"]
NO_ROLLS = pd.DataFrame(columns=["date", "series", "status"])


def flat_rates(dates: pd.DatetimeIndex, **levels: float) -> pd.DataFrame:
    return pd.DataFrame({k: v for k, v in levels.items()}, index=dates)


def test_bp_conversion():
    assert to_bp(1.0) == 100.0
    assert to_bp(0.0125) == pytest.approx(1.25)
    clean = flat_rates(pd.bdate_range("2024-01-01", periods=1), nz_swap_2y=4.00, nz_swap_5y=4.25)
    assert instrument_level(clean, {"nz_swap_5y": 1, "nz_swap_2y": -1}).iloc[0] == pytest.approx(25.0)


def test_butterfly_formula():
    """2s5s10s = 2 x 5y - 2y - 10y, in bp."""
    clean = flat_rates(pd.bdate_range("2024-01-01", periods=1), au_gov_2y=3.0, au_gov_5y=4.0, au_gov_10y=4.5)
    fly = instrument_level(clean, DEFS["au_gov_2s5s10s"]["legs"]).iloc[0]
    assert fly == pytest.approx(2 * 400 - 300 - 450)       # = 50bp


def test_definitions_use_expected_series_and_never_adj():
    check_definitions(DEFS)                                 # config passes
    assert all(not s.endswith("_adj") for d in DEFS.values() for s in d["legs"])
    for tenor in ("2y", "5y", "10y"):
        assert DEFS[f"au_nz_gov_{tenor}"]["legs"] == {f"au_gov_{tenor}": 1, f"nz_gov_{tenor}": -1}
        assert DEFS[f"nz_swap_spread_{tenor}"]["legs"] == {f"nz_swap_{tenor}": 1, f"nz_gov_{tenor}": -1}
    assert all(s.startswith("nz_swap_") for n, d in DEFS.items() if n.startswith("nz_swap_2s") for s in d["legs"])
    with pytest.raises(ValueError):
        check_definitions({"bad": {"group": "x", "legs": {"nz_gov_2y_adj": 1, "nz_swap_2y": -1}}})


def synthetic_rates() -> pd.DataFrame:
    dates = pd.bdate_range("2013-08-26", periods=30)
    rng = np.random.default_rng(0)
    cols = ["au_gov_2y", "au_gov_3y", "au_gov_5y", "au_gov_10y", "nz_gov_2y", "nz_gov_5y", "nz_gov_10y",
            "nz_swap_2y", "nz_swap_5y", "nz_swap_10y"]
    df = pd.DataFrame(3 + rng.normal(0, 0.05, (len(dates), len(cols))).cumsum(axis=0), index=dates, columns=cols)
    df.loc[dates[-5]:, [c for c in cols if c.startswith("au_")]] = np.nan   # RBA not yet published
    return df


def test_no_values_beyond_inputs_last_real_date():
    rates = synthetic_rates()
    long, summary = build_instruments(rates, DEFS, "2013-09-02", NO_ROLLS, 20, 2)
    au_last = rates["au_gov_10y"].last_valid_index()
    nz_last = rates["nz_swap_10y"].last_valid_index()
    for name in DEFS:
        last = long.loc[long["instrument"] == name, "date"].max()
        has_au = any(s.startswith("au_") for s in DEFS[name]["legs"])
        assert last == (au_last if has_au else nz_last), name
    s = summary.set_index("instrument")
    assert pd.Timestamp(s.loc["au_nz_gov_10y", "as_of"]) == au_last
    assert pd.Timestamp(s.loc["au_nz_gov_10y", "start"]) == pd.Timestamp("2013-09-02")   # au_start respected


def test_forward_filled_inputs_are_marked():
    rates = synthetic_rates()
    day = rates.index[10]
    rates.loc[day, "nz_gov_5y"] = np.nan                     # NZ holiday for the bond
    long, _ = build_instruments(rates, DEFS, "2013-09-02", NO_ROLLS, 20, 2)
    row = long[(long["instrument"] == "au_nz_gov_5y") & (long["date"] == day)].iloc[0]
    assert row["inputs_filled"] and row["value_bp"] == pytest.approx(
        100 * (rates.loc[day, "au_gov_5y"] - rates["nz_gov_5y"].iloc[9]), abs=0.01)


def test_roll_flags_and_block_window():
    index = pd.bdate_range("2024-01-01", periods=40)
    rolls = pd.DataFrame({"date": [index[5], index[30]], "series": ["nz_gov_2y", "nz_gov_2y"],
                          "status": ["confirmed", "pending"]})
    f = roll_flags(index, ["nz_gov_2y"], rolls, block_days=20)
    assert f["roll_day"].sum() == 2
    assert f["post_roll_block"].iloc[5:26].all() and not f["post_roll_block"].iloc[26:30].any()
    assert f["post_roll_block"].iloc[30:].all()               # pending blocks until reviewed
    assert not roll_flags(index, ["nz_gov_5y"], rolls, 20)["roll_day"].any()   # other bond: no flags
    assert not roll_flags(index, [], rolls, 20)["roll_day"].any()              # swap-only: no flags


def test_rejected_rolls_are_ignored():
    index = pd.bdate_range("2024-01-01", periods=10)
    rolls = pd.DataFrame({"date": [index[3]], "series": ["nz_gov_5y"], "status": ["rejected"]})
    assert not roll_flags(index, ["nz_gov_5y"], rolls, 20).any(axis=None)
