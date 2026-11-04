"""Tests for src/pca.py - no look-ahead, PC1/PC2 neutrality, sign stability."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.pca import (fit_pca, fix_signs, fly_weights, frozen_zscores, rolling_pca,
                     weight_sentence, window_ok, yield_changes)
from src.spreads import to_bp

SPEC = {
    "fly_name": "au_gov_pca_fly",
    "belly": "au_gov_5y",
    "wings": ["au_gov_2y", "au_gov_10y"],
    "tenors": ["au_gov_2y", "au_gov_3y", "au_gov_5y", "au_gov_10y"],
}
PCA_CFG = {"window": 250, "min_real_fraction": 0.8, "max_gap_days": 10, "n_components": 3, "simple_wings": 0.5}
SCAN_CFG = {"windows": [60, 250], "min_real_fraction": 0.8, "max_gap_days": 10, "percentile_min_obs": 250}
TENORS = SPEC["tenors"]


def synthetic_yields(n: int = 420, seed: int = 0) -> pd.DataFrame:
    """Four-tenor curve whose daily changes are mostly level + slope + curvature."""
    dates = pd.bdate_range("2014-01-02", periods=n)
    rng = np.random.default_rng(seed)
    ages = np.array([2.0, 3.0, 5.0, 10.0])
    pc1 = np.ones(4); pc1 /= np.linalg.norm(pc1)
    pc2 = ages - ages.mean(); pc2 /= np.linalg.norm(pc2)
    pc3 = (ages - ages.mean()) ** 2; pc3 -= pc3.mean(); pc3 /= np.linalg.norm(pc3)
    chg = (rng.normal(0, 6, n)[:, None] * pc1
           + rng.normal(0, 1.8, n)[:, None] * pc2
           + rng.normal(0, 0.7, n)[:, None] * pc3
           + rng.normal(0, 0.12, (n, 4))) / 100.0
    return pd.DataFrame(3.0 + np.cumsum(chg, axis=0), index=dates, columns=TENORS)


def test_fix_signs_pc1_sum_positive_and_long_end_positive():
    loadings = np.array([
        [-0.5,  0.6,  0.2],
        [-0.5,  0.1, -0.3],
        [-0.5, -0.2, -0.4],
        [-0.5, -0.7,  0.5],
    ])
    out = fix_signs(loadings)
    assert out[:, 0].sum() > 0
    assert out[-1, 1] > 0
    assert out[-1, 2] > 0
    np.testing.assert_allclose(fix_signs(out), out)


def test_fly_weights_are_orthogonal_to_pc1_and_pc2():
    loadings = np.array([
        [0.50, -0.60,  0.30],   # 2y
        [0.50, -0.20, -0.10],   # 3y
        [0.50,  0.10, -0.50],   # 5y
        [0.50,  0.70,  0.40],   # 10y
    ])
    w2, w10 = fly_weights(loadings, TENORS, SPEC["belly"], SPEC["wings"])
    w = np.array([w2, 1.0, w10])
    legs = loadings[[0, 2, 3], :2]
    assert abs(w @ legs[:, 0]) < 1e-12
    assert abs(w @ legs[:, 1]) < 1e-12
    assert "sell" in weight_sentence(w2, w10) and "5y" in weight_sentence(w2, w10)


def test_filled_and_roll_days_are_excluded_from_the_fit():
    clean = synthetic_yields(n=40)
    raw = clean.copy()
    filled_day, roll_day = clean.index[10], clean.index[15]
    raw.loc[filled_day, "au_gov_2y"] = np.nan
    chg = yield_changes(clean, raw, TENORS, roll_days=pd.DatetimeIndex([roll_day]))
    assert chg.loc[filled_day].isna().all()
    assert chg.loc[roll_day].isna().all()
    assert chg.loc[clean.index[12]].notna().all()


def test_weights_neutralise_pc1_and_pc2_on_the_fitting_window():
    clean = synthetic_yields()
    chg = yield_changes(clean, clean, TENORS)
    weights, _ = rolling_pca(chg, SPEC, PCA_CFG)
    end = weights.index[-1]
    block = window_ok(chg, end, 250, 0.8, 10, [SPEC["belly"], *SPEC["wings"]])
    assert block is not None
    loadings, _ = fit_pca(block.to_numpy(), 3)
    w2, w10 = float(weights.at[end, "w_2y"]), float(weights.at[end, "w_10y"])
    idx = {n: i for i, n in enumerate(block.columns)}
    vec = np.zeros(len(block.columns))
    vec[idx[SPEC["wings"][0]]], vec[idx[SPEC["belly"]]], vec[idx[SPEC["wings"][1]]] = w2, 1.0, w10
    assert abs(vec @ loadings[:, 0]) < 1e-8
    assert abs(vec @ loadings[:, 1]) < 1e-8


def test_rolling_pca_has_no_lookahead():
    clean = synthetic_yields()
    chg = yield_changes(clean, clean, TENORS)
    w_full, _ = rolling_pca(chg, SPEC, PCA_CFG)
    w_cut, _ = rolling_pca(chg.iloc[:-40], SPEC, PCA_CFG)
    cols = ["w_2y", "w_10y", "var_pc1", "var_pc2", "pc1_2y", "pc2_10y"]
    pd.testing.assert_frame_equal(w_full.loc[w_cut.index, cols], w_cut[cols], atol=1e-10, check_freq=False)

    chg_future = chg.copy()
    chg_future.iloc[-40:] = 99.0
    w_alt, _ = rolling_pca(chg_future, SPEC, PCA_CFG)
    pd.testing.assert_frame_equal(w_alt.loc[w_cut.index, cols], w_cut[cols], atol=1e-10, check_freq=False)


def test_sign_stability_across_overlapping_windows():
    clean = synthetic_yields()
    chg = yield_changes(clean, clean, TENORS)
    w, _ = rolling_pca(chg, SPEC, PCA_CFG)
    assert (w["pc1_2y"] > 0).mean() > 0.99
    assert (w["pc1_10y"] > 0).mean() > 0.99
    assert (w["pc2_10y"] > 0).mean() > 0.99
    consecutive = np.sign(w["pc1_2y"]) * np.sign(w["pc1_2y"].shift())
    assert consecutive.dropna().mean() > 0.98


def test_frozen_zscore_rebuilds_history_with_date_t_weights():
    n = 300
    dates = pd.bdate_range("2020-01-01", periods=n)
    rng = np.random.default_rng(1)
    y2 = pd.Series(3.0 + rng.normal(0, 0.04, n).cumsum(), index=dates)
    y5 = pd.Series(3.2 + rng.normal(0, 0.04, n).cumsum(), index=dates)
    y10 = pd.Series(3.6 + rng.normal(0, 0.04, n).cumsum(), index=dates)
    fill = pd.Series(False, index=dates)
    w2_t, w10_t = -0.62, -0.48
    weights = pd.DataFrame({
        "w_2y": np.linspace(-0.90, w2_t, n),
        "w_10y": np.linspace(-0.30, w10_t, n),
    }, index=dates)
    z = frozen_zscores(y2, y5, y10, fill, weights.iloc[[-1]], SCAN_CFG)
    rebuilt = to_bp(w2_t * y2 + y5 + w10_t * y10)
    window = rebuilt.iloc[-250:]
    assert z["z_250"].iloc[-1] == pytest.approx((window.iloc[-1] - window.mean()) / window.std())
    # a stitched series (each day its own weights) would not match this window mean
    stitched = to_bp(weights["w_2y"] * y2 + y5 + weights["w_10y"] * y10)
    assert z["mean_250"].iloc[-1] != pytest.approx(stitched.iloc[-250:].mean())
