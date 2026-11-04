"""run_all.py stops on fetch/validation failure and does not call later steps."""

from __future__ import annotations

from unittest.mock import patch

import run_all


def test_run_all_stops_when_fetch_fails():
    cfg = {"paths": {"validation_file": "data/processed/validation_report.csv"},
           "scanner": {"top_n": 5}}
    with patch("src.fetch.run", return_value=1) as fetch, \
            patch("src.rolls.run") as rolls, \
            patch("src.spreads.run") as spreads, \
            patch("src.pca.run") as pca, \
            patch("src.scanner.run") as scanner, \
            patch("src.residual.run") as residual, \
            patch("src.backtest.run") as backtest, \
            patch("src.robustness.run") as robustness, \
            patch.object(run_all, "load_config", return_value=cfg), \
            patch.object(run_all, "_explain_stop"):
        rc = run_all.run(cfg)
    assert rc == 1
    fetch.assert_called_once()
    rolls.assert_not_called()
    spreads.assert_not_called()
    pca.assert_not_called()
    scanner.assert_not_called()
    residual.assert_not_called()
    backtest.assert_not_called()
    robustness.assert_not_called()
