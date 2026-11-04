"""
run_all.py - one command for the full AU/NZ rates RV pipeline.

Order:
  fetch RBA (HTTP) and the newest RBNZ B2 file in data/manual/
  -> validate (stop and keep previous outputs on error)
  -> rolls (new candidates marked pending)
  -> instruments
  -> PCA
  -> scanner
  -> residuals
  -> backtests
  -> robustness report / README charts

Run from the project root:   python run_all.py
"""

from __future__ import annotations

import sys
from typing import Any

import pandas as pd

from src.utils import get_logger, load_config, project_path

log = get_logger("run_all")


def _validation_errors(cfg: dict[str, Any]) -> pd.DataFrame:
    path = project_path(cfg["paths"]["validation_file"])
    if not path.exists():
        return pd.DataFrame()
    report = pd.read_csv(path)
    if "severity" not in report.columns:
        return pd.DataFrame()
    return report[report["severity"] == "error"]


def _explain_stop(cfg: dict[str, Any], fetch_rc: int) -> None:
    print("\nSTOPPED. Previous processed files and outputs were left unchanged.\n")
    errors = _validation_errors(cfg)
    if len(errors):
        print("Validation errors:")
        cols = [c for c in ("check", "series", "date", "detail") if c in errors.columns]
        print(errors[cols].to_string(index=False))
        print()
        return
    print("A source failed before rates.csv could be updated (RBA download / manual F2,")
    print("or no RBNZ B2 file in data/manual/). Check the log above.")
    print(f"(fetch exit code {fetch_rc})\n")


def _latest_residual_signals(cfg: dict[str, Any]) -> str:
    path = project_path(cfg["paths"]["residual_history_file"])
    if not path.exists():
        return "  (no residual history yet)"
    h = pd.read_csv(path, parse_dates=["date"])
    lines = []
    for name, g in h.groupby("instrument", sort=False):
        row = g.sort_values("date").iloc[-1]
        z = row.get("z_250", float("nan"))
        c = row.get("carry_3m_bp", float("nan"))
        fade = "fade" if pd.notna(z) and abs(z) > 2 else "no fade"
        carry_ok = ""
        if pd.notna(z) and abs(z) > 2 and pd.notna(c):
            side = -1 if z > 0 else 1
            carry_ok = "carry with you" if side * c >= 0 else "blocked by carry"
        lines.append(
            f"  {name} as of {pd.Timestamp(row['date']).date()}: "
            f"z_250={z:+.2f}  3m carry of long residual={c:+.1f}bp  ({fade}"
            + (f", {carry_ok}" if carry_ok else "") + ")"
        )
    return "\n".join(lines) if lines else "  (empty)"


def _print_summary(cfg: dict[str, Any], scan: pd.DataFrame | None, rolls: pd.DataFrame | None,
                   n_new_rolls: int) -> None:
    paths = {k: project_path(v) for k, v in cfg["paths"].items()}
    rates = pd.read_csv(paths["rates_file"], parse_dates=["date"]) if paths["rates_file"].exists() else None
    print("\n========== pipeline summary ==========")
    if rates is not None and "date" in rates.columns:
        indexed = rates.set_index("date")
        au = [c for c in indexed.columns if c.startswith("au_")]
        nz = [c for c in indexed.columns if c.startswith("nz_")]
        au_last = indexed[au].dropna(how="all").index.max() if au else pd.NaT
        nz_last = indexed[nz].dropna(how="all").index.max() if nz else pd.NaT
        print(f"Data dates: AU {pd.Timestamp(au_last).date() if pd.notna(au_last) else 'n/a'}  "
              f"NZ {pd.Timestamp(nz_last).date() if pd.notna(nz_last) else 'n/a'}  "
              f"({indexed.index.min().date()} to {indexed.index.max().date()})")
    if rolls is not None:
        pending = rolls[rolls["status"] == "pending"]
        print(f"Roll candidates: {n_new_rolls} new this run, {len(pending)} pending review")
        for r in pending.itertuples():
            print(f"  pending {pd.Timestamp(r.date).date()} {r.series}")
    if scan is not None and len(scan):
        print("Top scanner signals:")
        for r in scan.head(cfg["scanner"]["top_n"]).itertuples():
            flag = " STRONG" if getattr(r, "signal", False) else ""
            z = getattr(r, "z_250", float("nan"))
            print(f"  {r.rank}. {r.instrument}{flag}: z_250={z:+.2f}  {r.trade_status}")
    print("Current residual signals:")
    print(_latest_residual_signals(cfg))
    print("======================================\n")


def run(cfg: dict[str, Any] | None = None) -> int:
    cfg = cfg or load_config()

    from src import fetch
    print("1/8  Fetch RBA F2 and newest RBNZ B2 in data/manual/")
    rc = fetch.run(cfg)
    if rc != 0:
        _explain_stop(cfg, rc)
        return 1

    from src import rolls as rolls_mod
    print("2/8  Rolls (new candidates -> pending)")
    before = rolls_mod.load_roll_dates(project_path(cfg["paths"]["roll_dates_file"]))
    n_before = int((before["status"] == "pending").sum()) if len(before) else 0
    rolls = rolls_mod.run(cfg)
    n_after = int((rolls["status"] == "pending").sum()) if len(rolls) else 0
    n_new = max(0, n_after - n_before)

    from src import spreads
    print("3/8  Instruments")
    spreads.run(cfg)

    from src import pca
    print("4/8  PCA butterflies")
    pca.run(cfg)

    from src import scanner
    print("5/8  Scanner")
    scan = scanner.run(cfg)

    from src import residual
    print("6/8  Policy-hedged residuals")
    residual.run(cfg)

    from src import backtest
    print("7/8  Backtests")
    backtest.run(cfg)

    from src import robustness
    print("8/8  Robustness report (does not retune)")
    robustness.run(cfg)

    _print_summary(cfg, scan, rolls, n_new)
    return 0


if __name__ == "__main__":
    sys.exit(run())
