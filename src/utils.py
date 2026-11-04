"""Small helpers shared by every module: config loading, logging and file I/O."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path = PROJECT_ROOT / "config.yaml") -> dict[str, Any]:
    """Read config.yaml into a nested dictionary."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def project_path(relative: str | Path) -> Path:
    """Turn a path from config.yaml (relative to the project root) into an absolute path."""
    return PROJECT_ROOT / relative


def get_logger(name: str) -> logging.Logger:
    """Return a logger that prints 'time LEVEL module: message' to the console."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("kaleido", "choreographer"):   # chart-export libraries log every step
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logging.getLogger(name)


def read_rates(path: str | Path) -> pd.DataFrame | None:
    """Load rates.csv (dates as index, one column per series). Returns None if it doesn't exist."""
    path = Path(path)
    if not path.exists():
        return None
    return pd.read_csv(path, index_col="date", parse_dates=["date"]).sort_index()


def with_gap_breaks(data: pd.DataFrame | pd.Series, max_days: int = 7) -> pd.DataFrame | pd.Series:
    """For charts: insert an empty row before any jump of more than `max_days` calendar days,
    so plotted lines break at data gaps instead of drawing a straight line across them."""
    idx = data.index
    starts = idx[1:][np.diff(idx.values) > np.timedelta64(max_days, "D")]
    if len(starts) == 0:
        return data
    blank = data.iloc[:0].reindex(starts - pd.Timedelta(days=1))
    return pd.concat([data, blank]).sort_index()


def write_rates(df: pd.DataFrame, path: str | Path) -> None:
    """Save rates in a stable format (sorted dates, ISO date strings, 4 decimal places)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.sort_index().copy()
    out.index = out.index.strftime("%Y-%m-%d")
    out.index.name = "date"
    out.to_csv(path, float_format="%.4f")
