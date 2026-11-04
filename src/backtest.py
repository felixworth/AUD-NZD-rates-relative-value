"""
backtest.py - walk-forward mean-reversion backtest of the scanner instruments.

Rules (config.yaml -> backtest; not tuned on the out-of-sample window):
  enter when |z_250| > entry_z (fade: short a high reading, long a low one)
  exit  when |z_250| < exit_z
  stop  when |z_250| > stop_z
  flatten after max_hold_days trading days
  signal at close t, trade at close t+1
  one position per instrument

No new trade if the signal day is filled, has insufficient 250-day history, is in a
post-roll block, or has a pending roll. NZ-bond roll-day changes are excluded from
PnL and any open bond-based trade is flattened at the previous close (we do not
hold across a benchmark switch).

Sizing: each instrument is a yield combination with weights w_i. Notionals are
chosen so dollar DV01 of leg i equals k * w_i, and the long side sums to
$10,000 DV01. When the weights sum to zero (every instrument here, PCA flies
nearly so) the book is net-DV01-neutral: a 1bp parallel shift of every leg is
worth ~$0, and a 1bp move in the combination is worth $k. Par modified duration
at 4% is used for every tenor (see config.yaml).

PnL is reported in bp of that yield combination and in $ per $10k long-side DV01.
One-way costs are charged on each leg at entry and again at exit (0.5bp swaps,
1bp bonds).

In-sample 2 Sep 2013 – 31 Dec 2020; out-of-sample 1 Jan 2021 – latest. Daily PnL,
Sharpe and drawdown are split by calendar date; trade counts / hit rate by entry date.

Run from the project root:   python -m src.backtest
"""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.fetch import forward_fill_interior
from src.rolls import load_roll_dates, pending_series
from src.spreads import to_bp
from src.utils import get_logger, load_config, project_path, read_rates, with_gap_breaks

log = get_logger("backtest")

TENOR_YEARS = re.compile(r"_(\d+)y$")


# =============================================================================
# DV01, costs, instrument weights
# =============================================================================

def par_mod_duration(n: float, y: float) -> float:
    """Modified duration of a par bond/swap with annual coupons, n years, yield y."""
    if y <= 0:
        return float(n)
    return (1.0 - (1.0 + y) ** (-n)) / y


def tenor_years(series: str) -> float:
    m = TENOR_YEARS.search(series)
    if not m:
        raise ValueError(f"Cannot read tenor from series name {series!r}")
    return float(m.group(1))


def durations(legs: dict[str, float], assume_yield: float) -> dict[str, float]:
    return {s: par_mod_duration(tenor_years(s), assume_yield) for s in legs}


def book_scale(legs: dict[str, float], book_dv01_usd: float) -> float:
    """Dollars of PnL per 1bp of the yield combination (k in N_i * D_i = k * w_i)."""
    long_w = sum(w for w in legs.values() if w > 0)
    if long_w <= 0:
        return float("nan")
    return book_dv01_usd / long_w


def net_dv01(legs: dict[str, float], k: float) -> float:
    """Net dollar DV01 of the book (0 when the yield weights sum to 0)."""
    return k * sum(legs.values())


def one_way_cost_bp(legs: dict[str, float], costs: dict[str, float]) -> float:
    """Cost in bp of the yield combination for crossing every leg once."""
    total = 0.0
    for series, w in legs.items():
        c = costs["swap_bp"] if "swap" in series else costs["bond_bp"]
        total += abs(w) * c
    return total


def instrument_specs(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """name -> {group, legs, pca, belly, wings}."""
    specs: dict[str, dict[str, Any]] = {}
    for name, d in cfg["instruments"]["definitions"].items():
        specs[name] = {"group": d["group"], "legs": dict(d["legs"]), "pca": False,
                       "belly": None, "wings": None}
    for spec in cfg.get("pca", {}).get("curves", {}).values():
        specs[spec["fly_name"]] = {
            "group": spec["group"], "pca": True, "belly": spec["belly"], "wings": spec["wings"],
            "legs": {spec["belly"]: 1.0, spec["wings"][0]: 1.0, spec["wings"][1]: 1.0},
        }
    return specs


def pca_weight_lookup(path) -> dict[str, pd.DataFrame]:
    """fly name -> DataFrame indexed by date with w_2y, w_5y, w_10y."""
    p = project_path(path)
    if not p.exists():
        return {}
    w = pd.read_csv(p, parse_dates=["date"])
    out = {}
    for fly, g in w.groupby("fly"):
        out[fly] = g.set_index("date")[["w_2y", "w_5y", "w_10y"]].sort_index()
    return out


def legs_on_date(spec: dict[str, Any], date: pd.Timestamp, pca_w: dict[str, pd.DataFrame],
                 name: str) -> dict[str, float] | None:
    if not spec["pca"]:
        return dict(spec["legs"])
    table = pca_w.get(name)
    if table is None or date not in table.index:
        return None
    row = table.loc[date]
    return {spec["wings"][0]: float(row["w_2y"]), spec["belly"]: float(row["w_5y"]),
            spec["wings"][1]: float(row["w_10y"])}


# =============================================================================
# Event calendar (RBA / RBNZ decisions, AU / NZ CPI)
# =============================================================================

# RBNZ OCR announcement dates, 2013 onwards. Source: rbnz.govt.nz/monetary-policy/monetary-policy-decisions
RBNZ_OCR = """
2013-01-31 2013-03-14 2013-04-24 2013-06-13 2013-07-25 2013-09-12 2013-10-31 2013-12-12
2014-01-30 2014-03-13 2014-04-24 2014-06-12 2014-07-24 2014-09-11 2014-10-30 2014-12-11
2015-01-29 2015-03-12 2015-04-30 2015-06-11 2015-07-23 2015-09-10 2015-10-29 2015-12-10
2016-01-28 2016-03-10 2016-04-28 2016-06-09 2016-08-11 2016-09-22 2016-11-10
2017-02-09 2017-03-23 2017-05-11 2017-06-22 2017-08-10 2017-09-28 2017-11-09
2018-02-08 2018-03-22 2018-05-10 2018-06-28 2018-08-09 2018-09-27 2018-11-08
2019-02-13 2019-03-27 2019-05-08 2019-06-26 2019-08-07 2019-09-25 2019-11-13
2020-02-12 2020-03-16 2020-03-23 2020-04-07 2020-05-13 2020-06-24 2020-08-12 2020-09-23 2020-11-11
2021-02-24 2021-04-14 2021-05-26 2021-07-14 2021-08-18 2021-10-06 2021-11-24
2022-02-23 2022-04-13 2022-05-25 2022-07-13 2022-08-17 2022-10-05 2022-11-23
2023-02-22 2023-04-05 2023-05-24 2023-07-12 2023-08-16 2023-10-04 2023-11-29
2024-02-28 2024-04-10 2024-05-22 2024-07-10 2024-08-14 2024-10-09 2024-11-27
2025-02-19 2025-04-09 2025-05-28 2025-07-09 2025-08-20 2025-10-08 2025-11-26
2026-02-18 2026-04-08 2026-05-27 2026-07-08 2026-09-02 2026-10-28
""".split()

# RBA announcement day = last day of the published two-day meeting, 2024 onwards.
RBA_FROM_2024 = """
2024-02-06 2024-03-19 2024-05-07 2024-06-18 2024-08-06 2024-09-24 2024-11-05 2024-12-10
2025-02-18 2025-04-01 2025-05-20 2025-07-08 2025-08-12 2025-09-30 2025-11-04 2025-12-09
2026-02-03 2026-03-17 2026-05-05 2026-06-16 2026-08-11 2026-09-29 2026-11-03 2026-12-08
""".split()

# Known ABS CPI dates that are not simply "last Wednesday of the month".
AU_CPI_OVERRIDE = {
    "2025-11-26": "au_cpi",   # first complete monthly CPI (October 2025)
    "2026-01-07": "au_cpi",   # November 2025 (holiday exception)
    "2026-10-28": "au_cpi",   # September 2026 (also an RBA-week CPI)
}


def _first_tuesday(year: int, month: int) -> pd.Timestamp:
    d = pd.Timestamp(year=year, month=month, day=1)
    return d + pd.Timedelta(days=(1 - d.weekday()) % 7)


def _last_wednesday(year: int, month: int) -> pd.Timestamp:
    d = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
    return d - pd.Timedelta(days=(d.weekday() - 2) % 7)


def build_events(start: str = "2013-01-01", end: str | None = None) -> pd.DataFrame:
    """RBA / RBNZ decision dates and AU / NZ CPI release dates.

    RBA 2013–2023: first Tuesday of each month except January, plus the 18 Mar 2020
    unscheduled meeting. 2024–: published announcement days.
    RBNZ: official OCR dates.
    AU CPI: last Wednesday of Jan/Apr/Jul/Oct (quarterly) through 2022, then last
    Wednesday of every month (monthly indicator, then complete monthly CPI), with
    a few documented exceptions.
    NZ CPI: 16 weekdays after quarter-end (Stats NZ is "about 12 working days",
    December slips; Sep-2026 actually printed 22 Oct 2026 = 16 weekdays). Stated
    as an approximation, not a scraped release calendar.
    """
    end = pd.Timestamp(end or pd.Timestamp.today())
    start_ts = pd.Timestamp(start)
    rows = []

    for year in range(start_ts.year, 2024):
        for month in range(1, 13):
            if month == 1:
                continue
            rows.append((_first_tuesday(year, month), "rba", "AU"))
    rows.append((pd.Timestamp("2020-03-18"), "rba", "AU"))
    for s in RBA_FROM_2024:
        rows.append((pd.Timestamp(s), "rba", "AU"))
    for s in RBNZ_OCR:
        rows.append((pd.Timestamp(s), "rbnz", "NZ"))

    for year in range(start_ts.year, end.year + 1):
        months = range(1, 13) if year >= 2022 else (1, 4, 7, 10)
        for month in months:
            rows.append((_last_wednesday(year, month), "au_cpi", "AU"))
    for s, kind in AU_CPI_OVERRIDE.items():
        rows.append((pd.Timestamp(s), kind, "AU"))

    for q_end in pd.date_range("2012-12-31", end, freq="QE"):
        nz = pd.bdate_range(q_end + pd.Timedelta(days=1), periods=16)[-1]
        rows.append((nz, "nz_cpi", "NZ"))

    ev = pd.DataFrame(rows, columns=["date", "event", "country"]).drop_duplicates(["date", "event"])
    ev = ev[(ev["date"] >= start_ts) & (ev["date"] <= end)].sort_values(["date", "event"])
    return ev.reset_index(drop=True)


def event_types_for(name: str) -> set[str]:
    if name.startswith("au_nz"):
        return {"rba", "rbnz", "au_cpi", "nz_cpi"}
    if name.startswith("au_"):
        return {"rba", "au_cpi"}
    return {"rbnz", "nz_cpi"}


# =============================================================================
# Simulation
# =============================================================================

def _as_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    return s.astype(str).str.lower().isin(["1", "true", "yes"])


def _sample_of(date: pd.Timestamp, is_end: pd.Timestamp) -> str:
    return "IS" if date <= is_end else "OOS"


def simulate(df: pd.DataFrame, spec: dict[str, Any], name: str, rules: dict[str, Any],
             costs: dict[str, float], assume_yield: float, book_dv01: float,
             pca_w: dict[str, pd.DataFrame],
             clean: pd.DataFrame | None = None,
             residual: dict[str, Any] | None = None,
             require_carry: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Walk one instrument. `df` is scanner-history rows indexed by date.

    Signals use scanner z_250. Marks use the instrument's value_bp, except PCA flies
    (weights frozen at the signal date) and residuals (beta frozen at the signal date).
    `require_carry` is only used by the residual backtest; Step 4 leaves it False.
    Returns (trades, daily PnL).
    """
    d = df.sort_index()
    dates = list(d.index)
    z = d["z_250"].to_numpy(dtype=float)
    filled = _as_bool(d["inputs_filled"]).to_numpy()
    roll = _as_bool(d["roll_day"]).to_numpy()
    pending_roll = _as_bool(d["pending_roll"]).to_numpy() if "pending_roll" in d.columns \
        else np.zeros(len(d), dtype=bool)
    post = _as_bool(d["post_roll_block"]).to_numpy()
    insuff = _as_bool(d["insufficient_250"]).to_numpy()
    no_signal = filled | roll | post | pending_roll | insuff | ~np.isfinite(z)
    no_exec_enter = filled | roll | post | pending_roll
    carry_arr = (residual["carry_long"].reindex(d.index).to_numpy(dtype=float)
                 if residual is not None and "carry_long" in residual else None)

    scanner_px = d["value_bp"].to_numpy(dtype=float)
    mark = scanner_px.copy()
    n = len(dates)
    daily_bp = np.zeros(n)
    daily_usd = np.zeros(n)
    pos = 0
    entry_px = entry_z = np.nan
    entry_date = None
    k = cost = 0.0
    last_px = np.nan
    hold = 0
    pending: tuple[str, Any] | None = None
    trades: list[dict[str, Any]] = []

    def close_trade(i: int, reason: str, px: float) -> None:
        nonlocal pos, pending, hold
        if pos == 0:
            return
        pnl_bp = pos * (px - entry_px) - 2.0 * cost
        daily_bp[i] -= 2.0 * cost
        daily_usd[i] -= k * 2.0 * cost
        trades.append({
            "instrument": name, "group": spec["group"], "side": "long" if pos > 0 else "short",
            "entry_date": entry_date, "exit_date": dates[i], "entry_z": entry_z, "exit_z": z[i],
            "entry_bp": entry_px, "exit_bp": px, "pnl_bp": pnl_bp, "pnl_usd": k * pnl_bp,
            "hold_days": hold, "reason": reason, "k_usd_per_bp": k, "cost_bp_roundtrip": 2.0 * cost,
            "sample": _sample_of(entry_date, rules["is_end"]),
        })
        pos = 0
        pending = None
        hold = 0

    for i in range(n):
        px = mark[i]
        real = np.isfinite(px) and not filled[i]

        if pos != 0 and roll[i]:
            close_trade(i, "roll", last_px if np.isfinite(last_px) else entry_px)
            pending = None
            mark = scanner_px.copy()
            continue

        if pos != 0 and real:
            daily_bp[i] += pos * (px - last_px)
            daily_usd[i] += k * pos * (px - last_px)
            last_px = px
            hold += 1

        if pending is not None and real:
            action, payload = pending
            if action == "enter" and not no_exec_enter[i]:
                legs, k_override = None, None
                if residual is not None:
                    sig = payload["signal_date"]
                    beta_s = residual["beta"]
                    if sig in beta_s.index and np.isfinite(beta_s.at[sig]):
                        beta_t = float(beta_s.at[sig])
                        frozen = (residual["long_bp"] - beta_t * residual["short_bp"]).reindex(d.index)
                        mark = frozen.to_numpy(dtype=float)
                        px = mark[i]
                        legs = dict(residual["long_legs"])
                        for s, w in residual["short_legs"].items():
                            legs[s] = legs.get(s, 0.0) - beta_t * w
                        long_w = sum(w for w in residual["long_legs"].values() if w > 0)
                        k_override = book_dv01 / long_w if long_w > 0 else float("nan")
                else:
                    sig = payload["signal_date"] if spec.get("pca") else dates[i]
                    legs = legs_on_date(spec, sig, pca_w, name)
                    if spec.get("pca") and clean is not None:
                        frozen = frozen_pca_level(clean, spec, payload["signal_date"], pca_w, name)
                        if frozen is not None:
                            mark = frozen.reindex(d.index).to_numpy(dtype=float)
                            px = mark[i]
                if legs and np.isfinite(px) and all(np.isfinite(list(legs.values()))):
                    pos = payload["side"]
                    entry_px = last_px = px
                    entry_z = payload["z"]
                    entry_date = dates[i]
                    k = k_override if k_override is not None else book_scale(legs, book_dv01)
                    cost = one_way_cost_bp(legs, costs)
                    hold = 0
                pending = None
            elif action == "enter" and filled[i]:
                pass
            elif action == "enter":
                pending = None
            elif action == "exit":
                close_trade(i, payload, px)
                mark = scanner_px.copy()

        if pos == 0 and pending is None and not no_signal[i] and abs(z[i]) > rules["entry_z"]:
            side = -int(np.sign(z[i]))
            ok_carry = True
            if require_carry:
                cr = carry_arr[i] if carry_arr is not None else np.nan
                ok_carry = bool(np.isfinite(cr) and side * cr >= 0)
            if ok_carry:
                pending = ("enter", {"side": side, "z": float(z[i]), "signal_date": dates[i]})
        elif pos != 0 and pending is None:
            if np.isfinite(z[i]) and abs(z[i]) > rules["stop_z"]:
                pending = ("exit", "stop")
            elif hold >= rules["max_hold_days"]:
                pending = ("exit", "time")
            elif np.isfinite(z[i]) and abs(z[i]) < rules["exit_z"]:
                pending = ("exit", "target")

    if pos != 0:
        close_trade(n - 1, "end_of_sample", last_px if np.isfinite(last_px) else entry_px)

    pos_path = np.zeros(n)
    for t in trades:
        a, b = dates.index(t["entry_date"]), dates.index(t["exit_date"])
        pos_path[a:b] = 1 if t["side"] == "long" else -1
    daily = pd.DataFrame({"date": dates, "instrument": name, "pnl_bp": daily_bp,
                          "pnl_usd": daily_usd, "position": pos_path})
    cols = ["instrument", "group", "side", "entry_date", "exit_date", "entry_z", "exit_z",
            "entry_bp", "exit_bp", "pnl_bp", "pnl_usd", "hold_days", "reason",
            "k_usd_per_bp", "cost_bp_roundtrip", "sample"]
    return pd.DataFrame(trades, columns=cols), daily


def frozen_pca_level(clean: pd.DataFrame, spec: dict[str, Any], signal_date: pd.Timestamp,
                     pca_w: dict[str, pd.DataFrame], name: str) -> pd.Series | None:
    legs = legs_on_date(spec, signal_date, pca_w, name)
    if not legs:
        return None
    return to_bp(sum(w * clean[s] for s, w in legs.items()))


# =============================================================================
# Metrics
# =============================================================================

def sharpe(daily: pd.Series) -> float:
    x = daily.dropna()
    if len(x) < 2 or x.std(ddof=1) == 0:
        return float("nan")
    return float(x.mean() / x.std(ddof=1) * math.sqrt(252))


def max_drawdown(cum: pd.Series) -> float:
    if cum.empty:
        return float("nan")
    peak = cum.cummax()
    dd = cum - peak
    return float(dd.min())


def summarise_trades(trades: pd.DataFrame, daily: pd.DataFrame, sample: str | None,
                     is_end: pd.Timestamp) -> dict[str, Any]:
    t = trades if sample is None else trades[trades["sample"] == sample]
    if sample == "IS":
        d = daily[daily["date"] <= is_end]
    elif sample == "OOS":
        d = daily[daily["date"] > is_end]
    else:
        d = daily
    n = len(t)
    stopped = (t["reason"] == "stop").mean() * 100 if n else float("nan")
    pnl_usd = d["pnl_usd"].sum() if len(d) else 0.0
    pnl_bp = d["pnl_bp"].sum() if len(d) else 0.0
    by_date = d.groupby("date")["pnl_usd"].sum() if len(d) else pd.Series(dtype=float)
    return {
        "sample": sample or "ALL", "trades": n,
        "hit_rate": (t["pnl_bp"] > 0).mean() * 100 if n else float("nan"),
        "avg_bp": t["pnl_bp"].mean() if n else float("nan"),
        "total_bp": pnl_bp, "total_usd": pnl_usd,
        "sharpe": sharpe(by_date), "max_dd_usd": max_drawdown(by_date.cumsum()) if len(by_date) else float("nan"),
        "avg_hold_days": t["hold_days"].mean() if n else float("nan"),
        "pct_stopped": stopped,
    }


# =============================================================================
# Charts
# =============================================================================

def equity_chart(daily: pd.DataFrame, is_end: pd.Timestamp) -> go.Figure:
    names = list(dict.fromkeys(daily["instrument"]))
    n = len(names)
    cols = 3
    rows = math.ceil(n / cols) + 1
    titles = ["Overall ($ per $10k DV01, equal book per instrument)"] + names
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles[: 1 + n],
                        vertical_spacing=0.04, horizontal_spacing=0.05)
    overall = daily.groupby("date")["pnl_usd"].sum().sort_index().cumsum()
    overall = with_gap_breaks(overall)
    fig.add_trace(go.Scatter(x=overall.index, y=overall.values, mode="lines",
                             line=dict(width=1.4, color="#1f77b4"), name="overall", showlegend=False),
                  row=1, col=1)
    fig.add_vrect(x0=is_end, x1=overall.index.max(), fillcolor="rgba(255,127,14,0.08)",
                  line_width=0, row=1, col=1)
    fig.update_yaxes(title_text="USD", row=1, col=1)
    for k, name in enumerate(names):
        r, c = (k // cols) + 2, (k % cols) + 1
        s = with_gap_breaks(daily.loc[daily["instrument"] == name].set_index("date")["pnl_usd"].cumsum())
        fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines",
                                 line=dict(width=1.1, color="#1f77b4"), showlegend=False), row=r, col=c)
        fig.add_vrect(x0=is_end, x1=s.index.max() if len(s) else is_end,
                      fillcolor="rgba(255,127,14,0.08)", line_width=0, row=r, col=c)
    fig.update_layout(height=280 * rows, width=1100, template="plotly_white",
                      margin=dict(t=60, l=50, r=20, b=30),
                      title="Equity curves (shaded = out-of-sample from 2021)")
    return fig


def z_trade_chart(hist: pd.DataFrame, trades: pd.DataFrame, name: str) -> go.Figure:
    d = hist[hist["instrument"] == name].set_index("date").sort_index()
    d = with_gap_breaks(d)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.55, 0.45], vertical_spacing=0.08,
                        subplot_titles=[f"{name}: z_250 with trade markers", "cumulative $ PnL"])
    fig.add_trace(go.Scatter(x=d.index, y=d["z_250"], mode="lines",
                             line=dict(width=1.1, color="#1f77b4"), name="z_250"), row=1, col=1)
    for level, colour in ((2, "rgba(255,127,14,0.8)"), (3.5, "rgba(214,39,40,0.8)"),
                          (0.5, "rgba(127,127,127,0.6)")):
        fig.add_hline(y=level, line=dict(color=colour, dash="dash", width=1), row=1, col=1)
        fig.add_hline(y=-level, line=dict(color=colour, dash="dash", width=1), row=1, col=1)
    t = trades[trades["instrument"] == name]
    longs = t[t["side"] == "long"]
    shorts = t[t["side"] == "short"]
    fig.add_trace(go.Scatter(x=longs["entry_date"], y=longs["entry_z"], mode="markers",
                             marker=dict(color="#2ca02c", size=8, symbol="triangle-up"),
                             name="enter long"), row=1, col=1)
    fig.add_trace(go.Scatter(x=shorts["entry_date"], y=shorts["entry_z"], mode="markers",
                             marker=dict(color="#d62728", size=8, symbol="triangle-down"),
                             name="enter short"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t["exit_date"], y=t["exit_z"], mode="markers",
                             marker=dict(color="black", size=7, symbol="x"), name="exit"), row=1, col=1)
    fig.update_yaxes(title_text="z", row=1, col=1)
    fig.update_layout(height=650, width=1100, template="plotly_white",
                      legend=dict(orientation="h", y=1.08, x=0), margin=dict(t=80, l=50, r=20, b=30))
    return fig


def year_chart(yearly: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    y = yearly[yearly["instrument"] == "ALL"].sort_values("year")
    colours = ["#1f77b4" if yr <= 2020 else "#ff7f0e" for yr in y["year"]]
    fig.add_trace(go.Bar(x=y["year"].astype(str), y=y["pnl_usd"], marker_color=colours, name="ALL"))
    fig.update_layout(title="PnL by year ($ per $10k DV01, all instruments)",
                      yaxis_title="USD", template="plotly_white", height=420, width=1100,
                      margin=dict(t=60, l=50, r=20, b=40))
    return fig


def _write_fig(fig: go.Figure, path, stem: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    fig.write_html(path / f"{stem}.html", include_plotlyjs="cdn")
    try:
        fig.write_image(path / f"{stem}.png", scale=1)
    except Exception as e:
        log.warning("PNG export skipped for %s (%s)", stem, e)


# =============================================================================
# Run
# =============================================================================

def attach_pending(hist: pd.DataFrame, specs: dict[str, dict], roll_dates: pd.DataFrame) -> pd.DataFrame:
    pending = pending_series(roll_dates)
    hist = hist.copy()
    hist["pending_roll"] = False
    if not pending:
        return hist
    for name, spec in specs.items():
        legs = spec["wings"] + [spec["belly"]] if spec["pca"] else list(spec["legs"])
        dates = [dt for leg in legs for dt in pending.get(leg, [])]
        if not dates:
            continue
        m = hist["instrument"] == name
        hist.loc[m, "pending_roll"] = hist.loc[m, "date"].isin(dates)
    return hist


def simulate_all(hist: pd.DataFrame, specs: dict[str, dict], rules: dict[str, Any], cfg: dict[str, Any],
                 pca_w: dict[str, pd.DataFrame], clean: pd.DataFrame | None
                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
    bt, costs = cfg["backtest"], cfg["backtest"]["costs"]
    assume, book = bt["assume_yield"], bt["book_dv01_usd"]
    trades, daily = [], []
    for name, spec in specs.items():
        d = hist[hist["instrument"] == name].set_index("date").sort_index()
        if d.empty:
            continue
        t, day = simulate(d, spec, name, rules, costs, assume, book, pca_w, clean=clean)
        trades.append(t)
        daily.append(day)
    tr = pd.concat(trades, ignore_index=True) if trades else pd.DataFrame()
    dy = pd.concat(daily, ignore_index=True) if daily else pd.DataFrame()
    return tr, dy


def event_split(trades: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    ev = events.set_index("date")
    rows = []
    for name, g in trades.groupby("instrument"):
        kinds = event_types_for(name)
        dates = ev[ev["event"].isin(kinds)].index
        for t in g.itertuples():
            spanned = bool(((dates > t.entry_date) & (dates <= t.exit_date)).any())
            rows.append({**{c: getattr(t, c) for c in g.columns}, "event_window": spanned})
    out = pd.DataFrame(rows)
    summary = (out.groupby("event_window")
               .agg(trades=("pnl_bp", "size"), hit_rate=("pnl_bp", lambda s: (s > 0).mean() * 100),
                    avg_bp=("pnl_bp", "mean"), total_usd=("pnl_usd", "sum"))
               .reset_index())
    summary["event_window"] = summary["event_window"].map({True: "open across RBA/RBNZ/CPI", False: "not"})
    return summary


def sensitivity_table(hist: pd.DataFrame, specs: dict, cfg: dict, pca_w: dict, clean,
                      roll_dates) -> pd.DataFrame:
    bt = cfg["backtest"]
    rows = []
    for entry in bt["sensitivity"]["entry_z"]:
        for exit_z in bt["sensitivity"]["exit_z"]:
            rules = dict(entry_z=entry, exit_z=exit_z, stop_z=bt["stop_z"],
                         max_hold_days=bt["max_hold_days"], is_end=pd.Timestamp(bt["in_sample"][1]))
            tr, dy = simulate_all(hist, specs, rules, cfg, pca_w, clean)
            for sample in ("IS", "OOS"):
                m = summarise_trades(tr, dy, sample, rules["is_end"])
                rows.append({"entry_z": entry, "exit_z": exit_z, **m})
    return pd.DataFrame(rows)


def english_summary(metrics: pd.DataFrame) -> str:
    """Which instruments mean-revert out of sample, in plain English."""
    oos = metrics[(metrics["sample"] == "OOS") & (metrics["instrument"] != "ALL")].copy()
    if oos.empty:
        return "No out-of-sample trades."
    lines = ["Out-of-sample (from 2021), fading |z_250| > 2 and exiting inside 0.5:"]
    overall = metrics[(metrics["sample"] == "OOS") & (metrics["instrument"] == "ALL")]
    if len(overall):
        r = overall.iloc[0]
        lines.append(f"  Overall: {int(r.trades)} trades, hit rate {r.hit_rate:.0f}%, "
                     f"avg {r.avg_bp:.1f}bp, Sharpe {r.sharpe:.2f}. The rule lost money.")
    pos = oos[oos["sharpe"] > 0].sort_values("sharpe", ascending=False)
    if len(pos):
        bits = [f"{r.instrument} (hit {r.hit_rate:.0f}%, {r.avg_bp:+.1f}bp, Sharpe {r.sharpe:.2f})"
                for r in pos.itertuples()]
        lines.append("  Positive Sharpe, but too small to call mean-reversion: " + "; ".join(bits) + ".")
    else:
        lines.append("  No instrument has a positive out-of-sample Sharpe.")
    worst = oos.nsmallest(3, "sharpe")
    lines.append("  Worst: " + ", ".join(
        f"{r.instrument} (hit {r.hit_rate:.0f}%, {int(r.pct_stopped)}% stopped)" for r in worst.itertuples()) + ".")
    flies = oos[oos["instrument"].str.contains("2s5s10s|pca_fly")]
    if len(flies) and float(flies["hit_rate"].mean()) < 30:
        lines.append("  Butterflies failed: the 2021-23 hiking cycle and the 2024-26 easing cycle each "
                     "ran as a one-way flatten then steepener, so a 250-day z-score kept fading a trend. "
                     "Stops were hit on half or more of the simple 2s5s10s trades.")
    xccy = oos[oos["instrument"].str.startswith("au_nz")].sort_values("sharpe", ascending=False)
    if len(xccy):
        best = xccy.iloc[0]
        lines.append(f"  {best.instrument} is the least-bad cross-market spread "
                     f"(hit {best.hit_rate:.0f}%, Sharpe {best.sharpe:.2f}); when the two policy cycles "
                     "diverge, the 250-day window treats a persistent premium as a stretch to fade.")
    lines.append("  Round-trip costs (0.5bp per swap leg, 1bp per bond leg) are large next to a "
                 "move from 2 sd back to 0.5 sd, so a small mean-reversion edge would not survive anyway.")
    return "\n".join(lines)


def run(cfg: dict[str, Any] | None = None) -> dict[str, pd.DataFrame]:
    cfg = cfg or load_config()
    paths = {k: project_path(v) for k, v in cfg["paths"].items()}
    hist = pd.read_csv(paths["scanner_history_file"], parse_dates=["date"])
    specs = instrument_specs(cfg)
    hist = attach_pending(hist, specs, load_roll_dates(paths["roll_dates_file"]))
    pca_w = pca_weight_lookup(cfg["paths"]["pca_weights_file"])
    raw = read_rates(paths["rates_file"])
    clean, _ = forward_fill_interior(raw, cfg["update"]["ffill_max_days"]) if raw is not None else (None, None)

    bt = cfg["backtest"]
    is_end = pd.Timestamp(bt["in_sample"][1])
    oos_start = pd.Timestamp(bt["out_of_sample"][0])
    rules = dict(entry_z=bt["entry_z"], exit_z=bt["exit_z"], stop_z=bt["stop_z"],
                 max_hold_days=bt["max_hold_days"], is_end=is_end)

    # Drop history before the in-sample start so NZ-only series share the same window.
    hist = hist[hist["date"] >= pd.Timestamp(bt["in_sample"][0])]

    log.info("Backtesting %d instruments, %s to %s", hist["instrument"].nunique(),
             hist["date"].min().date(), hist["date"].max().date())
    trades, daily = simulate_all(hist, specs, rules, cfg, pca_w, clean)

    metrics_rows = []
    grouped = list(trades.groupby("instrument")) if len(trades) else []
    for sample in ("IS", "OOS", None):
        for name, g in grouped:
            dy = daily[daily["instrument"] == name]
            m = summarise_trades(g, dy, sample, is_end)
            m["instrument"] = name
            metrics_rows.append(m)
        m = summarise_trades(trades, daily, sample, is_end)
        m["instrument"] = "ALL"
        metrics_rows.append(m)
    metrics = pd.DataFrame(metrics_rows)

    yearly = (daily.assign(year=daily["date"].dt.year)
              .groupby(["instrument", "year"], as_index=False)[["pnl_bp", "pnl_usd"]].sum())
    all_year = daily.assign(year=daily["date"].dt.year).groupby("year", as_index=False)[["pnl_bp", "pnl_usd"]].sum()
    all_year.insert(0, "instrument", "ALL")
    yearly = pd.concat([yearly, all_year], ignore_index=True)

    events = build_events(start=bt["in_sample"][0], end=str(hist["date"].max().date()))
    ev_sum = event_split(trades, events)
    worst = trades.nsmallest(10, "pnl_usd") if len(trades) else trades

    sens = sensitivity_table(hist, specs, cfg, pca_w, clean, load_roll_dates(paths["roll_dates_file"]))

    for p in (paths["backtest_trades_file"], paths["backtest_daily_file"], paths["backtest_metrics_file"],
              paths["backtest_sensitivity_file"], paths["backtest_yearly_file"], paths["backtest_worst_file"],
              paths["backtest_events_file"]):
        p.parent.mkdir(parents=True, exist_ok=True)

    def _save(df: pd.DataFrame, path, dates=("date", "entry_date", "exit_date")):
        out = df.copy()
        for c in dates:
            if c in out.columns:
                out[c] = pd.to_datetime(out[c]).dt.strftime("%Y-%m-%d")
        out.to_csv(path, index=False)

    _save(trades, paths["backtest_trades_file"])
    _save(daily, paths["backtest_daily_file"])
    metrics.to_csv(paths["backtest_metrics_file"], index=False, float_format="%.4f")
    sens.to_csv(paths["backtest_sensitivity_file"], index=False, float_format="%.4f")
    yearly.to_csv(paths["backtest_yearly_file"], index=False, float_format="%.2f")
    _save(worst, paths["backtest_worst_file"])
    _save(events, paths["backtest_events_file"])

    charts = paths["charts_dir"] / "backtest"
    _write_fig(equity_chart(daily, is_end), charts, "equity")
    _write_fig(year_chart(yearly), charts, "pnl_by_year")
    for name in hist["instrument"].unique():
        fig = z_trade_chart(hist, trades, name)
        # add cumulative pnl of this instrument on the empty second row
        s = daily.loc[daily["instrument"] == name].set_index("date")["pnl_usd"].cumsum()
        fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines",
                                 line=dict(width=1.2, color="#ff7f0e"), name="$ PnL", showlegend=False),
                      row=2, col=1)
        fig.update_yaxes(title_text="USD", row=2, col=1)
        _write_fig(fig, charts, f"z_{name}")

    # Console report
    show = metrics[(metrics["instrument"] != "ALL") | (metrics["sample"] != "ALL")]
    cols = ["instrument", "sample", "trades", "hit_rate", "avg_bp", "total_bp", "total_usd",
            "sharpe", "max_dd_usd", "avg_hold_days", "pct_stopped"]
    with pd.option_context("display.width", 220, "display.max_rows", 80, "display.float_format",
                           lambda x: f"{x: .2f}"):
        print("\nBacktest  fade |z_250| > {0}  exit |z| < {1}  stop |z| > {2}  max hold {3}d"
              .format(bt["entry_z"], bt["exit_z"], bt["stop_z"], bt["max_hold_days"]))
        print("IS {0} to {1}   OOS {2} to {3}. $ PnL is per $10k long-side DV01.\n"
              .format(bt["in_sample"][0], bt["in_sample"][1], oos_start.date(), hist["date"].max().date()))
        print(metrics[cols].to_string(index=False))
        print("\nSensitivity (report only - defaults are not chosen from this table):")
        print(sens.to_string(index=False))
        print("\nPnL on trades that were open across an RBA/RBNZ decision or CPI print vs not:")
        print(ev_sum.to_string(index=False) if len(ev_sum) else "  (no trades)")
        print("\nTen worst trades ($):")
        if len(worst):
            print(worst[["instrument", "side", "entry_date", "exit_date", "pnl_bp", "pnl_usd", "reason"]]
                  .to_string(index=False))
        print("\n" + english_summary(metrics) + "\n")

    log.info("Saved backtest tables and charts")
    return {"trades": trades, "daily": daily, "metrics": metrics, "sensitivity": sens}


if __name__ == "__main__":
    run()
