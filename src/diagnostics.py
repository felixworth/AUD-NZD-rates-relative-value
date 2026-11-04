"""
diagnostics.py - one-off data checks that support decisions in the README.

rba_join_check: the RBA F2 history file ends on 17-May-2013 and the current file
starts on 20-May-2013 (10y) / 2-Sep-2013 (2y, 3y, 5y). The two files may use
different methods (the history file's series are described as "N years maturity",
the current file's as "interpolated, N years maturity"), so we look for a jump
at the join that NZ swaps don't explain.

For each AU tenor it shows the 10 published values either side of the join, the
day-on-day change, the change in the matching NZ swap, and the AU-NZ government
bond spread. A change is flagged if |AU change| > 5bp AND |AU change - NZ swap change| > 5bp.
NZ 3y swaps aren't in our data, so AU 3y is compared with 2/3 x NZ 2y + 1/3 x NZ 5y swap.
AU-NZ spread LEVELS use raw NZ bond yields (what the market quoted); spread CHANGES
use back-adjusted NZ yields, because a confirmed NZ 5y roll on 1-Aug-2013 falls
inside the 2013 gap.

event_window_check: the same comparison around dates when RBA changed the F2
methodology inside the current file (31-Mar-2023 and 29-Dec-2023).

Run from the project root:
  python -m src.diagnostics                                 2013 file join
  python -m src.diagnostics --events 2023-03-31 2023-12-29  methodology-change dates
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.utils import get_logger, load_config, project_path, read_rates

log = get_logger("diagnostics")

SWAP_FOR = {"2y": {"nz_swap_2y": 1.0}, "3y": {"nz_swap_2y": 2 / 3, "nz_swap_5y": 1 / 3},
            "5y": {"nz_swap_5y": 1.0}, "10y": {"nz_swap_10y": 1.0}}


def rba_join_check(rates: pd.DataFrame, adjusted: pd.DataFrame, history_end: str, n: int = 10,
                   flag_bp: float = 5.0) -> pd.DataFrame:
    """Table of AU yields around the 2013 file join, with unexplained moves flagged."""
    end = pd.Timestamp(history_end)
    tables = []
    for tenor, weights in SWAP_FOR.items():
        au = rates[f"au_gov_{tenor}"].dropna()
        before, after = au[au.index <= end].tail(n + 1), au[au.index > end].head(n)
        window = pd.concat([before, after])
        swap = sum(w * rates[s] for s, w in weights.items()).ffill(limit=2).reindex(window.index)
        t = pd.DataFrame({"tenor": tenor, "file": ["history"] * len(before) + ["current"] * len(after),
                          "au_yield_pct": window})
        t["au_change_bp"] = window.diff() * 100
        t["nz_swap_change_bp"] = swap.diff() * 100
        t["unexplained_bp"] = t["au_change_bp"] - t["nz_swap_change_bp"]
        if f"nz_gov_{tenor}" in rates:
            t["au_nz_spread_raw_bp"] = (window - rates[f"nz_gov_{tenor}"].reindex(window.index)) * 100
            adj_spread = (window - adjusted[f"nz_gov_{tenor}_adj"].reindex(window.index)) * 100
            t["au_nz_spread_change_bp"] = adj_spread.diff()   # change measured with roll-adjusted NZ
        else:
            t["au_nz_spread_raw_bp"] = t["au_nz_spread_change_bp"] = float("nan")
        t["flag"] = ((t["au_change_bp"].abs() > flag_bp) & (t["unexplained_bp"].abs() > flag_bp)).map({True: "FLAG", False: ""})
        t["note"] = ""
        t.loc[t.index[len(before)], "note"] = f"first value after join (previous value {before.index[-1].date()})"
        tables.append(t.iloc[1:])          # first row only provides the starting value for diff()
    out = pd.concat(tables)
    out.index.name = "date"
    return out.round(1)


def event_window_check(rates: pd.DataFrame, event: str, n: int = 10, flag_bp: float = 5.0) -> pd.DataFrame:
    """AU yields vs NZ swaps for `n` published AU values before `event` and `n` from it onwards.

    Used for dates when RBA changed the F2 methodology inside a single continuous file
    (31-Mar-2023, 29-Dec-2023). Same flag as the join check: |AU change| > flag_bp AND
    |AU change - NZ swap change| > flag_bp.
    """
    day = pd.Timestamp(event)
    tables = []
    for tenor, weights in SWAP_FOR.items():
        au = rates[f"au_gov_{tenor}"].dropna()
        before, after = au[au.index < day].tail(n + 1), au[au.index >= day].head(n)
        window = pd.concat([before, after])
        swap = sum(w * rates[s] for s, w in weights.items()).ffill(limit=2).reindex(window.index)
        t = pd.DataFrame({"event": event, "tenor": tenor,
                          "side": ["before"] * len(before) + ["from event"] * len(after),
                          "au_yield_pct": window,
                          "nz_swap_pct": swap})
        t["au_change_bp"] = window.diff() * 100
        t["nz_swap_change_bp"] = swap.diff() * 100
        t["unexplained_bp"] = t["au_change_bp"] - t["nz_swap_change_bp"]
        t["flag"] = ((t["au_change_bp"].abs() > flag_bp) & (t["unexplained_bp"].abs() > flag_bp)).map({True: "FLAG", False: ""})
        tables.append(t.iloc[1:])
    out = pd.concat(tables)
    out.index.name = "date"
    return out.round(3)


def baseline_flag_rate(rates: pd.DataFrame, start: str, end: str, flag_bp: float = 5.0) -> pd.Series:
    """How often the same flag fires on ordinary days (for comparison with the join)."""
    rates = rates.loc[start:end]
    rate = {}
    for tenor, weights in SWAP_FOR.items():
        pair = pd.concat([rates[f"au_gov_{tenor}"], sum(w * rates[s] for s, w in weights.items())], axis=1).dropna()
        d = pair.diff() * 100
        flag = (d.iloc[:, 0].abs() > flag_bp) & ((d.iloc[:, 0] - d.iloc[:, 1]).abs() > flag_bp)
        rate[tenor] = flag.mean()
    return pd.Series(rate, name=f"share of days flagged {start} to {end}")


def run_events(events: list[str], cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    """Event-window check around RBA F2 methodology-change dates; saves and prints the result."""
    cfg = cfg or load_config()
    rates = read_rates(project_path(cfg["paths"]["rates_file"]))
    table = pd.concat([event_window_check(rates, e) for e in events])
    out_dir = project_path(cfg["paths"]["diagnostics_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "rba_method_change_windows.csv")

    with pd.option_context("display.width", 250, "display.max_rows", 200):
        for (event, tenor), t in table.groupby(["event", "tenor"], sort=False):
            print(f"\n=== AU {tenor} around {event} ===")
            print(t.drop(columns=["event", "tenor"]).assign(date=t.index.date).set_index("date").to_string())
        print("\nFlags per event and tenor:")
        print(table.assign(flagged=table["flag"] == "FLAG").groupby(["event", "tenor"], sort=False)["flagged"]
              .sum().unstack().to_string())
        print("\nBaseline - share of ordinary days flagged in 2023-2024:")
        print((baseline_flag_rate(rates, "2023-01-01", "2024-12-31") * 100).round(1).astype(str).add("%").to_string())
    log.info("Saved %s", out_dir / "rba_method_change_windows.csv")
    return table


def run(cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    rates = read_rates(project_path(cfg["paths"]["rates_file"]))
    adjusted = read_rates(project_path(cfg["paths"]["rates_adjusted_file"]))
    table = rba_join_check(rates, adjusted, cfg["rba"]["history_end"])
    out_dir = project_path(cfg["paths"]["diagnostics_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "rba_2013_join.csv")

    with pd.option_context("display.width", 250, "display.max_rows", 200):
        for tenor, t in table.groupby("tenor", sort=False):
            print(f"\n=== AU {tenor} around the 2013 file join ===")
            print(t.drop(columns="tenor").assign(date=t.index.date).set_index("date").to_string())
        print()
        base = pd.concat([baseline_flag_rate(rates, "2011-01-01", "2013-05-17"),
                          baseline_flag_rate(rates, "2013-09-02", "2015-12-31")], axis=1)
        print("Baseline - share of ordinary days that would be flagged:")
        print((base * 100).round(1).astype(str).add("%").to_string())
    log.info("Saved %s", out_dir / "rba_2013_join.csv")
    return table


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="One-off data diagnostics")
    parser.add_argument("--events", nargs="+", metavar="DATE",
                        help="check AU vs NZ swaps around these dates instead of the 2013 file join")
    args = parser.parse_args()
    run_events(args.events) if args.events else run()
