"""
fetch.py - get central-bank yield data and maintain data/processed/rates.csv.

What it does, in order:
1. RBA table F2 (Australian government bond yields): downloaded automatically with
   plain HTTP requests. If the download fails, it looks for an F2 file in data/manual/.
2. RBNZ table B2 (NZ bond yields and swap rates): RBNZ restricts automated access,
   so these files are downloaded by hand and placed in data/manual/.
3. Every column is checked against the source's own description (e.g. the RBA
   "Title" row must say "10 year bond", not "Indexed Bond") before it is used.
4. The new data is "upserted" into rates.csv:
     - new dates are added and previously missing values are filled in;
     - values in the most recent 30 calendar days may be overwritten (sources revise);
     - older history is never changed - differences are logged as warnings.
5. Validation runs (see validate.py). If anything fails badly, rates.csv is left untouched.

rates.csv holds the values exactly as published, in PERCENT, with gaps left empty.
Forward-filling of small gaps happens when the data is read (load_clean_rates), so
the stored file always matches the source.

Run from the project root:   python -m src.fetch
"""

from __future__ import annotations

import csv
import fnmatch
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from src.utils import get_logger, load_config, project_path, read_rates, write_rates
from src.validate import validate

log = get_logger("fetch")


class SourceError(Exception):
    """Raised when a source cannot be downloaded, parsed or trusted."""


@dataclass
class SourceResult:
    """Everything we got from one central bank on this run."""

    data: pd.DataFrame                 # index = date, columns = clean names, values in percent
    meta: list[dict[str, Any]]         # one dict per clean name: id, original name, units...
    files: list[str] = field(default_factory=list)
    method: str = "download"           # "download" or "manual"
    published: str | None = None       # publication date stated inside the source file


# =============================================================================
# Downloading (RBA only)
# =============================================================================

def http_get(url: str, dest: Path, http_cfg: dict[str, Any]) -> Path:
    """Download `url` to `dest`, retrying politely with exponential backoff.

    Raises SourceError if every attempt fails or the server returns an HTML page
    (which usually means an error or access-restriction page, not data).
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": http_cfg["user_agent"]}
    wait = http_cfg["backoff_seconds"]
    last_error: Exception | None = None
    for attempt in range(1, http_cfg["max_retries"] + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=http_cfg["timeout_seconds"])
            resp.raise_for_status()
            if "text/html" in resp.headers.get("content-type", ""):
                raise SourceError(f"{url} returned an HTML page instead of a data file")
            dest.write_bytes(resp.content)
            log.info("Downloaded %s (%.0f KB)", url, len(resp.content) / 1024)
            return dest
        except (requests.RequestException, SourceError) as e:
            last_error = e
            log.warning("Attempt %d/%d failed for %s: %s", attempt, http_cfg["max_retries"], url, e)
            if attempt < http_cfg["max_retries"]:
                time.sleep(wait)
                wait *= 2
    raise SourceError(f"Could not download {url}: {last_error}")


def find_manual_files(manual_dir: Path, patterns: list[str]) -> list[Path]:
    """Return files in data/manual/ matching any of the glob patterns (case-insensitive)."""
    if not manual_dir.exists():
        return []
    return sorted(
        p for p in manual_dir.iterdir()
        if p.is_file() and any(fnmatch.fnmatch(p.name.lower(), pat.lower()) for pat in patterns)
    )


# =============================================================================
# Parsing RBA F2
# =============================================================================

def parse_rba_csv(path: Path, series_cfg: dict[str, dict]) -> tuple[pd.DataFrame, list[dict], str | None]:
    """Parse the current RBA F2 CSV.

    File layout: ~11 metadata rows ("Title", "Description", "Units", ...,
    "Publication date", "Series ID"), then one row per date like "28-Oct-2026,4.951,...".
    Returns (data in percent, metadata per series, publication date).
    """
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.reader(f)]

    labels = {r[0].strip(): i for i, r in enumerate(rows[:30]) if r}
    for needed in ("Title", "Series ID"):
        if needed not in labels:
            raise SourceError(f"{path.name}: no '{needed}' row found - has the RBA changed the format?")
    ids = [c.strip() for c in rows[labels["Series ID"]]]
    titles = rows[labels["Title"]]
    units = rows[labels["Units"]] if "Units" in labels else [""] * len(ids)
    published = None
    if "Publication date" in labels:
        published = str(pd.to_datetime(rows[labels["Publication date"]][1], format="%d-%b-%Y").date())

    cols, meta = {}, []
    for clean, spec in series_cfg.items():
        if spec["id"] not in ids:
            raise SourceError(f"{path.name}: series {spec['id']} ({clean}) not found")
        j = ids.index(spec["id"])
        title = titles[j]
        if spec["must_contain"].lower() not in title.lower() or "indexed" in title.lower():
            raise SourceError(f"{path.name}: {spec['id']} is titled '{title}', expected '{spec['must_contain']}'")
        cols[clean] = j
        meta.append({"clean_name": clean, "series_id": spec["id"], "original_name": title,
                     "units": units[j], "source": "RBA", "table": "F2"})

    records = []
    for r in rows[labels["Series ID"] + 1:]:
        if not r or not r[0].strip():
            continue
        try:
            d = datetime.strptime(r[0].strip(), "%d-%b-%Y")
        except ValueError:
            continue  # not a data row
        records.append([d] + [_to_float(r[j]) if j < len(r) else np.nan for j in cols.values()])

    data = pd.DataFrame(records, columns=["date", *cols]).set_index("date").sort_index()
    return data, meta, published


def parse_rba_xls(path: Path, series_cfg: dict[str, dict]) -> pd.DataFrame:
    """Parse the RBA F2 history file (1995 to 17-May-2013).

    Layout: metadata rows, a "Mnemonic" row holding the series IDs, then data.
    The tenor labels ("2 yrs", "Indexed") sit a few rows above the Mnemonic row.
    """
    raw = pd.read_excel(path, header=None, engine="calamine")
    text = raw.fillna("").astype(str)
    first_col = text[0].str.strip()
    if not (first_col == "Mnemonic").any():
        raise SourceError(f"{path.name}: no 'Mnemonic' row found")
    hdr = int(np.flatnonzero(first_col == "Mnemonic")[0])
    ids = text.iloc[hdr].str.strip().tolist()
    tenor_labels = text.iloc[:hdr].apply(lambda c: " ".join(v for v in c if "yrs" in v or "Indexed" in v))

    cols = {}
    for clean, spec in series_cfg.items():
        if spec["id"] not in ids:
            raise SourceError(f"{path.name}: series {spec['id']} ({clean}) not found")
        j = ids.index(spec["id"])
        if "Indexed" in tenor_labels[j]:
            raise SourceError(f"{path.name}: {spec['id']} looks like an indexed bond")
        cols[clean] = j

    body = raw.iloc[hdr + 1:]
    data = pd.DataFrame({clean: pd.to_numeric(body[j], errors="coerce") for clean, j in cols.items()})
    data.index = pd.to_datetime(body[0], errors="coerce")
    data = data[data.index.notna()]
    data.index.name = "date"
    return data.sort_index()


def get_rba(cfg: dict[str, Any], existing: pd.DataFrame | None) -> SourceResult:
    """Download (or, as a fallback, read manually supplied) RBA F2 data."""
    rcfg, series_cfg = cfg["rba"], cfg["rba"]["series"]
    raw_dir = project_path(cfg["paths"]["raw_dir"])

    try:
        current = http_get(rcfg["current_url"], raw_dir / "f2-data.csv", cfg["http"])
        method = "download"
    except SourceError as e:
        log.error("%s", e)
        manual = [p for p in find_manual_files(project_path(cfg["paths"]["manual_dir"]), rcfg["manual_patterns"])
                  if p.suffix.lower() == ".csv"]
        if not manual:
            raise SourceError("RBA download failed and no manual F2 CSV in data/manual/") from e
        current, method = manual[-1], "manual"
        log.warning("Using manual RBA file %s", current.name)

    data, meta, published = parse_rba_csv(current, series_cfg)
    files = [current.name]

    # The 1995-2013 history file only needs fetching until rates.csv contains it.
    history_end = pd.Timestamp(rcfg["history_end"])
    have_history = (existing is not None and "au_gov_10y" in existing
                    and existing["au_gov_10y"].first_valid_index() is not None
                    and existing["au_gov_10y"].first_valid_index() <= history_end)
    if not have_history:
        hist_path = raw_dir / "f02dhist.xls"
        if not hist_path.exists():
            http_get(rcfg["history_url"], hist_path, cfg["http"])
        hist = parse_rba_xls(hist_path, series_cfg)
        data = data.combine_first(hist)   # the current file wins on any overlapping date
        files.append(hist_path.name)

    return SourceResult(data=data, meta=meta, files=files, method=method, published=published)


# =============================================================================
# Parsing RBNZ B2 (manual files)
# =============================================================================

def parse_rbnz_file(path: Path, series_cfg: dict[str, dict], optional: set[str] | None = None
                    ) -> tuple[pd.DataFrame, list[dict], str | None]:
    """Parse one RBNZ B2 file (.xlsx as downloaded, or the 'Data' sheet saved as .csv).

    Layout of the 'Data' sheet: 5 header rows - Group, Series, Notes, Unit, Series Id -
    then one row per date with the date in the first column.
    `optional` clean names (PCA extra tenors) are skipped if this file does not have them.
    Returns (data in percent, metadata per series, publication date or None).
    """
    if path.suffix.lower() == ".csv":
        raw = pd.read_csv(path, header=None)
    else:
        raw = pd.read_excel(path, sheet_name="Data", header=None, engine="calamine")

    text = raw.fillna("").astype(str)
    first_col = text[0].str.strip().str.lower()
    if not (first_col == "series id").any():
        raise SourceError(f"{path.name}: no 'Series Id' row found - is this the B2 'Data' sheet?")
    sid_row = int(np.flatnonzero(first_col == "series id")[0])
    ids = text.iloc[sid_row].str.strip().tolist()
    groups = text.iloc[sid_row - 4].str.strip().tolist()
    names = text.iloc[sid_row - 3].str.strip().tolist()
    units = text.iloc[sid_row - 1].str.strip().tolist()

    optional = set(optional or [])
    cols, meta = {}, []
    for clean, spec in series_cfg.items():
        if spec["id"] not in ids:
            if clean in optional:
                log.info("%s: optional series %s (%s) not in this file — skipped", path.name, spec["id"], clean)
                continue
            raise SourceError(f"{path.name}: series {spec['id']} ({clean}) not found")
        j = ids.index(spec["id"])
        if spec["group_contains"].lower() not in groups[j].lower() or names[j].lower() != spec["name_is"].lower():
            raise SourceError(
                f"{path.name}: {spec['id']} is '{groups[j]} / {names[j]}', expected "
                f"'{spec['group_contains']}' / '{spec['name_is']}'"
            )
        cols[clean] = j
        meta.append({"clean_name": clean, "series_id": spec["id"], "original_name": f"{groups[j]} - {names[j]}",
                     "units": units[j], "source": "RBNZ", "table": "B2"})

    body = raw.iloc[sid_row + 1:]
    data = pd.DataFrame({clean: pd.to_numeric(body[j], errors="coerce") for clean, j in cols.items()})
    data.index = pd.to_datetime(body[0], errors="coerce")
    data = data[data.index.notna()]
    data.index.name = "date"

    published = None
    if path.suffix.lower() != ".csv":
        try:
            desc = pd.read_excel(path, sheet_name="Table Description", header=None, engine="calamine")
            hit = np.argwhere(desc.fillna("").astype(str).apply(lambda c: c.str.strip()).values == "Published Date")
            if len(hit):
                r, c = hit[0]
                # the value sits to the right of the label (older layouts: below it)
                for value in (desc.iat[r, c + 1] if c + 1 < desc.shape[1] else None,
                              desc.iat[r + 1, c] if r + 1 < desc.shape[0] else None):
                    stamp = pd.to_datetime(value, errors="coerce")
                    if pd.notna(stamp):
                        published = str(stamp.date())
                        break
        except (ValueError, KeyError):
            pass  # description sheet missing: fall back to the last data date below
    return data.sort_index(), meta, published


def get_rbnz(cfg: dict[str, Any]) -> SourceResult:
    """Read every B2 file in data/manual/ and combine them.

    Normally there are two files: the 1985-2017 history and the 2018-current file.
    Files are combined oldest-first by their stated publication date, so on any
    overlapping date the most recently published file wins.
    """
    rcfg = cfg["rbnz"]
    files = find_manual_files(project_path(cfg["paths"]["manual_dir"]), rcfg["manual_patterns"])
    if not files:
        raise SourceError("No RBNZ B2 file found in data/manual/ (see README: manual NZ data step)")

    parsed = []
    for p in files:
        series_cfg = {**rcfg["series"], **rcfg.get("pca_series", {})}
        data, meta, published = parse_rbnz_file(p, series_cfg, optional=set(rcfg.get("pca_series", {})))
        published = published or str(data.index.max().date())
        parsed.append((published, p.name, data, meta))
        log.info("RBNZ file %s: published %s, data %s to %s",
                 p.name, published, data.index.min().date(), data.index.max().date())
    parsed.sort(key=lambda t: t[0])

    combined = parsed[0][2]
    for _, _, data, _ in parsed[1:]:
        combined = data.combine_first(combined)
    return SourceResult(data=combined, meta=parsed[-1][3], files=[t[1] for t in parsed],
                        method="manual", published=parsed[-1][0])


# =============================================================================
# Upsert and clean loading
# =============================================================================

def upsert(existing: pd.DataFrame | None, incoming: pd.DataFrame, today: pd.Timestamp,
           window_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge freshly parsed data into the stored history.

    Rules (per date and series):
      - stored value missing, source has one       -> fill it in
      - both present and different, date is recent  -> take the source value (revision)
      - both present and different, date is older   -> keep the stored value, log a conflict
      - source missing a value we already have      -> keep the stored value
    "Recent" means within `window_days` calendar days before `today`.

    Returns (merged data, event log with one row per changed or conflicting cell).
    """
    if existing is None:
        existing = pd.DataFrame(index=pd.DatetimeIndex([], name="date"))
    cols = list(dict.fromkeys([*existing.columns, *incoming.columns]))
    dates = existing.index.union(incoming.index)
    old = existing.reindex(index=dates, columns=cols)
    new = incoming.reindex(index=dates, columns=cols)

    recent = pd.Series(dates >= today - pd.Timedelta(days=window_days), index=dates)
    recent_2d = pd.DataFrame(np.repeat(recent.values[:, None], len(cols), axis=1), index=dates, columns=cols)

    fill = old.isna() & new.notna()
    differ = old.notna() & new.notna() & (old.round(6) != new.round(6))
    overwrite = differ & recent_2d
    conflict = differ & ~recent_2d

    merged = old.mask(fill | overwrite, new)
    merged = merged.dropna(how="all")
    merged.index.name = "date"

    events = []
    for action, mask in (("filled", fill), ("revised_recent", overwrite), ("conflict_kept_old", conflict)):
        stacked = mask.stack()
        for (d, s) in stacked[stacked].index:
            events.append({"date": d, "series": s, "old": old.at[d, s], "new": new.at[d, s], "action": action})
    return merged, pd.DataFrame(events, columns=["date", "series", "old", "new", "action"])


def forward_fill_interior(df: pd.DataFrame, limit: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Forward-fill gaps of at most `limit` rows, but only INSIDE each series' history.

    A series is never extended past its last real observation (e.g. AU data that
    the RBA hasn't published yet stays empty). Returns (filled data, mask of gaps
    that remain empty after filling, i.e. the gaps that need flagging).
    """
    filled = df.ffill(limit=limit)
    inside = pd.DataFrame(False, index=df.index, columns=df.columns)
    for col in df.columns:
        first, last = df[col].first_valid_index(), df[col].last_valid_index()
        if last is None:
            continue
        filled.loc[filled.index > last, col] = np.nan
        inside[col] = (df.index >= first) & (df.index <= last)
    remaining_gaps = filled.isna() & inside
    return filled, remaining_gaps


def load_clean_rates(cfg: dict[str, Any] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read rates.csv and apply the interior forward-fill. This is what later steps use.

    Returns (filled rates in percent, mask of remaining gaps).
    """
    cfg = cfg or load_config()
    rates = read_rates(project_path(cfg["paths"]["rates_file"]))
    if rates is None:
        raise FileNotFoundError("rates.csv not found - run `python -m src.fetch` first")
    return forward_fill_interior(rates, cfg["update"]["ffill_max_days"])


def last_dates(rates: pd.DataFrame) -> dict[str, pd.Timestamp | None]:
    """Last date with a real (published) value, per series."""
    return {c: rates[c].last_valid_index() for c in rates.columns}


# =============================================================================
# Main run
# =============================================================================

def _to_float(x: str) -> float:
    x = x.strip()
    return float(x) if x else np.nan


def build_series_meta(rates: pd.DataFrame, metas: list[dict]) -> pd.DataFrame:
    """One row per clean series: where it comes from and how much data it has."""
    meta = pd.DataFrame(metas).drop_duplicates("clean_name").set_index("clean_name").reindex(rates.columns)
    meta["first_date"] = [rates[c].first_valid_index() for c in rates.columns]
    meta["last_date"] = [rates[c].last_valid_index() for c in rates.columns]
    meta["n_obs"] = rates.count()
    meta["stored_units"] = "percent per annum"
    for c in ("first_date", "last_date"):
        meta[c] = pd.to_datetime(meta[c]).dt.strftime("%Y-%m-%d")
    meta.index.name = "clean_name"
    return meta


def print_summary(rates: pd.DataFrame, meta: pd.DataFrame, ffill_limit: int) -> None:
    """Print the mapping (clean name -> source series) and data coverage."""
    _, gaps = forward_fill_interior(rates, ffill_limit)
    table = pd.DataFrame({
        "source": meta["source"] + " " + meta["table"],
        "series_id": meta["series_id"],
        "original_name": meta["original_name"].str.slice(0, 52),
        "first": meta["first_date"],
        "last": meta["last_date"],
        "n_obs": meta["n_obs"],
        "gaps_after_ffill": gaps.sum(),
    })
    with pd.option_context("display.width", 250, "display.max_colwidth", 60):
        print(f"\nrates.csv: {rates.shape[0]} dates x {rates.shape[1]} series, "
              f"{rates.index.min().date()} to {rates.index.max().date()} (values in percent)\n")
        print(table.to_string())
        print()


def run(cfg: dict[str, Any] | None = None, today: pd.Timestamp | None = None) -> int:
    """Fetch, upsert, validate and save. Returns 0 on success, 1 if nothing was saved."""
    cfg = cfg or load_config()
    today = (today or pd.Timestamp.today()).normalize()
    paths = {k: project_path(v) for k, v in cfg["paths"].items()}
    existing = read_rates(paths["rates_file"])

    results: dict[str, SourceResult] = {}
    failed = []
    for name, getter in (("rba", lambda: get_rba(cfg, existing)), ("rbnz", lambda: get_rbnz(cfg))):
        try:
            results[name] = getter()
        except SourceError as e:
            log.error("%s: %s", name.upper(), e)
            failed.append(name)
    if failed:
        log.error("Source(s) failed: %s. rates.csv left unchanged (previous data kept).", ", ".join(failed))
        return 1

    order = [*cfg["rba"]["series"], *cfg["rbnz"]["series"], *cfg["rbnz"].get("pca_series", {})]
    incoming = pd.concat([r.data for r in results.values()], axis=1, sort=True).reindex(columns=order)
    merged, events = upsert(existing, incoming, today, cfg["update"]["overwrite_window_days"])
    merged = merged.reindex(columns=list(dict.fromkeys([*order, *merged.columns])))

    report = validate(merged, cfg, today, events)
    paths["validation_file"].parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(paths["validation_file"], index=False)
    if (report["severity"] == "error").any():
        log.error("Validation found errors (see %s). rates.csv left unchanged.", paths["validation_file"].name)
        return 1

    write_rates(merged, paths["rates_file"])
    meta = build_series_meta(merged, [m for r in results.values() for m in r.meta])
    meta.to_csv(paths["series_meta_file"])
    status = {
        "run_time_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        **{name: {"method": r.method, "files": r.files, "published": r.published,
                  "last_data_date": str(max(d for d in last_dates(r.data).values() if d is not None).date())}
           for name, r in results.items()},
    }
    paths["source_status_file"].write_text(json.dumps(status, indent=2), encoding="utf-8")

    counts = events["action"].value_counts().to_dict() if len(events) else {}
    log.info("Saved %s: %d rows x %d series. Changes: %s", paths["rates_file"].name, *merged.shape, counts or "none")
    print_summary(merged, meta, cfg["update"]["ffill_max_days"])
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
