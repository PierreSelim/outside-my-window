from __future__ import annotations

import json
import os
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum, StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Any

import polars as pl
import requests


def _bundle_dir() -> Path:
    """Root holding the read-only `data/` directory: the PyInstaller bundle when frozen, else the repo."""
    meipass = getattr(sys, "_MEIPASS", None)
    return Path(meipass) if meipass else Path(__file__).parent.parent


def _default_cache_dir() -> Path:
    """Downloads live beside the code in a checkout, and in a writable per-user directory when packaged."""
    override = os.environ.get("OMW_CACHE_DIR")
    if override:
        return Path(override)
    return _bundle_dir() / "data" / "cache"


# Stable data.gouv.fr permalink: redirects to whatever storage host is live, so a
# host migration never breaks us. Rebuild data/resources.json with
# scripts/build_resource_index.py if the dataset re-issues resources with new ids.
DATAGOUV_RESOURCE_URL = "https://www.data.gouv.fr/api/1/datasets/r"
BUNDLE_DIR = _bundle_dir()
DATA_DIR = BUNDLE_DIR / "data"
ASSETS_DIR = BUNDLE_DIR / "assets"
CACHE_DIR = _default_cache_dir()

# (dept, period.value) → data.gouv.fr resource-id, built by scripts/build_resource_index.py
_RESOURCE_INDEX: dict[str, dict[str, str]] = json.loads((DATA_DIR / "resources.json").read_text(encoding="utf-8"))

# Columns we actually need for visualisation — the raw files have 60 columns
KEEP_COLS: list[str] = [
    "NUM_POSTE",
    "NOM_USUEL",
    "LAT",
    "LON",
    "ALTI",
    "AAAAMMJJ",
    "RR",
    "TN",
    "TX",
    "FFM",
    "FXY",
]

# Coordinates stay Float64; measurements are reported to 0.1 and a department is ~1.5 M rows,
# so Float32 halves the resident size of a cached department at no readable precision cost.
_PARSE_COORD_COLS: list[str] = ["LAT", "LON"]
_PARSE_MEASURE_COLS: list[str] = ["RR", "TN", "TX", "FFM", "FXY"]
_PARSE_INT_COLS: list[str] = ["NUM_POSTE", "ALTI"]

# Mapping from raw dataset column names to human-readable equivalents
COLUMN_RENAME: dict[str, str] = {
    "NUM_POSTE": "station_id",
    "NOM_USUEL": "station_name",
    "LAT": "lat",
    "LON": "lon",
    "ALTI": "altitude",
    "TN": "temp_min",
    "TX": "temp_max",
    "RR": "precipitation",
    "FFM": "wind_mean",
    "FXY": "wind_gust",
}

# Hot-day thresholds shared across charts and analytics
HOT_DAY_TMIN: float = 20.0
HOT_DAY_TMAX: float = 35.0


class Period(StrEnum):
    HISTORICAL = "1852-1949"
    MODERN = "previous-1950-2024"
    LATEST = "latest-2025-2026"


@dataclass(frozen=True)
class Truncated:
    """Configuration for one granularity, the value type of the Granularity members.

    `truncate_expr` is None for the identity granularity: there is no Polars truncation string
    that means "leave the daily rows alone". Callers use Granularity, not this type.
    """

    label: str
    truncate_expr: str | None
    title_suffix: str


class Granularity(Enum):
    """Granularity levels for time-series aggregation.

    DAY is the identity — no truncation applied. WEEK and MONTH collapse daily rows to coarser
    periods, each measurement using the operation that preserves its meaning (see `aggregate`).
    """

    DAY = Truncated("day", None, "")
    WEEK = Truncated("week", "1w", " (weekly)")
    MONTH = Truncated("month", "1mo", " (monthly)")

    @property
    def label(self) -> str:
        return self.value.label

    @property
    def truncate_expr(self) -> str | None:
        return self.value.truncate_expr

    @property
    def title_suffix(self) -> str:
        return self.value.title_suffix

    def per_unit(self, unit: str) -> str:
        """Axis unit for a quantity that accumulates over the period: `mm` → `mm/month`."""
        return unit if self is Granularity.DAY else f"{unit}/{self.label}"


_GRANULARITY_BY_LABEL: dict[str, Granularity] = {g.label: g for g in Granularity}


def granularity_from(value: str) -> Granularity:
    """Parse a RadioItems string value into a Granularity.

    Returns Granularity.DAY for any unrecognised value.
    """
    return _GRANULARITY_BY_LABEL.get(value, Granularity.DAY)


@dataclass(frozen=True)
class Station:
    station_id: int
    name: str
    lat: float
    lon: float
    altitude: int


_META_COLS: list[str] = ["lat", "lon", "altitude"]


def _total_or_null(col: str) -> pl.Expr:
    """Sum, but an all-null group stays null: `sum()` alone reports 0 mm for a station that measures no rain."""
    return pl.when(pl.col(col).is_not_null().any()).then(pl.col(col).sum()).otherwise(None).alias(col)


# What a measurement means once a period is collapsed: a week of rain is a total, a week of
# gusts is the strongest one, a week of temperatures is an average.
_MEASUREMENT_AGGS: dict[str, pl.Expr] = {
    "temp_min": pl.col("temp_min").mean(),
    "temp_max": pl.col("temp_max").mean(),
    "precipitation": _total_or_null("precipitation"),
    "wind_mean": pl.col("wind_mean").mean(),
    "wind_gust": pl.col("wind_gust").max(),
}


_LATEST_TTL_SECONDS: int = 6 * 3600


def _file_url(dept: str, period: Period) -> str | None:
    """Stable permalink for (dept, period), or None if the resource is unknown."""
    resource_id = _RESOURCE_INDEX.get(dept, {}).get(period.value)
    if resource_id is None:
        return None
    return f"{DATAGOUV_RESOURCE_URL}/{resource_id}"


def _cache_path(dept: str, period: Period) -> Path:
    return CACHE_DIR / f"Q_{dept}_{period.value}_RR-T-Vent.csv.gz"


def _is_stale(path: Path) -> bool:
    return time.time() - path.stat().st_mtime > _LATEST_TTL_SECONDS


def _download(dept: str, period: Period) -> Path | None:
    url = _file_url(dept, period)
    if url is None:
        return None
    response = requests.get(url, timeout=30)
    if response.status_code != 200:
        return None
    path = _cache_path(dept, period)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)
    return path


def _fetch(dept: str, period: Period) -> Path | None:
    """Return the local cache path for (dept, period), downloading if needed.

    For Period.LATEST, re-downloads if the cached file is older than _LATEST_TTL_SECONDS.
    On network failure during refresh, falls back to the stale file rather than returning None.
    Returns None when the remote file does not exist (e.g. no historical data
    for that department) — never raises for expected HTTP failures.
    """
    path = _cache_path(dept, period)
    if path.exists():
        if period != Period.LATEST or not _is_stale(path):
            return path
        return _download(dept, period) or path
    return _download(dept, period)


def _parse(path: Path) -> pl.DataFrame | None:
    """Parse a cached CSV.gz file into a tidy Polars DataFrame.

    Returns None if the file cannot be read (e.g. corrupt download).
    """
    try:
        df = pl.read_csv(path, separator=";", null_values=["mq"], infer_schema_length=1000)

        available = [c for c in KEEP_COLS if c in df.columns]
        cast_exprs = (
            [pl.col(c).cast(pl.Float64) for c in _PARSE_COORD_COLS if c in df.columns]
            + [pl.col(c).cast(pl.Float32) for c in _PARSE_MEASURE_COLS if c in df.columns]
            + [pl.col(c).cast(pl.Int32) for c in _PARSE_INT_COLS if c in df.columns]
            + [pl.col("AAAAMMJJ").cast(pl.String).str.to_date(format="%Y%m%d").alias("DATE")]
        )
        rename = {raw: readable for raw, readable in COLUMN_RENAME.items() if raw in df.columns}
        return df.select(available).with_columns(cast_exprs).drop("AAAAMMJJ").rename(rename)
    except (pl.exceptions.PolarsError, OSError):
        return None


def aggregate(df: pl.DataFrame, granularity: Granularity) -> pl.DataFrame:
    """Resample a station DataFrame to weekly or monthly values.

    Granularity.DAY is the identity — returns df unchanged. For WEEK or MONTH, each measurement
    column is collapsed with the operation that preserves its meaning (see `_MEASUREMENT_AGGS`):
    temperatures average, precipitation totals, gusts take the period maximum. Metadata columns
    keep their first value (constant per station). Result is sorted by (station_id, DATE).

    ponytail: a period with missing days yields a proportionally understated precipitation total.
    Surface per-period coverage if partial months turn out to mislead.
    """
    truncate_expr = granularity.truncate_expr
    if truncate_expr is None:
        return df

    numeric_aggs = [expr for col, expr in _MEASUREMENT_AGGS.items() if col in df.columns]
    meta_aggs = [pl.col(c).first() for c in _META_COLS if c in df.columns]

    return (
        df.with_columns(pl.col("DATE").dt.truncate(truncate_expr))
        .group_by(["station_id", "station_name", "DATE"])
        .agg(numeric_aggs + meta_aggs)
        .sort(["station_id", "DATE"])
    )


def load_department(dept: str) -> pl.DataFrame | None:
    """Load and merge all time periods for a department into one DataFrame.

    Fetches missing periods from the remote source and caches them locally.
    Returns None if no data could be loaded for this department.
    """
    frames: list[pl.DataFrame] = []
    for period in Period:
        path = _fetch(dept, period)
        if path is None:
            continue
        df = _parse(path)
        if df is not None:
            frames.append(df)

    if not frames:
        return None

    return pl.concat(frames, how="diagonal").sort(["station_id", "DATE"])


@dataclass(frozen=True)
class _CacheEntry:
    df: pl.DataFrame | None
    loaded_at: float

    def is_fresh(self, now: float) -> bool:
        return now - self.loaded_at < _LATEST_TTL_SECONDS


# A parsed department holds every station at daily resolution since 1809 — 100–165 MB each.
# Browsing is local in practice, so keep the few most recent and let the rest be re-read from disk.
MAX_CACHED_DEPTS: int = 3

_dept_cache: OrderedDict[str, _CacheEntry] = OrderedDict()
_dept_locks: dict[str, threading.Lock] = {}
# Guards the two dicts above. Held only for dict bookkeeping, never across a load — the long
# work happens under the per-department lock so different departments still load in parallel.
_registry_lock = threading.Lock()


def _cached_entry(dept: str, now: float) -> _CacheEntry | None:
    with _registry_lock:
        entry = _dept_cache.get(dept)
        if entry is None or not entry.is_fresh(now):
            return None
        _dept_cache.move_to_end(dept)
        return entry


def _store_entry(dept: str, entry: _CacheEntry) -> None:
    with _registry_lock:
        _dept_cache[dept] = entry
        _dept_cache.move_to_end(dept)
        while len(_dept_cache) > MAX_CACHED_DEPTS:
            _dept_cache.popitem(last=False)


def load_department_cached(dept: str) -> pl.DataFrame | None:
    """Return a department DataFrame from a bounded in-process cache, loading on first access.

    Holds at most MAX_CACHED_DEPTS departments, evicting least-recently-used; entries expire after
    _LATEST_TTL_SECONDS so that Period.LATEST data stays fresh. Thread-safe: concurrent requests for
    different departments do not block each other; concurrent requests for the same department wait
    on a per-key lock.
    """
    now = time.time()
    entry = _cached_entry(dept, now)
    if entry is not None:
        return entry.df
    with _registry_lock:
        lock = _dept_locks.setdefault(dept, threading.Lock())
    with lock:
        entry = _cached_entry(dept, time.time())
        if entry is None:
            entry = _CacheEntry(load_department(dept), time.time())
            _store_entry(dept, entry)
    return entry.df


def clear_cache() -> None:
    """Force the next request to fetch fresh LATEST-period data instead of serving cached data.

    Drops the in-process cache and deletes disk-cached LATEST files, which would
    otherwise be served as-is for up to _LATEST_TTL_SECONDS.
    """
    with _registry_lock:
        _dept_cache.clear()
    for path in CACHE_DIR.glob(f"*{Period.LATEST.value}*"):
        path.unlink(missing_ok=True)


@dataclass(frozen=True)
class RecordSpan:
    """How far back a station's observations run. Present or absent as a whole — the index is
    built without it by `build_station_index.py --fast`, never with half of it."""

    first_year: int
    last_year: int

    def __post_init__(self) -> None:
        if self.first_year > self.last_year:
            raise ValueError(f"record span ends before it starts: {self.first_year}–{self.last_year}")

    @property
    def n_years(self) -> int:
        return self.last_year - self.first_year + 1


@dataclass(frozen=True)
class IndexedStation:
    """One entry of the pre-built station index: where a station is, and how long it has run."""

    station_id: int
    name: str
    dept: str
    lat: float
    lon: float
    altitude: int
    span: RecordSpan | None


def _indexed_station(entry: Any) -> IndexedStation | None:
    """Read one index entry, or None if it is malformed — a stale index is not a crash.

    An inverted span raises out of `RecordSpan` and is caught here with the other value errors.
    """
    if not isinstance(entry, dict):
        return None
    try:
        first, last = entry.get("first_year"), entry.get("last_year")
        return IndexedStation(
            station_id=int(entry["station_id"]),
            name=str(entry["station_name"]),
            dept=str(entry["dept"]),
            lat=float(entry["lat"]),
            lon=float(entry["lon"]),
            altitude=int(entry["altitude"]),
            span=RecordSpan(int(first), int(last)) if first is not None and last is not None else None,
        )
    except (KeyError, TypeError, ValueError):
        return None


@lru_cache(maxsize=1)
def station_index() -> list[IndexedStation]:
    """The pre-built index of every station, or an empty list when it has not been generated yet.

    Built by scripts/build_station_index.py and committed; the app never fetches it at runtime.
    """
    path = DATA_DIR / "stations.json"
    if not path.exists():
        return []
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, list):
        return []
    return [station for station in map(_indexed_station, parsed) if station is not None]


def stations_from(df: pl.DataFrame) -> list[Station]:
    """Extract the unique list of stations from a department DataFrame."""
    return [
        Station(
            station_id=row["station_id"],
            name=row["station_name"],
            lat=row["lat"],
            lon=row["lon"],
            altitude=row["altitude"],
        )
        for row in (
            df.select(["station_id", "station_name", "lat", "lon", "altitude"])
            .unique(subset=["station_id"])
            .sort("station_name")
            .to_dicts()
        )
    ]
