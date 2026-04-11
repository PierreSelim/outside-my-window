from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import polars as pl
import requests

BASE_URL = "https://object.files.data.gouv.fr/meteofrance/data/synchro_ftp/BASE/QUOT"
CACHE_DIR = Path(__file__).parent.parent / "data" / "cache"

# Columns we actually need for visualisation — the raw files have 60 columns
KEEP_COLS: list[str] = [
    "NUM_POSTE", "NOM_USUEL", "LAT", "LON", "ALTI", "AAAAMMJJ",
    "RR", "TN", "TX", "TAMPLI", "FFM", "FXY",
]

# Mapping from raw dataset column names to human-readable equivalents
COLUMN_RENAME: dict[str, str] = {
    "NUM_POSTE": "station_id",
    "NOM_USUEL": "station_name",
    "LAT":       "lat",
    "LON":       "lon",
    "ALTI":      "altitude",
    "TN":        "temp_min",
    "TX":        "temp_max",
    "TAMPLI":    "temp_amplitude",
    "RR":        "precipitation",
    "FFM":       "wind_mean",
    "FXY":       "wind_gust",
}


class Period(str, Enum):
    HISTORICAL = "1852-1949"
    MODERN = "previous-1950-2024"
    LATEST = "latest-2025-2026"


@dataclass(frozen=True)
class Truncated:
    """A granularity that collapses daily rows to a coarser period by averaging.

    ``label`` is the RadioItems wire value; ``truncate_expr`` is passed directly
    to Polars ``dt.truncate``; ``title_suffix`` is appended to chart titles.
    """

    label: str
    truncate_expr: str
    title_suffix: str


class Granularity:
    """Namespace of granularity constants.

    DAY is None — the identity case, no truncation needed.
    WEEK and MONTH are Truncated instances carrying all derived values.
    """

    DAY: None = None
    WEEK: Truncated = Truncated("week", "1w", " (weekly avg)")
    MONTH: Truncated = Truncated("month", "1mo", " (monthly avg)")


_GRANULARITY_BY_LABEL: dict[str, Truncated] = {
    g.label: g for g in [Granularity.WEEK, Granularity.MONTH]
}


def granularity_from(value: str) -> Truncated | None:
    """Parse a RadioItems string value into a Truncated granularity.

    Returns None for the daily (identity) case — any value not in the
    lookup is treated as DAY.
    """
    return _GRANULARITY_BY_LABEL.get(value)


@dataclass(frozen=True)
class Station:
    station_id: int
    name: str
    lat: float
    lon: float
    altitude: int


_NUMERIC_COLS: list[str] = [
    "temp_min", "temp_max", "temp_amplitude",
    "precipitation", "wind_mean", "wind_gust",
]
_META_COLS: list[str] = ["lat", "lon", "altitude"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _file_url(dept: str, period: Period) -> str:
    return f"{BASE_URL}/Q_{dept}_{period.value}_RR-T-Vent.csv.gz"


def _cache_path(dept: str, period: Period) -> Path:
    return CACHE_DIR / f"Q_{dept}_{period.value}_RR-T-Vent.csv.gz"


def _fetch(dept: str, period: Period) -> Path | None:
    """Return the local cache path for (dept, period), downloading if needed.

    Returns None when the remote file does not exist (e.g. no historical data
    for that department) — never raises for expected HTTP failures.
    """
    path = _cache_path(dept, period)
    if path.exists():
        return path

    url = _file_url(dept, period)
    response = requests.get(url, timeout=30)
    if response.status_code != 200:
        return None

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)
    return path


def _parse(path: Path) -> pl.DataFrame | None:
    """Parse a cached CSV.gz file into a tidy Polars DataFrame.

    Returns None if the file cannot be read (e.g. corrupt download).
    """
    try:
        df = pl.read_csv(path, separator=";", null_values=["mq"], infer_schema_length=1000)

        # Keep only the columns we care about (some periods may lack a few cols)
        available = [c for c in KEEP_COLS if c in df.columns]
        df = df.select(available)

        # Normalise numeric columns to consistent types across all periods
        float_cols = ["LAT", "LON", "RR", "TN", "TX", "TAMPLI", "FFM", "FXY"]
        int_cols = ["NUM_POSTE", "ALTI"]
        df = df.with_columns(
            [pl.col(c).cast(pl.Float64) for c in float_cols if c in df.columns]
            + [pl.col(c).cast(pl.Int32) for c in int_cols if c in df.columns]
        )

        # Parse YYYYMMDD integer → proper Date column
        df = df.with_columns(
            pl.col("AAAAMMJJ").cast(pl.String).str.to_date(format="%Y%m%d").alias("DATE")
        ).drop("AAAAMMJJ")

        # Rename to human-readable column names
        rename = {raw: readable for raw, readable in COLUMN_RENAME.items() if raw in df.columns}
        return df.rename(rename)
    except (pl.exceptions.PolarsError, OSError):
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def aggregate(df: pl.DataFrame, granularity: Truncated | None) -> pl.DataFrame:
    """Resample a station DataFrame to weekly or monthly averages.

    None (DAY) is the identity — returns df unchanged. For a Truncated
    granularity, each numeric measurement column is averaged; metadata columns
    keep their first value (they are constant per station). The result is
    sorted by (station_id, DATE).
    """
    if granularity is None:
        return df

    numeric_aggs = [pl.col(c).mean() for c in _NUMERIC_COLS if c in df.columns]
    meta_aggs = [pl.col(c).first() for c in _META_COLS if c in df.columns]

    return (
        df.with_columns(pl.col("DATE").dt.truncate(granularity.truncate_expr))
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


_dept_cache: dict[str, pl.DataFrame | None] = {}
_dept_cache_lock = threading.Lock()


def load_department_cached(dept: str) -> pl.DataFrame | None:
    """Return a department DataFrame from an in-process cache, loading on first access.

    Thread-safe: concurrent requests for the same department will not trigger
    duplicate downloads.
    """
    with _dept_cache_lock:
        if dept not in _dept_cache:
            _dept_cache[dept] = load_department(dept)
    return _dept_cache[dept]


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
