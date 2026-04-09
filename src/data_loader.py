from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import polars as pl
import requests

BASE_URL = "https://object.files.data.gouv.fr/meteofrance/data/synchro_ftp/BASE/QUOT"
CACHE_DIR = Path("data/cache")

# Columns we actually need for visualisation — the raw files have 60 columns
KEEP_COLS: list[str] = [
    "NUM_POSTE", "NOM_USUEL", "LAT", "LON", "ALTI", "AAAAMMJJ",
    "RR", "TN", "TX", "TM", "TAMPLI", "FFM", "FXY", "DXY",
]

# Mapping from raw dataset column names to human-readable equivalents
COLUMN_RENAME: dict[str, str] = {
    "NUM_POSTE": "station_id",
    "NOM_USUEL": "station_name",
    "ALTI":      "altitude",
    "TN":        "temp_min",
    "TX":        "temp_max",
    "TM":        "temp_mean",
    "TAMPLI":    "temp_amplitude",
    "RR":        "precipitation",
    "FFM":       "wind_mean",
    "FXY":       "wind_gust",
    "DXY":       "wind_gust_dir",
}


class Period(str, Enum):
    HISTORICAL = "1852-1949"
    MODERN = "previous-1950-2024"
    LATEST = "latest-2025-2026"


@dataclass(frozen=True)
class Station:
    num_poste: int
    name: str
    lat: float
    lon: float
    altitude: int


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
        float_cols = ["LAT", "LON", "RR", "TN", "TX", "TM", "TAMPLI", "FFM", "FXY", "DXY"]
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
    except Exception:  # noqa: BLE001 — any I/O or schema failure, not business logic
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


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


def stations_from(df: pl.DataFrame) -> list[Station]:
    """Extract the unique list of stations from a department DataFrame."""
    return [
        Station(
            num_poste=row["station_id"],
            name=row["station_name"],
            lat=row["LAT"],
            lon=row["LON"],
            altitude=row["altitude"],
        )
        for row in (
            df.select(["station_id", "station_name", "LAT", "LON", "altitude"])
            .unique(subset=["station_id"])
            .sort("station_name")
            .to_dicts()
        )
    ]
