from __future__ import annotations

import gzip
import textwrap
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest

from src.data_loader import Station, _parse, load_department, stations_from

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CSV_HEADER = "NUM_POSTE;NOM_USUEL;LAT;LON;ALTI;AAAAMMJJ;RR;TN;TX;TM;TAMPLI;FFM;FXY;DXY"
_CSV_ROWS = [
    "31001;TOULOUSE;43.6;1.44;152;20200101;0.0;-1.0;8.0;3.5;9.0;3.0;8.0;180",
    "31001;TOULOUSE;43.6;1.44;152;20200102;2.5;0.5;10.5;5.0;10.0;5.0;12.0;270",
    "31002;BLAGNAC;43.63;1.37;151;20200101;mq;-2.0;7.0;mq;9.0;mq;mq;mq",
]


def _make_csv_gz(tmp_path: Path, rows: list[str] | None = None) -> Path:
    """Write a minimal raw RR-T-Vent CSV.gz to tmp_path and return its path."""
    content = "\n".join([_CSV_HEADER] + (rows or _CSV_ROWS))
    path = tmp_path / "test.csv.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(content.encode()))
    return path


# ---------------------------------------------------------------------------
# _parse
# ---------------------------------------------------------------------------


def test_parse_renames_columns(tmp_path: Path) -> None:
    df = _parse(_make_csv_gz(tmp_path))
    assert df is not None
    assert "temp_min" in df.columns
    assert "temp_max" in df.columns
    assert "temp_mean" in df.columns
    assert "precipitation" in df.columns
    assert "wind_mean" in df.columns
    assert "wind_gust" in df.columns
    assert "station_id" in df.columns
    assert "station_name" in df.columns
    assert "altitude" in df.columns


def test_parse_raw_names_absent(tmp_path: Path) -> None:
    df = _parse(_make_csv_gz(tmp_path))
    assert df is not None
    for raw in ("TN", "TX", "TM", "RR", "FFM", "FXY", "NUM_POSTE", "NOM_USUEL", "ALTI", "AAAAMMJJ"):
        assert raw not in df.columns


def test_parse_drops_quality_flags(tmp_path: Path) -> None:
    df = _parse(_make_csv_gz(tmp_path))
    assert df is not None
    for col in ("QRR", "QTN", "QTX", "QTM", "QFFM", "QFXY"):
        assert col not in df.columns


def test_parse_date_column_type(tmp_path: Path) -> None:
    df = _parse(_make_csv_gz(tmp_path))
    assert df is not None
    assert df["DATE"].dtype == pl.Date


def test_parse_mq_sentinel_becomes_null(tmp_path: Path) -> None:
    df = _parse(_make_csv_gz(tmp_path))
    assert df is not None
    blagnac = df.filter(pl.col("station_id") == 31002)
    assert blagnac["precipitation"].is_null().all()
    assert blagnac["wind_mean"].is_null().all()


def test_parse_returns_none_for_corrupt_file(tmp_path: Path) -> None:
    bad = tmp_path / "bad.csv.gz"
    bad.write_bytes(b"not a gzip file")
    assert _parse(bad) is None


# ---------------------------------------------------------------------------
# stations_from
# ---------------------------------------------------------------------------


def test_stations_from_returns_one_per_station(sample_df: pl.DataFrame) -> None:
    stations = stations_from(sample_df)
    assert len(stations) == 2


def test_stations_from_sorted_by_name(sample_df: pl.DataFrame) -> None:
    stations = stations_from(sample_df)
    names = [s.name for s in stations]
    assert names == sorted(names)


def test_stations_from_fields(sample_df: pl.DataFrame) -> None:
    stations = stations_from(sample_df)
    toulouse = next(s for s in stations if s.name == "TOULOUSE")
    assert toulouse.num_poste == 31001
    assert toulouse.lat == pytest.approx(43.60)
    assert toulouse.altitude == 152


def test_stations_from_returns_station_dataclass(sample_df: pl.DataFrame) -> None:
    stations = stations_from(sample_df)
    assert all(isinstance(s, Station) for s in stations)


# ---------------------------------------------------------------------------
# load_department
# ---------------------------------------------------------------------------


def test_load_department_returns_none_when_all_fetches_fail() -> None:
    with patch("src.data_loader._fetch", return_value=None):
        result = load_department("99")
    assert result is None


def test_load_department_merges_periods(tmp_path: Path) -> None:
    # Two periods, each with 1 row — result should have 2 rows
    file_a = _make_csv_gz(tmp_path / "a", ["31001;TOULOUSE;43.6;1.44;152;20200101;0.0;-1.0;8.0;3.5;9.0;3.0;8.0;180"])
    file_b = _make_csv_gz(tmp_path / "b", ["31001;TOULOUSE;43.6;1.44;152;19500101;0.5;-2.0;6.0;2.0;8.0;2.0;5.0;90"])

    with patch("src.data_loader._fetch", side_effect=[None, file_a, file_b]):
        result = load_department("31")

    assert result is not None
    assert len(result) == 2
