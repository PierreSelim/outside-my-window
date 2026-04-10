from __future__ import annotations

import gzip
import textwrap
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest

from src.data_loader import Granularity, Station, _parse, aggregate, load_department, stations_from

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
    assert toulouse.station_id == 31001
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


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------

# Multi-week fixture: station 31001 with 3 weeks of data (Mon–Sun blocks)
# Week 1: 2020-01-06 (Mon) and 2020-01-07
# Week 2: 2020-01-13
# Week 3 (different station): 2020-01-06
@pytest.fixture
def multi_week_df() -> pl.DataFrame:
    from datetime import date

    return pl.DataFrame(
        {
            "station_id":     [31001, 31001, 31001, 31002],
            "station_name":   ["TOULOUSE", "TOULOUSE", "TOULOUSE", "BLAGNAC"],
            "lat":            [43.60, 43.60, 43.60, 43.63],
            "lon":            [1.44,  1.44,  1.44,  1.37],
            "altitude":       [152,   152,   152,   151],
            "DATE":           [date(2020, 1, 6), date(2020, 1, 7), date(2020, 1, 13), date(2020, 1, 6)],
            "temp_min":       [0.0,   2.0,  4.0,  1.0],
            "temp_max":       [8.0,  10.0, 12.0,  9.0],
            "temp_mean":      [4.0,   6.0,  8.0,  5.0],
            "temp_amplitude": [8.0,   8.0,  8.0,  8.0],
            "precipitation":  [1.0,   3.0,  2.0,  0.5],
            "wind_mean":      [2.0,   4.0,  3.0,  None],
            "wind_gust":      [5.0,  10.0,  7.0,  None],
        },
        schema={
            "station_id":     pl.Int32,
            "station_name":   pl.String,
            "lat":            pl.Float64,
            "lon":            pl.Float64,
            "altitude":       pl.Int32,
            "DATE":           pl.Date,
            "temp_min":       pl.Float64,
            "temp_max":       pl.Float64,
            "temp_mean":      pl.Float64,
            "temp_amplitude": pl.Float64,
            "precipitation":  pl.Float64,
            "wind_mean":      pl.Float64,
            "wind_gust":      pl.Float64,
        },
    )


def test_aggregate_day_is_identity(sample_df: pl.DataFrame) -> None:
    result = aggregate(sample_df, Granularity.DAY)
    assert result is sample_df


def test_aggregate_week_reduces_row_count(multi_week_df: pl.DataFrame) -> None:
    # station 31001: rows on Jan 6+7 (week 1) and Jan 13 (week 2) → 2 rows
    # station 31002: row on Jan 6 (week 1) → 1 row
    # total: 3 rows
    result = aggregate(multi_week_df, Granularity.WEEK)
    assert len(result) == 3


def test_aggregate_week_averages_values(multi_week_df: pl.DataFrame) -> None:
    result = aggregate(multi_week_df, Granularity.WEEK)
    toulouse = result.filter(pl.col("station_id") == 31001).sort("DATE")
    # Week 1 for Toulouse: Jan 6 and Jan 7 → mean temp_min = (0+2)/2 = 1.0
    week1 = toulouse.row(0, named=True)
    assert week1["temp_min"] == pytest.approx(1.0)
    assert week1["precipitation"] == pytest.approx(2.0)  # (1+3)/2


def test_aggregate_month_groups_all_january(multi_week_df: pl.DataFrame) -> None:
    # All rows are in January 2020 — each station should collapse to 1 row
    result = aggregate(multi_week_df, Granularity.MONTH)
    assert len(result) == 2  # one row per station


def test_aggregate_month_averages_values(multi_week_df: pl.DataFrame) -> None:
    result = aggregate(multi_week_df, Granularity.MONTH)
    toulouse = result.filter(pl.col("station_id") == 31001).row(0, named=True)
    # Jan 6, 7, 13 → temp_min mean = (0+2+4)/3
    assert toulouse["temp_min"] == pytest.approx(2.0)
    assert toulouse["wind_mean"] == pytest.approx(3.0)  # (2+4+3)/3


def test_aggregate_preserves_null_columns(multi_week_df: pl.DataFrame) -> None:
    # BLAGNAC has null wind — aggregated result should still be null
    result = aggregate(multi_week_df, Granularity.WEEK)
    blagnac = result.filter(pl.col("station_id") == 31002).row(0, named=True)
    assert blagnac["wind_mean"] is None
    assert blagnac["wind_gust"] is None


def test_aggregate_week_date_is_week_start(multi_week_df: pl.DataFrame) -> None:
    from datetime import date

    result = aggregate(multi_week_df, Granularity.WEEK)
    toulouse = result.filter(pl.col("station_id") == 31001).sort("DATE")
    # 2020-01-06 is a Monday — truncate("1w") should yield 2020-01-06
    assert toulouse["DATE"][0] == date(2020, 1, 6)


def test_granularity_title_suffix() -> None:
    assert Granularity.DAY.title_suffix == ""
    assert Granularity.WEEK.title_suffix == " (weekly avg)"
    assert Granularity.MONTH.title_suffix == " (monthly avg)"


def test_granularity_truncate_expr() -> None:
    assert Granularity.DAY.truncate_expr is None
    assert Granularity.WEEK.truncate_expr == "1w"
    assert Granularity.MONTH.truncate_expr == "1mo"
