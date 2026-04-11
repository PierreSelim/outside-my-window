from __future__ import annotations

from datetime import date

import polars as pl
import plotly.graph_objects as go
import pytest

from src.charts import empty_figure, precipitation_figure, temperature_figure, wind_figure


# ---------------------------------------------------------------------------
# temperature_figure
# ---------------------------------------------------------------------------


def test_temperature_figure_returns_figure(sample_df: pl.DataFrame) -> None:
    fig = temperature_figure(sample_df, "TOULOUSE")
    assert isinstance(fig, go.Figure)


def test_temperature_figure_has_min_max_traces(sample_df: pl.DataFrame) -> None:
    fig = temperature_figure(sample_df, "TOULOUSE")
    assert len(fig.data) == 2  # max + min band
    assert all(isinstance(t, go.Scatter) for t in fig.data)


def test_temperature_figure_title_contains_station(sample_df: pl.DataFrame) -> None:
    fig = temperature_figure(sample_df, "TOULOUSE")
    assert "TOULOUSE" in fig.layout.title.text


# ---------------------------------------------------------------------------
# temperature_figure — hot day highlighting
# ---------------------------------------------------------------------------


@pytest.fixture
def hot_day_df(sample_df: pl.DataFrame) -> pl.DataFrame:
    """sample_df with one day modified to meet the hot-day rule (Tmin≥20, Tmax≥35)."""
    return sample_df.with_columns([
        pl.when(pl.col("DATE") == date(2020, 1, 1))
          .then(pl.lit(21.0)).otherwise(pl.col("temp_min")).cast(pl.Float64).alias("temp_min"),
        pl.when(pl.col("DATE") == date(2020, 1, 1))
          .then(pl.lit(36.0)).otherwise(pl.col("temp_max")).cast(pl.Float64).alias("temp_max"),
    ])


def test_temperature_figure_no_hot_day_shapes(sample_df: pl.DataFrame) -> None:
    fig = temperature_figure(sample_df, "TOULOUSE")
    assert len(fig.layout.shapes) == 0


def test_temperature_figure_hot_day_adds_one_shape(hot_day_df: pl.DataFrame) -> None:
    toulouse = hot_day_df.filter(pl.col("station_id") == 31001)
    fig = temperature_figure(toulouse, "TOULOUSE")
    assert len(fig.layout.shapes) == 1


def test_temperature_figure_hot_day_adds_legend_entry(hot_day_df: pl.DataFrame) -> None:
    toulouse = hot_day_df.filter(pl.col("station_id") == 31001)
    fig = temperature_figure(toulouse, "TOULOUSE")
    trace_names = [t.name for t in fig.data]
    assert any("hot day" in name.lower() for name in trace_names)


def test_temperature_figure_no_legend_entry_without_hot_days(sample_df: pl.DataFrame) -> None:
    fig = temperature_figure(sample_df, "TOULOUSE")
    trace_names = [t.name for t in fig.data]
    assert not any("hot day" in name.lower() for name in trace_names)


def test_temperature_figure_multiple_hot_days_add_multiple_shapes(sample_df: pl.DataFrame) -> None:
    df = sample_df.with_columns([
        pl.lit(21.0).cast(pl.Float64).alias("temp_min"),
        pl.lit(36.0).cast(pl.Float64).alias("temp_max"),
    ])
    toulouse = df.filter(pl.col("station_id") == 31001)  # 3 rows → 3 hot days
    fig = temperature_figure(toulouse, "TOULOUSE")
    assert len(fig.layout.shapes) == 3


# ---------------------------------------------------------------------------
# precipitation_figure
# ---------------------------------------------------------------------------


def test_precipitation_figure_returns_figure(sample_df: pl.DataFrame) -> None:
    fig = precipitation_figure(sample_df, "TOULOUSE")
    assert isinstance(fig, go.Figure)


def test_precipitation_figure_has_single_bar_trace(sample_df: pl.DataFrame) -> None:
    fig = precipitation_figure(sample_df, "TOULOUSE")
    assert len(fig.data) == 1
    assert isinstance(fig.data[0], go.Bar)


def test_precipitation_figure_title_contains_station(sample_df: pl.DataFrame) -> None:
    fig = precipitation_figure(sample_df, "TOULOUSE")
    assert "TOULOUSE" in fig.layout.title.text


# ---------------------------------------------------------------------------
# wind_figure
# ---------------------------------------------------------------------------


def test_wind_figure_returns_figure(sample_df: pl.DataFrame) -> None:
    fig = wind_figure(sample_df, "TOULOUSE")
    assert isinstance(fig, go.Figure)


def test_wind_figure_has_gust_and_mean_traces(sample_df: pl.DataFrame) -> None:
    df = sample_df.filter(pl.col("station_id") == 31001)  # only station with wind data
    fig = wind_figure(df, "TOULOUSE")
    assert len(fig.data) == 2


def test_wind_figure_no_data_when_all_null(sample_df: pl.DataFrame) -> None:
    df = sample_df.filter(pl.col("station_id") == 31002)  # wind cols are all null
    fig = wind_figure(df, "BLAGNAC")
    assert len(fig.data) == 0
    assert "no data" in fig.layout.title.text.lower()


def test_wind_figure_title_contains_station(sample_df: pl.DataFrame) -> None:
    fig = wind_figure(sample_df, "TOULOUSE")
    assert "TOULOUSE" in fig.layout.title.text


# ---------------------------------------------------------------------------
# empty_figure
# ---------------------------------------------------------------------------


def test_empty_figure_returns_figure() -> None:
    fig = empty_figure("Nothing here")
    assert isinstance(fig, go.Figure)


def test_empty_figure_has_annotation() -> None:
    fig = empty_figure("Nothing here")
    assert len(fig.layout.annotations) == 1
    assert fig.layout.annotations[0].text == "Nothing here"


def test_empty_figure_axes_hidden() -> None:
    fig = empty_figure()
    assert fig.layout.xaxis.visible is False
    assert fig.layout.yaxis.visible is False


# ---------------------------------------------------------------------------
# temperature_figure — band consistency
# ---------------------------------------------------------------------------


def test_temperature_band_traces_share_same_dates_when_nulls_differ(sample_df: pl.DataFrame) -> None:
    """Max and Min traces must cover identical dates even when TX/TN have different null patterns."""
    from datetime import date

    df = sample_df.filter(pl.col("station_id") == 31001).with_columns(
        pl.when(pl.col("DATE") == date(2020, 1, 3))
        .then(None)
        .otherwise(pl.col("temp_max"))
        .cast(pl.Float64)
        .alias("temp_max")
    )
    fig = temperature_figure(df, "TOULOUSE")
    max_trace = next(t for t in fig.data if t.name == "Max")
    min_trace = next(t for t in fig.data if t.name == "Min")
    assert list(max_trace.x) == list(min_trace.x)
