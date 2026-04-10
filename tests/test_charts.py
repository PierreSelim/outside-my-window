from __future__ import annotations

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


def test_temperature_figure_has_min_max_mean_traces(sample_df: pl.DataFrame) -> None:
    fig = temperature_figure(sample_df, "TOULOUSE")
    assert len(fig.data) == 3  # max, min, mean
    assert all(isinstance(t, go.Scatter) for t in fig.data)


def test_temperature_figure_omits_mean_when_all_null(sample_df: pl.DataFrame) -> None:
    df = sample_df.with_columns(pl.lit(None).cast(pl.Float64).alias("temp_mean"))
    fig = temperature_figure(df, "TOULOUSE")
    assert len(fig.data) == 2  # max + min only


def test_temperature_figure_title_contains_station(sample_df: pl.DataFrame) -> None:
    fig = temperature_figure(sample_df, "TOULOUSE")
    assert "TOULOUSE" in fig.layout.title.text


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
