from __future__ import annotations

from datetime import date

import plotly.graph_objects as go
import polars as pl
import pytest

from src.charts import (
    empty_figure,
    hot_cold_yearly_figure,
    monthly_avg_temp_by_decade_figure,
    monthly_avg_temp_figure,
    precipitation_figure,
    temperature_figure,
    wind_figure,
)
from src.transforms import linear_trend

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
    return sample_df.with_columns(
        [
            pl.when(pl.col("DATE") == date(2020, 1, 1))
            .then(pl.lit(21.0))
            .otherwise(pl.col("temp_min"))
            .cast(pl.Float64)
            .alias("temp_min"),
            pl.when(pl.col("DATE") == date(2020, 1, 1))
            .then(pl.lit(36.0))
            .otherwise(pl.col("temp_max"))
            .cast(pl.Float64)
            .alias("temp_max"),
        ]
    )


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
    df = sample_df.with_columns(
        [
            pl.lit(21.0).cast(pl.Float64).alias("temp_min"),
            pl.lit(36.0).cast(pl.Float64).alias("temp_max"),
        ]
    )
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
# hot_cold_yearly_figure
# ---------------------------------------------------------------------------


def test_hot_cold_yearly_figure_returns_figure(sample_df: pl.DataFrame) -> None:
    fig = hot_cold_yearly_figure(sample_df, "TOULOUSE")
    assert isinstance(fig, go.Figure)


def test_hot_cold_yearly_figure_has_two_line_traces(sample_df: pl.DataFrame) -> None:
    fig = hot_cold_yearly_figure(sample_df, "TOULOUSE")
    assert len(fig.data) == 2
    assert all(isinstance(t, go.Scatter) for t in fig.data)


def test_hot_cold_yearly_figure_title_contains_station(sample_df: pl.DataFrame) -> None:
    fig = hot_cold_yearly_figure(sample_df, "TOULOUSE")
    assert "TOULOUSE" in fig.layout.title.text


def test_hot_cold_yearly_figure_counts_cold_days(sample_df: pl.DataFrame) -> None:
    """sample_df has temp_min = [-1.0, 0.5, 2.0, -2.0, 1.0] — 2 negative values in year 2020."""
    df = sample_df.filter(pl.col("DATE").dt.year() == 2020)
    fig = hot_cold_yearly_figure(df, "TOULOUSE")
    cold_trace = next(t for t in fig.data if "cold" in t.name.lower())
    total_cold = sum(cold_trace.y)
    assert total_cold == 2  # -1.0 and -2.0


def test_hot_cold_yearly_figure_counts_hot_days(sample_df: pl.DataFrame) -> None:
    """Inject one hot day (Tmin≥20, Tmax≥35) for a single station and verify count."""
    from datetime import date

    df = sample_df.filter(pl.col("station_id") == 31001).with_columns(
        [
            pl.when(pl.col("DATE") == date(2020, 1, 1))
            .then(pl.lit(21.0))
            .otherwise(pl.col("temp_min"))
            .cast(pl.Float64)
            .alias("temp_min"),
            pl.when(pl.col("DATE") == date(2020, 1, 1))
            .then(pl.lit(36.0))
            .otherwise(pl.col("temp_max"))
            .cast(pl.Float64)
            .alias("temp_max"),
        ]
    )
    fig = hot_cold_yearly_figure(df, "TOULOUSE")
    hot_trace = next(t for t in fig.data if "hot" in t.name.lower())
    assert sum(hot_trace.y) == 1


def test_hot_cold_yearly_figure_no_trend_by_default(sample_df: pl.DataFrame) -> None:
    fig = hot_cold_yearly_figure(sample_df, "TOULOUSE")
    assert len(fig.data) == 2


def test_hot_cold_yearly_figure_trend_adds_two_traces(multi_decade_df: pl.DataFrame) -> None:
    fig = hot_cold_yearly_figure(multi_decade_df, "TOULOUSE", show_trend=True)
    assert len(fig.data) == 4
    trend_traces = [t for t in fig.data if "trend" in t.name.lower()]
    assert len(trend_traces) == 2


def test_hot_cold_yearly_figure_trend_traces_are_dashed_lines(multi_decade_df: pl.DataFrame) -> None:
    fig = hot_cold_yearly_figure(multi_decade_df, "TOULOUSE", show_trend=True)
    for trace in fig.data:
        if "trend" in trace.name.lower():
            assert trace.line.dash == "dash"
            assert trace.mode == "lines"


def test_hot_cold_yearly_figure_trend_same_x_as_data(multi_decade_df: pl.DataFrame) -> None:
    fig = hot_cold_yearly_figure(multi_decade_df, "TOULOUSE", show_trend=True)
    data_x = fig.data[0].x
    for trace in fig.data:
        if "trend" in trace.name.lower():
            assert list(trace.x) == list(data_x)


# ---------------------------------------------------------------------------
# _linear_trend
# ---------------------------------------------------------------------------


def test_linear_trend_perfect_slope() -> None:
    result = linear_trend([1.0, 2.0, 3.0], [2.0, 4.0, 6.0])
    assert abs(result.slope - 2.0) < 1e-9
    assert abs(result.intercept - 0.0) < 1e-9
    assert abs(result.r_squared - 1.0) < 1e-9


def test_linear_trend_flat() -> None:
    result = linear_trend([1.0, 2.0, 3.0], [5.0, 5.0, 5.0])
    assert abs(result.slope) < 1e-9
    assert abs(result.intercept - 5.0) < 1e-9
    assert result.r_squared == 0.0


def test_linear_trend_single_point_returns_zero_slope() -> None:
    result = linear_trend([2020.0], [10.0])
    assert result.slope == 0.0
    assert result.intercept == 10.0
    assert result.r_squared == 0.0


def test_linear_trend_stable_with_large_x() -> None:
    """Centered OLS must not lose precision for year-scale x values."""
    x = [2000.0, 2010.0, 2020.0]
    y = [2000.0, 2010.0, 2020.0]  # slope=1, intercept=0
    result = linear_trend(x, y)
    assert abs(result.slope - 1.0) < 1e-9
    assert abs(result.intercept) < 1e-6
    assert abs(result.r_squared - 1.0) < 1e-9


def test_hot_cold_yearly_figure_x_axis_is_years(sample_df: pl.DataFrame) -> None:
    fig = hot_cold_yearly_figure(sample_df, "TOULOUSE")
    for trace in fig.data:
        assert all(isinstance(y, int) for y in trace.x)


# ---------------------------------------------------------------------------
# monthly_avg_temp_figure
# ---------------------------------------------------------------------------


def test_monthly_avg_temp_figure_returns_figure(sample_df: pl.DataFrame) -> None:
    fig = monthly_avg_temp_figure(sample_df, "TOULOUSE")
    assert isinstance(fig, go.Figure)


def test_monthly_avg_temp_figure_has_two_named_line_traces(sample_df: pl.DataFrame) -> None:
    """Two main traces (Avg Tmax, Avg Tmin) + four invisible sigma-band boundary traces = 6 total."""
    fig = monthly_avg_temp_figure(sample_df, "TOULOUSE")
    assert all(isinstance(t, go.Scatter) for t in fig.data)
    named = [t for t in fig.data if t.name]
    assert len(named) == 2
    assert any("tmax" in t.name.lower() for t in named)
    assert any("tmin" in t.name.lower() for t in named)
    sigma_band = [t for t in fig.data if not t.name]
    assert len(sigma_band) == 4


def test_monthly_avg_temp_figure_title_contains_station(sample_df: pl.DataFrame) -> None:
    fig = monthly_avg_temp_figure(sample_df, "TOULOUSE")
    assert "TOULOUSE" in fig.layout.title.text


def test_monthly_avg_temp_figure_x_axis_always_all_12_months(sample_df: pl.DataFrame) -> None:
    """Even when data covers only January, all 12 month labels must appear."""
    fig = monthly_avg_temp_figure(sample_df, "TOULOUSE")
    expected = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    for trace in fig.data:
        assert list(trace.x) == expected


def test_monthly_avg_temp_figure_avg_tmin_correct(sample_df: pl.DataFrame) -> None:
    """January temp_min values: [-1.0, 0.5, 2.0, -2.0, 1.0] → mean = 0.1."""
    fig = monthly_avg_temp_figure(sample_df, "TOULOUSE")
    tmin_trace = next(t for t in fig.data if "tmin" in t.name.lower())
    assert abs(tmin_trace.y[0] - 0.1) < 1e-9


def test_monthly_avg_temp_figure_tmax_above_tmin(sample_df: pl.DataFrame) -> None:
    """For months that have data, Tmax must exceed Tmin; None months are skipped."""
    fig = monthly_avg_temp_figure(sample_df, "TOULOUSE")
    tmax_trace = next(t for t in fig.data if "tmax" in t.name.lower())
    tmin_trace = next(t for t in fig.data if "tmin" in t.name.lower())
    for tmax_val, tmin_val in zip(tmax_trace.y, tmin_trace.y, strict=True):
        if tmax_val is not None and tmin_val is not None:
            assert tmax_val > tmin_val


# ---------------------------------------------------------------------------
# monthly_avg_temp_by_decade_figure
# ---------------------------------------------------------------------------


@pytest.fixture
def multi_decade_df(sample_df: pl.DataFrame) -> pl.DataFrame:
    """sample_df (year 2020) extended with identical rows shifted to 2010 to create two decades."""
    older = sample_df.with_columns(
        pl.col("DATE").map_elements(lambda d: date(d.year - 10, d.month, d.day), return_dtype=pl.Date).alias("DATE")
    )
    return pl.concat([sample_df, older])


def test_monthly_avg_temp_by_decade_figure_returns_figure(sample_df: pl.DataFrame) -> None:
    fig = monthly_avg_temp_by_decade_figure(sample_df, "TOULOUSE")
    assert isinstance(fig, go.Figure)


def test_monthly_avg_temp_by_decade_figure_two_traces_per_decade(multi_decade_df: pl.DataFrame) -> None:
    """Two decades → 4 traces (Tmax + Tmin per decade)."""
    fig = monthly_avg_temp_by_decade_figure(multi_decade_df, "TOULOUSE")
    assert len(fig.data) == 4


def test_monthly_avg_temp_by_decade_figure_one_decade_two_traces(sample_df: pl.DataFrame) -> None:
    """sample_df has only 2020 data → one decade (2020s) → 2 traces."""
    fig = monthly_avg_temp_by_decade_figure(sample_df, "TOULOUSE")
    assert len(fig.data) == 2


def test_monthly_avg_temp_by_decade_figure_tmin_is_dashed(sample_df: pl.DataFrame) -> None:
    fig = monthly_avg_temp_by_decade_figure(sample_df, "TOULOUSE")
    tmin_trace = next(t for t in fig.data if "tmin" in t.name.lower())
    assert tmin_trace.line.dash == "dash"


def test_monthly_avg_temp_by_decade_figure_tmax_is_solid(sample_df: pl.DataFrame) -> None:
    fig = monthly_avg_temp_by_decade_figure(sample_df, "TOULOUSE")
    tmax_trace = next(t for t in fig.data if "tmax" in t.name.lower())
    assert tmax_trace.line.dash is None or tmax_trace.line.dash == "solid"


def test_monthly_avg_temp_by_decade_figure_title_contains_station(sample_df: pl.DataFrame) -> None:
    fig = monthly_avg_temp_by_decade_figure(sample_df, "TOULOUSE")
    assert "TOULOUSE" in fig.layout.title.text


def test_monthly_avg_temp_by_decade_figure_tmax_colors_distinct_from_tmin(multi_decade_df: pl.DataFrame) -> None:
    """Tmax and Tmin traces for the same decade must use different colours (separate gradients)."""
    fig = monthly_avg_temp_by_decade_figure(multi_decade_df, "TOULOUSE")
    for decade_label in {"2010s", "2020s"}:
        tmax = next(t for t in fig.data if t.name == f"{decade_label} Tmax")
        tmin = next(t for t in fig.data if t.name == f"{decade_label} Tmin")
        assert tmax.line.color != tmin.line.color


def test_monthly_avg_temp_by_decade_figure_newer_decade_redder_tmax(multi_decade_df: pl.DataFrame) -> None:
    """Newer decade Tmax should be closer to deep red (lower green channel) than older decade."""
    fig = monthly_avg_temp_by_decade_figure(multi_decade_df, "TOULOUSE")
    tmax_2010 = next(t for t in fig.data if t.name == "2010s Tmax")
    tmax_2020 = next(t for t in fig.data if t.name == "2020s Tmax")

    # Parse "rgb(r,g,b)" and compare green channel: orange has more green than deep red
    def green(color: str) -> int:
        return int(color.split(",")[1].strip())

    assert green(tmax_2010.line.color) > green(tmax_2020.line.color)


def test_monthly_avg_temp_by_decade_figure_newer_decade_darker_tmin(multi_decade_df: pl.DataFrame) -> None:
    """Newer decade Tmin should be closer to dark blue (lower blue channel value due to RGB mix)."""
    fig = monthly_avg_temp_by_decade_figure(multi_decade_df, "TOULOUSE")
    tmin_2010 = next(t for t in fig.data if t.name == "2010s Tmin")
    tmin_2020 = next(t for t in fig.data if t.name == "2020s Tmin")

    # Light blue has higher red channel than dark blue
    def red(color: str) -> int:
        return int(color.split(",")[0].replace("rgb(", "").strip())

    assert red(tmin_2010.line.color) > red(tmin_2020.line.color)


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
