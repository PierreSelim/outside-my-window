from __future__ import annotations

import time
from datetime import date, timedelta

import plotly.graph_objects as go
import polars as pl
import pytest

from src.charts import (
    density_comparison_figure,
    empty_figure,
    hot_cold_yearly_figure,
    monthly_avg_temp_by_decade_figure,
    monthly_avg_temp_figure,
    precipitation_figure,
    temperature_figure,
    wind_figure,
)
from src.data_loader import Granularity
from src.transforms import TmaxOnly


def yearly_fig(df: pl.DataFrame, station: str = "TOULOUSE", **kwargs: object) -> go.Figure:
    """hot_cold_yearly_figure with the coverage guard off — these fixtures are a handful of days."""
    kwargs.setdefault("min_coverage", 0.0)
    return hot_cold_yearly_figure(df, station, **kwargs)  # type: ignore[arg-type]


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


def test_temperature_figure_merges_consecutive_hot_days_into_one_band(sample_df: pl.DataFrame) -> None:
    """A heatwave is one block, not five stripes — and it keeps the shape count bounded."""
    df = sample_df.with_columns(
        [
            pl.lit(21.0).cast(pl.Float64).alias("temp_min"),
            pl.lit(36.0).cast(pl.Float64).alias("temp_max"),
        ]
    )
    toulouse = df.filter(pl.col("station_id") == 31001)  # 3 consecutive hot days
    fig = temperature_figure(toulouse, "TOULOUSE")
    assert len(fig.layout.shapes) == 1
    assert fig.layout.shapes[0].x0 == date(2020, 1, 1)
    assert fig.layout.shapes[0].x1 == date(2020, 1, 4)


def test_temperature_figure_separates_hot_days_that_are_not_consecutive(sample_df: pl.DataFrame) -> None:
    df = sample_df.filter(pl.col("station_id") == 31001).with_columns(
        pl.when(pl.col("DATE") == date(2020, 1, 2))
        .then(pl.lit(10.0))
        .otherwise(pl.lit(36.0))
        .cast(pl.Float64)
        .alias("temp_max"),
        pl.lit(21.0).cast(pl.Float64).alias("temp_min"),
    )
    fig = temperature_figure(df, "TOULOUSE")
    assert len(fig.layout.shapes) == 2


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
    fig = yearly_fig(sample_df, "TOULOUSE")
    assert isinstance(fig, go.Figure)


def test_hot_cold_yearly_figure_has_three_line_traces(sample_df: pl.DataFrame) -> None:
    fig = yearly_fig(sample_df, "TOULOUSE")
    assert len(fig.data) == 3  # hot days, tropical nights, frost days
    assert all(isinstance(t, go.Scatter) for t in fig.data)


def test_hot_cold_yearly_figure_counts_tropical_nights() -> None:
    df = pl.DataFrame(
        {
            "DATE": [date(2020, 7, 1), date(2020, 7, 2)],
            "temp_min": [21.0, 12.0],
            "temp_max": [28.0, 24.0],
        },
        schema={"DATE": pl.Date, "temp_min": pl.Float64, "temp_max": pl.Float64},
    )
    fig = yearly_fig(df, "TOULOUSE")
    tropical = next(t for t in fig.data if "tropical" in t.name.lower())
    assert sum(v for v in tropical.y if v is not None) == 1


def test_hot_cold_yearly_figure_counts_are_drawn_straight() -> None:
    # Annual counts are integers on a discrete axis: a spline invents values between them.
    fig = yearly_fig(sample_year_df(), "TOULOUSE")
    assert all(t.line.shape is None for t in fig.data)


def test_hot_cold_yearly_figure_title_contains_station(sample_df: pl.DataFrame) -> None:
    fig = yearly_fig(sample_df, "TOULOUSE")
    assert "TOULOUSE" in fig.layout.title.text


def test_hot_cold_yearly_figure_counts_cold_days(sample_df: pl.DataFrame) -> None:
    """sample_df has temp_min = [-1.0, 0.5, 2.0, -2.0, 1.0] — 2 negative values in year 2020."""
    df = sample_df.filter(pl.col("DATE").dt.year() == 2020)
    fig = yearly_fig(df, "TOULOUSE")
    cold_trace = next(t for t in fig.data if "frost" in t.name.lower())
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
    fig = yearly_fig(df, "TOULOUSE")
    hot_trace = next(t for t in fig.data if "hot" in t.name.lower())
    assert sum(hot_trace.y) == 1


def test_hot_cold_yearly_figure_no_trend_by_default(sample_df: pl.DataFrame) -> None:
    fig = yearly_fig(sample_df, "TOULOUSE")
    assert len(fig.data) == 3


def test_hot_cold_yearly_figure_trend_adds_one_trace_per_series(multi_decade_df: pl.DataFrame) -> None:
    fig = yearly_fig(multi_decade_df, "TOULOUSE", show_trend=True)
    assert len(fig.data) == 6
    trend_traces = [t for t in fig.data if "trend" in t.name.lower()]
    assert len(trend_traces) == 3


def test_hot_cold_yearly_figure_trend_traces_are_dashed_lines(multi_decade_df: pl.DataFrame) -> None:
    fig = yearly_fig(multi_decade_df, "TOULOUSE", show_trend=True)
    for trace in fig.data:
        if "trend" in trace.name.lower():
            assert trace.line.dash == "dash"
            assert trace.mode == "lines"


def test_hot_cold_yearly_figure_trend_same_x_as_data(multi_decade_df: pl.DataFrame) -> None:
    fig = yearly_fig(multi_decade_df, "TOULOUSE", show_trend=True)
    data_x = fig.data[0].x
    for trace in fig.data:
        if "trend" in trace.name.lower():
            assert list(trace.x) == list(data_x)


def test_hot_cold_yearly_figure_x_axis_is_years(sample_df: pl.DataFrame) -> None:
    fig = yearly_fig(sample_df, "TOULOUSE")
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


# ---------------------------------------------------------------------------
# hot_cold_yearly_figure — current_year provisional point
# ---------------------------------------------------------------------------


@pytest.fixture
def two_year_df() -> pl.DataFrame:
    """DataFrame with a complete year (2019) and a provisional current year (2020).

    2019-07-01: hot day under default definition (tmin=21≥20, tmax=36≥35).
    2019-08-01: cold day (tmin=-1<0).
    2020-07-01: hot day under default definition (provisional, i.e. current year).
    """
    return pl.DataFrame(
        {
            "DATE": [date(2019, 7, 1), date(2019, 8, 1), date(2020, 7, 1)],
            "temp_min": [21.0, -1.0, 21.0],
            "temp_max": [36.0, 5.0, 36.0],
        },
        schema={"DATE": pl.Date, "temp_min": pl.Float64, "temp_max": pl.Float64},
    )


def test_hot_cold_yearly_figure_current_year_none_gives_two_traces(sample_df: pl.DataFrame) -> None:
    """Explicit current_year=None: provisional path off, only 2 solid traces returned."""
    fig = yearly_fig(sample_df, "TOULOUSE", current_year=None)
    assert len(fig.data) == 3


def test_hot_cold_yearly_figure_current_year_adds_dotted_traces(two_year_df: pl.DataFrame) -> None:
    """With current_year set and data for that year, one dotted connector per counted series."""
    fig = yearly_fig(two_year_df, "STATION", current_year=2020)
    dot_traces = [t for t in fig.data if getattr(t.line, "dash", None) == "dot"]
    assert len(dot_traces) == 3


def test_hot_cold_yearly_figure_total_traces_with_current_year(two_year_df: pl.DataFrame) -> None:
    """3 solid + 3 dotted when current_year is set and present in the data."""
    fig = yearly_fig(two_year_df, "STATION", current_year=2020)
    assert len(fig.data) == 6


def test_hot_cold_yearly_figure_dotted_traces_have_open_circle_marker(two_year_df: pl.DataFrame) -> None:
    fig = yearly_fig(two_year_df, "STATION", current_year=2020)
    dot_traces = [t for t in fig.data if getattr(t.line, "dash", None) == "dot"]
    for trace in dot_traces:
        assert trace.marker.symbol == "circle-open"


def test_hot_cold_yearly_figure_dotted_traces_not_in_legend(two_year_df: pl.DataFrame) -> None:
    fig = yearly_fig(two_year_df, "STATION", current_year=2020)
    dot_traces = [t for t in fig.data if getattr(t.line, "dash", None) == "dot"]
    for trace in dot_traces:
        assert trace.showlegend is False


def test_hot_cold_yearly_figure_solid_hot_trace_excludes_current_year(two_year_df: pl.DataFrame) -> None:
    """The current year carries no solid value — it is drawn only as the provisional point."""
    fig = yearly_fig(two_year_df, "STATION", current_year=2020)
    solid_hot = next(t for t in fig.data if "hot" in t.name.lower() and getattr(t.line, "dash", None) != "dot")
    assert solid_hot.y[list(solid_hot.x).index(2020)] is None


def test_hot_cold_yearly_figure_solid_cold_trace_excludes_current_year(two_year_df: pl.DataFrame) -> None:
    """The frost series follows the same provisional rule as the hot one."""
    fig = yearly_fig(two_year_df, "STATION", current_year=2020)
    solid = next(t for t in fig.data if "frost" in t.name.lower() and getattr(t.line, "dash", None) != "dot")
    assert solid.y[list(solid.x).index(2020)] is None


def test_hot_cold_yearly_figure_no_dotted_traces_when_current_year_absent(two_year_df: pl.DataFrame) -> None:
    """If current_year has no matching row in the aggregated data, no dotted trace is added."""
    fig = yearly_fig(two_year_df, "STATION", current_year=2025)
    dot_traces = [t for t in fig.data if getattr(t.line, "dash", None) == "dot"]
    assert len(dot_traces) == 0


# ---------------------------------------------------------------------------
# hot_cold_yearly_figure — HotDayDefinition
# ---------------------------------------------------------------------------


def test_hot_cold_yearly_figure_hot_trace_name_contains_definition_label() -> None:
    """Solid hot trace name must embed the definition's label string."""
    definition = TmaxOnly(32.0)
    df = pl.DataFrame(
        {
            "DATE": [date(2020, 7, 1)],
            "temp_min": [15.0],
            "temp_max": [33.0],
        },
        schema={"DATE": pl.Date, "temp_min": pl.Float64, "temp_max": pl.Float64},
    )
    fig = yearly_fig(df, "STATION", definition=definition)
    hot_trace = next(t for t in fig.data if "hot" in t.name.lower())
    assert definition.label in hot_trace.name


# ---------------------------------------------------------------------------
# density_comparison_figure
# ---------------------------------------------------------------------------

_SERIES_A = pl.Series([20.0, 21.0, 21.5, 22.0, 22.5, 23.0, 24.0])
_SERIES_B = pl.Series([24.0, 25.0, 25.5, 26.0, 26.5, 27.0, 28.0])


def test_density_comparison_figure_has_two_line_traces() -> None:
    fig = density_comparison_figure(_SERIES_A, _SERIES_B, "A", "B", "Title")
    assert len(fig.data) == 2
    assert all(trace.mode == "lines" for trace in fig.data)


def test_density_comparison_figure_curves_are_smooth_not_binned() -> None:
    """A histogram would give a handful of flat steps; a KDE gives a dense curve."""
    fig = density_comparison_figure(_SERIES_A, _SERIES_B, "A", "B", "Title")
    assert len(fig.data[0].x) > 100
    assert len(set(fig.data[0].y)) > 100


def test_density_comparison_figure_curves_are_filled_to_zero() -> None:
    fig = density_comparison_figure(_SERIES_A, _SERIES_B, "A", "B", "Title")
    assert all(trace.fill == "tozeroy" for trace in fig.data)


def test_density_comparison_figure_draws_a_mean_line_per_period() -> None:
    fig = density_comparison_figure(_SERIES_A, _SERIES_B, "A", "B", "Title")
    assert len(fig.layout.shapes) == 2
    assert {round(shape.x0, 1) for shape in fig.layout.shapes} == {22.0, 26.0}


def test_density_comparison_figure_uses_the_labels_as_trace_names() -> None:
    fig = density_comparison_figure(_SERIES_A, _SERIES_B, "1961-1990", "1995-2024", "Title")
    assert [trace.name for trace in fig.data] == ["1961-1990", "1995-2024"]


def test_density_comparison_figure_title() -> None:
    fig = density_comparison_figure(_SERIES_A, _SERIES_B, "A", "B", "Daily Tmax")
    assert fig.layout.title.text == "Daily Tmax"


def test_density_comparison_figure_separated_samples_peak_apart() -> None:
    fig = density_comparison_figure(_SERIES_A, _SERIES_B, "A", "B", "Title")
    peaks = [trace.x[max(range(len(trace.y)), key=lambda i: trace.y[i])] for trace in fig.data]
    assert peaks[0] < peaks[1]


def test_density_comparison_figure_empty_period_returns_placeholder() -> None:
    fig = density_comparison_figure(pl.Series([], dtype=pl.Float64), _SERIES_B, "A", "B", "Title")
    assert len(fig.data) == 0


def test_density_comparison_figure_all_null_period_returns_placeholder() -> None:
    fig = density_comparison_figure(_SERIES_A, pl.Series([None, None], dtype=pl.Float64), "A", "B", "Title")
    assert len(fig.data) == 0


def test_density_comparison_figure_constant_period_returns_placeholder() -> None:
    """A zero-variance sample has no bandwidth and therefore no density."""
    fig = density_comparison_figure(pl.Series([20.0] * 10), _SERIES_B, "A", "B", "Title")
    assert len(fig.data) == 0


def sample_year_df(years: tuple[int, ...] = (2020, 2021), days: int = 366) -> pl.DataFrame:
    """A fully observed record: `days` consecutive days from 1 January of each year."""
    dates = [date(y, 1, 1) + timedelta(days=i) for y in years for i in range(days)]
    return pl.DataFrame(
        {"DATE": dates, "temp_min": [5.0] * len(dates), "temp_max": [20.0] * len(dates)},
        schema={"DATE": pl.Date, "temp_min": pl.Float64, "temp_max": pl.Float64},
    )


def test_hot_cold_yearly_figure_excludes_years_the_station_barely_observed() -> None:
    """A year of 20 observed days is a measure of downtime, not of a cool summer."""
    full = sample_year_df((2020,))
    sparse = pl.DataFrame(
        {
            "DATE": [date(2021, 1, 1) + timedelta(days=i) for i in range(20)],
            "temp_min": [5.0] * 20,
            "temp_max": [20.0] * 20,
        },
        schema={"DATE": pl.Date, "temp_min": pl.Float64, "temp_max": pl.Float64},
    )
    fig = hot_cold_yearly_figure(pl.concat([full, sparse]), "TOULOUSE")
    hot = next(t for t in fig.data if "hot" in t.name.lower())
    assert hot.y[list(hot.x).index(2021)] is None
    assert hot.y[list(hot.x).index(2020)] is not None


def test_hot_cold_yearly_figure_marks_the_excluded_years() -> None:
    sparse = pl.DataFrame(
        {"DATE": [date(2021, 1, 1)], "temp_min": [5.0], "temp_max": [20.0]},
        schema={"DATE": pl.Date, "temp_min": pl.Float64, "temp_max": pl.Float64},
    )
    fig = hot_cold_yearly_figure(pl.concat([sample_year_df((2020,)), sparse]), "TOULOUSE")
    assert len(fig.layout.shapes) == 1
    assert any("excluded" in (t.name or "") for t in fig.data)


def test_hot_cold_yearly_figure_trend_ignores_the_excluded_years() -> None:
    """The regression must run on the same population as the solid line."""
    fig = hot_cold_yearly_figure(sample_year_df((2020, 2021, 2022)), "TOULOUSE", show_trend=True)
    trend = next(t for t in fig.data if "trend" in (t.name or "").lower())
    assert list(trend.x) == [2020, 2021, 2022]


def test_temperature_figure_keeps_the_gap_in_a_broken_record(sample_df: pl.DataFrame) -> None:
    """A missing day is a hole in the line, not a straight segment drawn across it."""
    df = sample_df.filter(pl.col("station_id") == 31001).with_columns(
        pl.when(pl.col("DATE") == date(2020, 1, 2))
        .then(None)
        .otherwise(pl.col("temp_max"))
        .cast(pl.Float64)
        .alias("temp_max")
    )
    fig = temperature_figure(df, "TOULOUSE")
    max_trace = next(t for t in fig.data if t.name == "Max")
    assert len(max_trace.y) == 3
    assert max_trace.y[1] is None


def test_precipitation_figure_labels_the_accumulation_period(sample_df: pl.DataFrame) -> None:
    daily = precipitation_figure(sample_df, "TOULOUSE", Granularity.DAY)
    monthly = precipitation_figure(sample_df, "TOULOUSE", Granularity.MONTH)
    assert daily.layout.yaxis.title.text == "mm"
    assert monthly.layout.yaxis.title.text == "mm/month"


def test_decade_figure_hides_tmin_behind_the_legend() -> None:
    fig = monthly_avg_temp_by_decade_figure(sample_year_df((2000, 2010)), "TOULOUSE")
    tmin = [t for t in fig.data if "Tmin" in t.name]
    assert tmin and all(t.visible == "legendonly" for t in tmin)


def test_decade_figure_draws_only_the_selected_decades() -> None:
    df = sample_year_df((2000, 2010, 2020))
    fig = monthly_avg_temp_by_decade_figure(df, "TOULOUSE", selected=[2000, 2020])
    labels = {t.name.split()[0] for t in fig.data}
    assert labels == {"2000s", "2020s"}


def test_temperature_figure_stays_fast_with_many_hot_days() -> None:
    """Shapes are assigned in one call: `add_vrect` per day made this quadratic (~130 s)."""
    days = 20 * 365
    dates = [date(2006, 1, 1) + timedelta(days=i) for i in range(days)]
    df = pl.DataFrame(
        {"DATE": dates, "temp_min": [18.0] * days, "temp_max": [31.0 if i % 3 else 12.0 for i in range(days)]},
        schema={"DATE": pl.Date, "temp_min": pl.Float64, "temp_max": pl.Float64},
    )
    start = time.perf_counter()
    fig = temperature_figure(df, "TOULOUSE")
    elapsed = time.perf_counter() - start
    assert len(fig.layout.shapes) > 1000
    assert elapsed < 5.0
