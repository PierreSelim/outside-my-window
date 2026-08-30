from __future__ import annotations

import math
from datetime import date, timedelta

import polars as pl
import pytest

from src.transforms import (
    DEFAULT_HOT_DAY,
    DEFAULT_WINDOW,
    HOT_DAY_OPTIONS,
    DayWindow,
    Density,
    Distribution,
    LinearTrend,
    TmaxAndTmin,
    TmaxOnly,
    YearSpan,
    describe,
    gaussian_kde,
    hot_day_from,
    linear_trend,
    silverman_bandwidth,
    window_filter,
    year_filter,
    yearly_hot_cold,
)

# ---------------------------------------------------------------------------
# linear_trend
# ---------------------------------------------------------------------------


def test_linear_trend_returns_dataclass() -> None:
    result = linear_trend([1.0, 2.0, 3.0], [2.0, 4.0, 6.0])
    assert isinstance(result, LinearTrend)


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


def test_linear_trend_filters_none_y() -> None:
    result = linear_trend([1.0, 2.0, 3.0], [2.0, None, 6.0])
    assert abs(result.slope - 2.0) < 1e-9


def test_linear_trend_zero_variance_x_returns_mean() -> None:
    result = linear_trend([5.0, 5.0, 5.0], [1.0, 2.0, 3.0])
    assert result.slope == 0.0
    assert abs(result.intercept - 2.0) < 1e-9


# ---------------------------------------------------------------------------
# yearly_hot_cold
# ---------------------------------------------------------------------------


@pytest.fixture
def daily_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "DATE": [date(2020, 1, 1), date(2020, 6, 1), date(2021, 7, 1), date(2021, 8, 1)],
            "temp_min": [-1.0, 21.0, 0.5, 21.0],
            "temp_max": [5.0, 36.0, 8.0, 36.0],
        },
        schema={"DATE": pl.Date, "temp_min": pl.Float64, "temp_max": pl.Float64},
    )


def test_yearly_hot_cold_returns_dataframe(daily_df: pl.DataFrame) -> None:
    result = yearly_hot_cold(daily_df)
    assert isinstance(result, pl.DataFrame)
    assert set(result.columns) >= {"year", "hot_days", "cold_days"}


def test_yearly_hot_cold_sorted_by_year(daily_df: pl.DataFrame) -> None:
    result = yearly_hot_cold(daily_df)
    years = result["year"].to_list()
    assert years == sorted(years)


def test_yearly_hot_cold_counts_hot_days(daily_df: pl.DataFrame) -> None:
    result = yearly_hot_cold(daily_df)
    row_2020 = result.filter(pl.col("year") == 2020).row(0, named=True)
    row_2021 = result.filter(pl.col("year") == 2021).row(0, named=True)
    assert row_2020["hot_days"] == 1
    assert row_2021["hot_days"] == 1


def test_yearly_hot_cold_counts_cold_days(daily_df: pl.DataFrame) -> None:
    result = yearly_hot_cold(daily_df)
    row_2020 = result.filter(pl.col("year") == 2020).row(0, named=True)
    assert row_2020["cold_days"] == 1  # only -1.0 is below 0


def test_yearly_hot_cold_no_false_positives(daily_df: pl.DataFrame) -> None:
    result = yearly_hot_cold(daily_df)
    row_2021 = result.filter(pl.col("year") == 2021).row(0, named=True)
    assert row_2021["cold_days"] == 0


# ---------------------------------------------------------------------------
# HotDayDefinition — label properties
# ---------------------------------------------------------------------------


def test_tmax_only_label() -> None:
    assert TmaxOnly(32.0).label == "Tmax≥32°C"


def test_tmax_only_label_large_value() -> None:
    assert TmaxOnly(40.0).label == "Tmax≥40°C"


def test_tmax_and_tmin_label() -> None:
    assert TmaxAndTmin(tmax_min=35.0, tmin_min=20.0).label == "Tmin≥20°C & Tmax≥35°C"


# ---------------------------------------------------------------------------
# hot_day_from
# ---------------------------------------------------------------------------


def test_hot_day_from_known_tmax_only_label_returns_matching_option() -> None:
    tmax_only = TmaxOnly(32.0)
    assert hot_day_from(tmax_only.label) == tmax_only


def test_hot_day_from_default_label_returns_default() -> None:
    assert hot_day_from(DEFAULT_HOT_DAY.label) == DEFAULT_HOT_DAY


def test_hot_day_from_unknown_label_returns_default() -> None:
    assert hot_day_from("not a real label") == DEFAULT_HOT_DAY


def test_hot_day_from_all_options_round_trip() -> None:
    """Every option in HOT_DAY_OPTIONS must be recoverable from its own label."""
    for option in HOT_DAY_OPTIONS:
        assert hot_day_from(option.label) == option


# ---------------------------------------------------------------------------
# yearly_hot_cold — HotDayDefinition
# ---------------------------------------------------------------------------


@pytest.fixture
def tmax_only_df() -> pl.DataFrame:
    """DataFrame with one day where tmax=33≥32 but tmin=15<20.

    This day qualifies under TmaxOnly(32) but NOT under the default TmaxAndTmin(35, 20),
    which requires both tmax≥35 AND tmin≥20.
    """
    return pl.DataFrame(
        {
            "DATE": [date(2023, 7, 1), date(2023, 7, 2)],
            "temp_min": [15.0, 0.5],
            "temp_max": [33.0, 10.0],
        },
        schema={"DATE": pl.Date, "temp_min": pl.Float64, "temp_max": pl.Float64},
    )


def test_yearly_hot_cold_tmax_only_counts_differently_from_default(tmax_only_df: pl.DataFrame) -> None:
    """Day with tmax=33, tmin=15: counted by TmaxOnly(32) but not by TmaxAndTmin(35, 20)."""
    result_default = yearly_hot_cold(tmax_only_df)
    result_tmax_only = yearly_hot_cold(tmax_only_df, TmaxOnly(32.0))
    row_default = result_default.filter(pl.col("year") == 2023).row(0, named=True)
    row_tmax_only = result_tmax_only.filter(pl.col("year") == 2023).row(0, named=True)
    assert row_default["hot_days"] == 0
    assert row_tmax_only["hot_days"] == 1


def test_yearly_hot_cold_tmax_only_cold_days_unchanged(tmax_only_df: pl.DataFrame) -> None:
    """cold_days definition (temp_min < 0) is unaffected by the HotDayDefinition choice."""
    result_default = yearly_hot_cold(tmax_only_df)
    result_tmax_only = yearly_hot_cold(tmax_only_df, TmaxOnly(32.0))
    assert result_default["cold_days"].to_list() == result_tmax_only["cold_days"].to_list()


# ---------------------------------------------------------------------------
# DayWindow
# ---------------------------------------------------------------------------


def test_day_window_from_dates_encodes_mmdd() -> None:
    window = DayWindow.from_dates(date(2000, 6, 1), date(2000, 8, 31))
    assert (window.start_md, window.end_md) == (601, 831)


def test_day_window_from_dates_ignores_the_year() -> None:
    assert DayWindow.from_dates(date(1999, 6, 1), date(2024, 8, 31)) == DEFAULT_WINDOW


def test_day_window_label() -> None:
    assert DayWindow(601, 831).label == "1 Jun \u2013 31 Aug"


def test_day_window_wraps_when_end_precedes_start() -> None:
    assert DayWindow(1201, 228).wraps
    assert not DayWindow(601, 831).wraps


# ---------------------------------------------------------------------------
# window_filter
# ---------------------------------------------------------------------------


def _daily_df(start: date, days: int) -> pl.DataFrame:
    dates = [start + timedelta(days=i) for i in range(days)]
    return pl.DataFrame({"DATE": dates, "temp_max": [float(i % 30) for i in range(days)]})


def test_window_filter_keeps_only_dates_inside_the_window() -> None:
    df = _daily_df(date(2020, 1, 1), 366)
    result = window_filter(df, YearSpan(2020, 2020), DayWindow(601, 831))
    months = set(result["DATE"].dt.month().to_list())
    assert months == {6, 7, 8}
    assert result.height == 30 + 31 + 31


def test_window_filter_respects_year_bounds() -> None:
    df = _daily_df(date(2018, 1, 1), 365 * 4)
    result = window_filter(df, YearSpan(2019, 2020), DayWindow(601, 831))
    assert set(result["DATE"].dt.year().to_list()) == {2019, 2020}


def test_window_filter_wrap_around_keeps_december_and_january() -> None:
    df = _daily_df(date(2020, 1, 1), 366)
    result = window_filter(df, YearSpan(2020, 2020), DayWindow(1201, 228))
    months = set(result["DATE"].dt.month().to_list())
    assert months == {12, 1, 2}


def test_window_filter_no_match_returns_empty_frame() -> None:
    df = _daily_df(date(2020, 6, 1), 30)
    result = window_filter(df, YearSpan(2020, 2020), DayWindow(101, 131))
    assert result.is_empty()


def test_year_span_of_orders_an_inverted_pair() -> None:
    """The end year typed before the start is a user mid-edit, not an error."""
    assert YearSpan.of(2020, 1991) == YearSpan(1991, 2020)


def test_year_span_label_collapses_a_single_year() -> None:
    assert YearSpan(1991, 2020).label == "1991–2020"
    assert YearSpan(2020, 2020).label == "2020"


def test_year_filter_keeps_only_the_years_in_span() -> None:
    df = _daily_df(date(2018, 1, 1), 365 * 4)
    assert set(year_filter(df, YearSpan(2019, 2020))["DATE"].dt.year().to_list()) == {2019, 2020}


def test_window_filter_month_boundary_does_not_overflow_int8() -> None:
    """month * 100 must not wrap: dt.month() is Int8, so December (1200) needs a wider type."""
    df = _daily_df(date(2020, 1, 1), 366)
    result = window_filter(df, YearSpan(2020, 2020), DayWindow(1215, 1231))
    assert result.height == 17


# ---------------------------------------------------------------------------
# describe
# ---------------------------------------------------------------------------


def test_describe_returns_distribution() -> None:
    result = describe(pl.Series([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert isinstance(result, Distribution)
    assert result.n == 5
    assert result.mean == pytest.approx(3.0)
    assert result.median == pytest.approx(3.0)


def test_describe_quantiles() -> None:
    result = describe(pl.Series([float(i) for i in range(1, 101)]))
    assert result is not None
    assert result.p10 == pytest.approx(10.0, abs=1.0)
    assert result.p90 == pytest.approx(90.0, abs=1.0)


def test_describe_ignores_nulls() -> None:
    result = describe(pl.Series([1.0, None, 3.0]))
    assert result is not None
    assert result.n == 2
    assert result.mean == pytest.approx(2.0)


def test_describe_all_null_returns_none() -> None:
    assert describe(pl.Series([None, None], dtype=pl.Float64)) is None


def test_describe_empty_returns_none() -> None:
    assert describe(pl.Series([], dtype=pl.Float64)) is None


# ---------------------------------------------------------------------------
# silverman_bandwidth
# ---------------------------------------------------------------------------


def test_silverman_bandwidth_is_positive_for_a_spread_sample() -> None:
    assert silverman_bandwidth(pl.Series([1.0, 2.0, 3.0, 4.0, 5.0])) > 0


def test_silverman_bandwidth_is_zero_for_a_constant_sample() -> None:
    assert silverman_bandwidth(pl.Series([7.0] * 10)) == 0.0


def test_silverman_bandwidth_shrinks_as_the_sample_grows() -> None:
    small = silverman_bandwidth(pl.Series([float(i % 10) for i in range(20)]))
    large = silverman_bandwidth(pl.Series([float(i % 10) for i in range(2000)]))
    assert large < small


def test_silverman_bandwidth_uses_the_robust_scale_against_outliers() -> None:
    """One absurd outlier inflates the stdev; the IQR-based scale must keep the bandwidth sane."""
    clean = pl.Series([float(i % 10) for i in range(200)])
    with_outlier = pl.Series([float(i % 10) for i in range(199)] + [10_000.0])
    assert silverman_bandwidth(with_outlier) < 2 * silverman_bandwidth(clean)


# ---------------------------------------------------------------------------
# gaussian_kde
# ---------------------------------------------------------------------------

_SAMPLE = pl.Series([float(v) for v in (18, 19, 20, 20, 21, 21, 21, 22, 22, 23, 24)])


def test_gaussian_kde_returns_a_density() -> None:
    result = gaussian_kde(_SAMPLE)
    assert isinstance(result, Density)
    assert len(result.x) == len(result.y)


def test_gaussian_kde_grid_is_dense_and_ordered() -> None:
    result = gaussian_kde(_SAMPLE)
    assert result is not None
    assert len(result.x) > 100
    assert result.x == sorted(result.x)


def test_gaussian_kde_is_non_negative() -> None:
    result = gaussian_kde(_SAMPLE)
    assert result is not None
    assert all(value >= 0 for value in result.y)


def test_gaussian_kde_integrates_to_one() -> None:
    """The defining property of a PDF — trapezoid rule over the grid."""
    result = gaussian_kde(_SAMPLE)
    assert result is not None
    step = result.x[1] - result.x[0]
    area = sum((result.y[i] + result.y[i + 1]) / 2 * step for i in range(len(result.y) - 1))
    assert area == pytest.approx(1.0, abs=0.01)


def test_gaussian_kde_peaks_near_the_mode() -> None:
    result = gaussian_kde(_SAMPLE)
    assert result is not None
    peak = result.x[max(range(len(result.y)), key=lambda i: result.y[i])]
    assert peak == pytest.approx(21.0, abs=1.0)


def test_gaussian_kde_grid_extends_past_the_extremes() -> None:
    result = gaussian_kde(_SAMPLE)
    assert result is not None
    assert result.x[0] < 18.0
    assert result.x[-1] > 24.0


def test_gaussian_kde_honours_an_explicit_bandwidth() -> None:
    wide = gaussian_kde(_SAMPLE, bandwidth=5.0)
    narrow = gaussian_kde(_SAMPLE, bandwidth=0.3)
    assert wide is not None and narrow is not None
    assert wide.bandwidth == 5.0
    assert max(narrow.y) > max(wide.y)


def test_gaussian_kde_matches_the_direct_kernel_sum() -> None:
    """The binned-by-0.1-°C shortcut must agree with the textbook O(n*grid) formula."""
    result = gaussian_kde(_SAMPLE, bandwidth=1.0)
    assert result is not None
    values = _SAMPLE.to_list()
    norm = 1.0 / (len(values) * 1.0 * math.sqrt(2 * math.pi))
    for index in (0, 60, 128, 200, 255):
        expected = norm * sum(math.exp(-0.5 * ((result.x[index] - v) / 1.0) ** 2) for v in values)
        assert result.y[index] == pytest.approx(expected, abs=1e-9)


def test_gaussian_kde_ignores_nulls() -> None:
    assert gaussian_kde(pl.Series([18.0, None, 24.0])) is not None


def test_gaussian_kde_single_value_returns_none() -> None:
    assert gaussian_kde(pl.Series([20.0])) is None


def test_gaussian_kde_constant_sample_returns_none() -> None:
    assert gaussian_kde(pl.Series([20.0] * 50)) is None


def test_gaussian_kde_empty_returns_none() -> None:
    assert gaussian_kde(pl.Series([], dtype=pl.Float64)) is None
