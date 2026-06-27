from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from src.transforms import (
    DEFAULT_HOT_DAY,
    HOT_DAY_OPTIONS,
    LinearTrend,
    TmaxAndTmin,
    TmaxOnly,
    hot_day_from,
    linear_trend,
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
