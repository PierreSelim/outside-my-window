from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from src.transforms import LinearTrend, linear_trend, yearly_hot_cold


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
    return pl.DataFrame({
        "DATE": [date(2020, 1, 1), date(2020, 6, 1), date(2021, 7, 1), date(2021, 8, 1)],
        "temp_min": [-1.0, 21.0, 0.5, 21.0],
        "temp_max": [5.0,  36.0, 8.0, 36.0],
    }, schema={"DATE": pl.Date, "temp_min": pl.Float64, "temp_max": pl.Float64})


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
