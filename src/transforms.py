from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from src.data_loader import HOT_DAY_TMAX, HOT_DAY_TMIN


@dataclass(frozen=True)
class LinearTrend:
    slope: float
    intercept: float
    r_squared: float


def linear_trend(x: list[float | int], y: list[float | int | None]) -> LinearTrend:
    """Return OLS linear regression for paired x/y data.

    Filters out pairs where y is None. Centers x before computing to avoid
    catastrophic cancellation with large x-values (e.g. years). Returns
    slope=0 and r_squared=0 when there are fewer than 2 valid points or zero x-variance.
    """
    pairs = [(float(xi), float(yi)) for xi, yi in zip(x, y, strict=True) if yi is not None]
    n = len(pairs)
    if n < 2:
        return LinearTrend(0.0, pairs[0][1] if n == 1 else 0.0, 0.0)
    xs = [p[0] for p in pairs]
    mean_x = sum(xs) / n
    xc = [xi - mean_x for xi in xs]
    yv = [p[1] for p in pairs]
    sum_xc2 = sum(v * v for v in xc)
    if sum_xc2 == 0:
        return LinearTrend(0.0, sum(yv) / n, 0.0)
    slope = sum(xci * yi for xci, yi in zip(xc, yv, strict=True)) / sum_xc2
    intercept = sum(yv) / n - slope * mean_x
    mean_y = sum(yv) / n
    ss_tot = sum((yi - mean_y) ** 2 for yi in yv)
    r_squared = (
        0.0
        if ss_tot == 0
        else 1.0 - sum((yi - (slope * xi + intercept)) ** 2 for xi, yi in zip(xs, yv, strict=True)) / ss_tot
    )
    return LinearTrend(slope, intercept, r_squared)


def yearly_hot_cold(df: pl.DataFrame) -> pl.DataFrame:
    """Aggregate daily station data into yearly hot-day and cold-day counts.

    Returns a DataFrame with columns: year, hot_days, cold_days, sorted ascending by year.
    """
    return (
        df.with_columns(pl.col("DATE").dt.year().alias("year"))
        .group_by("year")
        .agg(
            [
                (
                    pl.col("temp_min").is_not_null()
                    & pl.col("temp_max").is_not_null()
                    & (pl.col("temp_min") >= HOT_DAY_TMIN)
                    & (pl.col("temp_max") >= HOT_DAY_TMAX)
                )
                .sum()
                .alias("hot_days"),
                (pl.col("temp_min").is_not_null() & (pl.col("temp_min") < 0)).sum().alias("cold_days"),
            ]
        )
        .sort("year")
    )
