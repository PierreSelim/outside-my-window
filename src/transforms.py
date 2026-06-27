from __future__ import annotations

from dataclasses import dataclass
from typing import assert_never

import polars as pl

from src.data_loader import HOT_DAY_TMAX, HOT_DAY_TMIN


@dataclass(frozen=True)
class LinearTrend:
    slope: float
    intercept: float
    r_squared: float


@dataclass(frozen=True)
class TmaxOnly:
    tmax_min: float

    @property
    def label(self) -> str:
        return f"Tmax≥{self.tmax_min:.0f}°C"


@dataclass(frozen=True)
class TmaxAndTmin:
    tmax_min: float
    tmin_min: float

    @property
    def label(self) -> str:
        return f"Tmin≥{self.tmin_min:.0f}°C & Tmax≥{self.tmax_min:.0f}°C"


type HotDayDefinition = TmaxOnly | TmaxAndTmin

DEFAULT_HOT_DAY: HotDayDefinition = TmaxAndTmin(tmax_min=HOT_DAY_TMAX, tmin_min=HOT_DAY_TMIN)
HOT_DAY_OPTIONS: list[HotDayDefinition] = [
    DEFAULT_HOT_DAY,
    TmaxOnly(32.0),
    TmaxOnly(35.0),
    TmaxOnly(36.0),
    TmaxOnly(37.0),
    TmaxOnly(38.0),
    TmaxOnly(39.0),
    TmaxOnly(40.0),
]


def hot_day_from(label: str) -> HotDayDefinition:
    return next((d for d in HOT_DAY_OPTIONS if d.label == label), DEFAULT_HOT_DAY)


def _hot_predicate(d: HotDayDefinition) -> pl.Expr:
    match d:
        case TmaxOnly():
            return pl.col("temp_max").is_not_null() & (pl.col("temp_max") >= d.tmax_min)
        case TmaxAndTmin():
            return (
                pl.col("temp_max").is_not_null()
                & (pl.col("temp_max") >= d.tmax_min)
                & pl.col("temp_min").is_not_null()
                & (pl.col("temp_min") >= d.tmin_min)
            )
        case _:
            assert_never(d)


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


def yearly_hot_cold(df: pl.DataFrame, definition: HotDayDefinition = DEFAULT_HOT_DAY) -> pl.DataFrame:
    """Aggregate daily station data into yearly hot-day and cold-day counts.

    Returns a DataFrame with columns: year, hot_days, cold_days, sorted ascending by year.
    """
    return (
        df.with_columns(pl.col("DATE").dt.year().alias("year"))
        .group_by("year")
        .agg(
            [
                _hot_predicate(definition).sum().alias("hot_days"),
                (pl.col("temp_min").is_not_null() & (pl.col("temp_min") < 0)).sum().alias("cold_days"),
            ]
        )
        .sort("year")
    )
