from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Any, assert_never

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


_MONTH_ABBR: tuple[str, ...] = (
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
)


@dataclass(frozen=True)
class DayWindow:
    """A calendar window inside a year, year-agnostic. Stored as MMDD integers (601 = 1 June).

    MMDD comparison is leap-year proof, and a window whose end precedes its start
    (e.g. 1 Dec → 28 Feb) is a legal wrap-around rather than an invalid state.
    """

    start_md: int
    end_md: int

    @classmethod
    def from_dates(cls, start: date, end: date) -> DayWindow:
        return cls(start.month * 100 + start.day, end.month * 100 + end.day)

    @property
    def wraps(self) -> bool:
        return self.end_md < self.start_md

    @property
    def label(self) -> str:
        return f"{_md_label(self.start_md)} – {_md_label(self.end_md)}"


DEFAULT_WINDOW: DayWindow = DayWindow(601, 831)


def _md_label(md: int) -> str:
    return f"{md % 100} {_MONTH_ABBR[md // 100 - 1]}"


def _in_window(window: DayWindow) -> pl.Expr:
    # cast: dt.month()/dt.day() are Int8, and month * 100 overflows it
    md = pl.col("DATE").dt.month().cast(pl.Int32) * 100 + pl.col("DATE").dt.day().cast(pl.Int32)
    if window.wraps:
        return (md >= window.start_md) | (md <= window.end_md)
    return (md >= window.start_md) & (md <= window.end_md)


@dataclass(frozen=True)
class YearSpan:
    """An inclusive range of calendar years, ordered by construction — build it with `of`."""

    start: int
    end: int

    @classmethod
    def of(cls, start: int, end: int) -> YearSpan:
        """An inverted pair is a user mid-edit, not an error: order it instead of rejecting it."""
        return cls(start, end) if start <= end else cls(end, start)

    @property
    def label(self) -> str:
        return str(self.start) if self.start == self.end else f"{self.start}–{self.end}"


def _in_years(span: YearSpan) -> pl.Expr:
    year = pl.col("DATE").dt.year()
    return (year >= span.start) & (year <= span.end)


def year_filter(df: pl.DataFrame, span: YearSpan) -> pl.DataFrame:
    """Rows whose DATE falls within the inclusive year span."""
    return df.filter(_in_years(span))


def window_filter(df: pl.DataFrame, span: YearSpan, window: DayWindow) -> pl.DataFrame:
    """Rows whose DATE falls within the year span and inside the calendar window.

    ponytail: a wrapping window assigns each day to its own calendar year, so "Dec–Feb" pairs
    the December of one winter with the Jan/Feb of the next at the two range boundaries. Over a
    multi-decade density that is noise. For per-winter attribution, shift years for md < start_md.
    """
    return df.filter(_in_years(span) & _in_window(window))


@dataclass(frozen=True)
class Distribution:
    n: int
    mean: float
    median: float
    p10: float
    p90: float


def _stat(value: Any) -> float:
    """Narrow a Polars aggregation to float: on a non-empty numeric series it is never null."""
    return float(value)


def describe(series: pl.Series) -> Distribution | None:
    """Summary statistics of a numeric series, or None when it holds no non-null value."""
    values = series.drop_nulls()
    if values.is_empty():
        return None
    return Distribution(
        n=values.len(),
        mean=_stat(values.mean()),
        median=_stat(values.median()),
        p10=_stat(values.quantile(0.10)),
        p90=_stat(values.quantile(0.90)),
    )


_KDE_GRID_POINTS: int = 256
_KDE_TAIL: float = 3.0
# Météo-France reports temperatures at 0.1 °C, so collapsing onto that grid is lossless here
# and caps the kernel sum at a few hundred distinct values instead of tens of thousands of days.
_KDE_RESOLUTION: int = 1


@dataclass(frozen=True)
class Density:
    x: list[float]
    y: list[float]
    bandwidth: float


def silverman_bandwidth(series: pl.Series) -> float:
    """Silverman's rule of thumb, using the smaller of stdev and IQR/1.349 as the scale.

    The robust scale keeps the bandwidth from being inflated by the long warm tail that is
    exactly what a two-period comparison is meant to show.
    """
    std = _stat(series.std() or 0.0)
    q75, q25 = series.quantile(0.75), series.quantile(0.25)
    iqr = float(q75 - q25) if q75 is not None and q25 is not None else 0.0
    scale = min(std, iqr / 1.349) if iqr > 0 else std
    return 0.9 * scale * series.len() ** -0.2


def gaussian_kde(series: pl.Series, bandwidth: float | None = None) -> Density | None:
    """Smooth probability density of a sample, evaluated on a fixed grid.

    Returns None when the sample is too small or degenerate (a single repeated value) to carry
    a density — an expected outcome for a short record, not an error.
    """
    values = series.drop_nulls()
    if values.len() < 2:
        return None
    bw = silverman_bandwidth(values) if bandwidth is None else bandwidth
    if bw <= 0:
        return None

    grouped = values.round(_KDE_RESOLUTION).rename("value").value_counts()
    pairs = list(zip(grouped["value"].to_list(), grouped["count"].to_list(), strict=True))

    lo = _stat(values.min()) - _KDE_TAIL * bw
    hi = _stat(values.max()) + _KDE_TAIL * bw
    step = (hi - lo) / (_KDE_GRID_POINTS - 1)
    norm = 1.0 / (values.len() * bw * math.sqrt(2.0 * math.pi))

    x = [lo + i * step for i in range(_KDE_GRID_POINTS)]
    y = [norm * sum(c * math.exp(-0.5 * ((g - v) / bw) ** 2) for v, c in pairs) for g in x]
    return Density(x, y, bw)
