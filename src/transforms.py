from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta
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

# Tmax ≥ 30 °C ("jour de forte chaleur") is the default because it is where the signal lives:
# at REVEL it runs 37–77 days a year, while Tmin≥20 & Tmax≥35 yields 0–9 — a count so low the
# chart shows mostly sampling noise. The stricter pairs remain available in the dropdown.
DEFAULT_HOT_DAY: HotDayDefinition = TmaxOnly(30.0)
HEATWAVE_HOT_DAY: HotDayDefinition = TmaxAndTmin(tmax_min=HOT_DAY_TMAX, tmin_min=HOT_DAY_TMIN)
HOT_DAY_OPTIONS: list[HotDayDefinition] = [
    DEFAULT_HOT_DAY,
    TmaxOnly(32.0),
    TmaxOnly(35.0),
    TmaxOnly(36.0),
    TmaxOnly(38.0),
    TmaxOnly(40.0),
    HEATWAVE_HOT_DAY,
]

# A tropical night is the warm-tail counterpart of a frost day, and it is the cleanest warming
# signal in the record — it gets its own series rather than being folded into the hot-day rule.
TROPICAL_NIGHT_TMIN: float = HOT_DAY_TMIN
FROST_TMIN: float = 0.0

# Below this share of days carrying a temperature observation, a year's counts are not a
# measurement of the weather but of the station's downtime.
MIN_YEAR_COVERAGE: float = 0.9


def hot_day_from(label: str) -> HotDayDefinition:
    """Parse a dropdown label back into its definition, falling back to the default when unknown."""
    return next((d for d in HOT_DAY_OPTIONS if d.label == label), DEFAULT_HOT_DAY)


def hot_day_predicate(d: HotDayDefinition) -> pl.Expr:
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


def linear_trend(x: Sequence[float | int], y: Sequence[float | int | None]) -> LinearTrend | None:
    """Return OLS linear regression for paired x/y data, or None when there is no trend to fit.

    Filters out pairs where y is None. Centers x before computing to avoid catastrophic
    cancellation with large x-values (e.g. years). Fewer than 2 valid pairs, or x-values that
    are all the same, describe no line — None says so rather than reporting a flat one.
    """
    pairs = [(float(xi), float(yi)) for xi, yi in zip(x, y, strict=True) if yi is not None]
    n = len(pairs)
    if n < 2:
        return None
    xs = [p[0] for p in pairs]
    mean_x = sum(xs) / n
    xc = [xi - mean_x for xi in xs]
    yv = [p[1] for p in pairs]
    sum_xc2 = sum(v * v for v in xc)
    if sum_xc2 == 0:
        return None
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
    """Aggregate daily station data into yearly extreme-day counts, with the coverage behind them.

    Returns columns: year, hot_days, cold_days, tropical_nights, observed_days, coverage —
    sorted ascending by year. A missing day counts as "not hot", so a year the station spent
    offline reads as a genuine minimum; `coverage` (share of the year's days carrying any
    temperature reading) is what lets callers tell the two apart — see `is_complete`.
    """
    observed = pl.col("temp_min").is_not_null() | pl.col("temp_max").is_not_null()
    return (
        df.with_columns(pl.col("DATE").dt.year().alias("year"))
        .group_by("year")
        .agg(
            [
                hot_day_predicate(definition).sum().alias("hot_days"),
                (pl.col("temp_min").is_not_null() & (pl.col("temp_min") < FROST_TMIN)).sum().alias("cold_days"),
                (pl.col("temp_min").is_not_null() & (pl.col("temp_min") >= TROPICAL_NIGHT_TMIN))
                .sum()
                .alias("tropical_nights"),
                observed.sum().alias("observed_days"),
            ]
        )
        .with_columns((pl.col("observed_days") / pl.date(pl.col("year"), 12, 31).dt.ordinal_day()).alias("coverage"))
        .sort("year")
    )


def is_complete(min_coverage: float = MIN_YEAR_COVERAGE) -> pl.Expr:
    """Predicate over a `yearly_hot_cold` frame: the year is measured well enough to be counted."""
    return pl.col("coverage") >= min_coverage


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


@dataclass(frozen=True)
class DatedValue:
    when: date
    value: float


@dataclass(frozen=True)
class Streak:
    days: int
    start: date
    end: date


@dataclass(frozen=True)
class StationRecords:
    """The all-time superlatives of a station record. Every field is None when the
    underlying column is absent or entirely null — a short or partial record, not an error."""

    hottest_day: DatedValue | None
    coldest_night: DatedValue | None
    wettest_day: DatedValue | None
    strongest_gust: DatedValue | None
    longest_hot_streak: Streak | None


def _extreme(df: pl.DataFrame, col: str, *, largest: bool) -> DatedValue | None:
    """The record high (or low) of a column, reported at its most recent occurrence."""
    if col not in df.columns:
        return None
    rows = df.select(["DATE", col]).drop_nulls()
    if rows.is_empty():
        return None
    row = rows.sort([col, "DATE"], descending=[largest, True]).row(0, named=True)
    return DatedValue(when=row["DATE"], value=float(row[col]))


def longest_streak(df: pl.DataFrame, predicate: pl.Expr) -> Streak | None:
    """Longest run of calendar-consecutive days satisfying a predicate.

    A missing day breaks the run: an unobserved day is not evidence that the run continued.
    Ties resolve to the earliest run. Returns None when the predicate's columns are absent.
    """
    if "DATE" not in df.columns:
        return None
    days = df.with_columns(predicate.alias("hit")).select(["DATE", "hit"]).filter(pl.col("hit")).sort("DATE")
    if days.is_empty():
        return None
    islands = (
        days.with_columns((pl.col("DATE") - pl.duration(days=pl.int_range(pl.len()))).alias("island"))
        .group_by("island")
        .agg(pl.len().alias("days"), pl.col("DATE").min().alias("start"), pl.col("DATE").max().alias("end"))
        .sort(["days", "start"], descending=[True, False])
    )
    best = islands.row(0, named=True)
    return Streak(days=best["days"], start=best["start"], end=best["end"])


def station_records(df: pl.DataFrame, definition: HotDayDefinition = DEFAULT_HOT_DAY) -> StationRecords:
    """All-time extremes of one station's record, over whatever period `df` already covers."""
    temps_present = {"temp_min", "temp_max"} <= set(df.columns)
    return StationRecords(
        hottest_day=_extreme(df, "temp_max", largest=True),
        coldest_night=_extreme(df, "temp_min", largest=False),
        wettest_day=_extreme(df, "precipitation", largest=True),
        strongest_gust=_extreme(df, "wind_gust", largest=True),
        longest_hot_streak=longest_streak(df, hot_day_predicate(definition)) if temps_present else None,
    )


# WMO standard climate normal: the current decade-aligned 30-year period, 2001–2030 from 2031 on.
REFERENCE_PERIOD: YearSpan = YearSpan(1991, 2020)
NORMAL_HALF_WINDOW_DAYS: int = 7


@dataclass(frozen=True)
class DayNormal:
    """The expected temperatures for one calendar day, averaged over a reference period."""

    span: YearSpan
    temp_max: float | None
    temp_min: float | None
    n_days: int


@dataclass(frozen=True)
class DayRank:
    """Where a value sits among all years' observations of the same calendar day. rank 1 = warmest."""

    rank: int
    of: int


@dataclass(frozen=True)
class LatestObservation:
    when: date
    temp_max: float | None
    temp_min: float | None
    precipitation: float | None


@dataclass(frozen=True)
class LatestVsNormal:
    observation: LatestObservation
    normal: DayNormal
    rank: DayRank | None

    @property
    def anomaly_max(self) -> float | None:
        return _difference(self.observation.temp_max, self.normal.temp_max)

    @property
    def anomaly_min(self) -> float | None:
        return _difference(self.observation.temp_min, self.normal.temp_min)


def _difference(observed: float | None, reference: float | None) -> float | None:
    return None if observed is None or reference is None else observed - reference


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def latest_observation(df: pl.DataFrame) -> LatestObservation | None:
    """The most recent day carrying a temperature reading, or None for a record with none."""
    rows = df.filter(pl.col("temp_max").is_not_null() | pl.col("temp_min").is_not_null()).sort("DATE")
    if rows.is_empty():
        return None
    row = rows.row(rows.height - 1, named=True)
    return LatestObservation(
        when=row["DATE"],
        temp_max=_optional_float(row["temp_max"]),
        temp_min=_optional_float(row["temp_min"]),
        precipitation=_optional_float(row.get("precipitation")),
    )


def day_normal(
    df: pl.DataFrame,
    day: date,
    reference: YearSpan = REFERENCE_PERIOD,
    half_window: int = NORMAL_HALF_WINDOW_DAYS,
) -> DayNormal:
    """Mean Tmax/Tmin for a calendar day, over a ±`half_window` window across the reference years.

    A record too short or too recent to cover the reference period falls back to its own span
    rather than reporting nothing — the returned `span` says which was used.
    """
    window = DayWindow.from_dates(day - timedelta(days=half_window), day + timedelta(days=half_window))
    span = reference
    sample = window_filter(df, span, window)
    if sample.is_empty():
        span = _record_span(df) or reference
        sample = window_filter(df, span, window)
    return DayNormal(
        span=span,
        temp_max=_optional_float(sample["temp_max"].mean()) if not sample.is_empty() else None,
        temp_min=_optional_float(sample["temp_min"].mean()) if not sample.is_empty() else None,
        n_days=sample.height,
    )


def _record_span(df: pl.DataFrame) -> YearSpan | None:
    first, last = df["DATE"].min(), df["DATE"].max()
    if isinstance(first, date) and isinstance(last, date):
        return YearSpan.of(first.year, last.year)
    return None


def day_rank(df: pl.DataFrame, day: date, value: float | None, column: str = "temp_max") -> DayRank | None:
    """Rank of `value` among every year's reading for the same calendar day (1 = warmest)."""
    if value is None:
        return None
    same_day = df.filter(
        (pl.col("DATE").dt.month() == day.month) & (pl.col("DATE").dt.day() == day.day) & pl.col(column).is_not_null()
    )
    if same_day.is_empty():
        return None
    warmer = same_day.filter(pl.col(column) > value).height
    return DayRank(rank=warmer + 1, of=same_day.height)


def latest_vs_normal(df: pl.DataFrame, reference: YearSpan = REFERENCE_PERIOD) -> LatestVsNormal | None:
    """The station's most recent day, read against the climate normal for that calendar day."""
    observation = latest_observation(df)
    if observation is None:
        return None
    return LatestVsNormal(
        observation=observation,
        normal=day_normal(df, observation.when, reference),
        rank=day_rank(df, observation.when, observation.temp_max),
    )


def decades_in(df: pl.DataFrame) -> list[int]:
    """Every decade the record touches, ascending: 1950 stands for the 1950s."""
    if df.is_empty():
        return []
    decades = df.select(((pl.col("DATE").dt.year() // 10) * 10).alias("decade"))["decade"].unique().to_list()
    return sorted(decades)


def has_wind(df: pl.DataFrame) -> bool:
    """Whether this station measures wind at all — many do not, and an empty axis is not a chart."""
    return any(col in df.columns and df[col].is_not_null().any() for col in ("wind_mean", "wind_gust"))
