from __future__ import annotations

from datetime import date
from typing import Any
from urllib.parse import parse_qs

import plotly.graph_objects as go
import polars as pl
from dash import Dash, Input, Output, State, dcc, html

from src.charts import (
    density_comparison_figure,
    empty_figure,
    hot_cold_yearly_figure,
    monthly_avg_temp_by_decade_figure,
    monthly_avg_temp_figure,
    precipitation_figure,
    temperature_figure,
    wind_figure,
    yearly_series,
)
from src.data_loader import (
    Granularity,
    Station,
    aggregate,
    granularity_from,
    load_department_cached,
    stations_from,
)
from src.departments import DEPT_NAMES
from src.pages.components import station_search
from src.transforms import (
    DEFAULT_HOT_DAY,
    DEFAULT_WINDOW,
    HOT_DAY_OPTIONS,
    DatedValue,
    DayWindow,
    Distribution,
    LatestVsNormal,
    LinearTrend,
    StationRecords,
    Streak,
    YearSpan,
    decades_in,
    describe,
    has_wind,
    hot_day_from,
    is_complete,
    latest_vs_normal,
    linear_trend,
    station_records,
    window_filter,
    year_filter,
    yearly_hot_cold,
)

_DEFAULT_YEAR_WINDOW: int = 20
_COMPARISON_SPAN: int = 30
_NO_DATA = "No data available"
_COMPARISON_TAB = "comparison"
_FALLBACK_YEARS: YearSpan = YearSpan(1950, 2026)
_DECADES_SHOWN: int = 3
# ponytail: DatePickerRange needs a year; 2000 is a leap year so 29 Feb stays selectable.
# Only month/day are read back -- see DayWindow.from_dates.
_WINDOW_REF_YEAR: int = 2000
_MAX_SLIDER_MARKS: int = 8
_SLIDER_MARK_STEPS: tuple[int, ...] = (10, 20, 25, 50, 100)
_CELL: dict[str, str] = {"paddingRight": "1.5rem"}


def _chart_card(graph_id: str) -> html.Div:
    """Each chart carries its own loading state — the page-level one leaves stale charts on screen."""
    return html.Div(
        id=f"{graph_id}-card",
        className="card card--flush",
        children=[
            dcc.Loading(
                type="circle",
                delay_show=250,
                children=dcc.Graph(id=graph_id, config={"displayModeBar": False}),
            ),
        ],
    )


def _stat_tile(label: str, value: str, detail: str = "", tone: str = "") -> html.Div:
    return html.Div(
        className=f"stat-tile {tone}".strip(),
        children=[
            html.Span(label, className="stat-label"),
            html.Span(value, className="stat-value"),
            html.Span(detail, className="stat-detail"),
        ],
    )


def _signed(value: float | None, unit: str = "°C") -> str:
    return "—" if value is None else f"{value:+.1f} {unit}"


def _tone(value: float | None) -> str:
    if value is None:
        return ""
    return "stat-tile--warm" if value >= 0 else "stat-tile--cool"


def _ordinal(n: int) -> str:
    suffix = "th" if 11 <= n % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _degrees(value: float | None) -> str:
    return "—" if value is None else f"{value:.1f} °C"


def _usual_tile(delta: float | None, normal: float | None, what: str, span_label: str) -> html.Div:
    """The anomaly, said in words: a reader should not have to know what "anomaly" means."""
    if delta is None:
        label, value = "Compared with usual", "—"
    else:
        label = "Warmer than usual" if delta >= 0 else "Colder than usual"
        value = f"{abs(delta):.1f} °C"
    detail = "no long-term average on record"
    if normal is not None:
        detail = f"usual {what} for this date {normal:.1f} °C ({span_label})"
    return _stat_tile(label, value, detail, _tone(delta))


def _latest_tiles(latest: LatestVsNormal) -> list[html.Div]:
    """The station's most recent day, read against the normal for that calendar day.

    This is the only place in the app that answers "what is it doing now, and is that unusual" —
    the question the daily-refreshed LATEST period exists to serve.
    """
    obs, normal = latest.observation, latest.normal
    day_label = obs.when.strftime("%d %b %Y")
    tiles = [
        _stat_tile("Daytime high", _degrees(obs.temp_max), day_label),
        _usual_tile(latest.anomaly_max, normal.temp_max, "high", normal.span.label),
        _stat_tile("Overnight low", _degrees(obs.temp_min), day_label),
        _usual_tile(latest.anomaly_min, normal.temp_min, "low", normal.span.label),
    ]
    if latest.rank is not None:
        tiles.append(
            _stat_tile(
                f"Rank for {obs.when.strftime('%d %b')}",
                f"{_ordinal(latest.rank.rank)} warmest",
                f"of the {latest.rank.of} years this date was recorded",
            )
        )
    return tiles


def _dated(value: DatedValue | None, unit: str) -> str:
    return "—" if value is None else f"{value.value:.1f} {unit} · {value.when.strftime('%d %b %Y')}"


def _streak(streak: Streak | None) -> str:
    if streak is None:
        return "—"
    return f"{streak.days} days · {streak.start.strftime('%d %b')} – {streak.end.strftime('%d %b %Y')}"


def _records_table(records: StationRecords, definition_label: str) -> html.Table:
    rows = [
        ("Hottest day", _dated(records.hottest_day, "°C")),
        ("Coldest night", _dated(records.coldest_night, "°C")),
        ("Wettest day", _dated(records.wettest_day, "mm")),
        ("Strongest gust", _dated(records.strongest_gust, "m/s")),
        (f"Longest run of {definition_label}", _streak(records.longest_hot_streak)),
    ]
    return html.Table(
        html.Tbody([html.Tr([html.Td(name, style=_CELL), html.Td(value)]) for name, value in rows]),
        className="records-table",
    )


def _record_years(df: pl.DataFrame | None) -> YearSpan:
    """First and last year covered by a DataFrame, or a sane span when there is no data."""
    first = df["DATE"].min() if df is not None else None
    last = df["DATE"].max() if df is not None else None
    if isinstance(first, date) and isinstance(last, date):
        return YearSpan.of(first.year, last.year)
    return _FALLBACK_YEARS


def _mark_years(record: YearSpan, step: int) -> range:
    """Round years inside the record at `step` intervals, starting on the first multiple of `step`."""
    return range(record.start + (-record.start) % step, record.end + 1, step)


def _year_marks(record: YearSpan) -> dict[int, str]:
    """Decade marks thinned until they fit: 22 four-digit labels in one row is an unreadable smear."""
    step = next(
        (s for s in _SLIDER_MARK_STEPS if len(_mark_years(record, s)) <= _MAX_SLIDER_MARKS),
        _SLIDER_MARK_STEPS[-1],
    )
    return {y: str(y) for y in _mark_years(record, step)} or {
        record.start: str(record.start),
        record.end: str(record.end),
    }


def _year_input(component_id: str, value: int, record: YearSpan) -> dcc.Input:
    return dcc.Input(
        id=component_id,
        type="number",
        min=record.start,
        max=record.end,
        step=1,
        value=value,
        debounce=True,
        className="year-input",
    )


def _period_group(label: str, prefix: str, span: YearSpan, record: YearSpan) -> html.Div:
    """Two year boxes rather than a range slider: periods are typed exactly, not scrubbed."""
    return html.Div(
        className="control-group control-group--period",
        children=[
            html.Label(label, className="control-label"),
            html.Div(
                className="period-range",
                children=[
                    _year_input(f"{prefix}-start", span.start, record),
                    html.Span("–", className="period-dash"),
                    _year_input(f"{prefix}-end", span.end, record),
                ],
            ),
        ],
    )


def layout(search: str = "") -> html.Div:
    """Build the station detail page.

    Parses dept and station from the URL search string, loads department data,
    and pre-populates controls so the initial chart render fires immediately.
    """
    params = parse_qs(search.lstrip("?"))
    dept: str | None = params.get("dept", [None])[0]
    raw_station = params.get("station", [None])[0]
    initial_station: int | None = int(raw_station) if raw_station else None

    df = load_department_cached(dept) if dept else None
    stations: list[Station] = stations_from(df) if df is not None else []

    record = _record_years(df)
    year_min, year_max = record.start, record.end
    marks = _year_marks(record)

    valid_ids = {s.station_id for s in stations}
    if initial_station not in valid_ids:
        initial_station = stations[0].station_id if stations else None

    span_a = YearSpan.of(year_min, min(year_max, year_min + _COMPARISON_SPAN - 1))
    span_b = YearSpan.of(max(year_min, year_max - _COMPARISON_SPAN + 1), year_max)
    station_options = [{"label": s.name, "value": s.station_id} for s in stations]
    dept_label = f"{DEPT_NAMES.get(dept, dept)} ({dept})" if dept else ""

    return html.Div(
        className="page-container",
        children=[
            html.Div(
                className="page-nav",
                children=[
                    dcc.Link("← Back to map", href="/", className="back-link"),
                    html.Span("·", className="page-nav-sep"),
                    html.Span(dept_label, className="page-nav-dept"),
                    html.Div(station_search("Jump to any station in France…"), className="page-nav-search"),
                ],
            ),
            dcc.Store(id="dept-store", data=dept),
            dcc.Loading(type="circle", delay_show=250, children=html.Div(id="station-header")),
            html.Div(
                className="card",
                children=[
                    html.Div(
                        className="controls-row",
                        children=[
                            html.Div(
                                className="control-group control-group--station",
                                children=[
                                    html.Label("Station", className="control-label"),
                                    dcc.Dropdown(
                                        id="station-dropdown",
                                        options=station_options,
                                        value=initial_station,
                                        clearable=False,
                                    ),
                                ],
                            ),
                            html.Div(
                                id="year-slider-group",
                                className="control-group control-group--slider",
                                children=[
                                    html.Label("Year range", className="control-label"),
                                    dcc.RangeSlider(
                                        id="year-slider",
                                        min=year_min,
                                        max=year_max,
                                        step=1,
                                        value=[max(year_min, year_max - _DEFAULT_YEAR_WINDOW), year_max],
                                        marks=marks,
                                        tooltip={"placement": "top", "always_visible": True},
                                    ),
                                ],
                            ),
                            html.Div(
                                className="control-group control-group--definition",
                                children=[
                                    html.Label("Hot day", className="control-label"),
                                    dcc.Dropdown(
                                        id="hot-day-definition",
                                        options=[{"label": d.label, "value": d.label} for d in HOT_DAY_OPTIONS],
                                        value=DEFAULT_HOT_DAY.label,
                                        clearable=False,
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
            dcc.Tabs(
                id="station-tabs",
                value=_COMPARISON_TAB,
                className="station-tabs",
                children=[
                    dcc.Tab(
                        label="Then vs now",
                        value=_COMPARISON_TAB,
                        className="station-tab station-tab--feature",
                        selected_className="station-tab station-tab--feature station-tab--selected",
                        style={},
                        selected_style={},
                        children=[
                            html.Div(
                                className="card",
                                children=[
                                    html.Div(
                                        className="controls-row",
                                        children=[
                                            html.Div(
                                                className="control-group",
                                                children=[
                                                    html.Label("Season window (shared)", className="control-label"),
                                                    dcc.DatePickerRange(
                                                        id="cmp-window",
                                                        display_format="DD MMM",
                                                        min_date_allowed=date(_WINDOW_REF_YEAR, 1, 1).isoformat(),
                                                        max_date_allowed=date(_WINDOW_REF_YEAR, 12, 31).isoformat(),
                                                        start_date=date(_WINDOW_REF_YEAR, 6, 1).isoformat(),
                                                        end_date=date(_WINDOW_REF_YEAR, 8, 31).isoformat(),
                                                    ),
                                                ],
                                            ),
                                            _period_group("Period A", "cmp-a", span_a, record),
                                            _period_group("Period B", "cmp-b", span_b, record),
                                            html.Span(
                                                f"Record: {year_min}–{year_max}",
                                                className="control-hint",
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                            html.Div(id="cmp-headline"),
                            html.Div(
                                className="chart-pair",
                                children=[
                                    _chart_card("chart-density-tmax"),
                                    _chart_card("chart-density-tmin"),
                                ],
                            ),
                            html.Div(id="cmp-stats"),
                        ],
                    ),
                    dcc.Tab(
                        label="Observations",
                        value="observations",
                        className="station-tab",
                        selected_className="station-tab station-tab--selected",
                        style={},
                        selected_style={},
                        children=[
                            html.Div(
                                className="card",
                                children=[
                                    html.Div(
                                        className="controls-row",
                                        children=[
                                            html.Div(
                                                className="control-group control-group--granularity",
                                                children=[
                                                    html.Label("Granularity", className="control-label"),
                                                    dcc.RadioItems(
                                                        id="granularity-radio",
                                                        className="granularity-pills",
                                                        options=[
                                                            {"label": "Day", "value": Granularity.DAY.label},
                                                            {"label": "Week", "value": Granularity.WEEK.label},
                                                            {"label": "Month", "value": Granularity.MONTH.label},
                                                        ],
                                                        value=Granularity.DAY.label,
                                                        inline=True,
                                                    ),
                                                ],
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                            _chart_card("chart-temperature"),
                            _chart_card("chart-precipitation"),
                            _chart_card("chart-wind"),
                        ],
                    ),
                    dcc.Tab(
                        label="Yearly extremes",
                        value="yearly-extremes",
                        className="station-tab",
                        selected_className="station-tab station-tab--selected",
                        style={},
                        selected_style={},
                        children=[
                            html.Div(
                                className="card",
                                children=[
                                    html.Div(
                                        className="controls-row",
                                        children=[
                                            html.Div(
                                                className="control-group",
                                                children=[
                                                    dcc.Checklist(
                                                        id="yearly-trend-toggle",
                                                        options=[{"label": "  Show tendency lines", "value": "show"}],
                                                        value=[],
                                                        inline=True,
                                                    ),
                                                ],
                                            ),
                                            html.Span(
                                                "Years with under 90 % of days observed are excluded — "
                                                "an offline station is not a cool year.",
                                                className="control-hint",
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                            _chart_card("chart-hot-cold-yearly"),
                            html.Div(id="trend-stats"),
                        ],
                    ),
                    dcc.Tab(
                        label="Monthly averages",
                        value="monthly-avg",
                        className="station-tab",
                        selected_className="station-tab station-tab--selected",
                        style={},
                        selected_style={},
                        children=[
                            _chart_card("chart-monthly-avg-temp"),
                            html.Div(
                                className="card",
                                children=[
                                    html.Div(
                                        className="controls-row",
                                        children=[
                                            html.Div(
                                                className="control-group control-group--decades",
                                                children=[
                                                    html.Label("Decades compared", className="control-label"),
                                                    dcc.Dropdown(
                                                        id="decade-select",
                                                        options=[],
                                                        value=[],
                                                        multi=True,
                                                        placeholder="All decades in range",
                                                    ),
                                                ],
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                            _chart_card("chart-monthly-avg-temp-decade"),
                        ],
                    ),
                ],
            ),
        ],
    )


def _station_df(station_id: int | None, dept: str | None) -> pl.DataFrame | None:
    """Return the full record for one station, or None if inputs are invalid / no data."""
    if dept is None or station_id is None:
        return None
    df_full = load_department_cached(dept)
    if df_full is None:
        return None
    df = df_full.filter(pl.col("station_id") == station_id)
    return df if not df.is_empty() else None


def _filtered_station_df(station_id: int | None, year_range: list[int], dept: str | None) -> pl.DataFrame | None:
    """Return a station+year-filtered DataFrame, or None if inputs are invalid / no data."""
    df = _station_df(station_id, dept)
    if df is None:
        return None
    df = year_filter(df, YearSpan.of(*year_range))
    return df if not df.is_empty() else None


def update_station_header(station_id: int | None, dept: str | None, definition_label: str) -> list[Any]:
    """The card above the tabs: what the station is doing now, and its all-time extremes.

    Independent of the year range — a record is a record whatever window the charts are showing.
    """
    df = _station_df(station_id, dept)
    if df is None:
        return []
    definition = hot_day_from(definition_label)
    latest = latest_vs_normal(df)
    record = _record_years(df)
    summary = f"Record {record.label} · {df.height:,} days observed".replace(",", " ")
    return [
        html.Div(
            className="card station-header",
            children=[
                html.Div(
                    className="station-header-title",
                    children=[
                        html.Strong(df["station_name"][0]),
                        html.Span(summary, className="station-header-summary"),
                    ],
                ),
                html.Div(className="stat-row", children=_latest_tiles(latest) if latest is not None else []),
                _records_table(station_records(df, definition), definition.label),
            ],
        )
    ]


def update_charts(
    station_id: int | None,
    year_range: list[int],
    dept: str | None,
    granularity_value: str,
    definition_label: str = DEFAULT_HOT_DAY.label,
) -> tuple[go.Figure, go.Figure, go.Figure, dict[str, str]]:
    """Render the three observation charts, hiding the wind card for stations that measure no wind."""
    df = _filtered_station_df(station_id, year_range, dept)
    if df is None:
        placeholder = empty_figure(_NO_DATA)
        return placeholder, placeholder, placeholder, {"display": "none"}

    granularity: Granularity = granularity_from(granularity_value)
    windy = has_wind(df)
    df = aggregate(df, granularity)
    station_name = df["station_name"][0]
    return (
        temperature_figure(df, station_name, granularity, hot_day_from(definition_label)),
        precipitation_figure(df, station_name, granularity),
        wind_figure(df, station_name, granularity) if windy else empty_figure(_NO_DATA),
        {} if windy else {"display": "none"},
    )


def update_yearly_chart(
    station_id: int | None,
    year_range: list[int],
    dept: str | None,
    trend_values: list[str] | None,
    definition_label: str = DEFAULT_HOT_DAY.label,
) -> tuple[go.Figure, list[Any]]:
    """Render the yearly hot/cold days chart, with optional trend lines.

    The current (incomplete) year is shown as a provisional dotted point when the year-range
    upper bound includes it. Trend statistics are computed on complete years only.
    Returns (figure, trend_stats_children) — the stats card is empty when trends are off.
    """
    df = _filtered_station_df(station_id, year_range, dept)
    if df is None:
        return empty_figure(_NO_DATA), []

    current_year = date.today().year
    definition = hot_day_from(definition_label)
    show_trend = "show" in (trend_values or [])
    provisional = current_year if YearSpan.of(*year_range).end >= current_year else None

    fig = hot_cold_yearly_figure(
        df,
        df["station_name"][0],
        definition=definition,
        show_trend=show_trend,
        current_year=provisional,
    )

    if not show_trend:
        return fig, []

    # Same population as the chart's solid lines: complete years only, current year excluded.
    agg = yearly_hot_cold(df, definition).filter(is_complete() & (pl.col("year") < current_year))
    years_f = [float(y) for y in agg["year"].to_list()]
    if len(years_f) < 2:
        return fig, []

    trends = ((s.label, linear_trend(years_f, agg[s.column].to_list())) for s in yearly_series(definition))
    rows = [
        html.Tr(
            [
                html.Td(label, style=_CELL),
                html.Td(_slope_label(trend), style=_CELL),
                html.Td(f"R² = {trend.r_squared:.2f}"),
            ]
        )
        for label, trend in trends
        if trend is not None
    ]

    stats_card = html.Div(
        className="card",
        children=[
            html.Strong(f"Tendency over {len(years_f)} fully observed years"),
            html.Table(html.Tbody(rows), style={"marginTop": "0.35rem"}),
        ],
    )
    return fig, [stats_card]


def _slope_label(trend: LinearTrend) -> str:
    sign = "+" if trend.slope >= 0 else "−"
    return f"{sign}{abs(trend.slope):.2f} days/yr ({sign}{abs(trend.slope) * 10:.1f} per decade)"


def update_decade_options(
    station_id: int | None,
    year_range: list[int],
    dept: str | None,
) -> tuple[list[dict[str, Any]], list[int]]:
    """Offer the decades the current range covers, pre-selecting the oldest, middle and newest.

    Drawing every decade at once is what made this chart a legend with a plot attached.
    """
    df = _filtered_station_df(station_id, year_range, dept)
    decades = decades_in(df) if df is not None else []
    options = [{"label": f"{d}s", "value": d} for d in decades]
    if len(decades) <= _DECADES_SHOWN:
        return options, decades
    return options, [decades[0], decades[len(decades) // 2], decades[-1]]


def update_monthly_charts(
    station_id: int | None,
    year_range: list[int],
    dept: str | None,
    decades: list[int] | None = None,
) -> tuple[go.Figure, go.Figure]:
    """Render the two monthly average temperature charts."""
    df = _filtered_station_df(station_id, year_range, dept)
    if df is None:
        placeholder = empty_figure(_NO_DATA)
        return placeholder, placeholder
    station_name = df["station_name"][0]
    return (
        monthly_avg_temp_figure(df, station_name),
        monthly_avg_temp_by_decade_figure(df, station_name, decades or None),
    )


def _window_from(start_date: str | None, end_date: str | None) -> DayWindow:
    if not start_date or not end_date:
        return DEFAULT_WINDOW
    return DayWindow.from_dates(date.fromisoformat(start_date[:10]), date.fromisoformat(end_date[:10]))


def _typed_span(start: int | None, end: int | None, fallback: YearSpan) -> YearSpan:
    """A year box cleared to empty falls back to the record bound rather than blanking the chart."""
    return YearSpan.of(
        start if start is not None else fallback.start,
        end if end is not None else fallback.end,
    )


def _describe_cell(dist: Distribution | None) -> str:
    if dist is None:
        return "no data"
    return (
        f"{dist.n} days \u00b7 average {dist.mean:.1f} \u00b7 median {dist.median:.1f} "
        f"\u00b7 warmest tenth above {dist.p90:.1f} \u00b0C"
    )


def _headline(
    rows: list[tuple[str, Distribution | None, Distribution | None]],
    label_a: str,
    label_b: str,
    overlapping: bool,
) -> list[Any]:
    """\u0394 mean and \u0394 p90 as tiles: the shift is the answer this tab exists for, not a table cell."""
    tiles: list[html.Div] = []
    for name, a, b in rows:
        if a is None or b is None:
            continue
        delta_mean, delta_p90 = b.mean - a.mean, b.p90 - a.p90
        shift = f"{label_a} \u2192 {label_b}"
        tiles.append(_stat_tile(f"Average {name.lower()}", _signed(delta_mean), shift, _tone(delta_mean)))
        tiles.append(_stat_tile(f"Warmest tenth of {name.lower()}", _signed(delta_p90), shift, _tone(delta_p90)))
    if not tiles:
        return []
    children: list[Any] = [html.Div(className="stat-row", children=tiles)]
    if overlapping:
        children.append(
            html.Span(
                "The two periods overlap \u2014 part of the same years is being compared with itself.",
                className="control-hint control-hint--warning",
            )
        )
    return [html.Div(className="card", children=children)]


def _delta_cell(a: Distribution | None, b: Distribution | None) -> str:
    if a is None or b is None:
        return "\u2014"
    return f"average {b.mean - a.mean:+.1f} \u00b7 warmest tenth {b.p90 - a.p90:+.1f} \u00b0C"


def _stats_card(
    rows: list[tuple[str, Distribution | None, Distribution | None]],
    label_a: str,
    label_b: str,
) -> list[Any]:
    """Numeric companion to the density overlay: where each period sits and how far apart they are."""
    if all(a is None and b is None for _, a, b in rows):
        return []
    header = html.Tr(
        [
            html.Th(""),
            html.Th(label_a, style=_CELL),
            html.Th(label_b, style=_CELL),
            html.Th("Shift"),
        ],
        style={"textAlign": "left"},
    )
    body = [
        html.Tr(
            [
                html.Td(html.Strong(name), style=_CELL),
                html.Td(_describe_cell(a), style=_CELL),
                html.Td(_describe_cell(b), style=_CELL),
                html.Td(_delta_cell(a, b)),
            ]
        )
        for name, a, b in rows
    ]
    return [html.Div(className="card", children=[html.Table([html.Thead(header), html.Tbody(body)])])]


def update_comparison_charts(
    station_id: int | None,
    dept: str | None,
    a_start: int | None,
    a_end: int | None,
    b_start: int | None,
    b_end: int | None,
    start_date: str | None,
    end_date: str | None,
) -> tuple[go.Figure, go.Figure, list[Any], list[Any]]:
    """Overlay the Tmax and Tmin densities of two year ranges over a shared calendar window."""
    df = _station_df(station_id, dept)
    if df is None:
        placeholder = empty_figure(_NO_DATA)
        return placeholder, placeholder, [], []

    record = _record_years(df)
    span_a = _typed_span(a_start, a_end, record)
    span_b = _typed_span(b_start, b_end, record)
    window = _window_from(start_date, end_date)
    df_a = window_filter(df, span_a, window)
    df_b = window_filter(df, span_b, window)
    label_a, label_b = span_a.label, span_b.label
    station_name = df["station_name"][0]

    def figure(col: str, what: str) -> go.Figure:
        return density_comparison_figure(
            df_a[col],
            df_b[col],
            label_a,
            label_b,
            f"{what} distribution, {window.label} \u2014 {station_name}",
        )

    rows = [
        (what, describe(df_a[col]), describe(df_b[col]))
        for what, col in (("Daytime highs", "temp_max"), ("Overnight lows", "temp_min"))
    ]
    overlapping = span_a.end >= span_b.start and span_b.end >= span_a.start
    return (
        figure("temp_max", "Daytime highs"),
        figure("temp_min", "Overnight lows"),
        _stats_card(rows, label_a, label_b),
        _headline(rows, label_a, label_b, overlapping),
    )


def year_slider_style(tab: str | None) -> dict[str, str]:
    """Hide the page-level year range on the tab that brings its own two periods."""
    return {"display": "none"} if tab == _COMPARISON_TAB else {}


def register_callbacks(app: Dash) -> None:
    app.callback(
        Output("year-slider-group", "style"),
        Input("station-tabs", "value"),
    )(year_slider_style)

    app.callback(
        Output("station-header", "children"),
        Input("station-dropdown", "value"),
        State("dept-store", "data"),
        Input("hot-day-definition", "value"),
    )(update_station_header)

    app.callback(
        Output("chart-temperature", "figure"),
        Output("chart-precipitation", "figure"),
        Output("chart-wind", "figure"),
        Output("chart-wind-card", "style"),
        Input("station-dropdown", "value"),
        Input("year-slider", "value"),
        State("dept-store", "data"),
        Input("granularity-radio", "value"),
        Input("hot-day-definition", "value"),
    )(update_charts)

    app.callback(
        Output("chart-hot-cold-yearly", "figure"),
        Output("trend-stats", "children"),
        Input("station-dropdown", "value"),
        Input("year-slider", "value"),
        State("dept-store", "data"),
        Input("yearly-trend-toggle", "value"),
        Input("hot-day-definition", "value"),
    )(update_yearly_chart)

    app.callback(
        Output("decade-select", "options"),
        Output("decade-select", "value"),
        Input("station-dropdown", "value"),
        Input("year-slider", "value"),
        State("dept-store", "data"),
    )(update_decade_options)

    app.callback(
        Output("chart-monthly-avg-temp", "figure"),
        Output("chart-monthly-avg-temp-decade", "figure"),
        Input("station-dropdown", "value"),
        Input("year-slider", "value"),
        State("dept-store", "data"),
        Input("decade-select", "value"),
    )(update_monthly_charts)

    app.callback(
        Output("chart-density-tmax", "figure"),
        Output("chart-density-tmin", "figure"),
        Output("cmp-stats", "children"),
        Output("cmp-headline", "children"),
        Input("station-dropdown", "value"),
        State("dept-store", "data"),
        Input("cmp-a-start", "value"),
        Input("cmp-a-end", "value"),
        Input("cmp-b-start", "value"),
        Input("cmp-b-end", "value"),
        Input("cmp-window", "start_date"),
        Input("cmp-window", "end_date"),
    )(update_comparison_charts)
