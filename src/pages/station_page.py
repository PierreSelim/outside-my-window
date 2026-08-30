from __future__ import annotations

import datetime
from datetime import date as _date
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
from src.transforms import (
    DEFAULT_HOT_DAY,
    DEFAULT_WINDOW,
    HOT_DAY_OPTIONS,
    DayWindow,
    Distribution,
    LinearTrend,
    YearSpan,
    describe,
    hot_day_from,
    linear_trend,
    window_filter,
    year_filter,
    yearly_hot_cold,
)

_DEFAULT_YEAR_WINDOW: int = 20
_COMPARISON_SPAN: int = 30
_NO_DATA = "No data available"
_COMPARISON_TAB = "comparison"
_FALLBACK_YEARS: YearSpan = YearSpan(1950, 2026)
# ponytail: DatePickerRange needs a year; 2000 is a leap year so 29 Feb stays selectable.
# Only month/day are read back -- see DayWindow.from_dates.
_WINDOW_REF_YEAR: int = 2000
_CELL: dict[str, str] = {"paddingRight": "1.5rem"}


def _chart_card(graph_id: str) -> html.Div:
    return html.Div(
        className="card card--flush",
        children=[
            dcc.Graph(id=graph_id, config={"displayModeBar": False}),
        ],
    )


def _record_years(df: pl.DataFrame | None) -> YearSpan:
    """First and last year covered by a DataFrame, or a sane span when there is no data."""
    first = df["DATE"].min() if df is not None else None
    last = df["DATE"].max() if df is not None else None
    if isinstance(first, _date) and isinstance(last, _date):
        return YearSpan.of(first.year, last.year)
    return _FALLBACK_YEARS


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
    marks = {y: str(y) for y in range(year_min, year_max + 1, 10)}

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
                ],
            ),
            dcc.Store(id="dept-store", data=dept),
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
                                        tooltip={"placement": "bottom", "always_visible": True},
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
            dcc.Tabs(
                id="station-tabs",
                value="observations",
                className="station-tabs",
                children=[
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
                                                    html.Label("Hot day definition", className="control-label"),
                                                    dcc.Dropdown(
                                                        id="hot-day-definition",
                                                        options=[
                                                            {"label": d.label, "value": d.label}
                                                            for d in HOT_DAY_OPTIONS
                                                        ],
                                                        value=DEFAULT_HOT_DAY.label,
                                                        clearable=False,
                                                    ),
                                                ],
                                            ),
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
                            _chart_card("chart-monthly-avg-temp-decade"),
                        ],
                    ),
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
                                                        min_date_allowed=_date(_WINDOW_REF_YEAR, 1, 1).isoformat(),
                                                        max_date_allowed=_date(_WINDOW_REF_YEAR, 12, 31).isoformat(),
                                                        start_date=_date(_WINDOW_REF_YEAR, 6, 1).isoformat(),
                                                        end_date=_date(_WINDOW_REF_YEAR, 8, 31).isoformat(),
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
                            _chart_card("chart-density-tmax"),
                            _chart_card("chart-density-tmin"),
                            html.Div(id="cmp-stats"),
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


def update_charts(
    station_id: int | None,
    year_range: list[int],
    dept: str | None,
    granularity_value: str,
) -> tuple[go.Figure, go.Figure, go.Figure]:
    df = _filtered_station_df(station_id, year_range, dept)
    if df is None:
        placeholder = empty_figure(_NO_DATA)
        return placeholder, placeholder, placeholder

    granularity: Granularity = granularity_from(granularity_value)
    df = aggregate(df, granularity)
    station_name = df["station_name"][0]
    return (
        temperature_figure(df, station_name, granularity),
        precipitation_figure(df, station_name, granularity),
        wind_figure(df, station_name, granularity),
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

    current_year = datetime.date.today().year
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

    df_complete = df.filter(pl.col("DATE").dt.year() < current_year)
    if df_complete.is_empty():
        return fig, []

    agg = yearly_hot_cold(df_complete, definition)
    years_f = [float(y) for y in agg["year"].to_list()]
    if len(years_f) < 2:
        return fig, []

    rows = []
    for label, col in [("Hot days", "hot_days"), ("Cold days", "cold_days")]:
        trend: LinearTrend = linear_trend(years_f, agg[col].to_list())
        sign = "+" if trend.slope >= 0 else "−"
        rows.append(
            html.Tr(
                [
                    html.Td(label, style={"paddingRight": "1.5rem"}),
                    html.Td(f"{sign}{abs(trend.slope):.2f} days/yr", style={"paddingRight": "1.5rem"}),
                    html.Td(f"R² = {trend.r_squared:.2f}"),
                ]
            )
        )

    stats_card = html.Div(
        className="card",
        children=[
            html.Strong("Tendency lines — "),
            html.Table(html.Tbody(rows), style={"display": "inline-table", "marginLeft": "0.5rem"}),
        ],
    )
    return fig, [stats_card]


def update_monthly_charts(
    station_id: int | None,
    year_range: list[int],
    dept: str | None,
) -> tuple[go.Figure, go.Figure]:
    """Render the two monthly average temperature charts."""
    df = _filtered_station_df(station_id, year_range, dept)
    if df is None:
        placeholder = empty_figure(_NO_DATA)
        return placeholder, placeholder
    station_name = df["station_name"][0]
    return (
        monthly_avg_temp_figure(df, station_name),
        monthly_avg_temp_by_decade_figure(df, station_name),
    )


def _window_from(start_date: str | None, end_date: str | None) -> DayWindow:
    if not start_date or not end_date:
        return DEFAULT_WINDOW
    return DayWindow.from_dates(_date.fromisoformat(start_date[:10]), _date.fromisoformat(end_date[:10]))


def _typed_span(start: int | None, end: int | None, fallback: YearSpan) -> YearSpan:
    """A year box cleared to empty falls back to the record bound rather than blanking the chart."""
    return YearSpan.of(
        start if start is not None else fallback.start,
        end if end is not None else fallback.end,
    )


def _describe_cell(dist: Distribution | None) -> str:
    if dist is None:
        return "no data"
    return f"n={dist.n} \u00b7 mean {dist.mean:.1f} \u00b7 median {dist.median:.1f} \u00b7 p90 {dist.p90:.1f} \u00b0C"


def _delta_cell(a: Distribution | None, b: Distribution | None) -> str:
    if a is None or b is None:
        return "\u2014"
    return f"\u0394 mean {b.mean - a.mean:+.1f} \u00b7 \u0394 p90 {b.p90 - a.p90:+.1f} \u00b0C"


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
) -> tuple[go.Figure, go.Figure, list[Any]]:
    """Overlay the Tmax and Tmin densities of two year ranges over a shared calendar window."""
    df = _station_df(station_id, dept)
    if df is None:
        placeholder = empty_figure(_NO_DATA)
        return placeholder, placeholder, []

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
        (what, describe(df_a[col]), describe(df_b[col])) for what, col in (("Tmax", "temp_max"), ("Tmin", "temp_min"))
    ]
    return figure("temp_max", "Daily Tmax"), figure("temp_min", "Daily Tmin"), _stats_card(rows, label_a, label_b)


def year_slider_style(tab: str | None) -> dict[str, str]:
    """Hide the page-level year range on the tab that brings its own two periods."""
    return {"display": "none"} if tab == _COMPARISON_TAB else {}


def register_callbacks(app: Dash) -> None:
    app.callback(
        Output("year-slider-group", "style"),
        Input("station-tabs", "value"),
    )(year_slider_style)

    app.callback(
        Output("chart-temperature", "figure"),
        Output("chart-precipitation", "figure"),
        Output("chart-wind", "figure"),
        Input("station-dropdown", "value"),
        Input("year-slider", "value"),
        State("dept-store", "data"),
        Input("granularity-radio", "value"),
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
        Output("chart-monthly-avg-temp", "figure"),
        Output("chart-monthly-avg-temp-decade", "figure"),
        Input("station-dropdown", "value"),
        Input("year-slider", "value"),
        State("dept-store", "data"),
    )(update_monthly_charts)

    app.callback(
        Output("chart-density-tmax", "figure"),
        Output("chart-density-tmin", "figure"),
        Output("cmp-stats", "children"),
        Input("station-dropdown", "value"),
        State("dept-store", "data"),
        Input("cmp-a-start", "value"),
        Input("cmp-a-end", "value"),
        Input("cmp-b-start", "value"),
        Input("cmp-b-end", "value"),
        Input("cmp-window", "start_date"),
        Input("cmp-window", "end_date"),
    )(update_comparison_charts)
