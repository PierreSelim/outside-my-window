from __future__ import annotations

import datetime
from datetime import date as _date
from typing import Any
from urllib.parse import parse_qs

import plotly.graph_objects as go
import polars as pl
from dash import Dash, Input, Output, State, dcc, html

from src.charts import (
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
from src.transforms import LinearTrend, linear_trend, yearly_hot_cold

_DEFAULT_YEAR_WINDOW: int = 20
_NO_DATA = "No data available"


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------


def _chart_card(graph_id: str) -> html.Div:
    return html.Div(
        className="card card--flush",
        children=[
            dcc.Graph(id=graph_id, config={"displayModeBar": False}),
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

    _date_min = df["DATE"].min() if df is not None else None
    _date_max = df["DATE"].max() if df is not None else None
    year_min = _date_min.year if isinstance(_date_min, _date) else 1950
    year_max = _date_max.year if isinstance(_date_max, _date) else 2026
    marks = {y: str(y) for y in range(year_min, year_max + 1, 10)}

    valid_ids = {s.station_id for s in stations}
    if initial_station not in valid_ids:
        initial_station = stations[0].station_id if stations else None

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
                                    dcc.Checklist(
                                        id="yearly-trend-toggle",
                                        options=[{"label": "  Show tendency lines", "value": "show"}],
                                        value=[],
                                        inline=True,
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
                ],
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


def _filtered_station_df(station_id: int | None, year_range: list[int], dept: str | None) -> pl.DataFrame | None:
    """Return a station+year-filtered DataFrame, or None if inputs are invalid / no data."""
    if dept is None or station_id is None:
        return None
    df_full = load_department_cached(dept)
    if df_full is None:
        return None
    year_start, year_end = year_range
    df = df_full.filter(
        (pl.col("station_id") == station_id)
        & (pl.col("DATE").dt.year() >= year_start)
        & (pl.col("DATE").dt.year() <= year_end)
    )
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
) -> tuple[go.Figure, list[Any]]:
    """Render the yearly hot/cold days chart, with optional trend lines.

    The current (incomplete) year is excluded so partial counts don't skew the chart.
    Returns (figure, trend_stats_children) — the stats card is empty when trends are off.
    """
    df = _filtered_station_df(station_id, year_range, dept)
    if df is None:
        return empty_figure(_NO_DATA), []
    current_year = datetime.date.today().year
    df = df.filter(pl.col("DATE").dt.year() < current_year)
    if df.is_empty():
        return empty_figure(_NO_DATA), []

    show_trend = "show" in (trend_values or [])
    fig = hot_cold_yearly_figure(df, df["station_name"][0], show_trend=show_trend)

    if not show_trend:
        return fig, []

    agg = yearly_hot_cold(df)
    years_f = [float(y) for y in agg["year"].to_list()]

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


def register_callbacks(app: Dash) -> None:
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
    )(update_yearly_chart)

    app.callback(
        Output("chart-monthly-avg-temp", "figure"),
        Output("chart-monthly-avg-temp-decade", "figure"),
        Input("station-dropdown", "value"),
        Input("year-slider", "value"),
        State("dept-store", "data"),
    )(update_monthly_charts)
