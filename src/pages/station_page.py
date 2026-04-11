from __future__ import annotations

import datetime
from urllib.parse import parse_qs

import plotly.graph_objects as go
import polars as pl
from dash import Dash, Input, Output, State, dcc, html

from src.charts import (
    _linear_trend,
    empty_figure,
    hot_cold_yearly_figure,
    monthly_avg_temp_by_decade_figure,
    monthly_avg_temp_figure,
    precipitation_figure,
    temperature_figure,
    wind_figure,
)
from src.data_loader import Granularity, Station, Truncated, aggregate, granularity_from, load_department_cached, stations_from
from src.departments import DEPT_NAMES

_DEFAULT_YEAR_WINDOW: int = 20

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


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
    year_min = _date_min.year if _date_min is not None else 1950
    year_max = _date_max.year if _date_max is not None else 2026
    marks = {y: str(y) for y in range(year_min, year_max + 1, 10)}

    valid_ids = {s.station_id for s in stations}
    if initial_station not in valid_ids:
        initial_station = stations[0].station_id if stations else None

    station_options = [{"label": s.name, "value": s.station_id} for s in stations]

    dept_label = f"{DEPT_NAMES.get(dept, dept)} ({dept})" if dept else ""

    return html.Div(
        className="page-container",
        children=[
            # Navigation breadcrumb
            html.Div(
                className="page-nav",
                children=[
                    dcc.Link("← Back to map", href="/", className="back-link"),
                    html.Span("·", className="page-nav-sep"),
                    html.Span(dept_label, className="page-nav-dept"),
                ],
            ),

            # Store dept for use in update_charts
            dcc.Store(id="dept-store", data=dept),

            # Controls card — station and year range apply to all tabs
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

            # Tabbed chart area
            dcc.Tabs(
                id="station-tabs",
                value="observations",
                children=[
                    dcc.Tab(
                        label="Observations",
                        value="observations",
                        children=[
                            # Granularity control — only meaningful for this tab
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
                                                            {"label": "Day",   "value": "day"},
                                                            {"label": "Week",  "value": Granularity.WEEK.label},
                                                            {"label": "Month", "value": Granularity.MONTH.label},
                                                        ],
                                                        value="day",
                                                        inline=True,
                                                    ),
                                                ],
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                            html.Div(className="card card--flush", children=[
                                dcc.Graph(id="chart-temperature", config={"displayModeBar": False}),
                            ]),
                            html.Div(className="card card--flush", children=[
                                dcc.Graph(id="chart-precipitation", config={"displayModeBar": False}),
                            ]),
                            html.Div(className="card card--flush", children=[
                                dcc.Graph(id="chart-wind", config={"displayModeBar": False}),
                            ]),
                        ],
                    ),
                    dcc.Tab(
                        label="Yearly extremes",
                        value="yearly-extremes",
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
                            html.Div(className="card card--flush", children=[
                                dcc.Graph(id="chart-hot-cold-yearly", config={"displayModeBar": False}),
                            ]),
                            html.Div(id="trend-stats"),
                        ],
                    ),
                    dcc.Tab(
                        label="Monthly averages",
                        value="monthly-avg",
                        children=[
                            html.Div(className="card card--flush", children=[
                                dcc.Graph(id="chart-monthly-avg-temp", config={"displayModeBar": False}),
                            ]),
                            html.Div(className="card card--flush", children=[
                                dcc.Graph(id="chart-monthly-avg-temp-decade", config={"displayModeBar": False}),
                            ]),
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
) -> tuple:
    df = _filtered_station_df(station_id, year_range, dept)
    if df is None:
        placeholder = empty_figure("No data available")
        return placeholder, placeholder, placeholder

    granularity: Truncated | None = granularity_from(granularity_value)
    df = aggregate(df, granularity)
    label = granularity.title_suffix if granularity else ""
    station_name = df["station_name"][0]
    return (
        temperature_figure(df, station_name, label),
        precipitation_figure(df, station_name, label),
        wind_figure(df, station_name, label),
    )


def update_yearly_chart(
    station_id: int | None,
    year_range: list[int],
    dept: str | None,
    trend_values: list[str] | None,
) -> tuple[go.Figure, list]:
    """Render the yearly hot/cold days chart, with optional trend lines.

    The current (incomplete) year is excluded so partial counts don't skew the chart.
    Returns (figure, trend_stats_children) — the stats card is empty when trends are off.
    """
    df = _filtered_station_df(station_id, year_range, dept)
    if df is None:
        return empty_figure("No data available"), []
    current_year = datetime.date.today().year
    df = df.filter(pl.col("DATE").dt.year() < current_year)
    if df.is_empty():
        return empty_figure("No data available"), []

    show_trend = "show" in (trend_values or [])
    fig = hot_cold_yearly_figure(df, df["station_name"][0], show_trend=show_trend)

    if not show_trend:
        return fig, []

    yearly = (
        df.with_columns(pl.col("DATE").dt.year().alias("year"))
        .group_by("year")
        .agg([
            (
                pl.col("temp_min").is_not_null()
                & pl.col("temp_max").is_not_null()
                & (pl.col("temp_min") >= 20.0)
                & (pl.col("temp_max") >= 35.0)
            ).sum().alias("hot_days"),
            (pl.col("temp_min").is_not_null() & (pl.col("temp_min") < 0)).sum().alias("cold_days"),
        ])
        .sort("year")
    )
    years_f = [float(y) for y in yearly["year"].to_list()]

    rows = []
    for label, col in [("Hot days", "hot_days"), ("Cold days", "cold_days")]:
        slope, _, r_squared = _linear_trend(years_f, yearly[col].to_list())
        sign = "+" if slope >= 0 else "−"
        rows.append(html.Tr([
            html.Td(label, style={"paddingRight": "1.5rem"}),
            html.Td(f"{sign}{abs(slope):.2f} days/yr", style={"paddingRight": "1.5rem"}),
            html.Td(f"R² = {r_squared:.2f}"),
        ]))

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
) -> tuple:
    """Render the two monthly average temperature charts."""
    df = _filtered_station_df(station_id, year_range, dept)
    if df is None:
        placeholder = empty_figure("No data available")
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
