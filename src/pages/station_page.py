from __future__ import annotations

from urllib.parse import parse_qs

import polars as pl
from dash import Dash, Input, Output, State, dcc, html

from src.charts import empty_figure, precipitation_figure, temperature_figure, wind_figure
from src.data_loader import Granularity, Station, aggregate, load_department, stations_from
from src.departments import DEPT_NAMES

# ---------------------------------------------------------------------------
# Server-side data cache — one DataFrame per department, loaded on first visit
# ---------------------------------------------------------------------------

_dept_cache: dict[str, pl.DataFrame | None] = {}


def _load_cached(dept: str) -> pl.DataFrame | None:
    if dept not in _dept_cache:
        _dept_cache[dept] = load_department(dept)
    return _dept_cache[dept]


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

_FALLBACK_MARKS = {y: str(y) for y in range(1950, 2027, 10)}


def layout(search: str = "") -> html.Div:
    """Build the station detail page.

    Parses dept and station from the URL search string, loads department data,
    and pre-populates controls so the initial chart render fires immediately.
    """
    params = parse_qs(search.lstrip("?"))
    dept: str | None = params.get("dept", [None])[0]
    raw_station = params.get("station", [None])[0]
    initial_station: int | None = int(raw_station) if raw_station else None

    df = _load_cached(dept) if dept else None
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

    return html.Div(
        style={"fontFamily": "sans-serif", "maxWidth": "1400px", "margin": "0 auto", "padding": "1rem"},
        children=[
            # Navigation
            html.Div(
                style={"marginBottom": "1rem"},
                children=[
                    dcc.Link("← Back to map", href="/", style={"color": "#4C9BE8", "textDecoration": "none"}),
                    html.Span(
                        f"  ·  {DEPT_NAMES.get(dept, dept)} ({dept})" if dept else "",
                        style={"color": "#888", "marginLeft": "0.5rem"},
                    ),
                ],
            ),

            # Store dept for use in update_charts
            dcc.Store(id="dept-store", data=dept),

            # Controls row
            html.Div(
                style={"display": "flex", "gap": "2rem", "alignItems": "flex-start", "marginBottom": "1.5rem"},
                children=[
                    html.Div(
                        style={"flex": "0 0 320px"},
                        children=[
                            html.Label("Station", style={"fontWeight": "bold"}),
                            dcc.Dropdown(
                                id="station-dropdown",
                                options=station_options,
                                value=initial_station,
                                clearable=False,
                            ),
                        ],
                    ),
                    html.Div(
                        style={"flex": "1"},
                        children=[
                            html.Label("Year range", style={"fontWeight": "bold"}),
                            dcc.RangeSlider(
                                id="year-slider",
                                min=year_min,
                                max=year_max,
                                step=1,
                                value=[max(year_min, year_max - 10), year_max],
                                marks=marks,
                                tooltip={"placement": "bottom", "always_visible": True},
                            ),
                        ],
                    ),
                    html.Div(
                        style={"flex": "0 0 220px"},
                        children=[
                            html.Label("Granularity", style={"fontWeight": "bold"}),
                            dcc.RadioItems(
                                id="granularity-radio",
                                options=[
                                    {"label": "Day",   "value": Granularity.DAY.value},
                                    {"label": "Week",  "value": Granularity.WEEK.value},
                                    {"label": "Month", "value": Granularity.MONTH.value},
                                ],
                                value=Granularity.DAY.value,
                                inline=True,
                                inputStyle={"marginRight": "4px"},
                                labelStyle={"marginRight": "12px"},
                            ),
                        ],
                    ),
                ],
            ),

            # Charts
            dcc.Graph(id="chart-temperature", config={"displayModeBar": False}),
            dcc.Graph(id="chart-precipitation", config={"displayModeBar": False}),
            dcc.Graph(id="chart-wind", config={"displayModeBar": False}),
        ],
    )


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


def update_charts(
    station_id: int | None,
    year_range: list[int],
    dept: str | None,
    granularity_value: str,
) -> tuple:
    if dept is None or station_id is None:
        placeholder = empty_figure("No data available")
        return placeholder, placeholder, placeholder

    df_full = _load_cached(dept)
    if df_full is None:
        placeholder = empty_figure("No data available")
        return placeholder, placeholder, placeholder

    year_start, year_end = year_range
    df = df_full.filter(
        (pl.col("station_id") == station_id)
        & (pl.col("DATE").dt.year() >= year_start)
        & (pl.col("DATE").dt.year() <= year_end)
    )

    if df.is_empty():
        placeholder = empty_figure("No data for this station / period")
        return placeholder, placeholder, placeholder

    granularity = Granularity(granularity_value)
    df = aggregate(df, granularity)
    label = granularity.title_suffix
    station_name = df["station_name"][0]
    return (
        temperature_figure(df, station_name, label),
        precipitation_figure(df, station_name, label),
        wind_figure(df, station_name, label),
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
