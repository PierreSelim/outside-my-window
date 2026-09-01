from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html
from dash.exceptions import PreventUpdate

from src.data_loader import IndexedStation, station_index
from src.pages.components import station_href, station_search

_MISSING_INDEX = "Station index not found — run: uv run python scripts/build_station_index.py"
_UNIFORM_COLOR = "#2C6079"
_RECORD_SCALE: list[list[Any]] = [[0.0, "#D9C7A7"], [0.5, "#7C9099"], [1.0, "#1E3040"]]


def _record_lengths(stations: list[IndexedStation]) -> list[int] | None:
    """Years of record per station, or None unless every station has one — the colour array
    must line up with the marker array, so a partial index falls back to a flat colour."""
    lengths = [s.span.n_years for s in stations if s.span is not None]
    return lengths if len(lengths) == len(stations) else None


def _marker(stations: list[IndexedStation]) -> dict[str, Any]:
    """Colour carries record length when the index has it: a 170-year record and a 3-year one
    are not the same offer, and the map is otherwise 2 000 identical dots."""
    lengths = _record_lengths(stations)
    if lengths is None:
        return {"size": 7, "color": _UNIFORM_COLOR, "opacity": 0.75}
    return {
        "size": 7,
        "opacity": 0.85,
        "color": lengths,
        "colorscale": _RECORD_SCALE,
        "cmin": 0,
        "colorbar": {
            "title": {"text": "Years of<br>record", "side": "right"},
            "thickness": 10,
            "outlinewidth": 0,
            "x": 0.99,
            "bgcolor": "rgba(251,248,243,0.85)",
        },
    }


def _map_figure() -> go.Figure:
    stations = station_index()
    if not stations:
        fig = go.Figure()
        fig.update_layout(
            annotations=[{"text": _MISSING_INDEX, "showarrow": False, "font": {"size": 14}}],
            xaxis={"visible": False},
            yaxis={"visible": False},
        )
        return fig

    has_span = _record_lengths(stations) is not None
    fig = go.Figure(
        go.Scattermap(
            lat=[s.lat for s in stations],
            lon=[s.lon for s in stations],
            mode="markers",
            marker=_marker(stations),
            text=[s.name for s in stations],
            customdata=[
                [
                    s.dept,
                    s.station_id,
                    s.altitude,
                    s.span.first_year if s.span else "?",
                    s.span.n_years if s.span else "?",
                ]
                for s in stations
            ],
            hovertemplate=(
                "<b>%{text}</b><br>Department %{customdata[0]}<br>Altitude %{customdata[2]} m"
                + ("<br>Record since %{customdata[3]} (%{customdata[4]} years)" if has_span else "")
                + "<extra>Click to open</extra>"
            ),
        )
    )
    fig.update_layout(
        map={"style": "open-street-map", "center": {"lat": 46.5, "lon": 2.5}, "zoom": 5},
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        clickmode="event",
    )
    return fig


def layout() -> html.Div:
    return html.Div(
        children=[
            html.Div(
                className="map-overlay",
                children=[
                    station_search("Search a station by name…"),
                    html.Span("or click any station on the map", className="map-overlay-hint"),
                ],
            ),
            dcc.Graph(
                id="station-map",
                figure=_map_figure(),
                config={"displayModeBar": False, "scrollZoom": True},
                style={"height": "calc(100vh - 52px)"},
            ),
        ],
    )


def on_map_click(click_data: dict[str, Any] | None) -> str:
    if not click_data or "points" not in click_data:
        raise PreventUpdate
    point = click_data["points"][0]
    dept, station_id = point["customdata"][0], point["customdata"][1]
    return station_href(dept, station_id)


def register_callbacks(app: Dash) -> None:
    app.callback(
        Output("url", "href"),
        Input("station-map", "clickData"),
        prevent_initial_call=True,
    )(on_map_click)
