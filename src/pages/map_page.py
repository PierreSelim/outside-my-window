from __future__ import annotations

import json
from pathlib import Path

import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html
from dash.exceptions import PreventUpdate

_STATION_INDEX = Path(__file__).parent.parent.parent / "data" / "stations.json"


def _load_stations() -> list[dict]:
    if not _STATION_INDEX.exists():
        return []
    return json.loads(_STATION_INDEX.read_text(encoding="utf-8"))


_STATIONS: list[dict] = _load_stations()


def _map_figure() -> go.Figure:
    if not _STATIONS:
        fig = go.Figure()
        fig.update_layout(
            annotations=[dict(
                text="Station index not found — run: uv run python scripts/build_station_index.py",
                showarrow=False, font=dict(size=14),
            )],
            xaxis=dict(visible=False), yaxis=dict(visible=False),
        )
        return fig

    fig = go.Figure(
        go.Scattermap(
            lat=[s["lat"] for s in _STATIONS],
            lon=[s["lon"] for s in _STATIONS],
            mode="markers",
            marker=dict(size=7, color="#4C9BE8", opacity=0.75),
            text=[s["station_name"] for s in _STATIONS],
            customdata=[[s["dept"], s["station_id"], s["altitude"]] for s in _STATIONS],
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Département : %{customdata[0]}<br>"
                "Altitude : %{customdata[2]} m"
                "<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        map=dict(style="open-street-map", center=dict(lat=46.5, lon=2.5), zoom=5),
        margin=dict(r=0, t=0, l=0, b=0),
        clickmode="event",
    )
    return fig


def layout() -> html.Div:
    return html.Div(
        children=[
            dcc.Graph(
                id="station-map",
                figure=_map_figure(),
                config={"displayModeBar": False, "scrollZoom": True},
                style={"height": "calc(100vh - 52px)"},
            ),
        ],
    )


def on_map_click(click_data: dict | None) -> str:
    if not click_data or "points" not in click_data:
        raise PreventUpdate
    point = click_data["points"][0]
    dept, station_id, _ = point["customdata"]
    return f"/station?dept={dept}&station={station_id}"


def register_callbacks(app: Dash) -> None:
    app.callback(
        Output("url", "href"),
        Input("station-map", "clickData"),
        prevent_initial_call=True,
    )(on_map_click)
