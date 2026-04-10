from __future__ import annotations

import plotly.graph_objects as go
import pytest
from dash import html, dcc
from dash.exceptions import PreventUpdate

import src.pages.map_page as map_page
from src.pages.map_page import _map_figure, layout, on_map_click

_SAMPLE_STATIONS = [
    {"station_id": 31001, "station_name": "TOULOUSE", "dept": "31", "lat": 43.6, "lon": 1.44, "altitude": 152},
    {"station_id": 31002, "station_name": "BLAGNAC",  "dept": "31", "lat": 43.63, "lon": 1.37, "altitude": 151},
]


@pytest.fixture(autouse=True)
def patch_stations(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(map_page, "_STATIONS", _SAMPLE_STATIONS)


# ---------------------------------------------------------------------------
# _map_figure
# ---------------------------------------------------------------------------


def test_map_figure_with_stations_returns_scattermap() -> None:
    fig = _map_figure()
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert isinstance(fig.data[0], go.Scattermap)


def test_map_figure_with_stations_has_correct_point_count() -> None:
    fig = _map_figure()
    assert len(fig.data[0].lat) == 2
    assert len(fig.data[0].lon) == 2


def test_map_figure_customdata_contains_dept_and_station_id() -> None:
    fig = _map_figure()
    customdata = fig.data[0].customdata
    assert customdata[0][0] == "31"
    assert customdata[0][1] == 31001


def test_map_figure_no_stations_returns_annotation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(map_page, "_STATIONS", [])
    fig = _map_figure()
    assert len(fig.data) == 0
    assert len(fig.layout.annotations) == 1


# ---------------------------------------------------------------------------
# layout
# ---------------------------------------------------------------------------


def test_layout_returns_div() -> None:
    result = layout()
    assert isinstance(result, html.Div)


def test_layout_contains_graph() -> None:
    result = layout()
    graphs = [c for c in result.children if isinstance(c, dcc.Graph)]
    assert len(graphs) == 1
    assert graphs[0].id == "station-map"


def test_layout_graph_has_scroll_zoom() -> None:
    result = layout()
    graph = next(c for c in result.children if isinstance(c, dcc.Graph))
    assert graph.config.get("scrollZoom") is True


# ---------------------------------------------------------------------------
# on_map_click
# ---------------------------------------------------------------------------


def test_on_map_click_builds_correct_url() -> None:
    click_data = {"points": [{"customdata": ["31", 31001, 152]}]}
    url = on_map_click(click_data)
    assert url == "/station?dept=31&station=31001"


def test_on_map_click_none_raises_prevent_update() -> None:
    with pytest.raises(PreventUpdate):
        on_map_click(None)


def test_on_map_click_empty_raises_prevent_update() -> None:
    with pytest.raises(PreventUpdate):
        on_map_click({})


def test_on_map_click_no_points_key_raises_prevent_update() -> None:
    with pytest.raises(PreventUpdate):
        on_map_click({"foo": "bar"})
