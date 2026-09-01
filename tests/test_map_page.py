from __future__ import annotations

from dataclasses import replace

import plotly.graph_objects as go
import pytest
from dash import dcc, html
from dash.exceptions import PreventUpdate

import src.pages.components as components
import src.pages.map_page as map_page
from src.data_loader import IndexedStation, RecordSpan
from src.pages.map_page import _map_figure, layout, on_map_click
from tests.conftest import find_component

_SAMPLE_STATIONS = [
    IndexedStation(31001, "TOULOUSE", "31", 43.6, 1.44, 152, span=None),
    IndexedStation(31002, "BLAGNAC", "31", 43.63, 1.37, 151, span=None),
]

_SPANNED_STATIONS = [
    replace(s, span=RecordSpan(first_year=1950 + i, last_year=2025)) for i, s in enumerate(_SAMPLE_STATIONS)
]


@pytest.fixture(autouse=True)
def patch_stations(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(map_page, "station_index", lambda: _SAMPLE_STATIONS)
    monkeypatch.setattr(components, "station_index", lambda: _SAMPLE_STATIONS)


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
    monkeypatch.setattr(map_page, "station_index", lambda: [])
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
    graph = find_component(layout(), dcc.Graph)
    assert graph is not None
    assert graph.id == "station-map"


def test_layout_graph_has_scroll_zoom() -> None:
    graph = find_component(layout(), dcc.Graph)
    assert graph is not None
    assert graph.config.get("scrollZoom") is True


def test_layout_offers_a_station_search() -> None:
    search = find_component(layout(), dcc.Dropdown)
    assert search is not None
    assert search.id == "station-search"
    assert search.options[0]["value"] == "/station?dept=31&station=31001"


def test_map_figure_colours_by_record_length_when_the_index_has_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(map_page, "station_index", lambda: _SPANNED_STATIONS)
    marker = _map_figure().data[0].marker
    assert list(marker.color) == [76, 75]
    assert marker.colorscale is not None


def test_map_figure_falls_back_to_one_colour_without_record_length() -> None:
    marker = _map_figure().data[0].marker
    assert isinstance(marker.color, str)


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
