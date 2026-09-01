from __future__ import annotations

import pytest
from dash.exceptions import PreventUpdate

import src.pages.components as components
from src.data_loader import IndexedStation
from src.pages.components import navigate_to_station, station_href, station_search

_STATIONS = [
    IndexedStation(31001, "TOULOUSE", "31", 43.6, 1.44, 152, span=None),
    IndexedStation(1089001, "AMBERIEU", "01", 45.9, 5.3, 250, span=None),
]


@pytest.fixture(autouse=True)
def patch_index(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(components, "station_index", lambda: _STATIONS)


def test_station_href_builds_the_query_string() -> None:
    assert station_href("31", 31001) == "/station?dept=31&station=31001"


def test_station_search_offers_every_station_in_the_index() -> None:
    options = station_search().options
    assert len(options) == 2
    assert options[0]["label"] == "TOULOUSE (31)"


def test_station_search_crosses_department_boundaries() -> None:
    """The point of the control: a station page can reach a station in another department."""
    values = [o["value"] for o in station_search().options]
    assert "/station?dept=01&station=1089001" in values


def test_navigate_to_station_returns_the_selected_href() -> None:
    assert navigate_to_station("/station?dept=31&station=31001") == "/station?dept=31&station=31001"


def test_navigate_to_station_ignores_an_empty_selection() -> None:
    with pytest.raises(PreventUpdate):
        navigate_to_station(None)
