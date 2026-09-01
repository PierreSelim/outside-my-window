"""Components shared by the map and station pages."""

from __future__ import annotations

from dash import Dash, Input, Output, dcc
from dash.exceptions import PreventUpdate

from src.data_loader import station_index

_SEARCH_ID = "station-search"


def station_href(dept: str, station_id: int) -> str:
    return f"/station?dept={dept}&station={station_id}"


def station_search(placeholder: str = "Search a station…") -> dcc.Dropdown:
    """Type-ahead over every station in France.

    The dropdown is the only control that crosses department boundaries: without it the map is
    the sole way in, and a station page can only reach the ~20 stations of its own department.
    """
    return dcc.Dropdown(
        id=_SEARCH_ID,
        options=[
            {"label": f"{s.name} ({s.dept})", "value": station_href(s.dept, s.station_id)} for s in station_index()
        ],
        placeholder=placeholder,
        value=None,
        className="station-search",
        optionHeight=32,
        maxHeight=320,
    )


def navigate_to_station(href: str | None) -> str:
    if not href:
        raise PreventUpdate
    return href


def register_callbacks(app: Dash) -> None:
    """Registered once for the whole app — both pages render the same search component id."""
    app.callback(
        Output("url", "href", allow_duplicate=True),
        Input(_SEARCH_ID, "value"),
        prevent_initial_call=True,
    )(navigate_to_station)
