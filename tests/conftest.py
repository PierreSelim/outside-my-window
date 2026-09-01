from __future__ import annotations

from datetime import date
from typing import Any

import polars as pl
import pytest


def find_component[T](node: Any, cls: type[T]) -> T | None:
    """Depth-first search for the first Dash component of a given type."""
    if isinstance(node, cls):
        return node
    children = getattr(node, "children", None)
    if isinstance(children, list):
        for child in children:
            result = find_component(child, cls)
            if result is not None:
                return result
    elif children is not None and not isinstance(children, str):
        return find_component(children, cls)
    return None


def find_by_id(node: Any, component_id: str) -> Any:
    """Depth-first search for the first Dash component carrying a given id."""
    if getattr(node, "id", None) == component_id:
        return node
    children = getattr(node, "children", None)
    candidates = children if isinstance(children, list) else [children]
    for child in candidates:
        if child is None or isinstance(child, str):
            continue
        found = find_by_id(child, component_id)
        if found is not None:
            return found
    return None


@pytest.fixture
def sample_df() -> pl.DataFrame:
    """Minimal DataFrame as returned by load_department() — columns already renamed."""
    return pl.DataFrame(
        {
            "station_id": [31001, 31001, 31001, 31002, 31002],
            "station_name": ["TOULOUSE", "TOULOUSE", "TOULOUSE", "BLAGNAC", "BLAGNAC"],
            "lat": [43.60, 43.60, 43.60, 43.63, 43.63],
            "lon": [1.44, 1.44, 1.44, 1.37, 1.37],
            "altitude": [152, 152, 152, 151, 151],
            "DATE": [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 1), date(2020, 1, 2)],
            "temp_min": [-1.0, 0.5, 2.0, -2.0, 1.0],
            "temp_max": [8.0, 10.5, 12.0, 7.0, 9.0],
            "precipitation": [0.0, 2.5, 0.0, 0.0, 1.0],
            "wind_mean": [3.0, 5.0, 2.0, None, None],
            "wind_gust": [8.0, 12.0, 6.0, None, None],
        },
        schema={
            "station_id": pl.Int32,
            "station_name": pl.String,
            "lat": pl.Float64,
            "lon": pl.Float64,
            "altitude": pl.Int32,
            "DATE": pl.Date,
            "temp_min": pl.Float64,
            "temp_max": pl.Float64,
            "precipitation": pl.Float64,
            "wind_mean": pl.Float64,
            "wind_gust": pl.Float64,
        },
    )
