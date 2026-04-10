from __future__ import annotations

from unittest.mock import patch

import plotly.graph_objects as go
import pytest
from dash import dcc, html

import src.pages.station_page as station_page
from src.data_loader import Granularity
from src.pages.station_page import layout, update_charts
from tests.conftest import find_component


# ---------------------------------------------------------------------------
# layout
# ---------------------------------------------------------------------------


def test_layout_returns_div(sample_df) -> None:
    with patch("src.pages.station_page._load_cached", return_value=sample_df):
        result = layout("?dept=31&station=31001")
    assert isinstance(result, html.Div)


def test_layout_dept_store_has_correct_value(sample_df) -> None:
    with patch("src.pages.station_page._load_cached", return_value=sample_df):
        result = layout("?dept=31&station=31001")
    store = next(c for c in result.children if isinstance(c, dcc.Store))
    assert store.data == "31"


def test_layout_station_dropdown_pre_populated(sample_df) -> None:
    with patch("src.pages.station_page._load_cached", return_value=sample_df):
        result = layout("?dept=31&station=31001")
    dropdown = find_component(result, dcc.Dropdown)
    assert dropdown is not None
    assert len(dropdown.options) == 2  # two stations in sample_df


def test_layout_invalid_station_falls_back_to_first(sample_df) -> None:
    with patch("src.pages.station_page._load_cached", return_value=sample_df):
        result = layout("?dept=31&station=99999")
    dropdown = find_component(result, dcc.Dropdown)
    assert dropdown is not None
    assert dropdown.value in {31001, 31002}


def test_layout_empty_search_uses_fallback_years() -> None:
    with patch("src.pages.station_page._load_cached", return_value=None):
        result = layout("")
    slider = find_component(result, dcc.RangeSlider)
    assert slider is not None
    assert slider.min == 1950
    assert slider.max == 2026


def test_layout_empty_dataframe_uses_fallback_years(sample_df) -> None:
    """An empty DataFrame (non-None but zero rows) must not crash on .min().year."""
    with patch("src.pages.station_page._load_cached", return_value=sample_df.clear()):
        result = layout("?dept=31&station=31001")
    slider = find_component(result, dcc.RangeSlider)
    assert slider is not None
    assert slider.min == 1950
    assert slider.max == 2026


def test_layout_header_shows_department_name(sample_df) -> None:
    with patch("src.pages.station_page._load_cached", return_value=sample_df):
        result = layout("?dept=31&station=31001")
    # Dept 31 is Haute-Garonne — its name must appear in a Span in the nav header
    span = find_component(result, html.Span)
    assert span is not None
    assert "Haute-Garonne" in (span.children or "")


def test_layout_contains_back_link(sample_df) -> None:
    with patch("src.pages.station_page._load_cached", return_value=sample_df):
        result = layout("?dept=31&station=31001")

    def find_link(node):
        if isinstance(node, dcc.Link) and getattr(node, "href", None) == "/":
            return node
        for child in getattr(node, "children", []) or []:
            found = find_link(child)
            if found:
                return found
        return None

    assert find_link(result) is not None


# ---------------------------------------------------------------------------
# update_charts
# ---------------------------------------------------------------------------


def test_update_charts_returns_three_figures(sample_df) -> None:
    with patch("src.pages.station_page._load_cached", return_value=sample_df):
        temp, precip, wind = update_charts(31001, [2020, 2020], "31", Granularity.DAY.value)
    assert all(isinstance(f, go.Figure) for f in (temp, precip, wind))


def test_update_charts_no_dept_returns_placeholders() -> None:
    temp, precip, wind = update_charts(31001, [2020, 2020], None, Granularity.DAY.value)
    assert all(len(f.data) == 0 for f in (temp, precip, wind))


def test_update_charts_no_station_returns_placeholders(sample_df) -> None:
    with patch("src.pages.station_page._load_cached", return_value=sample_df):
        temp, precip, wind = update_charts(None, [2020, 2020], "31", Granularity.DAY.value)
    assert all(len(f.data) == 0 for f in (temp, precip, wind))


def test_update_charts_load_failure_returns_placeholders() -> None:
    with patch("src.pages.station_page._load_cached", return_value=None):
        temp, precip, wind = update_charts(31001, [2020, 2020], "31", Granularity.DAY.value)
    assert all(len(f.data) == 0 for f in (temp, precip, wind))


def test_update_charts_year_filter_applied(sample_df) -> None:
    # sample_df has dates only in 2020 — filtering to 2021 should yield empty/placeholder
    with patch("src.pages.station_page._load_cached", return_value=sample_df):
        temp, precip, wind = update_charts(31001, [2021, 2021], "31", Granularity.DAY.value)
    assert all(len(f.data) == 0 for f in (temp, precip, wind))


def test_update_charts_calls_load_cached_per_invocation(sample_df) -> None:
    # Cache logic lives inside _load_cached itself; update_charts always delegates to it.
    with patch("src.pages.station_page._load_cached", return_value=sample_df) as mock_load:
        update_charts(31001, [2020, 2020], "31", Granularity.DAY.value)
        update_charts(31001, [2020, 2020], "31", Granularity.DAY.value)
    assert mock_load.call_count == 2


# ---------------------------------------------------------------------------
# Granularity control
# ---------------------------------------------------------------------------


def test_layout_contains_granularity_radio(sample_df) -> None:
    with patch("src.pages.station_page._load_cached", return_value=sample_df):
        result = layout("?dept=31&station=31001")
    radio = find_component(result, dcc.RadioItems)
    assert radio is not None
    assert radio.id == "granularity-radio"


def test_layout_granularity_radio_default_is_day(sample_df) -> None:
    with patch("src.pages.station_page._load_cached", return_value=sample_df):
        result = layout("?dept=31&station=31001")
    radio = find_component(result, dcc.RadioItems)
    assert radio is not None
    assert radio.value == Granularity.DAY.value


def test_layout_granularity_radio_has_three_options(sample_df) -> None:
    with patch("src.pages.station_page._load_cached", return_value=sample_df):
        result = layout("?dept=31&station=31001")
    radio = find_component(result, dcc.RadioItems)
    assert radio is not None
    assert len(radio.options) == 3
    option_values = {opt["value"] for opt in radio.options}
    assert option_values == {Granularity.DAY.value, Granularity.WEEK.value, Granularity.MONTH.value}


def test_update_charts_weekly_returns_figures(sample_df) -> None:
    with patch("src.pages.station_page._load_cached", return_value=sample_df):
        temp, precip, wind = update_charts(31001, [2020, 2020], "31", Granularity.WEEK.value)
    assert all(isinstance(f, go.Figure) for f in (temp, precip, wind))


def test_update_charts_monthly_title_suffix(sample_df) -> None:
    with patch("src.pages.station_page._load_cached", return_value=sample_df):
        temp, _, _ = update_charts(31001, [2020, 2020], "31", Granularity.MONTH.value)
    assert "(monthly avg)" in temp.layout.title.text


def test_update_charts_weekly_title_suffix(sample_df) -> None:
    with patch("src.pages.station_page._load_cached", return_value=sample_df):
        temp, _, _ = update_charts(31001, [2020, 2020], "31", Granularity.WEEK.value)
    assert "(weekly avg)" in temp.layout.title.text


def test_update_charts_daily_no_title_suffix(sample_df) -> None:
    with patch("src.pages.station_page._load_cached", return_value=sample_df):
        temp, _, _ = update_charts(31001, [2020, 2020], "31", Granularity.DAY.value)
    assert "avg" not in temp.layout.title.text
