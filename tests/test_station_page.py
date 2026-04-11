from __future__ import annotations

from unittest.mock import patch

import plotly.graph_objects as go
import pytest
from dash import dcc, html

import src.pages.station_page as station_page
from src.data_loader import Granularity, Truncated
from src.pages.station_page import layout, update_charts, update_monthly_charts, update_yearly_chart
from tests.conftest import find_component


# ---------------------------------------------------------------------------
# layout
# ---------------------------------------------------------------------------


def test_layout_returns_div(sample_df) -> None:
    with patch("src.pages.station_page.load_department_cached", return_value=sample_df):
        result = layout("?dept=31&station=31001")
    assert isinstance(result, html.Div)


def test_layout_dept_store_has_correct_value(sample_df) -> None:
    with patch("src.pages.station_page.load_department_cached", return_value=sample_df):
        result = layout("?dept=31&station=31001")
    store = next(c for c in result.children if isinstance(c, dcc.Store))
    assert store.data == "31"


def test_layout_station_dropdown_pre_populated(sample_df) -> None:
    with patch("src.pages.station_page.load_department_cached", return_value=sample_df):
        result = layout("?dept=31&station=31001")
    dropdown = find_component(result, dcc.Dropdown)
    assert dropdown is not None
    assert len(dropdown.options) == 2  # two stations in sample_df


def test_layout_invalid_station_falls_back_to_first(sample_df) -> None:
    with patch("src.pages.station_page.load_department_cached", return_value=sample_df):
        result = layout("?dept=31&station=99999")
    dropdown = find_component(result, dcc.Dropdown)
    assert dropdown is not None
    assert dropdown.value in {31001, 31002}


def test_layout_empty_search_uses_fallback_years() -> None:
    with patch("src.pages.station_page.load_department_cached", return_value=None):
        result = layout("")
    slider = find_component(result, dcc.RangeSlider)
    assert slider is not None
    assert slider.min == 1950
    assert slider.max == 2026


def test_layout_empty_dataframe_uses_fallback_years(sample_df) -> None:
    """An empty DataFrame (non-None but zero rows) must not crash on .min().year."""
    with patch("src.pages.station_page.load_department_cached", return_value=sample_df.clear()):
        result = layout("?dept=31&station=31001")
    slider = find_component(result, dcc.RangeSlider)
    assert slider is not None
    assert slider.min == 1950
    assert slider.max == 2026


def test_layout_header_shows_department_name(sample_df) -> None:
    with patch("src.pages.station_page.load_department_cached", return_value=sample_df):
        result = layout("?dept=31&station=31001")

    def find_dept_span(node) -> bool:
        if isinstance(node, html.Span) and "Haute-Garonne" in (node.children or ""):
            return True
        for child in getattr(node, "children", []) or []:
            if find_dept_span(child):
                return True
        return False

    assert find_dept_span(result)


def test_layout_contains_back_link(sample_df) -> None:
    with patch("src.pages.station_page.load_department_cached", return_value=sample_df):
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
    with patch("src.pages.station_page.load_department_cached", return_value=sample_df):
        temp, precip, wind = update_charts(31001, [2020, 2020], "31", "day")
    assert all(isinstance(f, go.Figure) for f in (temp, precip, wind))


def test_update_charts_no_dept_returns_placeholders() -> None:
    temp, precip, wind = update_charts(31001, [2020, 2020], None, "day")
    assert all(len(f.data) == 0 for f in (temp, precip, wind))


def test_update_charts_no_station_returns_placeholders(sample_df) -> None:
    with patch("src.pages.station_page.load_department_cached", return_value=sample_df):
        temp, precip, wind = update_charts(None, [2020, 2020], "31", "day")
    assert all(len(f.data) == 0 for f in (temp, precip, wind))


def test_update_charts_load_failure_returns_placeholders() -> None:
    with patch("src.pages.station_page.load_department_cached", return_value=None):
        temp, precip, wind = update_charts(31001, [2020, 2020], "31", "day")
    assert all(len(f.data) == 0 for f in (temp, precip, wind))


def test_update_charts_year_filter_applied(sample_df) -> None:
    # sample_df has dates only in 2020 — filtering to 2021 should yield empty/placeholder
    with patch("src.pages.station_page.load_department_cached", return_value=sample_df):
        temp, precip, wind = update_charts(31001, [2021, 2021], "31", "day")
    assert all(len(f.data) == 0 for f in (temp, precip, wind))


def test_update_charts_calls_load_department_cached_per_invocation(sample_df) -> None:
    # update_charts always delegates to load_department_cached; caching is its responsibility.
    with patch("src.pages.station_page.load_department_cached", return_value=sample_df) as mock_load:
        update_charts(31001, [2020, 2020], "31", "day")
        update_charts(31001, [2020, 2020], "31", "day")
    assert mock_load.call_count == 2


# ---------------------------------------------------------------------------
# Granularity control
# ---------------------------------------------------------------------------


def test_layout_contains_granularity_radio(sample_df) -> None:
    with patch("src.pages.station_page.load_department_cached", return_value=sample_df):
        result = layout("?dept=31&station=31001")
    radio = find_component(result, dcc.RadioItems)
    assert radio is not None
    assert radio.id == "granularity-radio"


def test_layout_granularity_radio_default_is_day(sample_df) -> None:
    with patch("src.pages.station_page.load_department_cached", return_value=sample_df):
        result = layout("?dept=31&station=31001")
    radio = find_component(result, dcc.RadioItems)
    assert radio is not None
    assert radio.value == "day"


def test_layout_granularity_radio_has_three_options(sample_df) -> None:
    with patch("src.pages.station_page.load_department_cached", return_value=sample_df):
        result = layout("?dept=31&station=31001")
    radio = find_component(result, dcc.RadioItems)
    assert radio is not None
    assert len(radio.options) == 3
    option_values = {opt["value"] for opt in radio.options}
    assert option_values == {"day", Granularity.WEEK.label, Granularity.MONTH.label}


def test_update_charts_weekly_returns_figures(sample_df) -> None:
    with patch("src.pages.station_page.load_department_cached", return_value=sample_df):
        temp, precip, wind = update_charts(31001, [2020, 2020], "31", Granularity.WEEK.label)
    assert all(isinstance(f, go.Figure) for f in (temp, precip, wind))


def test_update_charts_monthly_title_suffix(sample_df) -> None:
    with patch("src.pages.station_page.load_department_cached", return_value=sample_df):
        temp, _, _ = update_charts(31001, [2020, 2020], "31", Granularity.MONTH.label)
    assert "(monthly avg)" in temp.layout.title.text


def test_update_charts_weekly_title_suffix(sample_df) -> None:
    with patch("src.pages.station_page.load_department_cached", return_value=sample_df):
        temp, _, _ = update_charts(31001, [2020, 2020], "31", Granularity.WEEK.label)
    assert "(weekly avg)" in temp.layout.title.text


def test_update_charts_daily_no_title_suffix(sample_df) -> None:
    with patch("src.pages.station_page.load_department_cached", return_value=sample_df):
        temp, _, _ = update_charts(31001, [2020, 2020], "31", "day")
    assert "avg" not in temp.layout.title.text


# ---------------------------------------------------------------------------
# update_yearly_chart
# ---------------------------------------------------------------------------


def test_update_yearly_chart_returns_figure(sample_df) -> None:
    with patch("src.pages.station_page.load_department_cached", return_value=sample_df):
        fig, stats = update_yearly_chart(31001, [2020, 2020], "31", [])
    assert isinstance(fig, go.Figure)
    assert stats == []


def test_update_yearly_chart_no_data_returns_placeholder() -> None:
    fig, stats = update_yearly_chart(None, [2020, 2020], None, [])
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 0
    assert stats == []


def test_update_yearly_chart_trend_toggle_adds_traces(sample_df) -> None:
    # sample_df has only 1 year so trend guard is not met; just confirm no crash
    with patch("src.pages.station_page.load_department_cached", return_value=sample_df):
        fig, stats = update_yearly_chart(31001, [2020, 2020], "31", ["show"])
    assert isinstance(fig, go.Figure)
    assert isinstance(stats, list)


# ---------------------------------------------------------------------------
# update_monthly_charts
# ---------------------------------------------------------------------------


def test_update_monthly_charts_returns_two_figures(sample_df) -> None:
    with patch("src.pages.station_page.load_department_cached", return_value=sample_df):
        monthly, by_decade = update_monthly_charts(31001, [2020, 2020], "31")
    assert isinstance(monthly, go.Figure)
    assert isinstance(by_decade, go.Figure)


def test_update_monthly_charts_no_data_returns_placeholders() -> None:
    monthly, by_decade = update_monthly_charts(None, [2020, 2020], None)
    assert len(monthly.data) == 0
    assert len(by_decade.data) == 0
