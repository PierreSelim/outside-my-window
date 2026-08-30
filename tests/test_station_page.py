from __future__ import annotations

import re
from datetime import date, timedelta
from unittest.mock import patch

import plotly.graph_objects as go
import polars as pl
from dash import Dash, dcc, html

from src.data_loader import Granularity
from src.pages.station_page import (
    layout,
    register_callbacks,
    update_charts,
    update_comparison_charts,
    update_monthly_charts,
    update_yearly_chart,
    year_slider_style,
)
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


def test_update_yearly_chart_one_complete_year_trend_stats_empty(sample_df) -> None:
    """Exactly 1 complete year: trend toggle ON must yield an empty stats card.

    The chart's regression guard requires >= 2 complete years; the stats card must
    mirror that guard so it never shows +0.00 days/yr rows for a degenerate fit.
    """
    with patch("src.pages.station_page.load_department_cached", return_value=sample_df):
        fig, stats = update_yearly_chart(31001, [2020, 2020], "31", ["show"])
    assert isinstance(fig, go.Figure)
    assert stats == []


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


# ---------------------------------------------------------------------------
# update_comparison_charts
# ---------------------------------------------------------------------------


def _long_record_df() -> pl.DataFrame:
    days = [date(1961, 1, 1) + timedelta(days=i) for i in range(365 * 60)]
    n = len(days)
    return pl.DataFrame(
        {
            "station_id": pl.Series([31001] * n, dtype=pl.Int32),
            "station_name": ["TOULOUSE"] * n,
            "lat": [43.6] * n,
            "lon": [1.44] * n,
            "altitude": pl.Series([152] * n, dtype=pl.Int32),
            "DATE": days,
            "temp_min": [5.0 + (d.year - 1961) * 0.03 + (d.timetuple().tm_yday % 7) for d in days],
            "temp_max": [15.0 + (d.year - 1961) * 0.05 + (d.timetuple().tm_yday % 9) for d in days],
            "temp_amplitude": [10.0] * n,
            "precipitation": [0.0] * n,
            "wind_mean": [1.0] * n,
            "wind_gust": [2.0] * n,
        }
    )


def test_layout_has_a_comparison_tab(sample_df) -> None:
    with patch("src.pages.station_page.load_department_cached", return_value=sample_df):
        result = layout("?dept=31&station=31001")
    tabs = find_component(result, dcc.Tabs)
    assert tabs is not None
    assert [tab.value for tab in tabs.children] == [
        "observations",
        "yearly-extremes",
        "monthly-avg",
        "comparison",
    ]


def test_update_comparison_charts_returns_two_figures_and_stats() -> None:
    with patch("src.pages.station_page.load_department_cached", return_value=_long_record_df()):
        tmax, tmin, stats = update_comparison_charts(31001, "31", 1961, 1990, 1991, 2020, "2000-06-01", "2000-08-31")
    assert isinstance(tmax, go.Figure)
    assert isinstance(tmin, go.Figure)
    assert len(tmax.data) == 2
    assert stats


def test_update_comparison_charts_title_carries_the_window_label() -> None:
    with patch("src.pages.station_page.load_department_cached", return_value=_long_record_df()):
        tmax, _, _ = update_comparison_charts(31001, "31", 1961, 1990, 1991, 2020, "2000-06-01", "2000-08-31")
    assert "1 Jun \u2013 31 Aug" in tmax.layout.title.text
    assert "TOULOUSE" in tmax.layout.title.text


def test_update_comparison_charts_identical_periods_show_no_shift() -> None:
    with patch("src.pages.station_page.load_department_cached", return_value=_long_record_df()):
        _, _, stats = update_comparison_charts(31001, "31", 1991, 2020, 1991, 2020, "2000-06-01", "2000-08-31")
    text = str(stats[0])
    assert "+0.0" in text


def test_update_comparison_charts_missing_window_falls_back_to_default() -> None:
    with patch("src.pages.station_page.load_department_cached", return_value=_long_record_df()):
        tmax, _, _ = update_comparison_charts(31001, "31", 1961, 1990, 1991, 2020, None, None)
    assert "1 Jun \u2013 31 Aug" in tmax.layout.title.text


def test_update_comparison_charts_no_data_returns_placeholders() -> None:
    tmax, tmin, stats = update_comparison_charts(None, None, 1961, 1990, 1991, 2020, None, None)
    assert len(tmax.data) == 0
    assert len(tmin.data) == 0
    assert stats == []


def test_update_comparison_charts_empty_period_returns_placeholders() -> None:
    with patch("src.pages.station_page.load_department_cached", return_value=_long_record_df()):
        tmax, _, _ = update_comparison_charts(31001, "31", 1800, 1810, 1991, 2020, "2000-06-01", "2000-08-31")
    assert len(tmax.data) == 0


def test_update_comparison_charts_all_null_temperatures_returns_no_stats() -> None:
    df = _long_record_df().with_columns(
        pl.lit(None, dtype=pl.Float64).alias("temp_max"),
        pl.lit(None, dtype=pl.Float64).alias("temp_min"),
    )
    with patch("src.pages.station_page.load_department_cached", return_value=df):
        tmax, tmin, stats = update_comparison_charts(31001, "31", 1961, 1990, 1991, 2020, "2000-06-01", "2000-08-31")
    assert len(tmax.data) == 0
    assert len(tmin.data) == 0
    assert stats == []


# ---------------------------------------------------------------------------
# register_callbacks
# ---------------------------------------------------------------------------


def _component_ids(node: object) -> set[str]:
    found = set()
    component_id = getattr(node, "id", None)
    if isinstance(component_id, str):
        found.add(component_id)
    children = getattr(node, "children", None)
    for child in children if isinstance(children, list) else [children]:
        if child is not None and not isinstance(child, str):
            found |= _component_ids(child)
    return found


def _wired_ids(app: Dash) -> set[str]:
    ids = set()
    for entry in app.callback_map.values():
        ids |= {ref["id"] for ref in entry["inputs"] + entry["state"]}
    for entry in app._callback_list:
        ids |= set(re.findall(r"([a-z0-9-]+)\.[a-z_]+", entry["output"]))
    return ids


def test_register_callbacks_wires_every_id_present_in_the_layout(sample_df) -> None:
    """A component id typo would leave a chart silently dead in the browser."""
    app = Dash(__name__, suppress_callback_exceptions=True)
    register_callbacks(app)
    with patch("src.pages.station_page.load_department_cached", return_value=sample_df):
        page = layout("?dept=31&station=31001")
    assert _wired_ids(app) <= _component_ids(page)


def test_register_callbacks_registers_the_comparison_callback(sample_df) -> None:
    app = Dash(__name__, suppress_callback_exceptions=True)
    register_callbacks(app)
    outputs = [entry["output"] for entry in app._callback_list]
    assert any("chart-density-tmax" in o and "cmp-stats" in o for o in outputs)


# ---------------------------------------------------------------------------
# tab navigation
# ---------------------------------------------------------------------------


def _tabs(node: object) -> dcc.Tabs | None:
    if isinstance(node, dcc.Tabs):
        return node
    for child in getattr(node, "children", None) or []:
        if not isinstance(child, str):
            found = _tabs(child)
            if found is not None:
                return found
    return None


def test_comparison_is_the_last_tab_and_is_accented(sample_df) -> None:
    """It sits at the end on purpose, so the accent is what makes it findable."""
    with patch("src.pages.station_page.load_department_cached", return_value=sample_df):
        page = layout("?dept=31&station=31001")
    last = _tabs(page).children[-1]
    assert last.value == "comparison"
    assert last.label == "Then vs now"
    assert "station-tab--feature" in last.className
    assert "station-tab--feature" in last.selected_className


def test_year_slider_is_hidden_on_the_comparison_tab() -> None:
    """The page-level year range is not an input there; showing it would be a lie."""
    assert year_slider_style("comparison") == {"display": "none"}


def test_year_slider_is_visible_on_every_other_tab() -> None:
    assert all(year_slider_style(tab) == {} for tab in ("observations", "yearly-extremes", "monthly-avg", None))


def _find_by_id(node: object, component_id: str) -> object | None:
    if getattr(node, "id", None) == component_id:
        return node
    children = getattr(node, "children", None)
    for child in children if isinstance(children, list) else [children]:
        if child is not None and not isinstance(child, str):
            found = _find_by_id(child, component_id)
            if found is not None:
                return found
    return None


def test_update_comparison_charts_swaps_inverted_year_boxes() -> None:
    """Typing the end year first is a valid intermediate state, not an error."""
    with patch("src.pages.station_page.load_department_cached", return_value=_long_record_df()):
        forward, _, _ = update_comparison_charts(31001, "31", 1961, 1990, 1991, 2020, None, None)
        inverted, _, _ = update_comparison_charts(31001, "31", 1990, 1961, 2020, 1991, None, None)
    assert forward.data[0].x == inverted.data[0].x


def test_update_comparison_charts_cleared_year_box_falls_back_to_the_record() -> None:
    with patch("src.pages.station_page.load_department_cached", return_value=_long_record_df()):
        tmax, _, _ = update_comparison_charts(31001, "31", None, None, 1991, 2020, None, None)
    assert len(tmax.data) == 2


def test_layout_period_boxes_are_year_inputs(sample_df) -> None:
    """Two typed boxes per period; a RangeSlider was unreadable at this width."""
    with patch("src.pages.station_page.load_department_cached", return_value=sample_df):
        page = layout("?dept=31&station=31001")
    boxes = [_find_by_id(page, box) for box in ("cmp-a-start", "cmp-a-end", "cmp-b-start", "cmp-b-end")]
    assert all(isinstance(b, dcc.Input) and b.type == "number" and b.debounce for b in boxes)
