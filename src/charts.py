from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Literal

import plotly.graph_objects as go
import polars as pl

from src.data_loader import Granularity
from src.transforms import (
    DEFAULT_HOT_DAY,
    FROST_TMIN,
    MIN_YEAR_COVERAGE,
    TROPICAL_NIGHT_TMIN,
    Density,
    HotDayDefinition,
    LinearTrend,
    gaussian_kde,
    hot_day_predicate,
    linear_trend,
    yearly_hot_cold,
)

_BLUE = "#2C6079"
_RED = "#B4442C"
_GREEN = "#6E675C"
_ORANGE = "#C98A2B"
_GREY = "#9C948A"
# Rain is not temperature: it gets its own hue rather than borrowing the cool of the Tmin traces.
# Lighter too — a bar is a block of colour where a trace is a 0.75px line.
_RAIN = "#7FA3B3"
_BAND_FILL = "rgba(44, 96, 121, 0.13)"
_HOT_DAY_FILL = "rgba(180, 68, 44, 0.16)"
_PARTIAL_FILL = "rgba(110, 103, 92, 0.14)"
_TMAX_SIGMA_FILL = "rgba(180, 68, 44, 0.13)"
_TMIN_SIGMA_FILL = "rgba(44, 96, 121, 0.13)"

_GRID_BASE: dict[str, Any] = {"showgrid": True, "gridcolor": "#E5DCCD", "gridwidth": 1}
_YAXIS_GRID: dict[str, Any] = {
    **_GRID_BASE,
    "zeroline": True,
    "zerolinecolor": "#C9BCA6",
    "zerolinewidth": 1.5,
    "minor": {"showgrid": True, "gridcolor": "rgba(229,220,205,0.45)", "gridwidth": 0.5},
}
_XAXIS_GRID: dict[str, Any] = {**_GRID_BASE}

# Decade chart gradients: oldest → newest
_TMAX_GRADIENT_START: tuple[int, int, int] = (222, 168, 106)
_TMAX_GRADIENT_END: tuple[int, int, int] = (124, 38, 22)
_TMIN_GRADIENT_START: tuple[int, int, int] = (156, 190, 205)
_TMIN_GRADIENT_END: tuple[int, int, int] = (23, 55, 73)

_MONTH_LABELS: list[str] = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
_MONTH_SPINE: pl.DataFrame = pl.DataFrame({"month": list(range(1, 13))}, schema={"month": pl.UInt32})


_FONT_FAMILY = "system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
_TITLE_FAMILY = "'Source Serif 4', Georgia, 'Times New Roman', serif"
_PLOT_BG = "rgba(0, 0, 0, 0)"


def _apply_standard_layout(
    fig: go.Figure,
    title: str,
    yaxis_title: str,
    *,
    legend: bool = True,
    **extra_layout: object,
) -> None:
    layout_kwargs: dict[str, Any] = {
        "title": {
            "text": title,
            "font": {"family": _TITLE_FAMILY, "size": 16, "weight": 600, "color": "#191714"},
            "x": 0,
            "xanchor": "left",
            "pad": {"l": 4},
        },
        "yaxis_title": yaxis_title,
        "hovermode": "x unified",
        "margin": {"t": 52, "b": 24, "l": 52, "r": 8},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": _PLOT_BG,
        "font": {"family": _FONT_FAMILY, "size": 12, "color": "#6E675C"},
        "hoverlabel": {"bgcolor": "#191714", "font_color": "#FBF8F3", "bordercolor": "#191714", "font_size": 12},
        **extra_layout,
    }
    if legend:
        layout_kwargs["legend"] = {
            "orientation": "h",
            "y": 1.06,
            "x": 1,
            "xanchor": "right",
            "font": {"size": 11},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
        }
    fig.update_layout(**layout_kwargs)


def _gradient_color(t: float, start: tuple[int, int, int], end: tuple[int, int, int]) -> str:
    """Linearly interpolate between two RGB colours. t=0 → start, t=1 → end."""
    r = int(start[0] + t * (end[0] - start[0]))
    g = int(start[1] + t * (end[1] - start[1]))
    b = int(start[2] + t * (end[2] - start[2]))
    return f"rgb({r},{g},{b})"


def _hot_day_dates(df: pl.DataFrame, definition: HotDayDefinition) -> list[date]:
    """Dates matching the hot-day definition currently selected on the page."""
    return df.filter(hot_day_predicate(definition))["DATE"].to_list()


def _band_shapes(spans: Sequence[tuple[Any, Any]], fillcolor: str) -> list[dict[str, Any]]:
    """Background bands as plain dicts, to be assigned in one `update_layout(shapes=...)`.

    `add_vrect` revalidates the whole shape list on every call, so drawing a few hundred bands
    one at a time is quadratic: 900 hot days took ~130 s before this was one assignment.
    """
    return [
        {
            "type": "rect",
            "xref": "x",
            "yref": "paper",
            "y0": 0,
            "y1": 1,
            "x0": x0,
            "x1": x1,
            "fillcolor": fillcolor,
            "line": {"width": 0},
            "layer": "below",
        }
        for x0, x1 in spans
    ]


def _date_runs(dates: list[date]) -> list[tuple[date, date]]:
    """Collapse sorted dates into (start, end-exclusive) runs of consecutive days.

    A heatwave is one block, not five stripes — and it is a few hundred shapes instead of a
    few thousand.
    """
    runs: list[tuple[date, date]] = []
    for d in dates:
        if runs and d == runs[-1][1]:
            runs[-1] = (runs[-1][0], d + timedelta(days=1))
        else:
            runs.append((d, d + timedelta(days=1)))
    return runs


def _series(df: pl.DataFrame, col: str) -> tuple[list[date], list[float | None]]:
    """Return (dates, values) lists for a column, keeping nulls.

    Nulls are kept so a gap in the record renders as a gap: dropping them draws a confident
    straight line across the years a station was offline.
    """
    return df["DATE"].to_list(), df[col].to_list()


def _add_temp_trace(
    fig: go.Figure,
    month_labels: list[Any],
    y: list[Any],
    name: str,
    color: str,
    dash: str = "",
    visible: Literal[True, False, "legendonly"] = True,
) -> None:
    line: dict[str, Any] = {"color": color, "width": 1.5, "shape": "spline"}
    if dash:
        line["dash"] = dash
    fig.add_trace(
        go.Scatter(
            x=month_labels,
            y=y,
            name=name,
            line=line,
            mode="lines+markers",
            marker={"size": 4},
            visible=visible,
        )
    )


def _add_sigma_band(
    fig: go.Figure,
    x: list[Any],
    avg: list[Any],
    std: list[Any],
    fillcolor: str,
    n_sigma: float = 2.0,
) -> None:
    """Add a ±n_sigma shaded band using two invisible boundary traces."""
    upper = [a + n_sigma * s if a is not None and s is not None else None for a, s in zip(avg, std, strict=True)]
    lower = [a - n_sigma * s if a is not None and s is not None else None for a, s in zip(avg, std, strict=True)]
    invisible_line: dict[str, Any] = {"color": "rgba(0,0,0,0)", "width": 0, "shape": "spline"}
    fig.add_trace(
        go.Scatter(
            x=x,
            y=upper,
            name="",
            mode="lines",
            line=invisible_line,
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=lower,
            name="",
            mode="lines",
            line=invisible_line,
            fill="tonexty",
            fillcolor=fillcolor,
            showlegend=False,
            hoverinfo="skip",
        )
    )


def temperature_figure(
    df: pl.DataFrame,
    station_name: str,
    granularity: Granularity = Granularity.DAY,
    definition: HotDayDefinition = DEFAULT_HOT_DAY,
) -> go.Figure:
    """Line chart: TN/TX shaded band, with the days matching `definition` highlighted."""
    fig = go.Figure()

    # A row missing either bound breaks the band rather than being dropped: fill="tonexty" spans
    # the null, so the shading stops at a gap instead of bridging it.
    band = df.with_columns(
        pl.when(pl.col("temp_min").is_not_null() & pl.col("temp_max").is_not_null())
        .then(pl.col(c))
        .otherwise(None)
        .alias(c)
        for c in ("temp_min", "temp_max")
    )
    dates = band["DATE"].to_list()

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=band["temp_max"].to_list(),
            name="Max",
            line={"color": _RED, "width": 0.75},
            mode="lines",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=band["temp_min"].to_list(),
            name="Min",
            line={"color": _BLUE, "width": 0.75},
            fill="tonexty",
            fillcolor=_BAND_FILL,
            mode="lines",
        )
    )

    hot_days = _hot_day_dates(band, definition)
    if hot_days:
        fig.update_layout(shapes=_band_shapes(_date_runs(hot_days), _HOT_DAY_FILL))
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker={"color": _HOT_DAY_FILL, "size": 10, "symbol": "square"},
                name=f"Hot day ({definition.label})",
            )
        )

    _apply_standard_layout(fig, f"Temperature — {station_name}{granularity.title_suffix}", "°C")
    fig.update_yaxes(dtick=5, **_YAXIS_GRID)
    return fig


def precipitation_figure(
    df: pl.DataFrame,
    station_name: str,
    granularity: Granularity = Granularity.DAY,
) -> go.Figure:
    """Bar chart: precipitation — a daily depth, or the accumulated total of a week or month."""
    dates, precip = _series(df, "precipitation")
    unit = granularity.per_unit("mm")
    fig = go.Figure(
        go.Bar(
            x=dates,
            y=precip,
            name=f"Precipitation ({unit})",
            marker_color=_RAIN,
            marker_line_width=0,
            hovertemplate="%{y:.1f} " + unit + "<extra></extra>",
        )
    )
    _apply_standard_layout(
        fig,
        f"Precipitation — {station_name}{granularity.title_suffix}",
        unit,
        legend=False,
        bargap=0,
    )
    fig.update_yaxes(**_YAXIS_GRID)
    return fig


def wind_figure(
    df: pl.DataFrame,
    station_name: str,
    granularity: Granularity = Granularity.DAY,
) -> go.Figure:
    """Line chart: FFM mean wind speed and FXY max gust."""
    fig = go.Figure()

    has_mean = df["wind_mean"].is_not_null().any() if "wind_mean" in df.columns else False
    has_gust = df["wind_gust"].is_not_null().any() if "wind_gust" in df.columns else False

    if has_gust:
        dates_gust, gust = _series(df, "wind_gust")
        fig.add_trace(
            go.Scatter(
                x=dates_gust,
                y=gust,
                name="Max gust",
                line={"color": _RED, "width": 0.75},
                mode="lines",
            )
        )
    if has_mean:
        dates_mean, mean = _series(df, "wind_mean")
        fig.add_trace(
            go.Scatter(
                x=dates_mean,
                y=mean,
                name="Mean",
                line={"color": _GREEN, "width": 1},
                mode="lines",
            )
        )

    no_data = not has_mean and not has_gust
    _apply_standard_layout(
        fig,
        f"Wind — {station_name}{granularity.title_suffix}" + (" (no data for this station)" if no_data else ""),
        "m/s",
    )
    fig.update_yaxes(dtick=5, **_YAXIS_GRID)
    return fig


def _consecutive_runs(values: list[int]) -> list[tuple[int, int]]:
    """Collapse a sorted list of ints into (first, last) runs of consecutive values."""
    runs: list[tuple[int, int]] = []
    for v in values:
        if runs and v == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], v)
        else:
            runs.append((v, v))
    return runs


@dataclass(frozen=True)
class CountedSeries:
    """One counted series of the yearly tab: the column it reads, its legend text, its colour.

    Named rather than a 3-tuple because all three are strings — a positional swap of label and
    colour is a bug nothing else would catch.
    """

    column: str
    label: str
    color: str


def yearly_series(definition: HotDayDefinition) -> list[CountedSeries]:
    """The three counted series of the yearly tab."""
    return [
        CountedSeries("hot_days", f"Hot days ({definition.label})", _RED),
        CountedSeries("tropical_nights", f"Tropical nights (Tmin≥{TROPICAL_NIGHT_TMIN:.0f}°C)", _ORANGE),
        CountedSeries("cold_days", f"Frost days (Tmin<{FROST_TMIN:.0f}°C)", _BLUE),
    ]


def hot_cold_yearly_figure(
    df: pl.DataFrame,
    station_name: str,
    *,
    definition: HotDayDefinition = DEFAULT_HOT_DAY,
    show_trend: bool = False,
    current_year: int | None = None,
    min_coverage: float = MIN_YEAR_COVERAGE,
) -> go.Figure:
    """Line chart: yearly counts of hot days, tropical nights and frost days.

    Three kinds of year are drawn differently, because they mean different things:
    fully-observed years carry the solid lines and the regression; years below `min_coverage`
    are left out of both — a station that was offline did not have a cool year — and marked with
    a grey band; the current year, if included, is a dotted provisional point.
    """
    series = yearly_series(definition)
    agg = yearly_hot_cold(df, definition)
    years: list[int] = agg["year"].to_list()
    coverage: list[float] = agg["coverage"].to_list()

    is_provisional = [y == current_year for y in years]
    is_measured = [c >= min_coverage and not p for c, p in zip(coverage, is_provisional, strict=True)]
    measured_years = [y for y, m in zip(years, is_measured, strict=True) if m]
    counts: dict[str, list[int]] = {s.column: agg[s.column].to_list() for s in series}
    measured_counts: dict[str, list[int]] = {
        col: [v for v, m in zip(values, is_measured, strict=True) if m] for col, values in counts.items()
    }

    fig = go.Figure()
    for s in series:
        fig.add_trace(
            go.Scatter(
                x=years,
                y=[v if m else None for v, m in zip(counts[s.column], is_measured, strict=True)],
                name=s.label,
                line={"color": s.color, "width": 1.5},
                mode="lines+markers",
                marker={"size": 4},
                connectgaps=False,
            )
        )

    sparse_years = [y for y, c, p in zip(years, coverage, is_provisional, strict=True) if c < min_coverage and not p]
    if sparse_years:
        runs = [(first - 0.5, last + 0.5) for first, last in _consecutive_runs(sparse_years)]
        fig.update_layout(shapes=_band_shapes(runs, _PARTIAL_FILL))
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker={"color": _PARTIAL_FILL, "size": 10, "symbol": "square"},
                name=f"Record under {min_coverage:.0%} of days — excluded",
            )
        )

    provisional_index = next((i for i, p in enumerate(is_provisional) if p), None)
    if provisional_index is not None:
        for s in series:
            _add_provisional_point(
                fig,
                measured_years,
                measured_counts[s.column],
                years[provisional_index],
                counts[s.column][provisional_index],
                s.color,
            )

    if show_trend:
        years_f = [float(y) for y in measured_years]
        for s in series:
            trend: LinearTrend | None = linear_trend(years_f, measured_counts[s.column])
            if trend is None:
                continue
            sign = "+" if trend.slope >= 0 else "−"
            fig.add_trace(
                go.Scatter(
                    x=measured_years,
                    y=[trend.slope * y + trend.intercept for y in years_f],
                    name=f"{s.label} — trend",
                    line={"color": s.color, "width": 1, "dash": "dash"},
                    mode="lines",
                    showlegend=False,
                    hovertemplate=(
                        f"{s.label} trend<br>slope: {sign}{abs(trend.slope):.2f} days/yr"
                        f"<br>R²: {trend.r_squared:.2f}<extra></extra>"
                    ),
                )
            )

    _apply_standard_layout(fig, f"Yearly extreme days — {station_name}", "Days")
    fig.update_yaxes(dtick=5, **_YAXIS_GRID)
    fig.update_xaxes(dtick=5, **_XAXIS_GRID)
    return fig


def _add_provisional_point(
    fig: go.Figure,
    measured_years: list[int],
    measured_counts: list[int],
    year: int,
    value: int,
    color: str,
) -> None:
    """Dotted connector to a hollow marker: this year is still being recorded."""
    if measured_years:
        x, y = [measured_years[-1], year], [measured_counts[-1], value]
        mode = "lines+markers"
        # The anchor is already described by the solid trace; only the endpoint says "(partial)".
        hover: str | list[str] = ["<extra></extra>", "(partial)<extra></extra>"]
    else:
        x, y = [year], [value]
        mode = "markers"
        hover = "(partial)<extra></extra>"
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode=mode,
            line={"dash": "dot", "color": color, "width": 1.5},
            marker={"symbol": "circle-open", "size": 6},
            showlegend=False,
            hovertemplate=hover,
        )
    )


def monthly_avg_temp_figure(df: pl.DataFrame, station_name: str) -> go.Figure:
    """Line chart: average temp_min and temp_max by month of year, with ±2σ bands."""
    monthly = _MONTH_SPINE.join(
        df.with_columns(pl.col("DATE").dt.month().alias("month"))
        .group_by("month")
        .agg(
            [
                pl.col("temp_min").mean().alias("avg_temp_min"),
                pl.col("temp_max").mean().alias("avg_temp_max"),
                pl.col("temp_min").std().alias("std_temp_min"),
                pl.col("temp_max").std().alias("std_temp_max"),
            ]
        ),
        on="month",
        how="left",
    )
    month_labels = [_MONTH_LABELS[m - 1] for m in monthly["month"].to_list()]
    avg_tmax = monthly["avg_temp_max"].to_list()
    avg_tmin = monthly["avg_temp_min"].to_list()
    std_tmax = monthly["std_temp_max"].to_list()
    std_tmin = monthly["std_temp_min"].to_list()

    fig = go.Figure()
    _add_sigma_band(fig, month_labels, avg_tmax, std_tmax, _TMAX_SIGMA_FILL)
    _add_sigma_band(fig, month_labels, avg_tmin, std_tmin, _TMIN_SIGMA_FILL)
    fig.add_trace(
        go.Scatter(
            x=month_labels,
            y=avg_tmax,
            name="Avg Tmax",
            line={"color": _RED, "width": 1.5, "shape": "spline"},
            mode="lines+markers",
            marker={"size": 5},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=month_labels,
            y=avg_tmin,
            name="Avg Tmin",
            line={"color": _BLUE, "width": 1.5, "shape": "spline"},
            mode="lines+markers",
            marker={"size": 5},
        )
    )
    _apply_standard_layout(fig, f"Monthly average temperatures — {station_name}", "°C")
    fig.update_yaxes(dtick=2, **_YAXIS_GRID)
    return fig


def monthly_avg_temp_by_decade_figure(
    df: pl.DataFrame,
    station_name: str,
    selected: list[int] | None = None,
) -> go.Figure:
    """Line chart: average temp_min (dashed) and temp_max (solid) by month, one colour per decade.

    `selected` restricts which decades are drawn; Tmin traces start collapsed into the legend,
    since sixteen gradient-coloured lines are a legend, not a chart.
    """
    decade_monthly = (
        df.with_columns(
            [
                pl.col("DATE").dt.month().alias("month"),
                ((pl.col("DATE").dt.year() // 10) * 10).alias("decade"),
            ]
        )
        .group_by(["decade", "month"])
        .agg(
            [
                pl.col("temp_min").mean().alias("avg_temp_min"),
                pl.col("temp_max").mean().alias("avg_temp_max"),
            ]
        )
        .sort(["decade", "month"])
    )

    decades = sorted(decade_monthly["decade"].unique().to_list())
    if selected is not None:
        decades = [d for d in decades if d in set(selected)]
    n = len(decades)
    fig = go.Figure()

    for i, decade in enumerate(decades):
        t = i / (n - 1) if n > 1 else 0.0
        tmax_color = _gradient_color(t, _TMAX_GRADIENT_START, _TMAX_GRADIENT_END)
        tmin_color = _gradient_color(t, _TMIN_GRADIENT_START, _TMIN_GRADIENT_END)
        sub = _MONTH_SPINE.join(
            decade_monthly.filter(pl.col("decade") == decade).select(["month", "avg_temp_min", "avg_temp_max"]),
            on="month",
            how="left",
        )
        month_labels = [_MONTH_LABELS[m - 1] for m in sub["month"].to_list()]
        label = f"{decade}s"
        _add_temp_trace(fig, month_labels, sub["avg_temp_max"].to_list(), f"{label} Tmax", tmax_color)
        _add_temp_trace(
            fig,
            month_labels,
            sub["avg_temp_min"].to_list(),
            f"{label} Tmin",
            tmin_color,
            dash="dash",
            visible="legendonly",
        )

    _apply_standard_layout(fig, f"Monthly average temperatures by decade — {station_name}", "°C")
    fig.update_yaxes(dtick=2, **_YAXIS_GRID)
    return fig


_DENSITY_FILL_A = "rgba(76, 155, 232, 0.22)"
_DENSITY_FILL_B = "rgba(232, 92, 76, 0.22)"


def _add_density_trace(fig: go.Figure, density: Density, name: str, color: str, fill: str) -> None:
    fig.add_trace(
        go.Scatter(
            x=density.x,
            y=density.y,
            name=name,
            mode="lines",
            line={"color": color, "width": 1.75, "shape": "spline"},
            fill="tozeroy",
            fillcolor=fill,
            hovertemplate="%{x:.1f} °C<br>density %{y:.3f}<extra>" + name + "</extra>",
        )
    )


def density_comparison_figure(
    series_a: pl.Series,
    series_b: pl.Series,
    label_a: str,
    label_b: str,
    title: str,
) -> go.Figure:
    """Overlay the smoothed probability density of two temperature samples."""
    density_a = gaussian_kde(series_a)
    density_b = gaussian_kde(series_b)
    if density_a is None or density_b is None:
        return empty_figure("Not enough data in one of the two periods")

    fig = go.Figure()
    _add_density_trace(fig, density_a, label_a, _BLUE, _DENSITY_FILL_A)
    _add_density_trace(fig, density_b, label_b, _RED, _DENSITY_FILL_B)

    for series, color in ((series_a, _BLUE), (series_b, _RED)):
        mean = series.drop_nulls().mean()
        if isinstance(mean, (int, float)):
            fig.add_vline(x=float(mean), line={"color": color, "width": 1.5, "dash": "dash"})

    _apply_standard_layout(fig, title, "Probability density", hovermode="x")
    fig.update_xaxes(title_text="°C", dtick=5, **_XAXIS_GRID)
    fig.update_yaxes(rangemode="tozero", **_YAXIS_GRID)
    return fig


def empty_figure(message: str = "No data") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        annotations=[{"text": message, "showarrow": False, "font": {"size": 16, "color": _GREY}}],
        xaxis={"visible": False},
        yaxis={"visible": False},
        margin={"t": 50, "b": 30},
    )
    return fig
