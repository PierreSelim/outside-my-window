from __future__ import annotations

from datetime import timedelta
from typing import Any

import plotly.graph_objects as go
import polars as pl

from src.data_loader import HOT_DAY_TMAX, HOT_DAY_TMIN, Granularity
from src.transforms import DEFAULT_HOT_DAY, HotDayDefinition, LinearTrend, linear_trend, yearly_hot_cold

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

_BLUE = "#4C9BE8"
_RED = "#E85C4C"
_GREEN = "#5CB85C"
_GREY = "#AAAAAA"
_BAND_FILL = "rgba(76, 155, 232, 0.15)"
_HOT_DAY_FILL = "rgba(230, 80, 0, 0.15)"
_TMAX_SIGMA_FILL = "rgba(232, 92, 76, 0.15)"
_TMIN_SIGMA_FILL = "rgba(76, 155, 232, 0.15)"

# ---------------------------------------------------------------------------
# Grid / axis style
# ---------------------------------------------------------------------------

_GRID_BASE: dict[str, Any] = {"showgrid": True, "gridcolor": "#E2E8F0", "gridwidth": 1}
_YAXIS_GRID: dict[str, Any] = {
    **_GRID_BASE,
    "zeroline": True,
    "zerolinecolor": "#CBD5E1",
    "zerolinewidth": 1.5,
    "minor": {"showgrid": True, "gridcolor": "rgba(226,232,240,0.5)", "gridwidth": 0.5},
}
_XAXIS_GRID: dict[str, Any] = {**_GRID_BASE}

# Decade chart gradients: oldest → newest
_TMAX_GRADIENT_START: tuple[int, int, int] = (255, 140, 0)  # orange
_TMAX_GRADIENT_END: tuple[int, int, int] = (160, 0, 0)  # deep red
_TMIN_GRADIENT_START: tuple[int, int, int] = (173, 216, 230)  # light blue
_TMIN_GRADIENT_END: tuple[int, int, int] = (0, 0, 139)  # dark blue

_MONTH_LABELS: list[str] = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
_MONTH_SPINE: pl.DataFrame = pl.DataFrame({"month": list(range(1, 13))}, schema={"month": pl.UInt32})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


_FONT_FAMILY = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"
_PLOT_BG = "rgba(248, 250, 252, 1)"


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
            "font": {"size": 13, "weight": 600, "color": "#0F172A"},
            "x": 0,
            "xanchor": "left",
            "pad": {"l": 4},
        },
        "yaxis_title": yaxis_title,
        "hovermode": "x unified",
        "margin": {"t": 48, "b": 24, "l": 56, "r": 16},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": _PLOT_BG,
        "font": {"family": _FONT_FAMILY, "size": 12, "color": "#374151"},
        "hoverlabel": {"bgcolor": "#1E293B", "font_color": "#F8FAFC", "bordercolor": "#1E293B", "font_size": 12},
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


def _hot_day_dates(df: pl.DataFrame) -> list[Any]:
    """Return dates where temp_min ≥ HOT_DAY_TMIN and temp_max ≥ HOT_DAY_TMAX."""
    return df.filter((pl.col("temp_min") >= HOT_DAY_TMIN) & (pl.col("temp_max") >= HOT_DAY_TMAX))["DATE"].to_list()


def _series(df: pl.DataFrame, col: str) -> tuple[list[Any], list[Any]]:
    """Return (dates, values) lists for a column, dropping nulls."""
    sub = df.filter(df[col].is_not_null())
    return sub["DATE"].to_list(), sub[col].to_list()


def _add_temp_trace(
    fig: go.Figure,
    month_labels: list[Any],
    y: list[Any],
    name: str,
    color: str,
    dash: str = "",
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


# ---------------------------------------------------------------------------
# Individual figures
# ---------------------------------------------------------------------------


def temperature_figure(
    df: pl.DataFrame,
    station_name: str,
    granularity: Granularity = Granularity.DAY,
) -> go.Figure:
    """Line chart: TN/TX shaded band."""
    fig = go.Figure()

    # Plotly filled-area bands require two traces: upper (temp_max) drawn first;
    # the lower trace uses fill="tonexty" to shade between them.
    df_band = df.filter(pl.col("temp_min").is_not_null() & pl.col("temp_max").is_not_null())
    dates = df_band["DATE"].to_list()

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=df_band["temp_max"].to_list(),
            name="Max",
            line={"color": _RED, "width": 0.75},
            mode="lines",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=df_band["temp_min"].to_list(),
            name="Min",
            line={"color": _BLUE, "width": 0.75},
            fill="tonexty",
            fillcolor=_BAND_FILL,
            mode="lines",
        )
    )

    hot_days = _hot_day_dates(df_band)
    for d in hot_days:
        fig.add_vrect(x0=d, x1=d + timedelta(days=1), fillcolor=_HOT_DAY_FILL, layer="below", line_width=0)
    if hot_days:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker={"color": _HOT_DAY_FILL, "size": 10, "symbol": "square"},
                name=f"Hot day (Tmin≥{HOT_DAY_TMIN:.0f}°C, Tmax≥{HOT_DAY_TMAX:.0f}°C)",
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
    """Bar chart: daily precipitation."""
    dates, precip = _series(df, "precipitation")
    fig = go.Figure(
        go.Bar(
            x=dates,
            y=precip,
            name="Precipitation (mm)",
            marker_color=_BLUE,
            marker_line_width=0,
        )
    )
    _apply_standard_layout(
        fig,
        f"Precipitation — {station_name}{granularity.title_suffix}",
        "mm",
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


def hot_cold_yearly_figure(
    df: pl.DataFrame,
    station_name: str,
    *,
    definition: HotDayDefinition = DEFAULT_HOT_DAY,
    show_trend: bool = False,
    current_year: int | None = None,
) -> go.Figure:
    """Line chart: yearly count of hot days and days with temp_min < 0.

    When current_year is provided and a row for that year exists in the data, it is rendered
    as a provisional point connected by a dotted line to the last complete year.
    """
    agg = yearly_hot_cold(df, definition)
    years = agg["year"].to_list()
    hot = agg["hot_days"].to_list()
    cold = agg["cold_days"].to_list()

    if current_year is None:
        complete_years = years
        complete_hot = hot
        complete_cold = cold
        provisional = None
    else:
        c_idx = [i for i, y in enumerate(years) if y < current_year]
        p_idx = next((i for i, y in enumerate(years) if y == current_year), None)
        complete_years = [years[i] for i in c_idx]
        complete_hot = [hot[i] for i in c_idx]
        complete_cold = [cold[i] for i in c_idx]
        provisional = (years[p_idx], hot[p_idx], cold[p_idx]) if p_idx is not None else None

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=complete_years,
            y=complete_hot,
            name=f"Hot days ({definition.label})",
            line={"color": _RED, "width": 1.5, "shape": "spline"},
            mode="lines+markers",
            marker={"size": 4},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=complete_years,
            y=complete_cold,
            name="Cold days (Tmin < 0°C)",
            line={"color": _BLUE, "width": 1.5, "shape": "spline"},
            mode="lines+markers",
            marker={"size": 4},
        )
    )

    if provisional is not None:
        prov_year, prov_hot, prov_cold = provisional
        for series_complete, prov_val, color in [
            (complete_hot, prov_hot, _RED),
            (complete_cold, prov_cold, _BLUE),
        ]:
            if complete_years:
                x_dot = [complete_years[-1], prov_year]
                y_dot = [series_complete[-1], prov_val]
                dot_mode = "lines+markers"
                # Suppress hover on the anchor (already shown by the solid trace);
                # only the provisional endpoint carries the "(partial)" label.
                hover: str | list[str] = ["<extra></extra>", "(partial)<extra></extra>"]
            else:
                x_dot = [prov_year]
                y_dot = [prov_val]
                dot_mode = "markers"
                hover = "(partial)<extra></extra>"
            fig.add_trace(
                go.Scatter(
                    x=x_dot,
                    y=y_dot,
                    mode=dot_mode,
                    line={"dash": "dot", "color": color, "width": 1.5},
                    marker={"symbol": "circle-open", "size": 6},
                    showlegend=False,
                    hovertemplate=hover,
                )
            )

    complete_years_f = [float(y) for y in complete_years]
    if show_trend and len(complete_years_f) >= 2:
        for label, counts, color in [
            ("Hot days trend", complete_hot, _RED),
            ("Cold days trend", complete_cold, _BLUE),
        ]:
            trend: LinearTrend = linear_trend(complete_years_f, counts)
            trend_y = [trend.slope * y + trend.intercept for y in complete_years_f]
            sign = "+" if trend.slope >= 0 else "−"
            hover = (
                f"{label}<br>slope: {sign}{abs(trend.slope):.2f} days/yr<br>R²: {trend.r_squared:.2f}<extra></extra>"  # noqa: E501
            )
            fig.add_trace(
                go.Scatter(
                    x=complete_years,
                    y=trend_y,
                    name=label,
                    line={"color": color, "width": 1, "dash": "dash"},
                    mode="lines",
                    hovertemplate=hover,
                )
            )

    _apply_standard_layout(fig, f"Yearly extreme days — {station_name}", "Days")
    fig.update_yaxes(dtick=5, **_YAXIS_GRID)
    fig.update_xaxes(dtick=5, **_XAXIS_GRID)
    return fig


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


def monthly_avg_temp_by_decade_figure(df: pl.DataFrame, station_name: str) -> go.Figure:
    """Line chart: average temp_min (dashed) and temp_max (solid) by month, one colour per decade."""
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
        _add_temp_trace(fig, month_labels, sub["avg_temp_min"].to_list(), f"{label} Tmin", tmin_color, dash="dash")

    _apply_standard_layout(fig, f"Monthly average temperatures by decade — {station_name}", "°C")
    fig.update_yaxes(dtick=2, **_YAXIS_GRID)
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
