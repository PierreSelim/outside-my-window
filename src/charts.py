from __future__ import annotations

from datetime import timedelta

import polars as pl
import plotly.graph_objects as go


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

_BLUE = "#4C9BE8"
_RED = "#E85C4C"
_GREEN = "#5CB85C"
_GREY = "#AAAAAA"
_BAND_FILL = "rgba(76, 155, 232, 0.15)"

# ---------------------------------------------------------------------------
# Grid / axis style helpers
# ---------------------------------------------------------------------------

_YAXIS_GRID: dict = {
    "showgrid": True,
    "gridcolor": "rgba(180,180,180,0.4)",
    "gridwidth": 0.5,
    "zeroline": True,
    "zerolinecolor": "rgba(150,150,150,0.6)",
    "zerolinewidth": 1,
    "minor": {"showgrid": True, "gridcolor": "rgba(210,210,210,0.25)", "gridwidth": 0.5},
}
_XAXIS_GRID: dict = {
    "showgrid": True,
    "gridcolor": "rgba(180,180,180,0.4)",
    "gridwidth": 0.5,
}

# Hot day: temp_min ≥ 20 °C AND temp_max ≥ 35 °C
_HOT_DAY_TMIN: float = 20.0
_HOT_DAY_TMAX: float = 35.0
_HOT_DAY_FILL: str = "rgba(230, 80, 0, 0.15)"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _linear_trend(x: list[float | int], y: list[float | int | None]) -> tuple[float, float, float]:
    """Return (slope, intercept, r_squared) for a simple OLS linear regression.

    Filters out pairs where y is None. Centers x before computing to avoid
    catastrophic cancellation with large x-values (e.g. years). Returns
    (0, mean_y, 0) when there are fewer than 2 valid points or zero x-variance.
    R² is 0 when SS_tot is zero (constant y).
    """
    pairs = [(float(xi), float(yi)) for xi, yi in zip(x, y) if yi is not None]
    n = len(pairs)
    if n < 2:
        return 0.0, pairs[0][1] if n == 1 else 0.0, 0.0
    mean_x = sum(p[0] for p in pairs) / n
    xc = [p[0] - mean_x for p in pairs]
    yv = [p[1] for p in pairs]
    sum_xc2 = sum(v * v for v in xc)
    if sum_xc2 == 0:
        return 0.0, sum(yv) / n, 0.0
    slope = sum(xci * yi for xci, yi in zip(xc, yv)) / sum_xc2
    intercept = sum(yv) / n - slope * mean_x
    mean_y = sum(yv) / n
    ss_tot = sum((yi - mean_y) ** 2 for yi in yv)
    r_squared = 0.0 if ss_tot == 0 else 1.0 - sum((yi - (slope * xi + intercept)) ** 2 for xi, yi in zip(x, yv)) / ss_tot
    return slope, intercept, r_squared


def _hot_day_dates(df: pl.DataFrame) -> list:
    """Return dates where temp_min ≥ 20 °C and temp_max ≥ 35 °C."""
    return (
        df.filter(
            pl.col("temp_min").is_not_null()
            & pl.col("temp_max").is_not_null()
            & (pl.col("temp_min") >= _HOT_DAY_TMIN)
            & (pl.col("temp_max") >= _HOT_DAY_TMAX)
        )["DATE"].to_list()
    )


def _series(df: pl.DataFrame, col: str) -> tuple[list, list]:
    """Return (dates, values) lists for a column, dropping nulls."""
    mask = df[col].is_not_null()
    sub = df.filter(mask)
    return sub["DATE"].to_list(), sub[col].to_list()


# ---------------------------------------------------------------------------
# Individual figures
# ---------------------------------------------------------------------------


def temperature_figure(df: pl.DataFrame, station_name: str, granularity_label: str = "") -> go.Figure:
    """Line chart: TN/TX shaded band."""
    fig = go.Figure()

    # Filter jointly so both band traces cover identical dates
    df_band = df.filter(pl.col("temp_min").is_not_null() & pl.col("temp_max").is_not_null())
    dates = df_band["DATE"].to_list()
    temp_min = df_band["temp_min"].to_list()
    temp_max = df_band["temp_max"].to_list()

    # temp_max upper bound of the band
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=temp_max,
            name="Max",
            line={"color": _RED, "width": 0.75},
            mode="lines",
        )
    )

    # temp_min lower bound — fill to temp_max creates the band
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=temp_min,
            name="Min",
            line={"color": _BLUE, "width": 0.75},
            fill="tonexty",
            fillcolor=_BAND_FILL,
            mode="lines",
        )
    )

    hot_days = _hot_day_dates(df)
    for d in hot_days:
        fig.add_vrect(x0=d, x1=d + timedelta(days=1), fillcolor=_HOT_DAY_FILL, layer="below", line_width=0)
    if hot_days:
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None],
                mode="markers",
                marker={"color": _HOT_DAY_FILL, "size": 10, "symbol": "square"},
                name=f"Hot day (Tmin≥{_HOT_DAY_TMIN:.0f}°C, Tmax≥{_HOT_DAY_TMAX:.0f}°C)",
            )
        )

    fig.update_layout(
        title=f"Temperature — {station_name}{granularity_label}",
        yaxis_title="°C",
        legend={"orientation": "h", "y": 1.02, "x": 1, "xanchor": "right"},
        hovermode="x unified",
        margin={"t": 50, "b": 30},
    )
    fig.update_yaxes(dtick=5, **_YAXIS_GRID)
    return fig


def precipitation_figure(df: pl.DataFrame, station_name: str, granularity_label: str = "") -> go.Figure:
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
    fig.update_layout(
        title=f"Precipitation — {station_name}{granularity_label}",
        yaxis_title="mm",
        bargap=0,
        hovermode="x unified",
        margin={"t": 50, "b": 30},
    )
    fig.update_yaxes(**_YAXIS_GRID)
    return fig


def wind_figure(df: pl.DataFrame, station_name: str, granularity_label: str = "") -> go.Figure:
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
    fig.update_layout(
        title=f"Wind — {station_name}{granularity_label}" + (" (no data for this station)" if no_data else ""),
        yaxis_title="m/s",
        legend={"orientation": "h", "y": 1.02, "x": 1, "xanchor": "right"},
        hovermode="x unified",
        margin={"t": 50, "b": 30},
    )
    fig.update_yaxes(dtick=5, **_YAXIS_GRID)
    return fig


_MONTH_LABELS: list[str] = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
# Full 12-month spine used to guarantee Jan–Dec appear even when data starts mid-year
_MONTH_SPINE: pl.DataFrame = pl.DataFrame({"month": list(range(1, 13))}, schema={"month": pl.UInt32})

# Decade chart gradients: oldest → newest
_TMAX_GRADIENT_START: tuple[int, int, int] = (255, 140, 0)   # orange
_TMAX_GRADIENT_END: tuple[int, int, int] = (160, 0, 0)        # deep red
_TMIN_GRADIENT_START: tuple[int, int, int] = (173, 216, 230)  # light blue
_TMIN_GRADIENT_END: tuple[int, int, int] = (0, 0, 139)        # dark blue


def _gradient_color(t: float, start: tuple[int, int, int], end: tuple[int, int, int]) -> str:
    """Linearly interpolate between two RGB colours. t=0 → start, t=1 → end."""
    r = int(start[0] + t * (end[0] - start[0]))
    g = int(start[1] + t * (end[1] - start[1]))
    b = int(start[2] + t * (end[2] - start[2]))
    return f"rgb({r},{g},{b})"


def hot_cold_yearly_figure(df: pl.DataFrame, station_name: str, show_trend: bool = False) -> go.Figure:
    """Line chart: yearly count of hot days and days with temp_min < 0."""
    yearly = (
        df.with_columns(pl.col("DATE").dt.year().alias("year"))
        .group_by("year")
        .agg(
            [
                (
                    pl.col("temp_min").is_not_null()
                    & pl.col("temp_max").is_not_null()
                    & (pl.col("temp_min") >= _HOT_DAY_TMIN)
                    & (pl.col("temp_max") >= _HOT_DAY_TMAX)
                )
                .sum()
                .alias("hot_days"),
                (pl.col("temp_min").is_not_null() & (pl.col("temp_min") < 0)).sum().alias("cold_days"),
            ]
        )
        .sort("year")
    )

    years = yearly["year"].to_list()
    hot_days = yearly["hot_days"].to_list()
    cold_days = yearly["cold_days"].to_list()
    years_f = [float(y) for y in years]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=years,
            y=hot_days,
            name=f"Hot days (Tmin≥{_HOT_DAY_TMIN:.0f}°C & Tmax≥{_HOT_DAY_TMAX:.0f}°C)",
            line={"color": _RED, "width": 1.5, "shape": "spline"},
            mode="lines+markers",
            marker={"size": 4},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=years,
            y=cold_days,
            name="Cold days (Tmin < 0°C)",
            line={"color": _BLUE, "width": 1.5, "shape": "spline"},
            mode="lines+markers",
            marker={"size": 4},
        )
    )
    if show_trend and len(years_f) >= 2:
        for label, counts, color in [
            ("Hot days trend", hot_days, _RED),
            ("Cold days trend", cold_days, _BLUE),
        ]:
            slope, intercept, r_squared = _linear_trend(years_f, counts)
            trend_y = [slope * y + intercept for y in years_f]
            sign = "+" if slope >= 0 else "−"
            hover = f"{label}<br>slope: {sign}{abs(slope):.2f} days/yr<br>R²: {r_squared:.2f}<extra></extra>"
            fig.add_trace(
                go.Scatter(
                    x=years,
                    y=trend_y,
                    name=label,
                    line={"color": color, "width": 1, "dash": "dash"},
                    mode="lines",
                    hovertemplate=hover,
                )
            )
    fig.update_layout(
        title=f"Yearly extreme days — {station_name}",
        yaxis_title="Days",
        legend={"orientation": "h", "y": 1.02, "x": 1, "xanchor": "right"},
        hovermode="x unified",
        margin={"t": 50, "b": 30},
    )
    fig.update_yaxes(dtick=5, **_YAXIS_GRID)
    fig.update_xaxes(dtick=5, **_XAXIS_GRID)
    return fig


def monthly_avg_temp_figure(df: pl.DataFrame, station_name: str) -> go.Figure:
    """Line chart: average temp_min and temp_max by month of year."""
    monthly = _MONTH_SPINE.join(
        df.with_columns(pl.col("DATE").dt.month().alias("month"))
        .group_by("month")
        .agg(
            [
                pl.col("temp_min").mean().alias("avg_temp_min"),
                pl.col("temp_max").mean().alias("avg_temp_max"),
            ]
        ),
        on="month",
        how="left",
    )

    month_labels = [_MONTH_LABELS[m - 1] for m in monthly["month"].to_list()]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=month_labels,
            y=monthly["avg_temp_max"].to_list(),
            name="Avg Tmax",
            line={"color": _RED, "width": 1.5, "shape": "spline"},
            mode="lines+markers",
            marker={"size": 5},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=month_labels,
            y=monthly["avg_temp_min"].to_list(),
            name="Avg Tmin",
            line={"color": _BLUE, "width": 1.5, "shape": "spline"},
            mode="lines+markers",
            marker={"size": 5},
        )
    )
    fig.update_layout(
        title=f"Monthly average temperatures — {station_name}",
        yaxis_title="°C",
        legend={"orientation": "h", "y": 1.02, "x": 1, "xanchor": "right"},
        hovermode="x unified",
        margin={"t": 50, "b": 30},
    )
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
        fig.add_trace(
            go.Scatter(
                x=month_labels,
                y=sub["avg_temp_max"].to_list(),
                name=f"{label} Tmax",
                line={"color": tmax_color, "width": 1.5, "shape": "spline"},
                mode="lines+markers",
                marker={"size": 4},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=month_labels,
                y=sub["avg_temp_min"].to_list(),
                name=f"{label} Tmin",
                line={"color": tmin_color, "width": 1.5, "dash": "dash", "shape": "spline"},
                mode="lines+markers",
                marker={"size": 4},
            )
        )

    fig.update_layout(
        title=f"Monthly average temperatures by decade — {station_name}",
        yaxis_title="°C",
        legend={"orientation": "h", "y": 1.02, "x": 1, "xanchor": "right"},
        hovermode="x unified",
        margin={"t": 50, "b": 30},
    )
    fig.update_yaxes(dtick=2, **_YAXIS_GRID)
    return fig


# ---------------------------------------------------------------------------
# Empty placeholder
# ---------------------------------------------------------------------------


def empty_figure(message: str = "No data") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        annotations=[{"text": message, "showarrow": False, "font": {"size": 16, "color": _GREY}}],
        xaxis={"visible": False},
        yaxis={"visible": False},
        margin={"t": 50, "b": 30},
    )
    return fig
