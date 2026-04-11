from __future__ import annotations

from datetime import timedelta

import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

_BLUE = "#4C9BE8"
_RED = "#E85C4C"
_GREEN = "#5CB85C"
_GREY = "#AAAAAA"
_BAND_FILL = "rgba(76, 155, 232, 0.15)"

# Hot day: temp_min ≥ 20 °C AND temp_max ≥ 35 °C
_HOT_DAY_TMIN: float = 20.0
_HOT_DAY_TMAX: float = 35.0
_HOT_DAY_FILL: str = "rgba(230, 80, 0, 0.15)"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
    dates_min, temp_min = df_band["DATE"].to_list(), df_band["temp_min"].to_list()
    dates_max, temp_max = df_band["DATE"].to_list(), df_band["temp_max"].to_list()

    # temp_max upper bound of the band
    fig.add_trace(
        go.Scatter(
            x=dates_max,
            y=temp_max,
            name="Max",
            line=dict(color=_RED, width=1),
            mode="lines",
        )
    )

    # temp_min lower bound — fill to temp_max creates the band
    fig.add_trace(
        go.Scatter(
            x=dates_min,
            y=temp_min,
            name="Min",
            line=dict(color=_BLUE, width=1),
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
                marker=dict(color=_HOT_DAY_FILL, size=10, symbol="square"),
                name=f"Hot day (Tmin≥{_HOT_DAY_TMIN:.0f}°C, Tmax≥{_HOT_DAY_TMAX:.0f}°C)",
            )
        )

    fig.update_layout(
        title=f"Temperature — {station_name}{granularity_label}",
        yaxis_title="°C",
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
        hovermode="x unified",
        margin=dict(t=50, b=30),
    )
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
        margin=dict(t=50, b=30),
    )
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
                line=dict(color=_RED, width=1),
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
                line=dict(color=_GREEN, width=1.5),
                mode="lines",
            )
        )

    no_data = not has_mean and not has_gust
    fig.update_layout(
        title=f"Wind — {station_name}{granularity_label}" + (" (no data for this station)" if no_data else ""),
        yaxis_title="m/s",
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
        hovermode="x unified",
        margin=dict(t=50, b=30),
    )
    return fig


# ---------------------------------------------------------------------------
# Empty placeholder
# ---------------------------------------------------------------------------


def empty_figure(message: str = "No data") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        annotations=[dict(text=message, showarrow=False, font=dict(size=16, color=_GREY))],
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(t=50, b=30),
    )
    return fig
