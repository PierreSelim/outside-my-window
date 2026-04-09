from __future__ import annotations

import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

_BLUE = "#4C9BE8"
_RED = "#E85C4C"
_ORANGE = "#F5A623"
_GREEN = "#5CB85C"
_GREY = "#AAAAAA"
_BAND_FILL = "rgba(76, 155, 232, 0.15)"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _series(df: pl.DataFrame, col: str) -> tuple[list, list]:
    """Return (dates, values) lists for a column, dropping nulls."""
    mask = df[col].is_not_null()
    sub = df.filter(mask)
    return sub["DATE"].to_list(), sub[col].to_list()


# ---------------------------------------------------------------------------
# Individual figures
# ---------------------------------------------------------------------------


def temperature_figure(df: pl.DataFrame, station_name: str) -> go.Figure:
    """Line chart: TN/TX shaded band + TM mean line."""
    fig = go.Figure()

    dates_min, temp_min = _series(df, "temp_min")
    dates_max, temp_max = _series(df, "temp_max")

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

    # temp_mean (optional — many stations lack it)
    if df["temp_mean"].is_not_null().any():
        dates_mean, temp_mean = _series(df, "temp_mean")
        fig.add_trace(
            go.Scatter(
                x=dates_mean,
                y=temp_mean,
                name="Mean",
                line=dict(color=_ORANGE, width=1.5),
                mode="lines",
            )
        )

    fig.update_layout(
        title=f"Temperature — {station_name}",
        yaxis_title="°C",
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
        hovermode="x unified",
        margin=dict(t=50, b=30),
    )
    return fig


def precipitation_figure(df: pl.DataFrame, station_name: str) -> go.Figure:
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
        title=f"Precipitation — {station_name}",
        yaxis_title="mm",
        bargap=0,
        hovermode="x unified",
        margin=dict(t=50, b=30),
    )
    return fig


def wind_figure(df: pl.DataFrame, station_name: str) -> go.Figure:
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
        title=f"Wind — {station_name}" + (" (no data for this station)" if no_data else ""),
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
