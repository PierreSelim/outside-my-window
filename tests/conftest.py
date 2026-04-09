from __future__ import annotations

from datetime import date

import polars as pl
import pytest


@pytest.fixture
def sample_df() -> pl.DataFrame:
    """Minimal DataFrame as returned by load_department() — columns already renamed."""
    return pl.DataFrame(
        {
            "station_id":      [31001, 31001, 31001, 31002, 31002],
            "station_name":    ["TOULOUSE", "TOULOUSE", "TOULOUSE", "BLAGNAC", "BLAGNAC"],
            "LAT":             [43.60, 43.60, 43.60, 43.63, 43.63],
            "LON":             [1.44,  1.44,  1.44,  1.37,  1.37],
            "altitude":        [152,   152,   152,   151,   151],
            "DATE":            [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3),
                                date(2020, 1, 1), date(2020, 1, 2)],
            "temp_min":        [-1.0, 0.5,  2.0, -2.0,  1.0],
            "temp_max":        [8.0,  10.5, 12.0, 7.0,   9.0],
            "temp_mean":       [3.5,  5.0,  7.0,  2.5,   5.0],
            "temp_amplitude":  [9.0,  10.0, 10.0, 9.0,   8.0],
            "precipitation":   [0.0,  2.5,  0.0,  0.0,   1.0],
            "wind_mean":       [3.0,  5.0,  2.0,  None,  None],
            "wind_gust":       [8.0,  12.0, 6.0,  None,  None],
            "wind_gust_dir":   [180.0, 270.0, 90.0, None, None],
        },
        schema={
            "station_id":     pl.Int32,
            "station_name":   pl.String,
            "LAT":            pl.Float64,
            "LON":            pl.Float64,
            "altitude":       pl.Int32,
            "DATE":           pl.Date,
            "temp_min":       pl.Float64,
            "temp_max":       pl.Float64,
            "temp_mean":      pl.Float64,
            "temp_amplitude": pl.Float64,
            "precipitation":  pl.Float64,
            "wind_mean":      pl.Float64,
            "wind_gust":      pl.Float64,
            "wind_gust_dir":  pl.Float64,
        },
    )
