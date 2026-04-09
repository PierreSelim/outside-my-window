from __future__ import annotations

import polars as pl
from dash import Dash, Input, Output, State, dcc, html

from src.charts import empty_figure, precipitation_figure, temperature_figure, wind_figure
from src.data_loader import Station, load_department, stations_from

# ---------------------------------------------------------------------------
# Department definitions
# ---------------------------------------------------------------------------

_DEPT_NAMES: dict[str, str] = {
    "01": "Ain", "02": "Aisne", "03": "Allier", "04": "Alpes-de-Haute-Provence",
    "05": "Hautes-Alpes", "06": "Alpes-Maritimes", "07": "Ardèche", "08": "Ardennes",
    "09": "Ariège", "10": "Aube", "11": "Aude", "12": "Aveyron",
    "13": "Bouches-du-Rhône", "14": "Calvados", "15": "Cantal", "16": "Charente",
    "17": "Charente-Maritime", "18": "Cher", "19": "Corrèze",
    "2A": "Corse-du-Sud", "2B": "Haute-Corse",
    "21": "Côte-d'Or", "22": "Côtes-d'Armor", "23": "Creuse", "24": "Dordogne",
    "25": "Doubs", "26": "Drôme", "27": "Eure", "28": "Eure-et-Loir",
    "29": "Finistère", "30": "Gard", "31": "Haute-Garonne", "32": "Gers",
    "33": "Gironde", "34": "Hérault", "35": "Ille-et-Vilaine", "36": "Indre",
    "37": "Indre-et-Loire", "38": "Isère", "39": "Jura", "40": "Landes",
    "41": "Loir-et-Cher", "42": "Loire", "43": "Haute-Loire", "44": "Loire-Atlantique",
    "45": "Loiret", "46": "Lot", "47": "Lot-et-Garonne", "48": "Lozère",
    "49": "Maine-et-Loire", "50": "Manche", "51": "Marne", "52": "Haute-Marne",
    "53": "Mayenne", "54": "Meurthe-et-Moselle", "55": "Meuse", "56": "Morbihan",
    "57": "Moselle", "58": "Nièvre", "59": "Nord", "60": "Oise",
    "61": "Orne", "62": "Pas-de-Calais", "63": "Puy-de-Dôme", "64": "Pyrénées-Atlantiques",
    "65": "Hautes-Pyrénées", "66": "Pyrénées-Orientales", "67": "Bas-Rhin",
    "68": "Haut-Rhin", "69": "Rhône", "70": "Haute-Saône", "71": "Saône-et-Loire",
    "72": "Sarthe", "73": "Savoie", "74": "Haute-Savoie", "75": "Paris",
    "76": "Seine-Maritime", "77": "Seine-et-Marne", "78": "Yvelines",
    "79": "Deux-Sèvres", "80": "Somme", "81": "Tarn", "82": "Tarn-et-Garonne",
    "83": "Var", "84": "Vaucluse", "85": "Vendée", "86": "Vienne",
    "87": "Haute-Vienne", "88": "Vosges", "89": "Yonne", "90": "Territoire de Belfort",
    "91": "Essonne", "92": "Hauts-de-Seine", "93": "Seine-Saint-Denis",
    "94": "Val-de-Marne", "95": "Val-d'Oise",
    "971": "Guadeloupe", "972": "Martinique", "973": "Guyane",
    "974": "La Réunion", "976": "Mayotte",
}

DEPT_OPTIONS = [{"label": f"{code} — {name}", "value": code} for code, name in _DEPT_NAMES.items()]
DEFAULT_DEPT = "31"

# ---------------------------------------------------------------------------
# Server-side data cache (populated lazily per department)
# ---------------------------------------------------------------------------

_dept_cache: dict[str, pl.DataFrame | None] = {}


def _load_cached(dept: str) -> pl.DataFrame | None:
    if dept not in _dept_cache:
        _dept_cache[dept] = load_department(dept)
    return _dept_cache[dept]


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------

app = Dash(__name__, title="Outside My Window")

app.layout = html.Div(
    style={"fontFamily": "sans-serif", "maxWidth": "1400px", "margin": "0 auto", "padding": "1rem"},
    children=[
        html.H2("Météo France", style={"marginBottom": "1rem"}),

        # Department selector (outside loading — always interactive)
        html.Div(
            style={"marginBottom": "1rem"},
            children=[
                html.Label("Département", style={"fontWeight": "bold"}),
                dcc.Dropdown(
                    id="dept-dropdown",
                    options=DEPT_OPTIONS,
                    value=DEFAULT_DEPT,
                    clearable=False,
                    style={"maxWidth": "400px"},
                ),
            ],
        ),

        # Loading wrapper covers station/year controls + charts
        dcc.Loading(
            type="circle",
            children=[
                # Station + year range row
                html.Div(
                    style={
                        "display": "flex", "gap": "2rem",
                        "alignItems": "flex-start", "marginBottom": "1.5rem",
                    },
                    children=[
                        html.Div(
                            style={"flex": "0 0 320px"},
                            children=[
                                html.Label("Station", style={"fontWeight": "bold"}),
                                dcc.Dropdown(id="station-dropdown", options=[], value=None, clearable=False),
                            ],
                        ),
                        html.Div(
                            style={"flex": "1"},
                            children=[
                                html.Label("Year range", style={"fontWeight": "bold"}),
                                dcc.RangeSlider(
                                    id="year-slider",
                                    min=1950,
                                    max=2026,
                                    step=1,
                                    value=[2016, 2026],
                                    marks={y: str(y) for y in range(1950, 2027, 10)},
                                    tooltip={"placement": "bottom", "always_visible": True},
                                ),
                            ],
                        ),
                    ],
                ),

                # Charts
                dcc.Graph(id="chart-temperature", config={"displayModeBar": False}),
                dcc.Graph(id="chart-precipitation", config={"displayModeBar": False}),
                dcc.Graph(id="chart-wind", config={"displayModeBar": False}),
            ],
        ),
    ],
)

# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


@app.callback(
    Output("station-dropdown", "options"),
    Output("station-dropdown", "value"),
    Output("year-slider", "min"),
    Output("year-slider", "max"),
    Output("year-slider", "marks"),
    Output("year-slider", "value"),
    Input("dept-dropdown", "value"),
)
def on_dept_change(dept: str | None) -> tuple:
    """Load department data (fetching from remote if not cached) and populate controls."""
    _fallback_marks = {y: str(y) for y in range(1950, 2027, 10)}
    if dept is None:
        return [], None, 1950, 2026, _fallback_marks, [2016, 2026]

    df = _load_cached(dept)
    if df is None:
        return [], None, 1950, 2026, _fallback_marks, [2016, 2026]

    stations: list[Station] = stations_from(df)
    year_min = int(df["DATE"].min().year)  # type: ignore[union-attr]
    year_max = int(df["DATE"].max().year)  # type: ignore[union-attr]
    marks = {y: str(y) for y in range(year_min, year_max + 1, 10)}
    year_value = [max(year_min, year_max - 10), year_max]
    options = [{"label": s.name, "value": s.num_poste} for s in stations]
    first_station = stations[0].num_poste if stations else None

    return options, first_station, year_min, year_max, marks, year_value


@app.callback(
    Output("chart-temperature", "figure"),
    Output("chart-precipitation", "figure"),
    Output("chart-wind", "figure"),
    Input("station-dropdown", "value"),
    Input("year-slider", "value"),
    State("dept-dropdown", "value"),
)
def update_charts(station_id: int | None, year_range: list[int], dept: str | None) -> tuple:
    if dept is None or station_id is None:
        placeholder = empty_figure("No data available")
        return placeholder, placeholder, placeholder

    df_full = _load_cached(dept)
    if df_full is None:
        placeholder = empty_figure("No data available")
        return placeholder, placeholder, placeholder

    year_start, year_end = year_range
    df = df_full.filter(
        (pl.col("station_id") == station_id)
        & (pl.col("DATE").dt.year() >= year_start)
        & (pl.col("DATE").dt.year() <= year_end)
    )

    if df.is_empty():
        placeholder = empty_figure("No data for this station / period")
        return placeholder, placeholder, placeholder

    station_name = df["station_name"][0]
    return (
        temperature_figure(df, station_name),
        precipitation_figure(df, station_name),
        wind_figure(df, station_name),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)
