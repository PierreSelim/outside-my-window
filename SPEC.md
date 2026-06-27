# Outside My Window — Weather Visualization App

## Goal
A visualization app for French daily climatological open data from Météo-France,
published on [data.gouv.fr](https://www.data.gouv.fr/datasets/donnees-climatologiques-de-base-quotidiennes)).

## Data Source
- **Dataset**: Données climatologiques de base quotidiennes (Météo-France)
- **Scope**: `RR-T-Vent` files only (precipitation, temperature, wind)
- **URL pattern**: `https://object.files.data.gouv.fr/meteofrance/data/synchro_ftp/BASE/QUOT/Q_{dept}_{period}_RR-T-Vent.csv.gz`
- **Periods per department**:
  - `1852-1949` — historical, annual updates
  - `previous-1950-2024` — modern, monthly updates
  - `latest-2025-2026` — current year, daily updates
- **Coverage**: All French departments (01–95 + overseas)
- **License**: Open Licence / Etalab 2.0

## Tech Stack
- **Language**: Python
- **Package manager**: uv (`pyproject.toml`)
- **Web framework**: Dash (Plotly)
- **Data processing**: Polars
- **Charts**: Plotly (`plotly.express` and `plotly.graph_objects`)
- **Linter/formatter**: ruff (line length 120)

## Project Structure
```
outside-my-window/
├── app.py                      # Dash entry point — routing + layout shell
├── scripts/
│   └── build_station_index.py  # One-off script: fetch latest period per dept, emit data/stations.json
├── data/
│   ├── stations.json           # Pre-built station index (committed to repo)
│   └── cache/                  # Downloaded .csv.gz files (gitignored)
├── src/
│   ├── data_loader.py          # Fetch → cache → parse → return Polars DataFrame
│   ├── charts.py               # Plotly figure builders (temperature, precipitation, wind)
│   └── pages/
│       ├── map_page.py         # Landing page: station map
│       └── station_page.py     # Station detail page: charts + controls
├── tests/
├── pyproject.toml
└── SPEC.md
```

## Schema (RR-T-Vent files)
Separator: `;` — Missing value sentinel: `mq`

| Column | Description |
|---|---|
| `NUM_POSTE` | Station ID (int) |
| `NOM_USUEL` | Station name (str) |
| `LAT`, `LON` | Coordinates (float) |
| `ALTI` | Altitude in metres (int) |
| `AAAAMMJJ` | Date as YYYYMMDD integer — must be parsed |
| `RR` | Daily precipitation (mm) |
| `TN` / `TX` | Min / Max temperature (°C) |
| `TAMPLI` | Temperature amplitude TX−TN |
| `FFM` | Mean wind speed (m/s) |
| `FXY` | Max wind gust (m/s) |
| `DXY` | Direction of max gust (°) |
| `Q*` | Quality flag for each measure: `1` = valid, `9` = estimated, `null` = missing |

Dept 31 (latest period): **22 stations**, 9 235 rows, 2025-01-01 → 2026-04-07.
Wind columns have significant nulls (not all stations measure them).

## Station Index (`data/stations.json`)
Pre-built file committed to the repo. Generated once (and regenerated when station network changes) by `scripts/build_station_index.py`.

Schema — list of objects:
```json
[
  { "station_id": 31001, "station_name": "TOULOUSE", "dept": "31",
    "lat": 43.6, "lon": 1.44, "altitude": 152 },
  ...
]
```

The script fetches only the `latest-2025-2026` period per department (smallest files), extracts unique stations, and writes the index. It processes departments sequentially (one HTTP request at a time) and skips departments that return no data. The app itself never triggers a bulk download — it only loads one department at a time on demand.

## App Pages & Navigation
Multi-page routing via `dcc.Location` + a top-level callback in `app.py` that renders the correct page layout based on `pathname`.

| Path | Page |
|---|---|
| `/` | Map — landing page |
| `/station` | Station detail — query params `?dept=31&station=31001` |

### Map page (`/`)
- Loads `data/stations.json` at import time (fast — committed file)
- Renders a `px.scatter_mapbox` (OpenStreetMap tiles, no token required) with one marker per station
- Hover tooltip: station name, department, altitude
- Click on a marker → navigates to `/station?dept={dept}&station={station_id}`

### Station detail page (`/station?dept=…&station=…`)
- Back-to-map link at the top
- Station dropdown (all stations in the department) — changing it updates the URL
- Year range slider — default window: last 20 years
- Three tabs: **Daily charts**, **Yearly extremes**, **Monthly averages**
- Department data loaded lazily on first visit, cached server-side in `_dept_cache`

#### Daily charts tab
- Granularity radio (Day / Week / Month) — default Day
- Three charts: temperature band, precipitation bar, wind lines

#### Temporal aggregation
When Week or Month is selected, daily rows are grouped by truncated date period and all
numeric measurement columns (`temp_min`, `temp_max`, `temp_amplitude`,
`precipitation`, `wind_mean`, `wind_gust`) are averaged (`mean`). Metadata columns
(`lat`, `lon`, `altitude`) keep their first value (constant per station). Chart titles
gain a suffix: ` (weekly avg)` or ` (monthly avg)`.

#### Yearly extremes tab
- Line chart: yearly count of hot days and cold days (Tmin < 0 °C)
- **Configurable hot-day definition** — a dropdown lets the user choose from eight options:
  - Default: Tmin ≥ 20 °C **and** Tmax ≥ 35 °C (historical, matches the Observations shading)
  - Tmax-only thresholds: 32, 35, 36, 37, 38, 39, 40 °C
  - Represented by the sum type `HotDayDefinition = TmaxOnly | TmaxAndTmin` in `transforms.py`
  - `HOT_DAY_OPTIONS` lists all eight options; `hot_day_from(label)` resolves a label string back to
    the matching option (returns `DEFAULT_HOT_DAY` for unrecognised labels, never raises)
- **Provisional current-year rendering** — when the year-range upper bound includes the current year:
  - Complete years (year < current year) are drawn as solid traces
  - The current (partial) year is rendered as a dotted connector from the last complete year to a
    hollow circle marker (`circle-open`), indicating provisional/partial data
  - If no data exists for the current year in the filtered range, no dotted trace is added
  - Edge case: if there are zero complete years and only a provisional row, a single hollow marker
    is rendered (no connector line)
  - When the year-range upper bound is below the current year, the provisional path is skipped
    entirely and all years are treated as complete
- Optional tendency lines (OLS linear regression) toggled via a checklist
  - Regression is computed over **complete years only** (current year excluded), even when the
    provisional point is shown on the chart
  - When trends are shown: slope and R² appear both as a hover tooltip on each trend trace
    and in a summary card below the chart
- The Observations tab temperature-chart hot-day shading is intentionally left on the original
  default rule (Tmin ≥ 20 °C and Tmax ≥ 35 °C) regardless of the dropdown selection

#### Monthly averages tab
- Average Tmin / Tmax by month of year, with **±2σ shaded bands** showing inter-annual variability
- Average Tmin / Tmax by month of year, broken down by decade

## Architecture Decisions
- **Station index**: pre-built JSON committed to repo; never fetched at runtime by the app
- **Data fetching**: on-demand per department, cached locally as `.csv.gz`
- **DataFrame**: Polars throughout; only converted to lists/arrays at chart-build time
- **Time periods**: merged into a single DataFrame per department on load
- **Column names**: raw Météo-France names (TN, TX, …) renamed to readable names
  (temp_min, temp_max, …) inside `_parse()` — see `COLUMN_RENAME` in `data_loader.py`
- **Map tiles**: OpenStreetMap via `px.scatter_mapbox` — no Mapbox token needed
- **Temporal aggregation**: `aggregate(df, granularity)` in `data_loader.py`; called in
  `station_page.update_charts` after the station/year filter. `Granularity.DAY` is the
  identity (returns df unchanged). Week uses `dt.truncate("1w")`, month uses `"1mo"`.
- **Trend statistics**: `linear_trend` in `transforms.py` returns a `LinearTrend` dataclass
  `(slope, intercept, r_squared)`. Slope and R² are surfaced in two places: hover tooltip on
  the trend trace, and a summary card rendered below the yearly extremes chart via a second
  Dash `Output`. When a provisional current year is shown, trend regression uses complete years only.
- **HotDayDefinition sum type**: `TmaxOnly | TmaxAndTmin` in `transforms.py`. `_hot_predicate`
  maps a definition to a Polars expression; `yearly_hot_cold` accepts a `definition` parameter
  (default `DEFAULT_HOT_DAY = TmaxAndTmin(35.0, 20.0)`). The `hot_cold_yearly_figure` function
  in `charts.py` also accepts `definition` and embeds `definition.label` in the hot-trace name.
- **±2σ bands**: `_add_sigma_band(fig, x, avg, std, fillcolor, n_sigma=2.0)` in `charts.py`
  draws a shaded inter-annual variability envelope using two invisible boundary `go.Scatter`
  traces with `fill="tonexty"`. Called by `monthly_avg_temp_figure` for both Tmax (orange
  fill `_TMAX_SIGMA_FILL`) and Tmin (blue fill `_TMIN_SIGMA_FILL`). The standard deviation
  is computed per calendar month via `pl.col(…).std()` in the `group_by("month").agg(…)`
  step.

