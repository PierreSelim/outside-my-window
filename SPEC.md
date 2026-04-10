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
| `TN` / `TX` / `TM` | Min / Max / Mean temperature (°C) |
| `TAMPLI` | Temperature amplitude TX−TN |
| `FFM` | Mean wind speed (m/s) |
| `FXY` | Max wind gust (m/s) |
| `DXY` | Direction of max gust (°) |
| `Q*` | Quality flag for each measure: `1` = valid, `9` = estimated, `null` = missing |

Dept 31 (latest period): **22 stations**, 9 235 rows, 2025-01-01 → 2026-04-07.
`TM` and wind columns have significant nulls (not all stations measure them).

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
- Year range slider
- Granularity radio (Day / Week / Month) — default Day
- Three charts: temperature band, precipitation bar, wind lines
- Department data loaded lazily on first visit, cached server-side in `_dept_cache`

#### Temporal aggregation
When Week or Month is selected, daily rows are grouped by truncated date period and all
numeric measurement columns (`temp_min`, `temp_max`, `temp_mean`, `temp_amplitude`,
`precipitation`, `wind_mean`, `wind_gust`) are averaged (`mean`). Metadata columns
(`lat`, `lon`, `altitude`) keep their first value (constant per station). Chart titles
gain a suffix: ` (weekly avg)` or ` (monthly avg)`.

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

## Development Plan
1. ✅ Schema exploration
2. ✅ Project setup
3. ✅ Data loader — fetch, cache, parse, merge, rename columns
4. ✅ Single-dept Dash app — station dropdown, year slider, 3 charts
5. ✅ Expand to all departments — dept dropdown, lazy server-side cache
6. ✅ Unit tests — pytest + coverage (src/ at 89%)
7. `scripts/build_station_index.py` — generate `data/stations.json`
8. Refactor `app.py` into `src/pages/map_page.py` + `src/pages/station_page.py`
9. Multi-page routing in `app.py` via `dcc.Location`
10. Map page — `px.scatter_mapbox`, hover info, click → navigate
11. Station page — back link, URL-driven station selection
