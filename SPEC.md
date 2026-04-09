# Outside My Window — Weather Visualization App

## Goal
A visualization app for French daily climatological open data from Météo-France,
published on [data.gouv.fr](https://www.data.gouv.fr/datasets/donnees-climatologiques-de-base-quotidiennes).

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
- **Charts**: Plotly (via `plotly.express` and `plotly.graph_objects`)
- **Linter/formatter**: ruff (line length 120)

## Project Structure
```
outside-my-window/
├── app.py                  # Dash entry point
├── data/
│   └── cache/              # Downloaded .csv.gz files (gitignored)
├── src/
│   ├── data_loader.py      # Fetch → cache → parse → return Polars DataFrame
│   └── charts.py           # Plotly figure builders
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

## Architecture Decisions
- **Data fetching**: On-demand per department, cached locally as `.csv.gz`
- **DataFrame**: Polars throughout; only converted to lists/arrays at chart-build time
- **Time periods**: Merged into a single DataFrame per department on load
- **Missing values**: To be determined after schema inspection (likely `mq` sentinel)

## Development Plan
1. ✅ Schema exploration — inspect dept 31 latest file to understand columns
2. ✅ Project setup — `uv init`, add dependencies
3. ✅ Data loader — fetch, cache, parse, merge periods, clean
4. ✅ Dash app (dept 31 only) — station dropdown, date range, temperature/precipitation/wind charts
5. ✅ Expand to all departments — dept dropdown (01–95 + overseas), server-side lazy cache per dept

## App Behaviour (final)
- Dept dropdown always interactive; changing it triggers data fetch + spinner on lower controls/charts
- `_dept_cache` dict holds loaded DataFrames in memory for the process lifetime (no re-download within session)
- Station dropdown and year slider are repopulated from the loaded data via `on_dept_change` callback
- Charts update via `update_charts` callback (station + year range as inputs, dept as State)

## First Test Department
**Dept 31** (Haute-Garonne) — used for initial development and schema validation.
