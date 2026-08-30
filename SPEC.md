# Outside My Window — Weather Visualization App

## Goal
A visualization app for French daily climatological open data from Météo-France,
published on [data.gouv.fr](https://www.data.gouv.fr/datasets/donnees-climatologiques-de-base-quotidiennes)).

## Data Source
- **Dataset**: Données climatologiques de base quotidiennes (Météo-France)
- **Scope**: `RR-T-Vent` files only (precipitation, temperature, wind)
- **URL pattern**: stable data.gouv.fr permalink `https://www.data.gouv.fr/api/1/datasets/r/{resource-id}` (302-redirects to the live storage host, so a host migration never breaks us). The `(dept, period) → resource-id` map lives in `data/resources.json`, built by `scripts/build_resource_index.py`. Replaced the former hardcoded storage host after the `object.files.data.gouv.fr` mirror stopped syncing in June 2026.
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
- Year range slider — default window: last 20 years. It applies to the first three tabs only and
  is **hidden** while the *Then vs now* tab is active (`year_slider_style`), since that tab brings
  its own two period sliders. Station is the only control that is global to every tab.
- Four tabs: **Daily charts**, **Yearly extremes**, **Monthly averages**, **Then vs now**
  (tab `value` remains `"comparison"`). *Then vs now* is deliberately **last** — the page reads
  raw data → extremes → averages → shift — but is right-aligned and accent-coloured
  (`.station-tab--feature`) so it is spotted without being the landing view.
- Department data loaded lazily on first visit, cached server-side in `_dept_cache`
- LATEST-period data refreshes automatically after 6h (`_LATEST_TTL_SECONDS`); a `POST /api/refresh`
  route (used by the Electron app's Ctrl/Cmd+Shift+R shortcut) force-clears the in-process cache and
  deletes the disk-cached LATEST file so the next load re-fetches immediately — see `clear_cache()`
  in `data_loader.py`

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

#### Then vs now tab
Answers "has the whole distribution shifted?" — a question the monthly averages cannot,
since a mean rising because every day warmed and a mean rising because the hot tail doubled
look identical there.

- Overlays the **probability density** of daily Tmax (first chart) and daily Tmin (second) for
  **two year ranges**, both restricted to the **same calendar window**
- Controls, all on one row: a shared season window (`dcc.DatePickerRange`, default 1 Jun – 31 Aug)
  and **two year boxes per period** (`dcc.Input(type="number")`, `debounce=True`; defaults: oldest
  30 years vs newest 30 years of the record), followed by a `Record: <min>–<max>` hint
- **Typed boxes, not `RangeSlider`s**: a slider is a scrubbing control, but comparison periods are
  chosen exactly (1961–1990 vs 1991–2020). Two sliders side by side rendered their decade marks
  illegibly, and stacking them only made the unreadable control taller. `min`/`max` on the native
  number input keeps entry inside the record without any validation code.
- `_typed_span(start, end, fallback)` turns the two boxes into a `YearSpan`: a cleared box falls
  back to the record bound, and `YearSpan.of` orders an inverted pair rather than rejecting it —
  invalid input is a state the callback absorbs, never an exception
- The page-level "Year range" slider is **not** an input to this tab — the period boxes replace it,
  and the slider is hidden while the tab is active rather than sitting there inert
- Densities are **smooth KDE curves** (`go.Scatter`, spline, filled to zero), not histograms —
  see `gaussian_kde` below
- A dashed vertical line marks each period's mean
- Below the charts, a stats card gives `n`, mean, median and p90 per period per variable, plus
  the Δ mean / Δ p90 shift between them
- Either period being empty, all-null, or constant for that variable renders the placeholder
  figure and an empty stats card — no exception

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
- **`YearSpan` (frozen dataclass)**: an inclusive `start`/`end` year pair, ordered by construction
  via `YearSpan.of`. It replaces the `tuple[int, int]` / `list[int]` year ranges that let
  `start > end` be represented, and carries the `label` used in chart legends and stats headers.
  The two filters that consume it live in `transforms.py`: `year_filter(df, span)` for the three
  single-period tabs, and `window_filter(df, span, window)` — the same year predicate plus a
  `DayWindow` — for the comparison tab. Dash still hands the page-level `RangeSlider` value over as
  a `list[int]`, so `_filtered_station_df` narrows it with `YearSpan.of(*year_range)` at the
  callback boundary.
- **`DayWindow` (MMDD ints)**: a year-agnostic calendar window in `transforms.py`, stored as
  `start_md`/`end_md` integers (`601` = 1 June). MMDD comparison is leap-year proof, and a window
  whose end precedes its start (1 Dec → 28 Feb) is a legal *wrap-around* rather than an invalid
  state — `_in_window` switches from `&` to `|`. Note `dt.month()` is `Int8` in Polars, so both
  `month` and `day` are cast to `Int32` before `month * 100`; without the cast December overflows.
  A wrapping window assigns each day to its own calendar year, so "Dec–Feb, 1991–2020" pairs the
  December of one winter with the Jan/Feb of the next at the two range boundaries — negligible
  over a multi-decade density, but it is a known simplification.
- **`DatePickerRange` reference year**: the component requires a year, so the window picker is
  pinned to 2000 (a leap year, keeping 29 Feb selectable) and only month/day are read back via
  `DayWindow.from_dates`. See `_WINDOW_REF_YEAR` in `station_page.py`.
- **`gaussian_kde(series, bandwidth=None) -> Density | None`**: hand-rolled Gaussian kernel
  density estimate in `transforms.py`, in the same spirit as `linear_trend` — the project has no
  numpy/scipy. Evaluated on a fixed 256-point grid spanning the sample extremes ± 3 bandwidths.
  Returns `None` for a sample with fewer than 2 values or zero spread (no bandwidth ⇒ no density).
  - **Cost**: the naive O(n × grid) kernel sum is ~0.5 M `exp()` calls per curve in pure Python.
    Instead the sample is first collapsed with `value_counts()` onto its native 0.1 °C
    resolution — lossless for Météo-France data — so the sum runs over a few hundred *distinct*
    temperatures rather than tens of thousands of days. Measured: ~57 ms for the whole
    two-chart callback over a 64-year record (2 760 samples per period).
  - **Bandwidth**: Silverman's rule of thumb `0.9 · scale · n^(-1/5)` with
    `scale = min(stdev, IQR/1.349)`. The robust scale matters here: the warm tail a two-period
    comparison is meant to reveal would otherwise inflate the stdev and oversmooth it away.
  - `test_gaussian_kde_matches_the_direct_kernel_sum` pins the binned shortcut against the
    textbook formula, and `test_gaussian_kde_integrates_to_one` pins the defining PDF property.
- **`describe(series) -> Distribution | None`**: returns `None` for an all-null/empty series
  rather than raising, matching the `load_department_cached` / `_filtered_station_df` convention.
- **`_station_df` / `_filtered_station_df` split**: `station_page.py` filters by station alone in
  `_station_df`; `_filtered_station_df` layers the page-level year range on top. The comparison
  callback uses the former, since it applies its own two year ranges.
- **±2σ bands**: `_add_sigma_band(fig, x, avg, std, fillcolor, n_sigma=2.0)` in `charts.py`
  draws a shaded inter-annual variability envelope using two invisible boundary `go.Scatter`
  traces with `fill="tonexty"`. Called by `monthly_avg_temp_figure` for both Tmax (orange
  fill `_TMAX_SIGMA_FILL`) and Tmin (blue fill `_TMIN_SIGMA_FILL`). The standard deviation
  is computed per calendar month via `pl.col(…).std()` in the `group_by("month").agg(…)`
  step.

