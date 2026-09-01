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
├── server.py                   # waitress entry point used by the desktop app
├── server.spec                 # PyInstaller spec: bundles server.py + data + assets
├── electron/main.js            # Desktop shell: spawns the sidecar, opens the window
├── scripts/
│   ├── build_station_index.py  # One-off script: emit data/stations.json
│   └── build_resource_index.py # One-off script: emit data/resources.json
├── data/
│   ├── stations.json           # Pre-built station index (committed to repo)
│   ├── resources.json          # (dept, period) → data.gouv.fr resource id (committed)
│   └── cache/                  # Downloaded .csv.gz files (gitignored)
├── src/
│   ├── data_loader.py          # Fetch → cache → parse → return Polars DataFrame
│   ├── transforms.py           # Pure computation: trends, densities, records, normals
│   ├── charts.py               # Plotly figure builders
│   └── pages/
│       ├── components.py       # Shared components: the all-France station search
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
    "lat": 43.6, "lon": 1.44, "altitude": 152,
    "first_year": 1923, "last_year": 2026, "n_years": 104 },
  ...
]
```

`first_year` / `last_year` describe the station's record span, and drive the map's marker colour.
`data_loader.station_index()` reads the file into `IndexedStation` objects, each carrying a
`RecordSpan | None` — the span is present as a whole or absent as a whole, never half of it, and
`n_years` is derived from it rather than trusted from the file. A malformed entry is skipped
rather than crashing the map. The span years are only knowable from the historical files, so the
default run reads **all three periods** of every department (~1.5 GB of downloads, cached under `data/cache/`, one HTTP
request at a time). `--fast` reads the latest period alone and omits the span fields; the
map then falls back to a single marker colour. `--depts` and `--out` restrict a run for testing.

The app itself never triggers a bulk download — it loads one department at a time, on demand.

## App Pages & Navigation
Multi-page routing via `dcc.Location` + a top-level callback in `app.py` that renders the correct page layout based on `pathname`.

| Path | Page |
|---|---|
| `/` | Map — landing page |
| `/station` | Station detail — query params `?dept=31&station=31001` |

### Map page (`/`)
- Reads `data/stations.json` through `data_loader.station_index()` (`lru_cache`d — committed file)
- Renders a `go.Scattermap` (OpenStreetMap tiles, no token required) with one marker per station
- **Markers are coloured by record length** (`RecordSpan.n_years`, a sand→slate→navy scale with a
  colorbar) when every station in the index carries a span, and fall back to one flat colour
  otherwise — the colour array has to line up with the marker array — a 170-year record and a
  3-year one are not the same offer, and an all-blue map says nothing about which is which
- Hover tooltip: station name, department, altitude, record span; `<extra>Click to open</extra>`
  makes the click affordance explicit
- A **station search** (`components.station_search`) floats over the map: a type-ahead over all
  ~2 100 stations, since spotting one dot on a country-scale map is not a way to find a town
- Click on a marker → navigates to `/station?dept={dept}&station={station_id}`

### Station detail page (`/station?dept=…&station=…`)
- Back-to-map link at the top, plus the same all-France **station search** in the page nav — the
  only control that crosses department boundaries (the station dropdown is dept-scoped)
- A **station header card** above the tabs, independent of the year range (`update_station_header`):
  - The latest observation read against the normal for that calendar day, and the rank of that day
    among all years of the record. This is the only view in the app answering "what is it doing
    now, and is that unusual", which is the question the daily-refreshed LATEST period exists to
    serve.
  - The five tiles are worded for a reader who does not know the jargon: *Daytime high* /
    *Overnight low* rather than Tmax/Tmin, and *Warmer than usual* / *Colder than usual* carrying
    an unsigned magnitude rather than *Anomaly* carrying a signed one. The sign lives in the
    label, the reference period in the detail line (`usual high for this date 26.6 °C (1991–2020)`).
    Sign still drives colour: warm tiles print their value in the warm accent, cool in the cool one.
  - The all-time records: hottest day, coldest night, wettest day, strongest gust, and the longest
    run of hot days under the selected definition.
- **Hot-day definition dropdown at page level** — one definition now drives three things: the
  Observations shading, the Yearly extremes counts, and the records card. It used to sit inside the
  Yearly tab while the Observations chart shaded a different, hard-coded rule.
- Station dropdown (all stations in the department) — changing it updates the URL
- Year range slider — default window: last 20 years. It applies to the first three tabs only and
  is **hidden** while the *Then vs now* tab is active (`year_slider_style`), since that tab brings
  its own two period sliders. Station is the only control that is global to every tab.
  - `_year_marks` picks the mark interval (10/20/25/50/100 years) so that at most
    `_MAX_SLIDER_MARKS` (8) labels are drawn: a fixed decade step put 22 four-digit labels in one
    row on a station recording since 1809, and they overlapped into a smear.
  - The value tooltips are placed **above** the handles; marks are below. Both defaulted to the
    bottom and collided.
- Four tabs: **Then vs now**, **Daily charts**, **Yearly extremes**, **Monthly averages**
  (tab `value` remains `"comparison"`). *Then vs now* is **first and selected by default** — it is
  the reason to open a station page — and is accent-coloured (`.station-tab--feature`); the
  remaining tabs keep the raw data → extremes → averages order for drilling down.
- Department data loaded lazily on first visit, cached server-side in `_dept_cache`
- LATEST-period data refreshes automatically after 6h (`_LATEST_TTL_SECONDS`); a `POST /api/refresh`
  route (used by the Electron app's Ctrl/Cmd+Shift+R shortcut) force-clears the in-process cache and
  deletes the disk-cached LATEST file so the next load re-fetches immediately — see `clear_cache()`
  in `data_loader.py`

#### Daily charts tab
- Granularity radio (Day / Week / Month) — default Day
- Three charts: temperature band, precipitation bar, wind lines
- The wind card is **hidden entirely** for a station that measures no wind (`has_wind`), rather
  than rendering an empty axis box
- Nulls are **kept** in every series (`charts._series`): a day the station did not observe is a
  gap in the line, not a straight segment drawn confidently across it

#### Temporal aggregation
When Week or Month is selected, daily rows are grouped by truncated date period and each
measurement is collapsed with **the operation that preserves its meaning** (`_MEASUREMENT_AGGS`
in `data_loader.py`):

| Column | Aggregation | Why |
|---|---|---|
| `temp_min`, `temp_max`, `wind_mean` | `mean` | an average is what a mean temperature is |
| `precipitation` | `sum` (null-preserving) | a month of rain is a total — averaging daily mm reported 1.6 mm for a 50.1 mm month |
| `wind_gust` | `max` | FXY is already a daily maximum; the mean of maxima describes no wind that ever blew |

`_total_or_null` keeps an all-null group null, so a station that measures no rain reports "no
data" rather than a confident 0 mm. Metadata columns (`lat`, `lon`, `altitude`) keep their first
value. Chart titles gain ` (weekly)` / ` (monthly)`, and the precipitation axis is relabelled
`mm/week` / `mm/month` via `Granularity.per_unit`.

A period with missing days still yields a proportionally understated total — a known
simplification, marked with a `ponytail:` note in `aggregate`.

#### Yearly extremes tab
- Line chart: yearly counts of **three** series — hot days, tropical nights (Tmin ≥ 20 °C) and
  frost days (Tmin < 0 °C). Tropical nights are the warm-tail counterpart of a frost day and the
  cleanest warming signal in the record, so they get their own series rather than being folded
  into the hot-day rule. `charts.yearly_series(definition)` is the single source of the three
  (column, label, colour) triples, shared by the figure and the tendency table.
- Counts are drawn with `shape: "linear"`: a spline over annual integer counts draws values that
  do not exist.
- **Configurable hot-day definition**, at page level (see above):
  - Default: **Tmax ≥ 30 °C** (*jour de forte chaleur*). It replaced Tmin≥20 & Tmax≥35, which at a
    typical station yields 0–9 days a year — a count so low the chart is mostly sampling noise,
    where Tmax≥30 runs 37–77 and shows the trend. The old pair remains as `HEATWAVE_HOT_DAY`.
  - Other options: Tmax-only at 32, 35, 36, 38, 40 °C, plus `HEATWAVE_HOT_DAY`
  - Represented by the sum type `HotDayDefinition = TmaxOnly | TmaxAndTmin` in `transforms.py`
  - `HOT_DAY_OPTIONS` lists every option; `hot_day_from(label)` resolves a label string back to
    the matching option (returns `DEFAULT_HOT_DAY` for unrecognised labels, never raises)

- **Coverage guard** — a missing day counts as "not hot", so a year the station spent offline
  reads as a genuine minimum. `yearly_hot_cold` therefore also returns `observed_days` and
  `coverage` (share of the year's days carrying any temperature reading, leap years included),
  and `is_complete(min_coverage=MIN_YEAR_COVERAGE)` is the predicate that separates a measured
  year from downtime. Years below 90 % coverage are:
  - left out of the plotted lines (the y value becomes `None`, so the line *breaks* rather than
    bridging the gap — `connectgaps=False`),
  - marked by a grey `add_vrect` band spanning each run of excluded years, with one legend proxy,
  - and excluded from the regression, which is the point: at REVEL, 25 of 113 years are below the
    threshold and five have **zero** temperature observations, each of which previously plotted as
    a hard 0 and dragged the trend line.
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
  - Regression runs on exactly the population the solid lines show: fully observed years, current
    year excluded
  - When trends are shown: slope and R² appear both as a hover tooltip on each trend trace and in
    a summary card below the chart, which states the number of fully observed years it used and
    gives the slope per year *and* per decade
- The Observations tab shading and this tab now share one definition (the page-level dropdown);
  they used to disagree.

#### Monthly averages tab
- Average Tmin / Tmax by month of year, with **±2σ shaded bands** showing inter-annual variability
- Average Tmin / Tmax by month of year, broken down by decade:
  - a **decade multi-select** (`update_decade_options`) offers the decades the current year range
    covers and pre-selects the oldest, middle and newest. Drawing every decade at once made this a
    legend with a plot attached — up to 18 gradient-coloured traces with no way to isolate two.
  - Tmin traces start at `visible="legendonly"`, halving the legend and keeping the data one click
    away.

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
- **Above** the charts, a headline card gives the mean and p90 shift per variable as stat tiles:
  the shift is the answer this tab exists for, and it used to sit in the fourth column of a plain
  table below two full-width charts. Labelled in plain words — *Average daytime highs* and
  *Warmest tenth of daytime highs*, not `Δ mean Tmax` / `Δ p90 Tmax`
- The two densities sit **side by side** on a wide screen (`.chart-pair`, CSS grid `auto-fit`
  `minmax(440px, 1fr)`), so the Tmin chart is not below the fold
- An **overlap warning** appears when the two periods intersect — comparing 1990–2020 with
  2000–2026 silently compares part of the same years with themselves
- Below the charts, a stats card gives day count, mean, median and p90 per period per variable,
  plus the shift between them — same plain wording (`… days · average … · median … · warmest
  tenth above … °C`)
- Either period being empty, all-null, or constant for that variable renders the placeholder
  figure and an empty stats card — no exception

## Visual design (`assets/style.css`, `charts.py`)
The app is styled as a printed climate record rather than a dashboard: a warm paper ground
(`--paper #FBF8F3`), hairline rules instead of boxes, and one warm/cool accent pair that means
temperature and nothing else.

- **No cards.** `.card` is a block with a bottom hairline and vertical rhythm — no surface colour,
  no shadow, no radius, no border box. Sections are separated by rules and space.
- **Type.** Source Serif 4 for the wordmark, station name, chart titles and every stat value;
  the system sans for UI text and axis labels. Numbers are `tabular-nums` throughout.
- **Colour.** Warm `#B4442C` and cool `#2C6079` carry temperature — in the charts, in the stat
  values, and nowhere decorative. Precipitation is not temperature, so it does not borrow the
  cool: rain has its own lighter water blue `#7FA3B3`, lighter because a bar is a block of colour
  where a trace is a hairline. Neutrals are warm greys keyed to the paper, not blue-greys.
- **Charts sit on the page ground**: `plot_bgcolor` and `paper_bgcolor` are both transparent, so a
  figure has no plate of its own; only its gridlines (`#E5DCCD`) separate it from the page.
- Tone classes colour the stat **value**, replacing the coloured left border they used to draw.

## Architecture Decisions
- **Station index**: pre-built JSON committed to repo; never fetched at runtime by the app
- **Data fetching**: on-demand per department, cached locally as `.csv.gz`
- **DataFrame**: Polars throughout; only converted to lists/arrays at chart-build time
- **Time periods**: merged into a single DataFrame per department on load
- **Column names**: raw Météo-France names (TN, TX, …) renamed to readable names
  (temp_min, temp_max, …) inside `_parse()` — see `COLUMN_RENAME` in `data_loader.py`
- **Map tiles**: OpenStreetMap via `go.Scattermap` — no Mapbox token needed
- **Temporal aggregation**: `aggregate(df, granularity)` in `data_loader.py`; called in
  `station_page.update_charts` after the station/year filter. `Granularity.DAY` is the
  identity: its `truncate_expr` is `None` — there is no truncation string meaning "leave the
  daily rows alone" — and `aggregate` returns df unchanged. Week uses `dt.truncate("1w")`,
  month uses `"1mo"`.
- **Trend statistics**: `linear_trend` in `transforms.py` returns a `LinearTrend` dataclass
  `(slope, intercept, r_squared)`, or `None` when the data describes no line — fewer than two
  non-null points, or x-values that are all the same. Callers skip the trace and the stats row
  rather than drawing a fabricated flat trend. Slope and R² are surfaced in two places: hover tooltip on
  the trend trace, and a summary card rendered below the yearly extremes chart via a second
  Dash `Output`. When a provisional current year is shown, trend regression uses complete years only.
- **HotDayDefinition sum type**: `TmaxOnly | TmaxAndTmin` in `transforms.py`. `hot_day_predicate`
  maps a definition to a Polars expression; `yearly_hot_cold` accepts a `definition` parameter
  (default `DEFAULT_HOT_DAY = TmaxOnly(30.0)`). The `hot_cold_yearly_figure` function
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
  `Distribution` carries `n`, `mean`, `median`, `p90` — `p10` was computed and never rendered.
- **Station records** (`station_records -> StationRecords`): hottest day, coldest night, wettest
  day, strongest gust (each a `DatedValue | None`, reported at its most recent occurrence), and
  the longest run of hot days (`Streak | None`). `longest_streak` groups calendar-consecutive days
  with the gaps-and-islands trick (`DATE - int_range(len)`), so an unobserved day **breaks** the
  run rather than being assumed hot.
- **Latest vs normal** (`latest_vs_normal -> LatestVsNormal | None`): the station's most recent
  day with a temperature, the `day_normal` for that calendar day (mean Tmax/Tmin over a ±7-day
  window across `REFERENCE_PERIOD`, reusing `DayWindow` so the wrap-around case is
  already handled), and `day_rank`, the rank of that day among every year's reading for the same
  calendar day. A record too short or too recent to cover the reference falls back to its own span
  and reports which span it used, rather than reporting nothing. `REFERENCE_PERIOD = 1991–2020` is
  the WMO standard climate normal: the World Meteorological Organization defines a normal as the
  30-year mean over the most recent decade-aligned period, which has been 1991–2020 since 2021 and
  moves to 2001–2030 in 2031. Using it means "usual" here is the same baseline Météo-France and
  every other national service publish, so an anomaly read on this page is comparable with theirs.
- **Bounded department cache**: `_dept_cache` is an `OrderedDict[str, _CacheEntry]` holding at most
  `MAX_CACHED_DEPTS` (3) departments, least-recently-used evicted. A parsed department is 100–165 MB
  resident (every station, daily, since 1809), so the previous unbounded dict reached ~1.3 GB after
  ten departments and never released any of it. TTL semantics are unchanged.
- **Float32 measurements**: `LAT`/`LON` stay `Float64`; `RR`/`TN`/`TX`/`FFM`/`FXY` are parsed as
  `Float32`, halving a cached department at no readable precision cost (the source is reported to
  0.1). `TAMPLI` / `temp_amplitude` is no longer read at all — it was fetched, parsed, renamed and
  aggregated, and charted nowhere.
- **Packaged-app paths**: `_bundle_dir()` returns `sys._MEIPASS` when frozen and the repository
  root otherwise, so `DATA_DIR` and `ASSETS_DIR` resolve inside a PyInstaller bundle (the Dash app
  passes `assets_folder=ASSETS_DIR` explicitly for the same reason). `CACHE_DIR` honours
  `OMW_CACHE_DIR`, which the desktop shell points at the per-user data directory — an installed
  app must not write downloads next to its own binary.
- **`_station_df` / `_filtered_station_df` split**: `station_page.py` filters by station alone in
  `_station_df`; `_filtered_station_df` layers the page-level year range on top. The comparison
  callback uses the former, since it applies its own two year ranges.
- **±2σ bands**: `_add_sigma_band(fig, x, avg, std, fillcolor, n_sigma=2.0)` in `charts.py`
  draws a shaded inter-annual variability envelope using two invisible boundary `go.Scatter`
  traces with `fill="tonexty"`. Called by `monthly_avg_temp_figure` for both Tmax (orange
  fill `_TMAX_SIGMA_FILL`) and Tmin (blue fill `_TMIN_SIGMA_FILL`). The standard deviation
  is computed per calendar month via `pl.col(…).std()` in the `group_by("month").agg(…)`
  step.

## Desktop app

`npm start` runs the Electron shell against the checkout (`uv run python server.py`).
`npm run dist` produces an **installer that needs neither Python nor uv on the target machine**:

1. `npm run build:server` freezes `server.py` with PyInstaller (`server.spec`) into
   `sidecar/omw-server/`, bundling `data/stations.json`, `data/resources.json` and `assets/`.
   `collect_all` is required for dash, plotly, polars, narwhals and waitress — they ship templates,
   JS bundles and package metadata that no import graph reveals.
2. `electron-builder` packages `electron/` plus that directory as `extraResources` → `resources/server`,
   emitting an NSIS installer (Windows), a dmg (macOS) or an AppImage (Linux) under `release/`.

`electron/main.js` picks the sidecar by `app.isPackaged`: the frozen executable in
`process.resourcesPath/server` when installed, `uv run python server.py` in a checkout. Either way
it reads the chosen port from the sidecar's first line of stdout, waits for the socket, and sets
`OMW_CACHE_DIR` to `app.getPath('userData')/cache`. Ctrl/Cmd+Shift+R posts to `/api/refresh` and
reloads; it is a window-scoped `before-input-event` handler rather than a `globalShortcut`, so the
chord is not held hostage system-wide while the app runs, and `preventDefault` stops the default
menu's force-reload from reloading the page without refreshing the data behind it. If the sidecar
fails to start, the rejection surfaces as an error dialog and the app quits — it does not hang
windowless. Closing the window kills the sidecar.
