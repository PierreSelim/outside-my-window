# Outside My Window

[![CI](https://github.com/PierreSelim/outside-my-window/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/PierreSelim/outside-my-window/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/PierreSelim/outside-my-window/branch/main/graph/badge.svg)](https://codecov.io/gh/PierreSelim/outside-my-window)

Visualization app for French daily climatological data from Météo-France ([data.gouv.fr](https://www.data.gouv.fr/datasets/donnees-climatologiques-de-base-quotidiennes)).

Select a department, a weather station, and a year range to explore temperature, precipitation, and wind history, or compare the temperature distribution of two periods.

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)

## Build the station index

`data/stations.json` is committed, so this is only needed when the station network changes:

```bash
uv run python scripts/build_station_index.py          # full: positions + record span
uv run python scripts/build_station_index.py --fast   # positions only, much quicker
```

The full run reads all three periods of every department (~1.5 GB of downloads, cached under
`data/cache/`, one request at a time) because a station's first year is only knowable from the
historical files. It records `first_year` / `last_year` / `n_years`, which the map uses to colour
markers by record length. `--fast` reads the latest period alone and omits those fields; the map
then falls back to a single colour. Either way the index lists the stations still reporting today.

## Build the resource index

Downloads use stable data.gouv.fr permalinks resolved from `data/resources.json`. Rebuild it only if the dataset re-issues resources with new ids:

```bash
uv run python scripts/build_resource_index.py
```

## Run the app

```bash
uv run python app.py
```

Then open [http://127.0.0.1:8050](http://127.0.0.1:8050) in your browser.

The landing page is a map of all stations. Click a station to view its temperature, precipitation, and wind history. Department data is fetched from Météo-France on first visit and cached locally under `data/cache/`.

## Run as a desktop app (Electron)

**Additional requirements:** [Node.js](https://nodejs.org/) (includes npm)

```bash
npm install        # install Electron + electron-builder (one-time)
npm start          # launch the desktop window against this checkout
```

This starts a local [waitress](https://docs.pylonsproject.org/projects/waitress/) server on a free
port and opens it in an Electron window. The Python sidecar is killed when you close the window.

Data is cached for 6 hours; press **Ctrl+Shift+R** (**Cmd+Shift+R** on macOS) to force-refresh the
latest data immediately instead of waiting for the cache to expire.

The `uv run python app.py` workflow above still works for browser-based development.

## Build an installable desktop app

```bash
npm run dist       # installer for the current platform, into release/
npm run dist:dir   # unpacked app only, into release/, for a quick check
```

`npm run dist` first freezes the Python server with PyInstaller into `sidecar/omw-server/`
(`npm run build:server` does that step alone), then has electron-builder package it as an
NSIS installer on Windows, a dmg on macOS, or an AppImage on Linux.

**The result needs neither Python nor uv on the target machine** — the interpreter, the
dependencies, the station index and the stylesheet are all inside the bundle. Expect ~600 MB
unpacked, most of it Electron and Chromium. Downloaded weather data is written to the per-user
application data directory, never next to the installed binary.

## Run the tests

```bash
uv run pytest tests/ --cov=src --cov-report=term-missing
```

## Features

### Station map
Browse every Météo-France station still reporting, across metropolitan France and the overseas
departments. Markers are coloured by how long the station's record runs, so a 100-year series is
visible at a glance. Search by name from the map or from any station page — the search is the one
control that crosses department boundaries.

![Station map](docs/screenshots/outside-my-window_station_map.png)

### Station header
Every station page opens with what the station is doing *now*: the latest observation read against
the 1991–2020 WMO standard normal for that same calendar day, the anomaly, and where the day
ranks among every year of the record. Below it, the all-time records — hottest day, coldest
night, wettest day, strongest gust, longest run of hot days.

### Daily observations
Explore daily temperature (min/max band), precipitation, and wind for any station and year range.
Switch between daily, weekly, and monthly granularity: temperatures average, precipitation totals
(`mm/week`, `mm/month`), and gusts report the period's peak. Gaps in the record are drawn as gaps.

![Daily observations](docs/screenshots/outside-my-window_station_observation.png)

### Yearly extremes
Track hot days (Tmax ≥ 30 °C by default, seven other definitions available), tropical nights
(Tmin ≥ 20 °C) and frost days (Tmin < 0 °C) year by year. Years in which the station observed
under 90 % of days are excluded and shaded grey rather than counted as cool ones. Optional trend
lines show the long-term evolution with slope and R², computed on the fully observed years only.

![Yearly extremes](docs/screenshots/outside-my-window_station_extremes.png)

### Monthly averages
Compare average Tmin and Tmax by month of year, either over the full record or broken down by
decade — pick which decades to overlay rather than reading all of them at once.

![Monthly averages](docs/screenshots/outside-my-window_station_averages.png)

### Comparison
Overlay the smoothed probability density of daily Tmax and Tmin for two year ranges over the same
season window (e.g. 1 Jun – 31 Aug, 1961–1990 against 1995–2024), so the shift of the whole
distribution — and of its hot tail — is directly readable. Curves are Gaussian kernel density
estimates with a Silverman bandwidth. A stats line below gives the mean, median and p90 of each
period plus the shift between them.
