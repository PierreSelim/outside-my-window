# Outside My Window

Visualization app for French daily climatological data from Météo-France ([data.gouv.fr](https://www.data.gouv.fr/datasets/donnees-climatologiques-de-base-quotidiennes)).

Select a department, a weather station, and a year range to explore temperature, precipitation, and wind history.

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)

## Build the station index

The map requires a pre-built index of all stations. Run this once (and again if the station network changes):

```bash
uv run python scripts/build_station_index.py
```

This fetches the latest-period file for each department sequentially and writes `data/stations.json`.

## Run the app

```bash
uv run python app.py
```

Then open [http://127.0.0.1:8050](http://127.0.0.1:8050) in your browser.

The landing page is a map of all stations. Click a station to view its temperature, precipitation, and wind history. Department data is fetched from Météo-France on first visit and cached locally under `data/cache/`.

## Run the tests

```bash
uv run pytest tests/ --cov=src --cov-report=term-missing
```
