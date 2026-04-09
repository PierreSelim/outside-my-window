# Outside My Window

Visualization app for French daily climatological data from Météo-France ([data.gouv.fr](https://www.data.gouv.fr/datasets/donnees-climatologiques-de-base-quotidiennes)).

Select a department, a weather station, and a year range to explore temperature, precipitation, and wind history.

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)

## Run the app

```bash
uv run python app.py
```

Then open [http://127.0.0.1:8050](http://127.0.0.1:8050) in your browser.

Data files are fetched from Météo-France on first use and cached locally under `data/cache/`.

## Run the tests

```bash
uv run pytest tests/ --cov=src --cov-report=term-missing
```
