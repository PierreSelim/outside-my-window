# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run the Dash app
uv run python app.py          # then open http://127.0.0.1:8050

# Build the station index (once, or when station network changes)
uv run python scripts/build_station_index.py

# Tests
uv run pytest tests/ --cov=src --cov-report=term-missing   # full suite with coverage
uv run pytest tests/test_transforms.py                      # single file
uv run pytest tests/test_transforms.py::test_linear_trend   # single test

# Lint & format
uv run ruff check src/ tests/          # lint
uv run ruff format src/ tests/         # format
uv run mypy src/                       # type-check
```

## Architecture

Data flows in one direction: **fetch → parse → transform → chart → page**.

```
data_loader.py  →  transforms.py  →  charts.py  →  pages/
```

- **`src/data_loader.py`** — all I/O. Fetches `.csv.gz` files from Météo-France by `(dept, Period)`, caches them on disk under `data/cache/`, parses them into Polars DataFrames with renamed columns, merges all three time periods per department, and exposes a thread-safe in-process cache (`_dept_cache`) for hot reloads. Also owns the domain types: `Station`, `Period`, `Granularity`.
- **`src/transforms.py`** — pure computation. `yearly_hot_cold()` and `linear_trend()` operate on Polars DataFrames/plain lists with no I/O.
- **`src/charts.py`** — Plotly figure builders. One function per chart type; they receive a filtered Polars DataFrame and return a `go.Figure`. Never loads data itself.
- **`src/departments.py`** — static mapping of department codes to names.
- **`src/pages/map_page.py`** — landing page. Reads the pre-built `data/stations.json` (committed to repo, built by `scripts/build_station_index.py`). Click → navigate to `/station?dept=…&station=…`.
- **`src/pages/station_page.py`** — detail page. Reads dept+station from URL query params, calls `load_department_cached()`, registers three Dash callbacks for observations, yearly extremes, and monthly averages.
- **`app.py`** — Dash shell: `dcc.Location` + a single top-level callback routing `/` to `map_page.layout()` and `/station` to `station_page.layout(search)`. Both pages register their own callbacks via `register_callbacks(app)`.

### Two-layer caching

1. **Disk**: raw `.csv.gz` files in `data/cache/` — skipped on subsequent requests.
2. **In-process**: parsed `pl.DataFrame` per department in `_dept_cache` (dict), protected by per-key `threading.Lock` so concurrent Dash requests for the same department don't double-load.

### DataFrame schema

After `_parse()`, every DataFrame has these columns (raw Météo-France names are renamed inside `_parse()` via `COLUMN_RENAME`):

| Column | Type | Notes |
|---|---|---|
| `station_id` | `Int32` | |
| `station_name` | `String` | |
| `lat`, `lon` | `Float64` | |
| `altitude` | `Int32` | |
| `DATE` | `Date` | parsed from `AAAAMMJJ` |
| `temp_min`, `temp_max`, `temp_amplitude` | `Float64` | nullable |
| `precipitation` | `Float64` | nullable |
| `wind_mean`, `wind_gust` | `Float64` | nullable; many stations lack wind data |

## Domain types

- **`Period(str, Enum)`** — `HISTORICAL` / `MODERN` / `LATEST`; used to build remote URLs and cache paths.
- **`Granularity(Enum)`** — `DAY` / `WEEK` / `MONTH`; value is a `Truncated` dataclass holding the Polars truncate expression and a chart title suffix. `Granularity.DAY` is the identity (no aggregation).
- **`Station(frozen dataclass)`** — `station_id`, `name`, `lat`, `lon`, `altitude`.
- **`LinearTrend(frozen dataclass)`** — `slope`, `intercept`, `r_squared`; produced by `linear_trend()`.

## Documentation

Keep documentation up to date as part of every change:

- **`SPEC.md`**: update whenever a new feature is added or an existing one changes — describe behaviour, data flow, and any architectural decisions made. Do this before moving on.
- **`README.md`**: update whenever the build process changes (new dependencies, new `uv` commands, environment setup steps) or a new CLI script is added under `scripts/`.

## Testing

Every new module or function added to `src/` must be covered by unit tests in `tests/`:

- Use pytest with plain functions (no `TestCase`)
- Use fixtures in `conftest.py` for shared test data (e.g. `sample_df`)
- Mock network calls — tests must never hit the network
- Run with coverage: `uv run pytest tests/ --cov=src --cov-report=term-missing`

## Python coding standards

- **Formatter**: ruff, line length 120
- **Type hints**: all function signatures and variable declarations where non-obvious
- **Data modelling**: use `dataclass` and `Enum` to represent domain concepts; avoid raw dicts for structured data
- **Error handling**: avoid using exceptions for business logic flow control; prefer returning `None`, a result type, or an `Enum` variant to signal expected failure states
- **Style**: prefer functional programming (map, filter, comprehensions, pure functions) where it keeps the code readable; do not over-abstract or force FP patterns when a simple loop is clearer
