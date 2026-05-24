# CLAUDE.md

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
Write Python code to a senior developer standard:

- **Formatter**: ruff, line length 120
- **Type hints**: all function signatures and variable declarations where non-obvious
- **Data modelling**: use `dataclass` and `Enum` to represent domain concepts; avoid raw dicts for structured data
- **Error handling**: avoid using exceptions for business logic flow control; prefer returning `None`, a result type, or an `Enum` variant to signal expected failure states
- **Style**: prefer functional programming (map, filter, comprehensions, pure functions) where it keeps the code readable; do not over-abstract or force FP patterns when a simple loop is clearer
