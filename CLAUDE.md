# CLAUDE.md

## Project spec
Keep `SPEC.md` up to date as decisions are made. Any time a new architectural choice, data finding, dependency, or structural change is agreed upon, update the relevant section of `SPEC.md` before moving on.

## Python coding standards
Write Python code to a senior developer standard:

- **Formatter**: ruff, line length 120
- **Type hints**: all function signatures and variable declarations where non-obvious
- **Data modelling**: use `dataclass` and `Enum` to represent domain concepts; avoid raw dicts for structured data
- **Error handling**: avoid using exceptions for business logic flow control; prefer returning `None`, a result type, or an `Enum` variant to signal expected failure states
- **Style**: prefer functional programming (map, filter, comprehensions, pure functions) where it keeps the code readable; do not over-abstract or force FP patterns when a simple loop is clearer
