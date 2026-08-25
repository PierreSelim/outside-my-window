#!/usr/bin/env python
"""Build data/resources.json — a (dept, period) → data.gouv.fr resource-id map.

The download URLs are stable permalinks (``/api/1/datasets/r/{id}``) that
redirect to whatever storage host is live, so a host migration never breaks us.
Run once, or when the dataset re-issues resources with new ids:

    uv run python scripts/build_resource_index.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import Period

DATASET_API = "https://www.data.gouv.fr/api/1/datasets/donnees-climatologiques-de-base-quotidiennes/"
OUTPUT = Path(__file__).parent.parent / "data" / "resources.json"

_TITLE_RE = re.compile(r"QUOT_departement_(\w+)_periode_(.+)_RR-T-Vent$")

# Catalog period tokens → our Period enum
_TOKEN_TO_PERIOD: dict[str, Period] = {
    "avant-1949": Period.HISTORICAL,
    "1950-2024": Period.MODERN,
    "2025-2026": Period.LATEST,
}


def main() -> None:
    resources = requests.get(DATASET_API, timeout=30).json()["resources"]

    index: dict[str, dict[str, str]] = {}
    for r in resources:
        m = _TITLE_RE.match(r["title"])
        if m is None:
            continue
        dept, token = m.group(1), m.group(2)
        period = _TOKEN_TO_PERIOD.get(token)
        if period is None:
            continue
        index.setdefault(dept, {})[period.value] = r["id"]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Done — {len(index)} departments written to {OUTPUT}")


if __name__ == "__main__":
    main()
