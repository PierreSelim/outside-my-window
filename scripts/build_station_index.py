#!/usr/bin/env python
"""Build data/stations.json: every station, where it is, and how long its record runs.

Run once (or after the station network changes):
    uv run python scripts/build_station_index.py
    uv run python scripts/build_station_index.py --fast   # positions only, no record span

The full run reads all three periods of every department (~1.5 GB of downloads, cached under
data/cache), because a station's first year is only knowable from the historical files. --fast
reads the latest period alone and omits first_year/n_years; the map then falls back to a single
marker colour.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import Period, _fetch, _parse, load_department, stations_from
from src.departments import DEPT_NAMES

OUTPUT = Path(__file__).parent.parent / "data" / "stations.json"


def _latest_only(dept: str) -> pl.DataFrame | None:
    path = _fetch(dept, Period.LATEST)
    return _parse(path) if path is not None else None


def _record_spans(df: pl.DataFrame) -> dict[int, tuple[int, int]]:
    """station_id → (first year, last year) of its observations."""
    spans = df.group_by("station_id").agg(
        pl.col("DATE").min().dt.year().alias("first"), pl.col("DATE").max().dt.year().alias("last")
    )
    return {row["station_id"]: (row["first"], row["last"]) for row in spans.to_dicts()}


def _entries(dept: str, active: pl.DataFrame, full: pl.DataFrame | None) -> list[dict[str, Any]]:
    """One entry per station still reporting, with its span measured over the whole record.

    Stations are taken from the latest period only: a station that closed in 1881 cannot say
    what it is doing outside your window today. The span comes from the merged periods, so a
    long-running station is still shown as long-running.
    """
    spans = _record_spans(full) if full is not None else {}
    entries = []
    for s in stations_from(active):
        entry: dict[str, Any] = {
            "station_id": s.station_id,
            "station_name": s.name,
            "dept": dept,
            "lat": s.lat,
            "lon": s.lon,
            "altitude": s.altitude,
        }
        span = spans.get(s.station_id)
        if span is not None:
            entry["first_year"], entry["last_year"] = span
            entry["n_years"] = span[1] - span[0] + 1
        entries.append(entry)
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true", help="latest period only; omit the record span")
    parser.add_argument("--depts", nargs="*", help="restrict to these department codes")
    parser.add_argument("--out", type=Path, default=OUTPUT, help="where to write the index")
    args = parser.parse_args()

    depts = args.depts or list(DEPT_NAMES.keys())
    all_stations: list[dict[str, Any]] = []

    for i, dept in enumerate(depts, 1):
        print(f"[{i:>3}/{len(depts)}] dept {dept:<4}", end=" … ", flush=True)
        active = _latest_only(dept)
        if active is None:
            print("no data")
            continue
        entries = _entries(dept, active, None if args.fast else load_department(dept))
        all_stations.extend(entries)
        print(f"{len(entries)} stations")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(all_stations, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nDone — {len(all_stations)} stations written to {OUTPUT}")


if __name__ == "__main__":
    main()
