#!/usr/bin/env python
"""Build data/stations.json by fetching the latest-period file for each department.

Run once (or after the station network changes):
    uv run python scripts/build_station_index.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import Period, _fetch, _parse, stations_from
from src.departments import DEPT_NAMES

OUTPUT = Path(__file__).parent.parent / "data" / "stations.json"


def main() -> None:
    all_stations: list[dict] = []
    depts = list(DEPT_NAMES.keys())

    for i, dept in enumerate(depts, 1):
        print(f"[{i:>3}/{len(depts)}] dept {dept:<4}", end=" … ", flush=True)

        path = _fetch(dept, Period.LATEST)
        if path is None:
            print("no file")
            continue

        df = _parse(path)
        if df is None:
            print("parse error")
            continue

        stations = stations_from(df)
        for s in stations:
            all_stations.append({
                "station_id": s.num_poste,
                "station_name": s.name,
                "dept": dept,
                "lat": s.lat,
                "lon": s.lon,
                "altitude": s.altitude,
            })
        print(f"{len(stations)} stations")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(all_stations, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nDone — {len(all_stations)} stations written to {OUTPUT}")


if __name__ == "__main__":
    main()
