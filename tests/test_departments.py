from __future__ import annotations

from src.departments import DEPT_NAMES


def test_dept_names_contains_metropolitan_departments() -> None:
    for code in ["01", "31", "75", "95"]:
        assert code in DEPT_NAMES


def test_dept_names_contains_corsica() -> None:
    assert "2A" in DEPT_NAMES
    assert "2B" in DEPT_NAMES


def test_dept_names_contains_overseas() -> None:
    for code in ["971", "972", "973", "974", "976"]:
        assert code in DEPT_NAMES


def test_dept_names_excludes_20() -> None:
    # Dept 20 was split into 2A/2B — should not appear
    assert "20" not in DEPT_NAMES


def test_dept_names_values_are_non_empty_strings() -> None:
    assert all(isinstance(name, str) and name for name in DEPT_NAMES.values())
