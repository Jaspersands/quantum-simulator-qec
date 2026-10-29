"""The recordings the README's tables are generated from, checked against the engine that ships:
each is stamped with the release that made it, and a few of its cheapest points are measured
again here and must come out the same, shot for shot. A recording an older engine made, and
this one does not reproduce, fails here before it reaches a table."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"


def tool(name):
    """A module from tools/, imported by path (the tools import their neighbours by name)."""
    if str(TOOLS) not in sys.path:
        sys.path.insert(0, str(TOOLS))
    spec = importlib.util.spec_from_file_location(f"recorded_{name}", TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def data(path):
    return json.loads((ROOT / path).read_text())


# The rows whose search the 1.5.0 recording left blank (its search kept every set it visited and
# gave up); the memory-budgeted search finds each in under a second.
FAST_SEARCH_ROWS = [
    ("surface, rotated (Stim's)", 3), ("surface, rotated (Stim's)", 5), ("surface, unrotated (Stim's)", 5),
    ("colour code XYZ (Stim's)", 5), ("colour code XYZ (Stim's)", 7), ("surface, rotated, SD6", 5), ("surface, XZZX, SD6", 5),
]


@pytest.mark.parametrize("memory, d", FAST_SEARCH_ROWS)
def test_distance_table_search_column_is_the_search(memory, d):
    distances = tool("distances")
    doc = data("data/distances.json")
    row = next(m for m in doc["memories"] if (m["memory"], m["code_distance"]) == (memory, d))
    make = next(make for name, dd, make in distances.memories(False) if (name, dd) == (memory, d))
    circuit = make()
    assert row["faults"] == circuit.detector_error_model().num_errors
    found = circuit.search_for_undetectable_logical_errors(**doc["search_limits"])
    assert row["search"] == len(found)
    assert row["search"] == row["exact"]
