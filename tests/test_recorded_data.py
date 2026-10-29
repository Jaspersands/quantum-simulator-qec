"""The recordings the README's tables are generated from, checked against the engine that ships:
each is stamped with the release that made it, and a few of its cheapest points are measured
again here and must come out the same, shot for shot. A recording an older engine made, and
this one does not reproduce, fails here before it reaches a table."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import numpy as np
import pytest

import stabilizer_qec as sq

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


def test_every_rerun_recording_is_from_this_release():
    tables = tool("readme_tables")
    assert tables.package_version() == sq.__version__
    assert tables.stale_data() == []
    for path in tables.RERUN_EACH_RELEASE:
        assert tables.major_minor(data(path)["version"]) == tables.major_minor(sq.__version__), path


def test_stale_data_names_old_and_unstamped_recordings(tmp_path, monkeypatch):
    tables = tool("readme_tables")
    (tmp_path / "data").mkdir()
    for name, doc in (("old", {"version": "2.1.9"}), ("unstamped", {"points": {}}), ("same", {"version": "2.2.0"}), ("patch", {"version": "2.2.3"}), ("newer", {"version": "2.3.0"})):
        (tmp_path / "data" / f"{name}.json").write_text(json.dumps(doc))
    monkeypatch.setattr(tables, "ROOT", tmp_path)
    paths = [f"data/{n}.json" for n in ("old", "unstamped", "same", "patch", "newer")]
    stale = tables.stale_data(paths, current="2.2.1")
    assert [m.split(" ")[0] for m in stale] == ["data/old.json", "data/unstamped.json"]
    assert "recorded by 2.1.9, the package is 2.2.1" in stale[0]
    assert "recorded by an unstamped version" in stale[1]


# The rows the memory-budgeted search settles in under a second, among them the five the 1.5.0
# recording left blank (its search kept every set it visited and gave up).
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


@pytest.mark.parametrize("key", ["d3/T2/p0.003/correlated", "d3/T2/p0.003/plain", "d5/T2/p0.003/correlated", "d3/T2/p0.002/correlated", "d3/T3/p0.003/correlated"])
def test_surgery_timing_points_reproduce(key):
    pytest.importorskip("stim")
    pytest.importorskip("pymatching")
    surgery = tool("surgery")
    rec = data("data/surgery/results.json")["points"][key]
    shots, counts = surgery.run_point(rec["d"], rec["merged"], rec["p"], rec["matcher"] == "correlated", 2_000_000, 300.0)
    assert (shots, counts) == (rec["shots"], rec["failures"])


@pytest.mark.parametrize("key", ["cnot/d3/T2/p0.003/z/correlated", "cnot/d3/T3/p0.003/z/correlated"])
def test_surgery_program_points_reproduce(key):
    pytest.importorskip("stim")
    pytest.importorskip("pymatching")
    surgery = tool("surgery")
    rec = data("data/surgery/programs.json")["points"][key]
    i, (_, _, text, corr, window) = next((i, p) for i, p in enumerate(surgery.program_points()) if p[0] == key)
    shots, counts, _ = surgery.program_point(text, 7_000_003 * (i + 1), corr, 2_000_000, 300.0, window)
    assert (shots, counts) == (rec["shots"], rec["failures"])


@pytest.mark.parametrize("eps", ["0.01", "0.05"])
def test_distillation_twirl_column_reproduces(eps):
    # The twirl of Tsim's distillation circuit, whose R_X and T_DAG act on five qubits at once:
    # the recording must be the twirl that keeps each such gate once.
    clifford_t = tool("clifford_t")
    doc = data("data/clifford_t/distill.json")
    plain = clifford_t.ours_tsim.Circuit(clifford_t.distillation_text(p=0.0) + "\n" + "\n".join(f"OBSERVABLE_INCLUDE({k}) rec[{k - 5}]" for k in range(5)))
    _, ref = plain.reference_detector_and_observable_signs()
    ref = np.asarray(ref, dtype=bool)
    d, o = clifford_t.over_rotate(plain, float(eps)).twirled().compile_stabilizer_rank_sampler(seed=0).sample(doc["shots"], separate_observables=True)
    outcomes = o ^ ref[None, :]
    accept = ~d.any(axis=1) & np.all(outcomes[:, 1:5] == np.array([True, False, True, True]), axis=1)
    got = clifford_t.rates(d, outcomes, None, accept)
    rec = doc["unencoded"][eps]["twirl"]
    assert (got["observables"][0], got["acceptance"]) == (rec["infidelity"], rec["acceptance"])
