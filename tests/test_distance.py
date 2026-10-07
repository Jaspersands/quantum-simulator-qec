"""Circuit distance: the smallest set of faults that flips an observable and no detector, found
as Stim finds it: over the graph-like pieces of a decomposed model (``shortest_graphlike_error``)
and by Stim's search over hyperedges (``search_for_undetectable_logical_errors``)."""

from __future__ import annotations

import pytest

import stabilizer_qec as sq

stim = pytest.importorskip("stim")

GRAPHLIKE = [(t, d) for t in ["repetition_code:memory", "surface_code:rotated_memory_x", "surface_code:rotated_memory_z", "surface_code:unrotated_memory_z"] for d in (3, 5)]


def undetected_logical(dem_text):
    """The faults' symptoms cancel, and they flip an observable."""
    dets, obs = set(), set()
    for line in dem_text.splitlines():
        if line.startswith("error"):
            for t in line.split(")", 1)[1].replace("^", " ").split():
                (dets if t[0] == "D" else obs).symmetric_difference_update({t})
    return not dets and bool(obs)


def noisy(task, d):
    return stim.Circuit.generated(task, distance=d, rounds=d, after_clifford_depolarization=0.001, before_measure_flip_probability=0.001)


@pytest.mark.parametrize("task,d", GRAPHLIKE)
def test_graphlike_distance_equals_stims(task, d):
    c = noisy(task, d)
    ours = sq.Circuit(str(c))
    found = ours.detector_error_model(decompose_errors=True).shortest_graphlike_error()
    assert found.num_errors == len(c.shortest_graphlike_error()) == d
    assert undetected_logical(str(found))
    explained = ours.shortest_graphlike_error()
    assert len(explained) == d
    assert all(e.circuit_error_locations for e in explained)


@pytest.mark.parametrize("d", [3, 5, 7])
def test_graphlike_ignores_hyperedges_as_stim_does(d):
    # The colour code's models have hyperedges and observable-only pieces, both skipped.
    c = stim.Circuit.generated("color_code:memory_xyz", distance=d, rounds=d, after_clifford_depolarization=0.001)
    ours = sq.Circuit(str(c))
    assert len(ours.shortest_graphlike_error()) == len(c.shortest_graphlike_error())
    with pytest.raises(ValueError):
        sq.DetectorErrorModel("error(0.1) D0 D1 D2 L0").shortest_graphlike_error(ignore_ungraphlike_errors=False)


def test_canonicalized_errors_have_one_location():
    c = noisy("surface_code:rotated_memory_z", 3)
    for e in sq.Circuit(str(c)).shortest_graphlike_error(canonicalize_circuit_errors=True):
        assert len(e.circuit_error_locations) == 1


def test_no_logical_error_is_an_error():
    with pytest.raises(ValueError, match="undetectable"):
        sq.DetectorErrorModel("error(0.1) D0 D1\nerror(0.1) D1").shortest_graphlike_error()


SEARCH = dict(dont_explore_detection_event_sets_with_size_above=4, dont_explore_edges_with_degree_above=4, dont_explore_edges_increasing_symptom_degree=False)

HYPER = [
    ("color d=3", lambda: stim.Circuit.generated("color_code:memory_xyz", distance=3, rounds=3, after_clifford_depolarization=0.001)),
    ("color d=5", lambda: stim.Circuit.generated("color_code:memory_xyz", distance=5, rounds=3, after_clifford_depolarization=0.001)),
    ("surface d=5", lambda: noisy("surface_code:rotated_memory_x", 5)),
    ("steane", lambda: stim.Circuit(str(sq.CssCode.color_code(3).memory_circuit(2, 0.001)))),
    ("bb72", lambda: stim.Circuit(str(sq.BivariateBicycleCode("72").memory_circuit(1, 0.001)))),
]


@pytest.mark.parametrize("name,make", HYPER)
def test_heuristic_search_matches_stims_length(name, make):
    c = make()
    for kw in (SEARCH, {**SEARCH, "dont_explore_edges_increasing_symptom_degree": True}):
        try:
            want = len(c.search_for_undetectable_logical_errors(**kw))
        except ValueError:
            # Where Stim's search finds nothing within its limits, neither does this one.
            with pytest.raises(ValueError, match="no undetectable"):
                sq.Circuit(str(c)).search_for_undetectable_logical_errors(**kw)
            continue
        got = sq.Circuit(str(c)).search_for_undetectable_logical_errors(**kw)
        assert len(got) == want, (name, kw)
        assert all(e.circuit_error_locations for e in got)


def test_search_on_models_and_limits():
    dem = sq.DetectorErrorModel("error(0.1) D0 D1 D2\nerror(0.1) D0\nerror(0.1) D1 D2 L0")
    found = dem.search_for_undetectable_logical_errors(**SEARCH)
    assert found.num_errors == 3 and undetected_logical(str(found))
    with pytest.raises(ValueError, match="undetectable"):
        dem.search_for_undetectable_logical_errors(**{**SEARCH, "dont_explore_edges_with_degree_above": 2})
