"""Circuit distance: the smallest set of faults that flips an observable and no detector, found
as Stim finds it: over the graph-like pieces of a decomposed model (``shortest_graphlike_error``,
Stim's own answer, fault for fault and character for character) and by Stim's search over
hyperedges (``search_for_undetectable_logical_errors``)."""

from __future__ import annotations

import pytest

import stabilizer_qec as sq

stim = pytest.importorskip("stim")
# Stim 1.16 orders an explained fault's locations as these do; 1.15 (the last for Python 3.9) does not.
EXPLAINS_AS_STIM = tuple(int(x) for x in stim.__version__.split(".")[:2]) >= (1, 16)

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
    assert found.num_errors == d
    assert undetected_logical(str(found))
    # Stim's own faults, in Stim's order, explained as Stim explains them.
    assert str(found) == str(c.detector_error_model(decompose_errors=True).shortest_graphlike_error())
    for canonicalize in (True, False) if EXPLAINS_AS_STIM else ():
        want = [str(e) for e in c.shortest_graphlike_error(canonicalize_circuit_errors=canonicalize)]
        assert [str(e).rstrip("\n") for e in ours.shortest_graphlike_error(canonicalize_circuit_errors=canonicalize)] == want


GENERATED = ["repetition_code:memory", "surface_code:rotated_memory_x", "surface_code:unrotated_memory_z", "color_code:memory_xyz"]


@pytest.mark.parametrize("task", GENERATED)
@pytest.mark.parametrize("decompose", [True, False])
def test_graphlike_models_are_stims_character_for_character(task, decompose):
    for d in (3, 5, 7):
        model = noisy(task, d).detector_error_model(decompose_errors=decompose, ignore_decomposition_failures=True)
        for ignore in (True, False):
            try:
                want = str(model.shortest_graphlike_error(ignore_ungraphlike_errors=ignore))
            except ValueError:
                with pytest.raises(ValueError):
                    sq.DetectorErrorModel(str(model)).shortest_graphlike_error(ignore_ungraphlike_errors=ignore)
                continue
            assert str(sq.DetectorErrorModel(str(model)).shortest_graphlike_error(ignore_ungraphlike_errors=ignore)) == want, (task, d, ignore)


def test_graphlike_random_models_and_circuits_are_stims():
    import random

    from test_stim_gates import random_case

    rng = random.Random(5)
    for _ in range(300):
        nd = rng.randint(2, 12)
        lines = []
        for _ in range(rng.randint(2, 25)):
            ds = rng.sample(range(nd), min(rng.choice([1, 1, 2, 2, 2, 3]), nd))
            obs = [f"L{j}" for j in range(2) if rng.random() < 0.25]
            lines.append(f"error({rng.choice([0.1, 0.01, 0])}) " + " ".join([f"D{x}" for x in ds] + obs))
        text = "\n".join(lines + [f"detector D{nd - 1}"])
        for ignore in (True, False):
            try:
                want = str(stim.DetectorErrorModel(text).shortest_graphlike_error(ignore_ungraphlike_errors=ignore))
            except ValueError:
                with pytest.raises(ValueError):
                    sq.DetectorErrorModel(text).shortest_graphlike_error(ignore_ungraphlike_errors=ignore)
                continue
            assert str(sq.DetectorErrorModel(text).shortest_graphlike_error(ignore_ungraphlike_errors=ignore)) == want, text
    for seed in range(100 if EXPLAINS_AS_STIM else 0):
        text = random_case(seed, disjoint=False)
        try:
            want = [str(e) for e in stim.Circuit(text).shortest_graphlike_error()]
        except ValueError:
            continue
        assert [str(e).rstrip("\n") for e in sq.Circuit(text).shortest_graphlike_error()] == want, text


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
