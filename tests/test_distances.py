"""The exact distance of a model by integer programming, checked against the graph-like search
where both apply, against brute force on small models, and on codes of known distance."""

from __future__ import annotations

import itertools

import pytest

import stabilizer_qec as sq

pytest.importorskip("scipy")


@pytest.mark.parametrize("d", [3, 5])
def test_milp_agrees_with_graphlike_on_surface_codes(d):
    c = sq.Circuit.generated("surface_code:rotated_memory_z", distance=d, rounds=d, after_clifford_depolarization=0.001)
    dem = c.detector_error_model(decompose_errors=True)
    assert dem.distance() == dem.shortest_graphlike_error().num_errors == d
    assert dem.distance(method="graphlike") == d


def brute_force(faults):
    for k in range(1, len(faults) + 1):
        for chosen in itertools.combinations(faults, k):
            dets, obs = set(), set()
            for f_dets, f_obs in chosen:
                dets ^= f_dets
                obs ^= f_obs
            if not dets and obs:
                return k
    return None


def test_milp_equals_brute_force_on_random_models():
    import random

    rng = random.Random(7)
    for _ in range(30):
        faults = []
        for _ in range(rng.randint(3, 9)):
            dets = frozenset(rng.sample(range(5), rng.randint(1, 3)))
            obs = frozenset({0}) if rng.random() < 0.3 else frozenset()
            faults.append((dets, obs))
        text = "\n".join("error(0.1) " + " ".join([f"D{d}" for d in sorted(f[0])] + [f"L{k}" for k in sorted(f[1])]) for f in faults)
        want = brute_force(faults)
        dem = sq.DetectorErrorModel(text)
        if want is None:
            with pytest.raises(ValueError, match="no undetectable"):
                dem.distance()
        else:
            assert dem.distance() == want, text


@pytest.mark.parametrize("d,distance", [(3, 2), (5, 3)])
def test_colour_code_memory_distances(d, distance):
    # Stim's generated XYZ colour-code memory has circuit distance below the code's: two
    # correlated two-qubit faults at d = 3, three at d = 5. The graph-like search cannot see it
    # (the faults are hyperedges); the integer program proves it, and Stim's own search agrees.
    c = sq.Circuit.generated("color_code:memory_xyz", distance=d, rounds=d, after_clifford_depolarization=0.001)
    assert c.detector_error_model().distance() == distance
    stim = pytest.importorskip("stim")
    kw = dict(dont_explore_detection_event_sets_with_size_above=6, dont_explore_edges_with_degree_above=6, dont_explore_edges_increasing_symptom_degree=False)
    assert len(stim.Circuit(str(c)).search_for_undetectable_logical_errors(**kw)) == distance


def test_bad_method():
    with pytest.raises(ValueError, match="method"):
        sq.DetectorErrorModel("error(0.1) D0 L0").distance(method="guess")
