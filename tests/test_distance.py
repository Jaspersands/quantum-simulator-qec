"""Circuit distance: the smallest set of faults that flips an observable and no detector, found
as Stim finds it: over the graph-like pieces of a decomposed model (``shortest_graphlike_error``,
Stim's own answer, fault for fault and character for character) and by Stim's search over
hyperedges (``search_for_undetectable_logical_errors``)."""

from __future__ import annotations

import numpy as np
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
    with pytest.raises(ValueError, match="Failed to find any graphlike logical errors"):
        sq.DetectorErrorModel("error(0.1) D0 D1\nerror(0.1) D1").shortest_graphlike_error()


SEARCH = dict(dont_explore_detection_event_sets_with_size_above=4, dont_explore_edges_with_degree_above=4, dont_explore_edges_increasing_symptom_degree=False)

HYPER = [
    ("color d=3", lambda: stim.Circuit.generated("color_code:memory_xyz", distance=3, rounds=3, after_clifford_depolarization=0.001)),
    ("color d=5", lambda: stim.Circuit.generated("color_code:memory_xyz", distance=5, rounds=3, after_clifford_depolarization=0.001)),
    ("surface d=5", lambda: noisy("surface_code:rotated_memory_x", 5)),
    ("steane", lambda: stim.Circuit(str(sq.CssCode.color_code(3).memory_circuit(2, 0.001)))),
    ("bb72", lambda: stim.Circuit(str(sq.BivariateBicycleCode("72").memory_circuit(1, 0.001)))),
]


def _terms(errors):
    return [[str(t.dem_target) for t in e.dem_error_terms] for e in errors]


@pytest.mark.parametrize("name,make", HYPER)
def test_heuristic_search_finds_stims_faults(name, make):
    # Stim's steps, so Stim's answer, fault for fault (among equally short ones too).
    c = make()
    for kw in (SEARCH, {**SEARCH, "dont_explore_edges_increasing_symptom_degree": True}, {**SEARCH, "dont_explore_detection_event_sets_with_size_above": 2, "dont_explore_edges_with_degree_above": 2}):
        try:
            want = c.search_for_undetectable_logical_errors(**kw)
        except ValueError as ex:
            # Where Stim's search finds nothing within its limits, neither does this one.
            with pytest.raises(ValueError) as ours:
                sq.Circuit(str(c)).search_for_undetectable_logical_errors(**kw)
            assert str(ours.value) == str(ex)
            continue
        got = sq.Circuit(str(c)).search_for_undetectable_logical_errors(**kw)
        assert _terms(got) == _terms(want), (name, kw)
        assert all(e.circuit_error_locations for e in got)


@pytest.mark.parametrize("seed", range(60))
def test_heuristic_search_finds_stims_faults_on_random_models(seed):
    # A circuit with any model: a qubit per detector and observable, each fault an E on them.
    rng = np.random.default_rng(seed)
    n, nobs = int(rng.integers(2, 10)), int(rng.integers(1, 4))
    lines = []
    for _ in range(int(rng.integers(2, 19))):
        dets = sorted(rng.choice(n, size=int(rng.integers(1, min(4, n) + 1)), replace=False).tolist())
        obs = [k for k in range(nobs) if rng.random() < 0.3]
        lines.append("E(0.01) " + " ".join(f"X{q}" for q in dets + [n + k for k in obs]))
    lines.append("M " + " ".join(map(str, range(n + nobs))))
    lines += [f"DETECTOR rec[{q - n - nobs}]" for q in range(n)] + [f"OBSERVABLE_INCLUDE({k}) rec[{k - nobs}]" for k in range(nobs)]
    c = stim.Circuit("\n".join(lines))
    for limits in [(4, 4, False), (3, 3, True), (6, 3, False), (2, 2, False)]:
        kw = dict(zip(SEARCH, limits))
        try:
            want = _terms(c.search_for_undetectable_logical_errors(**kw))
        except ValueError as ex:
            want = str(ex)
        try:
            got = _terms(sq.Circuit(str(c)).search_for_undetectable_logical_errors(**kw))
        except ValueError as ex:
            got = str(ex)
        assert got == want, (limits, str(c))


def test_search_on_models_and_limits():
    dem = sq.DetectorErrorModel("error(0.1) D0 D1 D2\nerror(0.1) D0\nerror(0.1) D1 D2 L0")
    found = dem.search_for_undetectable_logical_errors(**SEARCH)
    assert found.num_errors == 3 and undetected_logical(str(found))
    with pytest.raises(ValueError, match="Failed to find any logical errors"):
        dem.search_for_undetectable_logical_errors(**{**SEARCH, "dont_explore_edges_with_degree_above": 2})


def test_faults_unchanged_on_corpus():
    # 2.1.0's corpus (recorded by tools/search_corpus.py): 1,200 random models at five sets of
    # limits, 3,804 searches that find faults and 2,196 that find none, Stim's answers and
    # 2.1.0's alike. 2.2 stores the visited sets in columns; the steps, so the answers, are the same.
    import gzip
    import json
    import pathlib
    import sys

    root = pathlib.Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "tools"))
    import search_corpus

    data = json.loads(gzip.decompress((root / "tests" / "data" / "search_corpus.json.gz").read_bytes()))
    assert sum(isinstance(a, list) for row in data["answers"] for a in row) == 3804
    for model, want in zip(data["models"], data["answers"]):
        text = search_corpus.circuit_text(*model)
        assert [search_corpus.answer(sq, text, limits) for limits in data["limits"]] == want, text


def test_memory_budget_message():
    # Stim's search keeps every set it visits for as long as it runs; this one keeps them within
    # max_memory_bytes (by default half the machine's memory). Running out raises Stim's message
    # for a search that finds nothing, with the budget and the sets visited.
    with pytest.raises(ValueError) as nothing:
        stim.Circuit("R 0 1\nX_ERROR(0.1) 0\nM 0 1\nDETECTOR rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-1]").search_for_undetectable_logical_errors(**SEARCH)
    c = noisy("surface_code:rotated_memory_x", 5)
    ours = sq.Circuit(str(c))
    model = sq.DetectorErrorModel(str(c.detector_error_model()))
    # The d = 5 surface code's search keeps about 204,000 sets before it finds its five faults,
    # which takes a budget of 12.5 MB.
    for budget, states in [(4096, 64), (100_000, 2030), (10**7, 138_858)]:
        for run in (ours.search_for_undetectable_logical_errors, model.search_for_undetectable_logical_errors):
            with pytest.raises(ValueError) as got:
                run(**SEARCH, max_memory_bytes=budget)
            assert str(got.value) == f"{nothing.value} (memory budget of {budget} bytes reached after {states} states)"
    # Within its budget, the search takes the same steps: Stim's faults.
    want = _terms(c.search_for_undetectable_logical_errors(**SEARCH))
    assert len(want) == 5
    assert _terms(ours.search_for_undetectable_logical_errors(**SEARCH, max_memory_bytes=10**8)) == want
    assert _terms(ours.search_for_undetectable_logical_errors(**SEARCH, max_memory_bytes=None)) == want
    for budget in (10**8, None):
        assert [line.split()[1:] for line in str(model.search_for_undetectable_logical_errors(**SEARCH, max_memory_bytes=budget)).splitlines()] == want
    with pytest.raises(TypeError, match="max_memory_bytes must be an integer"):
        ours.search_for_undetectable_logical_errors(**SEARCH, max_memory_bytes=1e6)
    with pytest.raises(ValueError, match="max_memory_bytes must be at least 0"):
        model.search_for_undetectable_logical_errors(**SEARCH, max_memory_bytes=-1)


def _search_outcome(model, budget):
    """The faults found (as text), or the sets stored when the budget ran out."""
    import re

    try:
        return str(model.search_for_undetectable_logical_errors(**SEARCH, max_memory_bytes=budget))
    except ValueError as e:
        return int(re.fullmatch(r"Failed to find any logical errors\. \(memory budget of \d+ bytes reached after (\d+) states\)", str(e)).group(1))


def test_memory_budget_is_monotone():
    # A larger budget never stores fewer sets, so never fails where a smaller one finds the
    # faults: every budget from 0 bytes on a model whose answer comes after one set (the answer
    # is returned, not stored; 164 bytes failed "after 2 states" when it was stored), and on the
    # d = 3 color code's model every budget to 4,000 bytes and windows of 300 above, past the
    # least that finds its faults.
    one = sq.DetectorErrorModel("error(0.1) D0 L0\nerror(0.1) D0")
    color = sq.DetectorErrorModel(str(HYPER[0][1]().detector_error_model()))
    windows = [range(b, b + 300) for b in (8_000, 20_000, 40_000, 70_000, 100_000)]
    for model, budgets in [(one, range(0, 1_000)), (color, [b for r in [range(0, 4_000), *windows] for b in r])]:
        want = str(model.search_for_undetectable_logical_errors(**SEARCH))
        stored, least = 0, None
        for budget in budgets:
            got = _search_outcome(model, budget)
            if isinstance(got, str):
                assert got == want, budget
                least = budget if least is None else least
            else:
                assert least is None, (budget, got)
                assert got >= stored, (budget, got, stored)
                stored = got
        assert least is not None
    assert _search_outcome(one, 164) == "error(1) D0\nerror(1) D0 L0"


def test_memory_budget_leaves_graphlike_limits_alone():
    # Limits of two and two take Stim's graph-like search, which keeps no sets: the budget does
    # not cover it, and even 0 bytes finds Stim's faults.
    c = noisy("surface_code:rotated_memory_x", 3)
    limits = dict(dont_explore_detection_event_sets_with_size_above=2, dont_explore_edges_with_degree_above=2, dont_explore_edges_increasing_symptom_degree=False)
    want = _terms(c.search_for_undetectable_logical_errors(**limits))
    assert len(want) == 3
    assert _terms(sq.Circuit(str(c)).search_for_undetectable_logical_errors(**limits, max_memory_bytes=0)) == want
    model = sq.DetectorErrorModel(str(c.detector_error_model()))
    assert [line.split()[1:] for line in str(model.search_for_undetectable_logical_errors(**limits, max_memory_bytes=0)).splitlines()] == want


@pytest.mark.parametrize("text", [
    "R 0\nM 0\nDETECTOR rec[-1]",
    "R 0\nX_ERROR(0.1) 0\nM 0\nDETECTOR rec[-1]",
    "R 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]",
    "R 0 1\nX_ERROR(0.1) 0\nM 0 1\nDETECTOR rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-1]",
    "R 0 1 2\nX_ERROR(0.1) 0\nM 0 1 2\nDETECTOR rec[-3] rec[-2] rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]",
    "R 0 1 2\nX_ERROR(0.1) 0\nCX 0 1 0 2\nM 0 1 2\nDETECTOR rec[-3]\nDETECTOR rec[-2]\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]",
])
def test_searches_that_find_nothing_say_so_as_stim(text):
    # Stim's message, with its warnings for a model that makes the search vacuous.
    for run in (
        lambda c: c.shortest_graphlike_error(),
        lambda c: c.detector_error_model().shortest_graphlike_error(),
        lambda c: c.search_for_undetectable_logical_errors(dont_explore_detection_event_sets_with_size_above=4, dont_explore_edges_with_degree_above=4, dont_explore_edges_increasing_symptom_degree=False),
    ):
        with pytest.raises(ValueError) as want:
            run(stim.Circuit(text))
        with pytest.raises(ValueError) as got:
            run(sq.Circuit(text))
        assert str(got.value) == str(want.value)
