"""The PyMatching API's ``Matching`` against PyMatching 2.4 itself, shot by shot, ties included:
it decodes on ``src/pm_exact``, a line-for-line transliteration of PyMatching's sparse blossom
(its radix-heap queue with last-in-first-out ties, its blossom cycle direction, its two-ended
path search and its edge cancellation), so ``decode_to_edges_array`` gives PyMatching's edges in
PyMatching's order, ``decode_to_matched_dets_array`` its pairs in its order, and ``decode`` /
``decode_batch`` its predictions and weights. The cases: surface-code memories, colour codes'
Möbius matching graphs (Chromobius's), boundary-heavy repetition codes, negative weights, weights
all zero, a disconnected graph (shots refused, with PyMatching's message), and graphs built by
hand with merges, boundary nodes, integer weights (ties everywhere) and more than 64 fault ids."""

from __future__ import annotations

import functools
import os
import random

import numpy as np
import pytest

import stabilizer_qec as sq

pymatching = pytest.importorskip("pymatching", minversion="2.4")
stim = pytest.importorskip("stim")

MERGES = ["independent", "smallest-weight", "keep-original", "replace"]


def outcome(f):
    """A call's value, or its ValueError's message (both sides must fail alike)."""
    try:
        return "ok", f()
    except ValueError as e:
        return "error", str(e)


def model_case(text: str, dets: np.ndarray):
    return sq.Matching(sq.DetectorErrorModel(text)), pymatching.Matching.from_detector_error_model(stim.DetectorErrorModel(text)), dets.astype(np.uint8)


def surface(d: int, p: float, shots: int = 1000):
    c = stim.Circuit.generated(
        "surface_code:rotated_memory_x", distance=d, rounds=d, after_clifford_depolarization=p,
        before_round_data_depolarization=p, before_measure_flip_probability=p, after_reset_flip_probability=p,
    )  # fmt: skip
    text = str(c.detector_error_model(decompose_errors=True))
    dets = c.compile_detector_sampler(seed=1000 * d + int(p * 1e4)).sample(shots)
    return model_case(text, dets)


def colour(d: int, p: float, shots: int = 1000):
    # The Möbius matching graph our Chromobius port builds (Chromobius's `to_matching_graph`),
    # each fired colour-code detector exciting its two Möbius nodes, as Chromobius decodes.
    cc = sq.CssCode.color_code(d).memory_circuit(d, p, annotate_colors=True)
    mobius = str(sq.ColorMatching(cc.detector_error_model()).mobius_model)
    fired = cc.compile_detector_sampler(seed=31 * d).sample(shots, separate_observables=True)[0].astype(bool)
    n = stim.DetectorErrorModel(mobius).num_detectors
    dets = np.zeros((shots, n), dtype=np.uint8)
    dets[:, 0 : 2 * fired.shape[1] : 2] = fired
    dets[:, 1 : 2 * fired.shape[1] : 2] = fired
    return model_case(mobius, dets)


def repetition_model(shots: int = 1000):
    c = stim.Circuit.generated("repetition_code:memory", distance=9, rounds=9, after_clifford_depolarization=0.04, before_measure_flip_probability=0.04)
    return model_case(str(c.detector_error_model(decompose_errors=True)), c.compile_detector_sampler(seed=5).sample(shots))


def repetition_check_matrix(shots: int = 1000):
    # Unit weights and a boundary node: every syndrome has many equally good matchings.
    n = 11
    H = np.zeros((n - 1, n), dtype=np.uint8)
    for k in range(n - 1):
        H[k, k] = H[k, k + 1] = 1
    kw = dict(repetitions=6, timelike_weights=1.0, measurement_error_probabilities=0.05, error_probabilities=0.05)
    ours, theirs = sq.Matching.from_check_matrix(H, **kw), pymatching.Matching.from_check_matrix(H, **kw)
    rng = np.random.default_rng(6)
    return ours, theirs, (rng.random((shots, theirs.num_detectors)) < 0.15).astype(np.uint8)


def negative_model(shots: int = 600):
    # A d = 5 surface model with every seventh fault's probability above 1/2: negative weights.
    c = stim.Circuit.generated("surface_code:rotated_memory_z", distance=5, rounds=5, after_clifford_depolarization=0.006, before_measure_flip_probability=0.006)
    dem = c.detector_error_model(decompose_errors=True)
    out = stim.DetectorErrorModel()
    for k, inst in enumerate(dem.flattened()):
        if inst.type == "error" and k % 7 == 3:
            out.append("error", 0.8, inst.targets_copy())
        else:
            out.append(inst)
    rng = np.random.default_rng(7)
    return model_case(str(out), (rng.random((shots, out.num_detectors)) < 0.08).astype(np.uint8))


def by_hand(seed: int, *, integral: bool, signed: bool, wide: bool, shots: int = 300):
    rng = random.Random(seed)
    n = rng.randint(6, 30)
    nf = rng.randint(66, 80) if wide else rng.randint(1, 8)
    ours, theirs = sq.Matching(), pymatching.Matching()
    for _ in range(rng.randint(n, 3 * n)):
        w = float(rng.randint(-2 if signed else 0, 4)) if integral else rng.uniform(-2.0 if signed else 0.05, 6.0)
        faults = set(rng.sample(range(nf), rng.randint(0, min(2, nf))))
        p = rng.choice([None, rng.uniform(0.01, 0.4)])
        merge = rng.choice(MERGES)
        if rng.random() < 0.2:
            u = rng.randrange(n)
            for m in (ours, theirs):
                m.add_boundary_edge(u, fault_ids=faults, weight=w, error_probability=p, merge_strategy=merge)
        else:
            u, v = rng.sample(range(n), 2)
            for m in (ours, theirs):
                m.add_edge(u, v, fault_ids=faults, weight=w, error_probability=p, merge_strategy=merge)
    if rng.random() < 0.4:
        b = set(rng.sample(range(n), rng.randint(1, 2)))
        ours.set_boundary_nodes(b)
        theirs.set_boundary_nodes(b)
    ours.ensure_num_fault_ids(nf)
    theirs.ensure_num_fault_ids(nf)
    nd = theirs.num_detectors
    srng = np.random.default_rng(seed)
    return ours, theirs, (srng.random((shots, nd)) < rng.choice([0.1, 0.25, 0.4])).astype(np.uint8)


def zero_weights(shots: int = 500):
    # Every edge of weight 0 (a 4 x 4 grid with boundary edges on one side): all matchings tie.
    ours, theirs = sq.Matching(), pymatching.Matching()
    for m in (ours, theirs):
        for r in range(4):
            for c in range(4):
                k = 4 * r + c
                if c < 3:
                    m.add_edge(k, k + 1, fault_ids={k % 3}, weight=0.0)
                if r < 3:
                    m.add_edge(k, k + 4, fault_ids=set(), weight=0.0)
            m.add_boundary_edge(4 * r, fault_ids={3}, weight=0.0)
    rng = np.random.default_rng(8)
    return ours, theirs, (rng.random((shots, 16)) < 0.3).astype(np.uint8)


def disconnected(shots: int = 500):
    # A component with a boundary, one without (odd parity there: refused), and isolated nodes.
    ours, theirs = sq.Matching(), pymatching.Matching()
    for m in (ours, theirs):
        for k in range(5):
            m.add_edge(k, k + 1, fault_ids={0}, weight=1.0 + k % 2)
        m.add_boundary_edge(0, fault_ids={1}, weight=2.5)
        for k in range(8, 13):
            m.add_edge(k, k + 1, fault_ids={2}, weight=1.5)
        m.add_edge(15, 16, fault_ids=set(), weight=0.5)
    rng = np.random.default_rng(9)
    return ours, theirs, (rng.random((shots, theirs.num_detectors)) < 0.15).astype(np.uint8)


BUILDERS = {}
for _d in (3, 5, 7):
    for _p in (0.001, 0.005, 0.01):
        BUILDERS[f"surface_d{_d}_p{_p}"] = functools.partial(surface, _d, _p)
for _d, _p in ((3, 0.01), (5, 0.005), (5, 0.01)):
    BUILDERS[f"colour_d{_d}_p{_p}"] = functools.partial(colour, _d, _p)
BUILDERS["repetition_model"] = repetition_model
BUILDERS["repetition_check_matrix"] = repetition_check_matrix
BUILDERS["negative_weights"] = negative_model
BUILDERS["zero_weights"] = zero_weights
BUILDERS["disconnected"] = disconnected
for _s in range(12):
    BUILDERS[f"by_hand_{_s}"] = functools.partial(by_hand, 500 + _s, integral=_s % 2 == 0, signed=_s % 3 == 0, wide=_s % 4 == 1)


@functools.lru_cache(maxsize=None)
def case(name: str):
    return BUILDERS[name]()


def compare(name, ours_f, theirs_f):
    ours, theirs, dets = case(name)
    differ = []
    for k, s in enumerate(dets):
        a, b = outcome(lambda: ours_f(ours, s)), outcome(lambda: theirs_f(theirs, s))
        if a != b:
            differ.append(k)
    assert not differ, f"{name}: {len(differ)} of {len(dets)} shots differ (first: {differ[:5]})"


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_edges_equal_pymatching(name):
    f = lambda m, s: m.decode_to_edges_array(s).tolist()  # noqa: E731
    compare(name, f, f)


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_matched_dets_equal_pymatching(name):
    f = lambda m, s: (m.decode_to_matched_dets_array(s).tolist(), m.decode_to_matched_dets_dict(s))  # noqa: E731
    compare(name, f, f)


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_decode_and_weight_equal_pymatching(name):
    ours, theirs, dets = case(name)
    f = lambda m, s: (lambda r: (r[0].tolist(), r[1]))(m.decode(s, return_weight=True))  # noqa: E731
    compare(name, f, f)
    a = outcome(lambda: ours.decode_batch(dets, return_weights=True))
    b = outcome(lambda: theirs.decode_batch(dets, return_weights=True))
    assert a[0] == b[0], (a, b)
    if a[0] == "ok":
        assert np.array_equal(a[1][0], b[1][0]) and np.array_equal(a[1][1], b[1][1])
        packed = np.packbits(dets, axis=1, bitorder="little")
        assert np.array_equal(
            ours.decode_batch(packed, bit_packed_shots=True, bit_packed_predictions=True),
            theirs.decode_batch(packed, bit_packed_shots=True, bit_packed_predictions=True),
        )
    else:
        # PyMatching's message, after the first failing shot's index.
        first = next(k for k, s in enumerate(dets) if outcome(lambda: theirs.decode(s))[0] == "error")
        assert a[1] == f"shot {first}: {b[1]}", (a, b)


@pytest.mark.skipif(not os.environ.get("STABILIZER_QEC_SLOW"), reason="slow: set STABILIZER_QEC_SLOW=1")
def test_many_more_shots_and_graphs():
    # 20,000 shots of each model and 400 more graphs built by hand: edges, pairs, predictions and
    # weights, shot by shot.
    builders = [functools.partial(surface, d, p, 20_000) for d in (3, 5, 7) for p in (0.001, 0.005, 0.01, 0.02)]
    builders += [functools.partial(colour, d, p, 20_000) for d, p in ((3, 0.01), (5, 0.01), (7, 0.005))]
    builders += [functools.partial(repetition_model, 20_000), functools.partial(repetition_check_matrix, 20_000), functools.partial(negative_model, 20_000)]
    builders += [functools.partial(by_hand, 20_000 + s, integral=s % 2 == 0, signed=s % 3 == 0, wide=s % 4 == 1, shots=100) for s in range(400)]
    for make in builders:
        ours, theirs, dets = make()
        for s in dets:
            for f in (
                lambda m: m.decode_to_edges_array(s).tolist(),
                lambda m: m.decode_to_matched_dets_array(s).tolist(),
                lambda m: (lambda r: (r[0].tolist(), r[1]))(m.decode(s, return_weight=True)),
            ):
                assert outcome(lambda: f(ours)) == outcome(lambda: f(theirs)), (make, s)


def test_threads_do_not_change_answers():
    ours, theirs, dets = case("surface_d5_p0.01")
    want, want_w = theirs.decode_batch(dets, return_weights=True)
    for threads in (1, 2, 3, 0):
        got, got_w = ours.decode_batch(dets, return_weights=True, threads=threads)
        assert np.array_equal(got, want) and np.array_equal(got_w, want_w), threads
    # And a graph with ties everywhere, at every thread count.
    ours, theirs, dets = case("repetition_check_matrix")
    want = theirs.decode_batch(dets)
    for threads in (1, 4, 0):
        assert np.array_equal(ours.decode_batch(dets, threads=threads), want), threads
    # A batch that fails names its earliest failing shot, at every thread count.
    m = sq.Matching()
    m.add_edge(0, 1)
    m.add_edge(2, 3)
    shots = np.zeros((40, 4), dtype=np.uint8)
    shots[[7, 23, 31], 2] = 1
    for threads in (1, 2, 4, 0):
        with pytest.raises(ValueError, match=r"^shot 7: No perfect matching could be found"):
            m.decode_batch(shots, threads=threads)
