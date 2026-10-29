"""The PyMatching API's ``Matching`` against PyMatching 2.4 itself, shot by shot, ties included:
it decodes on ``src/pm_exact``, a line-for-line transliteration of PyMatching's sparse blossom
(its radix-heap queue with last-in-first-out ties, its blossom cycle direction, its two-ended
path search and its edge cancellation), so ``decode_to_edges_array`` gives PyMatching's edges in
PyMatching's order, ``decode_to_matched_dets_array`` its pairs in its order, and ``decode`` /
``decode_batch`` its predictions and weights. The cases: surface-code memories, colour codes'
Möbius matching graphs (Chromobius's), boundary-heavy repetition codes, negative weights, weights
all zero, a disconnected graph (shots refused, with PyMatching's message), graphs built by hand
with merges, boundary nodes, integer weights (ties everywhere) and more than 64 fault ids, and
models loaded with ``enable_correlations=True`` and decoded plainly (PyMatching's correlated load
orders each edge's ends and counts the weights edges imply for each other in its weight scale)."""

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


def rare_correlations_text(seed: int, *, high_first: bool = False) -> str:
    # A line of common faults beside rare correlated ones, D_k D_k+1 ^ D_k+2 D_k+3 at 1e-9 to
    # 1e-4: the weights those imply are the graph's heaviest. `high_first` writes every piece's
    # higher detector first: a correlated load puts the ends in order, a plain load keeps them.
    rng = random.Random(seed)
    pair = lambda a, b: f"D{b} D{a}" if high_first else f"D{a} D{b}"  # noqa: E731
    lines = []
    n = rng.randint(8, 20)
    for k in range(n - 1):
        lines.append(f"error({rng.choice([0.3, 0.3, 0.35, rng.uniform(0.25, 0.45)]):.6f}) {pair(k, k + 1)} L{k % 2}")
    for k in range(n - 2):
        if rng.random() < 0.5:
            lines.append(f"error({rng.choice([0.3, 0.35, rng.uniform(0.25, 0.45)]):.6f}) {pair(k, k + 2)}")
    for k in range(n):
        if rng.random() < 0.6:
            lines.append(f"error({rng.choice([0.3, 0.35, rng.uniform(0.25, 0.45)]):.6f}) D{k}")
    for k in range(n - 3):
        if rng.random() < 0.4:
            lines.append(f"error({rng.choice([1e-9, 1e-6, 1e-4]):g}) {pair(k, k + 1)} ^ {pair(k + 2, k + 3)}")
    return "\n".join(lines)


def correlated_load(text: str):
    return (
        sq.Matching.from_detector_error_model(sq.DetectorErrorModel(text), enable_correlations=True),
        pymatching.Matching.from_detector_error_model(stim.DetectorErrorModel(text), enable_correlations=True),
    )


def correlated_rare(seed: int, shots: int = 400, *, high_first: bool = False, changed: bool = False):
    # Plain decodes on a model loaded with correlations: PyMatching's correlated load orders each
    # edge's ends and counts the implied weights in its weight scale, so these differ from a plain
    # load's. `changed`: edges added afterwards, which keep the implied weights.
    ours, theirs = correlated_load(rare_correlations_text(seed, high_first=high_first))
    if changed:
        n = theirs.num_detectors
        for m in (ours, theirs):
            m.add_edge(n - 1, 0, fault_ids={1}, weight=0.75, merge_strategy="independent")
            m.add_edge(1, 0, fault_ids={0}, weight=0.4, error_probability=0.4, merge_strategy="independent")
    dets = (np.random.default_rng(seed).random((shots, theirs.num_detectors)) < 0.4).astype(np.uint8)
    return ours, theirs, dets


def correlated_surface(d: int, p: float, shots: int = 1000):
    c = stim.Circuit.generated("surface_code:rotated_memory_x", distance=d, rounds=d, after_clifford_depolarization=p, before_measure_flip_probability=p)
    return (*correlated_load(str(c.detector_error_model(decompose_errors=True))), c.compile_detector_sampler(seed=77 * d).sample(shots).astype(np.uint8))


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
# Seeds whose plain decodes on a plain load's scale differ from PyMatching's: predictions (20,
# 26), edges and pairs (all).
for _s in (3, 13, 19, 20, 23, 26):
    BUILDERS[f"correlated_rare_{_s}"] = functools.partial(correlated_rare, _s, high_first=_s % 2 == 1)
BUILDERS["correlated_rare_changed"] = functools.partial(correlated_rare, 20, high_first=True, changed=True)
BUILDERS["correlated_surface_d5_p0.01"] = functools.partial(correlated_surface, 5, 0.01)
CORRELATED = sorted(k for k in BUILDERS if k.startswith("correlated"))


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


@pytest.mark.parametrize(
    "graph,bits,recovers",
    [
        ("line", 0b100000, True), ("boundary", 0b100001, True), ("negative", 0b100001, True),
        ("line", 0b100101, False), ("line", 0b100010, False), ("boundary", 0b100101, False),
        ("negative", 0b100100, False), ("isolated", 0b100010, False),
    ],
)  # fmt: skip
def test_a_failed_bit_packed_batch_leaves_the_decoder_working(graph, bits, recovers):
    # A padding bit past the last node (bit 5 of 3 nodes) fails the batch with PyMatching's
    # message, after "shot 0: ". PyMatching first makes a region at each of the shot's earlier
    # detection events and then throws, so it recovers only when it made none: no earlier event,
    # one at a boundary node, or one at a node a negative edge marked (it only unmarks it). Ours
    # refuses before making any (deliberately), so its later decodes are always a fresh decoder's;
    # where PyMatching recovers, they are also that same PyMatching decoder's.
    def build(mod):
        m = mod.Matching()
        if graph == "isolated":  # node 1 has no edges
            m.add_edge(0, 2, fault_ids={0})
        else:
            m.add_edge(0, 1, fault_ids={0}, weight=-1.0 if graph == "negative" else 1.0)
            m.add_edge(1, 2, fault_ids={1})
        m.add_boundary_edge(0, fault_ids={2})
        m.add_boundary_edge(2, fault_ids={3})
        if graph == "boundary":
            m.set_boundary_nodes({0})
        return m

    def after(m):
        return [
            outcome(lambda: plain_value(m.decode(s, return_weight=True)))
            for s in ([1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0], [1, 1, 0])
        ] + [outcome(lambda: plain_value(m.decode_batch(np.array([[0b001], [0b100], [0b101]], dtype=np.uint8), bit_packed_shots=True, return_weights=True)))]

    # The bad shot first, so at every thread count it falls to the decoder kept for later decodes.
    bad = np.array([[bits], [0b001], [0b100], [0b000]], dtype=np.uint8)
    ours, theirs, fresh = build(sq), build(pymatching), build(pymatching)
    want_error = outcome(lambda: theirs.decode_batch(bad, bit_packed_shots=True))
    assert want_error[0] == "error" and "index" in want_error[1] and "5" in want_error[1], want_error
    for threads in (1, 4):
        assert outcome(lambda: ours.decode_batch(bad, bit_packed_shots=True, threads=threads)) == ("error", "shot 0: " + want_error[1]), threads
        assert after(ours) == after(fresh), threads
    if recovers:
        assert after(theirs) == after(fresh)


def rows(m):
    return [(u, v, d["fault_ids"], d["weight"], d["error_probability"]) for u, v, d in m.edges()]


def close(a: float, b: float) -> bool:
    # PyMatching's arm64 build fuses multiply-adds in its merge arithmetic: a last-bit difference
    # in a merged edge's weight or probability (as tests/test_pymatching_differential.py).
    return a == b or abs(a - b) <= 1e-12 * max(1.0, abs(a), abs(b))


def same_rows(a, b) -> bool:
    """The same edges in the same order, ends and fault ids exactly."""
    return len(a) == len(b) and all(x[:3] == y[:3] and close(x[3], y[3]) and close(x[4], y[4]) for x, y in zip(a, b))


@pytest.mark.parametrize("name", CORRELATED)
def test_edges_of_a_correlated_load_equal_pymatching(name):
    # Each edge's ends in order (node1 < node2), as PyMatching's correlated load writes them.
    ours, theirs, _ = case(name)
    a, b = rows(ours), rows(theirs)
    assert [r[:3] for r in a] == [r[:3] for r in b]
    assert same_rows(a, b), [(x, y) for x, y in zip(a, b) if x != y][:3]
    assert ours.num_edges == theirs.num_edges and ours.num_fault_ids == theirs.num_fault_ids


def test_a_correlated_load_differs_from_a_plain_one():
    # The cases above are not vacuous: the same model loaded plainly has its ends as written, a
    # different weight scale, and other answers on some shots.
    text = rare_correlations_text(20, high_first=True)
    plain, correlated = sq.Matching(sq.DetectorErrorModel(text)), sq.Matching(sq.DetectorErrorModel(text), enable_correlations=True)
    assert [r[:2] for r in rows(plain)] != [r[:2] for r in rows(correlated)]
    assert [(min(u, v), max(u, v)) if v is not None else (u, v) for u, v, *_ in rows(plain)] == [r[:2] for r in rows(correlated)]
    dets = (np.random.default_rng(1).random((300, plain.num_detectors)) < 0.4).astype(np.uint8)
    assert not np.array_equal(plain.decode_batch(dets, return_weights=True)[1], correlated.decode_batch(dets, return_weights=True)[1])


def test_a_correlated_load_refuses_as_pymatching():
    for text in (
        "error(0.6) D0 D1",
        "error(0.1) D0 D1 D2",
        "error(0.1) D0 D1 ^ D2 D3 D4",
        "error(0.1) L0 ^ D1",
        "error(0.1) D0 ^ L0",
        "error(0.6) D0\nerror(0.1) D0 D1 D2",
        "error(0.1) D0 D1 D2\nerror(0.6) D0",
    ):
        a = outcome(lambda: rows(sq.Matching.from_detector_error_model(sq.DetectorErrorModel(text), enable_correlations=True)))
        b = outcome(lambda: rows(pymatching.Matching.from_detector_error_model(stim.DetectorErrorModel(text), enable_correlations=True)))
        assert a == b and a[0] == "error", (text, a, b)
    # Undetectable alone, or impossible: ignored, as PyMatching.
    text = "error(0.1) L0\nerror(0) D0 D1 D2\nerror(0.5) D1 D0"
    ours, theirs = correlated_load(text)
    assert rows(ours) == rows(theirs) == [(0, 1, set(), 0.0, 0.5)]
    # Loaded, but a weight implied for a self-loop (not an edge of the matching graph) stops
    # PyMatching building its decoder: every decode is refused, with its message, correlated ones
    # too (they take the decoder, `get_mwpm_with_search_graph`, before they reweight).
    ours, theirs = correlated_load("error(0.1) D0 D0 ^ D1\nerror(0.2) D0\nerror(0.2) D1 D2")
    s = np.array([1, 0, 1], dtype=np.uint8)
    for f in (
        lambda m: m.decode(s).tolist(), lambda m: m.decode_to_edges_array(s).tolist(), lambda m: m.decode_batch(s[None]).tolist(),
        lambda m: m.decode(s, enable_correlations=True).tolist(), lambda m: m.decode_to_edges_array(s, enable_correlations=True).tolist(),
        lambda m: m.decode_batch(s[None], enable_correlations=True).tolist(),
    ):  # fmt: skip
        assert outcome(lambda: f(ours)) == outcome(lambda: f(theirs)) == ("error", "Failed to find neighbor.")


def test_a_changed_correlated_load_pickles_with_its_implied_weights():
    import pickle

    ours, theirs, dets = case("correlated_rare_changed")
    again = pickle.loads(pickle.dumps(ours))
    assert rows(again) == rows(ours) and same_rows(rows(again), rows(theirs))
    want = theirs.decode_batch(dets, return_weights=True)
    got = again.decode_batch(dets, return_weights=True)
    assert np.array_equal(got[0], want[0]) and np.array_equal(got[1], want[1])


SQUARE_DEM = "error(0.1) D0 D1 L0\nerror(0.1) D1 D2\nerror(0.1) D2 D3 L1\nerror(0.1) D3 D0"


def plain_value(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (tuple, list)):
        return [plain_value(v) for v in x]
    if isinstance(x, dict):
        return sorted(x.items())
    return x


@pytest.mark.parametrize(
    "load,steps",
    [
        ("square", "d e d"), ("square", "d e b"), ("square", "d e b4"), ("square", "d e bp"), ("square", "e d"),
        ("square", "d e p q d"), ("square", "b e b"), ("square", "d e d e d"), ("square", "d e x d"),
        ("square", "d x e d"), ("square", "x e d"), ("square", "p e d"), ("square", "d e d x d b"),
        ("square", "d e a d"), ("square", "x e x d"),
        ("plain", "d e d"), ("plain", "d e c d"), ("plain", "d e cb d"), ("plain", "d e cx d"), ("plain", "d c e d"),
        ("correlated", "d e d"), ("correlated", "d e c d"), ("correlated", "d e cb b"), ("correlated", "d e cx d"),
        ("correlated", "c e d"), ("correlated", "d e x d"),
    ],
)  # fmt: skip
def test_ensure_num_fault_ids_keeps_the_built_decoder_as_pymatching(load, steps):
    # PyMatching's `set_min_num_observables` does not mark its decoder for rebuilding, so one built
    # before keeps its own fault-id count: it fills only those of the longer output, and on ties
    # takes its own path (by mask up to 64 fault ids, by the search graph above). A change to the
    # graph rebuilds it, and so does `get_mwpm_with_search_graph` (`decode_to_edges_array` and every
    # correlated call) when it was built without the search graph. On this square, [1, 0, 1, 0]
    # has two matchings of weight 2 and the two paths pick different ones.
    def build(mod):
        if load == "square":
            m = mod.Matching()
            for (u, v), f in zip(((0, 1), (1, 2), (2, 3), (3, 0)), ({0}, set(), {1}, set())):
                m.add_edge(u, v, fault_ids=f, weight=1.0)
            return m
        dem = (sq.DetectorErrorModel if mod is sq else stim.DetectorErrorModel)(SQUARE_DEM)
        return mod.Matching.from_detector_error_model(dem, enable_correlations=load == "correlated")

    s = np.array([1, 0, 1, 0], dtype=np.uint8)
    shots = np.tile(s, (50, 1))
    packed = np.packbits(s, bitorder="little")[None, :]

    def batch(m, rows, **kw):
        if isinstance(m, sq.Matching) and kw.pop("threads4", False):
            kw["threads"] = 4
        kw.pop("threads4", None)
        return m.decode_batch(rows, return_weights=True, **kw)

    ops = {
        "d": lambda m: m.decode(s, return_weight=True),
        "b": lambda m: batch(m, shots),
        "b4": lambda m: batch(m, shots, threads4=True),
        "bp": lambda m: batch(m, packed, bit_packed_shots=True, bit_packed_predictions=True),
        "e": lambda m: m.ensure_num_fault_ids(70),
        "x": lambda m: m.decode_to_edges_array(s),
        "p": lambda m: m.decode_to_matched_dets_array(s),
        "q": lambda m: m.decode_to_matched_dets_dict(s),
        "a": lambda m: m.add_edge(0, 2, fault_ids={2}, weight=5.0),
        # Correlated calls (their answers are the correlated matcher's; only whether they decode or
        # refuse is compared here).
        "c": lambda m: m.decode(s, enable_correlations=True) is None,
        "cb": lambda m: batch(m, shots, enable_correlations=True) is None,
        "cx": lambda m: m.decode_to_edges_array(s, enable_correlations=True) is None,
    }
    ours, theirs = build(sq), build(pymatching)
    got = [outcome(lambda: plain_value(ops[k](ours))) for k in steps.split()]
    want = [outcome(lambda: plain_value(ops[k](theirs))) for k in steps.split()]
    assert got == want, (load, steps)


def test_ensure_num_fault_ids_cases_are_not_vacuous():
    # The kept decoder answers otherwise than a fresh one: the sequences above see it.
    s = np.array([1, 0, 1, 0], dtype=np.uint8)
    for mod in (sq, pymatching):
        dem = (sq.DetectorErrorModel if mod is sq else stim.DetectorErrorModel)(SQUARE_DEM)
        kept, fresh = mod.Matching.from_detector_error_model(dem), mod.Matching.from_detector_error_model(dem)
        kept.decode(s)
        kept.ensure_num_fault_ids(70)
        fresh.ensure_num_fault_ids(70)
        assert kept.decode(s)[:2].tolist() == [1, 0] and fresh.decode(s)[:2].tolist() == [0, 1], mod


@pytest.mark.parametrize("negative", [False, True])
@pytest.mark.parametrize("ids", [{79}, set(range(64, 71)), set(range(64, 72)), set(range(64, 81)), {1, 2, 3, 4, 5, 79, 87, 95}])
def test_fault_ids_past_the_count_are_dropped(ids, negative):
    # A replace merge can put fault ids past the count on an edge (merges do not extend it).
    # PyMatching shifts `1 << id` for them, which C++ leaves undefined (its arm64 wheel drops them
    # eight at a time and wraps the rest), so no rule matches it; ours drops them, as the search
    # graph does above 64 fault ids, and never reports id - 64 (a real fault id) in their place.
    m = sq.Matching()
    m.add_edge(0, 1, fault_ids={0}, weight=1.0)
    m.add_boundary_edge(0, weight=0.1 if negative else 1.0)
    m.add_boundary_edge(1, weight=0.1 if negative else 1.0)
    m.add_edge(0, 1, fault_ids=ids, weight=-1.0 if negative else 1.0, merge_strategy="replace")
    m.ensure_num_fault_ids(64)
    syndrome = [0, 0] if negative else [1, 1]
    want = sorted(i for i in ids if i < 64)
    assert np.flatnonzero(m.decode(syndrome)).tolist() == want
    assert np.flatnonzero(m.decode_batch(np.array([syndrome] * 3, dtype=np.uint8))[2]).tolist() == want
