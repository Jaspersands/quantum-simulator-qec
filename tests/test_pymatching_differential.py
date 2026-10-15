"""PyMatching's graph API against PyMatching 2.4 itself: random graphs built by hand (signed,
integer and float weights, boundary nodes, merges, more than 64 fault ids), check matrices with
repetitions, networkx round trips and models. Predictions and weights must be identical; edge
and matched-detector sets too wherever the weights make the matching unique."""

from __future__ import annotations

import random

import numpy as np
import pytest

import stabilizer_qec as sq

pymatching = pytest.importorskip("pymatching", minversion="2.4")

MERGES = ["disallow", "independent", "smallest-weight", "keep-original", "replace"]


def random_graph(rng: random.Random, *, signed: bool, integral: bool, wide: bool):
    n = rng.randint(2, 14)
    nf = rng.randint(66, 90) if wide else rng.randint(1, 8)
    ours, theirs = sq.Matching(), pymatching.Matching()
    for _ in range(rng.randint(n, 3 * n)):
        w = float(rng.randint(-3 if signed else 0, 9)) if integral else rng.uniform(-2.0 if signed else 0.05, 6.0)
        faults = set(rng.sample(range(nf), rng.randint(0, min(2, nf))))
        p = rng.choice([None, rng.uniform(0.01, 0.4)])
        merge = rng.choice(MERGES[1:])
        if rng.random() < 0.25:
            u = rng.randrange(n)
            args = (u,), dict(fault_ids=faults, weight=w, error_probability=p, merge_strategy=merge)
            ours.add_boundary_edge(*args[0], **args[1])
            theirs.add_boundary_edge(*args[0], **args[1])
        else:
            u, v = rng.sample(range(n), 2)
            ours.add_edge(u, v, fault_ids=faults, weight=w, error_probability=p, merge_strategy=merge)
            theirs.add_edge(u, v, fault_ids=faults, weight=w, error_probability=p, merge_strategy=merge)
    if rng.random() < 0.3:
        b = set(rng.sample(range(n), rng.randint(1, 2)))
        ours.set_boundary_nodes(b)
        theirs.set_boundary_nodes(b)
    ours.ensure_num_fault_ids(nf)
    theirs.ensure_num_fault_ids(nf)
    return ours, theirs


class Failed(str):
    pass


def outcome(f):
    try:
        return f()
    except ValueError as e:
        return Failed(str(e))


def close(a: float, b: float) -> bool:
    # PyMatching's arm64 build fuses multiply-adds in its merge arithmetic: a last-bit difference.
    return a == b or abs(a - b) <= 1e-12 * max(1.0, abs(a), abs(b))


def same_edges(ours, theirs):
    a, b = ours.edges(), theirs.edges()
    assert len(a) == len(b)
    for (u1, v1, d1), (u2, v2, d2) in zip(a, b):
        assert (u1, v1, d1["fault_ids"]) == (u2, v2, d2["fault_ids"])
        assert close(d1["weight"], d2["weight"]) and close(d1["error_probability"], d2["error_probability"]), (d1, d2)


def same_decodes(ours, theirs, rng, shots=12, edges=False, predictions=True):
    """Weights always agree; predictions (and edges) only where the matching is unique, which
    integer weights rarely make it (equally good matchings: PyMatching and this package may
    each pick a different one)."""
    assert (ours.num_nodes, ours.num_edges, ours.num_fault_ids, ours.num_detectors, ours.boundary) == (theirs.num_nodes, theirs.num_edges, theirs.num_fault_ids, theirs.num_detectors, theirs.boundary)
    same_edges(ours, theirs)
    nd = theirs.num_detectors
    for _ in range(shots):
        syndrome = np.array([rng.random() < 0.3 for _ in range(nd)], dtype=np.uint8)
        a = outcome(lambda: ours.decode(syndrome, return_weight=True))
        b = outcome(lambda: theirs.decode(syndrome, return_weight=True))
        if isinstance(b, Failed):
            assert isinstance(a, Failed), (a, b)
            continue
        assert not isinstance(a, Failed), (syndrome, a, b)
        assert close(a[1], b[1]), (syndrome, a, b)
        if predictions:
            assert np.array_equal(a[0], b[0]), (syndrome, a, b)
        if edges:
            ea = {tuple(sorted(e)) for e in ours.decode_to_edges_array(syndrome).tolist()}
            eb = {tuple(sorted(e)) for e in theirs.decode_to_edges_array(syndrome).tolist()}
            assert ea == eb, syndrome


@pytest.mark.parametrize("seed", range(150))
def test_random_graphs(seed):
    rng = random.Random(seed)
    integral = seed % 2 == 0
    ours, theirs = random_graph(rng, signed=seed % 3 == 0, integral=integral, wide=seed % 5 == 0)
    same_decodes(ours, theirs, rng, predictions=not integral)


@pytest.mark.parametrize("seed", range(60))
def test_unique_matchings_give_the_same_edges_and_pairs(seed):
    # Float weights spread widely make ties (and so different but equally good answers) rare.
    rng = random.Random(1000 + seed)
    ours, theirs = sq.Matching(), pymatching.Matching()
    n = rng.randint(3, 12)
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < 0.4:
                w = rng.uniform(0.1, 10)
                ours.add_edge(u, v, fault_ids={u % 4}, weight=w)
                theirs.add_edge(u, v, fault_ids={u % 4}, weight=w)
        w = rng.uniform(0.1, 10)
        ours.add_boundary_edge(u, fault_ids={4}, weight=w)
        theirs.add_boundary_edge(u, fault_ids={4}, weight=w)
    same_decodes(ours, theirs, rng, edges=True)
    for _ in range(10):
        syndrome = np.array([rng.random() < 0.4 for _ in range(n)], dtype=np.uint8)
        a = ours.decode_to_matched_dets_dict(syndrome)
        b = theirs.decode_to_matched_dets_dict(syndrome)
        assert a == b
        pa = {tuple(sorted(p)) for p in ours.decode_to_matched_dets_array(syndrome).tolist()}
        pb = {tuple(sorted(p)) for p in theirs.decode_to_matched_dets_array(syndrome).tolist()}
        assert pa == pb


@pytest.mark.parametrize("seed", range(40))
def test_check_matrices(seed):
    scipy = pytest.importorskip("scipy.sparse")
    rng = np.random.default_rng(seed)
    rows, cols = int(rng.integers(3, 9)), int(rng.integers(4, 14))
    H = np.zeros((rows, cols), dtype=np.uint8)
    for c in range(cols):
        for r in rng.choice(rows, size=int(rng.integers(1, 3)), replace=False):
            H[r, c] = 1
    weights = rng.uniform(0.5, 3, cols)
    kw = dict(weights=weights, error_probabilities=float(rng.uniform(0.01, 0.2)))
    if seed % 2:
        kw.update(repetitions=int(rng.integers(2, 4)), timelike_weights=rng.uniform(0.5, 2, rows), measurement_error_probabilities=0.05)
    virtual = seed % 3 == 0
    for H_ in (H, scipy.csc_matrix(H)):
        ours = sq.Matching.from_check_matrix(H_, use_virtual_boundary_node=virtual, **kw)
        theirs = pymatching.Matching.from_check_matrix(H_, use_virtual_boundary_node=virtual, **kw)
        same_decodes(ours, theirs, random.Random(seed))
    # A faults matrix, and the 2-d (checks x rounds) syndrome.
    F = (rng.random((3, cols)) < 0.4).astype(np.uint8)
    ours = sq.Matching(H, weights=weights, faults_matrix=F)
    theirs = pymatching.Matching(H, weights=weights, faults_matrix=F)
    same_decodes(ours, theirs, random.Random(seed))


@pytest.mark.parametrize("seed", range(20))
def test_networkx_round_trips(seed):
    nx = pytest.importorskip("networkx")
    rng = random.Random(2000 + seed)
    ours, theirs = random_graph(rng, signed=False, integral=False, wide=False)
    g_ours, g_theirs = ours.to_networkx(), theirs.to_networkx()
    assert nx.utils.graphs_equal(g_ours, g_theirs)
    same_decodes(sq.Matching(g_theirs), pymatching.Matching(g_theirs), rng)
    same_decodes(sq.Matching.from_networkx(g_ours), pymatching.Matching.from_networkx(g_ours), rng)


@pytest.mark.parametrize("task", ["surface_code:rotated_memory_x", "repetition_code:memory", "color_code:memory_xyz"])
def test_models_and_circuits(task):
    stim = pytest.importorskip("stim")
    c = stim.Circuit.generated(task, distance=5, rounds=4, after_clifford_depolarization=0.005, before_measure_flip_probability=0.005)
    dem = c.detector_error_model(decompose_errors=True)
    dets, _ = c.compile_detector_sampler(seed=3).sample(3000, separate_observables=True)
    pairs = [(sq.Matching(sq.DetectorErrorModel(str(dem))), pymatching.Matching(dem)), (sq.Matching.from_stim_circuit(c), pymatching.Matching.from_stim_circuit(c))]
    for ours, theirs in pairs:
        a, wa = ours.decode_batch(dets, return_weights=True)
        b, wb = theirs.decode_batch(dets, return_weights=True)
        assert np.array_equal(a, b) and np.array_equal(wa, wb)
        assert np.array_equal(ours.decode_batch(np.packbits(dets, axis=1, bitorder="little"), bit_packed_shots=True, bit_packed_predictions=True), theirs.decode_batch(np.packbits(dets, axis=1, bitorder="little"), bit_packed_shots=True, bit_packed_predictions=True))


def test_correlated_edges_are_the_correlated_matchings():
    # decode_to_edges_array(enable_correlations=True) is pass two's matching, as PyMatching's:
    # its edges flip the observables decode(enable_correlations=True) predicts (it was pass
    # one's, which on 77 of these 2000 shots flips others; found by stabilizer_qec.conformance).
    stim = pytest.importorskip("stim")
    c = stim.Circuit.generated("surface_code:rotated_memory_z", distance=5, rounds=5, after_clifford_depolarization=0.005, before_measure_flip_probability=0.005)
    dem = c.detector_error_model(decompose_errors=True)
    dets, _ = c.compile_detector_sampler(seed=3).sample(2000, separate_observables=True)
    ours = sq.Matching.from_detector_error_model(sq.DetectorErrorModel(str(dem)), enable_correlations=True)
    theirs = pymatching.Matching.from_detector_error_model(dem, enable_correlations=True)
    edge_obs = {}
    for inst in dem.flattened():
        if inst.type == "error":
            for part in str(inst).split("^"):
                d = [int(t[1:]) for t in part.split() if t.startswith("D")]
                o = sum(1 for t in part.split() if t == "L0") % 2
                edge_obs.setdefault((min(d), max(d)) if len(d) == 2 else (d[0], -1), o)
    differ = 0
    for s in dets.astype(np.uint8):
        o = ours.decode(s, enable_correlations=True)
        flips = sum(edge_obs[(min(u, v), max(u, v)) if min(u, v) >= 0 else (max(u, v), -1)] for u, v in ours.decode_to_edges_array(s, enable_correlations=True)) % 2
        assert flips == o[0]
        differ += not np.array_equal(o, theirs.decode(s, enable_correlations=True))
    assert differ <= 4  # (ties)


def test_errors_and_repr():
    m = sq.Matching()
    m.add_edge(0, 1)
    with pytest.raises(ValueError, match="already exists"):
        m.add_edge(1, 0)
    with pytest.raises(ValueError, match="not recognised"):
        m.add_edge(0, 1, merge_strategy="nope")
    with pytest.warns(UserWarning, match="exceeds maximum edge weight"):
        m.add_edge(2, 3, weight=2.0**25)
    assert repr(m) == "<stabilizer_qec.Matching object with 2 detectors, 0 boundary nodes, and 1 edge>"
    with pytest.raises(ValueError, match="No perfect matching"):
        m.decode([1, 0])
    with pytest.raises(ValueError, match="enable_correlations"):
        m.decode([1, 1], enable_correlations=True)


def test_noise_is_seeded():
    from stabilizer_qec import pymatching as pm

    m = sq.Matching()
    for k in range(10):
        m.add_edge(k, k + 1, fault_ids={k}, error_probability=0.3)
    pm.set_seed(5)
    a = m.add_noise()
    pm.set_seed(5)
    b = m.add_noise()
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    m.add_edge(0, 5, weight=1.0)
    assert m.add_noise() is None


@pytest.mark.parametrize("enable_correlations", [False, True])
def test_command_line(tmp_path, enable_correlations):
    stim = pytest.importorskip("stim")
    from stabilizer_qec import pymatching as ours

    c = stim.Circuit.generated("surface_code:rotated_memory_z", distance=3, rounds=3, after_clifford_depolarization=0.02)
    (tmp_path / "c.dem").write_text(str(c.detector_error_model(decompose_errors=True)))
    dets, obs = c.compile_detector_sampler(seed=1).sample(200, separate_observables=True)
    stim.write_shot_data_file(data=dets, path=str(tmp_path / "d.b8"), format="b8", num_detectors=c.num_detectors)
    stim.write_shot_data_file(data=obs, path=str(tmp_path / "o.01"), format="01", num_observables=c.num_observables)
    both = np.concatenate([dets, obs], axis=1)
    stim.write_shot_data_file(data=both, path=str(tmp_path / "da.01"), format="01", num_detectors=c.num_detectors, num_observables=c.num_observables)
    extra = ["--enable_correlations"] if enable_correlations else []
    runs = [
        ["predict", "--dem", "c.dem", "--in", "d.b8", "--in_format", "b8", "--out", "OUT"],
        ["predict", "--dem", "c.dem", "--in", "d.b8", "--in_format", "b8", "--out_format", "b8", "--out", "OUT"],
        ["count_mistakes", "--dem", "c.dem", "--in", "d.b8", "--in_format", "b8", "--obs_in", "o.01", "--out", "OUT"],
        ["count_mistakes", "--dem", "c.dem", "--in", "da.01", "--in_includes_appended_observables", "--out", "OUT"],
    ]
    for run in runs:
        outs = []
        for name, cli in (("ours", ours.cli), ("theirs", pymatching.cli)):
            args = [str(tmp_path / a) if a.endswith((".dem", ".b8", ".01")) else a for a in run + extra]
            args[args.index("OUT")] = str(tmp_path / name)
            cli(command_line_args=args)
            outs.append((tmp_path / name).read_bytes())
        assert outs[0] == outs[1], run
    # PyMatching 2.4's predict fails on appended observables ("Hit index from data is too
    # large"); here they are stripped, and the predictions are those of the bare events.
    ours.cli(command_line_args=["predict", "--dem", str(tmp_path / "c.dem"), "--in", str(tmp_path / "da.01"), "--in_includes_appended_observables", "--out", str(tmp_path / "appended")] + extra)
    ours.cli(command_line_args=["predict", "--dem", str(tmp_path / "c.dem"), "--in", str(tmp_path / "d.b8"), "--in_format", "b8", "--out", str(tmp_path / "bare")] + extra)
    assert (tmp_path / "appended").read_bytes() == (tmp_path / "bare").read_bytes()
    with pytest.raises(ValueError, match="Unrecognized command"):
        ours.cli(command_line_args=["nope"])


def test_api_surface():
    """Every public name of PyMatching's module and of its Matching, with every parameter and
    its default."""
    import inspect

    from stabilizer_qec import pymatching as ours

    assert [n for n in dir(pymatching) if not n.startswith("_") and not hasattr(ours, n)] == []
    for name in dir(pymatching.Matching):
        if name.startswith("_") and name not in ("__init__", "__repr__", "__reduce__"):
            continue
        assert hasattr(sq.Matching, name), name
        try:
            theirs = inspect.signature(getattr(pymatching.Matching, name)).parameters
        except (TypeError, ValueError):
            continue
        mine = inspect.signature(getattr(sq.Matching, name)).parameters
        for p, param in theirs.items():
            assert p in mine, (name, p)
            if param.default is not inspect.Parameter.empty:
                assert repr(mine[p].default) == repr(param.default), (name, p)
