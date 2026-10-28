"""stabilizer_qec.mwpf against MWPF 0.2.12: the same classes and methods; every example code's graph,
positions, snapshot and random errors seed for seed; the solvers' subgraphs, weight bounds and
snapshots on random syndromes (with erasures, heralds and override weights), their configurations
and the errors for bad ones; JSON and pickles of initializers and syndromes; matrices as tables;
benchmark suites and HTML exports."""

from __future__ import annotations

import json
import sys
import pickle
import random

import pytest

from conftest import reference

ref = reference("mwpf")

import stabilizer_qec.mwpf as mw  # noqa: E402

# QEC-Playground's codes: built on the qecp crate, which this package does not include
EXCLUDED = {"QECPlaygroundCode"}
CODES = ["CodeCapacityRepetitionCode", "CodeCapacityPlanarCode", "CodeCapacityDepolarizePlanarCode", "CodeCapacityColorCode", "CodeCapacityTailoredCode"]
SOLVERS = ["SolverSerialJointSingleHair", "SolverSerialSingleHair", "SolverSerialUnionFind"]


def code_pair(name, d, p=0.05):
    if name == "CodeCapacityTailoredCode":
        return ref.CodeCapacityTailoredCode(d=d, pxy=p / 10, pz=p), mw.CodeCapacityTailoredCode(d=d, pxy=p / 10, pz=p)
    return getattr(ref, name)(d=d, p=p), getattr(mw, name)(d=d, p=p)


def outcome(f):
    """A call's result, or its exception's type name and message."""
    try:
        return ("ok", f())
    except BaseException as e:  # noqa: BLE001 (PanicException derives from BaseException)
        return (type(e).__name__, str(e))


def test_api_surface():
    for name in dir(ref.mwpf):
        if name.startswith("__") or name in EXCLUDED:
            continue
        assert hasattr(mw, name), name
        a, b = getattr(ref.mwpf, name), getattr(mw, name)
        if isinstance(a, type):
            # public names (frozendict's and frozenlist's classes, which MWPF re-exports, are written
            # in Python here, with private helpers of their own). frozendict's are its pure-Python
            # implementation's, which it uses from Python 3.11; below that its C extension, with
            # fewer methods, may stand in.
            if name == "frozendict" and getattr(sys.modules["frozendict"], "c_ext", False):
                continue
            assert {m for m in dir(a) if not m.startswith("_")} == {m for m in dir(b) if not m.startswith("_")}, name


@pytest.mark.parametrize("name", CODES)
@pytest.mark.parametrize("d", [3, 5, 7])
def test_codes(name, d):
    a, b = code_pair(name, d)
    assert repr(a) == repr(b)
    assert repr(a.get_initializer()) == repr(b.get_initializer())
    assert repr(a.get_positions()) == repr(b.get_positions())
    assert a.snapshot() == b.snapshot() and a.snapshot(False) == b.snapshot(False)
    assert a.vertex_num() == b.vertex_num()
    for seed in range(30):
        assert repr(a.generate_random_errors(seed)) == repr(b.generate_random_errors(seed)), seed
        assert repr(a.get_syndrome()) == repr(b.get_syndrome())
    a.set_erasure_probability(0.1)
    b.set_erasure_probability(0.1)
    for seed in range(30):
        assert repr(a.generate_random_errors(seed)) == repr(b.generate_random_errors(seed)), seed


def bounds(weight_range):
    """A weight range's bounds as floats (each package's `Rational` compares only with its own)."""
    return weight_range.lower.float(), weight_range.upper.float()


def solve_both(name, d, seeds, config=None, erasure=0.0, p=0.05):
    a, b = code_pair(name, d, p)
    if erasure:
        a.set_erasure_probability(erasure)
        b.set_erasure_probability(erasure)
    for solver_name in SOLVERS:
        sa = getattr(ref, solver_name)(a.get_initializer(), config)
        sb = getattr(mw, solver_name)(b.get_initializer(), config)
        assert sa.config == sb.config
        for seed in seeds:
            a.generate_random_errors(seed)
            b.generate_random_errors(seed)
            sa.solve(a.get_syndrome())
            sb.solve(b.get_syndrome())
            ra, rb = sa.subgraph_range(), sb.subgraph_range()
            assert bounds(ra[1]) == bounds(rb[1]), (solver_name, seed)
            assert list(ra[0]) == list(rb[0]), (solver_name, seed)
            sa.clear()
            sb.clear()


@pytest.mark.parametrize("name", CODES)
@pytest.mark.parametrize("d", [3, 5])
def test_solvers(name, d):
    solve_both(name, d, range(20))


@pytest.mark.parametrize("name", ["CodeCapacityPlanarCode", "CodeCapacityColorCode"])
def test_solvers_erasures(name):
    solve_both(name, 5, range(20), erasure=0.05)


@pytest.mark.parametrize("config", [{}, {"cluster_node_limit": 0}, {"cluster_node_limit": 5}, {"timeout": 0.0}, {"primal": {"cluster_node_limit": 3}}, {"only_solve_primal_once": True}])
def test_solver_configs(config):
    solve_both("CodeCapacityColorCode", 5, range(10), config=config, p=0.1)


def small_initializer(lib):
    return lib.SolverInitializer(4, [lib.HyperEdge([0, 1], 1.5), lib.HyperEdge([1, 2], 2.0), lib.HyperEdge([2, 3, 0], 3.0), lib.HyperEdge([3], 0.5)])


CONFIGS = [
    None, {}, {"cluster_node_limit": 7}, {"timeout": 2.5}, {"primal": [1.0]}, {"primal": [1.0, 5, True]}, {"primal": [1.0, 5, True, 4]}, {"primal": ["x"]},
    {"primal": True}, {"primal": {}}, {"primal": {"cluster_node_limit": 7}}, {"primal": [2, 3]}, {"primal": [None]}, {"primal": {"only_solve_primal_once": True, "timeout": 3}},
    {"primal": {"timeout": "1.5"}}, {"primal": {"cluster_node_limit": 2**62}}, {"timeout": 2**62}, {"cluster_node_limit": True}, {"primal": {"timeout": 1.0, "timeout2": 1}},
    {"primal": [1.0, -1]}, {"primal": [1.0, 1.5]}, {"primal": [1, 2, 0]}, {"primal": [1, 2, "x"]}, {"primal": {"timeout": [1]}}, {"primal": {"timeout": {}}},
    {"primal": {"cluster_node_limit": "x"}}, {"primal": {"cluster_node_limit": None}}, {"timeout": 1e308, "primal": None}, {"timeout": 5e-324}, {"cluster_node_limit": 0},
    {"bogus": 1}, {"timeout": -1.5}, {"cluster_node_limit": -1}, {"cluster_node_limit": 1.5}, {"timeout": "x"}, {"timeout": None}, "x", 3, [1],
]


@pytest.mark.parametrize("solver_name", SOLVERS)
def test_config_errors(solver_name):
    for config in CONFIGS:
        a = outcome(lambda: getattr(ref, solver_name)(small_initializer(ref), config).config)
        b = outcome(lambda: getattr(mw, solver_name)(small_initializer(mw), config).config)
        assert a == b, config


def test_syndromes_and_bounds():
    for lib_pair in [(ref, mw)]:
        results = []
        for lib in lib_pair:
            init = small_initializer(lib)
            solver = lib.SolverSerialJointSingleHair(init)
            out = []
            for syndrome in [
                lib.SyndromePattern([0, 1]),
                lib.SyndromePattern([0, 3]),
                lib.SyndromePattern([1, 3], erasures=[2]),
                lib.SyndromePattern([0, 2], heralds=[]),
                lib.SyndromePattern([0, 1], override_weights=[0.5, 1.0, 1.0, 1.0]),
                lib.SyndromePattern([2]),
            ]:
                out.append((repr(syndrome), outcome(lambda: (lambda r: (list(r[0]), bounds(r[1])))((solver.solve(syndrome), solver.subgraph_range())[1]))))
                solver.clear()
            results.append(out)
        assert results[0] == results[1]


def test_json_and_pickles():
    # `from_json` takes the Python object the JSON text decodes to (MWPF's own round trip through a
    # string panics, on both packages alike)
    for make in [small_initializer, lambda lib: lib.CodeCapacityColorCode(d=5, p=0.1).get_initializer()]:
        a, b = make(ref), make(mw)
        assert a.to_json() == b.to_json()
        assert repr(ref.SolverInitializer.from_json(json.loads(a.to_json()))) == repr(mw.SolverInitializer.from_json(json.loads(b.to_json())))
        assert outcome(lambda: ref.SolverInitializer.from_json(a.to_json())) == outcome(lambda: mw.SolverInitializer.from_json(b.to_json()))
        assert repr(pickle.loads(pickle.dumps(b))) == repr(b)
    for text in ['{"vertex_num": 2}', '{"vertex_num": 2, "weighted_edges": []}', '{"vertex_num": -1, "weighted_edges": []}', '{"vertex_num": 2, "weighted_edges": [[[0, 1], 1.5]]}',
                 '{"vertex_num": 2, "weighted_edges": [{"vertices": [0, 1], "weight": 1.5}]}', '{"vertex_num": 2, "weighted_edges": [], "x": 1}', "[2, []]", "[2, [[[0, 1], 1.5]], []]", "3", "null"]:
        value = json.loads(text)
        assert outcome(lambda: repr(ref.SolverInitializer.from_json(value))) == outcome(lambda: repr(mw.SolverInitializer.from_json(value))), text
    a, b = ref.SolverSerialJointSingleHair(small_initializer(ref)), mw.SolverSerialJointSingleHair(small_initializer(mw))
    assert repr(pickle.loads(pickle.dumps(b)).config) == repr(a.config)


def test_matrices():
    for lib in (ref, mw):
        m = lib.BasicMatrix()
        m.add_constraint(0, [1, 4, 16], True)
        m.add_constraint(1, [4, 23], False)
        m.add_constraint(2, [1, 23], True)
        lib.result = (str(m), repr(m), m.snapshot() if hasattr(m, "snapshot") else None)
        e = lib.EchelonMatrix()
        for v in [1, 4, 16, 23]:
            e.add_variable(v)
        e.add_constraint(0, [1, 4, 16], True)
        e.add_constraint(1, [4, 23], False)
        e.add_constraint(2, [1, 23], True)
        lib.echelon = (str(e), repr(e))
    assert ref.result == mw.result
    assert ref.echelon == mw.echelon


def test_visualizer_and_html(tmp_path):
    out = []
    for lib in (ref, mw):
        code = lib.CodeCapacityTailoredCode(d=5, pxy=0.001, pz=0.1)
        code.set_defect_vertices([10, 11, 16, 17])
        solver = lib.SolverSerialJointSingleHair(code.get_initializer(), {"cluster_node_limit": 100, "timeout": 10.0})
        visualizer = lib.Visualizer(positions=code.get_positions())
        visualizer.snapshot("syndrome", code)
        solver.solve(code.get_syndrome(), visualizer)
        subgraph, bound = solver.subgraph_range(visualizer)
        data = visualizer.get_visualizer_data()
        assert data["snapshots"]
        out.append((list(subgraph), bounds(bound), data))
    assert out[0] == out[1]


def test_html_export_compresses_as_mwpf():
    # MWPF gzips the visualizer data with flate2's GzEncoder at its default level (miniz_oxide's
    # deflate, level 6): the same bytes here, for text short and long, plain and repetitive, and
    # for a visualizer's data and the HTML it writes.
    rng = random.Random(7)
    texts = ["", "a", "hello world " * 10, "x" * 47, "x" * 48, "\u00e9t\u00e9 " * 999, "".join(rng.choice("ab") for _ in range(70000)),
             "".join(rng.choice("0123456789,[]{}\":.") for _ in range(33000)), "".join(chr(rng.randint(32, 126)) for _ in range(200000)),
             json.dumps([{"w": rng.randint(1, 1000), "s": rng.random() < 0.1} for _ in range(30000)])]
    for text in texts:
        assert mw.HTMLExport.compress_content(text) == ref.HTMLExport.compress_content(text), len(text)
        assert ref.HTMLExport.decompress_content(mw.HTMLExport.compress_content(text)) == text
    # A solved visualizer's data, compressed by each (MWPF's own subgraph order varies run to run,
    # with its hash maps' seeds, so the data is this package's), and a page of a code's snapshot.
    code = mw.CodeCapacityPlanarCode(15, 0.08)
    code.generate_random_errors(11)
    solver = mw.SolverSerialJointSingleHair(code.get_initializer())
    visualizer = mw.Visualizer(positions=code.get_positions())
    solver.solve(code.get_syndrome(), visualizer)
    solver.subgraph(visualizer)
    data = json.dumps(visualizer.get_visualizer_data())
    assert mw.HTMLExport.compress_content(data) == ref.HTMLExport.compress_content(data)
    pages = []
    for lib in (ref, mw):
        code = lib.CodeCapacityPlanarCode(15, 0.08)
        code.generate_random_errors(11)
        visualizer = lib.Visualizer(positions=code.get_positions())
        visualizer.snapshot("syndrome", code)
        pages.append(visualizer.generate_html())
    assert len(pages[1]) < 500_000 and pages[0] == pages[1]


@pytest.mark.parametrize("seed", range(20, 60, 10))
def test_depolarizing_planar_more_seeds(seed):
    # the single-hair plugins' linear programs at their most degenerate (an optimum among many, and
    # right-hand sides zero up to round-off): every relaxer the optimizer returns, and so every
    # subgraph and bound, is MWPF's (larger codes take MWPF itself minutes on some seeds)
    for solver_name in ["SolverSerialJointSingleHair", "SolverSerialSingleHair"]:
        results = []
        for lib in (ref, mw):
            code = lib.CodeCapacityDepolarizePlanarCode(d=5, p=0.05)
            solver = getattr(lib, solver_name)(code.get_initializer())
            for s in range(seed, seed + 10):
                code.generate_random_errors(s)
                solver.solve(code.get_syndrome())
                subgraph, bound = solver.subgraph_range()
                results.append((list(subgraph), bounds(bound)))
                solver.clear()
        assert results[:10] == results[10:], solver_name


@pytest.mark.parametrize("decoder", [dict(decoder_type="SolverSerialJointSingleHair"), dict(decoder_type="SolverSerialSingleHair"), dict(decoder_type="SolverSerialUnionFind"), dict(decoder_type="SolverSerialJointSingleHair", c=5), dict(bp=True)])
def test_sinter_decoder_on_circuits(decoder):
    import numpy as np

    import stabilizer_qec

    real_stim = pytest.importorskip("stim")
    if not hasattr(ref, "SinterMWPFDecoder"):
        pytest.skip("MWPF's sinter decoders need its `stim` extra (frozendict, frozenlist)")

    for task, distance in [("surface_code:rotated_memory_x", 3), ("surface_code:rotated_memory_z", 5), ("color_code:memory_xyz", 3)]:
        circuit = real_stim.Circuit.generated(task, distance=distance, rounds=distance, after_clifford_depolarization=0.006, before_measure_flip_probability=0.006)
        dem_text = str(circuit.detector_error_model(decompose_errors=False))
        dets = circuit.compile_detector_sampler(seed=distance).sample(200, bit_packed=True)
        predictions = []
        for lib, stim_lib in ((ref, real_stim), (mw, stabilizer_qec)):
            compiled = lib.SinterMWPFDecoder(**decoder).compile_decoder_for_dem(dem=stim_lib.DetectorErrorModel(dem_text))
            predictions.append(np.asarray(compiled.decode_shots_bit_packed(bit_packed_detection_event_data=dets)))
        assert np.array_equal(predictions[0], predictions[1]), (task, distance)


def test_stims_own_objects():
    """Sinter, and MWPF's users, give its code Stim's own circuits and models: they are read as
    this package's, so the predictions are MWPF's (not all zero, as when the observables of
    Stim's targets went unseen) and the helpers take and give Stim's circuits."""
    import numpy as np

    real_stim = pytest.importorskip("stim")
    if not hasattr(ref, "SinterMWPFDecoder"):
        pytest.skip("MWPF's sinter decoders need its `stim` extra (frozendict, frozenlist)")
    import stabilizer_qec
    from test_mwpf_stim_objects import HERALDED

    # MWPF's own test (tests/python/test_sinter.py): both detectors lit means L0 flipped.
    dem = real_stim.DetectorErrorModel("error(0.02) D0\nerror(0.02) D0 D1 L0\nerror(0.1) D1\nerror(0.01) D1 L0\n")
    lit = np.packbits(np.array([[1, 1]]), axis=-1, bitorder="little")
    for lib in (ref, mw):
        compiled = lib.SinterMWPFDecoder(panic_action=lib.PanicAction.RAISE).compile_decoder_for_dem(dem=dem)
        assert compiled.predictor.get_observable_bits(compiled.decode_shots_bit_packed(bit_packed_detection_event_data=lit)) == [1]

    circuit = real_stim.Circuit.generated("surface_code:rotated_memory_x", distance=3, rounds=3, after_clifford_depolarization=0.01, before_measure_flip_probability=0.01)
    dem = circuit.detector_error_model(decompose_errors=False)
    dets, obs = circuit.compile_detector_sampler(seed=3).sample(300, separate_observables=True, bit_packed=True)
    real = ref.SinterMWPFDecoder().compile_decoder_for_dem(dem=dem).decode_shots_bit_packed(bit_packed_detection_event_data=dets)
    mine = mw.SinterMWPFDecoder().compile_decoder_for_dem(dem=dem).decode_shots_bit_packed(bit_packed_detection_event_data=dets)
    ours = mw.SinterMWPFDecoder().compile_decoder_for_dem(dem=stabilizer_qec.DetectorErrorModel(str(dem))).decode_shots_bit_packed(bit_packed_detection_event_data=dets)
    assert np.array_equal(real, mine) and np.array_equal(mine, ours)
    assert np.any(mine != 0) and (mine != obs).any(axis=1).sum() < obs.any(axis=1).sum()

    heralded = real_stim.Circuit(HERALDED)
    hdem = heralded.detector_error_model(approximate_disjoint_errors=True)
    hdets = heralded.compile_detector_sampler(seed=4).sample(200, bit_packed=True)
    predictions = []
    for lib in (ref, mw):
        decoder = lib.SinterMWPFDecoder(pass_circuit=True).with_circuit(heralded)
        predictions.append(decoder.compile_decoder_for_dem(dem=hdem).decode_shots_bit_packed(bit_packed_detection_event_data=hdets))
    assert np.array_equal(predictions[0], predictions[1])
    assert str(ref.heralded_dem.HeraldedDetectorErrorModel.of(heralded)) == str(mw.heralded_dem.HeraldedDetectorErrorModel.of(heralded))
    assert str(ref.ref_circuit.RefCircuit.of(heralded)) == str(mw.ref_circuit.RefCircuit.of(heralded))
    for helper in ("add_herald_detectors", "remove_herald_detectors"):
        a, b = getattr(ref.heralded_dem, helper)(heralded), getattr(mw.heralded_dem, helper)(heralded)
        assert isinstance(b, real_stim.Circuit) and a == b, helper
    target = real_stim.GateTarget(0)
    a = ref.ref_circuit.RefInstruction.new_heralded_erase((target,), 0.1)
    b = mw.ref_circuit.RefInstruction.new_heralded_erase((target,), 0.1)
    assert str(ref.ref_circuit.RefCircuit.of([a])) == str(mw.ref_circuit.RefCircuit.of([b]))


def test_random_solver_inputs():
    rng = random.Random(5)
    for trial in range(40):
        n = rng.randint(2, 7)
        edges = [(sorted(rng.sample(range(n), rng.randint(1, min(3, n)))), rng.choice([0.5, 1.0, 1.5, 2.0, 3.0])) for _ in range(rng.randint(n, 2 * n))]
        defects = sorted(rng.sample(range(n), rng.randint(0, n)))
        res = []
        for lib in (ref, mw):
            init = lib.SolverInitializer(n, [lib.HyperEdge(v, w) for v, w in edges])
            solver = lib.SolverSerialJointSingleHair(init)
            res.append(outcome(lambda: (lambda r: (list(r[0]), bounds(r[1])))((solver.solve(lib.SyndromePattern(defects)), solver.subgraph_range())[1])))
        assert res[0][0] == res[1][0], (trial, res)
        if res[0][0] == "ok":
            assert res[0][1][1:] == res[1][1][1:], (trial, edges, defects)
