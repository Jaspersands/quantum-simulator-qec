"""stabilizer_qec.fusion_blossom against fusion_blossom 0.2.13: the same classes and methods; every
example code's graph, positions, snapshot and random errors seed for seed; the serial solver's
matchings, subgraphs, dual sums and snapshots, with erasures, dynamic weights and any tree-size
limit; the partitioned solvers' results; the visualizer's files byte for byte; the JSON text of
initializers and partitions; error-pattern files; Stim's bit-packed prediction; and sinter's
fusion_blossom decoder."""

from __future__ import annotations

import os
import random

import numpy as np
import pytest

from conftest import reference

ref = reference("fusion_blossom")

import stabilizer_qec.fusion_blossom as fb  # noqa: E402

PLAIN = ["CodeCapacityRepetitionCode", "CodeCapacityPlanarCode", "CodeCapacityRotatedCode"]
TIMED = ["PhenomenologicalPlanarCode", "CircuitLevelPlanarCode", "PhenomenologicalRotatedCode"]


def code_pair(name, d, p=0.05, **kwargs):
    if name in TIMED:
        return getattr(ref, name)(d=d, noisy_measurements=d, p=p, **kwargs), getattr(fb, name)(d=d, noisy_measurements=d, p=p, **kwargs)
    return getattr(ref, name)(d=d, p=p, **kwargs), getattr(fb, name)(d=d, p=p, **kwargs)


def test_api_surface():
    for name in dir(ref):
        if name.startswith("__") or name in ("fusion_blossom", "helper"):
            continue
        assert hasattr(fb, name), name
        a, b = getattr(ref, name), getattr(fb, name)
        if isinstance(a, type) and name != "PyMut":
            assert {m for m in dir(a) if not m.startswith("__")} == {m for m in dir(b) if not m.startswith("__")}, name
    for name in dir(ref.helper):
        if not name.startswith("_") and callable(getattr(ref.helper, name)):
            assert callable(getattr(fb.helper, name, None)), name
    assert set(ref.helper.visualizer_website) == set(fb.helper.visualizer_website)
    assert hasattr(fb.helper, "bottle")


@pytest.mark.parametrize("name", PLAIN + TIMED)
@pytest.mark.parametrize("d", [3, 5])
def test_codes(name, d):
    a, b = code_pair(name, d)
    assert repr(a) == repr(b)
    assert repr(a.get_initializer()) == repr(b.get_initializer())
    assert repr(a.get_positions()) == repr(b.get_positions())
    assert a.snapshot() == b.snapshot() and a.snapshot(False) == b.snapshot(False)
    assert a.vertex_num() == b.vertex_num() and a.sanity_check() == b.sanity_check()
    for seed in range(25):
        assert repr(a.generate_random_errors(seed)) == repr(b.generate_random_errors(seed)), seed
    a.set_erasure_probability(0.1)
    b.set_erasure_probability(0.1)
    for seed in range(25):
        assert repr(a.generate_random_errors(seed)) == repr(b.generate_random_errors(seed)), seed
    edges = list(range(0, len(a.edges), 3))
    assert repr(a.generate_errors(edges)) == repr(b.generate_errors(edges))
    assert a.get_defect_vertices() == b.get_defect_vertices() and a.get_erasures() == b.get_erasures()
    a.compute_weights(1000)
    b.compute_weights(1000)
    assert repr(a.edges) == repr(b.edges)


def test_circuit_level_diagonal():
    a = ref.CircuitLevelPlanarCode.new_diagonal(5, 5, 0.01, 500, 0.002)
    b = fb.CircuitLevelPlanarCode.new_diagonal(5, 5, 0.01, 500, 0.002)
    assert repr(a) == repr(b)
    assert repr(a.generate_random_errors(7)) == repr(b.generate_random_errors(7))


def same_solver_state(sa, sb, ma, mb):
    pa, pb = sa.perfect_matching(), sb.perfect_matching()
    assert pa.peer_matchings == pb.peer_matchings
    assert pa.virtual_matchings == pb.virtual_matchings
    assert sa.subgraph() == sb.subgraph()
    assert sa.sum_dual_variables() == sb.sum_dual_variables()
    assert sa.snapshot() == sb.snapshot()


@pytest.mark.parametrize("name", PLAIN + TIMED)
@pytest.mark.parametrize("max_tree_size", [None, 0, 1, 5])
def test_serial_solver(name, max_tree_size):
    a, b = code_pair(name, 5, p=0.08)
    a.set_erasure_probability(0.05)
    b.set_erasure_probability(0.05)
    sa = ref.SolverSerial(a.get_initializer(), max_tree_size=max_tree_size)
    sb = fb.SolverSerial(b.get_initializer(), max_tree_size=max_tree_size)
    rng = random.Random(1)
    for seed in range(15):
        ya, yb = a.generate_random_errors(seed), b.generate_random_errors(seed)
        if seed % 3 == 1:
            weights = [(rng.randrange(len(a.edges)), 2 * rng.randrange(0, 600)) for _ in range(4)]
            ya = ref.SyndromePattern(ya.defect_vertices, dynamic_weights=weights)
            yb = fb.SyndromePattern(yb.defect_vertices, dynamic_weights=weights)
        sa.solve(ya)
        sb.solve(yb)
        same_solver_state(sa, sb, ya, yb)
        sa.clear()
        sb.clear()


def test_visualizer_files(tmp_path):
    a, b = code_pair("PhenomenologicalPlanarCode", 3, p=0.1)
    syndrome = 11
    for module, code, folder in ((ref, a, tmp_path / "ref"), (fb, b, tmp_path / "ours")):
        folder.mkdir()
        visualizer = module.Visualizer(filepath=str(folder / "v.json"), positions=code.get_positions())
        solver = module.SolverSerial(code.get_initializer())
        solver.solve(code.generate_random_errors(syndrome), visualizer)
        solver.subgraph(visualizer)
        visualizer.snapshot("code", code)
        visualizer.snapshot_combined("both", [code, solver])
        visualizer.snapshot_value("value", code.snapshot())
        visualizer.snapshot_combined_value("values", [code.snapshot(), solver.snapshot()])
    assert (tmp_path / "ref" / "v.json").read_text() == (tmp_path / "ours" / "v.json").read_text()
    positions = [ref.VisualizePosition(1.5, -2, 3), ref.VisualizePosition(0.25, 4, 1)]
    ours = [fb.VisualizePosition(1.5, -2, 3), fb.VisualizePosition(0.25, 4, 1)]
    assert repr(ref.center_positions(positions)) == repr(fb.center_positions(ours))


def partition_pair(d=5, rounds=6):
    a, b = code_pair("PhenomenologicalPlanarCode", d, p=0.03)
    out = []
    for module, code in ((ref, a), (fb, b)):
        initializer = code.get_initializer()
        n = initializer.vertex_num
        layer = n // (d + 1)
        config = module.PartitionConfig(n)
        config.partitions = [module.VertexRange(0, 2 * layer), module.VertexRange(3 * layer, n)]
        config.fusions = [(0, 1)]
        out.append((module, code, initializer, config))
    return out


def test_partitions_and_json():
    (_, _, ia, ca), (_, _, ib, cb) = partition_pair()
    assert ia.to_json() == ib.to_json()
    assert ca.to_json() == cb.to_json()
    assert repr(ca) == repr(cb)
    info_a, info_b = ca.info(), cb.info()
    assert info_a.to_json() == info_b.to_json() and repr(info_a) == repr(info_b)
    for ua, ub in zip(info_a.units, info_b.units):
        assert ua.to_json() == ub.to_json()
        assert repr(fb.PartitionUnitInfo.from_json(ub.to_json())) == repr(ua)
    assert repr(fb.PartitionInfo.from_json(info_a.to_json())) == repr(info_a)
    assert repr(fb.SolverInitializer.from_json(ia.to_json())) == repr(ia)
    syndrome = list(range(0, ia.vertex_num, 7))
    assert repr(info_a.partition_syndrome_unordered(ref.SyndromePattern(syndrome))) == repr(info_b.partition_syndrome_unordered(fb.SyndromePattern(syndrome)))
    ra, rb = ref.VertexRange(2, 9), fb.VertexRange(2, 9)
    assert repr(ra.fuse(ref.VertexRange(11, 20))) == repr(rb.fuse(fb.VertexRange(11, 20)))
    assert (ra.len(), ra.contains_any([1, 9]), ra.contains_any([3])) == (rb.len(), rb.contains_any([1, 9]), rb.contains_any([3]))
    ra.bias_by(3)
    rb.bias_by(3)
    assert repr(ra) == repr(rb) and repr(ref.DefectRange.new_length(4, 6)) == repr(fb.DefectRange.new_length(4, 6))
    with pytest.raises(TypeError):
        fb.PartitionConfig.from_json('{"vertex_num": 3, "partitions": [], "fusions": [], "extra": 1}')


@pytest.mark.parametrize("which", ["SolverDualParallel", "SolverParallel"])
def test_parallel_solvers(which):
    pairs = partition_pair()
    solvers = []
    for module, code, initializer, config in pairs:
        solvers.append(getattr(module, which)(initializer, config.info(), {}))
    (_, a, ia, _), (_, b, _, _) = pairs
    serial = ref.SolverSerial(ia)
    for seed in range(20):
        ya, yb = a.generate_random_errors(seed), b.generate_random_errors(seed)
        solvers[0].solve(ya)
        solvers[1].solve(yb)
        serial.solve(ya)
        assert solvers[0].subgraph() == solvers[1].subgraph()
        assert solvers[0].sum_dual_variables() == solvers[1].sum_dual_variables() == serial.sum_dual_variables()
        if which == "SolverParallel":
            assert solvers[0].defect_perfect_matching() == solvers[1].defect_perfect_matching()
        for s in solvers + [serial]:
            s.clear()


def test_error_pattern_reader(tmp_path):
    a, _ = code_pair("CodeCapacityPlanarCode", 5, p=0.1)
    lines = ["Syndrome Pattern v1.0   <initializer> <positions> <syndrome_pattern>*", a.get_initializer().to_json(),
             "[" + ",".join(f'{{"i":{p.i},"j":{p.j},"t":{p.t}}}' for p in a.get_positions()) + "]"]
    lines += ['{"defect_vertices":%s,"erasures":[],"dynamic_weights":[]}' % a.generate_random_errors(s).defect_vertices for s in range(5)]
    path = tmp_path / "patterns.txt"
    path.write_text("\n".join(lines) + "\n")
    ra, rb = ref.ErrorPatternReader(str(path), cyclic_syndrome=True), fb.ErrorPatternReader(str(path), cyclic_syndrome=True)
    assert repr(ra) == repr(rb)
    for _ in range(8):
        assert repr(ra.generate_random_errors()) == repr(rb.generate_random_errors())


def test_stim_bit_packed_prediction(tmp_path):
    a, b = code_pair("PhenomenologicalRotatedCode", 5, p=0.02)
    num_dets = a.vertex_num()
    rng = np.random.default_rng(3)
    shots = 40
    dets = np.zeros((shots, num_dets), dtype=np.uint8)
    for s in range(shots):
        y = a.generate_random_errors(s)
        dets[s, y.defect_vertices] = 1
    (tmp_path / "in.b8").write_bytes(np.packbits(dets, axis=1, bitorder="little").tobytes())
    masks = [int(x) for x in rng.integers(0, 4, len(a.edges))]
    for module, code, out in ((ref, a, "ref.b8"), (fb, b, "ours.b8")):
        module.SolverSerial(code.get_initializer()).stim_integration_predict_bit_packed_data(str(tmp_path / "in.b8"), str(tmp_path / out), masks, shots, num_dets, 2)
    assert (tmp_path / "ref.b8").read_bytes() == (tmp_path / "ours.b8").read_bytes()


def test_sinter_decoder(monkeypatch):
    import stabilizer_qec as stim
    from stabilizer_qec.sinter._decoding import _decoding_fusion_blossom as decoding

    circuit = stim.Circuit.generated("surface_code:rotated_memory_x", distance=5, rounds=5, after_clifford_depolarization=0.01)
    dem = circuit.detector_error_model(decompose_errors=True)
    dets, _ = circuit.compile_detector_sampler(seed=5).sample(300, separate_observables=True, bit_packed=True)
    ours = decoding.FusionBlossomDecoder().compile_decoder_for_dem(dem=dem).decode_shots_bit_packed(bit_packed_detection_event_data=dets)
    monkeypatch.setattr(decoding, "fusion_blossom", ref)
    theirs = decoding.FusionBlossomDecoder().compile_decoder_for_dem(dem=dem).decode_shots_bit_packed(bit_packed_detection_event_data=dets)
    assert np.array_equal(ours, theirs)


def test_weight_of_p_and_names():
    for p in (0.001, 0.1, 0.5):
        assert fb.weight_of_p(p) == ref.weight_of_p(p)
    assert fb.static_visualize_data_filename() == ref.static_visualize_data_filename()
    assert len(fb.auto_visualize_data_filename()) == len(ref.auto_visualize_data_filename())
    assert repr(fb.SyndromePattern([1, 2], erasures=[3])) == repr(ref.SyndromePattern([1, 2], erasures=[3]))
    assert repr(fb.SyndromePattern(syndrome_vertices=[4])) == repr(ref.SyndromePattern(syndrome_vertices=[4]))
    for module in (ref, fb):
        with pytest.raises(BaseException):
            module.SyndromePattern([1], syndrome_vertices=[2])
    assert fb.helper.__file__.endswith(os.path.join("helper", "__init__.py"))
