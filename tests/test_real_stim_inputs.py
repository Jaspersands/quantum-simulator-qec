"""The ports take Stim's own objects too: where their originals check ``isinstance(x, stim.Circuit)``
(or ``stim.DetectorErrorModel``, ``stim.DemInstruction``), their ``stim`` is this package, so a
real Stim circuit or model was refused (TypeError, ValueError, AssertionError) or, where the check
picks a branch, silently taken for something else. Each such entry reads a Stim object from its
text first; given one, a port answers as it does given this package's copy."""

from __future__ import annotations

import numpy as np
import pytest

import stabilizer_qec as sq

stim = pytest.importorskip("stim", minversion="1.16")

TASKS = [
    ("surface_code:rotated_memory_z", 3, 2),
    ("repetition_code:memory", 5, 3),
    ("color_code:memory_xyz", 3, 2),
]


def real_circuit(task="surface_code:rotated_memory_z", d=3, r=2):
    return stim.Circuit.generated(task, distance=d, rounds=r, after_clifford_depolarization=0.01, before_measure_flip_probability=0.01, after_reset_flip_probability=0.005)


def mine(x):
    """This package's copy of a Stim object."""
    return {"Circuit": sq.Circuit, "DetectorErrorModel": sq.DetectorErrorModel}[type(x).__name__](str(x))


def mine_instr(instr):
    """This package's copy of a Stim instruction (read from a model's text: an instruction's own
    rounds its probability to 6 digits)."""
    m = stim.DetectorErrorModel()
    m.append(instr)
    return sq.DetectorErrorModel(str(m))[0]


def same(a, b):
    """Outcomes as text (this package's objects print like Stim's), containers element by element."""
    def norm(x):
        if isinstance(x, (sq.Circuit, sq.DetectorErrorModel, sq.DemInstruction, sq.DemTarget, stim.Circuit, stim.DetectorErrorModel, stim.DemInstruction)):
            return str(x)
        if isinstance(x, dict):
            return {norm(k): norm(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return type(x)(norm(v) for v in x)
        if isinstance(x, set):
            return {norm(v) for v in x}
        if isinstance(x, np.ndarray):
            return x.tolist()
        return x

    return norm(a) == norm(b)


def outcome(fn, *args):
    """``fn(*args)``, or the exception it raises (its type and message): the same failure, where
    there is one, is the expectation."""
    try:
        return fn(*args)
    except Exception as e:  # noqa: BLE001
        return ("raised", type(e).__name__, str(e))


# qec-util (stabilizer_qec.analysis)

CIRCUIT_FUNCS = [
    ("remove_gauge_detectors", lambda c: (c,)),
    ("remove_detectors", lambda c: (c, [0, 2])),
    ("remove_observables", lambda c: (c,)),
    ("observables_to_detectors", lambda c: (c,)),
    ("move_observables_to_end", lambda c: (c,)),
    ("move_first_resets_to_beginning", lambda c: (c,)),
    ("format_rec_targets", lambda c: (c,)),
    ("merge_observable_definitions", lambda c: (c,)),
    ("redefine_observables", lambda c: (c, {0: [0, 1]})),
]


@pytest.mark.parametrize("task,d,r", TASKS)
@pytest.mark.parametrize("name,args", CIRCUIT_FUNCS, ids=[f[0] for f in CIRCUIT_FUNCS])
def test_qec_util_circuit_tools_take_stim_circuits(task, d, r, name, args):
    from stabilizer_qec.analysis import circuits

    fn = getattr(circuits, name)
    c = real_circuit(task, d, r)
    if name == "redefine_observables" and task != "repetition_code:memory":
        pytest.skip("qec-util's test case is a repetition code's")
    got = outcome(fn, *args(c))
    assert same(got, outcome(fn, *args(mine(c))))
    assert not (isinstance(got, tuple) and got[:2] == ("raised", "TypeError"))


DEM_FUNCS = [
    ("remove_gauge_detectors", lambda m: (m,)),
    ("get_max_weight_hyperedge", lambda m: (m,)),
    ("disjoint_graphs", lambda m: (m,)),
    ("get_flippable_detectors", lambda m: (m,)),
    ("get_flippable_observables", lambda m: (m,)),
    ("contains_only_edges", lambda m: (m,)),
    ("observables_to_detectors", lambda m: (m,)),
    ("get_errors_triggering_detectors", lambda m: (m,)),
    ("only_errors", lambda m: (m,)),
    ("remove_hyperedges", lambda m: (m,)),
    ("separate_edges_and_hyperedges", lambda m: (m,)),
    ("decomposed_graphlike_dem", lambda m: (m,)),
    ("get_dem_subgraph", lambda m: (m, [0, 1, 2, 5])),
    ("dem_difference", lambda m: (m, m[:-3])),
    ("detectors_to_observables", lambda m: (m, 0)),
    ("prepare_distance2_dem_for_pymatching", lambda m: (m,)),
    ("is_instr_in_dem", lambda m: (m[0], m)),
]


@pytest.mark.parametrize("task,d,r", TASKS)
@pytest.mark.parametrize("name,args", DEM_FUNCS, ids=[f[0] for f in DEM_FUNCS])
def test_qec_util_model_tools_take_stim_models(task, d, r, name, args):
    from stabilizer_qec.analysis import dems

    fn = getattr(dems, name)
    model = real_circuit(task, d, r).detector_error_model(decompose_errors=True)
    assert same(outcome(fn, *args(model)), outcome(fn, *args(mine(model))))


def test_qec_util_hyperedge_decomposition_takes_stim_models():
    from stabilizer_qec.analysis import dem_instrs, dems

    model = real_circuit().detector_error_model(decompose_errors=False)
    got = dems.decompose_hyperedges_to_edges(model, None, True)
    assert isinstance(got, sq.DetectorErrorModel)
    assert same(got, dems.decompose_hyperedges_to_edges(mine(model), None, True))
    edges, hyperedges = dems.separate_edges_and_hyperedges(model)
    for instr in [i for i in model.flattened() if i.type == "error" and len(i.targets_copy()) > 2][:8]:
        assert same(outcome(dem_instrs.decompose_hyperedge_to_edges, instr, stim.DetectorErrorModel(str(edges)), True), outcome(dem_instrs.decompose_hyperedge_to_edges, mine_instr(instr), edges, True))


@pytest.mark.parametrize("name,extra", [("get_detectors", ()), ("get_observables", ()), ("has_separator", ()), ("decomposed_detectors", ()), ("decomposed_observables", ()), ("decomposed_instrs", ()), ("sorted_dem_instr", ()), ("remove_detectors", ([0, 1, 2],)), ("detectors_to_observables", ({0: [0]},))])
def test_qec_util_instruction_tools_take_stim_instructions(name, extra):
    from stabilizer_qec.analysis import dem_instrs

    fn = getattr(dem_instrs, name)
    model = real_circuit().detector_error_model(decompose_errors=True)
    errors = [i for i in model.flattened() if i.type == "error"]
    for instr in errors[:: max(1, len(errors) // 12)]:
        assert same(outcome(fn, instr, *extra), outcome(fn, mine_instr(instr), *extra)), (name, str(instr))


def test_qec_util_distance_takes_stim_models():
    pytest.importorskip("scipy")
    from stabilizer_qec.analysis import distance

    model = real_circuit("repetition_code:memory", 3, 2).detector_error_model()
    d, errors = distance.get_circuit_distance(model)
    assert d == 3 and isinstance(errors, sq.DetectorErrorModel)
    d0, _ = distance.get_circuit_distance_observable(model, 0)
    assert d0 == 3


def test_qec_util_sampler_takes_stim_models():
    from stabilizer_qec.analysis import samplers

    class NoCorrection:
        def decode_batch(self, syndromes):
            return np.zeros((len(syndromes), 1), dtype=bool)

    model = real_circuit("repetition_code:memory", 3, 2).detector_error_model()
    failures, kept, samples, _ = samplers.sample_failures(model, NoCorrection(), max_samples=200, batch_size=100)
    assert samples == kept == 200 and 0 <= failures <= 200


# MQT QECC

def test_mqt_circuits_take_stim_circuits():
    from stabilizer_qec.mqt_qecc.circuit_synthesis.cat_states import fault_gens_from_circuit
    from stabilizer_qec.mqt_qecc.circuit_synthesis.circuits import CliffordIsometry, CNOTCircuit

    cnots = stim.Circuit("CX 0 1 1 2 0 3\nCX 2 3")
    assert str(CNOTCircuit.from_stim_circuit(cnots).to_stim_circuit()) == str(CNOTCircuit.from_stim_circuit(mine(cnots)).to_stim_circuit())
    assert fault_gens_from_circuit(cnots) == fault_gens_from_circuit(mine(cnots))
    clifford = stim.Circuit("H 0\nCX 0 1\nS 1\nCZ 1 2")
    assert str(CliffordIsometry.from_stim_circuit(clifford).to_stim_circuit()) == str(CliffordIsometry.from_stim_circuit(mine(clifford)).to_stim_circuit())


def test_mqt_dem_to_matrices_takes_stim_models():
    pytest.importorskip("scipy")
    from stabilizer_qec.mqt_qecc.cc_decoder.stim_interface.dem_to_matrices import detector_error_model_to_check_matrices

    model = real_circuit("color_code:memory_xyz", 3, 2).detector_error_model(decompose_errors=True)
    a, b = detector_error_model_to_check_matrices(model), detector_error_model_to_check_matrices(mine(model))
    for field in ("check_matrix", "observables_matrix", "edge_check_matrix", "edge_observables_matrix", "hyperedge_to_edge_matrix"):
        assert (getattr(a, field) != getattr(b, field)).nnz == 0, field
    assert np.array_equal(a.priors, b.priors)


# qLDPC

def test_qldpc_takes_stim_models_and_circuits():
    from stabilizer_qec.qldpc import circuits, decoders

    c = real_circuit()
    model = c.detector_error_model()
    a, b = decoders.DetectorErrorModelArrays(model), decoders.DetectorErrorModelArrays(mine(model))
    assert (a.detector_flip_matrix != b.detector_flip_matrix).nnz == 0
    assert np.array_equal(a.error_probs, b.error_probs)
    assert (decoders.DetectorErrorModelArrays(c).detector_flip_matrix != decoders.DetectorErrorModelArrays(mine(c)).detector_flip_matrix).nnz == 0
    model = real_circuit("repetition_code:memory", 5, 3).detector_error_model()  # graphlike, for matching
    syndromes = mine(model).compile_sampler(seed=5).sample(50)[0]
    for build in (lambda m: decoders.lookup(max_weight=2).build(m), lambda m: decoders.mwpm().build(m)):
        assert np.array_equal(build(model).decode_batch(syndromes), build(mine(model)).decode_batch(syndromes))
    noiseless = c.without_noise()
    noise = circuits.DepolarizingNoiseModel(0.01)
    assert str(noise.noisy_circuit(noiseless)) == str(noise.noisy_circuit(mine(noiseless)))
    assert str(circuits.with_remapped_qubits(c, {q: q + 1 for q in range(c.num_qubits)})) == str(circuits.with_remapped_qubits(mine(c), {q: q + 1 for q in range(c.num_qubits)}))


def test_qldpc_noise_rule_takes_stim_circuits():
    from stabilizer_qec.qldpc.circuits.noise_model import NoiseRule

    after = stim.Circuit("DEPOLARIZE1(0.01) 0")
    assert str(NoiseRule(after=after).after) == str(NoiseRule(after=mine(after)).after)


# MWPF, Relay-BP, sinter

def test_mwpf_ref_circuit_takes_stim_circuits():
    from stabilizer_qec import mwpf
    from stabilizer_qec.mwpf.sinter_decoders import SinterMWPFDecoder

    c = real_circuit()
    assert str(mwpf.ref_circuit.RefCircuit.of(c)) == str(mwpf.ref_circuit.RefCircuit.of(mine(c)))
    assert str(SinterMWPFDecoder().with_circuit(c).circuit) == str(c)


def test_relay_bp_noise_takes_stim_circuits():
    from stabilizer_qec.relay_bp.stim.noise import insert_uniform_academic_circuit_noise

    c = real_circuit("repetition_code:memory", 3, 10).without_noise()
    assert "REPEAT" in str(c)
    assert str(insert_uniform_academic_circuit_noise(c, 0.01)) == str(insert_uniform_academic_circuit_noise(mine(c), 0.01))


@pytest.mark.parametrize("module", ["_decoding_fusion_blossom", "_decoding_mwpf"])
def test_sinter_model_flattening_takes_stim_models(module):
    import importlib

    flatten = importlib.import_module(f"stabilizer_qec.sinter._decoding.{module}").iter_flatten_model
    model = real_circuit("repetition_code:memory", 3, 10).detector_error_model()
    assert "repeat" in str(model)
    seen = []
    for m in (model, mine(model)):
        errors, coords = [], []
        flatten(m, lambda p, d, o: errors.append((p, list(d), list(o))), lambda det, cs: coords.append((det, list(cs))))
        seen.append((errors, coords))
    assert seen[0] == seen[1] and len(seen[0][0]) == model.flattened().num_errors  # not silently skipped


# Deltakit

def test_deltakit_takes_stim_circuits():
    from stabilizer_qec import deltakit_stim
    from stabilizer_qec.deltakit import circuit as sp
    from stabilizer_qec.deltakit.explorer.types import Measurements

    c = real_circuit()
    assert str(sp.Circuit.from_stim_circuit(c).as_stim_circuit()) == str(sp.Circuit.from_stim_circuit(deltakit_stim.Circuit(str(c))).as_stim_circuit())
    shots = mine(c).compile_sampler(seed=3).sample(4).astype(np.uint8).tolist()
    dets, obs = Measurements(shots).to_detectors_and_observables(c)
    dets2, obs2 = Measurements(shots).to_detectors_and_observables(deltakit_stim.Circuit(str(c)))
    assert np.array_equal(dets.as_numpy(), dets2.as_numpy()) and np.array_equal(obs.as_numpy(), obs2.as_numpy())


def test_ours_leaves_everything_else_alone():
    from stabilizer_qec._util import ours

    c = sq.Circuit("H 0")
    assert ours(c) is c and ours("H 0") == "H 0" and ours(None) is None
    for x in (stim.Circuit("H 0\nM(0.25) 0"), stim.DetectorErrorModel("error(0.125) D0 L0"), stim.DetectorErrorModel("error(0.125) D0 L0")[0]):
        y = ours(x)
        assert type(y).__module__.split(".")[0] == "stabilizer_qec" and str(y) == str(x)
