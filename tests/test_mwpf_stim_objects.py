"""stabilizer_qec.mwpf given Stim's own objects, as sinter and MWPF's users give them: a
stim.DetectorErrorModel decodes as this package's model does (its observables were once unseen,
every prediction 0), and the reference circuits, heralded models and herald helpers take Stim's
circuits and targets, the helpers giving Stim's circuits back. (test_mwpf_package.py compares
the same against MWPF itself.)"""

from __future__ import annotations

import numpy as np
import pytest

stim = pytest.importorskip("stim")

import stabilizer_qec as sq  # noqa: E402
import stabilizer_qec.mwpf as mw  # noqa: E402

if not hasattr(mw, "SinterMWPFDecoder"):
    pytest.skip("MWPF's sinter helpers need Python 3.10", allow_module_level=True)

# A circuit of MWPF's heralded-model tests (tests/python/test_heralded_dem.py).
HERALDED = """\
R 0 1 2 3 4
TICK
CX 0 1 2 3
TICK
CX 2 1 4 3
TICK
MR 1 3
DETECTOR(1, 0) rec[-2]
DETECTOR(3, 0) rec[-1]
DEPOLARIZE1(0.02) 2
HERALDED_ERASE(0.01) 0
DETECTOR rec[-1]
M 0 2 4
DETECTOR(1, 1) rec[-2] rec[-3] rec[-6]
DETECTOR(3, 1) rec[-1] rec[-2] rec[-5]
OBSERVABLE_INCLUDE(0) rec[-1]
"""


def decode(decoder, dem, dets):
    return decoder.compile_decoder_for_dem(dem=dem).decode_shots_bit_packed(bit_packed_detection_event_data=dets)


def test_sinter_decoder_takes_stims_model():
    # MWPF's own test (tests/python/test_sinter.py): both detectors lit means L0 flipped.
    dem = stim.DetectorErrorModel("error(0.02) D0\nerror(0.02) D0 D1 L0\nerror(0.1) D1\nerror(0.01) D1 L0\n")
    lit = np.packbits(np.array([[1, 1]]), axis=-1, bitorder="little")
    compiled = mw.SinterMWPFDecoder(panic_action=mw.PanicAction.RAISE).compile_decoder_for_dem(dem=dem)
    assert compiled.predictor.get_observable_bits(compiled.decode_shots_bit_packed(bit_packed_detection_event_data=lit)) == [1]
    circuit = stim.Circuit.generated("surface_code:rotated_memory_x", distance=3, rounds=3, after_clifford_depolarization=0.01, before_measure_flip_probability=0.01)
    dem = circuit.detector_error_model(decompose_errors=False)
    dets, obs = circuit.compile_detector_sampler(seed=3).sample(300, separate_observables=True, bit_packed=True)
    for decoder in (mw.SinterMWPFDecoder(), mw.SinterHUFDecoder(), mw.SinterMWPFDecoder(bp=True)):
        theirs, ours = decode(decoder, dem, dets), decode(decoder, sq.DetectorErrorModel(str(dem)), dets)
        assert np.array_equal(theirs, ours)
        assert (theirs != obs).any(axis=1).sum() < obs.any(axis=1).sum() / 2
    model = mw.ref_circuit.RefDetectorErrorModel.of(dem=dem)
    assert model.hyperedges == mw.ref_circuit.RefDetectorErrorModel.of(dem=sq.DetectorErrorModel(str(dem))).hyperedges
    assert any(e.observables for e in model.hyperedges)


def test_heralded_helpers_take_stims_circuits():
    theirs, ours = stim.Circuit(HERALDED), sq.Circuit(HERALDED)
    dem = theirs.detector_error_model(approximate_disjoint_errors=True)
    dets = theirs.compile_detector_sampler(seed=4).sample(200, bit_packed=True)
    a = decode(mw.SinterMWPFDecoder(pass_circuit=True).with_circuit(theirs), dem, dets)
    b = decode(mw.SinterMWPFDecoder(pass_circuit=True).with_circuit(ours), sq.DetectorErrorModel(str(dem)), dets)
    assert np.array_equal(a, b)
    assert str(mw.heralded_dem.HeraldedDetectorErrorModel.of(theirs)) == str(mw.heralded_dem.HeraldedDetectorErrorModel.of(ours))
    assert str(mw.ref_circuit.RefCircuit.of(theirs)) == str(mw.ref_circuit.RefCircuit.of(ours))
    for helper in ("add_herald_detectors", "remove_herald_detectors"):
        x, y = getattr(mw.heralded_dem, helper)(theirs), getattr(mw.heralded_dem, helper)(ours)
        assert isinstance(x, stim.Circuit) and isinstance(y, sq.Circuit) and x == stim.Circuit(str(y)), helper
    a = mw.ref_circuit.RefInstruction.new_heralded_erase((stim.GateTarget(0), stim.GateTarget(3)), 0.1)
    b = mw.ref_circuit.RefInstruction.new_heralded_erase((sq.GateTarget(0), sq.GateTarget(3)), 0.1)
    assert str(mw.ref_circuit.RefCircuit.of([a])) == str(mw.ref_circuit.RefCircuit.of([b]))
