from __future__ import annotations

import pytest

import stabilizer_qec as sq
from conftest import assert_same_model

BELL = "R 0 1\nH 0\nCX 0 1\nM 0 1\nDETECTOR rec[-1] rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-1]\n"


def test_counts_and_text():
    c = sq.Circuit(BELL)
    assert (c.num_qubits, c.num_measurements, c.num_detectors, c.num_observables, c.num_sweep_bits) == (2, 2, 1, 1, 0)
    assert sq.Circuit(str(c)) == c
    assert c != sq.Circuit("M 0")
    assert c.__eq__("not a circuit") is NotImplemented
    assert "Circuit(" in repr(c)


def test_counts_go_through_loops_without_unrolling():
    c = sq.Circuit("REPEAT 1000000000 {\n  M 0\n  DETECTOR rec[-1]\n}\nCX sweep[5] 0")
    assert c.num_measurements == 10**9
    assert c.num_detectors == 10**9
    assert c.num_sweep_bits == 6


def test_from_file(tmp_path):
    path = tmp_path / "bell.stim"
    path.write_text(BELL)
    assert sq.Circuit.from_file(path) == sq.Circuit(BELL)
    (tmp_path / "bell.dem").write_text("error(0.1) D0 L0\n")
    assert sq.DetectorErrorModel.from_file(tmp_path / "bell.dem").num_errors == 1


def test_bad_circuits():
    with pytest.raises(ValueError, match="unsupported instruction"):
        sq.Circuit("FOO 0")
    # A circuit reading before its first measurement can be held (a piece of a larger one), but
    # not sampled, analysed or converted.
    early = sq.Circuit("M 0\nDETECTOR rec[-2]")
    assert early.num_detectors == 1
    for use in (early.detector_error_model, early.compile_detector_sampler, early.compile_m2d_converter):
        with pytest.raises(ValueError, match="before the first measurement"):
            use()
    with pytest.raises(TypeError):
        sq.Circuit(3)


def test_stim_objects_in_and_out():
    stim = pytest.importorskip("stim")
    c = sq.Circuit(stim.Circuit(BELL))
    assert stim.Circuit(str(c)) == stim.Circuit(BELL)
    dem = sq.DetectorErrorModel(stim.DetectorErrorModel("error(0.125) D0 D1 L0\ndetector D2"))
    assert (dem.num_detectors, dem.num_observables, dem.num_errors) == (3, 1, 1)


@pytest.mark.parametrize("code", ["rotated", "xzzx"])
@pytest.mark.parametrize("noise", ["sd6", "current"])
@pytest.mark.parametrize("decompose", [False, True])
def test_error_models_equal_stims(code, noise, decompose):
    stim = pytest.importorskip("stim")
    c = sq.memory_circuit(code, distance=3, rounds=3, p=0.003, noise=noise)
    ours = str(c.detector_error_model(decompose_errors=decompose))
    theirs = str(stim.Circuit(str(c)).detector_error_model(decompose_errors=decompose).flattened())
    assert_same_model(ours, theirs)


def test_dem_text_round_trip(d3):
    dem = d3.detector_error_model(decompose_errors=True)
    again = sq.DetectorErrorModel(str(dem))
    assert again == dem
    assert (again.num_detectors, again.num_observables, again.num_errors) == (dem.num_detectors, 1, dem.num_errors)
