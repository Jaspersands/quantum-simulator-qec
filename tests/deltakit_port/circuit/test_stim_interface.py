# (c) Copyright Riverlane 2020-2025.
# modified for stabilizer_qec: its Stim is stabilizer_qec.deltakit_stim and its PyMatching this package's, and deltakit_{circuit,core,decode,explorer} are stabilizer_qec.deltakit.{circuit,core,decode,explorer}.
from stabilizer_qec import deltakit_stim as stim
import pytest


@pytest.mark.parametrize(
    ("stim_circuit1", "stim_circuit2"),
    [
        (
            stim.Circuit("I 0\nM 0\nDETECTOR rec[-1]"),
            stim.Circuit("X_ERROR(0) 0\nM 0\nDETECTOR rec[-1]"),
        ),
        (
            stim.Circuit("I 0\nMZ(1) 0\nDETECTOR rec[-1]"),
            stim.Circuit("X_ERROR(1) 0\nMZ 0\nDETECTOR rec[-1]"),
        ),
        (
            stim.Circuit("X 0\nM 0\nDETECTOR rec[-1]"),
            stim.Circuit("X 0\nM(0) 0\nDETECTOR rec[-1]"),
        ),
        (
            stim.Circuit("X 0\nM !0\nDETECTOR rec[-1]"),
            stim.Circuit("X 0\nM(0) 0\nDETECTOR rec[-1]"),
        ),
        (
            stim.Circuit("X 0\nMZ(1) 0\nDETECTOR rec[-1]"),
            stim.Circuit("X_ERROR(1) 0\nMZ(0) 0\nDETECTOR rec[-1]"),
        ),
    ],
)
def test_stim_detector_error_model_is_equal_when_error_probability_is_equal(
    stim_circuit1, stim_circuit2
):
    assert stim_circuit1.detector_error_model() == stim_circuit2.detector_error_model()
