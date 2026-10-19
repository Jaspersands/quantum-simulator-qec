# CUDA-Q QEC's custom_repetition_code_fine_grain_noise.py (NVIDIA/cudaqx 0.8.1,
# docs/sphinx/examples/qec/python/; Apache License 2.0) registers CUDA-Q kernels for a Python code,
# its stabilizer round applying noise of its own. This package cannot run CUDA-Q kernels, so a
# memory experiment on such a code must say that it ran its stand-ins instead, never stay quiet.
import warnings

import numpy as np
import pytest

from stabilizer_qec import cudaq_qec as qec

calls = []


def x_logical(logical):
    calls.append("x")


def prep0(logical):
    calls.append("prep0")


def prep1(logical):
    calls.append("prep1")


def stabilizer_round(logical):
    calls.append("stabilizer_round")  # (CUDA-Q's applies DepolarizationChannel(0.1) here)


@qec.code("custom_repetition_code_kernels_test")
class MyRepetitionCode:

    def __init__(self, **kwargs):
        qec.Code.__init__(self)
        self.distance = kwargs.get("distance", 3)
        d = self.distance
        self.stabilizers = ["I" * i + "ZZ" + "I" * (d - i - 2) for i in range(d - 1)]
        self.pauli_observables = ["Z" * d]
        self.operation_encodings = {qec.operation.prep0: prep0, qec.operation.prep1: prep1, qec.operation.x: x_logical, qec.operation.stabilizer_round: stabilizer_round}

    def get_num_data_qubits(self):
        return self.distance

    def get_num_ancilla_x_qubits(self):
        return 0

    def get_num_ancilla_z_qubits(self):
        return self.distance - 1

    def get_num_ancilla_qubits(self):
        return self.distance - 1

    def get_num_x_stabilizers(self):
        return 0

    def get_num_z_stabilizers(self):
        return self.distance - 1


@pytest.fixture
def code():
    return qec.get_code("custom_repetition_code_kernels_test", distance=3)


@pytest.fixture
def noise():
    n = qec.NoiseModel()
    n.add_all_qubit_channel("x", qec.Depolarization2(0.01), 1)
    return n


def test_every_memory_experiment_warns(code, noise):
    experiments = [
        lambda: qec.sample_memory_circuit(code, qec.operation.prep0, 10, 3, noise),
        lambda: qec.z_sample_memory_circuit(code, qec.operation.prep0, 10, 3, noise),
        lambda: qec.dem_from_memory_circuit(code, qec.operation.prep0, 3, noise),
        lambda: qec.z_dem_from_memory_circuit(code, qec.operation.prep0, 3, noise),
        lambda: qec.decoder_context_from_memory_circuit(code, qec.operation.prep0, 3, noise),
        lambda: qec.memory_circuit(code, qec.operation.prep1, 2, noise),
    ]
    for run in experiments:
        with pytest.warns(qec.KernelNotRunWarning, match=r"the prep\d and stabilizer_round kernels registered for code 'custom_repetition_code_kernels_test' are not run") as record:
            run()
        assert "apply_noise" in str(record[0].message)
    assert calls == []  # the registered kernels were indeed never called


def test_warning_can_be_made_an_error(code, noise):
    with warnings.catch_warnings():
        warnings.simplefilter("error", qec.KernelNotRunWarning)
        with pytest.raises(qec.KernelNotRunWarning):
            qec.sample_memory_circuit(code, qec.operation.prep0, 10, 3, noise)


def test_built_in_codes_do_not_warn(noise):
    with warnings.catch_warnings():
        warnings.simplefilter("error", qec.KernelNotRunWarning)
        for name in ("repetition", "steane", "surface_code"):
            qec.sample_memory_circuit(qec.get_code(name, distance=3), qec.operation.prep0, 4, 2, noise)


def test_stand_in_round_is_still_the_codes_memory(code, noise):
    # What does run: the code's own stabilizers measured each round, so the syndromes are
    # silent without noise.
    with pytest.warns(qec.KernelNotRunWarning):
        syndromes, data = qec.sample_memory_circuit(code, qec.operation.prep0, 20, 4)
    assert syndromes.shape[0] == 20 and not syndromes.any() and not data.any()
