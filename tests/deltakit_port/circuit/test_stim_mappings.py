# (c) Copyright Riverlane 2020-2025.
# modified for stabilizer_qec: its Stim is stabilizer_qec.deltakit_stim and its PyMatching this package's, and deltakit_{circuit,core,decode,explorer} are stabilizer_qec.deltakit.{circuit,core,decode,explorer}.
from stabilizer_qec.deltakit.circuit.gates import GATE_MAPPING
from stabilizer_qec.deltakit.circuit.noise_channels import NOISE_CHANNEL_MAPPING

ALL_STIM_GATE_STRINGS = (
    "I",
    "X",
    "Y",
    "Z",
    "C_XYZ",
    "C_ZYX",
    "H",
    "H_XY",
    "H_XZ",
    "H_YZ",
    "S",
    "SQRT_X",
    "SQRT_X_DAG",
    "SQRT_Y",
    "SQRT_Y_DAG",
    "SQRT_Z",
    "SQRT_Z_DAG",
    "S_DAG",
    "CNOT",
    "CX",
    "CXSWAP",
    "CY",
    "CZ",
    "CZSWAP",
    "ISWAP",
    "ISWAP_DAG",
    "SQRT_XX",
    "SQRT_XX_DAG",
    "SQRT_YY",
    "SQRT_YY_DAG",
    "SQRT_ZZ",
    "SQRT_ZZ_DAG",
    "SWAP",
    "XCX",
    "XCY",
    "XCZ",
    "YCX",
    "YCY",
    "YCZ",
    "ZCX",
    "ZCY",
    "ZCZ",
    "M",
    "MPP",
    "MR",
    "MRX",
    "MRY",
    "MRZ",
    "MX",
    "MY",
    "MZ",
    "R",
    "RX",
    "RY",
    "RZ",
)

ALL_STIM_NOISE_CHANNEL_STRINGS = (
    "CORRELATED_ERROR",
    "DEPOLARIZE1",
    "DEPOLARIZE2",
    "E",
    "ELSE_CORRELATED_ERROR",
    "PAULI_CHANNEL_1",
    "PAULI_CHANNEL_2",
    "X_ERROR",
    "Y_ERROR",
    "Z_ERROR",
)


def test_every_valid_stim_gate_can_be_mapped_to_a_deltakit_circuit_gate():
    assert all(stim_string in GATE_MAPPING for stim_string in ALL_STIM_GATE_STRINGS)


def test_every_valid_stim_noise_channel_can_be_mapped_to_a_deltakit_circuit_noise_channel():
    assert all(
        stim_string in NOISE_CHANNEL_MAPPING
        for stim_string in ALL_STIM_NOISE_CHANNEL_STRINGS
    )
