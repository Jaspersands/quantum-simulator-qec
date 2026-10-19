"""Tsim's circuit surface on this package's circuits: ``to_matrix`` against Tsim's gate matrices
(its ``test_gate_unitaries.py`` and ``test_circuit.py`` cases), ``tcount``, ``is_clifford``,
``inverse`` of tagged gates, and the detector error model with non-deterministic observables
(its ``test_dem.py``, ported)."""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

import stabilizer_qec as sq
from stabilizer_qec import tsim

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from tsim_port.gate_matrices import ROT_GATE_MATRICES, SINGLE_QUBIT_GATE_MATRICES, TWO_QUBIT_GATE_MATRICES  # noqa: E402

X, Y, Z = (SINGLE_QUBIT_GATE_MATRICES[k] for k in "XYZ")


def equal_up_to_phase(u: np.ndarray, v: np.ndarray, atol: float = 1e-10) -> bool:
    product = u @ v.conj().T
    phase = product.flat[np.argmax(np.abs(product))]
    return abs(abs(phase) - 1) < 1e-8 and np.allclose(product, phase * np.eye(u.shape[0]), atol=atol)


# Tsim checks its gates against these matrices entry by entry in modulus (its Bell-state
# trick); a Clifford gate's global phase is the decomposition's here, so up to a phase.
@pytest.mark.parametrize("gate", sorted(SINGLE_QUBIT_GATE_MATRICES))
def test_single_qubit_gates(gate):
    assert equal_up_to_phase(tsim.Circuit(f"{gate} 0").to_matrix(), SINGLE_QUBIT_GATE_MATRICES[gate])
    if gate in ("T", "T_DAG", "S", "S_DAG", "H", "X", "Y", "Z", "I"):
        assert np.allclose(tsim.Circuit(f"{gate} 0").to_matrix(), SINGLE_QUBIT_GATE_MATRICES[gate])


@pytest.mark.parametrize("gate", sorted(TWO_QUBIT_GATE_MATRICES))
def test_two_qubit_gates(gate):
    # Tsim's matrices have qubit 0 as the most significant bit, as to_matrix's default.
    assert equal_up_to_phase(tsim.Circuit(f"{gate} 0 1").to_matrix(), TWO_QUBIT_GATE_MATRICES[gate])
    swapped = TWO_QUBIT_GATE_MATRICES[gate].reshape(2, 2, 2, 2).transpose(1, 0, 3, 2).reshape(4, 4)
    assert equal_up_to_phase(sq.Circuit(f"{gate} 0 1").to_matrix(endian="little"), swapped)


@pytest.mark.parametrize("gate", ["R_X", "R_Y", "R_Z"])
def test_rotations(gate):
    assert np.allclose(tsim.Circuit(f"{gate}(0.345) 0").to_matrix(), ROT_GATE_MATRICES[gate](0.345))


@pytest.mark.parametrize("gate", ["R_XX", "R_YY", "R_ZZ"])
def test_pair_rotations(gate):
    assert np.allclose(tsim.Circuit(f"{gate}(0.345) 0 1").to_matrix(), ROT_GATE_MATRICES[gate](0.345))


def test_r_pauli_and_u3():
    expected = ROT_GATE_MATRICES["R_PAULI"](0.345, np.kron(X, Y))
    assert np.allclose(tsim.Circuit("R_PAULI(0.345) X0*Y1").to_matrix(), expected)
    assert np.allclose(tsim.Circuit("U3(0.345, 0.245, 0.495) 0").to_matrix(), ROT_GATE_MATRICES["U3"](0.345, 0.245, 0.495))


def tpp(p: np.ndarray, dagger: bool = False) -> np.ndarray:
    phase = np.exp(-1j * np.pi / 4) if dagger else np.exp(1j * np.pi / 4)
    return ((1 + phase) * np.eye(p.shape[0]) + (1 - phase) * p) / 2


TPP_CASES = {
    "TPP Z0": SINGLE_QUBIT_GATE_MATRICES["T"],
    "TPP_DAG Z0": SINGLE_QUBIT_GATE_MATRICES["T_DAG"],
    "TPP !Z0": SINGLE_QUBIT_GATE_MATRICES["T_DAG"],
    "TPP_DAG !Z0": SINGLE_QUBIT_GATE_MATRICES["T"],
    "TPP X0": tpp(X),
    "TPP_DAG X0": tpp(X, True),
    "TPP !X0": tpp(X, True),
    "TPP Y0": tpp(Y),
    "TPP_DAG !Y0": tpp(Y),
    "TPP X0*X1": tpp(np.kron(X, X)),
    "TPP_DAG X0*X1": tpp(np.kron(X, X), True),
    "TPP !X0*X1": tpp(np.kron(X, X), True),
    "TPP Y0*Y1": tpp(np.kron(Y, Y)),
    "TPP Z0*Z1": tpp(np.kron(Z, Z)),
    "TPP_DAG !Z0*Z1": tpp(np.kron(Z, Z)),
    "SPP Z0": SINGLE_QUBIT_GATE_MATRICES["S"],
    "SPP !X0": SINGLE_QUBIT_GATE_MATRICES["SQRT_X_DAG"],
    "SPP X0*X1": TWO_QUBIT_GATE_MATRICES["SQRT_XX"],
    "SPP_DAG Z0*Z1": TWO_QUBIT_GATE_MATRICES["SQRT_ZZ_DAG"],
}


@pytest.mark.parametrize("instruction", sorted(TPP_CASES))
def test_tpp_and_spp(instruction):
    assert equal_up_to_phase(tsim.Circuit(instruction).to_matrix(), TPP_CASES[instruction])


def test_ccz_and_ccx_matrices():
    ccz = np.eye(8, dtype=complex)
    ccz[-1, -1] = -1
    ccx = np.eye(8, dtype=complex)
    ccx[6, 6] = ccx[7, 7] = 0
    ccx[6, 7] = ccx[7, 6] = 1
    for text, m in (("CCZ 0 1 2", ccz), ("CCX 0 1 2", ccx), ("CCZ[tag] 0 1 2", ccz), ("CCX[tag] 0 1 2", ccx)):
        c = tsim.Circuit(text)
        assert equal_up_to_phase(c.to_matrix(), m), text
        assert c.tcount() == 7
        assert not c.is_clifford
    # Through the sampler: a Toffoli on |110⟩ reads |111⟩, and on |101⟩ stays.
    c = tsim.Circuit("R 0 1 2\nX 0 1\nCCX 0 1 2\nM 0 1 2")
    assert np.array_equal(c.compile_sampler(seed=1).sample(50), np.ones((50, 3), dtype=bool))
    c = tsim.Circuit("R 0 1 2\nX 0 2\nCCX 0 1 2\nM 0 1 2")
    assert np.array_equal(c.compile_sampler(seed=1).sample(50), np.tile([True, False, True], (50, 1)))


def test_to_matrix_rules():
    assert sq.Circuit("").to_matrix().shape == (1, 1)
    assert sq.Circuit("TICK\nQUBIT_COORDS(1, 2) 1\nH 0").to_matrix().shape == (4, 4)
    t = sq.Circuit("I[T] 0").to_tensor()
    assert t.shape == (2, 2) and np.allclose(t, SINGLE_QUBIT_GATE_MATRICES["T"])
    assert sq.Circuit("CX 0 1\nI[T] 1").to_tensor().shape == (2, 2, 2, 2)
    for text in ("M 0", "R 0", "X_ERROR(0.1) 0", "MPP X0*X1"):
        with pytest.raises(ValueError, match="no matrix"):
            sq.Circuit(text).to_matrix()
    with pytest.raises(ValueError, match="at most 10"):
        sq.Circuit("H 11").to_matrix()
    with pytest.raises(ValueError, match="endian"):
        sq.Circuit("H 0").to_matrix(endian="middle")
    # Stim's tableau agrees on Clifford circuits, in either endianness.
    c = sq.Circuit("H 0\nCX 0 1\nS 1\nSQRT_Y 2\nCZ 2 0")
    for endian in ("big", "little"):
        assert equal_up_to_phase(c.to_matrix(endian=endian), c.to_tableau().to_unitary_matrix(endian=endian).astype(complex))


def test_tcount_and_is_clifford():
    assert tsim.Circuit("H 0\nCNOT 0 1").tcount() == 0
    assert tsim.Circuit("H 0\nT 0\nT 1\nT 0").tcount() == 3
    # Every non-Clifford phase counts, whatever its angle (R_X(0.3) too; of U3's three, φ = 0.5 is Clifford).
    assert tsim.Circuit("T_DAG 0\nTPP X0*Z1\nR_Z(0.25) 0\nR_Z(0.75) 0\nR_Z(-1.25) 0\nR_X(0.3) 0\nU3(0.25, 0.5, 0.75) 0").tcount() == 8
    assert tsim.Circuit("R_Z(0.5) 0\nR_XX(0.5) 0 1\nU3(0.5, 0.5, 1.0) 0").tcount() == 0
    assert sq.Circuit("I_ERROR[R_Z(theta=0.39269908169872414)] 0").tcount() == 1
    assert tsim.Circuit("H 0\nCNOT 0 1\nM 0 1\nDETECTOR rec[-1]").is_clifford
    assert tsim.Circuit("R_Z(0.5) 0\nR_X(-1.5) 0\nU3(0.5, -1.0, 1.5) 0").is_clifford
    for text in ("T 0", "TPP Z0", "TPP_DAG X0*Y1", "H 0\nR_Z(0.25) 0\nCNOT 0 1", "U3(0.5, 0.25, 1.0) 0"):
        assert not tsim.Circuit(text).is_clifford, text
    assert sq.Circuit("I[mytag] 0\nH 0\nI[abc123] 0\nCNOT 0 1\nM 0 1").is_clifford
    assert sq.Circuit("DEPOLARIZE1(0.1) 0\nI_ERROR[AMPLITUDE_DAMPING(gamma=0.1)] 0").is_clifford
    assert not sq.Circuit("I_ERROR[R_Z(theta=0.1)] 0").is_clifford
    assert sq.Circuit("REPEAT 3 {\n    I[T] 0\n}").tcount() == 3


@pytest.mark.parametrize("text", ["R_Z(0.25) 0", "H 0", "R_X(0.3) 0", "R_Y(-0.5) 0", "U3(0.3, 0.24, 0.49) 0", "T 0", "T_DAG 0", "TPP Z0", "TPP_DAG X0*Y1", "R_XX(0.345) 0 1", "R_PAULI(0.2) X0*Z1*Y2", "H 0\nS 0\nCCZ 0 1 2\nT 1\nCX 1 0"])
def test_inverse_undoes_tagged_gates(text):
    c = tsim.Circuit(text)
    inv = c.inverse()
    m = (c + inv).to_matrix()
    assert equal_up_to_phase(m, np.eye(m.shape[0])), text
    if "\n" not in text:
        assert len(inv) == len(c)
        assert inv.inverse() == c


def test_inverse_rejects_non_unitary_tags():
    with pytest.raises(ValueError, match="not unitary"):
        sq.Circuit("I[LEAK(p=0.1)] 0").inverse()
    assert str(sq.Circuit("I[mine] 0\nH 0").inverse()) == "H 0\nI[mine] 0"
    assert str(sq.Circuit("I[U3(theta=1.0, phi=2.0, lambda=3.0)] 0").inverse()) == "I[U3(theta=-1.0, phi=-3.0, lambda=-2.0)] 0"


# Tsim's test_dem.py.
def test_dem_with_non_deterministic_observable():
    with_nondet = sq.Circuit("""
        RX 6
        S 6
        H 6
        R 0 1 2 3 4 5
        SQRT_Y_DAG 0 1 2 3 4 5
        CZ 1 2 3 4 5 6
        SQRT_Y 6
        CZ 0 3 2 5 4 6
        DEPOLARIZE2(0.01) 0 3 2 5 4 6
        SQRT_Y 2 3 4 5 6
        DEPOLARIZE1(0.01) 0 1 2 3 4 5 6
        CZ 0 1 2 3 4 5
        DEPOLARIZE2(0.01) 0 1 2 3 4 5
        SQRT_Y 1 2 4
        M 0 1 2
        OBSERVABLE_INCLUDE(0) rec[-3]
        M 3 4 5 6
        DETECTOR rec[-7] rec[-6] rec[-5] rec[-4]
        DETECTOR rec[-6] rec[-5] rec[-3] rec[-2]
        DETECTOR rec[-5] rec[-4] rec[-3] rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-6] rec[-2]
    """)
    deterministic = sq.Circuit("\n".join(str(with_nondet).splitlines()[3:]))
    dem = with_nondet.detector_error_model(allow_non_deterministic_observables=True)
    assert dem.approx_equals(deterministic.detector_error_model(), atol=1e-12)
    assert tsim.get_detector_error_model(with_nondet).approx_equals(dem, atol=1e-12)
    with pytest.raises(ValueError):
        with_nondet.detector_error_model()
    with pytest.raises(ValueError, match="not supported"):
        with_nondet.detector_error_model(allow_non_deterministic_observables=True, decompose_errors=True)


def test_dem_gauge_detectors_kept_and_observable_gauges_dropped():
    c = sq.Circuit("R 0 1\nH 0\nCNOT 0 1\nH 1\nX_ERROR(0.01) 0\nM 0 1\nOBSERVABLE_INCLUDE(0) rec[-1] rec[-2]\nDETECTOR rec[-2]")
    assert c.detector_error_model(allow_non_deterministic_observables=True).approx_equals(sq.DetectorErrorModel("error(0.5) D0\n error(0.01) D0 L0"), atol=1e-12)


def test_dem_no_errors():
    f = lambda text: str(sq.Circuit(text).detector_error_model(allow_non_deterministic_observables=True))  # noqa: E731
    assert f("R 0 1\nM 0 1\nOBSERVABLE_INCLUDE(0) rec[-1]") == "logical_observable L0"
    assert f("R 0 1\nM 0 1\nDETECTOR rec[-1]") == "detector D0"
    assert f("R 0 1\nM 0 1\nOBSERVABLE_INCLUDE(0) rec[-1]\nOBSERVABLE_INCLUDE(1) rec[-1]") == "logical_observable L0\nlogical_observable L1"


def test_dem_observable_eaten_as_gauge_is_an_error():
    with pytest.raises(ValueError, match=r"The number of observables changed after conversion\."):
        sq.Circuit("R 0\nH 0\nX_ERROR(0.01) 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]").detector_error_model(allow_non_deterministic_observables=True)
    with pytest.raises(ValueError, match=r"The number of observables changed after conversion\."):
        sq.Circuit("R 0 1 2 3\nH 0 1 2 3\nZ_ERROR(0.01) 0\nMPP Z0*Z1 Z2*Z3\nM 0 1\nOBSERVABLE_INCLUDE(0) rec[-3]\nDETECTOR rec[-4] rec[-2] rec[-1]").detector_error_model(allow_non_deterministic_observables=True)


def test_dem_counts_records_of_every_measuring_instruction():
    cases = [
        "R 0 1 2\nH 0 1 2\nX_ERROR(0.01) 0\nMPP X0*X1*X2\nOBSERVABLE_INCLUDE(0) rec[-1]\nM 0\nDETECTOR rec[-1] rec[-2]",
        "R 0 1 2 3 4\nH 0 1 2 3 4\nX_ERROR(0.01) 0\nMPP X0*X1 Z2*Z3 Y4\nOBSERVABLE_INCLUDE(0) rec[-3]\nM 0\nDETECTOR rec[-1] rec[-4]",
        "R 0 1 2 3\nH 0 1 2 3\nX_ERROR(0.01) 0\nMPP X0*X1\nOBSERVABLE_INCLUDE(0) rec[-1]\nMPP X2*X3\nDETECTOR rec[-1] rec[-2]",
    ]
    for text in cases:
        dem = sq.Circuit(text).detector_error_model(allow_non_deterministic_observables=True)
        assert dem.num_detectors == 1 and dem.num_observables == 1
        assert "D0" in str(dem) and "L0" in str(dem)
    for pair in ("MXX", "MYY", "MZZ"):
        c = sq.Circuit(f"R 0 1\nX_ERROR(0.1) 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]\n{pair} 0 1\nDETECTOR rec[-1]")
        assert c.detector_error_model(allow_non_deterministic_observables=True).approx_equals(c.detector_error_model(allow_gauge_detectors=True), atol=1e-12)
    for text, kw in (
        ("R 0 1 2 3\nX_ERROR(0.1) 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]\nMXX 0 1 2 3\nDETECTOR rec[-1]\nDETECTOR rec[-2]", {}),
        ("R 0\nX_ERROR(0.1) 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]\nMPAD 0\nDETECTOR rec[-1]", {}),
        ("R 0 1\nX_ERROR(0.1) 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]\nMRZ 1\nDETECTOR rec[-1]", {}),
        ("R 0 1 2\nX_ERROR(0.1) 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]\nHERALDED_ERASE(0.05) 1 2\nDETECTOR rec[-1]\nDETECTOR rec[-2]", dict(approximate_disjoint_errors=True)),
        ("R 0 1 2\nX_ERROR(0.1) 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]\nHERALDED_PAULI_CHANNEL_1(0.01, 0.02, 0.03, 0.04) 1 2\nDETECTOR rec[-1]\nDETECTOR rec[-2]", dict(approximate_disjoint_errors=True)),
    ):
        c = sq.Circuit(text)
        assert c.detector_error_model(allow_non_deterministic_observables=True, **kw).approx_equals(c.detector_error_model(allow_gauge_detectors=True, **kw), atol=1e-12)


def test_tsim_circuit_reads_both_forms():
    a = tsim.Circuit("R 0\nT 0\nR_X(0.25) 0\nM 0")
    b = tsim.Circuit("R 0\nS[T] 0\nI[R_X(theta=0.25*pi)] 0\nM 0")
    assert a == b and a.tcount() == 2
    assert str(sq.Circuit(a)) == "R 0\nI[T] 0\nI[R_X(theta=0.39269908169872414)] 0\nM 0"
    assert str(a) == str(b) == "R 0\nT 0\nR_X(0.25) 0\nM 0"


def test_tsim_conveniences():
    # Tsim's from_stim_program, cast_to_stim, stim_circuit and without_annotations.
    c = tsim.Circuit.from_stim_program(("H 0\nS[T] 0\nI[R_Z(theta=0.5*pi)] 1\nM 0 1\nDETECTOR rec[-1]\nREPEAT 2 {\n    M 0\n    DETECTOR rec[-1]\n}\nOBSERVABLE_INCLUDE(0) rec[-2]"))
    assert c.tcount() == 1 and str(sq.Circuit(c)).count("I[T]") == 1 and str(c).splitlines()[1] == "T 0"
    assert c.cast_to_stim() is c
    expanded = c.stim_circuit
    assert "R_Z" not in str(expanded) and "I[T]" in str(expanded) and equal_up_to_phase(expanded[:3].to_matrix(), c[:3].to_matrix())
    bare = c.without_annotations()
    assert bare.num_detectors == 0 and bare.num_observables == 0 and bare.num_measurements == c.num_measurements
    assert "REPEAT 2" in str(bare) and str(c.without_annotations().without_annotations()) == str(bare)
    stim = pytest.importorskip("stim")
    t = pytest.importorskip("tsim")
    ref = t.Circuit("H 0\nT 0\nR_Z(0.5) 1\nM 0 1\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-2]")
    ours = tsim.Circuit.from_stim_program(ref._stim_circ)
    assert ours.num_detectors == ref.num_detectors and ours.tcount() == ref.tcount()
    assert str(ours.without_annotations().without_noise()) == str(tsim.Circuit(str(ref.without_annotations()._stim_circ)))
    assert isinstance(stim.Circuit(str(ours.stim_circuit)), stim.Circuit)
