"""Process matrices (W8): known channels' closed forms, the four forms' conversions, fidelities,
and PECOS's DensityMatrix and ProcessMatrix simulators against PECOS's own SparseSim."""

from __future__ import annotations

import math

import numpy as np
import pytest

import stabilizer_qec as sq

H = np.array([[1, 1], [1, -1]]) / math.sqrt(2)
# CX with qubit 0 the control, little-endian (qubit 0 the least significant bit of an index).
CX01 = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])


def test_unitaries():
    np.testing.assert_allclose(sq.Circuit("H 0").process_matrix(), sq.choi_of_unitary(H), atol=1e-12)
    u = CX01 @ np.kron(np.eye(2), H)
    np.testing.assert_allclose(sq.Circuit("H 0\nCX 0 1").process_matrix(), sq.choi_of_unitary(u), atol=1e-12)
    # The Hadamard's PTM swaps X and Z and negates Y.
    np.testing.assert_allclose(sq.Circuit("H 0").process_matrix("ptm"), [[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0], [0, 1, 0, 0]], atol=1e-12)


def test_channels_closed_forms():
    p = 0.3
    np.testing.assert_allclose(sq.Circuit(f"DEPOLARIZE1({p}) 0").process_matrix("ptm"), np.diag([1, 1 - 4 * p / 3, 1 - 4 * p / 3, 1 - 4 * p / 3]), atol=1e-12)
    np.testing.assert_allclose(sq.Circuit("PAULI_CHANNEL_1(0.1, 0.2, 0.05) 0").process_matrix("chi"), np.diag([0.65, 0.1, 0.2, 0.05]), atol=1e-12)
    g, lam = 0.2, 0.3
    ptm = sq.Circuit(f"I_ERROR[AMPLITUDE_DAMPING(gamma={g})] 0\nI_ERROR[PHASE_DAMPING(lambda={lam})] 0").process_matrix("ptm")
    s = math.sqrt((1 - g) * (1 - lam))
    np.testing.assert_allclose(ptm, [[1, 0, 0, 0], [0, s, 0, 0], [0, 0, s, 0], [g, 0, 0, 1 - g]], atol=1e-12)
    # A measurement dephases; measure-and-correct is a reset.
    np.testing.assert_allclose(sq.Circuit("M 0").process_matrix("ptm"), np.diag([1, 0, 0, 1]), atol=1e-12)
    np.testing.assert_allclose(sq.Circuit("M 0\nCX rec[-1] 0").process_matrix(), sq.Circuit("R 0").process_matrix(), atol=1e-12)
    np.testing.assert_allclose(sq.Circuit("R 0").process_matrix("ptm"), [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]], atol=1e-12)
    # Two qubits: DEPOLARIZE2's χ puts 1 − p on II and p/15 on each other Pauli pair.
    chi = sq.Circuit("DEPOLARIZE2(0.15) 0 1").process_matrix("chi")
    np.testing.assert_allclose(chi, np.diag([0.85] + [0.01] * 15), atol=1e-12)


@pytest.mark.parametrize("seed", range(4))
def test_conversions_round_trip(seed):
    rng = np.random.default_rng(seed)
    gates = ["H 0", "S 1", "CX 0 1", "I_ERROR[AMPLITUDE_DAMPING(gamma=0.2)] 1", "DEPOLARIZE1(0.1) 0", "I[T] 0", "I_ERROR[R_Y(theta=0.4)] 1", "CZ 0 1", "M 0", "MR 1"]
    c = sq.Circuit("\n".join(gates[k] for k in rng.integers(len(gates), size=8)) + "\nI 1")
    choi = c.process_matrix()
    for kind in ("superoperator", "ptm", "chi"):
        m = c.process_matrix(kind)
        np.testing.assert_allclose(sq.convert_process(m, kind, "choi"), choi, atol=1e-12)
        np.testing.assert_allclose(sq.convert_process(choi, "choi", kind), m, atol=1e-12)
    # The superoperator applies the channel to a density matrix.
    psi = rng.normal(size=4) + 1j * rng.normal(size=4)
    psi /= np.linalg.norm(psi)
    rho = np.outer(psi, psi.conj())
    s = sq.DensityMatrixSimulator()
    s.set_density_matrix(rho)
    s.apply_channel(c)
    np.testing.assert_allclose((c.process_matrix("superoperator") @ rho.reshape(-1)).reshape(4, 4), s.density_matrix(), atol=1e-12)
    # A channel's PTM is real; its Choi matrix is positive with trace d; χ is positive with trace 1.
    assert np.isrealobj(c.process_matrix("ptm"))
    assert np.trace(choi).real == pytest.approx(4) and np.linalg.eigvalsh(choi).min() > -1e-12
    assert np.trace(c.process_matrix("chi")).real == pytest.approx(1)


def test_fidelities():
    p = 0.12
    ident = sq.choi_of_unitary(np.eye(2))
    noisy = sq.Circuit(f"DEPOLARIZE1({p}) 0").process_matrix()
    assert sq.process_fidelity(noisy, ident) == pytest.approx(1 - p)
    assert sq.average_gate_fidelity(noisy, ident) == pytest.approx((2 * (1 - p) + 1) / 3)
    a = sq.Circuit("H 0\nI_ERROR[AMPLITUDE_DAMPING(gamma=0.1)] 0").process_matrix("ptm")
    assert sq.process_fidelity(a, a, kind="ptm") == pytest.approx(1)
    # The Choi state's fidelity with a unitary is ⟨⟨U|J|U⟩⟩/d².
    u = sq.choi_of_unitary(H)
    assert sq.process_fidelity(sq.Circuit("H 0\nZ_ERROR(0.25) 0").process_matrix(), u) == pytest.approx(0.75)
    with pytest.raises(ValueError):
        sq.process_matrix("H 6")
    with pytest.raises(ValueError):
        sq.process_matrix("H 0", "kraus")


def test_pauli_basis():
    b = sq.pauli_basis(2)
    assert b.shape == (16, 4, 4)
    # Index 1 is X on qubit 0 (the least significant bit), index 4 X on qubit 1.
    np.testing.assert_allclose(b[1], np.kron(np.eye(2), [[0, 1], [1, 0]]))
    np.testing.assert_allclose(b[4], np.kron([[0, 1], [1, 0]], np.eye(2)))


# PECOS's simulators.

pc = pytest.importorskip("stabilizer_qec.pecos")


def _stabilizers(state):
    """A pySparseSim state's stabilizer generators as (sign, {qubit: letter}): a generator is
    (-1)^minus i^(signs_i) times its qubits' X^x Z^z, and XZ = -iY."""
    out = []
    st = state.stabs
    for g in range(state.num_qubits):
        xs, zs = st.row_x[g], st.row_z[g]
        letters = {q: "Y" if q in xs and q in zs else "X" if q in xs else "Z" for q in xs | zs}
        phase = (-1 if g in st.signs_minus else 1) * (1j if g in st.signs_i else 1) * (-1j) ** sum(v == "Y" for v in letters.values())
        assert abs(phase.imag) < 1e-12
        out.append((int(round(phase.real)), letters))
    return out


def _agree(sparse, dense):
    for sign, letters in _stabilizers(sparse):
        width = dense.num_qubits
        p = sq.PauliString("".join(letters.get(q, "_") for q in range(width)))
        assert dense._sim.peek_observable_expectation(p) == pytest.approx(sign, abs=1e-9), (sign, letters)


ONE = ["I", "X", "Y", "Z", "Q", "Qd", "R", "Rd", "S", "Sd", "H", "H2", "H3", "H4", "H5", "H6", "F1", "F1d", "F2", "F2d", "F3", "F3d", "F4", "F4d", "H+z+x", "H-z-x", "H+y-z", "H-y-z", "H-x+y", "H-x-y"]
TWO = ["CNOT", "CZ", "CY", "SWAP", "G", "G2", "II", "SqrtXX", "MS", "MSXX"]
INIT = ["init |0>", "init |1>", "init |+>", "init |->", "init |+i>", "init |-i>"]


@pytest.mark.parametrize("symbol", ONE + INIT)
def test_pecos_one_qubit_gates(symbol):
    for init in ("init |0>", "init |+>", "init |+i>", "init |-i>"):
        sparse, dense = pc.simulators.pySparseSim(1), pc.simulators.DensityMatrix(1)
        for s in (sparse, dense):
            s.run_gate(init, {0})
            s.run_gate(symbol, {0})
        _agree(sparse, dense)


@pytest.mark.parametrize("symbol", TWO)
def test_pecos_two_qubit_gates(symbol):
    rng = np.random.default_rng(len(symbol))
    for _ in range(12):
        sparse, dense = pc.simulators.pySparseSim(2), pc.simulators.DensityMatrix(2)
        prep = [(ONE[rng.integers(len(ONE))], q) for q in (0, 1)] + [("H", 0), ("CNOT", (0, 1))] * int(rng.integers(2)) + [(ONE[rng.integers(len(ONE))], q) for q in (0, 1)]
        for s in (sparse, dense):
            for g, loc in prep:
                s.run_gate(g, {loc})
            s.run_gate(symbol, {(0, 1)})
        _agree(sparse, dense)


@pytest.mark.parametrize("seed", range(8))
def test_pecos_random_circuits_with_forced_measurements(seed):
    rng = np.random.default_rng(seed)
    n = 4
    sparse, dense = pc.simulators.pySparseSim(n), pc.simulators.DensityMatrix(n)
    for _ in range(30):
        k = rng.integers(10)
        if k < 5:
            sym, loc, params = ONE[rng.integers(len(ONE))], int(rng.integers(n)), {}
        elif k < 8:
            a, b = (int(x) for x in rng.choice(n, 2, replace=False))
            sym, loc, params = TWO[rng.integers(len(TWO))], (a, b), {}
        elif k < 9:
            # PECOS's stabilizer simulators initialize by a forced measurement, the density matrix
            # by a reset: equal on a qubit just measured.
            sym, loc, params = INIT[rng.integers(len(INIT))], int(rng.integers(n)), {}
            assert sparse.run_gate("measure Z", {loc}, forced_outcome=0) == dense.run_gate("measure Z", {loc}, forced_outcome=0)
        else:
            sym, loc, params = ["measure X", "measure Y", "measure Z"][rng.integers(3)], int(rng.integers(n)), {"forced_outcome": int(rng.integers(2))}
        assert sparse.run_gate(sym, {loc}, **params) == dense.run_gate(sym, {loc}, **params), sym
    _agree(sparse, dense)


def test_pecos_init_is_a_reset():
    """The density matrix's initializations trace the qubit out (PECOS's stabilizer simulators
    measure it, forcing the outcome): of a Bell pair, the other qubit is left fully mixed."""
    dense = pc.simulators.DensityMatrix(2)
    dense.run_gate("H", {0})
    dense.run_gate("CNOT", {(0, 1)})
    dense.run_gate("init |+>", {0})
    np.testing.assert_allclose(dense.density_matrix(), np.kron(np.eye(2) / 2, np.full((2, 2), 0.5)), atol=1e-12)


def test_pecos_logical_runs():
    """PECOS's own flow (test_basic_logical_sim) on the density matrix: an ideal logical |0>,
    syndrome extraction with PECOS's errors, decoding and recovery, each output equal to
    SparseSim's on the same errors."""
    surface = pc.qeccs.Surface4444(distance=2)
    logic = pc.circuits.LogicalCircuit(suppress_warning=True)
    logic.append(surface.gate("ideal init |0>"))
    logic2 = pc.circuits.LogicalCircuit(suppress_warning=True)
    logic2.append(surface.gate("I", num_syn_extract=1))
    runner = pc.circuit_runners.Standard(seed=3)
    depolar = pc.error_gens.DepolarGen(model_level="code_capacity")
    decoder = pc.decoders.MWPM2D(surface)
    for trial in range(6):
        sparse = pc.simulators.pySparseSim(surface.num_qudits)
        dense = pc.simulators.DensityMatrix(surface.num_qudits)
        o1, _ = runner.run(sparse, logic)
        p1, _ = runner.run(dense, logic)
        assert dict(o1) == dict(p1)
        o2, errors = runner.run(sparse, logic2, error_gen=depolar, error_params={"p": 0.2})
        p2, _ = runner.run(dense, logic2, error_circuits=errors)
        assert dict(o2) == dict(p2)
        runner.run(sparse, decoder.decode(o2))
        runner.run(dense, decoder.decode(p2))
        _agree(sparse, dense)
        logical_z = surface.instruction("instr_syn_extract").final_logical_ops[0]["Z"]
        assert sparse.logical_sign(logical_z) == dense.logical_sign(logical_z)


def test_pecos_non_clifford_gates():
    t = 0.7
    for sym, u in [("RX", np.array([[math.cos(t / 2), -1j * math.sin(t / 2)], [-1j * math.sin(t / 2), math.cos(t / 2)]])), ("RZ", np.diag([np.exp(-0.5j * t), np.exp(0.5j * t)])), ("T", np.diag([1, np.exp(0.25j * math.pi)])), ("PhaseRot", np.diag([1, np.exp(1j * t)]))]:
        s = pc.simulators.DensityMatrix(1)
        s.run_gate("H", {0})
        s.run_gate(sym, {0}, angle=t)
        psi = u @ H @ np.array([1, 0])
        np.testing.assert_allclose(s.density_matrix(), np.outer(psi, psi.conj()), atol=1e-12)
    # RXX(θ) = exp(−iθ XX/2), and the Toffoli.
    s = pc.simulators.DensityMatrix(2)
    s.run_gate("RXX", {(0, 1)}, angle=t)
    assert s.probabilities()[3] == pytest.approx(math.sin(t / 2) ** 2)
    s = pc.simulators.DensityMatrix(3)
    s.run_gate("X", {0, 1})
    s.run_gate("TOFFOLI", {(0, 1, 2)})
    assert s.probabilities()[7] == pytest.approx(1)
    # Noise by Kraus operators; the logical sign of a mixed state is not ±1.
    s.apply_kraus([2], [np.sqrt(0.5) * np.eye(2), np.sqrt(0.5) * np.array([[1, 0], [0, -1]])])
    s.run_gate("H", {2})
    q = pc.circuits.QuantumCircuit([{"Z": {2}}])
    assert s.logical_sign(q, allow_float=True) == pytest.approx(0)


def test_pecos_process_matrix():
    pm = pc.simulators.ProcessMatrix(2)
    pm.run_gate("H", {0})
    pm.run_gate("CNOT", {(0, 1)})
    np.testing.assert_allclose(pm.choi(), sq.choi_of_unitary(CX01 @ np.kron(np.eye(2), H)), atol=1e-12)
    np.testing.assert_allclose(pm.ptm(), sq.Circuit("H 0\nCX 0 1").process_matrix("ptm"), atol=1e-12)
    pm = pc.simulators.ProcessMatrix(1)
    pm.apply_kraus([0], [math.sqrt(0.9) * np.eye(2), math.sqrt(0.1) * np.array([[0, 1], [1, 0]])])
    np.testing.assert_allclose(pm.chi(), np.diag([0.9, 0.1, 0, 0]), atol=1e-12)
    np.testing.assert_allclose(pm.superoperator(), sq.Circuit("X_ERROR(0.1) 0").process_matrix("superoperator"), atol=1e-12)
    with pytest.raises(ValueError):
        pc.simulators.ProcessMatrix(7)
