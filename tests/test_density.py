"""The density-matrix simulator (W8, X1): against Quantumsim's (its gates, amplitude and phase
damping, depolarizing and bit-flip idling), against the state vector's exact distribution and
its trajectories, against TableauSimulator on Clifford circuits, and the damping channels'
closed forms."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import stabilizer_qec as sq


def _quantumsim():
    pytest.importorskip("pytools")
    qs = pytest.importorskip("quantumsim")
    if getattr(qs, "__version__", "0.2.0") not in ("0.2.0",):
        pytest.skip("the test reproduces Quantumsim 0.2.0")
    import quantumsim.circuit as qc
    import quantumsim.sparsedm as sd

    return qc, sd


def _random_program(rng, n):
    """Random operations, each as (Quantumsim gate maker, this package's line)."""
    names = "abcdefgh"[:n]
    ops = []
    for _ in range(14):
        kind = rng.integers(10)
        q = int(rng.integers(n))
        r = int((q + 1 + rng.integers(n - 1)) % n)
        a, b = names[q], names[r]
        angle = float(rng.uniform(-math.pi, math.pi))
        t1 = float(rng.uniform(10, 50))
        t2 = float(rng.uniform(1, 2 * t1))
        dur = float(rng.uniform(0.5, 8))
        if kind == 0:
            ops.append((lambda t, a=a: qc_mod.Hadamard(a, time=t), f"H {q}"))
        elif kind in (1, 2, 3):
            cls, axis = [(qc_mod.RotateX, "X"), (qc_mod.RotateY, "Y"), (qc_mod.RotateZ, "Z")][kind - 1]
            ops.append((lambda t, a=a, cls=cls, angle=angle: cls(a, angle=angle, time=t), f"I[R_{axis}(theta={angle / 2!r})] {q}"))
        elif kind == 4:
            ops.append((lambda t, a=a, b=b: qc_mod.CPhase(a, b, time=t), f"CZ {q} {r}"))
        elif kind == 5:
            # Quantumsim's CNOT takes its control second.
            ops.append((lambda t, a=a, b=b: qc_mod.CNOT(a, b, time=t), f"CX {r} {q}"))
        elif kind == 6:
            ops.append((lambda t, a=a, dur=dur, t1=t1, t2=t2: qc_mod.AmpPhDamp(a, time=t, duration=dur, t1=t1, t2=t2), f"I_ERROR[THERMAL_RELAXATION(t={dur!r}, t1={t1!r}, t2={t2!r})] {q}"))
        elif kind == 7:
            g = 1 - math.exp(-dur / t1)
            ops.append((lambda t, a=a, dur=dur, t1=t1: qc_mod.DepolarizingNoise(a, time=t, duration=dur, t1=t1), f"DEPOLARIZE1({3 * g / 4!r}) {q}"))
        elif kind == 8:
            g = 1 - math.exp(-dur / t1)
            ops.append((lambda t, a=a, dur=dur, t1=t1: qc_mod.BitflipNoise(a, time=t, duration=dur, t1=t1), f"X_ERROR({g / 2!r}) {q}"))
        else:
            ops.append((lambda t, a=a, b=b: qc_mod.Swap(a, b, time=t), f"SWAP {q} {r}"))
    return names, ops


qc_mod = None


@pytest.mark.parametrize("seed", range(12))
def test_density_matrix_equals_quantumsim(seed):
    global qc_mod
    qc_mod, sd = _quantumsim()
    rng = np.random.default_rng(seed)
    n = 2 + seed % 3
    names, ops = _random_program(rng, n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = qc_mod.Circuit("random")
        for name in names:
            c.add_qubit(name)
        for t, (make, _) in enumerate(ops):
            c.add_gate(make(float(t)))
        c.order()
        dm = sd.SparseDM(list(names))
        for name in names:  # every qubit in the dense part, in order: qubit 0 the least significant
            dm.ensure_dense(name)
        c.apply_to(dm)
        theirs = dm.full_dm.to_array()
    text = "\n".join(line for _, line in ops) + f"\nI {n - 1}"
    ours = sq.Circuit(text).density_matrix()
    np.testing.assert_allclose(ours, theirs, atol=1e-10)


@pytest.mark.parametrize("seed", range(6))
def test_exact_distributions_agree(seed):
    rng = np.random.default_rng(100 + seed)
    gates = ["H", "S", "SQRT_X", "CX", "CZ", "I[T]", "I_ERROR[R_Y(theta=0.3)]", "DEPOLARIZE1(0.05)", "DEPOLARIZE2(0.04)", "I_ERROR[AMPLITUDE_DAMPING(gamma=0.2)]", "I_ERROR[PHASE_DAMPING(lambda=0.3)]", "I_ERROR[THERMAL_RELAXATION(t=3, t1=20, t2=15)]", "M(0.02)", "MX", "MR"]
    lines = ["R 0 1 2 3"]
    for _ in range(12):
        g = gates[rng.integers(len(gates))]
        if g in ("CX", "CZ", "DEPOLARIZE2(0.04)"):
            a, b = rng.choice(4, 2, replace=False)
            lines.append(f"{g} {a} {b}")
        else:
            lines.append(f"{g} {rng.integers(4)}")
    lines.append("M 0 1 2 3")
    m = sum(1 for line in lines if line.split("(")[0].split(" ")[0] in ("M", "MX", "MR")) + 3
    lines += [f"DETECTOR rec[-{k}]" for k in range(1, m + 1)]
    lines.append("OBSERVABLE_INCLUDE(0) rec[-1] rec[-2]")
    c = sq.Circuit("\n".join(lines))
    a = c.exact_distribution(method="density_matrix")
    b = c.exact_distribution(max_branches=1 << 24)
    assert set(a) == set(b)
    for k in a:
        assert a[k] == pytest.approx(b[k], abs=1e-12)


def test_trajectories_average_to_the_density_matrix():
    c = sq.Circuit(
        """
        RX 0
        R 1
        I_ERROR[THERMAL_RELAXATION(t=10, t1=20, t2=12)] 0
        CX 0 1
        I_ERROR[AMPLITUDE_DAMPING(gamma=0.3)] 1
        I_ERROR[PHASE_DAMPING(lambda=0.2)] 0
        H 0
        M 0 1
        DETECTOR rec[-1]
        DETECTOR rec[-2]
        """
    )
    exact = c.exact_distribution(method="density_matrix")
    shots = 200_000
    dets, _ = c.compile_exact_sampler(seed=5).sample(shots, separate_observables=True)
    for (d, _), p in exact.items():
        hits = np.mean(np.all(dets == np.array(d), axis=1))
        assert abs(hits - p) < 5 * math.sqrt(p * (1 - p) / shots) + 1e-9


@pytest.mark.parametrize("seed", range(5))
def test_clifford_circuits_equal_the_tableau_simulator(seed):
    rng = np.random.default_rng(seed)
    lines = []
    for _ in range(25):
        if rng.integers(2):
            lines.append(f"{['H', 'S', 'SQRT_Y', 'S_DAG', 'H_YZ', 'X', 'Y'][rng.integers(7)]} {rng.integers(4)}")
        else:
            a, b = rng.choice(4, 2, replace=False)
            lines.append(f"{['CX', 'CZ', 'CY', 'ISWAP', 'SQRT_XX', 'SWAP'][rng.integers(6)]} {a} {b}")
    circuit = sq.Circuit("\n".join(lines) + "\nI 3")
    sim = sq.TableauSimulator()
    sim.do(circuit)
    psi = sim.state_vector().astype(np.complex128)
    np.testing.assert_allclose(circuit.density_matrix(), np.outer(psi, psi.conj()), atol=1e-6)


def test_thermal_relaxation_closed_forms():
    t, t1, t2 = 7.0, 25.0, 18.0
    # T1: |1⟩ decays to |0⟩ as e^{−t/T1}.
    rho = sq.Circuit(f"X 0\nI_ERROR[THERMAL_RELAXATION(t={t}, t1={t1}, t2={t2})] 0").density_matrix()
    assert rho[1, 1].real == pytest.approx(math.exp(-t / t1), rel=1e-12)
    # T2: |+⟩'s coherence decays as e^{−t/T2}.
    rho = sq.Circuit(f"H 0\nI_ERROR[THERMAL_RELAXATION(t={t}, t1={t1}, t2={t2})] 0").density_matrix()
    assert 2 * abs(rho[0, 1]) == pytest.approx(math.exp(-t / t2), rel=1e-12)
    # T2 = 2 T1: no pure dephasing.
    rho = sq.Circuit("H 0\nI_ERROR[THERMAL_RELAXATION(t=3, t1=5, t2=10)] 0").density_matrix()
    assert 2 * abs(rho[0, 1]) == pytest.approx(math.exp(-3 / 10), rel=1e-12)
    rho = sq.Circuit("H 0\nI_ERROR[PHASE_DAMPING(lambda=0.36)] 0").density_matrix()
    assert 2 * abs(rho[0, 1]) == pytest.approx(0.8, rel=1e-12) and rho[1, 1].real == pytest.approx(0.5)
    for bad in ("THERMAL_RELAXATION(t=1, t1=2, t2=5)", "THERMAL_RELAXATION(t=1, t1=0, t2=1)", "PHASE_DAMPING(lambda=2)", "THERMAL_RELAXATION(t=1, t1=2)"):
        with pytest.raises(ValueError):
            sq.Circuit(f"I_ERROR[{bad}] 0").density_matrix()


def test_density_matrix_measurements_and_feedback():
    # Measure-and-correct leaves |0⟩; measurement alone leaves the qubit mixed.
    np.testing.assert_allclose(sq.Circuit("H 0\nM 0\nCX rec[-1] 0").density_matrix(), [[1, 0], [0, 0]], atol=1e-12)
    np.testing.assert_allclose(sq.Circuit("H 0\nM 0").density_matrix(), np.eye(2) / 2, atol=1e-12)
    np.testing.assert_allclose(sq.Circuit("H 0\nMR 0").density_matrix(), [[1, 0], [0, 0]], atol=1e-12)
    np.testing.assert_allclose(sq.Circuit("H 0\nMRX 0").density_matrix(), np.full((2, 2), 0.5), atol=1e-12)
    # Teleportation: qubit 2 ends in qubit 0's state.
    psi_line = "I[U3(theta=0.7, phi=0.3, lambda=-1.1)] 0"
    rho = sq.Circuit(f"{psi_line}\nH 1\nCX 1 2\nCX 0 1\nH 0\nM 0 1\nCX rec[-1] 2\nCZ rec[-2] 2").density_matrix()
    want = sq.Circuit(psi_line).density_matrix()
    np.testing.assert_allclose(sq.reduced_density_matrix(rho, [2]), want, atol=1e-12)
    with pytest.raises(ValueError, match="at most 12"):
        sq.Circuit("H 12").density_matrix()
    with pytest.raises(ValueError, match="leakage"):
        sq.Circuit("I_ERROR[LEAK(p=0.1)] 0").density_matrix()


def test_correlated_error_chains():
    c = sq.Circuit("E(0.1) X0\nELSE_CORRELATED_ERROR(0.2) X1\nELSE_CORRELATED_ERROR(0.5) X0 X1\nM 0 1\nDETECTOR rec[-2]\nDETECTOR rec[-1]")
    d = c.exact_distribution(method="density_matrix")
    want = {(True, False): 0.1, (False, True): 0.9 * 0.2, (True, True): 0.9 * 0.8 * 0.5, (False, False): 0.9 * 0.8 * 0.5}
    for (dets, _), p in d.items():
        assert p == pytest.approx(want[dets], abs=1e-12)


def test_simulator():
    s = sq.DensityMatrixSimulator(seed=3)
    s.do(sq.Circuit("H 0\nCX 0 1"))
    assert s.num_qubits == 2 and s.purity() == pytest.approx(1)
    assert s.peek_observable_expectation(sq.PauliString("XX")) == pytest.approx(1)
    assert s.peek_observable_expectation(sq.PauliString("-YY")) == pytest.approx(1)
    assert s.entropy([0]) == pytest.approx(1) and s.entropy() == pytest.approx(0, abs=1e-9)
    assert s.fidelity(np.array([1, 0, 0, 1]) / math.sqrt(2)) == pytest.approx(1)
    np.testing.assert_allclose(s.state_vector(), np.array([1, 0, 0, 1]) / math.sqrt(2), atol=1e-12)
    s.do("DEPOLARIZE1(0.75) 0")
    assert s.purity() == pytest.approx(0.25) and s.peek_observable_expectation("ZZ") == pytest.approx(0)
    with pytest.raises(ValueError, match="mixed"):
        s.state_vector()
    s.do("M 0 1")
    rec = s.current_measurement_record()
    assert len(rec) == 2
    np.testing.assert_allclose(s.probabilities(), np.eye(4)[rec[0] + 2 * rec[1]], atol=1e-12)
    # Endianness, growth and setting a state.
    s = sq.DensityMatrixSimulator()
    s.do("X 2")
    assert s.num_qubits == 3 and np.argmax(s.probabilities()) == 4 and np.argmax(s.probabilities(endian="big")) == 1
    rho = np.diag([0.25, 0.75]).astype(complex)
    s.set_density_matrix(rho)
    assert s.num_qubits == 1 and s.peek_observable_expectation("Z") == pytest.approx(-0.5)
    s.set_num_qubits(2)
    np.testing.assert_allclose(s.reduced_density_matrix([0]), rho)
    copy = s.copy()
    copy.do("X 0")
    assert s.peek_observable_expectation("Z") == pytest.approx(-0.5)
    with pytest.raises(ValueError):
        s.set_num_qubits(13)


def test_simulator_operators_and_measurements():
    s = sq.DensityMatrixSimulator(seed=7)
    # A CX as a matrix, qubit order little- and big-endian.
    cx = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])
    s.apply_unitary([0], np.array([[1, 1], [1, -1]]) / math.sqrt(2))
    s.apply_unitary([0, 1], cx)
    assert s.peek_observable_expectation("XX") == pytest.approx(1) and s.peek_observable_expectation("ZZ") == pytest.approx(1)
    t = sq.DensityMatrixSimulator()
    t.do("H 2\nCX 2 0")
    u = sq.DensityMatrixSimulator()
    u.apply_unitary([2], np.array([[1, 1], [1, -1]]) / math.sqrt(2))
    u.apply_unitary([2, 0], cx[[0, 2, 1, 3]][:, [0, 2, 1, 3]], endian="big")
    np.testing.assert_allclose(u.density_matrix(), t.density_matrix(), atol=1e-12)
    # Kraus operators: amplitude damping as the tag runs it.
    g = 0.3
    k = [np.array([[1, 0], [0, math.sqrt(1 - g)]]), np.array([[0, math.sqrt(g)], [0, 0]])]
    a = sq.DensityMatrixSimulator()
    a.do("H 0")
    a.apply_kraus([0], k)
    b = sq.DensityMatrixSimulator()
    b.do(f"H 0\nI_ERROR[AMPLITUDE_DAMPING(gamma={g})] 0")
    np.testing.assert_allclose(a.density_matrix(), b.density_matrix(), atol=1e-12)
    # Forced and drawn measurements, in each basis.
    m = sq.DensityMatrixSimulator(seed=1)
    m.do("H 0")
    assert m.measure(0, forced=True) is True and m.peek_observable_expectation("Z") == pytest.approx(-1)
    assert m.measure(0, forced=False) is True  # impossible: the outcome that can happen
    m.do("H 0")
    assert m.measure(0, basis="X") is True and m.peek_observable_expectation("X") == pytest.approx(-1)
    m.do("H_YZ 0")
    assert m.measure(0, basis="Y") in (True, False)
    assert m.current_measurement_record()[:3] == [True, True, True]
    # A channel: the measurement averaged away, nothing recorded.
    c = sq.DensityMatrixSimulator()
    c.do("H 0")
    c.apply_channel("M 0")
    np.testing.assert_allclose(c.density_matrix(), np.eye(2) / 2, atol=1e-12)
    assert c.current_measurement_record() == []
    with pytest.raises(ValueError):
        c.apply_unitary([0], np.eye(4))
