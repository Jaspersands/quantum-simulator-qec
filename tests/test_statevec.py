"""The exact state-vector simulator, against a dense density-matrix oracle written here (numpy,
a few qubits) and against the frame sampler on Clifford circuits."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

import stabilizer_qec as sq

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.diag([1, -1]).astype(complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S = np.diag([1, 1j])
PAULI = {"I": I2, "X": X, "Y": Y, "Z": Z}


def full(n: int, ops: dict) -> np.ndarray:
    """The operator acting as ops[q] on qubit q (identity elsewhere); bit q of a basis index is
    qubit q."""
    m = np.array([[1]], dtype=complex)
    for q in reversed(range(n)):
        m = np.kron(m, ops.get(q, I2))
    return m


def controlled(n: int, c: int, t: int, u: np.ndarray) -> np.ndarray:
    p0, p1 = np.diag([1, 0]).astype(complex), np.diag([0, 1]).astype(complex)
    return full(n, {c: p0}) + full(n, {c: p1, t: u})


class Oracle:
    """Density matrices by measurement record."""

    def __init__(self, n: int) -> None:
        self.n = n
        rho = np.zeros((2**n, 2**n), dtype=complex)
        rho[0, 0] = 1
        self.branches = {(): rho}

    def unitary(self, u: np.ndarray) -> None:
        self.branches = {r: u @ rho @ u.conj().T for r, rho in self.branches.items()}

    def channel(self, terms) -> None:
        """terms: (probability, operator); the rest of the probability does nothing."""
        rest = 1 - sum(p for p, _ in terms)
        self.branches = {r: rest * rho + sum(p * k @ rho @ k.conj().T for p, k in terms) for r, rho in self.branches.items()}

    def kraus(self, ks) -> None:
        self.branches = {r: sum(k @ rho @ k.conj().T for k in ks) for r, rho in self.branches.items()}

    def measure(self, q: int, basis: str, flip: float, reset: bool, record: bool = True) -> None:
        b = full(self.n, {q: H}) if basis == "X" else np.eye(2**self.n)
        proj = [b @ full(self.n, {q: np.diag([1, 0]).astype(complex)}) @ b, b @ full(self.n, {q: np.diag([0, 1]).astype(complex)}) @ b]
        flipper = b @ full(self.n, {q: X}) @ b
        out: dict = {}
        for r, rho in self.branches.items():
            for v in (0, 1):
                part = proj[v] @ rho @ proj[v]
                if reset and v:
                    part = flipper @ part @ flipper
                if not record:
                    out[r] = out.get(r, 0) + part
                    continue
                for bit, w in ((v, 1 - flip), (1 - v, flip)):
                    if w:
                        key = r + (bit,)
                        out[key] = out.get(key, 0) + w * part
        self.branches = out


def oracle_distribution(lines: list[str], n: int, detectors, observables, reference) -> dict:
    o = Oracle(n)
    for line in lines:
        name, *targets = line.split()
        qs = [int(t) for t in targets]
        if name == "H":
            o.unitary(full(n, {qs[0]: H}))
        elif name == "S":
            o.unitary(full(n, {qs[0]: S}))
        elif name in ("X", "Y", "Z"):
            o.unitary(full(n, {qs[0]: PAULI[name]}))
        elif name == "CX":
            o.unitary(controlled(n, qs[0], qs[1], X))
        elif name == "CZ":
            o.unitary(controlled(n, qs[0], qs[1], Z))
        elif name in ("M", "MX", "MR"):
            o.measure(qs[0], "X" if name == "MX" else "Z", 0.0, name == "MR")
        elif name.startswith("M("):
            o.measure(qs[0], "Z", float(name[2:-1]), False)
        elif name in ("R", "RX"):
            o.measure(qs[0], "X" if name == "RX" else "Z", 0.0, True, record=False)
        elif name.startswith("X_ERROR"):
            o.channel([(float(name[8:-1]), full(n, {qs[0]: X}))])
        elif name.startswith("DEPOLARIZE1"):
            p = float(name[12:-1])
            o.channel([(p / 3, full(n, {qs[0]: PAULI[c]})) for c in "XYZ"])
        elif name.startswith("DEPOLARIZE2"):
            p = float(name[12:-1])
            o.channel([(p / 15, full(n, {qs[0]: PAULI[a], qs[1]: PAULI[b]})) for a, b in itertools.product("IXYZ", repeat=2) if a + b != "II"])
        elif name.startswith("I_ERROR[R_") and "ZZ" not in name:
            axis, theta = name[10], float(name.split("theta=")[1].split(")")[0])
            p = full(n, {qs[0]: PAULI[axis]})
            o.unitary(np.cos(theta) * np.eye(2**n) - 1j * np.sin(theta) * p)
        elif name.startswith("II_ERROR[R_ZZ"):
            theta = float(name.split("theta=")[1].split(")")[0])
            p = full(n, {qs[0]: Z, qs[1]: Z})
            o.unitary(np.cos(theta) * np.eye(2**n) - 1j * np.sin(theta) * p)
        elif name == "I[T]":
            o.unitary(full(n, {qs[0]: np.diag([1, np.exp(1j * np.pi / 4)])}))
        elif name.startswith("I[U3"):
            th, ph, la = (float(v.split("=")[1]) for v in name[5:-2].split(","))
            u = np.array([[np.cos(th / 2), -np.exp(1j * la) * np.sin(th / 2)], [np.exp(1j * ph) * np.sin(th / 2), np.exp(1j * (ph + la)) * np.cos(th / 2)]])
            o.unitary(full(n, {qs[0]: u}))
        elif name.startswith("I_ERROR[AMPLITUDE_DAMPING"):
            g = float(name.split("gamma=")[1].split(")")[0])
            k0, k1 = np.array([[1, 0], [0, np.sqrt(1 - g)]]), np.array([[0, np.sqrt(g)], [0, 0]])
            o.kraus([full(n, {qs[0]: k0}), full(n, {qs[0]: k1})])
        else:
            raise AssertionError(name)
    nd = len(detectors)
    out: dict = {}
    for r, rho in o.branches.items():
        p = float(np.real(np.trace(rho)))
        if p < 1e-15:
            continue
        dets = tuple(bool(sum(r[i] for i in d) % 2 ^ reference[k]) for k, d in enumerate(detectors))
        obs = tuple(bool(sum(r[i] for i in d) % 2 ^ reference[nd + k]) for k, d in enumerate(observables))
        out[(dets, obs)] = out.get((dets, obs), 0) + p
    return out


GATES_1 = ["H", "S", "X", "Y", "Z", "I_ERROR[R_X(theta=0.3)]", "I_ERROR[R_Y(theta=-0.2)]", "I_ERROR[R_Z(theta=0.7)]", "I[T]", "I[U3(theta=0.4,phi=0.3,lambda=-0.5)]", "X_ERROR(0.1)", "DEPOLARIZE1(0.2)", "I_ERROR[AMPLITUDE_DAMPING(gamma=0.3)]", "R", "RX"]
GATES_2 = ["CX", "CZ", "DEPOLARIZE2(0.1)", "II_ERROR[R_ZZ(theta=0.4)]"]
MEAS = ["M", "MX", "MR", "M(0.1)"]


def random_case(rng):
    n = int(rng.integers(1, 4))
    lines, nm = [], 0
    for _ in range(int(rng.integers(4, 14))):
        u = rng.random()
        if u < 0.5:
            lines.append(f"{rng.choice(GATES_1)} {rng.integers(n)}")
        elif u < 0.75 and n > 1:
            a, b = rng.choice(n, 2, replace=False)
            lines.append(f"{rng.choice(GATES_2)} {a} {b}")
        else:
            lines.append(f"{rng.choice(MEAS)} {rng.integers(n)}")
            nm += 1
    lines.append(f"M {n - 1}")
    nm += 1
    detectors = [sorted(rng.choice(nm, int(rng.integers(1, min(nm, 2) + 1)), replace=False).tolist()) for _ in range(int(rng.integers(1, 4)))]
    observables = [sorted(rng.choice(nm, 1).tolist())]
    text = "\n".join(lines)
    for d in detectors:
        text += "\nDETECTOR " + " ".join(f"rec[{i - nm}]" for i in d)
    for o in observables:
        text += "\nOBSERVABLE_INCLUDE(0) " + " ".join(f"rec[{i - nm}]" for i in o)
    return n, lines, detectors, observables, nm, text


def reference(circuit: sq.Circuit, detectors, observables) -> list[bool]:
    """The noiseless reference run's parities (the state vector's detection events are the
    parities compared with them, as the converter's are)."""
    r = circuit.reference_sample()
    return [bool(sum(r[i] for i in d) % 2) for d in detectors + observables]


@pytest.mark.parametrize("seed", range(60))
def test_exact_distribution_equals_the_density_matrix(seed):
    rng = np.random.default_rng(seed)
    n, lines, detectors, observables, nm, text = random_case(rng)
    c = sq.Circuit(text)
    ours = c.exact_distribution()
    theirs = oracle_distribution(lines, n, detectors, observables, reference(c, detectors, observables))
    for key in set(ours) | set(theirs):
        assert abs(ours.get(key, 0) - theirs.get(key, 0)) < 1e-10, (text, key, ours.get(key), theirs.get(key))


@pytest.mark.parametrize("seed", range(6))
def test_samples_follow_the_exact_distribution(seed):
    rng = np.random.default_rng(100 + seed)
    *_, text = random_case(rng)
    c = sq.Circuit(text)
    exact = c.exact_distribution()
    shots = 4000
    d, o = c.compile_exact_sampler(seed=seed).sample(shots, separate_observables=True, threads=0)
    rows = [tuple(map(bool, r)) for r in d]
    for (dets, _), p in exact.items():
        hits = sum(r == dets for r in rows)
        marginal = sum(q for (k, _), q in exact.items() if k == dets)
        sigma = np.sqrt(marginal * (1 - marginal) / shots) + 1e-9
        assert abs(hits / shots - marginal) < 5 * sigma + 1e-12, (text, dets, hits / shots, marginal)


def test_clifford_circuits_agree_with_the_frame_sampler():
    c = sq.Circuit.generated("surface_code:rotated_memory_z", distance=3, rounds=2, after_clifford_depolarization=0.02, before_measure_flip_probability=0.02, after_reset_flip_probability=0.02)
    s = c.compile_exact_sampler(seed=3)
    assert s.num_qubits == 17
    shots = 1500
    ours = s.sample(shots, threads=0).mean(axis=0)
    theirs = c.compile_detector_sampler(seed=4).sample(200_000).mean(axis=0)
    sigma = np.sqrt(theirs * (1 - theirs) / shots) + 1e-3
    assert np.all(np.abs(ours - theirs) < 5 * sigma), (ours, theirs)


def test_threads_do_not_change_the_shots():
    c = sq.Circuit("R 0 1\nH 0\nI_ERROR[R_X(theta=0.4)] 1\nCX 0 1\nM 0 1\nDETECTOR rec[-1] rec[-2]")
    a = c.compile_exact_sampler(seed=9).sample(300, threads=1)
    b = c.compile_exact_sampler(seed=9).sample(300, threads=4)
    assert np.array_equal(a, b)


def test_leakage_is_refused():
    with pytest.raises(ValueError, match="leakage"):
        sq.Circuit("R 0\nI_ERROR[LEAK(p=0.1)] 0\nM 0").compile_exact_sampler()
