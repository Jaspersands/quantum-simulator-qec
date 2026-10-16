"""stabilizer_qec.tsim: Tsim's encoders on this package's circuits, verified by Tableau (each
encoding circuit maps the input qubit's X and Z to the code's logicals, and the reset qubits'
Z to its stabilizers, all with sign +), by Stim flows as Tsim's demos check them, and on the
encoded circuits Tsim's demos build (detectors deterministic, observables as encoded); and
Tsim's half-turn rotations, T and U3 read as this package's operations with Tsim's meaning
(exact distributions against a state vector written from Tsim's definitions)."""

import itertools
import math

import numpy as np
import pytest

import stabilizer_qec as sq
from stabilizer_qec.tsim import ColorEncoder5, SteaneEncoder, shorthand_to_stim, tsim_to_native

ENCODERS = [SteaneEncoder, ColorEncoder5]


def _pauli(n, letter, support):
    return sq.PauliString("".join(letter if q in support else "_" for q in range(n)))


def _unitary_part(encoder):
    # The encoding programs reset every qubit but the input at the start; the rest is unitary.
    text = shorthand_to_stim(encoder.encoding_program_text)
    reset = [line for line in text.splitlines() if line.strip().startswith("R ")]
    assert len(reset) == 1
    resets = {int(t) for t in reset[0].split()[1:]}
    body = "\n".join(line for line in text.splitlines() if not line.strip().startswith("R "))
    return resets, sq.Circuit(body).to_tableau()


@pytest.mark.parametrize("make", ENCODERS)
def test_encoding_maps_inputs_and_resets_by_tableau(make):
    enc = make()
    n, q = enc.n, enc.encoding_qubit
    resets, t = _unitary_part(enc)
    assert resets == set(range(n)) - {q}
    inverse = t.inverse()

    def pulled_back(p):
        return inverse(p)

    # Each stabilizer, pulled back through the encoding, is a +Z product on the reset qubits.
    for support in enc.stabilizer_generators:
        for letter in "XZ":
            back = pulled_back(_pauli(n, letter, support))
            assert back.sign == 1
            assert all(back[i] in (0, 3) for i in range(n)) and back[q] == 0, (letter, support, back)
    # Each logical, pulled back, is the input's Pauli times such a product.
    for letter, code in (("X", 1), ("Z", 3)):
        back = pulled_back(_pauli(n, letter, enc.observables[0]))
        assert back.sign == 1 and back[q] == code
        assert all(back[i] in (0, 3) for i in range(n) if i != q), (letter, back)


@pytest.mark.parametrize("make", ENCODERS)
def test_encoding_flows_as_tsims_demos_check(make):
    enc = make()
    n, q = enc.n, enc.encoding_qubit
    c = sq.Circuit(shorthand_to_stim(enc.encoding_program_text))
    flows = [sq.Flow(f"{_pauli(n, 'X', [q])} -> {_pauli(n, 'X', enc.observables[0])}"), sq.Flow(f"{_pauli(n, 'Z', [q])} -> {_pauli(n, 'Z', enc.observables[0])}")]
    flows += [sq.Flow(f"1 -> {_pauli(n, letter, s)}") for s in enc.stabilizer_generators for letter in "XZ"]
    assert c.has_all_flows(flows)
    assert not c.has_flow(sq.Flow(f"{_pauli(n, 'X', [q])} -> {_pauli(n, 'Z', enc.observables[0])}"))
    assert enc.encoding_flow_generators() == c.flow_generators()


@pytest.mark.parametrize("make", ENCODERS)
@pytest.mark.parametrize("state, logical, value", [("", "Z", 0), ("X", "Z", 1), ("H", "X", 0), ("X\nH", "X", 1)])
def test_initialize_prepares_the_logical_state(make, state, logical, value):
    enc = make()
    prep = "R 0" + (f"\n{state.replace(chr(10), ' 0' + chr(10))} 0" if state else "")
    enc.initialize(prep)
    enc.encode_transversally(("H 0\n" if logical == "X" else "") + "M 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]")
    c = enc.circuit
    assert c.num_qubits == enc.n and c.num_detectors == len(enc.stabilizer_generators)
    # Measured in the logical's basis, its checks are deterministic and the logical is the
    # prepared value (read from the measurements: the detector sampler reports flips).
    assert not c.compile_detector_sampler(seed=3).sample(64).any()
    m = c.compile_sampler(seed=3).sample(64)
    assert (np.bitwise_xor.reduce(m[:, enc.observables[0]], axis=1) == value).all()


def test_colour_code_distillation_as_tsims_demo():
    # magic_state_distillation.ipynb's circuit, its injected state Clifford (the T and rotation
    # make it non-Clifford; the structure is the demo's).
    enc = ColorEncoder5()
    enc.initialize("R 0 1 2 3 4\nH 0 1 2 3 4\nDEPOLARIZE1(0) 0 1 2 3 4")
    enc.encode_transversally(
        "SQRT_X 0 1 4\nCZ 0 1 2 3\nSQRT_Y 0 3\nCZ 0 2 3 4\nTICK\nSQRT_X_DAG 0\nCZ 0 4\nTICK\nCZ 1 3\nTICK\nSQRT_X_DAG 0 1 2 3 4\n"
        "M 0 1 2 3 4\n" + "".join(f"DETECTOR rec[-{k}]\n" for k in range(5, 0, -1)) + "".join(f"OBSERVABLE_INCLUDE({i}) rec[-{5 - i}]\n" for i in range(5))
    )
    c = enc.circuit
    assert (c.num_qubits, c.num_detectors, c.num_observables) == (85, 40, 5)
    det = c.compile_detector_sampler(seed=5).sample(64)
    assert not det.any()
    # The same program unencoded gives the logical outcomes' distribution.
    plain = sq.Circuit("R 0 1 2 3 4\nH 0 1 2 3 4\nSQRT_X 0 1 4\nCZ 0 1 2 3\nSQRT_Y 0 3\nCZ 0 2 3 4\nSQRT_X_DAG 0\nCZ 0 4\nCZ 1 3\nSQRT_X_DAG 0 1 2 3 4\nM 0 1 2 3 4")
    _, obs = c.compile_detector_sampler(seed=7).sample(4000, separate_observables=True)
    raw = plain.compile_sampler(seed=7).sample(4000)
    assert np.allclose(obs.mean(axis=0), raw.mean(axis=0), atol=0.06)


def test_encoded_t_is_this_packages_t():
    # Tsim's T (S[T] in its Stim) is this package's I[T], not S.
    enc = SteaneEncoder()
    enc.initialize("R 0\nH 0\nT 0")
    text = str(enc.circuit)
    assert "I[T] 6" in text and "S[T]" not in text


# Tsim's definitions, as a state vector: R_P(a) = exp(-i a pi P / 2), T = diag(1, e^{i pi / 4}),
# U3(a, b, c) = R_Z(b) R_Y(a) R_Z(c) up to phase, TPP P = exp(-i pi P / 8).
_P = {"I": np.eye(2), "X": np.array([[0, 1], [1, 0]]), "Y": np.array([[0, -1j], [1j, 0]]), "Z": np.diag([1, -1])}


def _rot(p, angle):
    return math.cos(angle) * np.eye(len(p)) - 1j * math.sin(angle) * p


def _kron_on(n, ops):
    m = np.array([[1.0]])
    for q in range(n):
        m = np.kron(m, ops.get(q, np.eye(2)))
    return m


def _reference(n, steps):
    psi = np.zeros(2**n, dtype=complex)
    psi[0] = 1
    for kind, arg, qubits in steps:
        if kind == "H":
            u = _kron_on(n, {qubits[0]: np.array([[1, 1], [1, -1]]) / math.sqrt(2)})
        elif kind == "CX":
            c, t = qubits
            u = np.zeros((2**n, 2**n))
            for i in range(2**n):
                bits = [(i >> (n - 1 - k)) & 1 for k in range(n)]
                if bits[c]:
                    bits[t] ^= 1
                u[sum(b << (n - 1 - k) for k, b in enumerate(bits)), i] = 1
        elif kind == "T":
            u = _kron_on(n, {qubits[0]: np.diag([1, np.exp(1j * math.pi / 4 * (1 if arg else -1))])})
        elif kind == "R":
            u = _rot(_kron_on(n, {qubits[0]: _P[arg[0]]}), arg[1] * math.pi / 2)
        elif kind == "U3":
            a, b, c = (x * math.pi for x in arg)
            rz = lambda x: np.diag([np.exp(-1j * x / 2), np.exp(1j * x / 2)])  # noqa: E731
            ry = np.array([[math.cos(a / 2), -math.sin(a / 2)], [math.sin(a / 2), math.cos(a / 2)]])
            u = _kron_on(n, {qubits[0]: rz(b) @ ry @ rz(c)})
        elif kind == "PP":
            pauli, angle = arg
            u = _rot(_kron_on(n, {q: _P[l] for q, l in zip(qubits, pauli)}), angle)
        psi = u @ psi
    return {format(i, f"0{n}b"): abs(a) ** 2 for i, a in enumerate(psi) if abs(a) ** 2 > 1e-12}


@pytest.mark.parametrize("seed", range(12))
def test_tsims_operations_read_with_tsims_meaning(seed):
    rng = np.random.default_rng(seed)
    n = 3
    steps, lines = [], []
    for _ in range(10):
        k = rng.integers(7)
        q = [int(x) for x in rng.permutation(n)[:2]]
        if k == 0:
            steps.append(("H", None, q[:1]))
            lines.append(f"H {q[0]}")
        elif k == 1:
            steps.append(("CX", None, q))
            lines.append(f"CX {q[0]} {q[1]}")
        elif k == 2:
            dag = bool(rng.integers(2))
            steps.append(("T", not dag, q[:1]))
            lines.append(f"{'T_DAG' if dag else 'T'} {q[0]}")
        elif k == 3:
            axis, a = "XYZ"[rng.integers(3)], round(float(rng.uniform(-1, 1)), 3)
            steps.append(("R", (axis, a), q[:1]))
            lines.append(f"R_{axis}({a}) {q[0]}")
        elif k == 4:
            a, b, c = (round(float(x), 3) for x in rng.uniform(-1, 1, 3))
            steps.append(("U3", (a, b, c), q[:1]))
            lines.append(f"U3({a}, {b}, {c}) {q[0]}")
        elif k == 5:
            pauli, a = "".join("XYZ"[i] for i in rng.integers(3, size=2)), round(float(rng.uniform(-1, 1)), 3)
            steps.append(("PP", (pauli, a * math.pi / 2), q))
            lines.append(f"R_PAULI({a}) {pauli[0]}{q[0]}*{pauli[1]}{q[1]}")
        else:
            pauli, dag = "".join("XYZ"[i] for i in rng.integers(3, size=2)), bool(rng.integers(2))
            steps.append(("PP", (pauli, -math.pi / 8 if dag else math.pi / 8), q))
            lines.append(f"{'TPP_DAG' if dag else 'TPP'} {pauli[0]}{q[0]}*{pauli[1]}{q[1]}")
    native = tsim_to_native(shorthand_to_stim("\n".join(lines)))
    # Outcomes from the exact distribution of a detector per measurement (events relative to
    # the reference sample, whose run reads the tags as identities).
    c = sq.Circuit(native + "\nM " + " ".join(map(str, range(n))) + "".join(f"\nDETECTOR rec[-{k}]" for k in range(n, 0, -1)))
    ref = c.reference_sample()
    got = {}
    for (events, _), p in c.exact_distribution().items():
        key = "".join(str(int(e) ^ int(r)) for e, r in zip(events, ref))
        got[key] = got.get(key, 0.0) + p
    want = _reference(n, steps)
    for key in set(want) | set(got):
        assert abs(got.get(key, 0.0) - want.get(key, 0.0)) < 1e-9, (native, key, got, want)


def test_tsim_to_native_forms():
    assert tsim_to_native("S[T] 0 1\nS_DAG[T:x] 2") == "I[T] 0 1\nI[T_DAG] 2"
    assert tsim_to_native("I[R_Z(theta=0.5*pi)] 0") == f"I[R_Z(theta={math.pi / 4!r})] 0"
    assert tsim_to_native("SPP_DAG[T] X0*!Y1 Z2") == f"I[R_PAULI(theta={math.pi / 8!r}, pauli=XY)] 0 1\nI[R_PAULI(theta={-math.pi / 8!r}, pauli=Z)] 2"
    assert tsim_to_native("H 0  # keep\nS 1") == "H 0  # keep\nS 1"
    assert list(itertools.islice(tsim_to_native("I[U3(theta=1.0*pi, phi=0.5*pi, lambda=0.0*pi)] 3").split("\n"), 1)) == [f"I[U3(theta={math.pi!r}, phi={math.pi / 2!r}, lambda=0.0)] 3"]
