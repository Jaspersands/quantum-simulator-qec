"""W7's cross-checks between the ported code libraries and this package's engine: each code's
distance by qLDPC's Brouwer–Zimmermann search, and the distance MQT QECC declares for its
constructions, equals the exact integer-programming distance of its code-capacity model (every
single-qubit X, Y and Z fault), and each encoding circuit qLDPC
synthesizes is checked on this package's ``Tableau``: its outputs are the code's stabilizers
and a logical basis, X and Z paired."""

import sys

import numpy as np
import pytest

import stabilizer_qec as sq

if sys.version_info < (3, 11):  # (at collection: the parameters are qLDPC's codes)
    pytest.skip("stabilizer_qec.qldpc needs Python 3.11, as qLDPC", allow_module_level=True)
for _module in ("scipy", "galois", "sympy", "cvxpy", "networkx", "diskcache", "platformdirs"):  # the qldpc extra
    pytest.importorskip(_module)


def codes():
    from stabilizer_qec.qldpc import codes as c

    yield "five-qubit", c.FiveQubitCode()
    yield "Steane", c.SteaneCode()
    yield "C4", c.C4Code()
    yield "C6", c.C6Code()
    yield "iceberg 6", c.IcebergCode(6)
    yield "quantum Hamming 4", c.QuantumHammingCode(4)
    yield "tetrahedral", c.TetrahedralCode()
    yield "surface 3", c.SurfaceCode(3, rotated=False)
    yield "rotated surface 5", c.SurfaceCode(5, rotated=True)
    yield "toric 3", c.ToricCode(3, rotated=False)
    yield "HGP(Hamming 3)", c.HGPCode(c.HammingCode(3))
    yield "quantum Golay", c.QuantumGolayCode()


def symplectic(code):
    """The code's stabilizers and logicals as rows [x | z] over GF(2)."""
    stabs = np.asarray(code.get_stabilizer_ops(symplectic=True), dtype=np.uint8) % 2
    logs = np.asarray(code.get_logical_ops(symplectic=True), dtype=np.uint8) % 2
    return stabs, logs


def anticommute(a, b):
    n = a.shape[-1] // 2
    return (a[..., :n] @ b[..., n:].T + a[..., n:] @ b[..., :n].T) % 2


def rank2(m):
    m = m.copy() % 2
    r = 0
    for c in range(m.shape[1]):
        piv = next((i for i in range(r, m.shape[0]) if m[i, c]), None)
        if piv is None:
            continue
        m[[r, piv]] = m[[piv, r]]
        for i in range(m.shape[0]):
            if i != r and m[i, c]:
                m[i] ^= m[r]
        r += 1
    return r


def capacity_model(stabs, logs):
    """Every single-qubit Pauli as a fault: the stabilizers it anticommutes with are its
    detectors, the logicals its observables."""
    n = stabs.shape[1] // 2
    lines = []
    for q in range(n):
        for x, z in ((1, 0), (1, 1), (0, 1)):
            p = np.zeros(2 * n, dtype=np.uint8)
            p[q], p[n + q] = x, z
            dets = np.flatnonzero(anticommute(p, stabs))
            obs = np.flatnonzero(anticommute(p, logs))
            if len(dets) or len(obs):
                lines.append("error(0.01) " + " ".join([f"D{d}" for d in dets] + [f"L{o}" for o in obs]))
    lines += [f"detector D{d}" for d in range(len(stabs))]
    lines += [f"logical_observable L{o}" for o in range(len(logs))]
    return sq.DetectorErrorModel("\n".join(lines))


@pytest.mark.parametrize("name,code", list(codes()), ids=lambda x: x if isinstance(x, str) else "")
def test_distance_by_brouwer_zimmermann_equals_integer_programming(name, code):
    stabs, logs = symplectic(code)
    assert int(code.get_distance()) == capacity_model(stabs, logs).distance(method="milp")


@pytest.mark.parametrize("name,code", list(codes()), ids=lambda x: x if isinstance(x, str) else "")
def test_encoders_on_this_packages_tableau(name, code):
    from stabilizer_qec.qldpc.circuits import get_encoding_circuit

    stabs, logs = symplectic(code)
    n, k = stabs.shape[1] // 2, logs.shape[0] // 2
    circuit = sq.Circuit(str(get_encoding_circuit(code)))
    t = circuit.to_tableau()

    def image(pauli, j):
        p = sq.PauliString(n)
        p[j] = pauli
        xs, zs = t(p).to_numpy()
        return np.concatenate([xs, zs]).astype(np.uint8)

    # Z on the inputs past the logical ones: the stabilizer group, all of it.
    zs = np.array([image("Z", j) for j in range(k, n)])
    assert not anticommute(zs, stabs).any()
    assert rank2(np.vstack([stabs, zs])) == rank2(stabs) == n - k
    # X and Z on the logical inputs: logicals (commuting with the stabilizers, outside their
    # group), paired as X_j, Z_j.
    xl = np.array([image("X", j) for j in range(k)]).reshape(k, 2 * n)
    zl = np.array([image("Z", j) for j in range(k)]).reshape(k, 2 * n)
    lo = np.vstack([xl, zl])
    assert not anticommute(lo, stabs).any()
    assert rank2(np.vstack([stabs, lo])) == n + k
    np.testing.assert_array_equal(anticommute(xl, zl), np.eye(k, dtype=np.uint8))


def mqt_codes():
    from stabilizer_qec.mqt_qecc import codes as m

    yield "rotated surface 3", m.RotatedSurfaceCode(3)
    yield "rotated surface 5", m.RotatedSurfaceCode(5)
    yield "hexagonal colour 5", m.HexagonalColorCode(5)
    yield "square-octagon colour 5", m.SquareOctagonColorCode(5)
    yield "quantum Hamming 4", m.construct_quantum_hamming_code(4)
    yield "iceberg 3", m.construct_iceberg_code(3)
    yield "many-hypercube 2", m.construct_many_hypercube_code(2)


@pytest.mark.parametrize("name,code", list(mqt_codes()), ids=lambda x: x if isinstance(x, str) else "")
def test_mqt_declared_distance_equals_integer_programming(name, code):
    n = code.n
    css = lambda h, x: np.hstack([h, np.zeros_like(h)]) if x else np.hstack([np.zeros_like(h), h])  # noqa: E731
    stabs = np.vstack([css(np.asarray(code.Hx) % 2, True), css(np.asarray(code.Hz) % 2, False)]).astype(np.uint8)
    logs = np.vstack([css(np.asarray(code.Lx) % 2, True), css(np.asarray(code.Lz) % 2, False)]).astype(np.uint8)
    assert stabs.shape[1] == 2 * n
    assert code.distance == capacity_model(stabs, logs).distance(method="milp")
