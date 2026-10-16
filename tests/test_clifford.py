"""QuantumClifford.jl's stabilizer-state structure and Clifford enumeration (stabilizer_qec.clifford):
its own tests (test_entanglement.jl, test_graphs.jl, test_enumerate.jl and its docstrings' pinned
values), the entropies against each other, the state vector's Schmidt spectrum and the density
matrix, graph states rebuilding their states, and the enumeration against Tableau.iter_all."""

import functools

import numpy as np
import pytest

import stabilizer_qec as sq

# Sizes around the 64-bit word boundary, as QuantumClifford's tests (they cost little here).
SIZES = [1, 2, 10, 63, 64, 65]


def _random_state(n, seed):
    """A random stabilizer state, signs included, from a seeded random Clifford circuit."""
    rng = np.random.default_rng(seed)
    c = sq.Circuit()
    c.append("I", [n - 1])
    for _ in range(4 * n + 4):
        q = int(rng.integers(n))
        k = rng.integers(4)
        if k == 0:
            c.append("H", [q])
        elif k == 1:
            c.append("S", [q])
        elif k == 2 and n > 1:
            a, b = (int(x) for x in rng.choice(n, 2, replace=False))
            c.append("CX", [a, b])
        else:
            c.append(["X", "Z"][rng.integers(2)], [q])
    return sq.Tableau.from_circuit(c).to_stabilizers()


def _canonical(ps):
    return [str(p) for p in sq.Tableau.from_stabilizers(ps).to_stabilizers(canonicalize=True)]


def _same_group(a, b):
    """Two lists of generators generate the same group, up to signs (mixed states too)."""
    n = max(len(p) for p in a + b)
    for p in a:
        if not all(p.commutes(q) for q in b):
            return False
    # Each of b is a product of a's: adding it to a does not raise the rank.
    return _rank(a, n) == _rank(a + b, n) == _rank(b, n)


def _rank(ps, n):
    rows = []
    for p in ps:
        xs, zs = p.to_numpy()
        row = np.zeros(2 * n, dtype=bool)
        row[: len(xs)], row[n : n + len(zs)] = xs, zs
        rows.append(row)
    return sq.clifford._rank_gf2(np.array(rows)) if rows else 0


# The clipped gauge.


def test_clip_pinned():
    s = ["-X_ZX_X", "+XXYZ__", "-YZ_Z_X", "-XZX__Y", "+_Z_Y_Y", "-____Z_"]
    assert [str(p) for p in sq.canonicalize_clip(s)] == ["-X_XY__", "+YZY___", "+_XZX__", "-_ZYX_Z", "-__YZ_X", "-____Z_"]


@pytest.mark.parametrize("n", SIZES)
def test_clipped_gauge_of_stabilizer_states(n):
    s = _random_state(n, n)
    clipped = sq.canonicalize_clip(s)
    assert _canonical(clipped) == _canonical(s)
    bg = sq.bigram(clipped, clip=False)
    assert all(np.count_nonzero(bg == j) == 2 for j in range(n))


@pytest.mark.parametrize("n", SIZES)
def test_entropy_of_mixed_states_clip_equals_rref(n):
    rng = np.random.default_rng(n)
    for k in range(4):
        s = _random_state(n, 100 * n + k)[: int(rng.integers(1, n + 1))]
        a, b = sorted(rng.integers(0, n, 2))
        sub = range(a, b + 1)
        assert sq.entanglement_entropy(s, sub, "clip", num_qubits=n) == sq.entanglement_entropy(s, sub, "rref", num_qubits=n)


@pytest.mark.parametrize("n", SIZES)
def test_entropy_of_pure_states_three_ways(n):
    rng = np.random.default_rng(n + 1)
    for k in range(4):
        s = _random_state(n, 200 * n + k)
        a, b = sorted(rng.integers(0, n, 2))
        sub = range(a, b + 1)
        assert sq.entanglement_entropy(s, sub, "clip") == sq.entanglement_entropy(s, sub, "graph") == sq.entanglement_entropy(s, sub, "rref")
        # Any subsystem, not only a contiguous one, for the graph and rref algorithms.
        sub = [int(q) for q in np.flatnonzero(rng.random(n) < 0.5)]
        assert sq.entanglement_entropy(s, sub, "graph") == sq.entanglement_entropy(s, sub, "rref")


def test_entropy_special_case():
    s = ["+XZZ_ZZ", "+ZX_ZZ_", "+Z_XZ_Z", "+_ZZXZZ", "+ZZ_ZXZ", "+Z_ZZZX"]
    for algorithm in ("clip", "graph", "rref"):
        assert sq.entanglement_entropy(s, range(3), algorithm) == 2


def _schmidt_entropy(ps, sub):
    n = len(ps[0])
    psi = sq.Tableau.from_stabilizers(ps).to_state_vector(endian="little").reshape((2,) * n)
    # Axis n - 1 - q is qubit q (qubit 0 the least significant bit).
    a = [n - 1 - q for q in sub]
    rest = [ax for ax in range(n) if ax not in a]
    m = np.transpose(psi, a + rest).reshape(2 ** len(a), -1)
    w = np.linalg.svd(m, compute_uv=False) ** 2
    w = w[w > 1e-12]
    return float(-(w * np.log2(w)).sum())


@pytest.mark.parametrize("n", [2, 3, 5, 7, 8])
def test_entropy_against_the_state_vector(n):
    rng = np.random.default_rng(n + 2)
    for k in range(6):
        s = _random_state(n, 300 * n + k)
        sub = [int(q) for q in np.flatnonzero(rng.random(n) < 0.5)]
        want = _schmidt_entropy(s, sub)
        assert sq.entanglement_entropy(s, sub, "rref") == pytest.approx(want, abs=1e-9)
        assert sq.entanglement_entropy(s, sub, "graph") == pytest.approx(want, abs=1e-9)
        if sub and sub == list(range(sub[0], sub[-1] + 1)):
            assert sq.entanglement_entropy(s, sub, "clip") == pytest.approx(want, abs=1e-9)


@functools.lru_cache(None)
def _pauli_matrices(n):
    return sq.pauli_basis(n)


def _mixed_rho(ps, n):
    rho = np.eye(2**n, dtype=complex)
    basis = _pauli_matrices(n)
    for p in ps:
        xs, zs = p.to_numpy()
        index = sum((1 if x and not z else 2 if x and z else 3 if z else 0) * 4**q for q, (x, z) in enumerate(zip(xs, zs)))
        rho = rho @ (np.eye(2**n) + p.sign.real * basis[index]) / 2
    return rho / np.trace(rho)


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_mixed_entropy_against_the_density_matrix(n):
    rng = np.random.default_rng(n + 3)
    for k in range(6):
        s = _random_state(n, 400 * n + k)[: int(rng.integers(1, n + 1))]
        rho = _mixed_rho(s, n)
        sub = [int(q) for q in np.flatnonzero(rng.random(n) < 0.5)]
        want = sq.von_neumann_entropy(sq.reduced_density_matrix(rho, sub), base=2) if sub else 0.0
        assert sq.entanglement_entropy(s, sub, "rref", num_qubits=n) == pytest.approx(want, abs=1e-9)


def test_mutual_information():
    ghz = ["XXXX", "ZZ__", "_ZZ_", "__ZZ"]
    assert sq.mutual_information(ghz, [0], [3]) == 1
    bell_pairs = ["XX__", "ZZ__", "__XX", "__ZZ"]
    assert sq.mutual_information(bell_pairs, [0], [1]) == 2
    assert sq.mutual_information(bell_pairs, [0], [2]) == 0
    with pytest.raises(ValueError):
        sq.mutual_information(ghz, [0, 1], [1])


def test_entropy_errors():
    with pytest.raises(ValueError, match="contiguous"):
        sq.entanglement_entropy(["XX", "ZZ"], [0, 2], "clip", num_qubits=3)
    with pytest.raises(ValueError, match="pure"):
        sq.entanglement_entropy(["XX"], [0], "graph")
    with pytest.raises(ValueError, match="algorithm"):
        sq.entanglement_entropy(["XX", "ZZ"], [0], "svd")


# Canonical forms.


@pytest.mark.parametrize("n", [3, 10, 65])
def test_rref_and_gott_keep_the_group(n):
    s = _random_state(n, 500 + n)
    rref, k = sq.canonicalize_rref(s, range(n // 2))
    assert _same_group(rref, s)
    assert all(not p.to_numpy()[0][: n // 2].any() and not p.to_numpy()[1][: n // 2].any() for p in rref[:k])
    gott, r, rank_z, xperm, zperm = sq.canonicalize_gott(s)
    assert r + rank_z == n
    perm = [xperm[q] for q in zperm]
    # Column k of the standard form is qubit perm[k]: undo it and the group is the input's.
    undone = [sq.PauliString.from_numpy(xs=p.to_numpy()[0][np.argsort(perm)], zs=p.to_numpy()[1][np.argsort(perm)], sign=p.sign) for p in gott]
    assert _same_group(undone, s)
    xs = np.array([p.to_numpy()[0] for p in gott])
    assert (xs[:r, :r] == np.eye(r, dtype=bool)).all() and not xs[r:].any()


# Graph states.


def _apply(t, ps):
    return [t(sq.PauliString(p)) for p in ps]


@pytest.mark.parametrize("n", SIZES)
def test_graph_state_conversion(n):
    s = _random_state(n, 600 + n)
    adj, h, ip, z = sq.graph_state(s)
    gate = sq.graph_gate(h, ip, z, n)
    graph = sq.Tableau.from_graph(adj).to_stabilizers()
    assert _canonical(_apply(gate, s)) == _canonical(graph)
    assert _canonical(_apply(gate.inverse(), graph)) == _canonical(s)


def test_graph_state_pinned():
    adj, h, ip, z = sq.graph_state(["+XXX", "+ZZ_", "-_ZZ"])
    assert [tuple(map(int, e)) for e in zip(*np.nonzero(np.triu(adj)))] == [(0, 1), (0, 2)]
    assert (h, ip, z) == ([1, 2], [], [2])
    # The second docstring: with a Y, the gates leave a graph state up to row products.
    s = ["+XXX", "+YZ_", "-_ZZ"]
    adj, h, ip, z = sq.graph_state(s)
    assert _canonical(_apply(sq.graph_gate(h, ip, z, 3), s)) == _canonical(sq.Tableau.from_graph(adj).to_stabilizers())
    # Gottesman's form permuting the qubits.
    s = ["-_XZ", "+_ZX", "+ZXZ"]
    adj, h, ip, z = sq.graph_state(s)
    assert _canonical(_apply(sq.graph_gate(h, ip, z, 3), s)) == _canonical(sq.Tableau.from_graph(adj).to_stabilizers())


@pytest.mark.parametrize("s", [["+Y"], ["-Y"], ["-Z"], ["+Y_", "-_Z"], ["XXXX", "ZZZZ", "ZXY_", "_X_X"]])
def test_graph_state_reconstruction(s):
    """Y needs the phase gate back; for -Z, the Z comes before the Hadamard (QuantumClifford's
    "direction and order" cases, the last the [[4,2,2]] code with logical Y and X fixed)."""
    n = len(s[0].lstrip("+-"))
    adj, h, ip, z = sq.graph_state(s)
    to_graph = sq.graph_gate(h, ip, z, n)
    graph = sq.Tableau.from_graph(adj).to_stabilizers()
    assert _canonical(_apply(to_graph, [sq.PauliString(p) for p in s])) == _canonical(graph)
    assert _canonical(_apply(to_graph.inverse(), graph)) == _canonical([sq.PauliString(p) for p in s])


def test_from_graph():
    t = sq.Tableau.from_graph(np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]]))
    assert [str(p) for p in t.to_stabilizers()] == ["+XZZ", "+ZX_", "+Z_X"]
    with pytest.raises(ValueError):
        sq.Tableau.from_graph(np.array([[0, 1], [0, 0]]))


# Enumeration.


def _key(t):
    return tuple(a.tobytes() for a in t.to_numpy())


def test_enumerate_one_qubit():
    unsigned = {_key(t) for t in sq.enumerate_cliffords(1)}
    assert unsigned == {_key(t) for t in sq.Tableau.iter_all(1, unsigned=True)}
    signed = {_key(t) for t in sq.enumerate_phases(sq.enumerate_cliffords(1))}
    assert signed == {_key(t) for t in sq.Tableau.iter_all(1)} and len(signed) == 24


def test_enumerate_two_qubits_is_a_bijection():
    ops = [_key(t) for t in sq.enumerate_cliffords(2)]
    assert len(ops) == len(set(ops)) == 720 == sq.clifford_cardinality(2, phases=False)
    assert set(ops) == {_key(t) for t in sq.Tableau.iter_all(2, unsigned=True)}
    assert sum(1 for _ in sq.enumerate_phases(sq.enumerate_cliffords(2))) == 11520


def _rows_text(t):
    n = len(t)
    return [str(t.x_output(q))[1:] for q in range(n)] + [str(t.z_output(q))[1:] for q in range(n)]


def test_enumerate_pinned_first():
    assert _rows_text(sq.enumerate_cliffords(3, 0)) == ["X__", "__Z", "_Z_", "Z__", "__X", "_X_"]
    assert _rows_text(next(iter(sq.enumerate_cliffords(5)))) == ["X____", "____Z", "___Z_", "__Z__", "_Z___", "Z____", "____X", "___X_", "__X__", "_X___"]


def test_cardinalities():
    for n in range(1, 11):
        assert sq.clifford_cardinality(n) == sq.clifford_cardinality(n, phases=False) * 2 ** (2 * n)
    assert sq.clifford_cardinality(1, phases=False) == 6
    with pytest.raises(IndexError):
        sq.enumerate_cliffords(1, 6)


def test_cosets_and_padding():
    cosets = list(sq.enumerate_cliffords(2, only_coset=True))
    assert len(cosets) == 15 * 8
    # Padded: the Clifford acts on the last n of padded_n qubits.
    for i in range(6):
        t = sq.enumerate_cliffords(1, i, padded_n=3)
        assert len(t) == 3 and t.x_output(0) == sq.PauliString("X__") and t.z_output(0) == sq.PauliString("Z__")


def test_symplectic_gram_schmidt():
    assert sq.symplectic_gram_schmidt("X", padded_n=3) == sq.Tableau(3)
    t = sq.symplectic_gram_schmidt("Z")
    assert t.x_output(0) == sq.PauliString("Z") and t.z_output(0) == sq.PauliString("X")
    rng = np.random.default_rng(7)
    for _ in range(20):
        n = int(rng.integers(1, 6))
        bits = rng.integers(0, 2, (2, n)).astype(bool)
        if not bits.any():
            continue
        p = sq.PauliString.from_numpy(xs=bits[0], zs=bits[1])
        t = sq.symplectic_gram_schmidt(p)
        out = t.x_output(0)
        assert (out.to_numpy()[0] == bits[0]).all() and (out.to_numpy()[1] == bits[1]).all()


def test_single_qubit_gates():
    assert sq.enumerate_single_qubit_gates(5) == sq.Tableau.from_named_gate("S")
    assert sq.enumerate_single_qubit_gates(3) == sq.Tableau.from_named_gate("H")
    assert sq.enumerate_single_qubit_gates(0) == sq.Tableau(1)
    t = sq.enumerate_single_qubit_gates(5, phases=(True, True))
    assert t.x_output(0) == sq.PauliString("-Y") and t.z_output(0) == sq.PauliString("-Z")
    assert sq.enumerate_single_qubit_gates(0, phases=(False, True)) == sq.Tableau.from_named_gate("X")
    assert {_key(sq.enumerate_single_qubit_gates(i)) for i in range(6)} == {_key(t) for t in sq.Tableau.iter_all(1, unsigned=True)}


def test_enumerate_phases():
    ts = list(sq.enumerate_phases(sq.Tableau.from_named_gate("CNOT")))
    assert len(ts) == len({_key(t) for t in ts}) == 16
    assert ts[0] == sq.Tableau.from_named_gate("CNOT")


def test_overcomplete_generators():
    # A cyclic repetition code's three checks are two independent ones: its entropies are the
    # independent pair's, every algorithm alike (the graph state's refusing a mixed state).
    checks = ["ZZ_", "_ZZ", "Z_Z"]
    for sub in ([0], [0, 1], [0, 1, 2]):
        assert sq.entanglement_entropy(checks, sub) == sq.entanglement_entropy(checks[:2], sub) == 1
    assert sq.entanglement_entropy(checks, [0, 1], "clip") == 1
    assert sq.mutual_information(checks, [0], [1]) == sq.mutual_information(checks[:2], [0], [1]) == 1
    with pytest.raises(ValueError, match="pure"):
        sq.entanglement_entropy(checks, [0], "graph")
    # A pure state listed with a repeat is still pure.
    ghz = ["XXX", "ZZ_", "_ZZ", "Z_Z"]
    assert sq.entanglement_entropy(ghz, [0], "graph") == 1
    assert sq.graph_state(ghz)[0].sum() == 4
    with pytest.raises(ValueError, match="−I"):
        sq.entanglement_entropy(["ZZ", "-ZZ"], [0])


def test_invalid_states_are_refused():
    for state in (["X", "Z"], ["XZ", "ZZ"]):
        for f in (sq.canonicalize_clip, sq.bigram, sq.graph_state, lambda s: sq.entanglement_entropy(s, [0])):
            with pytest.raises(ValueError, match="commute"):
                f(state)
    # One Pauli string is one generator, not its letters.
    assert sq.canonicalize_clip("XZ") == [sq.PauliString("XZ")]
    assert sq.entanglement_entropy("XX", [0]) == 1
    assert sq.canonicalize_clip(sq.PauliString("XYZ")) == [sq.PauliString("XYZ")]
    # A state read once (an iterator) gives what its list does.
    ghz = ["XXXX", "ZZ__", "_ZZ_", "__ZZ"]
    assert sq.mutual_information(iter(ghz), [0], [1], num_qubits=4) == sq.mutual_information(ghz, [0], [1]) == 1


def test_invalid_arguments_are_refused():
    with pytest.raises(ValueError):
        sq.graph_gate([4], [], [], 2)
    with pytest.raises(TypeError):
        list(sq.enumerate_phases("X"))
    with pytest.raises(ValueError):
        sq.clifford_cardinality(-1)
    with pytest.raises(TypeError):
        sq.enumerate_cliffords(1, 2.7)
    with pytest.raises(ValueError):
        sq.canonicalize_rref(["XZ", "ZX"], [-1])
    for bad in ([[0, np.nan], [np.nan, 0]], [[0, 2], [2, 0]], 0):
        with pytest.raises(ValueError):
            sq.Tableau.from_graph(bad)
    assert len(sq.Tableau.from_graph([])) == len(sq.Tableau.from_graph(np.zeros((0, 0)))) == 0
