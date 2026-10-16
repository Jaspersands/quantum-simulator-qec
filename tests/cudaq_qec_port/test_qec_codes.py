# Ported from CUDA-Q QEC (NVIDIA/cudaqx 0.8.1, libs/qec/unittests/test_qec.cpp; Copyright (c) 2024 -
# 2026 NVIDIA Corporation & Affiliates, Apache License 2.0), modified for stabilizer_qec: its code
# tests (the stabilizer and parity utilities, the Steane, repetition and surface codes, their
# orientations, schedules and memory experiments, and seeded code-capacity sampling) in Python
# on stabilizer_qec.cudaq_qec; C++-only tensor ranks read as Python's (0, n) arrays.

import numpy as np
import pytest

import stabilizer_qec.cudaq_qec as qec
from stabilizer_qec.cudaq_qec import XH, XV, ZH, ZV, amx, amz, empty

ORIENTATIONS = [XV, XH, ZV, ZH]


def overlap_parity(a, b):
    return len(set(a) & set(b)) % 2


def support_of(term):
    return [i for i, c in enumerate(str(term)[1:]) if c != "_"]


def test_construct_from_spin_ops():
    words = ["ZZZZIII", "XXXXIII", "IXXIXXI", "IIXXIXX", "IZZIZZI", "IIZZIZZ"]
    expected = np.array(
        [
            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1],
        ]
    )
    half = np.array([[1, 1, 1, 1, 0, 0, 0], [0, 1, 1, 0, 1, 1, 0], [0, 0, 1, 1, 0, 1, 1]])
    code = qec.get_code("steane", stabilizers=words)
    np.testing.assert_array_equal(code.get_parity(), expected)
    np.testing.assert_array_equal(code.get_parity_x(), half)
    np.testing.assert_array_equal(code.get_parity_z(), half)


def test_to_parity_matrix_edge_cases():
    code = qec.get_code("steane", stabilizers=["XXXIIII", "IXXXIII", "IIXXXII"])
    assert code.get_parity_z().shape == (0, 7)


def test_sample_memory_circuit_steane():
    steane = qec.get_code("steane")
    Lx, Lz = steane.get_observables_x(), steane.get_observables_z()
    n_shots, n_rounds = 10, 4
    syndromes, d = qec.sample_memory_circuit(steane, qec.operation.prep0, n_shots, n_rounds)
    assert syndromes.shape == (n_shots, 24)
    assert syndromes.sum() == 0
    assert not (Lz @ d.T % 2).any()
    _, d = qec.sample_memory_circuit(steane, qec.operation.prep1, n_shots, n_rounds)
    assert (Lz @ d.T % 2).sum() == n_shots
    _, d = qec.sample_memory_circuit(steane, qec.operation.prepp, n_shots, n_rounds)
    assert not (Lx @ d.T % 2).any()
    _, d = qec.sample_memory_circuit(steane, qec.operation.prepm, n_shots, n_rounds)
    assert (Lx @ d.T % 2).sum() == n_shots


def test_steane():
    steane = qec.get_code("steane")
    parity = steane.get_parity()
    assert parity.shape == (6, 14)
    np.testing.assert_array_equal(parity[:3, :7], [[1, 1, 1, 1, 0, 0, 0], [0, 1, 1, 0, 1, 1, 0], [0, 0, 1, 1, 0, 1, 1]])
    np.testing.assert_array_equal(steane.get_pauli_observables_matrix(), [[0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]])
    np.testing.assert_array_equal(steane.get_observables_x(), [[0, 0, 0, 0, 1, 1, 1]])
    np.testing.assert_array_equal(steane.get_observables_z(), [[0, 0, 0, 0, 1, 1, 1]])
    syndromes, _ = qec.sample_memory_circuit(steane, 10, 4)
    assert syndromes.sum() == 0
    # From stabilizers given as terms.
    words = ["ZZZZIII", "XXXXIII", "IXXIXXI", "IIXXIXX", "IZZIZZI", "IIZZIZZ"]
    terms = [qec._codes.PauliString(w) for w in sorted(words, key=qec._codes._sort_key)]
    code = qec.get_code("steane", stabilizers=terms)
    np.testing.assert_array_equal(code.get_parity(), parity)
    syndromes, _ = qec.sample_memory_circuit(code, 10, 4)
    assert syndromes.sum() == 0


def test_code_capacity():
    Hz = qec.get_code("steane").get_parity_z()
    syndromes, data = qec.sample_code_capacity(Hz, 10, 0)
    assert Hz.shape == (3, 7) and syndromes.shape == (10, 3) and data.shape == (10, 7)
    assert not data.any() and not syndromes.any()
    syndromes, data = qec.sample_code_capacity(Hz, 10, 0.15, 1337)
    np.testing.assert_array_equal(
        data,
        [[0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0]],
    )
    np.testing.assert_array_equal(syndromes, [[1, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 0], [0, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0], [0, 1, 1], [0, 0, 0]])
    syndromes, data = qec.sample_code_capacity(Hz, 10, 0.25, 1337)
    np.testing.assert_array_equal(
        data,
        [[0, 1, 0, 1, 1, 0, 0], [0, 0, 0, 1, 0, 0, 1], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0]],
    )
    np.testing.assert_array_equal(syndromes, [[0, 0, 1], [1, 0, 0], [1, 1, 1], [0, 1, 0], [0, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0], [0, 1, 1], [0, 0, 0]])


def test_code_capacity_with_code_object():
    steane = qec.get_code("steane")
    H = steane.get_parity()
    syndromes, data = qec.sample_code_capacity(steane, 5, 0.0)
    assert syndromes.shape == (5, H.shape[0]) and data.shape == (5, H.shape[1]) and not data.any() and not syndromes.any()
    a = qec.sample_code_capacity(steane, 8, 0.15, 1337)
    b = qec.sample_code_capacity(steane, 8, 0.15, 1337)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    rep = qec.get_code("repetition", distance=3)
    a = qec.sample_code_capacity(rep, 8, 0.2, 42)
    b = qec.sample_code_capacity(rep, 8, 0.2, 42)
    assert a[0].shape == (8, rep.get_parity().shape[0]) and all(np.array_equal(x, y) for x, y in zip(a, b))
    s1, d1 = qec.sample_code_capacity(steane, 3, 0.1)
    assert s1.shape == (3, H.shape[0]) and d1.shape == (3, H.shape[1])


def test_repetition():
    with pytest.raises(RuntimeError):
        qec.get_code("repetition")
    rep = qec.get_code("repetition", distance=9)
    assert (rep.get_num_x_stabilizers(), rep.get_num_z_stabilizers()) == (0, 8)
    assert [str(s)[1:].replace("_", "I") for s in rep.get_stabilizers()] == ["ZZIIIIIII", "IZZIIIIII", "IIZZIIIII", "IIIZZIIII", "IIIIZZIII", "IIIIIZZII", "IIIIIIZZI", "IIIIIIIZZ"]
    parity = rep.get_parity()
    assert parity.shape == (8, 18) and rep.get_parity_x().shape == (0, 9)
    expected = np.zeros((8, 18), dtype=np.uint8)
    for i in range(8):
        expected[i, i] = expected[i, i + 1] = 1
    np.testing.assert_array_equal(parity, expected)
    np.testing.assert_array_equal(rep.get_pauli_observables_matrix(), [[1] + [0] * 17])
    assert rep.get_observables_x().shape == (0, 9)
    np.testing.assert_array_equal(rep.get_observables_z(), [[1, 0, 0, 0, 0, 0, 0, 0, 0]])
    syndromes, data = qec.sample_memory_circuit(rep, 10, 4)
    assert syndromes.shape == (10, 40) and data.shape == (10, 9) and syndromes.sum() == 0


def test_two_qubit_bitflip_channel():
    qec.TwoQubitBitFlip(0.05)


def test_surface_code_orientation_from_string():
    f = qec.sc_orientation_from_str
    assert (f("XV"), f("o1"), f(" xh "), f("O2"), f("ZV"), f(" o3 "), f("zh"), f("O4")) == (XV, XV, XH, XH, ZV, ZV, ZH, ZH)
    with pytest.raises(RuntimeError):
        f("north")
    assert qec.get_code("surface_code", distance=3, orientation="O2").grid.orientation == XH


GRIDS = {
    XV: [[empty, amz, empty, empty], [empty, amx, amz, amx], [amx, amz, amx, empty], [empty, empty, amz, empty]],
    XH: [[empty, empty, amx, empty], [amz, amx, amz, empty], [empty, amz, amx, amz], [empty, amx, empty, empty]],
    ZV: [[empty, amx, empty, empty], [empty, amz, amx, amz], [amz, amx, amz, empty], [empty, empty, amx, empty]],
    ZH: [[empty, empty, amz, empty], [amx, amz, amx, empty], [empty, amx, amz, amx], [empty, amz, empty, empty]],
}


def test_surface_code_default_zh_layout():
    g = qec.stabilizer_grid(3)
    assert [g.roles[r * 4:(r + 1) * 4] for r in range(4)] == GRIDS[ZH]


@pytest.mark.parametrize("orientation", ORIENTATIONS)
def test_surface_code_orientation_algebra(orientation):
    d = 5
    g = qec.stabilizer_grid(d, orientation)
    assert len(g.x_stabilizers) == len(g.z_stabilizers) == (d * d - 1) // 2
    top, left = list(range(d)), list(range(0, d * d, d))
    horizontal = orientation in (XV, ZH)
    obs = g.get_spin_op_observables()
    assert len(obs) == 2
    lx, lz = support_of(obs[0]), support_of(obs[1])
    assert lx == (top if horizontal else left) and lz == (left if horizontal else top)
    assert overlap_parity(lx, lz) == 1
    for xs in g.x_stabilizers:
        assert overlap_parity(xs, lz) == 0
        assert all(overlap_parity(xs, zs) == 0 for zs in g.z_stabilizers)
    assert all(overlap_parity(zs, lx) == 0 for zs in g.z_stabilizers)


@pytest.mark.parametrize("orientation", ORIENTATIONS)
def test_surface_code_all_orientation_geometries(orientation):
    g = qec.stabilizer_grid(3, orientation)
    assert [g.roles[r * 4:(r + 1) * 4] for r in range(4)] == GRIDS[orientation]


def test_surface_code():
    with pytest.raises(RuntimeError):
        qec.get_code("surface_code")
    code = qec.get_code("surface_code", distance=3)
    np.testing.assert_array_equal(
        code.get_parity(),
        [
            [1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
        ],
    )
    np.testing.assert_array_equal(code.get_parity_x(), [[1, 0, 0, 1, 0, 0, 0, 0, 0], [0, 1, 1, 0, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 0, 1, 1, 0], [0, 0, 0, 0, 0, 1, 0, 0, 1]])
    np.testing.assert_array_equal(code.get_parity_z(), [[1, 1, 0, 1, 1, 0, 0, 0, 0], [0, 1, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 0, 1, 1], [0, 0, 0, 0, 0, 0, 1, 1, 0]])
    np.testing.assert_array_equal(code.get_pauli_observables_matrix(), [[1, 0, 0, 1, 0, 0, 1, 0, 0] + [0] * 9, [0] * 9 + [1, 1, 1, 0, 0, 0, 0, 0, 0]])
    np.testing.assert_array_equal(code.get_observables_x(), [[1, 1, 1, 0, 0, 0, 0, 0, 0]])
    np.testing.assert_array_equal(code.get_observables_z(), [[1, 0, 0, 1, 0, 0, 1, 0, 0]])
    assert (code.get_num_data_qubits(), code.get_num_ancilla_qubits(), code.get_num_ancilla_x_qubits(), code.get_num_ancilla_z_qubits()) == (9, 8, 4, 4)


def _spam(code, op, expected):
    L = code.get_observables_x() if op in (qec.operation.prepp, qec.operation.prepm) else code.get_observables_z()
    _, d = qec.sample_memory_circuit(code, op, 10, 4)
    return bool(((L @ d.T % 2)[0] == expected).all())


@pytest.mark.parametrize("name, options", [("steane", {}), ("surface_code", {"distance": 3})] + [("surface_code", {"distance": 3, "orientation": o}) for o in ("XV", "XH", "ZV")])
def test_spam(name, options):
    code = qec.get_code(name, **options)
    for op, value in ((qec.operation.prep0, 0), (qec.operation.prep1, 1), (qec.operation.prepp, 0), (qec.operation.prepm, 1)):
        assert _spam(code, op, value) and not _spam(code, op, 1 - value)


def test_repetition_spam():
    rep = qec.get_code("repetition", distance=9)
    assert _spam(rep, qec.operation.prep0, 0) and _spam(rep, qec.operation.prep1, 1)
    assert not _spam(rep, qec.operation.prep0, 1) and not _spam(rep, qec.operation.prep1, 0)


def test_steane_qubit_counts():
    s = qec.get_code("steane")
    assert (s.get_num_data_qubits(), s.get_num_ancilla_qubits(), s.get_num_ancilla_x_qubits(), s.get_num_ancilla_z_qubits()) == (7, 6, 3, 3)


@pytest.mark.parametrize("d, roles, stabs, data", [(3, 16, 4, 9), (5, 36, 12, 25), (17, 324, 144, 289)])
def test_stabilizer_grid(d, roles, stabs, data):
    g = qec.stabilizer_grid(d)
    for f in (g.format_stabilizer_grid, g.format_stabilizer_coords, g.format_stabilizer_indices, g.format_data_grid, g.format_stabilizers):
        assert f()
    assert (g.distance, g.grid_length, len(g.roles)) == (d, d + 1, roles)
    assert len(g.x_stab_coords) == len(g.z_stab_coords) == len(g.x_stab_indices) == len(g.z_stab_indices) == len(g.x_stabilizers) == len(g.z_stabilizers) == stabs
    assert len(g.data_coords) == len(g.data_indices) == data


def _check_schedule(sched, parity, d, vertical):
    assert sched.shape == parity.shape
    for s in range(sched.shape[0]):
        row = sched[s]
        np.testing.assert_array_equal(row != 0, parity[s] != 0)
        steps = row[row != 0]
        assert steps.max() <= 4 and len(set(steps.tolist())) == len(steps)
        support = [int(q) for step in range(1, 5) for q in np.flatnonzero(row == step)]
        assert len(support) in (2, 4)
        if len(support) == 4:
            a, b = support[2], support[3]
            if vertical:
                assert a % d == b % d and abs(a // d - b // d) == 1
            else:
                assert a // d == b // d and abs(a % d - b % d) == 1


@pytest.mark.parametrize("d", [3, 5, 7])
@pytest.mark.parametrize("orientation", ORIENTATIONS)
def test_cnot_schedule(d, orientation):
    g = qec.stabilizer_grid(d, orientation)
    code = qec.get_code("steane", stabilizers=[str(p)[1:].replace("_", "I") for p in g.get_spin_op_stabilizers()])
    px, pz = code.get_parity_x(), code.get_parity_z()
    sx, sz = g.get_cnot_schedule_x(), g.get_cnot_schedule_z()
    horizontal = orientation in (XV, ZH)
    _check_schedule(sx, px, d, horizontal)
    _check_schedule(sz, pz, d, not horizontal)
    for step in range(1, 5):
        touches = (sx == step).sum(axis=0) + (sz == step).sum(axis=0)
        assert touches.max() <= 1
    sc = qec.get_code("surface_code", distance=d, orientation=orientation.name)
    np.testing.assert_array_equal(sc.get_stabilizer_schedule_x(), sx)
    np.testing.assert_array_equal(sc.get_stabilizer_schedule_z(), sz)


def test_vec2d_operators():
    v1, v2, v3, v4 = qec.vec2d(2, 3), qec.vec2d(5, 7), qec.vec2d(2, 3), qec.vec2d(1, 3)
    s, diff = v1 + v2, v2 - v1
    assert (s.row, s.col, diff.row, diff.col) == (7, 10, 3, 4)
    assert v1 == v3 and not v1 == v2 and not v1 == v4
    assert v4 < v1 and v1 < v2 and not v1 < v3 and not v2 < v1
    v5, v6 = qec.vec2d(2, 1), qec.vec2d(2, 5)
    assert v5 < v1 and v1 < v6 and not v1 < v5
    assert [(v.row, v.col) for v in sorted([v1, v2, v4, v5, v6])] == [(1, 3), (2, 1), (2, 3), (2, 5), (5, 7)]
