"""The Pauli twirl of a circuit (`Circuit.twirled`) keeps its gates as they are, once, and twirls
its coherent errors alone; with merge, rotations that are one fault become one error, tagged
Clifford-angle gates being the Clifford gates they are. Also: the time reversal of a circuit's
tagged gates, and the coherent sampler's order 0."""

from __future__ import annotations

import math

import numpy as np
import pytest

import stabilizer_qec as sq


# Tagged gates of several targets, and of none but a noise channel (`I_ERROR[LEAK]`): no coherent
# error, so the twirl is the circuit itself. The parser fuses `I[T] 0` and `I[T] 1` into one.
GATES = [
    "I[T] 0 1",
    "I[T_DAG] 0 1 2",
    "I[R_Z(theta=0.1)] 0 1 2",
    "I[R_X(theta=0.3)] 0 1 2",
    "II[R_ZZ(theta=0.2)] 0 1 2 3",
    "I[R_PAULI(theta=0.3, pauli=XZ)] 0 1",
    "I[U3(theta=0.3, phi=0.2, lambda=0.1)] 0 1",
    "I[T] 0\nI[T] 1",
    "R 0 1\nI_ERROR[LEAK(p=0.1)] 0 1\nM 0 1",
]


@pytest.mark.parametrize("text", GATES)
def test_the_twirl_keeps_each_gate_once(text):
    c = sq.Circuit(text)
    assert str(c.twirled()) == str(c)
    assert c.twirled().num_non_clifford_gates() == c.num_non_clifford_gates()


def test_the_twirl_of_gates_and_errors_on_several_targets():
    """A gate of several targets is kept once, beside its errors twirled target by target."""
    c = sq.Circuit("I[T] 0 1\nI_ERROR[R_X(theta=0.1)] 0 1\nII[R_ZZ(theta=0.2)] 0 1 2 3")
    lines = str(c.twirled()).splitlines()
    assert lines[0] == "I[T] 0 1" and lines[-1] == "II[R_ZZ(theta=0.2)] 0 1 2 3" and len(lines) == 4
    for line, q in zip(lines[1:3], "01"):
        assert line.startswith("X_ERROR(") and line.endswith(f") {q}")
        assert abs(float(line.split("(")[1].split(")")[0]) - math.sin(0.1) ** 2) < 1e-6


def test_the_twirl_samples_what_the_circuit_samples_without_errors():
    """With no coherent error the twirl changes nothing: H T H on two qubits reads 1 with
    probability sin²(π/8) on each (not 1/2, as H S H, the T run twice, would)."""
    c = sq.Circuit("R 0 1\nH 0 1\nI[T] 0 1\nH 0 1\nM 0 1\nDETECTOR rec[-1]\nDETECTOR rec[-2]")
    for t in (c.twirled(), c.twirled(merge=True)):
        dist = t.exact_distribution(method="stabilizer_rank")
        assert dist.keys() == c.exact_distribution(method="stabilizer_rank").keys()
        for key, p in c.exact_distribution(method="stabilizer_rank").items():
            assert abs(dist[key] - p) < 1e-12, (str(t), key, dist[key], p)
        assert abs(dist[((False, False), ())] - math.cos(math.pi / 8) ** 4) < 1e-12


MERGE = """R 0 1
I_ERROR[R_X(theta=0.1)] 0
TICK
{H} 0
I_ERROR[R_Z(theta=0.15)] 0
TICK
{H} 0
{S} 1
I_ERROR[R_Y(theta=0.2)] 1
TICK
{S} 1
I_ERROR[R_X(theta=0.05)] 1
M 0 1
DETECTOR rec[-2]
DETECTOR rec[-1]
"""


def errors(c: sq.Circuit) -> list[str]:
    return [line for line in str(c).splitlines() if "_ERROR(" in line]


@pytest.mark.parametrize("tagged", [
    {"H": "I[U3(theta=1.5707963267948966, phi=0, lambda=3.141592653589793)]", "S": "I[R_Z(theta=0.7853981633974483)]"},
    {"H": "I[U3(theta=1.5707963267948966, phi=0, lambda=3.141592653589793)]", "S": "I[R_PAULI(theta=0.7853981633974483, pauli=Z)]"},
])
def test_the_merged_twirl_reads_tagged_clifford_gates_as_the_gates(tagged):
    """A tagged gate at a Clifford angle is the Clifford gate (as in the coherent samplers), so
    the rotations either side of it that are one fault merge as they do around the plain gate:
    X before H and Z after it are one error of sin²(0.25), Y before S and X after it one of
    sin²(0.25). The merged twirl keeps the tagged gates as written."""
    plain = sq.Circuit(MERGE.format(H="H", S="S"))
    ours = sq.Circuit(MERGE.format(**tagged))
    m = ours.twirled(merge=True)
    assert errors(m) == errors(plain.twirled(merge=True)) and len(errors(m)) == 2
    assert str(m).count("I[") == str(ours).count("I[")
    exact = plain.exact_distribution()
    for c in (m, plain.twirled(merge=True)):
        dist = c.exact_distribution()
        for key in exact.keys() | dist.keys():
            assert abs(dist.get(key, 0.0) - exact.get(key, 0.0)) < 1e-12, (key, dist, exact)


def test_the_merged_twirl_does_not_merge_across_a_non_clifford_gate():
    """A tagged gate at a non-Clifford angle is a barrier: its angle is never an error's (the
    gate still runs), and the errors either side of it are not one fault (they need not commute
    with it: Z either side of R_X). Around a Clifford gate they merge."""
    p = lambda c: sorted(float(line.split("(")[1].split(")")[0]) for line in errors(c))
    for gate, axis in [("I[R_X(theta=0.3)]", "X"), ("I[R_X(theta=0.3)]", "Z"), ("I[T]", "X")]:
        c = sq.Circuit(f"RX 0\nI_ERROR[R_{axis}(theta=0.1)] 0\n{gate} 0\nI_ERROR[R_{axis}(theta=0.2)] 0\nMX 0\nDETECTOR rec[-1]")
        m = c.twirled(merge=True)
        assert m.num_non_clifford_gates() == 1
        assert np.allclose(p(m), sorted([math.sin(0.1) ** 2, math.sin(0.2) ** 2]), atol=1e-6), (gate, axis, str(m))
    m = sq.Circuit("RX 0\nI_ERROR[R_X(theta=0.1)] 0\nI[R_X(theta=1.5707963267948966)] 0\nI_ERROR[R_X(theta=0.2)] 0\nMX 0\nDETECTOR rec[-1]").twirled(merge=True)
    assert np.allclose(p(m), [math.sin(0.3) ** 2], atol=1e-6), str(m)


# The time reversal of a tagged gate is its inverse (as `Circuit.inverse` writes it): each R_PAULI
# letter on its qubit, angles negated, T and T_DAG swapped.
REVERSED = [
    ("I[R_PAULI(theta=0.7853981633974483, pauli=XZ)] 0 1", "I[R_PAULI(theta=-0.7853981633974483, pauli=ZX)] 1 0"),
    ("II[R_PAULI(theta=0.7853981633974483, pauli=XZYZ)] 0 1 2 3", "II[R_PAULI(theta=-0.7853981633974483, pauli=YZXZ)] 2 3 0 1"),
    ("I[R_PAULI(theta=0.3, pauli=XYZ)] 0 1 2", "I[R_PAULI(theta=-0.3, pauli=ZYX)] 2 1 0"),
    ("I[T] 0 1", "I[T_DAG] 1 0"),
    ("I[R_Z(theta=0.7853981633974483)] 0", "I[R_Z(theta=-0.7853981633974483)] 0"),
    ("I[U3(theta=1.0, phi=2.0, lambda=3.0)] 0", "I[U3(theta=-1.0, phi=-3.0, lambda=-2.0)] 0"),
]


@pytest.mark.parametrize("text, expected", REVERSED)
def test_the_time_reversal_of_a_tagged_gate_is_its_inverse(text, expected):
    c = sq.Circuit(text)
    r = c.time_reversed_for_flows([])[0]
    assert str(r) == expected == str(c.inverse())
    u, v = r.to_matrix(), c.to_matrix()
    assert np.allclose(u @ v, np.eye(len(u)) * (u @ v)[0, 0]) and abs(abs((u @ v)[0, 0]) - 1) < 1e-9


def test_the_time_reversal_keeps_noise_tags():
    """A tagged channel that is not unitary (damping, leakage) is kept as the reversal keeps
    Stim's noise; a tag that is not ours is inert and kept."""
    for text in ["I_ERROR[AMPLITUDE_DAMPING(gamma=0.1)] 0 1", "I_ERROR[LEAK(p=0.1)] 0", "I[someone_else] 0 1"]:
        r = sq.Circuit(text).time_reversed_for_flows([])[0]
        assert str(r).split("]")[0] == text.split("]")[0]
