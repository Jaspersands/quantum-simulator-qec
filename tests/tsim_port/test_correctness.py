"""Regressions for aliases and bounded measured-matrix dispatch."""
from __future__ import annotations

import numpy as np
import pytest

import stabilizer_qec as sq
from stabilizer_qec import tsim


@pytest.mark.parametrize("alias,canonical", [
    ("SQRT_Z[T] 0", "S[T] 0"),
    ("SQRT_Z_DAG[T] 0", "S_DAG[T] 0"),
    ("s[T] 0", "S[T] 0"),
    ("spp[T] X0", "SPP[T] X0"),
    ("spp[R_PAULI(theta=0.3*pi)] X0", "SPP[R_PAULI(theta=0.3*pi)] X0"),
    ("i[R_Z(theta=0.3*pi)] 0", "I[R_Z(theta=0.3*pi)] 0"),
    ("H_XZ 0", "H 0"),
])
@pytest.mark.parametrize("entry", ["constructor", "append", "insert"])
def test_tagged_aliases_execute_canonical_operation(alias, canonical, entry):
    expected = tsim.Circuit(canonical)
    if entry == "constructor":
        circuit = tsim.Circuit(alias)
    elif entry == "insert":
        circuit = tsim.Circuit()
        name, targets = alias.split(" ", 1)
        name, sep, tag = name.partition("[")
        # Preserve a Stim instruction's raw tag/spelling without the native
        # instruction constructor interpreting its radians-only rotation syntax.
        instruction = sq.CircuitInstruction._make(name, tag[:-1] if sep else "", [], [sq.GateTarget(t)._v for t in targets.split()])
        circuit.insert(0, instruction)
    else:
        name, targets = alias.split(" ", 1)
        name, sep, tag = name.partition("[")
        circuit = tsim.Circuit()
        circuit.append(name, targets.split(), tag=tag[:-1] if sep else "")
    assert circuit == expected
    assert circuit.tcount() == expected.tcount()
    assert np.allclose(circuit.to_matrix(), expected.to_matrix(), atol=1e-12)


@pytest.mark.parametrize("gate", ["SQRT_Z[T]", "SQRT_Z_DAG[T]", "s[T]"])
def test_tagged_alias_sampler_is_non_clifford(gate):
    circuit = tsim.Circuit("H 0\n" + gate + " 0\nH 0\nM 0")
    samples = circuit.compile_sampler().sample(20000)
    assert samples.dtype == np.bool_
    # T and T_DAG give sin(pi/8)^2; the old S execution gave 1/2.
    assert abs(samples.mean() - np.sin(np.pi / 8) ** 2) < 0.025
