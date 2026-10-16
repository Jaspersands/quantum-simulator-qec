# (c) Copyright Riverlane 2020-2025.
# modified for stabilizer_qec: its Stim is stabilizer_qec.deltakit_stim and its PyMatching this package's, and deltakit_{circuit,core,decode,explorer} are stabilizer_qec.deltakit.{circuit,core,decode,explorer}.
from copy import copy

import pytest

from stabilizer_qec.deltakit.circuit import Circuit, GateLayer
from stabilizer_qec.deltakit.circuit.gates import CX, H, X


@pytest.fixture
def circuit() -> Circuit:
    inner_gate_layers = [GateLayer([H(0), X(1)]), GateLayer([CX(1, 0)])]
    gate_layers = [
        GateLayer([H(1), H(0)]),
        Circuit(inner_gate_layers, iterations=4),
        Circuit(inner_gate_layers, iterations=1),
        GateLayer([CX(0, 1)]),
        GateLayer([X(0)]),
    ]
    return Circuit(gate_layers, iterations=5)


def test_circuit_shallow_copy_clones_object(circuit):
    other_circuit = copy(circuit)
    assert circuit is not other_circuit


def test_circuit_shallow_copy_clones_layers(circuit):
    other_circuit = copy(circuit)
    for l1, l2 in zip(circuit.layers, other_circuit.layers, strict=True):
        assert l1 is not l2


def test_circuit_shallow_copy_clones_layers_recursively(circuit):
    other_circuit = copy(circuit)
    for l1, l2 in zip(
        circuit.layers[1].layers, other_circuit.layers[1].layers, strict=True
    ):
        assert l1 is not l2
