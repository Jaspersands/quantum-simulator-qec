"""Serialization keeps mutable tqec schedules and rebuilds stateless builders."""
from __future__ import annotations

import pickle
import pytest
from stabilizer_qec import tqec


def move():
    return tqec.gallery.move_rotation(tqec.Basis.Z)


@pytest.mark.parametrize("engine", [False, True])
@pytest.mark.parametrize("convention", ["fixed_bulk", "fixed_boundary"])
@pytest.mark.parametrize("after_generation", [False, True])
def test_compilation_pickle_preserves_next_generation(engine, convention, after_generation):
    # Move/rotation exposes schedule mutation; the engine wrapper uses memory,
    # which is within its supported graph family.
    graph = tqec.gallery.memory(tqec.Basis.Z) if engine else move()
    compiled = tqec.compile_block_graph(graph, 3, convention=convention) if engine else tqec.compile_block_graph(graph, convention=convention)
    if after_generation:
        compiled.generate_stim_circuit(1)
    restored = pickle.loads(pickle.dumps(compiled))
    original_computation = compiled.computation if engine else compiled
    restored_computation = restored.computation if engine else restored
    assert restored_computation.convention == original_computation.convention
    assert restored_computation.num_observables == original_computation.num_observables
    assert restored_computation.stabilizers == original_computation.stabilizers
    if engine:
        assert str(restored.circuit) == str(compiled.circuit)
        assert restored.steps == compiled.steps
    assert str(restored.generate_stim_circuit(1)) == str(compiled.generate_stim_circuit(1))
