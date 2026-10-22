"""Direct topological wrappers must honor or explicitly refuse engine rounds."""
from __future__ import annotations
import pytest
from stabilizer_qec import tqec


@pytest.mark.parametrize("graph,controls", [
    ("memory", {"rounds": 5}),
    ("cnot", {"rounds": 5}),
    ("cnot", {"merged": 5}),
])
def test_direct_round_overrides_have_same_refusal_as_engine_wrapper(graph, controls):
    graph = getattr(tqec.gallery, graph)(tqec.Basis.Z)
    direct = tqec.TopologicalComputationGraph(graph, **controls)
    compiled = tqec.compile_block_graph(graph, 3, **controls)
    with pytest.raises(ValueError, match="cannot honour") as expected:
        compiled.generate_stim_circuit(1)
    with pytest.raises(ValueError) as actual:
        direct.generate_stim_circuit(1)
    assert str(actual.value) == str(expected.value)


@pytest.mark.parametrize("controls", [{"rounds": 3}, {"merged": 5}, {"rounds": 3, "merged": 3}])
def test_direct_compatible_or_unused_round_overrides(controls):
    graph = tqec.gallery.memory(tqec.Basis.Z)
    direct = tqec.TopologicalComputationGraph(graph, **controls)
    expected = tqec.compile_block_graph(graph).generate_stim_circuit(1)
    assert str(direct.generate_stim_circuit(1)) == str(expected)
