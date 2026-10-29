"""Direct topological wrappers must honor or explicitly refuse engine rounds."""
from __future__ import annotations
import gzip
import pathlib
import re
import numpy as np
import pytest
from stabilizer_qec import tqec
from stabilizer_qec.tqec import LinearFunction

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def fixture(name):
    return gzip.decompress((FIXTURES / f"{name}.stim.gz").read_bytes()).decode()


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


def test_y_hint_does_not_recommend_an_unimplemented_route():
    graph = tqec.BlockGraph()
    graph.add_cube(tqec.Position3D(0, 0, 0), "Y")
    graph.add_cube(tqec.Position3D(0, 0, 1), "ZXZ")
    graph.add_pipe(tqec.Position3D(0, 0, 0), tqec.Position3D(0, 0, 1))
    with pytest.raises(tqec.TQECError, match="also has no Y-cube implementation"):
        graph.compile(3)


def test_spatial_hadamard_hint_names_supported_convention():
    graph = tqec.BlockGraph()
    graph.add_cube(tqec.Position3D(0, 0, 0), "ZXZ")
    graph.add_cube(tqec.Position3D(1, 0, 0), "XZX")
    graph.add_pipe(tqec.Position3D(0, 0, 0), tqec.Position3D(1, 0, 0))
    with pytest.raises(tqec.TQECError, match="fixed_boundary"):
        graph.compile(3)


@pytest.mark.parametrize("height,rounds", [(5, 7), (lambda k: 5, 7), (LinearFunction(2, 1), 5)], ids=["number", "callable", "linear"])
def test_numpy_distance_honours_block_temporal_height(height, rounds):
    # np.arange(3, 10, 2) gives NumPy integers: the height is honoured at them as at an int.
    graph = tqec.gallery.memory(tqec.Basis.Z)
    expected = str(tqec.compile_block_graph(graph, 3, rounds=rounds).circuit)
    assert str(tqec.compile_block_graph(graph, 3, block_temporal_height=height).circuit) == expected
    compiled = tqec.compile_block_graph(graph, np.int64(3), block_temporal_height=height)
    assert str(compiled.circuit) == expected and type(compiled.distance) is int
    assert str(graph.compile(np.int64(3), block_temporal_height=height)) == expected
    assert str(graph.compile(distance=np.int32(3), block_temporal_height=height)) == expected


def test_numpy_distance_honours_linear_height_on_the_topological_graph():
    graph = tqec.gallery.memory(tqec.Basis.Z)
    direct = tqec.TopologicalComputationGraph(graph, block_temporal_height=LinearFunction(2, 1))
    assert str(direct.compile(np.int64(3)).circuit) == str(tqec.compile_block_graph(graph, 3, rounds=5).circuit)


@pytest.mark.parametrize("distance", [3.0, np.float64(3.0), True, 1, 4, np.int64(4)], ids=repr)
@pytest.mark.parametrize("height", [None, 5])
def test_distance_that_is_not_an_odd_integer_is_refused(distance, height):
    graph = tqec.gallery.memory(tqec.Basis.Z)
    with pytest.raises(tqec.TQECError, match="odd distance of at least 3, not " + re.escape(repr(distance))):
        tqec.compile_block_graph(graph, distance, block_temporal_height=height)
    with pytest.raises(tqec.TQECError, match="odd distance of at least 3"):
        graph.compile(distance, block_temporal_height=height)


# Graphs only tqec's route compiles: spatial cubes, junctions, a Hadamard pipe; whether a time
# slice merges (a spatial pipe runs in it). The stability experiment is one spatial cube.
TQEC_ONLY = [
    ("move_rotation", lambda: tqec.gallery.move_rotation(tqec.Basis.Z), "move_rotation_Z_fixed_bulk_k1", True),
    ("stability", lambda: tqec.gallery.stability(tqec.Basis.Z), "stability_Z_fixed_bulk_k1", False),
    ("cz", lambda: tqec.gallery.cz("ZZ -> ZZ"), "cz_ZZZZ_fixed_bulk_k1", True),
]


@pytest.mark.parametrize("name,build,reference,merges", TQEC_ONLY, ids=[c[0] for c in TQEC_ONLY])
def test_direct_round_controls_on_graphs_only_tqec_compiles(name, build, reference, merges):
    graph = build()
    with pytest.raises(tqec.TQECError):
        graph.compile(3)  # the engine cannot express it
    unused = [] if merges else [{"merged": 5}, {"rounds": 3, "merged": 5}]
    # Controls matching tqec's 2k + 1 layout (or a merged no slice uses) pass through to tqec's
    # circuit.
    for controls in [{"rounds": 3}, {"merged": 3}, {"rounds": 3, "merged": 3}] + unused:
        assert str(tqec.TopologicalComputationGraph(graph, **controls).generate_stim_circuit(1)) == fixture(reference), controls
    # Others get the layout refusal, never the engine's refusal of the graph.
    for controls in [{"rounds": 5}, {"rounds": 5, "merged": 3}] + ([] if unused else [{"merged": 5}, {"rounds": 3, "merged": 5}]):
        with pytest.raises(ValueError, match=r"cannot honour that") as refused:
            tqec.TopologicalComputationGraph(graph, **controls).generate_stim_circuit(1)
        assert not isinstance(refused.value, tqec.TQECError), controls
        assert "2k + 1 = 3 rounds" in str(refused.value)
