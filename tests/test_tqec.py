"""The block-graph port (`stabilizer_qec.tqec`): tqec's gallery, correlation surfaces, port
filling, rotations, JSON and Collada files, and the compile to this package's surgery engine.
The differential tests run when tqec is installed."""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

import stabilizer_qec as sq
from stabilizer_qec import tqec
from stabilizer_qec.tqec import Basis, BlockGraph, Direction3D, Position3D, TQECError, ZXCube, gallery

try:
    import tqec as theirs
    from tqec import gallery as their_gallery
    from tqec.utils.enums import Basis as TheirBasis
    from tqec.utils.position import Direction3D as TheirDirection
except ImportError:  # pragma: no cover
    theirs = None

needs_tqec = pytest.mark.skipif(theirs is None, reason="tqec is not installed")


def structure(g) -> tuple:
    """A graph's cubes, pipes and ports as plain tuples (ours or tqec's)."""
    return (
        sorted((c.position.as_tuple(), str(c.kind), c.label) for c in g.cubes),
        sorted((p.u.position.as_tuple(), p.v.position.as_tuple(), str(p.kind)) for p in g.pipes),
        {k: v.as_tuple() for k, v in g.ports.items()},
    )


def spans(surfaces) -> list:
    return [sorted(((e.u.id, str(e.u.basis)), (e.v.id, str(e.v.basis))) for e in s.span) for s in surfaces]


GALLERY = [("memory", "X"), ("memory", "Z"), ("stability", "X"), ("stability", "Z")]
GALLERY += [(name, b) for name in ("cnot", "move_rotation", "three_cnots", "steane_encoding") for b in (None, "X", "Z")]
CZ_FLOWS = [None, "XI -> XZ", ["XI -> XZ", "IZ -> IZ"], ["ZI -> ZI", "IX -> ZX"], "ZZ -> ZZ"]


def ours(name, basis):
    if name == "cz":
        return gallery.cz(basis)
    return getattr(gallery, name)(None if basis is None else Basis(basis))


def tqecs(name, basis):
    if name == "cz":
        return their_gallery.cz(basis)
    return getattr(their_gallery, name)(None if basis is None else TheirBasis(basis))


# ---------------------------------------------------------------------------------------------
# Loom-free checks with pinned values
# ---------------------------------------------------------------------------------------------


def test_memory_and_stability_are_one_cube_with_one_surface():
    g = gallery.memory(Basis.Z)
    assert g.num_cubes == 1 and g.num_pipes == 0 and g.spacetime_volume == 1
    surfaces = g.find_correlation_surfaces()
    assert len(surfaces) == 1 and surfaces[0].is_single_node
    # A lone cube has no leaf (degree-1) cube, so its external stabilizer is the empty string,
    # as in tqec; the surface's basis is the observable's.
    assert surfaces[0].bases_at(0) == {Basis.Z} and surfaces[0].external_stabilizer_on_graph(g) == ""
    assert gallery.memory(Basis.X).find_correlation_surfaces()[0].bases_at(0) == {Basis.X}
    assert gallery.stability(Basis.Z).cubes[0].kind == ZXCube.from_str("ZZX")


def test_cnot_surfaces_and_stabilizers():
    g = gallery.cnot()
    assert g.num_cubes == 10 and g.num_pipes == 9 and g.num_ports == 4 and g.is_open
    assert g.ordered_ports == ["In_Control", "In_Target", "Out_Control", "Out_Target"]
    open_surfaces = g.find_correlation_surfaces()
    assert len(open_surfaces) == 4
    # Ports in label order: In_Control, In_Target, Out_Control, Out_Target. The generators
    # tqec finds, of the CNOT's flows Z_C -> Z_C, Z_T -> Z_C Z_T, X_C -> X_C X_T, X_T -> X_T.
    assert [s.external_stabilizer_on_graph(g) for s in open_surfaces] == ["XXXI", "XIXX", "ZIZI", "ZZIZ"]
    z = gallery.cnot(Basis.Z)
    assert not z.is_open and z.spacetime_volume == 10
    # Closed graphs read their leaf cubes in insertion order: In_Control, Out_Control, In_Target, Out_Target.
    assert [s.external_stabilizer_on_graph(z) for s in z.find_correlation_surfaces()] == ["ZZII", "IZZZ"]
    x = gallery.cnot(Basis.X)
    assert [s.external_stabilizer_on_graph(x) for s in x.find_correlation_surfaces()] == ["XXXI", "XXIX"]


def test_gallery_sizes():
    assert [len(ours(n, b).find_correlation_surfaces()) for n, b in GALLERY] == [1, 1, 1, 1, 4, 2, 2, 2, 1, 1, 6, 3, 3, 7, 3, 4]
    assert [len(gallery.cz(f).find_correlation_surfaces()) for f in CZ_FLOWS] == [4, 2, 2, 2, 2]
    assert gallery.steane_encoding().num_cubes == 19 and gallery.steane_encoding().num_ports == 7
    assert gallery.three_cnots().num_half_y_cubes == 0


def test_validate_catches_bad_graphs():
    g = BlockGraph("bad")
    g.add_cube(Position3D(0, 0, 0), "P", "a")
    g.add_cube(Position3D(0, 0, 1), "ZXZ")
    g.add_cube(Position3D(0, 0, 2), "P", "b")
    with pytest.raises(TQECError):
        g.add_cube(Position3D(1, 0, 0), "P", "a")
    g.add_pipe(Position3D(0, 0, 0), Position3D(0, 0, 1))
    g.add_pipe(Position3D(0, 0, 1), Position3D(0, 0, 2))
    g.validate()
    g.add_cube(Position3D(1, 0, 1), "XZZ")
    g.add_pipe(Position3D(0, 0, 1), Position3D(1, 0, 1), "OXZ")
    with pytest.raises(TQECError, match="mismatched colors"):
        g.validate()
    assert g.get_degree(Position3D(0, 0, 1)) == 3
    assert g.shift_by(dx=3).is_open and g.shift_by(dx=3).ports["a"] == Position3D(3, 0, 0)


def test_fill_ports_for_minimal_simulation_covers_every_generator():
    g = gallery.cnot()
    filled = g.fill_ports_for_minimal_simulation()
    assert [f.stabilizers for f in filled] == [["XXXI", "XIXX"], ["ZIZI", "ZZIZ"]]
    for f in filled:
        assert f.graph.num_ports == 0
        # On the filled graph the stabilizers read the leaf cubes in insertion order (In_Control,
        # Out_Control, In_Target, Out_Target), not the ports' label order, as in tqec.
        assert f.get_external_stabilizers() == [s[0] + s[2] + s[1] + s[3] for s in f.stabilizers]
    with pytest.raises(TQECError):
        gallery.cnot(Basis.Z).fill_ports_for_minimal_simulation()


def test_rotation_compose_and_relabel():
    g = gallery.cnot()
    four = g
    for _ in range(4):
        four = four.rotate(Direction3D.Z)
    assert structure(four)[:2] == structure(g)[:2]
    r = g.rotate(Direction3D.Z)
    assert {str(c.kind) for c in r.cubes if not c.is_port} == {"XZX", "XZZ"}
    composed = gallery.cnot().compose(gallery.move_rotation(), "Out_Target", "In")
    assert composed.num_cubes == 14 and set(composed.ports) == {"In_Control", "Out_Control", "In_Target", "Out"}
    g.relabel_cubes({"In_Control": "C_in"})
    assert "C_in" in g.ports and "In_Control" not in g.ports
    with pytest.raises(TQECError):
        g.relabel_cubes({"C_in": "In_Target"})


def test_json_and_dae_round_trips(tmp_path: pathlib.Path):
    for name, basis in [("cnot", None), ("cnot", "Z"), ("steane_encoding", None), ("move_rotation", "X")]:
        g = ours(name, basis)
        g.to_json(tmp_path / "g.json")
        data = json.loads((tmp_path / "g.json").read_text())
        assert data["name"] == g.name and len(data["cubes"]) == g.num_cubes and all(c["transform"] == [[1, 0, 0], [0, 1, 0], [0, 0, 1]] for c in data["cubes"])
        surfaces = g.find_correlation_surfaces()
        g.to_dae_file(tmp_path / "g.dae", show_correlation_surface=surfaces[0])
        # Both readers name the ports afresh (Port0, Port1, ...); the blocks come back as written.
        blocks = sorted((c.position.as_tuple(), str(c.kind)) for c in g.cubes)
        pipes = sorted((p.u.position.as_tuple(), p.v.position.as_tuple(), str(p.kind)) for p in g.pipes)
        for read in (BlockGraph.from_json(tmp_path / "g.json"), BlockGraph.from_dae_file(tmp_path / "g.dae")):
            assert sorted((c.position.as_tuple(), str(c.kind)) for c in read.cubes) == blocks
            assert sorted((p.u.position.as_tuple(), p.v.position.as_tuple(), str(p.kind)) for p in read.pipes) == pipes
            assert read.num_ports == g.num_ports and all(label.startswith("Port") for label in read.ports)
            assert spans(read.find_correlation_surfaces()) == spans(surfaces)


def test_block_kinds():
    assert str(tqec.block_kind_from_str("ozxh")) == "OZXH" and tqec.block_kind_from_str("y") == tqec.YHalfCube()
    assert tqec.block_kind_from_str("port") == tqec.Port()
    kind = tqec.PipeKind.from_str("XOZ")
    assert kind.direction == Direction3D.Y and kind.get_basis_along(Direction3D.X) == Basis.X and kind.get_basis_along(Direction3D.Y) is None
    assert tqec.PipeKind.from_str("XOZH").get_basis_along(Direction3D.X, at_head=False) == Basis.Z
    with pytest.raises(TQECError):
        ZXCube.from_str("ZZZ")


# ---------------------------------------------------------------------------------------------
# Compiling to the surgery engine
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("basis", ["Z", "X"])
def test_memory_compiles_to_a_memory_circuit(basis):
    g = gallery.memory(Basis(basis))
    compiled = tqec.compile_block_graph(g, 3, p=0.001)
    assert compiled.num_patches == 1 and compiled.num_merges == 0 and compiled.stabilizers == [""]
    assert compiled.observables == [[("logical", 0, basis.lower(), 0)]]
    c = compiled.circuit
    assert c.num_observables == 1 and c.num_detectors > 0
    assert c.detector_error_model().num_errors > 0
    assert len(c.shortest_graphlike_error()) == 3


@pytest.mark.parametrize("basis", ["Z", "X"])
def test_cnot_compiles_with_its_two_observables(basis):
    g = gallery.cnot(Basis(basis))
    compiled = tqec.compile_block_graph(g, 3, p=0.001)
    assert compiled.num_patches == 3 and compiled.num_merges == 2
    assert compiled.stabilizers == (["ZZII", "IZZZ"] if basis == "Z" else ["XXXI", "XXIX"])
    c = compiled.circuit
    assert c.num_observables == 2
    # Noiseless, every detector and observable is quiet.
    quiet = g.compile(3)
    d, o = quiet.compile_detector_sampler(seed=1).sample(64, separate_observables=True)
    assert not d.any() and not o.any()
    # The shortest undetectable graphlike error is the code distance.
    assert len(c.shortest_graphlike_error()) == 3
    # Decoded, the logical error rate is the engine's own CNOT's (same patches, same merges;
    # ours prepares the ancilla at its own time slice): a few percent at d = 3, p = 1e-3.
    reference = sq.surgery.cnot(3, merged=3, p=0.001, inputs=basis.lower())
    rates = []
    for circuit in (c, reference):
        d, o = circuit.compile_detector_sampler(seed=7).sample(4000, separate_observables=True)
        predicted = sq.Matching.from_detector_error_model(circuit.detector_error_model(decompose_errors=True)).decode_batch(d)
        rates.append(float((predicted != o).any(axis=1).mean()))
    assert rates[0] < 0.08 and abs(rates[0] - rates[1]) < 0.03, rates


def test_compile_refuses_what_the_engine_cannot_express():
    with pytest.raises(TQECError, match="open ports"):
        gallery.cnot().compile(3)
    with pytest.raises(TQECError, match="Hadamard"):
        gallery.cz("XI -> XZ").compile(3)
    with pytest.raises(TQECError, match="spatial"):
        gallery.three_cnots(Basis.Z).compile(3)
    with pytest.raises(TQECError, match="spatial"):
        gallery.move_rotation(Basis.Z).compile(3)
    with pytest.raises(TQECError, match="odd distance"):
        gallery.memory().compile(4)


# ---------------------------------------------------------------------------------------------
# Against tqec
# ---------------------------------------------------------------------------------------------


@needs_tqec
@pytest.mark.parametrize("name,basis", GALLERY + [("cz", f) for f in CZ_FLOWS])
def test_gallery_agrees_with_tqec(name, basis, tmp_path: pathlib.Path):
    o, t = ours(name, basis), tqecs(name, basis)
    assert structure(o) == structure(t)
    assert (o.num_cubes, o.num_pipes, o.num_ports, o.spacetime_volume, o.is_open, o.bounding_box_size()) == (t.num_cubes, t.num_pipes, t.num_ports, t.spacetime_volume, t.is_open, t.bounding_box_size())
    o.validate()
    so, st = o.find_correlation_surfaces(), t.find_correlation_surfaces()
    assert spans(so) == spans(st)
    assert [s.external_stabilizer_on_graph(o) for s in so] == [s.external_stabilizer_on_graph(t) for s in st]
    assert [s.area for s in so] == [s.area for s in st]
    # JSON: the same text; each reads the other's.
    o.to_json(tmp_path / "o.json")
    t.to_json(tmp_path / "t.json")
    assert json.loads((tmp_path / "o.json").read_text()) == json.loads((tmp_path / "t.json").read_text())
    if o.num_pipes:
        assert structure(theirs.BlockGraph.from_json(tmp_path / "o.json")) == structure(theirs.BlockGraph.from_json(tmp_path / "t.json"))
        assert structure(BlockGraph.from_json(tmp_path / "t.json")) == structure(theirs.BlockGraph.from_json(tmp_path / "o.json"))
    # Collada: each reads the other's file to the same graph.
    o.to_dae_file(tmp_path / "o.dae", show_correlation_surface=so[0])
    t.to_dae_file(tmp_path / "t.dae", show_correlation_surface=st[0])
    assert structure(theirs.BlockGraph.from_dae_file(tmp_path / "o.dae")) == structure(theirs.BlockGraph.from_dae_file(tmp_path / "t.dae"))
    assert structure(BlockGraph.from_dae_file(tmp_path / "t.dae")) == structure(BlockGraph.from_dae_file(tmp_path / "o.dae")) == structure(theirs.BlockGraph.from_dae_file(tmp_path / "o.dae"))
    if o.is_open:
        fo, ft = o.fill_ports_for_minimal_simulation(), t.fill_ports_for_minimal_simulation()
        assert [f.stabilizers for f in fo] == [f.stabilizers for f in ft]
        assert [structure(f.graph) for f in fo] == [structure(f.graph) for f in ft]
    for axis in ("X", "Y", "Z"):
        for ccw in (True, False):
            assert structure(o.rotate(getattr(Direction3D, axis), ccw, 1)) == structure(t.rotate(getattr(TheirDirection, axis), ccw, 1))
    assert structure(o.fix_shadowed_faces()) == structure(t.fix_shadowed_faces())


@needs_tqec
def test_compose_agrees_with_tqec():
    o = gallery.cnot().compose(gallery.move_rotation(), "Out_Target", "In")
    t = their_gallery.cnot().compose(their_gallery.move_rotation(), "Out_Target", "In")
    assert structure(o) == structure(t)


@needs_tqec
def test_steane_dae_reads_as_tqec_reads_it():
    assert structure(BlockGraph.from_dae_file(gallery.STEANE_CODE_DAE)) == structure(theirs.BlockGraph.from_dae_file(gallery.STEANE_CODE_DAE))


# ---------------------------------------------------------------------------------------------
# The 2.0 review's regressions
# ---------------------------------------------------------------------------------------------


def _row_of_patches(n: int, merge_at: list[int], height: int) -> BlockGraph:
    """`n` ZXZ patches side by side for `height` slices, merged along the row at the slices `merge_at`."""
    g = BlockGraph("row")
    for x in range(n):
        for z in range(height):
            g.add_cube(Position3D(x, 0, z), "ZXZ")
            if z:
                g.add_pipe(Position3D(x, 0, z - 1), Position3D(x, 0, z))
    for z in merge_at:
        for x in range(n - 1):
            g.add_pipe(Position3D(x, 0, z), Position3D(x + 1, 0, z))
    return g


def test_a_surface_crossing_two_seams_of_one_merge_gets_one_seam_term():
    # Three patches merged in a row: the ZZZ-type surface crosses both seams of the one merge. One
    # seam term covers every seam of its merge, so a second term would cancel the first (the
    # observable then depended on the seam qubits' resets).
    compiled = tqec.compile_block_graph(_row_of_patches(3, [1], 3), 3, p=0.001)
    assert compiled.num_merges == 1 and compiled.stabilizers == ["ZZZZZZ"]
    (terms,) = compiled.observables
    assert [t for t in terms if t[0] == "seam"] == [("seam", 0, terms[0][2], 0)] and terms[0][2] in (0, 1, 2)
    assert terms[1:] == [("logical", 0, "z", 0), ("logical", 1, "z", 0), ("logical", 2, "z", 0)]
    assert len(compiled.circuit.shortest_graphlike_error()) == 3
    d, o = tqec.compile_block_graph(_row_of_patches(3, [1], 3), 3).circuit.compile_detector_sampler(seed=1).sample(64, separate_observables=True)
    assert not d.any() and not o.any()


def test_the_same_pair_merging_twice_is_two_merges():
    # Merges are indexed per time slice: the pair (0, 1) merges at z = 1 and again at z = 3.
    compiled = tqec.compile_block_graph(_row_of_patches(2, [1, 3], 5), 3, p=0.001)
    assert compiled.num_merges == 2
    assert compiled.observables == [[("seam", 0, 0, 0), ("seam", 1, 0, 0), ("logical", 0, "z", 0), ("logical", 1, "z", 0)]]
    assert len(compiled.circuit.shortest_graphlike_error()) == 3
    d, o = tqec.compile_block_graph(_row_of_patches(2, [1, 3], 5), 3).circuit.compile_detector_sampler(seed=1).sample(64, separate_observables=True)
    assert not d.any() and not o.any()


def test_hadamard_direction_adjustment_is_an_involution():
    from stabilizer_qec.tqec.block_graph import block_kind_from_str
    from stabilizer_qec.tqec.rotations import adjust_hadamards_direction

    pairs = {"ZXOH": "XZOH", "XOZH": "ZOXH", "OXZH": "OZXH"}
    for a, b in pairs.items():
        assert str(adjust_hadamards_direction(block_kind_from_str(a))) == b
        assert str(adjust_hadamards_direction(block_kind_from_str(b))) == a  # the inverse pairs were missing


def test_explicit_observables_must_be_surfaces_of_the_graph():
    # Another graph's surfaces (the X-basis CNOT's on the Z-basis CNOT: the same pipes, the
    # other port fills) would compile into random observables.
    with pytest.raises(TQECError, match="not a correlation surface of this graph"):
        tqec.compile_block_graph(gallery.cnot(Basis.Z), 3, observables=gallery.cnot(Basis.X).find_correlation_surfaces())
    for b in (Basis.Z, Basis.X):
        own = gallery.cnot(b).find_correlation_surfaces()
        assert len(tqec.compile_block_graph(gallery.cnot(b), 3, observables=own).observables) == 2
    # A lone cube beside another component: tqec's search needs a leaf in every component.
    g = BlockGraph("g")
    g.add_cube(Position3D(0, 0, 1), "XZX")
    g.add_cube(Position3D(1, 0, 1), "XZZ")
    g.add_cube(Position3D(1, 0, 2), "XZZ")
    g.add_pipe(Position3D(1, 0, 1), Position3D(1, 0, 2))
    with pytest.raises(TQECError, match="has no pipe"):
        g.compile(3)


# ---------------------------------------------------------------------------------------------
# 2.1: what the tqec audit found (tqec 0.2.0's numbers recorded where tqec takes minutes)
# ---------------------------------------------------------------------------------------------


def _passthrough(graph_class, position, middle: str):
    """Two ZXZ patches three slices tall, joined at the middle slice through a routing cube."""
    g = graph_class("passthrough")
    for x in (0, 2):
        for z in range(3):
            g.add_cube(position(x, 0, z), "ZXZ")
            if z:
                g.add_pipe(position(x, 0, z - 1), position(x, 0, z))
    g.add_cube(position(1, 0, 1), middle)
    g.add_pipe(position(0, 0, 1), position(1, 0, 1))
    g.add_pipe(position(1, 0, 1), position(2, 0, 1))
    return g


def test_compile_fixes_shadowed_faces_first():
    # The routing cube drawn XXZ has its x faces hidden by the two pipes: tqec's compile runs
    # fix_shadowed_faces, which makes it ZXZ, and compiles (81 qubits, 184 detectors, one
    # observable, graphlike distance 3 at k = 1 in tqec 0.2.0).
    drawn = tqec.compile_block_graph(_passthrough(BlockGraph, Position3D, "XXZ"), 3, p=0.001)
    fixed = tqec.compile_block_graph(_passthrough(BlockGraph, Position3D, "ZXZ"), 3, p=0.001)
    assert str(drawn.circuit) == str(fixed.circuit) and drawn.stabilizers == ["ZZZZ"]
    c = drawn.circuit
    assert (c.num_qubits, c.num_detectors, c.num_observables) == (81, 184, 1)
    assert len(c.shortest_graphlike_error()) == 3
    # A cube that really is spatial is still refused, and the message says which kind it is.
    with pytest.raises(TQECError, match=r"spatial \(its spatial boundaries all of one basis\)"):
        gallery.stability(Basis.Z).compile(3)
    with pytest.raises(TQECError, match=r"spatial \(a junction of two spatial directions\)"):
        gallery.three_cnots(Basis.Z).compile(3)


def test_observables_take_tqecs_values():
    # tqec: "auto" (the default) is every correlation surface, None is none.
    g = gallery.cnot(Basis.Z)
    assert tqec.compile_block_graph(g, 3).circuit.num_observables == 2
    assert tqec.compile_block_graph(g, 3, observables="auto").circuit.num_observables == 2
    assert tqec.compile_block_graph(g, 3, observables=None).circuit.num_observables == 0
    assert tqec.compile_block_graph(g, 3, observables=[]).circuit.num_observables == 0
    assert g.compile(3, observables=None).num_observables == 0 and g.compile(3).num_observables == 2
    with pytest.raises(TQECError, match="'auto', None or a list"):
        tqec.compile_block_graph(g, 3, observables="all")


@needs_tqec
def test_shadowed_faces_and_observables_against_tqec():
    t = theirs.compile_block_graph(_passthrough(theirs.BlockGraph, theirs_position(), "XXZ"))
    assert len(t._observables) == 1
    assert theirs.compile_block_graph(_passthrough(theirs.BlockGraph, theirs_position(), "XXZ"), observables=None)._observables == []
    assert structure(_passthrough(BlockGraph, Position3D, "XXZ").fix_shadowed_faces()) == structure(_passthrough(theirs.BlockGraph, theirs_position(), "XXZ").fix_shadowed_faces())


def theirs_position():
    from tqec.utils.position import Position3D as TheirPosition

    return TheirPosition


def _two_cnots(module, fill: str):
    g = module.gallery.cnot().compose(module.gallery.cnot(), "Out_Control", "In_Control")
    g.fill_ports(module.ZXCube.from_str(fill) if module is tqec else _their_zx(fill))
    return g


def _their_zx(kind: str):
    from tqec.computation.cube import ZXCube as TheirZXCube

    return TheirZXCube.from_str(kind)


def _decoded_rates(circuit, shots: int, seed: int = 7) -> list[float]:
    d, o = circuit.compile_detector_sampler(seed=seed).sample(shots, separate_observables=True)
    predicted = sq.Matching.from_detector_error_model(circuit.detector_error_model(decompose_errors=True)).decode_batch(d)
    return [float(x) for x in (predicted != o).mean(axis=0)]


# tqec 0.2.0 on the two CNOTs (cnot().compose(cnot(), "Out_Control", "In_Control"), ports filled
# ZXZ or ZXX), k = 1, NoiseModel.uniform_depolarizing(0.001), PyMatching, 20 000 shots: 81 qubits,
# 464 detectors, 2 observables, graphlike distance 3.
TQEC_TWO_CNOTS = {"Z": (["ZIZI", "IZIZ"], [0.0163, 0.0402]), "X": (["XXXX", "XIXI"], [0.0451, 0.0423])}


@pytest.mark.parametrize("basis", ["Z", "X"])
def test_two_cnots_in_a_row_reuse_the_ancillas_tile(basis):
    g = _two_cnots(tqec, "ZX" + basis)
    compiled = tqec.compile_block_graph(g, 3, p=0.001)
    # The ancilla's tile holds two patches, one per CNOT, the first read out before the second
    # is prepared.
    assert compiled.tiles == [(0, 0), (1, 0), (1, 0), (1, 1)] and compiled.num_merges == 4
    stabilizers, rates = TQEC_TWO_CNOTS[basis]
    assert compiled.stabilizers == stabilizers
    c = compiled.circuit
    # The engine's layout spends more qubits than tqec's (113, not 81); the detectors are tqec's.
    assert (c.num_qubits, c.num_detectors, c.num_observables) == (113, 464, 2)
    assert len(c.shortest_graphlike_error()) == 3
    d, o = g.compile(3).compile_detector_sampler(seed=1).sample(128, separate_observables=True)
    assert not d.any() and not o.any()
    # Under tqec's own noise model, laid over the engine's circuit as tqec lays it over its own,
    # the logical error rates are tqec's within a factor of two (about 1.3 times tqec's: the
    # engine's rounds idle more moments).
    noisy = tqec.compile_block_graph(g).generate_stim_circuit(k=1, noise_model=tqec.NoiseModel.uniform_depolarizing(0.001))
    assert noisy.num_detectors == 464
    ours = _decoded_rates(noisy, 8000)
    assert all(r / 2 < x < 2 * r for x, r in zip(ours, rates)), (ours, rates)
    assert len(tqec.compile_block_graph(g, 5, p=0.001).circuit.shortest_graphlike_error()) == 5


def _stacked(first: str, second: str, x2: int = 0) -> BlockGraph:
    """A patch at (0, 0) for z = 0, 1, then one at (x2, 0) for z = 2, 3, not joined in time."""
    g = BlockGraph("stacked")
    for x, z, kind in ((0, 0, first), (0, 1, first), (x2, 2, second), (x2, 3, second)):
        g.add_cube(Position3D(x, 0, z), kind)
    g.add_pipe(Position3D(0, 0, 0), Position3D(0, 0, 1))
    g.add_pipe(Position3D(x2, 0, 2), Position3D(x2, 0, 3))
    return g


@pytest.mark.parametrize("first,second", [("ZXZ", "ZXZ"), ("ZXZ", "ZXX"), ("ZXX", "ZXZ")])
def test_a_tile_read_out_and_prepared_again_at_once(first, second):
    # The second patch is prepared the slice after the first is read out, so its first round
    # follows the first patch's last round on the same checks: it starts afresh, with every
    # detector the two patches have on two tiles (comparing across the readout instead loses
    # the first round's detectors when the basis changes, and the distance with them).
    together = tqec.compile_block_graph(_stacked(first, second), 3, p=0.001)
    apart = tqec.compile_block_graph(_stacked(first, second, x2=1), 3, p=0.001)
    assert together.tiles == [(0, 0), (0, 0)] and apart.tiles == [(0, 0), (0, 1)]
    assert together.circuit.num_detectors == apart.circuit.num_detectors == 96
    assert len(together.circuit.shortest_graphlike_error()) == 3
    d, o = tqec.compile_block_graph(_stacked(first, second), 3).circuit.compile_detector_sampler(seed=1).sample(64, separate_observables=True)
    assert not d.any() and not o.any()


def test_the_python_writer_is_the_engines_on_programs_the_engine_takes():
    from stabilizer_qec import _core
    from stabilizer_qec._circuit import Circuit
    from stabilizer_qec.tqec._program import program_text

    graphs = [gallery.memory(Basis.Z), gallery.memory(Basis.X), gallery.cnot(Basis.Z), gallery.cnot(Basis.X), gallery.cnot(Basis.Z).rotate(Direction3D.Z)]
    graphs += [_row_of_patches(3, [1], 3), _row_of_patches(2, [1, 3], 5), _row_of_patches(2, [0], 1), _passthrough(BlockGraph, Position3D, "ZXZ")]
    graphs += [f.graph for f in gallery.cnot().fill_ports_for_minimal_simulation()]
    for g in graphs:
        for d, p in ((3, 0.0), (3, 0.001), (5, 1e-5)):
            c = tqec.compile_block_graph(g, d, p=p)
            engine = _core.surgery_program(d, p, c.tiles, c.steps, c.observables)
            assert str(Circuit(program_text(d, p, c.tiles, c.steps, c.observables))) == str(Circuit(engine)), (g.name, d, p)
    # Its refusals are the engine's.
    c = tqec.compile_block_graph(gallery.cnot(Basis.Z), 3)
    for steps in ([("rounds", 3)], [("prepare", [0, 0], "z")], [("split",)], [("prepare", [0], "z"), ("measure", [0], "z"), ("prepare", [0], "z")]):
        with pytest.raises(ValueError) as python:
            program_text(3, 0.0, c.tiles, steps, [])
        with pytest.raises(ValueError) as engine:
            _core.surgery_program(3, 0.0, c.tiles, steps, [])
        assert str(python.value) == str(engine.value)
    with pytest.raises(ValueError, match="which patch 0 still holds"):
        program_text(3, 0.0, [(0, 0), (0, 0)], [("prepare", [0], "z"), ("prepare", [1], "z")], [])


@needs_tqec
def test_two_cnots_agree_with_tqec():
    for basis in ("Z", "X"):
        o, t = _two_cnots(tqec, "ZX" + basis), _two_cnots(theirs, "ZX" + basis)
        assert structure(o) == structure(t)
        so, st = o.find_correlation_surfaces(), t.find_correlation_surfaces()
        assert spans(so) == spans(st)
        assert [s.external_stabilizer_on_graph(o) for s in so] == [s.external_stabilizer_on_graph(t) for s in st] == TQEC_TWO_CNOTS[basis][0]


# tqec 0.2.0's NoiseModel laid over the engine's noiseless circuits at d = 3, p = 0.001: the
# SHA-256 of the noisy circuit's text (as this package writes it) and the noise's targets.
TQEC_NOISY = {
    ("memory", "uniform_depolarizing"): ("e7e63978d3d29b2fb38397fac7bad2204ff0672b3d5a96f89e58b910e42b3d76", {"X_ERROR": 33, "DEPOLARIZE1": 440, "DEPOLARIZE2": 144, "M": 33}),
    ("memory", "si1000"): ("d29d40c5d6a2f1b90c64f2a2cd2cf0739ddff8c997c9fe552c29a29aa1193038", {"X_ERROR": 33, "DEPOLARIZE1": 574, "DEPOLARIZE2": 144, "M": 33}),
    ("cnot", "uniform_depolarizing"): ("0a60b1f7c7bb96371e505bde3216094c11343cfa6f20f702bbea6846b8a5e948", {"Z_ERROR": 30, "DEPOLARIZE1": 9413, "X_ERROR": 267, "DEPOLARIZE2": 1632, "M": 276, "MX": 21}),
    ("two_cnots", "uniform_depolarizing"): ("eb88f05905b34fe3e9bfda4762481ae7fabe3c4fbca1d9edc5fc6e47bf761a22", {"X_ERROR": 504, "DEPOLARIZE1": 16308, "DEPOLARIZE2": 2976, "M": 522, "Z_ERROR": 24, "MX": 6}),
}


def _noiseless(name: str):
    graph = {"memory": lambda: gallery.memory(Basis.Z), "cnot": lambda: gallery.cnot(Basis.X), "two_cnots": lambda: _two_cnots(tqec, "ZXZ")}[name]()
    return tqec.compile_block_graph(graph, 3).circuit


def _noise_targets(circuit) -> dict:
    counts: dict = {}
    for inst in circuit.flattened():
        if inst.name in ("DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Z_ERROR") or (inst.name in ("M", "MX") and inst.gate_args_copy()):
            counts[inst.name] = counts.get(inst.name, 0) + len(inst.targets_copy())
    return counts


@pytest.mark.parametrize("name,model", sorted(TQEC_NOISY))
def test_noise_models_are_tqecs(name, model):
    import hashlib

    noisy = getattr(tqec.NoiseModel, model)(0.001).noisy_circuit(_noiseless(name))
    digest, counts = TQEC_NOISY[(name, model)]
    assert hashlib.sha256(str(noisy).encode()).hexdigest() == digest
    assert _noise_targets(noisy) == counts
    # si1000 has rules for Z-basis resets and measurements only, as in tqec.
    with pytest.raises(ValueError, match="No noise"):
        tqec.NoiseModel.si1000(0.001).noisy_circuit(_noiseless("cnot"))


def test_noise_models_keep_tqecs_moments_and_repeat_blocks():
    c = sq.Circuit("TICK\nTICK\nR 0 1 2\nTICK\nREPEAT 2 {\n  TICK\n  H 0\n  CX 1 2\n  TICK\n  MPP X0*X1 Y2\n}\nCX rec[-1] 0\nTICK\nTICK\nMX 0 1\nDETECTOR rec[-1]\n")
    noisy = tqec.NoiseModel.uniform_depolarizing(0.01).noisy_circuit(c)
    assert str(noisy) == (
        "DEPOLARIZE1(0.01) 0 1 2\nTICK\nDEPOLARIZE1(0.01) 0 1 2\nTICK\nR 0 1 2\nX_ERROR(0.01) 0 1 2\n"
        "REPEAT 2 {\n    DEPOLARIZE1(0.01) 0 1 2\n    TICK\n    H 0\n    CX 1 2\n    DEPOLARIZE1(0.01) 0\n    DEPOLARIZE2(0.01) 1 2\n    TICK\n    MPP(0.01) X0*X1 Y2\n}\n"
        "CX rec[-1] 0\nDEPOLARIZE1(0.01) 0 1 2\nTICK\nDEPOLARIZE1(0.01) 0 1 2\nTICK\nMX(0.01) 0 1\nDETECTOR rec[-1]\nDEPOLARIZE1(0.01) 2"
    )
    with pytest.raises(ValueError, match="multiple times without a TICK"):
        tqec.NoiseModel.uniform_depolarizing(0.01).noisy_circuit(sq.Circuit("H 0\nX 0"))


@needs_tqec
def test_noise_models_against_tqec():
    stim = pytest.importorskip("stim")
    # (The two CNOTs' noisy circuit is pinned by its fixture; tqec takes a minute over it.)
    circuits = [_noiseless(name) for name in ("memory", "cnot")]
    circuits.append(sq.Circuit("TICK\nTICK\nR 0 1 2\nTICK\nREPEAT 2 {\n  TICK\n  H 0\n  CX 1 2\n  TICK\n  MPP X0*X1 Y2\n}\nCX rec[-1] 0\nTICK\nTICK\nMX 0 1\nDETECTOR rec[-1]\n"))
    for c in circuits:
        for model in ("uniform_depolarizing", "si1000"):
            try:
                expected = str(getattr(theirs.NoiseModel, model)(0.001).noisy_circuit(stim.Circuit(str(c))))
            except ValueError:
                with pytest.raises(ValueError):
                    getattr(tqec.NoiseModel, model)(0.001).noisy_circuit(c)
                continue
            assert str(stim.Circuit(str(getattr(tqec.NoiseModel, model)(0.001).noisy_circuit(c)))) == expected


def test_tqecs_call_shape():
    from stabilizer_qec.tqec import FIXED_BOUNDARY_CONVENTION, FIXED_BULK_CONVENTION, NoiseModel, TopologicalComputationGraph, compile_block_graph

    g = gallery.cnot(Basis.Z)
    computation = compile_block_graph(g)
    assert isinstance(computation, TopologicalComputationGraph) and computation.num_observables == 2 and computation.stabilizers == ["ZZII", "IZZZ"]
    noiseless = computation.generate_stim_circuit(k=1)
    assert str(noiseless) == str(compile_block_graph(g, 3).circuit)
    noise = NoiseModel.uniform_depolarizing(0.001)
    noisy = computation.generate_stim_circuit(k=1, noise_model=noise)
    assert str(noisy) == str(noise.noisy_circuit(noiseless))
    assert (noisy.num_qubits, noisy.num_detectors, noisy.num_observables) == (113, 256, 2)
    assert len(computation.generate_stim_circuit(2, noise).shortest_graphlike_error()) == 5
    # tqec's arguments for its detector search are taken; a negative radius drops the detectors.
    assert str(computation.generate_stim_circuit(1, None, 2, None, "db.pkl", True, False, True)) == str(noiseless)
    assert computation.generate_stim_circuit(1, manhattan_radius=-1).num_detectors == 0
    # The conventions: tqec's two, positional as tqec places it, by name; both are the engine's layout.
    for convention in (FIXED_BULK_CONVENTION, FIXED_BOUNDARY_CONVENTION, "fixed_boundary"):
        assert str(compile_block_graph(g, convention).generate_stim_circuit(1)) == str(noiseless)
        assert compile_block_graph(g, convention=convention).convention.name == str(convention)
    with pytest.raises(TQECError, match="not one of tqec's"):
        compile_block_graph(g, convention="fixed_corners")
    # block_temporal_height: tqec's rounds between the initialisation and measurement layers.
    m = gallery.memory(Basis.Z)
    assert compile_block_graph(m).generate_stim_circuit(1).num_detectors == 24  # 2k + 1 rounds, as tqec
    assert compile_block_graph(m, block_temporal_height=5).generate_stim_circuit(1).num_detectors == 8 * 7
    assert compile_block_graph(m, None, "auto", lambda k: 2 * k - 1).generate_stim_circuit(2).num_detectors == 120
    with pytest.raises(TQECError, match="not both"):
        compile_block_graph(m, block_temporal_height=3, rounds=4)
    # The observables argument, the engine's p, and k.
    assert compile_block_graph(g, observables=None).generate_stim_circuit(1).num_observables == 0
    with pytest.raises(TQECError, match="give the distance"):
        compile_block_graph(g, p=0.001)
    with pytest.raises(TQECError, match="k must be a positive integer"):
        computation.generate_stim_circuit(0)
    # CompiledBlockGraph writes the same circuits; BlockGraph.compile takes k and noise_model.
    compiled = compile_block_graph(g, 3, p=0.001)
    assert str(compiled.generate_stim_circuit()) == str(noiseless) and str(compiled.generate_stim_circuit(1, noise)) == str(noisy)
    assert str(g.compile(k=1)) == str(noiseless) and str(g.compile(3, noise_model=noise)) == str(noisy)
    with pytest.raises(TQECError, match="one of them"):
        g.compile(3, k=1)
    with pytest.raises(TQECError, match="not both"):
        g.compile(3, p=0.001, noise_model=noise)
    # The checks are made when the graph is compiled, as tqec makes them.
    with pytest.raises(TQECError, match="Hadamard"):
        compile_block_graph(gallery.cz("XI -> XZ"))
    with pytest.raises(TQECError, match="open ports"):
        compile_block_graph(gallery.cnot())


@needs_tqec
def test_tqecs_conventions_and_noise_model_objects_are_taken():
    from tqec.compile.convention import FIXED_BOUNDARY_CONVENTION as THEIR_FIXED_BOUNDARY

    g = gallery.memory(Basis.Z)
    assert tqec.compile_block_graph(g, THEIR_FIXED_BOUNDARY).convention == tqec.FIXED_BOUNDARY_CONVENTION
    pytest.importorskip("stim")
    # tqec's own NoiseModel works on Stim's circuits; it is given Stim's and read back.
    assert str(g.compile(3, noise_model=theirs.NoiseModel.uniform_depolarizing(0.001))) == str(g.compile(3, noise_model=tqec.NoiseModel.uniform_depolarizing(0.001)))


# ---------------------------------------------------------------------------------------------
# Small API: positions, LinearFunction, CZ flows, sq.tqec, find_correlation_surfaces' arguments
# ---------------------------------------------------------------------------------------------


def test_positions_and_linear_functions():
    from stabilizer_qec.tqec import FloatPosition3D, LinearFunction, Position2D

    assert Position3D(1, 2, 3).as_2d() == Position2D(1, 2) and Position2D(1, 2).to_3d(4) == Position3D(1, 2, 4)
    assert Position2D(0, 0).is_neighbour(Position2D(0, 1)) and not Position2D(0, 0).is_neighbour(Position2D(1, 1))
    a = FloatPosition3D(0.5, 1.0, -2.0).as_array()
    assert a.dtype == np.float32 and a.tolist() == [0.5, 1.0, -2.0]
    f = LinearFunction(2, -1)
    assert (f(1), f.integer_eval(3), repr(f), repr(LinearFunction(0, 3)), repr(LinearFunction(2, 0))) == (1, 5, "2*x + -1", "3", "2*x")
    assert f + 1 == LinearFunction(2, 0) and f - LinearFunction(1, 1) == LinearFunction(1, -2) and 3 * f == LinearFunction(6, -3)
    assert f.intersection(LinearFunction(1, 0)) == 1 and f.intersection(LinearFunction(2, 5)) is None
    assert LinearFunction(4, 2).exact_integer_div(2) == LinearFunction(2, 1)
    with pytest.raises(TQECError, match="not a multiple of the offset"):
        LinearFunction(4, 1).exact_integer_div(2)
    assert LinearFunction.unambiguous_max_on_positives([LinearFunction(1, 0), LinearFunction(2, 1)]) == LinearFunction(2, 1)
    with pytest.raises(TQECError, match="ambiguous"):
        LinearFunction.unambiguous_max_on_positives([LinearFunction(1, 3), LinearFunction(2, 1)])
    assert LinearFunction.safe_mul(LinearFunction(0, 2), f) == LinearFunction(4, -2)
    # The block_temporal_height tqec passes; the default is 2k - 1.
    m = gallery.memory(Basis.Z)
    assert str(tqec.compile_block_graph(m, block_temporal_height=f).generate_stim_circuit(2)) == str(tqec.compile_block_graph(m).generate_stim_circuit(2))


def test_cz_flows_in_stims_syntax():
    import re

    def ports(g):
        return {label: str(g[pos].kind) for label, pos in (("In_1", Position3D(0, 0, 0)), ("In_2", Position3D(1, -1, 1)), ("Out_1", Position3D(0, 0, 2)), ("Out_2", Position3D(1, 1, 1)))}

    dense = ports(gallery.cz("XI -> XZ"))
    assert dense == {"In_1": "XZX", "In_2": "XZZ", "Out_1": "XZX", "Out_2": "XZZ"}
    for same in ("X_ -> XZ", "X0 -> X0*Z1", "+XI -> +XZ", "-XI -> -XZ", "xi -> xz", "XZ -> XI"):
        assert ports(gallery.cz(same)) == dense
    assert ports(gallery.cz("X1 -> Z0*X1")) == ports(gallery.cz("_X -> ZX"))
    assert ports(gallery.cz("1 -> 1")) == ports(gallery.cz("ZZ -> ZZ"))
    # Errors name the flow as Stim writes it, as tqec's do.
    for flow, text in (("XI -> XX", "X_ -> XX"), ("X0 -> X0", "X -> X"), ("-XI -> XZ", "-X_ -> XZ"), ("XI -> -XZ", "X_ -> -XZ"), ("X0 -> X0*Z1*Z2", "X -> XZZ")):
        with pytest.raises(TQECError, match=f"^{re.escape(text)} is not a valid flow for the CZ gate.$"):
            gallery.cz(flow)
    with pytest.raises(TQECError, match="qubits are 0 and 1"):
        gallery.cz("Z2 -> Z2")
    with pytest.raises(TQECError, match="Port 0 fails to support both X and Z observable"):
        gallery.cz(["X0 -> X0*Z1", "Z0 -> Z0"])
    # The two sparse flows tqec joins into conflicting ports are compatible on the gate's qubits.
    assert ports(gallery.cz(["X0 -> X0*Z1", "Z1 -> Z1"])) == dense


@needs_tqec
def test_cz_flows_agree_with_tqec():
    def outcome(module, flows):
        try:
            return structure(module.cz(flows))
        except Exception as e:  # noqa: BLE001
            return f"{type(e).__name__}: {e}"

    # tqec reads a sparse flow whose sides have different lengths into the wrong ports
    # ("X0 -> X0*Z1"), and fails on flows beyond the two qubits; the others are the same.
    for flows in ("XI -> XZ", "X_ -> XZ", "-XI -> -XZ", "X1 -> Z0*X1", "Z1 -> Z1", "1 -> 1", "XZ -> XI", "ZZ -> ZZ", ["ZI -> ZI", "IX -> ZX"], "XI -> XX", "X0 -> X0", "-XI -> XZ", "X0 -> X0*Z1*Z2", "XX -> YY", ["X_ -> XZ", "_Z -> _Z"]):
        assert outcome(gallery, flows) == outcome(their_gallery, flows), flows


def test_tqec_is_an_attribute_before_it_is_imported():
    import os
    import subprocess
    import sys

    code = "import stabilizer_qec as sq, sys; assert 'stabilizer_qec.tqec' not in sys.modules; print(sq.tqec.compile_block_graph(sq.tqec.gallery.memory()).generate_stim_circuit(1).num_detectors)"
    env = {**os.environ, "PYTHONPATH": str(pathlib.Path(sq.__file__).parent.parent)}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True)
    assert out.stdout.strip() == "24"
