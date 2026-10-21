"""The block-graph port (`stabilizer_qec.tqec`): tqec's gallery, correlation surfaces, port
filling, rotations, JSON and Collada files, and the compile to this package's surgery engine.
The differential tests run when tqec is installed."""

from __future__ import annotations

import inspect
import json
import pathlib
import re

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


def _tqec_fixture(case_id: str) -> str:
    """tqec's own circuit for a case of tests/tqec_port/fixtures (see tools/tqec_fixtures.py)."""
    import gzip

    return gzip.decompress((pathlib.Path(__file__).parent / "tqec_port" / "fixtures" / f"{case_id}.stim.gz").read_bytes()).decode()


# tqec 0.2.0's stabilizers for the two CNOTs (cnot().compose(cnot(), "Out_Control",
# "In_Control"), ports filled ZXZ or ZXX); its circuits at k = 1 are fixtures of tests/tqec_port.
TQEC_TWO_CNOTS = {"Z": ["ZIZI", "IZIZ"], "X": ["XXXX", "XIXI"]}


@pytest.mark.parametrize("basis", ["Z", "X"])
def test_two_cnots_in_a_row_reuse_the_ancillas_tile(basis):
    g = _two_cnots(tqec, "ZX" + basis)
    compiled = tqec.compile_block_graph(g, 3, p=0.001)
    # The ancilla's tile holds two patches, one per CNOT, the first read out before the second
    # is prepared.
    assert compiled.tiles == [(0, 0), (1, 0), (1, 0), (1, 1)] and compiled.num_merges == 4
    assert compiled.stabilizers == TQEC_TWO_CNOTS[basis]
    c = compiled.circuit
    # The engine's layout spends more qubits than tqec's (113, not 81); the detectors are tqec's.
    assert (c.num_qubits, c.num_detectors, c.num_observables) == (113, 464, 2)
    assert len(c.shortest_graphlike_error()) == 3
    d, o = g.compile(3).compile_detector_sampler(seed=1).sample(128, separate_observables=True)
    assert not d.any() and not o.any()
    assert len(tqec.compile_block_graph(g, 5, p=0.001).circuit.shortest_graphlike_error()) == 5
    # tqec's way (its plaquettes, 81 qubits): its own circuit, byte for byte.
    noiseless = tqec.compile_block_graph(g).generate_stim_circuit(k=1)
    assert str(noiseless) == _tqec_fixture(f"two_cnots_ZX{basis}_fixed_bulk_k1")
    noisy = tqec.NoiseModel.uniform_depolarizing(0.001).noisy_circuit(noiseless)
    assert (noisy.num_qubits, noisy.num_detectors, noisy.num_observables) == (81, 464, 2)


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
        assert [s.external_stabilizer_on_graph(o) for s in so] == [s.external_stabilizer_on_graph(t) for s in st] == TQEC_TWO_CNOTS[basis]


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
    from stabilizer_qec.tqec import FIXED_BOUNDARY_CONVENTION, FIXED_BULK_CONVENTION, LinearFunction, NoiseModel, TopologicalComputationGraph, compile_block_graph

    g = gallery.cnot(Basis.Z)
    computation = compile_block_graph(g)
    assert isinstance(computation, TopologicalComputationGraph) and computation.num_observables == 2 and computation.stabilizers == ["ZZII", "IZZZ"]
    # tqec's circuit, byte for byte (tests/tqec_port has the rest of the gallery).
    noise = NoiseModel.uniform_depolarizing(0.001)
    noisy = computation.generate_stim_circuit(k=1, noise_model=noise)
    assert str(noisy) == str(noise.noisy_circuit(sq.Circuit(_tqec_fixture("cnot_Z_fixed_bulk_k1"))))
    assert (noisy.num_qubits, noisy.num_detectors, noisy.num_observables) == (81, 256, 2)
    # tqec's arguments for its detector search, positional as tqec places them: the database is
    # in memory (a path is not used), and a radius that is not positive finds no detectors.
    m = gallery.memory(Basis.Z)
    memory = compile_block_graph(m)
    noiseless = memory.generate_stim_circuit(1)
    assert str(memory.generate_stim_circuit(1, None, 2, None, "db.pkl", True, False, True)) == str(noiseless) == _tqec_fixture("memory_Z_fixed_bulk_k1")
    assert memory.generate_stim_circuit(1, manhattan_radius=0).num_detectors == 0
    assert len(memory.generate_stim_circuit(2, noise).shortest_graphlike_error()) == 5
    with pytest.raises(TQECError, match="DetectorDatabase"):
        memory.generate_stim_circuit(1, detector_database={})
    # The conventions: tqec's two, positional as tqec places it, or by name.
    for convention in (FIXED_BOUNDARY_CONVENTION, "fixed_boundary"):
        assert str(compile_block_graph(m, convention).generate_stim_circuit(1)) == _tqec_fixture("memory_Z_fixed_boundary_k1")
        assert compile_block_graph(m, convention=convention).convention.name == str(convention)
    assert compile_block_graph(m, FIXED_BULK_CONVENTION).convention == FIXED_BULK_CONVENTION
    with pytest.raises(TQECError, match="not one of tqec's"):
        compile_block_graph(g, convention="fixed_corners")
    # block_temporal_height: tqec's LinearFunction of k, the rounds between the initialisation and
    # measurement layers (2k - 1 by default); tqec refuses a constant one.
    assert noiseless.num_detectors == 24
    assert str(compile_block_graph(m, None, "auto", LinearFunction(2, 1)).generate_stim_circuit(1)) == _tqec_fixture("memory_Z_fixed_bulk_k1_height2k1")
    with pytest.raises(TQECError, match="non-scalable dimension"):
        compile_block_graph(m, block_temporal_height=5)
    with pytest.raises(TQECError, match="must be a LinearFunction"):
        compile_block_graph(m, block_temporal_height=lambda k: 2 * k - 1)
    # A number or a callable, which 2.1 took on this route (its engine ran height + 2 rounds in
    # every time slice): the error (tqec's for the constant LinearFunction a number is), then a
    # hint. A LinearFunction and anything else get no hint. The engine still takes them, at a
    # distance, and still runs height + 2 rounds.
    def refusal(height) -> str:
        with pytest.raises(TQECError) as info:
            compile_block_graph(m, block_temporal_height=height)
        return re.sub(r"<[\w.]+\.(\w+) object at 0x[0-9a-f]+>", r"<\1 object>", str(info.value))

    manifest = json.loads((pathlib.Path(__file__).parent / "tqec_port" / "fixtures" / "manifest.json").read_text())
    constant = re.sub(r"<[\w.]+\.(\w+) object at 0x[0-9a-f]+>", r"<\1 object>", next(c for c in manifest["cases"] if c["id"] == "memory_Z_fixed_bulk_k1_height1")["error"]["message"])
    assert constant == "Cannot add the block as a cube. The provided block(<Block object>) has at least one non-scalable dimension."
    hint = "2.1 took a number or a callable of k here, and its engine ran height + 2 rounds in every time slice (height 5: 7 rounds); the engine still takes them, at a distance: compile_block_graph(graph, distance, block_temporal_height=...)."
    assert refusal(LinearFunction(0, 5)) == constant
    for height in (5, 2.5, np.int64(5)):
        assert refusal(height) == f"{constant} block_temporal_height={height!r} is a constant height, and tqec's is a LinearFunction of k (slope * k + offset, the rounds between a block's initialisation and measurement layers; LinearFunction(2, -1) by default). {hint}"
    def height(k):
        return 2 * k - 1

    assert refusal(height) == f"block_temporal_height must be a LinearFunction (slope * k + offset; tqec's default is LinearFunction(2, -1)), not {height!r}. {hint}"
    assert refusal("5") == "block_temporal_height must be a LinearFunction (slope * k + offset; tqec's default is LinearFunction(2, -1)), not '5'"
    assert str(compile_block_graph(m, 3, block_temporal_height=5).circuit) == str(compile_block_graph(m, 3, rounds=7).circuit)
    assert str(compile_block_graph(m, 5, block_temporal_height=height).circuit) == str(compile_block_graph(m, 5, rounds=5).circuit)
    with pytest.raises(TQECError, match="not both"):
        compile_block_graph(m, block_temporal_height=3, rounds=4)
    with pytest.raises(TQECError, match="give the distance too"):
        compile_block_graph(m, rounds=4)
    # The observables argument, the engine's p, and k.
    assert compile_block_graph(m, observables=None).generate_stim_circuit(1).num_observables == 0
    with pytest.raises(TQECError, match="'auto', None or a list"):
        compile_block_graph(m, observables="all")
    with pytest.raises(TQECError, match="give the distance"):
        compile_block_graph(g, p=0.001)
    for k in (0, 1.5, True):
        with pytest.raises(TQECError, match="k must be a positive integer"):
            computation.generate_stim_circuit(k)
    assert str(memory.generate_stim_circuit(np.int64(1))) == str(noiseless)
    # CompiledBlockGraph (the engine's) writes tqec's circuits too; BlockGraph.compile is the
    # engine's, taking k and noise_model.
    compiled = compile_block_graph(m, 3, p=0.001)
    assert str(compiled.generate_stim_circuit()) == str(noiseless)
    engine = compile_block_graph(g, 3).circuit
    assert str(g.compile(k=1)) == str(engine) and str(g.compile(3, noise_model=noise)) == str(noise.noisy_circuit(engine))
    with pytest.raises(TQECError, match="one of them"):
        g.compile(3, k=1)
    with pytest.raises(TQECError, match="not both"):
        g.compile(3, p=0.001, noise_model=noise)
    # tqec's checks are made when the graph is compiled, with tqec's errors; the engine's when it
    # compiles.
    with pytest.raises(TQECError, match="open ports"):
        compile_block_graph(gallery.cnot())
    y = BlockGraph("y")
    y.add_cube(Position3D(0, 0, 0), "Y")
    y.add_cube(Position3D(0, 0, 1), "ZXZ")
    y.add_pipe(Position3D(0, 0, 0), Position3D(0, 0, 1))
    with pytest.raises(NotImplementedError, match="^Y cube is not implemented.$"):
        compile_block_graph(y)
    cz = compile_block_graph(gallery.cz("ZZ -> ZZ"))
    with pytest.raises(TQECError, match="Hadamard"):
        cz.compile(3)


def test_compiled_block_graph_writes_tqecs_layout():
    """A ``CompiledBlockGraph`` (the engine's program) has two layouts: ``circuit``, the
    engine's, and ``generate_stim_circuit``, tqec's, with ``TopologicalComputationGraph``'s
    parameters (tqec's), refusing a program whose rounds tqec's layout cannot honour."""
    from stabilizer_qec.tqec import LinearFunction, TopologicalComputationGraph, compile_block_graph
    from stabilizer_qec.tqec.compile import CompiledBlockGraph

    ours = inspect.signature(CompiledBlockGraph.generate_stim_circuit).parameters
    tqecs = inspect.signature(TopologicalComputationGraph.generate_stim_circuit).parameters
    assert list(ours) == list(tqecs) == ["self", "k", "noise_model", "manhattan_radius", "detector_database", "database_path", "do_not_use_database", "only_use_database", "reschedule_measurements"]
    assert [p.default for name, p in ours.items() if name != "k"] == [p.default for name, p in tqecs.items() if name != "k"]
    assert ours["k"].default is None  # the program's scale
    # The engine's layout and tqec's: the CNOT at distance 3 on 113 qubits and on 81.
    cnot = gallery.cnot(Basis.Z)
    program = compile_block_graph(cnot, 3)
    assert program.circuit.num_qubits == 113
    assert str(program.generate_stim_circuit()) == _tqec_fixture("cnot_Z_fixed_bulk_k1")
    assert sq.Circuit(_tqec_fixture("cnot_Z_fixed_bulk_k1")).num_qubits == 81
    # tqec's parameters, positional as tqec places them.
    m = gallery.memory(Basis.Z)
    noiseless = _tqec_fixture("memory_Z_fixed_bulk_k1")
    compiled = compile_block_graph(m, 3)
    assert str(compiled.circuit) != noiseless
    assert str(compiled.generate_stim_circuit()) == str(compiled.generate_stim_circuit(1, None, 2, None, "db.pkl", True, False, True)) == noiseless
    assert compiled.generate_stim_circuit(manhattan_radius=0).num_detectors == 0
    assert str(compiled.generate_stim_circuit(1, reschedule_measurements=False)) == str(compile_block_graph(m).generate_stim_circuit(1, reschedule_measurements=False))
    noise = tqec.NoiseModel.uniform_depolarizing(0.001)
    assert str(compiled.generate_stim_circuit(noise_model=noise)) == str(noise.noisy_circuit(sq.Circuit(noiseless)))
    with pytest.raises(TypeError):
        compiled.generate_stim_circuit(1, rounds=5)
    with pytest.raises(TQECError, match="DetectorDatabase"):
        compiled.generate_stim_circuit(1, detector_database={})
    # Rounds tqec's layout cannot honour are refused, pointing at the engine's circuit...
    message = "this program was compiled with rounds=5, so at k = 1 each time slice runs 5 rounds; generate_stim_circuit writes tqec's layout, which runs 2k + 1 = 3 rounds in every time slice, merges included, and cannot honour that. The program's circuit, in the engine's layout, is .circuit (at its distance; compile_block_graph(graph, distance, ...) for another)"
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        compile_block_graph(m, 3, rounds=5).generate_stim_circuit()
    with pytest.raises(ValueError, match=r"^this program was compiled with rounds=3, so at k = 2 each time slice runs 3 rounds; .* 2k \+ 1 = 5 rounds"):
        compile_block_graph(m, 3, rounds=3).generate_stim_circuit(2)
    with pytest.raises(ValueError, match=r"^this program was compiled with merged=2, so at k = 1 each time slice runs 3 rounds and each merge 2; "):
        compile_block_graph(cnot, 3, merged=2).generate_stim_circuit()
    with pytest.raises(ValueError, match=r"^this program was compiled with rounds=3, merged=2, so at k = 1 each time slice runs 3 rounds and each merge 2; "):
        compile_block_graph(cnot, 3, rounds=3, merged=2).generate_stim_circuit()
    with pytest.raises(ValueError, match=r"^this program was compiled with block_temporal_height=5, so at k = 1 each time slice runs 7 rounds; "):
        compile_block_graph(m, 3, block_temporal_height=5).generate_stim_circuit()
    with pytest.raises(ValueError, match=r"block_temporal_height=<function .*, so at k = 2 each time slice runs 3 rounds; "):
        compile_block_graph(m, 3, block_temporal_height=lambda k: 1).generate_stim_circuit(2)
    # ...and rounds it can honour are written: tqec's own height, or tqec's rounds at k however
    # they were given (merged is not used by a graph without merges).
    assert str(compile_block_graph(m, 3, rounds=3).generate_stim_circuit()) == noiseless
    assert str(compile_block_graph(m, 3, merged=2).generate_stim_circuit()) == noiseless
    assert str(compile_block_graph(m, 3, block_temporal_height=1).generate_stim_circuit()) == noiseless
    assert str(compile_block_graph(m, 3, block_temporal_height=lambda k: 2 * k - 1).generate_stim_circuit()) == noiseless
    assert str(compile_block_graph(m, 3, block_temporal_height=LinearFunction(2, 1)).generate_stim_circuit()) == _tqec_fixture("memory_Z_fixed_bulk_k1_height2k1")
    with pytest.raises(TQECError, match="non-scalable dimension"):  # tqec's own height, which tqec refuses
        compile_block_graph(m, 3, block_temporal_height=LinearFunction(0, 1)).generate_stim_circuit()


@needs_tqec
def test_generate_stim_circuit_takes_tqecs_parameters():
    from tqec.compile.graph import TopologicalComputationGraph as Theirs

    from stabilizer_qec.tqec.compile import CompiledBlockGraph

    theirs_parameters = inspect.signature(Theirs.generate_stim_circuit).parameters
    for cls in (tqec.TopologicalComputationGraph, CompiledBlockGraph):
        parameters = inspect.signature(cls.generate_stim_circuit).parameters
        assert list(parameters) == list(theirs_parameters)
        # tqec's defaults, database_path's (tqec's database file) included, apart from the
        # engine's program's k (its own scale; tqec's k has no default).
        assert [p.default for name, p in parameters.items() if name != "k"] == [p.default for name, p in theirs_parameters.items() if name != "k"]
        assert parameters["k"].default is (None if cls is CompiledBlockGraph else inspect.Parameter.empty)
        assert all(p.kind == p.POSITIONAL_OR_KEYWORD for p in parameters.values())


@needs_tqec
def test_database_path_defaults_to_tqecs(monkeypatch: pytest.MonkeyPatch):
    """Every database_path this package takes for one of tqec's signatures defaults to tqec's
    DEFAULT_DETECTOR_DATABASE_PATH, computed as tqec computes it (and never read)."""
    import tqec.utils.paths as their_paths
    from tqec.compile.graph import TopologicalComputationGraph as Theirs
    from tqec.compile.tree.tree import LayerTree as TheirTree
    from tqec.simulation import generation as their_generation, simulation as their_simulation

    from stabilizer_qec.tqec._compile.graph import TopologicalComputationGraph as Port
    from stabilizer_qec.tqec._compile.tree.tree import LayerTree
    from stabilizer_qec.tqec._compile.utils import paths
    from stabilizer_qec.tqec.compile import CompiledBlockGraph
    from stabilizer_qec.tqec.simulation import generation, simulation

    assert paths.USER_DATA_PATH == their_paths.USER_DATA_PATH
    assert paths.DEFAULT_DETECTOR_DATABASE_PATH == their_paths.DEFAULT_DETECTOR_DATABASE_PATH
    pairs = [
        (tqec.TopologicalComputationGraph.generate_stim_circuit, Theirs.generate_stim_circuit),
        (CompiledBlockGraph.generate_stim_circuit, Theirs.generate_stim_circuit),
        (Port.generate_stim_circuit, Theirs.generate_stim_circuit),
        (LayerTree.generate_circuit, TheirTree.generate_circuit),
        (LayerTree._annotate_detectors, TheirTree._annotate_detectors),
        (generation.generate_stim_circuits_with_detectors, their_generation.generate_stim_circuits_with_detectors),
        (generation.generate_sinter_tasks, their_generation.generate_sinter_tasks),
        (simulation.start_simulation_using_sinter, their_simulation.start_simulation_using_sinter),
    ]
    for ours, theirs_function in pairs:
        default = inspect.signature(ours).parameters["database_path"].default
        assert default == inspect.signature(theirs_function).parameters["database_path"].default == their_paths.DEFAULT_DETECTOR_DATABASE_PATH, ours
    # tqec's environment variable, which tqec reads when it is imported.
    monkeypatch.setenv("TQEC_DETECTOR_DATABASE_PATH", "elsewhere/db.pkl")
    assert paths._get_database_path() == their_paths._get_database_path() == pathlib.Path("elsewhere/db.pkl")


def test_default_database_path_without_platformdirs(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path):
    """tqec's database path is platformdirs' user data directory for "TQEC"; without platformdirs
    (which tqec requires) it is platformdirs' rule for the platform. The port never reads or
    writes the path."""
    import os
    import sys

    from stabilizer_qec.tqec._compile.utils import paths

    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.delenv("TQEC_DETECTOR_DATABASE_PATH", raising=False)
    try:
        import platformdirs
    except ImportError:  # pragma: no cover
        platformdirs = None
    expected = None if platformdirs is None else platformdirs.user_data_path(appname="TQEC")
    assert paths._user_data_path("TQEC") == expected or expected is None
    assert paths._get_database_path() == paths.USER_DATA_PATH / "detector_database.pkl"
    monkeypatch.setitem(sys.modules, "platformdirs", None)  # import platformdirs now fails
    fallback = paths._user_data_path("TQEC")
    if expected is not None:
        assert fallback == expected
    if sys.platform == "darwin":
        assert fallback == pathlib.Path.home() / "Library" / "Application Support" / "TQEC"
    if sys.platform != "win32":
        # $XDG_DATA_HOME, when absolute, as platformdirs takes it on macOS and other Unix
        # systems (Windows has no XDG directories); a relative one is ignored.
        monkeypatch.setenv("XDG_DATA_HOME", " /xdg/data ")
        assert paths._user_data_path("TQEC") == pathlib.Path("/xdg/data/TQEC")
        monkeypatch.setenv("XDG_DATA_HOME", "relative/data")
        assert paths._user_data_path("TQEC") == fallback
    else:  # pragma: no cover
        monkeypatch.setenv("LOCALAPPDATA", "C:\\Users\\u\\AppData\\Local")
        assert paths._user_data_path("TQEC") == pathlib.Path("C:\\Users\\u\\AppData\\Local", "TQEC", "TQEC")
    monkeypatch.setenv("TQEC_DETECTOR_DATABASE_PATH", os.fspath(tmp_path / "db.pkl"))
    assert paths._get_database_path() == tmp_path / "db.pkl"
    # The path is taken for tqec's signature only: nothing is read or written there.
    memory = tqec.compile_block_graph(gallery.memory(Basis.Z))
    assert str(memory.generate_stim_circuit(1, database_path=tmp_path / "db.pkl")) == _tqec_fixture("memory_Z_fixed_bulk_k1")
    assert list(tmp_path.iterdir()) == []


@needs_tqec
def test_tqecs_conventions_and_noise_model_objects_are_taken():
    from tqec.compile.convention import FIXED_BOUNDARY_CONVENTION as THEIR_FIXED_BOUNDARY

    g = gallery.memory(Basis.Z)
    assert tqec.compile_block_graph(g, THEIR_FIXED_BOUNDARY).convention == tqec.FIXED_BOUNDARY_CONVENTION
    pytest.importorskip("stim")
    # tqec's own NoiseModel works on Stim's circuits; it is given Stim's and read back.
    assert str(g.compile(3, noise_model=theirs.NoiseModel.uniform_depolarizing(0.001))) == str(g.compile(3, noise_model=tqec.NoiseModel.uniform_depolarizing(0.001)))
    computation = tqec.compile_block_graph(g, block_temporal_height=theirs.utils.scale.LinearFunction(2, 1))
    assert str(computation.generate_stim_circuit(1)) == _tqec_fixture("memory_Z_fixed_bulk_k1_height2k1")
    assert str(computation.generate_stim_circuit(1, theirs.NoiseModel.uniform_depolarizing(0.001))) == str(computation.generate_stim_circuit(1, tqec.NoiseModel.uniform_depolarizing(0.001)))


# ---------------------------------------------------------------------------------------------
# tqec.simulation on this package's sinter
# ---------------------------------------------------------------------------------------------

# tqec 0.2.0's heuristic_custom_error_key for the gallery's graphs (the same here). Not the three
# CNOTs: two combinations tie there, and tqec breaks the tie by the order of a set of edges,
# which follows the string hashes of the process.
TQEC_ERROR_KEYS = {("cnot", None): "obs_mistake_mask=__E_", ("cnot", "Z"): "obs_mistake_mask=EE", ("cnot", "X"): "obs_mistake_mask=EE", ("steane_encoding", None): "obs_mistake_mask=_E___EE", ("move_rotation", None): "obs_mistake_mask=E_"}


def _split_rows():
    return [
        dict(strong_id="a", decoder="pymatching", json_metadata={"d": 3, "r": 3, "p": 0.002}, shots=1000, errors=30, discards=0, seconds=1.0, custom_counts={"obs_mistake_mask=E_": 10, "obs_mistake_mask=EE": 5, "obs_mistake_mask=_E": 15, "detection_events": 7}),
        dict(strong_id="a", decoder="pymatching", json_metadata={"d": 3, "r": 3, "p": 0.002}, shots=500, errors=12, discards=1, seconds=0.5, custom_counts={"obs_mistake_mask=E_": 4, "obs_mistake_mask=_E": 8}),
        dict(strong_id="b", decoder="pymatching", json_metadata={"d": 5, "r": 5, "p": 0.002}, shots=2000, errors=3, discards=0, seconds=2.0, custom_counts={"obs_mistake_mask=EE": 3}),
    ]


def _our_stats():
    import collections

    return [sq.sinter.TaskStats(**{**r, "custom_counts": collections.Counter(r["custom_counts"])}) for r in _split_rows()]


def _flat(split) -> list:
    return [[(s.strong_id, s.shots, s.errors, s.discards, s.seconds, dict(s.custom_counts), s.json_metadata) for s in stats] for stats in split]


def test_simulation_splits_the_observables_as_tqec_does():
    from stabilizer_qec.tqec.simulation import heuristic_custom_error_key, split_stats_for_observables
    from stabilizer_qec.tqec.simulation.split import split_counts_for_observables

    for (name, basis), key in TQEC_ERROR_KEYS.items():
        assert heuristic_custom_error_key(ours(name, basis).find_correlation_surfaces()) == key
    assert split_counts_for_observables({"obs_mistake_mask=E_E": 3, "obs_mistake_mask=_EE": 2, "detection_events": 9}, 3) == [3, 2, 5]
    assert _flat(split_stats_for_observables(_our_stats(), 2)) == [
        [("a", 1500, 19, 1, 1.5, {"detection_events": 7}, {"d": 3, "r": 3, "p": 0.002}), ("b", 2000, 3, 0, 2.0, {}, {"d": 5, "r": 5, "p": 0.002})],
        [("a", 1500, 28, 1, 1.5, {"detection_events": 7}, {"d": 3, "r": 3, "p": 0.002}), ("b", 2000, 3, 0, 2.0, {}, {"d": 5, "r": 5, "p": 0.002})],
    ]


def test_simulation_generates_tqecs_tasks_and_collects_them():
    from stabilizer_qec.tqec.simulation import start_simulation_using_sinter
    from stabilizer_qec.tqec.simulation.generation import generate_sinter_tasks, generate_stim_circuits_with_detectors
    from stabilizer_qec.tqec.simulation.threshold import get_logical_error_rate_per_shot

    computation = tqec.compile_block_graph(gallery.memory(Basis.Z))
    tasks = list(generate_sinter_tasks(computation, [1, 2], [0.001, 0.002], tqec.NoiseModel.uniform_depolarizing, 2))
    assert [t.json_metadata for t in tasks] == [{"d": d, "r": d, "p": p} for d in (3, 5) for p in (0.001, 0.002)]
    assert [str(t.circuit) for t in tasks[:2]] == [str(computation.generate_stim_circuit(1, tqec.NoiseModel.uniform_depolarizing(p))) for p in (0.001, 0.002)]
    assert [(k, p) for _, k, p in generate_stim_circuits_with_detectors(computation, [1], [0.001], tqec.NoiseModel.uniform_depolarizing, -1)] == [(1, 0.001)]
    # One list of statistics per observable, the shots shared, the errors each observable's.
    g = gallery.cnot(Basis.Z)
    split = start_simulation_using_sinter(g, [1], [0.002], tqec.NoiseModel.uniform_depolarizing, 2, num_workers=1, max_shots=1000)
    assert len(split) == 2 and [len(s) for s in split] == [1, 1]
    (a,), (b,) = split
    assert a.json_metadata == b.json_metadata == {"d": 3, "r": 3, "p": 0.002} and a.shots == b.shots >= 1000
    assert not any(k.startswith("obs_mistake_mask=") for k in a.custom_counts)
    (raw,) = start_simulation_using_sinter(g, [1], [0.002], tqec.NoiseModel.uniform_depolarizing, 2, num_workers=1, max_shots=1000, split_observable_stats=False)
    assert len(raw) == 1 and all(k.startswith("obs_mistake_mask=") and len(k) == len("obs_mistake_mask=") + 2 for k in raw[0].custom_counts)
    fit = get_logical_error_rate_per_shot(raw[0])
    assert fit.low <= fit.best <= fit.high
    nothing = get_logical_error_rate_per_shot(raw[0].with_edits(errors=0))
    assert np.isnan(nothing.best) and nothing.low == 0


def test_binary_search_threshold_brackets_the_memory_threshold():
    from stabilizer_qec.tqec.simulation.threshold import binary_search_threshold

    m = gallery.memory(Basis.Z)
    threshold, fits = binary_search_threshold(m, m.find_correlation_surfaces()[0], tqec.NoiseModel.uniform_depolarizing, minp=0.001, maxp=0.03, atol=0.005, rtol=0.2, num_workers=1, max_shots=4000, max_errors=200)
    assert 0.001 < threshold < 0.03
    assert sorted(fits) == [1, 2] and len(fits[1]) == len(fits[2]) >= 1
    assert [p for p, _ in fits[1]] == [p for p, _ in fits[2]] and fits[1][0][0] == pytest.approx(0.0155)


def test_sinter_stats_csv_files(tmp_path: pathlib.Path):
    from stabilizer_qec.tqec.simulation.io_utils.csv_files import write_sinter_stats_to_csv

    stats = _our_stats()
    path = tmp_path / "stats.csv"
    write_sinter_stats_to_csv(path, stats[:1])
    with pytest.raises(TQECError, match="already exists"):
        write_sinter_stats_to_csv(path, stats)
    write_sinter_stats_to_csv(path, stats[1:], if_file_exists="merge")
    merged = sorted(sq.sinter.read_stats_from_csv_files(path), key=lambda s: s.strong_id)
    assert [(s.strong_id, s.shots, s.errors) for s in merged] == [("a", 1500, 42), ("b", 2000, 3)]
    write_sinter_stats_to_csv(path, stats[2:], if_file_exists="overwrite")
    assert [s.strong_id for s in sq.sinter.read_stats_from_csv_files(path)] == ["b"]


def test_simulation_plots():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from stabilizer_qec.tqec.simulation import plot_observable_as_inset
    from stabilizer_qec.tqec.simulation.plotting.inset import plot_threshold_as_inset

    g = gallery.cnot(Basis.Z)
    fig, ax = plt.subplots()
    plot_observable_as_inset(ax, g.to_zx_graph(), g.find_correlation_surfaces()[1])
    plot_threshold_as_inset(ax, _our_stats(), zoom_bounds=(1e-3, 1e-1, 1e-2, 1e-3), threshold=5e-3)
    assert len(ax.child_axes) == 2
    fig2, ax2 = g.to_zx_graph().draw(title="CNOT")
    assert ax2.get_title() == "CNOT"
    plt.close(fig)
    plt.close(fig2)


@needs_tqec
def test_simulation_agrees_with_tqec():
    import collections

    import sinter as real_sinter
    from tqec.simulation import heuristic_custom_error_key as their_key
    from tqec.simulation import split_stats_for_observables as their_split

    from stabilizer_qec.tqec.simulation import heuristic_custom_error_key, split_stats_for_observables

    for name, basis in TQEC_ERROR_KEYS:
        assert heuristic_custom_error_key(ours(name, basis).find_correlation_surfaces()) == their_key(tqecs(name, basis).find_correlation_surfaces())
    mine = split_stats_for_observables(_our_stats(), 2)
    them = their_split([real_sinter.TaskStats(**{**r, "custom_counts": collections.Counter(r["custom_counts"])}) for r in _split_rows()], 2)
    assert _flat(mine) == _flat(them)
    assert [s.to_csv_line() for s in mine[1]] == [s.to_csv_line() for s in them[1]]


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


def test_find_correlation_surfaces_takes_tqecs_arguments():
    g = gallery.cnot()
    zx = g.to_zx_graph()
    assert spans(tqec.find_correlation_surfaces(zx, parallel=True)) == spans(tqec.find_correlation_surfaces(zx)) == spans(g.find_correlation_surfaces())
    with pytest.raises(NotImplementedError, match="reserved"):
        tqec.find_correlation_surfaces(zx, vertex_ordering=[{0}])


def test_module_docstring_examples():
    import doctest

    assert doctest.testmod(tqec, verbose=False).failed == 0


# ---------------------------------------------------------------------------------------------
# PyZX interop (PyZX optional) and block synthesis
# ---------------------------------------------------------------------------------------------


def test_block_synthesis_without_pyzx():
    from stabilizer_qec.tqec.interop.pyzx import block_synthesis

    for name, basis in (("cnot", "Z"), ("cnot", "X"), ("memory", "Z"), ("move_rotation", "X")):
        g = ours(name, basis)
        assert structure(g.to_zx_graph().to_block_graph()) == structure(block_synthesis(g.to_zx_graph())) == structure(g)
    # Open graphs come back with the ports named by vertex, as tqec names them.
    assert structure(gallery.cnot().to_zx_graph().to_block_graph())[2] == {"P_0": (0, 0, 0), "P_3": (0, 0, 3), "P_9": (1, 1, 3), "P_6": (1, 1, 0)}
    with pytest.raises(ValueError, match="requires specifying positions"):
        block_synthesis(object())


def test_pyzx_graphs_in_and_out():
    zx = pytest.importorskip("pyzx")
    from fractions import Fraction

    from stabilizer_qec.tqec.interop.pyzx import PositionedZX, block_synthesis, is_boundary, zx_to_pauli

    g = gallery.cnot(Basis.Z)
    z = g.to_zx_graph()
    pg = z.g
    assert list(pg.vertices()) == z.vertices() and sorted(pg.edges()) == sorted(z.edges())
    assert [str(zx_to_pauli(pg, v)) for v in pg.vertices()] == [str(z.kind(v)) for v in z.vertices()]
    assert spans(tqec.find_correlation_surfaces(pg)) == spans(g.find_correlation_surfaces())
    for s in g.find_correlation_surfaces():
        web = s.to_pauli_web(z)
        assert web.half_edges() == s.half_edge_paulis() and spans([tqec.CorrelationSurface.from_pauli_web(web)]) == spans([s])
    memory = gallery.memory(Basis.Z)
    single = memory.find_correlation_surfaces()[0]
    assert spans([tqec.CorrelationSurface.from_pauli_web(single.to_pauli_web(memory.to_zx_graph()))]) == spans([single])
    # An L of PyZX's: in, Z, X, out, and an S spider on top; its inputs and outputs name the ports.
    h = zx.Graph()
    vs = [h.add_vertex(t) for t in (zx.VertexType.BOUNDARY, zx.VertexType.Z, zx.VertexType.X, zx.VertexType.BOUNDARY, zx.VertexType.Z)]
    h.set_phase(vs[4], Fraction(1, 2))
    for e in ((0, 1), (1, 2), (2, 3), (2, 4)):
        h.add_edge(e)
    h.set_inputs((0,))
    h.set_outputs((3,))
    positions = {0: Position3D(-1, 0, 0), 1: Position3D(0, 0, 0), 2: Position3D(0, 1, 0), 3: Position3D(0, 2, 0), 4: Position3D(0, 1, 1)}
    assert structure(block_synthesis(h, positions=positions)) == structure(PositionedZX(h, positions).to_block_graph()) == (
        [((-1, 0, 0), "PORT", "IN_0"), ((0, 0, 0), "XXZ", ""), ((0, 1, 0), "XZZ", ""), ((0, 1, 1), "Y", ""), ((0, 2, 0), "PORT", "OUT_0")],
        [((-1, 0, 0), (0, 0, 0), "OXZ"), ((0, 0, 0), (0, 1, 0), "XOZ"), ((0, 1, 0), (0, 1, 1), "XZO"), ((0, 1, 0), (0, 2, 0), "XOZ")],
        {"IN_0": (-1, 0, 0), "OUT_0": (0, 2, 0)},
    )
    assert is_boundary(h, 0) and PositionedZX(h, positions).g is h
    with pytest.raises(ValueError, match="requires specifying positions"):
        block_synthesis(h)
    with pytest.raises(TQECError, match="do not match"):
        PositionedZX(h, {v: p for v, p in positions.items() if v != 4})
    with pytest.raises(TQECError, match="must be neighbors"):
        PositionedZX(h, {**positions, 4: Position3D(0, 1, 3)})
    bad = zx.Graph()
    bad.add_vertex(zx.VertexType.H_BOX)
    with pytest.raises(TQECError, match="Unsupported spider type and phase"):
        tqec.find_correlation_surfaces(bad)


@needs_tqec
@pytest.mark.parametrize("name,basis", GALLERY + [("cz", f) for f in CZ_FLOWS])
def test_pyzx_interop_agrees_with_tqec(name, basis):
    from tqec.computation.correlation import find_correlation_surfaces as their_find

    from stabilizer_qec.tqec.interop.pyzx import block_synthesis

    def shape(g):
        return [(v, str(g.type(v)), g.phase(v), [(u, str(g.edge_type((v, u)))) for u in g.neighbors(v)]) for v in g.vertices()]

    o, t = ours(name, basis), tqecs(name, basis)
    oz, tz = o.to_zx_graph(), t.to_zx_graph()
    assert shape(oz.g) == shape(tz.g)
    assert spans(tqec.find_correlation_surfaces(tz.g)) == spans(their_find(tz.g)) == spans(their_find(oz.g))
    for a, b in zip(o.find_correlation_surfaces(), t.find_correlation_surfaces()):
        assert a.to_pauli_web(oz).half_edges() == b.to_pauli_web(tz.g).half_edges()
    assert structure(oz.to_block_graph()) == structure(tz.to_block_graph())
    assert structure(block_synthesis(tz.g, positions={v: Position3D(*p.as_tuple()) for v, p in tz.positions.items()})) == structure(tz.to_block_graph())
