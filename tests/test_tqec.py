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
