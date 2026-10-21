"""Tsim's ZX graphs on this package's circuits: ``get_graph``, ``get_sampling_graph``, ``tcount``,
the ``pyzx`` diagrams and ``to_matrix`` of a circuit with measurements.

The expected values are recorded from Tsim (bloqade-tsim 0.1.5 on pyzx_param 0.9.3) by
tools/tsim_zx_fixtures.py into tests/tsim_port/zx_graphs.json.gz: for 40 circuits, the graph's
vertices (number, type, phase, phase variables, qubit, row) and edges (pair, type) in order, its
inputs, outputs and scalar, for ``get_graph()`` and both ``get_sampling_graph`` graphs, the builders'
record (measurement vertices, detectors, observables, error bits and channel probabilities), the
T-count, the matrices of the 12 circuits with measurements and the diagram graphs. The graphs are
compared on both backends: pyzx_param's (skipped where it is not installed) and this package's own;
the comparisons with Tsim itself are skipped where it is not installed.
"""

from __future__ import annotations

import gzip
import importlib.util
import json
import re
import sys
import types
import warnings
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

import stabilizer_qec as sq
from stabilizer_qec import tsim
from stabilizer_qec.tsim import zx
from stabilizer_qec.tsim.zx import EdgeType, VertexType, ZXGraph

FIXTURES = json.loads(gzip.decompress((Path(__file__).parent / "tsim_port" / "zx_graphs.json.gz").read_bytes()))
CIRCUITS = {c["name"]: c for c in FIXTURES["circuits"]}
NAMES = list(CIRCUITS)
MATRIX_NAMES = [n for n in NAMES if "matrix" in CIRCUITS[n]]

HAS_PYZX_PARAM = importlib.util.find_spec("pyzx_param") is not None
needs_pyzx_param = pytest.mark.skipif(not HAS_PYZX_PARAM, reason="pyzx_param is not installed")


@pytest.fixture(autouse=True)
def _numpy_print_options():
    """Importing pyzx_param (the default backend does) sets numpy's ``suppress=True`` for the process:
    kept from the tests that follow."""
    with np.printoptions():
        yield


def record(g) -> dict:
    """A graph's fields, as the fixtures hold them (phases Fractions, phase variables sets)."""
    return {
        "vertices": [[int(v), int(g.type(v)), Fraction(g.phase(v)), set(g._phaseVars[v]), g.qubit(v), g.row(v)] for v in g.vertices()],
        "edges": [[int(e[0]), int(e[1]), int(g.edge_type(e))] for e in g.edges()],
        "scalar": [int(g.scalar.power2), Fraction(g.scalar.phase)],
        "inputs": [int(v) for v in g.inputs()],
        "outputs": [int(v) for v in g.outputs()],
    }


def expected(raw: dict) -> dict:
    """A fixture graph with its phases as Fractions and its variables as sets."""
    return {
        "vertices": [[v, t, Fraction(p), set(names), q, r] for v, t, p, names, q, r in raw["vertices"]],
        "edges": raw["edges"],
        "scalar": [raw["scalar"][0], Fraction(raw["scalar"][1])],
        "inputs": raw["inputs"],
        "outputs": raw["outputs"],
    }


def assert_graph_equals(g, raw: dict) -> None:
    got, want = record(g), expected(raw)
    assert [v[0] for v in got["vertices"]] == [v[0] for v in want["vertices"]], "vertex numbers"
    for a, b in zip(got["vertices"], want["vertices"]):
        assert a == b, f"vertex {b[0]}"
    assert got["edges"] == want["edges"], "edges"
    assert got["scalar"] == want["scalar"], "scalar"
    assert got["inputs"] == want["inputs"] and got["outputs"] == want["outputs"], "inputs and outputs"


def signature_multiset(g) -> list:
    """What an isomorphism preserves, without the vertex numbers: each vertex's type, phase, variables,
    position and the (edge type, type, phase, variables) of each neighbour, sorted."""

    def sig(v):
        return (int(g.type(v)), Fraction(g.phase(v)), tuple(sorted(g._phaseVars[v])), float(g.qubit(v)), float(g.row(v)))

    out = []
    for v in g.vertices():
        around = sorted((int(g.edge_type(g.edge(v, w))),) + sig(w) for w in g.neighbors(v))
        out.append((sig(v), tuple(around)))
    return sorted(out)


def assert_built_equals(b, raw: dict) -> None:
    assert [int(v) for v in b.rec] == raw["rec"]
    assert [int(v) for v in b.silent_rec] == raw["silent_rec"]
    assert [int(v) for v in b.detectors] == raw["detectors"]
    assert {str(k): int(v) for k, v in b.observables_dict.items()} == raw["observables"]
    assert {str(k): int(v) for k, v in b.first_vertex.items()} == raw["first_vertex"]
    assert {str(k): int(v) for k, v in b.last_vertex.items()} == raw["last_vertex"]
    assert b.num_error_bits == raw["num_error_bits"]
    assert len(b.channel_probs) == len(raw["channel_probs"])
    for p, q in zip(b.channel_probs, raw["channel_probs"]):
        assert np.array_equal(p, np.array(q))


def check_circuit_on_backend(name: str, backend: str) -> None:
    entry = CIRCUITS[name]
    c = tsim.Circuit(entry["text"])
    assert_graph_equals(c.get_graph(backend=backend), entry["graph"])
    assert_graph_equals(c.get_sampling_graph(backend=backend), entry["sampling"])
    assert_graph_equals(c.get_sampling_graph(sample_detectors=True, backend=backend), entry["sampling_dets"])
    assert_built_equals(zx.parse_stim_circuit(c, backend=backend), entry["built"])


def assert_matrix_is_tsims(mine, ref) -> None:
    """``to_matrix`` is Tsim's: equal to it when it is the ZX tensor (a circuit with a measurement, a
    detector or observable, a reset after a qubit's first operation, feedback or a gate the engine
    has no matrix for), up to a global phase only when the unitary engine gives it; the ZX tensor
    (``_zx_matrix``) equal to Tsim's either way."""
    want = np.asarray(ref.to_matrix())
    original, used_zx = tsim.Circuit._zx_matrix, []

    def spy(self, endian, *built):
        used_zx.append(endian)
        return original(self, endian, *built)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(tsim.Circuit, "_zx_matrix", spy)
        got = mine.to_matrix()
    assert got.shape == want.shape, str(mine)
    assert np.allclose(mine._zx_matrix("big"), want, atol=1e-9, rtol=0), str(mine)
    if used_zx:
        assert np.allclose(got, want, atol=1e-9, rtol=0), str(mine)
    else:
        i = np.argmax(np.abs(want))
        phase = got.flat[i] / want.flat[i]
        assert abs(abs(phase) - 1) < 1e-9 and np.allclose(got, phase * want, atol=1e-9, rtol=0), str(mine)


def test_the_corpus_is_forty_circuits_and_twelve_matrices():
    assert len(NAMES) == 40 and len(MATRIX_NAMES) == 12
    assert FIXTURES["tsim"] == "0.1.5"


@needs_pyzx_param
@pytest.mark.parametrize("name", NAMES)
def test_graph_equals_tsim(name):
    """On pyzx_param's graph: Tsim's vertex numbers, types, phases (as Fractions), phase variables (as
    sets), positions, edges, scalar, inputs and outputs, for the graph and both sampling graphs."""
    import pyzx_param

    entry = CIRCUITS[name]
    g = tsim.Circuit(entry["text"]).get_graph(backend="pyzx_param")
    assert type(g) is pyzx_param.graph.graph_s.GraphS
    check_circuit_on_backend(name, "pyzx_param")


@pytest.mark.parametrize("name", NAMES)
def test_own_backend_isomorphic_and_equal_fields(name):
    """On this package's graph, built without pyzx_param: the same fields, vertex for vertex (the
    numbers a copy, a removal and a composition give are pyzx_param's), so the graphs are isomorphic
    with every field equal; the numbering-free signatures agree too."""
    entry = CIRCUITS[name]
    c = tsim.Circuit(entry["text"])
    assert type(c.get_graph(backend="own")) is ZXGraph
    check_circuit_on_backend(name, "own")
    for graph, raw in ((c.get_graph(backend="own"), entry["graph"]), (c.get_sampling_graph(backend="own"), entry["sampling"]), (c.get_sampling_graph(True, backend="own"), entry["sampling_dets"])):
        want = expected(raw)
        reference = ZXGraph()
        for v, t, p, names, q, r in want["vertices"]:
            reference.add_vertex(index=v, ty=t, qubit=q, row=r, phase=p, phaseVars=names)
        for s, t, et in want["edges"]:
            reference.add_edge((s, t), et)
        assert signature_multiset(graph) == signature_multiset(reference)


@pytest.mark.parametrize("name", MATRIX_NAMES)
def test_to_matrix_with_measurements_allclose(name):
    """Tsim's matrix of a circuit with measurements, detectors, resets, feedback, noise and
    ``MPP``: every measurement outcome and error bit read as 0, the scalar kept."""
    entry = CIRCUITS[name]
    m = entry["matrix"]
    want = (np.array(m["real"]) + 1j * np.array(m["imag"])).reshape(m["shape"])
    got = tsim.Circuit(entry["text"]).to_matrix()
    assert got.shape == tuple(m["shape"])
    assert np.allclose(got, want, atol=1e-12, rtol=0)
    # The tensor is the matrix with an index of size 2 per output and input qubit.
    t = tsim.Circuit(entry["text"]).to_tensor()
    assert t.shape == (2,) * (got.shape[0].bit_length() - 1 + got.shape[1].bit_length() - 1)
    assert np.array_equal(t.reshape(got.shape), got)


@needs_pyzx_param
@pytest.mark.parametrize("name", MATRIX_NAMES)
def test_contraction_agrees_on_pyzx_params_graph(name):
    """``tsim.zx.tensorfy`` of pyzx_param's graph is Tsim's matrix, and pyzx_param's own ``to_matrix``
    of it the same: the contraction is pyzx_param's."""
    entry = CIRCUITS[name]
    m = entry["matrix"]
    want = (np.array(m["real"]) + 1j * np.array(m["imag"])).reshape(m["shape"])
    g = zx.parse_stim_circuit(entry["text"], backend="pyzx_param").graph.copy()
    g.normalize()
    mine = zx.tensor_to_matrix(zx.tensorfy(g), g.num_inputs(), g.num_outputs())
    assert np.allclose(mine, want, atol=1e-12, rtol=0)
    assert np.allclose(g.to_matrix(), want, atol=1e-12, rtol=0)


@pytest.mark.parametrize("name", NAMES)
def test_tcount(name):
    """Tsim's T-count: the vertices of its ZX graph with a non-Clifford phase."""
    entry = CIRCUITS[name]
    c = tsim.Circuit(entry["text"])
    assert c.tcount() == entry["tcount"]
    assert zx.tcount(c.get_graph(backend="own")) == entry["tcount"]
    if HAS_PYZX_PARAM:
        import pyzx_param

        assert pyzx_param.tcount(c.get_graph(backend="pyzx_param")) == entry["tcount"]


def test_tcount_is_the_graphs_not_the_engines():
    # Tsim reads a coherent error as an identity; stabilizer_qec.Circuit counts it as the phase it is.
    text = "T 0\nI_ERROR[R_Z(theta=0.1)](0.5) 0\nR_Z(0.1) 1"
    assert sq.Circuit(text.replace("T 0", "I[T] 0").replace("R_Z(0.1) 1", "I[R_Z(theta=0.15707963267948966)] 1")).tcount() == 3
    assert tsim.Circuit(text).tcount() == 2


def test_the_frame_samplers_warning_still_counts_coherent_errors():
    # tsim.Circuit.tcount is Tsim's (no coherent errors), the warning of the frame sampler the engine's.
    c = tsim.Circuit("R 0\nH 0\nI_ERROR[R_Z(theta=0.5)] 0\nH 0\nM 0\nDETECTOR rec[-1]")
    assert c.tcount() == 0
    with pytest.warns(UserWarning, match="reads the circuit's coherent errors"):
        c.compile_detector_sampler(seed=1, device="cpu")


def test_diagram_reduced_needs_pyzx_param(monkeypatch):
    monkeypatch.setitem(sys.modules, "pyzx_param", None)
    c = tsim.Circuit("R 0\nH 0\nT 0\nM 0\nDETECTOR rec[-1]")
    for kind in ("pyzx-dets", "pyzx-meas"):
        with pytest.raises(ImportError, match="pyzx_param") as info:
            c.diagram(kind)
        assert kind in str(info.value) and "full_reduce" in str(info.value) and "pip install pyzx-param" in str(info.value)
    with pytest.raises(ImportError, match="pyzx_param backend"):
        c.get_graph(backend="pyzx_param")
    # The graphs and the matrix need nothing.
    assert type(c.get_graph()) is ZXGraph
    assert type(c.get_sampling_graph(True)) is ZXGraph
    assert c.to_matrix().shape == (2, 1)


def test_default_backend_is_the_installed_one(monkeypatch):
    c = tsim.Circuit("H 0\nT 0\nM 0")
    if HAS_PYZX_PARAM:
        import pyzx_param

        assert type(c.get_graph()) is pyzx_param.graph.graph_s.GraphS
        assert type(c.get_sampling_graph()) is pyzx_param.graph.graph_s.GraphS
    monkeypatch.setitem(sys.modules, "pyzx_param", None)
    assert type(c.get_graph()) is ZXGraph
    assert type(c.get_graph(backend="own")) is ZXGraph
    with pytest.raises(ValueError, match="unknown ZX graph backend"):
        c.get_graph(backend="quizx")
    with pytest.raises(ValueError, match="unknown ZX graph backend"):
        c.get_sampling_graph(backend="graph_tool")


@needs_pyzx_param
@pytest.mark.parametrize("name", NAMES)
def test_diagrams_equal_tsim(name):
    """``diagram("pyzx")``, ``("pyzx-dets")`` and ``("pyzx-meas")`` draw and return Tsim's graphs:
    the circuit's with its classical wires bold and a boundary beside each detector and observable,
    and the sampling graphs reduced by pyzx_param's ``full_reduce(paramSafe=True)``."""
    plt = pytest.importorskip("matplotlib.pyplot")
    previous = plt.get_backend()
    plt.switch_backend("Agg")  # pyzx_param shows the figure: off screen
    try:
        entry = CIRCUITS[name]
        c = tsim.Circuit(entry["text"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for kind, raw in entry["diagrams"].items():
                assert_graph_equals(c.diagram(kind), raw)
    finally:
        plt.close("all")
        plt.switch_backend(previous)


def test_diagram_pyzx_draws_with_pyzx_when_pyzx_param_is_missing(monkeypatch):
    pyzx = pytest.importorskip("pyzx")
    plt = pytest.importorskip("matplotlib.pyplot")
    monkeypatch.setitem(sys.modules, "pyzx_param", None)
    drawn = []
    monkeypatch.setattr(pyzx, "draw", lambda g, **kw: drawn.append((g, kw)))
    c = tsim.Circuit("R 0\nH 0\nT 0\nCX 0 1\nX_ERROR(0.1) 1\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]")
    g = c.diagram("pyzx", scale_horizontally=2.0)
    assert type(g) is ZXGraph and len(drawn) == 1
    plain, kwargs = drawn[0]
    assert type(plain).__module__.startswith("pyzx.") and kwargs == {}
    assert plain.num_vertices() == g.num_vertices() and plain.num_edges() == g.num_edges()
    # A boundary beside the detector and the observable, the classical wires bold.
    base = c.get_graph(backend="own")
    assert g.num_vertices() == base.num_vertices() + 2
    assert any(int(g.type(v)) in (int(VertexType.Z_BOLD), int(VertexType.X_BOLD)) for v in g.vertices())
    plt.close("all")


@pytest.mark.parametrize("name", NAMES)
def test_diagram_pyzx_on_this_packages_graph_equals_tsim(name, monkeypatch):
    """Without pyzx_param, ``diagram("pyzx")`` builds Tsim's diagram on this package's graph, vertex
    for vertex (the bold classical wires, a boundary beside each detector and observable, the last
    vertices lined up), and draws it with pyzx (stubbed here: the drawing is pyzx's)."""
    from stabilizer_qec.tsim.zx import diagram as diagrams

    monkeypatch.setitem(sys.modules, "pyzx_param", None)
    monkeypatch.setitem(sys.modules, "pyzx", types.ModuleType("pyzx"))
    drawn = []
    monkeypatch.setattr(diagrams, "draw", lambda g, **kwargs: drawn.append((g, kwargs)))
    g = tsim.Circuit(CIRCUITS[name]["text"]).diagram("pyzx")
    assert type(g) is ZXGraph
    assert len(drawn) == (1 if g.num_vertices() else 0) and all(d is g and kwargs == {} for d, kwargs in drawn)
    assert_graph_equals(g, CIRCUITS[name]["diagrams"]["pyzx"])


def test_a_pyzx_param_that_does_not_import_is_not_used(monkeypatch, tmp_path):
    """An installed pyzx_param that fails to import (here with the AttributeError of a numpy it
    predates) is as if it were not installed: the graphs are this package's, and what needs it says
    why it does not import."""
    broken = tmp_path / "pyzx_param"
    broken.mkdir()
    (broken / "__init__.py").write_text("raise AttributeError(\"module 'numpy' has no attribute 'float_'\")\n")
    for module in [m for m in sys.modules if m == "pyzx_param" or m.startswith("pyzx_param.")]:
        monkeypatch.delitem(sys.modules, module)
    monkeypatch.syspath_prepend(str(tmp_path))
    c = tsim.Circuit("R 0\nH 0\nT 0\nM 0\nDETECTOR rec[-1]")
    assert not zx.pyzx_param_available()
    assert type(c.get_graph()) is ZXGraph and type(c.get_sampling_graph(True)) is ZXGraph
    assert type(zx.parse_stim_circuit(c).graph) is ZXGraph
    with pytest.raises(ImportError) as info:
        c.get_graph(backend="pyzx_param")
    assert str(info.value) == "The pyzx_param backend needs pyzx_param (pip install pyzx-param), which is installed but does not import: AttributeError(\"module 'numpy' has no attribute 'float_'\")"
    assert isinstance(info.value.__cause__, AttributeError)
    with pytest.raises(ImportError, match="pyzx-dets.*which is installed but does not import: AttributeError"):
        c.diagram("pyzx-dets")
    # Not installed (or blocked): said so.
    monkeypatch.setitem(sys.modules, "pyzx_param", None)
    with pytest.raises(ImportError) as info:
        c.get_graph(backend="pyzx_param")
    assert str(info.value) == "The pyzx_param backend needs pyzx_param (pip install pyzx-param), which is not installed"


def test_diagram_pyzx_needs_a_drawing_library(monkeypatch):
    monkeypatch.setitem(sys.modules, "pyzx_param", None)
    monkeypatch.setitem(sys.modules, "pyzx", None)
    with pytest.raises(ImportError, match="pyzx"):
        tsim.Circuit("H 0\nM 0").diagram("pyzx")


def test_to_pyzx():
    pyzx = pytest.importorskip("pyzx")
    c = tsim.Circuit(CIRCUITS["repetition_code_noisy"]["text"])
    g = c.get_graph(backend="own")
    p = g.to_pyzx()
    assert type(p) is pyzx.graph.graph_s.GraphS
    assert p.num_vertices() == g.num_vertices() and p.num_edges() == g.num_edges()
    assert p.scalar.power2 == g.scalar.power2 and p.scalar.phase == g.scalar.phase
    seen_variable = False
    for v, pv in zip(g.vertices(), p.vertices()):
        assert int(p.type(pv)) == int(g.type(v))
        assert (p.qubit(pv), p.row(pv)) == (g.qubit(v), g.row(v))
        phase, names = p.phase(pv), g.phase_vars(v)
        if names:
            # The variables are boolean symbols added to the constant phase.
            seen_variable = True
            assert {str(x) for x in phase.free_vars()} == names
            constant = sum((coeff for coeff, term in phase.terms if not term.vars), Fraction(0))
            assert constant == Fraction(g.phase(v))
        else:
            assert Fraction(phase) == Fraction(g.phase(v))
    assert seen_variable
    # The bold graph converts to plain pyzx types.
    bold = zx.parse_stim_circuit(c, track_classical_wires=True, backend="own").graph
    pb = bold.to_pyzx()
    assert all(int(pb.type(v)) in (0, 1, 2) for v in pb.vertices())


def test_zx_graph_edge_rules_match_pyzx_param():
    """Parallel edges, self-loops, removal and renumbering behave as pyzx_param's GraphS."""
    pytest.importorskip("pyzx_param")
    from pyzx_param.graph.graph_s import GraphS
    from pyzx_param.utils import EdgeType as PE
    from pyzx_param.utils import VertexType as PV

    rng = np.random.default_rng(7)
    for _ in range(60):
        ours, theirs = ZXGraph(), GraphS()
        n = int(rng.integers(2, 6))
        for _v in range(n):
            ty = int(rng.choice([0, 1, 2]))
            ph = Fraction(int(rng.integers(0, 4)), 2)
            ours.add_vertex(VertexType(ty), qubit=_v, row=_v, phase=ph)
            theirs.add_vertex(PV(ty), qubit=_v, row=_v, phase=ph)
        for _e in range(int(rng.integers(1, 12))):
            s, t = (int(x) for x in rng.integers(0, n, size=2))
            et = int(rng.choice([1, 2]))
            outcome = []
            for g, E in ((ours, EdgeType), (theirs, PE)):
                try:
                    g.add_edge((s, t), E(et))
                    outcome.append("ok")
                except ValueError as exc:
                    outcome.append(str(exc).split(",")[0])
            assert outcome[0] == outcome[1]
        assert sorted(ours.edges()) == sorted(theirs.edges())
        assert [int(ours.edge_type(e)) for e in sorted(ours.edges())] == [int(theirs.edge_type(e)) for e in sorted(theirs.edges())]
        assert [Fraction(ours.phase(v)) for v in ours.vertices()] == [Fraction(theirs.phase(v)) for v in theirs.vertices()]
        assert ours.scalar.power2 == theirs.scalar.power2 and ours.scalar.phase == theirs.scalar.phase
        victim = int(rng.integers(0, n))
        ours.remove_vertex(victim)
        theirs.remove_vertex(victim)
        assert ours.add_vertex(VertexType.Z) == theirs.add_vertex(PV.Z)
        a, b = ours.copy(), theirs.copy()
        assert list(a.vertices()) == list(b.vertices()) and sorted(a.edges()) == sorted(b.edges())
        a, b = ours.adjoint(), theirs.adjoint()
        assert [Fraction(a.phase(v)) for v in a.vertices()] == [Fraction(b.phase(v)) for v in b.vertices()]
        assert a.scalar.power2 == b.scalar.power2 and a.scalar.phase == b.scalar.phase


def test_zx_graph_basics():
    g = ZXGraph()
    a = g.add_vertex(VertexType.Z, qubit=0, row=1, phase=Fraction(1, 4))
    b = g.add_vertex(VertexType.X, qubit=1, row=2, phase="rec[0]")
    assert g.phase(a) == Fraction(1, 4) and g.phase_vars(b) == {"rec[0]"} and g.phase(b) == 0
    g.add_edge((a, b), EdgeType.HADAMARD)
    assert g.num_vertices() == 2 and g.num_edges() == 1 and list(g.edges()) == [(0, 1)]
    assert g.edge_type((1, 0)) == EdgeType.HADAMARD and g.connected(0, 1)
    # A Hadamard self-loop on a spider adds a pi, a simple one nothing.
    g.add_edge((a, a), EdgeType.HADAMARD)
    g.add_edge((a, a), EdgeType.SIMPLE)
    assert g.phase(a) == Fraction(5, 4) and g.num_edges() == 1
    with pytest.raises(ValueError, match="self-loop"):
        g.add_vertex(VertexType.BOUNDARY)
        g.add_edge((2, 2))
    # phase_vars is a copy; the scalar is 1.
    names = g.phase_vars(b)
    names.add("x")
    assert g.phase_vars(b) == {"rec[0]"}
    assert g.scalar.power2 == 0 and g.scalar.phase == 0 and g.scalar.to_number() == 1
    # A phase is taken mod 2.
    g.set_phase(a, Fraction(5, 2))
    assert g.phase(a) == Fraction(1, 2)
    h = g.clone()
    h.set_phase(a, 0)
    assert g.phase(a) == Fraction(1, 2)


def test_tensor_of_small_graphs():
    # A Hadamard edge between two boundaries is the Hadamard matrix; a Z spider with phase pi/2 and two
    # legs the S gate; the scalar multiplies.
    g = ZXGraph()
    i = g.add_vertex(VertexType.BOUNDARY, qubit=0, row=0)
    o = g.add_vertex(VertexType.BOUNDARY, qubit=0, row=1)
    g.add_edge((i, o), EdgeType.HADAMARD)
    g.set_inputs([i])
    g.set_outputs([o])
    assert np.allclose(g.to_matrix(), np.array([[1, 1], [1, -1]]) / np.sqrt(2))
    s = ZXGraph()
    i = s.add_vertex(VertexType.BOUNDARY, qubit=0, row=0)
    z = s.add_vertex(VertexType.Z, qubit=0, row=1, phase=Fraction(1, 2))
    o = s.add_vertex(VertexType.BOUNDARY, qubit=0, row=2)
    s.add_edge((i, z))
    s.add_edge((z, o))
    s.set_inputs([i])
    s.set_outputs([o])
    s.scalar.add_power(2)
    assert np.allclose(s.to_matrix(), 2 * np.diag([1, 1j]))
    assert np.allclose(s.to_matrix(preserve_scalar=False), np.diag([1, 1j]))
    # Phase variables are read as 0.
    s.set_phase(z, "e0")
    assert np.allclose(s.to_matrix(), 2 * np.eye(2))
    # An isolated spider is a number: Z(phase 0) is 1 + 1 = 2.
    n = ZXGraph()
    n.add_vertex(VertexType.Z, qubit=0, row=0)
    assert np.allclose(n.to_tensor(), 2)
    # A vertex that is neither a spider nor an input or output is refused.
    bad = ZXGraph()
    bad.add_vertex(VertexType.H_BOX)
    with pytest.raises(ValueError, match="not a Z or X spider"):
        bad.to_tensor()
    lone = ZXGraph()
    lone.add_vertex(VertexType.BOUNDARY)
    with pytest.raises(ValueError, match="no inputs or outputs set"):
        lone.to_tensor()


def test_to_matrix_measurement_semantics():
    h = 1 / np.sqrt(2)
    # A measurement is the identity on its qubit times 1/sqrt(2), so H then M is H/sqrt(2).
    assert np.allclose(tsim.Circuit("H 0\nM 0").to_matrix(), np.array([[0.5, 0.5], [0.5, -0.5]]))
    # A detector on the outcome projects it to 0; a reset projects and prepares.
    assert np.allclose(tsim.Circuit("H 0\nM 0\nDETECTOR rec[-1]").to_matrix(), [[h, h], [0, 0]])
    assert np.allclose(tsim.Circuit("H 0\nR 0").to_matrix(), [[1, 0], [0, 0]])
    assert np.allclose(tsim.Circuit("H 0\nMR 0").to_matrix(), [[h, 0], [0, 0]])
    # MPP adds the aux output qubit (qubit -2 of the graph, the most significant index).
    assert tsim.Circuit("H 0\nMPP X0*Z1").to_matrix().shape == (8, 4)
    # Feedback is the controlled gate.
    fb = tsim.Circuit("H 0\nM 0\nCX rec[-1] 1").to_matrix()
    assert fb.shape == (4, 4) and np.allclose(fb[:2, :2], [[0.5, 0], [0, 0.5]])
    # Noise is read as no error, whichever qubit it names first; a reset after noise projects.
    assert tsim.Circuit("X_ERROR(0.2) 0\nM 0").to_matrix().shape == (2, 2)
    assert np.allclose(tsim.Circuit("X_ERROR(0.2) 0\nM 0").to_matrix(), np.eye(2) * h)
    assert tsim.Circuit("Z_ERROR(0.037) 1\nR 1").to_matrix().shape == (2, 2)
    assert tsim.Circuit("I 0\nR 0").to_matrix().shape == (2, 2)
    assert tsim.Circuit("C_XYZ 1\nII 0 1").to_matrix().shape == (4, 4)
    # A circuit that resets every qubit first, then measures, is a state's column.
    assert tsim.Circuit("R 0 1\nH 0\nCX 0 1\nM 0 1").to_matrix().shape == (4, 1)


# Circuits with no measurement whose detectors and observables XOR nothing, and Tsim's matrices of them.
ANNOTATED_UNMEASURED = [
    "DETECTOR",
    "DETECTOR\nH 0",
    "H 0\nOBSERVABLE_INCLUDE(1)",
    "OBSERVABLE_INCLUDE(2)",
    "DETECTOR(1)\nR 0\nH 0",
    "REPEAT 3 {\n    DETECTOR\n}",
    "H 0\nREPEAT 2 {\n    OBSERVABLE_INCLUDE(0)\n    T 0\n}",
    "OBSERVABLE_INCLUDE(1)\nOBSERVABLE_INCLUDE(1)",
    "OBSERVABLE_INCLUDE(2)\nOBSERVABLE_INCLUDE(0)",
    "DETECTOR(1, 2)\nSHIFT_COORDS(1)\nDETECTOR\nCX 0 1",
    "OBSERVABLE_INCLUDE(1)\nX_ERROR(0.1) 0",
    "DETECTOR\nII 0 1",
    "R 0 1\nH 0\nCX 0 1\nDETECTOR(0, 0, 0)\nTPP X0*Z1",
]


def test_to_matrix_of_annotations_without_measurements():
    """A detector or an observable is an X spider of Tsim's graph, so one that XORs no measurement
    stands alone, a factor of 2 (an ``OBSERVABLE_INCLUDE(k)`` makes the observables 0 to k), with or
    without a measurement elsewhere in the circuit. Tsim's values."""
    h = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    for text, want in (
        ("DETECTOR", [[2]]),
        ("DETECTOR\nH 0", 2 * h),
        ("H 0\nOBSERVABLE_INCLUDE(1)", 4 * h),
        ("OBSERVABLE_INCLUDE(2)", [[8]]),
        ("DETECTOR(1)\nR 0\nH 0", 2 * np.sqrt(0.5) * np.ones((2, 1))),
        ("REPEAT 3 {\n    DETECTOR\n}", [[8]]),
        ("OBSERVABLE_INCLUDE(1)\nOBSERVABLE_INCLUDE(1)", [[4]]),
        ("OBSERVABLE_INCLUDE(2)\nOBSERVABLE_INCLUDE(0)", [[8]]),
        ("DETECTOR\nII 0 1", 2 * np.eye(4)),
        ("M 0\nDETECTOR\nOBSERVABLE_INCLUDE(3)", 16 * np.sqrt(2) * np.eye(2)),
    ):
        got = tsim.Circuit(text).to_matrix()
        assert got.shape == np.shape(want) and np.allclose(got, want, atol=1e-12, rtol=0), text
    # An observable of Pauli targets is Tsim's error, measurement or not.
    for text in ("OBSERVABLE_INCLUDE(0) X0", "H 0\nOBSERVABLE_INCLUDE(1) Z0"):
        with pytest.raises(ValueError, match="OBSERVABLE_INCLUDE with Pauli targets is not supported in Tsim"):
            tsim.Circuit(text).to_matrix()


@pytest.mark.parametrize("text", ANNOTATED_UNMEASURED)
def test_to_matrix_of_annotations_equals_tsim(text):
    real = pytest.importorskip("tsim")
    want = np.asarray(real.Circuit(text).to_matrix())
    got = tsim.Circuit(text).to_matrix()
    assert got.shape == want.shape and np.allclose(got, want, atol=1e-9, rtol=0)


def test_to_matrix_of_loops_is_the_unrolled_circuits():
    looped = tsim.Circuit("REPEAT 3 {\n    H 0\n    T 0\n    M 0\n    R 1\n    CX 0 1\n}")
    unrolled = tsim.Circuit("H 0\nT 0\nM 0\nR 1\nCX 0 1\n" * 3)
    assert np.allclose(looped.to_matrix(), unrolled.to_matrix())
    assert looped.tcount() == unrolled.tcount() == 3
    # A loop whose second pass resets a qubit its first pass used is a reset after the first operation.
    assert tsim.Circuit("REPEAT 2 {\n    H 0\n    R 0\n}").to_matrix().shape == (2, 2)
    assert np.allclose(tsim.Circuit("REPEAT 2 {\n    H 0\n    R 0\n}").to_matrix(), tsim.Circuit("H 0\nR 0\nH 0\nR 0").to_matrix())


def test_to_matrix_endianness_of_the_zx_path():
    c = tsim.Circuit("H 0\nT 1\nCX 0 1\nM 1\nR 0\nH 1")
    big, little = c.to_matrix(), c.to_matrix(endian="little")
    assert big.shape == little.shape == (4, 4)

    def reverse(m):
        n, k = m.shape[0].bit_length() - 1, m.shape[1].bit_length() - 1
        t = m.reshape((2,) * (n + k))
        t = t.transpose(list(range(n - 1, -1, -1)) + list(range(n + k - 1, n - 1, -1)))
        return t.reshape(m.shape)

    assert np.allclose(little, reverse(big))
    assert np.array_equal(c.to_tensor(endian="little"), little.reshape((2,) * 4))
    with pytest.raises(ValueError, match="endian"):
        c.to_matrix(endian="middle")


@pytest.mark.parametrize("text", ["H 0\nT 1\nCX 0 1\nS 1", "R 0 1\nH 0\nCX 0 1\nR_Z(0.3) 1", "H 0\nCCZ 0 1 2", "RX 0\nR_X(0.3) 1\nTPP X0*Z1"])
def test_zx_path_agrees_with_the_unitary_engine(text):
    """The contraction of the ZX graph and the engine's unitary are the same map up to a global phase."""
    c = tsim.Circuit(text)
    zx_matrix, engine = c._zx_matrix("big"), c.to_matrix()
    assert zx_matrix.shape == engine.shape
    i = np.argmax(np.abs(engine))
    phase = zx_matrix.flat[i] / engine.flat[i]
    assert abs(abs(phase) - 1) < 1e-9 and np.allclose(zx_matrix, phase * engine, atol=1e-9)


def test_to_matrix_size_limits():
    wide = tsim.Circuit("H " + " ".join(map(str, range(12))) + "\nM 0")
    with pytest.raises(ValueError, match="at most 2"):
        wide.to_matrix()
    # A spider of more legs than a tensor can hold is refused, not allocated.
    g = ZXGraph()
    hub = g.add_vertex(VertexType.Z, qubit=0, row=5)
    for k in range(30):
        leaf = g.add_vertex(VertexType.X, qubit=k + 1, row=k)
        g.add_edge((hub, leaf))
    with pytest.raises(ValueError, match="2\\^30 entries"):
        g.to_tensor()
    # The tensor of the inputs is held before any spider: 14 wires (2^28 entries) are refused, as is
    # any intermediate tensor of more than 2^26 (1 GiB of complex128, which tensordot copies).
    wires = ZXGraph()
    ins, outs = [], []
    for q in range(14):
        i, o = wires.add_vertex(VertexType.BOUNDARY, qubit=q, row=0), wires.add_vertex(VertexType.BOUNDARY, qubit=q, row=2)
        z = wires.add_vertex(VertexType.Z, qubit=q, row=1)
        wires.add_edge((i, z))
        wires.add_edge((z, o))
        ins.append(i)
        outs.append(o)
    wires.set_inputs(ins)
    wires.set_outputs(outs)
    with pytest.raises(ValueError, match="^the contraction needs a tensor of more than 2\\^26 entries$"):
        zx.tensorfy(wires)
    # Three wires (2^6 entries for the inputs) contract to the identity.
    small = ZXGraph()
    ins, outs = [], []
    for q in range(3):
        i, o = small.add_vertex(VertexType.BOUNDARY, qubit=q, row=0), small.add_vertex(VertexType.BOUNDARY, qubit=q, row=2)
        z = small.add_vertex(VertexType.Z, qubit=q, row=1)
        small.add_edge((i, z))
        small.add_edge((z, o))
        ins.append(i)
        outs.append(o)
    small.set_inputs(ins)
    small.set_outputs(outs)
    assert np.allclose(zx.tensor_to_matrix(zx.tensorfy(small), 3, 3), np.eye(8))


def test_parser_errors_are_tsims():
    with pytest.raises(NotImplementedError, match="Sweep bit targets"):
        tsim.Circuit("CX sweep[0] 1").get_graph()
    with pytest.raises(ValueError, match="OBSERVABLE_INCLUDE with Pauli targets is not supported in Tsim"):
        tsim.Circuit("OBSERVABLE_INCLUDE(0) X0").get_graph()
    # A record entry from before the circuit is Tsim's IndexError, naming the instruction.
    for text in ("M 0\nDETECTOR rec[-2]", "M 0\nOBSERVABLE_INCLUDE(0) rec[-3]", "M 0\nCX rec[-2] 1", "DETECTOR rec[-1]"):
        with pytest.raises(IndexError, match=r"reads rec\[-\d\], but only") as info:
            tsim.Circuit(text).get_graph()
        assert isinstance(info.value, ValueError)


def chain_probabilities(probabilities: list) -> dict:
    """Tsim's ``correlated_error_probs`` entries that are not 0: index 0 (no branch fires) and 1 << i."""
    out, no_error_so_far = {}, 1.0
    for i, p in enumerate(probabilities):
        out[1 << i] = no_error_so_far * p
        no_error_so_far *= 1 - p
    out[0] = no_error_so_far
    return out


def test_a_long_chain_of_correlated_errors_builds_as_in_tsim():
    """A chain of k correlated errors is k error bits, as in Tsim; its 2^k probabilities (Tsim's
    ``channel_probs`` entry, an array) are kept as the chain's k for k over 24, an entry each."""
    from stabilizer_qec.tsim.zx.instructions import CorrelatedChainProbs, correlated_error_probs

    ps = [0.1 + 0.01 * i for i in range(31)]
    text = "H 0\nE(0.1) X0\n" + "".join(f"ELSE_CORRELATED_ERROR({p}) X0\n" for p in ps[1:]) + "M 0\nDETECTOR rec[-1]"
    c = tsim.Circuit(text)
    built = zx.parse_stim_circuit(c, backend="own")
    assert built.num_error_bits == 31
    assert sorted(set().union(*(c.get_graph(backend="own")._phaseVars.values()))) == sorted([f"e{i}" for i in range(31)] + ["det[0]", "rec[0]"])
    probs = built.channel_probs[-1]
    assert type(probs) is CorrelatedChainProbs and len(probs) == 2**31
    want = chain_probabilities(ps)
    for i, p in want.items():
        assert probs[i] == p and type(probs[i]) is np.float64
    assert probs[3] == probs[2**31 - 1] == probs[5 << 20] == 0.0 and probs[-(2**31)] == want[0] and probs[-1] == 0.0
    # The first 24 branches' entries are those of a chain of 24.
    short = correlated_error_probs(ps[:24])
    assert all(probs[1 << i] == short[1 << i] for i in range(24))
    for index in (2**31, -(2**31) - 1):
        with pytest.raises(IndexError, match="out of bounds"):
            probs[index]
    for read in (np.asarray, list, lambda p: p[0:2], lambda p: p[1.0]):
        with pytest.raises(ValueError, match="A chain of 31 correlated errors has 2\\^31 outcomes; their probabilities are held as an array for at most 24"):
            read(probs)
    # The matrix needs no probability: each error bit is read as 0, so it is the chain-free circuit's.
    assert np.allclose(c.to_matrix(), tsim.Circuit("H 0\nM 0\nDETECTOR rec[-1]").to_matrix(), atol=1e-12, rtol=0)
    assert np.allclose(c.to_matrix(), np.array([[1, 1], [0, 0]]) / np.sqrt(2), atol=1e-12, rtol=0)
    # A chain of 24 keeps Tsim's array.
    assert type(zx.parse_stim_circuit("E(0.1) X0\n" + "ELSE_CORRELATED_ERROR(0.1) X0\n" * 23, backend="own").channel_probs[-1]) is np.ndarray


@pytest.mark.parametrize("k", [25, 26])
def test_a_long_chain_of_correlated_errors_against_tsim(k):
    """Tsim builds a chain of more than 24 too (its array of 2^k probabilities allocated, not read):
    the same graph, entries and matrix."""
    real = pytest.importorskip("tsim")
    from tsim.core.parse import parse_stim_circuit as tsims_parse

    chain = "H 0\nE(0.1) X0\n" + "ELSE_CORRELATED_ERROR(0.2) Z0\n" * (k - 1)
    for text in (chain + "DETECTOR", chain + "M 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]"):
        ref, mine = real.Circuit(text), tsim.Circuit(text)
        assert record(ref.get_graph()) == record(mine.get_graph(backend="own"))
        theirs = tsims_parse(ref._stim_circ).channel_probs[-1]
        ours = zx.parse_stim_circuit(mine, backend="own").channel_probs[-1]
        assert len(ours) == len(theirs) == 2**k
        for i in [0, 3, 2**k - 1] + [1 << j for j in range(k)]:
            assert ours[i] == theirs[i], i
        assert np.allclose(mine.to_matrix(), np.asarray(ref.to_matrix()), atol=1e-12, rtol=0)
    # DETECTOR, an X spider alone, is a factor of 2: twice H.
    assert np.allclose(tsim.Circuit(chain + "DETECTOR").to_matrix(), np.sqrt(2) * np.array([[1, 1], [1, -1]]), atol=1e-12, rtol=0)


def test_instructions_are_read_back_as_tsims_half_turns():
    from stabilizer_qec.tsim.zx.parse import tsim_instructions

    c = tsim.Circuit("T 0\nT_DAG 1\nR_Z(0.3) 0\nR_X(1e-05) 1\nR_Y(-0.25) 2\nU3(0.1, 0.2, 0.3) 0\nR_PAULI(0.3) Z2*X0\nTPP_DAG X0*Y1\nR_ZZ(0.7) 0 1\nR_XX(0.25) 1 2\nH 0")
    lines = [f"{i.name}[{i.tag}] {' '.join(t.target_str() for t in i.targets_copy())}".replace(" * ", "*") for i in tsim_instructions(c)]
    assert lines == [
        "S[T] 0",
        "S_DAG[T] 1",
        "I[R_Z(theta=0.3*pi)] 0",
        "I[R_X(theta=1e-05*pi)] 1",
        "I[R_Y(theta=-0.25*pi)] 2",
        "I[U3(theta=0.1*pi, phi=0.2*pi, lambda=0.3*pi)] 0",
        "SPP[R_PAULI(theta=0.3*pi)] Z2*X0",
        "SPP_DAG[T] X0*Y1",
        "SPP[R_PAULI(theta=0.7*pi)] Z0*Z1",
        "SPP[R_PAULI(theta=0.25*pi)] X1*X2",
        "H[] 0",
    ]


PI_OVER_EIGHT = {  # a product rotation by +-pi/8: the TPP whose spiders it has, and its scalar phase
    "R_PAULI(0.25) X0*Y1": ("TPP X0*Y1", Fraction(15, 8)),
    "R_PAULI(-0.25) X0*Y1": ("TPP_DAG X0*Y1", Fraction(1, 8)),
    "R_XX(0.25) 0 1": ("TPP X0*X1", Fraction(15, 8)),
    "R_XX(-0.25) 0 1": ("TPP_DAG X0*X1", Fraction(1, 8)),
    "R_ZZ(0.25) 1 0": ("TPP Z1*Z0", Fraction(15, 8)),
    "R_PAULI(-0.25) Z0": ("TPP_DAG Z0", Fraction(1, 8)),
}


def test_a_pi_over_eight_product_rotation_is_r_pauli_not_tpp():
    """``R_PAULI(+-0.25)`` and ``R_XX/YY/ZZ(+-0.25)`` are Tsim's ``R_PAULI``: ``TPP``'s spiders with the
    scalar phase -+1/8 that ``TPP``'s graph has not (so the matrix is exp(-+i pi/8 P), not ``TPP``'s);
    printed and read back as written."""
    from stabilizer_qec.tsim.zx.parse import tsim_instructions

    for rotation, (tpp, scalar) in PI_OVER_EIGHT.items():
        c, t = tsim.Circuit(rotation + "\nM 0"), tsim.Circuit(tpp + "\nM 0")
        g, h = record(c.get_graph(backend="own")), record(t.get_graph(backend="own"))
        assert g["scalar"] == [h["scalar"][0], scalar] and h["scalar"][1] == 0, rotation
        assert g["vertices"] == h["vertices"] and g["edges"] == h["edges"], rotation
        assert str(c) == rotation + "\nM 0" and str(t) == tpp + "\nM 0"
        assert str(c.copy()) == str(c[0:1] + c[1:]) == str(c * 1) == rotation + "\nM 0"
        assert np.allclose(c.to_matrix(), np.exp(1j * np.pi * float(scalar)) * t.to_matrix(), atol=1e-12, rtol=0)
        assert c.tcount() == t.tcount() == 1
        [rot, _] = tsim_instructions(c)
        assert (rot.name, rot.tag) == ("SPP", f"R_PAULI(theta={'-' if '-' in rotation else ''}0.25*pi)")
    # Unmeasured, the engine's unitary is the rotation's: exp(-i pi/8 X X).
    xx = np.kron(np.array([[0, 1], [1, 0]]), np.array([[0, 1], [1, 0]]))
    want = np.cos(np.pi / 8) * np.eye(4) - 1j * np.sin(np.pi / 8) * xx
    assert np.allclose(tsim.Circuit("R_XX(0.25) 0 1").to_matrix(), want, atol=1e-12, rtol=0)
    assert np.allclose(tsim.Circuit("R_XX(0.25) 0 1").to_matrix(), tsim.Circuit("R_XX(0.25) 0 1\nDETECTOR").to_matrix() / 2, atol=1e-12, rtol=0)
    assert np.allclose(tsim.Circuit("R_PAULI(-0.25) X0*X1").to_matrix(), want.conj(), atol=1e-12, rtol=0)


PI_OVER_EIGHT_ERRORS = {
    "R_PAULI(0.25) Z0*Z0": "R_PAULI target qubits must be distinct within a product, got repeated qubit 0 in 'SPP[R_PAULI(theta=0.25*pi)] Z0*Z0'.",
    "R_PAULI(-0.25) Z0*Z0": "R_PAULI target qubits must be distinct within a product, got repeated qubit 0 in 'SPP[R_PAULI(theta=-0.25*pi)] Z0*Z0'.",
    "R_PAULI(0.25) Z0*X0": "R_PAULI target qubits must be distinct within a product, got repeated qubit 0 in 'SPP[R_PAULI(theta=0.25*pi)] Z0*X0'.",
    "R_PAULI(0.25) " + "*".join(f"Z{i}" for i in range(65)): "R_PAULI supports at most 64 qubits per instruction, got 65.",
}

# Consecutive instructions of the same name, tag and arguments are one instruction to Tsim (Stim fuses
# them, across a loop's unrolling and SHIFT_COORDS but not TICK): R_PAULI's 64 qubits are counted per
# fused instruction and its message quotes the fused one.
FUSED_ERRORS = {
    "\n".join(f"R_ZZ(0.3) {2 * i} {2 * i + 1}" for i in range(33)): "R_PAULI supports at most 64 qubits per instruction, got 66.",
    "\n".join(f"R_XX(0.25) {2 * i} {2 * i + 1}" for i in range(33)): "R_PAULI supports at most 64 qubits per instruction, got 66.",
    "R_PAULI(0.3) " + " ".join(f"X{2 * i}*Z{2 * i + 1}" for i in range(33)): "R_PAULI supports at most 64 qubits per instruction, got 66.",
    "REPEAT 33 {\n    R_ZZ(0.3) 0 1\n}": "R_PAULI supports at most 64 qubits per instruction, got 66.",
    "R_ZZ(0.3) 0 1\nSHIFT_COORDS(1)\nR_ZZ(0.3) 0 1\nR_PAULI(0.3) X0*X0": "R_PAULI target qubits must be distinct within a product, got repeated qubit 0 in 'SPP[R_PAULI(theta=0.3*pi)] Z0*Z1 Z0*Z1 X0*X0'.",
    "R_ZZ(0.3) 0 1\nTICK\nR_PAULI(0.3) X0*X0": "R_PAULI target qubits must be distinct within a product, got repeated qubit 0 in 'SPP[R_PAULI(theta=0.3*pi)] X0*X0'.",
    "TPP X0*X1\nTPP Z0*X0": "SPP[T] X0*X1 Z0*X0 acted on an anti-Hermitian operator",
}


# A product's ``!``s and a rotation's ``SPP_DAG`` are kept, so they fuse as in Tsim (an SPP_DAG not with the
# SPP of the negated angle, an inverted product with its own angle) and the messages quote them.
X_RANGE = lambda a, b, bang="": " ".join(f"{bang}X{i}" for i in range(a, b))  # noqa: E731
KEPT_ERRORS = {
    "SPP_DAG[R_PAULI(theta=0.3*pi)] X0\nR_PAULI(-0.3) X0*X0": "R_PAULI target qubits must be distinct within a product, got repeated qubit 0 in 'SPP[R_PAULI(theta=-0.3*pi)] X0*X0'.",
    "R_PAULI(0.3) X0*!X0": "R_PAULI target qubits must be distinct within a product, got repeated qubit 0 in 'SPP[R_PAULI(theta=0.3*pi)] X0*!X0'.",
    "SPP[R_PAULI(theta=0.3*pi)] !X0*!X0": "R_PAULI target qubits must be distinct within a product, got repeated qubit 0 in 'SPP[R_PAULI(theta=0.3*pi)] !X0*!X0'.",
    "SPP_DAG[R_PAULI(theta=0.25*pi)] !Z1\nSPP_DAG[R_PAULI(theta=0.25*pi)] Z0*!X0": "R_PAULI target qubits must be distinct within a product, got repeated qubit 0 in 'SPP_DAG[R_PAULI(theta=0.25*pi)] !Z1 Z0*!X0'.",
    f"SPP[R_PAULI(theta=0.3*pi)] {X_RANGE(0, 40, '!')}\nR_PAULI(0.3) {X_RANGE(40, 70)}": "R_PAULI supports at most 64 qubits per instruction, got 70.",
    f"SPP_DAG[R_PAULI(theta=0.3*pi)] {X_RANGE(0, 40)}\nSPP_DAG[R_PAULI(theta=0.3*pi)] {X_RANGE(40, 70)}": "R_PAULI supports at most 64 qubits per instruction, got 70.",
    "TPP !X0*Y0": "SPP[T] !X0*Y0 acted on an anti-Hermitian operator",
    "TPP !X1\nTPP X0*Y0": "SPP[T] !X1 X0*Y0 acted on an anti-Hermitian operator",
    "TPP !X1\nTPP_DAG X0*Y0": "SPP_DAG[T] X0*Y0 acted on an anti-Hermitian operator",
}


@pytest.mark.parametrize("text", list(PI_OVER_EIGHT_ERRORS) + list(FUSED_ERRORS) + list(KEPT_ERRORS))
def test_r_pauli_errors_are_tsims(text):
    message = {**PI_OVER_EIGHT_ERRORS, **FUSED_ERRORS, **KEPT_ERRORS}[text]
    c = tsim.Circuit(text)
    for backend in ["own"] + (["pyzx_param"] if HAS_PYZX_PARAM else []):
        with pytest.raises(ValueError) as info:
            c.get_graph(backend=backend)
        assert str(info.value) == message
        with pytest.raises(ValueError) as info:
            zx.parse_stim_circuit(c, backend=backend)
        assert str(info.value) == message
    if importlib.util.find_spec("tsim") is not None:
        import tsim as real

        with pytest.raises(type(info.value)) as theirs:
            real.Circuit(text).get_graph()
        assert str(theirs.value) == message


def test_fused_instructions_within_the_limit_build():
    """Thirty-two ``R_ZZ`` lines are one instruction of 64 qubits: they build, as Tsim's do, and the
    fused instruction is read back as one."""
    from stabilizer_qec.tsim.zx.parse import tsim_instructions

    text = "\n".join(f"R_ZZ(0.3) {2 * i} {2 * i + 1}" for i in range(32))
    c = tsim.Circuit(text)
    products = [f"Z{2 * i}*Z{2 * i + 1}" for i in range(32)]
    assert [str(i) for i in tsim_instructions(c)] == ["SPP[R_PAULI(theta=0.3*pi)] " + " ".join(products)]
    assert c.get_graph(backend="own").num_vertices() == 288
    split = tsim.Circuit(text.replace("\nR_ZZ(0.3) 2 3", "\nTICK\nR_ZZ(0.3) 2 3"))
    assert [str(i) for i in tsim_instructions(split)] == ["SPP[R_PAULI(theta=0.3*pi)] Z0*Z1", "TICK", "SPP[R_PAULI(theta=0.3*pi)] " + " ".join(products[1:])]
    # R_ZZ and R_PAULI of one angle are one instruction; another angle (one float apart) is another.
    other = "R_ZZ(0.3) 0 1\nR_PAULI(0.3) X0*Z1\nR_ZZ(0.30000000000000004) 0 1"
    assert [str(i) for i in tsim_instructions(tsim.Circuit(other))] == ["SPP[R_PAULI(theta=0.3*pi)] Z0*Z1 X0*Z1", "SPP[R_PAULI(theta=0.30000000000000004*pi)] Z0*Z1"]
    if importlib.util.find_spec("tsim") is not None:
        import tsim as real

        assert record(real.Circuit(text).get_graph()) == record(c.get_graph(backend="own"))
        assert record(real.Circuit(other).get_graph()) == record(tsim.Circuit(other).get_graph(backend="own"))
        assert [str(i) for i in real.Circuit(other)._stim_circ] == [str(i) for i in tsim_instructions(tsim.Circuit(other))]


# Tsim's product rotations as written: what this package holds (a ``!`` a lower-case letter, a rotation's
# SPP_DAG its arguments in the other order, R_PAULI(+-0.25) +-pi/8 with further digits), Tsim's instruction
# read back from it, the printing and the inverse's printing (Tsim's, but a product a line).
KEPT_AS_WRITTEN = {
    "SPP_DAG[R_PAULI(theta=0.3*pi)] X0": ("I[R_PAULI(pauli=X, theta=-0.47123889803846897)] 0", ["SPP_DAG[R_PAULI(theta=0.3*pi)] X0"], "SPP_DAG[R_PAULI(theta=0.3*pi)] X0", "R_PAULI(0.3) X0"),
    "SPP[R_PAULI(theta=0.3*pi)] !X0*Y1": ("I[R_PAULI(theta=-0.47123889803846897, pauli=xY)] 0 1", ["SPP[R_PAULI(theta=0.3*pi)] !X0*Y1"], "SPP[R_PAULI(theta=0.3*pi)] !X0*Y1", "SPP_DAG[R_PAULI(theta=0.3*pi)] Y1*!X0"),
    "R_PAULI(0.3) X0*!X1": ("I[R_PAULI(theta=-0.47123889803846897, pauli=Xx)] 0 1", ["SPP[R_PAULI(theta=0.3*pi)] X0*!X1"], "R_PAULI(0.3) X0*!X1", "SPP_DAG[R_PAULI(theta=0.3*pi)] !X1*X0"),
    "TPP !X0": ("I[R_PAULI(theta=-0.39269908169872414, pauli=x)] 0", ["SPP[T] !X0"], "TPP !X0", "TPP_DAG !X0"),
    "SPP_DAG[R_PAULI(theta=0.25*pi)] !Z0": ("I[R_PAULI(pauli=z, theta=0.392699081698724139)] 0", ["SPP_DAG[R_PAULI(theta=0.25*pi)] !Z0"], "SPP_DAG[R_PAULI(theta=0.25*pi)] !Z0", "SPP[R_PAULI(theta=0.25*pi)] !Z0"),
    "SPP_DAG[R_PAULI(theta=-0.0*pi)] X0": ("I[R_PAULI(pauli=X, theta=0.0)] 0", ["SPP_DAG[R_PAULI(theta=-0.0*pi)] X0"], "SPP_DAG[R_PAULI(theta=-0.0*pi)] X0", "R_PAULI(-0.0) X0"),
    "SPP[T] x0 * Y1": ("I[R_PAULI(theta=0.39269908169872414, pauli=XY)] 0 1", ["SPP[T] X0*Y1"], "TPP X0*Y1", "TPP_DAG Y1*X0"),
    "TPP_DAG !X0 X1": ("I[R_PAULI(theta=0.39269908169872414, pauli=x)] 0\nI[R_PAULI(theta=-0.39269908169872414, pauli=X)] 1", ["SPP_DAG[T] !X0 X1"], "TPP_DAG !X0\nTPP_DAG X1", "TPP X1\nTPP !X0"),
}


@pytest.mark.parametrize("text", list(KEPT_AS_WRITTEN))
def test_a_daggered_or_inverted_product_is_kept_as_written(text):
    """A rotation's ``SPP_DAG`` and a product's ``!``s are kept: read back as Tsim's instruction, printed
    as Tsim prints it (and read again as the same circuit), inverted as Stim inverts it (``SPP`` and
    ``SPP_DAG`` exchanged, the angle as written)."""
    from stabilizer_qec.tsim.zx.parse import tsim_instructions

    held, read, printed, inverse = KEPT_AS_WRITTEN[text]
    c = tsim.Circuit(text)
    assert c._stim_exact_text() == held
    assert [str(i) for i in tsim_instructions(c)] == read
    assert str(c) == printed and str(c.inverse()) == inverse
    assert tsim.Circuit(str(c)) == c and c.inverse().inverse() == c
    assert record(tsim.Circuit(str(c)).get_graph(backend="own")) == record(c.get_graph(backend="own"))
    assert np.allclose(c.inverse().to_matrix(), c.to_matrix().conj().T, atol=1e-12, rtol=0)
    if importlib.util.find_spec("tsim") is not None:
        import tsim as real

        ref = real.Circuit(text)
        assert [str(i) for i in ref._stim_circ] == read
        assert str(ref) == printed or "\n" in printed  # a product a line here (PRINTED_HERE)
        for backend in ["own"] + (["pyzx_param"] if HAS_PYZX_PARAM else []):
            assert record(ref.get_graph()) == record(c.get_graph(backend=backend)), backend
            assert record(ref.inverse().get_graph()) == record(c.inverse().get_graph(backend=backend)), backend
        assert record(real.Circuit(printed).get_graph()) == record(ref.get_graph())
        assert ref.tcount() == c.tcount()
        assert_matrix_is_tsims(c, ref)
        assert_matrix_is_tsims(c.inverse(), ref.inverse())


@pytest.mark.parametrize("gate, untagged, targets", [("SPP[T]", "SPP", "X0*"), ("SPP[T]", "SPP", "*X0"), ("TPP", "SPP", "X0*"), ("SPP_DAG[R_PAULI(theta=0.3*pi)]", "SPP_DAG", "X0*"), ("SPP[T]", "SPP", "2"), ("SPP[R_PAULI(theta=0.3*pi)]", "SPP", "X0 2")])
def test_a_product_that_does_not_parse_is_refused_as_spps(gate, untagged, targets):
    """A tagged product whose targets do not parse is refused as this package's parser refuses the
    untagged gate on them (a ``ValueError`` naming the line; Tsim's is Stim's ``ValueError``), not with
    an ``IndexError``."""
    with pytest.raises(ValueError) as plain:
        sq.Circuit(f"{untagged} {targets}")
    with pytest.raises(ValueError) as tagged:
        tsim.Circuit(f"{gate} {targets}")
    assert str(tagged.value) == str(plain.value) and str(tagged.value).startswith("line 1: ")
    if importlib.util.find_spec("tsim") is not None:
        import tsim as real

        with pytest.raises(ValueError):
            real.Circuit(f"{gate} {targets}")


def test_a_daggered_rotation_fuses_as_in_tsim():
    """``SPP_DAG[R_PAULI(theta=0.3*pi)]`` does not fuse with ``R_PAULI(-0.3)`` (the review's 70 qubits
    build, as in Tsim), and an inverted product fuses with its own angle, not the negated one."""
    from stabilizer_qec.tsim.zx.parse import tsim_instructions

    apart = f"SPP_DAG[R_PAULI(theta=0.3*pi)] {X_RANGE(0, 40)}\nR_PAULI(-0.3) {X_RANGE(40, 70)}"
    c = tsim.Circuit(apart)
    assert [str(i) for i in tsim_instructions(c)] == [f"SPP_DAG[R_PAULI(theta=0.3*pi)] {X_RANGE(0, 40)}", f"SPP[R_PAULI(theta=-0.3*pi)] {X_RANGE(40, 70)}"]
    assert c.get_graph(backend="own").num_vertices() == 210 and c.tcount() == 70
    together = "SPP[R_PAULI(theta=0.3*pi)] !X0\nR_PAULI(0.3) X1\nR_PAULI(-0.3) X2\nSPP_DAG[R_PAULI(theta=0.3*pi)] X3\nSPP_DAG[R_PAULI(theta=0.3*pi)] !Y4"
    fused = ["SPP[R_PAULI(theta=0.3*pi)] !X0 X1", "SPP[R_PAULI(theta=-0.3*pi)] X2", "SPP_DAG[R_PAULI(theta=0.3*pi)] X3 !Y4"]
    assert [str(i) for i in tsim_instructions(tsim.Circuit(together))] == fused
    if importlib.util.find_spec("tsim") is not None:
        import tsim as real

        for text in (apart, together):
            assert record(real.Circuit(text).get_graph()) == record(tsim.Circuit(text).get_graph(backend="own"))
            assert [str(i) for i in real.Circuit(text)._stim_circ] == [str(i) for i in tsim_instructions(tsim.Circuit(text))]


def test_a_native_pair_rotation_by_pi_over_eight_prints_as_the_tpp_it_is():
    """This package's own ``II[R_ZZ(theta=pi/8)]`` (pi/8 as ``repr`` writes it) is ``TPP`` to the ZX graph:
    it prints as ``TPP Z0*Z1``, which reads back to the same graph and matrix. With further digits it
    is ``R_ZZ(0.25)``, Tsim's R_PAULI, both ways."""
    cases = {
        "II[R_ZZ(theta=0.39269908169872414)] 0 1\nM 0": ("TPP Z0*Z1\nM 0", Fraction(0)),
        "II[R_XX(theta=-0.39269908169872414)] 0 1 2 3\nM 0": ("TPP_DAG X0*X1\nTPP_DAG X2*X3\nM 0", Fraction(0)),
        "II[R_ZZ(theta=0.392699081698724139)] 0 1\nM 0": ("R_ZZ(0.25) 0 1\nM 0", Fraction(15, 8)),
    }
    for native, (printed, scalar) in cases.items():
        c = tsim.Circuit(sq.Circuit(native))
        assert str(c) == printed
        again = tsim.Circuit(str(c))
        assert record(again.get_graph(backend="own")) == record(c.get_graph(backend="own"))
        assert record(c.get_graph(backend="own"))["scalar"][1] == scalar
        assert np.allclose(again.to_matrix(), c.to_matrix(), atol=1e-12, rtol=0)
        if importlib.util.find_spec("tsim") is not None:
            import tsim as real

            assert record(real.Circuit(printed).get_graph()) == record(c.get_graph(backend="own"))


# Unmeasured, the unitary engine would give a matrix of these; Tsim's builders refuse them.
REFUSED_BY_THE_BUILDERS = {
    "R_PAULI(0.3) Z0*Z0": (ValueError, "R_PAULI target qubits must be distinct within a product, got repeated qubit 0 in 'SPP[R_PAULI(theta=0.3*pi)] Z0*Z0'."),
    "R_PAULI(0.25) Z0*Z0": (ValueError, "R_PAULI target qubits must be distinct within a product, got repeated qubit 0 in 'SPP[R_PAULI(theta=0.25*pi)] Z0*Z0'."),
    "R_PAULI(0.3) X0*!X0": (ValueError, "R_PAULI target qubits must be distinct within a product, got repeated qubit 0 in 'SPP[R_PAULI(theta=0.3*pi)] X0*!X0'."),
    "H 1\nTPP X0*Y0": (ValueError, "SPP[T] X0*Y0 acted on an anti-Hermitian operator"),
    "SPP X0*Y0": (ValueError, "SPP X0*Y0 acted on an anti-Hermitian operator"),
    "SPP_DAG X1*Y1": (ValueError, "SPP_DAG X1*Y1 acted on an anti-Hermitian operator"),
    "MPP X0*Y0": (ValueError, "MPP X0*Y0 acted on an anti-Hermitian operator"),
    "H 0\nCX sweep[0] 0": (NotImplementedError, "Sweep bit targets (e.g. sweep[N]) are not supported in instruction 'CX sweep[0] 0'"),
    "R_PAULI(0.3) " + X_RANGE(0, 65): (ValueError, "R_PAULI supports at most 64 qubits per instruction, got 65."),
    "\n".join(f"R_ZZ(0.3) {2 * i} {2 * i + 1}" for i in range(33)): (ValueError, "R_PAULI supports at most 64 qubits per instruction, got 66."),
}


@pytest.mark.parametrize("text", list(REFUSED_BY_THE_BUILDERS))
def test_to_matrix_raises_what_tsims_builders_raise(text):
    kind, message = REFUSED_BY_THE_BUILDERS[text]
    c = tsim.Circuit(text)
    for method in (c.to_matrix, c.to_tensor):
        with pytest.raises(kind) as info:
            method()
        assert str(info.value) == message
    if importlib.util.find_spec("tsim") is not None:
        import tsim as real

        with pytest.raises(kind) as theirs:
            real.Circuit(text).to_matrix()
        assert str(theirs.value) == message


def test_to_matrix_of_a_product_naming_a_qubit_twice_is_tsims():
    """A product that cancels on a qubit no other instruction names leaves it out (``TPP Z0*Z0`` is
    [[1]]), and one naming a qubit twice keeps its sign (``TPP X0*Y0*X0`` is ``TPP_DAG Y0``, X Y X being
    -Y): Tsim's ZX tensor, where the unitary engine would keep the qubit and drop the sign."""
    h = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    y = np.array([[0, -1j], [1j, 0]])
    tpp_dag_y = np.exp(-1j * np.pi / 8) * (np.cos(np.pi / 8) * np.eye(2) + 1j * np.sin(np.pi / 8) * y)
    cases = {
        "TPP Z0*Z0": np.array([[1]]),
        "TPP !Z0*Z0": np.array([[1]]),
        "SPP_DAG Y0*Y0": np.array([[1]]),
        "TPP Z0*Z0\nH 1": h,
        "TPP X0*Y0*X0": tpp_dag_y,
        "H 0\nTPP X0*Y0*X0": tpp_dag_y @ h,
    }
    for text, want in cases.items():
        got = tsim.Circuit(text).to_matrix()
        assert got.shape == want.shape and np.allclose(got, want, atol=1e-12, rtol=0), text
        if importlib.util.find_spec("tsim") is not None:
            import tsim as real

            assert_matrix_is_tsims(tsim.Circuit(text), real.Circuit(text))


def test_the_detector_error_model_reads_a_product_as_tsims_stim_does():
    """Tsim's model is Stim's of its tagged circuit: a product rotation is its instruction's Clifford
    gate on its product, ``SPP_DAG`` and the ``!``s as written."""
    from stabilizer_qec.tsim._native import stim_reading

    c = tsim.Circuit("SPP_DAG[R_PAULI(theta=0.3*pi)] !X0*Y1\nR_PAULI(-0.3) Z2\nTPP !X3\nTPP_DAG Y4")
    assert stim_reading(c._stim_exact_text()) == "SPP_DAG !X0*Y1\nSPP Z2\nSPP !X3\nSPP_DAG Y4"
    text = "R 0 1\nH 0\nSPP_DAG[R_PAULI(theta=0.3*pi)] !X0*X1\nX_ERROR(0.1) 0\nTPP !Z1\nDEPOLARIZE1(0.05) 1\nM 0 1\nDETECTOR rec[-1] rec[-2]"
    model = tsim.Circuit(text).detector_error_model(allow_gauge_detectors=True)
    if importlib.util.find_spec("tsim") is not None:
        import tsim as real

        assert str(model) == str(real.Circuit(text).detector_error_model(allow_gauge_detectors=True))


# Printed here as written where Tsim's shorthand differs (the module's "Where it differs"): same-angle
# product rotations in a row stay lines of their own.
PRINTED_HERE = {
    "H 0 1\nR_PAULI(0.25) X0*Z1 Y1\nR_PAULI(0.25) Z0\nMPP X0*X1\nDETECTOR rec[-1]": "H 0 1\nR_PAULI(0.25) X0*Z1\nR_PAULI(0.25) Y1\nR_PAULI(0.25) Z0\nMPP X0*X1\nDETECTOR rec[-1]",
    "R_ZZ(0.3) 0 1\nR_ZZ(0.3) 1 2\nM 0": "R_ZZ(0.3) 0 1\nR_ZZ(0.3) 1 2\nM 0",
}


@pytest.mark.parametrize(
    "text",
    [
        "R_PAULI(0.25) X0*Z1\nM 0",
        "H 0\nR_PAULI(0.25) X0*Z1\nM 0",
        "H 0\nR_XX(0.25) 0 1\nM 0 1",
        "R_PAULI(-0.25) Z0\nM 0",
        "TPP X0*Z1\nM 0",
        "TPP_DAG X0\nR_PAULI(-0.25) X1\nM 0",
        "R_PAULI(0.25) X0*Y1",
        "R_YY(-0.25) 0 1\nR_ZZ(0.25) 1 2\nTPP X0*Z2",
        "R_PAULI(-0.25) X0*Z1\nR_Z(0.25) 1\nDETECTOR",
        "SPP_DAG[R_PAULI(theta=0.25*pi)] X0\nM 0",  # printed as Tsim prints it
        "SPP[R_PAULI(theta=-0.25*pi)] !X0*Z1\nSPP_DAG[R_PAULI(theta=0.25*pi)] Y1",
    ]
    + list(PRINTED_HERE),
)
def test_pi_over_eight_product_rotations_against_tsim(text):
    """Tsim's graphs, sampling graphs, matrix, T-count, printing and inverse."""
    real = pytest.importorskip("tsim")
    ref, mine = real.Circuit(text), tsim.Circuit(text)
    assert str(mine) == PRINTED_HERE.get(text, str(ref))
    for backend in ["own"] + (["pyzx_param"] if HAS_PYZX_PARAM else []):
        assert record(ref.get_graph()) == record(mine.get_graph(backend=backend)), backend
        for sample in (False, True):
            assert record(ref.get_sampling_graph(sample)) == record(mine.get_sampling_graph(sample, backend=backend)), (backend, sample)
    assert ref.tcount() == mine.tcount()
    assert_matrix_is_tsims(mine, ref)
    if "M" not in text and "DETECTOR" not in text:  # Tsim has no inverse of either
        assert record(ref.inverse().get_graph()) == record(mine.inverse().get_graph(backend="own"))
        assert_matrix_is_tsims(mine.inverse(), ref.inverse())


# Tsim's printing of the inverse: Stim inverts a product rotation's SPP to SPP_DAG, its tag (the angle as
# written) untouched.
INVERSES = {
    "R_PAULI(0.25) X0*Z1": "SPP_DAG[R_PAULI(theta=0.25*pi)] Z1*X0",
    "R_XX(0.25) 0 1\nR_Z(1.9967044602602857) 0": "R_Z(-1.9967044602602857) 0\nSPP_DAG[R_PAULI(theta=0.25*pi)] X1*X0",
    "U3(0.1, 0.2, 0.3) 0\nTPP X0*Z1\nR_PAULI(-0.25) Y1": "SPP_DAG[R_PAULI(theta=-0.25*pi)] Y1\nTPP_DAG Z1*X0\nU3(-0.1, -0.3, -0.2) 0",
    "REPEAT 2 {\n    R_PAULI(0.3) X0*Z1*Y2\n    T 0\n}": "REPEAT 2 {\n    T_DAG 0\n    SPP_DAG[R_PAULI(theta=0.3*pi)] Y2*Z1*X0\n}",
    "R_ZZ(0.3) 0 1\nR_Y(-0.7) 2\nU3(0.25, 0.5, -0.125) 1": "U3(-0.25, 0.125, -0.5) 1\nR_Y(0.7) 2\nSPP_DAG[R_PAULI(theta=0.3*pi)] Z1*Z0",
    "R_X(0.10044109572818205) 0\nR_PAULI(-1.4074629084552872) X0*Y1": "SPP_DAG[R_PAULI(theta=-1.4074629084552872*pi)] Y1*X0\nR_X(-0.10044109572818205) 0",
    # A zero is 0.0, as Tsim's float(-Fraction(0)).
    "R_Z(0.0) 0\nU3(0, -0.0, 0.5) 1\nR_X(-0.0) 2": "R_X(0.0) 2\nU3(0.0, -0.5, 0.0) 1\nR_Z(0.0) 0",
}
INVERTED_TWICE = {"R_Z(0.0) 0\nU3(0, -0.0, 0.5) 1\nR_X(-0.0) 2": "R_Z(0.0) 0\nU3(0.0, 0.0, 0.5) 1\nR_X(0.0) 2"}


@pytest.mark.parametrize("text", list(INVERSES))
def test_inverse_of_rotations_is_tsims(text):
    """The inverse negates each rotation's half-turns as written (a ``U3``'s are -theta, -lambda, -phi)
    and keeps ``R_PAULI(+-0.25)`` an ``R_PAULI``: Tsim's graph of its inverse, and the matrix's adjoint."""
    c = tsim.Circuit(text)
    inverse = c.inverse()
    assert str(inverse) == INVERSES[text]
    assert str(inverse.inverse()) == INVERTED_TWICE.get(text, text)
    assert np.allclose(inverse.to_matrix(), c.to_matrix().conj().T, atol=1e-12, rtol=0)
    if importlib.util.find_spec("tsim") is not None:
        import tsim as real

        theirs = real.Circuit(text).inverse()
        assert str(theirs) == INVERSES[text]
        assert record(theirs.get_graph()) == record(inverse.get_graph(backend="own"))
        assert_matrix_is_tsims(inverse, theirs)


# Literals whose radians are the float of a neighbouring number of half-turns: read back as written.
NEIGHBOURED_LITERALS = ["1.9967044602602857", "1.4074629084552872", "0.10044109572818205", "-0.17214200890901177"]


def test_angle_literals_are_kept_as_written():
    """A rotation's half-turns are the literal as written (``repr`` of its float), as in Tsim: printed,
    read back and in the graph (its phase the literal's Fraction), where several half-turns give the
    same radians."""
    from stabilizer_qec.tsim.zx.parse import tsim_instructions

    rng = np.random.default_rng(3)
    literals = NEIGHBOURED_LITERALS + [repr(float(rng.uniform(-3, 3))) for _ in range(150)] + [repr(float(10 ** rng.uniform(-12, 12))) for _ in range(50)]
    for x in literals:
        text = f"R_Z({x}) 0\nR_PAULI({x}) X0*Z1\nU3({x}, 0.5, {x}) 1\nR_YY({x}) 0 1"
        c = tsim.Circuit(text)
        assert str(c) == text and str(c.copy()) == text and str(tsim.Circuit(str(c))) == text
        assert [i.tag for i in tsim_instructions(c)] == [f"R_Z(theta={x}*pi)", f"R_PAULI(theta={x}*pi)", f"U3(theta={x}*pi, phi=0.5*pi, lambda={x}*pi)", f"R_PAULI(theta={x}*pi)"]
        g = record(tsim.Circuit(f"R_Z({x}) 0").get_graph(backend="own"))
        assert [v[2] for v in g["vertices"]] == [0, Fraction(x) % 2, 0] and g["scalar"][1] == (-Fraction(x) / 2) % 2, x
    # The review's example: Tsim's 19967044602602857/10^16, not 998352230130143/(5*10^14).
    g = tsim.Circuit("R_Z(1.9967044602602857) 0").get_graph(backend="own")
    assert record(g)["vertices"][1][2] == Fraction(19967044602602857, 10**16)


@pytest.mark.parametrize("seed", range(8))
def test_angle_literals_against_tsim(seed):
    real = pytest.importorskip("tsim")
    rng = np.random.default_rng(2000 + seed)

    def literal():
        if rng.random() < 0.7:
            return repr(float(rng.uniform(-3, 3)))
        return repr(float(10 ** rng.uniform(-12, 12)) * (1 if rng.random() < 0.5 else -1))

    for _ in range(25):
        x, y, z = literal(), literal(), literal()
        for text in (f"R_Z({x}) 0", f"R_X({x}) 1\nR_Y({y}) 0", f"U3({x}, {y}, {z}) 0", f"R_PAULI({x}) X0*Z1*Y2", f"R_YY({x}) 0 1\nR_XX({y}) 1 2\nR_ZZ({z}) 2 0", f"R_PAULI({x}) X0*Z1 Y2"):
            ref, mine = real.Circuit(text), tsim.Circuit(text)
            if not text.endswith(" Y2"):  # two products, two lines here (PRINTED_HERE)
                assert str(ref) == str(mine), text
            assert record(ref.get_graph()) == record(mine.get_graph(backend="own")), text
            assert record(ref.inverse().get_graph()) == record(mine.inverse().get_graph(backend="own")), text
            if not text.endswith(" Y2"):
                assert str(ref.inverse()) == str(mine.inverse()), text


def test_zx_graph_phases_are_exact_fractions():
    c = tsim.Circuit("R_Z(0.3) 0\nR_X(0.1) 1\nR_PAULI(0.345) X0*Y1\nU3(0.3, 0.24, 0.49) 0")
    g = c.get_graph(backend="own")
    phases = {Fraction(g.phase(v)) for v in g.vertices()}
    assert Fraction(3, 10) in phases and Fraction(1, 10) in phases and Fraction(69, 200) in phases
    assert Fraction(6, 25) in phases and Fraction(49, 100) in phases
    assert all(isinstance(g.phase(v), Fraction) for v in g.vertices())


# Random circuits against Tsim itself (installed: bloqade-tsim 0.1.5 with pyzx_param).
def random_circuit(rng, n: int, length: int) -> str:
    one = ["H", "S", "S_DAG", "X", "Y", "Z", "SQRT_X", "SQRT_Y_DAG", "C_XYZ", "C_NXYZ", "H_YZ", "H_NXY", "T", "T_DAG"]
    two = ["CX", "CY", "CZ", "SWAP", "ISWAP", "CXSWAP", "SQRT_XX", "SQRT_YY_DAG", "XCZ", "YCX"]
    lines = []
    measured = 0
    for _ in range(length):
        u, q = rng.random(), int(rng.integers(n))
        a, b = (int(x) for x in rng.choice(n, 2, replace=False)) if n > 1 else (0, 0)
        if u < 0.25:
            lines.append(f"{rng.choice(one)} {q}")
        elif u < 0.4 and n > 1:
            lines.append(f"{rng.choice(two)} {a} {b}")
        elif u < 0.48:
            lines.append(f"R_{rng.choice(list('XYZ'))}({round(float(rng.uniform(-2, 2)), 4)}) {q}")
        elif u < 0.52:
            lines.append(f"U3({round(float(rng.uniform(0, 2)), 3)}, {round(float(rng.uniform(0, 2)), 3)}, {round(float(rng.uniform(0, 2)), 3)}) {q}")
        elif u < 0.58 and n > 1:
            lines.append(f"{rng.choice(['TPP', 'TPP_DAG', 'SPP', 'SPP_DAG', 'R_PAULI(0.25)', 'R_PAULI(-0.25)'])} {rng.choice(list('XYZ'))}{a}*{rng.choice(list('XYZ'))}{b}")
        elif u < 0.62 and n > 1:
            lines.append(f"R_{rng.choice(list('XYZ')) * 2}({round(float(rng.uniform(-1, 1)), 3)}) {a} {b}")
        elif u < 0.72:
            lines.append(f"{rng.choice(['M', 'MX', 'MY', 'MR', 'MRX', 'MRY'])}{rng.choice(['', '(0.05)'])} {'!' if rng.random() < 0.2 else ''}{q}")
            measured += 1
        elif u < 0.77:
            lines.append(f"{rng.choice(['R', 'RX', 'RY'])} {q}")
        elif u < 0.81 and n > 1:
            lines.append(f"MPP {rng.choice(list('XYZ'))}{a}*{rng.choice(list('XYZ'))}{b}")
            measured += 1
        elif u < 0.84:
            lines.append(f"MPAD {int(rng.integers(2))}")
            measured += 1
        elif u < 0.88 and measured and n > 1:
            lines.append(f"{rng.choice(['CX', 'CZ', 'CY'])} rec[-{int(rng.integers(1, measured + 1))}] {q if q != a else b}")
        elif u < 0.93:
            lines.append(f"{rng.choice(['X_ERROR', 'Y_ERROR', 'Z_ERROR', 'DEPOLARIZE1', 'HERALDED_ERASE'])}({round(float(rng.uniform(0, 0.3)), 3)}) {q}")
            if lines[-1].startswith("HERALDED"):
                measured += 1
        elif u < 0.95 and n > 1:
            lines.append(f"DEPOLARIZE2(0.05) {a} {b}")
        elif u < 0.97 and n > 1:
            lines.append(f"E(0.1) X{a} Z{b}\nELSE_CORRELATED_ERROR(0.2) Y{a}")
        elif measured:
            ks = sorted({int(x) for x in rng.integers(1, measured + 1, size=int(rng.integers(0, 3)))})
            targets = " ".join(f"rec[-{k}]" for k in ks)
            lines.append(f"DETECTOR {targets}" if rng.random() < 0.5 else f"OBSERVABLE_INCLUDE({int(rng.integers(2))}) {targets}")
    return "\n".join(lines)


@pytest.mark.parametrize("seed", range(40))
def test_random_circuits_against_tsim(seed):
    real = pytest.importorskip("tsim")
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 4))
    text = random_circuit(rng, n, int(rng.integers(3, 20)))
    ref, mine = real.Circuit(text), tsim.Circuit(text)
    backends = ["own"] + (["pyzx_param"] if HAS_PYZX_PARAM else [])
    for backend in backends:
        for sample in (None, False, True):
            if sample is None:
                a, b = ref.get_graph(), mine.get_graph(backend=backend)
            else:
                a, b = ref.get_sampling_graph(sample), mine.get_sampling_graph(sample, backend=backend)
            assert record(a) == record(b), (backend, sample, text)
    assert ref.tcount() == mine.tcount(), text
    assert_matrix_is_tsims(mine, ref)


def random_unmeasured_annotated_circuit(rng, n: int, length: int) -> str:
    """A circuit of gates, rotations (``TPP`` and the ±0.25 ``R_PAULI`` among them), noise and resets,
    no measurement, with at least one detector or observable that XORs nothing."""
    one = ["H", "S", "S_DAG", "X", "SQRT_X", "C_XYZ", "H_YZ", "T", "T_DAG"]
    two = ["CX", "CY", "CZ", "SWAP", "ISWAP", "SQRT_XX", "XCZ"]
    annotations = ["DETECTOR", "DETECTOR(1, 2)", "OBSERVABLE_INCLUDE(0)", "OBSERVABLE_INCLUDE(1)"]
    lines = [str(rng.choice(annotations))]
    for _ in range(length):
        u, q = rng.random(), int(rng.integers(n))
        a, b = (int(x) for x in rng.choice(n, 2, replace=False)) if n > 1 else (0, 0)
        if u < 0.3:
            lines.append(f"{rng.choice(one)} {q}")
        elif u < 0.45 and n > 1:
            lines.append(f"{rng.choice(two)} {a} {b}")
        elif u < 0.55:
            lines.append(f"R_{rng.choice(list('XYZ'))}({round(float(rng.uniform(-2, 2)), 4)}) {q}")
        elif u < 0.6 and n > 1:
            lines.append(f"{rng.choice(['TPP', 'SPP_DAG', 'R_PAULI(0.25)', 'R_PAULI(-0.25)'])} {rng.choice(list('XYZ'))}{a}*{rng.choice(list('XYZ'))}{b}")
        elif u < 0.62 and n > 1:
            lines.append(f"R_{rng.choice(list('XYZ')) * 2}({rng.choice(['0.25', '-0.25'])}) {a} {b}")
        elif u < 0.68:
            lines.append(f"{rng.choice(['R', 'RX', 'RY'])} {q}")
        elif u < 0.76:
            lines.append(f"{rng.choice(['X_ERROR', 'Z_ERROR', 'DEPOLARIZE1'])}({round(float(rng.uniform(0, 0.3)), 3)}) {q}")
        elif u < 0.8:
            lines.append(f"QUBIT_COORDS(1, 2) {q}\nTICK")
        else:
            lines.append(str(rng.choice(annotations)))
    order = rng.permutation(len(lines))
    return "\n".join(lines[i] for i in order)


@pytest.mark.parametrize("seed", range(20))
def test_random_unmeasured_annotated_circuits_against_tsim(seed):
    """Without a measurement, a detector or observable still makes ``to_matrix`` Tsim's ZX tensor:
    equal to Tsim's, not up to a phase or a factor."""
    real = pytest.importorskip("tsim")
    rng = np.random.default_rng(1000 + seed)
    text = random_unmeasured_annotated_circuit(rng, int(rng.integers(1, 4)), int(rng.integers(2, 12)))
    want = np.asarray(real.Circuit(text).to_matrix())
    got = tsim.Circuit(text).to_matrix()
    assert got.shape == want.shape and np.allclose(got, want, atol=1e-9, rtol=0), text


PRODUCT_ANGLES = ["0.3", "-0.3", "0.25", "-0.25", "0.5", "0.0", "-0.0", "1.9967044602602857", "0.30000000000000004"]


def random_product_circuit(rng, n: int, length: int, distinct: bool, measure: bool) -> str:
    """Product rotations as Tsim writes them: ``SPP``/``SPP_DAG`` of ``R_PAULI`` and ``T`` in its tagged
    Stim, ``R_PAULI``, ``TPP``, ``TPP_DAG`` and ``R_PP`` in its shorthand, with ``!``s on factors (not
    the first in the shorthand, which Tsim does not read), a few angles (so that lines fuse), a qubit
    named twice in a product unless ``distinct``; gates, ``TICK`` and, if ``measure``, measurements
    and detectors between."""

    def product(lead: bool) -> str:
        k = int(rng.integers(1, 4))
        qubits = [int(q) for q in rng.choice(n, min(k, n), replace=False)] if distinct else [int(rng.integers(n)) for _ in range(k)]
        return "*".join(f"{'!' if rng.random() < 0.3 and (lead or j) else ''}{rng.choice(list('XYZ'))}{q}" for j, q in enumerate(qubits))

    def products(lead: bool = True) -> str:
        return " ".join(product(lead) for _ in range(int(rng.integers(1, 3))))

    lines, measured = [], 0
    for _ in range(length):
        u, a = rng.random(), str(rng.choice(PRODUCT_ANGLES))
        if u < 0.2:
            lines.append(f"{rng.choice(['SPP', 'SPP_DAG'])}[R_PAULI(theta={a}*pi)] {products()}")
        elif u < 0.35:
            lines.append(f"R_PAULI({a}) {products(lead=False)}")
        elif u < 0.5:
            lines.append(f"{rng.choice(['TPP', 'TPP_DAG', 'SPP', 'SPP_DAG'])} {products()}")
        elif u < 0.58 and n > 1:
            q0, q1 = (int(x) for x in rng.choice(n, 2, replace=False))
            lines.append(f"R_{rng.choice(list('XYZ')) * 2}({a}) {q0} {q1}")
        elif u < 0.7:
            lines.append(f"{rng.choice(['H', 'S', 'T', 'T_DAG', 'SQRT_X'])} {int(rng.integers(n))}")
        elif u < 0.75:
            lines.append("TICK")
        elif u < 0.8 and n > 1:
            q0, q1 = (int(x) for x in rng.choice(n, 2, replace=False))
            lines.append(f"CX {q0} {q1}")
        elif measure and u < 0.9:
            lines.append(f"M {int(rng.integers(n))}")
            measured += 1
        elif measure and measured and u < 0.95:
            lines.append("DETECTOR rec[-1]")
    return "\n".join(lines)


def outcome(f):
    """``("ok", value)``, or the exception's type and message."""
    try:
        return ("ok", f())
    except Exception as exc:  # noqa: BLE001 - compared with Tsim's
        return (type(exc).__name__, str(exc))


@pytest.mark.parametrize("seed", range(40))
def test_random_product_rotations_against_tsim(seed):
    """Tsim's graphs (or its error, type and message), sampling graphs, T-count, matrix and inverse, and
    Tsim reading this package's printing as the same circuit; the model (an anti-Hermitian product's
    message is this package's Stim's, its type Stim's)."""
    real = pytest.importorskip("tsim")
    rng = np.random.default_rng(5000 + seed)
    n, measure = int(rng.integers(1, 4)), bool(rng.random() < 0.4)
    text = random_product_circuit(rng, n, int(rng.integers(1, 8)), distinct=seed % 2 == 0, measure=measure)
    ref, mine = real.Circuit(text), tsim.Circuit(text)
    for backend in ["own"] + (["pyzx_param"] if HAS_PYZX_PARAM else []):
        assert outcome(lambda: record(ref.get_graph())) == outcome(lambda: record(mine.get_graph(backend=backend))), (backend, text)
        for sample in (False, True):
            assert outcome(lambda: record(ref.get_sampling_graph(sample))) == outcome(lambda: record(mine.get_sampling_graph(sample, backend=backend))), (backend, sample, text)
    assert outcome(ref.tcount) == outcome(mine.tcount), text
    built = outcome(lambda: ref.to_matrix())
    if built[0] == "ok":
        assert_matrix_is_tsims(mine, ref)
    else:
        assert outcome(mine.to_matrix) == built, text
    assert tsim.Circuit(str(mine)) == mine, text
    assert outcome(lambda: record(real.Circuit(str(mine)).get_graph())) == outcome(lambda: record(ref.get_graph())), text
    theirs, ours = outcome(lambda: str(ref.detector_error_model(allow_gauge_detectors=True))), outcome(lambda: str(mine.detector_error_model(allow_gauge_detectors=True)))
    assert ours[0] == theirs[0] and (ours == theirs or "anti-Hermitian" in theirs[1]), text
    if not measure:
        ref_inverse, inverse = ref.inverse(), mine.inverse()
        assert outcome(lambda: record(ref_inverse.get_graph())) == outcome(lambda: record(inverse.get_graph(backend="own"))), text
        if outcome(lambda: ref_inverse.to_matrix())[0] == "ok":
            assert_matrix_is_tsims(inverse, ref_inverse)


# The stored forms the core reads (src/nonpauli.rs ``parse``) and the spelling this package writes of
# each: an ``R_PAULI``'s ``_`` letter is an identity, as ``I`` is; empty arguments (``theta=0.1,``) are
# skipped; spaces around the name, the brackets, the commas and the ``=`` are read past; ``T()`` is T.
STORED_SPELLINGS = {
    "I[R_PAULI(theta=0.1, pauli=X_Z)] 0 1 2": "I[R_PAULI(theta=0.1, pauli=XIZ)] 0 1 2",
    "I[R_PAULI(theta=0.1, pauli=x_Z)] 0 1 2": "I[R_PAULI(theta=0.1, pauli=xIZ)] 0 1 2",
    "I[R_PAULI(pauli=_Y_, theta=-0.3)] 0 1 2": "I[R_PAULI(pauli=IYI, theta=-0.3)] 0 1 2",
    "I[R_PAULI(theta=0.39269908169872414, pauli=_X)] 0 1": "I[R_PAULI(theta=0.39269908169872414, pauli=IX)] 0 1",
    "I[R_PAULI(theta=-0.39269908169872414, pauli=Z_y)] 2 0 1": "I[R_PAULI(theta=-0.39269908169872414, pauli=ZIy)] 2 0 1",
    "I[R_PAULI(theta=0.1, pauli=XZ,)] 0 1": "I[R_PAULI(theta=0.1, pauli=XZ)] 0 1",
    "I[R_PAULI(,pauli=XZ,, theta=0.1)] 0 1": "I[R_PAULI(pauli=XZ, theta=0.1)] 0 1",
    "I[ R_PAULI ( theta = +.1 , pauli = Xz ) ] 0 1": "I[R_PAULI(theta=+.1, pauli=Xz)] 0 1",
    "I[R_PAULI(theta=0.1, pauli=__)] 0 1": "I[R_PAULI(theta=0.1, pauli=II)] 0 1",
    "I[R_PAULI(theta=0.1, pauli=X_)] 0 0": "I[R_PAULI(theta=0.1, pauli=XI)] 0 0",
    "I[R_X(theta=0.1,)] 0": "I[R_X(theta=0.1)] 0",
    "I[U3(theta=0.1, phi=0.2, lambda=0.3,)] 1": "I[U3(theta=0.1, phi=0.2, lambda=0.3)] 1",
    "II[R_ZZ( theta=0.1 ,)] 0 1": "II[R_ZZ(theta=0.1)] 0 1",
    "I[T()] 0": "I[T] 0",
    "I[ T_DAG ( , ) ] 1": "I[T_DAG] 1",
    # ``R_PAULI`` on ``II`` is one product over its targets, as on ``I`` (``parse``'s R_PAULI arm takes
    # either gate), pairs of distinct qubits that may repeat across pairs.
    "II[R_PAULI(theta=0.1, pauli=XZ)] 0 1": "I[R_PAULI(theta=0.1, pauli=XZ)] 0 1",
    "II[R_PAULI(theta=0.1, pauli=X_Z_)] 0 1 2 0": "I[R_PAULI(theta=0.1, pauli=XIZI)] 0 1 2 0",
    "II[R_PAULI(pauli=yX, theta=-0.3)] 2 0": "I[R_PAULI(pauli=yX, theta=-0.3)] 2 0",
    "II[R_PAULI(theta=-0.39269908169872414, pauli=ZxY_)] 1 0 2 1": "I[R_PAULI(theta=-0.39269908169872414, pauli=ZxYI)] 1 0 2 1",
    "II[ R_PAULI ( theta = +.1 ,, pauli = XzyX ) ] 0 1 2 0": "I[R_PAULI(theta=+.1, pauli=XzyX)] 0 1 2 0",
    "II[R_PAULI(theta=0.1, pauli=__)] 0 1": "I[R_PAULI(theta=0.1, pauli=II)] 0 1",
}


def stored_circuits(line: str) -> tuple:
    """A stored line in an unmeasured circuit (for ``to_matrix`` and the inverse) and in a measured one
    with noise and detectors (for the model)."""
    unmeasured = f"H 0 1 2\n{line}\nS 0"
    measured = f"R 0 1 2\nH 1\nX_ERROR(0.1) 2\n{line}\nH 0\nM 0 1 2\nDETECTOR rec[-3]\nDETECTOR rec[-2]\nDETECTOR rec[-1]"
    return unmeasured, measured


def assert_read_as_spelled(stored: str, spelled: str) -> None:
    """Every reader reads ``stored`` (a circuit in this package's text) as ``spelled``: printing, Stim's
    reading for the model, the graphs, the T-count, ``to_matrix`` (bit for bit), the model and the
    inverse (its printing, graph and matrix), which undoes the circuit."""
    from stabilizer_qec.tsim._native import stim_reading

    def same_matrix(a, b) -> None:
        a, b = outcome(a.to_matrix), outcome(b.to_matrix)
        assert a[0] == b[0] and (np.array_equal(a[1], b[1]) if a[0] == "ok" else a == b), stored

    def same_print(a, b) -> None:
        # The core fuses a line with the next of the same tag text, so the spelling's lines can be fused
        # where the stored ones (``I[ T ] 0`` then ``I[T] 0``) are not: then they print alike once read back.
        assert str(a) == str(b) if len(a) == len(b) else str(tsim.Circuit(str(a))) == str(b), stored

    mine, theirs = tsim.Circuit(sq.Circuit(stored)), tsim.Circuit(sq.Circuit(spelled))
    printed = "I[R_PAULI" not in str(theirs)  # an all-identity product prints as stored
    if printed:
        same_print(mine, theirs)
    # Stim's reading, as the model reads it (a circuit, consecutive lines of one gate fused).
    assert str(sq.Circuit(stim_reading(mine._stim_exact_text()))) == str(sq.Circuit(stim_reading(theirs._stim_exact_text()))), stored
    for backend in ["own"] + (["pyzx_param"] if HAS_PYZX_PARAM else []):
        assert outcome(lambda: record(mine.get_graph(backend=backend))) == outcome(lambda: record(theirs.get_graph(backend=backend))), (backend, stored)
    assert outcome(mine.tcount) == outcome(theirs.tcount), stored
    same_matrix(mine, theirs)
    model = outcome(lambda: str(mine.detector_error_model(allow_gauge_detectors=True)))
    assert model == outcome(lambda: str(theirs.detector_error_model(allow_gauge_detectors=True))), stored
    a, b = outcome(mine.inverse), outcome(theirs.inverse)
    assert a[0] == b[0] and (a[0] == "ok" or a == b), stored
    undone = outcome((mine + a[1]).to_matrix) if a[0] == "ok" else ("no inverse", "")
    if undone[0] == "ok":
        u = undone[1]
        assert abs(abs(u.flat[0]) - 1) < 1e-9 and np.allclose(u, u.flat[0] * np.eye(len(u)), atol=1e-9, rtol=0), stored
    if a[0] == "ok" and len(mine) == len(theirs):
        if printed:
            same_print(a[1], b[1])
        assert outcome(lambda: record(a[1].get_graph(backend="own"))) == outcome(lambda: record(b[1].get_graph(backend="own"))), stored
        same_matrix(a[1], b[1])
    elif a[0] == "ok":  # the core inverts lines it did not fuse in another order (``T_DAG 0 1``, not ``1 0``)
        assert np.allclose(a[1].to_matrix(), b[1].to_matrix(), atol=1e-12, rtol=0), stored


@pytest.mark.parametrize("stored", list(STORED_SPELLINGS))
def test_a_stored_form_the_core_reads_is_read_as_its_spelling(stored):
    """Each stored form is read as the core reads it: as the spelling this package writes of it, and
    (Tsim reading this package's printing of it) as Tsim's instruction."""
    for mine, theirs in zip(stored_circuits(stored), stored_circuits(STORED_SPELLINGS[stored])):
        assert_read_as_spelled(mine, theirs)
        if importlib.util.find_spec("tsim") is not None and "pauli=__" not in stored:
            import tsim as real

            c = tsim.Circuit(sq.Circuit(mine))
            ref = real.Circuit(str(c))
            assert outcome(lambda: record(ref.get_graph())) == outcome(lambda: record(c.get_graph(backend="own"))), stored
            assert str(c.detector_error_model(allow_gauge_detectors=True)) == str(ref.detector_error_model(allow_gauge_detectors=True)), stored
            if outcome(ref.to_matrix)[0] == "ok":
                assert_matrix_is_tsims(c, ref)
            inverse = outcome(ref.inverse)
            if inverse[0] == "ok" and outcome(inverse[1].to_matrix)[0] == "ok":
                assert str(c.inverse()) == str(inverse[1]), stored
                assert record(c.inverse().get_graph(backend="own")) == record(inverse[1].get_graph()), stored
                assert_matrix_is_tsims(c.inverse(), inverse[1])


def test_an_ii_r_pauli_is_one_product_as_in_the_core():
    """The review's examples: the core reads ``II[R_PAULI(...)]`` as it reads ``I[R_PAULI(...)]``, one
    product rotation over its targets (``nonpauli::parse``'s R_PAULI arm takes either gate), so every
    reader does. The model is Tsim's of ``R_PAULI(...) X0*Z2`` (Stim's reading ``SPP X0*Z2``, not the
    identity); the print is Tsim's shorthand, the T-count 1, ``to_matrix`` the ``I`` spelling's and
    Tsim's; the inverse is Tsim's inverse of the print, and undoes the circuit."""
    from stabilizer_qec.tsim._native import stim_reading

    measured = "R 0 1 2 3\nX_ERROR(0.1) 2\n{}[R_PAULI(theta=0.1, pauli={})] 0 1 2 3\nM 0 1 2\nDETECTOR rec[-3]\nDETECTOR rec[-1]"
    pair, single = (tsim.Circuit(sq.Circuit(measured.format(gate, letters))) for gate, letters in (("II", "X_Z_"), ("I", "XIZI")))
    assert stim_reading(pair._stim_exact_text()) == stim_reading(single._stim_exact_text())
    assert "SPP X0*Z2" in stim_reading(pair._stim_exact_text()).split("\n")
    model = str(pair.detector_error_model(allow_gauge_detectors=True))
    assert model == str(single.detector_error_model(allow_gauge_detectors=True)) == "error(0.5) D0\nerror(0.1) D0 D1"
    a, b = (tsim.Circuit(sq.Circuit(f"H 0\n{gate}[R_PAULI(theta=0.1, pauli=XZ)] 0 1")) for gate in ("II", "I"))
    assert str(a) == str(b) == "H 0\nR_PAULI(0.06366197723675814) X0*Z1"
    assert a.tcount() == b.tcount() == 1
    assert record(a.get_graph(backend="own")) == record(b.get_graph(backend="own"))
    assert np.array_equal(a.to_matrix(), b.to_matrix())
    want, got = a._zx_matrix("big"), a.to_matrix()
    i = np.argmax(np.abs(want))
    assert got.shape == (4, 4) and np.allclose(got, got.flat[i] / want.flat[i] * want, atol=1e-12, rtol=0)
    four = tsim.Circuit(sq.Circuit("H 0 1 2 3\nII[R_PAULI(theta=0.1, pauli=XZYy)] 0 1 2 3"))
    assert str(four) == "H 0 1 2 3\nR_PAULI(-0.06366197723675814) X0*Z1*Y2*!Y3"
    # Inverted as I's (the targets reversed, not II's pairs): the core's inverse of the I spelling.
    assert four.inverse()._stim_exact_text() == "I[R_PAULI(pauli=yYZX, theta=-0.1)] 3 2 1 0\nH 3 2 1 0"
    assert str(four.inverse()) == "SPP_DAG[R_PAULI(theta=-0.06366197723675814*pi)] !Y3*Y2*Z1*X0\nH 3 2 1 0"
    undone = (four + four.inverse()).to_matrix()
    assert np.allclose(undone, undone[0, 0] * np.eye(16), atol=1e-12, rtol=0) and abs(abs(undone[0, 0]) - 1) < 1e-12
    if importlib.util.find_spec("tsim") is not None:
        import tsim as real

        ref = real.Circuit("R 0 1 2 3\nX_ERROR(0.1) 2\nR_PAULI(0.06366197723675814) X0*Z2\nM 0 1 2\nDETECTOR rec[-3]\nDETECTOR rec[-1]")
        assert model == str(ref.detector_error_model(allow_gauge_detectors=True))
        ref = real.Circuit("H 0\nR_PAULI(0.06366197723675814) X0*Z1")
        assert record(a.get_graph(backend="own")) == record(ref.get_graph())
        assert_matrix_is_tsims(a, ref)
        ref = real.Circuit(str(four)).inverse()
        assert str(four.inverse()) == str(ref)
        assert record(four.inverse().get_graph(backend="own")) == record(ref.get_graph())
        assert_matrix_is_tsims(four.inverse(), ref)


COHERENT_PRODUCT_ERRORS = [
    "I_ERROR[R_PAULI(theta=0.1, pauli=XZ)] 0 1",
    "II_ERROR[R_PAULI(theta=0.1, pauli=X_,)] 0 1",
    "I_ERROR[ R_PAULI (pauli=_zY, theta=0.39269908169872414)] 0 1 2",
    "II_ERROR[R_PAULI(theta=-0.3, pauli=XYZ_)] 0 1 2 0",
]


@pytest.mark.parametrize("line", COHERENT_PRODUCT_ERRORS)
def test_a_coherent_product_error_is_no_product_rotation(line):
    """``R_PAULI`` on ``I_ERROR`` or ``II_ERROR`` is a coherent error (``nonpauli::parse``), which no
    reader takes for a product rotation: it prints and stays in Stim's reading as written, and its
    graphs, T-count, matrix and model are those of the untagged noise (Tsim's, reading the print); the
    inverse of the unmeasured circuit is the core's error for noise."""
    from stabilizer_qec.tsim._native import stim_reading

    untagged = re.sub(r"\[.*\]", "", line)
    for stored, plain, why in zip(stored_circuits(line), stored_circuits(untagged), ("noise", "resets or measurements")):
        c, p = tsim.Circuit(sq.Circuit(stored)), tsim.Circuit(sq.Circuit(plain))
        assert line in str(c).split("\n") and line in stim_reading(c._stim_exact_text()).split("\n")
        for backend in ["own"] + (["pyzx_param"] if HAS_PYZX_PARAM else []):
            assert record(c.get_graph(backend=backend)) == record(p.get_graph(backend=backend)), (backend, stored)
        assert c.tcount() == p.tcount() == 0
        assert np.array_equal(c.to_matrix(), p.to_matrix())
        assert str(c.detector_error_model(allow_gauge_detectors=True)) == str(p.detector_error_model(allow_gauge_detectors=True))
        with pytest.raises(ValueError, match=f"because it contains {why}"):
            c.inverse()
        if importlib.util.find_spec("tsim") is not None:
            import tsim as real

            ref = real.Circuit(str(c))
            assert record(ref.get_graph()) == record(c.get_graph(backend="own")), stored
            assert str(ref.detector_error_model(allow_gauge_detectors=True)) == str(c.detector_error_model(allow_gauge_detectors=True)), stored
            assert_matrix_is_tsims(c, ref)


def test_an_underscore_is_an_identity_letter_as_in_the_core():
    """The review's example: ``pauli=X_Z`` is ``pauli=XIZ`` (the core reads ``_`` as ``I``), whose model
    and matrix are those of Tsim's ``R_PAULI(...) X0*Z2``: Stim's reading of it is ``SPP X0*Z2``, not
    the identity, and ``to_matrix`` is Tsim's ZX tensor, not a parse error."""
    from stabilizer_qec.tsim._native import stim_reading

    measured = "R 0 1 2\nX_ERROR(0.1) 2\nI[R_PAULI(theta=0.1, pauli=X{}Z)] 0 1 2\nM 0 1 2\nDETECTOR rec[-3]\nDETECTOR rec[-1]"
    unmeasured = "H 0\nI[R_PAULI(theta=0.1, pauli=X{}Z)] 0 1 2"
    underscore, identity = (tsim.Circuit(sq.Circuit(measured.format(mid))) for mid in "_I")
    assert stim_reading(underscore._stim_exact_text()) == stim_reading(identity._stim_exact_text())
    assert "SPP X0*Z2" in stim_reading(underscore._stim_exact_text()).split("\n")
    model = str(underscore.detector_error_model(allow_gauge_detectors=True))
    assert model == str(identity.detector_error_model(allow_gauge_detectors=True)) == "error(0.5) D0\nerror(0.1) D0 D1"
    a, b = (tsim.Circuit(sq.Circuit(unmeasured.format(mid))) for mid in "_I")
    assert str(a) == str(b) == "H 0\nR_PAULI(0.06366197723675814) X0*Z2"
    assert a.to_matrix().shape == (4, 4) and np.array_equal(a.to_matrix(), b.to_matrix())
    assert record(a.get_graph(backend="own")) == record(b.get_graph(backend="own")) and a.tcount() == b.tcount() == 1
    if importlib.util.find_spec("tsim") is not None:
        import tsim as real

        ref = real.Circuit("R 0 1 2\nX_ERROR(0.1) 2\nR_PAULI(0.06366197723675814) X0*Z2\nM 0 1 2\nDETECTOR rec[-3]\nDETECTOR rec[-1]")
        assert model == str(ref.detector_error_model(allow_gauge_detectors=True))
        ref = real.Circuit("H 0\nR_PAULI(0.06366197723675814) X0*Z2")
        assert record(a.get_graph(backend="own")) == record(ref.get_graph())
        assert_matrix_is_tsims(a, ref)


STORED_THETAS = ["0.1", "+.1", "1.e-1", "1E-1", "-0.25", "0", "-0", "2", "0.39269908169872414", "-0.39269908169872414", "3.9269908169872414e-1", "0.392699081698724139", "0.47123889803846897", "-0.47123889803846897"]


def random_stored_form(rng, n: int) -> tuple:
    """A stored line as the core reads it, spelled at random (``_``, ``I``, ``i`` and case in the
    letters, the arguments' order, spaces, empty arguments, the angle's literal), and this package's
    spelling of it."""

    def ws() -> str:
        return str(rng.choice(["", "", " ", "  ", "\t"]))

    def tag(name: str, args: list, spelled_args: list) -> tuple:
        parts = [f"{ws()}{k}{ws()}={ws()}{v}{ws()}" for k, v in args]
        for _ in range(int(rng.integers(0, 3))):  # empty arguments, which the core skips
            parts.insert(int(rng.integers(0, len(parts) + 1)), ws())
        spelled = f"{name}({', '.join(f'{k}={v}' for k, v in spelled_args)})" if args else name
        if not args and rng.random() < 0.5:
            return f"{ws()}{name}{ws()}", spelled
        return f"{ws()}{name}{ws()}({','.join(parts)}){ws()}", spelled

    u, q = rng.random(), int(rng.integers(n))
    theta = str(rng.choice(STORED_THETAS))
    if u < 0.6:
        # ``parse``'s R_PAULI arm takes I, II (one product over its targets, as I's), and I_ERROR and
        # II_ERROR (a coherent error, no product rotation): II's targets are pairs of distinct qubits.
        gate = str(rng.choice(["I", "I", "I", "II", "II", "I_ERROR", "II_ERROR"] if n > 1 else ["I", "I", "I", "I_ERROR"]))
        if gate.startswith("II"):
            qubits = [int(x) for _ in range(int(rng.integers(1, 3))) for x in rng.choice(n, 2, replace=False)]
        else:
            k = int(rng.integers(1, min(n, 4) + 1))
            qubits = [int(x) for x in rng.choice(n, k, replace=False)] if rng.random() < 0.85 else [int(rng.integers(n)) for _ in range(k)]
        letters = "".join(str(rng.choice(list("XYZxyzXYZ_I_i"))) for _ in qubits)
        args = [("theta", theta), ("pauli", letters)]
        if rng.random() < 0.5:
            args.reverse()
        stored, spelled = tag("R_PAULI", args, [(k, v.replace("_", "I") if k == "pauli" else v) for k, v in args])
        targets = " ".join(map(str, qubits))
        if gate.endswith("_ERROR"):  # a coherent error keeps its tag as written (its own test)
            return f"{gate}[{stored}] {targets}", f"{gate}[{stored}] {targets}"
        return f"{gate}[{stored}] {targets}", f"{'I' if gate == 'II' else gate}[{spelled}] {targets}"
    if u < 0.7:
        args = [("theta", theta)]
        gate = str(rng.choice(["I", "I", "I_ERROR"]))
        stored, spelled = tag(str(rng.choice(["R_X", "R_Y", "R_Z"])), args, args)
        return f"{gate}[{stored}] {q}", f"{gate}[{stored if gate == 'I_ERROR' else spelled}] {q}"
    if u < 0.8:
        args = [("theta", theta), ("phi", str(rng.choice(STORED_THETAS))), ("lambda", str(rng.choice(STORED_THETAS)))]
        args = [args[i] for i in rng.permutation(3)]
        stored, spelled = tag("U3", args, args)
        return f"I[{stored}] {q}", f"I[{spelled}] {q}"
    if u < 0.9 and n > 1:
        q0, q1 = (int(x) for x in rng.choice(n, 2, replace=False))
        args = [("theta", theta)]
        gate = str(rng.choice(["II", "II", "II_ERROR"]))
        stored, spelled = tag(str(rng.choice(["R_XX", "R_YY", "R_ZZ"])), args, args)
        return f"{gate}[{stored}] {q0} {q1}", f"{gate}[{stored if gate == 'II_ERROR' else spelled}] {q0} {q1}"
    stored, spelled = tag(str(rng.choice(["T", "T_DAG"])), [], [])
    return f"I[{stored}] {q}", f"I[{spelled}] {q}"


@pytest.mark.parametrize("seed", range(40))
def test_random_stored_forms_are_read_as_the_core_reads_them(seed):
    """Random stored forms the core accepts: ``_product_rotation`` reads each product rotation's
    factors, ``!``s and angle as ``nonpauli::parse`` does (``_`` and ``I`` letters dropped, the others'
    case the ``!``s, the signed angle the stored radians'), and every reader reads the circuit as this
    package's spelling of it (and, Tsim reading this package's printing of it, as Tsim's circuit)."""
    import math

    from stabilizer_qec.tsim._native import stim_reading
    from stabilizer_qec.tsim.utils.program_text import _NATIVE_LINE, _NATIVE_TAG, _native_args, _product_rotation, _read_turns

    rng = np.random.default_rng(9000 + seed)
    n, accepted = int(rng.integers(1, 5)), 0
    for _ in range(12):
        lines = [random_stored_form(rng, n) for _ in range(int(rng.integers(1, 4)))]
        stored = "\n".join(s for s, _ in lines)
        try:
            sq.Circuit(stored)
        except ValueError:
            continue
        accepted += 1
        for line, spelled_line in lines:
            m = _NATIVE_LINE.match(line)
            if m is None:  # a coherent error (I_ERROR, II_ERROR): no product rotation, kept in Stim's reading
                assert re.match(r"II?_ERROR\[", line) and stim_reading(line) == line, line
                continue
            # II[R_PAULI(...)] is read as its I spelling, one product over the same targets.
            assert stim_reading(line) == stim_reading(spelled_line), line
            t = _NATIVE_TAG.match(m.group(3))
            if t.group(1) != "R_PAULI":
                continue
            args, targets = _native_args(t.group(2)), m.group(4).split()
            found = _product_rotation(args, targets)
            assert found is not None, line
            dagger, turns, factors = found
            letters = args["pauli"]
            assert factors == [(p.islower(), p.upper(), q) for p, q in zip(letters, targets) if p.upper() not in "I_"], line
            sign = (-1) ** (dagger + sum(inverted for inverted, _, _ in factors))
            if turns is None:
                assert abs(float(args["theta"])) == math.pi / 8 and sign * math.pi / 8 == float(args["theta"]), line
            else:
                assert sign * float(turns) == float(_read_turns(args["theta"], 0.5)), line
        prefix = "H " + " ".join(map(str, range(n))) + "\n"
        spelled = "\n".join(s for _, s in lines)
        assert_read_as_spelled(prefix + stored, prefix + spelled)
        measured = f"R {' '.join(map(str, range(n)))}\nX_ERROR(0.1) 0\n{prefix}{{}}\nM {' '.join(map(str, range(n)))}\nDETECTOR rec[-1]"
        assert_read_as_spelled(measured.format(stored), measured.format(spelled))
        if importlib.util.find_spec("tsim") is not None and "I[R_PAULI" not in str(tsim.Circuit(sq.Circuit(prefix + spelled))):
            import tsim as real

            for text in (prefix + stored, measured.format(stored)):
                c = tsim.Circuit(sq.Circuit(text))
                ref = real.Circuit(str(c))
                assert outcome(lambda: record(ref.get_graph())) == outcome(lambda: record(c.get_graph(backend="own"))), text
                built = outcome(ref.to_matrix)
                if built[0] == "ok":
                    assert_matrix_is_tsims(c, ref)
                else:
                    assert outcome(c.to_matrix) == built, text
                theirs, ours = outcome(lambda: str(ref.detector_error_model(allow_gauge_detectors=True))), outcome(lambda: str(c.detector_error_model(allow_gauge_detectors=True)))
                assert ours[0] == theirs[0] and (ours == theirs or "anti-Hermitian" in theirs[1]), text
    assert accepted >= 2


# Not Pauli products or sweep bits: ``to_matrix`` does not build the graph first (a tag naming
# ``SPP`` or ``R_PAULI`` on another gate, a coherent error, an inert ``II`` tag).
NO_GRAPH_FIRST = [
    "H[SPP] 0\nX[R_PAULI] 1",
    "H 0\nI_ERROR[R_PAULI(theta=0.1, pauli=XZ)] 0 1",
    "H 0\nI_ERROR[R_PAULI(theta=0.1, pauli=X_Z)] 0 1 2",
    "H 0\nII_ERROR[R_ZZ(theta=0.1)] 0 1",
    "H 0\nII[foo] 0 1",
    "H 0\nT 1\nR_X(0.3) 0\nCX 0 1",
    "H 0\nII_ERROR[R_PAULI(theta=0.1, pauli=XZ)] 0 1",
]
GRAPH_FIRST = ["SPP X0*Z1", "SPP_DAG X0", "H 0\nR_PAULI(0.3) X0*Z1", "R_ZZ(0.3) 0 1", "TPP X0*Y1", "TPP Z0*Z0\nH 1", "H 0\nI[R_PAULI(theta=0.1, pauli=X_Z)] 0 1 2", "MPP X0*Z1", "H 0\nII[R_PAULI(theta=0.1, pauli=XZ)] 0 1"]


def test_to_matrix_builds_the_graph_first_only_for_products_and_sweep_bits(monkeypatch):
    """The graph is built before the engine runs only for what Tsim's builders can refuse (Pauli
    products, sweep bits), read from the instructions rather than the text, and then once."""
    built = []
    original = zx.parse_stim_circuit

    def spy(*args, **kwargs):
        built.append(1)
        return original(*args, **kwargs)

    for text in NO_GRAPH_FIRST + GRAPH_FIRST:
        c = tsim.Circuit(text)
        with monkeypatch.context() as patch:
            patch.setattr(zx, "parse_stim_circuit", spy)
            built.clear()
            got = c.to_matrix()
        assert len(built) == (0 if text in NO_GRAPH_FIRST else 1), text
        want = c._zx_matrix("big")
        i = np.argmax(np.abs(want))
        phase = got.flat[i] / want.flat[i]
        assert got.shape == want.shape and abs(abs(phase) - 1) < 1e-9 and np.allclose(got, phase * want, atol=1e-9, rtol=0), text
    with monkeypatch.context() as patch:
        patch.setattr(zx, "parse_stim_circuit", spy)
        built.clear()
        with pytest.raises(NotImplementedError, match="Sweep bit targets"):
            tsim.Circuit("H 0\nCX sweep[0] 0").to_matrix()
    assert len(built) == 1


# A circuit with products and an instruction Tsim's builders cannot read (a tag the core leaves inert,
# which Tsim's ``parse_parametric_tag`` refuses): ``to_matrix`` is the engine's, as without the products
# (and as before the graph was built first); the graph is Tsim's error. (With a product rotation naming
# a qubit twice, Tsim's error: ``UNREAD_WITH_A_REPEATED_QUBIT``.)
UNREAD_WITH_PRODUCTS = {
    "H 0\nR_PAULI(0.3) X0*Z1\nI[foo(bar)] 1": "H 0\nR_PAULI(0.3) X0*Z1\nI 1",
    "H 0\nR_PAULI(0.3) X0*Z1\nI[R_PAULIX(theta=0.1)] 1": "H 0\nR_PAULI(0.3) X0*Z1\nI 1",
    "SPP[foo(bar)] X0*Z1\nH 0": "SPP X0*Z1\nH 0",
    "H 0\nR_ZZ(0.3) 0 1\nH[x] 1\nI[my_tag(x=1)] 0": "H 0\nR_ZZ(0.3) 0 1\nH 1\nI 0",
}


@pytest.mark.parametrize("text", list(UNREAD_WITH_PRODUCTS))
def test_to_matrix_of_products_with_an_instruction_tsims_builders_cannot_read(text):
    c, plain = tsim.Circuit(text), tsim.Circuit(UNREAD_WITH_PRODUCTS[text])
    assert np.array_equal(c.to_matrix(), plain.to_matrix())
    with pytest.raises(ValueError, match="^Could not parse instruction "):
        c.get_graph(backend="own")
    if importlib.util.find_spec("tsim") is not None:
        import tsim as real

        with pytest.raises(ValueError) as theirs:
            real.Circuit(text).get_graph()
        with pytest.raises(ValueError) as ours:
            c.get_graph(backend="own")
        assert str(ours.value) == str(theirs.value)


def test_a_products_refusal_before_an_instruction_tsims_builders_cannot_read_is_raised():
    """The builders read the circuit in order: a product they refuse before the instruction they cannot
    read is still Tsim's error (Tsim's ``to_matrix`` raises it too)."""
    text = "H 0\nR_PAULI(0.3) X0*X0\nI[foo(bar)] 1"
    message = "R_PAULI target qubits must be distinct within a product, got repeated qubit 0 in 'SPP[R_PAULI(theta=0.3*pi)] X0*X0'."
    with pytest.raises(ValueError) as info:
        tsim.Circuit(text).to_matrix()
    assert str(info.value) == message
    if importlib.util.find_spec("tsim") is not None:
        import tsim as real

        with pytest.raises(ValueError) as theirs:
            real.Circuit(text).to_matrix()
        assert str(theirs.value) == message


# A product rotation naming a qubit twice is the ZX tensor's matrix (``TPP X0*Y0*X0`` is exp(-i pi/8 (-Y)),
# a sign the unitary engine does not keep; ``TPP Z0*Z0`` cancels, leaving qubit 0 out of Tsim's graph and
# matrix). With an instruction Tsim's builders cannot read there is no graph, so ``to_matrix`` raises
# Tsim's error, before or after the product, as Tsim's ``to_matrix`` does.
UNREAD_WITH_A_REPEATED_QUBIT = [
    "H 0\nTPP X0*Y0*X0\nI[foo(bar)] 1",
    "H 0\nI[foo(bar)] 1\nTPP X0*Y0*X0",
    "H 0\nTPP X0*Y0*X0\nH 1\nI[R_PAULIX(theta=0.1)] 1",
    "H 1\nTPP Z0*Z0\nI[foo(bar)] 1",
    "H 0\nTPP Z0*Z0\nH 1\nI[foo(bar)] 1",
    "H 0\nR_PAULI(0.3) X0*Y1\nI[R_PAULI(theta=0.39269908169872414, pauli=XYX)] 0 0 0\nI[foo(bar)] 1",
    "H 0\nI[foo(bar)] 1\nII[R_PAULI(theta=0.1, pauli=XYZX)] 0 1 0 2",
]


# A product that leaves a qubit out of Tsim's graph without naming it twice in a rotation (an identity
# letter, an ``SPP`` that cancels on it) leaves it out of the engine's matrix too, so with an instruction
# Tsim's builders cannot read, ``to_matrix`` is the engine's: of the qubits Tsim's matrix of the circuit
# without that tag has, equal to it up to the engine's global phase (Tsim's builders apply each rotation up
# to one), as the engine's matrix of a circuit without products is.
UNREAD_WITH_A_QUBIT_LEFT_OUT = [
    "H 1\nSPP X0*X0\nI[foo(bar)] 1",
    "H 1\nSPP Z0*Z0*X1\nI[foo(bar)] 1",
    "H 0\nI[R_PAULI(theta=0.1, pauli=X_)] 0 1\nI[foo(bar)] 0",
    "H 0\nI[foo(bar)] 0\nII[R_PAULI(theta=0.39269908169872414, pauli=X_)] 0 1",
    "H 0\nII[R_PAULI(theta=0.1, pauli=X_Z_)] 0 1 2 3\nI[R_PAULIX(theta=0.1)] 0",
    "H 0\nII[R_PAULI(theta=0.1, pauli=__)] 1 2\nI[foo(bar)] 0",
]


@pytest.mark.parametrize("text", UNREAD_WITH_A_QUBIT_LEFT_OUT)
def test_to_matrix_of_a_qubit_left_out_with_an_instruction_tsims_builders_cannot_read(text):
    c = tsim.Circuit(text)
    plain = tsim.Circuit(re.sub(r"I\[(foo\(bar\)|R_PAULIX\(theta=0\.1\))\]", "I", text))
    got, want = c.to_matrix(), plain.to_matrix()
    assert np.allclose(want, plain._zx_matrix("big"), atol=1e-12, rtol=0)  # its ZX tensor, the qubit left out
    i = np.argmax(np.abs(want))
    phase = got.flat[i] / want.flat[i]
    assert got.shape == want.shape and abs(abs(phase) - 1) < 1e-9 and np.allclose(got, phase * want, atol=1e-9, rtol=0), text
    with pytest.raises(ValueError, match="^Could not parse instruction "):
        c.get_graph(backend="own")
    if importlib.util.find_spec("tsim") is not None and "pauli=__" not in text:
        import tsim as real

        assert_matrix_is_tsims(plain, real.Circuit(str(plain)))


@pytest.mark.parametrize("text", UNREAD_WITH_A_REPEATED_QUBIT)
def test_to_matrix_of_a_repeated_qubit_with_an_instruction_tsims_builders_cannot_read_raises(text):
    c = tsim.Circuit(text)
    with pytest.raises(ValueError) as ours:
        c.to_matrix()
    with pytest.raises(ValueError) as graph:
        c.get_graph(backend="own")
    assert str(ours.value) == str(graph.value) and str(ours.value).startswith("Could not parse instruction ")
    if importlib.util.find_spec("tsim") is not None:
        import tsim as real

        with pytest.raises(ValueError) as theirs:
            real.Circuit(str(c)).to_matrix()
        assert str(ours.value) == str(theirs.value)
        # With that instruction an untagged ``I``, the matrix is Tsim's (or its error).
        plain = tsim.Circuit(re.sub(r"I\[(foo\(bar\)|R_PAULIX\(theta=0\.1\))\]", "I", text))
        ref = real.Circuit(str(plain))
        if outcome(ref.to_matrix)[0] == "ok":
            assert_matrix_is_tsims(plain, ref)
        else:
            assert outcome(plain.to_matrix) == outcome(ref.to_matrix)
