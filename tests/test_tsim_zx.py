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
import sys
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
    # Correlated errors are chains of at most 24 branches here (2^k probabilities).
    long_chain = "E(0.1) X0\n" + "ELSE_CORRELATED_ERROR(0.1) X0\n" * 30
    with pytest.raises(ValueError, match="correlated errors has 2\\^31 outcomes"):
        tsim.Circuit(long_chain).get_graph()


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
        "SPP[T] X1*X2",
        "H[] 0",
    ]


def test_a_pi_over_eight_product_rotation_is_tpp():
    # One gate here: Tsim's graph for TPP (no scalar phase), not R_PAULI(0.25)'s (a scalar phase of -1/8).
    same_gate = {"R_PAULI(0.25) X0*Y1": "TPP X0*Y1", "R_XX(0.25) 0 1": "TPP X0*X1", "R_PAULI(-0.25) Z0": "TPP_DAG Z0"}
    for rotation, tpp in same_gate.items():
        graph = record(tsim.Circuit(rotation + "\nM 0").get_graph(backend="own"))
        assert graph == record(tsim.Circuit(tpp + "\nM 0").get_graph(backend="own")), rotation
        assert graph["scalar"][1] == 0


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
            lines.append(f"{rng.choice(['TPP', 'TPP_DAG', 'SPP', 'SPP_DAG'])} {rng.choice(list('XYZ'))}{a}*{rng.choice(list('XYZ'))}{b}")
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
    want = np.asarray(ref.to_matrix())
    got = mine.to_matrix()
    assert got.shape == want.shape, text
    if np.allclose(got, want, atol=1e-9):
        return
    # A circuit the unitary engine runs: Tsim's matrix up to a global phase.
    i = np.argmax(np.abs(want))
    phase = got.flat[i] / want.flat[i]
    assert abs(abs(phase) - 1) < 1e-6 and np.allclose(got, phase * want, atol=1e-6), text
