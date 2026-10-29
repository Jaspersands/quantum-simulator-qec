"""The Python facade's Circuit, DetectorErrorModel, PauliString, Tableau and Matching where they
once departed from Stim and PyMatching: copies keep their class without calling a subclass's
constructor (and Deltakit's circuit its leakage), the reference signs read tagged Clifford-angle gates as the reference run does, a
model's tags may hold '#', indexing parses once, model files with a stray non-UTF-8 byte in a
comment load, the WDIMACS writer takes a certain error, the exact distance skips faults that
never happen, and heavy networkx / rustworkx edges are left out with PyMatching's warning."""

from __future__ import annotations

import copy
import io
import pickle
import warnings

import numpy as np
import pytest

import stabilizer_qec as sq
import stabilizer_qec.deltakit_stim as dk
from stabilizer_qec import _circuit, _dem


# Subclasses at module level, so pickle finds them by name (as a user's would be).
class MyCircuit(sq.Circuit):
    pass


class MyModel(sq.DetectorErrorModel):
    pass


class MyPauliString(sq.PauliString):
    pass


class MyTableau(sq.Tableau):
    pass


# Subclasses whose constructors take other arguments than the text: a copy must not call them.
class HadamardOn(sq.Circuit):
    def __init__(self, qubit):
        super().__init__(f"H {int(qubit)}")


class OneError(sq.DetectorErrorModel):
    def __init__(self, p):
        super().__init__(f"error({float(p)}) D0")


class AllX(sq.PauliString):
    def __init__(self, n):
        super().__init__("X" * int(n))


class Identity(sq.Tableau):
    def __init__(self, n):
        super().__init__(int(n))


class FlippedMatching(sq.Matching):
    def decode(self, z, **kwargs):
        return 1 - super().decode(z, **kwargs)


_ROUND_TRIPS = [("pickle", lambda x: pickle.loads(pickle.dumps(x))), ("copy", copy.copy), ("deepcopy", copy.deepcopy)]


@pytest.mark.parametrize("how,again", _ROUND_TRIPS, ids=[h for h, _ in _ROUND_TRIPS])
def test_subclasses_copy_and_pickle_as_themselves(how, again):
    # As Stim's: each comes back as its own class, equal to what it was.
    for obj in (MyCircuit("H 0\nM 0"), MyModel("error(0.1) D0"), MyPauliString("+XYZ"), MyPauliString("-i_ZZ"), MyTableau(2), MyTableau.random(3)):
        y = again(obj)
        assert type(y) is type(obj) and y == obj, (how, obj)
    # As PyMatching's copy: the subclass, its overrides and its attributes kept.
    m = FlippedMatching(np.array([[1, 1, 0], [0, 1, 1]]))
    m.note = "kept"
    y = again(m)
    assert type(y) is FlippedMatching and y.note == "kept"
    assert y.decode([1, 0]).tolist() == m.decode([1, 0]).tolist() == [0, 1, 1]
    # The base classes are as they were.
    assert type(again(sq.Matching(np.array([[1, 1]])))) is sq.Matching
    assert type(again(sq.Tableau(1))) is sq.Tableau
    for obj in (sq.Circuit("H 0"), sq.DetectorErrorModel("error(0.1) D0"), sq.PauliString("-XZ")):
        assert type(again(obj)) is type(obj) and again(obj) == obj


@pytest.mark.parametrize("how,again", _ROUND_TRIPS, ids=[h for h, _ in _ROUND_TRIPS])
def test_subclasses_with_their_own_constructors_copy_without_calling_them(how, again):
    # As Stim 1.16's: the copy is the subclass, equal, its constructor not called with the text
    # (which here would raise), and the instance's own attributes not kept.
    for obj, text in ((HadamardOn(3), "H 3"), (OneError(0.1), "error(0.1) D0"), (AllX(2), "+XX"), (Identity(1), str(sq.Tableau(1)))):
        obj.note = "dropped"
        y = again(obj)
        assert type(y) is type(obj) and y == obj and str(y) == text, (how, obj)
        assert not hasattr(y, "note")
    # Deltakit's circuit and Tsim's, subclasses of Circuit, still come back as themselves.
    from stabilizer_qec import tsim
    for obj in (dk.Circuit("LEAKAGE(0.2) 0\nM 0"), tsim.Circuit("R 0\nT 0\nM 0")):
        y = again(obj)
        assert type(y) is type(obj) and y == obj and str(y) == str(obj)


@pytest.mark.parametrize("how,again", _ROUND_TRIPS, ids=[h for h, _ in _ROUND_TRIPS])
def test_deltakit_circuit_copies_keep_their_leakage(how, again):
    c = dk.Circuit("R 0 1\nLEAKAGE(0.2) 0\nCX 0 1\nHERALD_LEAKAGE_EVENT 0\nM 1\nDETECTOR rec[-1]\nDETECTOR rec[-2]")
    y = again(c)
    assert type(y) is dk.Circuit and y == c
    assert str(y) == str(c) and str(y.detector_error_model()) == str(c.detector_error_model()) == "error(0.5) D0 D1"
    rates = y.compile_detector_sampler(seed=1).sample(20000).mean(0)
    assert np.array_equal(rates, c.compile_detector_sampler(seed=1).sample(20000).mean(0))
    assert rates[1] > 0.15  # the herald fires on about 20% of shots


def test_deltakit_parallel_engine_samples_the_leakage():
    # Deltakit's engine sends its decoder managers, circuits and all, to a process pool.
    pytest.importorskip("pathos")
    import pandas  # noqa: F401  (the engine's results)

    from stabilizer_qec.deltakit.decode import PyMatchingDecoder
    from stabilizer_qec.deltakit.decode.analysis import RunAllAnalysisEngine, StimDecoderManager
    from stabilizer_qec.deltakit.decode.utils import parse_stim_circuit

    c = dk.Circuit.generated("repetition_code:memory", distance=5, rounds=5, after_clifford_depolarization=0.002, after_reset_leakage=0.03, after_clifford_leakage_and_relaxation=0.01)
    fails = {}
    for processes in (1, 2):
        g, logicals, sc = parse_stim_circuit(c)
        manager = StimDecoderManager(sc, PyMatchingDecoder(g, logicals), seed=1, batch_size=2000)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = RunAllAnalysisEngine("x", [manager], num_parallel_processes=processes, batch_size=2000, max_shots=4000).run()
        fails[processes] = int(df["fails"][0])
    # About 3% of shots fail with the leakage (Deltakit's own run: 582 and 598 of 20,000);
    # without it, none would.
    assert fails[1] > 50 and fails[2] > 50, fails


@pytest.mark.parametrize(
    "gate,quarter",
    [("I[R_X(theta=1.5707963267948966)]", "X"), ("I[R_Y(theta=3.141592653589793)]", "I"), ("I[R_X(theta=0.7853981633974483)]", "SQRT_X")],
)
def test_reference_signs_read_clifford_angle_tags_as_the_reference_run(gate, quarter):
    # R_P(θ) = exp(-iθP): θ = π/2 is the Pauli, θ = π/4 its square root, θ = π the identity.
    text = "R 0 1\n{} 0\nCX 0 1\nX_ERROR(0.1) 0\nM 0 1\nDETECTOR rec[-2]\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]"
    tagged, plain = sq.Circuit(text.format(gate)), sq.Circuit(text.format(quarter))
    ref = tagged.reference_sample()
    assert ref.tolist() == plain.reference_sample().tolist()
    d, o = tagged.reference_detector_and_observable_signs()
    assert d.tolist() == ref.tolist() and o.tolist() == [bool(ref[1])]
    assert tagged.count_determined_measurements() == plain.count_determined_measurements()
    meas = np.array([[1, 1], [0, 1], [0, 0]], dtype=bool)
    for skip in (False, True):
        got = tagged.compile_m2d_converter(skip_reference_sample=skip).convert(measurements=meas, append_observables=True)
        want = plain.compile_m2d_converter(skip_reference_sample=skip).convert(measurements=meas, append_observables=True)
        assert got.tolist() == want.tolist(), skip
    if quarter == "X":
        # Raw parities, as Stim's converter gives them on the X circuit.
        got = tagged.compile_m2d_converter(skip_reference_sample=True).convert(measurements=meas, append_observables=True)
        assert got.astype(int).tolist() == [[1, 1, 1], [0, 1, 1], [0, 0, 0]]


def test_model_tags_may_hold_a_hash():
    # Stim's tags end only at ']': a '#' in one is not a comment (Stim gives these answers).
    d = sq.DetectorErrorModel("detector[a#b](1, 2) D0\nerror[t#g](0.1) D0 D1\nerror(0.1) D1\n")
    assert len(d) == 3 and str(d) == "detector[a#b](1, 2) D0\nerror[t#g](0.1) D0 D1\nerror(0.1) D1"
    assert d.get_detector_coordinates() == {0: [1.0, 2.0], 1: []}
    assert d[1].tag == "t#g" and d[1].targets_copy() == [sq.DemTarget("D0"), sq.DemTarget("D1")]
    assert [it.type for it in d] == ["detector", "error", "error"]
    assert repr(d).startswith("stabilizer_qec.DetectorErrorModel(") and "error[t#g](0.1) D0 D1" in repr(d)
    assert str(d.rounded(1)) == "detector[a#b](1, 2) D0\nerror[t#g](0.1) D0 D1\nerror(0.1) D1"
    assert len(d + d) == 6 and d.approx_equals(sq.DetectorErrorModel(str(d)), atol=0)
    assert str(d.flattened()) == str(d)
    assert sq.DetectorErrorModel("error[#](0.1) D0").get_detector_coordinates() == {0: []}
    # Comments still end a line, after the tag and around blocks.
    e = sq.DetectorErrorModel("error[x](0.1) D0 # [not a tag\nrepeat 2 { # loop\n  error(0.2) D1 # [\n} # done\n")
    assert [it.type for it in e] == ["error", "repeat"] and e[0].targets_copy() == [sq.DemTarget("D0")]
    assert _dem.parse_items("# only\n\nerror(0.1) D0 L0 # c\n")[0].targets_copy() == [sq.DemTarget("D0"), sq.DemTarget("L0")]


def _counting(monkeypatch, module, name):
    calls = [0]
    real = getattr(module, name)

    def counted(*args):
        calls[0] += 1
        return real(*args)

    monkeypatch.setattr(module, name, counted)
    return calls


def test_circuit_indexing_parses_once_until_it_changes(monkeypatch):
    c = sq.Circuit("\n".join(f"H {q}\nTICK\nCX {q} {q + 1}\nTICK" for q in range(300)))
    calls = _counting(monkeypatch, _circuit._core, "circuit_items")
    items = [c[i] for i in range(len(c))]
    assert len(items) == 1200 and calls[0] == 1
    assert items[2] == sq.CircuitInstruction("CX", [0, 1]) and c[-1] == sq.CircuitInstruction("TICK")
    # Each way of changing the circuit is seen.
    c.append("M", [0])
    assert len(c) == 1201 and c[-1] == sq.CircuitInstruction("M", [0]) and calls[0] == 2
    c.append_from_stim_program_text("X 5")
    assert c[-1] == sq.CircuitInstruction("X", [5])
    c += sq.Circuit("Y 6")
    assert c[-1] == sq.CircuitInstruction("Y", [6])
    c.append(sq.Circuit("Z 7"))
    assert c[-1] == sq.CircuitInstruction("Z", [7])
    assert c.pop() == sq.CircuitInstruction("Z", [7]) and c[-1] == sq.CircuitInstruction("Y", [6])
    c.insert(0, sq.CircuitInstruction("S", [9]))
    assert c[0] == sq.CircuitInstruction("S", [9]) and len(c) == 1204
    c *= 2
    assert len(c) == 1 and isinstance(c[0], sq.CircuitRepeatBlock)
    c.clear()
    assert len(c) == 0
    # A copy changed leaves the original as it was.
    a = sq.Circuit("H 0")
    assert len(a) == 1
    b = a.copy()
    b.append("X", [1])
    assert len(a) == 1 and len(b) == 2
    # Deltakit's circuit indexes its own instructions the same way.
    k = dk.Circuit("R 0\nLEAKAGE(0.1) 0\nM 0")
    assert [k[i].name for i in range(len(k))] == ["R", "LEAKAGE", "M"]
    k.append("RELAX", [0], 0.2)
    assert k[-1].name == "RELAX"


def test_model_indexing_parses_once_until_it_changes(monkeypatch):
    d = sq.DetectorErrorModel("\n".join(f"error(0.01) D{i} D{i + 1}" for i in range(500)))
    calls = _counting(monkeypatch, _dem, "parse_items")
    items = [d[i] for i in range(len(d))]
    assert len(items) == 500 and calls[0] == 1 and items[7] == sq.DemInstruction("error", [0.01], [sq.DemTarget("D7"), sq.DemTarget("D8")])
    d.append("error", 0.2, [sq.DemTarget("D3")])
    assert len(d) == 501 and d[-1] == sq.DemInstruction("error", [0.2], [sq.DemTarget("D3")]) and calls[0] == 2
    d += sq.DetectorErrorModel("detector D0")
    assert d[-1].type == "detector"
    d.clear()
    assert len(d) == 0


def test_files_with_a_stray_byte_in_a_comment_load(tmp_path):
    # Stim passes over the bytes of a comment; a byte that is not UTF-8 is read as U+FFFD.
    dem_path = tmp_path / "u.dem"
    dem_path.write_bytes(b"error(0.1) D0 L0 # caf\xe9 \xff\nerror(0.1) D0 D1\nerror(0.1) D1\n")
    stim_path = tmp_path / "u.stim"
    stim_path.write_bytes(b"X_ERROR(0.1) 0 # caf\xe9\nM 0\nDETECTOR rec[-1]\n")
    assert sq.DetectorErrorModel.from_file(dem_path).num_errors == 3
    assert sq.DetectorErrorModel.from_file(str(dem_path)) == sq.DetectorErrorModel("error(0.1) D0 L0\nerror(0.1) D0 D1\nerror(0.1) D1")
    assert sq.Circuit.from_file(stim_path).num_detectors == 1
    with open(stim_path, "rb") as f:
        assert sq.Circuit.from_file(f) == sq.Circuit("X_ERROR(0.1) 0\nM 0\nDETECTOR rec[-1]")
    assert sq.DetectorErrorModel.from_file(io.StringIO("error(0.1) D0")).num_errors == 1
    assert sq.Matching.from_detector_error_model_file(dem_path).num_detectors == 2
    assert sq.Matching.from_stim_circuit_file(stim_path).num_detectors == 1
    # Valid UTF-8 is read as it is.
    tagged = tmp_path / "t.stim"
    tagged.write_bytes("H[café] 0\n".encode())
    assert sq.Circuit.from_file(tagged)[0].tag == "café"


@pytest.mark.parametrize(
    "text,kwargs,want",
    [
        # Stim 1.16's output (its round of inf/inf drops the soft clauses).
        ("X_ERROR(1) 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]", {}, "p wcnf 1 3 301\n301 -1 0\n301 1 0\n"),
        ("X_ERROR(1) 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]", {"quantization": 10, "format": "WDIMACS"}, "p wcnf 1 3 31\n31 -1 0\n31 1 0\n"),
        ("X_ERROR(0.1) 0\nX_ERROR(1) 1\nM 0 1\nDETECTOR rec[-1]\nDETECTOR rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-1]", {}, "p wcnf 2 5 501\n501 -1 0\n501 -2 0\n501 1 0\n"),
        ("X_ERROR(1) 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]", {}, "p wcnf 1 2 201\n201 1 0\n"),
    ],
)
def test_likeliest_error_sat_problem_takes_a_certain_error(text, kwargs, want):
    assert sq.Circuit(text).likeliest_error_sat_problem(**kwargs) == want


def test_likeliest_error_sat_problem_certain_errors_match_stim():
    stim = pytest.importorskip("stim")
    for text in ["X_ERROR(1) 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]", "X_ERROR(0.1) 0\nX_ERROR(1) 1\nM 0 1\nDETECTOR rec[-1]\nDETECTOR rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-1]"]:
        for kw in ({}, {"quantization": 10}):
            assert sq.Circuit(text).likeliest_error_sat_problem(**kw) == stim.Circuit(text).likeliest_error_sat_problem(**kw)


@pytest.mark.parametrize(
    "text,want",
    [
        ("error(0) L0\nerror(0.1) D0\nerror(0.1) D0 L0", 2),
        ("error(0) D0 L0\nerror(0.1) D0 D1\nerror(0.1) D1 L0\nerror(0.1) D0", 3),
        ("error[a)b](0.1) D0 L0\nerror(0.1) D0", 2),
    ],
)
def test_exact_distance_skips_impossible_faults_and_reads_tags(text, want):
    # Stim 1.16's shortest_graphlike_error gives 2, 3 and 2.
    pytest.importorskip("scipy")
    dem = sq.DetectorErrorModel(text)
    assert dem.distance() == dem.distance(method="graphlike") == dem.distance(method="search") == want
    # A decomposed fault (with a tag holding ')') is taken whole: its pieces' symptoms together.
    dem = sq.DetectorErrorModel("error[x)](0.1) D0 ^ D1 L0\nerror(0.1) D0\nerror(0.1) D1\nerror(0) L0")
    assert dem.distance() == dem.distance(method="search") == 3


def _heavy_warnings(build):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m = build()
    return m, [str(w.message) for w in caught if issubclass(w.category, UserWarning)]


def test_heavy_graph_edges_are_left_out_with_pymatchings_warning():
    nx = pytest.importorskip("networkx")
    g = nx.Graph()
    g.add_edge(0, 1, weight=1, fault_ids=0)
    g.add_edge(1, 2, weight=2e7, fault_ids=1)
    g.add_edge(2, 3, weight=1, fault_ids=2)
    g.add_edge(0, 3, weight=5, fault_ids=3)
    for build in (lambda: sq.Matching(g), lambda: sq.Matching.from_networkx(g)):
        m, said = _heavy_warnings(build)
        # PyMatching 2.4.0's words, its weight printed by std::to_string.
        assert said == ["Weight 20000000.000000 of edge (1, 2) exceeds maximum edge weight 16777215 and has not been added to the matching graph."]
        assert m.num_edges == 3 and (1, 2) not in [(u, v) for u, v, _ in m.edges()]
        assert m.decode([1, 0, 0, 1]).tolist() == [0, 0, 0, 1] and m.decode_to_edges_array([1, 0, 0, 1]).tolist() == [[3, 0]]
        assert m.decode([1, 1, 0, 0]).tolist() == [1, 0, 0, 0]
    m, said = _heavy_warnings(lambda: _added_by_hand())
    assert said == [
        "Weight 20000000.000000 of edge (0, 1) exceeds maximum edge weight 16777215 and has not been added to the matching graph.",
        "Weight -30000000.000000 of edge (2, None) exceeds maximum edge weight 16777215 and has not been added to the matching graph.",
    ]
    assert m.num_edges == 1


def _added_by_hand():
    m = sq.Matching()
    m.add_edge(0, 1, weight=2e7)
    m.add_edge(1, 2, fault_ids=0)
    m.add_boundary_edge(2, weight=-3e7)
    return m


def test_heavy_rustworkx_edges_are_left_out_with_pymatchings_warning():
    rx = pytest.importorskip("rustworkx")
    r = rx.PyGraph()
    r.add_nodes_from([{} for _ in range(4)])
    r.add_edge(0, 1, {"weight": 1, "fault_ids": 0})
    r.add_edge(1, 2, {"weight": -2e7, "fault_ids": 1})
    r.add_edge(2, 3, {"weight": 1, "fault_ids": 2})
    r.add_edge(0, 3, {"weight": 7, "fault_ids": 3})

    def build():
        m = sq.Matching()
        m.load_from_rustworkx(r)
        return m

    m, said = _heavy_warnings(build)
    assert said == ["Weight -20000000.000000 of edge (1, 2) exceeds maximum edge weight 16777215 and has not been added to the matching graph."]
    assert m.num_edges == 3 and [w["weight"] for _, _, w in m.edges()] == [1.0, 1.0, 7.0]
    assert m.decode([1, 0, 0, 1]).tolist() == [0, 0, 0, 1]
