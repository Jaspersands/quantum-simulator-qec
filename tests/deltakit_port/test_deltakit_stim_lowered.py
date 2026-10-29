"""stabilizer_qec.deltakit_stim's circuits hold Deltakit's instructions as lowered, tagged lines
(``I_ERROR[LEAKAGE(p=…)]``, ``MPAD[HERALD_LEAKAGE_EVENT(qubits=…)]`` and the rest). What a
circuit hands out must be Deltakit's instructions, as deltakit_stim 0.2.7 hands them out, not
the lowered lines: its diagrams, its flattened operations, its tableau and time reversal (which
refuse leakage as Deltakit's do), its explained errors (which Deltakit's own
``parse_explained_dem`` reads), ``approx_equals``. The expected values are deltakit_stim
0.2.7's own output for these circuits."""

import pytest

import stabilizer_qec as sq
import stabilizer_qec.deltakit_stim as stim
from stabilizer_qec.deltakit.core.decoding_graphs import parse_explained_dem

# --- diagrams --------------------------------------------------------------------------------

DIAGRAMS = [
    (
        "R 0 1\nLEAKAGE(0.1) 0\nCX(0.1, 0.05, 0.2, 0.01) 0 1\nRELAX(0.2) 0\nRL 1\nHERALD_LEAKAGE_EVENT(0.01) 0 1\nM 1\nDETECTOR rec[-1]\nDETECTOR rec[-2]\nDETECTOR rec[-3]",
        "q0: -R-LEAKAGE(0.1)-@(0.1,0.05,0.2,0.01)-RELAX(0.2)-HERALD_LEAKAGE_EVENT(0.01):rec[0]-----------------------------DETECTOR:D2=rec[0]-\n                    |\nq1: -R--------------X(0.1,0.05,0.2,0.01)-RL---------HERALD_LEAKAGE_EVENT(0.01):rec[1]-M:rec[2]-DETECTOR:D0=rec[2]-DETECTOR:D1=rec[1]-",
    ),
    (
        "R 0 1 2\nTICK\nLEAKAGE(0.05) 0 1\nCZ(0.2, 0.1, 0, 0) 1 2\nCY(0.1,0.1,0.1,0.1) 0 1\nSQRT_XX(0.1, 0.1, 0.1, 0.1) 0 2\nTICK\nHERALD_LEAKAGE_EVENT 2 !0\nM 0 1 2\nDETECTOR rec[-1] rec[-4]",
        "       /----------------------------------------------------------------------\\ /------------------------------------------------------------\\\nq0: -R-LEAKAGE(0.05)----------------@(0.1,0.1,0.1,0.1)-SQRT_XX(0.1,0.1,0.1,0.1)-HERALD_LEAKAGE_EVENT:rec[1]-M:rec[2]-DETECTOR:D0=rec[4]*rec[1]-\n                                    |                  |\nq1: -R-LEAKAGE(0.05)-@(0.2,0.1,0,0)-Y(0.1,0.1,0.1,0.1)-|----------------------------------------------------M:rec[3]---------------------------\n                     |                                 |\nq2: -R---------------@(0.2,0.1,0,0)--------------------SQRT_XX(0.1,0.1,0.1,0.1)-HERALD_LEAKAGE_EVENT:rec[0]-M:rec[4]---------------------------\n       \\----------------------------------------------------------------------/ \\------------------------------------------------------------/",
    ),
    (
        "R 0 1\nREPEAT 3 {\nLEAKAGE(0.02) 0\nCX 0 1\nHERALD_LEAKAGE_EVENT(0.01) 1\nMR 1\nDETECTOR rec[-1]\nDETECTOR rec[-2]\n}\nM 0\nDETECTOR rec[-1]",
        "       /REP 3                                                                                                                                                 \\\nq0: -R-|------LEAKAGE(0.02)-@---------------------------------------------------------------------------------------------------------------------------------|-M:rec[6]-DETECTOR:D6=rec[6]-\n       |                    |                                                                                                                                 |\nq1: -R-|--------------------X-HERALD_LEAKAGE_EVENT(0.01):rec[0+iter*2]-MR:rec[1+iter*2]-DETECTOR:D[0+iter*2]=rec[1+iter*2]-DETECTOR:D[1+iter*2]=rec[0+iter*2]-|-----------------------------\n       \\                                                                                                                                                      /",
    ),
    (
        "QUBIT_COORDS(1, 2) 0\nQUBIT_COORDS(3, 4) 1\nRL 0 1\nHERALD_LEAKAGE_EVENT 1 0\nDETECTOR(3, 4) rec[-2]",
        "q0: -QUBIT_COORDS(1,2)-RL-HERALD_LEAKAGE_EVENT:rec[1]-------------------------\n\nq1: -QUBIT_COORDS(3,4)-RL-HERALD_LEAKAGE_EVENT:rec[0]-DETECTOR(3,4):D0=rec[0]-",
    ),
]


@pytest.mark.parametrize(("text", "want"), DIAGRAMS)
def test_timeline_draws_deltakits_instructions(text, want):
    c = stim.Circuit(text)
    assert str(c.diagram()) == want
    assert str(c.diagram("timeline-text")) == want


def test_pictures_draw_deltakits_instructions():
    c = stim.Circuit(DIAGRAMS[0][0])
    for kind in ["timeline-svg", "timeslice-svg", "detslice-with-ops-svg"]:
        svg = str(c.diagram(kind, tick=range(0, 1)) if "slice" in kind else c.diagram(kind))
        for lowered in ["MPAD", "I_ERROR", "LEAKAGE(p=", "LEAKAGE_TRANSPORT", "I[RL]"]:
            assert lowered not in svg, (kind, lowered)
        for label in [">LEAKAGE", ">RELAX", ">RL<", ">HERALD_LEAKAGE_EVENT"]:
            assert label in svg, (kind, label)
    # The timeline names the arguments, and the herald of qubit 1 is drawn on qubit 1, as its
    # record is.
    svg = str(c.diagram("timeline-svg"))
    for label in ["LEAKAGE(0.1)", "RELAX(0.2)", "CX(0.1, 0.05, 0.2, 0.01)", "HERALD_LEAKAGE_EVENT(0.01):rec[0]", "HERALD_LEAKAGE_EVENT(0.01):rec[1]", "M:rec[2]"]:
        assert f">{label}<" in svg, label
    # A circuit of this package's own, as written, is drawn as Stim draws it.
    assert "MPAD(0.01):rec[1]" in str(sq.Circuit(c._stim_exact_text()).diagram())


# --- flattened operations, tableau, time reversal, approx_equals -------------------------------

FLATTENED_OPERATIONS = {
    "H 0\nLEAKAGE(0.1) 0\nRELAX(0.25) 1\nRL 0\nHERALD_LEAKAGE_EVENT !0 1\nHERALD_LEAKAGE_EVENT(0.01) 2\nCX(0.1,0.2,0.3,0.4) 0 1\nCZ(0,0,0,0) 0 1\nM 0": [
        ("H", [0], 0),
        ("LEAKAGE", [0], 0.1),
        ("RELAX", [1], 0.25),
        ("RL", [0], 0),
        ("HERALD_LEAKAGE_EVENT", [("inv", 0), 1], 0),
        ("HERALD_LEAKAGE_EVENT", [2], 0.01),
        ("CX", [0, 1], [0.1, 0.2, 0.3, 0.4]),
        ("CZ", [0, 1], [0.0, 0.0, 0.0, 0.0]),
        ("M", [0], 0),
    ],
    "REPEAT 2 {\nLEAKAGE(0.1) 0\nRL 0\n}": [("LEAKAGE", [0], 0.1), ("RL", [0], 0), ("LEAKAGE", [0], 0.1), ("RL", [0], 0)],
}


@pytest.mark.parametrize("text", list(FLATTENED_OPERATIONS))
def test_flattened_operations_are_deltakits(text):
    assert stim.Circuit(text).flattened_operations() == FLATTENED_OPERATIONS[text]


# deltakit_stim 0.2.7's to_tableau for each of: nothing ignored, ignore_noise, ignore_measurement,
# ignore_reset, measurement and noise, noise and reset, all three; (kind, first operation) where it
# refuses.
_ZX, _XZ, _CX = "+-xz-\n| ++\n| ZX", "+-xz-\n| ++\n| XZ", "+-xz-xz-\n| ++ ++\n| ZX _Z\n| _X XZ"
TABLEAU = {
    "H 0\nLEAKAGE(0.1) 0": [("noisy", "LEAKAGE(0.1) 0"), _ZX, ("noisy", "LEAKAGE(0.1) 0"), ("noisy", "LEAKAGE(0.1) 0"), _ZX, _ZX, _ZX],
    "H 0\nRL 0": [("reset", "RL 0"), ("reset", "RL 0"), ("reset", "RL 0"), _ZX, ("reset", "RL 0"), _ZX, _ZX],
    "H 0\nHERALD_LEAKAGE_EVENT 0": [("measurement", "HERALD_LEAKAGE_EVENT 0"), ("measurement", "HERALD_LEAKAGE_EVENT 0"), _ZX, ("measurement", "HERALD_LEAKAGE_EVENT 0"), _ZX, ("measurement", "HERALD_LEAKAGE_EVENT 0"), _ZX],
    "H 0\nHERALD_LEAKAGE_EVENT(0.1) 0": [("measurement", "HERALD_LEAKAGE_EVENT(0.1) 0"), ("measurement", "HERALD_LEAKAGE_EVENT(0.1) 0"), ("noisy", "HERALD_LEAKAGE_EVENT(0.1) 0"), ("measurement", "HERALD_LEAKAGE_EVENT(0.1) 0"), _ZX, ("measurement", "HERALD_LEAKAGE_EVENT(0.1) 0"), _ZX],
    "H 0\nCX(0.1,0.1,0.1,0.1) 0 1": [("noisy", "CX(0.1, 0.1, 0.1, 0.1) 0 1"), _CX, ("noisy", "CX(0.1, 0.1, 0.1, 0.1) 0 1"), ("noisy", "CX(0.1, 0.1, 0.1, 0.1) 0 1"), _CX, _CX, _CX],
    "H 0\nCX(0,0,0,0) 0 1": [_CX] * 7,
    "H 0\nLEAKAGE(0) 0": [_ZX] * 7,
    "X_ERROR(0.1) 0\nLEAKAGE(0.1) 0": [("noisy", "X_ERROR(0.1) 0"), _XZ, ("noisy", "X_ERROR(0.1) 0"), ("noisy", "X_ERROR(0.1) 0"), _XZ, _XZ, _XZ],
    "LEAKAGE(0.1) 0\nX_ERROR(0.1) 0": [("noisy", "LEAKAGE(0.1) 0"), _XZ, ("noisy", "LEAKAGE(0.1) 0"), ("noisy", "LEAKAGE(0.1) 0"), _XZ, _XZ, _XZ],
    "M 0\nLEAKAGE(0.1) 0": [("measurement", "M 0"), ("measurement", "M 0"), ("noisy", "LEAKAGE(0.1) 0"), ("measurement", "M 0"), _XZ, ("measurement", "M 0"), _XZ],
    "LEAKAGE(0.1) 0\nM 0": [("noisy", "LEAKAGE(0.1) 0"), ("measurement", "M 0"), ("noisy", "LEAKAGE(0.1) 0"), ("noisy", "LEAKAGE(0.1) 0"), _XZ, ("measurement", "M 0"), _XZ],
    "REPEAT 2 {\nH 0\nRL 0\n}": [("reset", "RL 0"), ("reset", "RL 0"), ("reset", "RL 0"), _XZ, ("reset", "RL 0"), _XZ, _XZ],
}
IGNORES = ["", "n", "m", "r", "mn", "nr", "mnr"]
_FLAG = {"noisy": "ignore_noise", "measurement": "ignore_measurement", "reset": "ignore_reset"}


@pytest.mark.parametrize("text", list(TABLEAU))
def test_to_tableau_refuses_leakage_as_deltakit_does(text):
    c = stim.Circuit(text)
    for ignore, want in zip(IGNORES, TABLEAU[text]):
        kwargs = {"ignore_noise": "n" in ignore, "ignore_measurement": "m" in ignore, "ignore_reset": "r" in ignore}
        if isinstance(want, str):
            assert str(c.to_tableau(**kwargs)) == want, ignore
            continue
        kind, first = want
        message = f"The circuit has no well-defined tableau because it contains {kind} operations.\nTo ignore {kind} operations, pass the argument {_FLAG[kind]}=True.\nThe first {kind} operation is: {first}"
        with pytest.raises(ValueError) as e:
            c.to_tableau(**kwargs)
        assert str(e.value) == message, ignore


TIME_REVERSED = {
    "R 0\nCX(0.1,0.2,0.3,0.4) 0 1\nM 0": "M 0\nCX(0.1, 0.2, 0.3, 0.4) 0 1\nM 0",
    "R 0\nHERALD_LEAKAGE_EVENT !0 1\nCX(0.1,0.2,0.3,0.4) 0 1\nRELAX(0.3) 1\nM 0": "RELAX(0.3) 1",
    "R 0\nHERALD_LEAKAGE_EVENT(0.1) 0\nM 0": "HERALD_LEAKAGE_EVENT(0.1) 0",
    "R 0\nLEAKAGE(0.1) 0\nM 0": "LEAKAGE(0.1) 0",
    "R 0\nLEAKAGE(0.1) 0\nRL 0\nM 0": "RL 0",
    "R 0\nRELAX(0.1) 0\nM 0": "RELAX(0.1) 0",
    "R 0\nRL 0\nM 0": "RL 0",
    "REPEAT 2 {\nRL 0\nLEAKAGE(0.2) 1\n}\nM 0": "LEAKAGE(0.2) 1",
}


@pytest.mark.parametrize("text", list(TIME_REVERSED))
def test_time_reversal_refuses_leakage_as_deltakit_does(text):
    c, want = stim.Circuit(text), TIME_REVERSED[text]
    if "\n" in want:
        got, flows = c.time_reversed_for_flows([])
        assert isinstance(got, stim.Circuit) and str(got) == want and flows == []
        return
    for kwargs in [{}, {"dont_turn_measurements_into_resets": True}]:
        with pytest.raises(ValueError) as e:
            c.time_reversed_for_flows([], **kwargs)
        assert str(e.value) == f"Don't know how to invert {want}"


APPROX = [
    ("R 0\nLEAKAGE(0.1) 0\nRL 0\nM 0", "R 0\nLEAKAGE(0.1001) 0\nRL 0\nM 0", True),
    ("R 0\nLEAKAGE(0.1) 0\nM 0", "R 0\nLEAKAGE(0.2) 0\nM 0", False),
    ("CX(0.1,0.2,0.3,0.4) 0 1", "CX(0.1001,0.2,0.3,0.4) 0 1", True),
    ("CX(0.1,0.2,0.3,0.4) 0 1", "CX(0.1,0.2,0.3,0.5) 0 1", False),
    ("RELAX(0.3) 1", "RELAX(0.305) 1", True),
    ("RELAX(0.3) 1", "LEAKAGE(0.3) 1", False),
    ("HERALD_LEAKAGE_EVENT(0.1) 0", "HERALD_LEAKAGE_EVENT(0.1001) 0", True),
    ("HERALD_LEAKAGE_EVENT 0", "HERALD_LEAKAGE_EVENT 1", False),
    ("LEAKAGE(0.1) 0\nX_ERROR(0.1) 0", "LEAKAGE(0.1) 0\nX_ERROR(0.2) 0", False),
]


@pytest.mark.parametrize(("a", "b", "want"), APPROX)
def test_approx_equals_compares_deltakits_arguments(a, b, want):
    assert stim.Circuit(a).approx_equals(stim.Circuit(b), atol=0.01) is want
    assert stim.Circuit(b).approx_equals(stim.Circuit(a), atol=0.01) is want


# --- explained errors ------------------------------------------------------------------------

# deltakit_stim 0.2.7 explains only Stim's own noise: its leakage, relaxation and heralds are not
# fault locations of its explained errors (a herald's flip, in the model as `error(p) D…`, is not
# explained at all).
EXPLAINED = [
    "R 0 1\nX_ERROR(0.1) 1\nLEAKAGE(0.1) 0\nM 1\nDETECTOR rec[-1]",
    "R 0 1\nX_ERROR(0.1) 1\nRL 0\nM 1\nDETECTOR rec[-1]",
    "R 0 1\nX_ERROR(0.1) 1\nCX(0.1,0.1,0.1,0.1) 0 1\nM 1\nDETECTOR rec[-1]",
    "R 0 1\nX_ERROR(0.1) 1\nHERALD_LEAKAGE_EVENT(0.01) 0\nM 1\nDETECTOR rec[-1]\nDETECTOR rec[-2]",
    "R 0 1\nX_ERROR(0.1) 1\nLEAKAGE(0.1) 0\nHERALD_LEAKAGE_EVENT 0\nM 1 0\nDETECTOR rec[-2]\nDETECTOR rec[-1]",
    "R 0 1\nX_ERROR(0.1) 1\nLEAKAGE(0.1) 0\nCX(0.1,0.1,0.1,0.1) 0 1\nHERALD_LEAKAGE_EVENT 1\nM 1\nDETECTOR rec[-1]\nDETECTOR rec[-2]",
]


@pytest.mark.parametrize("text", EXPLAINED)
def test_explained_errors_are_stims_noise_alone(text):
    explained = stim.Circuit(text).explain_detector_error_model_errors()
    got = [(str(e.dem_error_terms[0].dem_target), [loc.instruction_targets.gate for loc in e.circuit_error_locations], [loc.stack_frames[0].instruction_offset for loc in e.circuit_error_locations]) for e in explained]
    assert got == [("D0", ["X_ERROR"], [1])]


def test_deltakits_parse_explained_dem_reads_the_explanation():
    c = stim.Circuit("R 0 1\nX_ERROR(0.1) 1\nLEAKAGE(0.1) 0\nHERALD_LEAKAGE_EVENT(0.01) 0\nM 1\nDETECTOR(0, 0) rec[-1]\nDETECTOR(1, 0) rec[-2]")
    explained = c.explain_detector_error_model_errors()
    assert [loc.instruction_targets.gate for e in explained for loc in e.circuit_error_locations] == ["X_ERROR"]
    graph = parse_explained_dem(explained)
    assert len(graph.edges) == 1
    # With a filter, a fault only leakage explains is kept, with no Stim location.
    filtered = c.explain_detector_error_model_errors(dem_filter=sq.DetectorErrorModel("error(0.01) D1"))
    assert [[loc.instruction_targets.gate for loc in e.circuit_error_locations] for e in filtered] == [[]]
