"""Circuit and model diagrams: the timeline (text and SVG), detector slices (what each detector
compares at a moment, checked against Stim's on its generated codes) and the matching graph."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

import pytest

import stabilizer_qec as sq

BELL = "R 0 1\nH 0\nTICK\nCX 0 1\nTICK\nM 0 1\nDETECTOR rec[-1] rec[-2]"


def repetition(rounds=2):
    return sq.Circuit(
        "QUBIT_COORDS(0, 0) 0\nQUBIT_COORDS(1, 0) 1\nQUBIT_COORDS(2, 0) 2\n"
        "R 0 1 2\nTICK\nCX 0 1\nTICK\nCX 2 1\nTICK\nMR 1\nDETECTOR(1, 0, 0) rec[-1]\n"
        f"REPEAT {rounds} {{\n TICK\n CX 0 1\n TICK\n CX 2 1\n TICK\n DEPOLARIZE1(0.01) 0 2\n MR 1\n"
        " SHIFT_COORDS(0, 0, 1)\n DETECTOR(1, 0, 0) rec[-1] rec[-2]\n}\n"
        "M 0 2\nDETECTOR(1, 0, 1) rec[-1] rec[-2] rec[-3]\nOBSERVABLE_INCLUDE(0) rec[-1]"
    )


def test_timeline_text_numbers_measurements_and_draws_loops_once():
    text = str(sq.Circuit(BELL).diagram())
    assert "M:rec[0]" in text and "M:rec[1]" in text
    assert "DETECTOR:D0=rec[1]*rec[0]" in text
    looped = str(repetition(3).diagram("timeline-text"))
    assert "REP 3" in looped and looped.count("MR:") == 2


@pytest.mark.parametrize("kind", ["timeline-svg", "detslice-svg"])
def test_the_pictures_are_well_formed_svg(kind):
    d = repetition().diagram(kind, tick=3 if kind.startswith("detslice") else None)
    root = ET.fromstring(str(d))
    assert root.tag.endswith("svg") and float(root.get("width")) > 0
    assert d._repr_svg_() == str(d)


def test_text_diagrams_are_not_pictures():
    d = sq.Circuit(BELL).diagram("timeline-text")
    assert d._repr_svg_() is None and "timeline-text" in repr(d)


def test_detector_slices_of_a_repetition_code():
    c = repetition()
    first = str(c.diagram("detslice-text", tick=1)).splitlines()
    # Before the first round: D0 compares the parity it will read; D1 compares only the
    # ancilla, as the first round's reset fixes it; the observable is the last data qubit.
    assert first[0] == "D0: Z0 Z1 Z2" and "D1: Z1" in first and first[-1] == "L0: Z2"
    later = str(c.diagram("detslice-text", tick=4)).splitlines()
    # After the first round's measurement D0 is read and gone; D1 compares the parity again.
    assert not any(line.startswith("D0:") for line in later) and "D1: Z0 Z1 Z2" in later


def test_detslice_svg_draws_only_qubits_it_uses():
    c = sq.Circuit("QUBIT_COORDS(0, 0) 3\nQUBIT_COORDS(1, 0) 5\nR 3 5\nTICK\nM 3 5\nDETECTOR rec[-1] rec[-2]")
    svg = str(c.diagram("detslice-svg", tick=1))
    labels = re.findall(r">(\d+)</text>", svg)
    assert sorted(labels) == ["3", "5"]


def test_matchgraph_has_an_edge_per_graphlike_fault():
    dem = repetition().detector_error_model(decompose_errors=True)
    svg = str(dem.diagram("matchgraph-svg"))
    root = ET.fromstring(svg)
    titles = [t.text for t in root.iter("{http://www.w3.org/2000/svg}title")]
    assert sum(t.startswith("D") and "p=" in t for t in titles) > 0
    assert sum("p=" not in t for t in titles) == dem.num_detectors


def test_saving_and_bad_types(tmp_path):
    d = sq.Circuit(BELL).diagram("timeline-svg")
    d.save(tmp_path / "t.svg")
    assert (tmp_path / "t.svg").read_text(encoding="utf-8") == str(d)
    with pytest.raises(ValueError):
        sq.Circuit(BELL).diagram("nope")
    with pytest.raises(ValueError):
        sq.Circuit(BELL).detector_error_model().diagram("timeline-svg")
    with pytest.raises(TypeError):
        sq.Circuit(BELL).diagram(3)


def _stim_slice(stim_circuit, tick):
    out = {}
    for line in str(stim_circuit.diagram("detslice-text", tick=tick)).splitlines():
        m = re.match(r"\s*q(\d+):", line)
        if m:
            for p, kind, n in re.findall(r"([XYZ]):([DL])(\d+)", line):
                out.setdefault(f"{kind}{n}", set()).add((int(m.group(1)), p))
    return out


def _our_slice(circuit, tick):
    out = {}
    for line in str(circuit.diagram("detslice-text", tick=tick)).splitlines():
        name, rest = line.split(": ")
        if not name.startswith("L"):  # Stim's slices leave the observables out
            out[name] = {(int(t[1:]), t[0]) for t in rest.split()}
    return out


@pytest.mark.parametrize(
    "code",
    ["surface_code:rotated_memory_z", "surface_code:rotated_memory_x", "repetition_code:memory", "color_code:memory_xyz", "surface_code:unrotated_memory_z"],
)
def test_detector_slices_match_stims(code):
    stim = pytest.importorskip("stim")
    sc = stim.Circuit.generated(code, distance=3, rounds=3, after_clifford_depolarization=0.001)
    ours = sq.Circuit(str(sc))
    for tick in range(sc.num_ticks + 1):
        assert _our_slice(ours, tick) == _stim_slice(sc, tick), f"tick {tick}"


@pytest.mark.parametrize("kind", ["timeslice-svg", "detslice-with-ops-svg", "detslice-svg"])
def test_slices_over_ranges(kind):
    c = sq.Circuit.generated("surface_code:rotated_memory_z", distance=3, rounds=2, after_clifford_depolarization=0.001)
    root = ET.fromstring(str(c.diagram(kind, tick=range(1, 7), rows=2)))
    titles = [t.text for t in root.iter("{http://www.w3.org/2000/svg}title")]
    assert [t for t in titles if t.startswith("Tick")] == [f"Tick {k}" for k in range(1, 7)]
    if kind != "detslice-svg":
        texts = [t.text for t in root.iter("{http://www.w3.org/2000/svg}text")]
        assert "H" in texts or any(t and "MR" in t for t in texts)
    with pytest.raises(ValueError):
        c.diagram(kind, tick=range(3, 3))
