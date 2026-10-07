"""The text timeline is Stim's, character for character: its moments, labels, boxes over TICK
groups, loops drawn once in terms of ``iter``, on hand-made circuits that each pin one rule,
Stim's generated memories, and random circuits using every gate."""

from __future__ import annotations

import numpy as np
import pytest

import stabilizer_qec as sq

stim = pytest.importorskip("stim")

RULES = {
    "single": "H 0",
    "two gates on a qubit": "H 0\nS 0",
    "parallel": "H 0 1",
    "a CX spans its wires": "CX 0 2",
    "a CX blocks the wire it crosses": "CX 0 2\nH 1",
    "TICK groups are boxed": "H 0\nTICK\nH 0 1\nS 0\nTICK\nX 1",
    "a TICK always moves on": "TICK\nH 0",
    "two TICKs": "H 0\nTICK\nTICK\nH 0",
    "measurements and annotations": "M 0 1\nMR 2\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-2] rec[-3]",
    "noise": "X_ERROR(0.1) 0\nDEPOLARIZE2(0.01) 0 1\nE(0.1) X0 Z2\nPAULI_CHANNEL_1(0.1,0.2,0.3) 1",
    "a correlated error opens a moment": "H 0\nE(0.1) X1\nH 2",
    "coordinates": "QUBIT_COORDS(1, 2) 0\nQUBIT_COORDS(3) 1\nH 0 1\nM 0\nDETECTOR(4, 5) rec[-1]",
    "a detector at its qubit's coordinates": "QUBIT_COORDS(0, 0) 0\nQUBIT_COORDS(5, 5) 1\nM 0 1\nDETECTOR(5, 5) rec[-2]",
    "products": "MPP X0*Y1 Z2\nSPP X0*Z2",
    "a loop": "H 0\nREPEAT 3 {\n  CX 0 1\n  M 1\n  DETECTOR(0, 1) rec[-1]\n  SHIFT_COORDS(0, 1)\n}\nM 0",
    "a loop starting with TICK": "REPEAT 2 {\n  TICK\n  H 0\n}",
    "a loop ending with TICK": "REPEAT 2 {\n  H 0\n  TICK\n}\nH 0",
    "a TICK before a loop": "H 0\nTICK\nREPEAT 2 {\n  H 0\n}",
    "nested loops": "REPEAT 2 {\n  H 0\n  REPEAT 3 {\n    M 0\n    DETECTOR(1) rec[-1]\n    SHIFT_COORDS(1)\n  }\n  TICK\n}",
    "tags are not drawn": "H[a] 0\nX_ERROR[noise](0.1) 1",
    "feedback": "M 0\nCX rec[-1] 1\nCZ sweep[0] 2\nXCZ 0 rec[-1]",
    "explicit zero arguments": "M(0.1) 0\nMPAD 1 0\nMPP(0.1) !X0*Z1\nM(0) !2\nDETECTOR",
}


def same(text):
    want = str(stim.Circuit(text).diagram("timeline-text"))
    assert str(sq.Circuit(text).diagram("timeline-text")) == want + "\n", text


@pytest.mark.parametrize("name", list(RULES))
def test_rules(name):
    same(RULES[name])


@pytest.mark.parametrize("task", ["repetition_code:memory", "surface_code:rotated_memory_x", "surface_code:unrotated_memory_z", "color_code:memory_xyz"])
@pytest.mark.parametrize("rounds", [1, 2, 3, 5])
def test_generated_circuits(task, rounds):
    if task.startswith("color") and rounds < 2:
        return
    same(str(stim.Circuit.generated(task, distance=3, rounds=rounds, after_clifford_depolarization=0.001, before_measure_flip_probability=0.002)))


def test_random_circuits_using_every_gate():
    from test_stim_gates import random_case, tagged

    for seed in range(200):
        text = random_case(seed, disjoint=seed % 3 == 1)
        if seed % 3 == 2:
            text = tagged(np.random.default_rng(seed), text)
        same(text)
