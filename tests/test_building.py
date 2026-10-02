"""Circuits built in code, as in Stim: ``append``, ``+``, ``*``, Stim's objects appended."""

from __future__ import annotations

import numpy as np
import pytest

import stabilizer_qec as sq

stim = pytest.importorskip("stim")


def counts(c):
    return (c.num_qubits, c.num_measurements, c.num_detectors, c.num_observables, c.num_sweep_bits)


INSTRUCTIONS = [
    ("H", [0, 1], None),
    ("CX", [0, 2, 1, 3], None),
    ("M", [0, 1], 0.01),
    ("MR", [2], None),
    ("MX", ["!3"], None),
    ("DEPOLARIZE1", [0, 3], 0.02),
    ("DEPOLARIZE2", [0, 1], 0.03),
    ("PAULI_CHANNEL_1", [2], [0.01, 0.02, 0.03]),
    ("E", ["X0", "Z2"], 0.05),
    ("MPP", ["X0", "*", "Z1", "Y2"], None),
    ("DETECTOR", ["rec[-1]", "rec[-2]"], [1, 2, 0]),
    ("OBSERVABLE_INCLUDE", ["rec[-1]", "Z0"], 1),
    ("CZ", ["sweep[2]", 1], None),
    ("SHIFT_COORDS", [], [0, 0, 1]),
    ("TICK", [], None),
]


def stim_targets(targets):
    out = []
    for t in targets:
        if isinstance(t, int):
            out.append(t)
        elif t == "*":
            out.append(stim.target_combiner())
        elif t.startswith("rec["):
            out.append(stim.target_rec(int(t[4:-1])))
        elif t.startswith("sweep["):
            out.append(stim.target_sweep_bit(int(t[6:-1])))
        elif t.startswith("!"):
            out.append(stim.target_inv(int(t[1:])))
        else:
            out.append({"X": stim.target_x, "Y": stim.target_y, "Z": stim.target_z}[t[0]](int(t[1:])))
    return out


@pytest.mark.parametrize("seed", range(20))
def test_appends_build_what_stims_build(seed):
    rng = np.random.default_rng(seed)
    ours, theirs = sq.Circuit("M 0 1 2 3"), stim.Circuit("M 0 1 2 3")
    for _ in range(12):
        name, targets, arg = INSTRUCTIONS[rng.integers(len(INSTRUCTIONS))]
        tag = str(rng.choice(["", "t", "a b"]))
        ours.append(name, targets, arg, tag=tag)
        theirs.append(name, stim_targets(targets), [] if arg is None else arg, tag=tag)
        if rng.random() < 0.3:
            # The same instruction from Stim's GateTargets.
            ours.append(name, stim_targets(targets), arg, tag=tag)
            theirs.append(name, stim_targets(targets), [] if arg is None else arg, tag=tag)
    assert stim.Circuit(str(ours)) == theirs
    assert counts(ours) == counts(sq.Circuit(str(ours))) == (theirs.num_qubits, theirs.num_measurements, theirs.num_detectors, theirs.num_observables, theirs.num_sweep_bits)


@pytest.mark.parametrize("seed", range(20))
def test_sums_and_products_count_and_print_as_stims(seed):
    rng = np.random.default_rng(100 + seed)
    pieces = ["M 0 1\nDETECTOR rec[-1] rec[-3]", "H 0\nCX 0 1\nMR 1\nOBSERVABLE_INCLUDE(2) rec[-1]", "CZ sweep[4] 3\nX_ERROR(0.1) 3\nM 3", "TICK", "REPEAT 3 {\n M 2\n DETECTOR rec[-1]\n}"]
    ours, theirs = sq.Circuit("M 0 1 2 3"), stim.Circuit("M 0 1 2 3")
    for _ in range(6):
        k, n = rng.integers(len(pieces)), int(rng.integers(0, 4))
        a, b = sq.Circuit(pieces[k]), stim.Circuit(pieces[k])
        if rng.random() < 0.5:
            ours, theirs = ours + a * n, theirs + b * n
        else:
            ours += n * a
            theirs += n * b
        if rng.random() < 0.2:
            ours *= 2
            theirs *= 2
    assert stim.Circuit(str(ours)) == theirs
    assert counts(ours) == counts(sq.Circuit(str(ours)))
    assert (ours.num_measurements, ours.num_detectors, ours.num_observables) == (theirs.num_measurements, theirs.num_detectors, theirs.num_observables)


def test_a_memory_built_from_rounds_samples_and_decodes_as_the_generated_one():
    text = str(stim.Circuit.generated("repetition_code:memory", distance=5, rounds=8, after_clifford_depolarization=0.02, before_measure_flip_probability=0.02))
    parsed = stim.Circuit(text)
    first, loop, last = parsed[: next(i for i, x in enumerate(parsed) if isinstance(x, stim.CircuitRepeatBlock))], None, None
    for i, x in enumerate(parsed):
        if isinstance(x, stim.CircuitRepeatBlock):
            loop, last = x, parsed[i + 1 :]
    built = sq.Circuit(str(first)) + sq.Circuit(str(loop.body_copy())) * loop.repeat_count + sq.Circuit(str(last))
    assert built == sq.Circuit(text)
    via_stim = sq.Circuit()
    for x in parsed:
        via_stim.append(x)
    assert via_stim == sq.Circuit(text)
    dem = built.detector_error_model(decompose_errors=True)
    dets, obs = built.compile_detector_sampler(seed=3).sample(5000, separate_observables=True)
    assert (sq.Matching(dem).decode_batch(dets) != obs).any(axis=1).mean() < 0.05


def test_copies_are_independent_and_operators_do_not_mutate():
    a = sq.Circuit("H 0")
    b, c = a.copy(), a + sq.Circuit("X 0")
    b.append("M", 0)
    d = a * 3
    assert str(a) == "H 0\n" and a.num_measurements == 0
    assert b.num_measurements == 1 and str(c) == "H 0\nX 0\n" and "REPEAT 3" in str(d)
    assert str(a * 0) == "" and a * 1 == a and 2 * a == a * 2
    e = a
    e += sq.Circuit("Z 0")
    assert e is a and str(a) == "H 0\nZ 0\n"
    # A circuit added to itself, as Stim allows.
    a += a
    a.append(a)
    assert str(a) == "H 0\nZ 0\n" * 4 and a.num_qubits == 1


def test_bad_appends_change_nothing():
    c = sq.Circuit("M 0")
    before = str(c)
    bad = [
        (("FOO", 0), ValueError),
        (("H\nM", 0), ValueError),
        (("H 0", 1), ValueError),
        (("REPEAT", 2), ValueError),
        (("H", "0 1"), ValueError),
        (("H", "0#x"), ValueError),
        (("H", -1), ValueError),
        (("H", True), TypeError),
        (("H", 1.5), TypeError),
        (("X_ERROR", 0, float("nan")), ValueError),
        (("X_ERROR", 0, 2.0), ValueError),
        (("MPP", ["X0", "*"]), ValueError),
        (("CX", [0]), ValueError),
    ]
    for args, err in bad:
        with pytest.raises(err):
            c.append(*args)
    for tag in ["a]b", "a\nb"]:
        with pytest.raises(ValueError):
            c.append("H", 0, tag=tag)
    with pytest.raises(ValueError):
        c.append(sq.Circuit("H 0"), 1)
    with pytest.raises(TypeError):
        c.append(3)
    with pytest.raises(ValueError):
        c * -1
    assert str(c) == before
    assert c.__add__(3) is NotImplemented and c.__mul__("2") is NotImplemented


def test_stim_objects_append_whole():
    c = sq.Circuit()
    s = stim.Circuit("H[x] 0\nREPEAT[loop] 2 {\n CX 0 1\n M 1\n DETECTOR rec[-1]\n}")
    for x in s:
        c.append(x)
    c.append(stim.Circuit("M 0"))
    c.append_from_stim_program_text("X 0\nM 0")
    assert stim.Circuit(str(c)) == s + stim.Circuit("M 0\nX 0\nM 0")
