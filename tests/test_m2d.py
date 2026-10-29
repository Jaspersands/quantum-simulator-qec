from __future__ import annotations

import numpy as np
import pytest

import stabilizer_qec as sq

SWEPT = """R 0 1
CX sweep[0] 0
X_ERROR(0.2) 0 1
M 0 1
DETECTOR rec[-2]
DETECTOR rec[-1]
OBSERVABLE_INCLUDE(0) rec[-1]
"""


def test_matches_stims_m2d(d5):
    stim = pytest.importorskip("stim")
    sc = stim.Circuit(str(d5))
    meas = sc.compile_sampler(seed=4).sample(2000)
    want_d, want_o = sc.compile_m2d_converter().convert(measurements=meas, separate_observables=True)
    conv = d5.compile_m2d_converter()
    d, o = conv.convert(measurements=meas, separate_observables=True)
    assert np.array_equal(d, want_d) and np.array_equal(o, want_o)
    packed = np.packbits(meas, axis=1, bitorder="little")
    pd = conv.convert(measurements=packed, bit_packed=True)
    assert np.array_equal(np.unpackbits(pd, axis=1, count=d5.num_detectors, bitorder="little").astype(bool), want_d)


def test_sweep_bits():
    stim = pytest.importorskip("stim")
    sc = stim.Circuit(SWEPT)
    meas = sc.compile_sampler(seed=1).sample(500)
    sweeps = np.random.default_rng(1).random((500, 1)) < 0.5
    want = sc.compile_m2d_converter().convert(measurements=meas, sweep_bits=sweeps, append_observables=True)
    got = sq.Circuit(SWEPT).compile_m2d_converter().convert(measurements=meas, sweep_bits=sweeps, append_observables=True)
    assert np.array_equal(got, want)


def test_shape_errors(d3):
    conv = d3.compile_m2d_converter()
    with pytest.raises(ValueError):
        conv.convert(measurements=np.zeros((4, d3.num_measurements + 1), dtype=bool))
    with pytest.raises(ValueError):
        conv.convert(measurements=np.zeros(d3.num_measurements, dtype=bool))
    with pytest.raises(ValueError, match="dense tableau"):
        sq.Circuit("M 20000").compile_m2d_converter()



@pytest.mark.parametrize("basis, flips", [("MX", True), ("MY", True), ("MRX", True), ("MRY", False)])
def test_a_sweep_bit_reaching_a_collapsed_qubit_flips_as_stims_frame_does(basis, flips):
    # A detector that depends on a random outcome reads the sweep bit as Stim's frame simulator
    # does: the X the bit applies survives an X or Y measurement and flips the Z measurement
    # after it; an X reset keeps it too (it commutes with |+⟩), a Y reset turns it into nothing.
    stim = pytest.importorskip("stim")
    text = f"CX sweep[0] 0\n{basis} 0\nM 0\nDETECTOR rec[-1]"
    meas = np.zeros((2, 2), dtype=bool)
    sweeps = np.array([[0], [1]], dtype=bool)
    want = stim.Circuit(text).compile_m2d_converter().convert(measurements=meas, sweep_bits=sweeps, append_observables=False)
    assert want.tolist() == [[False], [flips]]
    got = sq.Circuit(text).compile_m2d_converter().convert(measurements=meas, sweep_bits=sweeps, append_observables=False)
    assert np.array_equal(got, want)


MEASURES = ("M", "MX", "MY", "MR", "MRX", "MRY", "MPP", "MXX", "MZZ", "MPAD")


def random_swept_case(rng, sweep_bits):
    """A random Clifford circuit with sweep-controlled gates, collapses in every basis,
    feedback, MPP, a loop, detectors on random records and observables with Pauli targets:
    most of its detectors depend on random outcomes."""
    n = rng.randint(2, 4)
    state = {"measured": 0}

    def detector():
        m = state["measured"]
        return "DETECTOR " + " ".join(f"rec[-{k}]" for k in rng.sample(range(1, m + 1), rng.randint(1, min(3, m))))

    def body(steps, records):
        out = []
        for _ in range(steps):
            q = rng.randrange(n)
            a, b = rng.sample(range(n), 2)
            s = rng.randrange(sweep_bits)
            kind = rng.random()
            if kind < 0.25:
                out.append(rng.choice([f"CX sweep[{s}] {q}", f"CY sweep[{s}] {q}", f"CZ sweep[{s}] {q}", f"XCZ {q} sweep[{s}]", f"YCZ {q} sweep[{s}]"]))
            elif kind < 0.45:
                out.append(rng.choice(["M", "MX", "MY", "MR", "MRX", "MRY"]) + f" {q}")
            elif kind < 0.52:
                out.append(rng.choice([f"MPP X{a}*Z{b}", f"MXX {a} {b}", f"MZZ {a} {b}", f"MPP Y{a}", "MPAD 1"]))
            elif kind < 0.58:
                out.append(rng.choice(["R", "RX", "RY"]) + f" {q}")
            elif kind < 0.66 and records and state["measured"]:
                out.append(f"{rng.choice(['CX', 'CY', 'CZ'])} rec[-{rng.randint(1, min(state['measured'], 3))}] {q}")
            elif kind < 0.85 or not (records and state["measured"]):
                out.append(rng.choice([f"H {q}", f"S {q}", f"CX {a} {b}", f"CZ {a} {b}", f"SQRT_X {q}", f"H_YZ {q}", f"ISWAP {a} {b}", f"C_XYZ {q}"]))
            else:
                out.append(detector())
            state["measured"] += out[-1].split()[0] in MEASURES
        return out

    # The last bit always does something, so the circuit has all the bits asked for.
    lines = [f"CX sweep[{sweep_bits - 1}] {rng.randrange(n)}"] + body(rng.randint(3, 10), True)
    if rng.random() < 0.5:
        # A loop of gates and measurements only, so its records need no counting per pass.
        before = state["measured"]
        inner = body(rng.randint(2, 6), False)
        reps = rng.randint(1, 3)
        state["measured"] = before + (state["measured"] - before) * reps
        lines += [f"REPEAT {reps} {{"] + [" " + line for line in inner] + ["}"]
    lines.append("M " + " ".join(map(str, range(n))))
    state["measured"] += n
    for _ in range(3):
        lines.append(detector())
    lines.append(f"OBSERVABLE_INCLUDE(0) rec[-1] {rng.choice(['X', 'Y', 'Z'])}{rng.randrange(n)}")
    lines.append(f"OBSERVABLE_INCLUDE(1) {rng.choice(['X', 'Y', 'Z'])}{rng.randrange(n)}")
    return "\n".join(lines)


@pytest.mark.parametrize("sweep_bits", [1, 3, 70])
def test_random_swept_circuits_convert_as_stims_converter_does(sweep_bits):
    import random

    stim = pytest.importorskip("stim")
    rng = random.Random(sweep_bits)
    compared = moved = 0
    for _ in range(150):
        text = random_swept_case(rng, sweep_bits)
        sc = stim.Circuit(text)
        try:
            conv = sc.compile_m2d_converter()
        except ValueError:
            continue
        meas = sc.compile_sampler(seed=5).sample(16)
        assert sc.num_sweep_bits == sweep_bits
        sweeps = np.random.default_rng(compared).random((16, sweep_bits)) < 0.5
        ours = sq.Circuit(text).compile_m2d_converter()
        # What the sweep bits flip, compared apart from the reference: where a random outcome
        # meets an entangled partner, Stim's reference sample is its tableau's own choice, which
        # this package's reference does not always make (a separate matter from the sweeps).
        flips = [
            c.convert(measurements=meas, sweep_bits=sweeps, append_observables=True) ^ c.convert(measurements=meas, sweep_bits=np.zeros_like(sweeps), append_observables=True)
            for c in (conv, ours)
        ]
        assert np.array_equal(flips[1], flips[0]), text
        compared += 1
        moved += bool(flips[0].any())
    assert compared >= 100 and moved >= 50, (compared, moved)
