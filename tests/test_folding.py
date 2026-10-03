"""Error models with their loops folded, as Stim folds them: the same ``repeat`` blocks, the
same declarations, and the same faults in every stretch between them."""

from __future__ import annotations

import re
import time

import numpy as np
import pytest

import stabilizer_qec as sq

stim = pytest.importorskip("stim")


def merged(run):
    acc = {}
    for tag, pieces, p in run:
        q = acc.get((tag, pieces), 0.0)
        acc[(tag, pieces)] = q * (1 - p) + p * (1 - q)
    return tuple(sorted((k[0], k[1], float(f"{v:.8g}")) for k, v in acc.items()))


def shape(text):
    """A model's structure: each run of errors as a set (pieces sorted, and a fault Stim lists
    once per order of its pieces merged into one), declarations and shifts as written, repeat
    blocks nested."""
    out, run = [], []
    stack = [out]
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.fullmatch(r"error(\[[^\]]*\])?\(([^)]*)\)(.*)", line)
        if m:
            pieces = tuple(sorted(tuple(sorted(p.split())) for p in m.group(3).split("^")))
            run.append((m.group(1) or "", pieces, float(m.group(2))))
            continue
        if run:
            stack[-1].append(("errors", merged(run)))
            run.clear()
        if line.startswith("repeat"):
            block = []
            stack[-1].append((line, block))
            stack.append(block)
        elif line == "}":
            stack.pop()
        else:
            stack[-1].append(re.sub(r"\(([^)]*)\)", lambda a: "(" + ", ".join(f"{float(v):.9g}" for v in a.group(1).split(",")) + ")", line))
    if run:
        stack[-1].append(("errors", merged(run)))
    return out


def generated(code, d, rounds, p=0.001):
    return stim.Circuit.generated(code, distance=d, rounds=rounds, after_clifford_depolarization=p, before_round_data_depolarization=2 * p, before_measure_flip_probability=3 * p, after_reset_flip_probability=p)


CODES = ["surface_code:rotated_memory_z", "surface_code:rotated_memory_x", "surface_code:unrotated_memory_z", "repetition_code:memory", "color_code:memory_xyz"]


def compare(text, **kwargs):
    ours, theirs = sq.Circuit(text), stim.Circuit(text)
    try:
        want = theirs.detector_error_model(**kwargs)
    except ValueError:
        with pytest.raises(ValueError):
            ours.detector_error_model(**kwargs)
        return None
    got = ours.detector_error_model(**kwargs)
    assert shape(str(got)) == shape(str(want))
    assert (got.num_detectors, got.num_observables) == (want.num_detectors, want.num_observables)
    assert got.num_errors == want.num_errors
    assert str(got).strip() == str(want).strip()
    assert shape(str(got.flattened())) == shape(str(want.flattened()))
    return got


@pytest.mark.parametrize("code", CODES)
@pytest.mark.parametrize("d,rounds", [(3, 2), (3, 10), (5, 30)])
@pytest.mark.parametrize("decompose", [False, True])
def test_generated_memories_fold_as_stims(code, d, rounds, decompose):
    got = compare(str(generated(code, d, rounds)), decompose_errors=decompose)
    if got is not None and rounds >= 10:
        assert "repeat" in str(got)


@pytest.mark.parametrize("decompose", [False, True])
def test_flatten_loops_as_stims(decompose):
    text = str(generated("surface_code:rotated_memory_z", 3, 12))
    got = compare(text, decompose_errors=decompose, flatten_loops=True)
    assert "repeat" not in str(got)


def test_nested_tagged_and_odd_loops():
    body = generated("surface_code:rotated_memory_z", 3, 5)
    loop = next(x for x in body if isinstance(x, stim.CircuitRepeatBlock))
    inner = str(loop.body_copy())
    head = str(body[: list(body).index(loop)])
    tail = str(body[list(body).index(loop) + 1 :])
    cases = [
        # Nested: a loop of loops.
        f"{head}\nREPEAT 3 {{\nREPEAT 5 {{\n{inner}\n}}\n}}\n{tail}",
        # Tagged loops and tagged noise inside them.
        f"{head}\nREPEAT[outer] 7 {{\n{inner.replace('DEPOLARIZE1', 'DEPOLARIZE1[idle]')}\n}}\n{tail}",
        # A count that whole periods do not divide, and a loop of one pass.
        f"{head}\nREPEAT 11 {{\n{inner}\n{inner}\n}}\nREPEAT 1 {{\n{inner}\n}}\n{tail}",
        # A loop with no detectors: its faults all flip the same detectors.
        "R 0 1\nREPEAT 50 {\n X_ERROR(0.01) 0\n Z_ERROR(0.02) 1\n TICK\n}\nM 0 1\nDETECTOR rec[-1]\nDETECTOR rec[-2]",
        # A loop that resets nothing: the observable gathers every round.
        "R 0\nREPEAT 40 {\n X_ERROR(0.01) 0\n M 0\n DETECTOR(0, 1) rec[-1]\n SHIFT_COORDS(0, 1)\n}\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]",
    ]
    for text in cases:
        for decompose in (False, True):
            compare(text, decompose_errors=decompose)


def test_a_long_memory_builds_fast_and_decodes_where_it_fits():
    big = generated("surface_code:rotated_memory_z", 11, 10_000)
    c = sq.Circuit(str(big))
    t = time.perf_counter()
    dem = c.detector_error_model(decompose_errors=True)
    assert time.perf_counter() - t < 2.0
    assert dem.num_detectors == 1_200_000 and str(dem).count("\n") < 20_000
    assert shape(str(dem)) == shape(str(big.detector_error_model(decompose_errors=True)))
    # 27 million faults unrolled: past what a decoder takes, and it says so.
    with pytest.raises(ValueError, match="unrolls past"):
        sq.Matching(dem)
    # A long memory that does fit decodes, from its folded model.
    long = generated("surface_code:rotated_memory_z", 3, 5_000, p=0.002)
    c = sq.Circuit(str(long))
    dem = c.detector_error_model(decompose_errors=True)
    dets, obs = c.compile_detector_sampler(seed=1).sample(64, separate_observables=True)
    pred = sq.Matching(dem).decode_batch(dets)
    import pymatching

    want = pymatching.Matching.from_detector_error_model(long.detector_error_model(decompose_errors=True)).decode_batch(dets)
    assert (pred != want).sum() <= 2  # ties aside


def test_folded_models_read_back_and_pickle():
    import pickle

    dem = sq.Circuit(str(generated("surface_code:rotated_memory_x", 3, 20))).detector_error_model(decompose_errors=True)
    again = sq.DetectorErrorModel(str(dem))
    assert str(again) == str(dem) and again.num_errors == dem.num_errors
    assert str(pickle.loads(pickle.dumps(dem))) == str(dem)
    dets = np.zeros((1, dem.num_detectors), dtype=bool)
    assert np.array_equal(sq.Matching(again).decode_batch(dets), sq.Matching(dem).decode_batch(dets))
    # Stim's own folded models read, counted through their loops.
    theirs = generated("color_code:memory_xyz", 5, 1000).detector_error_model()
    ours = sq.DetectorErrorModel(str(theirs))
    assert (ours.num_detectors, ours.num_errors) == (theirs.num_detectors, theirs.num_errors)


@pytest.mark.parametrize("seed", range(20))
def test_random_circuits_in_loops_fold_as_stims(seed):
    from test_stim_gates import random_case

    case = random_case(4000 + seed, disjoint=seed % 2 == 1)
    # Each pass prepares, scrambles, measures and unscrambles afresh; the loop around it, and a
    # second one reading the first's last records, are what is folded.
    text = f"REPEAT {5 + seed % 4} {{\n{case}\n}}\nREPEAT 3 {{\n{case}\nDETECTOR rec[-1] rec[-{1 + 2 * case.count('M ')}]\n}}"
    approximate = seed % 2 == 1
    for decompose in (False, True):
        compare(text, decompose_errors=decompose, approximate_disjoint_errors=approximate)
