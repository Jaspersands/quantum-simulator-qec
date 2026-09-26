"""The installed package, checked end to end against Stim and PyMatching.

Run from a fresh virtualenv with the wheel, numpy, stim and pymatching
installed, from a directory outside the repository (so the import is the
installed package, not a build lying around):

    python tools/smoke.py

It generates a d = 5 SD6 memory, and checks that:
  1. Stim reads our circuit and builds the same number of detectors and the
     same error mechanisms from it as our model has;
  2. raw measurements sampled by Stim convert to exactly Stim's detection events;
  3. our batch sampler's detection rate matches Stim's;
  4. on Stim's shots, our plain matcher ties PyMatching on every shot where they
     differ, and our correlated matcher's failures sit within noise of
     PyMatching's enable_correlations=True;
  5. window decoding, sliding and parallel, explains every defect and fails no
     more often than a small margin above global decoding;
  6. a streamed memory explains every defect.
It prints one line per check and exits non-zero on the first failure.
"""

import math
import sys
from importlib.metadata import version

import numpy as np
import pymatching
import stim

import stabilizer_qec as sq

SHOTS = 20_000
ONE = np.uint64(1)


def fail(message):
    print(f"FAIL: {message}")
    sys.exit(1)


def within(a, b, sigmas=4.0):
    """Two failure counts from the same number of shots agree within noise."""
    return abs(a - b) <= sigmas * math.sqrt(max(a + b, 1))


def mechanisms(dem):
    """{symptom: p} with the symptom as a sorted tuple of targets, identical
    symptoms merged (p xor q)."""
    out = {}
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        key = tuple(sorted(str(t) for t in inst.targets_copy()))
        q = inst.args_copy()[0]
        p = out.get(key, 0.0)
        out[key] = p * (1 - q) + q * (1 - p)
    return out


def main():
    print(f"stabilizer-qec {version('stabilizer-qec')} from {sq.__file__}")
    print(f"stim {stim.__version__}, pymatching {pymatching.__version__}, numpy {np.__version__}")
    d, p = 5, 0.004
    text = sq.generate_circuit("rotated", d, d, "sd6", p)
    circuit = stim.Circuit(text)
    dem = circuit.detector_error_model(decompose_errors=True)

    # 1. The same model: every symptom's probability, identical symptoms merged.
    ours = mechanisms(stim.DetectorErrorModel(sq.dem_from_circuit(text, False)))
    theirs = mechanisms(circuit.detector_error_model(decompose_errors=False))
    if set(ours) != set(theirs):
        fail(f"{len(set(ours) ^ set(theirs))} symptoms in one model and not the other")
    worst = max(abs(ours[k] - theirs[k]) / theirs[k] for k in theirs)
    if worst > 1e-9:
        fail(f"probabilities differ from Stim's by up to {worst:.1e} (relative)")
    print(f"ok  1. model: {dem.num_detectors} detectors, {len(ours)} mechanisms, as Stim's (worst {worst:.0e})")

    # 2. Measurements to detection events.
    meas = circuit.compile_sampler(seed=7).sample(SHOTS, bit_packed=True)
    their_dets, their_obs = circuit.compile_m2d_converter().convert(
        measurements=meas, separate_observables=True, bit_packed=True)
    our_dets, our_obs = sq.m2d_b8(text, meas.tobytes(), b"", SHOTS)
    if our_dets != their_dets.tobytes() or our_obs != their_obs.tobytes():
        fail("m2d differs from Stim's")
    print(f"ok  2. m2d: {SHOTS} shots bit for bit")

    # 3. The batch sampler.
    dets_b, _, seconds = sq.sample_b8_batch(text, SHOTS, 11)
    ours_rate = np.unpackbits(np.frombuffer(dets_b, np.uint8), bitorder="little").sum()
    stim_dets = circuit.compile_detector_sampler(seed=12).sample(SHOTS)
    theirs_rate = stim_dets.sum()
    if not within(int(ours_rate), int(theirs_rate), 5):
        fail(f"detection events: ours {ours_rate}, Stim's {theirs_rate}")
    print(f"ok  3. sampler: {ours_rate} detection events vs Stim's {theirs_rate} ({seconds * 1e6 / SHOTS:.2f} us/shot)")

    # 4. Matching.
    dets, obs = circuit.compile_detector_sampler(seed=13).sample(SHOTS, separate_observables=True)
    actual = obs[:, 0].astype(np.uint64)
    packed = np.packbits(dets, axis=1, bitorder="little").tobytes()
    plain = pymatching.Matching.from_detector_error_model(dem)
    pm = plain.decode_batch(dets)[:, 0].astype(np.uint64)
    pred_b, weight_b, errors, _ = sq.decode_b8(str(dem), packed, SHOTS, threads=0)
    pred = np.frombuffer(pred_b, "<u8") & ONE
    weights = np.frombuffer(weight_b, "<f8")
    if errors:
        fail(f"{errors} shots failed to decode")
    non_ties = 0
    disagree = np.nonzero(pred != pm)[0]
    for i in disagree:
        _, w = plain.decode(dets[i], return_weight=True)
        non_ties += abs(w - weights[i]) > 1e-3 * max(1.0, abs(w))
    if non_ties:
        fail(f"{non_ties} of {len(disagree)} disagreements with PyMatching are not ties")
    corr = pymatching.Matching.from_detector_error_model(dem, enable_correlations=True)
    pm_c = corr.decode_batch(dets, enable_correlations=True)[:, 0].astype(np.uint64)
    pred_c_b, _, errors_c, _ = sq.decode_b8(str(dem), packed, SHOTS, threads=0, correlated=True)
    pred_c = np.frombuffer(pred_c_b, "<u8") & ONE
    f = dict(plain=int((pred != actual).sum()), pm=int((pm != actual).sum()),
             corr=int((pred_c != actual).sum()), pm_c=int((pm_c != actual).sum()))
    if errors_c or not within(f["corr"], f["pm_c"]):
        fail(f"correlated failures {f['corr']} vs PyMatching's {f['pm_c']} ({errors_c} errors)")
    print(f"ok  4. matching: plain {f['plain']} failures (PyMatching {f['pm']}, {len(disagree)} ties), "
          f"correlated {f['corr']} (PyMatching {f['pm_c']})")

    # 5. Windows, on a longer memory.
    rounds = 4 * d
    long_text = sq.generate_circuit("rotated", d, rounds, "sd6", p)
    long = stim.Circuit(long_text)
    long_dem = str(long.detector_error_model(decompose_errors=True))
    dets, obs = long.compile_detector_sampler(seed=14).sample(SHOTS, separate_observables=True)
    actual = obs[:, 0].astype(np.uint64)
    packed = np.packbits(dets, axis=1, bitorder="little").tobytes()
    g_b, _, _, _ = sq.decode_b8(long_dem, packed, SHOTS, threads=0)
    g = int(((np.frombuffer(g_b, "<u8") & ONE) != actual).sum())
    parts = [f"global {g}"]
    for mode in ("sliding", "parallel"):
        w_b, unexplained, _, windows = sq.decode_b8_window(long_dem, packed, SHOTS, d, d, mode)
        w = int(((np.frombuffer(w_b, "<u8") & ONE) != actual).sum())
        if unexplained or w > g + 4 * math.sqrt(g + 1):
            fail(f"{mode} windows: {w} failures vs global {g}, {unexplained} unexplained")
        parts.append(f"{mode} {w} over {len(windows)} windows")
    print(f"ok  5. windows at {rounds} rounds: " + ", ".join(parts))

    # 6. A stream.
    failures, streams, unexplained, _, windows, wall = sq.stream_decode(
        "rotated", d, p, 500, d, d, "parallel", correlated=True, batches=2)
    if unexplained:
        fail(f"stream: {unexplained} defects unexplained")
    print(f"ok  6. stream: {streams} streams x 500 rounds, {failures} failures, {len(windows)} windows, {wall:.1f} s")
    print("all checks passed")


if __name__ == "__main__":
    main()
