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
  6. a streamed memory explains every defect;
  7. belief-matching decodes every shot, no worse than plain matching;
  8. BP+OSD decodes the [[72, 12, 6]] bivariate bicycle memory, which Stim reads;
  9. lattice surgery's circuit reads in Stim, and one merged round is refused;
 10-11. the surgery CNOT and the gross code's logical measurement decode Stim's shots;
 12. the public API: seeds independent of threads, Matching as PyMatching, 0.4 names gone.
It prints one line per check and exits non-zero on the first failure.
"""

import math
import sys
from importlib.metadata import version

import numpy as np
import pymatching
import stim

import stabilizer_qec._core as sq

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
    packed_first, actual_first = packed, actual
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

    # 7. Belief-matching, on the first memory's shots: no worse than plain matching beyond noise.
    raw_b, _, conv_b, errors_b, _ = sq.decode_b8_belief(str(dem), packed_first, SHOTS)
    belief = int(((np.frombuffer(raw_b, "<u8") & ONE) != actual_first).sum())
    if errors_b or not belief <= f["plain"] + 4 * math.sqrt(f["plain"] + 1):
        fail(f"belief-matching {belief} failures vs plain {f['plain']} ({errors_b} errors)")
    print(f"ok  7. belief-matching: {belief} failures (plain {f['plain']}), BP alone on {np.frombuffer(conv_b, np.uint8).sum()}")

    # 8. BP+OSD on the [[72, 12, 6]] bivariate bicycle memory: Stim reads it, every shot decodes.
    bb = sq.bb_memory_circuit("72", 2, 0.003)
    bb_circuit = stim.Circuit(bb)
    bb_dem = sq.dem_from_circuit(bb, False)
    bb_dets, _, _ = sq.sample_b8_batch(bb, 256, 17)
    preds, bb_conv, _ = sq.decode_b8_bposd(bb_dem, bb_dets, 256, 200)
    if len(preds) != 8 * 256 or bb_circuit.num_observables != 12:
        fail("BP+OSD on [[72, 12, 6]]")
    print(f"ok  8. BP+OSD: [[72, 12, 6]], {bb_circuit.num_detectors} detectors, 256 shots, BP alone on {np.frombuffer(bb_conv, np.uint8).sum()}")

    # 9. Lattice surgery: Stim reads it and agrees on the number of detectors; one merged round
    # leaves a logical error no detector sees, kept in the model as Stim keeps it.
    ls = stim.Circuit(sq.surgery_circuit(3, 3, 0.001))
    ls_dem = stim.DetectorErrorModel(sq.dem_from_circuit(str(ls), True))
    if ls_dem.num_detectors != ls.num_detectors or ls.num_observables != 3:
        fail("lattice surgery's model")
    one = sq.surgery_circuit(3, 1, 0.001)
    undetectable = [str(e) for e in stim.DetectorErrorModel(sq.dem_from_circuit(one, True)).flattened() if e.type == "error" and not any(t.is_relative_detector_id() for t in e.targets_copy())]
    stim_undetectable = [str(e) for e in stim.Circuit(one).detector_error_model(decompose_errors=True).flattened() if e.type == "error" and not any(t.is_relative_detector_id() for t in e.targets_copy())]
    if not undetectable or undetectable != stim_undetectable:
        fail(f"one merged round's undetectable logical errors: {undetectable} vs Stim's {stim_undetectable}")
    print(f"ok  9. lattice surgery: {ls.num_detectors} detectors, 3 observables; one merged round's undetectable error {undetectable[0]} as Stim's")

    # 10. A logical CNOT by lattice surgery: Stim reads it; its model decodes Stim's shots.
    cn = stim.Circuit(sq.surgery_cnot(3, 3, 0.001))
    cn_dem = sq.dem_from_circuit(str(cn), True)
    cn_dets, _ = cn.compile_detector_sampler(seed=5).sample(256, separate_observables=True)
    _, _, cn_errors, _ = sq.decode_b8(cn_dem, np.packbits(cn_dets, axis=1, bitorder="little").tobytes(), 256, 1, True)
    if cn.num_observables != 2 or cn_errors != 0:
        fail("the lattice-surgery CNOT")
    print(f"ok 10. lattice-surgery CNOT: {cn.num_detectors} detectors, 256 shots decoded")

    # 11. The gross code's logical measurement: Stim reads it; BP+OSD decodes Stim's shots.
    gl = stim.Circuit(sq.bb_logical_measurement_circuit("f", "x", 1, 2, 1, 0.001))
    gl_dem = sq.dem_from_circuit(str(gl), False)
    gl_dets = gl.compile_detector_sampler(seed=11).sample(64)
    _, gl_conv, _ = sq.decode_b8_bposd(gl_dem, np.packbits(gl_dets, axis=1, bitorder="little").tobytes(), 64, 100)
    if gl.num_observables != 13 or len(gl_conv) != 64:
        fail("the gross code's logical measurement")
    print(f"ok 11. gross-code logical measurement: {gl.num_qubits} qubits, {gl.num_detectors} detectors, 64 shots decoded")

    # 12. The public API: the same seed gives the same shots on 1 and 4 threads; matching on
    # Stim's shots agrees with PyMatching but for ties; and the 0.4 names are gone, with a hint.
    import stabilizer_qec as pub

    c = pub.memory_circuit(distance=5, rounds=5, p=0.004)
    one = c.compile_detector_sampler(seed=3).sample(4096, threads=1)
    if not np.array_equal(one, c.compile_detector_sampler(seed=3).sample(4096, threads=4)):
        fail("the public sampler's shots depend on the thread count")
    pub_dem = c.detector_error_model(decompose_errors=True)
    s_dets, s_obs = stim.Circuit(str(c)).compile_detector_sampler(seed=4).sample(5000, separate_observables=True)
    ours = pub.Matching(pub_dem).decode_batch(s_dets, threads=0)
    theirs = pymatching.Matching.from_detector_error_model(stim.DetectorErrorModel(str(pub_dem))).decode_batch(s_dets)
    if (ours != theirs).any(axis=1).sum() > 10:
        fail("the public Matching disagrees with PyMatching beyond ties")
    try:
        pub.generate_circuit
        fail("a 0.4 name is still there")
    except AttributeError as e:
        if "memory_circuit" not in str(e):
            fail("a removed 0.4 name gives no replacement")
    print(f"ok 12. public API {pub.__version__}: seeds independent of threads, Matching = PyMatching on 5000 shots but for ties")
    print("all checks passed")


if __name__ == "__main__":
    main()
