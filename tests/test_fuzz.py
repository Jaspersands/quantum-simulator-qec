"""Property-based fuzzing of the public API: whatever goes in, the only exceptions that come
out are ValueError and TypeError, and what is accepted behaves. ``HYPOTHESIS_PROFILE=long``
runs many more examples."""

from __future__ import annotations

import math
import os
import re

import numpy as np
import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

import stabilizer_qec as sq  # noqa: E402

settings.register_profile("default", max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
settings.register_profile("long", max_examples=3000, deadline=None, suppress_health_check=[HealthCheck.too_slow])
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))

ALLOWED = (ValueError, TypeError)

numbers = st.one_of(
    st.integers(0, 7).map(str),
    st.sampled_from(["0", "-1", "1e300", "nan", "inf", "0.5", "2", "16777215", "16777216", "4294967296", "", "+3"]),
)
probabilities = st.one_of(st.integers(0, 400).map(lambda k: str(k / 4000)), numbers)
target = st.one_of(
    st.integers(0, 5).map(str),
    st.integers(0, 4).map(lambda k: f"rec[-{k}]"),
    st.integers(0, 3).map(lambda k: f"sweep[{k}]"),
    st.integers(0, 5).map(lambda k: f"!{k}"),
    numbers,
)
targets = st.lists(target, min_size=1, max_size=5).map(" ".join)
gates = st.sampled_from(["H", "X", "Y", "Z", "I", "R", "RX", "M", "MX", "MR", "MRX", "CX", "CZ", "S", "MY", "SWAP"])
noise = st.sampled_from(["X_ERROR", "Y_ERROR", "Z_ERROR", "DEPOLARIZE1", "DEPOLARIZE2"])


@st.composite
def lines(draw, depth=0):
    kind = draw(st.integers(0, 9 if depth < 2 else 8))
    if kind <= 2:
        return f"{draw(gates)} {draw(targets)}"
    if kind == 3:
        return f"{draw(noise)}({draw(probabilities)}) {draw(targets)}"
    if kind == 4:
        p = [draw(probabilities) for _ in range(3)]
        return f"PAULI_CHANNEL_1({', '.join(p)}) {draw(targets)}"
    if kind == 5:
        return f"M({draw(probabilities)}) {draw(targets)}"
    if kind == 6:
        recs = " ".join(f"rec[-{draw(st.integers(1, 5))}]" for _ in range(draw(st.integers(1, 3))))
        return f"DETECTOR({draw(numbers)}, {draw(numbers)}) {recs}"
    if kind == 7:
        return f"OBSERVABLE_INCLUDE({draw(st.sampled_from(['0', '1', '63', '64', '-1']))}) rec[-{draw(st.integers(1, 3))}]"
    if kind == 8:
        return draw(
            st.sampled_from(
                ["TICK", "", "# comment", "}", "SHIFT_COORDS(0, 1)", "QUBIT_COORDS(1, 2) 0", "CX rec[-1] 2", "CZ 1 rec[-2]",
                 "XCZ 0 sweep[1]", "CY rec[-0] 1", "HERALDED_ERASE(0.1) 0 1", "HERALDED_PAULI_CHANNEL_1(0.1, 0.1, 0, 0.1) 2",
                 "E(0.1) X0\nELSE_CORRELATED_ERROR(0.2) Z1", "H[t] 0", "M[a#b](0.1) 1", "DETECTOR[d](1) rec[-1]",
                 "OBSERVABLE_INCLUDE(0) X0 !Z1", "OBSERVABLE_INCLUDE[o](1) rec[-1] Y2", "H[unclosed 0", "X[] 1", "H[x]y 0"]
            )
        )
    count = draw(st.one_of(st.integers(0, 4).map(str), st.just("1000000000"), numbers))
    body = "\n".join(draw(st.lists(lines(depth + 1), min_size=1, max_size=3)))
    tag = draw(st.sampled_from(["", "", "[loop]", "[]"]))
    return f"REPEAT{tag} {count} {{\n{body}\n}}"


circuits = st.lists(lines(), min_size=1, max_size=10).map(lambda ls: "R 0 1 2 3\nM 0 1\n" + "\n".join(ls))


@st.composite
def dems(draw):
    out = []
    for _ in range(draw(st.integers(1, 8))):
        kind = draw(st.integers(0, 5))
        if kind == 0:
            out.append(f"detector{draw(st.sampled_from(['', '[d]']))}({draw(numbers)}) D{draw(st.integers(0, 6))}")
        elif kind == 1:
            out.append(f"logical_observable{draw(st.sampled_from(['', '[o]']))} L{draw(st.sampled_from(['0', '1', '63', '64']))}")
        elif kind == 2:
            out.append(f"shift_detectors {draw(numbers)}")
        elif kind == 3:
            out.append(f"repeat {draw(numbers)} {{\nerror[t](0.1) D0 D1\nshift_detectors[s] 2\n}}")
        else:
            pieces = [
                " ".join(draw(st.lists(st.one_of(st.integers(0, 7).map(lambda d: f"D{d}"), st.integers(0, 2).map(lambda o: f"L{o}")), min_size=1, max_size=3)))
                for _ in range(draw(st.integers(1, 3)))
            ]
            out.append(f"error({draw(probabilities)}) {' ^ '.join(pieces)}")
    return "\n".join(out)


def survives(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except ALLOWED:
        pass


@given(circuits)
def test_any_circuit_text(text):
    try:
        c = sq.Circuit(text)
    except ALLOWED:
        return
    assert sq.Circuit(str(c)) == c
    # Loops are unrolled for an error model: a million passes is legitimate work, and slow.
    counts = [int(m) for m in re.findall(r"REPEAT(?:\[[^\]]*\])?\s+(\d+)", text)]
    # So are a circuit's per-qubit arrays: qubit 16,777,215 costs 16.7 million of them.
    modest = math.prod(counts) <= 10_000 and c.num_qubits <= 4096  # nested loops multiply
    for decompose in (False, True) if modest else ():
        try:
            dem = c.detector_error_model(decompose_errors=decompose)
        except ALLOWED:
            continue
        assert sq.DetectorErrorModel(str(dem)).num_detectors <= max(dem.num_detectors, 1) + 0
        survives(c.detector_error_model, decompose_errors=decompose, approximate_disjoint_errors=True)
        dets = np.zeros((2, dem.num_detectors), dtype=bool)
        dets[1, ::3] = True
        for make in (lambda: sq.Matching(dem), lambda: sq.Matching(dem, enable_correlations=True), lambda: sq.BeliefMatching(dem, max_bp_iters=3), lambda: sq.UnionFind(dem)):
            try:
                decoder = make()
            except ALLOWED:
                continue
            survives(decoder.decode_batch, dets)
    # Sampling runs loops pass by pass: only modest loops are sampled here.
    if c.num_qubits <= 256 and c.num_measurements <= 2048 and modest:
        try:
            sampler = c.compile_detector_sampler(seed=1)
        except ALLOWED:  # a record read before the first measurement: refused here, not built
            return
        d = sampler.sample(70, threads=2)
        assert d.shape == (70, c.num_detectors)
        try:
            conv = c.compile_m2d_converter()
        except ALLOWED:
            return
        meas = np.zeros((3, c.num_measurements), dtype=bool)
        survives(conv.convert, measurements=meas, sweep_bits=np.zeros((3, c.num_sweep_bits), dtype=bool))


@given(st.lists(st.tuples(circuits, st.integers(0, 3), st.booleans()), min_size=1, max_size=4))
def test_any_circuit_built_from_pieces(pieces):
    # Built with +, * and append, a circuit counts and prints as its text parsed whole does.
    built = sq.Circuit()
    for text, n, in_place in pieces:
        try:
            piece = sq.Circuit(text)
        except ALLOWED:
            continue
        try:
            if in_place:
                built += piece * n
            else:
                built = built + n * piece
            built.append("TICK")
        except ALLOWED:
            continue
    again = sq.Circuit(str(built))
    assert again == built
    counts = lambda c: (c.num_qubits, c.num_measurements, c.num_detectors, c.num_observables, c.num_sweep_bits)
    assert counts(again) == counts(built)


@given(dems())
def test_any_model_text(text):
    try:
        dem = sq.DetectorErrorModel(text)
    except ALLOWED:
        return
    if dem.num_errors > 10_000 or dem.num_detectors > 100_000:  # large, legitimately: slow to decode
        return
    again = sq.DetectorErrorModel(str(dem))
    assert again.num_errors <= dem.num_errors and again.num_observables == dem.num_observables
    shots = np.zeros((2, dem.num_detectors), dtype=bool)
    shots[1, ::2] = True
    for make in (
        lambda: sq.Matching(dem),
        lambda: sq.BpOsd(dem, max_iter=5, osd_order=2),
        lambda: sq.WindowMatching(dem, commit=1, buffer=1),
        lambda: sq.UnionFind(dem),
        lambda: sq.BpLsd(dem, max_iter=5, lsd_method="lsd_cs", lsd_order=3),
        lambda: sq.RelayBp(dem, legs=3, pre_iterations=5, iterations=5),
        lambda: sq.ColorMatching(dem),
        lambda: sq.SearchDecoder(dem, pqlimit=1000, num_det_orders=1),
    ):
        try:
            decoder = make()
        except ALLOWED:
            continue
        survives(decoder.decode_batch, shots)


anything = st.one_of(
    st.integers(-(2**70), 2**70),
    st.floats(allow_nan=True, allow_infinity=True),
    st.booleans(),
    st.none(),
    st.text(max_size=5),
    st.sampled_from(["z", "x", "y", "rotated", "xzzx", "gross", "72", "f", "gh", "f+gh", "sliding", "parallel"]),
)
small = st.one_of(st.integers(-3, 6), st.sampled_from([0, 1, 2, 3, 5]), anything)


@given(small, small, anything, anything)
def test_memory_circuit_arguments(distance, rounds, p, basis):
    survives(sq.memory_circuit, distance=distance, rounds=rounds, p=p, basis=basis)


@given(small, small, anything, st.sampled_from(["zz", "xx", "cnot", "repeated", "line"]))
def test_surgery_arguments(distance, merged, p, kind):
    f = {
        "zz": lambda: sq.surgery.zz_measurement(distance, merged=merged, p=p),
        "xx": lambda: sq.surgery.xx_measurement(distance, merged=merged, p=p),
        "cnot": lambda: sq.surgery.cnot(distance, merged=merged, p=p),
        "repeated": lambda: sq.surgery.repeated_zz(distance, k=merged, merged=merged, p=p),
        "line": lambda: sq.surgery.line(distance, n=merged, merged=merged, p=p),
    }[kind]
    survives(f)


@given(anything, small, anything, anything)
def test_bivariate_bicycle_arguments(name, cycles, p, operator):
    try:
        code = sq.BivariateBicycleCode(name)
    except ALLOWED:
        return
    survives(code.memory_circuit, cycles, p)
    survives(code.gauging, operator)
    survives(code.logical_measurement_circuit, operator, "x", pre=cycles, merged=cycles, post=cycles, p=p)


@given(st.integers(-2, 300), st.integers(-2, 9), anything, st.booleans(), st.booleans())
def test_sampler_arguments(shots, threads, seed, packed, separate):
    c = sq.memory_circuit(distance=3, rounds=2, p=0.01)
    try:
        s = c.compile_detector_sampler(seed=seed)
    except ALLOWED:
        return
    survives(s.sample, shots, threads=threads, bit_packed=packed, separate_observables=separate)


@given(
    st.integers(0, 3),
    st.integers(0, 30),
    st.sampled_from([np.bool_, np.uint8, np.int64, np.float64]),
    st.booleans(),
    st.integers(-1, 4),
)
def test_decoder_inputs(rows, cols, dtype, packed, threads):
    dem = sq.memory_circuit(distance=3, rounds=3, p=0.01).detector_error_model(decompose_errors=True)
    shots = np.ones((rows, cols), dtype=dtype)
    for decoder in (
        sq.Matching(dem),
        sq.BeliefMatching(dem),
        sq.BpOsd(dem, max_iter=5, osd_order=2),
        sq.BpLsd(dem, max_iter=5),
        sq.RelayBp(dem, legs=2),
        sq.SearchDecoder(dem, num_det_orders=1),
    ):
        survives(decoder.decode_batch, shots, bit_packed_shots=packed, threads=threads)
    survives(sq.Matching(dem).decode, np.ones(cols, dtype=dtype))


@given(st.integers(0, 6), st.integers(0, 6), anything, st.sampled_from(["product_sum", "minimum_sum", "other"]))
def test_check_matrix_decoders(m, n, rate, method):
    pcm = np.ones((m, n), dtype=np.uint8)
    for make in (
        lambda: sq.BpDecoder(pcm, error_rate=rate, bp_method=method),
        lambda: sq.BpOsdDecoder(pcm, error_rate=rate, bp_method=method),
        lambda: sq.BpLsdDecoder(pcm, error_rate=rate, bp_method=method, lsd_method="lsd_e", lsd_order=2),
        lambda: sq.RelayBpDecoder(pcm, error_rate=rate, legs=2),
    ):
        try:
            dec = make()
        except ALLOWED:
            continue
        survives(dec.decode, np.ones(m, dtype=np.uint8))
        survives(dec.decode, np.ones(m + 1, dtype=np.uint8))


def test_nan_and_inf_are_refused():
    for p in (math.nan, math.inf, -0.5):
        with pytest.raises(ValueError):
            sq.memory_circuit(distance=3, rounds=3, p=p)
