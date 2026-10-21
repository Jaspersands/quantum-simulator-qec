"""Tsim's ``Circuit`` and samplers (bloqade-tsim 0.1.5) on this package's engines: the class
(``isinstance``, results that are ``tsim.Circuit``s, ``from_file``), its shorthand ``str``,
``append`` of Tsim's gates, addition, the detector sampler's values and its reference keywords,
``CompiledStateProbs``, the model's reading of T as S, ``to_matrix``'s state columns and dropped
qubits, and the diagram. The expected values are Tsim's; the comparisons with Tsim itself are
skipped where it is not installed."""

from __future__ import annotations

import pickle
import re

import numpy as np
import pytest

import stabilizer_qec as sq
from stabilizer_qec import tsim
from stabilizer_qec.tsim import CompiledDetectorSampler, CompiledMeasurementSampler, CompiledStateProbs, stim_to_shorthand


@pytest.fixture
def real():
    return pytest.importorskip("tsim")


def equal_up_to_phase(u: np.ndarray, v: np.ndarray) -> bool:
    if u.shape != v.shape:
        return False
    i = np.argmax(np.abs(v))
    if abs(v.flat[i]) < 1e-12:
        return np.allclose(u, 0, atol=1e-9)
    phase = u.flat[i] / v.flat[i]
    return abs(abs(phase) - 1) < 1e-6 and np.allclose(u, phase * v, atol=1e-6)


# Tsim's str() of each (its shorthand, its floats canonical).
SHORTHAND = {
    "T 0 1": "T 0 1",
    "T_DAG 0": "T_DAG 0",
    "R_X(1) 0": "R_X(1.0) 0",
    "R_ZZ(0.3) 0 1": "R_ZZ(0.3) 0 1",
    "R_PAULI(0.3) X0*Y1": "R_PAULI(0.3) X0*Y1",
    "R_PAULI(0.3) X0*Z2*Y4": "R_PAULI(0.3) X0*Z2*Y4",
    "TPP X0*Y1": "TPP X0*Y1",
    "TPP_DAG Z0": "TPP_DAG Z0",
    "U3(0.1, 0.2, 0.3) 0": "U3(0.1, 0.2, 0.3) 0",
    "R_X(0.1) 0 1": "R_X(0.1) 0 1",
    "R_Y(-0.25) 3": "R_Y(-0.25) 3",
    "R_Z(1e-05) 0": "R_Z(1e-05) 0",
    "X_ERROR(0.1234567) 0": "X_ERROR(0.123457) 0",
    "CCZ 0 1 2": "CX 1 2\nT_DAG 2\nCX 0 2\nT 2\nCX 1 2\nT_DAG 2\nCX 0 2\nT 1 2\nCX 0 1\nT 0\nT_DAG 1\nCX 0 1",
    "REPEAT 2 {\n    T 0\n    R_X(0.3) 1\n}": "REPEAT 2 {\n    T 0\n    R_X(0.3) 1\n}",
    "S[T] 0\nI[R_X(theta=0.25*pi)] 1": "T 0\nR_X(0.25) 1",
    "R 0\nH 0\nT 0\nH 0\nM 0\nDETECTOR rec[-1]": "R 0\nH 0\nT 0\nH 0\nM 0\nDETECTOR rec[-1]",
}


def test_circuit_is_a_class_whose_results_are_tsim_circuits():
    c = tsim.Circuit("R 0\nH 0\nT 0\nH 0\nM 0")
    assert isinstance(c, tsim.Circuit) and isinstance(c, sq.Circuit)
    assert str(c) == "R 0\nH 0\nT 0\nH 0\nM 0"
    assert repr(c) == "tsim.Circuit('''\nR 0\nH 0\nT 0\nH 0\nM 0\n''')"
    t = tsim.Circuit("T 0\nR_X(0.1) 0")
    for out in (c[1:3], c.flattened(), c.copy(), c * 2, c.without_noise(), c + c, t.inverse(), pickle.loads(pickle.dumps(c))):
        assert type(out) is tsim.Circuit
    assert str(c[1:3]) == "H 0\nT 0" and str(t.inverse()) == "R_X(-0.1) 0\nT_DAG 0"
    assert pickle.loads(pickle.dumps(t)) == t
    assert str(tsim.Circuit("T 0\nM 0\nDETECTOR rec[-1]").without_annotations()) == "T 0\nM 0"
    # stim_circuit is a plain circuit, in this package's form (as Tsim's is a stim.Circuit).
    assert type(c.stim_circuit) is sq.Circuit and str(c.stim_circuit) == "R 0\nH 0\nI[T] 0\nH 0\nM 0"
    assert str(sq.Circuit(c)) == "R 0\nH 0\nI[T] 0\nH 0\nM 0"
    with pytest.raises(TypeError):
        tsim.Circuit(3)


@pytest.mark.parametrize("text", sorted(SHORTHAND))
def test_str_is_tsims_shorthand(text):
    assert str(tsim.Circuit(text)) == SHORTHAND[text]
    if "ERROR" not in text:  # a probability prints to 6 digits, as Stim's; an angle exactly
        assert tsim.Circuit(str(tsim.Circuit(text))) == tsim.Circuit(text)


@pytest.mark.parametrize("text", sorted(SHORTHAND))
def test_str_and_repr_match_tsim(real, text):
    assert str(tsim.Circuit(text)) == str(real.Circuit(text))
    assert repr(tsim.Circuit(text)) == repr(real.Circuit(text))


@pytest.mark.parametrize("text", sorted(SHORTHAND))
def test_stim_to_shorthand_reads_the_native_form(text):
    native = str(sq.Circuit(tsim.Circuit(text)))
    assert stim_to_shorthand(native) == SHORTHAND[text]


def test_stim_to_shorthand_of_native_pairs_and_products():
    assert stim_to_shorthand("II[R_ZZ(theta=0.47123889803846897)] 0 1 2 3") == "R_ZZ(0.3) 0 1\nR_ZZ(0.3) 2 3"
    assert stim_to_shorthand("I[T] 0 1\nI[T_DAG] 2") == "T 0 1\nT_DAG 2"
    # Two rotations of products on separate lines stay separate (they are not fused).
    c = tsim.Circuit("R_ZZ(0.5) 0 1\nR_ZZ(0.5) 2 3\nTPP X0*Y1\nTPP X2*Y3\nR_PAULI(0.3) X0\nR_PAULI(0.3) Z1")
    assert str(c) == "R_ZZ(0.5) 0 1\nR_ZZ(0.5) 2 3\nTPP X0*Y1\nTPP X2*Y3\nR_PAULI(0.3) X0\nR_PAULI(0.3) Z1"
    assert c.num_qubits == 4


def test_from_file_reads_shorthand(tmp_path):
    path = tmp_path / "c.stim"
    path.write_text("R 0\nT 0\nR_X(0.25) 0\nM 0\n")
    with open(path) as f:
        read = [tsim.Circuit.from_file(path), tsim.Circuit.from_file(filename=str(path)), tsim.Circuit.from_file(f)]
    for c in read:
        assert type(c) is tsim.Circuit and str(c) == "R 0\nT 0\nR_X(0.25) 0\nM 0"
    out = tmp_path / "out.stim"
    read[0].to_file(out)
    assert out.read_text() == "R 0\nI[T] 0\nI[R_X(theta=0.39269908169872414)] 0\nM 0\n"
    assert tsim.Circuit.from_file(out) == read[0] and sq.Circuit.from_file(out) == read[0]
    assert tsim.Circuit.from_stim_program(sq.Circuit("I[T] 0")) == tsim.Circuit("T 0")


# Tsim's append: its gates, then what str() gives.
APPENDS = [
    (("T", 0), "T 0"),
    (("T_DAG", [1]), "T_DAG 1"),
    (("R_X", 0, 0.1), "R_X(0.1) 0"),
    (("R_Y", [0, 1], [0.2]), "R_Y(0.2) 0 1"),
    (("R_Z", 2, 0.5), "R_Z(0.5) 2"),
    (("U3", 0, [0.1, 0.2, 0.3]), "U3(0.1, 0.2, 0.3) 0"),
    (("R_ZZ", [0, 2], 0.5), "R_ZZ(0.5) 0 2"),
    (("R_XX", [0, 1], 0.3), "R_XX(0.3) 0 1"),
    (("CCZ", [0, 1, 2]), SHORTHAND["CCZ 0 1 2"]),
    (("H", 0), "H 0"),
    (("S", 0, None), "S 0"),
    (("S", 0, None, "T"), "T 0"),
    (("I", 1, None, "R_Z(theta=0.25*pi)"), "R_Z(0.25) 1"),
]


def test_append_writes_tsims_gates():
    c = tsim.Circuit()
    for args, want in APPENDS:
        one = tsim.Circuit()
        one.append(*args[:3], **({"tag": args[3]} if len(args) > 3 else {}))
        assert str(one) == want, args
        c.append(*args[:3], **({"tag": args[3]} if len(args) > 3 else {}))
    assert str(c) == "\n".join(w for _, w in APPENDS)
    c = tsim.Circuit()
    c.append_from_stim_program_text("T 0\nR_X(0.1) 0")
    assert str(c) == "T 0\nR_X(0.1) 0"
    with pytest.raises(ValueError, match=r"^For R_ZZ gates, the two target qubits must be distinct, got 0 0\.$"):
        tsim.Circuit().append("R_ZZ", [0, 0], 0.5)
    with pytest.raises(ValueError, match=r"^For U3 gates, three rotation angles must be provided\.$"):
        tsim.Circuit().append("U3", [0], 0.5)
    with pytest.raises(ValueError, match=r"Could not parse 'R_X' in program text\."):
        tsim.Circuit("R_X 0")


def test_append_matches_tsim(real):
    stim = pytest.importorskip("stim")
    pauli = [
        ("TPP", [stim.target_x(0), stim.target_combiner(), stim.target_y(1)], None),
        ("TPP_DAG", [stim.target_z(0)], None),
        ("R_PAULI", [stim.target_x(0), stim.target_combiner(), stim.target_z(1)], 0.3),
    ]
    ours, theirs = tsim.Circuit(), real.Circuit()
    for args, _ in APPENDS:
        for c in (ours, theirs):
            c.append(*args[:3], **({"tag": args[3]} if len(args) > 3 else {}))
    for args in pauli:
        for c in (ours, theirs):
            c.append(*args)
    assert str(ours) == str(theirs)


def test_addition_with_stim_and_this_packages_circuits():
    f = tsim.Circuit("T 0")
    f += sq.Circuit("I[T] 1\nH 0")
    assert type(f) is tsim.Circuit and str(f) == "T 0 1\nH 0"
    g = f + sq.Circuit("I[T] 2")
    assert type(g) is tsim.Circuit and str(g) == "T 0 1\nH 0\nT 2"
    h = sq.Circuit("H 0") + tsim.Circuit("T 0")
    assert type(h) is tsim.Circuit and str(h) == "H 0\nT 0"
    stim = pytest.importorskip("stim")
    f = tsim.Circuit("T 0")
    f += stim.Circuit("S[T] 1\nI[R_X(theta=0.25*pi)] 0")
    assert type(f) is tsim.Circuit and str(f) == "T 0 1\nR_X(0.25) 0"
    g = tsim.Circuit("T 0") + stim.Circuit("H 0")
    assert type(g) is tsim.Circuit and str(g) == "T 0\nH 0"


# Detector values: Tsim reports each detector's and observable's parity, not its flip.
DETERMINISTIC = {
    "X 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]": [True, True],
    "R 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]": [False, False],
    "X 0\nT 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]": [True, True],
    "MPAD 1\nDETECTOR rec[-1]": [True],
    "T 0\nM !0\nDETECTOR rec[-1]": [True],
    "X 0 1\nCX 0 1\nT 1\nM 0 1\nDETECTOR rec[-1] rec[-2]\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-2]": [True, False, True],
}


@pytest.mark.parametrize("text", sorted(DETERMINISTIC))
def test_detector_sampler_reports_values(text):
    want = np.array(DETERMINISTIC[text])
    c = tsim.Circuit(text)
    out = c.compile_detector_sampler(seed=1).sample(20, append_observables=True)
    assert out.dtype == np.bool_ and np.array_equal(out, np.tile(want, (20, 1)))
    # The reference keywords give flips: none here.
    flips = c.compile_detector_sampler(seed=1).sample(20, append_observables=True, use_detector_reference_sample=True, use_observable_reference_sample=True)
    assert not flips.any()
    d, o = c.compile_detector_sampler(seed=1).sample(5, separate_observables=True, use_detector_reference_sample=True)
    assert not d.any() and np.array_equal(o, np.tile(want[c.num_detectors :], (5, 1)))
    # This package's circuits keep Stim's flips.
    assert not sq.Circuit(c).compile_detector_sampler(seed=1).sample(20, append_observables=True).any()
    assert not c.compile_detector_sampler(seed=1, device="cpu").sample(20, append_observables=True).any()


@pytest.mark.parametrize("text", sorted(DETERMINISTIC))
def test_detector_values_match_tsim(real, text):
    for kw in ({}, dict(use_detector_reference_sample=True), dict(use_observable_reference_sample=True), dict(use_detector_reference_sample=True, use_observable_reference_sample=True)):
        ours = tsim.Circuit(text).compile_detector_sampler(seed=0).sample(10, append_observables=True, **kw)
        theirs = np.asarray(real.Circuit(text).compile_detector_sampler(seed=0).sample(10, append_observables=True, **kw))
        assert np.array_equal(ours, theirs), kw


def test_random_detector_values_match_tsim_in_distribution(real):
    text = "R 0 1\nH 0\nT 0\nCX 0 1\nDEPOLARIZE1(0.05) 0 1\nT_DAG 1\nX 2\nCX 1 2\nX_ERROR(0.1) 2\nH 0\nM 0 1 2\nDETECTOR rec[-1] rec[-2]\nDETECTOR rec[-2] rec[-3]\nOBSERVABLE_INCLUDE(0) rec[-1]"
    shots = 20_000
    a = tsim.Circuit(text).compile_detector_sampler(seed=0).sample(shots, append_observables=True).mean(0)
    b = np.asarray(real.Circuit(text).compile_detector_sampler(seed=0).sample(shots, batch_size=shots, append_observables=True)).mean(0)
    z = np.abs(a - b) / np.sqrt(a * (1 - a) / shots + b * (1 - b) / shots + 1e-12)
    assert np.all(z < 5), (a, b)


def test_detector_sampler_keywords():
    s = tsim.Circuit("X 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]").compile_detector_sampler(seed=0, strategy="bss")
    assert isinstance(s, CompiledDetectorSampler) and s.num_detectors == 1 and s.num_observables == 1
    assert s.sample(3, append_observables=True, bit_packed=True).tolist() == [[3], [3], [3]]
    assert s.sample(2, append_observables=True, prepend_observables=True).tolist() == [[True] * 3] * 2
    assert s.sample(4, batch_size=1, postselection_mask=np.array([True]), append_observables=True).tolist() == [[True, True]] * 4
    with pytest.raises(ValueError, match=r"^batch_size must be at least 1, got 0$"):
        s.sample(4, batch_size=0)
    with pytest.raises(ValueError, match=r"^shots must be non-negative, got -1$"):
        s.sample(-1)
    with pytest.raises(ValueError, match=r"^postselection_mask must have shape \(1,\), got \(2,\)$"):
        s.sample(4, postselection_mask=np.array([True, False]))
    with pytest.raises(ValueError, match=r"^Can't specify separate_observables=True with append_observables=True or prepend_observables=True$"):
        s.sample(4, separate_observables=True, append_observables=True)
    m = tsim.Circuit("R 0 1\nX 0\nM 0 1").compile_sampler(seed=0, strategy="cutting")
    assert isinstance(m, CompiledMeasurementSampler)
    assert m.sample(5, batch_size=2).tolist() == [[True, False]] * 5
    with pytest.raises(ValueError, match=r"^batch_size must be at least 1, got 0$"):
        m.sample(2, batch_size=0)
    # Tsim's strategy is not checked (its ZX decomposition reads anything else as "cat5").
    assert tsim.Circuit("H 0\nT 0\nM 0").compile_sampler(strategy="other", seed=1).sample(4).shape == (4, 1)


def test_state_probs_per_noise_draw():
    noiseless = CompiledStateProbs(tsim.Circuit("R 0\nH 0\nT 0\nH 0\nM 0"), seed=1)
    p = noiseless.probability_of(np.array([True]), batch_size=3)
    assert p.shape == (3,) and np.allclose(p, (1 - np.cos(np.pi / 4)) / 2)
    noisy = CompiledStateProbs(tsim.Circuit("R 0\nH 0\nT 0\nH 0\nX_ERROR(0.1) 0\nM 0"), seed=1)
    p = noisy.probability_of(np.array([True]), batch_size=400)
    # Each draw is with or without the flip: P(1 | no flip) or P(1 | flip).
    lo, hi = (1 - np.cos(np.pi / 4)) / 2, (1 + np.cos(np.pi / 4)) / 2
    assert np.all(np.isclose(p, lo) | np.isclose(p, hi)) and 10 < np.isclose(p, hi).sum() < 80
    dets = CompiledStateProbs(tsim.Circuit("R 0 1\nH 0\nT 0\nH 0\nX_ERROR(0.1) 0\nM 0 1\nDETECTOR rec[-1] rec[-2]"), sample_detectors=True, seed=1)
    p = dets.probability_of(np.array([True]), batch_size=400)
    assert np.all(np.isclose(p, lo) | np.isclose(p, hi))
    # A seed's draws are the same whatever the state: the outcomes' probabilities sum to 1 in each.
    text = "R 0 1\nH 0\nT 0\nCX 0 1\nDEPOLARIZE1(0.2) 0 1\nH 1\nM 0 1"
    states = [np.array([a, b]) for a in (False, True) for b in (False, True)]
    total = sum(CompiledStateProbs(tsim.Circuit(text), seed=7).probability_of(s, batch_size=50) for s in states)
    assert np.allclose(total, 1)
    with pytest.raises(ValueError, match=r"^state must have shape \(1,\), got \(2,\)$"):
        noiseless.probability_of(np.array([True, False]), batch_size=2)
    with pytest.raises(ValueError, match=r"^batch_size must be at least 1, got 0$"):
        noiseless.probability_of(np.array([True]), batch_size=0)
    # A detector state no record gives has probability 0.
    never = CompiledStateProbs(tsim.Circuit("R 0\nM 0\nDETECTOR rec[-1]\nDETECTOR rec[-1]"), sample_detectors=True)
    assert never.probability_of(np.array([True, False]), batch_size=2).tolist() == [0.0, 0.0]


def test_state_probs_match_tsim(real):
    from tsim.sampler import CompiledStateProbs as TheirStateProbs

    for text, sample_detectors, state in (
        ("R 0\nH 0\nT 0\nH 0\nM 0", False, [True]),
        ("R 0 1\nH 0\nT 0\nCX 0 1\nT_DAG 1\nH 0 1\nM 0 1", False, [True, False]),
        ("R 0 1\nH 0\nT 0\nH 0\nM 0 1\nDETECTOR rec[-1] rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-2]", True, [True, True]),
        ("R 0 1\nH 0\nT 0\nH 0\nM 0 1\nDETECTOR rec[-1] rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-2]", True, [True, False]),
    ):
        ours = CompiledStateProbs(tsim.Circuit(text), sample_detectors=sample_detectors, seed=0).probability_of(np.array(state), batch_size=4)
        theirs = np.asarray(TheirStateProbs(real.Circuit(text), sample_detectors=sample_detectors, seed=0).probability_of(np.array(state), batch_size=4))
        assert ours.shape == theirs.shape and np.allclose(ours, theirs, atol=1e-6), (text, state)
    # With noise, the draws differ (Tsim's are JAX's); each is one of the same few values.
    for text, sample_detectors, state in (
        ("R 0\nH 0\nT 0\nH 0\nX_ERROR(0.1) 0\nM 0", False, [True]),
        ("R 0 1\nH 0\nT 0\nH 0\nX_ERROR(0.1) 0\nM 0 1\nDETECTOR rec[-1] rec[-2]", True, [True]),
    ):
        ours = CompiledStateProbs(tsim.Circuit(text), sample_detectors=sample_detectors, seed=0).probability_of(np.array(state), batch_size=200)
        theirs = np.asarray(TheirStateProbs(real.Circuit(text), sample_detectors=sample_detectors, seed=0).probability_of(np.array(state), batch_size=200))
        assert set(np.round(ours, 5)) == set(np.round(theirs.astype(np.float64), 5)), text


# Tsim's model: Stim's analysis of its tagged form (T as S, a rotation as I, a product rotation as SPP).
DEM_CASES = {
    "R 0\nH 0\nT 0\nH 0\nX_ERROR(0.01) 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]": "error(0.5) D0 L0",
    "RX 0 1\nT 0\nCX 0 1\nX_ERROR(0.1) 0 1\nM 0 1\nDETECTOR rec[-1] rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-1]": None,
    "R 0 1\nR_ZZ(0.3) 0 1\nX_ERROR(0.1) 0\nH 0\nM 0 1\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-2]": None,
    "R 0 1\nH 0\nTPP X0*X1\nDEPOLARIZE1(0.1) 0 1\nM 0 1\nDETECTOR rec[-1] rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-1]": None,
    "R 0\nR_X(0.5) 0\nX_ERROR(0.1) 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]": None,
    "R 0 1 2\nCCZ 0 1 2\nH 0\nDEPOLARIZE1(0.01) 0 1 2\nM 0 1 2\nDETECTOR rec[-1]\nDETECTOR rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-3]": None,
    "R 0\nU3(0.3, 0.2, 0.1) 0\nX_ERROR(0.1) 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]\nR 1\nX_ERROR(0.1) 1\nM 1\nDETECTOR rec[-1]": None,
}


def test_dem_reads_t_as_s():
    c = tsim.Circuit("R 0\nH 0\nT 0\nH 0\nX_ERROR(0.01) 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]")
    assert str(c.detector_error_model()) == "error(0.5) D0 L0"
    assert str(tsim.get_detector_error_model(c)) == "error(0.5) D0 L0"
    with pytest.raises(ValueError, match="Decomposition of error mechanisms is not supported"):
        c.detector_error_model(decompose_errors=True)
    # This package's circuit has no such random bit in its model.
    assert "error(0.5)" not in str(sq.Circuit(c).detector_error_model(allow_non_deterministic_observables=True))


def dem_outcome(circuit_type, text: str, **kw) -> str:
    """The model's text, or the error it raises (its type and first line)."""
    try:
        return str(circuit_type(text).detector_error_model(**kw))
    except ValueError as exc:
        return f"ValueError: {str(exc).splitlines()[0]}"


@pytest.mark.parametrize("text", sorted(DEM_CASES))
def test_dem_matches_tsim(real, text):
    for kw in ({}, dict(allow_gauge_detectors=True), dict(flatten_loops=True)):
        assert dem_outcome(tsim.Circuit, text, **kw) == dem_outcome(real.Circuit, text, **kw), kw


def test_encoder_dem_matches_tsim(real):
    from tsim.utils import encoder as their_encoders

    for name in ("SteaneEncoder", "ColorEncoder5"):
        theirs, ours = getattr(their_encoders, name)(), getattr(tsim, name)()
        for e in (theirs, ours):
            e.initialize("R 0 1\nT 0\nH 1")
            e.encode_transversally("CX 0 1\nDEPOLARIZE1(0.001) 0 1\nM 0 1\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-2]")
        assert type(ours.circuit) is tsim.Circuit and str(ours.circuit) == str(theirs.circuit)
        assert str(ours.circuit.detector_error_model()) == str(theirs.circuit.detector_error_model())


def test_to_matrix_states_and_dropped_qubits():
    h = np.sqrt(0.5)
    assert np.allclose(tsim.Circuit("R 0 1\nH 0\nCX 0 1").to_matrix(), [[h], [0], [0], [h]])
    assert np.allclose(tsim.Circuit("RY 0").to_matrix(), [[h], [1j * h]])
    rz = tsim.Circuit("RX 0\nR_Z(0.25) 0").to_matrix()
    assert np.allclose(rz, [[h * np.exp(-1j * np.pi / 8)], [h * np.exp(1j * np.pi / 8)]])
    for text, shape in (("R_ZZ(0.5) 0 2", (4, 4)), ("R 0\nH 1", (4, 2)), ("H 1\nR 0", (4, 2)), ("R 0\nCX 0 1", (4, 2)), ("", (1, 1)), ("TICK", (1, 1)), ("I 3", (2, 2)), ("H 0\nI 3", (4, 4)), ("H 2", (2, 2)), ("R 5\nH 5", (2, 1)), ("X_ERROR(0.1) 0", (2, 2)), ("R 0\nX_ERROR(0.1) 0", (2, 1)), ("QUBIT_COORDS(1, 2) 0\nH 0", (2, 2)), ("R 0\nTICK\nH 0\nSHIFT_COORDS(1)", (2, 1))):
        assert tsim.Circuit(text).to_matrix().shape == shape, text
    assert tsim.Circuit("R " + " ".join(map(str, range(15)))).to_matrix().shape == (1 << 15, 1)
    assert tsim.Circuit("R 0 1\nX 1").to_matrix(endian="little")[2, 0] == 1
    assert tsim.Circuit("R 0 1\nX 1").to_tensor().shape == (2, 2)
    # A measurement, a reset after a qubit's first operation or feedback is Tsim's ZX tensor of the
    # circuit (tests/test_tsim_zx.py), its values here as Tsim gives them.
    g = np.sqrt(0.5)
    assert np.allclose(tsim.Circuit("H 0\nR 0").to_matrix(), [[1, 0], [0, 0]])
    assert np.allclose(tsim.Circuit("R 0\nM 0").to_matrix(), [[g], [0]])
    assert np.allclose(tsim.Circuit("M 0\nCX rec[-1] 1").to_matrix(), g * np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]))
    assert np.allclose(tsim.Circuit("MR 0").to_matrix(), [[0.5, 0.5], [0, 0]])
    assert np.allclose(tsim.Circuit("R 0\nR 0").to_matrix(), [[g], [0]])
    with pytest.raises(ValueError, match="at most"):
        tsim.Circuit("R " + " ".join(map(str, range(23)))).to_matrix()
    # This package's circuits keep their unitary's rules.
    assert sq.Circuit("H 2").to_matrix().shape == (8, 8)


def test_to_matrix_matches_tsim(real):
    for text in ("RX 0\nR_Z(0.25) 0", "R 0 1\nH 0", "R 0\nH 1", "H 1\nR 0", "R_ZZ(0.5) 0 2", "H 0\nH 2", "H 2", "R 2\nH 2", "", "R 0\nCX 0 1", "RY 0", "R 0\nX_ERROR(0.1) 0", "I 3", "H 0\nI 3", "R 1\nH 0", "REPEAT 2 {\nH 0\n}", "H 0\nSPP X0*Z2", "R 0 1 2\nH 0\nT 0\nCX 0 1\nR_X(0.3) 2\nTPP X1*Y2", "X_ERROR(0.1) 0", "E(0.1) X0", "DEPOLARIZE2(0.1) 0 2\nH 0", "R 0\nDEPOLARIZE1(0.1) 0\nH 0"):
        ours, theirs = tsim.Circuit(text).to_matrix(), np.asarray(real.Circuit(text).to_matrix())
        assert equal_up_to_phase(ours, theirs), text


def test_diagram_labels_tagged_gates_and_keeps_stims_drawing():
    plain = "R 0 1\nH 0\nCX 0 1\nM 0 1\nDETECTOR rec[-1]"
    for kind in ("timeline-text", "timeline-svg"):
        assert str(tsim.Circuit(plain).diagram(kind)) == str(sq.Circuit(plain).diagram(kind))
    text = str(tsim.Circuit("T 0\nR_X(0.25) 1\nR_ZZ(0.1) 0 1").diagram("timeline-text"))
    assert "T" in text and "R_X" in text and "R_ZZ" in text and "SPP" not in text
    svg = str(tsim.Circuit("T 0\nT_DAG 1\nU3(0.1, 0.2, 0.3) 2").diagram("timeline-svg"))
    assert svg.startswith("<svg") and ">T<" in svg
    # (The ZX-graph diagram types are tests/test_tsim_zx.py's.)


def test_diagram_html_sizes():
    d = tsim.Circuit("H 0\nH 1").diagram("timeline-svg")
    html = d._repr_html_()
    assert str(d).startswith("<svg") and "data-tsim-zoom" in html and "height:110px" in html
    assert "height:100px" in tsim.Circuit("H 0\nH 1").diagram("timeline-svg", height=100)._repr_html_()
    fixed = tsim.Circuit("H 0\nH 1").diagram("timeline-svg", width=300, zoomable=False)._repr_html_()
    assert '<div style="width: 300px">' in fixed and "data-tsim-zoom" not in fixed


def test_diagram_html_is_tsims_wrapper(real):
    # The SVG is this package's (not Stim's, nor so Tsim's); around it, the notebook view is
    # Tsim's, sized as Tsim sizes it: Tsim's own wrappers of the same SVG.
    from tsim.utils import diagram as theirs

    unzoom = lambda html: re.sub(r'data-tsim-zoom="[0-9a-f]{12}"|tsim_[0-9a-f]{12}|[0-9a-f]{12}', "ID", html)  # noqa: E731
    for text in ("H 0\nH 1", "R 0 1 2\nH 0\nT 0\nCX 0 1\nCX 1 2\nR_X(0.3) 2\nM 0 1 2"):
        for kind in ("timeline-svg", "timeslice-svg"):
            for kw in ({}, dict(height=100), dict(width=300, zoomable=False), dict(width=250), dict(height=80, zoomable=False)):
                d = tsim.Circuit(text).diagram(kind, **kw)
                svg = str(d)
                width, height = kw.get("width"), kw.get("height")
                if width is None and height is None and kind == "timeline-svg":
                    height = min(30 * real.Circuit(text).num_qubits + 50, 700)
                if width is None and height is not None:
                    width = theirs._width_from_viewbox(svg, height)
                elif height is None and width is not None:
                    w, h = theirs._viewbox_size(svg)
                    height = width / w * h
                if kw.get("zoomable", True):
                    want = theirs.wrap_svg_zoomable(svg, width=width, height=height if height is not None else 700)
                else:
                    want = theirs.wrap_svg(svg, width=width, height=height)
                assert unzoom(d._repr_html_()) == unzoom(want), (text, kind, kw)
