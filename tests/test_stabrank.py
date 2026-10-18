"""The stabilizer-rank sampler (Clifford+T at any qubit count) against the state vector, bit
for bit on small circuits, and its surface."""

from __future__ import annotations

import numpy as np
import pytest

import stabilizer_qec as sq

GATES_1 = ["H", "S", "S_DAG", "X", "Y", "Z", "SQRT_X", "SQRT_Y_DAG", "H_YZ", "C_XYZ"]
GATES_2 = ["CX", "CZ", "CY", "SWAP", "ISWAP", "SQRT_XX", "XCZ"]
MEAS = ["M", "MX", "MR", "MRX", "M(0.1)", "MY"]
NOISE = ["DEPOLARIZE1(0.2)", "X_ERROR(0.15)", "Y_ERROR(0.1)", "PAULI_CHANNEL_1(0.1, 0.05, 0.02)", "HERALDED_ERASE(0.1)"]


def random_case(rng, n_max: int = 3, length=(6, 14)) -> str:
    """A random Clifford+T circuit with noise, measurements in every basis, resets and feedback."""
    n = int(rng.integers(1, n_max + 1))
    lines, nm = ["R " + " ".join(map(str, range(n)))], 0
    for _ in range(int(rng.integers(*length))):
        u = rng.random()
        q = int(rng.integers(n))
        if u < 0.28:
            lines.append(f"{rng.choice(GATES_1)} {q}")
        elif u < 0.45 and n > 1:
            a, b = rng.choice(n, 2, replace=False)
            lines.append(f"{rng.choice(GATES_2)} {a} {b}")
        elif u < 0.65:
            kind = rng.integers(6)
            if kind == 0:
                lines.append(f"I[T] {q}")
            elif kind == 1:
                lines.append(f"I[T_DAG] {q}")
            elif kind == 2:
                lines.append(f"I[R_{rng.choice(list('XYZ'))}(theta={rng.uniform(-1.5, 1.5)!r})] {q}")
            elif kind == 3:
                lines.append(f"I[U3(theta={rng.uniform(0, 6)!r}, phi={rng.uniform(0, 6)!r}, lambda={rng.uniform(0, 6)!r})] {q}")
            elif kind == 4 and n > 1:
                a, b = rng.choice(n, 2, replace=False)
                lines.append(f"II[R_{rng.choice(['XX', 'YY', 'ZZ'])}(theta={rng.uniform(-1.5, 1.5)!r})] {a} {b}")
            else:
                lines.append(f"I[R_PAULI(theta={rng.uniform(-1.5, 1.5)!r}, pauli={''.join(rng.choice(list('XYZ'), n))})] {' '.join(map(str, range(n)))}")
        elif u < 0.76:
            noise = rng.choice(NOISE)
            if noise.startswith("HERALDED"):
                nm += 1
            lines.append(f"{noise} {q}")
        elif u < 0.9:
            lines.append(f"{rng.choice(MEAS)} {q}")
            nm += 1
        elif u < 0.95:
            lines.append(f"{rng.choice(['R', 'RX'])} {q}")
        elif nm > 0:
            lines.append(f"{rng.choice(['CX', 'CZ', 'CY'])} rec[-{int(rng.integers(1, min(nm, 3) + 1))}] {q}")
        else:
            lines.append(f"MPAD 1")
            nm += 1
    lines.append(f"M {n - 1}")
    nm += 1
    for k in range(1, min(nm, 3) + 1):
        lines.append(f"DETECTOR rec[-{k}]")
    lines.append(f"OBSERVABLE_INCLUDE(0) rec[-1] rec[-{min(nm, 2)}]")
    return "\n".join(lines)


def same(a: dict, b: dict, text: str, tol: float = 1e-10) -> None:
    for key in set(a) | set(b):
        assert abs(a.get(key, 0) - b.get(key, 0)) < tol, (text, key, a.get(key), b.get(key))


@pytest.mark.parametrize("seed", range(80))
def test_exact_distribution_equals_the_state_vector(seed):
    text = random_case(np.random.default_rng(seed))
    c = sq.Circuit(text)
    same(c.exact_distribution(method="stabilizer_rank"), c.exact_distribution(), text)


@pytest.mark.parametrize("seed", range(5))
def test_samples_follow_the_exact_distribution(seed):
    text = random_case(np.random.default_rng(300 + seed), n_max=3, length=(8, 14))
    c = sq.Circuit(text)
    exact = c.exact_distribution()
    shots = 4000
    s = c.compile_stabilizer_rank_sampler(seed=seed)
    d, _ = s.sample(shots, separate_observables=True, threads=0)
    rows = [tuple(map(bool, r)) for r in d]
    for dets in {k for k, _ in exact}:
        hits = sum(r == dets for r in rows)
        marginal = sum(q for (k, _), q in exact.items() if k == dets)
        sigma = np.sqrt(marginal * (1 - marginal) / shots) + 1e-9
        assert abs(hits / shots - marginal) < 5 * sigma + 1e-12, (text, dets, hits / shots, marginal)
    assert s.peak_terms >= 1


def test_measurement_records_and_probabilities():
    c = sq.Circuit("R 0 1\nH 0\nI[T] 0\nCX 0 1\nI[T_DAG] 1\nI[R_Y(theta=0.3)] 0\nH 1\nM 0 1")
    s = c.compile_stabilizer_rank_sampler(seed=3)
    records = s.sample_measurements(20000, threads=0)
    keys = [(a, b) for a in (False, True) for b in (False, True)]
    probs = s.probability_of(np.array(keys))
    assert abs(probs.sum() - 1) < 1e-12
    # The state vector agrees on each record's probability.
    sv = sq.Circuit(str(c) + "\nDETECTOR rec[-2]\nDETECTOR rec[-1]").exact_distribution()
    for key, p in zip(keys, probs):
        assert abs(sv.get((key, ()), 0) - p) < 1e-12
        hits = np.mean(np.all(records == np.array(key), axis=1))
        assert abs(hits - p) < 5 * np.sqrt(p * (1 - p) / 20000) + 1e-3
    assert s.probability_of(np.array(keys[1])) == pytest.approx(probs[1])
    with pytest.raises(ValueError, match="2 bits"):
        s.probability_of([True])


def test_probability_with_noise_is_an_average():
    c = sq.Circuit("R 0\nH 0\nI[T] 0\nX_ERROR(0.3) 0\nH 0\nM 0")
    s = c.compile_stabilizer_rank_sampler()
    assert s.is_noisy if hasattr(s, "is_noisy") else True
    p1 = s.probability_of([True], samples=20000, seed=1)
    exact = sq.Circuit(str(c) + "\nDETECTOR rec[-1]").exact_distribution()[((True,), ())]
    assert abs(p1 - exact) < 0.01
    assert s.probability_of([True], samples=20000, seed=1) == p1


def test_samplers_dispatch_to_the_engine_on_non_clifford_gates():
    t = sq.Circuit("R 0\nH 0\nI[T] 0\nH 0\nM 0\nDETECTOR rec[-1]")
    assert t.num_non_clifford_gates() == 1
    assert isinstance(t.compile_detector_sampler(seed=1), sq.StabilizerRankSampler)
    assert t.compile_detector_sampler(seed=1).device == t.compile_exact_sampler(seed=1).device == "cpu"
    d = t.compile_detector_sampler(seed=1).sample(20000)
    assert abs(d.mean() - np.sin(np.pi / 8) ** 2) < 0.01
    m = t.compile_sampler(seed=1).sample(20000)
    assert abs(m.mean() - np.sin(np.pi / 8) ** 2) < 0.01
    # Flips relative to the reference (all 0 here) are the records themselves.
    assert t.compile_sampler(seed=1, skip_reference_sample=True).sample(100).shape == (100, 1)
    # Stim's reading: the tag inert.
    assert t.without_tags().compile_detector_sampler(seed=1).sample(100).sum() == 0
    assert t.without_tags().num_non_clifford_gates() == 0
    # Clifford angles are Clifford; coherent errors are not gates.
    assert sq.Circuit("I[R_Z(theta=0.7853981633974483)] 0\nI_ERROR[R_Z(theta=0.1)] 0\nM 0").num_non_clifford_gates() == 0
    assert sq.Circuit("I[U3(theta=1.5707963267948966, phi=0, lambda=3.141592653589793)] 0\nM 0").num_non_clifford_gates() == 0
    assert sq.Circuit("I[U3(theta=1, phi=0, lambda=3.141592653589793)] 0\nM 0").num_non_clifford_gates() == 1
    with pytest.warns(UserWarning, match="coherent errors"):
        assert isinstance(sq.Circuit("I_ERROR[R_Z(theta=0.1)] 0\nM 0").compile_detector_sampler(), sq.DetectorSampler)


def test_rank_on_code_states_and_many_qubits():
    steane = "R 0 1 2 3 4 5 6\nH 4 5 6\nCX 4 0 4 1 4 2\nCX 5 0 5 1 5 3\nCX 6 0 6 2 6 3\nI[T] 0 1 2 3 4 5 6\nMX 0 1 2 3 4 5 6"
    s = sq.Circuit(steane).compile_stabilizer_rank_sampler(seed=0)
    s.sample(10)
    assert s.num_non_clifford == 7
    assert 1 < s.peak_terms <= 8
    # 120 qubits in a GHZ chain, four T gates: a handful of terms.
    n = 120
    text = "R " + " ".join(map(str, range(n))) + "\nH 0\n" + "\n".join(f"CX {q - 1} {q}" for q in range(1, n))
    text += "\nI[T] 0 40 80 119\nH 0 40 80 119\nM " + " ".join(map(str, range(n)))
    text += "\nDETECTOR rec[-1] rec[-2]"
    s = sq.Circuit(text).compile_stabilizer_rank_sampler(seed=0)
    d = s.sample(200, threads=0)
    assert d.shape == (200, 1) and s.num_qubits == n and s.peak_terms <= 16


def test_threads_do_not_change_the_shots():
    text = random_case(np.random.default_rng(77), n_max=3, length=(10, 12))
    c = sq.Circuit(text)
    a = c.compile_stabilizer_rank_sampler(seed=5).sample(300, threads=1)
    b = c.compile_stabilizer_rank_sampler(seed=5).sample(300, threads=0)
    assert np.array_equal(a, b)
    s = c.compile_stabilizer_rank_sampler(seed=5)
    parts = np.concatenate([s.sample(100), s.sample(200)])
    assert np.array_equal(parts, a)


def test_refusals():
    with pytest.raises(ValueError, match="density matrix"):
        sq.Circuit("I_ERROR[AMPLITUDE_DAMPING(gamma=0.1)] 0\nM 0").compile_stabilizer_rank_sampler()
    with pytest.raises(ValueError, match="leakage sampler"):
        sq.Circuit("I_ERROR[LEAK(p=0.1)] 0\nM 0").compile_stabilizer_rank_sampler()
    with pytest.raises(ValueError, match="Pauli targets"):
        sq.Circuit("H 0\nM 0\nOBSERVABLE_INCLUDE(0) Z0").compile_stabilizer_rank_sampler()
    with pytest.raises(ValueError, match="method must be"):
        sq.Circuit("M 0").exact_distribution(method="magic")


# ---------------------------------------------------------------------------------------------
# The 2.0 review's regressions
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tagged,plain,body",
    [
        ("I[R_Z(theta=1.5707963267948966)] 0", "Z 0", "H 0\n{g}\nH 0\nM 0"),
        ("I[R_X(theta=1.5707963267948966)] 0", "X 0", "{g}\nM 0"),
        ("I[R_Z(theta=0.7853981633974483)] 0\nI[R_Z(theta=0.7853981633974483)] 0", "S 0\nS 0", "H 0\n{g}\nH 0\nM 0"),
        ("I[R_Z(theta=0.7853981633974483)] 0", "S 0", "H 0\n{g}\nMY 0"),
        ("I[U3(theta=3.141592653589793, phi=0, lambda=3.141592653589793)] 0", "X 0", "{g}\nM 0"),
        ("II[R_ZZ(theta=0.7853981633974483)] 0 1", "SPP Z0*Z1", "H 0 1\n{g}\nMPP Y0*Z1\nMPP Z0*Y1\nDETECTOR rec[-2]"),
    ],
)
def test_clifford_angle_tagged_gates_are_the_gates_they_are_in_every_engine(tagged, plain, body):
    # Stim reads the tag as nothing; here the gate is run everywhere: the reference run, the
    # frame samplers, the detector error model and the exact engines agree with the plain
    # Clifford circuit (each a deterministic measurement of what the gate made).
    head, tail = "R 0 1 2\n", "\nDETECTOR rec[-1]\nX_ERROR(0.1) 2\nM 2\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]"
    c, ref = sq.Circuit(head + body.format(g=tagged) + tail), sq.Circuit(head + body.format(g=plain) + tail)
    assert c.num_non_clifford_gates() == 0 and c.is_clifford
    assert c.reference_sample().tolist() == ref.reference_sample().tolist()
    assert str(c.detector_error_model()) == str(ref.detector_error_model())
    assert isinstance(c.compile_detector_sampler(seed=1), sq.DetectorSampler)
    a = c.compile_detector_sampler(seed=1).sample(4000, separate_observables=True)
    b = ref.compile_detector_sampler(seed=1).sample(4000, separate_observables=True)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert np.array_equal(c.compile_sampler(seed=2).sample(100), ref.compile_sampler(seed=2).sample(100))
    for method in ("state_vector", "stabilizer_rank", "density_matrix"):
        assert c.exact_distribution(method=method) == pytest.approx(ref.exact_distribution(method=method))
    assert c.compile_coherent_sampler(seed=1).num_locations == 0  # no refusal of the Clifford-angle gate
    leaky = c.compile_leakage_sampler(seed=1).sample(10), ref.compile_leakage_sampler(seed=1).sample(10)
    assert all(np.array_equal(a, b) for a, b in zip(leaky[0], leaky[1]))


def test_the_frame_sampler_warns_on_coherent_errors_and_runs_them_beside_a_t_gate():
    coherent = sq.Circuit("R 0\nH 0\nI_ERROR[R_Z(theta=0.5)] 0\nH 0\nM 0\nDETECTOR rec[-1]")
    with pytest.warns(UserWarning, match="reads the circuit's coherent errors"):
        s = coherent.compile_detector_sampler(seed=1)
    assert isinstance(s, sq.DetectorSampler) and s.sample(100).sum() == 0
    with pytest.warns(UserWarning, match="coherent errors"):
        coherent.compile_sampler(seed=1)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        coherent.twirled().compile_detector_sampler(seed=1)
        coherent.without_tags().compile_detector_sampler(seed=1)
        sq.Circuit("R 0\nH 0\nI[T] 0\nH 0\nM 0\nDETECTOR rec[-1]").compile_detector_sampler(seed=1)
    # With a non-Clifford gate the stabilizer-rank sampler runs the error exactly: sin²(0.5).
    with_t = sq.Circuit("R 0\nH 0\nI_ERROR[R_Z(theta=0.5)] 0\nI[T] 0\nI[T_DAG] 0\nH 0\nM 0\nDETECTOR rec[-1]")
    assert isinstance(with_t.compile_detector_sampler(seed=1), sq.StabilizerRankSampler)
    assert abs(with_t.compile_detector_sampler(seed=1).sample(20000).mean() - np.sin(0.5) ** 2) < 0.01


def test_a_t_written_as_an_error_is_refused():
    with pytest.raises(ValueError, match="T is a gate, written I\\[T\\]"):
        sq.Circuit("I_ERROR[T] 0\nM 0")
    with pytest.raises(ValueError, match="T_DAG is a gate"):
        sq.Circuit("I_ERROR[T_DAG] 0\nM 0")
