"""The perturbative expansion (Circuit.perturbative_distribution, perturbative_logical_error_rate):
exact at full order, short by O(p^(K+1)) below it, its polynomials' coefficients the exact Taylor
coefficients, and QuantumClifford.jl's pinned purification polynomials."""

import numpy as np
import pytest

import stabilizer_qec as sq

# A repetition code round with every kind of site: Pauli channels on one qubit and two, a
# measurement flip, a heralded error, a chain of correlated errors and feedback.
REPETITION = """
R 0 1 2 3 4
X_ERROR({p}) 0 1 2
PAULI_CHANNEL_1({p}, {p}, {p}) 1
CX 0 3 1 3 1 4 2 4
DEPOLARIZE2({p}) 1 4
M({p}) 3 4
HERALDED_PAULI_CHANNEL_1({p}, {p}, 0, {p}) 2
E({p}) X0 X1
ELSE_CORRELATED_ERROR({p}) X2
CX rec[-1] 2
DEPOLARIZE1({p}) 0
M 0 1 2
DETECTOR rec[-6]
DETECTOR rec[-5]
DETECTOR rec[-3] rec[-2]
OBSERVABLE_INCLUDE(0) rec[-1]
"""


def _circuit(p, chain=True):
    text = REPETITION.format(p=p)
    if not chain:
        text = text.replace(f"ELSE_CORRELATED_ERROR({p}) X2\n", "")
    return sq.Circuit(text)


def _close(a, b, tol):
    assert set(a) == set(b)
    for k in a:
        assert a[k] == pytest.approx(b[k], abs=tol), k


def test_full_order_is_exact():
    c = _circuit(0.03)
    exact = c.exact_distribution()
    pe = c.perturbative_distribution(order=40)
    _close({k: v for k, v in pe.items() if v > 1e-15}, exact, 1e-12)
    assert sum(pe.values()) == pytest.approx(1, abs=1e-13)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_truncation_misses_order_k_plus_one(order):
    # The missing mass is the fault sets of more than K faults: O(p^(K+1)), halving p dividing
    # it by about 2^(K+1).
    missing = []
    for p in (0.004, 0.002):
        pe = _circuit(p).perturbative_distribution(order=order)
        missing.append(1 - sum(pe.values()))
    assert 2 ** (order + 1) * 0.8 < missing[0] / missing[1] < 2 ** (order + 1) * 1.25
    # Every weight is a lower bound.
    exact = _circuit(0.004).exact_distribution()
    pe = _circuit(0.004).perturbative_distribution(order=order)
    assert all(pe[k] <= exact.get(k, 0) + 1e-15 for k in pe)


def test_polynomials_are_taylor_coefficients():
    c = _circuit(0.05)
    full = c.perturbative_distribution(order=40, symbolic=True, degree="full")
    exact = c.exact_distribution()
    # The full expansion is the exact distribution as a polynomial in the noise scale.
    for k, poly in full.items():
        assert poly(1.0) == pytest.approx(exact.get(k, 0), abs=1e-13)
    # The order-2 polynomials' coefficients are its first ones.
    cut = c.perturbative_distribution(order=2, symbolic=True)
    for k, poly in cut.items():
        assert poly.degree() <= 2
        want = np.zeros(3)
        n = min(3, len(full[k].coef))
        want[:n] = full[k].coef[:n]
        np.testing.assert_allclose(np.pad(poly.coef, (0, 3 - len(poly.coef))), want, atol=1e-13)
    # The scale is the circuit's noise: x = 0.5 is the circuit at half the noise (each fault's
    # probability scaled, which for an ELSE_CORRELATED_ERROR is not its parameter's).
    full = _circuit(0.05, chain=False).perturbative_distribution(order=40, symbolic=True, degree="full")
    half = _circuit(0.025, chain=False).exact_distribution()
    for k, v in half.items():
        assert full[k](0.5) == pytest.approx(v, abs=1e-13)


def test_full_polynomial_at_order_k_is_the_numeric_sum():
    c = _circuit(0.02)
    for order in (1, 2):
        num = c.perturbative_distribution(order=order)
        sym = c.perturbative_distribution(order=order, symbolic=True, degree="full")
        _close({k: v(1.0) for k, v in sym.items()}, num, 1e-14)


def test_quantumclifford_bell_purification():
    """QuantumClifford.jl's symbolic test (test_noisycircuits.jl): two Bell pairs, each qubit
    depolarized (X, Y, Z each with probability e), CNOTs 1->3 and 2->4, a Bell measurement of
    3 and 4 in X (failure if odd), then 1 and 2 verified as the good Bell pair (false success
    if not), at its default max_order 1. Here e = x / 4 with DEPOLARIZE1(0.75 x)."""
    c = sq.Circuit(
        """
        H 0 2
        CX 0 1 2 3
        DEPOLARIZE1(0.75) 0 1 2 3
        CX 0 2 1 3
        MX 2 3
        DETECTOR rec[-1] rec[-2]
        MPP X0*X1 Z0*Z1
        OBSERVABLE_INCLUDE(0) rec[-2]
        OBSERVABLE_INCLUDE(1) rec[-1]
        """
    )
    pe = c.perturbative_distribution(order=1, symbolic=True, degree="full")
    zero = np.polynomial.Polynomial([0.0])
    failure = sum((w for (d, o), w in pe.items() if d[0]), zero)
    false_success = sum((w for (d, o), w in pe.items() if not d[0] and any(o)), zero)
    true_success = sum((w for (d, o), w in pe.items() if not d[0] and not any(o)), zero)
    scale = 0.25 ** np.arange(5)
    np.testing.assert_allclose(false_success.coef, np.array([0, 6, -54, 162, -162]) * scale, atol=1e-14)
    np.testing.assert_allclose(failure.coef, np.array([0, 4, -36, 108, -108]) * scale, atol=1e-14)
    np.testing.assert_allclose(true_success.coef, np.array([1, -10, 36, -54, 27]) * scale, atol=1e-14)


def test_logical_error_rate():
    c = sq.Circuit.generated("repetition_code:memory", distance=5, rounds=3, before_round_data_depolarization=0.001, before_measure_flip_probability=0.001)
    # Two faults are corrected at distance 5, three not always.
    assert sq.perturbative_logical_error_rate(c, order=2) == 0.0
    rate = sq.perturbative_logical_error_rate(c, order=3, symbolic=True)
    assert rate.coef[3] > 0
    # At full order on a small circuit, the decoder's exact failure rate.
    small = sq.Circuit.generated("repetition_code:memory", distance=3, rounds=1, before_round_data_depolarization=0.05, before_measure_flip_probability=0.05)
    m = sq.Matching.from_detector_error_model(small.detector_error_model(decompose_errors=True))
    exact = small.exact_distribution()
    want = 0.0
    for (d, o), p in exact.items():
        if (m.decode(np.array(d, dtype=np.uint8)).astype(bool) != np.array(o)).any():
            want += p
    assert sq.perturbative_logical_error_rate(small, order=40) == pytest.approx(want, abs=1e-13)
    # A decoder as a function: predicting no flip fails whenever the observable flips.
    flips = sum(p for (d, o), p in exact.items() if any(o))
    got = sq.perturbative_logical_error_rate(small, order=40, decoder=lambda dets: np.zeros((len(dets), 1), dtype=bool))
    assert got == pytest.approx(flips, abs=1e-13)


def test_surface_code_rate_matches_sampling():
    # The order-3 expansion of a d = 3 memory against sampled shots: the sampled rate is the
    # expansion's plus what more faults add.
    c = sq.memory_circuit(distance=3, rounds=2, p=0.002)
    rate = sq.perturbative_logical_error_rate(c, order=3)
    dets, obs = c.compile_detector_sampler(seed=5).sample(1_000_000, separate_observables=True)
    m = sq.Matching.from_detector_error_model(c.detector_error_model(decompose_errors=True))
    sampled = np.mean((m.decode_batch(dets) != obs).any(axis=1))
    sigma = np.sqrt(sampled / 1_000_000)
    assert rate - 4 * sigma < sampled < rate * 1.3 + 4 * sigma


def test_refusals():
    with pytest.raises(ValueError, match="Clifford"):
        sq.Circuit("R 0\nI[T] 0\nM 0\nDETECTOR rec[-1]").perturbative_distribution()
    with pytest.raises(ValueError):
        # A detector that is not deterministic.
        sq.Circuit("H 0\nX_ERROR(0.1) 0\nM 0\nDETECTOR rec[-1]").perturbative_distribution()
    with pytest.raises(ValueError, match="degree"):
        sq.Circuit("M 0").perturbative_distribution(degree=2)
    with pytest.raises(ValueError, match="degree"):
        sq.Circuit("M 0").perturbative_distribution(symbolic=True, degree="all")
    with pytest.raises(ValueError, match="terms"):
        _circuit(0.01).perturbative_distribution(order=3, max_terms=4)


def test_a_huge_degree_is_full():
    c = sq.Circuit("R 0\nX_ERROR(0.1) 0\nM 0\nDETECTOR rec[-1]")
    full = c.perturbative_distribution(order=1, symbolic=True, degree="full")
    huge = c.perturbative_distribution(order=1, symbolic=True, degree=2**64)
    assert {k: v.coef.tolist() for k, v in full.items()} == {k: v.coef.tolist() for k, v in huge.items()}
