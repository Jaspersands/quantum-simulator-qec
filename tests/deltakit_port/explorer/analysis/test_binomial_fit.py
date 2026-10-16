# modified for stabilizer_qec: its Stim is stabilizer_qec.deltakit_stim and its PyMatching this package's, and deltakit_{circuit,core,decode,explorer} are stabilizer_qec.deltakit.{circuit,core,decode,explorer}.
import math

import numpy as np
import pytest

from stabilizer_qec.deltakit.explorer.analysis import (
    ConfidenceInterval,
    calculate_lep_asymmetric,
    fit_binomial,
    fit_binomial_batch,
    fit_leppr_and_spam,
)
from stabilizer_qec.deltakit.explorer.analysis._binomial_fit import log_binomial


def _model_counts(
    eps: float, spam: float, rounds: list[int], shots: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rounds_arr = np.asarray(rounds)
    fidelity = (1 - 2 * spam) * (1 - 2 * eps) ** rounds_arr
    fails = np.round((1 - fidelity) / 2 * shots).astype(int)
    return rounds_arr, fails, np.full(len(rounds), shots)


class TestLogBinomial:
    def test_matches_sinter_documented_values(self) -> None:
        # From sinter._probability_util.log_binomial docstring examples.
        assert log_binomial(p=0.5, num_trials=100, num_successes=50) == pytest.approx(
            -2.5308762, abs=1e-5
        )
        assert log_binomial(
            p=0.2, num_trials=1_000_000, num_successes=1_000
        ) == pytest.approx(-216626.97, rel=1e-6)

    def test_zero_probability_with_hits_is_neg_inf(self) -> None:
        assert log_binomial(p=0.0, num_trials=10, num_successes=1) == -math.inf

    def test_certain_probability_with_misses_is_neg_inf(self) -> None:
        assert log_binomial(p=1.0, num_trials=10, num_successes=1) == -math.inf


class TestFitBinomial:
    def test_matches_sinter_balanced_example(self) -> None:
        # sinter.fit_binomial(num_shots=10, num_hits=5, factor=9) -> 0.202/0.5/0.798
        fit = fit_binomial(num_shots=10, num_hits=5, max_likelihood_factor=9.0)
        assert fit.best == pytest.approx(0.5)
        assert fit.low == pytest.approx(0.202, abs=1e-3)
        assert fit.high == pytest.approx(0.798, abs=1e-3)

    def test_upward_skewed_near_zero(self) -> None:
        # sinter.fit_binomial(1e8, 2, 1000) -> low=2e-10, best=2e-8, high=1.259e-7
        fit = fit_binomial(num_shots=100_000_000, num_hits=2)
        assert fit.best == pytest.approx(2e-8)
        assert fit.high == pytest.approx(1.259e-7, rel=1e-3)
        assert (fit.high - fit.best) > 5 * (fit.best - fit.low)

    def test_low_bound_is_non_negative(self) -> None:
        fit = fit_binomial(num_shots=1_000_000, num_hits=1)
        assert fit.low >= 0

    def test_zero_hits(self) -> None:
        fit = fit_binomial(num_shots=1_000_000, num_hits=0)
        assert fit.low == 0.0
        assert fit.best == 0.0
        assert fit.high > 0

    def test_all_hits(self) -> None:
        fit = fit_binomial(num_shots=100, num_hits=100)
        assert fit.high == 1.0
        assert fit.best == 1.0

    def test_zero_shots(self) -> None:
        assert fit_binomial(num_shots=0, num_hits=0) == ConfidenceInterval(
            low=0.0, best=0.5, high=1.0
        )

    @pytest.mark.parametrize(
        ("shots", "hits", "factor"),
        [(10, 11, 1000.0), (-1, 0, 1000.0), (10, 5, 0.5)],
    )
    def test_invalid_inputs_raise(self, shots: int, hits: int, factor: float) -> None:
        with pytest.raises(ValueError):
            fit_binomial(num_shots=shots, num_hits=hits, max_likelihood_factor=factor)


class TestFitBinomialBatch:
    def test_shapes_and_values_match_scalar(self) -> None:
        shots = np.array([1000, 5000])
        hits = np.array([3, 40])
        low, best, high = fit_binomial_batch(shots, hits)
        assert low.shape == best.shape == high.shape == (2,)
        scalar = fit_binomial(num_shots=1000, num_hits=3)
        assert low[0] == pytest.approx(scalar.low)
        assert high[0] == pytest.approx(scalar.high)

    def test_mismatched_shapes_raise(self) -> None:
        with pytest.raises(ValueError):
            fit_binomial_batch([1000, 2000], [3])


class TestCalculateLepAsymmetric:
    def test_best_is_ratio(self) -> None:
        low, best, high = calculate_lep_asymmetric([2, 151, 34], [500000] * 3)
        np.testing.assert_allclose(best, np.array([2, 151, 34]) / 500000)
        assert np.all(low >= 0)
        assert np.all(low <= best)
        assert np.all(best <= high)

    def test_asymmetric_when_few_hits(self) -> None:
        low, best, high = calculate_lep_asymmetric([2], [1_000_000])
        lower_margin = best[0] - low[0]
        upper_margin = high[0] - best[0]
        assert upper_margin != pytest.approx(lower_margin, rel=0.2)

    def test_accepts_zero_fails(self) -> None:
        low, best, high = calculate_lep_asymmetric([0], [1_000_000])
        assert low[0] == 0.0
        assert best[0] == 0.0
        assert high[0] > 0

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            calculate_lep_asymmetric([1, 2], [1000])

    def test_negative_fails_raise(self) -> None:
        with pytest.raises(ValueError):
            calculate_lep_asymmetric([-1], [1000])


class TestFitLepprAndSpam:
    def test_recovers_true_rate(self) -> None:
        rounds, fails, shots = _model_counts(0.05, 0.01, [2, 6, 10, 14], 200_000)
        leppr, spam = fit_leppr_and_spam(rounds, fails, shots)
        assert leppr.best == pytest.approx(0.05, rel=0.05)
        assert spam.best == pytest.approx(0.01, rel=0.3)

    def test_brackets_and_non_negative(self) -> None:
        rounds, fails, shots = _model_counts(1e-6, 1e-7, [2, 6, 10, 14], 1_000_000)
        leppr, spam = fit_leppr_and_spam(rounds, fails, shots)
        for fit in (leppr, spam):
            assert 0 <= fit.low <= fit.best <= fit.high

    def test_asymmetric_in_rare_regime(self) -> None:
        rounds, fails, shots = _model_counts(1e-6, 1e-7, [2, 6, 10, 14], 1_000_000)
        leppr, _ = fit_leppr_and_spam(rounds, fails, shots)
        assert (leppr.best - leppr.low) != pytest.approx(
            leppr.high - leppr.best, rel=0.2
        )

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            fit_leppr_and_spam([2, 4], [1], [1000])

    def test_fixed_spam_pins_spam_and_tightens_leppr(self) -> None:
        rounds, fails, shots = _model_counts(1e-6, 1e-7, [2, 6, 10, 14], 1_000_000)
        free_leppr, _ = fit_leppr_and_spam(rounds, fails, shots)
        fixed_leppr, fixed_spam = fit_leppr_and_spam(
            rounds, fails, shots, fixed_spam=1e-7
        )
        assert fixed_spam.low == fixed_spam.best == fixed_spam.high
        # Holding SPAM fixed removes the correlation, so the lower bound cannot be
        # looser than the free fit.
        assert fixed_leppr.low >= free_leppr.low
