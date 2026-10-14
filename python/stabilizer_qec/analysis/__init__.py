"""Analysis of QEC experiments: qec-util's tools (``circuits``, ``dems``, ``dem_instrs``,
``data_processing``, ``distance``, ``mod2``, ``performance``, ``samplers``, ``threshold``) and
deltakit's analysis (Λ and logical-error-per-round fits, quops prediction, binomial fits,
``pij``, ``decoding``, ``error_budget``), ported to need only numpy."""

# Ported from qec-util 0.5.0 (Copyright (c) 2023 MarcSerraPeralta, MIT License; see THIRD_PARTY_NOTICES.md),
# modified for stabilizer_qec: Stim and PyMatching are this package's, and numpy stands in for its other dependencies.
from . import (
    circuits,
    data_processing,
    dem_instrs,
    dems,
    distance,
    mod2,
    performance,
    samplers,
    threshold,
)

__version__ = "0.5.0"
__all__ = [
    "performance",
    "data_processing",
    "distance",
    "dems",
    "circuits",
    "dem_instrs",
    "threshold",
    "mod2",
    "samplers",
]

from . import decoding, error_budget, pij
from ._dk._analysis import get_exp_fit, get_lambda_fit
from ._dk._binomial_fit import DEFAULT_MAX_LIKELIHOOD_FACTOR, ConfidenceInterval, fit_binomial, fit_binomial_batch, fit_leppr_and_spam
from ._dk._lambda import LambdaData, calculate_lambda_and_lambda_stddev, calculate_lambda_asymmetric
from ._dk._leppr import (
    LogicalErrorProbabilityPerRoundData,
    calculate_lep_and_lep_stddev,
    calculate_lep_asymmetric,
    compute_logical_error_per_round,
    fit_logical_error_per_round_asymmetric,
    simulate_different_round_numbers_for_lep_per_round_estimation,
)
from ._dk._quops import predict_distance_for_quops, predict_quops_at_distance, predict_quops_interval
