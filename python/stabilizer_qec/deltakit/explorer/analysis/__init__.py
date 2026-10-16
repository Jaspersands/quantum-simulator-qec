# (c) Copyright Riverlane 2020-2026. All rights reserved.
# modified for stabilizer_qec: its Stim is stabilizer_qec.deltakit_stim and its PyMatching this package's, and deltakit_{circuit,core,decode,explorer} are stabilizer_qec.deltakit.{circuit,core,decode,explorer}.
"""Description of ``deltakit.explorer.analysis`` namespace here."""

from stabilizer_qec.deltakit.explorer.analysis._analysis import (
    get_exp_fit,
    get_lambda_fit,
)
from stabilizer_qec.deltakit.explorer.analysis._binomial_fit import (
    DEFAULT_MAX_LIKELIHOOD_FACTOR,
    ConfidenceInterval,
    fit_binomial,
    fit_binomial_batch,
    fit_leppr_and_spam,
)
from stabilizer_qec.deltakit.explorer.analysis._lambda import (
    LambdaData,
    calculate_lambda_and_lambda_stddev,
    calculate_lambda_asymmetric,
)
from stabilizer_qec.deltakit.explorer.analysis._leppr import (
    LogicalErrorProbabilityPerRoundData,
    calculate_lep_and_lep_stddev,
    calculate_lep_asymmetric,
    compute_logical_error_per_round,
    fit_logical_error_per_round_asymmetric,
    simulate_different_round_numbers_for_lep_per_round_estimation,
)
from stabilizer_qec.deltakit.explorer.analysis._quops import (
    predict_distance_for_quops,
    predict_quops_at_distance,
    predict_quops_interval,
)

from . import error_budget

# List only public members in `__all__`.
__all__ = [s for s in dir() if not s.startswith("_")]
