# (c) Copyright Riverlane 2020-2026. All rights reserved.
# modified for stabilizer_qec: its Stim is stabilizer_qec.deltakit_stim and its PyMatching this package's, and deltakit_{circuit,core,decode,explorer} are stabilizer_qec.deltakit.{circuit,core,decode,explorer}.
"""Description of ``deltakit.explorer.visualisation`` namespace here."""

from stabilizer_qec.deltakit.explorer.plotting._correlation_matrix import (
    correlation_matrix,
    defect_diagram,
    defect_rates,
)
from stabilizer_qec.deltakit.explorer.plotting._detection_on_patch import (
    DetectionProbabilityAggregation,
    plot_detection_probability_on_patch,
)
from stabilizer_qec.deltakit.explorer.plotting._lambda import plot_lambda
from stabilizer_qec.deltakit.explorer.plotting._leppr import plot_leppr
from stabilizer_qec.deltakit.explorer.plotting.plotting import plot
from stabilizer_qec.deltakit.explorer.plotting.results import (
    LambdaResult,
    LogicalErrorProbabilityPerRoundResult,
    interpolate_lambda,
    interpolate_leppr,
)

# List only public members in `__all__`.
__all__ = [s for s in dir() if not s.startswith("_")]
