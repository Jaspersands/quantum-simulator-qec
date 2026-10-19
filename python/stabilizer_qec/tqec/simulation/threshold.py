# Ported from tqec (tqec/tqec 0.2.0, src/tqec/simulation/threshold.py; Apache License 2.0, see
# LICENSE in the tqec directory), modified for stabilizer_qec: this package's compiler and sinter.
"""tqec's search for a computation's threshold, by bisection on the noise strength."""

from __future__ import annotations

import multiprocessing
from collections.abc import Callable, Iterable, Sequence
from math import isclose
from typing import TYPE_CHECKING, Optional

from ... import sinter
from .._errors import TQECError
from ..compile import FIXED_BULK_CONVENTION, Convention, compile_block_graph

if TYPE_CHECKING:
    from ..block_graph import BlockGraph
    from ..correlation import CorrelationSurface
    from ..noise_model import NoiseModel


def _is_only_floats(seq: Sequence[Optional[float]]) -> bool:
    return not any(elem is None for elem in seq)


def _is_sorted(seq: Sequence[float]) -> bool:
    if not seq:
        return True
    val = seq[0]
    for i in range(1, len(seq)):
        if seq[i] < val:
            return False
        val = seq[i]
    return True


def get_logical_error_rate_per_shot(stat: "sinter.TaskStats", max_likelihood_factor: float = 1e3) -> "sinter.Fit":
    """The logical error rate per shot of ``stat`` with its error bars
    (:func:`stabilizer_qec.sinter.fit_binomial`; the best estimate is NaN without errors)."""
    result = sinter.fit_binomial(num_shots=stat.shots - stat.discards, num_hits=stat.errors, max_likelihood_factor=max_likelihood_factor)
    if stat.errors == 0:
        result = sinter.Fit(low=result.low, high=result.high, best=float("nan"))
    return result


def binary_search_threshold(
    block_graph: "BlockGraph",
    observable: "CorrelationSurface",
    noise_model_factory: Callable[[float], "NoiseModel"],
    minp: float = 1e-5,
    maxp: float = 0.1,
    ks: Sequence[int] = (1, 2),
    atol: float = 1e-4,
    rtol: float = 1e-4,
    manhattan_radius: int = 2,
    convention: Convention = FIXED_BULK_CONVENTION,
    detector_database=None,
    num_workers: int = multiprocessing.cpu_count(),
    max_shots: int = 10_000_000,
    max_errors: int = 5_000,
    decoders: Iterable[str] = ("pymatching",),
) -> tuple[float, dict[int, list[tuple[float, "sinter.Fit"]]]]:
    """The threshold of ``observable`` on ``block_graph``, by bisection on ``p`` in
    ``[minp, maxp]`` as tqec searches it: at the midpoint, if the logical error rate rises with
    ``k`` (or is above 0.4 at every ``k``) the threshold is below it, if it falls with ``k``
    above; it stops when the bounds are close (``atol``, ``rtol``) or the ``ks`` disagree.

    Returns the estimate and, for each ``k``, the ``(p, fit)`` pairs computed on the way. The
    estimates' error bars are not used, so tolerances below the sampling noise of
    ``max_shots`` and ``max_errors`` mean little."""
    compiled_graph = compile_block_graph(block_graph, convention, [observable])
    ks = tuple(sorted(ks))
    noiseless_circuits = [compiled_graph.generate_stim_circuit(k, manhattan_radius=manhattan_radius, detector_database=detector_database) for k in ks]
    computed_logical_errors: dict[int, list[tuple[float, sinter.Fit]]] = {k: [] for k in ks}
    while not isclose(minp, maxp, rel_tol=rtol, abs_tol=atol):
        midp = (minp + maxp) / 2
        noise_model = noise_model_factory(midp)
        stats = sinter.collect(
            num_workers=num_workers,
            tasks=[sinter.Task(circuit=noise_model.noisy_circuit(circ), json_metadata={"d": 2 * k + 1, "r": 2 * k + 1, "p": midp}) for k, circ in zip(ks, noiseless_circuits)],
            max_shots=max_shots,
            max_errors=max_errors,
            decoders=decoders,
            hint_num_tasks=len(ks),
        )
        logical_errors_fits = [get_logical_error_rate_per_shot(stat) for stat in sorted(stats, key=lambda s: s.json_metadata["d"])]
        for i, k in enumerate(ks):
            computed_logical_errors[k].append((midp, logical_errors_fits[i]))
        logical_errors = [lerr_fit.best for lerr_fit in logical_errors_fits]
        if not _is_only_floats(logical_errors):
            raise TQECError("One of the computed logical errors is None. That likely means that you need more shots in order to have at least a few errors on all the simulations.")
        if all(lerr is not None and lerr > 0.4 for lerr in logical_errors):
            # The logical error rate saturates around 0.5: p is too high.
            maxp = midp
        elif _is_sorted([lerr for lerr in logical_errors if lerr is not None]):
            # Larger k, larger logical error rate: above the threshold.
            maxp = midp
        elif _is_sorted([lerr for lerr in logical_errors if lerr is not None][::-1]):
            # Larger k, smaller logical error rate: below the threshold.
            minp = midp
        else:
            # No clear ordering (three or more ks, some below and some above): stop here.
            break
    return (minp + maxp) / 2, computed_logical_errors
