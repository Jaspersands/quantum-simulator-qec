"""Logical error rates by the perturbative expansion (``Circuit.perturbative_distribution``)."""

from __future__ import annotations

from typing import Any

import numpy as np


def perturbative_logical_error_rate(circuit: Any, order: int = 2, *, decoder: Any = None, symbolic: bool = False, degree: Any = None, max_terms: int = 1 << 22) -> Any:
    """The probability that ``decoder`` mispredicts the observables, summed over every set of
    at most ``order`` faults: each outcome of ``circuit.perturbative_distribution(order)`` is
    decoded once and its weight counted when the prediction is wrong. Without sampling noise, so
    exact to O(p^(order+1)) however small the rate (a lower bound, short of the exact rate by
    the fault sets of more than ``order`` faults).

    ``decoder`` is anything with ``decode_batch`` (detection events, one row per outcome, to
    predicted observable flips) or a function doing the same; by default matching on the
    circuit's detector error model. With ``symbolic`` the rate is a
    ``numpy.polynomial.Polynomial`` in the noise scale x (``degree`` as
    ``perturbative_distribution``'s), the decoder fixed.

    >>> import stabilizer_qec as sq
    >>> c = sq.Circuit.generated("repetition_code:memory", distance=3, rounds=2, before_measure_flip_probability=0.001, after_reset_flip_probability=0.001)
    >>> sq.perturbative_logical_error_rate(c, order=1)  # one fault is always corrected
    0.0
    >>> rate = sq.perturbative_logical_error_rate(c, order=2, symbolic=True)
    >>> bool(rate.coef[2] > 0)  # two can be miscorrected: O(p^2)
    True
    """
    from ._circuit import Circuit

    c = circuit if isinstance(circuit, Circuit) else Circuit(str(circuit))
    dist = c.perturbative_distribution(order, symbolic=symbolic, degree=degree, max_terms=max_terms)
    if decoder is None:
        from ._matching import Matching

        decoder = Matching.from_detector_error_model(c.detector_error_model(decompose_errors=True))
    keys = list(dist)
    dets = np.array([k[0] for k in keys], dtype=np.uint8).reshape(len(keys), c.num_detectors)
    obs = np.array([k[1] for k in keys], dtype=bool).reshape(len(keys), c.num_observables)
    run = decoder.decode_batch if hasattr(decoder, "decode_batch") else decoder
    predicted = np.asarray(run(dets)).astype(bool).reshape(len(keys), c.num_observables)
    wrong = (predicted != obs).any(axis=1)
    total: Any = np.polynomial.Polynomial([0.0]) if symbolic else 0.0
    for key, w in zip(keys, wrong):
        if w:
            total = total + dist[key]
    return total
