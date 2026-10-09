"""Rates from weighted shots, as the coherent sampler gives them."""

from __future__ import annotations

from typing import Any

import numpy as np


def weighted_rate(hits: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    """The weighted rate of ``hits`` (bools, one per shot) and its standard error:
    Σ w·hit / Σ w, the error from the delta method for a ratio of sums."""
    hits = np.asarray(hits, dtype=float)
    w = np.asarray(weights, dtype=float)
    if hits.shape != w.shape:
        raise ValueError(f"hits has shape {hits.shape} and weights {w.shape}")
    total = w.sum()
    if total <= 0:
        raise ValueError("the weights sum to zero")
    p = float((w * hits).sum() / total)
    return p, float(np.sqrt(((w * (hits - p)) ** 2).sum()) / total)


def weighted_logical_error_rate(decoder: Any, detections: np.ndarray, observables: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    """A decoder's logical error rate on weighted shots (from ``CoherentSampler.sample``):
    the weighted share of shots it predicts wrongly, and its standard error. ``decoder`` is any
    of the package's decoders, or anything with ``decode_batch``."""
    predictions = np.asarray(decoder.decode_batch(detections))
    wrong = (predictions != np.asarray(observables)).reshape(len(weights), -1).any(axis=1)
    return weighted_rate(wrong, weights)


def effective_sample_size(weights: np.ndarray) -> float:
    """(Σ w)² / Σ w²: how many unweighted shots the weighted ones are worth."""
    w = np.asarray(weights, dtype=float)
    return float(w.sum() ** 2 / (w**2).sum())
