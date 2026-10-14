"""Running a decoder on a circuit's shots: deltakit's ``run_decoding_on_circuit`` and
``EmpiricalDecodingErrorDistribution`` (deltakit-decode 0.9.2, Copyright Riverlane, Apache
License 2.0; see THIRD_PARTY_NOTICES.md), with this package's samplers and decoders. For sweeps
over many tasks, worker processes and saved results, use ``stabilizer_qec.sinter``."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from stabilizer_qec.analysis._dk._empirical_decoding_error_distribution import EmpiricalDecodingErrorDistribution

__all__ = ["EmpiricalDecodingErrorDistribution", "run_decoding_on_circuit"]


def run_decoding_on_circuit(
    circuit: Any,
    max_shots: int,
    decoder: Any,
    max_batch_size: int = 10_000,
    target_rse: Optional[float] = None,
    min_fails: int = 10,
    *,
    seed: Optional[int] = None,
    progress: bool = False,
) -> Dict[str, int]:
    """Samples the circuit's detection events in batches and decodes them: ``{"shots",
    "fails", "fails_log_0", ...}``. Stops after ``max_shots``, or once every observable's
    relative standard error is at most ``target_rse`` with more than ``min_fails`` failures.
    ``decoder`` has ``decode_batch_to_logical_flip`` (deltakit's) or ``decode_batch`` (this
    package's and PyMatching's). ``progress`` shows tqdm's bar (deltakit always does)."""
    from stabilizer_qec import Circuit

    circuit = Circuit(circuit)
    sampler = circuit.compile_detector_sampler(seed=seed)
    batches = [max_batch_size] * (max_shots // max_batch_size)
    if (remaining_shots := max_shots - sum(batches)) > 0:
        batches.append(remaining_shots)
    decode = getattr(decoder, "decode_batch_to_logical_flip", None) or decoder.decode_batch
    result: Dict[str, int] = {"shots": 0, "fails": 0}
    result.update({f"fails_log_{log}": 0 for log in range(circuit.num_observables)})
    pbar: Any = batches
    if progress:
        from tqdm import tqdm

        pbar = tqdm(batches)
    for batch_size in pbar:
        syndrome, actual_observables = sampler.sample(shots=batch_size, separate_observables=True)
        predicted_observables = np.asarray(decode(syndrome)).astype(bool)
        result["shots"] += batch_size
        result["fails"] += int(np.sum(np.any(predicted_observables != actual_observables, axis=1)))
        fails_per_observable = np.sum(predicted_observables != actual_observables, axis=0).astype(np.int_)
        rse = 0.0
        min_fails_any_log = 0
        for log, fails in enumerate(fails_per_observable):
            log_fails_name = f"fails_log_{log}"
            result[log_fails_name] += int(fails)
            lep = result[log_fails_name] / result["shots"]
            if lep > 0:
                rse_log = np.sqrt(lep * (1 - lep) / result["shots"]) / lep
                rse = max(rse, rse_log)
                min_fails_any_log = max(min_fails_any_log, result[log_fails_name])
        if progress:
            pbar.set_description(f"Shots: {result['shots']}; Fails: {result['fails']}; RSE: {rse:.3f}")
        if target_rse is not None and rse <= target_rse and min_fails_any_log > min_fails:
            break
    return result
