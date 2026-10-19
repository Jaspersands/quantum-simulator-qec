# Ported from Tsim (QuEraComputing/tsim, bloqade-tsim 0.1.5, src/tsim/sampler.py; Apache License 2.0;
# see LICENSE in this directory), modified for stabilizer_qec: the samplers are this package's (the
# frame sampler on Clifford circuits, the stabilizer-rank sampler on non-Clifford ones) under Tsim's
# interface and conventions. Tsim's compiled ZX programs, its JAX batches and its postselection fast
# path are not ported: ``batch_size`` and ``postselection_mask`` are checked as Tsim checks them,
# ``strategy`` is accepted (Tsim does not check it either), and none of them changes a sample (every
# row is a full sample).
"""Tsim's samplers on this package's engines: ``CompiledMeasurementSampler`` (measurement
records), ``CompiledDetectorSampler`` (each detector's and observable's value, the parity of its
measurements, as Tsim reports it) and ``CompiledStateProbs`` (an outcome's probability given each of
a batch of noise draws)."""

from __future__ import annotations

from typing import Any, Union

import numpy as np

from stabilizer_qec import _circuit
from stabilizer_qec._util import call, count, seed_of
from stabilizer_qec.tsim.circuit import Circuit

# The most measurement records a detector state of CompiledStateProbs sums over.
MAX_RECORDS_PER_STATE = 1 << 16


def _as_circuit(circuit: Any) -> _circuit.Circuit:
    return circuit if isinstance(circuit, _circuit.Circuit) else Circuit(circuit)


def _check_shots(shots: Any) -> int:
    if isinstance(shots, bool) or not isinstance(shots, (int, np.integer)):
        raise TypeError(f"shots must be an integer, not {type(shots).__name__}")
    if shots < 0:
        raise ValueError(f"shots must be non-negative, got {shots}")
    return int(shots)


def _check_batch_size(batch_size: Any) -> None:
    if batch_size is not None and batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")


def _maybe_bit_pack(array: np.ndarray, *, bit_packed: bool) -> np.ndarray:
    """Optionally bit-pack a boolean array."""
    if not bit_packed:
        return array
    return np.packbits(array.astype(np.bool_), axis=1, bitorder="little")


def _converter(circuit: _circuit.Circuit) -> _circuit.MeasurementsToDetectionEventsConverter:
    # Parities of records alone (an all-zero reference); the tags change no detector's records.
    return _circuit.Circuit.without_tags(circuit).compile_m2d_converter(skip_reference_sample=True)


def reference_signs(circuit: _circuit.Circuit) -> tuple:
    """The parity of each detector and observable in the circuit's reference run
    (``reference_sample``), from which this package's samplers report flips: a flip XOR its
    sign is the detector's value."""
    ref = _circuit.Circuit.reference_sample(circuit)
    d, o = _converter(circuit).convert(measurements=ref[None, :], separate_observables=True)
    return d[0], o[0]


def parity_matrix(circuit: _circuit.Circuit) -> np.ndarray:
    """The (detectors + observables) × measurements matrix over GF(2) whose row k has a 1 for each
    measurement detector (then observable) k is the parity of."""
    m = circuit.num_measurements
    d, o = _converter(circuit).convert(measurements=np.eye(m, dtype=bool), separate_observables=True)
    return np.concatenate([d, o], axis=1).T.astype(np.uint8)


def _affine_solutions(h: np.ndarray, s: np.ndarray) -> Union[tuple, None]:
    """A record x0 with h x0 = s (mod 2) and a basis of h's kernel, or None if there is none."""
    rows, cols = h.shape
    a = np.concatenate([h % 2, (np.asarray(s, dtype=np.uint8) % 2)[:, None]], axis=1).astype(np.uint8)
    pivots = []
    r = 0
    for c in range(cols):
        if r == rows:
            break
        hits = np.flatnonzero(a[r:, c])
        if hits.size == 0:
            continue
        p = r + hits[0]
        if p != r:
            a[[r, p]] = a[[p, r]]
        mask = a[:, c].astype(bool)
        mask[r] = False
        a[mask] ^= a[r]
        pivots.append(c)
        r += 1
    if a[r:, -1].any():
        return None
    x0 = np.zeros(cols, dtype=np.uint8)
    for i, c in enumerate(pivots):
        x0[c] = a[i, -1]
    pivot_set = set(pivots)
    basis = []
    for f in range(cols):
        if f in pivot_set:
            continue
        v = np.zeros(cols, dtype=np.uint8)
        v[f] = 1
        for i, c in enumerate(pivots):
            v[c] = a[i, f]
        basis.append(v)
    return x0, np.array(basis, dtype=np.uint8).reshape(len(basis), cols)


class CompiledMeasurementSampler:
    """Samples measurement outcomes from a quantum circuit: each shot's record, as Tsim's
    ``CompiledMeasurementSampler`` (this package's sampler of raw records: the frame sampler on a
    Clifford circuit, the stabilizer-rank sampler on a non-Clifford one)."""

    def __init__(self, circuit: Any, *, strategy: str = "cat5", seed: Union[int, None] = None) -> None:
        """Create a measurement sampler.

        Args:
            circuit: The quantum circuit to compile.
            strategy: Stabilizer rank decomposition strategy.
                "cat5", "bss" or "cutting": accepted and unused (one engine here; Tsim does not check it).
            seed: Random seed for the sampler. The same seed gives the same shots whatever
                the batch size.

        """
        self.circuit = _as_circuit(circuit)
        self._inner = _circuit.MeasurementSampler(self.circuit, seed_of(seed))

    def __repr__(self) -> str:
        return f"CompiledMeasurementSampler({self.circuit.num_measurements} measurements of {self.circuit.num_qubits} qubits, {self.circuit.num_non_clifford_gates()} non-Clifford gates)"

    @property
    def num_measurements(self) -> int:
        return self.circuit.num_measurements

    def sample(self, shots: int, *, batch_size: Union[int, None] = None, bit_packed: bool = False, threads: int = 1) -> np.ndarray:
        """Sample measurement outcomes from the circuit.

        Args:
            shots: The number of times to sample every measurement in the circuit.
            batch_size: Tsim's batch size: checked (at least 1) and otherwise unused; this
                package's samplers draw shot by shot from their seed.
            bit_packed: (this package's) the records bit-packed, 8 to a byte, little-endian.
            threads: (this package's) worker threads; the shots do not depend on it.

        Returns:
            A numpy array containing the measurement samples.

        """
        shots = _check_shots(shots)
        _check_batch_size(batch_size)
        return self._inner.sample(shots, bit_packed=bool(bit_packed), threads=count(threads, "threads"))


class CompiledDetectorSampler:
    """Samples detector and observable outcomes from a quantum circuit, as Tsim's
    ``CompiledDetectorSampler``: by default each detector's and observable's value, the parity
    of its measurements (``X 0; M 0; DETECTOR rec[-1]`` reads True), not Stim's flip from a
    noiseless reference run; ``use_detector_reference_sample`` / ``use_observable_reference_sample``
    give the flips. The shots are this package's samplers': the frame sampler on a Clifford
    circuit, the stabilizer-rank sampler on a non-Clifford one."""

    def __init__(self, circuit: Any, *, strategy: str = "cat5", seed: Union[int, None] = None) -> None:
        """Create a detector sampler.

        Args:
            circuit: The quantum circuit to compile.
            strategy: Stabilizer rank decomposition strategy.
                "cat5", "bss" or "cutting": accepted and unused (one engine here; Tsim does not check it).
            seed: Random seed for the sampler. The same seed gives the same shots whatever
                the batch size.

        """
        self.circuit = _as_circuit(circuit)
        self._seed = seed_of(seed)
        self._inner = _circuit.Circuit.compile_detector_sampler(self.circuit, seed=self._seed)
        self._num_detectors = self.circuit.num_detectors
        self._num_observables = self.circuit.num_observables
        self._non_clifford = self.circuit.num_non_clifford_gates() > 0
        self._signs: Union[tuple, None] = None
        self._noiseless: Any = None
        self._noiseless_signs: Union[tuple, None] = None

    def __repr__(self) -> str:
        engine = "stabilizer-rank" if self._non_clifford else "frame"
        return f"CompiledDetectorSampler({self._num_detectors} detectors, {self._num_observables} observables, {engine} sampler)"

    @property
    def num_detectors(self) -> int:
        return self._num_detectors

    @property
    def num_observables(self) -> int:
        return self._num_observables

    def _reference_signs(self) -> tuple:
        if self._signs is None:
            self._signs = reference_signs(self.circuit)
        return self._signs

    def _reference_values(self) -> tuple:
        """One noiseless shot's detector and observable values, as Tsim draws its reference
        sample (with no error) for each ``sample`` call: random where a non-Clifford gate leaves
        a detector random."""
        if self._noiseless is None:
            ideal = _circuit.Circuit.without_noise(self.circuit)
            self._noiseless = _circuit.Circuit.compile_detector_sampler(ideal, seed=self._seed ^ 0x5DEECE66D)
            self._noiseless_signs = reference_signs(ideal)
        d, o = self._noiseless.sample(1, separate_observables=True)
        sd, so = self._noiseless_signs
        return d[0] ^ sd, o[0] ^ so

    def sample(
        self,
        shots: int,
        *,
        batch_size: Union[int, None] = None,
        prepend_observables: bool = False,
        append_observables: bool = False,
        separate_observables: bool = False,
        bit_packed: bool = False,
        use_detector_reference_sample: bool = False,
        use_observable_reference_sample: bool = False,
        postselection_mask: Any = None,
        threads: int = 1,
    ) -> Any:
        """Return detector samples from the circuit.

        The circuit must define the detectors using DETECTOR instructions. Observables
        defined by OBSERVABLE_INCLUDE instructions can also be included in the results
        as honorary detectors.

        Args:
            shots: The number of times to sample every detector in the circuit.
            batch_size: Tsim's batch size: checked (at least 1) and otherwise unused; this
                package's samplers draw shot by shot from their seed.
            separate_observables: Defaults to False. When set to True, the return value
                is a (detection_events, observable_flips) tuple instead of a flat
                detection_events array.
            prepend_observables: Defaults to false. When set, observables are included
                with the detectors and are placed at the start of the results.
            append_observables: Defaults to false. When set, observables are included
                with the detectors and are placed at the end of the results.
            bit_packed: Defaults to false. When set, results are bit-packed.
            use_detector_reference_sample: Defaults to False. When True, a noiseless
                reference sample is computed and XORed with detector outcomes so that
                results represent deviations from the noiseless baseline (Stim's detection
                events). This should only be used when detectors are deterministic: as in
                Tsim, the reference is one noiseless shot drawn for each call, so a detector
                a non-Clifford gate leaves random changes with it. On a Clifford circuit the
                result is Stim's events exactly (its reference run's).
            use_observable_reference_sample: Defaults to False. The same for the
                observables.
            postselection_mask: Optional boolean array of length ``num_detectors``, checked
                as Tsim checks it. Tsim uses it to skip sampling shots a masked detector
                discards (returning those rows with their other columns False); here every
                row is a full sample, so re-applying the mask to the returned detector
                columns gives the surviving shots, as in Tsim.
            threads: (this package's) worker threads; the shots do not depend on it.

        Returns:
            A numpy array or tuple of numpy arrays containing the samples.

        Raises:
            ValueError: If ``separate_observables`` is combined with
                ``prepend_observables`` or ``append_observables``.

        """
        if separate_observables and (prepend_observables or append_observables):
            raise ValueError("Can't specify separate_observables=True with append_observables=True or prepend_observables=True")
        shots = _check_shots(shots)
        _check_batch_size(batch_size)
        if postselection_mask is not None:
            mask = np.asarray(postselection_mask, dtype=np.bool_)
            if mask.shape != (self._num_detectors,):
                raise ValueError(f"postselection_mask must have shape ({self._num_detectors},), got {mask.shape}")

        det_samples, obs_samples = self._inner.sample(shots, separate_observables=True, threads=count(threads, "threads"))
        # The samplers' flips from the reference run, made values (or flips from Tsim's reference).
        if not (use_detector_reference_sample and use_observable_reference_sample):
            sd, so = self._reference_signs()
            if not use_detector_reference_sample:
                det_samples = det_samples ^ sd
            if not use_observable_reference_sample:
                obs_samples = obs_samples ^ so
        if self._non_clifford and (use_detector_reference_sample or use_observable_reference_sample):
            rd, ro = self._reference_values()
            sd, so = self._reference_signs()
            if use_detector_reference_sample:
                det_samples = det_samples ^ (rd ^ sd)
            if use_observable_reference_sample:
                obs_samples = obs_samples ^ (ro ^ so)

        if prepend_observables and append_observables:
            combined = np.concatenate([obs_samples, det_samples, obs_samples], axis=1)
            return _maybe_bit_pack(combined, bit_packed=bit_packed)
        if append_observables:
            combined = np.concatenate([det_samples, obs_samples], axis=1)
            return _maybe_bit_pack(combined, bit_packed=bit_packed)
        if prepend_observables:
            combined = np.concatenate([obs_samples, det_samples], axis=1)
            return _maybe_bit_pack(combined, bit_packed=bit_packed)
        if separate_observables:
            return (
                _maybe_bit_pack(det_samples, bit_packed=bit_packed),
                _maybe_bit_pack(obs_samples, bit_packed=bit_packed),
            )

        return _maybe_bit_pack(det_samples, bit_packed=bit_packed)


class CompiledStateProbs:
    """Computes measurement probabilities for a given state, as Tsim's ``CompiledStateProbs``:
    for each of a batch of noise draws f, the exact probability P(state | f) of an outcome (a
    measurement record, or with ``sample_detectors`` the detectors' and observables' values),
    from this package's stabilizer-rank engine. Their mean estimates P(state).

    With ``sample_detectors``, P(state | f) sums the probabilities of the measurement records
    whose parities are the state: at most 2^16 of them (a state's records are 2^k, k the
    measurements no detector or observable pins down, a reset-free circuit's unread
    measurements among them). A reset of a qubit left in superposition (``H 0`` then ``R 0``)
    draws its unread outcome along with the noise, so a draw's probability is given that outcome
    too; the mean is unchanged."""

    def __init__(self, circuit: Any, *, sample_detectors: bool = False, strategy: str = "cat5", seed: Union[int, None] = None) -> None:
        """Create a probability estimator.

        Args:
            circuit: The quantum circuit to compile.
            sample_detectors: If True, compute detector/observable probabilities.
            strategy: Stabilizer rank decomposition strategy.
                "cat5", "bss" or "cutting": accepted and unused (one engine here; Tsim does not check it).
            seed: Random seed. The same seed gives the same draws.

        """
        self.circuit = _as_circuit(circuit)
        self._sample_detectors = bool(sample_detectors)
        self._rng = np.random.default_rng(seed_of(seed))
        self._sampler = _circuit.StabilizerRankSampler(self.circuit, 0)
        self._noisy = _circuit.Circuit.without_noise(self.circuit)._stim_exact_text() != self.circuit._stim_exact_text()
        if self._sample_detectors:
            self._h = parity_matrix(self.circuit)
            self._num_outputs = self._h.shape[0]
        else:
            self._h = None
            self._num_outputs = self.circuit.num_measurements

    @property
    def num_outputs(self) -> int:
        """The state's length: the measurements, or the detectors then the observables."""
        return self._num_outputs

    def _records(self, state: np.ndarray) -> list:
        if not self._sample_detectors:
            return [[bool(b) for b in state]]
        solved = _affine_solutions(self._h, state.astype(np.uint8))
        if solved is None:
            return []
        x0, basis = solved
        k = basis.shape[0]
        if (1 << k) > MAX_RECORDS_PER_STATE:
            raise ValueError(f"the detector state is the parity of 2^{k} measurement records; CompiledStateProbs(sample_detectors=True) sums at most {MAX_RECORDS_PER_STATE} of them")
        coeffs = ((np.arange(1 << k)[:, None] >> np.arange(k)) & 1).astype(np.uint8)
        records = (x0[None, :] + coeffs @ basis) % 2
        return [[bool(b) for b in row] for row in records]

    def probability_of(self, state: Any, *, batch_size: int) -> np.ndarray:
        """Compute probabilities for a batch of error samples given a measurement state.

        Args:
            state: The measurement outcome state to compute probability for: a bool array of
                shape (num_measurements,), or (num_detectors + num_observables,) with
                ``sample_detectors``.
            batch_size: Number of error samples to use for estimation.

        Returns:
            Array of probabilities P(state | error_sample) for each error sample.

        """
        if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)):
            raise TypeError(f"batch_size must be an integer, not {type(batch_size).__name__}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}")
        state = np.asarray(state)
        if state.shape != (self._num_outputs,):
            raise ValueError(f"state must have shape ({self._num_outputs},), got {state.shape}")
        state = state != 0 if state.dtype != np.bool_ else state
        records = self._records(state)
        batch_size = int(batch_size)
        if not records:
            return np.zeros(batch_size)
        seeds = self._rng.integers(0, 1 << 62, size=batch_size if self._noisy else 1)
        probs = np.array([sum(call(self._sampler._s.probability_of, records, 1, int(s))) for s in seeds], dtype=np.float64)
        return probs if self._noisy else np.full(batch_size, probs[0])
