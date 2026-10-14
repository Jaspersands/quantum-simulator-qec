"""Subset sampling of a noisy circuit: the logical error rate as a sum over how many faults
occur, each count's failure rate sampled with exactly that many faults (Heußen, Winter et al.'s
"subset sampling", as in qsample, on this package's circuits, flip simulator and decoders).

The circuit's noise channels are its fault locations, grouped by their fault probability. A
subset is a number of faults in each group; its probability is a product of binomials,
``A_w = Π C(N_i, w_i) p_i^w_i (1 - p_i)^(N_i - w_i)``, and its failure rate ``f_w`` is sampled
by running the circuit with exactly those faults, at locations and Paulis drawn as the channels
draw them, and decoding. Then ``p_L = Σ_w A_w f_w`` over the subsets sampled, with a lower
bound (the subsets never sampled never fail) and an upper bound (they always fail). Since only
``A_w`` depends on the noise strength, one run gives the whole curve: ``estimate(scale)``
multiplies every probability by ``scale``. At low noise this needs far fewer shots than
sampling directly, which mostly samples the fault-free subset.

    s = SubsetSampler(circuit)        # decoded by matching on the circuit's model
    s.run(shots=20_000, max_weight=6)
    s.estimate()                      # SubsetEstimate(value, stddev, lower, upper)
    s.estimate(scale=0.1)             # the same circuit at a tenth of the noise
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

__all__ = ["FaultLocation", "SubsetEstimate", "SubsetSampler"]

_PAULIS_2 = [a + b for a in "IXYZ" for b in "IXYZ"][1:]
_MEASURE_FLIP = {"M": "X", "MR": "X", "MX": "Z", "MRX": "Z", "MY": "X", "MRY": "X"}


@dataclass
class FaultLocation:
    """Where a fault can happen: the qubits, the Paulis the fault may be (``"X"``, ``"XZ"``
    for two qubits, ...) with their probabilities given a fault, and the fault's probability."""

    instruction: int
    qubits: Tuple[int, ...]
    paulis: Tuple[str, ...]
    weights: Tuple[float, ...]
    probability: float


class SubsetEstimate(NamedTuple):
    """The logical error rate from the subsets sampled (``value``), its statistical standard
    deviation, and bounds for the subsets not sampled: ``lower`` (they never fail, = ``value``)
    and ``upper`` (they always fail)."""

    value: float
    stddev: float
    lower: float
    upper: float


def _locations(circuit: Any) -> Tuple[List[Tuple[str, Any]], List[FaultLocation]]:
    """The flattened circuit as noiseless steps and fault locations."""
    from stabilizer_qec import Circuit, CircuitInstruction

    steps: List[Tuple[str, Any]] = []
    locations: List[FaultLocation] = []
    pending = Circuit()

    def flush() -> None:
        nonlocal pending
        if len(pending):
            steps.append(("ops", pending))
            pending = Circuit()

    for k, instr in enumerate(circuit.flattened()):
        name, args = instr.name, instr.gate_args_copy()
        qubits = [t.value for t in instr.targets_copy() if t.is_qubit_target]
        found: List[FaultLocation] = []
        if name in ("X_ERROR", "Y_ERROR", "Z_ERROR"):
            found = [FaultLocation(k, (q,), (name[0],), (1.0,), args[0]) for q in qubits]
        elif name == "DEPOLARIZE1":
            found = [FaultLocation(k, (q,), ("X", "Y", "Z"), (1 / 3,) * 3, args[0]) for q in qubits]
        elif name == "DEPOLARIZE2":
            found = [FaultLocation(k, (a, b), tuple(_PAULIS_2), (1 / 15,) * 15, args[0]) for a, b in zip(qubits[::2], qubits[1::2])]
        elif name == "PAULI_CHANNEL_1":
            total = sum(args)
            found = [FaultLocation(k, (q,), ("X", "Y", "Z"), tuple(a / total for a in args), total) for q in qubits] if total > 0 else []
        elif name == "PAULI_CHANNEL_2":
            total = sum(args)
            found = [FaultLocation(k, (a, b), tuple(_PAULIS_2), tuple(x / total for x in args), total) for a, b in zip(qubits[::2], qubits[1::2])] if total > 0 else []
        elif name in ("E", "CORRELATED_ERROR"):
            targets = instr.targets_copy()
            paulis = "".join("X" if t.is_x_target else "Y" if t.is_y_target else "Z" for t in targets)
            found = [FaultLocation(k, tuple(t.value for t in targets), (paulis,), (1.0,), args[0])]
        elif name in ("ELSE_CORRELATED_ERROR", "HERALDED_ERASE", "HERALDED_PAULI_CHANNEL_1") or (instr.name == "MPP" and args and args[0] > 0):
            raise ValueError(f"subset sampling does not take {name}: its faults are not independent locations")
        elif name in _MEASURE_FLIP and args and args[0] > 0:
            # A noisy measurement: a flip before it, in its basis.
            flush()
            steps.append(("faults", [len(locations) + i for i in range(len(qubits))]))
            locations.extend(FaultLocation(k, (q,), (_MEASURE_FLIP[name],), (1.0,), args[0]) for q in qubits)
            pending.append(CircuitInstruction(name, instr.targets_copy(), []))
            continue
        if found:
            found = [f for f in found if f.probability > 0]
            if found:
                flush()
                steps.append(("faults", list(range(len(locations), len(locations) + len(found)))))
                locations.extend(found)
            continue
        pending.append(instr)
    flush()
    return steps, locations


def _binomial(n: int, k: int, p: float) -> float:
    if k > n or p < 0:
        return 0.0
    if p == 0:
        return 1.0 if k == 0 else 0.0
    if p >= 1:
        return 1.0 if k == n else 0.0
    return math.exp(math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) + k * math.log(p) + (n - k) * math.log1p(-p))


class SubsetSampler:
    """Subset sampling of ``circuit``'s logical error rate (see the module docstring).
    ``decoder`` has ``decode_batch`` (default: ``Matching`` on the circuit's model)."""

    def __init__(self, circuit: Any, decoder: Any = None, *, seed: Optional[int] = None) -> None:
        from stabilizer_qec import Circuit, Matching

        self.circuit = Circuit(circuit)
        self.steps, self.locations = _locations(self.circuit)
        groups: Dict[float, List[int]] = {}
        for i, loc in enumerate(self.locations):
            groups.setdefault(loc.probability, []).append(i)
        self.probabilities: Tuple[float, ...] = tuple(sorted(groups))
        self.groups: Tuple[Tuple[int, ...], ...] = tuple(tuple(groups[p]) for p in self.probabilities)
        self.decoder = decoder if decoder is not None else Matching(self.circuit.detector_error_model(decompose_errors=True))
        self._rng = np.random.default_rng(seed)
        self._seed = seed
        self.counts: Dict[Tuple[int, ...], List[int]] = {}  # subset -> [failures, shots]

    @property
    def num_locations(self) -> Tuple[int, ...]:
        """How many fault locations each group (by probability) has."""
        return tuple(len(g) for g in self.groups)

    def subset_probability(self, subset: Sequence[int], scale: float = 1.0) -> float:
        """``A_w``: the probability that exactly ``subset[i]`` faults occur in group ``i``."""
        return math.prod(_binomial(n, w, p * scale) for n, w, p in zip(self.num_locations, subset, self.probabilities))

    def subsets(self, max_weight: int) -> List[Tuple[int, ...]]:
        """Every subset of at most ``max_weight`` faults in all, fewest first."""
        out = [w for w in product(*(range(min(n, max_weight) + 1) for n in self.num_locations)) if sum(w) <= max_weight]
        return sorted(out, key=lambda w: (sum(w), w))

    def run(self, shots: int, *, max_weight: Optional[int] = None, scale: float = 1.0, batch_size: int = 1024, min_shots_per_subset: int = 100) -> None:
        """Samples ``shots`` shots over the subsets of 1 to ``max_weight`` faults, shared
        among them in proportion to their probability at ``scale`` (at least
        ``min_shots_per_subset`` each). The fault-free subset never fails a valid circuit,
        and is checked, not sampled. With ``max_weight`` None, enough subsets to leave at most
        1e-9 of the probability unsampled at ``scale`` (at most 12 faults)."""
        if max_weight is None:
            max_weight = 1
            while max_weight < 12 and 1 - sum(self.subset_probability(w, scale) for w in self.subsets(max_weight)) > 1e-9:
                max_weight += 1
        candidates = [w for w in self.subsets(max_weight) if sum(w) > 0]
        weights = np.array([self.subset_probability(w, scale) for w in candidates])
        if not candidates or weights.sum() == 0:
            raise ValueError("the circuit has no faults to sample")
        allocation = np.maximum(np.round(shots * weights / weights.sum()).astype(int), min_shots_per_subset)
        self._check_fault_free()
        for w, n in zip(candidates, allocation):
            for start in range(0, int(n), batch_size):
                fails = self._sample(w, min(batch_size, int(n) - start))
                entry = self.counts.setdefault(w, [0, 0])
                entry[0] += fails
                entry[1] += min(batch_size, int(n) - start)

    def _check_fault_free(self) -> None:
        if (0,) * len(self.groups) in self.counts:
            return
        fails = self._sample((0,) * len(self.groups), 64)
        if fails:
            raise ValueError("the circuit fails without faults: its detectors or observables are not deterministic, or the decoder errs on no detection events")
        self.counts[(0,) * len(self.groups)] = [0, 64]

    def _sample(self, subset: Sequence[int], shots: int) -> int:
        from stabilizer_qec import FlipSimulator

        # The faults of each shot: location -> (shots, Pauli index).
        chosen: Dict[int, List[Tuple[int, int]]] = {}
        for g, w in zip(self.groups, subset):
            if w == 0:
                continue
            for shot in range(shots):
                for loc in self._rng.choice(len(g), size=w, replace=False):
                    index = g[int(loc)]
                    weights = self.locations[index].weights
                    pauli = int(self._rng.choice(len(weights), p=weights)) if len(weights) > 1 else 0
                    chosen.setdefault(index, []).append((shot, pauli))
        sim = FlipSimulator(batch_size=shots, num_qubits=self.circuit.num_qubits, seed=int(self._rng.integers(2**62)))
        n = max(self.circuit.num_qubits, 1)
        for kind, step in self.steps:
            if kind == "ops":
                sim.do(step)
                continue
            xs = np.zeros((n, shots), dtype=bool)
            zs = np.zeros((n, shots), dtype=bool)
            any_fault = False
            for index in step:
                for shot, k in chosen.get(index, ()):
                    loc = self.locations[index]
                    for q, p in zip(loc.qubits, loc.paulis[k]):
                        if p in "XY":
                            xs[q, shot] ^= True
                        if p in "ZY":
                            zs[q, shot] ^= True
                    any_fault = True
            if any_fault:
                if xs.any():
                    sim.broadcast_pauli_errors(pauli="X", mask=xs)
                if zs.any():
                    sim.broadcast_pauli_errors(pauli="Z", mask=zs)
        dets = sim.get_detector_flips().T
        obs = sim.get_observable_flips().T
        predictions = np.asarray(self.decoder.decode_batch(dets)).astype(bool)
        return int(np.count_nonzero(np.any(predictions != obs, axis=1)))

    def estimate(self, scale: float = 1.0) -> SubsetEstimate:
        """The logical error rate with every fault probability multiplied by ``scale``."""
        value = variance = covered = 0.0
        for w, (fails, n) in self.counts.items():
            a = self.subset_probability(w, scale)
            covered += a
            if n:
                f = fails / n
                value += a * f
                # Wilson's variance (as qsample), which a failure-free subset leaves above zero.
                variance += a * a * (n * f * (1 - f) + 0.25) / (n + 1) ** 2 if sum(w) else 0.0
        return SubsetEstimate(value, math.sqrt(variance), value, value + max(0.0, 1.0 - covered))

    def failure_rates(self) -> Dict[Tuple[int, ...], Tuple[int, int]]:
        """Each sampled subset's (failures, shots)."""
        return {w: (f, n) for w, (f, n) in sorted(self.counts.items())}
