# Ported from Qualtran (quantumlib/Qualtran 0.7.0, qualtran/surface_code/*.py and the GateCounts of
# qualtran/resource_counting/_bloq_counts.py; Copyright 2023 Google LLC, Apache License 2.0; see LICENSE in this
# directory), modified for stabilizer_qec: dataclasses in place of attrs, plain ints and floats in place of sympy,
# a numpy density matrix in place of Cirq for the 15-to-1 factory, the Bloq-based `AlgorithmSummary.from_bloq`
# left out.
"""Qualtran's surface-code cost models: a QEC scheme (logical error rate from distance), physical
parameters, data blocks (Gidney–Fowler's simple block; Litinski's compact, intermediate and fast
blocks), magic-state factories (Gidney–Fowler's catalyzed CCZ-to-2T factory, Litinski's 15-to-1
with its density-matrix error model, several factories at once), rotation cost models, the
`PhysicalCostModel` that combines them, and the Gidney–Fowler spreadsheet's and Beverland et
al.'s estimates."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from functools import lru_cache, reduce
from typing import Callable, Iterable, Iterator, Optional

import numpy as np

# ---------------------------------------------------------------------------------------------
# Gate counts and algorithm summaries
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class GateCounts:
    """Counts of the gates a surface-code compilation pays for: T (and T†), Toffoli, CSWAP, And,
    Clifford, single-qubit rotations and measurements."""

    t: int = 0
    toffoli: int = 0
    cswap: int = 0
    and_bloq: int = 0
    clifford: int = 0
    rotation: int = 0
    measurement: int = 0

    def __add__(self, other: "GateCounts") -> "GateCounts":
        if not isinstance(other, GateCounts):
            raise TypeError(f"Can only add other `GateCounts` objects, not {other!r}")
        return GateCounts(self.t + other.t, self.toffoli + other.toffoli, self.cswap + other.cswap, self.and_bloq + other.and_bloq, self.clifford + other.clifford, self.rotation + other.rotation, self.measurement + other.measurement)

    def __mul__(self, other) -> "GateCounts":
        return GateCounts(other * self.t, other * self.toffoli, other * self.cswap, other * self.and_bloq, other * self.clifford, other * self.rotation, other * self.measurement)

    __rmul__ = __mul__

    def asdict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v}

    def __str__(self) -> str:
        parts = [f"{k}: {v}" for k, v in self.asdict().items()]
        return ", ".join(parts) if parts else "-"

    def total_t_count(self, ts_per_toffoli: int = 4, ts_per_cswap: int = 4, ts_per_and_bloq: int = 4, ts_per_rotation: int = 11) -> int:
        """Every gate as T gates (a rotation 11 by default: the mixed-fallback protocol at 10⁻³)."""
        return self.t + ts_per_toffoli * self.toffoli + ts_per_cswap * self.cswap + ts_per_and_bloq * self.and_bloq + ts_per_rotation * self.rotation

    def total_t_and_ccz_count(self, ts_per_rotation: int = 11) -> dict:
        return {"n_t": self.t + ts_per_rotation * self.rotation, "n_ccz": self.toffoli + self.cswap + self.and_bloq}

    def total_toffoli_only(self) -> int:
        if self.t:
            raise ValueError(f"{self} contains T counts.")
        if self.rotation:
            raise ValueError(f"{self} contains rotations.")
        return self.toffoli + self.cswap + self.and_bloq

    def total_beverland_count(self) -> dict:
        """Beverland et al.'s M_meas, M_R, M_T, M_Tof and D_R (the rotation layers, taken as the
        rotations)."""
        toffoli = self.toffoli + self.and_bloq + self.cswap
        return {"meas": self.measurement, "R": self.rotation, "T": self.t, "Tof": toffoli, "D_R": self.rotation}


@dataclass(frozen=True)
class AlgorithmSummary:
    """The logical costs of an algorithm: its qubits, its gate counts, and optionally the number
    of layers with a rotation in them (the rotations themselves by default)."""

    n_algo_qubits: int
    n_logical_gates: GateCounts
    n_rotation_layers: Optional[int] = None


# ---------------------------------------------------------------------------------------------
# QEC schemes and physical parameters
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class QECScheme:
    """The logical error rate ``a (p / p*)^((d + 1) / 2)`` of a distance-``d`` code at physical
    error ``p``: ``error_rate_scaler`` a, ``error_rate_threshold`` p*."""

    error_rate_scaler: float
    error_rate_threshold: float

    def logical_error_rate(self, code_distance: int, physical_error: float) -> float:
        return self.error_rate_scaler * math.pow(physical_error / self.error_rate_threshold, (code_distance + 1) / 2)

    def code_distance_from_budget(self, physical_error: float, budget: float) -> int:
        """The smallest odd distance (at least 3) whose logical error rate is within ``budget``."""
        r = math.log(budget / self.error_rate_scaler) / math.log(physical_error / self.error_rate_threshold)
        d = 2 * math.ceil(r) - 1
        return 3 if d < 3 else d

    def physical_qubits(self, code_distance: int) -> int:
        return 2 * code_distance**2

    @classmethod
    def make_gidney_fowler(cls) -> "QECScheme":
        return cls(error_rate_scaler=0.1, error_rate_threshold=0.01)

    @classmethod
    def make_beverland_et_al(cls) -> "QECScheme":
        return cls(error_rate_scaler=0.03, error_rate_threshold=0.01)


@dataclass(frozen=True)
class LogicalErrorModel:
    """A QEC scheme at one physical error rate: callable on a code distance."""

    physical_error: float
    qec_scheme: QECScheme

    def __call__(self, code_distance: int) -> float:
        return self.qec_scheme.logical_error_rate(code_distance=code_distance, physical_error=self.physical_error)


@dataclass(frozen=True)
class PhysicalParameters:
    physical_error: float = 1e-3
    cycle_time_us: float = 1.0

    @classmethod
    def make_beverland_et_al(cls, qubit_modality: str = "superconducting", optimistic_err_rate: bool = False) -> "PhysicalParameters":
        phys_err = 1e-4 if optimistic_err_rate else 1e-3
        if qubit_modality == "ion":
            t_gate_ns, t_meas_ns = 100_000, 100_000
        elif qubit_modality == "superconducting":
            t_gate_ns, t_meas_ns = 50, 100
        elif qubit_modality == "majorana":
            phys_err = 1e-6 if optimistic_err_rate else 1e-4
            t_gate_ns, t_meas_ns = 100, 100
        else:
            raise ValueError(f"Unknown qubit modality {qubit_modality}. Must be one of 'ion', 'superconducting', or 'majorana'.")
        cycle_time_ns = 4 * t_gate_ns + 2 * t_meas_ns
        return cls(physical_error=phys_err, cycle_time_us=cycle_time_ns / 1000.0)

    @classmethod
    def make_gidney_fowler(cls, optimistic_err_rate: bool = False) -> "PhysicalParameters":
        return cls(physical_error=1e-4 if optimistic_err_rate else 1e-3, cycle_time_us=1.0)


# ---------------------------------------------------------------------------------------------
# Data blocks
# ---------------------------------------------------------------------------------------------


class DataBlock:
    """How the algorithm's qubits are laid out in surface-code tiles: the tiles for a number of
    algorithm qubits and the steps (of ``data_d`` cycles) a magic state takes to consume."""

    data_d: int

    def n_tiles(self, n_algo_qubits: int) -> int:
        raise NotImplementedError

    @property
    def n_steps_to_consume_a_magic_state(self) -> int:
        raise NotImplementedError

    def n_cycles(self, n_logical_gates: GateCounts, logical_error_model: LogicalErrorModel) -> int:
        counts = n_logical_gates.total_t_and_ccz_count()
        return self.data_d * (counts["n_t"] + counts["n_ccz"]) * self.n_steps_to_consume_a_magic_state

    def n_physical_qubits(self, n_algo_qubits: int) -> int:
        return 2 * self.data_d**2 * self.n_tiles(n_algo_qubits)

    def data_error(self, n_algo_qubits: int, n_cycles: int, logical_error_model: LogicalErrorModel) -> float:
        return self.n_tiles(n_algo_qubits) * n_cycles * logical_error_model(self.data_d)


@dataclass(frozen=True)
class SimpleDataBlock(DataBlock):
    """One tile per algorithm qubit plus a routing overhead (Gidney–Fowler's spreadsheet)."""

    data_d: int
    routing_overhead: float = 0.5

    def n_tiles(self, n_algo_qubits: int) -> int:
        return math.ceil((1 + self.routing_overhead) * n_algo_qubits)

    @property
    def n_steps_to_consume_a_magic_state(self) -> int:
        return 0


@dataclass(frozen=True)
class CompactDataBlock(DataBlock):
    """Litinski's compact block: 1.5 tiles per qubit, 9 steps per magic state."""

    data_d: int

    def n_tiles(self, n_algo_qubits: int) -> int:
        return math.ceil(1.5 * n_algo_qubits)

    @property
    def n_steps_to_consume_a_magic_state(self) -> int:
        return 9


@dataclass(frozen=True)
class IntermediateDataBlock(DataBlock):
    """Litinski's intermediate block: 2 tiles per qubit, 5 steps per magic state."""

    data_d: int

    def n_tiles(self, n_algo_qubits: int) -> int:
        return math.ceil(2 * n_algo_qubits)

    @property
    def n_steps_to_consume_a_magic_state(self) -> int:
        return 5


@dataclass(frozen=True)
class FastDataBlock(DataBlock):
    """Litinski's fast block: ``2n + √(8n) + 1`` tiles, one step per magic state."""

    data_d: int

    @staticmethod
    def get_n_tiles(n_algo_qubits: int) -> int:
        return math.ceil(2 * n_algo_qubits + math.sqrt(8 * n_algo_qubits) + 1)

    def n_tiles(self, n_algo_qubits: int) -> int:
        return self.get_n_tiles(n_algo_qubits)

    @property
    def n_steps_to_consume_a_magic_state(self) -> int:
        return 1

    @classmethod
    def from_error_budget(cls, error_budget: float, n_algo_qubits: int, qec_scheme: QECScheme, phys_err_rate: float) -> "FastDataBlock":
        q = FastDataBlock.get_n_tiles(n_algo_qubits)
        return cls(data_d=qec_scheme.code_distance_from_budget(phys_err_rate, error_budget / q))


# ---------------------------------------------------------------------------------------------
# Magic-state factories
# ---------------------------------------------------------------------------------------------


class MagicStateFactory:
    """A factory's physical qubits, the cycles to make the magic states the gates need, and the
    error it leaves in them."""

    def n_physical_qubits(self) -> int:
        raise NotImplementedError

    def n_cycles(self, n_logical_gates: GateCounts, logical_error_model: LogicalErrorModel) -> int:
        raise NotImplementedError

    def factory_error(self, n_logical_gates: GateCounts, logical_error_model: LogicalErrorModel) -> float:
        raise NotImplementedError


@dataclass(frozen=True)
class CCZ2TFactory(MagicStateFactory):
    """Gidney and Fowler's two-level factory: six level-1 T factories at distance
    ``distillation_l1_d`` feeding a level-2 CCZ factory at ``distillation_l2_d``
    (arXiv:1812.01238)."""

    distillation_l1_d: int = 15
    distillation_l2_d: int = 31

    def l0_state_injection_error(self, error_model: LogicalErrorModel) -> float:
        return error_model.physical_error

    def l0_topo_error_t_gate(self, error_model: LogicalErrorModel) -> float:
        return 100 * error_model(self.distillation_l1_d // 2)

    def l0_error(self, error_model: LogicalErrorModel) -> float:
        return self.l0_state_injection_error(error_model) + self.l0_topo_error_t_gate(error_model)

    def l1_topo_error_factory(self, error_model: LogicalErrorModel) -> float:
        return 1000 * error_model(self.distillation_l1_d)

    def l1_topo_error_t_gate(self, error_model: LogicalErrorModel) -> float:
        return 100 * error_model(self.distillation_l1_d)

    def l1_distillation_error(self, error_model: LogicalErrorModel) -> float:
        return 35 * self.l0_error(error_model) ** 3

    def l1_error(self, error_model: LogicalErrorModel) -> float:
        return self.l1_topo_error_factory(error_model) + self.l1_topo_error_t_gate(error_model) + self.l1_distillation_error(error_model)

    def l2_error(self, error_model: LogicalErrorModel) -> float:
        return 1000 * error_model(self.distillation_l2_d) + 28 * self.l1_error(error_model) ** 2

    def n_physical_qubits(self) -> int:
        l1 = 4 * 8 * 2 * self.distillation_l1_d**2
        l2 = 4 * 8 * 2 * self.distillation_l2_d**2
        return 6 * l1 + l2

    def factory_error(self, n_logical_gates: GateCounts, logical_error_model: LogicalErrorModel) -> float:
        counts = n_logical_gates.total_t_and_ccz_count()
        return self.l2_error(logical_error_model) * (counts["n_ccz"] + math.ceil(counts["n_t"] / 2))

    def n_cycles(self, n_logical_gates: GateCounts, logical_error_model: LogicalErrorModel) -> int:
        distillation_d = max(2 * self.distillation_l1_d + 1, self.distillation_l2_d)
        counts = n_logical_gates.total_t_and_ccz_count()
        n_ccz_states = counts["n_ccz"] + math.ceil(counts["n_t"] / 2)
        catalyzations = math.ceil(counts["n_t"] / 2)
        return math.ceil((n_ccz_states * 5.5 + catalyzations) * distillation_d)


_PAULI = {"I": np.eye(2, dtype=complex), "X": np.array([[0, 1], [1, 0]], dtype=complex), "Z": np.array([[1, 0], [0, -1]], dtype=complex)}


def _pauli_matrix(pauli_string: str) -> np.ndarray:
    return reduce(np.kron, (_PAULI[c] for c in pauli_string))


def _rotation_unitary(pauli_string: str, theta: float) -> np.ndarray:
    """exp(i θ P)."""
    return math.cos(theta) * np.eye(1 << len(pauli_string)) + 1j * math.sin(theta) * _pauli_matrix(pauli_string)


class _Density:
    """A density matrix of n qubits with the two channels the 15-to-1 model uses."""

    def __init__(self, n: int) -> None:
        self.n = n
        plus = np.full((2,), 1 / math.sqrt(2), dtype=complex)
        state = reduce(np.kron, [plus] * n)
        self.rho = np.outer(state, state.conj())

    def noisy_rotation(self, pauli_string: str, p1: float, p2: float, p3: float) -> None:
        """A π/8 rotation about the Pauli that overshoots to 5π/8, −π/8 or 3π/8 with
        probabilities p1, p2, p3."""
        probabilities = [1 - p1 - p2 - p3, p1, p2, p3]
        out = np.zeros_like(self.rho)
        for p, k in zip(probabilities, (1, 5, -1, 3)):
            u = _rotation_unitary(pauli_string, k * math.pi / 8)
            out += p * (u @ self.rho @ u.conj().T)
        self.rho = out

    def storage_error(self, kind: str, probabilities) -> None:
        """An X or Z flip on each qubit with its probability."""
        if kind not in ("Z", "X"):
            raise ValueError(f"kind must be Z or X not {kind}")
        for q, p in enumerate(probabilities):
            if p == 0:
                continue
            letters = ["I"] * self.n
            letters[q] = kind
            e = _pauli_matrix("".join(letters))
            self.rho = (1 - p) * self.rho + p * (e @ self.rho @ e.conj().T)


@dataclass(frozen=True)
class FifteenToOne(MagicStateFactory):
    """Litinski's 15-to-1 T factory (arXiv:1905.06903): ``d_X`` and ``d_Z`` the patch sides,
    ``d_m`` the measurement rounds per surgery; its failure and output error from the paper's
    density-matrix error model of the eleven π/8 rotations."""

    d_X: int
    d_Z: int
    d_m: int

    def __post_init__(self) -> None:
        assert 0 < self.d_X <= 3 * self.d_m
        assert self.d_m > 0
        assert self.d_Z > 0

    def n_physical_qubits(self) -> int:
        return 2 * (self.d_X + 4 * self.d_Z) * 3 * self.d_X + 4 * self.d_m

    def _final_state(self, m: LogicalErrorModel) -> np.ndarray:
        return _fifteen_to_one_state(self.d_X, self.d_Z, self.d_m, m.physical_error, m(self.d_X), m(self.d_Z), m(self.d_m))

    def p_fail(self, logical_error_model: LogicalErrorModel) -> float:
        projector = np.kron(np.eye(2), np.ones((16, 16)) / 16)
        return float(np.real(1 - np.trace(projector @ self._final_state(logical_error_model))))

    def p_out(self, logical_error_model: LogicalErrorModel) -> float:
        projector = np.kron(np.eye(2), np.ones((16, 16)) / 16)
        rho = self._final_state(logical_error_model)
        projected = (projector @ rho @ projector.conj().T) / (1 - self.p_fail(logical_error_model))
        t_state = np.array([1, np.exp(-1j * np.pi / 4)]).reshape((1, 2)) / math.sqrt(2)
        target = np.kron(t_state.conj().T @ t_state, np.ones((16, 16)) / 16)
        return float(np.real(1 - np.trace(projected @ target)))

    def n_cycles(self, n_logical_gates: GateCounts, logical_error_model: LogicalErrorModel) -> int:
        """The expected cycles, failures included (page 11 of the paper)."""
        return int(math.ceil(n_logical_gates.total_t_count() * 6 * self.d_m / (1 - self.p_fail(logical_error_model))))

    def factory_error(self, n_logical_gates: GateCounts, logical_error_model: LogicalErrorModel) -> float:
        return self.p_out(logical_error_model) * n_logical_gates.total_t_count()


@lru_cache(maxsize=64)
def _fifteen_to_one_state(d_X: int, d_Z: int, d_m: int, phys_err: float, px: float, pz: float, pm: float) -> np.ndarray:
    """The 15-to-1 protocol's final density matrix under the paper's error model (its
    supplementary material's probabilities)."""
    s = _Density(5)
    e3 = phys_err / 3
    half_z = [0, 0.5 * (d_X / d_Z) * pz * d_m, 0.5 * (d_X / d_Z) * pz * d_m, 0.5 * (d_X / d_Z) * pz * d_m, 0]
    half_x = [0, 0.5 * (d_Z / d_X) * px * d_m, 0.5 * (d_Z / d_X) * px * d_m, 0.5 * (d_Z / d_X) * px * d_m, 0]
    full_x = [0.5 * px * d_m] + [0.5 * (d_Z / d_X) * px * d_m] * 4
    full_z = [0.5 * px * d_m] + [0.5 * (d_X / d_Z) * pz * d_m] * 4
    single = (e3 + 0.5 * (d_m / d_Z) * pz * d_m, e3 + 0.5 * d_Z * pm, e3)

    def multi(width: float):
        return (e3 + 0.5 * pm * d_m, e3 + 0.5 * pm * d_m + 0.5 * width * d_X / d_m * pm, e3)

    s.noisy_rotation("IZIII", *single)
    s.noisy_rotation("IIZII", *single)
    s.noisy_rotation("IIIZI", *single)
    s.noisy_rotation("IZZZI", *multi(3 * d_Z))
    s.storage_error("X", half_x)
    s.storage_error("Z", half_z)
    s.noisy_rotation("ZZZII", *multi(d_X + 2 * d_Z))
    s.noisy_rotation("ZZIZI", *multi(d_X + 3 * d_Z))
    s.storage_error("Z", [0.5 * ((d_X + 2 * d_Z) + (d_X + 3 * d_Z)) / d_X * px * d_m, 0, 0, 0, 0])
    s.storage_error("X", [0.5 * px * d_m, 0.5 * (d_Z / d_X) * px * d_m, 0.5 * (d_Z / d_X) * px * d_m, 0.5 * (d_Z / d_X) * px * d_m, 0])
    s.storage_error("Z", [0.5 * px * d_m, 0.5 * (d_X / d_Z) * pz * d_m, 0.5 * (d_X / d_Z) * pz * d_m, 0.5 * (d_X / d_Z) * pz * d_m, 0])
    s.noisy_rotation("ZIZZI", *multi(d_X + 3 * d_Z))
    s.noisy_rotation("ZIIZZ", *multi(d_X + 4 * d_Z))
    s.noisy_rotation("IIIIZ", *single)
    s.storage_error("Z", [0.5 * ((d_X + 3 * d_Z) + (d_X + 4 * d_Z)) / d_X * px * d_m, 0, 0, 0, 0])
    s.storage_error("X", full_x)
    s.storage_error("Z", full_z)
    s.noisy_rotation("ZZIIZ", *multi(d_X + 4 * d_Z))
    s.noisy_rotation("ZIZIZ", *multi(d_X + 4 * d_Z))
    s.storage_error("Z", [0.5 * ((d_X + 4 * d_Z) + (d_X + 4 * d_Z)) / d_X * px * d_m, 0, 0, 0, 0])
    s.storage_error("X", full_x)
    s.storage_error("Z", full_z)
    s.noisy_rotation("ZZZZZ", *multi(d_X + 4 * d_Z))
    s.noisy_rotation("IIZZZ", *multi(3 * d_Z))
    s.storage_error("Z", [0.5 * (d_X + 4 * d_Z) / d_X * px * d_m, 0, 0, 0, 0])
    s.storage_error("X", [0.5 * px * (d_m + 2 * d_X)] + [0.5 * (d_Z / d_X) * px * d_m] * 4)
    s.storage_error("Z", [0.5 * px * (d_m + 2 * d_X)] + [0.5 * (d_X / d_Z) * pz * d_m] * 4)
    s.noisy_rotation("IZIZZ", *multi(4 * d_Z))
    s.noisy_rotation("IZZIZ", *multi(4 * d_Z))
    s.storage_error("X", [0] + [0.5 * (d_Z / d_X) * px * d_m] * 4)
    s.storage_error("Z", [0] + [0.5 * (d_X / d_Z) * pz * d_m] * 4)
    return s.rho


@dataclass(frozen=True)
class MultiFactory(MagicStateFactory):
    """``n_factories`` copies of one factory: the qubits multiply, the cycles divide, the error
    stays."""

    base_factory: MagicStateFactory
    n_factories: int

    def n_physical_qubits(self) -> int:
        return self.base_factory.n_physical_qubits() * self.n_factories

    def n_cycles(self, n_logical_gates: GateCounts, logical_error_model: LogicalErrorModel) -> int:
        return math.ceil(self.base_factory.n_cycles(n_logical_gates, logical_error_model) / self.n_factories)

    def factory_error(self, n_logical_gates: GateCounts, logical_error_model: LogicalErrorModel) -> float:
        return self.base_factory.factory_error(n_logical_gates, logical_error_model)


# ---------------------------------------------------------------------------------------------
# Rotation cost models
# ---------------------------------------------------------------------------------------------


class RotationCostModel:
    def rotation_cost(self, error_budget: float) -> GateCounts:
        raise NotImplementedError

    def preparation_overhead(self, error_budget: float) -> GateCounts:
        raise NotImplementedError


@dataclass(frozen=True)
class RotationLogarithmicModel(RotationCostModel):
    """``#T = −slope · log₂(budget) + overhead`` per rotation, no preparation."""

    slope: float
    overhead: float
    gateset: Optional[str] = None

    def rotation_cost(self, error_budget: float) -> GateCounts:
        return GateCounts(t=math.ceil(-self.slope * math.log2(error_budget) + self.overhead))

    def preparation_overhead(self, error_budget: float) -> GateCounts:
        return GateCounts()


@dataclass(frozen=True)
class ConstantWithOverheadRotationCost(RotationCostModel):
    """``bitsize − 2`` Toffolis per rotation with a prepared phase-gradient state whose own
    rotations cost ``overhead_rotation_cost``."""

    bitsize: int
    overhead_rotation_cost: RotationCostModel

    def rotation_cost(self, error_budget: float) -> GateCounts:
        return GateCounts(toffoli=max(self.bitsize - 2, 0))

    def preparation_overhead(self, error_budget: float) -> GateCounts:
        return self.bitsize * self.overhead_rotation_cost.rotation_cost(error_budget / self.bitsize)


BeverlandEtAlRotationCost = RotationLogarithmicModel(slope=0.53, overhead=5.3, gateset="Clifford+T")
SevenDigitsOfPrecisionConstantCost = ConstantWithOverheadRotationCost(bitsize=7, overhead_rotation_cost=BeverlandEtAlRotationCost)


# ---------------------------------------------------------------------------------------------
# The physical cost model
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PhysicalCostsSummary:
    failure_prob: float
    footprint: int
    duration_hr: float

    @property
    def qubit_hours(self) -> float:
        return self.footprint * self.duration_hr


@dataclass(frozen=True)
class PhysicalCostModel:
    """Physical costs from an algorithm's counts: the cycles are the larger of the factory's
    generation and the data block's consumption, the qubits the sum of both, the error the sum
    of the factory's and the data block's over those cycles."""

    physical_params: PhysicalParameters
    data_block: DataBlock
    factory: MagicStateFactory
    qec_scheme: QECScheme

    @property
    def logical_error_model(self) -> LogicalErrorModel:
        return LogicalErrorModel(physical_error=self.physical_params.physical_error, qec_scheme=self.qec_scheme)

    def _base(self, algo: AlgorithmSummary) -> tuple[int, int, float]:
        model = self.logical_error_model
        n_cycles = int(max(self.factory.n_cycles(algo.n_logical_gates, model), self.data_block.n_cycles(algo.n_logical_gates, model)))
        n_phys = self.factory.n_physical_qubits() + self.data_block.n_physical_qubits(algo.n_algo_qubits)
        error = self.factory.factory_error(algo.n_logical_gates, model) + self.data_block.data_error(algo.n_algo_qubits, n_cycles, model)
        return n_cycles, n_phys, error

    def n_cycles(self, algo_summary: AlgorithmSummary) -> int:
        return self._base(algo_summary)[0]

    def duration_hr(self, algo_summary: AlgorithmSummary) -> float:
        return self.physical_params.cycle_time_us * self.n_cycles(algo_summary) / (1_000_000 * 60 * 60)

    def n_phys_qubits(self, algo_summary: AlgorithmSummary) -> int:
        return self._base(algo_summary)[1]

    def error(self, algo_summary: AlgorithmSummary) -> float:
        return self._base(algo_summary)[2]

    def summary(self, algo_summary: AlgorithmSummary) -> PhysicalCostsSummary:
        n_cycles, n_phys, error = self._base(algo_summary)
        return PhysicalCostsSummary(failure_prob=error, footprint=n_phys, duration_hr=self.physical_params.cycle_time_us * n_cycles / (1_000_000 * 60 * 60))

    @classmethod
    def make_gidney_fowler(cls, data_d: int) -> "PhysicalCostModel":
        return cls(PhysicalParameters.make_gidney_fowler(), SimpleDataBlock(data_d=data_d), CCZ2TFactory(), QECScheme.make_gidney_fowler())

    @classmethod
    def make_beverland_et_al(cls, data_d: int, data_block_name: str = "compact", factory_ds: tuple = (9, 3, 3)) -> "PhysicalCostModel":
        blocks = {"fast": FastDataBlock, "compact": CompactDataBlock, "intermediate": IntermediateDataBlock}
        if data_block_name not in blocks:
            raise ValueError(f"Unknown data block '{data_block_name}'")
        return cls(PhysicalParameters.make_beverland_et_al(), blocks[data_block_name](data_d=data_d), FifteenToOne(*factory_ds), QECScheme.make_beverland_et_al())


# ---------------------------------------------------------------------------------------------
# Gidney–Fowler estimates
# ---------------------------------------------------------------------------------------------


def get_ccz2t_costs(*, n_logical_gates: GateCounts, n_algo_qubits: int, phys_err: float, cycle_time_us: float, factory: MagicStateFactory, data_block: DataBlock) -> PhysicalCostsSummary:
    """Gidney–Fowler's spacetime cost and failure probability for the given factory and data block."""
    model = PhysicalCostModel(PhysicalParameters(physical_error=phys_err, cycle_time_us=cycle_time_us), data_block, factory, QECScheme.make_gidney_fowler())
    return model.summary(AlgorithmSummary(n_algo_qubits=n_algo_qubits, n_logical_gates=n_logical_gates))


def get_ccz2t_costs_from_error_budget(*, n_logical_gates: GateCounts, n_algo_qubits: int, phys_err: float = 1e-3, error_budget: float = 1e-2, cycle_time_us: float = 1.0, routing_overhead: float = 0.5, factory: Optional[MagicStateFactory] = None, data_block: Optional[DataBlock] = None) -> PhysicalCostsSummary:
    """The catalyzed-CCZ paper's prescription: the factory's error is fixed by its parameters,
    the data distance is chosen from the budget that remains."""
    if factory is None:
        factory = CCZ2TFactory()
    qec_scheme = QECScheme.make_gidney_fowler()
    err_model = LogicalErrorModel(physical_error=phys_err, qec_scheme=qec_scheme)
    factory_error = factory.factory_error(n_logical_gates, err_model)
    n_cycles = factory.n_cycles(n_logical_gates, err_model)
    if data_block is None:
        err_budget = error_budget - factory_error
        if err_budget < 0:
            raise ValueError(f"Factory error {factory_error} is larger than the error budget {error_budget}")
        n_logical_qubits = math.ceil((1 + routing_overhead) * n_algo_qubits)
        target_err_per_round = err_budget / (n_logical_qubits * n_cycles)
        data_d = qec_scheme.code_distance_from_budget(physical_error=phys_err, budget=target_err_per_round)
        data_block = SimpleDataBlock(data_d=data_d, routing_overhead=routing_overhead)
    return get_ccz2t_costs(n_logical_gates=n_logical_gates, n_algo_qubits=n_algo_qubits, phys_err=phys_err, cycle_time_us=cycle_time_us, factory=factory, data_block=data_block)


def iter_ccz2t_factories(l1_start: int = 5, l1_stop: int = 25, l2_stop: int = 41, *, n_factories: int = 1) -> Iterator[MagicStateFactory]:
    if n_factories < 1:
        raise ValueError("The number of factories should be a positive integer")
    for l1 in range(l1_start, l1_stop, 2):
        for l2 in range(l1 + 2, l2_stop, 2):
            base = CCZ2TFactory(distillation_l1_d=l1, distillation_l2_d=l2)
            yield base if n_factories == 1 else MultiFactory(base_factory=base, n_factories=n_factories)


def iter_simple_data_blocks(d_start: int = 7, d_stop: int = 35) -> Iterator[SimpleDataBlock]:
    for d in range(d_start, d_stop, 2):
        yield SimpleDataBlock(data_d=d)


def get_ccz2t_costs_from_grid_search(*, n_logical_gates: GateCounts, n_algo_qubits: int, phys_err: float = 1e-3, error_budget: float = 1e-2, cycle_time_us: float = 1.0, factory_iter: Optional[Iterable[MagicStateFactory]] = None, data_block_iter: Optional[Iterable[DataBlock]] = None, cost_function: Callable[[PhysicalCostsSummary], float] = lambda pc: pc.qubit_hours) -> tuple[PhysicalCostsSummary, MagicStateFactory, DataBlock]:
    """The factory and data block that minimise ``cost_function`` within the error budget."""
    factories = tuple(iter_ccz2t_factories()) if factory_iter is None else tuple(factory_iter)
    blocks = tuple(iter_simple_data_blocks()) if data_block_iter is None else tuple(data_block_iter)
    best: Optional[PhysicalCostsSummary] = None
    best_params = None
    for factory in factories:
        for data_block in blocks:
            cost = get_ccz2t_costs(n_logical_gates=n_logical_gates, n_algo_qubits=n_algo_qubits, phys_err=phys_err, cycle_time_us=cycle_time_us, factory=factory, data_block=data_block)
            if cost.failure_prob > error_budget:
                continue
            if best is None or cost_function(cost) < cost_function(best):
                best, best_params = cost, (factory, data_block)
    if best is None or best_params is None:
        raise ValueError("No valid factories found!")
    return best, best_params[0], best_params[1]


# ---------------------------------------------------------------------------------------------
# Beverland et al.
# ---------------------------------------------------------------------------------------------


def minimum_time_steps(*, error_budget: float, alg: AlgorithmSummary, rotation_model: RotationCostModel) -> int:
    """Beverland et al.'s equation D3: M_meas + M_R + M_T + 3 M_Tof + D_R × the T cost of a
    rotation at ε/(3 M_R)."""
    M = alg.n_logical_gates.total_beverland_count()
    c_min = M["meas"] + M["R"] + M["T"] + 3 * M["Tof"]
    eps_syn = error_budget / 3
    if M["R"] > 0:
        rot_err_budget = eps_syn / M["R"]
        rotation_cost = rotation_model.rotation_cost(rot_err_budget) + rotation_model.preparation_overhead(rot_err_budget)
        d_r = alg.n_rotation_layers if alg.n_rotation_layers is not None else M["D_R"]
        c_min += math.ceil(d_r * (rotation_cost.t + 4 * rotation_cost.toffoli))
    return c_min


def code_distance(*, error_budget: float, time_steps: float, alg: AlgorithmSummary, qec_scheme: QECScheme, physical_error: float) -> int:
    """The distance at which Q·C·P(d) = ε/3 on the fast data block's Q tiles."""
    q = FastDataBlock.get_n_tiles(alg.n_algo_qubits)
    return qec_scheme.code_distance_from_budget(physical_error, error_budget / (3 * q * time_steps))


def n_discrete_logical_gates(*, eps_syn: float, alg: AlgorithmSummary, rotation_model: RotationCostModel) -> GateCounts:
    """The counts with every rotation synthesised into T and Toffoli gates."""
    n_rotations = alg.n_logical_gates.rotation
    ret = replace(alg.n_logical_gates, rotation=0)
    if n_rotations > 0:
        ret = ret + rotation_model.preparation_overhead(eps_syn) + n_rotations * rotation_model.rotation_cost(eps_syn / n_rotations)
    return ret


def t_states(*, error_budget: float, alg: AlgorithmSummary, rotation_model: RotationCostModel) -> int:
    """Beverland et al.'s D4: the T states the algorithm consumes, rotations synthesised at ε/3."""
    return n_discrete_logical_gates(eps_syn=error_budget / 3, alg=alg, rotation_model=rotation_model).total_t_count()
