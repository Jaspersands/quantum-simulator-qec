# Ported from Qualtran (quantumlib/Qualtran 0.7.0, qualtran/surface_code/*.py, the GateCounts of
# qualtran/resource_counting/_bloq_counts.py and the TComplexity of qualtran/cirq_interop/t_complexity_protocol.py;
# Copyright 2023 Google LLC, Apache License 2.0; see LICENSE in the parent directory), modified for stabilizer_qec:
# dataclasses in place of attrs, plain ints and floats in place of sympy, numpy matrices and a numpy density matrix
# in place of Cirq's channels and simulator for the 15-to-1 factory; `AlgorithmSummary.from_bloq` counts a Bloq
# with Qualtran's own resource counting, so it needs Qualtran installed.
"""The implementation of `stabilizer_qec.estimate.qualtran`: every class and function of Qualtran's
``qualtran.surface_code`` (but its ``ui``), which the package and its submodules re-export under
Qualtran's names."""

from __future__ import annotations

import abc
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
            try:
                other = _as_gate_counts(other)  # Qualtran's own GateCounts adds too
            except (TypeError, ValueError):
                raise TypeError(f"Can only add other `GateCounts` objects, not {other!r}") from None
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

    def to_legacy_t_complexity(self, ts_per_toffoli: int = 4, ts_per_cswap: int = 7, ts_per_and_bloq: int = 4, cliffords_per_and_bloq: int = 9, cliffords_per_cswap: int = 10) -> "TComplexity":
        """The counts coalesced into a legacy `TComplexity` of T gates, rotations and Cliffords (measurements
        among the Cliffords). As Qualtran's, a CSWAP costs 7 T gates here but 4 in `total_t_count`."""
        return TComplexity(t=self.t + ts_per_toffoli * self.toffoli + ts_per_cswap * self.cswap + ts_per_and_bloq * self.and_bloq, rotations=self.rotation, clifford=self.clifford + self.measurement + cliffords_per_and_bloq * self.and_bloq + cliffords_per_cswap * self.cswap)

    def total_beverland_count(self) -> dict:
        """Beverland et al.'s M_meas, M_R, M_T, M_Tof and D_R (the rotation layers, taken as the
        rotations)."""
        toffoli = self.toffoli + self.and_bloq + self.cswap
        return {"meas": self.measurement, "R": self.rotation, "T": self.t, "Tof": toffoli, "D_R": self.rotation}


@dataclass(frozen=True)
class TComplexity:
    """Qualtran's legacy count of T gates, Clifford gates and single-qubit rotations."""

    t: int = 0
    clifford: int = 0
    rotations: int = 0

    @staticmethod
    def rotation_cost(eps: float) -> int:
        """T gates per rotation synthesised to within ``eps``: ``⌈1.149 log₂(1/ε) + 9.2⌉``."""
        return int(np.ceil(1.149 * math.log2(1.0 / eps) + 9.2))

    def t_incl_rotations(self, eps: float = 1e-11) -> int:
        """The T gates with every rotation synthesised to within ``eps``."""
        return int(np.ceil(self.t + self.rotation_cost(eps) * self.rotations))

    def __add__(self, other: "TComplexity") -> "TComplexity":
        return TComplexity(self.t + other.t, self.clifford + other.clifford, self.rotations + other.rotations)

    def __mul__(self, other: int) -> "TComplexity":
        return TComplexity(self.t * other, self.clifford * other, self.rotations * other)

    __rmul__ = __mul__

    def asdict(self) -> dict:
        return {"t": self.t, "rotations": self.rotations, "clifford": self.clifford}

    def __str__(self) -> str:
        return f"T-count:   {self.t:g}\nRotations: {self.rotations:g}\nCliffords: {self.clifford:g}\n"


_GATE_COUNT_FIELDS = ("t", "toffoli", "cswap", "and_bloq", "clifford", "rotation", "measurement")


def _as_gate_counts(counts) -> "GateCounts":
    """This module's `GateCounts` for Qualtran's own (``qualtran.resource_counting.GateCounts``) or anything else
    with its seven counts as attributes; a `GateCounts` as it is."""
    if isinstance(counts, GateCounts):
        return counts
    try:
        return GateCounts(**{name: _plain_count(name, getattr(counts, name)) for name in _GATE_COUNT_FIELDS})
    except AttributeError:
        raise TypeError(f"expected GateCounts (the counts t, toffoli, cswap, and_bloq, clifford, rotation and measurement), not {counts!r}") from None


def _plain_count(name: str, value) -> int:
    """A count Qualtran's resource counting returned (an int or a sympy number) as an int."""
    if isinstance(value, (int, np.integer)):
        return int(value)
    try:
        as_int = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"the {name} count {value} is symbolic; this port counts in plain integers") from None
    if as_int != value:
        raise ValueError(f"the {name} count {value} is not an integer")
    return as_int


@dataclass(frozen=True)
class AlgorithmSummary:
    """The logical costs of an algorithm: its qubits, its gate counts, and optionally the number
    of layers with a rotation in them (the rotations themselves by default)."""

    n_algo_qubits: int
    n_logical_gates: GateCounts
    n_rotation_layers: Optional[int] = None

    @staticmethod
    def from_bloq(bloq) -> "AlgorithmSummary":
        """The summary of a Qualtran Bloq: its qubits (Qualtran's `QubitCount`) and its gate counts (its
        `QECGatesCost`), counted by Qualtran's resource counting, so Qualtran must be installed. The counts must
        come out as numbers; a symbolic one raises ValueError."""
        try:
            from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value
        except ImportError as ex:
            raise ImportError("AlgorithmSummary.from_bloq counts the Bloq with Qualtran's resource counting, which needs Qualtran: pip install qualtran") from ex
        counts = get_cost_value(bloq, QECGatesCost())
        qubits = int(get_cost_value(bloq, QubitCount()))
        gates = _as_gate_counts(counts)
        return AlgorithmSummary(n_algo_qubits=qubits, n_logical_gates=gates)


# ---------------------------------------------------------------------------------------------
# QEC schemes and physical parameters
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class QECScheme:
    """The logical error rate ``a (p / p*)^((d + 1) / 2)`` of a distance-``d`` code at physical
    error ``p``: ``error_rate_scaler`` a, ``error_rate_threshold`` p*."""

    error_rate_scaler: float
    error_rate_threshold: float

    def __repr__(self) -> str:  # as Qualtran's attrs class prints it: both numbers `:g`
        return f"QECScheme(error_rate_scaler={self.error_rate_scaler:g}, error_rate_threshold={self.error_rate_threshold:g})"

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


@dataclass(frozen=True, init=False)
class LogicalErrorModel:
    """A QEC scheme at one physical error rate: callable on a code distance. As Qualtran's, the scheme is the
    private ``_qec_scheme`` (given as ``qec_scheme``; `qec_scheme` reads it too)."""

    physical_error: float
    _qec_scheme: QECScheme

    def __init__(self, physical_error: float, qec_scheme: QECScheme) -> None:
        object.__setattr__(self, "physical_error", physical_error)
        object.__setattr__(self, "_qec_scheme", qec_scheme)

    @property
    def qec_scheme(self) -> QECScheme:
        return self._qec_scheme

    def __call__(self, code_distance: int) -> float:
        return self._qec_scheme.logical_error_rate(code_distance=code_distance, physical_error=self.physical_error)


@dataclass(frozen=True)
class PhysicalParameters:
    physical_error: float = 1e-3
    cycle_time_us: float = 1.0

    def __repr__(self) -> str:  # as Qualtran's attrs class prints it: the error rate `:g`
        return f"PhysicalParameters(physical_error={self.physical_error:g}, cycle_time_us={self.cycle_time_us!r})"

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


class DataBlock(metaclass=abc.ABCMeta):
    """How the algorithm's qubits are laid out in surface-code tiles: the tiles for a number of
    algorithm qubits and the steps (of ``data_d`` cycles) a magic state takes to consume. Abstract, as
    Qualtran's: a block defines `data_d`, `n_tiles` and `n_steps_to_consume_a_magic_state`."""

    @property
    @abc.abstractmethod
    def data_d(self) -> int:
        """The code distance of the data block's tiles."""

    @abc.abstractmethod
    def n_tiles(self, n_algo_qubits: int) -> int:
        """The surface-code tiles that hold `n_algo_qubits` algorithm qubits."""

    @property
    @abc.abstractmethod
    def n_steps_to_consume_a_magic_state(self) -> int:
        """The steps (of `data_d` cycles each) a magic state takes to consume."""

    # The blocks below keep `data_d` in a slot (so that it overrides the abstract property, as an attrs field
    # does Qualtran's); pickle and copy, at every protocol, take the slot with the instance dict and set both
    # past the frozen dataclass's __setattr__.
    def __getstate__(self):
        slots = {k: getattr(self, k) for c in type(self).__mro__ for k in c.__dict__.get("__slots__", ()) if hasattr(self, k)}
        return (getattr(self, "__dict__", None) or None, slots or None)

    def __setstate__(self, state) -> None:
        dict_state, slot_state = state if isinstance(state, tuple) else (state, None)
        if dict_state:
            self.__dict__.update(dict_state)
        for key, value in (slot_state or {}).items():
            object.__setattr__(self, key, value)

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

    __slots__ = ("data_d",)
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

    __slots__ = ("data_d",)
    data_d: int

    def n_tiles(self, n_algo_qubits: int) -> int:
        return math.ceil(1.5 * n_algo_qubits)

    @property
    def n_steps_to_consume_a_magic_state(self) -> int:
        return 9


@dataclass(frozen=True)
class IntermediateDataBlock(DataBlock):
    """Litinski's intermediate block: 2 tiles per qubit, 5 steps per magic state."""

    __slots__ = ("data_d",)
    data_d: int

    def n_tiles(self, n_algo_qubits: int) -> int:
        return math.ceil(2 * n_algo_qubits)

    @property
    def n_steps_to_consume_a_magic_state(self) -> int:
        return 5


@dataclass(frozen=True)
class FastDataBlock(DataBlock):
    """Litinski's fast block: ``2n + √(8n) + 1`` tiles, one step per magic state."""

    __slots__ = ("data_d",)
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


class MagicStateFactory(metaclass=abc.ABCMeta):
    """A factory's physical qubits, the cycles to make the magic states the gates need, and the
    error it leaves in them. Abstract, as Qualtran's."""

    @abc.abstractmethod
    def n_physical_qubits(self) -> int:
        """The physical qubits of the factory."""

    @abc.abstractmethod
    def n_cycles(self, n_logical_gates: GateCounts, logical_error_model: LogicalErrorModel) -> int:
        """The cycles to make the magic states the gates need."""

    @abc.abstractmethod
    def factory_error(self, n_logical_gates: GateCounts, logical_error_model: LogicalErrorModel) -> float:
        """The error the factory leaves in those magic states."""


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


_PAULI = {"I": np.eye(2, dtype=complex), "X": np.array([[0, 1], [1, 0]], dtype=complex), "Y": np.array([[0, -1j], [1j, 0]], dtype=complex), "Z": np.array([[1, 0], [0, -1]], dtype=complex)}


def _pauli_matrix(pauli_string: str) -> np.ndarray:
    return reduce(np.kron, (_PAULI[c] for c in pauli_string))


def _rotation_unitary(pauli_string: str, theta: float) -> np.ndarray:
    """exp(i θ P)."""
    return math.cos(theta) * np.eye(1 << len(pauli_string)) + 1j * math.sin(theta) * _pauli_matrix(pauli_string)


class NoisyPauliRotation:
    """Qualtran's channel (``t_factory_utils.NoisyPauliRotation``, a Cirq gate there) as matrices: the π/8
    rotation ``exp(i π/8 P)`` about a Pauli string (its first letter the most significant qubit, as Cirq orders
    them) that overshoots to 5π/8, −π/8 or 3π/8 with probabilities p1, p2, p3 (each in [0, 1], their sum at
    most 1). `mixture` is Cirq's ``_mixture_``, `kraus` its Kraus operators, `apply` the channel on a density
    matrix."""

    def __init__(self, pauli_string: str, p1: float, p2: float, p3: float):
        for p in (p1, p2, p3):
            if not 0.0 <= p <= 1.0:
                raise ValueError(f"a channel probability must be between 0 and 1, got {p!r}")
        if p1 + p2 + p3 > 1.0 + 1e-12:
            raise ValueError(f"channel probabilities {p1!r}, {p2!r}, {p3!r} sum past 1")
        if not pauli_string or set(pauli_string) - set(_PAULI):
            raise ValueError(f"a Pauli string is of I, X, Y and Z, not {pauli_string!r}")
        self.pauli_string = pauli_string
        self._probabilities = [1 - p1 - p2 - p3, p1, p2, p3]
        self._unitaries = [_rotation_unitary(pauli_string, t * math.pi / 8) for t in (1, 5, -1, 3)]

    def num_qubits(self) -> int:
        return len(self.pauli_string)

    def mixture(self) -> tuple:
        """The (probability, unitary) pairs: the rotation by π/8, 5π/8, −π/8 and 3π/8."""
        return tuple(zip(self._probabilities, self._unitaries))

    def kraus(self) -> tuple:
        return tuple(math.sqrt(max(p, 0.0)) * u for p, u in self.mixture())

    def apply(self, rho: np.ndarray) -> np.ndarray:
        out = np.zeros_like(rho)
        for p, u in self.mixture():
            out += p * (u @ rho @ u.conj().T)
        return out


@dataclass(frozen=True)
class PauliFlipChannel:
    """One of `storage_error`'s channels (Cirq's ``AsymmetricDepolarizingChannel`` with only ``p_x`` or
    ``p_z`` there): an X or Z flip with probability ``p`` on qubit ``qubit``, ``ρ → (1 − p) ρ + p E ρ E†``."""

    kind: str
    p: float
    qubit: int

    def __post_init__(self) -> None:
        if self.kind not in ("Z", "X"):
            raise ValueError(f"kind must be Z or X not {self.kind}")
        if not 0.0 <= self.p <= 1.0:
            raise ValueError(f"a flip probability must be between 0 and 1, got {self.p!r}")

    def mixture(self) -> tuple:
        """The single-qubit (probability, unitary) pairs: nothing, or the flip."""
        return ((1 - self.p, _PAULI["I"]), (self.p, _PAULI[self.kind]))

    def kraus(self) -> tuple:
        return tuple(math.sqrt(p) * u for p, u in self.mixture())

    def apply(self, rho: np.ndarray) -> np.ndarray:
        """The flip on qubit `qubit` of an n-qubit density matrix (qubit 0 the most significant)."""
        if self.p == 0:
            return rho
        n = int(rho.shape[0]).bit_length() - 1
        letters = ["I"] * n
        letters[self.qubit] = self.kind
        e = _pauli_matrix("".join(letters))
        return (1 - self.p) * rho + self.p * (e @ rho @ e.conj().T)


def storage_error(kind: str, probabilities, qubits=None) -> list:
    """Qualtran's storage errors (``t_factory_utils.storage_error``): a `PauliFlipChannel` of the kind (Z or X)
    on each qubit with its probability, the qubits 0, 1, … unless given (as many as the probabilities)."""
    if kind not in ("Z", "X"):
        raise ValueError(f"kind must be Z or X not {kind}")
    probabilities = list(probabilities)
    qubits = list(range(len(probabilities))) if qubits is None else list(qubits)
    if len(qubits) != len(probabilities):
        raise ValueError(f"{len(probabilities)} probabilities for {len(qubits)} qubits")
    return [PauliFlipChannel(kind, p, q) for p, q in zip(probabilities, qubits)]


class _Density:
    """A density matrix of n qubits, |+⟩ on each, under the two channels the 15-to-1 model uses."""

    def __init__(self, n: int) -> None:
        self.n = n
        plus = np.full((2,), 1 / math.sqrt(2), dtype=complex)
        state = reduce(np.kron, [plus] * n)
        self.rho = np.outer(state, state.conj())

    def noisy_rotation(self, pauli_string: str, p1: float, p2: float, p3: float) -> None:
        self.rho = NoisyPauliRotation(pauli_string, p1, p2, p3).apply(self.rho)

    def storage_error(self, kind: str, probabilities) -> None:
        for channel in storage_error(kind, probabilities):
            self.rho = channel.apply(self.rho)


@dataclass(frozen=True)
class FifteenToOne(MagicStateFactory):
    """Litinski's 15-to-1 T factory (arXiv:1905.06903): ``d_X`` and ``d_Z`` the patch sides,
    ``d_m`` the measurement rounds per surgery; its failure and output error from the paper's
    density-matrix error model of the eleven π/8 rotations.

    The density matrix is evolved here with numpy, not with Cirq's simulator as in Qualtran, so the last bits
    of `p_fail` and `p_out` can differ from Qualtran's: `p_fail` agrees to about 1e-11 relative, and `p_out`
    (one minus a trace near 1) to about 1e-16 absolute, which is a large relative difference when `p_out` is
    itself near 1e-14 (a few percent at ``d_X=9, d_Z=5, d_m=3`` and a 1e-5 physical error). Everything else
    here, `n_cycles` included, has given Qualtran's numbers exactly."""

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

    def n_cycles(self, n_logical_gates: GateCounts, logical_error_model: LogicalErrorModel) -> float:
        """The expected cycles, failures included (page 11 of the paper), as Qualtran gives them: a numpy float
        (``np.ceil``)."""
        return np.ceil(n_logical_gates.total_t_count() * 6 * self.d_m / (1 - self.p_fail(logical_error_model)))

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


class RotationCostModel(metaclass=abc.ABCMeta):
    """The gates a single-qubit rotation costs within an error budget, and the gates to prepare whatever
    the rotations share. Abstract, as Qualtran's."""

    @abc.abstractmethod
    def rotation_cost(self, error_budget: float) -> GateCounts:
        """The gates one rotation costs within `error_budget`."""

    @abc.abstractmethod
    def preparation_overhead(self, error_budget: float) -> GateCounts:
        """The gates paid once, before the rotations."""


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


def get_ccz2t_costs_from_grid_search(*, n_logical_gates: GateCounts, n_algo_qubits: int, phys_err: float = 1e-3, error_budget: float = 1e-2, cycle_time_us: float = 1.0, factory_iter: Iterable[MagicStateFactory] = tuple(iter_ccz2t_factories()), data_block_iter: Iterable[DataBlock] = tuple(iter_simple_data_blocks()), cost_function: Callable[[PhysicalCostsSummary], float] = lambda pc: pc.qubit_hours) -> tuple[PhysicalCostsSummary, MagicStateFactory, DataBlock]:
    """The factory and data block that minimise ``cost_function`` within the error budget. As Qualtran's, the
    blocks are iterated again for each factory, so a one-shot iterator of blocks is used up by the first factory
    (and ``ValueError: No valid factories found!`` follows when none of its blocks fits)."""
    best: Optional[PhysicalCostsSummary] = None
    best_params = None
    for factory in factory_iter:
        for data_block in data_block_iter:
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
    """The counts with every rotation synthesised into T and Toffoli gates (Qualtran's own `GateCounts` in the
    summary is read as this module's)."""
    counts = _as_gate_counts(alg.n_logical_gates)
    n_rotations = counts.rotation
    ret = replace(counts, rotation=0)
    if n_rotations > 0:
        ret = ret + rotation_model.preparation_overhead(eps_syn) + n_rotations * rotation_model.rotation_cost(eps_syn / n_rotations)
    return ret


def t_states(*, error_budget: float, alg: AlgorithmSummary, rotation_model: RotationCostModel) -> int:
    """Beverland et al.'s D4: the T states the algorithm consumes, rotations synthesised at ε/3."""
    return n_discrete_logical_gates(eps_syn=error_budget / 3, alg=alg, rotation_model=rotation_model).total_t_count()
