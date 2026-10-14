"""qsample's simulators, on this package's ``TableauSimulator``: its gate names (``init``,
``measure``, ``H``, ``S``, ``Sd``, ``X``, ``Y``, ``Z``, ``I``, ``Q`` = √X, ``Qd``, ``R`` = √Y,
``Rd``, ``CNOT``, ``MSd``) and its measurement results."""

from __future__ import annotations

from typing import Optional

from stabilizer_qec.qsample._mixin import CircuitRunnerMixin

__all__ = ["MeasureResult", "StabilizerSimulator", "StatevectorSimulator"]


class MeasureResult:
    """A measurement's ``value`` and whether it was ``determined`` (not random)."""

    def __init__(self, value: bool, determined: bool) -> None:
        self.value = bool(value)
        self.determined = bool(determined)

    def __bool__(self) -> bool:
        return self.value

    def __eq__(self, other: object) -> bool:
        if isinstance(other, (bool, int)):
            return self.value == bool(other)
        return isinstance(other, MeasureResult) and (self.value, self.determined) == (other.value, other.determined)

    def __str__(self) -> str:
        return f"{self.value} ({['random', 'determined'][self.determined]})"

    def __repr__(self) -> str:
        return f"MeasureResult(value={self.value!r}, determined={self.determined!r})"


class StabilizerSimulator(CircuitRunnerMixin):
    """A stabilizer state of ``num_qubits`` qubits, all |0⟩ (qsample's CHP simulator, on this
    package's tableau simulator). ``seed`` fixes its measurements' randomness."""

    def __init__(self, num_qubits: int, seed: Optional[int] = None) -> None:
        import numpy as np

        from stabilizer_qec import TableauSimulator

        if seed is None:  # numpy's global seed (qsample's way to reproduce) fixes these too
            seed = int(np.random.randint(0, 2**63 - 1, dtype=np.int64))
        self.num_qubits = num_qubits
        self._sim = TableauSimulator(seed=seed)
        self._sim.set_num_qubits(num_qubits)

    def init(self, qubit: int) -> None:
        self._sim.reset(qubit)

    def measure(self, qubit: int) -> MeasureResult:
        determined = self._sim.peek_z(qubit) != 0
        return MeasureResult(self._sim.measure(qubit), determined)

    def H(self, qubit: int) -> None:
        self._sim.h(qubit)

    def S(self, qubit: int) -> None:
        self._sim.s(qubit)

    def Sd(self, qubit: int) -> None:
        self._sim.s_dag(qubit)

    def X(self, qubit: int) -> None:
        self._sim.x(qubit)

    def Y(self, qubit: int) -> None:
        self._sim.y(qubit)

    def Z(self, qubit: int) -> None:
        self._sim.z(qubit)

    def I(self, qubit: int) -> None:  # noqa: E743 - qsample's gate name
        pass

    def Q(self, qubit: int) -> None:
        self._sim.sqrt_x(qubit)

    def Qd(self, qubit: int) -> None:
        self._sim.sqrt_x_dag(qubit)

    def R(self, qubit: int) -> None:
        self._sim.sqrt_y(qubit)

    def Rd(self, qubit: int) -> None:
        self._sim.sqrt_y_dag(qubit)

    def CNOT(self, control: int, target: int) -> None:
        self._sim.cx(control, target)

    def MSd(self, qubitA: int, qubitB: int) -> None:
        self.Rd(qubitA)
        self.CNOT(qubitA, qubitB)
        self.R(qubitA)
        self.Qd(qubitA)
        self.Qd(qubitB)

    def __getattr__(self, name: str):
        if name in ("T", "Rx", "Ry", "Rz"):
            raise NotImplementedError(f"{name} is not a Clifford gate: the stabilizer simulator cannot apply it")
        raise AttributeError(name)


class StatevectorSimulator(CircuitRunnerMixin):  # pragma: no cover - exercised only by non-Clifford protocols
    """qsample's state-vector simulator needs projectq; here its non-Clifford gates are not
    available, and the Clifford ones run on the stabilizer simulator."""

    def __new__(cls, num_qubits: int, seed: Optional[int] = None):
        return StabilizerSimulator(num_qubits, seed)
