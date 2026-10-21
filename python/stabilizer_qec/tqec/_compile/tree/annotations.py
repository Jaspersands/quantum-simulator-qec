# Ported from tqec (tqec/tqec 0.2.0, src/tqec/compile/tree/annotations.py; Apache License 2.0, see LICENSE in stabilizer_qec/tqec),
# modified for stabilizer_qec.
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import stabilizer_qec as stim

from stabilizer_qec.tqec._compile.circuit.measurement_map import MeasurementRecordsMap
from stabilizer_qec.tqec._compile.circuit.qubit import GridQubit
from stabilizer_qec.tqec._compile.circuit.qubit_map import QubitMap
from stabilizer_qec.tqec._compile.circuit.schedule.circuit import ScheduledCircuit
from stabilizer_qec.tqec._compile.detectors.detector import Detector
from stabilizer_qec.tqec._compile.observables.builder import Observable
from stabilizer_qec.tqec._compile.plaquette.rpng.rpng import PauliBasis
from stabilizer_qec.tqec._compile.utils.coordinates import StimCoordinates
from stabilizer_qec.tqec._errors import TQECError


@dataclass(frozen=True)
class DetectorAnnotation:
    """An annotation including all the necessary information to build a ``DETECTOR`` instruction."""

    coordinates: StimCoordinates
    measurement_offsets: list[int]

    def __post_init__(self) -> None:
        if any(m >= 0 for m in self.measurement_offsets):
            raise TQECError("Expected strictly negative measurement offsets.")

    def to_instruction(self) -> stim.CircuitInstruction:
        """Return the ``DETECTOR`` instruction represented by ``self``."""
        return stim.CircuitInstruction(
            "DETECTOR",
            [stim.target_rec(offset) for offset in self.measurement_offsets],
            self.coordinates.to_stim_coordinates(),
        )

    @staticmethod
    def from_detector(
        detector: Detector, measurement_records: MeasurementRecordsMap
    ) -> DetectorAnnotation:
        """Create a :class:`DetectorAnnotation` from a detector and measurement records."""
        return DetectorAnnotation(
            detector.coordinates,
            sorted([measurement_records[m.qubit][m.offset] for m in detector.measurements]),
        )


@dataclass(frozen=True)
class Polygon:
    """A polygon representing a stabilizer region in Crumble."""

    basis: PauliBasis
    qubits: frozenset[GridQubit]

    def _sorted_qubits(self) -> list[GridQubit]:  # pragma: no cover
        """Return the qubits in a sorted order that can be used to draw the polygon."""
        cx = sum(q.x for q in self.qubits) / len(self.qubits)
        cy = sum(q.y for q in self.qubits) / len(self.qubits)
        return sorted(self.qubits, key=lambda q: math.atan2(q.y - cy, q.x - cx))

    def to_crumble_url_string(self, qubit_map: QubitMap) -> str:
        """Convert the polygon to the representation in a crumble url."""
        # default grey color for polygons with no basis information
        rgba = [0, 0, 0, 0.25]
        rgba["xyz".index(self.basis.value)] = 1
        rgba_str = ",".join(str(i) for i in rgba)
        qubits_idx = [qubit_map[q] for q in self._sorted_qubits()]
        qubits_str = "_".join(str(i) for i in qubits_idx)
        return f"POLYGON({rgba_str}){qubits_str};"


@dataclass
class LayerNodeAnnotations:
    circuit: ScheduledCircuit | None = None
    detectors: list[DetectorAnnotation] = field(default_factory=list)
    observables: list[Observable] = field(default_factory=list)
    polygons: list[Polygon] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation of ``self``."""
        return {  # pragma: no cover
            "circuit_str": (str(self.circuit.get_circuit()) if self.circuit is not None else None),
            "detectors": self.detectors,
            "observables": self.observables,
            "polygons": self.polygons,
        }


@dataclass
class LayerTreeAnnotations:
    qubit_map: QubitMap | None = None

    @property
    def has_qubit_map(self) -> bool:  # pragma: no cover
        """Return ``True`` if the qubit map annotation has been set."""
        return self.qubit_map is not None

    def to_dict(self) -> dict[str, Any]:  # pragma: no cover
        """Return a dictionary representation of ``self``."""
        ret: dict[str, Any] = {"qubit_map": None}
        if self.qubit_map is not None:
            ret["qubit_map"] = {
                i: (
                    q.x,
                    q.y,
                )
                for i, q in self.qubit_map.i2q.items()
            }
        return ret
