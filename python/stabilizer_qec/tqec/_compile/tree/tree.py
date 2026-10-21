# Ported from tqec (tqec/tqec 0.2.0, src/tqec/compile/tree/tree.py; Apache License 2.0, see LICENSE in stabilizer_qec/tqec),
# modified for stabilizer_qec: the detector database is in memory only (no file is read or
# written), detectors are computed in this process (see compute.py), and the Crumble URL and SVG
# methods, which need tqec's visualisation package, are not ported.
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Union

import stabilizer_qec as stim

from stabilizer_qec.tqec._compile.circuit.qubit import GridQubit
from stabilizer_qec.tqec._compile.circuit.qubit_map import QubitMap
from stabilizer_qec.tqec._compile.blocks.layers.composed.sequenced import SequencedLayers
from stabilizer_qec.tqec._compile.detectors.database import CURRENT_DATABASE_VERSION, DetectorDatabase
from stabilizer_qec.tqec._compile.observables.abstract_observable import AbstractObservable
from stabilizer_qec.tqec._compile.observables.builder import ObservableBuilder
from stabilizer_qec.tqec._compile.tree.annotations import LayerTreeAnnotations
from stabilizer_qec.tqec._compile.tree.annotators.circuit import AnnotateCircuitOnLayerNode
from stabilizer_qec.tqec._compile.tree.annotators.detectors import AnnotateDetectorsOnLayerNode
from stabilizer_qec.tqec._compile.tree.annotators.observables import annotate_observable
from stabilizer_qec.tqec._compile.tree.node import LayerNode, NodeWalker
from stabilizer_qec.tqec._errors import TQECError

# tqec runs its detector search in a pool of ``cpu_count() // 2 + 1`` processes when the database
# is empty: each situation computed afresh and returned through pickle. compute.py runs that path
# in this process; any count above 1 selects it.
_FRESH_DATABASE_PROCESS_COUNT = 2


class QubitLister(NodeWalker):
    def __init__(self, k: int):
        """Keep in memory all the qubits used by the nodes explored.

        Args:
            k: scaling factor used to explore the quantum circuits.

        """
        super().__init__()
        self._k = k
        self._seen_qubits: set[GridQubit] = set()

    def visit_node(self, node: LayerNode) -> None:
        if not node.is_leaf:
            return
        annotations = node.get_annotations(self._k)
        if annotations.circuit is None:
            raise TQECError("Cannot list qubits without the circuit annotation.")
        self._seen_qubits |= annotations.circuit.qubits

    @property
    def seen_qubits(self) -> set[GridQubit]:
        """Return all the qubits seen when exploring."""
        return self._seen_qubits


class LayerTree:
    def __init__(
        self,
        root: SequencedLayers,
        observable_builder: ObservableBuilder,
        abstract_observables: list[AbstractObservable] | None = None,
        annotations: Mapping[int, LayerTreeAnnotations] | None = None,
    ):
        """Represent a computation as a tree.

        Note:
            It is expected that the root is a
            :class:`~tqec.compile.blocks.layer.composed.sequenced.SequencedLayers`
            with each of its children representing the computation on a timespan
            equivalent to the timespan of a cube.

        Args:
            root: root node of the tree.
            abstract_observables: a list of abstract observables to be compiled into
                observables. If set to ``None``, no observables will be compiled
                into the circuit.
            annotations: a mapping from positive integers representing the value
                of ``k``, the scaling factor, to annotations computed for that
                value of ``k``.
            observable_builder: the style of the surface code patch.

        """
        self._root = LayerNode(root)
        self._abstract_observables = abstract_observables or []
        self._annotations = dict(annotations) if annotations is not None else {}
        self._observable_builder = observable_builder

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation of ``self``."""
        return {  # pragma: no cover
            "root": self._root.to_dict(),
            "abstract_observables": self._abstract_observables,
            "annotations": {k: annotation.to_dict() for k, annotation in self._annotations.items()},
        }

    def _annotate_circuits(self, k: int, reschedule_measurements: bool = True) -> None:
        self._root.walk(
            AnnotateCircuitOnLayerNode(k, reschedule_measurements=reschedule_measurements)
        )

    def _annotate_qubit_map(self, k: int) -> None:
        self._get_annotation(k).qubit_map = self._get_global_qubit_map(k)

    def _get_global_qubit_map(self, k: int) -> QubitMap:
        qubit_lister = QubitLister(k)
        self._root.walk(qubit_lister)
        return QubitMap.from_qubits(sorted(qubit_lister.seen_qubits))

    def _annotate_observables(self, k: int) -> None:
        for obs_idx, observable in enumerate(self._abstract_observables):
            annotate_observable(self._root, k, observable, obs_idx, self._observable_builder)

    def _annotate_detectors(
        self,
        k: int,
        manhattan_radius: int = 2,
        detector_database: DetectorDatabase | None = None,
        database_path: Path | None = None,
        only_use_database: bool = False,
        lookback: int = 2,
        parallel_process_count: int = 1,
    ) -> None:
        if manhattan_radius <= 0:
            return  # pragma: no cover
        self._root.walk(
            AnnotateDetectorsOnLayerNode(
                k,
                manhattan_radius,
                detector_database,
                only_use_database,
                lookback,
                parallel_process_count,
            )
        )

    def _generate_annotations(
        self,
        k: int,
        manhattan_radius: int = 2,
        detector_database: DetectorDatabase | None = None,
        database_path: Path | None = None,
        only_use_database: bool = False,
        lookback: int = 2,
        parallel_process_count: int = 1,
        reschedule_measurements: bool = True,
    ) -> None:
        """Annotate the tree with circuits, qubit maps, detectors and observables."""
        # If already annotated, no need to re-annotate.
        if k in self._annotations:
            return  # pragma: no cover
        # Else, perform all the needed computations.
        self._annotate_circuits(k, reschedule_measurements=reschedule_measurements)
        self._annotate_qubit_map(k)
        # This method will also update the detector_database and save it to disk at database_path.
        self._annotate_detectors(
            k,
            manhattan_radius,
            detector_database,
            database_path,
            only_use_database,
            lookback,
            parallel_process_count,
        )
        self._annotate_observables(k)

    def generate_circuit(
        self,
        k: int,
        include_qubit_coords: bool = True,
        manhattan_radius: int = 2,
        detector_database: DetectorDatabase | None = None,
        database_path: Union[str, Path, None] = None,
        do_not_use_database: bool = False,
        only_use_database: bool = False,
        lookback: int = 2,
        reschedule_measurements: bool = True,
    ) -> stim.Circuit:
        """Generate the quantum circuit representing ``self``.

        This method first annotates the tree according to the provided arguments
        and then use these annotations to generate the final quantum circuit.

        Args:
            k: scaling factor.
            include_qubit_coords: whether to include ``QUBIT_COORDS`` annotations
                in the returned quantum circuit or not. Default to ``True``.
            manhattan_radius: Parameter for the automatic computation of detectors.
                Should be large enough so that flows canceling each other to
                form a detector are strictly contained in plaquettes that are at
                most at a distance of ``manhattan_radius`` from the central
                plaquette. Detector computation runtime grows with this parameter,
                so you should try to keep it to its minimum. A value too low might
                produce invalid detectors.
            detector_database: an instance to retrieve from / store in detectors
                that are computed as part of the circuit generation. If not given,
                an empty one, in memory.
            database_path: accepted for tqec's signature and not used: this port never reads
                or writes a database file.
            do_not_use_database: if ``True``, no database is used.
            only_use_database: if ``True``, only detectors from the database
                will be used. An error will be raised if a situation that is not
                registered in the database is encountered.
            lookback: number of QEC rounds to consider to try to find detectors.
                Including more rounds increases computation time.
            reschedule_measurements: whether to reschedule measurements in a ``LayoutLayer``
                to be in the same moment. Since each plaquette may have its own measurement
                schedule, setting this may be necessary for hardware that requires
                measurements to be synchronous.

        Returns:
            a ``stim.Circuit`` instance implementing the computation described
            by ``self``.

        """
        # tqec reads its database from ``database_path`` (or its default file) and writes it back
        # there; this port keeps it in memory: the given ``detector_database``, else an empty one.
        if not do_not_use_database:
            if detector_database is None:
                detector_database = DetectorDatabase()
            loaded_version = detector_database.version
            current_version = CURRENT_DATABASE_VERSION
            if loaded_version != current_version:
                raise TQECError(
                    f"The detector database on disk you have specified is incompatible with"
                    f" the version in the TQEC code you are running. The version of the disk"
                    f" database is {loaded_version}, while the version in the TQEC code is "
                    f"{current_version}."
                )
        else:
            detector_database = None

        # Enable parallel processing only if the detector database is empty or None,
        # as current parallelization is effective only in this case.
        parallel_process_count = (
            _FRESH_DATABASE_PROCESS_COUNT
            if (detector_database is None or len(detector_database) == 0)
            else 1
        )

        self._generate_annotations(
            k,
            manhattan_radius,
            detector_database=detector_database,
            database_path=None,
            only_use_database=only_use_database,
            lookback=lookback,
            parallel_process_count=parallel_process_count,
            reschedule_measurements=reschedule_measurements,
        )
        annotations = self._get_annotation(k)
        assert annotations.qubit_map is not None

        circuit = stim.Circuit()
        if include_qubit_coords:
            circuit += annotations.qubit_map.to_circuit()
        circuit += self._root.generate_circuit(k, annotations.qubit_map)
        return circuit

    def _get_annotation(self, k: int) -> LayerTreeAnnotations:
        return self._annotations.setdefault(k, LayerTreeAnnotations())
