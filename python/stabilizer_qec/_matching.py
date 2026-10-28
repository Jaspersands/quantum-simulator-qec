"""``Matching``: PyMatching 2.4's ``Matching``, whole. A matching graph built by hand (edges,
boundary edges, boundary nodes, fault ids, merge strategies), from a check matrix (with
repetitions, timelike weights and a faults matrix), from networkx or rustworkx, or from a
detector error model or circuit; queried; decoded to fault ids and weights, to edges, or to
matched detection events, on PyMatching's own sparse blossom transliterated (``src/pm_exact/``,
the graph built as PyMatching 2.4 builds it, ``src/user_graph.rs``): its answers, ties and
orders included. ``enable_correlations`` decodes on this package's correlated matcher."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from . import _core
from ._util import call, count

_rng = np.random.default_rng()
_MAX_WEIGHT = float((1 << 24) - 1)


def _too_heavy(weight: float, a: Any, b: Any, stacklevel: int) -> bool:
    """Whether an edge's weight is past the largest a matching takes, warning as PyMatching
    does (its weight printed as C++'s ``std::to_string`` prints it) when it is."""
    if not abs(float(weight)) > _MAX_WEIGHT:
        return False
    warnings.warn(f"Weight {float(weight):f} of edge ({a}, {b}) exceeds maximum edge weight {int(_MAX_WEIGHT)} and has not been added to the matching graph.", stacklevel=stacklevel)
    return True


def set_seed(seed: int) -> None:
    """Seeds the generator ``Matching.add_noise`` uses (PyMatching's ``set_seed``)."""
    global _rng
    _rng = np.random.default_rng(int(seed))


def randomize() -> None:
    """Reseeds that generator from the system's entropy (PyMatching's ``randomize``)."""
    global _rng
    _rng = np.random.default_rng()


def rand_float(from_: float, to: float) -> float:
    """A uniform float in [from_, to) from that generator (PyMatching's ``rand_float``)."""
    return float(_rng.uniform(from_, to))


def _fault_set(fault_ids: Any) -> Set[int]:
    if fault_ids is None:
        return set()
    if isinstance(fault_ids, (int, np.integer)):
        return set() if fault_ids == -1 else {int(fault_ids)}
    try:
        out = set(fault_ids)
    except TypeError:
        raise TypeError(f"fault_ids property must be an int or a set of int (or convertible to a set), not {fault_ids}") from None
    if not all(isinstance(q, (int, np.integer)) for q in out):
        raise TypeError(f"fault_ids must be a set of ints, not {fault_ids}")
    return {int(q) for q in out}


def _csc(matrix: Any) -> Tuple[int, int, np.ndarray, np.ndarray]:
    """(rows, columns, indptr, indices) of a matrix's nonzero pattern, column by column."""
    try:
        from scipy.sparse import csc_matrix, issparse

        if issparse(matrix) or not isinstance(matrix, np.ndarray):
            m = csc_matrix(matrix)
        else:
            m = csc_matrix(matrix)
        m.eliminate_zeros()
        m.sort_indices()
        bad = m.data[m.data != 1]
        if len(bad):
            # PyMatching's check, its entries read as 8-bit integers.
            v = int(np.asarray(bad[:1]).astype(np.uint8)[0])
            if v == 0:
                raise ValueError("`matrix.data` must only contain ones, but a zero was found. First call `matrix.eliminate_zeros()` before using this method.")
            raise ValueError(f"`matrix.data` must only contain ones, but the element {v} was found.")
        return m.shape[0], m.shape[1], np.asarray(m.indptr), np.asarray(m.indices)
    except ImportError:
        a = np.asarray(matrix)
        if a.ndim != 2:
            raise TypeError("`check_matrix` must be convertible to a `scipy.sparse.csc_matrix`") from None
        indptr, indices = [0], []
        for c in range(a.shape[1]):
            rows = np.flatnonzero(a[:, c])
            indices.extend(rows.tolist())
            indptr.append(len(indices))
        return a.shape[0], a.shape[1], np.array(indptr), np.array(indices, dtype=np.int64)


class Matching:
    """Exact minimum-weight perfect matching, as ``pymatching.Matching``: on a graph built by
    hand or from a check matrix, networkx / rustworkx graph, detector error model or circuit.

    Weights are PyMatching's: integer weights as they are, otherwise scaled so the heaviest is
    2^24 − 1 and rounded; negative weights allowed. A model's pieces become edges of weight
    ln((1 − p)/p), parallel ones merged as independent mechanisms. ``enable_correlations`` at
    construction (from a model) makes correlated matching (PyMatching 2.4's second pass)
    available; ask for it at decode time with ``enable_correlations=True``. The graph is then
    PyMatching's correlated load (each edge's ends in order, and the weights edges imply for each
    other counted in the weight scale), so decoding it plainly gives PyMatching's answers too.

    >>> import stabilizer_qec as sq
    >>> m = sq.Matching()
    >>> m.add_edge(0, 1, fault_ids={0}, weight=1.0)
    >>> m.add_edge(1, 2, fault_ids={1}, weight=1.0)
    >>> m.add_boundary_edge(0, weight=1.0)
    >>> m.add_boundary_edge(2, fault_ids={2}, weight=1.0)
    >>> m.decode([0, 1, 1])
    array([0, 1, 1], dtype=uint8)
    >>> m
    <stabilizer_qec.Matching object with 3 detectors, 0 boundary nodes, and 4 edges>
    """

    __slots__ = ("_g", "_corr", "_dem_text", "__weakref__")

    def __init__(
        self,
        graph: Any = None,
        weights: Any = None,
        error_probabilities: Any = None,
        repetitions: Optional[int] = None,
        timelike_weights: Any = None,
        measurement_error_probabilities: Any = None,
        *,
        enable_correlations: bool = False,
        **kwargs: Any,
    ) -> None:
        self._g = _core.UserGraph()
        self._corr = None
        self._dem_text = None
        if graph is None:
            graph = kwargs.pop("H", None)
            if graph is None:
                return
        if self._load_object(graph, enable_correlations):
            return
        self.load_from_check_matrix(graph, weights, error_probabilities, repetitions, timelike_weights, measurement_error_probabilities, **kwargs)

    def _load_object(self, graph: Any, enable_correlations: bool) -> bool:
        from ._circuit import Circuit, DetectorErrorModel

        kind = type(graph).__name__
        if kind == "DetectorErrorModel":
            self._load_from_detector_error_model(graph, enable_correlations=enable_correlations)
            return True
        if kind == "Circuit":
            self._load_from_detector_error_model(Circuit(graph).detector_error_model(decompose_errors=True), enable_correlations=enable_correlations)
            return True
        if isinstance(graph, str):
            self._load_from_detector_error_model(DetectorErrorModel(graph), enable_correlations=enable_correlations)
            return True
        try:
            import networkx as nx

            if isinstance(graph, nx.Graph):
                self.load_from_networkx(graph)
                return True
        except ImportError:
            pass
        try:
            import rustworkx as rx

            if isinstance(graph, rx.PyGraph):
                self.load_from_rustworkx(graph)
                return True
        except ImportError:
            pass
        return False

    # -- Construction --------------------------------------------------------------------

    def _load_from_detector_error_model(self, model: Any, *, enable_correlations: bool = False) -> None:
        from ._circuit import DetectorErrorModel

        dem = model if isinstance(model, DetectorErrorModel) else DetectorErrorModel(str(model))
        # PyMatching's correlated load orders each edge's ends and counts the weights edges imply
        # for each other in its weight scale, so plain decodes on it differ from a plain load's.
        self._g = call(_core.UserGraph.from_dem, str(dem), bool(enable_correlations))
        self._dem_text = str(dem)
        self._corr = call(_core.Matcher, dem._d, True) if enable_correlations else None

    @staticmethod
    def from_detector_error_model(model: Any, *, enable_correlations: bool = False) -> "Matching":
        m = Matching()
        m._load_from_detector_error_model(model, enable_correlations=enable_correlations)
        return m

    @staticmethod
    def from_detector_error_model_file(dem_path: Union[str, Path], *, enable_correlations: bool = False) -> "Matching":
        from ._circuit import DetectorErrorModel

        return Matching.from_detector_error_model(DetectorErrorModel.from_file(str(dem_path)), enable_correlations=enable_correlations)

    @staticmethod
    def from_stim_circuit(circuit: Any, *, enable_correlations: bool = False) -> "Matching":
        from ._circuit import Circuit

        if type(circuit).__name__ != "Circuit":
            raise TypeError(f"`circuit` must be a `stim.Circuit`. Instead, got {type(circuit)}")
        return Matching.from_detector_error_model(Circuit(circuit).detector_error_model(decompose_errors=True), enable_correlations=enable_correlations)

    @staticmethod
    def from_stim_circuit_file(stim_circuit_path: Union[str, Path], *, enable_correlations: bool = False) -> "Matching":
        from ._circuit import Circuit

        return Matching.from_stim_circuit(Circuit.from_file(str(stim_circuit_path)), enable_correlations=enable_correlations)

    @staticmethod
    def from_check_matrix(
        check_matrix: Any,
        weights: Any = None,
        error_probabilities: Any = None,
        repetitions: Optional[int] = None,
        timelike_weights: Any = None,
        measurement_error_probabilities: Any = None,
        *,
        faults_matrix: Any = None,
        merge_strategy: str = "smallest-weight",
        use_virtual_boundary_node: bool = False,
        **kwargs: Any,
    ) -> "Matching":
        m = Matching()
        m.load_from_check_matrix(
            check_matrix, weights, error_probabilities, repetitions, timelike_weights, measurement_error_probabilities,
            faults_matrix=faults_matrix, merge_strategy=merge_strategy, use_virtual_boundary_node=use_virtual_boundary_node, **kwargs,
        )  # fmt: skip
        return m

    def load_from_check_matrix(
        self,
        check_matrix: Any = None,
        weights: Any = None,
        error_probabilities: Any = None,
        repetitions: Optional[int] = None,
        timelike_weights: Any = None,
        measurement_error_probabilities: Any = None,
        *,
        faults_matrix: Any = None,
        merge_strategy: str = "smallest-weight",
        use_virtual_boundary_node: bool = False,
        **kwargs: Any,
    ) -> None:
        """PyMatching's ``load_from_check_matrix``: each column of the check matrix an edge (two
        ones) or a boundary edge (one); ``repetitions`` stacks that many copies joined by
        timelike edges; ``faults_matrix`` gives each column's fault ids (default: its index)."""
        if check_matrix is None:
            check_matrix = kwargs.get("H")
            if check_matrix is None:
                raise ValueError("No check_matrix provided")
        try:
            rows, cols, indptr, indices = _csc(check_matrix)
        except TypeError:
            raise TypeError("`check_matrix` must be convertible to a `scipy.sparse.csc_matrix`") from None
        if faults_matrix is None:
            f_rows, f_indptr, f_indices = cols, np.arange(cols + 1), np.arange(cols)
        else:
            f_rows, f_cols, f_indptr, f_indices = _csc(faults_matrix)
            if f_cols != cols:
                raise ValueError(f"`faults_matrix` array with shape ({f_rows}, {f_cols}) must have the same number of columns as the check matrix, which has shape ({rows}, {cols}).")
        slw = kwargs.get("spacelike_weights")
        if weights is None and slw is not None:
            weights = slw
        elif weights is not None and slw is not None:
            raise ValueError("Both `weights` and `spacelike_weights` were provided as arguments, but these two arguments are equivalent. Please provide only `weights` as an argument, as the `spacelike_weights` argument has been deprecated.")
        weights = 1.0 if weights is None else weights
        if isinstance(weights, (int, float, np.integer, np.floating)):
            weights = np.ones(cols, dtype=float) * weights
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (cols,):
            raise ValueError(f"The size of the `weights` array ({weights.size}) should match the number of columns in the check matrix ({cols})")
        if error_probabilities is None:
            error_probabilities = np.ones(cols) * -1
        elif isinstance(error_probabilities, (int, float, np.integer, np.floating)):
            error_probabilities = np.ones(cols) * error_probabilities
        error_probabilities = np.asarray(error_probabilities, dtype=float)
        if error_probabilities.shape != (cols,):
            raise ValueError(f"The size of the `error_probabilities` array ({error_probabilities.size}) should match the number of columns in the check matrix ({cols})")
        reps = 1 if repetitions is None else int(repetitions)
        t_weights = p_meas = None
        if reps > 1:
            tw = 1.0 if timelike_weights is None else timelike_weights
            if isinstance(tw, (int, float, np.integer, np.floating)):
                t_weights = np.ones(rows, dtype=float) * tw
            elif isinstance(tw, (np.ndarray, list)):
                t_weights = np.array(tw, dtype=float)
            else:
                raise ValueError("timelike_weights should be a float or a 1d numpy array")
            mep = kwargs.get("measurement_error_probability")
            if measurement_error_probabilities is not None and mep is not None:
                raise ValueError("Both `measurement_error_probabilities` and `measurement_error_probability` were provided as arguments. Please provide `measurement_error_probabilities` instead of `measurement_error_probability` as an argument, as use of `measurement_error_probability` has been deprecated.")
            if measurement_error_probabilities is None and mep is not None:
                measurement_error_probabilities = mep
            pm = measurement_error_probabilities if measurement_error_probabilities is not None else -1
            if isinstance(pm, (int, float, np.integer, np.floating)):
                p_meas = np.ones(rows, dtype=float) * pm
            elif isinstance(pm, (np.ndarray, list)):
                p_meas = np.array(pm, dtype=float)
            else:
                raise ValueError("measurement_error_probabilities should be a float or 1d numpy array")
            if t_weights.shape != (rows,):
                raise ValueError(f"timelike_weights has length {t_weights.size} but its length must equal the number of columns in the check matrix ({rows}).")
            if p_meas.shape != (rows,):
                raise ValueError(f"`measurement_error_probabilities` has length {p_meas.size} but its length must equal the number of columns in the check matrix ({rows}).")
        num_detectors = rows * reps
        g = _core.UserGraph(num_detectors, f_rows)
        for rep in range(reps):
            off = rows * rep
            for c in range(cols):
                dets = indices[indptr[c] : indptr[c + 1]]
                obs = [int(x) for x in f_indices[f_indptr[c] : f_indptr[c + 1]]]
                w, p = float(weights[c]), float(error_probabilities[c])
                if len(dets) == 2:
                    call(g.add_edge, int(dets[0]) + off, int(dets[1]) + off, obs, w, p, merge_strategy)
                elif len(dets) == 1:
                    if use_virtual_boundary_node:
                        call(g.add_boundary_edge, int(dets[0]) + off, obs, w, p, merge_strategy)
                    else:
                        call(g.add_edge, int(dets[0]) + off, num_detectors, obs, w, p, merge_strategy)
                elif len(dets) != 0:
                    raise ValueError(f"`check_matrix` must contain at most two ones per column, but column {c} has {len(dets)} ones.")
        if reps > 1:
            for rep in range(reps - 1):
                for r in range(rows):
                    call(g.add_edge, r + rep * rows, r + (rep + 1) * rows, [], float(t_weights[r]), float(p_meas[r]), merge_strategy)
        if not use_virtual_boundary_node and g.num_nodes == num_detectors + 1:
            g.set_boundary([num_detectors])
        self._g, self._corr, self._dem_text = g, None, None

    @staticmethod
    def from_networkx(graph: Any, *, min_num_fault_ids: Optional[int] = None) -> "Matching":
        m = Matching()
        m.load_from_networkx(graph, min_num_fault_ids=min_num_fault_ids)
        return m

    def load_from_networkx(self, graph: Any, *, min_num_fault_ids: Optional[int] = None) -> None:
        """Edges from a networkx graph's edges and their ``fault_ids``, ``weight`` and
        ``error_probability`` attributes; nodes with ``is_boundary`` set are boundary nodes."""
        import networkx as nx

        if not isinstance(graph, nx.Graph):
            raise TypeError("G must be a NetworkX graph")
        boundary = [i for i, attr in graph.nodes(data=True) if attr.get("is_boundary", False)]
        self._load_edges(graph.number_of_nodes(), boundary, graph.edges(data=True), min_num_fault_ids)

    def load_from_retworkx(self, graph: Any, *, min_num_fault_ids: Optional[int] = None) -> None:
        warnings.warn("`pymatching.Matching.load_from_retworkx` is now deprecated since the `retworkx` library has been renamed to `rustworkx`. Please use `pymatching.Matching.load_from_rustworkx` instead.", DeprecationWarning, stacklevel=2)
        self.load_from_rustworkx(graph, min_num_fault_ids=min_num_fault_ids)

    def load_from_rustworkx(self, graph: Any, *, min_num_fault_ids: Optional[int] = None) -> None:
        """As ``load_from_networkx``, from a rustworkx ``PyGraph``."""
        try:
            import rustworkx as rx
        except ImportError:
            raise ImportError("rustworkx must be installed to use Matching.load_from_rustworkx") from None
        if not isinstance(graph, rx.PyGraph):
            raise TypeError("G must be a rustworkx graph")
        boundary = [i for i in graph.node_indices() if graph[i].get("is_boundary", False)]
        self._load_edges(len(graph), boundary, graph.weighted_edge_list(), min_num_fault_ids)

    def _load_edges(self, num_nodes: int, boundary: List[int], edges: Any, min_num_fault_ids: Optional[int]) -> None:
        g = _core.UserGraph(num_nodes, 0 if min_num_fault_ids is None else int(min_num_fault_ids))
        g.set_boundary(boundary)
        for u, v, attr in edges:
            if "fault_ids" in attr and "qubit_id" in attr:
                raise ValueError("Both `fault_ids` and `qubit_id` were provided as edge attributes, however use of `qubit_id` has been deprecated in favour of `fault_ids`. Please only supply `fault_ids` as an edge attribute.")
            faults = _fault_set(attr["qubit_id"] if "fault_ids" not in attr and "qubit_id" in attr else attr.get("fault_ids", set()))
            weight = float(attr.get("weight", 1))
            # As PyMatching's graph loaders: an edge too heavy to match is left out, with a
            # warning, and the rest of the graph decodes.
            if _too_heavy(weight, u, v, 3):
                continue
            call(g.add_edge, int(u), int(v), sorted(faults), weight, float(attr.get("error_probability", -1)), "smallest-weight")
        self._g, self._corr, self._dem_text = g, None, None

    # -- Editing and queries -------------------------------------------------------------

    def add_edge(self, node1: int, node2: int, fault_ids: Any = None, weight: float = 1.0, error_probability: Optional[float] = None, *, merge_strategy: str = "disallow", **kwargs: Any) -> None:
        """Adds an edge (PyMatching's ``add_edge``); ``merge_strategy`` says what happens to an
        edge already there: ``disallow`` (an error), ``independent``, ``smallest-weight``,
        ``keep-original`` or ``replace``."""
        if fault_ids is not None and "qubit_id" in kwargs:
            raise ValueError("Both `fault_ids` and `qubit_id` were provided as arguments. Please provide `fault_ids` instead of `qubit_id` as an argument, as use of `qubit_id` has been deprecated.")
        if fault_ids is None and "qubit_id" in kwargs:
            fault_ids = kwargs["qubit_id"]
        if node1 < 0 or node2 < 0:
            raise ValueError("Node indices must be non-negative.")
        if _too_heavy(weight, node1, node2, 3):
            return
        p = -1.0 if error_probability is None else float(error_probability)
        if call(self._g.add_edge, int(node1), int(node2), sorted(_fault_set(fault_ids)), float(weight), p, merge_strategy):
            self._corr = None

    def add_boundary_edge(self, node: int, fault_ids: Any = None, weight: float = 1.0, error_probability: Optional[float] = None, *, merge_strategy: str = "disallow", **kwargs: Any) -> None:
        """Adds an edge from ``node`` to the boundary (PyMatching's ``add_boundary_edge``)."""
        if node < 0:
            raise ValueError("Node index must be non-negative.")
        if _too_heavy(weight, node, None, 3):
            return
        p = -1.0 if error_probability is None else float(error_probability)
        if call(self._g.add_boundary_edge, int(node), sorted(_fault_set(fault_ids)), float(weight), p, merge_strategy):
            self._corr = None

    def set_boundary_nodes(self, nodes: Set[int]) -> None:
        """The boundary nodes: decoding treats an edge to one as an edge to the boundary."""
        self._g.set_boundary(sorted(int(n) for n in nodes))
        self._corr = None

    def ensure_num_fault_ids(self, min_num_fault_ids: int) -> None:
        self._g.ensure_num_fault_ids(int(min_num_fault_ids))

    @property
    def boundary(self) -> Set[int]:
        return set(self._g.boundary())

    @property
    def num_nodes(self) -> int:
        return self._g.num_nodes

    @property
    def num_edges(self) -> int:
        return self._g.num_edges

    @property
    def num_fault_ids(self) -> int:
        return self._g.num_fault_ids

    @property
    def num_detectors(self) -> int:
        return self._g.num_detectors

    @property
    def num_observables(self) -> int:
        """``num_fault_ids``, by this package's other decoders' name."""
        return self._g.num_fault_ids

    def has_edge(self, node1: int, node2: int) -> bool:
        return self._g.edge_data(int(node1), int(node2)) is not None

    def has_boundary_edge(self, node: int) -> bool:
        return self._g.edge_data(int(node), None) is not None

    def get_edge_data(self, node1: int, node2: int) -> Dict[str, Any]:
        if node1 >= self.num_nodes:
            raise ValueError(f"node1 ({node1}) not in graph")
        d = self._g.edge_data(int(node1), int(node2))
        if d is None:
            raise ValueError(f"Edge ({node1}, {node2}) not in graph.")
        return {"fault_ids": set(d[0]), "weight": d[1], "error_probability": d[2]}

    def get_boundary_edge_data(self, node: int) -> Dict[str, Any]:
        if node >= self.num_nodes:
            raise ValueError(f"node ({node}) not in graph")
        d = self._g.edge_data(int(node), None)
        if d is None:
            raise ValueError(f"Boundary edge ({node},) not in graph.")
        return {"fault_ids": set(d[0]), "weight": d[1], "error_probability": d[2]}

    def edges(self) -> List[Tuple[int, Optional[int], Dict[str, Any]]]:
        """Every edge, as ``(node1, node2 or None for the boundary, attributes)``."""
        return [(u, v, {"fault_ids": set(f), "weight": w, "error_probability": -1.0 if p < 0 or p > 1 else p}) for u, v, f, w, p in self._g.edges()]

    def to_networkx(self) -> Any:
        """A networkx graph of the edges; the boundary is node ``num_nodes`` if any edge meets it."""
        import networkx as nx

        graph = nx.Graph()
        n = self.num_nodes
        virtual = False
        for u, v, data in self.edges():
            if v is None:
                graph.add_edge(u, n, **data)
                virtual = True
            else:
                graph.add_edge(u, v, **data)
        boundary = self.boundary
        for i in graph.nodes:
            graph.nodes[i]["is_boundary"] = i in boundary
        if virtual:
            graph.nodes[n]["is_boundary"] = True
        return graph

    def to_retworkx(self) -> Any:
        warnings.warn("`pymatching.Matching.to_retworkx` is now deprecated since the `retworkx` library has been renamed to `rustworkx`. Please use `pymatching.Matching.to_rustworkx` instead.", DeprecationWarning, stacklevel=2)
        return self.to_rustworkx()

    def to_rustworkx(self) -> Any:
        """A rustworkx ``PyGraph`` of the edges, as ``to_networkx``."""
        try:
            import rustworkx as rx
        except ImportError:
            raise ImportError("rustworkx must be installed to use Matching.to_rustworkx.") from None
        graph = rx.PyGraph(multigraph=False)
        n = self.num_nodes
        virtual = False
        edges = []
        for u, v, data in self.edges():
            if v is None:
                edges.append((u, n, data))
                virtual = True
            else:
                edges.append((u, v, data))
        graph.add_nodes_from([{} for _ in range(n + virtual)])
        graph.extend_from_weighted_edge_list(edges)
        boundary = self.boundary
        for i in graph.node_indices():
            graph[i]["is_boundary"] = i in boundary
        if virtual:
            graph[n]["is_boundary"] = True
        return graph

    # -- Decoding ------------------------------------------------------------------------

    def _events(self, z: Any) -> np.ndarray:
        try:
            z = np.array(z, dtype=np.uint8)
        except ValueError:
            raise ValueError(f"Syndrome must be of type numpy.ndarray or convertible to numpy.ndarray, not {z}") from None
        nd = self.num_detectors
        if z.ndim == 1 and nd <= z.shape[0] <= nd + len(self.boundary):
            return z.nonzero()[0]
        if z.ndim == 2 and z.shape[0] * z.shape[1] == nd:
            times, checks = z.T.nonzero()
            return times * z.shape[0] + checks
        raise ValueError(f"The shape ({z.shape}) of the syndrome vector z is not valid.")

    def _check_correlations(self, enable_correlations: bool) -> bool:
        if not enable_correlations:
            return False
        # PyMatching's correlated calls first take its decoder with the search graph
        # (`get_mwpm_with_search_graph`), rebuilding one built without it, and only then decode
        # (or refuse); the plain decoder here follows, though the correlated matcher decodes.
        call(self._g.mwpm_with_search_graph)
        if self._corr is None:
            raise ValueError(
                "Attempting to decode with `enable_correlations=True`, however the decoder has not been configured to decode using correlations. Ensure that you also set `enable_correlations=True` when loading the stim circuit or detector error model. For example: `matching = pymatching.Matching.from_detector_error_model(dem, enable_correlations=True)`"
            )
        return True

    def decode(self, z: Any, *, return_weight: bool = False, enable_correlations: bool = False, **kwargs: Any) -> Any:
        """One syndrome (a 1-d array over the detectors, or 2-d over checks × rounds) to the
        fault ids its matching flips, as uint8 0/1; with ``return_weight``, also its weight."""
        events = self._events(z)
        if self._check_correlations(enable_correlations):
            dets = np.zeros((1, self.num_detectors), dtype=np.uint8)
            dets[0, events[events < self.num_detectors]] = 1
            pred, w = self.decode_batch(dets, return_weights=True, enable_correlations=True)
            return (pred[0], float(w[0])) if return_weight else pred[0]
        raw, weight = call(self._g.decode, [int(e) for e in events])
        correction = np.frombuffer(raw, dtype=np.uint8).copy()
        return (correction, weight) if return_weight else correction

    def decode_batch(
        self,
        shots: Any,
        *,
        return_weights: bool = False,
        bit_packed_shots: bool = False,
        bit_packed_predictions: bool = False,
        enable_correlations: bool = False,
        threads: int = 1,
    ) -> Any:
        """Shots (shots × detectors, uint8 0/1 or bool; or bit-packed rows) to predictions
        (shots × fault ids, uint8; or bit-packed), as PyMatching's ``decode_batch``; with
        ``return_weights``, each shot's matching weight too. ``threads = 0`` uses every core."""
        shots = np.asarray(shots)
        if shots.ndim != 2:
            raise ValueError(f"`shots` array should have two dimensions, not {shots.ndim}")
        nd, nn, nf = self.num_detectors, self.num_nodes, self.num_fault_ids
        n = shots.shape[0]
        if bit_packed_shots:
            lo, hi = (nd + 7) >> 3, (nn + 7) >> 3
            if not lo <= shots.shape[1] <= hi:
                raise ValueError(f"bit_packed `shots` array should have at least {lo} columns (ceil(num_detectors/8)) and at most {hi} columns, (ceil(num_nodes/8)). Instead it has {shots.shape[1]} columns.")
            packed = np.ascontiguousarray(shots, dtype=np.uint8)
        else:
            if not nd <= shots.shape[1] <= nn:
                raise ValueError(f"`shots` array should have at least {nd} columns (the number of detectors), and no more than {nn} columns (the number of nodes), but instead has {shots.shape[1]} columns")
            packed = np.packbits(shots.astype(np.uint8).astype(bool), axis=1, bitorder="little")
        if self._check_correlations(enable_correlations):
            dets = np.unpackbits(packed, axis=1, count=nd, bitorder="little")
            raw = np.packbits(dets, axis=1, bitorder="little").tobytes()
            preds, weights, failed = call(self._corr.decode_batch, raw, n, count(threads, "threads"))
            if failed:
                raise ValueError("No perfect matching could be found. This likely means that the syndrome has odd parity in the support of a connected component without a boundary.")
            from ._util import u64_to_rows

            out = u64_to_rows(preds, n, min(nf, 64), bit_packed_predictions)
            width = (nf + 7) // 8 if bit_packed_predictions else nf
            if out.shape[1] < width:
                out = np.pad(out, ((0, 0), (0, width - out.shape[1])))
            w = np.frombuffer(weights, dtype="<f8").copy()
            return (out, w) if return_weights else out
        preds, weights = call(self._g.decode_batch, packed.tobytes(), n, packed.shape[1], count(threads, "threads"))
        out = np.frombuffer(preds, dtype=np.uint8).reshape(n, (nf + 7) // 8)
        out = out.copy() if bit_packed_predictions else np.unpackbits(out, axis=1, count=nf, bitorder="little")
        if return_weights:
            return out, np.frombuffer(weights, dtype="<f8").copy()
        return out

    def decode_async(self, shots: Any, **kwargs: Any) -> Any:
        """``decode_batch(shots, **kwargs)`` on a background thread: a ``realtime.AsyncResult``
        whose ``get()`` is the predictions. A single syndrome gives one row."""
        from .realtime import decode_async

        s = np.asarray(shots)
        return decode_async(self, s.reshape(1, -1) if s.ndim == 1 else shots, **kwargs)

    def decode_to_edges_array(self, syndrome: Any, *, enable_correlations: bool = False) -> np.ndarray:
        """The edges the correction flips, as an (edges × 2) int64 array of node pairs (-1: the
        boundary)."""
        events = self._events(syndrome)
        if self._check_correlations(enable_correlations):
            edges = call(self._corr.decode_to_edges, [int(e) for e in events[events < self.num_detectors]])
            return np.array(edges, dtype=np.int64).reshape(-1, 2)
        edges = call(self._g.decode_to_edges, [int(e) for e in events])
        return np.array(edges, dtype=np.int64).reshape(-1, 2)

    def decode_to_matched_dets_array(self, syndrome: Any) -> np.ndarray:
        """The matched pairs of detection events, as an (pairs × 2) int64 array (-1: the
        boundary)."""
        pairs = call(self._g.decode_to_matched_pairs, [int(e) for e in self._events(syndrome)])
        return np.array(pairs, dtype=np.int64).reshape(-1, 2)

    def decode_to_matched_dets_dict(self, syndrome: Any) -> Dict[int, Optional[int]]:
        """The same as a dict: each detection event to its partner (None: the boundary)."""
        out: Dict[int, Optional[int]] = {}
        for a, b in call(self._g.decode_to_matched_pairs, [int(e) for e in self._events(syndrome)]):
            if b == -1:
                out[a] = None
            else:
                out[a] = b
                out[b] = a
        return out

    def add_noise(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Samples each edge's error with its probability (None unless every edge has one):
        ``(noise over the fault ids, syndrome over the nodes)``, the boundary nodes' zeroed.
        Seeded by ``set_seed``."""
        if not self._g.all_edges_have_error_probabilities():
            return None
        noise = np.zeros(self.num_fault_ids, dtype=np.uint8)
        syndrome = np.zeros(self.num_nodes, dtype=np.uint8)
        for u, v, f, _, p in self._g.edges():
            if _rng.random() < p:
                for k in f:
                    noise[k] ^= 1
                syndrome[u] ^= 1
                if v is not None:
                    syndrome[v] ^= 1
        for b in self.boundary:
            syndrome[b] = 0
        return noise, syndrome

    def draw(self) -> None:
        """Draws the graph with matplotlib and networkx (as PyMatching's ``draw``)."""
        import matplotlib
        import networkx as nx

        warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        G = self.to_networkx()
        pos = nx.spectral_layout(G, weight=None)
        c = "#bfbfbf"
        ncolors = ["w" if n[1]["is_boundary"] else c for n in G.nodes(data=True)]
        nx.draw_networkx_nodes(G, pos=pos, node_color=ncolors, edgecolors=c)
        nx.draw_networkx_labels(G, pos=pos)
        weights = np.array([e[2]["weight"] for e in G.edges(data=True)])
        normalised_weights = 0.2 + 2 * weights / np.max(weights)
        nx.draw_networkx_edges(G, pos=pos, width=normalised_weights)

        def qid_to_str(qid: Set[int]) -> str:
            if len(qid) == 0:
                return ""
            if len(qid) == 1:
                return str(next(iter(qid)))
            return str(qid)

        edge_labels = {(s, t): qid_to_str(d["fault_ids"]) for s, t, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels)

    def __repr__(self) -> str:
        m, b, e = self.num_detectors, len(self.boundary), self.num_edges
        return f"<stabilizer_qec.Matching object with {m} detector{'s' if m != 1 else ''}, {b} boundary node{'s' if b != 1 else ''}, and {e} edge{'s' if e != 1 else ''}>"

    def __reduce__(self) -> tuple:
        dem = self._dem_text if self._corr is not None else None
        state = (self.num_nodes, self.num_fault_ids, sorted(self.boundary), self._g.edges(), dem)
        implied = self._g.implied_weights() if dem is None else []
        if type(self) is Matching:
            return (_rebuild, state + (implied,) if implied else state)
        # A subclass is rebuilt as itself, with its own attributes, as PyMatching's copy keeps
        # them (its decode and the rest overridden as they were).
        extra = getattr(self, "__dict__", None)
        return (_rebuild_as, (type(self),) + state + (implied,), dict(extra) if extra else None)


def _rebuild_as(cls: type, *state: Any) -> Matching:
    """A ``Matching`` subclass's copy: ``_rebuild``'s graph in an instance of ``cls``."""
    m = _rebuild(*state)
    out = cls.__new__(cls)
    out._g, out._corr, out._dem_text = m._g, m._corr, m._dem_text
    return out


def _rebuild(num_nodes: int, num_fault_ids: int, boundary: list, edges: list, correlated_dem: Optional[str], implied: Any = ()) -> Matching:
    if correlated_dem is not None:
        m = Matching.from_detector_error_model(correlated_dem, enable_correlations=True)
        m.ensure_num_fault_ids(num_fault_ids)
        return m
    m = Matching()
    g = _core.UserGraph(num_nodes, num_fault_ids)
    g.set_boundary(boundary)
    for u, v, f, w, p in edges:
        if v is None:
            g.add_boundary_edge(u, f, w, p, "disallow")
        else:
            g.add_edge(u, v, f, w, p, "disallow")
    # The weights a correlated load implied (kept by a graph changed since), which its scale counts.
    by_edge: Dict[int, list] = {}
    for e, u, v, w in implied:
        by_edge.setdefault(e, []).append((u, v, w))
    for e, rules in by_edge.items():
        g.set_implied_weights(e, rules)
    m._g = g
    return m


class _CorrelatedMatching:
    """A model's ``Matching`` that always decodes with correlations (for the decoder kinds
    named ``correlated_matching``)."""

    __slots__ = ("matching", "__weakref__")

    def __init__(self, model: Any) -> None:
        self.matching = Matching.from_detector_error_model(model, enable_correlations=True)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.matching, name)

    def decode(self, z: Any, **kwargs: Any) -> Any:
        return self.matching.decode(z, enable_correlations=True, **kwargs)

    def decode_async(self, shots: Any, **kwargs: Any) -> Any:
        from .realtime import decode_async

        s = np.asarray(shots)
        return decode_async(self, s.reshape(1, -1) if s.ndim == 1 else shots, **kwargs)

    def decode_batch(self, shots: Any, **kwargs: Any) -> Any:
        return self.matching.decode_batch(shots, enable_correlations=True, **kwargs)

    def __reduce__(self) -> tuple:
        return (_CorrelatedMatching, (self.matching._dem_text,))

