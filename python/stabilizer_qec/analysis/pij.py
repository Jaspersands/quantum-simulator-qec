"""Error models from detection events: deltakit's p_ij method (Google's, "Exponential
suppression of bit or phase errors with cyclic error correction", Nature 595 (2021),
supplement), with this package's models in place of deltakit's decoding graphs.

``generate_expectation_data`` counts how often each detector, and each pair (up to
``max_degree``-tuple) of detectors, fires together; ``calculate_pij_values`` turns those into
the probability of each edge (or hyperedge of up to four detectors) of a model;
``create_dem_from_pij`` writes the model with those probabilities. The comparison helpers
(``pijs_edge_diff``, ``pij_edges_max_diff``, ``pij_and_dem_edge_diff``,
``dem_and_pij_edges_max_diff``), ``create_correlation_matrix`` and
``plot_correlation_matrix`` are deltakit's. The formulas are deltakit-decode 0.9.2's
(Copyright Riverlane, Apache License 2.0; see THIRD_PARTY_NOTICES.md)."""

from __future__ import annotations

import math
import warnings
from collections import Counter, defaultdict
from itertools import chain, combinations, islice, product
from typing import Any, Callable, Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

PijData = Dict[FrozenSet[int], float]


# -- Expectation values -------------------------------------------------------------------


def generate_expectation_data(samples: Any, only_even: bool = False, only_odd: bool = False, max_degree: int = 2, num_processes: int = 1) -> PijData:
    """The fraction of shots in which each set of up to ``max_degree`` detectors all fire:
    ``{frozenset({i}): <x_i>, frozenset({i, j}): <x_i x_j>, ...}``. ``samples`` is each shot's
    fired detectors (lists of indices), or a (shots, detectors) bool array.
    ``only_even``/``only_odd`` keep every other shot (deltakit's options). ``num_processes`` is
    accepted for deltakit's signature; the counting runs in this process."""
    if isinstance(samples, np.ndarray) and samples.ndim == 2:
        samples = [np.flatnonzero(row).tolist() for row in samples]
    if only_even and only_odd:
        warnings.warn("Both only_odd and only_even are True. Selecting whole batch.", stacklevel=2)
    elif only_even:
        samples = islice(samples, 0, None, 2)
    elif only_odd:
        samples = islice(samples, 1, None, 2)
    counter: Counter = Counter()
    num_samples = 0
    for s in samples:
        s = sorted(s)
        counter.update(chain.from_iterable(combinations(s, weight) for weight in range(1, max_degree + 1)))
        num_samples += 1
    return {frozenset(key): counter[key] / num_samples for key in counter}


# -- The formulas (deltakit's) ------------------------------------------------------------


def _n1_formula(exp_values: PijData, keys: Sequence[FrozenSet[int]]) -> float:
    return math.prod((1 - (2 * exp_values.get(n, 0.0))) for n in keys)


def _d1_formula(exp_values: PijData, keys: Sequence[FrozenSet[int]]) -> float:
    tuple_keys = [tuple(x) for x in keys]
    return math.prod(
        1 - (2 * exp_values.get(frozenset(n[:1]), 0.0)) - (2 * exp_values.get(frozenset(n[1:]), 0.0)) + (4 * exp_values.get(frozenset(n), 0.0))
        for n in tuple_keys
    )


def _n2_formula(exp_values: PijData, key: FrozenSet[int], keys: Sequence[FrozenSet[int]]) -> float:
    return (
        1
        - (2 * math.fsum(exp_values.get(frozenset((n,)), 0.0) for n in key))
        + (4 * math.fsum(exp_values.get(x, 0.0) for x in keys))
        - (8 * exp_values.get(key, 0.0))
    )


def _calculate_pij_degree_2(key: FrozenSet[int], exp_values: PijData) -> float:
    xi_key, xj_key = (frozenset((x,)) for x in key)
    numerator = float(_n1_formula(exp_values, [xi_key, xj_key]))
    denominator = _d1_formula(exp_values, [key])
    return 0.5 - 0.5 * math.sqrt(numerator / denominator)


def _calculate_pij_degree_3(key: FrozenSet[int], exp_values: PijData) -> float:
    key_combinations = [frozenset(x) for x in combinations(key, 2)]
    i, j, k = (frozenset((x,)) for x in key)
    n1 = _n1_formula(exp_values, [i, j, k])
    n2 = _n2_formula(exp_values, key, key_combinations)
    d1 = _d1_formula(exp_values, key_combinations)
    inner_b = math.pow(((n1 * n2) / d1), 0.25)
    return 0.5 * (1 - inner_b)


def _calculate_pij_degree_4(key: FrozenSet[int], exp_values: PijData) -> float:
    w2_nodes = [frozenset(x) for x in combinations(key, 2)]
    w3_nodes = [frozenset(x) for x in combinations(key, 3)]
    i, j, k, m = (frozenset((x,)) for x in key)
    n1 = _n1_formula(exp_values, [i, j, k, m])
    n2 = math.prod(_n2_formula(exp_values, n3, [frozenset(x) for x in combinations(n3, 2)]) for n3 in w3_nodes)
    d1 = _d1_formula(exp_values, w2_nodes)
    d2 = (
        1
        - (2 * math.fsum(exp_values.get(n, 0.0) for n in [i, j, k, m]))
        + (4 * math.fsum(exp_values.get(n2, 0.0) for n2 in w2_nodes))
        - (8 * math.fsum(exp_values.get(n3, 0.0) for n3 in w3_nodes))
        + (16 * exp_values.get(key, 0.0))
    )
    inner_b = math.pow((n1 * n2) / (d1 * d2), 1 / 8)
    return 0.5 * (1 - inner_b)


def _degree_formula(max_degree: int) -> Callable[[FrozenSet[int], PijData], float]:
    formulas = {2: _calculate_pij_degree_2, 3: _calculate_pij_degree_3, 4: _calculate_pij_degree_4}
    if max_degree not in formulas:
        raise NotImplementedError(f"{max_degree} is not a valid degree, must be between 2-4 inclusive.")
    return formulas[max_degree]


def _calculate_g_value(p: float, q: float) -> float:
    return p + q - (2 * p * q)


def _calculate_p_i_sigma(nodes: List[float]) -> List[float]:
    if len(nodes) == 1:
        return nodes
    if len(nodes) == 2:
        return [_calculate_g_value(nodes[0], nodes[1])]
    return _calculate_p_i_sigma([nodes[0], *_calculate_p_i_sigma(nodes[1:])])


def _calculate_edge_prob_with_higher_degrees(edge: FrozenSet[int], pij_data: PijData, min_prob: float) -> float:
    connected_edges = [pij_data[k] for k in pij_data if edge < k]
    if len(connected_edges) == 0:
        return pij_data.get(edge, 0.0)
    pi_sigma = _calculate_p_i_sigma(connected_edges)[0]
    if math.isclose(pi_sigma, 0.5, rel_tol=0.05):
        warnings.warn("pi_sigma is converging to 0.5. Please consider trimming the number of edges used in calculations.", stacklevel=3)
        return max(min_prob, 0.0)
    return (pij_data.get(edge, 0.0) - pi_sigma) / (1 - (2 * pi_sigma))


# -- Models as edge lists -----------------------------------------------------------------


def model_edges(model: Any, is_hypergraph: bool = False) -> Dict[FrozenSet[int], Set[int]]:
    """A model's edges, in order of first appearance: each error's detectors (graph-like
    pieces, split at ``^``, unless ``is_hypergraph``), mapped to the observables they flip."""
    out: Dict[FrozenSet[int], Set[int]] = {}
    for instr in model.flattened():
        if instr.type != "error":
            continue
        groups: List[List[Any]] = [[]]
        for t in instr.targets_copy():
            if t.is_separator():
                groups.append([])
            else:
                groups[-1].append(t)
        if is_hypergraph:
            groups = [[t for g in groups for t in g]]
        for g in groups:
            dets: Set[int] = set()
            obs: Set[int] = set()
            for t in g:
                (dets if t.is_relative_detector_id() else obs).symmetric_difference_update({t.val})
            if not dets or (not is_hypergraph and len(dets) > 2):
                continue
            out.setdefault(frozenset(dets), obs)
    return out


def calculate_pij_values(exp_values: PijData, graph: Any = None, min_prob: float = -math.inf, max_degree: int = 2, noise_floor_graph: Any = None, *, is_hypergraph: bool = False) -> PijData:
    """Each edge's probability from the expectation values: the edges of ``graph`` (a
    ``DetectorErrorModel``; graph-like pieces, or with ``is_hypergraph`` whole errors) of up to
    ``max_degree`` detectors, or every set in ``exp_values``. Edges of fewer detectors than
    ``max_degree`` are corrected for the larger ones containing them. ``noise_floor_graph``'s
    probabilities, and ``min_prob``, bound each result from below. As deltakit's."""
    pij_func = _degree_formula(max_degree)
    noise_floor_edges: PijData = {}
    if noise_floor_graph is not None:
        probs = _edge_probabilities(noise_floor_graph, is_hypergraph)
        noise_floor_edges = {k: v for k, v in probs.items() if len(k) <= max_degree}
    edges_to_calc = list(exp_values)
    if len(edges_to_calc) == 0:
        return {}
    if graph is not None:
        edges_to_calc = [e for e in model_edges(graph, is_hypergraph) if len(e) <= max_degree]
    pij_data: PijData = {}
    edges_for_recalc: List[FrozenSet[int]] = []
    for key in edges_to_calc:
        if len(key) < max_degree:
            edges_for_recalc.append(key)
            if len(key) == 1:
                pij_data[key] = exp_values[key]
            elif len(key) == 2:
                pij_data[key] = _calculate_pij_degree_2(key, exp_values)
            elif len(key) == 3:
                pij_data[key] = _calculate_pij_degree_3(key, exp_values)
            continue
        pij_val = pij_func(key, exp_values)
        pij_data[key] = max(noise_floor_edges.get(key, min_prob), pij_val)
    for edge in sorted(edges_for_recalc, reverse=True, key=len):
        adjusted_pi = _calculate_edge_prob_with_higher_degrees(edge, pij_data, min_prob)
        pij_data[edge] = max(noise_floor_edges.get(edge, min_prob), adjusted_pi)
    return pij_data


def _edge_probabilities(model: Any, is_hypergraph: bool = False) -> PijData:
    """Each edge's probability in the model, parallel errors combined (p + q - 2pq)."""
    out: PijData = {}
    for instr in model.flattened():
        if instr.type != "error":
            continue
        p = instr.args_copy()[0]
        groups: List[List[Any]] = [[]]
        for t in instr.targets_copy():
            if t.is_separator():
                groups.append([])
            else:
                groups[-1].append(t)
        if is_hypergraph:
            groups = [[t for g in groups for t in g]]
        for g in groups:
            dets: Set[int] = set()
            for t in g:
                if t.is_relative_detector_id():
                    dets.symmetric_difference_update({t.val})
            if not dets or (not is_hypergraph and len(dets) > 2):
                continue
            key = frozenset(dets)
            out[key] = _calculate_g_value(out[key], p) if key in out else p
    return out


def create_dem_from_pij(pij_data: PijData, graph: Any, logicals: Optional[Sequence[Iterable[FrozenSet[int]]]] = None, *, is_hypergraph: bool = False) -> Any:
    """A model with an error per edge of ``pij_data`` at its probability, flipping the
    observables the same edge of ``graph`` (a ``DetectorErrorModel``) flips, or those of
    ``logicals`` (for each observable, the edges that flip it); then ``graph``'s detectors'
    coordinates, and observables no edge flips declared."""
    from stabilizer_qec import DetectorErrorModel

    edges = model_edges(graph, is_hypergraph)
    if logicals is not None:
        flips: Dict[FrozenSet[int], Set[int]] = defaultdict(set)
        for i, edge_set in enumerate(logicals):
            for e in edge_set:
                flips[frozenset(e)].add(i)
        num_observables = len(logicals)
    else:
        flips = {e: o for e, o in edges.items()}
        num_observables = graph.num_observables
    lines = []
    for edge, p in pij_data.items():
        obs = sorted(flips.get(edge, ()))
        lines.append(f"error({p!r}) " + " ".join([f"D{i}" for i in sorted(edge)] + [f"L{i}" for i in obs]))
    coords = graph.get_detector_coordinates()
    for d in sorted(coords):
        if coords[d]:
            lines.append(f"detector({', '.join(repr(float(c)) for c in coords[d])}) D{d}")
    flipped = {o for e in pij_data for o in flips.get(e, ())}
    for i in range(num_observables):
        if i not in flipped:
            lines.append(f"logical_observable L{i}")
    return DetectorErrorModel("\n".join(lines))


# -- Comparisons ---------------------------------------------------------------------------


def pijs_edge_diff(pij1: PijData, pij2: PijData) -> Tuple[set, set]:
    """The edges only the first has, and those only the second has."""
    set_pij1, set_pij2 = set(pij1), set(pij2)
    return set_pij1.difference(set_pij2), set_pij2.difference(set_pij1)


def pij_edges_max_diff(pij1: PijData, pij2: PijData) -> float:
    """The largest difference in probability between the same edge of two."""
    first_diff, second_diff = pijs_edge_diff(pij1, pij2)
    if first_diff or second_diff:
        raise ValueError(f"Pij matrices do not contain identical edges so cannot compare for maximum difference. Diff: {first_diff, second_diff}")
    return max((abs(prob - pij2[edge]) for edge, prob in pij1.items()), default=0.0)


def pij_and_dem_edge_diff(dem: Any, pij: PijData, is_hypergraph: bool = False) -> Tuple[set, set]:
    """The model's edges that ``pij`` lacks, and ``pij``'s edges the model lacks."""
    graph_edges = set(model_edges(dem, is_hypergraph))
    pij_edges = set(pij)
    return graph_edges.difference(pij_edges), pij_edges.difference(graph_edges)


def dem_and_pij_edges_max_diff(dem: Any, pij: PijData, is_hypergraph: bool = False) -> float:
    """The largest difference in probability between an edge of the model and of ``pij``."""
    first_diff, second_diff = pij_and_dem_edge_diff(dem, pij, is_hypergraph=is_hypergraph)
    if first_diff or second_diff:
        raise ValueError(f"Pij matrices do not contain identical edges so cannot compare for maximum difference. Diff: {first_diff, second_diff}")
    graph_edges = _edge_probabilities(dem, is_hypergraph)
    return max((abs(prob - pij[edge]) for edge, prob in graph_edges.items()), default=0.0)


# -- The correlation matrix ------------------------------------------------------------------


def create_correlation_matrix(pij_data: PijData, graph: Any, plot_boundary_edges: bool = False) -> Tuple[np.ndarray, Dict[Tuple[float, ...], List[int]]]:
    """``pij_data`` as a matrix, detectors grouped by their spatial coordinates (all but the
    last, the time), and that grouping: as deltakit's (Google's Fig. 2c)."""
    mapping: Dict[Tuple[float, ...], List[int]] = defaultdict(list)
    for k, v in graph.get_detector_coordinates().items():
        mapping[tuple(v[:-1])].append(k)
    for detectors in mapping.values():
        detectors.sort()
    if len(mapping) == 0:
        return np.array([]), {}
    if not all(len(x) == len(next(iter(mapping.values()))) for x in mapping.values()):
        raise ValueError("Inconsistent qubit time mapping")

    def coord_to_pij(x: Tuple[int, int]) -> float:
        xf = frozenset((x[0],)) if x[0] == x[1] else frozenset(x)
        return pij_data.get(xf, 0.0)

    major = len(mapping)
    minor = len(next(iter(mapping.values())))
    matrix = np.zeros((major * minor, major * minor))
    for i, coord in enumerate(product(sorted(mapping), repeat=2)):
        for j, qubit in enumerate(product(mapping[coord[0]], mapping[coord[1]])):
            row = ((i // major) * minor) + (j // minor)
            col = ((i % major) * minor) + (j % minor)
            matrix[row][col] = coord_to_pij(qubit)
    if not plot_boundary_edges:
        np.fill_diagonal(matrix, 0.0)
    return matrix, dict(mapping)


def plot_correlation_matrix(matrix: Any, major_minor_mapping: Dict[Tuple[float, ...], List[int]], labels: Sequence[str] = (), ax: Any = None) -> Any:
    """The matrix as an image, a block per qubit (deltakit's plot, without seaborn). Returns
    the axes."""
    import matplotlib.pyplot as plt

    matrix = np.asarray(matrix, dtype=float)
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))
    image = ax.imshow(np.where(matrix > 0, matrix, np.nan), cmap="viridis", interpolation="nearest")
    plt.colorbar(image, ax=ax, label="$p_{ij}$")
    minor = len(next(iter(major_minor_mapping.values()))) if major_minor_mapping else 1
    for k in range(1, len(major_minor_mapping)):
        ax.axhline(k * minor - 0.5, color="white", linewidth=0.5)
        ax.axvline(k * minor - 0.5, color="white", linewidth=0.5)
    names = list(labels) or [str(c) for c in sorted(major_minor_mapping)]
    ticks = [k * minor + (minor - 1) / 2 for k in range(len(names))]
    ax.set_xticks(ticks, names, rotation=90)
    ax.set_yticks(ticks, names)
    return ax
