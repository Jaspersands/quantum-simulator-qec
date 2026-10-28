# Ported from qecsim 1.0b9 (qecsim/qecsim; David K. Tuckett, BSD 3-Clause; see THIRD_PARTY_NOTICES.md), modified for
# stabilizer_qec: qecsim is stabilizer_qec.qecsim; without Blossom V's library, this package's exact
# matcher (src/blossom.rs) stands in for it, so the Blossom V interface is always available; mwpm numbers the nodes in
# order of first appearance (so a seeded run repeats).
"""
This module provides a somewhat low-level Python interface for Blossom V, a fast C++ implementation of minimum-weight
perfect matching in a graph, due to Vladimir Kolmogorov. (Typically, this module is not invoked directly, see rather the
functions of :mod:`qecsim.graphtools`.)

    Vladimir Kolmogorov. "Blossom V: A new implementation of a minimum cost perfect matching algorithm."
    In *Mathematical Programming Computation* (MPC), July 2009, 1(1):43-67.

The licence for Blossom V does not permit public redistribution of the code or its derivatives (see the Blossom V site
http://pub.ist.ac.at/~vnk/software.html and the Blossom V README.TXT file for full details of the license and citation
requirements). Therefore, Blossom V is not packaged with qecsim.

If your use case satisfies the license requirements of Blossom V, see section :ref:`install_blossom` for details on how
to configure qecsim to use Blossom V. When performance is not a concern, the Python package NetworkX,
https://networkx.github.io/, may be sufficient and is already installed as a qecsim dependency.

The functions in this module assume the presence of a C++ library ``libpypm.so`` in one of the locations searched by
:func:`qecsim.util.load_clib`. All functions, except :func:`available` will fail with an :class:`OSError` if the library
cannot be loaded.
"""
import ctypes
import decimal
import functools
import logging

from stabilizer_qec.qecsim import util

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def _libpypm():
    """C++ pypm library (lazy-loaded), with configured function argument and return types.

    :return: C++ pypm library
    :rtype: ctypes.CDLL
    :raises OSError: if Blossom V library cannot be loaded.
    """
    # load pypm library
    lib = util.load_clib('libpypm.so')
    # declare function argument types
    lib.infty.argtypes = None
    lib.infty.restype = ctypes.c_int
    lib.mwpm.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.POINTER(ctypes.c_int),
                         ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int))
    lib.mwpm.restype = None
    return lib


@functools.lru_cache(maxsize=1)
def _have_libpypm():
    try:
        _libpypm()
        return True
    except OSError:
        return False


# The stand-in's infinity (Blossom V's own when its library is loaded).
_INFTY = (1 << 30) - 1


@functools.lru_cache(maxsize=1)
def available():
    """
    :return: Availability of Blossom V (its library, or this package's exact matcher standing in).
    :rtype: bool
    """
    return True


@functools.lru_cache(maxsize=1)
def infty():
    """Integer that represents infinity (Blossom V algorithm).

    Note:

    * This can be useful when converting float to int weights for MWPM.

    :return: Value that represents infinity.
    :rtype: int
    :raises OSError: if Blossom V library cannot be loaded.
    """
    return _libpypm().infty() if _have_libpypm() else _INFTY


def mwpm_ids(edges):
    """Minimum Weight Perfect Matching using node ids (Blossom V algorithm).

    Notes:

    * Node ids are assumed to form a contiguous set of non-negative integers starting at zero, e.g.  {0, 1, ...}.
    * All nodes are assumed to participate in at least one edge.

    :param edges: Edges as [(node_id, node_id, weight), ...].
    :type edges: list of (int, int, int)
    :return: Set of matches as {(node_id, node_id), ...}. (Each tuple is sorted.)
    :rtype: set of (int, int)
    :raises OSError: if Blossom V library cannot be loaded.
    """
    # extract and sort node ids
    node_ids = sorted(set(id for (id_a, id_b, _) in edges for id in (id_a, id_b)))
    # count n_nodes
    n_nodes = len(node_ids)
    # check node ids form contiguous set of non-negative integers starting at zero
    assert n_nodes == 0 or (node_ids[0] == 0 and node_ids[-1] == n_nodes - 1), (
        'Node ids are not a contiguous set of non-negative integers starting at zero.')
    # count n_edges
    n_edges = len(edges)
    # unzip edges
    nodes_a, nodes_b, weights = zip(*edges) if n_edges else ([], [], [])
    if not _have_libpypm():
        return _mwpm_ids_standin(n_nodes, list(edges))
    # prepare array types
    mates_array_type = ctypes.c_int * n_nodes
    edges_array_type = ctypes.c_int * n_edges
    # prepare empty mates
    mates_array = mates_array_type()
    # call C interface
    _libpypm().mwpm(ctypes.c_int(n_nodes), mates_array, ctypes.c_int(n_edges),
                    edges_array_type(*nodes_a), edges_array_type(*nodes_b), edges_array_type(*weights))
    # structure of mates: mates_array[i] = j means ith node matches jth node
    # convert to more useful format: e.g. convert [1, 0, 3, 2] to {(0, 1), (2, 3)}
    mates = {tuple(sorted((a, b))) for a, b in enumerate(mates_array)}
    return mates


def mwpm(edges):
    """Minimum Weight Perfect Matching using node objects (Blossom V algorithm).

    :param edges: Edges as [(node, node, weight), ...].
    :type edges: list of (object, object, int)
    :return: Set of matches as {(node, node), ...}.
    :rtype: set of (object, object)
    :raises OSError: if Blossom V library cannot be loaded.
    """
    # list of nodes without duplicates
    # (stabilizer_qec: in order of first appearance, not a set's order: the CMWPM and SMWPM decoders' nodes hash by
    # identity, so a set's order, and with it which of equal-weight matchings is returned, changed from process to
    # process, and a seeded run did not repeat.)
    nodes = list(dict.fromkeys(node for (node_a, node_b, _) in edges for node in (node_a, node_b)))
    # dict of node to id
    node_to_id = dict((n, i) for i, n in enumerate(nodes))
    # edges using ids
    edge_ids = [(node_to_id[node_a], node_to_id[node_b], weight) for node_a, node_b, weight in edges]
    # mwpm using ids
    mate_ids = mwpm_ids(edge_ids)
    # matches using objects
    mates = {(nodes[node_id_a], nodes[node_id_b]) for node_id_a, node_id_b in mate_ids}
    return mates


def weight_to_int_fn(weights):
    """Given an iterable of weights, return a function that scales all weights by a multiplicative factor and rounds to
    integers such that the largest (absolute) weight is an order of magnitude smaller than :func:`infty`.

    Notes:

    * The returned function is useful to convert float weights to ints for the Blossom V implementation.
    * If all weights are integers and the largest (absolute) weight is at least an order of magnitude smaller than
      :func:`infty` then the identity function is returned.
    * If the scaling function would scale the smallest (absolute) non-zero weight to zero or less than 3 significant
      figures, then a warning is logged.

    :param weights: Weights (positive and/or negative)
    :type weights: iterable of int or float
    :return: Function with signature weight_to_int(float or int) -> int
    :rtype: function
    :raises OSError: if Blossom V library cannot be loaded.
    """
    # extract absolute non-zero weights
    abs_non_zero_wts = {abs(wt) for wt in weights if wt != 0}

    # if all weights are zero
    if len(abs_non_zero_wts) == 0:
        # return zero (int) function
        return lambda wt: 0

    # extract smallest and largest absolute non-zero weights
    min_abs_non_zero_wt = min(abs_non_zero_wts)
    max_abs_non_zero_wt = max(abs_non_zero_wts)

    # if largest (absolute) weight is less than "infty" and all weight are ints
    if max_abs_non_zero_wt < infty() / 10 and all(isinstance(wt, int) for wt in weights):
        # return identity function
        return lambda wt: wt

    # define scaling so largest (absolute) weight is an order of magnitude smaller than "infty"
    scaling = infty() / 10 / max_abs_non_zero_wt

    # define _weight_to_int using scaling
    def _weight_to_int(weight):
        # multiply weight by scaling (round to nearest with ties going away from zero).
        return int(decimal.Decimal(weight * scaling).to_integral_value(rounding=decimal.ROUND_HALF_UP))

    # warn if smallest (absolute) weight is zero or less than 3 significant figures.
    scaled_min_abs_non_zero_wt = _weight_to_int(min_abs_non_zero_wt)
    if scaled_min_abs_non_zero_wt == 0:
        logger.warning('SCALED MINIMUM ABSOLUTE NON-ZERO WEIGHT IS ZERO')
    elif scaled_min_abs_non_zero_wt < 100:
        logger.warning('SCALED MINIMUM ABSOLUTE NON-ZERO WEIGHT LESS THAN 3 S.F.:{}'.format(scaled_min_abs_non_zero_wt))

    return _weight_to_int


def _mwpm_ids_standin(n_nodes, edges):
    """Minimum weight perfect matching by this package's exact matchers, in Blossom V's form:
    {(node_id, node_id), ...}, each tuple sorted.

    Up to 512 nodes, the dense blossom (src/blossom.rs, which caps its O(n^2) memory and O(n^3)
    time there); past it, Fusion Blossom on the same graph with every node a defect, which is
    exact too and stays fast at thousands of nodes. (NetworkX only if neither applies.)"""
    if n_nodes == 0:
        return set()
    from stabilizer_qec import _core

    if n_nodes <= 512:
        mates = _core.blossom_mwpm_ids(n_nodes, [(int(a), int(b), int(w)) for a, b, w in edges])
        return {tuple(sorted((a, b))) for a, b in enumerate(mates)}
    mates = _mwpm_ids_fusion(n_nodes, edges)
    if mates is not None:
        return mates
    import networkx as nx

    graph = nx.Graph()
    top = max(abs(w) for _, _, w in edges) + 1 if edges else 1
    for a, b, w in edges:
        graph.add_edge(a, b, weight=top - w)
    mates = nx.max_weight_matching(graph, maxcardinality=True)
    return {tuple(sorted(m)) for m in mates}


def _mwpm_ids_fusion(n_nodes, edges):
    """Minimum weight perfect matching by Fusion Blossom, or None when the graph has no perfect
    matching (or its weights would overflow).

    Fusion Blossom matches defects along shortest paths. With every node a defect and each weight
    raised to 2(w - low + M), M = (n/2)W + 1 (W the weight range), a matching that uses any path
    of two or more edges costs at least 2M(n/2 + 1), more than any perfect matching of direct
    edges (at most n(W + M)); so its optimum is one of direct edges, and minimum in the original
    weights too, since every perfect matching's raised weight is twice its own plus n(M - low)."""
    best = {}
    for a, b, w in edges:  # (the lightest of parallel edges; no loops)
        a, b, w = int(a), int(b), int(w)
        key = (min(a, b), max(a, b))
        if a != b and (key not in best or w < best[key]):
            best[key] = w
    if n_nodes % 2 or not best:
        return None
    # each connected component needs an even number of nodes (else Fusion Blossom, with no
    # boundary to match to, has no answer)
    parent = list(range(n_nodes))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in best:
        parent[find(a)] = find(b)
    sizes = {}
    for x in range(n_nodes):
        sizes[find(x)] = sizes.get(find(x), 0) + 1
    if any(size % 2 for size in sizes.values()):
        return None
    low = min(best.values())
    span = max(best.values()) - low
    big = (n_nodes // 2) * span + 1
    if 2 * (span + big) * n_nodes >= 1 << 62:
        return None
    from stabilizer_qec import fusion_blossom as fb

    solver = fb.SolverSerial(fb.SolverInitializer(n_nodes, [(a, b, 2 * (w - low + big)) for (a, b), w in best.items()], []))
    solver.solve(fb.SyndromePattern(list(range(n_nodes))))
    mates = {tuple(sorted(pair)) for pair in solver.perfect_matching().peer_matchings}
    if len(mates) != n_nodes // 2 or len({x for pair in mates for x in pair}) != n_nodes or any(pair not in best for pair in mates):
        return None  # (no perfect matching of direct edges: the graph has none)
    return mates
