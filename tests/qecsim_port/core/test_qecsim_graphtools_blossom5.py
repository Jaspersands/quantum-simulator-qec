# Ported from qecsim 1.0b9 (qecsim/qecsim; David K. Tuckett, BSD 3-Clause; see THIRD_PARTY_NOTICES.md), modified for
# stabilizer_qec: qecsim is stabilizer_qec.qecsim.
import math

from stabilizer_qec.qecsim.graphtools import blossom5


def test_blossom5_available():
    available = blossom5.available()
    print(available)
    assert isinstance(available, bool)


def test_blossom5_infty():
    infty = blossom5.infty()
    print(infty)
    assert isinstance(infty, int)
    assert infty > 0


def test_blossom5_mwpm_ids():
    edges = [(1, 2, 10), (1, 3, 25), (0, 2, 56), (0, 1, 15), (2, 3, 6)]
    mates = blossom5.mwpm_ids(edges)
    expected = {(0, 1), (2, 3)}
    assert mates == expected


def test_blossom5_mwpm_ids_no_edges():
    edges = []
    mates = blossom5.mwpm_ids(edges)
    expected = set()
    assert mates == expected


def test_blossom5_mwpm_ids_negative_weights():
    edges = [(1, 2, -90), (1, 3, -75), (0, 2, -44), (0, 1, -85), (2, 3, -94)]
    mates = blossom5.mwpm_ids(edges)
    expected = {(0, 1), (2, 3)}
    assert mates == expected


def test_blossom5_mwpm():
    edges = [('b', 'c', 10), ('b', 'd', 25), ('a', 'c', 56), ('a', 'b', 15), ('c', 'd', 6)]
    mates = blossom5.mwpm(edges)
    sorted_mates = {tuple(sorted(match)) for match in mates}
    expected = {('a', 'b'), ('c', 'd')}
    assert sorted_mates == expected


def test_blossom5_weight_to_int_fn():
    weights = [0, 0, 0, 0, 4.394449154672438, 8.788898309344876, 10.986122886681095, 4.394449154672438,
               6.591673732008657, 2.197224577336219, 0, 0, 2.197224577336219, 6.591673732008657, 4.394449154672438, 0,
               4.394449154672438, 6.591673732008657, 2.197224577336219, 0, 0, 0, 0, 0, 2.197224577336219,
               6.591673732008657, 10.986122886681095, 4.394449154672438, 8.788898309344876, 4.394449154672438,
               2.197224577336219, 6.591673732008657, 10.986122886681095, 4.394449154672438, 8.788898309344876,
               4.394449154672438, 4.394449154672438, 6.591673732008657, 2.197224577336219, 2.197224577336219,
               6.591673732008657, 4.394449154672438, 4.394449154672438, 8.788898309344876, 10.986122886681095,
               4.394449154672438, 6.591673732008657, 2.197224577336219]
    _weight_to_int = blossom5.weight_to_int_fn(weights)
    scaled_int_weights = [_weight_to_int(wt) for wt in weights]
    assert max(scaled_int_weights) < blossom5.infty() / 10, (
        'Max absolute weight is not smaller than infty by 1 order of magnitude')
    ratio_min_to_max_weight = min(wt for wt in weights if wt != 0) / max(weights)
    ratio_min_to_max_scaled_weight = min(wt for wt in scaled_int_weights if wt != 0) / max(scaled_int_weights)
    assert math.isclose(ratio_min_to_max_weight, ratio_min_to_max_scaled_weight, rel_tol=1e-7), (
        'Ratio of min / max for non-zero weights and scaled weights is not close')


def test_blossom5_weight_to_int_fn_warning_zero(caplog):
    weights = [0, 2, blossom5.infty()]
    _weight_to_int = blossom5.weight_to_int_fn(weights)
    scaled_int_weights = [_weight_to_int(wt) for wt in weights]
    print(scaled_int_weights)
    print(caplog.text)
    assert 'SCALED MINIMUM ABSOLUTE NON-ZERO WEIGHT IS ZERO' in caplog.text


def test_blossom5_weight_to_int_fn_warning_less_than_3sf(caplog):
    weights = [0, 294, blossom5.infty()]
    _weight_to_int = blossom5.weight_to_int_fn(weights)
    scaled_int_weights = [_weight_to_int(wt) for wt in weights]
    print(scaled_int_weights)
    print(caplog.text)
    assert 'SCALED MINIMUM ABSOLUTE NON-ZERO WEIGHT LESS THAN 3 S.F.' in caplog.text


# (stabilizer_qec: without Blossom V's library, past the dense blossom's 512 nodes, Fusion Blossom stands in, exactly.)
def _random_graph(rng, n_nodes, extra, low=-50, high=1000):
    perm = rng.permutation(n_nodes).tolist()
    edges = [(perm[i], perm[i + 1], int(rng.integers(low, high))) for i in range(0, n_nodes, 2)]
    edges += [(int(a), int(b), int(rng.integers(low, high))) for a, b in rng.integers(0, n_nodes, (extra, 2)) if a != b]
    return edges


def _weight(edges, mates):
    lightest = {}
    for a, b, w in edges:
        key = (min(a, b), max(a, b))
        lightest[key] = min(w, lightest.get(key, w))
    assert len(mates) * 2 == len({x for m in mates for x in m}) and all(m in lightest for m in mates)
    return sum(lightest[m] for m in mates)


def test_blossom5_mwpm_ids_fusion_equals_dense_blossom():
    import numpy as np

    from stabilizer_qec import _core

    rng = np.random.default_rng(7)
    for n_nodes, extra in [(2, 0), (10, 30), (100, 0), (100, 600), (300, 3000), (120, None)]:
        edges = _random_graph(rng, n_nodes, extra or 0)
        if extra is None:  # the complete graph, a weight per pair (no triangle inequality)
            edges = [(a, b, int(rng.integers(0, 10 ** 6))) for a in range(n_nodes) for b in range(a + 1, n_nodes)]
        dense = {tuple(sorted(m)) for m in enumerate(_core.blossom_mwpm_ids(n_nodes, edges))}
        fusion = blossom5._mwpm_ids_fusion(n_nodes, edges)
        assert _weight(edges, fusion) == _weight(edges, dense), (n_nodes, extra)


def test_blossom5_mwpm_ids_past_512_nodes():
    import networkx as nx
    import numpy as np

    rng = np.random.default_rng(11)
    for n_nodes, extra in [(514, 1500), (1200, 4000)]:
        edges = _random_graph(rng, n_nodes, extra)
        mates = blossom5.mwpm_ids(edges)
        assert len(mates) == n_nodes // 2
        graph = nx.Graph()
        for a, b, w in edges:
            if not graph.has_edge(a, b) or 2000 - w > graph[a][b]['weight']:
                graph.add_edge(a, b, weight=2000 - w)
        expected = {tuple(sorted(m)) for m in nx.max_weight_matching(graph, maxcardinality=True)}
        assert _weight(edges, mates) == _weight(edges, expected)
    # no perfect matching (a star of 4 nodes; two triangles): as before, NetworkX's maximum-cardinality one
    edges = _random_graph(rng, 518, 2000)
    for extra, n_nodes, n_mates in [([(518, 519, 1), (518, 520, 1), (518, 521, 1)], 522, 260),
                                    ([(518, 519, 1), (519, 520, 1), (520, 518, 1), (521, 522, 1), (522, 523, 1), (523, 521, 1)], 524, 261)]:
        assert blossom5._mwpm_ids_fusion(n_nodes, edges + extra) is None
        assert len(blossom5.mwpm_ids(edges + extra)) == n_mates


def test_blossom5_run_ftp_rotated_planar_13(monkeypatch):
    # qecsim's run-ftp example, `run-ftp -r5 "rotated_planar(13,13)" 13 ...`, matches graphs of thousands of nodes.
    import numpy as np

    from stabilizer_qec.qecsim import app
    from stabilizer_qec.qecsim.models.generic import BitPhaseFlipErrorModel
    from stabilizer_qec.qecsim.models.rotatedplanar import RotatedPlanarCode, RotatedPlanarSMWPMDecoder

    sizes, fusion = [], blossom5._mwpm_ids_fusion

    def spy(n_nodes, edges):
        sizes.append(n_nodes)
        return fusion(n_nodes, edges)

    monkeypatch.setattr(blossom5, '_mwpm_ids_fusion', spy)
    data = app.run_once_ftp(RotatedPlanarCode(13, 13), 13, BitPhaseFlipErrorModel(), RotatedPlanarSMWPMDecoder(), 0.1,
                            rng=np.random.default_rng(5))
    assert data['success'] and sizes and min(sizes) > 512


class _HashedNode:
    """A node object hashed as given and equal only to itself (as the CMWPM and SMWPM decoders'
    cluster nodes are, whose hashes are their memory addresses)."""

    def __init__(self, label, hash_value):
        self.label, self.hash_value = label, hash_value

    def __hash__(self):
        return self.hash_value


def test_blossom5_mwpm_ties_independent_of_node_hashes():
    # stabilizer_qec: a ring of six equal edges has two minimum-weight perfect matchings. The
    # matching chosen depends only on the edges given (their nodes numbered in order of first
    # appearance), never on the nodes' hashes, so that a seeded run that matches identity-hashed
    # nodes repeats from process to process.
    import itertools

    chosen = set()
    for hashes in itertools.permutations(range(6)):
        nodes = [_HashedNode(label, h) for label, h in zip('abcdef', hashes)]
        edges = [(nodes[i], nodes[(i + 1) % 6], 1) for i in range(6)]
        chosen.add(frozenset(frozenset((a.label, b.label)) for a, b in blossom5.mwpm(edges)))
    assert len(chosen) == 1


def test_blossom5_seeded_smwpm_and_cmwpm_runs_repeat():
    # stabilizer_qec: the same seed gives the same result each time, for the decoders that match
    # identity-hashed nodes through the Blossom V interface.
    from stabilizer_qec.qecsim import app
    from stabilizer_qec.qecsim.models.generic import BiasedDepolarizingErrorModel, DepolarizingErrorModel
    from stabilizer_qec.qecsim.models.planar import PlanarCMWPMDecoder, PlanarCode
    from stabilizer_qec.qecsim.models.rotatedplanar import RotatedPlanarCode, RotatedPlanarSMWPMDecoder

    for code, error_model, decoder, p in ((RotatedPlanarCode(4, 4), BiasedDepolarizingErrorModel(10), RotatedPlanarSMWPMDecoder(), 0.12),
                                          (PlanarCode(5, 5), DepolarizingErrorModel(), PlanarCMWPMDecoder(), 0.12)):
        runs = [app.run(code, error_model, decoder, p, max_runs=150, random_seed=5) for _ in range(3)]
        assert len({(r['n_fail'], tuple(r['n_logical_commutations'])) for r in runs}) == 1, [r['n_fail'] for r in runs]
