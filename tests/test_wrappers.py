"""The built-in wrappers: ``CompositeDecoder`` (a model's independent parts decoded apart, exact)
and ``SubgraphDecoder`` (qLDPC's: chosen detector subgraphs, predictions combined by exclusive
or)."""

import functools
import pickle
import warnings

import numpy as np
import pytest

import stabilizer_qec as sq


def side_by_side(*dems):
    """Models of separate blocks as one: detectors and observables numbered on."""
    lines, nd, no = [], 0, 0
    for dem in dems:
        for inst in dem.flattened():
            if inst.type == "error":
                ts = []
                for t in inst.targets_copy():
                    ts.append("^" if t.is_separator() else f"D{t.val + nd}" if t.is_relative_detector_id() else f"L{t.val + no}")
                lines.append(f"error({inst.args_copy()[0]!r}) " + " ".join(ts))
        nd += dem.num_detectors
        no += dem.num_observables
        lines.append(f"detector D{nd - 1}")
    return sq.DetectorErrorModel("\n".join(lines))


def shots(dem, n, seed):
    dets, obs, _ = dem.compile_sampler(seed=seed).sample(n)
    return dets, obs


@pytest.mark.parametrize("name", ["matching", "union_find", "bposd"])
def test_composite_of_blocks_is_each_block_decoded_alone(name):
    # Graph decoders on decomposed models, whose components are each block's two sectors;
    # BP+OSD on whole faults, which tie a block's sectors together, so the blocks are the parts.
    decompose = name != "bposd"
    a = sq.memory_circuit(distance=3, rounds=3, p=0.01).detector_error_model(decompose_errors=decompose)
    b = sq.memory_circuit(distance=5, rounds=2, p=0.004, basis="x").detector_error_model(decompose_errors=decompose)
    dem = side_by_side(a, b)
    dets, obs = shots(dem, 3000, 1)
    # (BP+OSD as a callable, cut to 30 iterations to keep the test short)
    cls = {"matching": sq.Matching, "bposd": functools.partial(sq.BpOsd, max_iter=30), "union_find": sq.UnionFind}[name]
    comp = sq.CompositeDecoder(dem, decoder=name if decompose else cls)
    if decompose:
        # the sectors no observable crosses are not decoded
        assert comp.num_subgraphs == 4 and sorted(o for _, o in comp.subgraphs) == [[], [], [0], [1]], comp.subgraphs
    else:
        assert [o for _, o in comp.subgraphs] == [[0], [1]]
    alone = np.concatenate([cls(a).decode_batch(dets[:, : a.num_detectors]), cls(b).decode_batch(dets[:, a.num_detectors :])], axis=1)
    np.testing.assert_array_equal(comp.decode_batch(dets, threads=2), alone)
    if name == "matching":
        np.testing.assert_array_equal(alone, sq.Matching(dem).decode_batch(dets))
    assert 0 < int((alone != obs).any(axis=1).sum()) < 600


def test_decomposed_faults_keep_sectors_apart():
    # An undecomposed model joins a code's X and Z sectors through its Y faults; decomposed,
    # the sectors are separate components.
    c = sq.memory_circuit(distance=3, rounds=3, p=0.01)
    assert sq.CompositeDecoder(c.detector_error_model(decompose_errors=True)).num_subgraphs == 2
    assert sq.CompositeDecoder(c.detector_error_model(), decoder="bposd").num_subgraphs == 1


def test_subgraphs_as_given():
    a = sq.memory_circuit(distance=3, rounds=3, p=0.01).detector_error_model(decompose_errors=True)
    dem = side_by_side(a, a)
    dets, obs = shots(dem, 2000, 2)
    comp = sq.CompositeDecoder(dem)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sub = sq.SubgraphDecoder(dem, [d for d, _ in comp.subgraphs], [o for _, o in comp.subgraphs])
    np.testing.assert_array_equal(sub.decode_batch(dets), comp.decode_batch(dets))
    # bit-packed in and out, one shot at a time, and pickled
    packed = np.packbits(dets, axis=1, bitorder="little")
    np.testing.assert_array_equal(np.unpackbits(sub.decode_batch(packed, bit_packed_shots=True, bit_packed_predictions=True), axis=1, count=2, bitorder="little"), comp.decode_batch(dets))
    np.testing.assert_array_equal(sub.decode(dets[7]), comp.decode_batch(dets[7:8])[0])
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(sub)).decode_batch(dets), sub.decode_batch(dets))
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(comp)).decode_batch(dets), sub.decode_batch(dets))
    # a callable inner decoder
    custom = sq.SubgraphDecoder(dem, [d for d, _ in comp.subgraphs], [o for _, o in comp.subgraphs], decoder=sq.UnionFind)
    assert int((custom.decode_batch(dets) != obs).any(axis=1).sum()) < 1.3 * int((sub.decode_batch(dets) != obs).any(axis=1).sum())


def test_subgraph_warnings_and_errors():
    dem = sq.DetectorErrorModel("error(0.1) D0 L0\nerror(0.1) D0 D1\nerror(0.1) D1 D2 L0\nerror(0.1) D2")
    with pytest.warns(UserWarning, match="cancel"):
        sq.SubgraphDecoder(dem, [[0, 1], [1, 2]])
    with pytest.warns(UserWarning, match="1 detectors are in no subgraph"):
        sq.SubgraphDecoder(dem, [[0, 1]])
    with pytest.raises(ValueError, match="2 subgraphs of detectors but 1 sets"):
        sq.SubgraphDecoder(dem, [[0], [1, 2]], [[0]])
    with pytest.raises(ValueError, match="detectors must be in 0..2"):
        sq.SubgraphDecoder(dem, [[0, 3]])
    with pytest.raises(ValueError, match="decoder must be one of"):
        sq.CompositeDecoder(dem, decoder="nope")
    with pytest.raises(TypeError, match="decoder must be a name"):
        sq.CompositeDecoder(dem, decoder=3)
