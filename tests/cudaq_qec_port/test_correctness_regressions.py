# New stabilizer_qec correctness regressions; tiny ONNX messages use the existing test helpers.
# CUDA-Q source contracts: NVIDIA cudaq-qec 0.8.1, Apache-2.0 (see test_trt_decoder.py).
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pytest

from stabilizer_qec import cudaq_qec as qec
from stabilizer_qec.cudaq_qec._onnx import OnnxModel, _mlas_product
from test_trt_decoder import FLOAT, FP8, INT64, UINT8, _int, _len, _str, model_path, node, onnx_model, tensor, value_info, write


def _bounded_python(source, timeout=10):
    # Explicit overlay under -I also works for the separately repaired Python 3.9 launcher.
    source = 'import sys; sys.path[:] = ' + repr(sys.path) + '\n' + source
    p = subprocess.Popen([sys.executable, '-I', '-c', source], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    deadline = time.monotonic() + timeout
    try:
        while p.poll() is None:
            if time.monotonic() >= deadline:
                pytest.fail('own regression process exceeded %ss' % timeout)
            rss = subprocess.run(['ps', '-o', 'rss=', '-p', str(p.pid)], capture_output=True, text=True).stdout.strip()
            if rss and int(rss) * 1024 > 2 * 1024**3:
                pytest.fail('own regression process exceeded 2 GiB RSS')
            time.sleep(0.03)
        out, err = p.communicate()
        assert p.returncode == 0, err
        return out
    finally:
        if p.poll() is None:
            p.kill()
        p.communicate()


def _realtime_config(path, batch=1, global_name=None):
    dc = qec.decoder_config()
    dc.id, dc.type, dc.block_size, dc.syndrome_size = 0, 'trt_decoder', 3, 3
    dc.H_sparse = [0, -1, 1, -1, 2, -1]
    dc.O_sparse = [0, -1]
    dc.D_sparse = [0, -1, 1, -1, 2, -1]
    params = dict(onnx_load_path=path, batch_size=batch, use_cuda_graph=False)
    if global_name is not None:
        params['global_decoder'] = global_name
    dc.decoder_custom_args = params
    return dc


def _configure(dc):
    cfg = qec.multi_decoder_config()
    cfg.decoders = [dc]
    return qec.configure_decoders(cfg)


@pytest.mark.parametrize('name', ['no_such_decoder', 'single_error_lut'])
def test_realtime_synthesizes_named_global_params(name):
    dc = _realtime_config(model_path('identity_dynamic.onnx'), global_name=name)
    # Neither the public assignment nor schema preparation materializes this map.
    assert 'global_decoder_params' not in dc.decoder_custom_args
    if name == 'no_such_decoder':
        assert 'global_decoder_params' not in dc._args_for_decoder()
    else:
        assert 'global_decoder_params' in dc._args_for_decoder()
    try:
        if name == 'no_such_decoder':
            with pytest.warns(RuntimeWarning, match='invalid decoder requested: no_such_decoder'):
                assert _configure(dc) == 4
        else:
            # O describes one pre_L, leaving two residuals, too few for this global decoder.
            # Unlike an ignored global name, constructing it reaches its residual-size check.
            with pytest.warns(RuntimeWarning, match='num_observables.*global_decoder.syndrome_size'):
                assert _configure(dc) == 4
    finally:
        qec.finalize_decoders()


def test_realtime_batch_warning_and_slot_zero():
    dc = _realtime_config(model_path('identity_dynamic.onnx'), batch=4)
    expected = ('trt_decoder batch_size > 1 has no effect on the realtime decode path '
                '(one syndrome is decoded per call); the extra batch slots are '
                'zero-padded and discarded. Use batch_size = 1 for realtime.')
    try:
        with pytest.warns(RuntimeWarning, match='trt_decoder batch_size > 1') as recorded:
            assert _configure(dc) == 0
        assert [str(w.message) for w in recorded] == [expected]
        for syndrome, expected_flip in [([1, 0, 0], True), ([0, 1, 1], False), ([1, 1, 1], True)]:
            qec.enqueue_syndromes(0, syndrome)
            assert list(qec.get_corrections(0, 1, True)) == [expected_flip]
    finally:
        qec.finalize_decoders()


def test_realtime_global_dem_forwarding_affects_observables(tmp_path):
    # Same H as the repetition model, but a global constructed from H alone has no O or priors.
    dem = 'error(0.1) D0\nerror(0.1) D0 D1\nerror(0.1) D1 D2\nerror(0.1) D2 L0\n'
    dem_path = tmp_path / 'repetition.dem'
    dem_path.write_text(dem)
    weights = np.array([[1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
    path = write(tmp_path, 'pre.onnx', onnx_model([node('MatMul', ['x', 'W'], ['y'])],
        [value_info('x', FLOAT, [1, 3])], [value_info('y', FLOAT, [1, 4])], [tensor('W', weights, FLOAT)]))
    direct = qec.get_decoder('trt_decoder', dem, onnx_load_path=path, batch_size=1,
        use_cuda_graph=False, global_decoder='pymatching', global_decoder_params={'stim_dem': dem})
    without = qec.get_decoder('trt_decoder', dem, onnx_load_path=path, batch_size=1,
        use_cuda_graph=False, global_decoder='pymatching', global_decoder_params={})
    dc = _realtime_config(path, global_name='pymatching')
    dc.stim_dem_path, dc.H_sparse, dc.O_sparse = str(dem_path), [], []
    dc.block_size = dc.syndrome_size = 0
    try:
        assert _configure(dc) == 0
        witnessed_difference = False
        for syndrome in ([0, 0, 1], [1, 0, 0], [0, 0, 0]):
            expected = (direct.decode(syndrome).result > 0.5).tolist()
            witnessed_difference |= expected != (without.decode(syndrome).result > 0.5).tolist()
            qec.enqueue_syndromes(0, syndrome)
            assert list(qec.get_corrections(0, 1, True)) == expected
        assert witnessed_difference
    finally:
        qec.finalize_decoders()


@pytest.mark.parametrize("op", ["MatMul", "Gemm"])
def test_empty_huge_k_public_decoder_returns_under_watchdog(tmp_path, op):
    # An empty initializer has no storage despite its enormous contracting dimension.
    k = 2**39
    b = _int(1, k) + _int(1, 0) + _int(2, FLOAT) + _str(8, 'b') + _len(9, b'')
    data = onnx_model([node(op, ['x', 'b'], ['y'])], [value_info('x', FLOAT, [0, k])], [value_info('y', FLOAT, [0, 0])], [b])
    path = write(tmp_path, 'empty.onnx', data)
    out = _bounded_python("import numpy as np\nfrom stabilizer_qec import cudaq_qec as q\n"
        "from stabilizer_qec.cudaq_qec._onnx import load_onnx\n"
        "q.get_decoder('trt_decoder', np.zeros((0,0), np.uint8), onnx_load_path=" + repr(path) + ")\n"
        "assert load_onnx(" + repr(path) + ").run(np.empty((0,2**39),np.float32)).shape == (0,0)\nprint('returned')")
    assert out.strip() == 'returned'


@pytest.mark.parametrize('ashape,bshape', [((0, 7), (7, 4)), ((3, 7), (7, 0)), ((0, 3, 7), (1, 7, 4))])
def test_empty_product_shapes(ashape, bshape):
    a, b = np.empty(ashape, np.float32), np.empty(bshape, np.float32)
    assert _mlas_product(a, b, alpha=float('inf')).shape == np.matmul(a, b).shape


def test_empty_product_still_validates_shapes_and_addend():
    with pytest.raises(ValueError, match='do not multiply'):
        _mlas_product(np.empty((0, 4)), np.empty((3, 0)))
    with pytest.raises(ValueError):
        _mlas_product(np.empty((0, 2, 4)), np.empty((3, 4, 0)))
    with pytest.raises(ValueError):
        _mlas_product(np.empty((0, 4)), np.empty((4, 0)), y=np.zeros(3))
    np.testing.assert_array_equal(_mlas_product(np.empty((2, 0)), np.empty((0, 3)), alpha=3, y=np.full((2, 3), 7, np.float32)), np.full((2, 3), 7))


@pytest.mark.parametrize('saturate', [0, 1])
def test_cast_fp8_all_nan_payloads_and_controls(saturate):
    bits = np.array([0x7f800001, 0xff800001, 0x7fa00000, 0xffa00000, 0x7fc00000, 0xffc00000,
                     0x7fffffff, 0xffffffff, 0x7f800000, 0xff800000, 0x43fa0000, 0xc3fa0000, 0x3f800000], np.uint32)
    recorded = np.load(Path(model_path("correctness")) / "fp8_nan_controls.npz")
    np.testing.assert_array_equal(bits, recorded["bits"])
    assert str(recorded["meta_ort"]) == "1.31.0"
    x = bits.view(np.float32)
    model = OnnxModel(onnx_model([node('Cast', ['x'], ['fp8'], to=FP8, saturate=saturate), node('Cast', ['fp8'], ['y'], to=FLOAT)],
        [value_info('x', FLOAT, [len(x)])], [value_info('y', FLOAT, [len(x)])]))
    y = model.run(x)
    assert np.isnan(y[:8]).all()
    expected = [448, -448, 448, -448, 1] if saturate else [np.nan, np.nan, np.nan, np.nan, 1]
    np.testing.assert_array_equal(y[8:], expected)
    np.testing.assert_array_equal(y, recorded["y" + str(saturate)])


@pytest.mark.parametrize('form', ['float64', 'float32', 'int32', 'int64', 'bool', 'list', 'fortran', 'strided', 'sparse'])
def test_trt_observable_matrix_documented_coercions(tmp_path, form):
    # Two rows make the F-order and noncontiguous cases distinct in memory layout.
    O = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.uint8)
    if form in ('float64', 'float32', 'int32', 'int64', 'bool'):
        O = O.astype(form)
    elif form == 'list':
        O = O.tolist()
    elif form == 'fortran':
        O = np.asfortranarray(O)
    elif form == 'strided':
        O = np.repeat(O, 2, axis=1)[:, ::2]
    elif form == 'sparse':
        O = pytest.importorskip('scipy.sparse').csr_matrix(O)
    d = qec.get_decoder('trt_decoder', np.eye(3, dtype=np.uint8), O=O,
        onnx_load_path=model_path('identity_dynamic.onnx'), batch_size=1, use_cuda_graph=False)
    np.testing.assert_array_equal(d.decode([1, 0, 1]).result, [1, 0])


@pytest.mark.parametrize('opset,accepted', [(13, False), (14, True)])
def test_public_reshape_allowzero_version_boundary(tmp_path, opset, accepted):
    data = onnx_model([node('Reshape', ['x', 's'], ['y'], allowzero=1)], [value_info('x', FLOAT, [1, 3])],
        [value_info('y', FLOAT, [1, 3])], [tensor('s', np.array([1, 3], np.int64), INT64)], opset=opset)
    path = write(tmp_path, 'reshape.onnx', data)
    if not accepted:
        with pytest.raises(ValueError, match='allowzero.*requires opset 14'):
            OnnxModel(data)
        with pytest.raises(RuntimeError, match='Failed to parse ONNX model'):
            qec.get_decoder('trt_decoder', np.eye(3, dtype=np.uint8), onnx_load_path=path)
    else:
        d = qec.get_decoder('trt_decoder', np.eye(3, dtype=np.uint8), onnx_load_path=path)
        np.testing.assert_array_equal(d.decode([0, 1, 0]).result, [0, 1, 0])


@pytest.mark.parametrize('op,attrs,opset', [('Cast', {'to': FLOAT, 'saturate': 1}, 18),
    ('QuantizeLinear', {'saturate': 1}, 18), ('QuantizeLinear', {'output_dtype': UINT8}, 20),
    ('QuantizeLinear', {'block_size': 0}, 20), ('DequantizeLinear', {'block_size': 0}, 20)])
def test_versioned_attributes_refused_before_evaluation(op, attrs, opset):
    ins = ['x'] if op == 'Cast' else ['x', 's']
    data = onnx_model([node(op, ins, ['y'], **attrs)], [value_info('x', FLOAT, [1])], [value_info('y', FLOAT, [1])],
        [] if op == 'Cast' else [tensor('s', np.array(1, np.float32), FLOAT)], opset=opset)
    with pytest.raises(ValueError, match='requires opset'):
        OnnxModel(data)


@pytest.mark.parametrize('opset', [0, 12, 22, 999])
def test_unsupported_opset_refused(opset):
    with pytest.raises(ValueError, match='unsupported ONNX opset'):
        OnnxModel(onnx_model([node('Identity', ['x'], ['y'])], [value_info('x', FLOAT, [1])], [value_info('y', FLOAT, [1])], opset=opset))


@pytest.mark.parametrize('domain', ['wrong.domain', 'ai.onnx.ml'])
def test_wrong_node_domain_refused(domain):
    data = onnx_model([node('Identity', ['x'], ['y']) + _str(7, domain)], [value_info('x', FLOAT, [1])], [value_info('y', FLOAT, [1])])
    with pytest.raises(ValueError, match='unsupported ONNX op'):
        OnnxModel(data)


def test_unknown_attribute_and_invalid_flag_refused():
    for attrs, message in [({'unknown': 0}, 'not supported'), ({'allowzero': 2}, 'must be 0 or 1')]:
        data = onnx_model([node('Reshape', ['x', 's'], ['y'], **attrs)], [value_info('x', FLOAT, [1])], [value_info('y', FLOAT, [1])], [tensor('s', np.array([1], np.int64), INT64)])
        with pytest.raises(ValueError, match=message):
            OnnxModel(data)


@pytest.mark.parametrize('opset', [13, 18])
def test_fp8_type_refused_before_introduction(opset):
    data = onnx_model([node('Cast', ['x'], ['y'], to=FP8)], [value_info('x', FLOAT, [1])], [value_info('y', FP8, [1])], opset=opset)
    with pytest.raises(ValueError, match='target type.*not supported'):
        OnnxModel(data)


@pytest.mark.parametrize('field,value,exc,message', [
    ('indices', [-1], RuntimeError, 'indices contains a negative index'),
    ('indices', [3], ValueError, 'column index out of range'),
    ('indices', [2**32], RuntimeError, 'indices exceeds sparse_binary_matrix index_type'),
    ('indptr', [-1, 1, 1], RuntimeError, 'indptr contains a negative index'),
    ('indptr', [0, 1], ValueError, 'row_ptrs must have size num_rows'),
    ('indptr', [1, 1, 1], ValueError, r'row_ptrs\[0\] must be zero'),
    ('indptr', [0, 1, 0], ValueError, 'row_ptrs must be non-decreasing'),
    ('indptr', [0, 1, 2], ValueError, 'last row_ptr must equal nnz'),
])
def test_css_sparse_setters_reject_malformed_csr_safely(field, value, exc, message):
    sp = pytest.importorskip('scipy.sparse')
    csr = sp.csr_matrix(([1], [0], [0, 1, 1]), shape=(2, 3))
    setattr(csr, field, np.array(value, np.int64))
    c = qec.CssCodes()
    for prop in ('hz', 'hx', 'lz', 'lx'):
        setattr(c, prop, np.eye(2, dtype=np.uint8))
        with pytest.raises(exc, match=message):
            setattr(c, prop, csr)
        # A failed assignment leaves the previously stored matrix intact.
        np.testing.assert_array_equal(getattr(c, prop), np.eye(2, dtype=np.uint8))


@pytest.mark.parametrize('shape', [(0, 0), (0, 3), (3, 0), (2, 3)])
def test_css_sparse_shapes_and_dense_coercions(shape):
    sp = pytest.importorskip('scipy.sparse')
    c = qec.CssCodes()
    c.hz = sp.csr_matrix(shape, dtype=np.uint8)
    assert c.hz.shape == shape
    assert c.hz.dtype == np.uint8
    if shape == (2, 3):
        a = np.array([[1, 0, 2], [0, 1, 0]], dtype=np.float64)
        for given in (a, a.tolist(), np.asfortranarray(a), sp.csc_matrix(a)):
            c.hz = given
            np.testing.assert_array_equal(c.hz, a != 0)


@pytest.mark.parametrize('failed_mapping', ['connection', 'named'])
def test_failed_public_yaml_mapping_registers_names_and_collisions(failed_mapping):
    if failed_mapping == 'connection':
        bad = 'dem_chunks:\n  seam: {from: costarring, to: [x]}\n  connections: []\n  phases: []\n'
    else:
        bad = ('dem_chunks:\n  seam: {from: next_round, to: prev_round}\n  connections: []\n'
               '  phases: [{name: costarring, spec: {num_faults: [x], error_rates: []}}]\n')
    prefix = 'id: 0\ntype: single_error_lut\n'
    # Process isolation preserves the global registration table of the surrounding test suite.
    source = "from stabilizer_qec import cudaq_qec as q\n"
    source += "try:\n q.decoder_config.from_yaml_str(" + repr(prefix + bad) + ")\nexcept RuntimeError:\n pass\nelse:\n raise AssertionError('malformed mapping accepted')\n"
    source += "try:\n q.DemChunksSpec().get_phase(q.SeamId('costarring'))\nexcept ValueError as e:\n assert str(e) == 'dem_chunks_spec::get_phase: phase costarring not found', str(e)\n"
    collision = prefix + 'dem_chunks:\n  seam: {from: liquid, to: prev_round}\n  connections: []\n  phases: []\n'
    source += "try:\n q.decoder_config.from_yaml_str(" + repr(collision) + ")\nexcept (RuntimeError, ValueError) as e:\n assert \"collision at 0x5e4daa9d between 'costarring' and 'liquid'\" in str(e), str(e)\nelse:\n raise AssertionError('collision was not registered')\nprint('registered')\n"
    assert _bounded_python(source).strip() == 'registered'


@pytest.mark.parametrize('op', ['MatMul', 'Gemm'])
def test_computed_weights_against_float64_mathematical_oracle(op):
    rng = np.random.default_rng(182)
    x = rng.normal(size=(3, 600)).astype(np.float32)
    # B is computed from the graph input, so an ORT optimizer cannot prepack it as a constant.
    nodes = [node('Reshape', ['x', 's'], ['b']), node(op, ['x', 'b'], ['y'])]
    m = OnnxModel(onnx_model(nodes, [value_info('x', FLOAT, [3, 600])], [value_info('y', FLOAT, [3, 3])],
        [tensor('s', np.array([600, 3], np.int64), INT64)]))
    expected = x.astype(np.float64) @ x.reshape(600, 3).astype(np.float64)
    np.testing.assert_allclose(m.run(x), expected, rtol=0, atol=1e-4)


def test_weight_dequantization_mathematical_oracle_and_binary_threshold(tmp_path):
    z = np.load(Path(model_path('correctness')) / 'weightdq_threshold.npz')
    assert str(z['meta_ort']) == '1.31.0'
    x, w, scale, zero, bias = (z[k] for k in ('x', 'w', 's', 'z', 'bias'))
    nodes = [node('DequantizeLinear', ['w', 's', 'z'], ['b']), node('MatMul', ['x', 'b'], ['mm']), node('Add', ['mm', 'bias'], ['y'])]
    data = onnx_model(nodes, [value_info('x', FLOAT, list(x.shape))], [value_info('y', FLOAT, [3, 20])],
        [tensor('w', w, 3), tensor('s', scale, FLOAT), tensor('z', zero, 3), tensor('bias', bias, FLOAT)])
    # Mathematical oracle starts with the graph's *typed*, float32 dequantized weights.
    dequantized = ((w.astype(np.int64) - zero).astype(np.float32) * scale).astype(np.float32)
    expected = x.astype(np.float64) @ dequantized.astype(np.float64) + bias.astype(np.float64)
    y = OnnxModel(data).run(x)
    np.testing.assert_allclose(y, expected, rtol=0, atol=1e-4)
    assert np.isin(x, [0, 1]).all()
    d = qec.get_decoder('trt_decoder', np.zeros((600, 20), np.uint8),
        onnx_load_path=write(tmp_path, 'threshold.onnx', data), batch_size=3, use_cuda_graph=False)
    port = d.decode(x[0]).result
    np.testing.assert_array_equal(port, y[0].astype(np.float64))
    assert port[5] == 0.49976348876953125
    assert z['ort_all'][0, 5] == 0.500244140625
    assert z['ort_disabled'][0, 5] == 0.49971771240234375
    assert not port[5] > 0.5
    assert z['ort_all'][0, 5] > 0.5
    assert not z['ort_disabled'][0, 5] > 0.5


@pytest.mark.parametrize('version', [0, 6, 11, 999])
def test_unsupported_ir_version_refused(version):
    data = onnx_model([node('Identity', ['x'], ['y'])], [value_info('x', FLOAT, [1])], [value_info('y', FLOAT, [1])])
    # ModelProto singular fields merge; the last scalar is authoritative.
    with pytest.raises(ValueError, match='unsupported ONNX IR version'):
        OnnxModel(data + _int(1, version))


@pytest.mark.parametrize('op,typ,opset', [('Add', UINT8, 13), ('Relu', 3, 13), ('MatMul', UINT8, 19), ('Flatten', FP8, 19)])
def test_operator_tensor_types_refused_before_evaluation(op, typ, opset):
    inputs = ['x', 'x'] if op in ('Add', 'MatMul') else ['x']
    data = onnx_model([node(op, inputs, ['y'])], [value_info('x', typ, [1, 1])], [value_info('y', typ, [1, 1])], opset=opset)
    with pytest.raises(ValueError, match='type.*not supported at opset'):
        OnnxModel(data)


def test_duplicate_and_wrong_domain_imports_refused():
    data = onnx_model([node('Identity', ['x'], ['y'])], [value_info('x', FLOAT, [1])], [value_info('y', FLOAT, [1])])
    for extra, message in [(_len(8, _int(2, 19)), 'exactly once'), (_len(8, _str(1, 'ai.onnx.ml') + _int(2, 1)), 'operator-set domain')]:
        with pytest.raises(ValueError, match=message):
            OnnxModel(data + extra)


def test_integer_gemm_evaluator_refuses_before_any_arithmetic():
    data = onnx_model([node('Gemm', ['x', 'b'], ['y'])], [value_info('x', 6, [1, 1])], [value_info('y', 6, [1, 1])],
        [tensor('b', np.ones((1, 1), np.int32), 6)])
    m = OnnxModel(data)
    # Even a feed whose conversion would fail must never reach graph evaluation.
    with pytest.raises(ValueError, match='integer Gemm evaluation is not supported'):
        m.run(object())


@pytest.mark.parametrize('op,opset,attrs', [('QuantizeLinear', 19, {'saturate': 1}),
    ('QuantizeLinear', 21, {'block_size': 0, 'output_dtype': UINT8}),
    ('DequantizeLinear', 21, {'block_size': 0})])
def test_versioned_quantization_attributes_valid_controls(op, opset, attrs):
    typ, out = (FLOAT, UINT8) if op == 'QuantizeLinear' else (UINT8, FLOAT)
    m = OnnxModel(onnx_model([node(op, ['x', 's'], ['y'], **attrs)], [value_info('x', typ, [1])],
        [value_info('y', out, [1])], [tensor('s', np.array(1, np.float32), FLOAT)], opset=opset))
    np.testing.assert_array_equal(m.run([1]), [1])


@pytest.mark.parametrize('beta,expected', [(0.0, 0.0), (2.0, 14.0)])
def test_zero_k_gemm_preserves_alpha_beta_and_bias(beta, expected):
    m = OnnxModel(onnx_model([node('Gemm', ['x', 'b', 'c'], ['y'], alpha=3.0, beta=beta)],
        [value_info('x', FLOAT, [2, 0])], [value_info('y', FLOAT, [2, 3])],
        [tensor('b', np.empty((0, 3), np.float32), FLOAT), tensor('c', np.full((2, 3), 7, np.float32), FLOAT)]))
    np.testing.assert_array_equal(m.run(np.empty((2, 0), np.float32)), np.full((2, 3), expected))
