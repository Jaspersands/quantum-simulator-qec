# CUDA-Q QEC's deprecated typed decoder configurations (NVIDIA/cudaqx 0.8.1,
# libs/qec/python/tests/test_decoding_config_deprecated.py; Apache License 2.0), re-typed and
# abridged.
import math

import pytest

from stabilizer_qec import cudaq_qec as qec

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def nv_qldpc_dc():
    dc = qec.decoder_config()
    dc.id = 0
    dc.type = "nv-qldpc-decoder"
    dc.block_size = 10
    dc.syndrome_size = 3
    dc.H_sparse = [1, 2, 3, -1, 6, 7, 8, -1, -1]
    return dc


@pytest.mark.filterwarnings("default::DeprecationWarning")
def test_deprecated_typed_configs_warn_on_construction():
    for cls in (
        qec.nv_qldpc_decoder_config, qec.multi_error_lut_config, qec.trt_decoder_config, qec.pymatching_config, qec.chromobius_config,
        qec.qecrt.config.srelay_bp_config, qec.qecrt.config.single_error_lut_config, qec.qecrt.config.sliding_window_config,
    ):
        with pytest.warns(DeprecationWarning, match="is deprecated; assign a plain dict"):
            cls()


def test_deprecated_config_matches_dict_built_yaml():
    cfg = qec.nv_qldpc_decoder_config()
    cfg.use_sparsity = True
    cfg.error_rate = 0.01
    cfg.max_iterations = 50
    cfg.bp_seed = -1
    cfg.srelay_config = qec.qecrt.config.srelay_bp_config()
    cfg.srelay_config.pre_iter = 5
    cfg.srelay_config.stopping_criterion = "NConv"
    old_style = nv_qldpc_dc()
    old_style.set_decoder_custom_args(cfg)
    new_style = nv_qldpc_dc()
    new_style.decoder_custom_args = {"use_sparsity": True, "error_rate": 0.01, "max_iterations": 50, "bp_seed": -1, "srelay_config": {"pre_iter": 5, "stopping_criterion": "NConv"}}
    assert old_style.to_yaml_str() == new_style.to_yaml_str()
    assert old_style == new_style


def test_deprecated_config_assignable_before_type():
    cfg = qec.pymatching_config()
    cfg.error_rate_vec = [0.1, 0.2, 0.3]
    cfg.merge_strategy = "smallest_weight"
    dc = qec.decoder_config()
    dc.decoder_custom_args = cfg  # (converted through its own schema, pymatching's)
    dc.type = "pymatching"
    assert list(dc.decoder_custom_args["error_rate_vec"]) == [0.1, 0.2, 0.3]
    assert dc.decoder_custom_args["merge_strategy"] == "smallest_weight"
    dc.validate_custom_args()


def test_unset_fields_read_as_none_and_unknown_ones_raise():
    cfg = qec.nv_qldpc_decoder_config()
    assert cfg.max_iterations is None
    cfg.max_iterations = 50
    assert cfg.max_iterations == 50
    cfg.max_iterations = None
    assert cfg.to_heterogeneous_map() == {}
    pm = qec.pymatching_config()
    with pytest.raises(AttributeError):
        pm.merge_stratgey = "smallest_weight"
    with pytest.raises(AttributeError):
        _ = pm.merge_stratgey


def test_setting_wrong_types_raises_typeerror():
    nv = qec.nv_qldpc_decoder_config()
    for name, value in (("max_iterations", "ten"), ("use_sparsity", "True"), ("error_rate", "0.1"), ("error_rate_vec", [0.1, "nope"]), ("error_rate_vec", 3.14)):
        with pytest.raises(TypeError):
            setattr(nv, name, value)
    nv.error_rate = 1  # (an int is a number)
    assert math.isclose(nv.error_rate, 1.0)


def test_sliding_window_inner_params_collapse_and_roundtrip():
    cfg = qec.qecrt.config.sliding_window_config()
    cfg.window_size = 3
    cfg.step_size = 1
    cfg.error_rate_vec = [0.1, 0.2, 0.3]
    cfg.inner_decoder_name = "multi_error_lut"
    cfg.multi_error_lut_params = qec.multi_error_lut_config()
    cfg.multi_error_lut_params.lut_error_depth = 2
    assert cfg.to_heterogeneous_map() == {"window_size": 3, "step_size": 1, "error_rate_vec": [0.1, 0.2, 0.3], "inner_decoder_name": "multi_error_lut", "inner_decoder_params": {"lut_error_depth": 2}}
    dc = qec.decoder_config()
    dc.id = 0
    dc.type = "sliding_window"
    dc.block_size = 3
    dc.syndrome_size = 3
    dc.H_sparse = [0, -1, 1, -1, 2, -1]
    dc.set_decoder_custom_args(cfg)
    dc.validate_custom_args()
    assert qec.decoder_config.from_yaml_str(dc.to_yaml_str()).decoder_custom_args["inner_decoder_params"]["lut_error_depth"] == 2
    source = {"window_size": 3, "error_rate_vec": [0.1, 0.2, 0.3], "inner_decoder_name": "multi_error_lut", "inner_decoder_params": {"lut_error_depth": 2}}
    back = qec.qecrt.config.sliding_window_config.from_heterogeneous_map(source)
    assert back.multi_error_lut_params.lut_error_depth == 2 and back.to_heterogeneous_map() == source


def test_schema_validation_still_applies():
    cfg = qec.qecrt.config.sliding_window_config()
    cfg.window_size = 2
    cfg.step_size = 5
    cfg.error_rate_vec = [0.1]
    cfg.inner_decoder_name = "single_error_lut"
    dc = qec.decoder_config()
    dc.type = "sliding_window"
    dc.set_decoder_custom_args(cfg)
    with pytest.raises(RuntimeError, match="step_size"):
        dc.validate_custom_args()


def test_trt_decoder_config_keeps_its_global_decoder_section():
    cfg = qec.trt_decoder_config({"global_decoder": "pymatching", "precision": "fp16"})
    assert cfg.to_heterogeneous_map() == {"precision": "fp16", "global_decoder": "pymatching", "global_decoder_params": {}}
    with pytest.raises(RuntimeError, match="global_decoder_params does not support global_decoder 'my_plugin'"):
        qec.trt_decoder_config({"global_decoder": "my_plugin", "global_decoder_params": {}})


def test_configure_with_a_deprecated_config():
    lut = qec.multi_error_lut_config()
    lut.lut_error_depth = 2
    dc = qec.decoder_config()
    dc.id = 0
    dc.type = "multi_error_lut"
    dc.block_size = 10
    dc.syndrome_size = 3
    dc.H_sparse = [1, 2, 3, -1, 6, 7, 8, -1, -1]
    dc.D_sparse = qec.generate_timelike_sparse_detector_matrix(3, 2, include_first_round=False)
    dc.set_decoder_custom_args(lut)
    mdc = qec.multi_decoder_config()
    mdc.decoders = [dc]
    try:
        assert qec.configure_decoders(mdc) == 0
    finally:
        qec.finalize_decoders()
