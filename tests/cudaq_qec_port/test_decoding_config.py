# CUDA-Q QEC's real-time decoding configuration (NVIDIA/cudaqx 0.8.1, libs/qec/python/tests:
# test_decoding_config.py, test_decoders_yaml.py; Apache License 2.0), re-typed and adapted:
# decoder_config, multi_decoder_config, their YAML, and the decoder parameter schemas.
import json
import math

import numpy as np
import pytest

from stabilizer_qec import cudaq_qec as qec
from stabilizer_qec.cudaq_qec import _yaml


def qec_yaml_for(dc):
    mdc = qec.multi_decoder_config()
    mdc.decoders = [dc]
    return mdc.to_yaml_str()


def pymatching_dc():
    dc = qec.decoder_config()
    dc.id = 0
    dc.type = "pymatching"
    dc.block_size = 3
    dc.syndrome_size = 3
    dc.H_sparse = [0, -1, 1, -1, 2, -1]
    dc.O_sparse = [0, -1, 1, -1, 2, -1]
    dc.D_sparse = [0, -1, 1, -1, 2, -1]
    return dc


def empty_lut_dc(decoder_id):
    config = qec.decoder_config()
    config.id = decoder_id
    config.type = "single_error_lut"
    config.block_size = 20
    config.syndrome_size = 10
    config.H_sparse = qec.pcm_to_sparse_vec(np.zeros((10, 20), dtype=np.uint8))
    config.O_sparse = qec.pcm_to_sparse_vec(np.zeros((2, 20), dtype=np.uint8))
    config.D_sparse = qec.generate_timelike_sparse_detector_matrix(10, 2, include_first_round=False)
    return config


def check_roundtrip(multi):
    text = multi.to_yaml_str(200)
    back = qec.multi_decoder_config.from_yaml_str(text)
    assert back.to_yaml_str(200) == text
    assert back == multi


def check_creation(multi):
    assert qec.configure_decoders(multi) == 0
    qec.finalize_decoders()


# -- Schemas ----------------------------------------------------------------------------------------


def test_nv_qldpc_decoder_param_schema_introspection():
    by_key = {e["key"]: e for e in qec.decoder_param_schema("nv-qldpc-decoder")}
    assert by_key["max_iterations"]["kind"] == "int32"
    assert by_key["error_rate_vec"]["kind"] == "float64_vec"
    assert by_key["explicit_gammas"]["kind"] == "float64_matrix"
    assert by_key["srelay_config"]["kind"] == "subschema" and by_key["srelay_config"]["subschema"] == "srelay_bp"


def test_decoder_param_schema_introspection():
    by_key = {e["key"]: e for e in qec.decoder_param_schema("sliding_window")}
    assert by_key["error_rate_vec"]["required"] is True
    assert by_key["inner_decoder_params"]["kind"] == "discriminated"
    assert by_key["inner_decoder_params"]["discriminator"] == "inner_decoder_name"
    assert qec.decoder_param_schema("no-such-decoder") is None
    assert qec.registered_decoder_schemas() == sorted(
        ["chromobius", "multi_error_lut", "nv-fusion-decoder", "nv-qldpc-decoder", "pymatching", "single_error_lut", "single_error_lut_example", "sliding_window", "srelay_bp"]
    )
    # Every decoder with a schema is one this package builds (the nested srelay_bp aside).
    from stabilizer_qec.cudaq_qec._decoders import _NATIVE

    assert set(qec.registered_decoder_schemas()) - {"srelay_bp"} <= set(_NATIVE)


# -- decoder_custom_args ----------------------------------------------------------------------------


def test_decoder_custom_args_is_a_dict():
    dc = qec.decoder_config()
    assert dc.decoder_custom_args == {}
    dc.decoder_custom_args = {"lut_error_depth": 2}
    assert dc.decoder_custom_args == {"lut_error_depth": 2}
    dc.set_decoder_custom_args({"lut_error_depth": 3})
    assert dc.decoder_custom_args == {"lut_error_depth": 3}


def test_decoder_config_yaml_roundtrip_and_custom_args():
    dc = qec.decoder_config()
    dc.id = 0
    dc.type = "nv-qldpc-decoder"
    dc.block_size = 10
    dc.syndrome_size = 3
    dc.H_sparse = [1, 2, 3, -1, 6, 7, 8, -1, -1]
    dc.decoder_custom_args = {
        "use_sparsity": True,
        "error_rate": 0.01,
        "max_iterations": 50,
        "error_rate_vec": [0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1],
        "srelay_config": {"pre_iter": 5, "stopping_criterion": "NConv"},
    }
    dc2 = qec.decoder_config.from_yaml_str(dc.to_yaml_str())
    assert (dc2.id, dc2.type, dc2.block_size, dc2.syndrome_size) == (0, "nv-qldpc-decoder", 10, 3)
    args = dc2.decoder_custom_args
    assert args["use_sparsity"] is True and math.isclose(args["error_rate"], 0.01) and args["max_iterations"] == 50
    assert args["srelay_config"] == {"pre_iter": 5, "stopping_criterion": "NConv"}
    assert isinstance(args["error_rate_vec"], np.ndarray) and args["error_rate_vec"].dtype == np.float64
    assert dc2 == dc


def test_pymatching_config_yaml_roundtrip():
    dc = pymatching_dc()
    dc.decoder_custom_args = {"error_rate_vec": [0.1, 0.2, 0.3], "merge_strategy": "smallest_weight"}
    dc2 = qec.decoder_config.from_yaml_str(dc.to_yaml_str())
    assert list(dc2.decoder_custom_args["error_rate_vec"]) == [0.1, 0.2, 0.3]
    assert dc2.decoder_custom_args["merge_strategy"] == "smallest_weight"


def test_unknown_custom_arg_key_is_rejected():
    dc = pymatching_dc()
    dc.decoder_custom_args = {"merge_strategy": "smallest_weight"}
    misspelled = qec_yaml_for(dc).replace("merge_strategy", "merge_stratgey")
    with pytest.raises(RuntimeError, match="Unknown key 'merge_stratgey' in 'pymatching' parameters"):
        qec.multi_decoder_config.from_yaml_str(misspelled)
    # An unknown key of the decoder entry itself is LLVM YAML's error.
    with pytest.raises(RuntimeError, match=r"Invalid decoder configuration YAML: Invalid argument \(YAML:\d+:\d+: unknown key 'foo'\)"):
        qec.multi_decoder_config.from_yaml_str(qec_yaml_for(pymatching_dc()).replace("type:", "foo: 1\n    type:"))


def test_validate_custom_args_checks_dict_built_configs():
    dc = qec.decoder_config()
    dc.type = "pymatching"
    dc.decoder_custom_args = {"merge_strategy": "smallest_weight"}
    dc.validate_custom_args()
    dc.decoder_custom_args = {"merge_stratgey": "smallest_weight"}
    with pytest.raises(RuntimeError, match="Unknown key 'merge_stratgey' in 'pymatching' parameters."):
        dc.validate_custom_args()
    dc.type = "decoder_without_registered_schema"
    dc.decoder_custom_args = {"anything": 1}
    with pytest.raises(RuntimeError, match="no registered parameter schema"):
        dc.validate_custom_args()
    mdc = qec.multi_decoder_config()
    mdc.decoders = [dc]
    with pytest.raises(RuntimeError):
        mdc.validate_custom_args()


def test_custom_args_dict_values_convert_to_schema_types():
    dc = qec.decoder_config()
    dc.id = 0
    dc.type = "nv-qldpc-decoder"
    dc.block_size = 1
    dc.syndrome_size = 1
    dc.H_sparse = [0, -1]
    dc.O_sparse = [0, -1]
    dc.D_sparse = [0, -1]
    dc.decoder_custom_args = {"clip_value": 2}  # an int for a float parameter
    assert "clip_value:      2" in dc.to_yaml_str() and isinstance(dc.decoder_custom_args["clip_value"], float)
    dc.decoder_custom_args = {"bp_seed": -1}  # a negative int32
    assert dc.decoder_custom_args == {"bp_seed": -1}
    assert qec.multi_decoder_config.from_yaml_str(qec_yaml_for(dc)).decoders[0].decoder_custom_args == {"bp_seed": -1}
    with pytest.raises(RuntimeError, match="Parameter 'bp_seed' of 'nv-qldpc-decoder' expects a 32-bit int value."):
        dc.decoder_custom_args = {"bp_seed": "oops"}
    with pytest.raises(RuntimeError, match="bp_seed"):
        dc.decoder_custom_args = {"bp_seed": 2**31}


def test_generic_conversion_without_a_schema():
    dc = qec.decoder_config()
    dc.decoder_custom_args = {"a": True, "b": 3, "c": 0.5, "d": "s", "e": [1, 2], "f": [[1.0], [2.0, 3.0]], "g": {"h": 1}, "empty": []}
    args = dc.decoder_custom_args
    assert args["a"] is True and args["b"] == 3 and args["c"] == 0.5 and args["d"] == "s" and "empty" not in args
    assert args["e"].tolist() == [1.0, 2.0] and args["f"] == [[1.0], [2.0, 3.0]] and args["g"] == {"h": 1}
    with pytest.raises(RuntimeError, match="size_t"):
        dc.decoder_custom_args = {"b": -1}
    with pytest.raises(RuntimeError):
        dc.decoder_custom_args = {"x": object()}


def test_validate_custom_args_checks_value_kinds():
    dc = qec.decoder_config()
    dc.decoder_custom_args = {"clip_value": 2}  # type not set yet: stored as an unsigned int
    dc.type = "nv-qldpc-decoder"
    with pytest.raises(RuntimeError, match="Key 'clip_value' in 'nv-qldpc-decoder' parameters does not hold a float value"):
        dc.validate_custom_args()
    dc.decoder_custom_args = {"clip_value": 2}
    dc.validate_custom_args()


def test_validate_custom_args_runs_schema_validate_hook():
    dc = qec.decoder_config()
    dc.type = "sliding_window"
    base = {"error_rate_vec": [0.01, 0.01], "inner_decoder_name": "single_error_lut"}
    dc.decoder_custom_args = dict(base, window_size=4, step_size=2)
    dc.validate_custom_args()
    dc.decoder_custom_args = dict(base, window_size=2, step_size=4)
    with pytest.raises(RuntimeError, match=r"step_size \(4\) must be between 1 and window_size \(2\)"):
        dc.validate_custom_args()
    dc.decoder_custom_args = dict(base, num_syndromes_per_round=2, num_boundary_syndromes=3)
    with pytest.raises(RuntimeError, match="num_boundary_syndromes"):
        dc.validate_custom_args()
    dc.decoder_custom_args = {"inner_decoder_name": "single_error_lut"}
    with pytest.raises(RuntimeError, match="Missing required key 'error_rate_vec'"):
        dc.validate_custom_args()
    dc.decoder_custom_args = dict(base, inner_decoder_name="nope", inner_decoder_params={})
    with pytest.raises(RuntimeError, match="does not support inner_decoder_name 'nope'"):
        dc.validate_custom_args()


def test_non_schema_keys_dropped_from_emission_and_decoder_params():
    dc = qec.decoder_config()
    dc.id = 0
    dc.type = "multi_error_lut"
    dc.block_size = 3
    dc.syndrome_size = 3
    dc.H_sparse = [0, -1, 1, -1, 2, -1]
    dc.decoder_custom_args = {"lut_error_depth": 2, "not_a_real_param": 42}
    with pytest.warns(RuntimeWarning, match="Key 'not_a_real_param' is not in the 'multi_error_lut' parameter schema"):
        text = qec_yaml_for(dc)
    assert "lut_error_depth" in text and "not_a_real_param" not in text
    assert dc.decoder_custom_args["not_a_real_param"] == 42
    with pytest.raises(RuntimeError, match="not_a_real_param"):
        dc.validate_custom_args()


def test_args_for_a_type_without_a_schema_are_omitted_with_a_warning():
    dc = pymatching_dc()
    dc.type = "invalid-decoder"
    dc.decoder_custom_args = {"max_iterations": 50}
    with pytest.warns(RuntimeWarning, match="no parameter schema is registered"):
        assert "decoder_custom_args" not in dc.to_yaml_str()


# -- YAML -------------------------------------------------------------------------------------------


def test_decoder_config_from_yaml_str_rejects_malformed_yaml():
    for text in ("id: [oops", "decoders: 3", "decoders:\n  - id: x\n    type: t\n", "a: b: c\n", ""):
        with pytest.raises(RuntimeError, match="Invalid decoder configuration YAML: Invalid argument"):
            qec.multi_decoder_config.from_yaml_str(text)


def test_yaml_format_is_cudaqs():
    dc = empty_lut_dc(0)
    dc.type = "multi_error_lut"
    dc.decoder_custom_args = {"lut_error_depth": 2}
    mdc = qec.multi_decoder_config()
    mdc.decoders = [dc]
    lines = mdc.to_yaml_str().splitlines()
    assert lines[:4] == ["---", "decoders:", "  - id:              0", "    type:            multi_error_lut"]
    assert "    H_sparse:        [ -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 ]" in lines
    assert lines[-3:] == ["    decoder_custom_args:", "      lut_error_depth: 2", "..."]
    # Long flow sequences wrap past the column, under the opening bracket.
    long = qec.decoder_config.from_yaml_str(mdc.to_yaml_str().replace("---\ndecoders:\n  - ", "").replace("\n    ", "\n"))
    text = long.to_yaml_str(40)
    d_lines = text[text.index("D_sparse:") :].splitlines()
    assert d_lines[0].startswith("D_sparse:        [ 0, 10, -1, ") and d_lines[0].endswith(", ") and len(d_lines[0]) > 40
    assert d_lines[1].startswith(" " * 19) and d_lines[1][19] != " "


def test_yaml_reader_agrees_with_pyyaml():
    yaml = pytest.importorskip("yaml")

    def plain(n):
        if n is None or n.kind == "null":
            return ""
        if n.kind == "scalar":
            return n.value
        if n.kind == "seq":
            return [plain(x) for x in n.value]
        return {k: plain(v) for k, _, v in n.value}

    docs = [
        "---\ndecoders:\n  - id:              0\n    H_sparse:        [ 0, 1, -1,\n                       2, -1 ]\n    args:\n      - '--a'\n      - \"b c\"\n    m:\n      x: {}\n      y: []\n...\n",
        "decoders:\n- id: 0\n  type: 'it''s'   # comment\n  d: {from: init, to: bulk}\n",
        "a: b\n  c\nd:\n- - x\n  - y\n- z\n",
        "key: 'a # b' # c\nother: \"x\\ty\"\n",
    ]
    for d in docs:
        assert plain(_yaml.parse(d)) == yaml.load(d, Loader=yaml.BaseLoader)
    for bad in ("id: [oops", "a:\n  b: 1\n c: 2\n", "a: b: c\n"):
        with pytest.raises(_yaml.YamlError):
            _yaml.parse(bad)
        with pytest.raises(yaml.YAMLError):
            yaml.load(bad, Loader=yaml.BaseLoader)


def test_yaml_values_follow_llvm_scalars():
    base = "id: {id}\ntype: single_error_lut\nblock_size: {b}\nsyndrome_size: 0\nH_sparse: []\nO_sparse: []\nD_sparse: []\n"
    assert qec.decoder_config.from_yaml_str(base.format(id="0x10", b=0)).id == 16
    for bad_id, msg in (("1.5", "invalid number"), ("[1]", "unexpected scalar")):
        with pytest.raises(RuntimeError, match=msg):
            qec.decoder_config.from_yaml_str(base.format(id=bad_id, b=0))
    with pytest.raises(RuntimeError, match="invalid number"):
        qec.decoder_config.from_yaml_str(base.format(id=0, b=-1))
    with pytest.raises(RuntimeError, match="missing required key 'type'"):
        qec.decoder_config.from_yaml_str("id: 0\n")
    with pytest.raises(RuntimeError, match="duplicated mapping key|duplicate"):
        qec.decoder_config.from_yaml_str("id: 0\nid: 1\ntype: x\n")


def test_dispatch_cuda_device_and_transport_roundtrip():
    dc = pymatching_dc()
    dc.dispatch = qec.decoder_dispatch.device_graph
    dc.cuda_device_id = 1
    mdc = qec.multi_decoder_config()
    mdc.decoders = [dc]
    mdc.transport.provider = "udp"
    mdc.transport.args = ["--num-slots=8"]
    mdc.transport.device_graph.provider = "gpu_roce"
    mdc.transport.device_graph.args = ["--pinned-rings"]
    text = mdc.to_yaml_str()
    assert "dispatch:        device_graph" in text and "cuda_device_id:  1" in text and "  args:\n    - '--num-slots=8'" in text
    check_roundtrip(mdc)
    assert qec.decoder_config.from_yaml_str(dc.to_yaml_str().replace("cuda_device_id:  1", "cuda_device_id:  <none>")).cuda_device_id is None


def test_multi_decoder_config_yaml_roundtrip():
    d1 = qec.decoder_config()
    d1.id = 0
    d1.type = "nv-qldpc-decoder"
    d1.block_size = 10
    d1.syndrome_size = 3
    d1.H_sparse = [1, 2, 3, -1, 6, 7, 8, -1, -1]
    d1.decoder_custom_args = {"use_sparsity": True, "error_rate": 0.01, "error_rate_vec": [0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1], "max_iterations": 50}
    d2 = qec.decoder_config()
    d2.id = 1
    d2.type = "multi_error_lut"
    d2.block_size = 10
    d2.syndrome_size = 3
    d2.H_sparse = [1, 2, 3, -1, 6, 7, 8, -1, -1]
    d2.decoder_custom_args = {"lut_error_depth": 3}
    mdc = qec.multi_decoder_config()
    mdc.decoders = [d1, d2]
    mdc2 = qec.multi_decoder_config.from_yaml_str(mdc.to_yaml_str())
    assert sorted(d.id for d in mdc2.decoders) == [0, 1] and mdc2 == mdc
    # The decoders list is copied in and out.
    got = mdc2.decoders
    got[0].id = 7
    assert mdc2.decoders[0].id == 0


def test_nv_qldpc_full_settings_roundtrip():
    config = empty_lut_dc(0)
    config.type = "nv-qldpc-decoder"
    config.decoder_custom_args = {
        "use_sparsity": True, "max_iterations": 50, "use_osd": True, "osd_order": 60, "osd_method": 3, "error_rate_vec": [0.1] * 20, "n_threads": 128,
        "bp_batch_size": 1, "osd_batch_size": 16, "iter_per_check": 2, "clip_value": 10.0, "bp_method": 3, "scale_factor": 1.0, "proc_float": "fp64",
        "gamma0": 0.0, "gamma_dist": [0.1, 0.2], "srelay_config": {"pre_iter": 5, "num_sets": 10, "stopping_criterion": "NConv", "stop_nconv": 10},
        "explicit_gammas": [[0.1] * 20 for _ in range(10)], "bp_seed": 42, "composition": 1,
    }
    mdc = qec.multi_decoder_config()
    mdc.decoders = [config, empty_lut_dc(1)]
    check_roundtrip(mdc)
    # (CUDA-Q builds it; this port refuses settings its nv-qldpc-decoder lacks, here
    # iter_per_check and clip_value, so configuring says why and returns 4.)
    with pytest.warns(RuntimeWarning, match="Error initializing decoders: .*nv-qldpc-decoder"):
        assert qec.configure_decoders(mdc) == 4
    qec.finalize_decoders()


def test_lut_decoders_roundtrip_and_build():
    for typ, args in (("multi_error_lut", {"lut_error_depth": 2}), ("single_error_lut", {})):
        config = empty_lut_dc(0)
        config.type = typ
        config.decoder_custom_args = args
        mdc = qec.multi_decoder_config()
        mdc.decoders = [config]
        check_roundtrip(mdc)
        check_creation(mdc)


def test_sliding_window_decoder():
    n_rounds, n_errs, n_syn = 4, 30, 10
    pcm = qec.generate_random_pcm(n_rounds=n_rounds, n_errs_per_round=n_errs, n_syndromes_per_round=n_syn, weight=3, seed=13)
    pcm = qec.sort_pcm_columns(pcm, n_syn)
    config = empty_lut_dc(0)
    config.type = "sliding_window"
    config.block_size = n_rounds * n_errs
    config.syndrome_size = n_rounds * n_syn
    config.H_sparse = qec.pcm_to_sparse_vec(pcm)
    config.O_sparse = qec.pcm_to_sparse_vec(np.zeros((2, config.block_size), dtype=np.uint8))
    config.D_sparse = qec.generate_timelike_sparse_detector_matrix(config.syndrome_size, 2, include_first_round=False)
    config.decoder_custom_args = {
        "window_size": 1, "step_size": 1, "num_syndromes_per_round": n_syn, "straddle_start_round": False, "straddle_end_round": True,
        "error_rate_vec": [0.1] * config.block_size, "inner_decoder_name": "multi_error_lut", "inner_decoder_params": {"lut_error_depth": 2},
    }
    mdc = qec.multi_decoder_config()
    mdc.decoders = [config]
    check_roundtrip(mdc)
    check_creation(mdc)


def test_incomplete_flat_config_is_rejected():
    complete = {
        "block_size": "block_size: 3",
        "syndrome_size": "syndrome_size: 2",
        "H_sparse": "H_sparse: [ 0, -1, 1, -1 ]",
        "O_sparse": "O_sparse: [ 0, -1 ]",
        "D_sparse": "D_sparse: [ 0, -1, 0, 1, -1 ]",
    }
    header = "id: 0\ntype: single_error_lut\n"
    assert qec.decoder_config.from_yaml_str(header + "\n".join(complete.values()))
    for omitted in complete:
        with pytest.raises(RuntimeError, match=f"decoder 0 is missing required field\\(s\\): {omitted}"):
            qec.decoder_config.from_yaml_str(header + "\n".join(v for k, v in complete.items() if k != omitted))
    with pytest.raises(RuntimeError, match="missing required field"):
        qec.decoder_config.from_yaml_str(header)


def test_flat_config_checks():
    header = "id: 0\ntype: single_error_lut\nblock_size: 3\n"
    with pytest.raises(RuntimeError, match="Number of rows in H_sparse vector is not equal to syndrome_size: 1 != 2"):
        qec.decoder_config.from_yaml_str(header + "syndrome_size: 2\nH_sparse: [0, -1]\nO_sparse: []\nD_sparse: []\n")
    with pytest.raises(RuntimeError, match="Value in O_sparse vector is out of range: 3"):
        qec.decoder_config.from_yaml_str(header + "syndrome_size: 1\nH_sparse: [0, -1]\nO_sparse: [3, -1]\nD_sparse: []\n")
    with pytest.raises(RuntimeError, match="D_sparse row is empty for decoder 0"):
        qec.decoder_config.from_yaml_str(header + "syndrome_size: 2\nH_sparse: [0, -1, 1, -1]\nO_sparse: []\nD_sparse: [0, -1, -1]\n")
    dc = pymatching_dc()
    dc.H_sparse = [0, -1]
    with pytest.raises(RuntimeError, match="Number of rows in H_sparse"):
        dc.to_yaml_str()


def test_stim_dem_path_form():
    dc = qec.decoder_config()
    dc.id = 2
    dc.type = "chromobius"
    dc.stim_dem_path = "/tmp/a b/model.dem"
    dc.D_sparse = [0, -1, 1, -1]
    text = dc.to_yaml_str()
    assert "block_size" not in text and "H_sparse" not in text and "stim_dem_path:   '/tmp/a b/model.dem'" in text
    assert qec.decoder_config.from_yaml_str(text) == dc
    dc.block_size = 3
    with pytest.raises(RuntimeError, match="block_size must not be set for decoder 2 because it is derived from stim_dem_path"):
        dc.to_yaml_str()
    with pytest.raises(RuntimeError, match="D_sparse is required for decoder 2 alongside stim_dem_path"):
        qec.decoder_config.from_yaml_str("id: 2\ntype: chromobius\nstim_dem_path: m.dem\n")


def test_dem_chunks_is_refused_and_expand_dem_chunks_is_a_no_op():
    with pytest.raises(RuntimeError, match="dem_chunks is not supported"):
        qec.decoder_config.from_yaml_str("id: 0\ntype: single_error_lut\ndem_chunks:\n  num_rounds: 3\n")
    dc = pymatching_dc()
    with pytest.raises(RuntimeError, match="dem_chunks is not supported"):
        dc.dem_chunks = {}
    assert dc.dem_chunks is None
    before = dc.H_sparse
    assert qec.expand_dem_chunks(dc) is None and qec.qecrt.config.expand_dem_chunks(dc) is None
    assert dc.H_sparse == before


# -- JSON Schema ------------------------------------------------------------------------------------


def test_decoder_config_json_schema_validates_yaml_documents():
    jsonschema = pytest.importorskip("jsonschema")
    yaml = pytest.importorskip("yaml")
    schema = json.loads(qec.decoder_config_json_schema())
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(schema)
    assert set(qec.registered_decoder_schemas()) == set(schema["$defs"]["decoder_params"])
    dc = pymatching_dc()
    dc.decoder_custom_args = {"error_rate_vec": [0.1, 0.1, 0.1], "merge_strategy": "smallest_weight"}
    validator.validate(yaml.safe_load(qec_yaml_for(dc)))

    def broken(edit):
        doc = yaml.safe_load(qec_yaml_for(dc))
        edit(doc)
        with pytest.raises(jsonschema.ValidationError):
            validator.validate(doc)

    broken(lambda d: d["decoders"][0]["decoder_custom_args"].update(merge_stratgey=d["decoders"][0]["decoder_custom_args"].pop("merge_strategy")))
    broken(lambda d: d["decoders"][0].update(type="sliding_window", decoder_custom_args={"window_size": 2}))
    broken(lambda d: d["decoders"][0].update(type="decoder_without_registered_schema"))
    broken(lambda d: d["decoders"][0].pop("type"))
    broken(lambda d: d["decoders"][0].update(dispatch="gpu_roce"))
    broken(lambda d: d.update(transport={"bogus": 1}))
