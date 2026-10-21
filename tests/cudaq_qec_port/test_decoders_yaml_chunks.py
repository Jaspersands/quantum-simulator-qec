# CUDA-Q QEC's chunk-form decoder configurations (NVIDIA/cudaqx 0.8.1, libs/qec: python/tests/
# test_decoders_yaml.py lines 280-557 and unittests/test_decoders_yaml.cpp lines 1365-1975, the
# dem_chunks and stim_dem_path tests; Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates,
# Apache License 2.0), re-typed and adapted: ``decoder_config.dem_chunks`` (a ``DemChunksSpec``) read
# from and written to YAML, ``expand_dem_chunks`` and the checks around them, the decoder built from
# them. The C++ cases name the test-only ``sample_decoder``; here it is ``single_error_lut``. Added:
# malformed documents (unknown fields, missing keys, wrong types, truncations) raise CUDA-Q's errors.
import json
import re
import warnings

import numpy as np
import pytest

from stabilizer_qec import cudaq_qec as qec
from stabilizer_qec.cudaq_qec import _realtime

NEXT, PREV = qec.seam_name.next_round, qec.seam_name.prev_round


@pytest.fixture(autouse=True)
def _finalize():
    yield
    qec.finalize_decoders()


# -- the d=5 repetition code as phases (test_decoders_yaml.py) --------------------------------------------------

REP5_CHECKS = 4
BULK_H_SPARSE = "[ 0, 1, 5, -1, 1, 2, 6, -1, 2, 3, 7, -1, 3, 4, 8, -1 ]"
BULK_RATES = "[ 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02 ]"


def chunk_form_yaml(num_rounds):
    """A d=5 repetition code written as phases rather than flat matrices. num_rounds lives inside
    dem_chunks, alongside the phases and the connection graph it expands. Pass None to omit it (the
    streaming form)."""
    rounds = "" if num_rounds is None else f"  num_rounds: {num_rounds}\n"
    return f"""
id: 0
type: single_error_lut
dem_chunks:
  seam:
    from: next_round
    to: prev_round
  connections:
    - {{from: init, to: bulk}}
    - {{from: bulk, to: bulk}}
    - {{from: bulk, to: final}}
{rounds}  phases:
    - name: init
      spec:
        num_faults: 9
        H_sparse: {BULK_H_SPARSE}
        O_sparse: [ 0, -1 ]
        error_rates: {BULK_RATES}
    - name: bulk
      spec:
        num_faults: 9
        H_sparse: {BULK_H_SPARSE}
        O_sparse: [ 0, -1 ]
        error_rates: {BULK_RATES}
    - name: final
      spec:
        num_faults: 5
        H_sparse: [ 0, 1, -1, 1, 2, -1, 2, 3, -1, 3, 4, -1 ]
        O_sparse: [ 0, -1 ]
        error_rates: [ 0.02, 0.02, 0.02, 0.02, 0.02 ]
"""


def expected_syndrome_size(num_rounds):
    """init's prev_round seam gives the initial-state detector band, and each inter-chunk boundary
    gives one more: num_rounds bands total."""
    return num_rounds * REP5_CHECKS


def expected_block_size(num_rounds):
    """init plus (num_rounds - 2) bulk copies at 9 faults each, and a destructively measured final
    phase with only its 5 data columns."""
    return 9 * (num_rounds - 1) + 5


def test_chunk_form_parses_with_no_flat_matrices():
    config = qec.decoder_config.from_yaml_str(chunk_form_yaml(5))

    assert config.dem_chunks is not None
    assert config.dem_chunks.num_rounds == 5
    assert len(config.dem_chunks.phases) == 3
    assert len(config.dem_chunks.connections) == 3
    assert config.dem_chunks.has_repeating_phase()
    assert config.dem_chunks.get_phase(qec.SeamId("init")).num_faults == 9
    assert config.dem_chunks.get_phase(qec.SeamId("final")).num_faults == 5
    # The flat fields are derived, so they stay unset until expansion.
    assert config.block_size == 0
    assert config.syndrome_size == 0
    assert len(config.H_sparse) == 0


def test_expand_dem_chunks_fills_the_flat_fields():
    config = qec.decoder_config.from_yaml_str(chunk_form_yaml(5))
    closed = qec.qecrt.config.expand_dem_chunks(config)

    assert closed is not None
    assert config.syndrome_size == expected_syndrome_size(5)
    assert config.block_size == expected_block_size(5)
    assert len(closed.error_rates) == config.block_size
    # One -1 terminator per row, so H and D are as tall as the DEM.
    assert config.H_sparse.count(-1) == config.syndrome_size
    assert config.D_sparse.count(-1) == config.syndrome_size
    assert config.O_sparse.count(-1) == 1


def test_both_forms_warns_and_flat_form_wins():
    """A leftover H_sparse disables the entire chunk spec. expand_dem_chunks() produces that same
    state, so it has to keep parsing, but it must say so."""
    yaml = (
        chunk_form_yaml(5)
        + """block_size: 2
syndrome_size: 2
H_sparse: [ 0, -1, 1, -1 ]
O_sparse: [ 0, -1 ]
D_sparse: [ 0, -1, 1, -1 ]
"""
    )
    with pytest.warns(RuntimeWarning, match="both H_sparse and dem_chunks") as record:
        config = qec.decoder_config.from_yaml_str(yaml)
    # (CUDA-Q's whole message, decoder id included.)
    assert str(record[0].message) == (
        "decoder 0 sets both H_sparse and dem_chunks. The flat H_sparse matrix is what decoders are built from, and dem_chunks is ignored, including its num_rounds. "
        "This is expected for a config already expanded by expand_dem_chunks(); otherwise drop H_sparse to decode from dem_chunks."
    )
    # The flat matrix is what a decoder would be built from, and dem_chunks survives so the document still round-trips.
    assert config.syndrome_size == 2
    assert list(config.H_sparse) == [0, -1, 1, -1]
    assert config.dem_chunks is not None


def test_expand_dem_chunks_is_a_no_op_on_a_flat_config():
    config = qec.decoder_config.from_yaml_str(chunk_form_yaml(3))
    qec.qecrt.config.expand_dem_chunks(config)
    flat_H = list(config.H_sparse)

    # Already flat now, so a second call must leave it alone.
    assert qec.qecrt.config.expand_dem_chunks(config) is None
    assert list(config.H_sparse) == flat_H


def test_same_chunks_serve_different_round_counts():
    sizes = {}
    # 3 is the shortest sequence these phases expand to (init, bulk, final).
    for rounds in (3, 4, 5, 8):
        config = qec.decoder_config.from_yaml_str(chunk_form_yaml(rounds))
        qec.qecrt.config.expand_dem_chunks(config)
        sizes[rounds] = (config.syndrome_size, config.block_size)

    for rounds, (syndrome_size, block_size) in sizes.items():
        assert syndrome_size == expected_syndrome_size(rounds)
        assert block_size == expected_block_size(rounds)


def test_chunk_form_builds_a_working_decoder():
    config = qec.decoder_config.from_yaml_str(chunk_form_yaml(5))
    multi_config = qec.multi_decoder_config()
    multi_config.decoders = [config]

    assert qec.qecrt.config.configure_decoders(multi_config) == 0
    qec.qecrt.config.finalize_decoders()


def test_num_rounds_is_optional_until_expansion():
    # Streaming configs leave num_rounds out: the round count is unknown when the file is written.
    # Expansion is what needs it, because the repeating phase has no other way to know how many
    # copies to emit.
    config = qec.decoder_config.from_yaml_str(chunk_form_yaml(None))
    assert config.dem_chunks.num_rounds is None

    with pytest.raises(RuntimeError, match="Cannot expand dem_chunks for decoder 0: dem_chunks_spec::phase_sequence: num_rounds required when a self-loop is present"):
        qec.qecrt.config.expand_dem_chunks(config)


def test_num_rounds_is_rejected_outside_dem_chunks():
    yaml_str = """
id: 0
type: single_error_lut
num_rounds: 5
block_size: 3
syndrome_size: 2
H_sparse: [ 0, -1, 1, -1 ]
O_sparse: [ 0, -1 ]
D_sparse: [ 0, -1, 0, 1, -1 ]
"""
    with pytest.raises(RuntimeError, match=r"YAML:\d+:\d+: unknown key 'num_rounds'"):
        qec.decoder_config.from_yaml_str(yaml_str)


def test_num_rounds_below_two_is_rejected():
    # num_rounds counts init and final as well as the bulk copies between them, so 2 is the
    # smallest value the parser accepts.
    for rounds in (0, 1):
        with pytest.raises(RuntimeError, match=rf"dem_chunks.num_rounds must be at least 2 for decoder 0; got {rounds}\."):
            qec.decoder_config.from_yaml_str(chunk_form_yaml(rounds))

    config = qec.decoder_config.from_yaml_str(chunk_form_yaml(2))
    assert config.dem_chunks.num_rounds == 2
    # These particular phases need three rounds to lay out, so 2 parses but cannot expand. The
    # parser only enforces the floor common to every spec.
    with pytest.raises(RuntimeError, match="Cannot expand dem_chunks for decoder 0: dem_chunks_spec::phase_sequence: num_rounds 2 too small for 3 non-repeating phases"):
        qec.qecrt.config.expand_dem_chunks(config)


def test_incomplete_flat_config_is_rejected():
    # dem_chunks made the flat fields optional in the mapping; a flat config still has to name all
    # of them, or it parses into an empty DEM that only fails when a decoder is built from it.
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
        yaml_str = header + "\n".join(line for key, line in complete.items() if key != omitted)
        with pytest.raises(RuntimeError, match=rf"decoder 0 is missing required field\(s\): {omitted}\. "):
            qec.decoder_config.from_yaml_str(yaml_str)


def test_config_describing_no_dem_at_all_is_rejected():
    with pytest.raises(RuntimeError, match="decoder 0 is missing required field"):
        qec.decoder_config.from_yaml_str("id: 0\ntype: single_error_lut\n")


def test_expand_dem_chunks_is_exported_at_top_level():
    config = qec.decoder_config.from_yaml_str(chunk_form_yaml(5))
    assert qec.expand_dem_chunks(config) is not None
    assert config.block_size == expected_block_size(5)


def test_expand_dem_chunks_is_a_method_too():
    # (This package's addition beside CUDA-Q's function: the same expansion, the same closed model.)
    config = qec.decoder_config.from_yaml_str(chunk_form_yaml(5))
    closed = config.expand_dem_chunks()
    assert closed is not None and closed.num_error_mechanisms() == expected_block_size(5)
    assert config.block_size == expected_block_size(5)
    assert config.expand_dem_chunks() is None
    assert qec.decoder_config().expand_dem_chunks() is None


def test_derived_fields_are_rejected_in_chunk_form():
    with_block_size = chunk_form_yaml(5).replace("type: single_error_lut", f"type: single_error_lut\nblock_size: {expected_block_size(5)}")
    with pytest.raises(RuntimeError, match="block_size must not be set for decoder 0 because it is derived from dem_chunks. Remove it, or describe the whole experiment with H_sparse instead of dem_chunks."):
        qec.decoder_config.from_yaml_str(with_block_size)


def _phase_entry(name, num_faults, h_sparse):
    entry = qec.PhaseSpecEntry()
    entry.id = qec.SeamId(name)
    entry.spec.num_faults = num_faults
    entry.spec.H_sparse = h_sparse
    entry.spec.O_sparse = [0, -1]
    entry.spec.error_rates = [0.02] * num_faults
    return entry


def _connection(from_name, to_name):
    connection = qec.PhaseConnection()
    connection.from_phase = qec.SeamId(from_name)
    connection.to_phase = qec.SeamId(to_name)
    return connection


def programmatic_config(rounds=4):
    config = qec.decoder_config()
    config.id = 0
    config.type = "single_error_lut"

    bulk_h = [0, 1, 5, -1, 1, 2, 6, -1, 2, 3, 7, -1, 3, 4, 8, -1]
    seam = qec.SeamConnection()
    seam.from_seam = qec.seam_name.next_round
    seam.to_seam = qec.seam_name.prev_round

    spec = qec.DemChunksSpec()
    spec.phases = [
        _phase_entry("init", 9, bulk_h),
        _phase_entry("bulk", 9, bulk_h),
        _phase_entry("final", 5, [0, 1, -1, 1, 2, -1, 2, 3, -1, 3, 4, -1]),
    ]
    spec.connections = [
        _connection("init", "bulk"),
        _connection("bulk", "bulk"),
        _connection("bulk", "final"),
    ]
    spec.seam = seam
    spec.num_rounds = rounds
    config.dem_chunks = spec
    return config


def test_chunk_form_can_be_built_programmatically():
    config = programmatic_config(4)
    qec.qecrt.config.expand_dem_chunks(config)
    assert config.syndrome_size == expected_syndrome_size(4)
    assert config.block_size == expected_block_size(4)


# -- dem_chunks: named-phase DEM for a repeated-round decomposition (test_decoders_yaml.cpp) ----------------------------------------


def dem_chunks_yaml(num_rounds=3, type_="single_error_lut", extra=""):
    """A simple 1-check, 2-fault code: H = [[f0, f1]], f0 flips the observable. Phases: init, bulk,
    final, each with num_faults=2 and H_sparse=[0,1,-1]."""
    return f"""
decoders:
  - id: 0
    type: {type_}
{extra}    dem_chunks:
      seam:
        from: next_round
        to: prev_round
      connections:
        - {{from: init, to: bulk}}
        - {{from: bulk, to: bulk}}
        - {{from: bulk, to: final}}
      num_rounds: {num_rounds}
      phases:
        - name: init
          spec:
            num_faults: 2
            H_sparse: [0, 1, -1]
            O_sparse: [0, -1]
            error_rates: [0.01, 0.01]
        - name: bulk
          spec:
            num_faults: 2
            H_sparse: [0, 1, -1]
            O_sparse: [0, -1]
            error_rates: [0.01, 0.01]
        - name: final
          spec:
            num_faults: 2
            H_sparse: [0, 1, -1]
            O_sparse: [0, -1]
            error_rates: [0.01, 0.01]
"""


def parse_one(yaml):
    config = qec.multi_decoder_config.from_yaml_str(yaml)
    assert len(config.decoders) == 1
    return config.decoders[0]


def test_parses_all_phases():
    config = qec.multi_decoder_config.from_yaml_str(dem_chunks_yaml(3))
    assert len(config.decoders) == 1
    chunks = config.decoders[0].dem_chunks
    assert chunks is not None
    assert chunks.num_rounds == 3
    assert len(chunks.phases) == 3
    assert len(chunks.connections) == 3
    assert not chunks.is_empty()
    assert chunks.has_repeating_phase()


def test_parsed_phases_expand_and_close():
    config = qec.multi_decoder_config.from_yaml_str(dem_chunks_yaml(4))
    spec = config.decoders[0].dem_chunks

    expanded = qec.dem_chunks_from_spec(spec)
    assert len(expanded) == 4

    flat = qec.dem_close_all(expanded)
    # init's prev_round band + 3 inter-chunk boundaries = 4 detectors; 4 rounds × 2 faults = 8 fault columns.
    assert flat.detector_error_matrix.shape == (4, 8)
    assert flat.observables_flips_matrix.shape[0] == 1


def test_section_is_optional():
    yaml = """
decoders:
  - id: 0
    type: multi_error_lut
    block_size: 3
    syndrome_size: 3
    H_sparse: [0, -1, 1, -1, 2, -1]
    O_sparse: [0, -1, 1, -1, 2, -1]
    D_sparse: [0, -1, 1, -1, 2, -1]
"""
    config = qec.multi_decoder_config.from_yaml_str(yaml)
    assert config.decoders[0].dem_chunks is None
    assert "dem_chunks" not in config.to_yaml_str(200)


def test_chunk_form_parses_with_no_flat_matrices_cpp():
    config = parse_one(dem_chunks_yaml(5))
    assert config.H_sparse == [] and config.O_sparse == [] and config.D_sparse == []
    assert config.block_size == 0 and config.syndrome_size == 0
    assert config.dem_chunks is not None
    assert config.dem_chunks.num_rounds == 5


def test_expands_to_a_self_consistent_flat_config():
    config = parse_one(dem_chunks_yaml(5))
    closed = qec.qecrt.config.expand_dem_chunks(config)

    assert closed is not None
    assert config.block_size > 0
    assert config.syndrome_size > 0
    assert config.H_sparse.count(-1) == config.syndrome_size
    assert config.O_sparse.count(-1) == 1
    # 5 rounds × 2 faults = 10 fault columns
    assert config.block_size == 10
    # init's prev_round band + 4 inter-chunk boundaries = 5 detectors
    assert config.syndrome_size == 5
    # Expanded config is flat, so re-expanding is a no-op.
    assert qec.qecrt.config.expand_dem_chunks(config) is None


def test_same_connections_different_round_counts():
    three, five = parse_one(dem_chunks_yaml(3)), parse_one(dem_chunks_yaml(5))
    qec.qecrt.config.expand_dem_chunks(three)
    qec.qecrt.config.expand_dem_chunks(five)
    assert three.syndrome_size < five.syndrome_size
    assert three.block_size == 6  # 3 rounds × 2 faults
    assert five.block_size == 10  # 5 rounds × 2 faults


@pytest.mark.parametrize("derived", ["    syndrome_size: 3\n", "    block_size: 10\n", "    O_sparse: [0, -1]\n", "    D_sparse: [0, -1]\n"])
def test_derived_fields_are_rejected(derived):
    yaml = "\ndecoders:\n  - id: 3\n    type: single_error_lut\n" + derived + dem_chunks_yaml(3)[dem_chunks_yaml(3).find("    dem_chunks:") :]
    key = derived.split(":")[0].strip()
    with pytest.raises(RuntimeError, match=rf"{key} must not be set for decoder 3 because it is derived from dem_chunks"):
        qec.multi_decoder_config.from_yaml_str(yaml)


def test_expanded_config_is_itself_a_valid_flat_config():
    config = parse_one(dem_chunks_yaml(3))
    qec.qecrt.config.expand_dem_chunks(config)

    wrapper = qec.multi_decoder_config()
    wrapper.decoders = [config]
    emitted = wrapper.to_yaml_str(200)
    assert "H_sparse" in emitted
    assert "dem_chunks" in emitted
    # (And it reads back as it was: flat, with its spec still there, which it says.)
    with pytest.warns(RuntimeWarning, match="both H_sparse and dem_chunks"):
        back = qec.multi_decoder_config.from_yaml_str(emitted)
    assert back == wrapper


def test_both_forms_warn_and_flat_form_wins_cpp():
    chunks = dem_chunks_yaml(3)
    insert_at = chunks.find("    dem_chunks:")
    assert insert_at >= 0
    yaml = chunks[:insert_at] + "    block_size: 2\n    syndrome_size: 2\n    H_sparse: [0, -1, 1, -1]\n    O_sparse: [0, -1]\n    D_sparse: [0, -1, 1, -1]\n" + chunks[insert_at:]

    with pytest.warns(RuntimeWarning, match="both H_sparse and dem_chunks"):
        config = parse_one(yaml)
    # The flat matrix wins, and dem_chunks survives so the document still round-trips.
    assert config.syndrome_size == 2
    assert config.H_sparse == [0, -1, 1, -1]
    assert config.dem_chunks is not None


def test_chunk_form_alone_does_not_warn():
    # The warning must stay quiet for a config that uses the chunk form as intended, otherwise it is noise on the common path.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        config = parse_one(dem_chunks_yaml(3))
    assert config.H_sparse == []


def test_h_sparse_still_required_without_chunks():
    yaml = """
decoders:
  - id: 7
    type: multi_error_lut
    block_size: 3
    syndrome_size: 3
    O_sparse: [0, -1, 1, -1, 2, -1]
    D_sparse: [0, -1, 1, -1, 2, -1]
"""
    with pytest.raises(RuntimeError, match="H_sparse") as e:
        qec.multi_decoder_config.from_yaml_str(yaml)
    assert str(e.value).startswith("decoder 7 is missing required field(s): H_sparse.")


@pytest.mark.parametrize(
    "omitted, body",
    [
        ("block_size", "    syndrome_size: 3\n    H_sparse: [0, -1, 1, -1, 2, -1]\n    O_sparse: [0, -1]\n    D_sparse: [0, -1, 1, -1, 2, -1]\n"),
        ("syndrome_size", "    block_size: 3\n    H_sparse: [0, -1, 1, -1, 2, -1]\n    O_sparse: [0, -1]\n    D_sparse: [0, -1, 1, -1, 2, -1]\n"),
        ("O_sparse", "    block_size: 3\n    syndrome_size: 3\n    H_sparse: [0, -1, 1, -1, 2, -1]\n    D_sparse: [0, -1, 1, -1, 2, -1]\n"),
        ("D_sparse", "    block_size: 3\n    syndrome_size: 3\n    H_sparse: [0, -1, 1, -1, 2, -1]\n    O_sparse: [0, -1]\n"),
    ],
)
def test_incomplete_flat_config_is_rejected_at_parse(omitted, body):
    yaml = "decoders:\n  - id: 0\n    type: multi_error_lut\n" + body
    with pytest.raises(RuntimeError, match=omitted):
        qec.multi_decoder_config.from_yaml_str(yaml)


def test_config_with_neither_form_is_rejected_at_parse():
    with pytest.raises(RuntimeError):
        qec.multi_decoder_config.from_yaml_str("\ndecoders:\n  - id: 0\n    type: multi_error_lut\n")


def test_num_rounds_below_two_is_rejected_cpp():
    # num_rounds must be at least 2 (init + final).
    for rounds in (0, 1):
        with pytest.raises(RuntimeError, match="num_rounds"):
            qec.multi_decoder_config.from_yaml_str(dem_chunks_yaml(rounds))
    # Two rounds (init + final, no bulk) is accepted.
    assert parse_one(dem_chunks_yaml(2)).dem_chunks.num_rounds == 2


def test_json_schema_num_rounds_minimum_matches_the_parser():
    schema = json.loads(qec.decoder_config_json_schema())
    assert schema["$defs"]["dem_chunks"]["properties"]["num_rounds"]["minimum"] == 2


def test_streaming_config_parses_without_num_rounds():
    yaml = dem_chunks_yaml(3)
    without = yaml[: yaml.find("      num_rounds:")] + yaml[yaml.find("      phases:") :]
    config = parse_one(without)
    assert config.dem_chunks is not None
    assert config.dem_chunks.num_rounds is None

    # The schema has to agree, or a validating tool would reject a streaming configuration the parser accepts.
    schema = json.loads(qec.decoder_config_json_schema())
    assert "num_rounds" not in schema["$defs"]["dem_chunks"]["required"]

    # Expansion is where the missing round count is caught.
    with pytest.raises(RuntimeError, match="num_rounds required when a self-loop is present"):
        qec.expand_dem_chunks(config)


def test_json_schema_validates_chunk_form_documents():
    jsonschema = pytest.importorskip("jsonschema")
    yaml = pytest.importorskip("yaml")
    validator = jsonschema.Draft202012Validator(json.loads(qec.decoder_config_json_schema()))
    document = yaml.safe_load(dem_chunks_yaml(4))
    assert list(validator.iter_errors(document)) == []
    parse_one(dem_chunks_yaml(4))
    # (num_rounds below the floor is the schema's error as well as the parser's.)
    assert list(validator.iter_errors(yaml.safe_load(dem_chunks_yaml(1))))


# -- stim_dem_path: the DEM text itself as the model a decoder is built from (test_decoders_yaml.cpp) --------------------------------

STIM_DEM = """
error(0.1) D0 D1
error(0.1) D0 D1 D2
error(0.1) D0 L0
error(0.1) D1 D2 D3
error(0.1) D2 D3
error(0.1) D3
detector(0, 0, 0, 1) D0
detector(1, 0, 0, 2) D1
detector(2, 0, 0, 0) D2
detector(3, 0, 0, 1) D3
"""
STIM_DEM_D_SPARSE = [0, -1, 1, -1, 2, -1, 3, -1]


@pytest.fixture
def dem_path(tmp_path):
    path = tmp_path / "model.dem"
    path.write_text(STIM_DEM, encoding="utf-8")
    return str(path)


def dem_form_yaml(path, extra_keys=""):
    return f"""
decoders:
  - id: 0
    type: multi_error_lut
    stim_dem_path: {path}
    D_sparse: [0, -1, 1, -1, 2, -1, 3, -1]
{extra_keys}"""


def test_parses_a_dem_form_document(dem_path):
    config = parse_one(dem_form_yaml(dem_path))
    assert config.stim_dem_path == dem_path
    assert config.H_sparse == [] and config.O_sparse == []
    assert config.block_size == 0 and config.syndrome_size == 0
    assert config.D_sparse == STIM_DEM_D_SPARSE


def test_emitted_document_re_parses_as_dem_form(dem_path):
    multi_config = qec.multi_decoder_config()
    multi_config.decoders = [parse_one(dem_form_yaml(dem_path))]

    # The fields DEM form derives have to stay absent, or the emitted document would come back as an incomplete flat configuration.
    emitted = multi_config.to_yaml_str(200)
    assert "stim_dem_path" in emitted
    for derived in ("H_sparse", "O_sparse", "block_size", "syndrome_size"):
        assert derived not in emitted, emitted

    round_tripped = qec.multi_decoder_config.from_yaml_str(emitted)
    assert round_tripped == multi_config
    assert round_tripped.to_yaml_str(200) == emitted


def test_rejects_stim_dem_path_together_with_dem_chunks(dem_path):
    # Each names the whole DEM on its own, and there is no precedence between them the way a flat H_sparse wins over dem_chunks.
    yaml = dem_chunks_yaml(3) + "    stim_dem_path: " + dem_path + "\n"
    with pytest.raises(RuntimeError, match="dem_chunks and stim_dem_path must not both be set for decoder 0; each describes the whole DEM on its own."):
        qec.multi_decoder_config.from_yaml_str(yaml)


@pytest.mark.parametrize("derived", ["    block_size: 6\n", "    syndrome_size: 4\n", "    H_sparse: [0, 1, 2, -1]\n", "    O_sparse: [2, -1]\n"])
def test_rejects_stim_dem_path_together_with_flat_matrices(dem_path, derived):
    with pytest.raises(RuntimeError, match=rf"{derived.split(':')[0].strip()} must not be set for decoder 0 because it is derived from stim_dem_path"):
        qec.multi_decoder_config.from_yaml_str(dem_form_yaml(dem_path, derived))


def test_requires_d_sparse_alongside_stim_dem_path(dem_path):
    # The DEM describes detectors but not which measurement bits form them, so this is the one model field DEM form cannot derive.
    yaml = f"\ndecoders:\n  - id: 0\n    type: multi_error_lut\n    stim_dem_path: {dem_path}\n"
    with pytest.raises(RuntimeError, match="D_sparse is required for decoder 0 alongside stim_dem_path"):
        qec.multi_decoder_config.from_yaml_str(yaml)


def test_json_schema_accepts_dem_form_alongside_the_other_two():
    schema = json.loads(qec.decoder_config_json_schema())
    assert "stim_dem_path" in schema["$defs"]["decoder_config"]["properties"]
    # The three forms are alternatives, so none of them may appear in the envelope's own required list.
    envelope = schema["$defs"]["decoder_config"]["required"]
    assert "id" in envelope and "type" in envelope
    for key in ("stim_dem_path", "dem_chunks", "H_sparse", "D_sparse"):
        assert key not in envelope, (key, envelope)


def test_derives_observables_for_a_decoder_that_installs_none(dem_path):
    # The LUT decoders return error frames and never install an observable mapping of their own, and
    # DEM form names no O_sparse to install for them. Without deriving one from the model, every
    # realtime correction would be empty.
    dc = qec.decoder_config()
    dc.id = 0
    dc.type = "multi_error_lut"
    dc.stim_dem_path = dem_path
    dc.D_sparse = STIM_DEM_D_SPARSE

    decoder = _realtime._create(dc)
    assert decoder.decoder.get_syndrome_size() == 4
    assert decoder.num_observables() == 1


def test_reports_a_stim_dem_path_that_is_not_a_file(tmp_path):
    dc = qec.decoder_config()
    dc.id = 0
    dc.type = "multi_error_lut"
    dc.stim_dem_path = str(tmp_path)
    dc.D_sparse = STIM_DEM_D_SPARSE

    # A directory opens as a stream and only fails on the first read, so the error has to name the key and the path.
    with pytest.raises(RuntimeError) as e:
        _realtime._create(dc)
    assert "stim_dem_path" in str(e.value) and dc.stim_dem_path in str(e.value)


def test_rejects_dem_chunks_with_stim_dem_path_at_construction(dem_path):
    # The parser's exclusivity rules reached the other way: decoder_config has writable fields, so a
    # caller can build one that never was a document. Four rounds close to four detectors, as many as
    # the model declares, so the D row check downstream cannot catch this.
    dc = parse_one(dem_chunks_yaml(4))
    dc.stim_dem_path = dem_path
    with pytest.raises(RuntimeError) as e:
        _realtime._create(dc)
    assert "stim_dem_path" in str(e.value) and "dem_chunks" in str(e.value)
    assert str(e.value) == "dem_chunks and O_sparse must not be set with stim_dem_path for decoder 0"
    # (configure_decoders reports it as a failed build.)
    multi = qec.multi_decoder_config()
    multi.decoders = [dc]
    with pytest.raises(RuntimeError, match="dem_chunks and stim_dem_path must not both be set for decoder 0"):
        multi.to_yaml_str()


def test_rejects_o_sparse_with_stim_dem_path_at_construction(dem_path):
    # block_size, syndrome_size and H_sparse are inert beside a DEM, so O_sparse is the only flat field
    # worth rejecting: it overwrites the mapping a DEM-native decoder installed for itself.
    dc = qec.decoder_config()
    dc.id = 0
    dc.type = "multi_error_lut"
    dc.stim_dem_path = dem_path
    dc.D_sparse = STIM_DEM_D_SPARSE
    dc.O_sparse = [2, -1]
    with pytest.raises(RuntimeError) as e:
        _realtime._create(dc)
    assert "O_sparse" in str(e.value) and "stim_dem_path" in str(e.value)


# -- the decoder built from a chunk-form configuration (realtime_decoding.cpp's create_realtime_decoder) -----------------------------------


def test_chunk_form_decoder_has_the_phases_priors_and_dimensions():
    config = qec.decoder_config.from_yaml_str(chunk_form_yaml(4))
    decoder = _realtime._create(config)
    assert decoder.decoder.get_syndrome_size() == expected_syndrome_size(4)
    assert decoder.decoder.get_block_size() == expected_block_size(4)
    assert decoder.num_observables() == 1
    assert decoder.num_msyn_per_decode() == 4 * REP5_CHECKS
    # (Expanded on a copy: the caller's configuration is still in chunk form.)
    assert config.H_sparse == [] and config.dem_chunks is not None
    # (The phases' priors reach the decoder the flat form would have had to pass as error_rate_vec.)
    assert np.allclose(decoder.decoder._inputs.rates, [0.02] * expected_block_size(4))


def test_chunk_form_decoder_decodes_a_shot_like_the_flat_config():
    chunk_config = qec.decoder_config.from_yaml_str(chunk_form_yaml(4))
    flat = qec.decoder_config.from_yaml_str(chunk_form_yaml(4))
    qec.expand_dem_chunks(flat)
    flat.dem_chunks = None
    rng = np.random.default_rng(7)
    shots = rng.integers(0, 2, size=(25, 4 * REP5_CHECKS), dtype=np.uint8)
    out = []
    for config in (chunk_config, flat):
        multi = qec.multi_decoder_config()
        multi.decoders = [config]
        assert qec.configure_decoders(multi) == 0
        results = []
        for shot in shots:
            qec.enqueue_syndromes(0, shot, 0)
            results.append(qec.get_corrections(0, 1, True))
        out.append(results)
        qec.finalize_decoders()
    assert out[0] == out[1]


def test_configure_from_str_and_status_of_a_bad_spec():
    assert qec.configure_decoders_from_str(dem_chunks_yaml(3)) == 0
    qec.finalize_decoders()
    bad = dem_chunks_yaml(3).replace("seam:\n        from: next_round\n        to: prev_round", "seam:\n        from: next_round\n        to: next_round")
    with pytest.warns(RuntimeWarning, match="Error initializing decoders: Cannot expand dem_chunks for decoder 0: extended_dem::add_seam: duplicate seam next_round"):
        assert qec.configure_decoders_from_str(bad) == 4
    # (A spec the chunks cannot close: seams of different widths between the phases.)
    uneven = dem_chunks_yaml(3).replace("H_sparse: [0, 1, -1]\n            O_sparse: [0, -1]\n            error_rates: [0.01, 0.01]\n        - name: final", "H_sparse: [0, 1, -1, 0, -1]\n            O_sparse: [0, -1]\n            error_rates: [0.01, 0.01]\n        - name: final")
    with pytest.warns(RuntimeWarning, match="Error initializing decoders: Cannot close dem_chunks for decoder 0:"):
        assert qec.configure_decoders_from_str(uneven) == 4


# -- YAML round trips ------------------------------------------------------------------------------------------------------------------------------


def test_chunk_form_round_trips_through_yaml():
    multi = qec.multi_decoder_config()
    multi.decoders = [qec.decoder_config.from_yaml_str(chunk_form_yaml(5))]
    text = multi.to_yaml_str(200)
    assert "H_sparse:        [ 0, 1, 5, -1" in text and "num_rounds:      5" in text
    # (The derived fields stay absent, which is what makes the document re-parse in the form it was written in.)
    assert "block_size" not in text and "syndrome_size" not in text and "D_sparse" not in text
    back = qec.multi_decoder_config.from_yaml_str(text)
    assert back == multi and back.to_yaml_str(200) == text
    assert text.splitlines()[:4] == ["---", "decoders:", "  - id:              0", "    type:            single_error_lut"]


def test_streaming_form_round_trips_without_num_rounds():
    multi = qec.multi_decoder_config()
    multi.decoders = [qec.decoder_config.from_yaml_str(chunk_form_yaml(None))]
    text = multi.to_yaml_str()
    assert "num_rounds" not in text
    back = qec.multi_decoder_config.from_yaml_str(text)
    assert back == multi and back.decoders[0].dem_chunks.num_rounds is None


def test_programmatic_chunk_form_round_trips_with_the_standard_names():
    config = programmatic_config(6)
    text = config.to_yaml_str()
    back = qec.decoder_config.from_yaml_str(text)
    assert back == config
    assert back.dem_chunks.seam.from_seam == NEXT and back.dem_chunks.seam.to_seam == PREV
    assert [e.id for e in back.dem_chunks.phases] == [qec.phase_name.init, qec.phase_name.bulk, qec.phase_name.final]


def test_names_other_than_the_registered_are_written_empty_as_cuda_q_writes_them():
    # (CUDA-Q has only the hashes when it writes, so every name is empty there; here a name registered for the hash, as the standard ones
    # and those a document has read are, is written. A name made only through SeamId(...) is not registered.)
    config = programmatic_config(3)
    spec = config.dem_chunks
    odd = _phase_entry("never_read_from_a_document", 9, [0, 1, -1])
    spec.phases = spec.phases + [odd]
    config.dem_chunks = spec
    text = config.to_yaml_str()
    assert re.search(r"- name:            ''\n", text)
    assert "name:            init" in text


def test_seam_specs_form_reads_writes_and_expands():
    yaml = """
id: 0
type: single_error_lut
dem_chunks:
  seam: {from: next_round, to: prev_round}
  connections: [{from: init, to: final}]
  phases:
    - name: init
      spec:
        num_faults: 3
        seam_specs:
          - name: prev_round
            spec:
              H_sparse: [0, 1, -1, 1, 2, -1]
          - name: next_round
            spec:
              H_sparse: [0, 1, -1, 1, 2, -1]
        O_sparse: [0, -1]
        error_rates: [0.1, 0.1, 0.1]
    - name: final
      spec:
        num_faults: 3
        H_sparse: [0, 1, -1, 1, 2, -1]
        O_sparse: [0, -1]
        error_rates: [0.1, 0.1, 0.1]
"""
    config = qec.decoder_config.from_yaml_str(yaml)
    entry = config.dem_chunks.phases[0].spec.seam_specs
    assert [e.id for e in entry] == [PREV, NEXT]
    assert entry[0].spec.H_sparse == [0, 1, -1, 1, 2, -1]
    back = qec.decoder_config.from_yaml_str(config.to_yaml_str())
    assert back == config
    closed = qec.expand_dem_chunks(config)
    assert closed.detector_error_matrix.shape == (4, 6)
    assert config.syndrome_size == 4 and config.block_size == 6


def test_equality_and_copies_include_dem_chunks():
    a = qec.decoder_config.from_yaml_str(chunk_form_yaml(5))
    b = qec.decoder_config.from_yaml_str(chunk_form_yaml(5))
    assert a == b
    b.dem_chunks.num_rounds = 6
    assert a != b
    c = qec.decoder_config.from_yaml_str(chunk_form_yaml(5))
    c.dem_chunks = None
    assert a != c
    multi = qec.multi_decoder_config()
    multi.decoders = [a]
    copied = multi.decoders[0]
    copied.dem_chunks.num_rounds = 9  # (a copy: the configuration keeps its own)
    assert multi.decoders[0].dem_chunks.num_rounds == 5
    import copy

    d = copy.deepcopy(a)
    d.dem_chunks.num_rounds = 7
    assert a.dem_chunks.num_rounds == 5
    with pytest.raises(TypeError, match="dem_chunks must be a DemChunksSpec or None, not dict"):
        a.dem_chunks = {}
    a.dem_chunks = a.dem_chunks  # (assigning copies the spec in)
    assert a.dem_chunks.num_rounds == 5


def test_dem_chunks_none_is_read_as_no_chunks():
    config = qec.decoder_config.from_yaml_str("id: 0\ntype: single_error_lut\ndem_chunks: <none>\nblock_size: 1\nsyndrome_size: 1\nH_sparse: [ 0, -1 ]\nO_sparse: [ 0, -1 ]\nD_sparse: [ 0, -1 ]\n")
    assert config.dem_chunks is None
    spec = qec.decoder_config.from_yaml_str(chunk_form_yaml(5).replace("num_rounds: 5", "num_rounds: <none>"))
    assert spec.dem_chunks.num_rounds is None


# -- malformed documents (the reference's error, never an IndexError or KeyError) ------------------------------------------------------------------------


def parse_error(yaml, pattern):
    with pytest.raises(RuntimeError, match=pattern):
        qec.decoder_config.from_yaml_str(yaml)


def line_of(yaml, needle):
    for i, line in enumerate(yaml.split("\n"), 1):
        if needle in line:
            return i, line.index(needle) + 1
    raise AssertionError(needle)


YAML_HEAD = "Invalid decoder configuration YAML: Invalid argument "


def test_unknown_fields_are_unknown_keys():
    base = chunk_form_yaml(5)
    for anchor, new_line, name in (
        ("  seam:", "  bogus: 1", "bogus"),
        ("      spec:", "      extra: 2", "extra"),
        ("        num_faults: 9", "        speed: fast", "speed"),
        ("    from: next_round", "    via: x", "via"),
        ("    - {from: init, to: bulk}", "    - {from: init, to: bulk, why: no}", "why"),
    ):
        yaml = base.replace(anchor, new_line + "\n" + anchor, 1) if "why" not in name else base.replace(anchor, new_line, 1)
        parse_error(yaml, rf"{re.escape(YAML_HEAD)}\(YAML:\d+:\d+: unknown key '{name}'\)")
    # (The location is the unknown key's own.)
    yaml = base.replace("  seam:", "  bogus: 1\n  seam:", 1)
    line, col = line_of(yaml, "bogus")
    parse_error(yaml, rf"\(YAML:{line}:{col}: unknown key 'bogus'\)")
    # (An unknown key beside the flat fields of a chunk-form decoder, and a top-level one.)
    parse_error(base + "surprise: 1\n", r"unknown key 'surprise'")


def test_missing_required_keys():
    base = chunk_form_yaml(5)
    cases = [
        ("  seam:\n    from: next_round\n    to: prev_round\n", "seam"),
        ("  connections:\n    - {from: init, to: bulk}\n    - {from: bulk, to: bulk}\n    - {from: bulk, to: final}\n", "connections"),
        ("    from: next_round\n", "from"),
        ("    to: prev_round\n", "to"),
        ("    - name: init\n      spec:", "name"),
        ("        num_faults: 9\n        H_sparse: " + BULK_H_SPARSE + "\n        O_sparse: [ 0, -1 ]\n        error_rates: " + BULK_RATES + "\n    - name: bulk", "num_faults"),
        ("        error_rates: [ 0.02, 0.02, 0.02, 0.02, 0.02 ]\n", "error_rates"),
    ]
    for chunk, key in cases:
        assert chunk in base, chunk
        yaml = base.replace(chunk, "" if key not in ("name", "num_faults") else (chunk.replace("- name: init\n      spec:", "- spec:") if key == "name" else chunk.replace("        num_faults: 9\n", "", 1)), 1)
        parse_error(yaml, rf"{re.escape(YAML_HEAD)}\(YAML:\d+:\d+: missing required key '{key}'\)")
    # (Missing `phases`, the case a truncated or renamed section gives; located at the mapping that lacks it.)
    yaml = base.replace("  phases:", "  phasez:", 1)
    line, col = line_of(yaml, "seam:")
    parse_error(yaml, rf"\(YAML:{line}:{col}: missing required key 'phases'\)")
    # (A `spec` without its sibling `name`; a phase with no `spec`.)
    parse_error(base.replace("      spec:\n        num_faults: 5", "      specs:\n        num_faults: 5"), r"missing required key 'spec'")
    parse_error("id: 0\ntype: single_error_lut\ndem_chunks:\n  num_rounds: 3\n", rf"{re.escape(YAML_HEAD)}\(YAML:4:3: missing required key 'seam'\)")


def test_wrong_types():
    base = chunk_form_yaml(5)

    def swap(old, new, count=1):
        assert old in base, old
        return base.replace(old, new, count)

    def small(body):
        return "id: 0\ntype: single_error_lut\ndem_chunks: " + body + "\n"

    spec = "{ seam: { from: a, to: b }, connections: [ { from: a, to: b } ], phases: [ { name: a, spec: { num_faults: 1, error_rates: [ 0.1 ], H_sparse: [ 0, -1 ] } } ] }"
    cases = [
        (swap("num_faults: 9", "num_faults: [ 9 ]"), "unexpected scalar"),
        (swap("num_faults: 9", "num_faults: nine"), "invalid number"),
        (swap("num_faults: 9", "num_faults: -9"), "invalid number"),
        (swap("num_faults: 9", "num_faults: 99999999999999999999999"), "invalid number"),
        (swap("num_rounds: 5", "num_rounds: five"), "invalid number"),
        (swap("num_rounds: 5", "num_rounds: -1"), "invalid number"),
        (swap("num_rounds: 5", "num_rounds: [ 5 ]"), "unexpected scalar"),
        (swap("H_sparse: " + BULK_H_SPARSE, "H_sparse: 5"), "not a sequence"),
        (swap("H_sparse: " + BULK_H_SPARSE, "H_sparse: [ 0, one, -1 ]"), "invalid number"),
        (swap("H_sparse: " + BULK_H_SPARSE, "H_sparse: [ 0, [ 1 ], -1 ]"), "unexpected scalar"),
        (swap("error_rates: " + BULK_RATES, "error_rates: [ 0.02, fast ]"), "invalid floating point number"),
        (swap("error_rates: " + BULK_RATES, "error_rates: 0.02"), "not a sequence"),
        (swap("    from: next_round\n", "    from: [ next_round ]\n"), "unexpected scalar"),
        (swap("    - name: init\n", "    - name: { a: b }\n"), "unexpected scalar"),
        (swap("    - {from: init, to: bulk}", "    - init"), "not a mapping"),
        (swap("  seam:\n    from: next_round\n    to: prev_round\n", "  seam: [ next_round, prev_round ]\n"), "not a mapping"),
        (swap("  seam:\n    from: next_round\n    to: prev_round\n", "  seam: next_round\n"), "not a mapping"),
        (small(spec.replace("connections: [ { from: a, to: b } ]", "connections: { from: a, to: b }")), "not a sequence"),
        (small(spec.replace("phases: [", "phases: 3, ignored: [")), "not a sequence"),
        (small(spec.replace("spec: { num_faults", "spec: 7, other: { num_faults")), "not a mapping"),
        (small(spec.replace("seam: { from: a, to: b }", "seam: 3")), "not a mapping"),
        (small("5"), "not a mapping"),
        (small("[ 1, 2 ]"), "not a mapping"),
        ("id: 0\ntype: single_error_lut\ndem_chunks:\n", "not a mapping"),
    ]
    for yaml, message in cases:
        with pytest.raises(RuntimeError) as e:
            qec.decoder_config.from_yaml_str(yaml)
        assert re.fullmatch(re.escape(YAML_HEAD) + rf"\(YAML:\d+:\d+: {message}\)", str(e.value)), (message, str(e.value), yaml)


def test_wrong_type_locations():
    base = chunk_form_yaml(5)
    yaml = base.replace("num_faults: 9", "num_faults: nine", 1)
    line, col = line_of(yaml, "nine")
    parse_error(yaml, rf"\(YAML:{line}:{col}: invalid number\)")
    yaml = base.replace("num_rounds: 5", "num_rounds: five", 1)
    line, col = line_of(yaml, "five")
    parse_error(yaml, rf"\(YAML:{line}:{col}: invalid number\)")


def test_post_parse_validation_errors_name_the_decoder():
    # (dem_chunks_spec's cross-phase checks run after the parse, as std::runtime_error.)
    def broken(old, new):
        return chunk_form_yaml(5).replace(old, new, 1)

    cases = [
        (broken("num_faults: 9", "num_faults: 0"), r"dem_chunks.init.num_faults must be positive|dem_chunks.init.error_rates has 9 entries but num_faults=0"),
        (broken("error_rates: " + BULK_RATES, "error_rates: [ 0.02 ]"), r"dem_chunks.init.error_rates has 1 entries but num_faults=9"),
        (broken("error_rates: " + BULK_RATES, "error_rates: [ 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 2.5 ]"), r"dem_chunks.init.error_rates\[8\] out of \[0,1\]"),
        (broken("H_sparse: " + BULK_H_SPARSE, "H_sparse: [ 0, 1, 9, -1 ]"), r"dem_chunks.init.H_sparse index 9 out of range"),
        (broken("H_sparse: " + BULK_H_SPARSE, "H_sparse: [ 0, 1, 5 ]"), r"dem_chunks.init.H_sparse must end with a -1 row terminator"),
        (broken("H_sparse: " + BULK_H_SPARSE, "H_sparse: [ 0, 0, -1 ]"), r"dem_chunks.init.H_sparse duplicate column index 0 in a row"),
        (broken("O_sparse: [ 0, -1 ]", "O_sparse: [ 0, 3, 3, -1 ]"), r"dem_chunks.init.O_sparse duplicate column index 3 in a row"),
    ]
    for yaml, message in cases:
        with pytest.raises(RuntimeError, match=rf"^Invalid dem_chunks for decoder 0: {message}"):
            qec.decoder_config.from_yaml_str(yaml)
    # A chain that is a cycle and one that branches load, and fail at expansion, as CUDA-Q has them:
    cycle = chunk_form_yaml(5).replace("{from: bulk, to: final}", "{from: bulk, to: init}")
    with pytest.raises(RuntimeError, match="Cannot expand dem_chunks for decoder 0: dem_chunks_spec::phase_sequence: connection graph is cyclic; phase init is reachable twice"):
        qec.expand_dem_chunks(qec.decoder_config.from_yaml_str(cycle))
    branching = chunk_form_yaml(5).replace("{from: bulk, to: final}", "{from: init, to: final}")
    with pytest.raises(RuntimeError, match="Cannot expand dem_chunks for decoder 0: dem_chunks_spec::phase_sequence: phase init has multiple non-self outgoing edges"):
        qec.expand_dem_chunks(qec.decoder_config.from_yaml_str(branching))
    # The decoder id, and a later decoder's parse error ahead of an earlier decoder's validation error:
    two = dem_chunks_yaml(3).replace("id: 0", "id: 4").replace("error_rates: [0.01, 0.01]", "error_rates: [0.01]", 1) + "  - id: 5\n    type: single_error_lut\n    block_size: 1\n    syndrome_size: 1\n    H_sparse: [0, -1]\n    O_sparse: [0, -1]\n    D_sparse: [0, -1]\n    surprise: 1\n"
    with pytest.raises(RuntimeError, match="unknown key 'surprise'"):
        qec.multi_decoder_config.from_yaml_str(two)
    with pytest.raises(RuntimeError, match="Invalid dem_chunks for decoder 4: dem_chunks.init.error_rates has 1 entries but num_faults=2"):
        qec.multi_decoder_config.from_yaml_str(two.split("  - id: 5")[0])


def test_chunk_form_checks_run_on_writing_too():
    config = qec.decoder_config.from_yaml_str(chunk_form_yaml(5))
    config.dem_chunks.num_rounds = 1
    with pytest.raises(RuntimeError, match=r"dem_chunks.num_rounds must be at least 2 for decoder 0; got 1\."):
        config.to_yaml_str()
    config = qec.decoder_config.from_yaml_str(chunk_form_yaml(5))
    config.block_size = 3
    with pytest.raises(RuntimeError, match="block_size must not be set for decoder 0 because it is derived from dem_chunks"):
        config.to_yaml_str()
    config = qec.decoder_config.from_yaml_str(chunk_form_yaml(5))
    config.stim_dem_path = "x.dem"
    with pytest.raises(RuntimeError, match="dem_chunks and stim_dem_path must not both be set for decoder 0"):
        config.to_yaml_str()


def test_truncated_documents_raise_only_the_reference_errors():
    # Every prefix of a valid document either reads or raises the YAML error the reference raises, never an IndexError or KeyError.
    text = chunk_form_yaml(5)
    outcomes = set()
    for n in range(len(text) + 1):
        try:
            qec.decoder_config.from_yaml_str(text[:n])
            outcomes.add("ok")
        except RuntimeError as e:
            outcomes.add(str(e).split(":")[0])
    assert "ok" in outcomes and any(o.startswith("Invalid") for o in outcomes)


def test_documents_with_a_line_dropped_or_garbled_raise_only_the_reference_errors():
    lines = chunk_form_yaml(5).split("\n")
    for i in range(len(lines)):
        for variant in ("drop", "scalar", "indent", "tail"):
            edited = list(lines)
            if variant == "drop":
                del edited[i]
            elif variant == "scalar":
                edited[i] = edited[i] + " zz"
            elif variant == "indent":
                edited[i] = "  " + edited[i]
            else:
                edited[i] = edited[i].split(":")[0] + ":"
            try:
                qec.decoder_config.from_yaml_str("\n".join(edited))
            except RuntimeError:
                pass


def test_random_mutations_raise_only_the_reference_errors():
    # (A few hundred seeded edits of a valid document: cut, spliced with YAML punctuation and keys, truncated. What reads must
    # write; what does not raises RuntimeError, the one error type every configuration error out of from_yaml_str has.)
    import random

    rng = random.Random(1234)
    alphabet = list("[]{}:,-#'\"\n \t0123456789abcxyz.<>&*!|%@`") + ["from", "to", "name", "spec", "phases", "seam", "null", "~", "-1", "1e400", "0x1f", "nan", ".inf"]
    read = 0
    for base in (chunk_form_yaml(5), dem_chunks_yaml(4)):
        for _ in range(300):
            text = base
            for _ in range(rng.randint(1, 4)):
                op, pos = rng.random(), rng.randrange(len(text) + 1)
                if op < 0.4:
                    text = text[:pos] + text[min(len(text), pos + rng.randint(1, 6)) :]
                elif op < 0.8:
                    text = text[:pos] + rng.choice(alphabet) + text[pos:]
                else:
                    text = text[: rng.randrange(len(text) + 1)]
            for parse in (qec.decoder_config.from_yaml_str, qec.multi_decoder_config.from_yaml_str):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        out = parse(text)
                except RuntimeError:
                    continue
                read += 1
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        out.to_yaml_str()
                except RuntimeError:
                    pass
    assert read > 0


def test_garbage_that_is_not_a_chunk_spec():
    for text in (
        "id: 0\ntype: single_error_lut\ndem_chunks: [ 1, 2 ]\n",
        "id: 0\ntype: single_error_lut\ndem_chunks: { seam: 3, connections: [], phases: [] }\n",
        "id: 0\ntype: single_error_lut\ndem_chunks: { seam: { from: a, to: b }, connections: [ [ ] ], phases: [ [ ] ] }\n",
        "id: 0\ntype: single_error_lut\ndem_chunks: { seam: { from: a, to: b }, connections: [], phases: [ { name: x, spec: { num_faults: 1, error_rates: [ 0.1 ], seam_specs: [ 1 ] } } ] }\n",
        "id: 0\ntype: single_error_lut\ndem_chunks: { seam: { from: a, to: b }, connections: [], phases: [ { name: x, spec: { num_faults: 1, error_rates: [ 0.1 ], seam_specs: [ { name: s } ] } } ] }\n",
    ):
        with pytest.raises(RuntimeError, match=re.escape(YAML_HEAD)):
            qec.decoder_config.from_yaml_str(text)
    # (An empty connection list, parsed fine, is the spec check's: connections must not be empty.)
    with pytest.raises(RuntimeError, match="Invalid dem_chunks for decoder 0: dem_chunks_spec: connections must not be empty"):
        qec.decoder_config.from_yaml_str("id: 0\ntype: single_error_lut\ndem_chunks: { seam: { from: a, to: b }, connections: [], phases: [ { name: x, spec: { num_faults: 1, error_rates: [ 0.1 ], H_sparse: [ 0, -1 ] } } ] }\n")
    with pytest.raises(RuntimeError, match="Invalid dem_chunks for decoder 0: dem_chunks_spec: phases must not be empty"):
        qec.decoder_config.from_yaml_str("id: 0\ntype: single_error_lut\ndem_chunks: { seam: { from: a, to: b }, connections: [ { from: a, to: b } ], phases: [] }\n")
    with pytest.raises(RuntimeError, match="Invalid dem_chunks for decoder 0: dem_chunks_spec: no phases or connections"):
        qec.decoder_config.from_yaml_str("id: 0\ntype: single_error_lut\ndem_chunks: { seam: { from: a, to: b }, connections: [], phases: [] }\n")
