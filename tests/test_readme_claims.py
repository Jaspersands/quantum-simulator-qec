"""The README, the package docstring and the technical report say what the code and the committed
data do.

Each check reads the fact a sentence rests on (a recorded run, what the fuzz harness drives, what a
reader does with a stray byte, what a program exits with) and then the sentence, so a change to
either side that makes them disagree fails here. tools/readme_tables.py --check holds the tables;
these hold the prose around them.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile

import pytest

import stabilizer_qec as sq
from tesseract_platform_helpers import stim_number

ROOT = pathlib.Path(__file__).resolve().parents[1]
README_TEXT = (ROOT / "README.md").read_text(encoding="utf-8")


def fold(text: str) -> str:
    """Line breaks and runs of spaces folded to single spaces (so a sentence may wrap)."""
    return re.sub(r"\s+", " ", text)


README = fold(README_TEXT)
REPORT_MD = fold((ROOT / "report" / "report.md").read_text(encoding="utf-8"))
PACKAGE_DOC = fold(sq.__doc__)


def bullet(start: str) -> str:
    """The README's top-level bullet that begins with ``start``, through its wrapped lines."""
    lines = README_TEXT.split("\n")
    i = next(i for i, line in enumerate(lines) if line.startswith(start))
    out = [lines[i]]
    for line in lines[i + 1:]:
        if line.startswith("- ") or line.startswith("#") or line == "":
            break
        out.append(line)
    return fold("\n".join(out))


def paragraph(start: str) -> str:
    """The README paragraph (up to a blank line) that begins with ``start``."""
    i = README_TEXT.index(start)
    return fold(README_TEXT[i:].split("\n\n", 1)[0])


def tables():
    spec = importlib.util.spec_from_file_location("readme_tables", ROOT / "tools" / "readme_tables.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


# B20.1: the promises name the compatibility policy CONTRIBUTING defines, and the README and the
# package docstring make the same ones (2.1's "any machine", "keeps working through every 2.x
# release" and "never a crash" are gone from both; an engine panic the core catches is RuntimeError).
def test_the_promises_name_the_policy_and_agree_with_the_package_docstring():
    from stabilizer_qec import _util

    class PanicException(BaseException):
        pass

    def panics():
        raise PanicException("boom")

    with pytest.raises(RuntimeError, match="please report it"):
        _util.call(panics)
    promises = bullet("- **Promises.**")
    assert "(CONTRIBUTING.md#compatibility-policy)" in promises
    assert "follows the compatibility policy with documented corrections/migrations" not in promises
    assert "under its documented compatibility policy" not in promises
    for text in (promises, PACKAGE_DOC):
        assert "Within a version, a seed and shot position give the same shots on any number of threads" in text, text
        assert "RuntimeError" in text and "a crash or a hang is a bug to report" in text, text
        assert "keeps its names and signatures through every 2.x release" in text, text
        assert "behaviour changes in a minor release only to correct it" in text, text
        assert "on any machine" not in text and "never a crash" not in text
    assert "CONTRIBUTING.md" in PACKAGE_DOC
    assert "follows the compatibility policy, with corrections" not in PACKAGE_DOC


# B20.2: the Decoders bullet quotes the latest recorded Apple-silicon run, as the Speed table does.
def test_the_decoders_bullet_quotes_the_latest_recorded_matching_ratios():
    runs = load("data/bench/runs.json")
    latest = [r for r in runs if r.get("kind") == "Darwin arm64"][-1]
    plain, correlated = (latest["results"][k]["ratio"] for k in ("matching", "correlated_matching"))
    linux = [r for r in runs if r.get("kind") == "Linux x86_64"][-1]
    text = bullet("- **Decoders**")
    assert f"({plain:.2f} in the recorded Apple-silicon run of {latest['version']})" in text, text
    assert f"({correlated:.2f};" in text
    assert all(round(linux["results"][k]["ratio"], 1) == 1.2 for k in ("matching", "correlated_matching"))
    assert "on Linux x86_64 both about 1.2" in text


# B20.3: the stabilizer-rank sampler is checked against the state vector's exact distribution and
# against Tsim in distribution (χ²), never bit for bit; and its injection and distillation
# measurements are 2.2.0's, so the bullet does not call them historical.
def test_the_clifford_t_bullet_says_how_the_sampler_is_checked():
    text = bullet("- **Clifford+T circuits, with coherent noise on every gate**")
    assert "bit for bit against the state vector and against Tsim" not in text
    assert "the state vector's distribution (to 10⁻⁹)" in text and "Tsim's in distribution (χ²)" in text
    for path in ("data/clifford_t/inject.json", "data/clifford_t/distill.json"):
        assert load(path)["version"].startswith("2.2"), path
    assert "predate the merged-proposal correction" not in text
    assert "re-recorded with the 2.2.0 wheel" in text


# B20.4: the randomized no-panic run and cargo fuzz drive src/fuzzing.rs, whose decoders are the
# src/sparse matcher (DemDecoder), belief-matching, BP+OSD and the window decoders; Matching's
# transliterated matcher (pm_exact) and the other decoders are not in it, and the README says so.
def test_the_ci_section_names_the_decoders_the_fuzz_harness_drives():
    harness = (ROOT / "src" / "fuzzing.rs").read_text(encoding="utf-8")
    imported = set(re.findall(r"^use crate::(\w+)", harness, re.M))
    assert {"dem_decoder", "belief", "osd", "window"} <= imported
    for absent in ("pm_exact", "user_graph", "lsd", "relay", "color", "search", "clustering", "union_find"):
        assert absent not in imported and f"crate::{absent}" not in harness, absent
    ci = fold(README_TEXT.split("## Continuous integration", 1)[1].split("\n## ", 1)[0])
    assert "every parser, model builder, sampler and decoder" not in ci
    assert "everything behind them" not in ci
    for named in ("`src/sparse/`", "belief-matching", "BP+OSD", "window decoders", "`Matching`'s transliterated matcher"):
        assert named in ci, named
    assert "are not in this run" in ci


TQEC_SPREADS = ("1.5 to 1.6 times on the Steane encoding", "2.8 to 3.1 on the CNOT", "5.9 to 6.6 on the memory", "1.9 to 5.0 on the CNOT", "5.0 to 8.8 on the memory")


# B20.5 and B20.6: tqec's fixed_bulk has no template for a spatial Hadamard pipe between two
# non-spatial cubes, along x or y; one at a spatial junction compiles in both. The speed is the
# measured spread of the timed call, as the CHANGELOG gives it, not a 6-16x ratio.
def test_the_tqec_paragraph_states_the_hadamard_rule_and_the_measured_speed():
    # tests/tqec_port/test_round_controls.py compiles each case the rule names, in the convention it names.
    rounds = (ROOT / "tests" / "tqec_port" / "test_round_controls.py").read_text(encoding="utf-8")
    assert '("y_regular",' in rounds and '("cz_junction",' in rounds
    changelog = fold((ROOT / "CHANGELOG.md").read_text(encoding="utf-8"))
    text = bullet("- **Fault-tolerant compilation, two ways**")
    assert "6–16" not in text and "6 to 16" not in text
    assert "Spatial Hadamards along x" not in text
    assert "spatial Hadamard pipe between two non-spatial cubes (along x or y) requires `fixed_boundary`" in text
    assert "`compile_block_graph(graph, convention=...).generate_stim_circuit(1)`" in text
    for spread in TQEC_SPREADS:
        assert spread in text and spread in changelog, spread
    assert "grows with load" not in text
    assert "Spatial Hadamards along x" not in REPORT_MD
    assert "A spatial Hadamard pipe between two non-spatial cubes, along x or y, requires fixed_boundary" in REPORT_MD


def test_direct_round_controls_are_described_on_every_graph_tqec_compiles():
    # B15.2: the direct controls are checked against tqec's 2k + 1 layout on the graph's time slices.
    assert "Direct rounds/merged controls receive the same layout refusals as the distance wrapper." not in README
    assert "Direct rounds/merged controls are checked against tqec's 2k + 1 layout on every graph tqec compiles" in README


# B20.7: the deliberate differences from Stim include the tagged identities the samplers run as
# gates (Stim reads them as identities), and the FlipSimulator row does not claim more than holds.
def test_the_stim_differences_list_the_tagged_identities():
    import numpy as np

    c = sq.Circuit("H 0\nI[T] 0\nI[T] 0\nI[T] 0\nI[T] 0\nH 0\nM 0")
    assert c.compile_sampler(seed=0).sample(8).all()  # four T are a Z: Stim samples all zeros
    assert not c.without_tags().compile_sampler(seed=0).sample(8).any()
    assert str(sq.Circuit("I[T] 0").inverse()) == "I[T_DAG] 0"
    f = sq.FlipSimulator(batch_size=3, num_qubits=2, disable_stabilizer_randomization=True)
    f.broadcast_pauli_errors(pauli="Y", mask=np.array([[True, False, True], [False, False, True]]), p=0.0)
    assert all(str(p) == "+__" for p in f.peek_pauli_flips())  # Stim applies the masked flips at p = 0
    text = paragraph("Where Stim's output depends on its random number generator")
    assert "Three differences are deliberate" not in text
    assert "Four differences are deliberate" in text
    for named in ("`I[T]`", "`without_tags()`", "`inverse()`", "`FlipSimulator.broadcast_pauli_errors`"):
        assert named in text, named
    row = next(line for line in README_TEXT.split("\n") if line.startswith("| `FlipSimulator`"))
    assert "`broadcast_pauli_errors` at `p=0`" in row


def run_module(module: str, *args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    return subprocess.run([sys.executable, "-m", module, *args], capture_output=True, text=True, env=env, timeout=120)


# B20.8: Chromobius's program aborts, as Chromobius's does, on the model faults Stim's parser throws
# std::out_of_range for; every other unparsable model is this package's message and exit 1.
def test_the_command_line_paragraph_says_where_chromobius_aborts():
    with tempfile.TemporaryDirectory() as tmp:
        bad = pathlib.Path(tmp) / "bad.dem"
        bad.write_text("foo D0\n")
        r = run_module("stabilizer_qec.chromobius", "describe_decoder", "--in", str(bad))
    assert r.returncode == 134 and "std::out_of_range: Unrecognized instruction name: foo" in r.stderr
    text = paragraph("The other tools' command lines are here too")
    assert "an unrecognized instruction name, a number of 2^60 or more, a `{` never closed or a `}` never opened" in text
    assert "status 134" in text


# B20.9: which readers replace a byte that is not UTF-8, and which still refuse it.
def test_the_utf8_sentence_names_the_readers_that_replace_and_the_one_that_refuses():
    from stabilizer_qec import chromobius

    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        dem = tmp / "m.dem"
        dem.write_bytes(b"# caf\xe9\nerror(0.1) D0 D1 L0\nerror(0.1) D0\nerror(0.1) D1\ndetector(0, 0, 0, 0) D0\ndetector(1, 0, 0, 1) D1\n")
        # (The model as Stim writes it on this machine: 0.1 has 19 digits on x86-64 Linux.)
        assert str(sq.DetectorErrorModel.from_file(dem)).startswith(f"error({stim_number(0.1)}) D0 D1 L0")
        sq.Matching.from_detector_error_model_file(str(dem))
        dets = tmp / "d.b8"
        dets.write_bytes(bytes([0b11]))
        with pytest.raises(UnicodeDecodeError):
            chromobius.sinter_decoders()["chromobius"].decode_via_files(num_shots=1, num_dets=2, num_obs=1, dem_path=dem, dets_b8_in_path=dets, obs_predictions_b8_out_path=tmp / "o.b8", tmp_dir=tmp)
    assert "Invalid UTF-8 model bytes use replacement, not promised native byte preservation." not in README
    text = paragraph("Animation's 1,080 frames/11 cases")
    for named in ("`Circuit.from_file`", "`DetectorErrorModel.from_file`", "PyMatching's file loaders", "the `stabilizer-qec` commands", "Tesseract's and Chromobius's programs", "U+FFFD"):
        assert named in text, named
    assert "Chromobius's sinter decoder" in text and "`decode_via_files`" in text


# B20.10: the m2d paragraph describes the converter as it is (src/m2d.rs's module comment).
def test_the_m2d_paragraph_describes_the_converter_as_it_is():
    source = (ROOT / "src" / "m2d.rs").read_text(encoding="utf-8")
    assert "dem_build::sweep_effects" in source and "fn sweep_frames" in source
    text = bullet("- **From raw measurements to detection events** (`src/m2d.rs`)")
    for stale in ("Seven more references", "one more run per bit", "refused as nondeterministic"):
        assert stale not in text, stale
    for named in ("every random outcome taken as 0", "backward walk", "Pauli frames", "`tests/test_m2d.py`"):
        assert named in text, named
    assert "src/m2d.rs raw measurements and sweep bits to detection events: a noiseless reference run and the sweep bits' Pauli frames" in README


# B20.11: the dense blossom holds 512 vertices, two per defect; past 256 defects decode_mwpm matches
# with pm_exact's sparse blossom, also exact.
def test_the_exact_matching_paragraph_says_how_it_stays_exact_past_256_defects():
    assert "pub const MAX_VERTICES: usize = 512;" in (ROOT / "src" / "blossom.rs").read_text(encoding="utf-8")
    assert "if 2 * unmatched.len() > crate::blossom::MAX_VERTICES" in (ROOT / "src" / "decoder.rs").read_text(encoding="utf-8")
    text = paragraph("**It is now exact at every defect count.**")
    assert "512 vertices, two per defect" in text and "past 256 defects" in text and "`src/pm_exact/`" in text


# B20.12: pm_exact's radius offsets are 64-bit, where PyMatching 2.4.0's int32_t never finishes.
def test_the_pm_exact_paragraph_lists_the_radius_divergence():
    text = paragraph("**PyMatching's own, transliterated (2.2).**")
    assert "64-bit" in text and "int32_t" in text and "131" in text and "`tests/test_matching_radius.py`" in text


# B20.13: the tables match their data, and the prose quoting them quotes the re-recorded numbers.
def test_the_readme_tables_and_the_prose_around_them_match_the_data():
    module = tables()
    assert module.stale_data() == []
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for name, header, fn in module.TABLES:
        body = "\n".join(fn()) + "\n"
        m = re.search("(" + header + r"\n\|---(?:\|---)*\|\n)((?:\|.*\|\n)+)", readme)
        assert m and m.group(2) == body, name
    points = load("data/surgery/results.json")["points"]
    t2 = [round(100 * points[f"d{d}/T2/p0.003/correlated"]["rate_outcome"]) for d in (3, 5, 7)]
    assert f"T = 2 fails {t2[0]}%, {t2[1]}% and {t2[2]}% of the time at d = 3, 5 and 7" in README
    memories = {m["memory"]: m for m in load("data/distances.json")["memories"]}
    colour = [v for k, v in memories.items() if k.startswith("colour code XYZ")]
    assert all(m.get("search") == m.get("exact") for m in colour)
    assert "Stim's search finds the same at all three" in README and "agrees where it reaches" not in README
    assert "Ours are 6, 9, 9 and 12" in README
    unencoded = load("data/clifford_t/distill.json")["unencoded"]["0.01"]
    for way in ("twirl", "coherent", "exact"):
        assert f"{unencoded[way]['acceptance']:.4f} /" in README or f"{unencoded[way]['acceptance']:.4f}, while" in README, way
    assert f"{unencoded['twirl']['infidelity']:.3g} /" in README
    # The per-shot column's joint-sampler times are an earlier, quieter recording's; the README says so.
    note = load("data/clifford_t/distill.json")["note"]
    assert "the 2026-10-23 recording's" in note
    assert "from the 2026-10-23 recording: the 2.2.0 re-recording drew the same joint-sampler samples" in README


def test_the_report_quotes_the_regenerated_tables():
    html = (ROOT / "report" / "report.html").read_text(encoding="utf-8")
    points = load("data/surgery/results.json")["points"]
    for d in (3, 5, 7):
        assert f"{100 * points[f'd{d}/T2/p0.003/correlated']['rate_outcome']:.2f}%" in html, d


# The fix batches' prose notes for README.md and the package docstring, each checked against the
# code or file it describes.
NOTES = [
    # B1 (round 2): the twirl keeps the gates; a non-Clifford gate's angle is never added into an error.
    ("a tagged gate at a Clifford angle is the Clifford gate there, and a non-Clifford gate is never added into an error", "gives the twirl, or the coherence-aware model with equivalent rotations added up before twirling, as a decoder's prior."),
    # B3.8: the integer program leaves out faults of probability 0.
    ("an integer program over every fault that can happen (probability above 0)", "`DetectorErrorModel.distance()`: an integer program over every fault, which"),
    # B4.3: detector indices as Stim writes them; decoders take up to 2^24 - 1.
    ("any number of observables, detector indices as Stim writes them, up to 2^60 - 1 (2^62 - 1 after shifts; decoders take up to 2^24 - 1)", None),
    # B6.1: the command line streams shots.
    ("shots streamed 1,024 at a time, so memory stays flat in `--shots`", None),
    # B9.3: the visualizer's index.js imports a patch only from its own origin.
    ("imports a `patch_url` only from its own origin", None),
    # B10: the ONNX evaluator's recorded corpus and the MatMul + Add fusion.
    ("bit for bit on 180 recorded cases of 23 models", "157 recorded cases of 17 models"),
    # B17: the live-references job.
    ("the comparisons with live Tsim, sinter 1.16.0 and relay-bp 0.2.2", None),
    # B18: the crates' notices.
    ("[THIRD_PARTY_CRATES.md](THIRD_PARTY_CRATES.md)", None),
    ("THIRD_PARTY_CRATES.md the Rust crates each wheel's module links", None),
    # B22: the joint sampler truncates without a warning.
    ("It does so without a warning: `max_contraction` has no effect on it, and only `exact_weights=False` reports the truncation", None),
    # B23.3: the highs feature is MWPF's LP solver.
    ("src/highs/ HiGHS 1.8.0, MWPF's LP solver (feature highs)", "HiGHS behind the ILP distance"),
]


@pytest.mark.parametrize("present, absent", NOTES)
def test_the_fix_batches_prose_notes_are_applied(present, absent):
    assert present in README, present
    if absent:
        assert absent not in README, absent


def test_the_noted_facts_hold_in_the_code():
    # The facts the notes above rest on, so a later change to the code fails here too.
    assert (ROOT / "THIRD_PARTY_CRATES.md").exists()
    assert "_BATCH = 1024" in (ROOT / "python" / "stabilizer_qec" / "_cli.py").read_text(encoding="utf-8")
    assert "HIGHS_MINOR=8" in (ROOT / "src" / "highs" / "Version.txt").read_text(encoding="utf-8")
    assert "refused patch from another origin" in (ROOT / "python" / "stabilizer_qec" / "fusion_blossom" / "visualizer" / "index.js").read_text(encoding="utf-8")
    assert "\n  live-references:\n" in (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    model = sq.DetectorErrorModel("error(0.1) D1099511627776 L0")
    assert model.num_detectors == 2**40 + 1
    k = 2**60 - 1
    assert str(sq.DetectorErrorModel(f"shift_detectors {k}\n" * 3 + f"error(0.1) D{k}").flattened()) == f"error({stim_number(0.1)}) D{4 * k}"
    with pytest.raises(ValueError, match="4611686018427387903"):
        sq.DetectorErrorModel(f"shift_detectors {k}\n" * 3 + f"shift_detectors 4\nerror(0.1) D{k}").flattened()
    with pytest.raises(ValueError, match="16777215"):
        sq.Matching.from_detector_error_model(model)
    limits = re.search(r"- \*\*Limits\.\*\*(.*?)- \*\*Stability", PACKAGE_DOC).group(1)
    assert "detector indices as Stim does, up to `2**60 - 1` as written and `2**62 - 1` after shifts" in limits and "`2**24 - 1`" in limits
