"""The paper, the crate's front page, CONTRIBUTING and the parity inventory say what the code does.

Each check reads the source fact a sentence rests on (which matcher a class decodes on, what a
Cargo feature switches, whether a policy the README names is defined) and then the sentence, so
a change to either side that makes them disagree fails here.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def text(path: str) -> str:
    """A file's text with its line breaks and runs of spaces folded to single spaces."""
    return re.sub(r"\s+", " ", (ROOT / path).read_text(encoding="utf-8"))


def test_paper_separates_independent_stages_from_ports():
    # 2.2's PyMatching interface decodes on PyMatching's own sparse blossom, transliterated, and
    # src/clifford/ holds Stim's algorithms ported: a comparison of those with their sources is
    # a port against its source, not two independent implementations.
    assert "line for line" in text("src/pm_exact/mod.rs")
    assert "ported from Stim 1.16.0" in text("THIRD_PARTY_NOTICES.md")
    paper = text("paper/paper.md")
    for overclaim in ("implements each stage independently", "Every stage has a test that compares it with an independent implementation", "compared with an independent reference implementation"):
        assert overclaim not in paper, overclaim
    ports = re.search(r"Others are ports((?:(?!\. ).)*)\. ", paper)
    assert ports, "the paper names no ported stages"
    for source in ("PyMatching's sparse blossom", "Stim's", "`ldpc`'s", "Relay-BP", "Tesseract", "Chromobius"):
        assert source in ports.group(1), source
    assert "A port's agreement with its source shows" in paper


def test_paper_and_crate_page_do_not_promise_identical_decoding_across_targets():
    # Python's Matching decodes on src/pm_exact/; the crate's Matching and the browser's decoders
    # are DemDecoder on src/sparse/, which can choose another equal-weight matching.
    assert "src/pm_exact/" in text("python/stabilizer_qec/_matching.py")
    assert "DemDecoder::new" in text("src/api/decoders.rs")
    assert "Option<DemDecoder>" in text("src/wasm_hw.rs")
    paper = text("paper/paper.md")
    assert "Sampling, model building and decoding run in the same process, so the browser, Python and Rust users all get the same results" not in paper
    assert re.search(r"equal-weight matchings: from 2\.2, Python's `Matching` makes PyMatching's choice", paper)
    crate = text("docs/crate.md")
    assert re.search(r"Python package's `Matching`[^.]*PyMatching's[^.]*equal-weight", crate)


def test_crate_page_says_what_the_highs_feature_does():
    # The feature switches MWPF's linear programs to HiGHS; the crate has no ILP distance (the
    # integer-programming distance is the Python package's, on SciPy's milp).
    cargo = text("Cargo.toml")
    assert re.search(r"# MWPF's linear programs by HiGHS[^\[]*highs = \[", cargo)
    for path in sorted((ROOT / "src").rglob("*.rs")):
        assert "milp" not in path.read_text(encoding="utf-8"), path
    assert "milp" in text("python/stabilizer_qec/_circuit.py")
    crate = text("docs/crate.md")
    assert "ILP circuit distance" not in crate
    assert re.search(r"the `highs` feature solves MWPF's linear programs with HiGHS", crate)


def test_the_compatibility_policy_the_docs_name_is_defined():
    # README and the package docstring send readers to "the compatibility policy"; CONTRIBUTING
    # defines it, and the crate's Stability section states the same terms.
    named = [p for p in ("README.md", "python/stabilizer_qec/__init__.py", "docs/crate.md") if "compatibility policy" in text(p)]
    assert named
    contributing = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    assert "\n## Compatibility policy\n" in contributing
    policy = re.sub(r"\s+", " ", contributing.split("\n## Compatibility policy\n", 1)[1].split("\n## ", 1)[0])
    for term in ("deprecated", "DeprecationWarning", "#[deprecated]", "upstream", "Changed", "seed", "cargo semver-checks"):
        assert term in policy, term
    crate = text("docs/crate.md")
    assert "keep compiling, with the same meaning, through every 2.x release" not in crate
    assert "CONTRIBUTING.md#compatibility-policy" in crate


def test_parity_inventory_f1_states_tqecs_hadamard_rule_and_no_unmeasured_speedup():
    inventory = (ROOT / "docs/superpowers/notes/2026-10-09-parity-inventory.md").read_text(encoding="utf-8")
    f1 = next(line for line in inventory.splitlines() if line.startswith("| F1 |"))
    assert "6 to 16 times" not in f1
    assert "X-directed spatial Hadamards require fixed_boundary" not in f1
    assert "Spatial Hadamard pipes between two non-spatial cubes, along x or y, require fixed_boundary" in f1
