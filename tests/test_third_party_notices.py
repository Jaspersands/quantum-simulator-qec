"""The licence notices the package ships: the Rust crates each wheel's compiled module links
(THIRD_PARTY_CRATES.md, written by tools/third_party_crates.py), the C++ libraries compiled in and
the ported code (THIRD_PARTY_NOTICES.md, estimate/LICENSE-qdk), and the licence metadata
(pyproject.toml, CITATION.cff)."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
NOTICES = ROOT / "THIRD_PARTY_NOTICES.md"
CRATES = ROOT / "THIRD_PARTY_CRATES.md"
TOOL = ROOT / "tools" / "third_party_crates.py"


def words(text):
    """Text with Markdown's quote markers dropped and white space collapsed, to find a licence
    text quoted in a notice whatever its line breaks."""
    text = re.sub(r"(?m)^\s*>\s?", "", text)
    return " ".join(text.split())


def section(title):
    text = NOTICES.read_text(encoding="utf-8")
    m = re.search(r"(?ms)^## " + re.escape(title) + r"\n(.*?)(?=^## |\Z)", text)
    assert m, "THIRD_PARTY_NOTICES.md has no section " + title
    return m.group(1)


def pyproject():
    return (ROOT / "pyproject.toml").read_text(encoding="utf-8")


def toml_list(text, key):
    m = re.search(r"(?ms)^" + re.escape(key) + r"\s*=\s*\[(.*?)\]", text)
    assert m, key
    return re.findall(r'"([^"]*)"', m.group(1))


def crate_table():
    """{(name, version): row cells} of THIRD_PARTY_CRATES.md's table, and the unlinked crates."""
    text = CRATES.read_text(encoding="utf-8")
    rows = {}
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if line.startswith("| ") and len(cells) == 6 and cells[0] != "Crate":
            rows[(cells[0].replace(" (procedural macro)", ""), cells[1])] = cells
    tail = text.split("## Crates of Cargo.lock that no wheel links", 1)[1]
    unlinked = set(re.findall(r"(?m)^- (\S+) (\S+)$", tail))
    return text, rows, unlinked


def test_wheels_ship_the_crate_notices():
    assert "THIRD_PARTY_CRATES.md" in toml_list(pyproject(), "license-files")
    assert CRATES.is_file()


def test_crate_notices_cover_cargo_lock():
    # Every package of Cargo.lock but this one is either listed as linked (with its licence) or as
    # linked by no wheel: a Cargo.lock the file was not regenerated for fails here.
    lock = set(re.findall(r'\[\[package\]\]\nname = "([^"]+)"\nversion = "([^"]+)"', (ROOT / "Cargo.lock").read_text(encoding="utf-8")))
    lock.discard(("stabilizer_qec", re.search(r'(?m)^version = "([^"]+)"', (ROOT / "Cargo.toml").read_text(encoding="utf-8")).group(1)))
    _, rows, unlinked = crate_table()
    assert not set(rows) & unlinked
    assert set(rows) | unlinked == lock


def test_each_linked_crate_has_its_licence_text():
    text, rows, _ = crate_table()
    carried = set()
    for line in re.findall(r"(?m)^Carried by: (.*)\.$", text):
        for who in line.split("; "):
            name, version = who.split(" ")[:2]
            carried.add((name, version))
    for key, cells in rows.items():
        if key in carried:
            continue
        # Only a crate whose package carries no licence file and that may be taken under Apache-2.0
        # (whose text is LICENSES/Apache-2.0.txt) has no text of its own here.
        assert "taken under Apache-2.0" in cells[5] and "Apache-2.0" in cells[2], key
    # The crates whose licences need their notice in binary copies, by name.
    for name in ["libloading", "libm", "ordered-float", "objc2", "objc2-encode", "objc2-foundation", "block2", "foldhash",
                 "pyo3", "naga", "wgpu", "rand", "codespan-reporting", "spirv"]:
        assert any(k[0] == name for k in rows), name
    for name in ["libloading", "libm", "ordered-float", "objc2", "objc2-encode", "objc2-foundation", "block2", "foldhash"]:
        assert any(k[0] == name for k in carried), name
    # Per target: libloading on Linux and Windows, objc2 on macOS alone.
    assert rows[next(k for k in rows if k[0] == "objc2")][3] == "macOS"
    assert "Windows" in rows[next(k for k in rows if k[0] == "libloading")][3] or rows[next(k for k in rows if k[0] == "libloading")][3] == "all"


def _cargo_resolves_offline():
    if shutil.which("cargo") is None:
        return False
    r = subprocess.run(["cargo", "metadata", "--format-version", "1", "--locked", "--offline"], cwd=ROOT, capture_output=True)
    return r.returncode == 0


@pytest.mark.skipif(not _cargo_resolves_offline(), reason="needs cargo with every crate of Cargo.lock in its registry")
def test_crate_notices_are_what_cargo_tree_gives():
    r = subprocess.run([sys.executable, str(TOOL), "--check"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_highs_licence_texts_are_quoted_not_cited():
    # The wheels compile HiGHS in but carry no src/highs/: its licence texts are quoted.
    highs = words(section("HiGHS"))
    for path in ["src/highs/LICENSE.txt", "src/highs/extern/filereaderlp/LICENSE", "src/highs/extern/pdqsort/license.txt"]:
        assert words((ROOT / path).read_text(encoding="utf-8")) in highs, path
    assert "THIRD_PARTY_CRATES.md" in section("Rust crates")


def test_cudaq_qec_notice_attribution():
    # Apache-2.0 4(d): the attribution of CUDA-Q QEC's NOTICE file.
    cudaq = words(section("CUDA-Q QEC"))
    assert "This product includes software developed by NVIDIA corporation and affiliates" in cudaq
    assert "NOTICE" in cudaq


def test_fasteval_port_has_its_notice():
    azure = words(section("Azure Quantum Resource Estimator"))
    assert "restricted `ast` evaluator" not in azure
    assert "fasteval 0.2.4" in azure and "Christopher Sebastian" in azure
    qdk = words((ROOT / "python" / "stabilizer_qec" / "estimate" / "LICENSE-qdk").read_text(encoding="utf-8"))
    assert "fasteval" in qdk and "Copyright (c) 2019 Christopher Sebastian" in qdk
    assert "The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software." in qdk.split("fasteval", 1)[1]
    src = (ROOT / "python" / "stabilizer_qec" / "estimate" / "azure.py").read_text(encoding="utf-8")
    assert not re.search(r"(?m)^\s*(import ast|from ast )", src)
    assert "fasteval 0.2.4" in src


def test_citation_licence_is_not_a_choice():
    # CFF 1.2.0 reads a list of licences as OR; the package is MIT AND Apache-2.0.
    cff = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    m = re.search(r"(?m)^license:(.*)$", cff)
    assert m and m.group(1).strip() == "MIT"
    after = cff[m.end():]
    assert not re.match(r"\n\s+-", after)
    assert "MIT AND Apache-2.0" in " ".join(cff.split())
    assert re.search(r'(?m)^license = "MIT AND Apache-2.0"$', pyproject())


