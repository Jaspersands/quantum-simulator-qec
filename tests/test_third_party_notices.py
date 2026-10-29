"""The licence notices the package ships: the Rust crates each wheel's compiled module links
(THIRD_PARTY_CRATES.md, written by tools/third_party_crates.py), the C++ libraries compiled in and
the ported code (THIRD_PARTY_NOTICES.md, estimate/LICENSE-qdk), and the licence metadata
(pyproject.toml, CITATION.cff)."""

from __future__ import annotations

import importlib.util
import os
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


def tool():
    spec = importlib.util.spec_from_file_location("third_party_crates", str(TOOL))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_tool(*args, env=None):
    return subprocess.run([sys.executable, str(TOOL)] + list(args), cwd=ROOT, capture_output=True, text=True, env=env)


needs_cargo = pytest.mark.skipif(shutil.which("cargo") is None, reason="needs cargo")


@needs_cargo
def test_crate_notices_are_what_cargo_tree_gives():
    # The tool needs every crate of every wheel target with the wheels' features; a registry that
    # holds only what a build for this host fetched cannot give them offline, and the tool says so
    # by its exit status rather than failing as an out-of-date file would.
    r = run_tool("--check")
    if r.returncode == tool().OFFLINE_UNAVAILABLE:
        pytest.skip("cargo's registry lacks crates the wheels' targets need (tools/third_party_crates.py --check --online fetches them)")
    assert r.returncode == 0, r.stdout + r.stderr


@needs_cargo
def test_crate_tool_says_when_cargo_cannot_answer_offline(tmp_path):
    # A CARGO_HOME without the crates (here an empty one) is the tool's distinct exit status, on
    # which the test above skips, not the status of a failure.
    r = run_tool("--check", env=dict(os.environ, CARGO_HOME=str(tmp_path)))
    assert r.returncode == tool().OFFLINE_UNAVAILABLE, r.stdout + r.stderr
    assert "--online" in r.stderr


def test_crate_tool_does_not_take_a_stale_lock_for_a_missing_crate():
    # cargo's words, offline: a crate's package it has not got, a package its index does not list,
    # and a Cargo.lock that --locked forbids updating (which must fail the test, not skip it).
    t = tool()
    assert t.offline_stop("error: failed to download `allocator-api2 v0.2.21`\n\nCaused by:\n  attempting to make an HTTP request, but --offline was specified\n")
    assert t.offline_stop("error: no matching package named `pyo3` found\nlocation searched: crates.io index\nrequired by package `stabilizer_qec v2.2.0 (/repo)`\n"
                          "As a reminder, you're using offline mode (--offline) which can sometimes cause surprising resolution failures, if this error is too confusing you may wish to retry without `--offline`.\n")
    assert not t.offline_stop("error: the lock file /repo/Cargo.lock needs to be updated but --locked was passed to prevent this\n"
                              "If you want to try to generate the lock file without accessing the network, remove the --locked flag and use --offline instead.\n")


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


def test_qldpc_extra_has_qldpc_numpy_floor():
    assert "numpy>=1.24; python_version >= '3.11'" in toml_list(pyproject(), "qldpc")


def test_qecsim_notice_says_where_config_is_read():
    qecsim = words(section("qecsim"))
    assert "$QECSIM_CFG" in qecsim and "~/.qecsim" in qecsim
    assert "logging_qecsim.ini" in qecsim and "libpypm" in qecsim
    assert "not the working directory" in qecsim
    # An empty QECSIM_CFG counts as unset (qecsim took it as the working directory), and the
    # Blossom V interface numbers its nodes in order of first appearance, not a set's order.
    port = ROOT / "python" / "stabilizer_qec" / "qecsim"
    assert re.search(r"(?m)^\s*if qecsim_cfg_path:", (port / "util.py").read_text(encoding="utf-8"))
    assert "an empty value counting as unset" in qecsim
    assert "order of first appearance" in words((port / "graphtools" / "blossom5.py").read_text(encoding="utf-8"))
    assert "numbers the nodes in order of first appearance" in qecsim


def test_mwpf_notice_names_the_deflate_it_adapts():
    # The HTML export's gzip is miniz_oxide 0.8.0's deflate written out in src/mwpf/shim.rs.
    shim = words((ROOT / "src" / "mwpf" / "shim.rs").read_text(encoding="utf-8"))
    assert "miniz_oxide` 0.8.0's deflate compressor" in shim and "2017 Frommi" in shim
    mwpf = words(section("MWPF"))
    assert "`miniz_oxide` 0.8.0's deflate compressor" in mwpf
    assert "Copyright (c) 2017 Frommi; MIT, Zlib or Apache-2.0" in mwpf


def test_onnxruntime_notice_counts_the_recordings():
    # The cases and models recorded from onnxruntime, as the recordings file holds them (its
    # meta__ entries are the versions, no case), and the fusion the evaluator emulates.
    import numpy as np

    with np.load(ROOT / "tests" / "cudaq_qec_port" / "models" / "ort_outputs.npz", allow_pickle=False) as z:
        keys = [k for k in z.files if not k.startswith("meta__")]
    cases = {k.rsplit("__", 1)[0] for k in keys}
    models = {k.split("__", 1)[0] for k in keys}
    ort = words(section("onnxruntime"))
    assert f"The {len(cases)} recordings on {len(models)} models in `tests/cudaq_qec_port/models/ort_outputs.npz`" in ort
    assert "MatMulAddFusion" in words((ROOT / "python" / "stabilizer_qec" / "cudaq_qec" / "_onnx.py").read_text(encoding="utf-8"))
    assert "the MatMul + Add fusion of its default optimizer" in ort and "once-rounded fused multiply-adds" in ort


def test_qsample_notice_says_save_needs_dill_for_lambdas():
    qsample = words(section("qsample"))
    assert "dill to save" in qsample
    assert "saving uses pickle, which cannot save lambda or nested check functions" in qsample
