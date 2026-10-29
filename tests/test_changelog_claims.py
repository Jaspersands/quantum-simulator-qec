"""The release notes say what the code and the committed data do.

CHANGELOG.md becomes the permanent record of a release the moment its tag is pushed, so its
checkable sentences are read here against their source: the rows of data/coherent/ and
data/decoders/check.json, the CI workflow, the compatibility policy CONTRIBUTING defines, and the
behaviour of the calls a sentence describes. A change to either side that makes them disagree
fails here.
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import numpy as np

import stabilizer_qec as sq

ROOT = Path(__file__).resolve().parents[1]
CHANGELOG = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")


def fold(text: str) -> str:
    """Line breaks and runs of spaces folded to single spaces (so a sentence may wrap)."""
    return re.sub(r"\s+", " ", text)


def release(version: str) -> str:
    """One release's section, from its heading to the next release's."""
    return CHANGELOG.split(f"\n## {version} ", 1)[1].split("\n## ", 1)[0]


def bullet(text: str, start: str) -> str:
    """The top-level bullet that begins with ``start``, through its wrapped and nested lines."""
    lines = text.split("\n")
    i = next(i for i, line in enumerate(lines) if line.startswith(start))
    out = [lines[i]]
    for line in lines[i + 1:]:
        if line.startswith("- ") or line.startswith("#") or line == "":
            break
        out.append(line)
    return fold("\n".join(out))


def test_the_preamble_names_the_compatibility_policy_contributing_defines():
    contributing = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    assert "\n## Compatibility policy\n" in contributing
    preamble = fold(CHANGELOG.split("\n## ", 1)[0])
    assert "(CONTRIBUTING.md#compatibility-policy)" in preamble
    promise = bullet(release("2.2.0").split("### Changed", 1)[1], "- **The compatibility promise.**")
    # 2.1's three promises, each as 2.1.0's README and package docstring worded it, and what 2.2 says instead.
    for old in ("keeps working through every 2.x release", "on any machine and any number of threads", "never a crash"):
        assert old in promise, old
    for named in ("block_temporal_height", "generate_stim_circuit", "rounds", "CONTRIBUTING.md#compatibility-policy", "bugs to report"):
        assert named in promise, named


def coherent_archive() -> dict:
    """Per data file, the (version, count) of the rows the current tables refuse."""
    spec = importlib.util.spec_from_file_location("readme_tables", ROOT / "tools" / "readme_tables.py")
    tables = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tables)
    revision = tables.COHERENT_GENERATION_REVISION
    out = {}
    for name in ("sweep", "crosstalk", "validate", "capacity"):
        data = json.loads((ROOT / "data" / "coherent" / f"{name}.json").read_text(encoding="utf-8"))
        old = [row.get("version", data.get("version")) for row in data["rows"] if row.get("generation_revision") != revision]
        current = [row for row in data["rows"] if row.get("generation_revision") == revision]
        assert all(row.get("version", data.get("version")) == "2.2.0" for row in current), name
        if old:
            out[name] = (set(old), len(old))
    return out


def test_measurement_provenance_describes_the_committed_coherent_rows():
    # B21.1: the d = 3 grids, validation and capacity were remeasured with the 2.2.0 wheel; only
    # 1.8.0's larger-distance sweep and crosstalk rows are archival, and no development-2.2 row ships.
    archive = coherent_archive()
    assert set(archive) == {"sweep", "crosstalk"}
    text = bullet(release("2.2.0"), "- **Measurement provenance.**")
    for stale in ("must refresh", "will refresh", "stamped 2.2.0", "until remeasured", "development-2.2 rows measured before"):
        assert stale not in text, stale
    assert "remeasured with the 2.2.0 wheel" in text
    for name, (versions, n) in archive.items():
        assert f"{n} larger-distance {name}" in text or f"{n} {name}" in text, (name, n)
        for version in versions:
            assert version in text
    assert "Git history" in text


def test_the_colour_matching_time_is_check_jsons():
    # B21.3: the ratio is sum(ours_s) / sum(chromobius_s) over the colour rows, as the README's
    # decoder table (tools/readme_tables.py) computes it from the same file.
    rows = json.loads((ROOT / "data" / "decoders" / "check.json").read_text(encoding="utf-8"))["sections"]["color"]["rows"]
    ratio = sum(r["ours_s"] for r in rows) / sum(r["chromobius_s"] for r in rows)
    shots = sum(r["shots"] for r in rows)
    entry = bullet(release("2.2.0"), "- **Chromobius on PyMatching 2.2.2's matcher")
    found = re.search(r"all ([\d,]+) shots of `tools/decoder_check.py`'s colour cases \(0 ties, at ([\d.]+) times Chromobius's time\)", entry)
    assert found, entry
    assert int(found.group(1).replace(",", "")) == shots
    assert found.group(2) == f"{ratio:.2f}"


def test_the_tqec_speed_is_the_measured_spread_of_the_timed_call():
    # B21.2: compile_block_graph(...).generate_stim_circuit(1), three runs a side in separate
    # processes: 1.5-1.6x (Steane), 2.8-3.1x (CNOT, tqec 9-10 s), 5.9-6.6x (memory); not 6 to 16
    # times, and no 53 s CNOT. The fixtures' recorded seconds include graph building and extra
    # regeneration, so they are not a timing of the call.
    entry = bullet(release("2.2.0"), "- **tqec's compile pipeline, byte for byte.**")
    assert "6 to 16 times" not in entry and "53 s" not in entry
    assert "generate_stim_circuit(1)" in entry
    for spread in ("1.5 to 1.6 times on the Steane encoding", "2.8 to 3.1 on the CNOT", "5.9 to 6.6 on the memory"):
        assert spread in entry, spread
    assert "grows with load" not in entry


def test_the_pm_exact_entry_lists_every_deliberate_divergence():
    # B21.4: PyMatching 2.4.0's compute_wrapped_radius returns int32_t, so a uniform chain of 131
    # or more nodes at a non-integer weight never finishes there; the port's offsets are i64.
    entry = bullet(release("2.2.0"), "- **PyMatching's sparse blossom, transliterated.**")
    assert "One further divergence" not in entry
    assert "Two further divergences are deliberate." in entry
    assert "radius" in entry and "131" in entry and "int32_t" in entry
    doc = (ROOT / "tests" / "test_matching_radius.py").read_text(encoding="utf-8").split('"""')[1]
    assert "PyMatching 2.4.0" in fold(doc) and "never finishes" in fold(doc)


def test_the_pm_exact_entry_says_uncounted_fault_ids_are_dropped():
    # B21.10: an id past the count, brought by a replace merge, is dropped, not reported as id mod 64.
    m = sq.Matching()
    m.add_edge(0, 1, fault_ids={0}, weight=-1.0)
    m.add_edge(0, 1, fault_ids={79}, weight=-2.0, merge_strategy="replace")
    m.ensure_num_fault_ids(64)
    prediction = m.decode(np.zeros(2, dtype=np.uint8))
    assert len(prediction) == 64 and not prediction.any()
    entry = bullet(release("2.2.0"), "- **PyMatching's sparse blossom, transliterated.**")
    assert "A fault id at or past the decoder's count" in entry and "is ignored" in entry


def test_spatial_hadamard_rule_is_tqecs():
    # B21.6: fixed_bulk lacks the template for a spatial Hadamard pipe between two non-spatial
    # cubes in either direction; one touching a spatial cube compiles in both conventions.
    entry = bullet(release("2.2.0"), "- **tqec's compile pipeline, byte for byte.**")
    assert "X-directed spatial Hadamards" not in entry
    assert "Hadamard pipe between two non-spatial cubes, along x or y, requires `fixed_boundary`" in entry


def test_count_mistakes_is_not_claimed_byte_for_byte():
    # B21.7: error messages carry this package's prefix and reader/parser messages, timings differ.
    entry = bullet(release("2.0.0"), "- **PyMatching's command line**")
    assert "output byte for byte" not in entry
    assert "error messages are this package's" in entry


def test_pecos_default_simulator_is_described_against_pypi_pecos():
    # B21.8: PECOS 0.2.0 from PyPI has no compiled simulator, so its SparseSim is the Python one.
    from stabilizer_qec.pecos import simulators

    assert simulators.SparseSim is not simulators.pySparseSim
    text = fold(release("2.0.0"))
    assert "(`cySparseSim`, the default `SparseSim`)" not in text
    assert "PECOS 0.2.0 installed from PyPI has no compiled C++, so its `SparseSim` is the Python one" in text


def test_panqec_test_count_says_how_the_slow_tests_run():
    # B21.9: the slow tests run only with PANQEC_RUNSLOW=1, which the live-references CI job sets.
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert 'PANQEC_RUNSLOW: "1"' in ci
    entry = fold(release("2.0.0"))
    assert "789 with its slow ones" not in entry
    assert "PANQEC_RUNSLOW=1" in entry


def test_models_past_64_observables_lists_the_leakage_samplers():
    # B21.11: the leakage samplers carry every observable, so they belong with what takes any number.
    c = sq.Circuit("R 0\nI_ERROR[LEAK(p=0.5)] 0\nM 0\nDETECTOR rec[-1]\n" + "".join(f"OBSERVABLE_INCLUDE({k}) rec[-1]\n" for k in range(70)))
    _, obs, _ = c.compile_leakage_sampler(seed=1).sample(4)
    assert obs.shape == (4, 70)
    entry = bullet(release("2.2.0").split("### Changed", 1)[1], "- **Models past 64 observables.**")
    assert "leakage samplers" in entry and "`compile_leakage_sampler`" in entry
