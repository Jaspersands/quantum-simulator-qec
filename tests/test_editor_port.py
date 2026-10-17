"""The site editor's fixtures (`tests/editor_port/crumble.json`, `js/editor-examples.js`) are what
`tools/editor_fixtures.py` writes from the package today: the Stim text, the Crumble URL
`to_crumble_url` writes, and what the engine says each Pauli placed before each layer flips. The
JavaScript side of the check (the editor writes the same URLs and propagates markers the same
way) runs in `tools/site-tests.mjs`."""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

import stabilizer_qec as sq

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests" / "editor_port" / "crumble.json"


@pytest.fixture(scope="module")
def generated():
    sys.path.insert(0, str(ROOT / "tools"))
    import editor_fixtures  # noqa: E402

    return editor_fixtures


def test_fixture_is_current(generated):
    recorded = json.loads(FIXTURE.read_text())
    fresh = {name: c for name, c in generated.examples().items()}
    assert [e["name"] for e in recorded] == list(fresh)
    for entry in recorded:
        c = fresh[entry["name"]]
        assert entry["stim"] == str(c)
        assert entry["url"] == c.to_crumble_url()
        assert entry["num_detectors"] == c.num_detectors and entry["num_observables"] == c.num_observables
        assert entry["cases"] == generated.propagation_cases(c)


def test_examples_file_matches_fixture():
    recorded = json.loads(FIXTURE.read_text())
    js = (ROOT / "js" / "editor-examples.js").read_text()
    for entry in recorded:
        assert json.dumps(entry["name"]) + ": " + json.dumps(entry["stim"]) + "," in js


def test_every_example_is_a_valid_circuit_with_deterministic_detectors():
    for entry in json.loads(FIXTURE.read_text()):
        c = sq.Circuit(entry["stim"])
        c.detector_error_model(decompose_errors=True)
        if entry["cases"]:
            dets, obs = c.compile_detector_sampler(seed=2).sample(50, separate_observables=True)
            assert not dets.any() and not obs.any(), entry["name"]
