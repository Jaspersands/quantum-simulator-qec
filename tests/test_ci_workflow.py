"""The CI workflow (.github/workflows/ci.yml) runs the differential tests that have no other guard.

The 18-entry test matrix installs its references from wheels only, and a reference it cannot install
turns its tests into skips, so a comparison can be dead on every runner while CI stays green. These
checks hold the workflow to running, somewhere and with installation failures fatal: the live Tsim
comparisons (bloqade-tsim, which the matrix never installs), sinter 1.16.0 and relay-bp 0.2.2 (both
publish only sources, which a wheels-only install never takes), and PanQEC's slow tests (skipped
unless PANQEC_RUNSLOW is set). The workflow is read as text, line by line, so that no YAML parser is
needed.
"""

from __future__ import annotations

import gzip
import json
import pathlib
import re

import pytest

from conftest import REFERENCE_VERSIONS

ROOT = pathlib.Path(__file__).parents[1]
CI = ROOT / ".github" / "workflows" / "ci.yml"
if not CI.exists():
    pytest.skip("needs the repository's .github/workflows/ci.yml", allow_module_level=True)

TSIM = json.loads(gzip.decompress((ROOT / "tests" / "tsim_port" / "zx_graphs.json.gz").read_bytes()))


def _jobs() -> dict:
    """Each job's lines, by its id (the two-space keys under ``jobs:``)."""
    jobs: dict = {}
    current = None
    inside = False
    for line in CI.read_text().splitlines():
        if line.startswith("jobs:"):
            inside = True
            continue
        if not inside:
            continue
        m = re.match(r"^  ([A-Za-z0-9_-]+):\s*$", line)
        if m:
            current = jobs.setdefault(m.group(1), [])
        elif current is not None and line.strip() and not line.startswith(" "):
            current = None
        if current is not None:
            current.append(line)
    return jobs


def _steps(job: list) -> list:
    """A job's steps, each as its text (from its ``- `` line to the next step's)."""
    steps: list = []
    for line in job:
        if re.match(r"^      - ", line):
            steps.append([])
        if steps:
            steps[-1].append(line)
    return ["\n".join(s) for s in steps]


def _commands(step: str) -> list:
    """The step's shell commands, comments dropped, continuation lines joined."""
    lines = [line.strip() for line in step.split("\n")]
    out: list = []
    for line in lines:
        if not line or line.startswith("#"):
            continue
        if out and out[-1].endswith("\\"):
            out[-1] = out[-1][:-1] + " " + line
        else:
            out.append(line)
    return out


def _hard_installs(step: str) -> list:
    """The step's ``pip install`` commands that fail the job when they fail (no ``||`` fallback, not
    inside a loop that turns a failure into a notice)."""
    return [c for c in _commands(step) if "pip install" in c and "||" not in c and not c.startswith("for ")]


def _allows_source(command: str, dist: str) -> bool:
    """Whether ``pip`` may build ``dist`` from its sources under ``command``'s format options."""
    no_binary = re.findall(r"--no-binary[= ](\S+)", command)
    if any(dist in v.split(",") or ":all:" in v.split(",") for v in no_binary):
        return True
    return "--only-binary" not in command


def _job_with(predicate) -> list:
    """The ids of the jobs one of whose steps satisfies ``predicate``."""
    return [name for name, job in _jobs().items() if any(predicate(s) for s in _steps(job))]


def _runs_tests(step: str, *paths: str) -> bool:
    """Whether one of the step's pytest commands names every one of ``paths``."""
    return any("pytest" in c and all(p in c.split() or p + "/" in c.split() for p in paths) for c in _commands(step))


def _fails_on(step: str, pattern: str) -> bool:
    """Whether the step, after its pytest run, fails when the run's output matches ``pattern``."""
    return any(re.search(r"\bgrep\b.*" + pattern, c) and "exit 1" in c for c in _commands(step))


def _guarded_run(job: list, *paths: str) -> bool:
    """A step of the job runs ``paths`` and fails when its summary reports any skip."""
    return any(_runs_tests(s, *paths) and _fails_on(s, re.escape("[0-9]+ skipped")) for s in _steps(job))


def test_a_job_runs_the_live_tsim_comparisons():
    """bloqade-tsim, at the version the fixtures were recorded with, installed where a failure fails
    the job, and every Tsim test file run there with a skip counted as a failure (24 comparisons
    with Tsim are dropped silently, not skipped, when ``tsim`` does not import)."""
    tsim_pin = f"bloqade-tsim=={TSIM['tsim']}"
    zx_pin = f"pyzx-param=={TSIM['pyzx_param']}"
    jobs = _jobs()
    installing = _job_with(lambda s: any(tsim_pin in c.split() and zx_pin in c.split() for c in _hard_installs(s)))
    assert installing, f"no job installs {tsim_pin} with {zx_pin} without a fallback"
    files = ["tests/test_tsim_zx.py", "tests/test_tsim_circuit.py", "tests/test_tsim_differential.py", "tests/test_tsim_encoders.py", "tests/test_tsim_surface.py", "tests/tsim_port"]
    running = [n for n in installing if _guarded_run(jobs[n], *files)]
    assert running, f"jobs {installing} install Tsim but none runs {files} failing on skips"
    assert any(re.search(r"import [^\n]*\btsim\b", "\n".join(jobs[n])) for n in running), "the job does not check that tsim imports"


def test_sinter_and_relay_bp_pins_are_built_from_their_sources():
    """sinter 1.16.0 and relay-bp 0.2.2 publish no wheels: a job installs them from their sources
    (relay-bp with a Rust toolchain) without a fallback and runs their differential files."""
    relay_pin = "relay-bp==" + REFERENCE_VERSIONS["relay_bp"][1]
    jobs = _jobs()

    def installs(step):
        return any(all(p in c.split() for p in ("sinter==1.16.0", relay_pin)) and _allows_source(c, "sinter") and _allows_source(c, "relay-bp") for c in _hard_installs(step))

    names = _job_with(installs)
    assert names, f"no job builds sinter==1.16.0 and {relay_pin} from their sources without a fallback"
    good = [n for n in names if "rust-toolchain" in "\n".join(jobs[n]) and _guarded_run(jobs[n], "tests/test_sinter_differential.py", "tests/test_relay_bp_package.py")]
    assert good, f"jobs {names} install them but none has Rust and runs their tests failing on skips"


def test_the_wheels_only_loop_names_no_pin_it_cannot_install():
    """The matrix's install loop takes wheels only: it names no reference pinned to a version that has
    none (relay-bp 0.2.2, sinter 1.16.0, and qldpc 0.4.1, which needs sinter 1.16)."""
    loops = []
    for job in _jobs().values():
        for s in _steps(job):
            commands = _commands(s)
            # A loop's line with the install command in its body.
            loops += [c + " " + commands[i + 1] for i, c in enumerate(commands[:-1]) if c.startswith("for p in ") and "pip install" in commands[i + 1]]
    assert loops
    for loop in loops:
        words = loop.split()
        for pin, source in (("relay-bp==0.2.2", "relay-bp"), ("sinter==1.16.0", "sinter"), ("qldpc==0.4.1", "sinter")):
            assert pin not in words or _allows_source(loop, source), f"{pin} is in a wheels-only loop"


def test_a_job_runs_panqecs_slow_tests():
    """PanQEC's slow tests (every decoder's single-qubit-error test among them) run in some job: a
    step sets PANQEC_RUNSLOW, runs tests/panqec_port and fails should any test still report the slow
    tests' skip (PanQEC's suite keeps skips of its own), in a job that checks its extra imports."""
    jobs = _jobs()

    def slow_step(step):
        return bool(re.search(r"PANQEC_RUNSLOW:\s*\"?1\"?", step)) and _runs_tests(step, "tests/panqec_port") and _fails_on(step, "needs PANQEC_RUNSLOW")

    good = _job_with(slow_step)
    assert good, "no step sets PANQEC_RUNSLOW, runs tests/panqec_port and fails on its slow tests' skip"
    # The text the step looks for is the skip's reason, as tests/panqec_port/conftest.py gives it.
    assert re.search(r"reason=\"needs PANQEC_RUNSLOW\b", (ROOT / "tests" / "panqec_port" / "conftest.py").read_text())
    # tests/conftest.py leaves tests/panqec_port uncollected (not skipped) without the panqec extra.
    for module in ("scipy", "pandas", "matplotlib", "click", "psutil", "tqdm"):
        assert any(re.search(r"import [^\n]*\b" + module + r"\b", "\n".join(jobs[n])) for n in good), f"the job does not check that {module} imports"
