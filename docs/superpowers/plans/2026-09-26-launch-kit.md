# Launch kit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the engine installable, continuously checked, and written up: a pip wheel with a typed interface, CI on every push, and a technical report whose every number comes from committed data.

**Architecture:**
- **Package.** maturin builds the existing PyO3 module as one abi3 wheel; the `.pyi` at the repository root ships in it.
- **Smoke tests.** Two end-to-end checks drive the engine the way users do: a Python one against Stim and PyMatching, and a Node one through the page's wrappers.
- **CI.** GitHub Actions runs the Rust tests, both smoke tests and the site tests.
- **The report.** It is Markdown with placeholders, filled by `tools/report.py` from `data/`. Pandoc makes it HTML, and headless Chrome prints the PDF.

**Tech Stack:**
- Rust, PyO3 0.22 (abi3-py39) and maturin 1.x;
- GitHub Actions: dtolnay/rust-toolchain, PyO3/maturin-action, pypa/gh-action-pypi-publish;
- Node 22;
- pandoc 3 and matplotlib;
- headless Chrome.

Spec: `docs/superpowers/specs/2026-09-26-launch-kit-design.md`.

## Global Constraints

- Package name `stabilizer-qec`, import name `stabilizer_qec`. One abi3 wheel, CPython ≥ 3.9, MIT.
- The Python module builds with the `python` Cargo profile (release, unstripped); a stripped arm64 build does not load on macOS.
- Publishing to PyPI is prepared, never performed: it needs the account owner.
- Every number in the report is computed from a committed file, or quoted word for word from the README. The build fails otherwise.
- No dataset is downloaded. The report and CI use only committed data (`data/willow-extract/` for the Willow check).

---

### Task 1: The package

**Files:**
- Create: `pyproject.toml`, `LICENSE`, `python/README.md`, `stabilizer_qec.pyi`, `tools/smoke.py`
- Modify: `Cargo.toml` (version 0.2.0, abi3-py39, crate metadata, `include` so the sdist carries only the crate), `src/py_api.rs` (docstrings), `.gitignore` (`/target-wheel`, `/dist`)

- [x] **Step 1:** Write `tools/smoke.py`. It must check:
  - the error model against Stim's (symptoms merged);
  - m2d bit for bit;
  - the batch sampler's detection count;
  - plain matching ties PyMatching, and correlated failures are within 4σ of PyMatching's;
  - windows explain every defect and are no worse than global beyond 4σ;
  - a stream explains every defect.
- [x] **Step 2:** `CARGO_TARGET_DIR=target-wheel maturin build --target universal2-apple-darwin --out dist`
- [x] **Step 3:** In a fresh venv (Python 3.13): `pip install dist/*.whl numpy stim pymatching`, then run `python tools/smoke.py` from outside the repository. Expected: `all checks passed`.
- [x] **Step 4:** Python 3.9 (system) and the x86_64 slice under Rosetta: import, sample, decode. The two must give identical failures from the same seed.
- [x] **Step 5:** `maturin sdist`, then `pip wheel` from the sdist.
- [x] **Step 6:** Commit.

### Task 2: CI

**Files:**
- Create: `.github/workflows/ci.yml`, `.github/workflows/wheels.yml`, `tools/wasm-smoke.mjs`
- Modify: `js/engine.js` (`globalThis.crypto`: `self` does not exist in Node)

- [x] **Step 1:** `tools/wasm-smoke.mjs` checks four things:
  - a phenomenological run;
  - each Willow extract's m2d, hashed against the manifest;
  - correlated decoding;
  - a window-decoded stream against global decoding.
- [x] **Step 2:** Run it on the committed engine and on a fresh build. Expected: `all checks passed`.
- [x] **Step 3:** Write `ci.yml` with two jobs:
  - the engine job: Rust tests, the wasm32 build, the site tests, and wasm-smoke on both engines;
  - the package job, on three OSes: build, install, and smoke from outside the repository, plus `xcheck --quick` on Linux.

  Write `wheels.yml`: manylinux x86_64/aarch64, macOS universal2, Windows x64 and an sdist, with trusted publishing on `v*` tags only.
- [x] **Step 4:** Validate both workflows as YAML, and run each step locally.
- [x] **Step 5:** Commit.

### Task 3: The report

**Files:**
- Create: `tools/report.py`, `report/report.md`, `report/template.html`, `report/report.css`, `report/figures/*.svg` (generated), `report/report.html` and `report/report.pdf` (generated)
- Modify: `tools/lambda.mjs` (`--json`: every fit, with intervals, the pairwise Λ and the validation), `.gitignore` (`/report/build`), `index.html` (links to the report), `README.md` (Install, CI and Technical report sections)

- [x] **Step 1:** `node tools/lambda.mjs 400 --json report/build/fits.json data/realtime/willow-windows.json` writes `willow`, `sycamore`, `validation` and `files`.
- [x] **Step 2:** `tools/report.py` builds the values and tables from `data/` and the fits. It draws four figures:
  - Willow's ε against d;
  - agreement with PyMatching;
  - decoding and sampling speed;
  - latency against cores.

  It fills `{{…}}` placeholders and exits naming any it cannot fill. It then runs pandoc (`--mathml --toc --number-sections`) and Chrome's `--print-to-pdf`.
- [x] **Step 3:** Write `report/report.md`, following the spec's outline.
- [x] **Step 4:** Build it, render every PDF page to an image, and read each one. Fix lists that lack a blank line, exponent tick labels, and the heading levels.
- [x] **Step 5:** Link it from the hero and the footer, and add the README sections.
- [x] **Step 6:** Rebuild after the million-round rerun, so the report quotes the working unexplained count. Then commit.
