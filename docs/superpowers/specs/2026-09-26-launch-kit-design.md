# Launch kit — a package, continuous integration, and a technical report

**Date:** 2026-09-26
**Status:** Decided under Jasper's standing instruction of 2026-09-26 ("do both or the biggest";
questions answered that way and marked).

## Why

The engine and the site are now worth someone's time: an exact sparse matcher and correlated
matching checked shot for shot against PyMatching, Google's hardware data decoded from raw readouts
with Λ fitted as Google fits it, and real-time window decoding measured against Willow's cycle.

What is missing is what makes them usable and checkable by someone else:
- a package they can install;
- tests that run on every change without this machine;
- a document that can be read in one sitting, cited, and handed to a colleague.

## Decisions

| Question | Decision |
|---|---|
| Package | **A pip-installable wheel** of the PyO3 module, built by maturin from a `pyproject.toml`: `pip install stabilizer-qec`, `import stabilizer_qec`. It is built and tested here from a fresh virtualenv. |
| Typed interface | **A `.pyi` stub** documenting every function's signature and meaning, so editors and readers see the API. |
| Publishing | **Prepared, not performed.** A tag-triggered workflow builds wheels for Linux, macOS (arm64 and x86_64) and Windows and publishes them to PyPI with trusted publishing. Publishing under Jasper's name is outward-facing and needs his account, so the workflow is inert until he enables it, and the README says how. |
| CI | **GitHub Actions on every push and pull request:** Rust tests; the wasm32 build; the site tests; a Node smoke test of the built WASM; the Python module built with `--profile python` and the quick cross-check against Stim and PyMatching. The committed WASM is not compared byte for byte (toolchains and paths differ across machines); instead the job builds its own and runs the site's WASM tests on it. |
| Report | **Both** a PDF and a web page. `report/report.md` is the source. `tools/report.py` renders its figures with matplotlib from the committed data. pandoc turns it into `report.html`, styled like the site and published with it, and headless Chrome prints the page to `report/report.pdf`. |
| Report scope | About ten pages covering the engine, how it is verified, and the results, with every number drawn from a committed file. The results are: thresholds; agreement with Stim and PyMatching; the sparse and correlated matchers; Google's data and Λ; throughput; and real time. |

## The report's outline

1. **Summary**: what the engine is, and the five results in five sentences.
2. **The engine**:
   - the stabilizer tableau and circuits in Stim's language;
   - error models built by walking circuits backwards;
   - the frame and batch samplers;
   - the sparse blossom matcher and correlated matching;
   - window decoders.
3. **How it is checked**:
   - the verification layers (exact oracles first, then references, then statistics);
   - the defects found and fixed, as the evidence that the layers work.
4. **Results:**
   - 4.1 Thresholds (SD6, rotated and XZZX), with the finite-size fit.
   - 4.2 Agreement with Stim (error models, every mechanism) and PyMatching (every disagreement a tie; plain and correlated).
   - 4.3 Google's hardware: the detection events rebuilt bit for bit, the Willow and Sycamore tables, Λ, and the validation against Sycamore's published numbers.
   - 4.4 Throughput: samplers against Stim, and the browser pool.
   - 4.5 Real time: latency against cores, accuracy against global decoding, the million-round stream.
5. **Limits and next steps**: belief-matching (D), the IBM gross code (F), lattice surgery (E), and the resource estimator.
6. **Reproducing everything**: commands, data sources and licences.

## Verification

- A fresh virtualenv installs the wheel and runs a smoke script: generate a circuit, sample it, decode it plainly and correlated, and compare with Stim and PyMatching.
- The CI workflow is validated as YAML, and its steps are run locally, in order.
- Every number in the report is taken from the committed data by `tools/report.py` or quoted from a README section that is. The report's build fails if a number it quotes is absent from its source.
