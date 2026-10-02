# Security

`stabilizer-qec` reads circuits, error models and shot data that may come from anywhere, so a
crash, a hang or unbounded memory on some input is a bug worth reporting, and one that could
be triggered by a file someone else supplies is a security bug.

## Reporting

Please report a vulnerability privately, through GitHub's
[private vulnerability reporting](https://github.com/Jaspersands/quantum-simulator-qec/security/advisories/new)
(the repository's Security tab, "Report a vulnerability"), rather than in a public issue.
Include the input that triggers it, the version (`stabilizer_qec.__version__`, or the crate's),
and what happens. You should hear back within a week.

## Supported versions

Fixes go into the latest 1.x release of the Python package and of the crate.

## What the code promises

Bad input raises `ValueError` or `TypeError` in Python and returns `Error` in Rust; it does
not panic, crash or run without bound. Sizes are capped (qubit indices, unrolled loops,
detectors), each cap stated where it applies. The parsers and decoders are fuzzed in CI
(`fuzz/` with cargo-fuzz, and `tests/test_fuzz.py` with Hypothesis).
