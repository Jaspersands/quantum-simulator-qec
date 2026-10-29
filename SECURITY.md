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

Fixes go into the latest 2.x release of the Python package and of the crate.

## What the code promises

Core validation uses `ValueError` or `TypeError` in Python and `Error` in Rust; ports may
raise their documented upstream exception types. Panics, crashes and hangs remain bugs to
report. Checked sizes do not guarantee recovery from every allocation or OS memory failure.
Sizes are capped (qubit indices, unrolled loops, detectors), each cap stated where it applies. The parsers and decoders are fuzzed in CI
(`fuzz/` with cargo-fuzz, and `tests/test_fuzz.py` with Hypothesis).

## Configuration that runs code

The qecsim port (`stabilizer_qec.qecsim`, `stabilizer-qec qecsim`) reads two files that are code:
a logging configuration, `logging_qecsim.ini`, whose handler arguments Python's
`logging.config.fileConfig` evaluates, and a Blossom V library, `clib/libpypm.so`, which it loads
with `ctypes` and uses in place of this package's matcher. It looks for them only in
`$QECSIM_CFG`, or in `~/.qecsim` when that is unset or empty, so point `QECSIM_CFG` only at a
directory you trust. Unlike qecsim, it does not search the working directory, and it treats an
empty `QECSIM_CFG` (which qecsim took as the working directory, and which a CI `env` entry for an
undefined variable gives) as unset, so running it inside a directory someone else supplied does not
run that directory's files. A relative `QECSIM_CFG`, such as `.` or `cfg`, is still read relative to
the working directory, as in qecsim, so give it as an absolute path.
