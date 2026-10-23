# Contributing

Issues and pull requests are welcome. The project's one rule is that a claim is checked: a
change to the engine comes with a test, and wherever a reference implementation exists (Stim,
PyMatching, `ldpc`, `beliefmatching`), the test compares against it.

## Questions, bugs and support

Ask a question, report a bug or request a feature by opening an issue at
[github.com/Jaspersands/quantum-simulator-qec/issues](https://github.com/Jaspersands/quantum-simulator-qec/issues).
A bug report is most useful with the smallest circuit or model that shows it, the version
(`stabilizer_qec.__version__`), and, where it applies, what Stim, PyMatching or `ldpc` give on
the same input.

## Building

```bash
cargo test --release                     # the Rust engine and the crate's API
pip install maturin && maturin develop --release
pip install pytest hypothesis stim pymatching sinter
pytest tests                             # the Python package, against the references
```

The site's WebAssembly engine is `cargo build --release --target wasm32-unknown-unknown
--no-default-features`, copied to the repository root; `node tools/site-tests.mjs` checks it.

## Conventions

- Bad input must produce a controlled error (core `ValueError` / `TypeError` / Rust `Error`,
  or a port's documented exception). Panics, crashes and hangs are bugs.
- The public APIs follow semantic versioning: the Python package's names without a leading
  underscore, and the crate's documented items (`cargo semver-checks` runs in CI). Anything
  to be removed is deprecated for at least one minor release first.
- Names follow the tools a user knows: Stim's for circuits and models, PyMatching's for
  matching, `ldpc`'s for BP and BP+OSD.
- Preserve seeded shots within each documented sampler/version contract; corrections that
  alter outcomes must say so in the changelog, with migration scope.
- Each change goes in `CHANGELOG.md`.

## Reporting a security problem

See [SECURITY.md](SECURITY.md).
