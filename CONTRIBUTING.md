# Contributing

Issues and pull requests are welcome. The project's one rule is that a claim is checked: a
change to the engine comes with a test, and wherever a reference implementation exists (Stim,
PyMatching, `ldpc`, `beliefmatching`), the test compares against it.

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

- Bad input is an error (`ValueError` / `Error`), never a panic or a crash.
- The public APIs follow semantic versioning: the Python package's names without a leading
  underscore, and the crate's documented items (`cargo semver-checks` runs in CI). Anything
  to be removed is deprecated for at least one minor release first.
- Names follow the tools a user knows: Stim's for circuits and models, PyMatching's for
  matching, `ldpc`'s for BP and BP+OSD.
- A seed's shots stay the same within 1.x; a change that alters them says so in the
  changelog.
- Each change goes in `CHANGELOG.md`.

## Reporting a security problem

See [SECURITY.md](SECURITY.md).
