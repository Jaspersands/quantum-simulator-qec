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
- A change keeps to the [compatibility policy](#compatibility-policy) below.
- Names follow the tools a user knows: Stim's for circuits and models, PyMatching's for
  matching, `ldpc`'s for BP and BP+OSD.
- Each change goes in `CHANGELOG.md`.

## Compatibility policy

The Python package and the Rust crate follow semantic versioning from 1.0, on these terms. The
public API is every name importable from `stabilizer_qec` and its subpackages without a leading
underscore, and every item in the crate's documentation. `stabilizer_qec._core`, the crate's
hidden modules, and names a module merely imports (a standard-library module, another
package's class) are not part of it.

- **Names.** A public name keeps working, with the same signature, through every 2.x release
  (`cargo semver-checks` checks the crate's in CI). One to be removed is deprecated first, with
  a `DeprecationWarning` (in Rust, `#[deprecated]`) naming its replacement, for at least one
  minor release before a major release removes it.
- **Behaviour.** What a public call returns or raises changes in a minor release only to correct
  it: to make a port do what its upstream (the tool and version it ports) does, its errors and
  refusals included, or to fix a defect. Each such change is a bullet under Changed or Fixed in
  `CHANGELOG.md` that says what changed, why, and what to call instead.
- **Seeds.** Within a version, a seed and shot position give the same shots on any number of
  threads. A correction to a sampler's algorithm can change its seeded shots; the changelog
  names the sampler and the version.

## Reporting a security problem

See [SECURITY.md](SECURITY.md).
