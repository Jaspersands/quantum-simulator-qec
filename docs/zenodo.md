# Archiving releases on Zenodo (a DOI for citing)

Zenodo archives each GitHub Release of the repository and gives it a DOI, plus one *concept*
DOI that always resolves to the latest version. `.zenodo.json` holds the metadata it uses
(title, author, description, licence, keywords, links); the version and date come from the
release.

Linking the repository is done once, by its owner, on Zenodo:

1. Sign in at [zenodo.org](https://zenodo.org) with GitHub ("Log in" → GitHub).
2. Open **GitHub** in the account menu ([zenodo.org/account/settings/github](https://zenodo.org/account/settings/github/)),
   press **Sync now**, and switch on `Jaspersands/quantum-simulator-qec`.
3. The next GitHub Release is archived automatically (within a few minutes of being
   published). Its record on Zenodo shows the version's DOI and the concept DOI.

Then, with the concept DOI (`10.5281/zenodo.NNNNNNN`):

- add `doi: 10.5281/zenodo.NNNNNNN` to `CITATION.cff`;
- add the badge to the top of `README.md`:
  `[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.NNNNNNN.svg)](https://doi.org/10.5281/zenodo.NNNNNNN)`.

Zenodo archives *every* published release. Each version therefore has one GitHub Release,
`v*`, whose notes cover the Python package and the crate. The `crate-v*` tag still publishes
the crate to crates.io, but gets no Release of its own, so a version is archived once.
