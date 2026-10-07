# Submitting to the Journal of Open Source Software (JOSS)

The paper is `paper/paper.md` with its references in `paper/paper.bib`. The workflow
`.github/workflows/paper.yml` builds it into the journal's PDF on every change and uploads it as
the run's artifact `paper`.

Submitting is done by the author, from their own account, at
[joss.theoj.org/papers/new](https://joss.theoj.org/papers/new). Before submitting:

**The paper.**

- [ ] Affiliation: `paper.md` says "Independent researcher". Replace it if there is a better one
      (a ROR identifier is optional).
- [ ] ORCID: add `orcid: 0000-...` under the author if you have one (recommended, not required).
- [ ] Read the *AI usage disclosure* section and change it to describe the work as you see it.
      JOSS requires the disclosure, and its wording is the author's to stand behind.
- [ ] The numbers in the paper come from the technical report (`report/report.md`, built
      4 October 2026). If the report is rebuilt with new data, check them again.
- [ ] Update the `date:` field to the submission date.
- [ ] The latest run of the Paper workflow is green, and its PDF reads correctly.

**The software (JOSS's review checklist).**

- [x] An OSI licence (MIT, `LICENSE`).
- [x] A public repository with an issue tracker.
- [x] Installation instructions, and a package on PyPI and crates.io.
- [x] Example usage (README, `tutorials/`, the API guide).
- [x] API documentation ([qcompiler.jaspersands.com/api](https://qcompiler.jaspersands.com/api/), docs.rs).
- [x] Automated tests, run in CI on every push.
- [x] Community guidelines: how to contribute, report issues and get support (`CONTRIBUTING.md`).
- [ ] An archived release with a DOI. Zenodo archives each GitHub Release from 1.6.0 on (see
      `docs/zenodo.md`); the DOI goes in `CITATION.cff` and the submission form.

**Submitting.**

1. Sign in to JOSS with GitHub and open a new submission.
2. Give the repository URL, the branch holding `paper/` (`master`), the software version and the
   archive DOI.
3. JOSS's editorial bot builds the paper from the repository; reviewers then open issues on the
   repository. Changes made in response are released as new versions, and the final version's
   archive DOI is given to the editor at acceptance.
