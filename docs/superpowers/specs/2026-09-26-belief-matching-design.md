# D — Belief propagation and belief-matching on Google's data

**Date:** 2026-09-26
**Status:** Decided under Jasper's standing instruction of 2026-09-26 ("do both or the biggest";
questions answered that way and marked).

## Why

Matching treats a circuit's faults as independent edges; correlated matching adds one round of
correction for faults that set off both halves of the graph. Belief-matching (Higgott, Bohdanowicz,
Kubica, Flammia and Campbell, PRX 13, 031007, 2023; arXiv:2203.04948) goes further:

1. It runs belief propagation on the whole error model: a hypergraph with every fault as it is,
   none of it split.
2. It re-weights the matching graph for this shot from BP's posterior marginals.
3. It matches.

On Sycamore's data it was one of the two decoders that made d = 5 beat d = 3. Google published its
predictions, so ours can be held to them.

BP is also the first half of BP+OSD, the decoder sub-project F needs for IBM's gross code. It is
built once, here, for both.

## Decisions

| Question | Decision |
|---|---|
| BP variant | **Both**: product-sum (the default; what belief-matching uses) and min-sum with a scaling factor. |
| Schedule | Parallel (flooding): every check updates, then every variable. Serial schedules are left out; the oracles below use flooding. |
| Iterations | **20** by default, as the `beliefmatching` package uses. Stop early when the hard decision reproduces the syndrome. |
| On convergence | If BP converges, its own correction is the answer, as in `beliefmatching`. Otherwise its posteriors re-weight the matching graph and the sparse matcher decides. |
| From posteriors to edge weights | Each graph edge's probability is the XOR-combination of the posteriors of the faults whose decomposition contains it, $p_e = \tfrac12\bigl(1 - \prod_i (1 - 2p_i)\bigr)$. Its weight is $\ln((1-p_e)/p_e)$. |
| Posteriors above ½ | **Negative weights, handled exactly** (biggest). An edge with $p_e > ½$ is assumed fired: its endpoints' detection events and its observables are flipped, and it is matched with weight $\lvert w\rvert$. This is how PyMatching handles them. Clamping at ½ would be simpler, but it is not the algorithm. |
| Per-shot weights | The matcher already keeps per-scratch weights for correlated matching. Belief-matching sets all of them per shot, on the model's fixed discretisation scale, and saturates weights beyond the scale's range. |
| Where BP runs | On the undecomposed hypergraph. A fault whose symptoms are written in pieces (`^`) counts as one variable with the XOR of its pieces as its symptom, and faults with identical symptoms merge. |

## Verification

1. **Oracle: `ldpc` and `beliefmatching`**, the Python packages from the paper's authors, installed
   into `.venv`.
   - **Posteriors.** On random small parity-check matrices and on surface-code error models, our
     flooding product-sum BP must give the same posterior LLRs as `ldpc`'s after every iteration,
     to floating-point rounding. Min-sum must agree too, with the same scaling factor.
   - **Belief-matching.** On identical shots, our predictions must equal `beliefmatching`'s except
     where matching ties. Every disagreement is explained as in check 5 of the cross-check: the
     weights of the two corrections under the shot's posterior weights are equal.
2. **Exact small cases.** A fault alone must be corrected. With zero iterations, BP must reproduce
   the priors exactly, and belief-matching must reduce to plain matching.
3. **Google's recorded belief-matching on Sycamore.** Every experiment is decoded with the pij
   priors, cross-fitted as Google used them. Failure rates, ε₃, ε₅ and Λ are set beside Google's
   recorded predictions on the same shots, with the rate at which the two agree shot by shot.
   Google's BP settings are not published in the dataset, so this is a statistical comparison, not
   an exact one.
4. **Willow.** Every experiment is decoded with Google's SI1000 prior and fitted as in section 11.
   Λ and ε₇ are set beside correlated matching and beside Google's decoders.

## Cost, and the one hedge

BP costs a pass over the Tanner graph for each iteration. The heaviest experiment is d = 7 at 250
rounds, with about 12,000 detectors. The run's cost is measured on it first:

- If all of Willow at 50,000 shots an experiment fits in about 8 hours on 10 cores, it runs whole.
- Otherwise the longer experiments are decoded on their first 10,000 shots, and the record says so.
  Sycamore always runs whole.

## Site and write-up

- **Section 11.** Figure 9 fits every decoder in the results files, so ours appears there as
  "ours, belief-matching" with each prior. Figure 10 adds belief-matching to its live decode of
  2,000 raw Willow shots at d = 3, 5 and 7, in WebAssembly.
- **README.** A "Belief-matching" section: method, the oracle checks, and the tables.
- **The report** gets a results subsection.
