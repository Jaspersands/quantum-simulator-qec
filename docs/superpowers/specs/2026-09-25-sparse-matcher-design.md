# Sub-project B1 — an exact sparse matcher

**Date:** 2026-09-25
**Status:** Approved (Jasper approved all three design sections interactively)

## Where this sits

The roadmap agreed on 2026-09-25, with nothing dropped:

| | Sub-project | State |
|---|---|---|
| A | General circuit path, checked against Stim and PyMatching | Done, merged and deployed |
| **B1** | **Exact sparse matcher (this spec)** | — |
| B2 | Google's Willow and Sycamore data, and Λ | Needs B1; datasets downloaded with approval |
| C | Real-time decoding: bit-parallel sampling, threaded WASM, streaming, latency | — |
| — | Launch kit: pip wheels, CI, written report | — |
| D | Correlated decoding, evaluated on the Willow data | — |
| F | IBM's gross code with BP+OSD | Reuses D's BP |
| E | Lattice surgery | — |
| — | Resource estimator driven by measured Λ | Needs B2 |

B was split in two when Jasper chose "sparse matcher first" over sliding-window decoding. Google's
d = 7, 250-round runs have about 12,000 detectors and on the order of a thousand defects per shot,
and the dense decoder from A takes at most 256 defects at O(n³). Rather than approximate its way
past that, the engine gets the exact algorithm PyMatching 2 uses.

## Problem

1. **The dense decoder does not scale.** `DemDecoder` runs Dijkstra from every defect and an
   O(n³) blossom over the complete defect graph. It refuses above 256 defects, and near that
   limit it takes seconds per shot.
2. **It is 18–136× slower than PyMatching** even where it works (check 4 of the A harness), and
   the gap grows with the patch.
3. **Every later sub-project needs this.** B2 decodes about 21 million real shots, C is about
   real-time decoding, and D's correlated matching reweights and re-runs a matcher.

## Decisions

| Question | Decision |
|---|---|
| Long runs | An exact sparse matcher first, then decode globally (not sliding windows, not short runs only). |
| Algorithm | Sparse blossom, from Higgott and Gidney, arXiv:2303.15933, which is PyMatching 2's core. |
| Oracle | Our dense blossom stays, as `decode_dense`: it is exact, verified, and the first thing the sparse matcher is checked against. |
| Speed | Exactness is non-negotiable. Speed is a measured target, within about 5× of PyMatching single-threaded, and reported as measured. |
| Crates | None new. Parallelism uses `std::thread`. |

## The algorithm, as this implementation follows it

Summarised from the paper, as the reference for the build.

**Detector graph.** One node per detector plus a virtual boundary. Each edge is a graph-like piece
of the error model, weighted `ln((1 − p)/p)` and carrying the observables it flips. Weights are
discretised to **even** integers, so that two regions growing toward each other collide at
`(w − r_u − r_v)/2`, an integer time. Parallel edges merge as A's decoder merges them.

**Graph-fill regions.** A region is a ball growing around a defect (or a blossom of regions). It has:
- a radius `y(t) = m·t + c`, with growth rate `m ∈ {+1, 0, −1}`: growing, frozen or shrinking;
- a shell-area stack, the nodes it owns directly, excluding its blossom children's;
- a blossom parent and children, and a blossom cycle of region edges.

**Per detector node:**
- its owning region;
- `S(v)`, the defect it was reached from;
- `l(v)`, the observables crossed on the way from `S(v)`;
- its radius of arrival `r_a(v)`.

The node's **local radius** is `r_L(v, t) = −r_a(v) + Σ_{A ∈ O(v)} y_A(t)`, summed over the owning
region and its blossom ancestors. When a region grows across edge `(u, v)`,
`l(v) := l(u) ⊕ l(u, v)`.

**Compressed edges.** A region edge stores only its endpoint defects, its observable mask and its
length. A collision across `(u, v)` yields the compressed edge `(S(u), S(v))` with observables
`l(u) ⊕ l(v) ⊕ l(u, v)`, all from O(1) node data.

**Flooder events.**
- **Arrive:** a growing region reaches an empty node `u`, when `r_L(u) = 0`.
- **Collide:** two regions meet across edge `(u, v)` when `r_L(u) + r_L(v) = w(u, v)`. A region
  hits the boundary when `r_L(u) = w(u, boundary)`.
- **Leave:** a shrinking region gives up the node at the top of its shell-area stack when that
  node's `r_L` reaches 0.
- **Implode:** a shrinking region or blossom reaches radius 0.

**Tracker.** A priority queue of *look-at-node* and *look-at-region* reminders, not of every
possible event. A reminder is queued only if it is earlier than the one already held for that
entity. When one fires, the flooder recomputes the entity's true next event, and a reminder made
stale by a rate change is discarded. The timeline advances by popping the minimum.

**Matcher: alternating trees over regions.** Growing regions have shrinking children; shrinking
regions have one growing child. The root is an unmatched growing region. The seven collision cases:

- **(a)** Growing R hits a matched pair (M₁, M₂): M₁ becomes R's child and shrinks, and M₂
  becomes M₁'s child and grows.
- **(b)** Growing R hits growing R′ in a different tree: augment. R and R′ match, and both trees
  dissolve into matched pairs.
- **(c)** Growing R hits growing R′ in the same tree: form a blossom from the cycle through their
  common ancestor. Shrinking regions of the tree that are not on the cycle become the blossom's
  children.
- **(d)** A shrinking blossom reaches radius 0: shatter it. The odd path from its tree-child to
  its tree-parent rejoins the tree, alternating growing and shrinking; the even path becomes
  matched pairs.
- **(e)** A trivial shrinking region R reaches radius 0: its child C must touch its parent P.
  Form the blossom (P, R, C), with the new blossom edge `(C_w, P_u)` carrying observables
  `l(u, v) ⊕ l(v, w)`.
- **(f)** Growing R hits the boundary: R matches to the boundary, and its tree dissolves into
  matched pairs.
- **(g)** Growing R hits region M, which is matched to the boundary: M matches to R instead, and
  the tree dissolves.

**Extraction.**
1. Shatter matched blossoms recursively. For a blossom B matched to R across `(B_u, R_v)`, the
   child holding `u` matches R, and the rest of the cycle pairs off.
2. The prediction is the XOR of the observable masks of all matched compressed edges (at most 64
   observables).
3. The weight is the sum of their lengths, reported in the float weights, as A's decoder does.

## Components

All additive, in a new `src/sparse/` module.

| File | Responsibility |
|---|---|
| `graph.rs` | `SparseGraph::from_dem(&Dem)`: nodes, adjacency, even-integer and float weights, observable masks, the boundary. Merges parallel edges as `DemDecoder::new` does. Rejects p > 0.5. |
| `flooder.rs` | Regions (radius equations, shells, blossom links), per-node state, and the four event types. Queries the tracker; hands collisions and implosions to the matcher. |
| `tracker.rs` | The reminder queue: a binary heap of `(time, entity, version)` with lazy invalidation. |
| `matcher.rs` | Alternating-tree bookkeeping and cases (a)–(g). |
| `extract.rs` | Recursive blossom shattering and the prediction. |
| `mod.rs` | `SparseMatcher::new(&Dem) -> Result<Self, String>` and `decode(&mut self, defects: &[u32]) -> Result<Prediction, DecodeError>`. Resets only the nodes and regions a shot touched. |

**Integration.** `DemDecoder::decode` uses the sparse matcher by default and keeps
`decode_dense` for verification. `decode_bools` and the error type are unchanged. A shot with an
odd component that has no route to the boundary returns `Err(Unmatchable)`, as now. The sparse
matcher has no defect ceiling, so `TooManyDefects` can come only from `decode_dense`.

## Verification

Five layers, cheapest first, each a hard pass or fail.

1. **Brute force.** On random small graphs of up to 14 edges, with and without boundaries,
   including odd components that cannot be matched: the total weight equals the exhaustive
   minimum, and the observables are among the optimal ones.
2. **Against the dense blossom.**
   - On 20,000 random error models.
   - On the surface-code memory circuits (rotated and XZZX, both bases, both noise models,
     d = 3, 5, 7), sampled by the frame sampler over a range of p reaching 256 defects.
   - Every shot must have equal total weight, within discretisation, and equal observables
     wherever the optimum is unique.
3. **Every single fault is corrected**, re-run with the sparse matcher on all 24 circuits
   up to d = 7.
4. **Against PyMatching on identical shots.** `tools/xcheck.py`'s check 2 uses the sparse matcher.
   Every disagreement must be a tie.
5. **Invariants, in debug builds:**
   - no region radius is negative;
   - every tree and blossom edge is tight, and no edge is over-full, so the dual is feasible;
   - every node in a shell area has that region as its recorded owner.

## Performance

- Speed is reported as measured, in check 4's table: µs per shot for PyMatching and for the sparse
  matcher, single-threaded, on d = 3, 5 and 7 under SD6 at p = 0.3% and 0.6%.
- The target is within about 5× of PyMatching.
- The rotated SD6 memory circuit at d = 9 is also measured, as the site's sweep uses it.
- Natively, multi-shot entry points decode in parallel with `std::thread`, one matcher per thread.

## What ships

- **Engine:** `src/sparse/`, and `DemDecoder` defaulting to it.
- **WASM:** rebuilt. Figure 8's live decoding rows, the SD6 sweep and the SD6 bench all use the
  sparse matcher.
- **SD6 sweep:** `SWEEP_RUNS[NOISE.SD6]` is re-set from newly measured cost. If it rises
  meaningfully, the README's SD6 threshold rows are re-measured with `tools/sweep.mjs`.
- **Harness:** `tools/xcheck.py` is re-run in full, and the reference and `report.txt` are
  regenerated.
- **README:** a section on the sparse matcher (what it is, the five layers, speed against
  PyMatching), with anything the checks catch written up like the defect sections.

## Out of scope

- **B2:** Willow and Sycamore ingestion, sweep-bit and Pauli-gate parser support, Λ, the new
  site section.
- **C:** bit-parallel sampling, threaded WASM, windowed and streaming decoding, latency.
- **D:** correlated matching.
- **The old per-code decoders** and every number they produce stay untouched.

## Risk

Sparse blossom is intricate. The likeliest failure is a rare collision case handled wrongly,
which shows up as a matching heavier than the dense optimum on some shot. Layer 2 exists to catch
exactly that, and so runs on a large, varied set of shots rather than a few dozen.
