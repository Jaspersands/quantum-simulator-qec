# Sub-project C — throughput, and real-time decoding

**Date:** 2026-09-26
**Status:** Decided under Jasper's standing instruction of 2026-09-26: "proceed with all your planning
and executing … assume the answer where you do both or the biggest unless it's super slow or super
large storage". Every decision below that would otherwise have been a question says which option was
taken and why.

## Where this sits

| | Sub-project | State |
|---|---|---|
| A | General circuit path, checked against Stim and PyMatching | Done, deployed |
| B | Sparse matcher, correlated matching, Google's Willow and Sycamore data, Λ | Done, deployed 2026-09-26 (master 744e019) |
| **C** | **Throughput and real-time decoding (this spec)** | — |
| — | Launch kit: pip wheels, CI, written report | — |
| D | Belief-matching and BP, evaluated on the Willow data | — |
| F | IBM's gross code with BP+OSD | — |
| E | Lattice surgery | — |
| — | Resource estimator driven by measured Λ | — |

## The question C answers

A quantum computer's decoder has a deadline. Willow runs one error-correction cycle every
1.1 µs, and a decoder that falls behind builds a backlog that grows without bound, which stalls
any computation waiting on a logical measurement. Google reported decoding a distance-5 memory in
real time over a million cycles, with an average latency of 63 µs (arXiv:2408.13687).

So C asks, of this engine:
1. **Throughput.** How many shots, and how many rounds, it can produce and decode per second, on
   one core and on many, natively and in the browser.
2. **Real time.** Whether it can decode a stream of rounds as they arrive at Willow's cadence, with
   how many cores, and with what latency. It is measured on Google's own recorded syndromes as
   well as on simulated ones.

## Decisions

| Question | Decision (and why) |
|---|---|
| Structure | Two plans under this one spec, as B2 was. **C1:** throughput, meaning a bit-parallel sampler and a worker pool in the browser. **C2:** real time, meaning window decoders, a streaming harness, long streams and a site section. Both, per the standing rule. |
| Threads in the browser | A **pool of workers, each with its own WebAssembly instance**. WebAssembly threads need SharedArrayBuffer, which needs cross-origin isolation headers that GitHub Pages cannot send. The only workaround is a service worker that rewrites responses, which is fragile. The pool gives the same parallel throughput for Monte Carlo, which shares nothing between shots. |
| Sampler | A new **bit-parallel frame sampler**: 64 shots per machine word, noise drawn by geometric skipping. It runs **`REPEAT` blocks without flattening them**, with a ring buffer of measurement records, so a million-round circuit costs constant memory and streams its detection events round by round. The existing sampler stays as the reference. |
| Window decoders | **Both** the sequential sliding window and the parallel ("sandwich") window decoder of Skoric et al. (arXiv:2209.08552) and Tan et al. (arXiv:2209.09219). **Both** plain and correlated matching inside a window. |
| Window sizes | Commit C = d rounds, buffer B = d rounds (the literature's default). Accuracy is also measured at B = d/2 and B = 2d, to show where it degrades. |
| Streams | **Both**: Google's recorded syndromes (Willow d = 3, 5, 7 at up to 250 rounds, where the full error model exists), and simulated streams of **a million rounds** at d = 5, like Google's real-time experiment. The long stream needs a window graph built once and reused (time translation), so it never needs the million-round error model. |
| Latency | Measured, not modelled: every window's decode time is timed natively. Rounds arrive every 1.1 µs in virtual time, and a scheduler over K cores gives each window's completion time. The latency (the last needed round's arrival to the commit) and the backlog follow from those. A real multithreaded run confirms the scheduler's throughput. |
| Site | Section 12, "Real time". A recorded figure: native results, cores needed and latency at d = 3, 5, 7, and Willow's stream. A live figure: a d = 5 stream sampled and window-decoded by the worker pool, with rounds per second against the 1.1 µs budget and a latency histogram. The threshold sweep and bench also use the pool, so more shots and the same page. |
| Crates | None new. Native threads use `std::thread::scope`. |

---

# Part 1 — throughput (Plan C1)

## The bit-parallel sampler (`src/batch_sampler.rs`)

**The frame.** Per qubit, two `u64` words, the X and Z components of the Pauli frame for 64 shots
at once. Each instruction acts on whole words:
- `H` swaps x and z.
- `CX c t`: `x[t] ^= x[c]; z[c] ^= z[t]`.
- `CZ a b`: `z[b] ^= x[a]; z[a] ^= x[b]`.
- Pauli gates and sweep-controlled X leave the frame alone, as in `FrameSampler`.
- Resets and measurements randomise the insensitive component with a random word, as
  `FrameSampler` does, so a nondeterministic detector shows up as a coin flip.
- `M(p)` records `x[q]` (or `z[q]` for `MX`) XOR a Bernoulli word.

**Noise.** A Bernoulli(p) word is drawn by geometric skipping when p < 1/4: the gap to the next set
lane is `floor(ln(u) / ln(1 − p))`, so a word at p = 10⁻³ costs about one random number, not 64.
Otherwise it is drawn by 64 comparisons.
- Composite channels (`DEPOLARIZE1`, `DEPOLARIZE2`, `PAULI_CHANNEL_1`) draw the word of lanes with
  any error, then a uniform or weighted Pauli for each set lane.
- The disjoint semantics are kept exactly, as `FrameSampler` keeps them.

**Structure.**
- The sampler walks the instruction tree and runs `REPEAT` bodies by iterating.
- Measurement records go into a ring buffer sized by the largest lookback any detector or
  observable uses. Detectors are evaluated when their instruction is reached, from lookbacks, so
  nothing is resolved to absolute indices.
- A `REPEAT 1000000` block therefore costs its body's size in memory.
- Detection events come out in order, and a callback receives each detector's word as it is
  evaluated. That is how streams are built.

**Outputs:**
- `sample_words`: detector and observable words for a batch of 64;
- `sample_b8(shots)`: Stim's b8 rows, by transposing 64 × 64 bit blocks;
- a stream mode for C2.

**Verification:**
1. **Exact agreement with `FrameSampler` on deterministic noise.** On random circuits whose
   channels are `X_ERROR`, `Y_ERROR` and `Z_ERROR` at p = 0 or 1, and `M(p)` at 0 or 1, every lane
   of every batch equals `FrameSampler`'s single shot, detector for detector. The gauge
   randomisation cannot matter to deterministic detectors, and those are the only ones generated.
2. **Marginals.** Each detector's firing rate, over 200,000 shots, equals the error model's exact
   prediction within 5σ, on the existing marginals test and on SD6 memory circuits.
3. **Against Stim.** The cross-check's check 3 (the χ² of per-detector rates against Stim's sampler,
   and logical error rates) is extended to the batch sampler.
4. **REPEAT without flattening.** On circuits with nested `REPEAT` blocks, the unflattened run and
   the run of the flattened circuit give identical words from the same seed. The noise is drawn in
   the same order.
5. **Speed**, reported as measured: shots per second against `FrameSampler` and against Stim's
   sampler, at d = 3, 5, 7 SD6.

## The browser pool (`js/pool.js`)

- A `Pool` of N workers, where N = `navigator.hardwareConcurrency` capped at 8, each running the
  existing `worker.js` with its own engine instance. It has the same `call(op, payload, onProgress)`
  interface as `Compute`, plus `map(op, payloads, onProgress)`, which spreads independent jobs over
  the workers and returns their results in order.
- **Uses:**
  - the threshold sweep's points are spread across the pool;
  - the bench's stream is split across the pool, combining counts;
  - section 11's live panel runs its three distances in parallel;
  - Figure 8's decoding rows run in parallel.
- Every result is a sum of independent shots, so splitting changes nothing but wall-clock time.
  The seeds stay independent, since each instance is seeded from the platform CSPRNG.
- **The general path (SD6)** samples with the batch sampler through a new export, keeping the old
  sampler as the reference behind its own export.
- **Verification:**
  - Pool results are the same statistics as one worker's: a site test checks the combining; a
    headless check shows the sweep finishing with the same fit within its interval.
  - Speed-ups are measured and reported in the README and the page's vitals.

---

# Part 2 — real time (Plan C2)

## Layers and window graphs (`src/window.rs`)

- **Layers.** A detector's layer is its time coordinate (the last one in its coordinates, as
  Stim's and Google's models both write it). Distinct times, in order, become layers 0, 1, 2, ….
  A model without time coordinates is refused, since windows need them.
- **Window graph.** For the layers `[a, b)`, a `SparseGraph` over the detectors in those layers,
  built from the full model's merged edges:
  - an edge with both ends inside is kept;
  - an edge from inside to a later layer (≥ b) becomes a boundary edge of its inside end, kept
    apart from any real boundary edge (two half-edges to the boundary). It is the **future
    boundary**, which lets a defect wait for a partner not yet seen;
  - an edge from inside to an earlier layer (< a) becomes a **past boundary** edge in the parallel
    decoder's windows, and is dropped in the sliding decoder, whose past is committed and closed;
  - each window graph keeps, per local edge, its original edge's id, so a committed edge is the
    full model's edge.

  The correlated rules come with it, restricted to edges inside the window. A rule naming an edge
  outside the window is dropped.

## Committing (`src/window.rs`)

- A window decodes its defects to edges (pass one's edge set, or pass two's for correlated
  matching, traced on the reweighted graph).
- **An edge is committed if its earlier end lies in the commit region.** Committing XORs its
  observables into the shot's prediction. Its other end, if it lies beyond the commit region,
  toggles that detector's defect: the correction reached into the next region, and the next window
  must see it.
- **Sliding.** Windows `[s, s + C + B)` for s = 0, C, 2C, …. The last window takes everything left
  and has no future boundary.
- **Parallel.**
  - Layer-A windows `[s − B, s + C + B)` with both boundaries virtual. Their commit regions
    `[s, s + C)` are spaced by `C + 2B`, and they decode independently, on as many cores as there
    are.
  - Layer-B windows cover the gaps between A's commit regions, with the A commits' toggles as
    extra defects and no virtual boundaries, since both neighbours are committed. They too decode
    independently.
  - Edge cases: the first and last windows have the real boundary where the stream starts and
    ends.
- **Verification:**
  1. **Exactness in the limit.** A window as large as the stream gives global decoding's
     prediction on every shot, for both decoders, plain and correlated.
  2. **Accuracy.** With B = d, the logical error rate of sliding and of parallel decoding equals
     global decoding's within statistical error (paired shot by shot: a McNemar test on the
     disagreements). This holds on SD6 circuits at d = 3, 5, 7 and on Willow's recorded data at
     d = 3, 5, 7 up to 250 rounds, fitted to ε and Λ with the B2 fit. At B = d/2 the degradation
     is measured and reported.
  3. **Commit bookkeeping.** On random small graphs, the committed edges XOR to a correction whose
     syndrome is the shot's defects. Every defect is explained exactly once.

## Streaming, latency and throughput (`src/stream.rs`, `tools/realtime.py`)

- **Timing.** Each window's decode is timed natively, single-threaded, with the window's defects
  already in hand.
- **Virtual-time schedule.**
  - Rounds arrive every τ = 1.1 µs, so window w becomes available when its last round arrives.
  - K workers take available windows in order: sliding windows in sequence, parallel windows by
    layer, with a layer-B window waiting for its two A neighbours.
  - A window's completion is its start plus its measured decode time. Latency is completion minus
    the arrival of the window's last round. Backlog is whether completion times fall ever further
    behind arrivals.
  - Reported per (d, K): mean and 99th-percentile latency, and whether it keeps up (the backlog
    stays bounded), for K = 1, 2, 4, 8.
- **A real run.** The same stream decoded with K real threads, to confirm the schedule's
  throughput and to catch contention the virtual schedule cannot see.
- **Streams:**
  - Willow's recorded d = 3, 5, 7 experiments at 250 rounds (every shot a stream);
  - simulated SD6 at d = 3, 5, 7, with p set so that the detection-event rate matches Willow's;
  - **a million-round** d = 5 stream from the batch sampler's stream mode.
- **Long streams and the window graph.** A million-round model is never built. The window graph
  for a bulk window comes from the model of a short circuit of the same code (enough rounds for the
  window to sit away from both time boundaries), and it is reused at every offset. That is correct
  because the circuit repeats: the bulk window graph at every offset is the same graph, which is
  verified by checking that the windows of a 60-round model at three different bulk offsets are
  identical, edge for edge, after shifting. The first and last windows come from the short model's
  own ends.

## Site: section 12, "Real time"

- **Recorded figure** (†): latency against cores at d = 3, 5, 7, with Willow's 1.1 µs budget and
  Google's published 63 µs at d = 5 marked, plus the million-round stream's result.
- **Live figure:** the pool samples a d = 5 SD6 stream and window-decodes it in chunks, showing:
  - rounds per second per worker and in total, against the 909,091 rounds per second that 1.1 µs
    demands;
  - the window latency histogram;
  - the logical error rate so far, against global decoding on the same rounds.

  The browser is slower than native and says so.

## Out of scope

- FPGA or ASIC decoders, and hardware latency beyond this machine's CPU.
- Decoding with leakage information, and neural-network decoders.
- Lattice surgery's multi-patch streams (E).

## Risks

1. **The window decoders' accuracy depends on getting the commit rule exactly right.** Layer 1
   (exactness in the limit) and layer 3 (bookkeeping) pin it before any accuracy number is quoted.
2. **Native latency depends on this machine.** The machine is named with every number, and the
   decode times are the raw material, so another machine can re-run the schedule.
3. **The browser pool multiplies memory by N instances.** Each is a few MB; N is capped at 8.
