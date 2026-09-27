# H — A matcher as fast as PyMatching

**Date:** 2026-09-27
**Status:** Proposed. Questions answered under Jasper's standing instruction ("do both or the
biggest"), marked where they were; for his review before a plan is written.

## Why

The sparse matcher is exact and agrees with PyMatching shot for shot but for ties. Speed is the
one place it still trails: single-threaded, plain matching takes 1.3 to 1.7 times PyMatching's
time per shot, and correlated matching 1.0 to 1.5 times. That comes after removing the per-event
allocations and quadratic scans, which bought 22 to 49%. For a company reading the page, "as fast
as the reference implementation, and checked against it" is a different claim from "within 2x".
It also moves every number downstream: the real-time section's cores-to-keep-up, the site's
sweeps, and the estimator's decoding cores.

## Goal

At every cross-check point (rotated and XZZX, SD6, d = 3, 5, 7, p = 0.3% and 0.6%, T = d), single
threaded:
- plain matching at or under PyMatching's time per shot;
- correlated matching at or under PyMatching's correlated mode.

This is to be reached without giving up exactness. If it is not reached, the page reports the ratio
it did reach. The target is not quietly moved.

## What is known about the gap

Sparse blossom (Higgott and Gidney, arXiv:2303.15933) is the algorithm both implementations run.
The differences are engineering, and the paper names the ones that matter:

1. **The event queue.** PyMatching's tracker is a radix heap over 32-bit cyclic times. Event times
   only increase, so a radix heap pushes and pops in amortised constant time. Ours is a binary
   heap of `(i64, Item)`, costing a logarithmic time per operation on 16-byte entries.
2. **State size.** Our detector node is about 56 bytes (i64 times and radii, u64 observables, and
   a u32 index for each of region, top, source, own). PyMatching's hot fields are smaller and sit
   together, so a region's growth touches fewer cache lines.
3. **Correlated matching's first pass.** It traces one shortest path per matched pair by one-sided
   Dijkstra from the pair's first defect. A search from both ends meets in about half the explored
   nodes.
4. **Unpacking shots.** `decode_b8` finds each shot's defects by testing every detector bit. At
   d = 11 that is 1,320 tests a shot; scanning words with `trailing_zeros` touches only the set
   bits.

The ranking above is a guess. Profiling decides which of these matter, before any of them is built.

## Design

### Components

- **`tools/matcher_bench.py`** (new, permanent). The benchmark this session used, made a tool:
  - rotated SD6 d = 3 to 11 and the cross-check's twelve points, single-threaded, against
    PyMatching on the same shots;
  - it writes `data/matcher/bench.json`: the times, and a fingerprint of every shot's integer
    matching weight;
  - `--check` compares the fingerprints to the recorded ones, fast enough for CI.
- **Profiling first.** One profile per regime (a small patch at low p, and d = 11 at p = 0.6%),
  with the macOS sampler on the Rust benchmark binary. The profile ranks the four items above. Each
  item is then built only if the profile says it is worth building, and measured after it lands.
- **`sparse/tracker.rs`.** A radix heap keyed by the event time, with the same `push`, `pop` and
  `clear` interface. Where the profile shows it pays, events at the same time are processed in a
  fixed order, so a decode stays deterministic.
- **`sparse/state.rs`.**
  - A compact node: u32 indices kept, with times and radii in the width their range needs. The
    range is checked where weights are built, and a model whose path weights would overflow is
    refused, not wrapped.
  - The fields `next_node_event` reads are grouped together.
- **`sparse/paths.rs`.** Bidirectional shortest paths for tracing matched pairs. The traced set is
  still a minimum-weight correction; only which of several shortest paths is traced may change.
- **`py_api.rs` and `wasm_*.rs`.** Defects found by word scanning.

### What "exact" means once tie order can change

A different queue order can find a different minimum-weight matching where two tie. So bit-identical
predictions are not the criterion. The criterion is that **every shot's optimal integer weight equals
the recorded one**. The optimum is unique even when the matching is not. On top of that, every
existing oracle keeps passing:
- the dense matcher's exact weights;
- brute force on small graphs;
- the dual-feasibility check after every event (`decode_checked`);
- the cross-check against PyMatching, where every disagreement must still be a tie.

## Decisions

| Question | Decision |
|---|---|
| Weight discretisation | **Keep ours** (2²⁰ per unit of ln((1−p)/p), even integers). PyMatching's coarser discretisation would make the two integer problems identical, but it would move our weights away from the true ones. Exactness comes first. |
| Parallel decoding of one shot (fusion blossom, arXiv:2305.08307) | **Out of scope here.** Window decoding already spreads one stream over cores. A parallel blossom for a single shot is its own project. |
| How far to go | **Biggest:** all four items, if profiling supports them, and the WebAssembly engine gets the same code, so the site speeds up too. |

## Verification

1. Every shot's integer weight equals the recorded fingerprint, over the benchmark corpus: about
   10⁶ shots, d = 3 to 11, plain and correlated.
2. The full cross-check (`tools/xcheck.py`, 100,000 shots a point) passes: models equal Stim's,
   and every disagreement with PyMatching is a tie.
3. The Rust suite, including the ignored d = 7 equivalence tests and the timing test.
4. The WebAssembly smoke test gives the same failure counts on Google's Willow data.

## Outputs

- README speed section and the report's speed figure, from the benchmark's data.
- The real-time data re-measured (`tools/realtime.py latency` and `million`), since the
  cores-to-keep-up follow the decode time. The estimator's cores follow from that.

## Out of scope

Approximate decoders (Union-Find, neural, pre-matching heuristics): the point is an exact matcher
that is also fast.
