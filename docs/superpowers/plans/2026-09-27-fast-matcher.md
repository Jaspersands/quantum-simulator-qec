# H — A Matcher as Fast as PyMatching: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** At every cross-check point, single-threaded, bring the sparse matcher's time per shot to at or under PyMatching's (plain and correlated), without changing one prediction.

**Architecture:** Profile-driven changes to the sparse matcher's hot path. A permanent benchmark fingerprints every prediction and weight, and every change must leave the fingerprints identical. Each performance change is kept only if it measures faster. The event queue becomes a radix heap that pops in exactly the binary heap's order. README tables are generated from data by a tool that CI checks.

**Tech Stack:** Rust (the crate, `examples/`), Python 3.13 (`.venv-f`: the engine by maturin, stim 1.16, pymatching 2.4), Node for the site tests, macOS `sample` for profiles.

## Global Constraints

- **Exactness first.** Every change must leave `tools/matcher_bench.py --check` passing: predictions and weights byte-identical to `data/matcher/bench.json` on every point, plain and correlated.
- **Keep only what measures faster.** A performance change is kept only if the benchmark's mean time at d = 11, p = 0.4% improves by at least 3% (plain or correlated, whichever it targets), with no point slower by more than 3%. Otherwise it is reverted, and the reason is recorded in the commit log of the revert.
- **Native timing.** On the recording machine the default Rust toolchain is x86_64 and runs under Rosetta. Every Rust timing uses `--target aarch64-apple-darwin`. Python timings are already native: maturin builds arm64.
- **Build the Python engine with the python profile:** `VIRTUAL_ENV=$PWD/.venv-f CARGO_TARGET_DIR=target-f .venv-f/bin/maturin develop --profile python`, never `--release`.
- **Honest outcome.** Out of scope, per the spec: bidirectional path search (the profile gives path tracing 7% of correlated time), changes to weight discretisation, and one-shot parallel matching. If the target is missed, the README and report state the ratio reached.

## Profile at the start (d = 11, p = 0.6%, arm64, `sample`, 6 s)

| function | plain | correlated |
|---|---|---|
| `next_node_event` | 63% | 60% |
| `run` (queue pops, dispatch) | 21% | 17% |
| `schedule_node` (queue pushes) | 6% | 5% |
| `trace_pairs` | — | 7% |

The rest is below 3% each. Native time per shot: 145 µs plain, 312 µs correlated. Under Rosetta it is 277 and 605.

---

### Task 1: The benchmark, the profiling binary, and a baseline

**Files:**
- Create: `tools/matcher_bench.py`
- Create: `examples/matcher_profile.rs` (replaces the throwaway copy)
- Create: `data/matcher/bench.json` (written by the tool)

**Interfaces:**
- Produces: `python tools/matcher_bench.py [--check] [--reps N] [--out PATH]`.
  - `--out` writes JSON: `{"generated", "engine_commit", "machine", "points": [{"d","p","shots","plain":{"ours_us","pm_us","ratio","digest"},"corr":{...}}]}`.
  - `--check` recomputes digests only (no PyMatching, one rep) and exits 1 on any difference from `data/matcher/bench.json`.
- Produces: `cargo run --release --no-default-features --target aarch64-apple-darwin --example matcher_profile -- profile D P plain|corr SECONDS`, and `... -- timing OUT.json`, which writes the native dense/sparse/correlated table for d = 3, 5, 7, 9 at p = 0.3% and 0.6%.

- [ ] **Step 1: Write `tools/matcher_bench.py`**

```python
"""The sparse matcher's speed against PyMatching, and a fingerprint of every
prediction and weight, so a faster matcher can be shown to decide nothing
differently.

    .venv-f/bin/python tools/matcher_bench.py            # time and fingerprint -> data/matcher/bench.json
    .venv-f/bin/python tools/matcher_bench.py --check    # fingerprints only, against the recorded ones

Rotated SD6 memories, T = d, shots sampled by Stim with a fixed seed. Times are
single-threaded, the best of --reps runs; ours is decode_b8's own clock (the
decode loop), PyMatching's is decode_batch's wall time.
"""
import argparse, hashlib, json, pathlib, sys, time

import numpy as np
import pymatching
import stim
import stabilizer_qec as sq

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from realtime import engine_commit, machine  # noqa: E402

OUT = ROOT / "data" / "matcher" / "bench.json"
POINTS = [(3, 0.003), (3, 0.006), (5, 0.003), (5, 0.006), (7, 0.003), (7, 0.006), (9, 0.004), (11, 0.004)]


def shots_for(d):
    return 100_000 if d <= 7 else 20_000


def point(d, p, reps, with_pm):
    shots = shots_for(d)
    text = sq.generate_circuit("rotated", d, d, "sd6", p, 0.5, "z")
    circuit = stim.Circuit(text)
    dem = circuit.detector_error_model(decompose_errors=True)
    dets, _ = circuit.compile_detector_sampler(seed=7).sample(shots, separate_observables=True)
    packed = np.packbits(dets, axis=1, bitorder="little").tobytes()
    row = dict(d=d, p=p, shots=shots)
    for key, corr in (("plain", False), ("corr", True)):
        best, digest = None, None
        for _ in range(reps):
            pred, w, _, seconds = sq.decode_b8(str(dem), packed, shots, 1, corr)
            h = hashlib.sha256(pred + w).hexdigest()[:16]
            if digest not in (None, h):
                raise SystemExit(f"d={d} p={p} {key}: two runs decided differently")
            digest, best = h, seconds if best is None else min(best, seconds)
        entry = dict(ours_us=best / shots * 1e6, digest=digest)
        if with_pm:
            m = pymatching.Matching.from_detector_error_model(dem, enable_correlations=corr)
            pm = min(_timed(lambda: m.decode_batch(dets, enable_correlations=corr)) for _ in range(reps))
            entry.update(pm_us=pm / shots * 1e6, ratio=best / pm)
        row[key] = entry
    return row


def _timed(fn):
    t = time.perf_counter()
    fn()
    return time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="fingerprints only, against data/matcher/bench.json")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", type=pathlib.Path, default=OUT)
    args = ap.parse_args()
    if args.check:
        recorded = {(r["d"], r["p"]): r for r in json.loads(OUT.read_text())["points"]}
        bad = 0
        for d, p in POINTS:
            r = point(d, p, 1, False)
            for key in ("plain", "corr"):
                same = r[key]["digest"] == recorded[(d, p)][key]["digest"]
                bad += not same
                print(f"{'ok ' if same else 'BAD'} d={d:<2} p={p} {key:<5} {r[key]['digest']}", flush=True)
        print("ALL FINGERPRINTS MATCH" if not bad else f"{bad} FINGERPRINTS DIFFER")
        return 1 if bad else 0
    points = []
    for d, p in POINTS:
        r = point(d, p, args.reps, True)
        points.append(r)
        print(f"d={d:<2} p={p}  plain {r['plain']['ours_us']:7.2f} us ({r['plain']['ratio']:.2f}x)  "
              f"corr {r['corr']['ours_us']:7.2f} us ({r['corr']['ratio']:.2f}x)", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    doc = dict(generated=time.strftime("%Y-%m-%d"), engine_commit=engine_commit(), machine=machine(),
               pymatching=pymatching.__version__, stim=stim.__version__, points=points)
    args.out.write_text(json.dumps(doc, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Write `examples/matcher_profile.rs`**

```rust
//! The sparse matcher, run natively for a profiler or a timing table.
//!
//! The recording machine's default Rust toolchain is x86_64 and runs under
//! Rosetta, which roughly halves the speed; always pass the native target:
//!
//!     cargo run --release --no-default-features --target aarch64-apple-darwin \
//!         --example matcher_profile -- profile 11 0.006 plain 12
//!     (then, in another shell: sample <pid> 6 -f profile.txt)
//!     cargo run ... --example matcher_profile -- timing data/matcher/native.json
use std::fmt::Write as _;

use stabilizer_qec::circuit::Basis;
use stabilizer_qec::dem::Dem;
use stabilizer_qec::dem_decoder::DemDecoder;
use stabilizer_qec::frame_sampler::FrameSampler;
use stabilizer_qec::memory::{generate, CodeKind, NoiseModel};
use stabilizer_qec::sparse::Scratch;
use stabilizer_qec::surface_code::Xorshift;

fn shots(d: usize, p: f64, n: usize) -> (DemDecoder, Vec<Vec<u32>>) {
    let c = generate(CodeKind::Rotated, d, d, NoiseModel::Sd6 { p }, Basis::Z).unwrap();
    let dec = DemDecoder::new(&Dem::from_circuit(&c).unwrap()).unwrap();
    let sampler = FrameSampler::new(&c).unwrap();
    let mut rng = Xorshift::new(3);
    let shots = (0..n)
        .map(|_| sampler.sample(&mut rng).detectors.iter().enumerate().filter(|x| *x.1).map(|x| x.0 as u32).collect())
        .collect();
    (dec, shots)
}

/// Mean microseconds a shot over the best of three passes.
fn time_us(shots: &[Vec<u32>], mut f: impl FnMut(&[u32])) -> f64 {
    (0..3)
        .map(|_| {
            let t = std::time::Instant::now();
            for s in shots {
                f(s);
            }
            t.elapsed().as_secs_f64() * 1e6 / shots.len() as f64
        })
        .fold(f64::INFINITY, f64::min)
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    match a.get(1).map(String::as_str) {
        Some("profile") => {
            let (d, p): (usize, f64) = (a[2].parse().unwrap(), a[3].parse().unwrap());
            let corr = a[4] == "corr";
            let secs: f64 = a[5].parse().unwrap();
            let (dec, shots) = shots(d, p, 4000);
            let mut s = Scratch::new(dec.graph());
            let t = std::time::Instant::now();
            let mut n = 0u64;
            while t.elapsed().as_secs_f64() < secs {
                for sh in &shots {
                    if corr {
                        dec.graph().decode_correlated(dec.correlations(), &mut s, sh).unwrap();
                    } else {
                        dec.graph().decode(&mut s, sh).unwrap();
                    }
                    n += 1;
                }
            }
            println!("{:.2} us/shot", t.elapsed().as_secs_f64() * 1e6 / n as f64);
        }
        Some("timing") => {
            let mut json = String::from("{\"points\": [\n");
            let mut first = true;
            for d in [3usize, 5, 7, 9] {
                for p in [0.003, 0.006] {
                    let (dec, shots) = shots(d, p, 2000);
                    let mut s = Scratch::new(dec.graph());
                    let sparse = time_us(&shots, |sh| { dec.graph().decode(&mut s, sh).unwrap(); });
                    let corr = time_us(&shots, |sh| { dec.graph().decode_correlated(dec.correlations(), &mut s, sh).unwrap(); });
                    let dense = time_us(&shots[..300], |sh| { let _ = dec.decode_dense(sh); });
                    println!("d = {d}, p = {p}: sparse {sparse:.2} us, correlated {corr:.2} us, dense {dense:.1} us");
                    let _ = write!(json, "{} {{\"d\": {d}, \"p\": {p}, \"sparse_us\": {sparse}, \"correlated_us\": {corr}, \"dense_us\": {dense}}}",
                        if first { "" } else { ",\n" });
                    first = false;
                }
            }
            json.push_str(&format!("\n], \"target\": \"{}\"}}\n", std::env::consts::ARCH));
            std::fs::write(&a[2], json).unwrap();
        }
        _ => eprintln!("usage: matcher_profile profile D P plain|corr SECONDS | timing OUT.json"),
    }
}
```

- [ ] **Step 3: Record the baseline**

Run:
```bash
VIRTUAL_ENV=$PWD/.venv-f CARGO_TARGET_DIR=target-f .venv-f/bin/maturin develop --profile python
.venv-f/bin/python tools/matcher_bench.py
.venv-f/bin/python tools/matcher_bench.py --check
cargo run --release --no-default-features --target aarch64-apple-darwin --example matcher_profile -- timing data/matcher/native.json
```
Expected:
- the first run prints eight points and writes `data/matcher/bench.json`;
- `--check` prints `ALL FINGERPRINTS MATCH`;
- `native.json` has `"target": "aarch64"`.

- [ ] **Step 4: Commit**

```bash
git add tools/matcher_bench.py examples/matcher_profile.rs data/matcher/bench.json data/matcher/native.json
git commit -m "feat(bench): the matcher benchmark with fingerprints, and a native profiling binary"
```

---

### Task 2: Defects found by scanning words, not bits

**Files:**
- Modify: `src/shots.rs` (add `defects_from_b8`)
- Modify: `src/py_api.rs:102-110` (`decode_packed`), `src/py_api.rs:562-568` (`decode_b8_belief`)
- Modify: `src/wasm_hw.rs:138`, `src/wasm_hw.rs:193`

**Interfaces:**
- Produces: `pub fn defects_from_b8(row: &[u8], num_bits: usize, out: &mut Vec<u32>)`, which appends the indices of set bits below `num_bits`, ascending.

- [ ] **Step 1: Write the failing test** (in `src/shots.rs`'s test module)

```rust
    #[test]
    fn defects_from_b8_equals_the_bitwise_scan() {
        let mut rng = crate::surface_code::Xorshift::new(9);
        for trial in 0..500 {
            let n = 1 + (rng.next_u64() % 300) as usize;
            let mut row: Vec<u8> = (0..n.div_ceil(8)).map(|_| rng.next_u64() as u8).collect();
            if trial % 2 == 0 {
                // Stray bits past num_bits must be ignored.
                if let Some(last) = row.last_mut() { *last |= 0x80; }
            }
            let slow: Vec<u32> = (0..n).filter(|&i| (row[i / 8] >> (i % 8)) & 1 == 1).map(|i| i as u32).collect();
            let mut fast = vec![7u32];
            defects_from_b8(&row, n, &mut fast);
            assert_eq!(&fast[1..], &slow[..], "n = {n}");
        }
    }
```

- [ ] **Step 2: Run it and see it fail**

Run: `cargo test --release --no-default-features defects_from_b8`
Expected: a compile error, `cannot find function defects_from_b8`.

- [ ] **Step 3: Implement**

```rust
/// Append the indices of `row`'s set bits below `num_bits`, ascending: a
/// b8 shot's detection events. Eight bytes at a time; bits past `num_bits`
/// are ignored.
pub fn defects_from_b8(row: &[u8], num_bits: usize, out: &mut Vec<u32>) {
    let bytes = num_bits.div_ceil(8).min(row.len());
    let mut base = 0usize;
    for chunk in row[..bytes].chunks(8) {
        let mut buf = [0u8; 8];
        buf[..chunk.len()].copy_from_slice(chunk);
        let mut word = u64::from_le_bytes(buf);
        let left = num_bits - base;
        if left < 64 {
            word &= (1u64 << left) - 1;
        }
        while word != 0 {
            out.push((base + word.trailing_zeros() as usize) as u32);
            word &= word - 1;
        }
        base += 64;
    }
}
```

Replace each bitwise loop with a call. In `decode_packed`:
```rust
                            defects.clear();
                            let row = &packed[s * stride..(s + 1) * stride];
                            crate::shots::defects_from_b8(row, nd, &mut defects);
```
Do the same in `decode_b8_belief`, and in `wasm_hw.rs` at both sites:
`defects.clear(); crate::shots::defects_from_b8(row, nd, &mut defects);`

- [ ] **Step 4: Run the tests and the fingerprints**

```bash
cargo test --release --no-default-features
VIRTUAL_ENV=$PWD/.venv-f CARGO_TARGET_DIR=target-f .venv-f/bin/maturin develop --profile python
.venv-f/bin/python tools/matcher_bench.py --check
```
Expected: all tests pass; `ALL FINGERPRINTS MATCH`.

- [ ] **Step 5: Measure** with `.venv-f/bin/python tools/matcher_bench.py --out /tmp/claude-bench-t2.json`, comparing to `data/matcher/bench.json`. Keep it if the Global Constraints' rule allows. The saving is largest at d = 3, where unpacking 24 bits is a real share of a 0.3 µs shot.

- [ ] **Step 6: Commit**

```bash
git add src/shots.rs src/py_api.rs src/wasm_hw.rs
git commit -m "perf: detection events unpacked a word at a time"
```

---

### Task 3: A tighter `next_node_event`

**Files:**
- Modify: `src/sparse/flooder.rs` (`next_node_event`, lines 157-205 today)

**Interfaces:** unchanged. The function returns exactly the same `(time, event)` for every state:
- the earliest time over v's edges;
- the first edge in adjacency order among equals, as `offer`'s strict `<` gives today.

- [ ] **Step 1: Rewrite the function**

```rust
    /// The next thing that happens across one of `v`'s edges, seen from `v`:
    /// the earliest, and among equals the first edge in adjacency order.
    pub(crate) fn next_node_event(&self, v: u32) -> Option<(i64, NodeEvent)> {
        let now = self.s.now;
        let nodes = &self.s.nodes;
        let regions = &self.s.regions;
        let nv = &nodes[v as usize];
        let v_owned = nv.top != NONE;
        let (lv, sv) = if v_owned {
            let r = &regions[nv.top as usize].radius;
            (r.at(now) + nv.wrapped, r.slope)
        } else {
            (0, 0)
        };
        let range = self.g.edges(v);
        let first = range.start;
        let to = &self.g.to[range.clone()];
        let w = &self.s.w[range];
        let mut best_t = i64::MAX;
        let mut best = None;
        for (k, (&u, &wt)) in to.iter().zip(w).enumerate() {
            let e = first + k;
            if u == BOUNDARY {
                if v_owned && sv > 0 {
                    let t = now + (wt - lv).max(0);
                    if t < best_t {
                        best_t = t;
                        best = Some(NodeEvent::Boundary { v, e });
                    }
                }
                continue;
            }
            let nu = &nodes[u as usize];
            if nu.top == NONE {
                if v_owned && sv > 0 {
                    let t = now + (wt - lv).max(0);
                    if t < best_t {
                        best_t = t;
                        best = Some(NodeEvent::Arrive { from: v, to: u, e });
                    }
                }
                continue;
            }
            // u is owned: one look at its top region serves both of its numbers.
            let ru = &regions[nu.top as usize].radius;
            let su = ru.slope;
            let lu = ru.at(now) + nu.wrapped;
            if !v_owned {
                if su > 0 {
                    let t = now + (wt - lu).max(0);
                    if t < best_t {
                        best_t = t;
                        best = Some(NodeEvent::Arrive { from: u, to: v, e });
                    }
                }
                continue;
            }
            if nu.top == nv.top {
                continue;
            }
            let rate = sv + su;
            if rate <= 0 {
                continue;
            }
            let gap = wt - lv - lu;
            debug_assert!(gap >= 0 && gap % rate == 0, "gap {gap} at rate {rate}");
            // Slopes are -1, 0 or 1, so a positive rate is 1 or 2.
            let g = gap.max(0);
            let t = now + if rate == 2 { g >> 1 } else { g };
            if t < best_t {
                best_t = t;
                best = Some(NodeEvent::Collide { v, u, e });
            }
        }
        best.map(|ev| (best_t, ev))
    }
```

- [ ] **Step 2: Run the tests.** `cargo test --release --no-default-features` passes, including `decode_checked`'s dual-feasibility checks in `sparse::tests`.
- [ ] **Step 3: Rebuild the engine and run `tools/matcher_bench.py --check`.** Expected: `ALL FINGERPRINTS MATCH`.
- [ ] **Step 4: Measure with `--out /tmp/claude-bench-t3.json`, and profile natively again** (`examples/matcher_profile` and `sample`). Keep or revert by the rule; the new share of `next_node_event` goes in the commit message.
- [ ] **Step 5: Commit** with `perf(sparse): next_node_event looks at a neighbour's region once, without bounds checks or division`.

---

### Task 4: A radix-heap event queue with the binary heap's exact pop order

**Files:**
- Modify: `src/sparse/tracker.rs` (the whole file)

**Interfaces:**
- Consumes: `Item`, `push(t: i64, item: Item)`, `pop() -> Option<(i64, Item)>`, `clear()`, as today.
- Produces: the same interface, `Tracker::default()` included. The behaviour is identical for monotone use: every push's time is at least the last popped time, which the flooder guarantees. Pops come in increasing `(time, Item)` order, exactly as `BinaryHeap<Reverse<(i64, Item)>>`'s.

- [ ] **Step 1: Write the failing test** (a new `#[cfg(test)] mod tests` in `tracker.rs`)

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::surface_code::Xorshift;

    /// Monotone use, as the flooder's: every push is at or after the last pop.
    #[test]
    fn pops_in_exactly_the_binary_heaps_order() {
        let mut rng = Xorshift::new(11);
        for trial in 0..300 {
            let mut radix = Tracker::default();
            let mut heap = std::collections::BinaryHeap::new();
            let mut now = 0i64;
            for _ in 0..2000 {
                if rng.next_u64() % 3 != 0 || heap.is_empty() {
                    // Bursts at the current time, and jumps of every scale.
                    let dt = match rng.next_u64() % 4 {
                        0 => 0,
                        1 => (rng.next_u64() % 8) as i64,
                        2 => (rng.next_u64() % 100_000) as i64,
                        _ => (rng.next_u64() % (1 << 40)) as i64,
                    };
                    let item = if rng.next_u64() % 2 == 0 {
                        Item::Node((rng.next_u64() % 50) as u32)
                    } else {
                        Item::Region((rng.next_u64() % 50) as u32)
                    };
                    radix.push(now + dt, item);
                    heap.push(std::cmp::Reverse((now + dt, item)));
                } else {
                    let a = radix.pop();
                    let b = heap.pop().map(|std::cmp::Reverse(x)| x);
                    assert_eq!(a, b, "trial {trial}");
                    now = a.unwrap().0;
                }
            }
            while let Some(std::cmp::Reverse(x)) = heap.pop() {
                assert_eq!(radix.pop(), Some(x), "trial {trial}, draining");
            }
            assert_eq!(radix.pop(), None);
        }
    }

    #[test]
    fn clear_empties_and_restarts_time() {
        let mut t = Tracker::default();
        t.push(5, Item::Node(1));
        t.push(9, Item::Region(2));
        assert_eq!(t.pop(), Some((5, Item::Node(1))));
        t.clear();
        assert_eq!(t.pop(), None);
        t.push(0, Item::Node(3));
        assert_eq!(t.pop(), Some((0, Item::Node(3))));
    }
}
```

- [ ] **Step 2: Run the tests against today's binary-heap tracker.** They pass, and pin the order the radix heap must keep:
`cargo test --release --no-default-features tracker`
Expected: PASS. The first test is the contract.

- [ ] **Step 3: Replace the implementation**

```rust
//! The event queue: reminders to look at a node or a region at a given time.
//!
//! Only the earliest reminder per entity is queued, and a reminder made stale
//! by a change of growth rate is not removed: it fires, the flooder recomputes,
//! and finds nothing due. That is the paper's tracker, and it keeps the queue
//! small.
//!
//! Times only increase: a reminder is never set before the last one taken. So
//! the queue is a radix heap. Bucket b ≥ 1 holds reminders whose time first
//! differs from the last time taken in bit b − 1, and taking the next time
//! empties only the lowest non-empty bucket into the ones below it. Pushes cost
//! O(1), and each reminder moves down at most 64 times. Reminders at the
//! current time wait in a small binary heap ordered by item, so they come out in
//! exactly the order `BinaryHeap<Reverse<(i64, Item)>>` gave, and a decode
//! takes the same steps as before.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Item {
    Node(u32),
    Region(u32),
}

pub(crate) struct Tracker {
    /// The last time taken; every queued time is at least this.
    last: i64,
    /// Reminders at time `last`, least item first.
    current: BinaryHeap<Reverse<Item>>,
    /// Bucket b holds times whose highest bit differing from `last` is b − 1.
    buckets: Vec<Vec<(i64, Item)>>,
    /// Which buckets hold anything, one bit per bucket.
    occupied: u64,
}

impl Default for Tracker {
    fn default() -> Self {
        Tracker { last: 0, current: BinaryHeap::new(), buckets: (0..64).map(|_| Vec::new()).collect(), occupied: 0 }
    }
}

impl Tracker {
    fn bucket(&self, t: i64) -> usize {
        64 - ((t ^ self.last) as u64).leading_zeros() as usize
    }

    pub fn clear(&mut self) {
        self.last = 0;
        self.current.clear();
        while self.occupied != 0 {
            let b = self.occupied.trailing_zeros() as usize;
            self.buckets[b].clear();
            self.occupied &= self.occupied - 1;
        }
    }

    pub fn push(&mut self, t: i64, item: Item) {
        debug_assert!(t >= self.last, "a reminder at {t} before the last time taken, {}", self.last);
        if t == self.last {
            self.current.push(Reverse(item));
        } else {
            let b = self.bucket(t);
            self.buckets[b].push((t, item));
            self.occupied |= 1 << b;
        }
    }

    pub fn pop(&mut self) -> Option<(i64, Item)> {
        if self.current.is_empty() {
            if self.occupied == 0 {
                return None;
            }
            // The lowest non-empty bucket holds the next time; empty it below.
            let b = self.occupied.trailing_zeros() as usize;
            self.occupied &= !(1 << b);
            let mut moving = std::mem::take(&mut self.buckets[b]);
            self.last = moving.iter().map(|x| x.0).min().expect("an occupied bucket is not empty");
            for &(t, item) in &moving {
                if t == self.last {
                    self.current.push(Reverse(item));
                } else {
                    let nb = self.bucket(t);
                    self.buckets[nb].push((t, item));
                    self.occupied |= 1 << nb;
                }
            }
            moving.clear();
            self.buckets[b] = moving;
        }
        self.current.pop().map(|Reverse(item)| (self.last, item))
    }
}
```

- [ ] **Step 4: Run the tracker tests, then the whole suite**

`cargo test --release --no-default-features` passes. The contract test passes against the new implementation.

- [ ] **Step 5: Fingerprints and measurement.** Run `--check` (expect `ALL FINGERPRINTS MATCH`), then measure. Keep or revert by the rule.
- [ ] **Step 6: Commit** with `perf(sparse): a radix-heap event queue, popping in the binary heap's exact order`.

---

### Task 5: Node state that `next_node_event` reads, packed tighter

**Files:**
- Modify: `src/sparse/state.rs` (`NodeState`, `Scratch`)
- Modify: `src/sparse/flooder.rs` (`reset`, `touch`, `schedule_node`, `run`, `leave`)

`NodeState` is 48 bytes today, and the queue's `queued: i64` and `dirty: bool` sit in it though `next_node_event` never reads them. Moving them into their own arrays in `Scratch` brings a node to 32 bytes, two to a cache line.

**Interfaces:**
- Produces: `Scratch.queued: Vec<i64>` and `Scratch.dirty: Vec<bool>`, one per node. `NodeState` loses its `queued` and `dirty` fields.

- [ ] **Step 1: Change the types**

In `state.rs`:
- remove `pub queued: i64,` and `pub dirty: bool,` from `NodeState` and from `NodeState::EMPTY`;
- add to `Scratch`:
```rust
    /// Per node: the time of its queued reminder (NO_TIME if none), and
    /// whether this decode has touched it. Kept apart from `nodes`, which the
    /// hot loop reads.
    pub(crate) queued: Vec<i64>,
    pub(crate) dirty: Vec<bool>,
```
- initialise them in `Scratch::new` with `queued: vec![NO_TIME; graph.num_nodes]` and `dirty: vec![false; graph.num_nodes]`.

In `flooder.rs`:
```rust
    // reset(): for each touched node
            self.s.nodes[v as usize] = NodeState::EMPTY;
            self.s.queued[v as usize] = NO_TIME;
            self.s.dirty[v as usize] = false;
    // touch()
        if !self.s.dirty[v as usize] {
            self.s.dirty[v as usize] = true;
            self.s.touched.push(v);
        }
    // schedule_node()
        if t < self.s.queued[v as usize] {
            self.touch(v);
            self.s.queued[v as usize] = t;
            self.s.queue.push(t, Item::Node(v));
        }
    // run(): Item::Node(v) branch
                    if self.s.queued[v as usize] != t {
                        continue;
                    }
                    self.s.queued[v as usize] = NO_TIME;
    // leave(): keeps the node's own region; its reminder and touch flag
    // live apart now and are left as they are
        let n = self.s.nodes[v as usize];
        self.s.nodes[v as usize] = NodeState { own: n.own, ..NodeState::EMPTY };
```
Fix any remaining uses the compiler reports. `extract.rs`, `paths.rs` and `matcher.rs` do not read either field.

- [ ] **Step 2: Test and fingerprint.** Run `cargo test --release --no-default-features` and `tools/matcher_bench.py --check`: all pass, and `ALL FINGERPRINTS MATCH`.
- [ ] **Step 3: Measure** at the benchmark's points and at Willow's size, `matcher_profile profile 7 0.003 corr 10` on a 250-round circuit. Keep or revert by the rule.
- [ ] **Step 4: Commit** with `perf(sparse): the queue's per-node fields apart from the node state the hot loop reads`.

---

### Task 6: README tables generated from the data, checked in CI

The README's tables were typed by hand, and three fell behind their data this week. This task makes them generated.

**Files:**
- Create: `tools/readme_tables.py`
- Modify: `.github/workflows/ci.yml` (one step in the site job)

**Interfaces:**
- Produces: `python3 tools/readme_tables.py [--write | --check]`. The standard library only. It regenerates the bodies of six README tables, each found by its header row:

| table | from |
|---|---|
| the cross-check's decoding table | `data/xcheck/reference.json` |
| the correlated table | `data/xcheck/reference.json` |
| the matcher speed table | `data/matcher/native.json`, with PyMatching's column from `reference.json` |
| the latency table | `data/realtime/latency.json` and `million.json` |
| the gross code table | `data/gross/results.json` |
| the gross code beside the surface code | `data/gross/results.json` |

  `--check` exits 1 and names each table whose README body differs.

- [ ] **Step 1: Write the tool.** It carries over the three generators validated this week, each of which reproduced the README from committed data before replacing it:

```python
"""The README's tables, generated from the committed data, so none can fall
behind what was measured.

    python3 tools/readme_tables.py            # print every table
    python3 tools/readme_tables.py --write    # replace them in README.md
    python3 tools/readme_tables.py --check    # exit 1 if any differs (CI)
"""
import json, math, pathlib, re, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
load = lambda path: json.loads((ROOT / path).read_text())


def sci(x):
    e = math.floor(math.log10(x)); m = x / 10 ** e
    if round(m, 1) >= 10:
        m, e = m / 10, e + 1
    return f"{m:.1f} × 10{str(e).replace('-', '⁻').translate(str.maketrans('0123456789', '⁰¹²³⁴⁵⁶⁷⁸⁹'))}"


def pct(k, n):
    return f"{k / n * 100:.3f}%"


def time_us(x):
    if x < 100:
        return f"{x:.1f} µs"
    if x < 1000:
        return f"{x:.0f} µs"
    return f"{x / 1000:.2f} ms" if x < 10_000 else f"{x / 1000:.1f} ms"


def gross():
    g = load("data/gross/results.json")
    rows = []
    for code, name in (("gross", "gross [[144, 12, 12]]"), ("72", "[[72, 12, 6]]")):
        for p in (0.002, 0.003, 0.004, 0.005, 0.006):
            q = g["points"][f"{code}/{p}"]; cs = q["results"]["bposd_cs7"]; o0 = q["results"]["bposd_0"]
            lo, hi = cs["pl_cycle_interval"]
            rows.append(f"| {name} | {p * 100:.1f}% | {q['shots']:,} | {cs['failures']} | {sci(cs['pl_cycle'])} "
                        f"[{sci(lo)}, {sci(hi)}] | {sci(o0['pl_cycle'])} | {cs['converged'] / q['shots'] * 100:.1f}% |")
    return rows


def gross_vs_surface():
    g = load("data/gross/results.json")
    rows = []
    for p in (0.002, 0.003, 0.004, 0.005, 0.006):
        gp = g["points"][f"gross/{p}"]["results"]["bposd_cs7"]["pl_cycle"]
        sp = g["surface"][f"surface-d11/{p}"]["results"]["correlated_matching"]["pl_cycle"]
        twelve = 1 - (1 - sp) ** 12
        rows.append(f"| {p * 100:.1f}% | {sci(gp)} | {sci(twelve)} | {twelve / gp:.1f}× |")
    return rows


def xcheck_plain():
    rows = []
    for r in load("data/xcheck/reference.json")["decoding"]:
        n = r["shots"]
        rows.append(f"| {r['d']} | {r['p'] * 100:.1f}% | {pct(r['pymatching_failures'], n)} | {pct(r['ours_failures'], n)} | "
                    f"{pct(r['ours_own_dem_failures'], n)} | {r['disagreements']} ({r['non_ties']}) | "
                    f"{pct(r['sampler']['our_failures'], n)} | {r['sampler']['z']:+.2f} | {r['pymatching_us']:.1f} µs | {r['ours_us']:.1f} µs |")
    return rows


def xcheck_corr():
    rows = []
    for r in load("data/xcheck/reference.json")["correlated"]:
        n = r["shots"]
        code = "rotated" if r["code"] == "rotated" else "XZZX"
        rows.append(f"| {code} | {r['d']} | {r['p'] * 100:.1f}% | {pct(r['pymatching_failures'], n)} | {pct(r['ours_failures'], n)} | "
                    f"{pct(r['plain_failures'], n)} | {r['disagreements']} | {r['pymatching_us']:.1f} | {r['ours_us']:.1f} |")
    return rows


def speed():
    native = load("data/matcher/native.json")["points"]
    pm = {(r["d"], r["p"]): r["pymatching_us"] for r in load("data/xcheck/reference.json")["decoding"]}
    rows = []
    for r in native:
        m = pm.get((r["d"], r["p"]))
        rows.append(f"| {r['d']} | {r['p'] * 100:.1f}% | {time_us(r['dense_us'])} | {time_us(r['sparse_us'])} | "
                    f"{time_us(m) if m is not None else '—'} |")
    return rows


def latency():
    lat, mil = load("data/realtime/latency.json"), load("data/realtime/million.json")
    us = lambda x: f"{x:.1f} µs" if x < 10 else f"{x:.0f} µs"
    whole = lambda x: f"{x:.0f} µs"
    cores = lambda k: f"{k} core" + ("" if k == 1 else "s")

    def first_k(by):
        return min(int(k) for k, v in by.items() if v["keeps_up"])

    rows = []
    for d, mode, m in ((3, "sliding", "plain"), (3, "parallel", "correlated"), (5, "parallel", "plain"),
                       (5, "parallel", "correlated"), (7, "parallel", "plain"), (7, "parallel", "correlated")):
        s = next(x for x in lat["streams"] if x["d"] == d and x["mode"] == mode and x["matcher"] == m)
        k = first_k(s["by_workers"]); w = s["by_workers"][str(k)]
        rows.append(f"| Willow d = {d} | {mode}, {m} | {us(s['window_us']['mean'])} | {cores(k)} | {whole(w['mean_us'])} | {whole(w['p99_us'])} |")
    for m, k2 in (("plain", 4), ("correlated", 8)):
        r = next(x for x in mil["runs"] if x["mode"] == "parallel" and x["matcher"] == m)
        k = first_k(r["by_workers"]); w = r["by_workers"][str(k)]; w2 = r["by_workers"][str(k2)]
        if k2 > k and w2["mean_us"] < 0.8 * w["mean_us"]:
            rows.append(f"| SD6 d = 5, 10⁶ rounds | parallel, {m} | {us(r['window_us']['mean'])} | {cores(k)} ({k2} for {whole(w2['mean_us'])}) | "
                        f"{whole(w['mean_us'])} ({whole(w2['mean_us'])} on {k2}) | {whole(w['p99_us'])} ({whole(w2['p99_us'])} on {k2}) |")
        else:
            rows.append(f"| SD6 d = 5, 10⁶ rounds | parallel, {m} | {us(r['window_us']['mean'])} | {cores(k)} | {whole(w['mean_us'])} | {whole(w['p99_us'])} |")
    return rows


# Each table's header row (as a regex), and the function giving its body.
TABLES = [
    ("gross", r"\| code \| p \| shots \| failures \| per cycle, BP\+OSD-CS \[95%\] \| per cycle, BP\+OSD-0 \| BP alone \|", gross),
    ("gross beside the surface code", r"\| p \| gross code: 12 logical qubits on 288 \| twelve d = 11 surface patches on 2,892 \| ratio \|", gross_vs_surface),
    ("cross-check, plain", r"\| d \| p \| PyMatching \| ours, on Stim's graph \|[^\n]*", xcheck_plain),
    ("cross-check, correlated", r"\| code \| d \| p \| PyMatching correlated \|[^\n]*", xcheck_corr),
    ("matcher speed", r"\| d \| p \| dense \| sparse \| PyMatching \|", speed),
    ("latency", r"\| stream \| decoder \| window decode \| keeps up on \| mean latency \| p99 \|", latency),
]


def main():
    readme = (ROOT / "README.md").read_text()
    out, stale = readme, []
    for name, header, fn in TABLES:
        body = "\n".join(fn()) + "\n"
        pattern = re.compile("(" + header + r"\n\|---(?:\|---)*\|\n)((?:\|.*\|\n)+)")
        m = pattern.search(out)
        if not m:
            raise SystemExit(f"{name}: header not found in README.md")
        if m.group(2) != body:
            stale.append(name)
        out = out[:m.start(2)] + body + out[m.end(2):]
        if "--write" not in sys.argv and "--check" not in sys.argv:
            print(f"## {name}\n{body}")
    if "--check" in sys.argv:
        for name in stale:
            print(f"README table out of date: {name}")
        return 1 if stale else 0
    if "--write" in sys.argv:
        (ROOT / "README.md").write_text(out)
        print(f"rewrote {len(stale)} table(s): {', '.join(stale) or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```
- [ ] **Step 2: `python3 tools/readme_tables.py --check` against today's README.** It must report every table as up to date except the speed table. That table changes only when `native.json` replaces the Rosetta numbers, in Task 7.
- [ ] **Step 3: Add to CI**, in the site job after the site tests:
```yaml
      - name: README tables match their data
        run: python3 tools/readme_tables.py --check
```
- [ ] **Step 4: Commit** with `feat(tools): README tables generated from the data, and checked in CI`.

---

### Task 7: Measure, publish, and say "native" only where it is true

**Files:**
- Modify: `data/matcher/bench.json`, `data/matcher/native.json`, `data/xcheck/reference.json`, `data/realtime/{latency,million}.json` (regenerated)
- Modify: `README.md`, `report/report.md`, `tools/report.py` (the speed sentences), `stabilizer_qec.wasm`, `report/*`

- [ ] **Step 1: Regenerate the measurements, one at a time on a quiet machine:**
```bash
VIRTUAL_ENV=$PWD/.venv-f CARGO_TARGET_DIR=target-f .venv-f/bin/maturin develop --profile python
.venv-f/bin/python tools/matcher_bench.py
.venv-f/bin/python tools/xcheck.py
.venv-f/bin/python tools/realtime.py latency
.venv-f/bin/python tools/realtime.py million
cargo run --release --no-default-features --target aarch64-apple-darwin --example matcher_profile -- timing data/matcher/native.json
cargo build --release --target wasm32-unknown-unknown --no-default-features && cp target/wasm32-unknown-unknown/release/stabilizer_qec.wasm .
node tools/wasm-smoke.mjs stabilizer_qec.wasm
```
Expected:
- the cross-check prints `ALL CHECKS PASSED`, with the same failure counts and ties as before;
- the WebAssembly smoke test prints `all checks passed`.

- [ ] **Step 2: `python3 tools/readme_tables.py --write`,** then update the prose from the printed values:
  - the ratios: "N to M times PyMatching's" in the features list, the Speed paragraph, and the correlated paragraph;
  - d = 7's per-shot time;
  - the all-core rate, measured as in this week's run;
  - the d = 9 "plain and correlated" sentence;
  - the throughput and latency bullets.

  Replace "Native, single-threaded" and "Natively, from `sparse::tests::timing`" with the example's command. Add one sentence saying the earlier table ran under Rosetta, and was about half the native speed.
- [ ] **Step 3: If the target (at or under 1.0 at every point) is met,** the features line says so. If it is not, it gives the range reached, and the report's "Limits" bullet keeps it honest.
- [ ] **Step 4: Rebuild the report after committing the data, and re-measure the browser figure.** The browser figure is Figure 12's window decoding: three headless runs, as in this week's run.
- [ ] **Step 5: Full verification:**
  - `cargo test --release --no-default-features`, plus the ignored d = 7 equivalence tests;
  - `tools/smoke.py` and `tools/xcheck.py --quick`;
  - `node tools/site-tests.mjs`, `tools/contrast.mjs` and `tools/readme_tables.py --check`;
  - a headless full-page check with no console errors.
- [ ] **Step 6: Commit, review at high effort, fix, merge to master, push, and check CI.**
