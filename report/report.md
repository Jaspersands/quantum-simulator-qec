---
title: "A surface-code simulator and decoder, checked against Stim, PyMatching and Google's hardware"
subtitle: "Exact and correlated matching, Google's Willow and Sycamore data, and window decoding at Willow's cycle"
author: Jasper Sands
date: "{{date}}"
version: "{{version}}"
commit: "{{commit}}"
description: "Technical report on stabilizer-qec: a Rust surface-code simulator and decoder, verified against Stim and PyMatching, run on Google's Willow and Sycamore data, and decoded in real time."
abstract: |
  `stabilizer-qec` is a surface-code simulator and decoder written in Rust. It runs natively, from
  Python, and in a web browser. Its results come in five parts.

  1. **Error models.** It builds error models identical to Stim's, fault by fault, on
     {{xc.circuits}} circuits (the largest relative difference is {{xc.max_rel}}).
  2. **Matching.** Its exact matcher agrees with PyMatching on every shot but ties:
     {{xc.plain_disagree}} disagreements in {{xc.shots}} shots, {{xc.plain_non_ties}} of them not
     ties. Its correlated matcher agrees with PyMatching 2.4's correlated mode just as closely, and
     fails {{xc.corr_gain}} less often than plain matching.
  3. **Google's hardware.** On Google's own recordings ({{g.experiments}} experiments,
     {{g.shots}} shots) it rebuilds the detection events bit for bit. Its correlated matcher,
     using Google's SI1000 prior on Willow, reaches Λ = {{w.corr.lambda}}. Google's own correlated
     matcher on the same prior reaches {{w.gcorr.l}}.
  4. **Throughput.** Its bit-parallel sampler takes {{speed.batch_d7}} a shot at d = 7, against
     Stim's {{speed.stim_d7}}.
  5. **Real time.** Parallel windows decode Willow's recorded d = 5 syndromes with correlated
     matching and keep up with its {{rt.cycle}} cycle on {{rt.d5.cores}} cores. A simulated
     million-round stream at the same detection rate holds a {{m.corr.mean8}} mean latency on 8
     cores. Windowed decoding reaches the same Λ as global decoding ({{win.corr.lambda}}).
---

# Introduction

A surface code protects a logical qubit by measuring checks on many physical qubits, round after
round, and handing the outcomes to a decoder. The decoder infers which errors most likely occurred
and whether they flipped the logical qubit. Two questions decide whether it is any good:

- **Is it correct?** Is its error model the circuit's, and is its matching exact?
- **Is it fast enough?** A quantum computer produces a round every microsecond or so, and a decoder
  that falls behind never catches up.

This report describes an engine built to answer both. It covers how the engine works, how it is
checked, and what it measures. Every number below is read from a file committed with the source,
and the build fails if one is missing. The same engine drives an interactive explainer that runs
it in the reader's browser, at [qcompiler.jaspersands.com](https://qcompiler.jaspersands.com).

The work is organised around three references the field already trusts:

- **Stim** (Gidney, 2021) for circuits, error models and sampling;
- **PyMatching 2** (Higgott and Gidney, 2023) for matching;
- **Google's published recordings** from Willow (2024) and Sycamore (2022) for real hardware.

Where this engine and a reference can be made to compute the same thing, they are compared exactly.
Where they cannot, the comparison is statistical, with its uncertainty stated.

# The engine

## Circuits and error models

The engine reads and writes the part of Stim's circuit language that memory experiments use:

- resets, `H`, `CX` and `CZ`, and measurements;
- depolarizing and Pauli channels;
- detectors and observables with their coordinates, `SHIFT_COORDS`, and `REPEAT` blocks;
- for Google's circuits, sweep-controlled `CX` and the Pauli gates.

Two generators write rotated and XZZX memory experiments. One uses the engine's own circuit-level
noise; the other uses SD6, the standard model in which published thresholds are quoted.

A **detector error model** lists every independent fault and the detectors and observables it
flips. The engine builds one by walking the circuit backwards once. For each qubit it carries the
set of detectors that an X or a Z error there would flip, as Stim's error analyzer does. Matching
needs graph-like faults, so a fault that sets off more than two detectors is split into pieces.
The engine splits them the way Stim does: by what the rest of the same noise channel can express,
then by Stim's global pass. The first, simpler split looked right and was not (section 3).

## Sampling

There are two samplers:

- **The frame sampler** propagates a Pauli frame through the circuit one shot at a time. It never
  looks at the error model, so the model and the sampler can disagree, and the checks can see it.
- **The batch sampler** keeps 64 shots in each machine word: two words per qubit, one for X and
  one for Z. It draws noise as whole words by geometric skipping, so a location at $p = 10^{-3}$
  costs about one random number, not 64. It runs `REPEAT` blocks without flattening them, with
  measurement records in a ring buffer sized by the longest lookback. A million-round memory costs
  one round's instructions, and its detection events stream out as they are made.

## Matching

Each graph edge is weighted $\ln((1-p)/p)$, discretised to even integers, and decoded by **sparse
blossom**, the algorithm PyMatching 2 is built on (Higgott and Gidney, arXiv:2303.15933):

- Every defect grows a region on the detector graph at the same rate, and a region's radius is its
  dual variable.
- Collisions between regions are found by the growth itself.
- Edmonds' alternating trees and blossoms act on regions rather than on vertices.

The earlier dense matcher (Dijkstra from every defect, then blossom on the complete graph) stays
in the code as the sparse one's first oracle.

**Correlated matching** follows PyMatching 2.4's `enable_correlations=True`, read from its source,
so that PyMatching can serve as the oracle. A Y error sets off checks in both halves of the
detector graph, and the error model records it as two pieces that fire together. Plain matching
forgets that; correlated matching uses it in two passes:

1. **Match once.** Trace a shortest path for each matched pair into one edge set.
2. **Lower the weights.** For every edge in that set, lower the weight of each edge it shares a
   fault with to $\ln((1-p_a)/p_a)$. Here $p_a = \min(1/2,\ \mathrm{joint}(c, a)/\mathrm{marginal}(c))$
   is the probability that edge $a$ fired, given that edge $c$ did.
3. **Match again** on the lowered weights, and read off the prediction.

## Window decoding

A global decoder waits for the experiment to end. A **window decoder** matches a few rounds at a
time:

- It decodes a window of *commit* + *buffer* rounds and keeps the edges of its correction that
  touch the commit region.
- It toggles their far ends, so a correction reaching past the region leaves a defect for the next
  window.
- An edge leaving the window becomes a boundary half-edge of its own, kept apart from the real
  boundary. A defect near the window's end can then wait for a partner not yet seen.

There are two schedules:

- **Sliding** windows run one after another, so each stream uses one core.
- **Parallel** windows (Skoric et al., arXiv:2209.08552; Tan et al., arXiv:2209.09219) come in two
  layers. Layer A's windows are spaced apart, with virtual boundaries on both sides. Layer B's fill
  the gaps once A has committed. Each layer's windows are independent, so one stream can use many
  cores.

Correlated matching works inside a window: the second pass is traced on the lowered weights, and
the correlation rules are restricted to the window's edges.

**Streaming.** A million-round experiment has no error model that fits in memory, and none is
needed: a memory circuit repeats. The stream decoder builds the model of a short *template* once,
and gives each window of the stream the graph of the template window in the same position. The
first and last windows come from the template's ends, the bulk from its middle. Rounds are pushed
in as the batch sampler makes them, and a window is decoded as soon as its last round has arrived
and the windows it depends on are done. Once no window still needs a round, the round is dropped,
and any defect left in it is counted as unexplained.

## Where it runs

The same Rust runs natively, from Python (a PyO3 module, `pip`-installable as one abi3 wheel for
Python 3.9 on), and as WebAssembly in the browser. The page has no build step. It runs the engine
in a pool of web workers, one per core but one, at most eight. The engine is compiled once and
shared by every worker.

# How it is checked

The checks come in three layers, strictest first:

- **Exact oracles.** Where an answer is unique, it is checked exactly:
  - The sparse matcher's optimal weight must *equal* the dense matcher's and a brute-force
    search's, on random graphs and on sampled surface-code shots.
  - Every single fault of every circuit, decoded alone, must predict its own logical flip.
  - One window as long as the stream must reproduce the global decoder's traced correction.
  - The batch sampler must reproduce the frame sampler's shots when every channel is made
    deterministic.
- **References.** Stim's error models must equal ours, and PyMatching's matchings must tie ours.
  Google's detection events must equal those we rebuild from their raw measurements.
- **Statistics.** Where only a rate can be compared, the comparison states its interval:
  per-detector χ² between samplers, Wilson intervals on logical error rates, and parametric
  bootstrap intervals on Λ.

The layers have caught real defects. They are listed in full in the repository's README, and four
of them show what each layer is for:

- **A decomposition that passed every unit test.** It split wide faults into their X and Z halves.
  It agreed with Stim fault for fault, because that comparison is about faults, not pieces. But the
  exhaustive single-fault check found faults at d = 3 that it decoded into logical errors: a Y
  error on a boundary qubit, kept whole, became an edge bridging the two matching graphs. The fix
  reproduces Stim's decomposition exactly, and the matching graphs in section 4.2 are the result.
- **Refitting that biased the intervals.** A bootstrap that let each redrawn data set re-choose its
  own points and weights biased the redrawn fidelities about 0.2% low. Averaged over eighteen
  patches, that made the intervals lopsided. It was caught before anything was quoted: the
  intervals had come out asymmetric where they should not have.
- **Parallel windows with no buffer.** They were found to leave corrections nowhere to land. The
  every-defect-explained check found them, and they are now refused.
- **A stream count that could not fail.** Code review found that the stream decoder dropped rounds
  without counting the defects left in them, so its "unexplained" count was zero by construction.
  It now counts them. A test that skips a window shows the leftover defect counted (and fails on the
  old code), and the million-round run below was repeated with the working count.

# Results

## Thresholds

SD6 thresholds from repeated sweeps over d = 3 to 9, each fitted by finite-size scaling. The fit
includes the leading correction to scaling, over the widest window the scaling form describes,
chosen by reduced χ² ({{pth.runs}} independent sweeps per code; `data/sweeps/`):

{{table:thresholds}}

The "uncorrected crossing" is where the curves cross without the correction term, which small
patches pull low; the corrected fit is the threshold.

## Agreement with Stim and PyMatching

**Error models.** The comparison covers Stim's own generated rotated memory, and ours for both
codes under both noise models, at d = 3, 5 and 7 with $p = 0.3\%$. For {{xc.all_identical}} of
them, both models list the same faults with the same probabilities, and the decomposed matching
graphs are identical edge for edge:

{{table:models}}

**Plain matching.** Stim sampled 100,000 shots at each point, and both decoders decoded the same
detection events. Two exact matchers can disagree only where two corrections tie in weight. There
were {{xc.plain_disagree}} disagreements in {{xc.shots}} shots, and every one is a tie: the two
matchings' weights are equal to within the discretisation noise.

{{table:decoding}}

**Correlated matching**, on {{xc.corr_shots}} shots: {{xc.corr_disagree}} disagreements, and none
of them a bug. For each, PyMatching's own first-pass edges went into our second pass:

- in {{xc.corr_path_ties}} cases this reproduces PyMatching's answer, because the two first passes
  traced different, equally short paths;
- the other {{xc.corr_weight_ties}} tie PyMatching's second-pass weight;
- {{xc.corr_non_ties}} are neither.

Correlated matching fails {{xc.corr_failures}} times against plain matching's
{{xc.plain_failures}}, {{xc.corr_gain}} fewer.

{{table:correlated}}

![Our logical error rate against PyMatching's on the same shots: plain matching at six points and correlated matching at twelve, SD6, 100,000 shots each. Every point lies on the diagonal.](figures/pymatching-agreement.svg){width=62%}

**Speed.** Single-threaded, our plain matcher takes {{xc.plain_ratio}} times PyMatching's time per
shot, and our correlated matcher {{xc.corr_ratio}} times PyMatching's correlated mode.
PyMatching's C++ has had years of tuning that this matcher has not. The gap is a roughly constant
factor that does not grow with d, as the figure below shows.

## Google's hardware

Google has published everything its chips recorded in two surface-code memory experiments:

- **Willow** (2024): d = 3, 5 and 7, 1 to 250 rounds;
- **Sycamore** (2022): d = 3 and 5, 1 to 25 rounds.

The data comes as raw measurements, sweep bits, circuits, error models and each of Google's
decoders' predictions. Both datasets are CC BY 4.0, from Zenodo records 13273331 and 6804040.

**Reading the chips' output.** A detector compares a parity of measurements with what a noiseless
run gives. The engine runs the ideal circuit once on its stabilizer tableau to get that reference,
and once more per sweep bit to get each bit's effect. This is independent of the backward walk that
builds error models. Over all {{g.experiments}} experiments and {{g.shots}} shots:

- the detection events and observable flips rebuilt from raw measurements equal Google's **bit for
  bit** ({{g.m2d_exact}} of {{g.experiments}} experiments);
- our model of every noisy circuit is identical to Stim's ({{g.models_same}} of
  {{g.experiments}}).

**One fit, Google's.** Λ is fitted the way Google fits it:

1. **Per patch and basis,** the logical fidelity $1 - 2P_L$ is fitted as $A(1-2\varepsilon)^r$ by
   weighted least squares on its logarithm, from round 10 for Willow and round 3 for Sycamore.
2. **Per distance,** ε is the mean over patches and bases.
3. **Λ** is $\exp(-2\,\mathrm{slope})$ of a line through $\ln \varepsilon$ against $d$.
4. **Intervals** come from a parametric bootstrap with 400 draws.

As validation, Google's own tensor-network predictions for Sycamore, fitted this way, give
ε₃ = {{val.e3}} and ε₅ = {{val.e5}}. The published values are {{val.pub3}} and {{val.pub5}}:
the same, to the digits published.

{{table:willow}}

![Willow's logical error per cycle against code distance, from every experiment's recorded shots, fitted as Google fits them. Bars are 95% bootstrap intervals.](figures/willow-lambda.svg){width=82%}

- **Correlated matching is most of the story.** With the same prior, it takes Λ from
  {{w.plain.l}} to {{w.corr.l}}, and ε at d = 7 from {{w.plain.e7}} to {{w.corr.e7}}.
- **Ours sits beside Google's own correlated matcher** on the same prior, {{w.corr.e7}} against
  {{w.gcorr.e7}} at d = 7. It is further behind at d = 3 ({{w.corr.e3}} against
  {{w.gcorr.e3}}), which is why its Λ comes out higher: Λ rewards improving with size, not being
  good.
- **Our own model of the noisy circuit**, built from Google's circuit with no fitting to the data,
  does about as well as Google's fitted SI1000 prior ({{w.own.e7}} at d = 7).
- **Google's best decoders are better still.** Libra reaches Λ = {{w.libra.l}} and ε₇ =
  {{w.libra.e7}}. The published neural-network decoder's 2.14 cannot be refitted here: its
  predictions are not in the dataset.

{{table:sycamore}}

On Sycamore, our plain matcher reproduces the PyMatching predictions Google recorded
({{s.plain.e3}} against {{s.pm.e3}} at d = 3). The two disagree on {{g.syc_disagree}} shots over
the whole dataset. On every one of them, our matching's weight equals the optimum PyMatching 2.4
finds ({{g.syc_not_optimal}} are not optimal): these are ties, broken differently by the older
PyMatching that Google used. Sycamore's Λ ≈ 1 was that paper's point, the first time a larger
surface code beat a smaller one at all, and only with the best decoders.

## Throughput

Sampling speed per shot, single-threaded, rotated SD6 with T = d. Stim is timed through its Python
API, compiled beforehand and writing packed bits; ours includes transposing batches into Stim's
b8 layout:

{{table:sampling}}

![Decoding (left) and sampling (right) per shot against distance, single-threaded, SD6 at p = 0.3%.](figures/speed.svg)

The batch sampler is {{speed.batch_vs_frame}} times as fast as the frame sampler, and faster than
Stim's as called from Python. Decoding, not sampling, is now almost all of a shot's cost.

On the page, work is split across a pool of workers. On the development machine the site's
section 7 threshold sweep ran {{readme:2.5 to 4.4 times faster}} with the pool than with one worker; the
spread comes from whether workers land on performance or efficiency cores.

## Real time

Willow runs a round of error correction every {{rt.cycle}}. Google has decoded a d = 5 memory in
real time over a million rounds with a 63 µs mean latency (arXiv:2408.13687), on its own hardware
and with its own definition of latency. Here, every window's decode time is measured natively on
{{rt.machine}}, one core at a time. Those times are then scheduled with rounds arriving every
{{rt.cycle}}:

- A window starts once its last round has arrived and, for layer B, both its layer-A neighbours are
  done.
- **Latency** is the time from the arrival of a window's last round to its commit.
- A stream **keeps up** if its latency does not grow along it.

The windows commit d rounds with a buffer of d. The streams are Willow's recorded syndromes, 2,000
shots at each of d = 3, 5 and 7, at 250 rounds, decoded with Google's SI1000 prior:

{{table:latency}}

![Mean window latency (points; bars reach the 99th percentile) against the cores given to one stream, parallel windows with correlated matching on Willow's recorded 250-round syndromes.](figures/latency.svg){width=82%}

- **Sliding windows** keep up only at d = 3 with plain matching. Everywhere else one core is too
  slow, which is why parallel windows exist.
- **At d = 5, parallel windows with correlated matching keep up on {{rt.d5.cores}} cores** with a
  {{rt.d5.mean}} mean latency. At d = 7 they need {{rt.d7.cores}}.

**A million rounds.** The stream is a rotated d = 5 memory under SD6 at *p* = {{m.p}}, the noise
at which its detectors fire as often as Willow's do ({{m.frac}} of detectors a round). There are
{{m.streams}} streams of {{m.rounds}} rounds each, sampled round by round and decoded as they
arrive:

{{table:million}}

- **Every defect of every stream is explained** ({{m.unexplained}} left, counted as rounds are
  dropped).
- **Correlated matching on 8 cores holds a {{m.corr.mean8}} mean latency** ({{m.corr.p998}} at the
  99th percentile) over a million rounds.
- **Throughput.** On all {{m.tput.cores}} cores, independent streams decode at {{m.tput.plain}}
  rounds a second with plain matching and {{m.tput.correlated}} with correlated. Willow's cycle
  demands {{m.willow_rate}} a second per stream.

**Accuracy.** Windowing costs nothing in Λ. The run covers {{win.experiments}} Willow
experiments (all but the single-round ones, which have nothing to window), {{win.shots}} shots
each. Every one was decoded globally and by sliding and parallel windows, fitted as above:

{{table:windows}}

- **Correlated matching gives the same Λ either way:** {{win.corr.lambda}} for parallel windows,
  {{win.global.lambda}} globally.
- **A buffer of d/2 is too short.** It raises ε₅ by {{win.half_cost}}, while a buffer of 2d gains
  nothing over d.
- **Simulated SD6 agrees.** At d = 3, 5 and 7, $p = 0.3\%$ and $0.5\%$, 50 rounds and
  {{win.sd6_shots}} shots, every window decoder's failures are within {{win.sd6_worst}} of the
  global decoder's on the same shots.
- **No defect was left unexplained** ({{win.unexplained}} over every Willow run).

# Limits and next steps

- **The matcher.** It is exact but not the fastest: it runs at {{xc.plain_ratio}} times
  PyMatching's single-threaded time. Correlated matching closes most of the gap to Google's
  correlated matcher, but Google's Harmony and Libra decoders do better still.
- **Latency is scheduled, not served.** The latency figures schedule decode times measured
  natively on one machine. They leave out the transport of syndromes from a quantum computer to
  the decoder, and they are not a claim about any particular control system.
- **The next decoders.** The next pieces of work, in order:
  - belief propagation and belief-matching on the Willow data;
  - the IBM "gross" bivariate-bicycle code with BP+OSD;
  - lattice surgery between patches;
  - a resource estimator driven by the Λ measured here.

# Reproducing everything

Every number in this report is filled in by `tools/report.py` from these committed files:

- `data/xcheck/reference.json`: the cross-check;
- `data/sweeps/`: thresholds;
- `data/google-results/`: Google's data, decoded;
- `data/realtime/`: latency, the million-round stream, and windowed accuracy.

The report fails to build if a value it quotes is missing. To regenerate the data:

```bash
python3 -m venv .venv && .venv/bin/pip install stim pymatching numpy maturin
VIRTUAL_ENV=$PWD/.venv CARGO_TARGET_DIR=target-py .venv/bin/maturin develop --profile python
.venv/bin/python tools/xcheck.py                       # Stim and PyMatching
.venv/bin/python tools/google.py run                   # Google's data (fetch it first)
.venv/bin/python tools/realtime.py latency             # window latency on Willow
.venv/bin/python tools/realtime.py million             # a million rounds
.venv/bin/python tools/realtime.py accuracy            # windowed against global
node tools/lambda.mjs                                  # every fit
python3 tools/report.py                                # this report
```

The package installs with `pip install stabilizer-qec` once published, or from a wheel built by
`maturin build`. Software: Stim {{stim}}, PyMatching {{pymatching}}. Google's datasets are
CC BY 4.0 (Google Quantum AI; Zenodo 13273331 and 6804040); the engine is MIT-licensed.
