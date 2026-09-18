# Instrument upgrade — the explainer as a living instrument

**Date:** 2026-09-18
**Status:** Approved (direction C, "Instrument"; Jasper asked for the plan and the build to run overnight)

## Problem

jaspersands.com now opens with a distance-27 surface code decoded live by this engine. Next to it
the explainer reads as a paper textbook: the top of the page is a title and four stat cards (two of
them "···" while throughput is measured), every figure sits in a boxed card with a beige inset,
nothing moves until it is clicked, and the page's real differentiator — every number computed in the
reader's browser — is invisible until the reader reaches the table in section 07.

The format is right and stays: ten sections in reading order, prose at measure, one figure per beat,
the rail with scroll-spy, every figure driven by the engine, no build step, no dependencies. The
identity stays: warm paper, ink, Lora / Inter / JetBrains Mono. Light only; there is no dark theme
and none is added.

## Decisions

| Question | Decision |
|---|---|
| Identity | Paper-white, unchanged. Upgrade within it. |
| The top of the page | A live lattice, ink on paper, replaces the four stat cards. The throughput vitals become one readout line under it. |
| Figures | Un-boxed: a hairline rule and a numbered caption line, the stage on the paper itself, full column width. |
| Read-only figures | Ambient: Figure 2 takes and corrects an error every few seconds; Figure 5 builds its stack and shows a lying readout on a slow cycle. Both stop the moment the reader touches them. |
| Section 07 | The load-time table becomes a live plot first — points land as cells are measured — with the table beneath as the data. |
| Section 09 | The bench streams: a run is chunked, and the logical error estimate, its interval and throughput are drawn as they converge. Stop button. |
| Palette | One pass: plaquette washes deep enough to carry the page; every label still clears 4.5:1 on the lightest surface it sits on. |
| Motion budget | Slow. Nothing on this page should read as an effect. `prefers-reduced-motion` gets a composed still frame everywhere. |

## Components

### 1. `js/opener.js` — the live lattice at the top

A canvas the full width of the main column, ~420 px tall on desktop, in the hero between the lede and
the readout line. A distance-*d* rotated surface code (d = 15 on desktop, 11 under 860 px, 9 under
560 px) sized so the patch fills the height; the canvas is wider than the patch, and the margin
either side carries nothing, which is the point: it is a diagram that happens to be running.

Drawing, in the page's tokens: plaquettes as `--x-soft` / `--z-soft` washes with `--rule-soft`
edges, boundary half-discs as the existing renderer draws them, data qubits as small `--surface`
discs with `--ink-3` rims. Errors fill the qubit with `--x` / `--z` / `--y` and stamp the letter as
`LatticeView` does. Fired checks fill `--defect-soft` with a `--defect` dot. Corrections are drawn
as `--ok` chains along the matched qubits, held briefly, then faded together with the errors they
cancel.

Behaviour, on the main-thread engine instance, with its own `Session`:

- Ambient noise: Poisson, about 0.8 errors per second over the patch, depolarising.
- Pointer noise: a Gaussian footprint under the pointer, bit-flip biased, so a sweep across the
  patch can build a chain. `touch-action: pan-y`; touch scrolling is never captured.
- A round every ~1.4 s: `decode(MWPM)` → animate correction → `clearErrors()`. `failed` from the
  engine increments a logical-error count and flashes the patch edge in `--fail-soft`.
- Readout line beneath, mono: `d = 15 · rounds N · physical errors N · logical errors N · MWPM x ms`
  followed by the throughput vitals once measured (`data noise N runs/s · phenomenological N runs/s
  · engine 210 KB`). The two throughput numbers are the same measurement `vitals.js` makes today,
  moved.
- Off-screen or hidden tab: the loop stops; it restarts on return.
- Reduced motion: one composed frame (a short chain, its defects, the matched correction), no loop,
  no pointer noise.
- Engine unavailable: the canvas is hidden and the existing boot-error banner does its job.

The opener has its own renderer rather than reusing `LatticeView`: it needs per-frame animation of
chains, a pointer footprint, and a patch four times larger than any figure, and it must not inherit
hover, hit-testing or keyboard cursors. The geometry (support map, plaquette polygons, boundary
half-discs) is read from the `Session` as `LatticeView` reads it, so the two draw the same shapes.

Pure helpers — Poisson sampling, the Gaussian footprint rate, chain extraction from a correction,
and the pointer-to-lattice projection — live in `js/opener-math.js` with no DOM so they can be
tested in Node.

### 2. Figures un-boxed

`.panel` is retired in favour of `.figure`:

```
<figure class="figure">
  <figcaption class="figure__cap">
    <span class="figure__num">Figure 2</span> <span class="figure__title">the lattice</span>
    <span class="figure__meta">read-only</span>
  </figcaption>
  <div class="figure__body"> …rig… </div>
</figure>
```

A hairline `--rule` above the caption, the caption in the mono small-caps voice, the body on the
paper with no border and no shadow. `.rig__stage` loses its sunk background and border; the canvas
sits directly on the paper. Controls keep their column. Tables keep a rule above and below and lose
the card shadow. The bench console keeps a light frame (it is a console) but loses the shadow.

### 3. Ambient figures

`js/ambient.js` exports one helper: `ambient({ view, session, every, act, until })` — a timer that
calls `act()` every `every` ms while the figure is on screen and the tab is visible, and stops for
good the first time the figure is interacted with (pointer down, key down, or any control change in
its section). Reduced motion: never starts.

- Figure 2 (anatomy): every 2.8 s, one X or Z error on a random qubit; after 1.1 s the correction
  (MWPM) is drawn as `LatticeView` already draws corrections; after another 0.9 s the patch is
  cleared. Hover pauses it (the hover readout is the figure's job) and it resumes on leave.
- Figure 5 (spacetime): on first sight, layers appear one at a time bottom to top over ~1.5 s.
  Then every 4 s a random plaquette in a middle round takes a lying readout, the amber pair and its
  worldline show for 2 s, and the patch clears.

Both figures keep every interaction they have now. The verdict banners and readouts are untouched
by the ambient layer: it only calls `toggleError` / `toggleMeasurementError` / `decode` /
`clearErrors` and `view.draw()`.

### 4. Section 07: plot first, table beneath

Above the results table, a `.figure` holding two `Plot` canvases side by side — *Data noise only*
and *Phenomenological, T = d* — each with three series (d = 3, 5, 7, the existing `--d*` tokens) of
logical error rate against p over the table's six rates, points with Wilson bars, joined by lines in
p order. Points appear as `compute.call('table', …)` reports each cell; the same callback that paints
the table paints the plot. Under 720 px the two plots stack.

The crossing is the figure: below threshold d = 7 is lowest, above it d = 7 is highest. No fit is
drawn here; the fitted sweep remains Figure 6 as it is. The table's status line and caption stay.

### 5. Section 09: the bench streams

Worker: a new op `stream` runs the same `runBenchmark` in chunks of `max(250, runs / 60)` shots,
yields to the event loop between chunks, and reports after each: `{ done, failures, seconds }`
cumulative, plus `chunkRunsPerSecond`. A `cancel` message with the job id sets a flag the loop
checks between chunks; the job then resolves with what it has. `Compute` gets `cancel(id)`, and
`call` returns a promise carrying its `id`.

Bench UI, replacing the four-row readout:

- A strip chart canvas (`data-bench-meter`, aspect 0.42): x = shots so far (0 → requested), y =
  logical error rate; the running estimate as an `--ink` line with its Wilson 95% band as a
  `--defect-soft` fill, so the reader watches the band narrow. The final value and interval are
  written at the right edge when the run completes.
- Live readout rows: logical error rate (running), 95% interval, throughput (last chunk, and the
  mean), shots done / requested, wall time.
- Run becomes Run / Stop. Stopped runs report what they measured; nothing is discarded.

The channel estimate below it is unchanged.

### 6. Palette pass

`--x-soft` #dfe7fa → #d3def6, `--z-soft` #f8e2df → #f4d6d1, `--defect-soft` unchanged, boundary
edge `--rule-soft` unchanged. `--ink-3` on the new washes must clear 4.5:1 (it clears 5.6:1 on the
current ones; the new ones are checked in the plan). The four `.vital` cards and their styles go.

## What does not change

Every section, its prose, its ids and the rail. Every computation, shot count and interval. The
engine and the worker protocol for the existing ops. Keyboard operation of the figures. The
accessible summaries `LatticeView` and `Plot` maintain. No build step; still `python3 -m http.server`.

## Testing

- Node: `tools/site-tests.mjs` covers `opener-math.js` (Poisson mean, footprint falls to zero
  outside its limit, chain extraction orders a boundary chain from the boundary) and the stream
  chunk planner (chunks cover the request exactly, last chunk never zero).
- Browser, headless Chrome through CDP: the opener boots and rounds advance; ambient figures stop
  after a synthetic pointer down; the table plot has 36 points when the status reads "Complete";
  a streamed bench run reports done === requested and a Stop mid-run resolves with done < requested;
  no console errors; screenshots of the top, Figure 2, section 07 and the bench at 1440 and 390 px.
- Contrast: computed for `--ink-3` on both new washes.

## Out of scope

Dark theme. Changes to the Rust engine or the wasm. Rewriting prose. The fitted sweep (Figure 6),
the bias comparison and the internals section, beyond the un-boxing that every figure gets.
