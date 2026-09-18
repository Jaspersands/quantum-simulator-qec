# Instrument Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the paper-white explainer into a living instrument — a live lattice at the top, un-boxed full-width figures, ambient read-only figures, a live plot over the threshold table, and a streaming bench — without changing a section, a number, or the identity.

**Architecture:** Every new piece hangs off the existing module layout: one new section-like module for the opener (`js/opener.js` + pure `js/opener-math.js`), one scheduling helper (`js/ambient.js`) used inside the two read-only sections, a chunked `stream` op in the worker with a `cancel` message, a small strip-chart renderer (`js/meter.js`), and a second use of the existing `Plot` in the threshold section. Structure stays in `index.html`; behaviour stays in `js/`; every colour stays a token in `css/styles.css`.

**Tech Stack:** Vanilla ES modules, canvas 2D, the existing `stabilizer_qec.wasm` through `js/engine.js`, a module Worker. No build step. Node 24 for the pure-function tests; headless Chrome through CDP for browser checks.

## Global Constraints

- Paper-white only. No dark theme. (spec: "Light only; there is no dark theme and none is added.")
- "Every colour, size, and border in this file comes from a token" — `css/styles.css` header rule. No hex below the token block.
- No build step, no dependencies; the page runs from `python3 -m http.server`.
- Every section, its prose, ids and the rail are unchanged. Every computation, shot count and interval is unchanged.
- `prefers-reduced-motion: reduce` gets a composed still frame everywhere; no loop starts.
- Text on a wash clears 4.5:1 against the lightest surface it sits on.
- Commit after every task with the message trailer `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.

---

## File map

| File | Responsibility |
|---|---|
| `css/styles.css` | tokens (palette pass), `.figure` replaces `.panel`, `.opener`, `.meter-chart`, `.results-plots`; `.vital*` removed |
| `index.html` | hero: opener canvas + readout line replaces `.vitals`; every `.panel` becomes a `.figure`; section 07 gains the two plot canvases; section 09 gains the meter canvas and Stop |
| `js/opener-math.js` (new) | pure: `poisson`, `footprintRate`, `chainsFromCorrection`, `layoutFor` |
| `js/opener.js` (new) | the live lattice at the top: renderer, noise, rounds, readout, reduced-motion frame |
| `js/sections/vitals.js` | unchanged API; fills inline spans now |
| `js/ambient.js` (new) | `ambient({root, canvas, every, act})` scheduler |
| `js/lattice.js` | `visibleRounds` option so the spacetime stack can reveal itself |
| `js/sections/anatomy.js`, `js/sections/spacetime.js` | ambient acts |
| `js/sections/threshold.js` | live plots over the results table |
| `js/stream.js` (new) | pure: `planChunks(total)` |
| `js/worker.js`, `js/compute.js` | `stream` op, `cancel` message, `Compute.cancel(id)` |
| `js/meter.js` (new) | strip-chart renderer for the streaming bench |
| `js/sections/bench.js` | Run/Stop, live readout, meter |
| `js/main.js` | boots the opener |
| `tools/site-tests.mjs` (new) | Node tests for the pure modules |

---

### Task 1: Palette pass and un-boxed figures

**Files:**
- Modify: `css/styles.css` (tokens block; section 6 Panels; section 9 `.vital*`, `.table-wrap`)
- Modify: `index.html` (every `<div class="panel">` … `</div>`)
- Create: `tools/contrast.mjs`

**Interfaces:**
- Produces: `.figure`, `.figure__cap`, `.figure__num`, `.figure__title`, `.figure__meta`, `.figure__body`, `.figure--console` (CSS classes used by every later task's markup).

- [ ] **Step 1: Write the contrast check**

`tools/contrast.mjs`:

```js
// WCAG contrast of --ink-3 against the plaquette washes. Run: node tools/contrast.mjs
import { readFileSync } from 'node:fs';
const css = readFileSync(new URL('../css/styles.css', import.meta.url), 'utf8');
const token = (name) => css.match(new RegExp(`${name}:\\s*(#[0-9a-f]{6})`, 'i'))[1];
const lum = (hex) => {
  const c = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255)
    .map((v) => (v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4));
  return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
};
const ratio = (a, b) => { const [hi, lo] = [lum(a), lum(b)].sort((x, y) => y - x); return (hi + 0.05) / (lo + 0.05); };
let failed = false;
for (const [text, on] of [['--ink-3', '--x-soft'], ['--ink-3', '--z-soft'], ['--ink-3', '--surface-sunk'], ['--defect-ink', '--defect-soft']]) {
  const r = ratio(token(text), token(on));
  console.log(`${text} on ${on}: ${r.toFixed(2)}:1 ${r >= 4.5 ? 'ok' : 'FAIL'}`);
  if (r < 4.5) failed = true;
}
process.exit(failed ? 1 : 0);
```

- [ ] **Step 2: Run it against the current tokens**

Run: `node tools/contrast.mjs`
Expected: four lines, all `ok` (the current washes pass).

- [ ] **Step 3: Deepen the washes**

In the tokens block of `css/styles.css`:

```css
  --x-soft: #d3def6;
  --z-soft: #f4d6d1;
```

- [ ] **Step 4: Run the contrast check again**

Run: `node tools/contrast.mjs`
Expected: all `ok`. If a wash fails, lighten it one step (`#d8e2f7`, `#f6dbd6`) and re-run; do not lower the bar.

- [ ] **Step 5: Replace the panel vocabulary in the stylesheet**

Replace section 6's `.panel`, `.panel__head`, `.panel__label`, `.panel__body` rules with:

```css
/* --------------------------------------------------------------------------
   6. Figures — the one container vocabulary
   A figure sits on the paper: a hairline above, a caption line in the mono
   voice, the stage with no border and no shadow. Only the bench console keeps
   a frame, because it is a console.
   -------------------------------------------------------------------------- */

.figure {
  margin: var(--s6) 0;
  padding-top: var(--s3);
  border-top: var(--rule-w) solid var(--rule);
}

.figure__cap {
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  gap: var(--s2) var(--s3);
  margin: 0 0 var(--s4);
  font-family: var(--font-mono);
  font-size: var(--t-micro);
  text-transform: uppercase;
  letter-spacing: 0.12em;
  font-weight: 600;
  color: var(--ink-2);
}
.figure__num { color: var(--ink); font-weight: 700; }
.figure__meta { margin-left: auto; color: var(--ink-3); font-weight: 500; }

.figure__body { min-width: 0; }

.figure--console .figure__body {
  border: 1px solid var(--rule-soft);
  padding: var(--s5);
  background: var(--surface);
}
```

And in `.rig__stage` drop the box:

```css
.rig__stage {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: var(--s3);
  min-width: 0;
}
```

And the table wrap:

```css
.table-wrap {
  overflow-x: auto;
  border-top: var(--rule-w) solid var(--rule);
  border-bottom: var(--rule-w) solid var(--rule);
  background: var(--surface);
}
```

Delete the `.vitals`, `.vital`, `.vital__label`, `.vital__value`, `.vital__unit`, `.vital__note` rules (Task 3 replaces the markup that used them).

- [ ] **Step 6: Convert every panel in `index.html`**

Eight panels, at the line numbers `grep -n 'class="panel"' index.html` reports. Each has this shape:

```html
  <div class="panel">
    <div class="panel__head">
      <span class="panel__label">Figure 2 · the lattice</span>
      <span class="panel__label">read-only</span>
    </div>
    <div class="panel__body">
      …
    </div>
  </div>
```

and becomes:

```html
  <figure class="figure">
    <figcaption class="figure__cap">
      <span class="figure__num">Figure 2</span>
      <span class="figure__title">the lattice</span>
      <span class="figure__meta">read-only</span>
    </figcaption>
    <div class="figure__body">
      …
    </div>
  </figure>
```

The first label splits at ` · ` into number and title. The bench's "Monte Carlo console" has no number: it becomes `<figure class="figure figure--console">` with `<span class="figure__num">Monte Carlo console</span>` and no title span. Do it with a script so the closing tags match:

```python
import re
p = 'index.html'; s = open(p).read()
def head(m):
    label, meta = m.group(1), m.group(2)
    console = ' figure--console' if '·' not in label else ''
    num, _, title = label.partition(' · ')
    parts = f'<span class="figure__num">{num}</span>'
    if title: parts += f'\n      <span class="figure__title">{title}</span>'
    parts += f'\n      <span class="figure__meta">{meta}</span>'
    return (f'<figure class="figure{console}">\n    <figcaption class="figure__cap">\n      {parts}\n    </figcaption>\n    <div class="figure__body">')
s2 = re.sub(r'<div class="panel">\s*<div class="panel__head">\s*<span class="panel__label">(.*?)</span>\s*<span class="panel__label">(.*?)</span>\s*</div>\s*<div class="panel__body">', head, s, flags=re.S)
assert s2.count('<figure class="figure') == 8, s2.count('<figure class="figure')
open(p, 'w').write(s2)
```

Then close each figure: the `</div>` that closed `.panel` must become `</figure>`. Each panel body is followed by `    </div>\n  </div>` (body close, panel close) at two-space indent; the panel close is the `  </div>` at column 2 that follows a `    </div>` at column 4 and precedes either a blank line + `<div class="aside">`, `<div class="prose">`, `<p`, `<h3`, `</section>` or a comment. Check by hand after the script: `grep -n -A1 '^    </div>$' index.html | grep -B1 '^[0-9]*-  </div>$'` lists the candidates; there must be exactly 8, one per figure, and each is replaced by `  </figure>`.

- [ ] **Step 7: Verify the markup**

Run: `python3 -c "import html.parser,sys; p=html.parser.HTMLParser(); p.feed(open('index.html').read()); print('parsed')"` and `grep -c '<figure class="figure' index.html; grep -c '</figure>' index.html; grep -c 'panel' index.html`
Expected: `parsed`, `8`, `8`, `0`.

- [ ] **Step 8: Look at it**

Serve (`python3 -m http.server 4174`) and screenshot Figure 2 and section 07 in headless Chrome at 1440 px (the `cdp-shot.mjs` driver in the session scratchpad, or `--screenshot`). Figures sit on the paper with a rule above and a mono caption; no card shadows anywhere except none; the results table has rules top and bottom only.

- [ ] **Step 9: Commit**

```bash
git add css/styles.css index.html tools/contrast.mjs
git commit -m "figures on the paper: un-boxed panels, deeper washes, contrast check"
```

---

### Task 2: Opener math

**Files:**
- Create: `js/opener-math.js`
- Create: `tools/site-tests.mjs`

**Interfaces:**
- Produces:
  - `poisson(lambda, rand = Math.random) → integer`
  - `footprintRate(dist, sigma, peak) → number` (0 beyond `3 * sigma`)
  - `layoutFor(width, height, d, pad) → { size, originX, originY, unit }` where `unit` is px per doubled-grid step (a cell is `2 * unit`)
  - `chainsFromCorrection(session, correctionX, correctionZ) → Array<{ type: 'X'|'Z', qubits: number[] }>` ordered end to end, boundary end first; uses `session.touchedBy`, `session.stabilizers[i].type` and the `STAB` codes (Z = 0, X = 1)
  - `pickPauli(rand, mix) → 'X'|'Z'|'Y'` for `mix = [['X', w], ['Z', w], ['Y', w]]`

- [ ] **Step 1: Write the failing tests**

`tools/site-tests.mjs`:

```js
// Node tests for the pure site modules. Run: node tools/site-tests.mjs
import assert from 'node:assert/strict';
import { poisson, footprintRate, layoutFor, chainsFromCorrection, pickPauli } from '../js/opener-math.js';

let passed = 0, failed = 0;
const test = (name, fn) => { try { fn(); passed++; console.log(`  ✓ ${name}`); } catch (e) { failed++; console.log(`  ✗ ${name}\n    ${e.message}`); } };
let seed = 20260918;
const rand = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296; };

test('poisson: mean within five standard errors over 50,000 draws', () => {
  for (const lambda of [0.04, 1, 4]) {
    let sum = 0; const N = 50000;
    for (let i = 0; i < N; i++) sum += poisson(lambda, rand);
    const tol = 5 * Math.sqrt(lambda / N);
    assert.ok(Math.abs(sum / N - lambda) < tol, `lambda ${lambda}: mean ${sum / N}`);
  }
  assert.equal(poisson(0, rand), 0);
});

test('footprintRate: peak at the pointer, zero beyond three sigma, monotone between', () => {
  assert.equal(footprintRate(0, 1.1, 8), 8);
  assert.equal(footprintRate(3.31, 1.1, 8), 0);
  let prev = Infinity;
  for (let r = 0; r < 3.3; r += 0.1) { const v = footprintRate(r, 1.1, 8); assert.ok(v <= prev); prev = v; }
});

test('layoutFor: the patch fills the height and is centred horizontally', () => {
  const L = layoutFor(1000, 420, 15, 20);
  assert.equal(L.size, 380);
  assert.equal(L.originX, 310);
  assert.equal(L.originY, 20);
  assert.ok(Math.abs(L.unit - 380 / 30) < 1e-9);
});

test('pickPauli follows the mix', () => {
  const mix = [['X', 0.8], ['Z', 0.15], ['Y', 0.05]];
  const n = { X: 0, Z: 0, Y: 0 };
  for (let i = 0; i < 20000; i++) n[pickPauli(rand, mix)]++;
  assert.ok(n.X > 15000 && n.Z > 2400 && n.Y > 700, JSON.stringify(n));
});

// A stand-in for engine.Session with the geometry of a d = 5 rotated patch:
// qubits at odd doubled coordinates, checks at even ones, checkerboard types.
function fakeSession(d) {
  const dataQubits = [], stabilizers = [];
  for (let r = 0; r < d; r++) for (let c = 0; c < d; c++) dataQubits.push({ idx: r * d + c, x: 2 * c + 1, y: 2 * r + 1 });
  let idx = 0;
  for (let y = 0; y <= 2 * d; y += 2) for (let x = 0; x <= 2 * d; x += 2) {
    const inside = x > 0 && x < 2 * d && y > 0 && y < 2 * d;
    const type = ((x / 2 + y / 2) % 2 === 0) ? 0 : 1;              // 0 = Z, 1 = X
    const onLR = (x === 0 || x === 2 * d) && y > 0 && y < 2 * d;    // X half-plaquettes left/right
    const onTB = (y === 0 || y === 2 * d) && x > 0 && x < 2 * d;    // Z half-plaquettes top/bottom
    if (inside || (onLR && type === 1) || (onTB && type === 0)) stabilizers.push({ idx: idx++, x, y, type });
  }
  const support = new Map(), touchedBy = new Map();
  for (const q of dataQubits) touchedBy.set(q.idx, []);
  for (const s of stabilizers) {
    const qs = dataQubits.filter((q) => Math.abs(q.x - s.x) === 1 && Math.abs(q.y - s.y) === 1).map((q) => q.idx);
    support.set(s.idx, qs);
    for (const q of qs) touchedBy.get(q).push(s.idx);
  }
  return { d, dataQubits, stabilizers, support, touchedBy, numData: d * d };
}

test('chainsFromCorrection: a row of X corrections from the left edge is one chain, boundary first', () => {
  const s = fakeSession(5);
  const cx = new Uint8Array(25), cz = new Uint8Array(25);
  cx[2 * 5 + 0] = 1; cx[2 * 5 + 1] = 1; cx[2 * 5 + 2] = 1;   // row 2, columns 0..2
  const chains = chainsFromCorrection(s, cx, cz);
  assert.equal(chains.length, 1);
  assert.equal(chains[0].type, 'X');
  assert.deepEqual(chains[0].qubits, [10, 11, 12]);
});

test('chainsFromCorrection: separate X and Z components come back as separate chains', () => {
  const s = fakeSession(5);
  const cx = new Uint8Array(25), cz = new Uint8Array(25);
  cx[12] = 1; cz[0] = 1; cz[5] = 1;                          // lone X in the middle; Z pair down column 0
  const chains = chainsFromCorrection(s, cx, cz);
  assert.equal(chains.length, 2);
  assert.deepEqual(chains.find((c) => c.type === 'X').qubits, [12]);
  assert.equal(chains.find((c) => c.type === 'Z').qubits.length, 2);
});

console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);
```

- [ ] **Step 2: Run to verify it fails**

Run: `node tools/site-tests.mjs`
Expected: `ERR_MODULE_NOT_FOUND` for `js/opener-math.js`.

- [ ] **Step 3: Implement `js/opener-math.js`**

```js
/**
 * Pure helpers for the live lattice at the top of the page. No DOM, no engine:
 * everything here takes plain numbers or a Session-shaped object, so it runs
 * under Node for tests.
 */

import { STAB } from './engine.js';

/** Knuth's Poisson sampler; fine for the small means used here. */
export function poisson(lambda, rand = Math.random) {
  if (lambda <= 0) return 0;
  const L = Math.exp(-lambda);
  let k = 0, p = 1;
  do { k++; p *= rand(); } while (p > L);
  return k - 1;
}

/** Gaussian error rate under the pointer, in errors/s/qubit; dist and sigma in cells. */
export function footprintRate(dist, sigma, peak) {
  if (dist > 3 * sigma) return 0;
  return peak * Math.exp(-(dist * dist) / (2 * sigma * sigma));
}

/**
 * Square patch of distance d filling the canvas height less a pad, centred
 * horizontally. `unit` is pixels per doubled-grid step: a qubit at engine
 * (x, y) draws at (originX + x * unit, originY + y * unit).
 */
export function layoutFor(width, height, d, pad) {
  const size = Math.max(0, Math.min(height - 2 * pad, width - 2 * pad));
  return {
    size,
    originX: (width - size) / 2,
    originY: (height - size) / 2,
    unit: size / (2 * d),
  };
}

export function pickPauli(rand, mix) {
  let r = rand();
  for (const [pauli, weight] of mix) { r -= weight; if (r <= 0) return pauli; }
  return mix[mix.length - 1][0];
}

/**
 * Group the qubits a correction flips into chains for drawing, one chain per
 * connected component, ordered end to end.
 *
 * Two flipped qubits are linked when they share a check of the type that sees
 * that Pauli: X flips are seen by Z checks and Z flips by X checks. That is the
 * same adjacency the matching used, so the chains drawn are the chains the
 * decoder chose. A chain that touches the boundary starts there.
 */
export function chainsFromCorrection(session, correctionX, correctionZ) {
  const out = [];
  const maxCoord = 2 * session.d - 1;
  const seesType = { X: STAB.Z, Z: STAB.X };

  for (const [type, mask] of [['X', correctionX], ['Z', correctionZ]]) {
    const active = [];
    for (let i = 0; i < session.numData; i++) if (mask[i]) active.push(i);
    if (!active.length) continue;

    const activeSet = new Set(active);
    const byCheck = new Map();                       // check idx -> active qubits it touches
    for (const q of active) {
      for (const s of session.touchedBy.get(q) ?? []) {
        if (session.stabilizers[s].type !== seesType[type]) continue;
        if (!byCheck.has(s)) byCheck.set(s, []);
        byCheck.get(s).push(q);
      }
    }
    const adj = new Map(active.map((q) => [q, new Set()]));
    for (const qs of byCheck.values()) {
      for (const a of qs) for (const b of qs) if (a !== b) adj.get(a).add(b);
    }

    const onBoundary = (q) => {
      const { x, y } = session.dataQubits[q];
      return type === 'X' ? (x === 1 || x === maxCoord) : (y === 1 || y === maxCoord);
    };

    const seen = new Set();
    for (const start of active) {
      if (seen.has(start)) continue;
      const comp = [];
      const queue = [start];
      seen.add(start);
      while (queue.length) {
        const q = queue.shift();
        comp.push(q);
        for (const n of adj.get(q)) if (!seen.has(n) && activeSet.has(n)) { seen.add(n); queue.push(n); }
      }
      // Walk from an end: a degree-1 qubit on the boundary, else any degree-1, else any boundary, else the first.
      const degree1 = comp.filter((q) => adj.get(q).size <= 1);
      const head = degree1.find(onBoundary) ?? degree1[0] ?? comp.find(onBoundary) ?? comp[0];
      const ordered = [head];
      const used = new Set([head]);
      let cur = head;
      while (ordered.length < comp.length) {
        const next = [...adj.get(cur)].find((n) => !used.has(n)) ?? comp.find((n) => !used.has(n));
        used.add(next); ordered.push(next); cur = next;
      }
      out.push({ type, qubits: ordered });
    }
  }
  return out;
}
```

- [ ] **Step 4: Run the tests**

Run: `node tools/site-tests.mjs`
Expected: `6 passed, 0 failed`. (`engine.js` imports cleanly under Node: its only top-level side effects are constant definitions and a `new URL(..., import.meta.url)`.)

- [ ] **Step 5: Commit**

```bash
git add js/opener-math.js tools/site-tests.mjs
git commit -m "opener math: poisson, footprint, layout and correction chains, with tests"
```

---

### Task 3: The live lattice at the top

**Files:**
- Create: `js/opener.js`
- Modify: `index.html` (hero: lines from `<div class="vitals pad-top">` through `<p class="rig__caption" data-vitals-note>…</p>`)
- Modify: `css/styles.css` (new `.opener` block after section 5 Typography)
- Modify: `js/main.js`
- Modify: `js/sections/vitals.js` (comment only; the selectors still resolve)

**Interfaces:**
- Consumes: `opener-math.js` (Task 2); `Session`, `DECODER`, `ERROR` from `engine.js`; `$` from `dom.js`.
- Produces: `initOpener(root, instance) → { stop }`; markup `[data-opener-canvas]`, `[data-opener-readout]` with spans `[data-opener-d]`, `[data-opener-rounds]`, `[data-opener-phys]`, `[data-opener-logic]`, `[data-opener-ms]`; the vitals spans `[data-vital-data]`, `[data-vital-phenom]`, `[data-vital-engine]` now inline in the readout line.

- [ ] **Step 1: Replace the vitals markup in the hero**

In `index.html`, replace from `  <div class="vitals pad-top">` through `  <p class="rig__caption" data-vitals-note>Measuring throughput on this device…</p>` with:

```html
  <div class="opener pad-top">
    <canvas class="opener__canvas" data-opener-canvas
      aria-label="A distance-15 rotated surface code taking random errors and correcting them with minimum-weight perfect matching, live. Move the pointer over it to add noise."></canvas>
    <p class="opener__readout" data-opener-readout>
      <span>d = <b data-opener-d>15</b></span>
      <span>rounds <b data-opener-rounds>0</b></span>
      <span>physical errors <b data-opener-phys>0</b></span>
      <span>logical errors <b data-opener-logic>0</b></span>
      <span>MWPM <b data-opener-ms>·</b></span>
      <span>data noise <b data-vital-data><span class="pending"></span></b></span>
      <span>phenomenological <b data-vital-phenom><span class="pending"></span></b></span>
      <span>engine <b data-vital-engine><span class="pending"></span></b></span>
    </p>
    <p class="rig__caption" data-vitals-note>
      Move the pointer over the patch to add errors; a chain across the whole width is a logical error.
      Measuring throughput on this device…
    </p>
  </div>
```

- [ ] **Step 2: Style it**

Append to section 5 of `css/styles.css`, after `.hero__lede strong`:

```css
/* The live lattice under the lede: a diagram that happens to be running. */
.opener { min-width: 0; }
.opener__canvas {
  width: 100%;
  height: 420px;
  touch-action: pan-y;
  cursor: crosshair;
}
.opener--still .opener__canvas { cursor: default; }
.opener__readout {
  display: flex;
  flex-wrap: wrap;
  gap: var(--s1) var(--s4);
  margin: var(--s3) 0 0;
  font-family: var(--font-mono);
  font-size: var(--t-micro);
  color: var(--ink-3);
}
.opener__readout b { color: var(--ink); font-weight: 600; }
.opener__readout b.flare { color: var(--fail); }
.opener__readout .vital__unit { color: var(--ink-3); font-weight: 400; }
@media (max-width: 860px) { .opener__canvas { height: 340px; } }
@media (max-width: 560px) { .opener__canvas { height: 280px; } }
```

Keep a single `.vital__unit` rule here (vitals.js still emits that class) and confirm no other `.vital*` rule survives: `grep -n 'vital' css/styles.css` shows only that line.

- [ ] **Step 3: Write `js/opener.js`**

```js
/**
 * The live lattice at the top of the page.
 *
 * A distance-d rotated patch on the main-thread engine, taking Poisson noise
 * and pointer noise, decoded by exact MWPM every round, drawn as an ink
 * diagram on the paper. It has its own renderer: the figures' LatticeView is
 * built for small patches with hover and keyboard cursors, and this one needs
 * neither, but does need per-frame animation of the decoder's chains.
 */

import { Session, DECODER, ERROR, STAB } from './engine.js';
import { poisson, footprintRate, layoutFor, chainsFromCorrection, pickPauli } from './opener-math.js';
import { $ } from './dom.js';

const C = {
  PAD: 18,
  AMBIENT: 0.8,                                           // errors / s over the patch
  CURSOR_PEAK: 6, CURSOR_SIGMA: 1.1,                      // errors / s / qubit at the pointer; cells
  PAULI: [['X', 0.45], ['Z', 0.45], ['Y', 0.10]],
  CURSOR_PAULI: [['X', 0.8], ['Z', 0.15], ['Y', 0.05]],   // bit-flip biased, so a sweep builds a chain
  MAX_PENDING: 90,
  ROUND: 1400, TICK: 50,                                  // ms
  T_ERR: 180, T_DEF: 220, T_CHAIN: 380, T_HOLD: 260, T_FADE: 320, T_FLASH: 600,
};

function distanceFor(width) {
  if (width < 560) return 9;
  if (width < 860) return 11;
  return 15;
}

function palette() {
  const css = getComputedStyle(document.documentElement);
  const get = (name, fallback) => (css.getPropertyValue(name).trim() || fallback);
  return {
    surface: get('--surface', '#fffefb'), ink: get('--ink', '#1c1d1f'), ink3: get('--ink-3', '#6b6d71'),
    rule: get('--rule-soft', '#ded9cd'), x: get('--x', '#2f5fd0'), xSoft: get('--x-soft', '#d3def6'),
    z: get('--z', '#c0392b'), zSoft: get('--z-soft', '#f4d6d1'), y: get('--y', '#7b4bb5'),
    defect: get('--defect', '#c98a06'), defectSoft: get('--defect-soft', '#fbeec6'),
    ok: get('--ok', '#2c7a4b'), fail: get('--fail', '#b23a2e'), failSoft: get('--fail-soft', '#f8e2df'),
  };
}

export function initOpener(root, instance) {
  const canvas = $('[data-opener-canvas]', root);
  if (!canvas) return null;
  const ctx = canvas.getContext('2d');
  const out = {
    d: $('[data-opener-d]', root), rounds: $('[data-opener-rounds]', root), phys: $('[data-opener-phys]', root),
    logic: $('[data-opener-logic]', root), ms: $('[data-opener-ms]', root),
  };
  const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const colors = palette();
  const now = () => performance.now();

  const S = {
    session: null, L: null, width: 0, height: 0, dpr: 1,
    pending: [], lit: new Map(), anim: null, flashAt: -Infinity,
    rounds: 0, phys: 0, logical: 0, decodeMs: null,
    pointer: { x: 0, y: 0, t: -Infinity }, running: false, tick: 0, raf: 0, dirty: true, lastTick: 0, nextRound: 0,
    onScreen: true, plaquettes: [],
  };

  /* -- sizing ------------------------------------------------------------ */
  function fit() {
    const rect = canvas.getBoundingClientRect();
    const w = rect.width || 800, h = rect.height || 420;
    S.dpr = Math.min(2, devicePixelRatio || 1);
    canvas.width = Math.round(w * S.dpr); canvas.height = Math.round(h * S.dpr);
    ctx.setTransform(S.dpr, 0, 0, S.dpr, 0, 0);
    S.width = w; S.height = h;
    const d = distanceFor(w);
    if (!S.session || S.session.d !== d) rebuild(d);
    S.L = layoutFor(w, h, d, C.PAD);
    S.plaquettes = S.session.stabilizers.map((st) => plaquette(st));
    S.dirty = true;
  }

  function rebuild(d) {
    S.session?.free();
    S.session = new Session(instance, { d, rounds: 1 });
    S.pending = []; S.lit = new Map(); S.anim = null;
    out.d.textContent = String(d);
    canvas.setAttribute('aria-label', canvas.getAttribute('aria-label').replace(/distance-\d+/, `distance-${d}`));
  }

  const at = (gx, gy) => ({ x: S.L.originX + gx * S.L.unit, y: S.L.originY + gy * S.L.unit });

  /** Polygon (or half-disc) for a check, in grid units, computed once per layout. */
  function plaquette(st) {
    const support = S.session.support.get(st.idx);
    const pts = support.map((q) => S.session.dataQubits[q]).map((q) => ({ x: q.x, y: q.y }));
    pts.sort((a, b) => Math.atan2(a.y - st.y, a.x - st.x) - Math.atan2(b.y - st.y, b.x - st.x));
    return { st, pts };
  }

  function tracePlaquette(p) {
    const { st, pts } = p;
    ctx.beginPath();
    if (pts.length >= 3) {
      pts.forEach((g, i) => { const q = at(g.x, g.y); i ? ctx.lineTo(q.x, q.y) : ctx.moveTo(q.x, q.y); });
      ctx.closePath();
    } else if (pts.length === 2) {
      const a = at(pts[0].x, pts[0].y), b = at(pts[1].x, pts[1].y), c = at(st.x, st.y);
      const mid = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
      const outv = { x: c.x - mid.x, y: c.y - mid.y }, len = Math.hypot(outv.x, outv.y) || 1;
      const bulge = Math.hypot(b.x - a.x, b.y - a.y) * 0.55;
      ctx.moveTo(a.x, a.y);
      ctx.quadraticCurveTo(mid.x + (outv.x / len) * bulge, mid.y + (outv.y / len) * bulge, b.x, b.y);
      ctx.closePath();
    }
  }

  /* -- noise ------------------------------------------------------------- */
  function inject(q, pauli, t) {
    if (S.pending.length >= C.MAX_PENDING) return 0;
    if (pauli === 'X' || pauli === 'Y') S.session.toggleError(q, ERROR.X);
    if (pauli === 'Z' || pauli === 'Y') S.session.toggleError(q, ERROR.Z);
    S.pending.push({ q, pauli, t0: t });
    return 1;
  }
  function refreshLit(t) {
    const { defects } = S.session.read();
    for (let i = 0; i < defects.length; i++) {
      if (defects[i]) { if (!S.lit.has(i)) S.lit.set(i, t); } else S.lit.delete(i);
    }
  }
  function tick() {
    if (!S.running) return;
    const t = now(), dt = Math.min(0.2, (t - S.lastTick) / 1000);
    S.lastTick = t;
    let n = 0;
    const k = poisson(C.AMBIENT * dt);
    for (let i = 0; i < k; i++) n += inject(Math.floor(Math.random() * S.session.numData), pickPauli(Math.random, C.PAULI), t);
    if (t - S.pointer.t < 400) {
      const gx = (S.pointer.x - S.L.originX) / S.L.unit, gy = (S.pointer.y - S.L.originY) / S.L.unit;
      for (const q of S.session.dataQubits) {
        const dist = Math.hypot((q.x - gx) / 2, (q.y - gy) / 2);
        const m = poisson(footprintRate(dist, C.CURSOR_SIGMA, C.CURSOR_PEAK) * dt);
        for (let j = 0; j < m; j++) n += inject(q.idx, pickPauli(Math.random, C.CURSOR_PAULI), t);
      }
      S.dirty = true;
    }
    if (n) { S.phys += n; refreshLit(t); S.dirty = true; }
    if (t >= S.nextRound) round(t);
    if (S.dirty || S.anim || t - S.flashAt < C.T_FLASH) frame();
    S.tick = setTimeout(tick, C.TICK);
  }

  function round(t) {
    S.rounds++;
    if (S.pending.length) {
      const t0 = now();
      const { failed, correctionX, correctionZ } = S.session.decode(DECODER.MWPM);
      S.decodeMs = now() - t0;
      const chains = chainsFromCorrection(S.session, correctionX, correctionZ);
      S.anim = { t0: t, chains, errors: S.pending, lit: [...S.lit.keys()], failed };
      S.session.clearErrors();
      S.pending = []; S.lit = new Map();
      if (failed) { S.logical++; S.flashAt = t; out.logic.classList.add('flare'); setTimeout(() => out.logic.classList.remove('flare'), C.T_FLASH); }
      S.dirty = true;
    }
    S.nextRound = t + C.ROUND * (0.85 + 0.3 * Math.random());
    readout();
  }

  function readout() {
    out.rounds.textContent = S.rounds.toLocaleString();
    out.phys.textContent = S.phys.toLocaleString();
    out.logic.textContent = S.logical.toLocaleString();
    out.ms.textContent = S.decodeMs == null ? '·' : `${S.decodeMs < 1 ? S.decodeMs.toFixed(2) : S.decodeMs.toFixed(1)} ms`;
  }

  /* -- drawing ----------------------------------------------------------- */
  const pauliColor = (p) => (p === 'X' ? colors.x : p === 'Z' ? colors.z : colors.y);
  const clamp = (v, a, b) => Math.min(b, Math.max(a, v));

  function drawStatic() {
    for (const p of S.plaquettes) {
      tracePlaquette(p);
      ctx.fillStyle = p.st.type === STAB.X ? colors.xSoft : colors.zSoft;
      ctx.fill();
      ctx.strokeStyle = colors.rule; ctx.lineWidth = 1; ctx.stroke();
    }
    const r = Math.max(2.2, Math.min(4, S.L.unit * 0.28));
    for (const q of S.session.dataQubits) {
      const c = at(q.x, q.y);
      ctx.beginPath(); ctx.arc(c.x, c.y, r, 0, Math.PI * 2);
      ctx.fillStyle = colors.surface; ctx.fill();
      ctx.strokeStyle = colors.ink3; ctx.lineWidth = 1; ctx.stroke();
    }
  }

  function drawLit(idx, alpha) {
    const p = S.plaquettes[idx];
    tracePlaquette(p);
    ctx.fillStyle = colors.defectSoft; ctx.globalAlpha = alpha; ctx.fill();
    ctx.strokeStyle = colors.defect; ctx.lineWidth = 1.5; ctx.stroke();
    const c = at(p.st.x, p.st.y);
    ctx.beginPath(); ctx.arc(c.x, c.y, Math.max(2.5, S.L.unit * 0.3), 0, Math.PI * 2);
    ctx.fillStyle = colors.defect; ctx.fill();
    ctx.globalAlpha = 1;
  }

  function drawError(e, grow, alpha) {
    const q = S.session.dataQubits[e.q], c = at(q.x, q.y);
    const r = Math.max(3, S.L.unit * (0.3 + 0.28 * grow));
    ctx.beginPath(); ctx.arc(c.x, c.y, r, 0, Math.PI * 2);
    ctx.globalAlpha = alpha; ctx.fillStyle = pauliColor(e.pauli); ctx.fill();
    if (r >= 6) {
      ctx.fillStyle = colors.surface; ctx.font = `700 ${Math.round(r * 1.15)}px "JetBrains Mono", monospace`;
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(e.pauli, c.x, c.y);
    }
    ctx.globalAlpha = 1;
  }

  function strokePartial(pts, frac) {
    const seg = []; let total = 0;
    for (let i = 1; i < pts.length; i++) { const l = Math.hypot(pts[i].x - pts[i - 1].x, pts[i].y - pts[i - 1].y); seg.push(l); total += l; }
    let left = total * frac;
    ctx.beginPath(); ctx.moveTo(pts[0].x, pts[0].y);
    for (let i = 1; i < pts.length && left > 0; i++) {
      if (left >= seg[i - 1]) { ctx.lineTo(pts[i].x, pts[i].y); left -= seg[i - 1]; }
      else { const f = left / seg[i - 1]; ctx.lineTo(pts[i - 1].x + (pts[i].x - pts[i - 1].x) * f, pts[i - 1].y + (pts[i].y - pts[i - 1].y) * f); left = 0; }
    }
    ctx.stroke();
  }

  function drawChains(chains, frac, alpha) {
    ctx.save(); ctx.globalAlpha = alpha; ctx.lineCap = 'round'; ctx.lineJoin = 'round';
    ctx.strokeStyle = colors.ok; ctx.lineWidth = Math.max(2, S.L.unit * 0.22);
    for (const ch of chains) {
      const pts = ch.qubits.map((q) => at(S.session.dataQubits[q].x, S.session.dataQubits[q].y));
      if (pts.length === 1) { ctx.beginPath(); ctx.arc(pts[0].x, pts[0].y, Math.max(5, S.L.unit * 0.45), 0, Math.PI * 2); ctx.stroke(); }
      else strokePartial(pts, frac);
    }
    ctx.restore();
  }

  function render(t) {
    ctx.clearRect(0, 0, S.width, S.height);
    S.dirty = false;
    drawStatic();
    for (const [idx, t0] of S.lit) drawLit(idx, clamp((t - t0) / C.T_DEF, 0, 1));
    for (const e of S.pending) drawError(e, clamp((t - e.t0) / C.T_ERR, 0, 1), 1);
    const a = S.anim;
    if (a) {
      const dt = t - a.t0, total = C.T_CHAIN + C.T_HOLD + C.T_FADE;
      if (dt >= total) S.anim = null;
      else {
        const alpha = dt < C.T_CHAIN + C.T_HOLD ? 1 : 1 - (dt - C.T_CHAIN - C.T_HOLD) / C.T_FADE;
        for (const idx of a.lit) drawLit(idx, alpha);
        for (const e of a.errors) drawError(e, 1, alpha);
        drawChains(a.chains, clamp(dt / C.T_CHAIN, 0, 1), alpha);
      }
    }
    const flash = t - S.flashAt;
    if (flash >= 0 && flash < C.T_FLASH) {
      ctx.save(); ctx.globalAlpha = 0.5 * (1 - flash / C.T_FLASH);
      ctx.strokeStyle = colors.fail; ctx.lineWidth = 6;
      ctx.strokeRect(S.L.originX - 3, S.L.originY - 3, S.L.size + 6, S.L.size + 6);
      ctx.restore();
    }
    if (t - S.pointer.t < 400) {
      const r = 2.4 * C.CURSOR_SIGMA * 2 * S.L.unit;
      const g = ctx.createRadialGradient(S.pointer.x, S.pointer.y, 0, S.pointer.x, S.pointer.y, r);
      g.addColorStop(0, colors.defectSoft); g.addColorStop(1, 'rgba(0,0,0,0)');
      ctx.save(); ctx.globalAlpha = 0.45; ctx.fillStyle = g; ctx.fillRect(S.pointer.x - r, S.pointer.y - r, 2 * r, 2 * r); ctx.restore();
    }
  }

  function frame() {
    if (S.raf) return;
    S.raf = requestAnimationFrame((t) => {
      S.raf = 0;
      render(t);
      if (S.anim || t - S.flashAt < C.T_FLASH || t - S.pointer.t < 400) frame();
    });
  }

  /* -- reduced motion: one composed frame ---------------------------------- */
  function still() {
    root.querySelector('.opener')?.classList.add('opener--still');
    const d = S.session.d, row = Math.floor(d / 2), col = Math.floor(d / 3);
    const t = now();
    for (let c = col; c < col + 3; c++) inject(row * d + c, 'X', t);
    for (let r = row - 5; r < row - 3; r++) inject(r * d + col + Math.floor(d / 2), 'Z', t);
    refreshLit(t);
    const { correctionX, correctionZ } = S.session.decode(DECODER.MWPM);
    S.anim = { t0: t - C.T_CHAIN, chains: chainsFromCorrection(S.session, correctionX, correctionZ), errors: S.pending, lit: [...S.lit.keys()], failed: false };
    S.pending = []; S.lit = new Map();
    S.session.clearErrors();
    render(t);                                         // one frame, held: anim is drawn at full and never advanced
    S.anim = null;
  }

  /* -- lifecycle ----------------------------------------------------------- */
  function start() {
    if (S.running || reduced || document.hidden || !S.onScreen) return;
    S.running = true; S.lastTick = now(); S.nextRound = now() + C.ROUND;
    clearTimeout(S.tick); S.tick = setTimeout(tick, C.TICK);
  }
  function stop() { S.running = false; clearTimeout(S.tick); }

  fit();
  readout();
  if (reduced) { still(); }
  else {
    canvas.addEventListener('pointermove', (e) => {
      const r = canvas.getBoundingClientRect();
      S.pointer = { x: e.clientX - r.left, y: e.clientY - r.top, t: now() };
      frame();
    }, { passive: true });
    canvas.addEventListener('pointerleave', () => { S.pointer.t = -Infinity; });
    new IntersectionObserver(([entry]) => { S.onScreen = entry.isIntersecting; if (S.onScreen) start(); else stop(); }, { threshold: 0.05 }).observe(canvas);
    document.addEventListener('visibilitychange', () => { if (document.hidden) stop(); else start(); });
    let rt = 0;
    addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(() => { fit(); frame(); }, 150); });
    start();
  }
  frame();

  return { stop, state: S };
}
```

- [ ] **Step 4: Boot it from `js/main.js`**

After `initAnatomy($('#anatomy'), instance);` and its siblings add, before them (the opener should start first):

```js
  import { initOpener } from './opener.js';        // at the top, with the other section imports
  …
  const opener = initOpener($('#overview'), instance);
  window.__opener = opener;                         // debug handle, as the figures have none; harmless
```

Place `initOpener` immediately after `instance = await instantiate()` succeeds, before `initAnatomy`.

- [ ] **Step 5: Check it in the browser**

Serve and run the CDP driver with this expression after a 6 s wait:

```js
const S = window.__opener.state;
const a = S.rounds; await new Promise(r => setTimeout(r, 3000));
return { d: S.session.d, roundsAdvanced: S.rounds > a, phys: S.phys, running: S.running,
  readout: document.querySelector('[data-opener-readout]').textContent.replace(/\s+/g, ' ').trim() };
```

Expected: `d: 15`, `roundsAdvanced: true`, `phys > 0`, readout showing `rounds N · physical errors N · … MWPM 0.xx ms · data noise N runs/s …` once vitals land. Screenshot the hero at 1440 px; then with `Emulation.setEmulatedMedia({ features: [{ name: 'prefers-reduced-motion', value: 'reduce' }] })` reload and confirm `running: false` and a chain is drawn (screenshot).

- [ ] **Step 6: Commit**

```bash
git add index.html css/styles.css js/opener.js js/main.js
git commit -m "the live lattice at the top, in place of the stat cards"
```

---

### Task 4: Ambient read-only figures

**Files:**
- Create: `js/ambient.js`
- Modify: `js/lattice.js` (`draw()` round loops; new `visibleRounds` field)
- Modify: `js/sections/anatomy.js`
- Modify: `js/sections/spacetime.js`

**Interfaces:**
- Produces: `ambient({ root, canvas, every, act }) → { stop, pause, resume }`. `act()` may return a promise; the next call is scheduled `every` ms after it settles. Stops for good on `pointerdown`/`keydown` on `canvas` or `change`/`click` on any `select, input, button` inside `root`. Never starts under reduced motion.
- `LatticeView.visibleRounds` (number, default `Infinity`): rounds `t >= visibleRounds` are not drawn.

- [ ] **Step 1: Write `js/ambient.js`**

```js
/**
 * A slow heartbeat for a read-only figure.
 *
 * Calls `act` every `every` milliseconds while the figure is on screen and
 * the tab is visible, waits for `act` to settle before counting the interval,
 * and stops for good the first time the reader touches the figure or any
 * control in its section — from then on the figure is theirs. Under reduced
 * motion it never starts.
 */
export function ambient({ root, canvas, every, act }) {
  const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
  let stopped = reduced, paused = false, onScreen = false, timer = 0, busy = false;

  const clear = () => { clearTimeout(timer); timer = 0; };
  const schedule = () => {
    clear();
    if (stopped || paused || !onScreen || document.hidden || busy) return;
    timer = setTimeout(async () => {
      timer = 0;
      if (stopped || paused || !onScreen || document.hidden) return;
      busy = true;
      try { await act(); } finally { busy = false; }
      schedule();
    }, every);
  };
  const stop = () => { stopped = true; clear(); };
  const pause = () => { paused = true; clear(); };
  const resume = () => { paused = false; schedule(); };

  if (!reduced) {
    new IntersectionObserver(([e]) => { onScreen = e.isIntersecting; schedule(); }, { threshold: 0.3 }).observe(canvas);
    document.addEventListener('visibilitychange', schedule);
    for (const type of ['pointerdown', 'keydown']) canvas.addEventListener(type, stop);
    for (const el of root.querySelectorAll('select, input, button')) {
      el.addEventListener('change', stop);
      el.addEventListener('click', stop);
    }
  }
  return { stop, pause, resume };
}
```

- [ ] **Step 2: Add `visibleRounds` to `LatticeView`**

In the constructor, after `this.hover = null;`: `this.visibleRounds = Infinity;`.
In `draw()`, change `for (let t = 0; t < rounds; t++) {` to `const shown = Math.min(rounds, this.visibleRounds);` … `for (let t = 0; t < shown; t++) {`. In `#drawWorldlines`, change `for (let t = 0; t < session.rounds - 1; t++)` to `for (let t = 0; t < Math.min(session.rounds, this.visibleRounds) - 1; t++)`. The call `if (rounds > 1) this.#drawWorldlines(state, nStab);` is unchanged.

- [ ] **Step 3: Anatomy breathes**

In `js/sections/anatomy.js`: import `DECODER` from `'../engine.js'` (alongside `Session`) and `ambient` from `'../ambient.js'`. After `rebuild();` at the end of `initAnatomy`, before the `return`:

```js
  // Ambient: one error, its correction, then clean — until the reader takes over.
  const wait = (ms) => new Promise((r) => setTimeout(r, ms));
  const heartbeat = ambient({
    root, canvas, every: 2800,
    act: async () => {
      if (view.hover) return;                         // the hover readout is the figure's job
      const q = Math.floor(Math.random() * session.numData);
      session.toggleError(q, Math.random() < 0.5 ? 0 : 1);
      view.draw();
      await wait(1100);
      if (!session.ptr) return;
      session.decode(DECODER.MWPM);
      view.draw();
      await wait(900);
      if (!session.ptr) return;
      session.clearErrors();
      view.draw();
    },
  });
  view.onHover = (target) => { describe(target); if (target) heartbeat.pause(); else heartbeat.resume(); };
```

and remove the earlier `view.onHover = describe;` line. `rebuild()` frees the session and makes a new one; the act reads `session` at call time, so it follows.

- [ ] **Step 4: Spacetime builds itself and shows a lying readout**

In `js/sections/spacetime.js`: import `ambient` from `'../ambient.js'`. At the end of `initSpacetime`, after `rebuild();`:

```js
  // Ambient: the stack reveals itself on first sight, then a readout lies now and then.
  const wait = (ms) => new Promise((r) => setTimeout(r, ms));
  let revealed = false;
  ambient({
    root, canvas, every: 4000,
    act: async () => {
      if (!revealed) {
        revealed = true;
        for (let k = 1; k <= session.rounds; k++) { view.visibleRounds = k; view.draw(); await wait(350); }
        view.visibleRounds = Infinity; view.draw();
        return;
      }
      const state = session.read();
      if (state.defects.some((v) => v)) return;       // the reader has left something in place
      const idx = Math.floor(Math.random() * session.numStab);
      const t = Math.max(0, Math.floor(session.rounds / 2) - 1);
      session.toggleMeasurementError(idx, t);
      update();
      await wait(2000);
      if (!session.ptr) return;
      session.clearErrors();
      update();
    },
  });
```

`update()` redraws and refreshes the readout; the verdict banner is left alone.

- [ ] **Step 5: Check it**

Serve; CDP expression after 2 s: scroll Figure 2 into view, wait 4.5 s, read `document.querySelector('[data-anatomy-canvas]').getAttribute('aria-label')` — it should at some sample read "1 qubit-rounds carry an error" or "checks are firing"; then dispatch `new PointerEvent('pointerdown')` on the canvas, wait 4 s, sample the label twice 3 s apart: both "No errors injected". Scroll Figure 5 into view; wait 2.5 s; `aria-label` mentions "over 3 rounds"; wait 6 s more and confirm at least one sample with "checks are firing".

- [ ] **Step 6: Commit**

```bash
git add js/ambient.js js/lattice.js js/sections/anatomy.js js/sections/spacetime.js
git commit -m "ambient figures: the lattice breathes, the spacetime stack builds itself"
```

---

### Task 5: Section 07 — plot first, table beneath

**Files:**
- Modify: `index.html` (section `#threshold`, before `<div class="table-wrap">`)
- Modify: `js/sections/threshold.js` (`initResultsTable`)
- Modify: `css/styles.css` (`.results-plots`)

**Interfaces:**
- Consumes: `Plot`, `plotLegend` from `plot.js`; `wilson` from `compute.js`.
- Produces: markup `[data-results-plot="0"]`, `[data-results-plot="1"]`, `[data-results-legend]`.

- [ ] **Step 1: Markup**

Insert before `  <div class="table-wrap">` in section 07:

```html
  <figure class="figure">
    <figcaption class="figure__cap">
      <span class="figure__num">Figure 5b</span>
      <span class="figure__title">the crossing, measured on load</span>
      <span class="figure__meta">Union-Find · 2,000 shots per point · Wilson 95%</span>
    </figcaption>
    <div class="figure__body">
      <div class="results-plots">
        <div class="rig__stage">
          <canvas data-results-plot="0" aria-label="Logical error rate against physical error rate under data noise, distances 3, 5 and 7"></canvas>
          <span class="results-plots__label">data noise only · perfect measurements</span>
        </div>
        <div class="rig__stage">
          <canvas data-results-plot="1" aria-label="Logical error rate against physical error rate under phenomenological noise, distances 3, 5 and 7"></canvas>
          <span class="results-plots__label">phenomenological · faulty measurements, T = d</span>
        </div>
      </div>
      <div class="legend tight" data-results-legend></div>
      <p class="rig__caption">
        Below the threshold the bigger patch is the lower curve; above it, the higher. The two panels
        are the two blocks of the table beneath, drawn as their cells are measured.
      </p>
    </div>
  </figure>
```

- [ ] **Step 2: Style**

Append to section 9 of the stylesheet:

```css
/* Two plots side by side over the results table. */
.results-plots { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: var(--s5); }
.results-plots__label { font-family: var(--font-mono); font-size: var(--t-micro); color: var(--ink-3); text-transform: uppercase; letter-spacing: 0.08em; }
@media (max-width: 720px) { .results-plots { grid-template-columns: minmax(0, 1fr); } }
```

- [ ] **Step 3: Draw the plots from the table's callback**

In `initResultsTable`, after `const results = new Map();`:

```js
  const plotCanvases = TABLE_MODELS.map((m) => $(`[data-results-plot="${m.noiseMode}"]`, root));
  const plots = plotCanvases.map((c) => {
    if (!c) return null;
    c.dataset.aspect = '0.7';
    return new Plot(c, {
      xLabel: 'physical error rate  p', yLabel: 'logical error rate  p_L',
      formatX: (v) => `${(v * 100).toFixed(0)}%`, formatY: (v) => `${(v * 100).toFixed(0)}%`, xTicks: 3,
    });
  });
  const plotLegendEl = $('[data-results-legend]', root);
  if (plotLegendEl) plotLegendEl.innerHTML = plotLegend(TABLE_DISTANCES.map((d) => ({ label: `d = ${d}`, color: SERIES_COLOR[d] })));

  function paintPlots() {
    TABLE_MODELS.forEach((model, i) => {
      const plot = plots[i];
      if (!plot) return;
      const series = TABLE_DISTANCES.map((d) => {
        const points = TABLE_PS
          .map((p) => ({ p, entry: results.get(key({ noiseMode: model.noiseMode, d, p })) }))
          .filter((x) => x.entry)
          .map(({ p, entry }) => { const ci = wilson(entry.pL, entry.runs); return { x: p, y: entry.pL, lo: ci.lo, hi: ci.hi }; });
        return { label: `d = ${d}`, color: SERIES_COLOR[d], points, line: points.length > 1 ? points.map(({ x, y }) => ({ x, y })) : undefined };
      });
      plot.render({ series, xRange: [0, TABLE_PS[TABLE_PS.length - 1] * 1.06], empty: 'Measuring…' });
    });
  }
  paintPlots();
```

and call `paintPlots();` wherever `paint();` is called after results land (the progress callback, the completion handler, and the failure handler).

- [ ] **Step 4: Check it**

Serve; wait for `[data-results-status]` to read "Complete" (≈ 40 s; poll every second up to 120 s in the CDP expression); then count points: both canvases have `aria-label` containing "d = 3 from", "d = 5 from", "d = 7 from". Screenshot section 07 at 1440 px and 390 px.

- [ ] **Step 5: Commit**

```bash
git add index.html css/styles.css js/sections/threshold.js
git commit -m "section 07: the crossing drawn live above the table"
```

---

### Task 6: The bench streams

**Files:**
- Create: `js/stream.js`
- Create: `js/meter.js`
- Modify: `tools/site-tests.mjs` (append)
- Modify: `js/worker.js` (`OPS.stream`, cancel handling)
- Modify: `js/compute.js` (`Compute.call` returns id; `Compute.cancel`)
- Modify: `js/sections/bench.js`
- Modify: `index.html` (bench: meter canvas, Stop)
- Modify: `css/styles.css` (`.meter-chart`)

**Interfaces:**
- Produces: `planChunks(total, { target = 60, min = 250 } = {}) → number[]`; worker op `stream` reporting `{ done, total, failures, seconds, chunkRunsPerSecond }` and resolving `{ runs, failures, rate, seconds, runsPerSecond, cancelled }`; message `{ op: 'cancel', id }`; `Compute.call` returns a promise with an `id` property; `Compute.cancel(id)`; `Meter(canvas).render({ total, samples, final })` where `samples = [{ done, rate, lo, hi }]`.

- [ ] **Step 1: Failing test for the chunk planner**

Append to `tools/site-tests.mjs` before the summary lines:

```js
import { planChunks } from '../js/stream.js';

test('planChunks covers the request exactly with no empty chunk', () => {
  for (const total of [1, 249, 250, 251, 5000, 200000]) {
    const chunks = planChunks(total);
    assert.equal(chunks.reduce((a, b) => a + b, 0), total, `total ${total}`);
    assert.ok(chunks.every((c) => c > 0), `total ${total} has an empty chunk`);
    assert.ok(chunks.length <= 61, `total ${total}: ${chunks.length} chunks`);
  }
  assert.deepEqual(planChunks(600), [250, 250, 100]);
});
```

(Move the `import` to the top of the file with the other import.)

- [ ] **Step 2: Run to verify it fails**

Run: `node tools/site-tests.mjs` — Expected: module not found for `js/stream.js`.

- [ ] **Step 3: Implement `js/stream.js`**

```js
/**
 * Chunking for a streamed Monte Carlo run: about `target` reports over the
 * whole run, never a chunk smaller than `min` shots (a report costs a message
 * and a repaint, and a 20-shot estimate tells nobody anything), and the last
 * chunk takes whatever is left.
 */
export function planChunks(total, { target = 60, min = 250 } = {}) {
  const size = Math.max(min, Math.ceil(total / target));
  const out = [];
  for (let done = 0; done < total; done += size) out.push(Math.min(size, total - done));
  return out;
}
```

- [ ] **Step 4: Run the tests** — Expected: `7 passed, 0 failed`.

- [ ] **Step 5: Worker: `stream` op and cancellation**

In `js/worker.js`: import `planChunks` from `'./stream.js'`. Add `const cancelled = new Set();` above `OPS`. Add the op:

```js
  /**
   * The bench's run, in chunks, reporting the running estimate after each so
   * the page can draw it converging. Between chunks the loop yields to the
   * event loop, which is what lets a cancel message land mid-run.
   */
  async stream(instance, config, report, control) {
    const total = config.runs;
    let done = 0, failures = 0, seconds = 0, last = 0;
    for (const n of planChunks(total)) {
      if (control.cancelled()) break;
      const r = runBenchmark(instance, { ...config, rounds: roundsFor(config), runs: n });
      done += n; failures += Math.round(r.rate * n); seconds += r.seconds; last = r.runsPerSecond;
      report({ done, total, failures, seconds, chunkRunsPerSecond: last });
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
    return {
      runs: done, failures, rate: done ? failures / done : 0, seconds,
      runsPerSecond: seconds > 0 ? Math.round(done / seconds) : 0, cancelled: control.cancelled(),
    };
  },
```

Replace `self.onmessage` with:

```js
self.onmessage = async (event) => {
  const { id, op, payload } = event.data;
  if (op === 'cancel') { cancelled.add(id); return; }
  const report = (progress) => self.postMessage({ id, type: 'progress', ...progress });
  const control = { cancelled: () => cancelled.has(id) };

  try {
    const handler = OPS[op];
    if (!handler) throw new Error(`unknown operation "${op}"`);
    const instance = await engine();
    const result = await handler(instance, payload ?? { ...DEFAULT_RUN }, report, control);
    self.postMessage({ id, type: 'done', result });
  } catch (error) {
    self.postMessage({ id, type: 'error', message: error?.message ?? String(error) });
  } finally {
    cancelled.delete(id);
  }
};
```

- [ ] **Step 6: Client: ids and cancel**

In `js/compute.js`, `call` becomes:

```js
  call(op, payload, onProgress) {
    if (this.dead) return Promise.reject(new Error(this.dead));
    const id = this.nextId++;
    const promise = new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject, onProgress });
      this.worker.postMessage({ id, op, payload });
    });
    promise.id = id;
    return promise;
  }

  /** Ask a streaming job to stop after its current chunk; it resolves with what it has. */
  cancel(id) {
    if (this.dead || !this.pending.has(id)) return;
    this.worker.postMessage({ id, op: 'cancel' });
  }
```

- [ ] **Step 7: The meter renderer `js/meter.js`**

```js
/**
 * Strip chart for a streamed Monte Carlo run: the running logical error rate
 * against shots so far, with its Wilson band, so the reader watches the
 * estimate settle rather than waiting for a number to appear.
 */
function palette() {
  const css = getComputedStyle(document.documentElement);
  const get = (name, fallback) => (css.getPropertyValue(name).trim() || fallback);
  return {
    ink: get('--ink', '#1c1d1f'), ink2: get('--ink-2', '#4a4c50'), ink3: get('--ink-3', '#6b6d71'),
    rule: get('--rule-soft', '#ded9cd'), band: get('--defect-soft', '#fbeec6'), surface: get('--surface', '#fffefb'),
  };
}
const M = { top: 14, right: 16, bottom: 30, left: 50 };

export class Meter {
  constructor(canvas) {
    this.canvas = canvas; this.ctx = canvas.getContext('2d'); this.colors = palette(); this.data = null;
    this.#fit();
    new ResizeObserver(() => { if (this.#fit() && this.data) this.render(this.data); }).observe(canvas);
  }
  #fit() {
    const dpr = devicePixelRatio || 1, rect = this.canvas.getBoundingClientRect();
    const w = rect.width || 480, h = w * Number(this.canvas.dataset.aspect || 0.42);
    if (this.width === w && this.height === h) return false;
    this.canvas.style.height = `${h}px`; this.canvas.width = Math.round(w * dpr); this.canvas.height = Math.round(h * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0); this.width = w; this.height = h; return true;
  }
  /** @param {{total:number, samples:Array<{done:number, rate:number, lo:number, hi:number}>, final?:boolean, empty?:string}} data */
  render(data) {
    this.data = data;
    const { ctx, width, height, colors } = this;
    ctx.clearRect(0, 0, width, height);
    const s = data.samples ?? [];
    const pw = width - M.left - M.right, ph = height - M.top - M.bottom;
    if (!s.length) {
      ctx.fillStyle = colors.ink3; ctx.font = '400 12px Inter, sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(data.empty ?? 'Run to watch the estimate settle.', width / 2, height / 2);
      this.canvas.setAttribute('aria-label', data.empty ?? 'No run yet.');
      return;
    }
    let yMax = Math.max(0.02, ...s.map((p) => p.hi));
    yMax = Math.min(1, Math.ceil(yMax * 50) / 50);
    const sx = (v) => M.left + (v / data.total) * pw, sy = (v) => M.top + ph - (v / yMax) * ph;
    ctx.font = '400 10px "JetBrains Mono", monospace'; ctx.fillStyle = colors.ink3;
    ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    for (let i = 0; i <= 4; i++) {
      const v = (yMax * i) / 4, y = sy(v);
      ctx.beginPath(); ctx.moveTo(M.left, y); ctx.lineTo(width - M.right, y); ctx.strokeStyle = colors.rule; ctx.lineWidth = i ? 0.6 : 1; ctx.stroke();
      ctx.fillText(`${(v * 100).toFixed(v < 0.1 ? 1 : 0)}%`, M.left - 6, y);
    }
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    for (let i = 0; i <= 4; i++) ctx.fillText(Math.round((data.total * i) / 4).toLocaleString(), sx((data.total * i) / 4), M.top + ph + 6);
    ctx.fillStyle = colors.ink2; ctx.font = '500 11px Inter, sans-serif';
    ctx.fillText('shots', M.left + pw / 2, height - 13);
    // band
    ctx.beginPath();
    s.forEach((p, i) => (i ? ctx.lineTo(sx(p.done), sy(p.hi)) : ctx.moveTo(sx(p.done), sy(p.hi))));
    for (let i = s.length - 1; i >= 0; i--) ctx.lineTo(sx(s[i].done), sy(s[i].lo));
    ctx.closePath(); ctx.fillStyle = colors.band; ctx.fill();
    // estimate
    ctx.beginPath();
    s.forEach((p, i) => (i ? ctx.lineTo(sx(p.done), sy(p.rate)) : ctx.moveTo(sx(p.done), sy(p.rate))));
    ctx.strokeStyle = colors.ink; ctx.lineWidth = 1.6; ctx.stroke();
    const last = s[s.length - 1];
    ctx.beginPath(); ctx.arc(sx(last.done), sy(last.rate), 3, 0, Math.PI * 2); ctx.fillStyle = colors.ink; ctx.fill();
    if (data.final) {
      ctx.font = '600 10px "JetBrains Mono", monospace'; ctx.textAlign = 'right'; ctx.textBaseline = 'bottom'; ctx.fillStyle = colors.ink;
      ctx.fillText(`${(last.rate * 100).toFixed(2)}%`, width - M.right, sy(last.rate) - 6);
    }
    this.canvas.setAttribute('aria-label', `Logical error rate estimate after ${last.done.toLocaleString()} of ${data.total.toLocaleString()} shots: `
      + `${(last.rate * 100).toFixed(2)}%, 95% interval ${(last.lo * 100).toFixed(2)}% to ${(last.hi * 100).toFixed(2)}%.`);
  }
}
```

- [ ] **Step 8: Bench markup**

In section 09, replace

```html
      <div class="rig rig--wide">
        <div class="stack">
          <button class="btn btn--primary" data-bench-run>Run Monte Carlo</button>
          <p class="status" role="status" data-bench-status>Ready.</p>
        </div>
        <div class="readout" aria-live="polite" data-bench-output></div>
      </div>
```

with

```html
      <div class="rig rig--wide">
        <div class="rig__stage">
          <canvas class="meter-chart" data-bench-meter aria-label="No run yet."></canvas>
        </div>
        <div class="stack">
          <button class="btn btn--primary" data-bench-run>Run Monte Carlo</button>
          <p class="status" role="status" data-bench-status>Ready.</p>
          <div class="readout" aria-live="polite" data-bench-output></div>
        </div>
      </div>
```

Style: `.meter-chart { width: 100%; }` in section 9 of the stylesheet.

- [ ] **Step 9: Bench behaviour**

In `js/sections/bench.js`: import `{ Meter }` from `'../meter.js'`. Replace the `runBtn.addEventListener('click', …)` block with:

```js
  const meterCanvas = $('[data-bench-meter]', root);
  const meter = meterCanvas ? new Meter(meterCanvas) : null;
  meter?.render({ total: 1, samples: [] });
  let job = null;

  const row = (k, v) => el('div', { class: 'readout__row' }, [el('span', { class: 'readout__key', text: k }), el('span', { class: 'readout__val', text: v })]);

  runBtn.addEventListener('click', async () => {
    if (job) { compute.cancel(job.id); runBtn.disabled = true; status.textContent = 'Stopping after this chunk…'; return; }
    const config = readConfig(root);
    const samples = [];
    let meanRate = 0;
    runBtn.textContent = 'Stop';
    status.textContent = `Running ${count(config.runs)} shots…`;
    const paint = (p, final = false) => {
      const ci = wilson(p.done ? p.failures / p.done : 0, p.done);
      meter?.render({ total: config.runs, samples, final });
      fill(output, [
        row('logical error rate', p.done ? percent(p.failures / p.done) : '—'),
        row('95% interval', p.done ? `${percent(ci.lo)} – ${percent(ci.hi)}` : '—'),
        row('throughput', `${count(p.chunkRunsPerSecond)} shots/s · mean ${count(meanRate)}`),
        row('shots', `${count(p.done)} / ${count(config.runs)}`),
        row('wall time', `${p.seconds.toFixed(2)} s`),
      ]);
    };
    try {
      job = compute.call('stream', config, (p) => {
        const ci = wilson(p.failures / p.done, p.done);
        samples.push({ done: p.done, rate: p.failures / p.done, lo: ci.lo, hi: ci.hi });
        meanRate = p.seconds > 0 ? Math.round(p.done / p.seconds) : 0;
        paint(p);
      });
      const result = await job;
      meanRate = result.runsPerSecond;
      paint({ done: result.runs, failures: result.failures, seconds: result.seconds, chunkRunsPerSecond: samples.length ? result.runsPerSecond : 0 }, true);
      status.textContent = `${result.cancelled ? 'Stopped' : 'Done'} · ${DECODER_NAME[config.decoder]} · ${NOISE_NAME[config.noiseMode]} · d = ${config.d}`
        + (result.cancelled ? ` · ${count(result.runs)} of ${count(config.runs)} shots` : '');
    } catch (error) {
      status.textContent = `Failed: ${error.message}`;
    } finally {
      job = null;
      runBtn.textContent = 'Run Monte Carlo';
      runBtn.disabled = false;
    }
  });
```

- [ ] **Step 10: Check it**

Serve; CDP: click `[data-bench-run]`; after 400 ms read `[data-bench-run].textContent === 'Stop'` and the meter `aria-label` mentions "of 5,000 shots"; wait for the status to start with "Done" (poll up to 60 s) and confirm the readout's shots row reads `5,000 / 5,000`. Second run: click Run, wait 300 ms, click again (Stop), wait for status "Stopped": the shots row shows fewer than 5,000. No console errors. Screenshot the bench.

- [ ] **Step 11: Commit**

```bash
git add js/stream.js js/meter.js js/worker.js js/compute.js js/sections/bench.js index.html css/styles.css tools/site-tests.mjs
git commit -m "the bench streams: chunked runs, a converging estimate, Stop"
```

---

### Task 7: Whole-page verification and hand-off

**Files:**
- Modify: `README.md` (the "Web explainer" bullet and a line under it)

- [ ] **Step 1: Run every check**

`node tools/site-tests.mjs` → all pass. `node tools/contrast.mjs` → all ok. Serve; CDP with `Runtime.consoleAPICalled` + `Runtime.exceptionThrown` listeners over a 60 s load: zero errors. Screenshots at 1440 and 390 px: hero, Figure 2, section 07, bench. Reduced-motion reload: opener still, ambient never fires (anatomy label stays "No errors injected" over 8 s).

- [ ] **Step 2: README**

Change the "Web explainer" bullet to: `**Web explainer**: `index.html` plus `css/` and `js/`. No build step, no dependencies. The lattice at the top of the page runs the engine live; every figure is driven by it; the threshold table is plotted as it is measured and the bench streams its estimate.` Add under the tests section (or create one): `node tools/site-tests.mjs` for the site's pure modules, `node tools/contrast.mjs` for the palette.

- [ ] **Step 3: Commit and push**

```bash
git add README.md
git commit -m "docs: the explainer runs live; site tests"
git push origin main
```

Then confirm `https://qcompiler.jaspersands.com/js/opener.js` returns 200 (GitHub Pages; allow a couple of minutes, and note Cloudflare caches a 404 for three minutes if polled too early).

---

## Self-review

- Spec §1 opener → Task 3 (renderer, noise, rounds, readout, reduced motion, off-screen, engine failure: the boot-error path is unchanged and the canvas simply stays blank since `initOpener` is only called after `instantiate()` succeeds). Vitals moved → Task 3 step 1.
- Spec §2 figures → Task 1. Spec §3 ambient → Task 4. Spec §4 plot → Task 5. Spec §5 stream → Task 6. Spec §6 palette → Task 1. Testing → Tasks 2, 6, 7. Names checked: `planChunks`, `Meter.render`, `Compute.cancel`, `ambient(...)`, `visibleRounds`, `initOpener` are used with the same signatures everywhere.
