/**
 * Section 13 — beyond the surface code: IBM's bivariate bicycle codes.
 *
 * Figure 13 draws the gross code on its torus: every cell holds an X check,
 * two data qubits and a Z check, and a check's six data qubits are drawn out
 * from it (hover or tap a check to choose it), wrapping around the torus.
 * Figure 14 reads what tools/gross.py measured natively: logical error per
 * syndrome cycle against physical error, BP+OSD-CS and BP+OSD-0, for the gross
 * code and [[72, 12, 6]] (data/gross/results.json). Figure 15 samples and
 * decodes [[72, 12, 6]] here, by BP+OSD in WebAssembly, one batch of 64 shots
 * at a time on every worker.
 */

import { $, fill, el, whenNear } from '../dom.js';
import { Plot, plotLegend } from '../plot.js';
import { GROSS, neighbours, position, torusDelta } from '../bb-geometry.js';

const DATA = new URL('../../data/gross/', import.meta.url);
const LIVE = { code: 0, cycles: 6, p: 0.005, maxIter: 10000, batchesPerWorker: 5 };
const SERIES = [
  { code: 'gross', decoder: 'bposd_cs7', label: 'gross [[144, 12, 12]], BP+OSD-CS', color: 'var(--x)' },
  { code: 'gross', decoder: 'bposd_0', label: 'gross, BP+OSD-0', color: 'var(--x)', dashed: true },
  { code: '72', decoder: 'bposd_cs7', label: '[[72, 12, 6]], BP+OSD-CS', color: 'var(--z)' },
  { code: '72', decoder: 'bposd_0', label: '[[72, 12, 6]], BP+OSD-0', color: 'var(--z)', dashed: true },
];

const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

function sci(x) {
  if (!Number.isFinite(x) || x <= 0) return '—';
  const e = Math.floor(Math.log10(x));
  return `${(x / 10 ** e).toFixed(1)}×10${String(e).replace('-', '⁻').replace(/\d/g, (d) => '⁰¹²³⁴⁵⁶⁷⁸⁹'[d])}`;
}

/** 0.001 → "10⁻³". */
function power(v) {
  const e = Math.round(Math.log10(v));
  return e === 0 ? '1' : `10${String(e).replace('-', '⁻').replace(/\d/g, (d) => '⁰¹²³⁴⁵⁶⁷⁸⁹'[d])}`;
}

/** Wilson interval for k of n, 95%. */
function wilson(k, n) {
  if (!n) return [0, 1];
  const z = 1.96, p = k / n, den = 1 + (z * z) / n;
  const mid = (p + (z * z) / (2 * n)) / den;
  const half = (z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n))) / den;
  return [Math.max(0, mid - half), Math.min(1, mid + half)];
}

const perCycle = (pl, cycles) => 1 - (1 - pl) ** (1 / cycles);

/* -- Figure 13: the torus ------------------------------------------------- */

function drawTorus(canvas, code, chosen) {
  const dpr = window.devicePixelRatio || 1;
  const width = canvas.clientWidth || 800;
  const cols = 2 * code.l, rows = 2 * code.m;
  const pad = 18;
  const step = (width - 2 * pad) / cols;
  const height = rows * step + 2 * pad;
  canvas.style.height = `${height}px`;
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);
  const at = (col, row) => [pad + (col + 0.5) * step, pad + (row + 0.5) * step];
  const colors = { X: cssVar('--x'), Z: cssVar('--z'), ink: cssVar('--ink'), ink3: cssVar('--ink-3'), rule: cssVar('--rule-soft') };

  // The torus: the box wraps left to right and top to bottom.
  ctx.save();
  ctx.strokeStyle = colors.rule;
  ctx.setLineDash([4, 4]);
  ctx.lineWidth = 1;
  ctx.strokeRect(pad, pad, cols * step, rows * step);
  ctx.restore();

  // The chosen checks' edges, each drawn the short way round the torus.
  ctx.save();
  ctx.beginPath();
  ctx.rect(pad, pad, cols * step, rows * step);
  ctx.clip();
  const highlighted = new Set();
  for (const { type, cell } of chosen) {
    const from = position(code, type, cell);
    for (const nb of neighbours(code, cell, type)) {
      const to = position(code, nb.side, nb.cell);
      highlighted.add(`${nb.side}${nb.cell}`);
      const dc = torusDelta(from.col, to.col, cols), dr = torusDelta(from.row, to.row, rows);
      ctx.strokeStyle = colors[type];
      ctx.lineWidth = 1.6;
      ctx.globalAlpha = 0.85;
      for (const [a, b] of [[from, { col: from.col + dc, row: from.row + dr }], [{ col: to.col - dc, row: to.row - dr }, to]]) {
        ctx.beginPath();
        ctx.moveTo(...at(a.col, a.row));
        ctx.lineTo(...at(b.col, b.row));
        ctx.stroke();
      }
    }
  }
  ctx.restore();
  ctx.globalAlpha = 1;

  // The qubits: checks as squares, data as circles (left filled, right open).
  const r = Math.max(2.5, step * 0.17);
  for (let cell = 0; cell < code.l * code.m; cell++) {
    for (const kind of ['X', 'Z', 'L', 'R']) {
      const { col, row } = position(code, kind, cell);
      const [x, y] = at(col, row);
      const isChosen = chosen.some((c) => c.type === kind && c.cell === cell);
      if (kind === 'X' || kind === 'Z') {
        const s = isChosen ? r * 1.9 : r * 1.25;
        ctx.fillStyle = colors[kind];
        ctx.globalAlpha = isChosen ? 1 : 0.35;
        ctx.fillRect(x - s, y - s, 2 * s, 2 * s);
        ctx.globalAlpha = 1;
      } else {
        const lit = highlighted.has(`${kind}${cell}`);
        ctx.beginPath();
        ctx.arc(x, y, lit ? r * 1.35 : r, 0, 2 * Math.PI);
        ctx.lineWidth = lit ? 2 : 1.2;
        ctx.strokeStyle = lit ? colors.ink : colors.ink3;
        ctx.fillStyle = kind === 'L' ? (lit ? colors.ink : colors.ink3) : cssVar('--surface');
        ctx.fill();
        ctx.stroke();
      }
    }
  }
  return { step, pad };
}

function initTorus(fig) {
  const canvas = $('[data-bb-canvas]', fig);
  const code = GROSS;
  let chosen = [{ type: 'X', cell: 4 * code.m + 2 }, { type: 'Z', cell: 8 * code.m + 3 }];
  let geometry = drawTorus(canvas, code, chosen);
  const redraw = () => { geometry = drawTorus(canvas, code, chosen); };
  new ResizeObserver(redraw).observe(canvas);
  const pick = (event) => {
    const rect = canvas.getBoundingClientRect();
    const col = Math.floor((event.clientX - rect.left - geometry.pad) / geometry.step);
    const row = Math.floor((event.clientY - rect.top - geometry.pad) / geometry.step);
    if (col < 0 || row < 0 || col >= 2 * code.l || row >= 2 * code.m) return;
    const type = col % 2 === 0 && row % 2 === 0 ? 'X' : col % 2 === 1 && row % 2 === 1 ? 'Z' : null;
    if (!type) return;
    const cell = Math.floor(col / 2) * code.m + Math.floor(row / 2);
    chosen = [...chosen.filter((c) => c.type !== type), { type, cell }];
    redraw();
    $('[data-bb-note]', fig).textContent = `${type} check at cell (${Math.floor(col / 2)}, ${Math.floor(row / 2)}): `
      + `its six data qubits, three on each side, reached ${type === 'X' ? 'through A = x³ + y + y² and B = y³ + x + x²' : 'through Bᵀ and Aᵀ'}.`;
  };
  canvas.addEventListener('pointermove', pick);
  canvas.addEventListener('pointerdown', pick);
}

/* -- Figure 14: recorded -------------------------------------------------- */

async function runRecorded(fig) {
  const status = $('[data-bb-fit-status]', fig);
  status.textContent = 'Loading the recorded measurements…';
  let doc;
  try {
    const response = await fetch(new URL('results.json', DATA));
    if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
    doc = await response.json();
  } catch (error) {
    status.textContent = `Recorded measurements unavailable (${error.message}).`;
    return;
  }
  const points = Object.values(doc.points);
  const series = SERIES.map((s) => {
    const pts = points.filter((q) => q.code === s.code && q.results[s.decoder]?.failures > 0)
      .sort((a, b) => a.p - b.p)
      .map((q) => {
        const r = q.results[s.decoder];
        return { x: q.p, y: r.pl_cycle, lo: r.pl_cycle_interval[0], hi: r.pl_cycle_interval[1] };
      });
    return { label: s.label, color: s.color, points: pts, line: pts, dashed: s.dashed };
  });
  const ps = [...new Set(points.map((q) => q.p))].sort((a, b) => a - b);
  series.push({
    label: '12 unprotected qubits, 1 − (1 − p)¹²',
    color: 'var(--ink-3)',
    points: [],
    line: ps.map((p) => ({ x: p, y: 1 - (1 - p) ** 12 })),
  });
  const plot = new Plot($('[data-bb-plot]', fig), {
    xLabel: 'physical error rate p',
    yLabel: 'logical error per syndrome cycle',
    yLog: true,
    xTickValues: ps,
    formatX: (v) => `${(v * 100).toFixed(1)}%`,
    // Decades only, as powers of ten: the axis spans five of them.
    formatY: (v) => (Math.abs(Math.log10(v) - Math.round(Math.log10(v))) < 1e-9 ? power(v) : ''),
  });
  plot.render({ series, xRange: [ps[0] - 0.0004, ps[ps.length - 1] + 0.0004] });
  $('[data-bb-legend]', fig).innerHTML = plotLegend(series);

  const rows = points.sort((a, b) => (a.code === b.code ? a.p - b.p : a.code === 'gross' ? -1 : 1)).map((q) => {
    const cs = q.results.bposd_cs7, o0 = q.results.bposd_0;
    return el('tr', {}, [
      el('th', { scope: 'row', text: `${q.code === 'gross' ? 'gross' : '[[72, 12, 6]]'}, ${q.cycles} cycles` }),
      el('td', { class: 'num', text: `${(q.p * 100).toFixed(1)}%` }),
      el('td', { class: 'num', text: q.shots.toLocaleString('en-US') }),
      el('td', { class: 'num', text: cs.failures.toLocaleString('en-US') }),
      el('td', { class: 'num' }, [document.createTextNode(sci(cs.pl_cycle)),
        el('span', { class: 'ci', text: `${sci(cs.pl_cycle_interval[0])}–${sci(cs.pl_cycle_interval[1])}` })]),
      el('td', { class: 'num', text: sci(o0.pl_cycle) }),
      el('td', { class: 'num', text: `${((cs.converged / q.shots) * 100).toFixed(1)}%` }),
    ]);
  });
  fill($('[data-bb-rows]', fig), rows);
  $('[data-bb-fit-foot]', fig).textContent = `† Measured natively on ${doc.machine.cpu} by tools/gross.py (engine `
    + `${doc.engine_commit}). Each shot is the paper's Z-basis memory: the data prepared in |0⟩, N_c syndrome cycles `
    + '(12 for the gross code, 6 for [[72, 12, 6]]) of the depth-8 circuit under circuit noise p, then the data read '
    + 'out. A shot fails if any of the 12 logical qubits comes out wrong, and the error per cycle is '
    + '1 − (1 − P_L)^(1/N_c), with a 95% Wilson interval. Both decoders see the same shots: BP+OSD-CS of order 7 and '
    + 'BP+OSD-0, min-sum BP with adaptive scaling for up to 10,000 iterations, the paper\'s settings. A point stops at '
    + '200 failures, or at its shot or time cap. Where the curve crosses the grey line, the code does as well as 12 '
    + 'unprotected qubits: its pseudo-threshold.';
  status.textContent = 'Loaded.';
}

/* -- Figure 15: live ------------------------------------------------------ */

async function runLive(fig, compute) {
  const status = $('[data-bb-live-status]', fig);
  const out = (name, text) => { $(`[data-bb-live-${name}]`, fig).textContent = text; };
  const workers = compute.size ?? 1;
  const jobs = Array.from({ length: workers }, () => ({ config: LIVE, batches: LIVE.batchesPerWorker }));
  const progress = new Array(workers).fill(null).map(() => ({ shots: 0, failures: 0 }));
  status.textContent = `Decoding on ${workers} workers…`;
  let results;
  try {
    results = await compute.map('bbrun', jobs, (i, p) => {
      progress[i] = p;
      const shots = progress.reduce((s, q) => s + q.shots, 0);
      const failures = progress.reduce((s, q) => s + q.failures, 0);
      status.textContent = `${shots.toLocaleString('en-US')} shots, ${failures} failures so far…`;
    });
  } catch (error) {
    status.textContent = `Failed: ${error.message}`;
    return;
  }
  const shots = results.reduce((s, r) => s + r.shots, 0);
  const failures = results.reduce((s, r) => s + r.failures, 0);
  const converged = results.reduce((s, r) => s + r.converged, 0);
  const seconds = results.reduce((s, r) => s + r.seconds, 0);
  const [lo, hi] = wilson(failures, shots);
  out('shots', shots.toLocaleString('en-US'));
  out('failures', failures.toLocaleString('en-US'));
  out('rate', `${sci(perCycle(failures / shots, LIVE.cycles))} (${sci(perCycle(lo, LIVE.cycles))}–${sci(perCycle(hi, LIVE.cycles))})`);
  out('bp', `${((converged / shots) * 100).toFixed(1)}%`);
  out('time', `${((seconds / shots) * 1000).toFixed(1)} ms`);
  out('model', `${results[0].detectors.toLocaleString('en-US')} detectors, ${results[0].faults.toLocaleString('en-US')} faults`);
  status.textContent = 'Done.';
}

export function initGross(root, compute) {
  const torusFig = $('[data-bb-torus]', root);
  const fitFig = $('[data-bb-fit]', root);
  const liveFig = $('[data-bb-live]', root);
  if (!torusFig || !fitFig || !liveFig) return;
  initTorus(torusFig);
  whenNear([fitFig, liveFig], () => runRecorded(fitFig).catch(() => {}).then(() => runLive(liveFig, compute)));
}
