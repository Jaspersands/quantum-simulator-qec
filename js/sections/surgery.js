/**
 * Section 14 — lattice surgery: two patches merged to measure Z⊗Z.
 *
 * Figure 16 draws the experiment phase by phase (prepare, merge, split, read
 * out) at d = 5: which checks are measured, the seam, and the new checks
 * along it whose product is the outcome. Figure 17 reads what
 * tools/surgery.py measured natively: how often the merge outcome is wrong
 * against the number of merged rounds T, at d = 3, 5, 7 (data/surgery/
 * results.json). Figure 18 samples and matches the whole experiment here.
 */

import { $, fill, el, whenNear } from '../dom.js';
import { Plot, plotLegend } from '../plot.js';
import { surgeryLayout } from '../surgery-geometry.js';

const DATA = new URL('../../data/surgery/', import.meta.url);
const COLORS = { 3: 'var(--d3)', 5: 'var(--d5)', 7: 'var(--d7)' };
const LIVE = { p: 0.003, batches: 40, configs: [{ d: 3, merged: 2 }, { d: 3, merged: 3 }, { d: 5, merged: 2 }, { d: 5, merged: 5 }] };

const PHASES = [
  {
    name: 'Prepare',
    text: 'Each patch is prepared in |0⟩ and measured on its own for d rounds. Its logical Z runs down a column, '
      + 'so the patches face each other across the seam with the boundaries whose small checks are X-type.',
  },
  {
    name: 'Merge',
    text: 'The seam is prepared in |+⟩ and the merged patch is measured for T rounds. The boundary X checks reach '
      + 'across onto the seam and stay predictable. The new Z checks along the seam (outlined) are each random, but '
      + 'their product is Z₁Z₂: the measurement. Repeating them for T rounds is what protects it.',
  },
  {
    name: 'Split',
    text: 'The seam is read out in X and the patches are measured apart again for d rounds. Each boundary X check '
      + 'is predicted by its last merged value times the two seam qubits it lost.',
  },
  {
    name: 'Read out',
    text: 'The data are read out in Z. The observables are the merge outcome and each patch\'s own Z, which '
      + 'the Z₁Z₂ measurement leaves alone.',
  },
];

const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

function drawPhase(canvas, g, phase) {
  const dpr = window.devicePixelRatio || 1;
  const width = canvas.clientWidth || 800;
  const pad = 22;
  const step = (width - 2 * pad) / g.width;
  const height = g.height * step + 2 * pad;
  canvas.style.height = `${height}px`;
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);
  const at = (x, y) => [pad + x * step, pad + y * step];
  const c = { X: cssVar('--x'), Z: cssVar('--z'), Xs: cssVar('--x-soft'), Zs: cssVar('--z-soft'), ink: cssVar('--ink'), ink3: cssVar('--ink-3'), rule: cssVar('--rule-soft'), amber: cssVar('--defect') };
  const merged = phase === 1;
  const checks = merged ? g.merged : g.patches;
  const isNew = new Set(g.newZ.map((q) => `${q.x},${q.y}`));

  // Plaquettes as diamonds (the rotated code's checks sit on the vertices of the data lattice).
  for (const ch of checks) {
    const pts = ch.support.map(([x, y]) => at(x, y));
    const [cx, cy] = at(ch.x, ch.y);
    ctx.beginPath();
    if (pts.length === 4) {
      const order = [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([dx, dy]) => at(ch.x + dx, ch.y + dy));
      order.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
    } else {
      ctx.moveTo(cx, cy);
      pts.forEach(([x, y]) => ctx.lineTo(x, y));
    }
    ctx.closePath();
    const fresh = merged && isNew.has(`${ch.x},${ch.y}`);
    ctx.fillStyle = ch.xType ? c.Xs : c.Zs;
    ctx.fill();
    ctx.strokeStyle = fresh ? c.ink : c.rule;
    ctx.lineWidth = fresh ? 2 : 1;
    ctx.stroke();
  }

  // The logical Z strings of each patch, and the seam.
  ctx.save();
  ctx.setLineDash([5, 4]);
  ctx.lineWidth = 1.5;
  ctx.strokeStyle = c.Z;
  for (const x of [g.p1[1], g.p2[0]]) {
    ctx.beginPath();
    ctx.moveTo(...at(x, 0.4));
    ctx.lineTo(...at(x, g.height - 0.4));
    ctx.stroke();
  }
  ctx.restore();
  ctx.fillStyle = c.Z;
  ctx.font = '600 11px "JetBrains Mono", monospace';
  ctx.textAlign = 'center';
  ctx.fillText('Z₁', ...at(g.p1[1], 0.1));
  ctx.fillText('Z₂', ...at(g.p2[0], 0.1));

  // Data qubits; the seam only while it is in use.
  const r = Math.max(2.5, step * 0.16);
  for (const q of g.data) {
    const live = !q.seam || phase === 1;
    const [x, y] = at(q.x, q.y);
    ctx.beginPath();
    ctx.arc(x, y, r, 0, 2 * Math.PI);
    ctx.fillStyle = live ? (q.seam ? c.amber : c.ink) : cssVar('--surface');
    ctx.strokeStyle = live ? (q.seam ? c.amber : c.ink) : c.rule;
    ctx.lineWidth = 1.2;
    ctx.fill();
    ctx.stroke();
  }
}

function initDiagram(fig) {
  const canvas = $('[data-ls-canvas]', fig);
  const g = surgeryLayout(5);
  let phase = 1;
  const buttons = $('[data-ls-phases]', fig);
  const note = $('[data-ls-note]', fig);
  const show = (k) => {
    phase = k;
    drawPhase(canvas, g, phase);
    note.textContent = PHASES[k].text;
  };
  fill(buttons, PHASES.map((ph, i) => el('label', { class: 'segmented__opt' }, [
    el('input', { type: 'radio', name: 'ls-phase', value: String(i), checked: i === phase, onchange: () => show(i) }),
    el('span', { text: `${i + 1}. ${ph.name}` }),
  ])));
  new ResizeObserver(() => drawPhase(canvas, g, phase)).observe(canvas);
  show(phase);
}

async function runRecorded(fig) {
  const status = $('[data-ls-fit-status]', fig);
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
  const series = [];
  for (const p of [0.003, 0.002]) {
    for (const d of [3, 5, 7]) {
      const pts = points.filter((q) => q.d === d && q.p === p && q.matcher === 'correlated')
        .sort((a, b) => a.merged - b.merged)
        .map((q) => ({ x: q.merged, y: q.rate_outcome, lo: q.interval_outcome[0], hi: q.interval_outcome[1] }));
      if (pts.length) series.push({ label: `d = ${d}, p = ${(p * 100).toFixed(1)}%`, color: COLORS[d], points: pts, line: pts, dashed: p === 0.002 });
    }
  }
  const plot = new Plot($('[data-ls-plot]', fig), {
    xLabel: 'merged rounds T',
    yLabel: 'merge outcome wrong, per shot',
    yLog: true,
    xTickValues: [2, 4, 6, 8, 10, 12, 14],
    formatX: (v) => `${v}`,
    formatY: (v) => `${(v * 100).toPrecision(1)}%`,
  });
  plot.render({ series, xRange: [1.5, 14.5], markers: [3, 5, 7].map((d) => ({ x: d, label: `T = ${d}` })) });
  $('[data-ls-legend]', fig).innerHTML = plotLegend(series);

  const rows = [];
  for (const p of [0.003, 0.002]) {
    for (const d of [3, 5, 7]) {
      const get = (T, m) => points.find((q) => q.d === d && q.p === p && q.merged === T && q.matcher === m);
      const cells = [2, d, 2 * d].map((T) => get(T, 'correlated'));
      if (cells.some((q) => !q)) continue;
      const plain = get(d, 'plain');
      const fmt = (q) => `${(q.rate_outcome * 100).toFixed(2)}%`;
      rows.push(el('tr', {}, [
        el('th', { scope: 'row', text: `d = ${d}, p = ${(p * 100).toFixed(1)}%` }),
        ...cells.map((q) => el('td', { class: 'num', text: fmt(q) })),
        el('td', { class: 'num', text: plain ? fmt(plain) : '—' }),
        el('td', { class: 'num', text: `${(cells[1].rate_patches * 100).toFixed(2)}%` }),
      ]));
    }
  }
  fill($('[data-ls-rows]', fig), rows);
  $('[data-ls-fit-foot]', fig).textContent = `† Measured natively on ${doc.machine.cpu} by tools/surgery.py (engine `
    + `${doc.engine_commit}). Each point: shots of the whole experiment (d rounds apart, T merged, d apart, read out; `
    + 'SD6 noise), sampled by the bit-parallel sampler and decoded by correlated matching (plain in its own column) '
    + 'on this engine\'s error model, which equals Stim\'s fault for fault; a point stops at 1,500 failures or its '
    + 'caps, and bars are 95% Wilson intervals. One merged round is not drawn: a single measurement error then flips '
    + 'the outcome with nothing after it to notice, so the error model refuses the circuit.';
  status.textContent = 'Loaded.';
}

async function runLive(fig, compute) {
  const status = $('[data-ls-live-status]', fig);
  const body = $('[data-ls-live-rows]', fig);
  const rowFor = new Map();
  fill(body, LIVE.configs.map((cfg) => {
    const tds = [el('td', { class: 'num', text: '—' }), el('td', { class: 'num', text: '—' }), el('td', { class: 'num', text: '—' })];
    rowFor.set(`${cfg.d}/${cfg.merged}`, tds);
    return el('tr', {}, [el('th', { scope: 'row', text: `d = ${cfg.d}, T = ${cfg.merged}` }), ...tds]);
  }));
  status.textContent = `Sampling and matching on ${compute.size ?? 1} workers…`;
  try {
    const jobs = LIVE.configs.map((cfg) => ({ config: { ...cfg, p: LIVE.p, correlated: true }, batches: LIVE.batches }));
    const results = await compute.map('surgery', jobs, (i, pr) => {
      const [shots] = rowFor.get(`${LIVE.configs[i].d}/${LIVE.configs[i].merged}`);
      shots.textContent = pr.shots.toLocaleString('en-US');
    });
    for (const r of results) {
      const [shots, outcome, time] = rowFor.get(`${r.d}/${r.merged}`);
      shots.textContent = r.shots.toLocaleString('en-US');
      outcome.textContent = `${r.outcome} (${((r.outcome / r.shots) * 100).toFixed(1)}%)`;
      time.textContent = `${((r.seconds / r.shots) * 1e6).toFixed(0)} µs`;
    }
    status.textContent = 'Done.';
  } catch (error) {
    status.textContent = `Failed: ${error.message}`;
  }
}

export function initSurgery(root, compute) {
  const diagram = $('[data-ls-diagram]', root);
  const fitFig = $('[data-ls-fit]', root);
  const liveFig = $('[data-ls-live]', root);
  if (!diagram || !fitFig || !liveFig) return;
  initDiagram(diagram);
  whenNear([fitFig, liveFig], () => runRecorded(fitFig).catch(() => {}).then(() => runLive(liveFig, compute)));
}
