/**
 * Section 14 — lattice surgery: two patches merged to measure Z⊗Z, and a
 * logical CNOT built from two such measurements.
 *
 * Figure 18 draws the Z⊗Z experiment phase by phase at d = 5. Figure 19
 * reads what tools/surgery.py measured natively: how often the merge outcome
 * is wrong against the number of merged rounds T (data/surgery/results.json).
 * Figure 20 draws the CNOT's program step by step, and Figure 21 reads its
 * measured failure, beside three patches held idle as long, and the law of
 * merges done in a row (data/surgery/programs.json). Figure 22 samples and
 * matches the Z⊗Z experiment and the CNOT here.
 */

import { $, fill, el, whenNear } from '../dom.js';
import { Plot, plotLegend } from '../plot.js';
import { surgeryLayout, cnotLayout } from '../surgery-geometry.js';

const DATA = new URL('../../data/surgery/', import.meta.url);
const COLORS = { 3: 'var(--d3)', 5: 'var(--d5)', 7: 'var(--d7)' };
const LIVE = {
  p: 0.003,
  batches: 40,
  configs: [
    { kind: 0, d: 3, merged: 2, label: 'Z⊗Z, d = 3, T = 2: the outcome' },
    { kind: 0, d: 3, merged: 3, label: 'Z⊗Z, d = 3, T = 3: the outcome' },
    { kind: 0, d: 5, merged: 2, label: 'Z⊗Z, d = 5, T = 2: the outcome' },
    { kind: 0, d: 5, merged: 5, label: 'Z⊗Z, d = 5, T = 5: the outcome' },
    { kind: 1, d: 3, merged: 3, label: 'CNOT on |0⟩|0⟩, d = 3: either observable' },
    { kind: 2, d: 3, merged: 3, label: 'CNOT on |+⟩|+⟩, d = 3: either observable' },
  ],
};

const CNOT_STEPS = [
  {
    name: 'Prepare',
    text: 'The control C and target T are prepared (here in |0⟩), the ancilla A in |+⟩, and each is measured on its '
      + 'own for d rounds.',
  },
  {
    name: 'Z_C Z_A',
    text: 'C and A merge side by side: the seam is prepared in |+⟩, and the new Z checks along it (outlined) multiply '
      + 'to Z on C\'s last column and A\'s first, Z_C Z_A. Its outcome m₁ is random; T is measured alone meanwhile.',
  },
  {
    name: 'X_A X_T',
    text: 'The C–A seam is read in X and parted; A and T merge one above the other, across a row of seam qubits in '
      + '|0⟩. The new X checks multiply to X on A\'s last row and T\'s first, X_A X_T, outcome m₂.',
  },
  {
    name: 'Read A',
    text: 'The A–T seam is read in Z and parted, and A is read out in Z (m₃). What is left is CNOT from C to T, up to '
      + 'Paulis fixed by m₁, m₂, m₃, which are tracked, not applied.',
  },
  {
    name: 'Read out',
    text: 'After d more rounds, C and T are read out. With |0⟩|0⟩ in, Z_C and Z_T ⊕ Z_A ⊕ m₁ (with the A–T seam\'s '
      + 'Z records on that column) must both come out +1; with |+⟩|+⟩ in, X_T and X_C X_T ⊕ m₂, X_C taken along the '
      + 'last row with the C–A seam\'s X record there.',
  },
];

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
  const rows = LIVE.configs.map(() => [el('td', { class: 'num', text: '—' }), el('td', { class: 'num', text: '—' }), el('td', { class: 'num', text: '—' })]);
  fill(body, LIVE.configs.map((cfg, i) => el('tr', {}, [el('th', { scope: 'row', text: cfg.label }), ...rows[i]])));
  status.textContent = `Sampling and matching on ${compute.size ?? 1} workers…`;
  try {
    const jobs = LIVE.configs.map(({ kind, d, merged }) => ({ config: { kind, d, merged, p: LIVE.p, correlated: true }, batches: LIVE.batches }));
    const results = await compute.map('surgery', jobs, (i, pr) => {
      rows[i][0].textContent = pr.shots.toLocaleString('en-US');
    });
    results.forEach((r, i) => {
      const [shots, wrong, time] = rows[i];
      // The Z⊗Z rows count the merge outcome; the CNOT rows a shot with either observable wrong.
      const n = r.kind === 0 ? r.outcome : r.any;
      shots.textContent = r.shots.toLocaleString('en-US');
      wrong.textContent = `${n} (${((n / r.shots) * 100).toFixed(1)}%)`;
      time.textContent = `${((r.seconds / r.shots) * 1e6).toFixed(0)} µs`;
    });
    status.textContent = 'Done.';
  } catch (error) {
    status.textContent = `Failed: ${error.message}`;
  }
}

function drawCnot(canvas, g, step) {
  const dpr = window.devicePixelRatio || 1;
  const width = canvas.clientWidth || 800;
  const pad = 22;
  const unit = Math.min((width - 2 * pad) / g.width, 34);
  const height = g.height * unit + 2 * pad;
  const left = (width - g.width * unit) / 2;
  canvas.style.height = `${height}px`;
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);
  const at = (x, y) => [left + x * unit, pad + y * unit];
  const c = { Xs: cssVar('--x-soft'), Zs: cssVar('--z-soft'), ink: cssVar('--ink'), ink3: cssVar('--ink-3'), rule: cssVar('--rule-soft'), amber: cssVar('--defect'), surface: cssVar('--surface') };
  const key = (q) => `${q.x},${q.y}`;
  const sets = [
    [...g.patches.C, ...g.patches.A, ...g.patches.T],
    [...g.ca, ...g.patches.T],
    [...g.patches.C, ...g.at],
    [...g.patches.C, ...g.patches.T],
    [...g.patches.C, ...g.patches.T],
  ];
  const fresh = new Set((step === 1 ? g.newCA : step === 2 ? g.newAT : []).map(key));
  for (const ch of sets[step]) {
    ctx.beginPath();
    if (ch.support.length === 4) {
      [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([dx, dy]) => at(ch.x + dx, ch.y + dy)).forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
    } else {
      ctx.moveTo(...at(ch.x, ch.y));
      ch.support.forEach(([x, y]) => ctx.lineTo(...at(x, y)));
    }
    ctx.closePath();
    ctx.fillStyle = ch.xType ? c.Xs : c.Zs;
    ctx.fill();
    const f = fresh.has(key(ch));
    ctx.strokeStyle = f ? c.ink : c.rule;
    ctx.lineWidth = f ? 2 : 1;
    ctx.stroke();
  }
  const r = Math.max(2.5, unit * 0.16);
  for (const q of g.data) {
    const seamLive = (q.role === 'ca' && step === 1) || (q.role === 'at' && step === 2);
    const gone = (q.role === 'A' && step >= 3) || ((q.role === 'ca' || q.role === 'at') && !seamLive);
    const [x, y] = at(q.x, q.y);
    ctx.beginPath();
    ctx.arc(x, y, r, 0, 2 * Math.PI);
    ctx.fillStyle = gone ? c.surface : seamLive ? c.amber : c.ink;
    ctx.strokeStyle = gone ? c.rule : seamLive ? c.amber : c.ink;
    ctx.lineWidth = 1.2;
    ctx.fill();
    ctx.stroke();
  }
  ctx.fillStyle = c.ink3;
  ctx.font = '600 13px "JetBrains Mono", monospace';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  // Each patch's name on a check's face, one step from the tile's centre, clear of the data.
  for (const [name, b] of [['C', g.C], ['A', g.A], ['T', g.T]]) {
    const [x, y] = at((b.x0 + b.x1) / 2 + 1, (b.y0 + b.y1) / 2 + 1);
    ctx.fillStyle = name === 'A' && step >= 3 ? c.rule : c.ink3;
    ctx.fillText(name, x, y);
  }
}

/** Text with each X_C or Z_T written as the operator with its patch as a subscript. */
const subscripted = (text) => text.split(/([XZ])_([A-Z])/).map((part, i) => (i % 3 === 2 ? el('sub', { text: part }) : part));

function initCnot(fig) {
  const canvas = $('[data-cnot-canvas]', fig);
  const g = cnotLayout(3);
  let step = 1;
  const note = $('[data-cnot-note]', fig);
  const show = (k) => {
    step = k;
    drawCnot(canvas, g, step);
    fill(note, subscripted(CNOT_STEPS[k].text));
  };
  fill($('[data-cnot-steps]', fig), CNOT_STEPS.map((st, i) => el('label', { class: 'segmented__opt' }, [
    el('input', { type: 'radio', name: 'cnot-step', value: String(i), checked: i === step, onchange: () => show(i) }),
    el('span', {}, [`${i + 1}. `, ...subscripted(st.name)]),
  ])));
  new ResizeObserver(() => drawCnot(canvas, g, step)).observe(canvas);
  show(step);
}

/** The per-merge failure q of merges in a row, from a least-squares line through ln(1 − P) against k. */
function perMerge(points) {
  const xs = points.map((q) => q.k), ys = points.map((q) => Math.log(1 - q.rate_any));
  const n = xs.length, mx = xs.reduce((a, b) => a + b) / n, my = ys.reduce((a, b) => a + b) / n;
  const slope = xs.reduce((a, x, i) => a + (x - mx) * (ys[i] - my), 0) / xs.reduce((a, x) => a + (x - mx) ** 2, 0);
  return 1 - Math.exp(slope);
}

async function runCnotRecorded(fig) {
  const status = $('[data-cnot-fit-status]', fig);
  status.textContent = 'Loading the recorded measurements…';
  let doc;
  try {
    const response = await fetch(new URL('programs.json', DATA));
    if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
    doc = await response.json();
  } catch (error) {
    status.textContent = `Recorded measurements unavailable (${error.message}).`;
    return;
  }
  const P = doc.points;
  const pct = (x) => `${(x * 100).toFixed(2)}%`;
  const three = (d, p) => 1 - (1 - P[`memory/d${d}/R${4 * d}/p${p}`].rate_any) ** 3;
  const series = [];
  for (const p of [0.003, 0.002]) {
    for (const inputs of ['z', 'x']) {
      const pts = [3, 5, 7].map((d) => {
        const q = P[`cnot/d${d}/T${d}/p${p}/${inputs}/correlated`];
        return { x: d, y: q.rate_any, lo: q.interval_any[0], hi: q.interval_any[1] };
      });
      series.push({ label: `CNOT, ${inputs === 'z' ? '|0⟩|0⟩' : '|+⟩|+⟩'} in, p = ${(p * 100).toFixed(1)}%`, color: inputs === 'z' ? 'var(--z)' : 'var(--x)', points: pts, line: pts, dashed: p === 0.002 });
    }
    const idle = [3, 5, 7].map((d) => ({ x: d, y: three(d, p) }));
    series.push({ label: `three patches idle as long, p = ${(p * 100).toFixed(1)}%`, color: 'var(--ink-3)', points: [], line: idle, dashed: p === 0.002 });
  }
  const plot = new Plot($('[data-cnot-plot]', fig), {
    xLabel: 'code distance d',
    yLabel: 'wrong, per CNOT',
    yLog: true,
    xTickValues: [3, 5, 7],
    formatX: (v) => `${v}`,
    // Decades as plain percentages: 0.1%, 1%, 10%.
    formatY: (v) => (v >= 0.01 ? `${Math.round(v * 100)}%` : `${+(v * 100).toPrecision(1)}%`),
  });
  plot.render({ series, xRange: [2.6, 7.4] });
  $('[data-cnot-legend]', fig).innerHTML = plotLegend(series);

  const rows = [];
  for (const p of [0.002, 0.003]) {
    for (const d of [3, 5, 7]) {
      const z = P[`cnot/d${d}/T${d}/p${p}/z/correlated`], x = P[`cnot/d${d}/T${d}/p${p}/x/correlated`];
      const zp = P[`cnot/d${d}/T${d}/p${p}/z/plain`];
      rows.push(el('tr', {}, [
        el('th', { scope: 'row', text: `d = ${d}, p = ${(p * 100).toFixed(1)}%` }),
        el('td', { class: 'num', text: pct(z.rate_any) }),
        el('td', { class: 'num', text: pct(x.rate_any) }),
        el('td', { class: 'num', text: pct(zp.rate_any) }),
        el('td', { class: 'num', text: pct(three(d, p)) }),
        el('td', { class: 'num', text: (z.rate_any / three(d, p)).toFixed(2) }),
      ]));
    }
  }
  fill($('[data-cnot-rows]', fig), rows);

  const seq = [3, 5].map((d) => {
    const pts = [1, 2, 4, 8].map((k) => P[`repeated/d${d}/k${k}/p0.003`]);
    return el('tr', {}, [
      el('th', { scope: 'row', text: `d = ${d}` }),
      ...pts.map((q) => el('td', { class: 'num', text: pct(q.rate_any) })),
      el('td', { class: 'num', text: pct(perMerge(pts)) }),
    ]);
  });
  fill($('[data-seq-rows]', fig), seq);
  const line3 = [3, 5, 7].map((d) => pct(P[`line/d${d}/n3/p0.003`].rate_any)).join(', ');
  const w = ['z', 'x'].map((i) => P[`windowed/cnot/d5/p0.003/${i}`]);
  $('[data-cnot-fit-foot]', fig).textContent = `† Measured natively on ${doc.machine.cpu} by tools/surgery.py programs (engine `
    + `${doc.engine_commit}): the whole program sampled by the bit-parallel sampler and decoded by correlated matching (plain `
    + 'in its own column) on this engine\'s error model, which equals Stim\'s fault for fault; each point stops at 1,500 '
    + 'shots with an observable wrong; T = d merged rounds per merge. "Three patches idle as long" holds three patches as '
    + 'memories for the CNOT\'s 4d rounds. Merges in a row: k Z⊗Z measurements on two patches, and the failure per merge '
    + 'fitted through ln(1 − P) against k, which is a straight line: each merge adds the same risk. Three patches merged in '
    + 'a row at once measure Z₁Z₂ and Z₂Z₃ together (the merged patch holds one logical qubit, so not the product Z₁Z₂Z₃), '
    + `and fail ${line3} at d = 3, 5, 7 and p = 0.3%. Window-decoded (parallel windows, correlated), the d = 5 CNOT `
    + `fails ${pct(w[0].windowed_rate_any)} and ${pct(w[1].windowed_rate_any)} against ${pct(w[0].rate_any)} and `
    + `${pct(w[1].rate_any)} decoded whole.`;
  status.textContent = 'Loaded.';
}

export function initSurgery(root, compute) {
  const diagram = $('[data-ls-diagram]', root);
  const fitFig = $('[data-ls-fit]', root);
  const cnotFig = $('[data-cnot-diagram]', root);
  const cnotFitFig = $('[data-cnot-fit]', root);
  const liveFig = $('[data-ls-live]', root);
  if (!diagram || !fitFig || !cnotFig || !cnotFitFig || !liveFig) return;
  initDiagram(diagram);
  initCnot(cnotFig);
  // A recorded figure that fails to draw says so, and the rest still run.
  const failed = (status) => (error) => { status.textContent = `Failed to draw: ${error.message}`; };
  whenNear([fitFig, cnotFitFig, liveFig], () => runRecorded(fitFig).catch(failed($('[data-ls-fit-status]', fitFig)))
    .then(() => runCnotRecorded(cnotFitFig).catch(failed($('[data-cnot-fit-status]', cnotFitFig))))
    .then(() => runLive(liveFig, compute)));
}
