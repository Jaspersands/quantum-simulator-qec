/**
 * Section 15 — what it would take: physical qubits, hours and decoding cores
 * for an algorithm, from what this project measured.
 *
 * Every input is read from the committed data the sections above draw: Λ
 * and ε at d = 7 fitted from Google's Willow counts (ours correlated, Google's
 * Libra, and our belief-matching), the cores that keep real-time decoding up
 * at d = 3, 5, 7 (section 12), and the lattice-surgery clock of d merged
 * rounds per operation (section 14). The model is js/estimator.js.
 */

import { $, fill, el, whenNear } from '../dom.js';
import { Plot, plotLegend } from '../plot.js';
import { estimate, physicalQubits, distanceFor, bigNumber, duration, modelFrom, coresFrom } from '../estimator.js';

const DATA = new URL('../../data/', import.meta.url);
const PRESETS = {
  small: { label: 'Small: 100 qubits, a million operations', qubits: 100, operations: 1e6 },
  medium: { label: 'Medium: 1,000 qubits, a billion operations', qubits: 1000, operations: 1e9 },
  large: { label: 'Large: 10,000 qubits, a trillion operations', qubits: 10000, operations: 1e12 },
};
const SOURCES = [
  { id: 'ours', file: 'google-results/willow.json', key: 'ours/si1000/correlated', label: 'ours, correlated matching', color: 'var(--x)' },
  { id: 'libra', file: 'google-results/willow.json', key: 'google/libra_decoder_with_rl_optimized_prior', label: "Google's Libra", color: 'var(--y)' },
  { id: 'belief', file: 'belief/willow.json', key: 'ours/si1000/belief', label: 'ours, belief-matching', color: 'var(--z)' },
];

async function loadJson(path) {
  const response = await fetch(new URL(path, DATA));
  if (!response.ok) throw new Error(`${path}: ${response.status}`);
  return response.json();
}

/** Point fits (no bootstrap): ε at d = 7 and Λ for each source. */
async function measuredModels() {
  const files = {};
  for (const s of SOURCES) files[s.file] ??= Object.values((await loadJson(s.file)).experiments);
  return Object.fromEntries(SOURCES.map((s) => [s.id, modelFrom(files[s.file], s.key)]));
}

async function measuredCores() {
  return coresFrom(await loadJson('realtime/latency.json'));
}

export function initEstimator(root) {
  const fig = $('[data-est]', root);
  const grid = $('[data-est-grid]', root);
  if (!fig || !grid) return;
  const input = (name) => $(`[data-est-${name}]`, fig);
  const status = $('[data-est-status]', fig);
  let models, cores, plot;

  function current() {
    const preset = PRESETS[input('preset').value];
    if (preset && input('preset').value !== 'custom') {
      input('qubits').value = preset.qubits;
      input('ops').value = String(Math.log10(preset.operations));
    }
    const algorithm = { qubits: Math.max(1, Number(input('qubits').value) || 1), operations: 10 ** Number(input('ops').value) };
    const source = input('source').value;
    const custom = source === 'custom';
    input('lambda').disabled = !custom;
    input('lambda').closest('.field').classList.toggle('field--inert', !custom);
    const base = models[custom ? 'ours' : source];
    const model = custom ? { ...base, lambda: Math.max(1.01, Number(input('lambda').value) || 2) } : base;
    const options = {
      budget: Number(input('budget').value),
      overhead: Math.max(1, Number(input('overhead').value) || 2),
      cycleSeconds: Math.max(0.01, Number(input('cycle').value) || 1.1) * 1e-6,
      cores,
    };
    return { algorithm, model, options };
  }

  function render() {
    const { algorithm, model, options } = current();
    const r = estimate(algorithm, model, options);
    const row = (label, value) => el('tr', {}, [el('th', { scope: 'row', text: label }), el('td', { class: 'num', text: value })]);
    fill($('[data-est-out]', fig), r ? [
      row('Λ and ε at d = 7', `${model.lambda.toFixed(2)}, ${(model.eps0 * 100).toFixed(3)}% per cycle`),
      row('Code distance needed', `d = ${r.d}`),
      row('Logical error per cycle at that d', r.epsilon.toExponential(1)),
      row('Physical qubits', bigNumber(r.physical)),
      row('Run time', duration(r.seconds)),
      row('Cores to decode in real time', r.cores == null ? '—' : bigNumber(r.cores)),
      row('Chance the whole run fails', `${(r.failure * 100).toPrecision(2)}%`),
    ] : [row('Code distance needed', 'none up to d = 201: Λ too small for this size')]);

    // Physical qubits against Λ for this algorithm, the measured Λs marked.
    const pts = [];
    for (let lambda = 1.3; lambda <= 4.001; lambda += 0.05) {
      const d = distanceFor(algorithm, { ...model, lambda }, options.budget);
      if (d) pts.push({ x: lambda, y: physicalQubits(d, algorithm, options.overhead) });
    }
    const series = [{ label: 'physical qubits needed', color: 'var(--ink)', points: [], line: pts }];
    // The measured Λs sit too close together to label on the chart; the legend names them.
    const markers = SOURCES.map((s) => ({ x: models[s.id].lambda, label: '' }));
    plot.render({ series, xRange: [1.3, 4], markers });
    const measured = [...SOURCES].sort((a, b) => models[a.id].lambda - models[b.id].lambda)
      .map((s) => `${s.label} ${models[s.id].lambda.toFixed(2)}`).join(', ');
    $('[data-est-legend]', fig).innerHTML = `${plotLegend(series)}<span class="legend__item">dashed: Λ measured on Willow, ${measured}</span>`;
  }

  function renderGrid() {
    const options = { budget: 0.01, overhead: 2, cycleSeconds: 1.1e-6, cores };
    const rows = [];
    for (const preset of Object.values(PRESETS)) {
      for (const s of SOURCES) {
        const r = estimate(preset, models[s.id], options);
        rows.push(el('tr', {}, [
          el('th', { scope: 'row', text: `${preset.label.split(':')[0]}, Λ = ${models[s.id].lambda.toFixed(2)} (${s.label})` }),
          el('td', { class: 'num', text: r ? `${r.d}` : '—' }),
          el('td', { class: 'num', text: r ? bigNumber(r.physical) : '—' }),
          el('td', { class: 'num', text: r ? duration(r.seconds) : '—' }),
          el('td', { class: 'num', text: r && r.cores != null ? bigNumber(r.cores) : '—' }),
        ]));
      }
    }
    fill($('[data-est-grid-rows]', grid), rows);
  }

  async function start() {
    status.textContent = 'Loading the measurements…';
    try {
      [models, cores] = await Promise.all([measuredModels(), measuredCores()]);
    } catch (error) {
      status.textContent = `Measurements unavailable (${error.message}).`;
      return;
    }
    plot = new Plot($('[data-est-plot]', fig), {
      xLabel: 'Λ, the suppression per two steps of distance',
      yLabel: 'physical qubits',
      yLog: true,
      xTickValues: [1.5, 2, 2.5, 3, 3.5, 4],
      formatX: (v) => v.toFixed(1),
      // Decades only, as powers of ten, short enough for the axis.
      formatY: (v) => (Math.abs(Math.log10(v) - Math.round(Math.log10(v))) < 1e-9
        ? `10${String(Math.round(Math.log10(v))).replace(/\d/g, (d) => '⁰¹²³⁴⁵⁶⁷⁸⁹'[d])}` : ''),
    });
    for (const name of ['preset', 'qubits', 'ops', 'budget', 'source', 'lambda', 'overhead', 'cycle']) {
      input(name).addEventListener('input', () => {
        if (name === 'qubits' || name === 'ops') input('preset').value = 'custom';
        render();
      });
    }
    render();
    renderGrid();
    status.textContent = `Measured inputs: cores to keep up at d = ${Object.keys(cores).join(', ')}: `
      + `${Object.values(cores).join(', ')}.`;
  }

  whenNear(fig, start, '300px 0px');
}
