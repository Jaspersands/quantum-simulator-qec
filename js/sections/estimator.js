/**
 * Section 15 — what it would take: physical qubits and hours for an algorithm,
 * from what this project measured and what the papers it is checked against
 * report.
 *
 * Figure 23 prices a chosen algorithm with js/estimator.js's fuller model:
 * magic-state factories (cultivation or Litinski's distillation), Litinski's
 * data blocks, the reaction each Toffoli waits for (this project's measured
 * decoder latency, extrapolated, plus a control delay), and idle storage.
 * Figure 24 checks the model against Gidney's RSA-2048 and Lee et al.'s
 * FeMoco, with the same cases tools/estimate.mjs prints. Figure 25 keeps the
 * simple model's point: under the Λ measured on Willow, distance and qubits
 * blow up, and Λ is the lever.
 */

import { $, fill, el, whenNear } from '../dom.js';
import {
  estimate, estimateFull, bigNumber, duration, modelFrom, coresFrom, latencyFrom, defaultOptions, validationCases,
  gidneyParallel,
} from '../estimator.js';

const DATA = new URL('../../data/', import.meta.url);
const PRESETS = {
  small: { label: 'Small: 100 qubits, a million operations', qubits: 100, operations: 1e6 },
  medium: { label: 'Medium: 1,000 qubits, a billion operations', qubits: 1000, operations: 1e9 },
  large: { label: 'Large: 10,000 qubits, a trillion operations', qubits: 10000, operations: 1e12 },
};
const WILLOW = [
  { id: 'ours', file: 'google-results/willow.json', key: 'ours/si1000/correlated', label: 'ours, correlated matching' },
  { id: 'libra', file: 'google-results/willow.json', key: 'google/libra_decoder_with_rl_optimized_prior', label: "Google's Libra" },
  { id: 'belief', file: 'belief/willow.json', key: 'ours/si1000/belief', label: 'ours, belief-matching' },
];

async function loadJson(path) {
  const response = await fetch(new URL(path, DATA));
  if (!response.ok) throw new Error(`${path}: ${response.status}`);
  return response.json();
}

async function loadAll() {
  const [sources, noise, latencyDoc, gross, distill] = await Promise.all([
    loadJson('estimate/sources.json'), loadJson('estimate/noise.json'), loadJson('realtime/latency.json'),
    loadJson('gross/results.json'), loadJson('estimate/distill.json'),
  ]);
  const files = {};
  for (const s of WILLOW) files[s.file] ??= Object.values((await loadJson(s.file)).experiments);
  const models = Object.fromEntries(WILLOW.map((s) => [s.id, modelFrom(files[s.file], s.key)]));
  return { sources, fit: noise.fit, latency: latencyFrom(latencyDoc), cores: coresFrom(latencyDoc), gross, distill, models };
}

const pct = (x, digits = 2) => `${(x * 100).toPrecision(digits)}%`;
/** Seconds as µs, ms or s, three figures. */
const short = (s) => (s < 1e-3 ? `${(s * 1e6).toPrecision(3)} µs` : s < 1 ? `${(s * 1e3).toPrecision(3)} ms` : `${s.toPrecision(3)} s`);
const sci = (x) => (x > 0 ? x.toExponential(1).replace('e-', ' × 10⁻').replace(/\d+$/, (e) => e.replace(/\d/g, (c) => '⁰¹²³⁴⁵⁶⁷⁸⁹'[c])) : '0');

/** Bars side by side, each as long as its value against the largest: alternatives, not parts. */
function compare(label, items) {
  const top = Math.max(...items.map((i) => i.value)) || 1;
  return el('div', { class: 'est-bar' }, [
    el('div', { class: 'est-bar__label', text: label }),
    ...items.map((i) => el('div', { class: 'est-bar__row' }, [
      el('div', { class: 'est-bar__track' }, [el('span', { class: `est-bar__part est-bar__part--${i.key}`, style: `width: ${(100 * i.value) / top}%` })]),
      el('div', { class: 'est-bar__legend', text: `${i.name}: ${i.text}` }),
    ])),
  ]);
}

/** A horizontal bar of labelled parts, widths in proportion. */
function bar(label, parts) {
  const total = parts.reduce((a, p) => a + p.value, 0) || 1;
  return el('div', { class: 'est-bar' }, [
    el('div', { class: 'est-bar__label', text: label }),
    el('div', { class: 'est-bar__track' }, parts.filter((p) => p.value > 0).map((p) => el('span', {
      class: `est-bar__part est-bar__part--${p.key}`, style: `width: ${(100 * p.value) / total}%`, title: `${p.name}: ${p.text}`,
    }))),
    el('div', { class: 'est-bar__legend' }, parts.filter((p) => p.value > 0).map((p) => el('span', {}, [
      el('span', { class: `est-bar__swatch est-bar__part--${p.key}` }), `${p.name} ${p.text}`,
    ]))),
  ]);
}

export function initEstimator(root) {
  const fig = $('[data-est]', root);
  const valid = $('[data-est-valid]', root);
  const grid = $('[data-est-grid]', root);
  if (!fig || !valid || !grid) return;
  const input = (name) => $(`[data-est-${name}]`, fig);
  const status = $('[data-est-status]', fig);
  let D;

  function algorithm() {
    const key = input('preset').value;
    const preset = D.sources.algorithms[key];
    if (preset) {
      // Shown rounded, computed exact, so the page gives the README's numbers for a preset.
      input('qubits').value = preset.qubits;
      input('toffolis').value = String(Math.round(Math.log10(preset.toffolis) * 10) / 10);
      input('hot').value = preset.hot ?? preset.qubits;
      return { qubits: preset.qubits, hot: preset.hot ?? preset.qubits, toffolis: preset.toffolis };
    }
    const qubits = Math.max(1, Number(input('qubits').value) || 1);
    return { qubits, hot: Math.min(qubits, Math.max(1, Number(input('hot').value) || qubits)), toffolis: 10 ** Number(input('toffolis').value) };
  }

  function options() {
    const o = defaultOptions(D.fit, D.latency, D.cores);
    const noise = input('noise').value;
    if (noise.startsWith('willow:')) {
      // Willow's own error per cycle and Λ; the factories are taken as at p = 0.1%.
      o.noise = { kind: 'measured', p: 1e-3, model: D.models[noise.slice(7)] };
    } else {
      o.noise = { kind: 'uniform', p: Number(noise), fit: D.fit };
    }
    o.factory = input('factory').value;
    o.block = input('block').value;
    o.storage = input('storage').value;
    o.budget = Number(input('budget').value);
    o.cycleSeconds = Math.max(0.01, Number(input('cycle').value) || 1) * 1e-6;
    o.controlSeconds = Math.max(0, Number(input('control').value) || 0) * 1e-6;
    const decoder = input('decoder').value;
    o.latencyOverrideUs = decoder === 'measured' ? null : Number(decoder);
    o.parallel = Math.max(1, Number(input('parallel').value) || 1);
    const g = D.gross.points[`gross/${o.noise.p}`];
    o.grossPerCycle = g ? g.results.bposd_cs7.pl_cycle : null;
    return o;
  }

  function render() {
    const alg = algorithm();
    const o = options();
    input('hot').closest('.field').classList.toggle('field--inert', o.storage === 'surface');
    const r = estimateFull(alg, o, D.sources);
    const row = (label, value) => el('tr', {}, [el('th', { scope: 'row', text: label }), el('td', { class: 'num', text: value })]);
    if (r.error) {
      fill($('[data-est-out]', fig), [row('No estimate', r.error)]);
      fill($('[data-est-bars]', fig), []);
      return;
    }
    fill($('[data-est-out]', fig), [
      row('Code distance', `d = ${r.d}, ε ${sci(r.epsilon)} per patch per cycle`),
      row('Physical qubits', bigNumber(r.qubits.total)),
      row('Run time', `${duration(r.seconds)}, ${short(r.perToffoli)} per Toffoli (${r.bound === 'clifford' ? 'lattice-surgery steps' : 'reaction'} bound)`),
      row('Magic states', `${r.factories} × ${r.factory.name}, ${r.factory.statesPerToffoli === 1 ? 'one CCZ' : 'four T states'} per Toffoli`),
      row('Chance the run fails', `${pct(r.failure.total)} (memory ${pct(r.failure.memory)}, magic states ${pct(r.failure.magic)}${o.storage === 'surface' ? '' : `, storage ${pct(r.failure.storage)}`})`),
      row('Cores to decode in real time', r.cores == null ? '—' : bigNumber(r.cores)),
    ]);
    fill($('[data-est-bars]', fig), [
      bar('Qubits', [
        { key: 'block', name: `${o.block} block, ${r.tiles.toLocaleString('en-US')} tiles`, value: r.qubits.block, text: bigNumber(r.qubits.block) },
        { key: 'factories', name: 'factories', value: r.qubits.factories, text: bigNumber(r.qubits.factories) },
        { key: 'storage', name: `${o.storage} storage`, value: r.qubits.storage, text: bigNumber(r.qubits.storage) },
      ]),
      compare('Each Toffoli waits for the longer of', [
        { key: 'block', name: 'its lattice-surgery steps', value: r.clifford, text: short(r.clifford) },
        { key: 'reaction', name: 'its reaction, decoding and control', value: r.reaction, text: short(r.reaction) },
      ]),
    ]);
  }

  function renderValidation() {
    const cases = validationCases(D.sources, defaultOptions(D.fit, D.latency, D.cores));
    fill($('[data-est-valid-rows]', valid), cases.map((c) => {
      const r = estimateFull(c.algorithm, c.options, D.sources);
      const who = c.group === 'rsa' ? 'RSA-2048' : 'FeMoco';
      if (r.error) {
        return el('tr', {}, [el('th', { scope: 'row', text: `${who}: ${c.label}` }), el('td', { colspan: '4', text: r.error })]);
      }
      return el('tr', {}, [
        el('th', { scope: 'row', text: `${who}: ${c.label}` }),
        el('td', { class: 'num', text: `${r.d}` }),
        el('td', { class: 'num', text: `${bigNumber(r.qubits.total)} (${(r.qubits.total / c.reference.qubits).toFixed(2)}×)` }),
        el('td', { class: 'num', text: `${duration(r.seconds)} (${(r.seconds / c.reference.seconds).toFixed(2)}×)` }),
        el('td', { class: 'num', text: r.bound === 'clifford' ? 'steps' : 'reaction' }),
      ]);
    }));
    const eps25 = D.fit.A * (1e-3 / D.fit.pth) ** 13;
    const d0 = D.distill.points.find((q) => q.p === 0.001);
    $('[data-est-valid-note]', valid).textContent = 'Gidney: 897,864 qubits and 12.07 hours per shot (arXiv:2505.15917). '
      + `His schedule runs ${gidneyParallel(D.sources).toFixed(1)} Toffolis in each 25 µs lattice-surgery step, and his ε is `
      + `1e-15 per patch per round at d = 25; this engine's, measured under uniform 0.1% noise and extrapolated, is ${sci(eps25)} `
      + `(ε = ${D.fit.A.toFixed(3)} (p / ${(D.fit.pth * 100).toFixed(2)}%)^((d+1)/2), fitted to d = 3 to 11). Lee et al.: about four `
      + 'million qubits and three days (arXiv:2011.03494). The first row is entirely Gidney\'s; each after it switches one option '
      + 'to this project\'s. The 15-to-1 law the distillation tables rest on is measured here too: this engine\'s samples and '
      + `an exact sum agree, and at p = 0.1% the output error is ${d0.output_over_35p3.toFixed(3)} × 35p³.`;
  }

  function renderGrid() {
    const opts = { budget: 0.01, overhead: 2, cycleSeconds: 1.1e-6, cores: D.cores };
    const rows = [];
    for (const preset of Object.values(PRESETS)) {
      for (const s of WILLOW) {
        const r = estimate(preset, D.models[s.id], opts);
        rows.push(el('tr', {}, [
          el('th', { scope: 'row', text: `${preset.label.split(':')[0]}, Λ = ${D.models[s.id].lambda.toFixed(2)} (${s.label})` }),
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
      D = await loadAll();
    } catch (error) {
      status.textContent = `Measurements unavailable (${error.message}).`;
      return;
    }
    for (const name of ['preset', 'qubits', 'hot', 'toffolis', 'noise', 'factory', 'block', 'storage', 'budget', 'cycle', 'control', 'decoder', 'parallel']) {
      input(name).addEventListener('input', () => {
        if (name === 'qubits' || name === 'toffolis' || name === 'hot') input('preset').value = 'custom';
        render();
      });
    }
    render();
    renderValidation();
    renderGrid();
    const ds = Object.keys(D.latency).join(', ');
    status.textContent = `Measured inputs: decoder latency (p99, one window) at d = ${ds}: `
      + `${Object.values(D.latency).map((x) => `${x.toFixed(0)} µs`).join(', ')}, extrapolated beyond as a power law.`;
  }

  whenNear([fig, valid, grid], start, '300px 0px');
}
