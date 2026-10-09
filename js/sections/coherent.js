/**
 * Section 16 — beyond Pauli noise: coherent errors at circuit level, against their twirl.
 *
 * Every figure reads what tools/coherent.py measured natively with the coherent sampler
 * (data/coherent/): Figure 26 the surface code's logical error under over-rotation, coherent
 * against twirled (sweep.json); Figure 27 the coherent sampler against exact answers
 * (validate.json); Figure 28 code capacity, coherent over twirled against distance
 * (capacity.json); Figure 29 ZZ crosstalk on every CNOT (crosstalk.json).
 */

import { $, fill, el, whenNear } from '../dom.js';
import { Plot, plotLegend } from '../plot.js';

const DATA = new URL('../../data/coherent/', import.meta.url);
const COLORS = ['var(--x)', 'var(--z)', 'var(--y)', 'var(--ok)'];

function sci(x) {
  if (!Number.isFinite(x) || x <= 0) return '—';
  const e = Math.floor(Math.log10(x));
  return `${(x / 10 ** e).toFixed(1)}×10${String(e).replace('-', '⁻').replace(/\d/g, (d) => '⁰¹²³⁴⁵⁶⁷⁸⁹'[d])}`;
}

function power(v) {
  const e = Math.round(Math.log10(v));
  return e === 0 ? '1' : `10${String(e).replace('-', '⁻').replace(/\d/g, (d) => '⁰¹²³⁴⁵⁶⁷⁸⁹'[d])}`;
}

const decades = (v) => (Math.abs(Math.log10(v) - Math.round(Math.log10(v))) < 1e-9 ? power(v) : '');

async function load(name, status) {
  status.textContent = 'Loading the recorded measurements…';
  try {
    const response = await fetch(new URL(name, DATA));
    if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
    const doc = await response.json();
    status.textContent = '';
    return doc;
  } catch (error) {
    status.textContent = `Recorded measurements unavailable (${error.message}).`;
    return null;
  }
}

/** Coherent (points with error bars, solid) and twirled (dashed) logical error against θ, by d. */
function sweepPlot(fig, doc, attr) {
  const ds = [...new Set(doc.rows.map((r) => r.d))].sort((a, b) => a - b);
  const series = [];
  ds.forEach((d, i) => {
    const rows = doc.rows.filter((r) => r.d === d && r.coherent.rate > 0).sort((a, b) => a.theta - b.theta);
    const pts = rows.map((r) => ({
      x: r.theta,
      y: r.coherent.rate,
      lo: Math.max(r.coherent.rate - 2 * r.coherent.stderr, r.coherent.rate / 10),
      hi: r.coherent.rate + 2 * r.coherent.stderr,
    }));
    series.push({ label: `d = ${d}, coherent`, color: COLORS[i % COLORS.length], points: pts, line: pts });
    const tw = rows.filter((r) => r.twirl.rate > 0).map((r) => ({ x: r.theta, y: r.twirl.rate }));
    series.push({ label: `d = ${d}, Pauli twirl`, color: COLORS[i % COLORS.length], points: [], line: tw, dashed: true });
  });
  const thetas = [...new Set(doc.rows.map((r) => r.theta))].sort((a, b) => a - b);
  // Evenly spaced ticks: the measured angles crowd together at the small end.
  const top = thetas[thetas.length - 1];
  const step = top <= 0.03 ? 0.005 : 0.01;
  const ticks = Array.from({ length: Math.floor(top / step + 1e-9) }, (_, i) => +((i + 1) * step).toFixed(4));
  const plot = new Plot($(`[${attr}-plot]`, fig), {
    xLabel: attr === 'data-zz' ? 'ZZ rotation θ per CNOT (radians)' : 'over-rotation θ per tick (radians)',
    yLabel: 'logical error per shot',
    yLog: true,
    xTickValues: ticks,
    formatX: (v) => `${v}`,
    formatY: decades,
  });
  plot.render({ series, xRange: [0, thetas[thetas.length - 1] * 1.05] });
  $(`[${attr}-legend]`, fig).innerHTML = plotLegend(series);
  const rows = doc.rows.map((r) => el('tr', {}, [
    el('th', { scope: 'row', text: `d = ${r.d}` }),
    el('td', { class: 'num', text: `${r.theta}` }),
    el('td', { class: 'num' }, [document.createTextNode(sci(r.coherent.rate)), el('span', { class: 'ci', text: `± ${sci(r.coherent.stderr)}` })]),
    el('td', { class: 'num', text: sci(r.twirl.rate) }),
    el('td', { class: 'num', text: r.twirl.rate > 0 ? `${(r.coherent.rate / r.twirl.rate).toFixed(1)}×` : '—' }),
    el('td', { class: 'num', text: sci(r.coherent.rate_merged) }),
    el('td', { class: 'num', text: `${Math.round(100 * r.coherent.ess)}%` }),
  ]));
  fill($(`[${attr}-rows]`, fig), rows);
}

async function runSweep(fig) {
  const doc = await load('sweep.json', $('[data-coh-status]', fig));
  if (doc) sweepPlot(fig, doc, 'data-coh');
}

async function runCrosstalk(fig) {
  const doc = await load('crosstalk.json', $('[data-zz-status]', fig));
  if (doc) sweepPlot(fig, doc, 'data-zz');
}

async function runValidate(fig) {
  const doc = await load('validate.json', $('[data-val-status]', fig));
  if (!doc) return;
  const rows = doc.rows.map((r) => {
    const z = (r.coherent.rate - r.exact) / Math.hypot(r.coherent.stderr, r.exact_stderr || 0);
    return el('tr', {}, [
      el('th', { scope: 'row', text: `${r.code}, d = ${r.d}, ${r.rounds} rounds` }),
      el('td', { class: 'num', text: `${r.theta}` }),
      el('td', { class: 'num', text: r.exact_stderr ? `${r.exact.toFixed(5)} ± ${r.exact_stderr.toFixed(5)}` : r.exact.toFixed(5) }),
      el('td', { class: 'num', text: `${r.coherent.rate.toFixed(5)} ± ${r.coherent.stderr.toFixed(5)}` }),
      el('td', { class: 'num', text: `${z >= 0 ? '+' : '−'}${Math.abs(z).toFixed(1)}σ` }),
      el('td', { class: 'num', text: r.twirl.rate.toFixed(5) }),
    ]);
  });
  fill($('[data-val-rows]', fig), rows);
}

async function runCapacity(fig) {
  const doc = await load('capacity.json', $('[data-cap-status]', fig));
  if (!doc) return;
  const thetas = [...new Set(doc.rows.map((r) => r.theta))].sort((a, b) => a - b);
  const series = thetas.map((t, i) => {
    // Points with too few twirled failures to give a ratio are left out.
    const pts = doc.rows.filter((r) => r.theta === t && r.twirl.failures >= 20 && r.coherent.rate > 0).sort((a, b) => a.d - b.d).map((r) => {
      // How far coherent is from twirled, relative: 0 is the twirl exactly.
      const ratio = r.coherent.rate / r.twirl.rate;
      const rel = Math.hypot(r.coherent.stderr / r.coherent.rate, r.twirl.stderr / r.twirl.rate);
      return { x: r.d, y: ratio - 1, lo: ratio * (1 - 2 * rel) - 1, hi: ratio * (1 + 2 * rel) - 1 };
    });
    return { label: `θ = ${t}`, color: COLORS[i % COLORS.length], points: pts, line: pts };
  });
  const ds = [...new Set(doc.rows.map((r) => r.d))].sort((a, b) => a - b);
  const plot = new Plot($('[data-cap-plot]', fig), {
    xLabel: 'code distance d',
    yLabel: 'coherent against twirled logical error',
    xTickValues: ds,
    formatX: (v) => `${v}`,
    formatY: (v) => `${v > 0 ? '+' : ''}${(v * 100).toFixed(0)}%`,
  });
  // A line at 0: the twirl exactly.
  series.push({ label: 'twirl exact', color: 'var(--ink-3)', points: [], line: [{ x: ds[0] - 0.5, y: 0 }, { x: ds[ds.length - 1] + 0.5, y: 0 }], dashed: true });
  plot.render({ series, xRange: [ds[0] - 0.5, ds[ds.length - 1] + 0.5], yRange: [-0.4, 0.4] });
  $('[data-cap-legend]', fig).innerHTML = plotLegend(series);
}

export function initCoherent(section) {
  if (!section) return;
  const run = (selector, fn) => {
    const fig = $(selector, section);
    if (fig) whenNear(fig, () => fn(fig));
  };
  run('[data-coh]', runSweep);
  run('[data-val]', runValidate);
  run('[data-cap]', runCapacity);
  run('[data-zz]', runCrosstalk);
}
