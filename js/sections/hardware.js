/**
 * Section 11 — Google's hardware data.
 *
 * Figure 9 reads the per-experiment failure counts recorded by
 * tools/google.py (every Willow and Sycamore experiment, every decoder, ours
 * and Google's) and fits logical error per cycle and Λ from them here, with the
 * same code the README's numbers come from (js/lambda-fit.js). The counts are
 * recorded, since decoding 27.5 million shots does not belong in a browser tab;
 * the fits are not.
 *
 * Figure 10 takes 2,000 raw Willow shots per distance, straight from the chip,
 * and does everything in the worker: raw measurements to detection events
 * (their SHA-256 checked against Google's), a model, and plain and correlated
 * matching, scored against the recorded outcomes and against Google's own
 * decoders on the same shots.
 */

import { $, fill, el } from '../dom.js';
import { Plot, plotLegend } from '../plot.js';
import { epsilonByDistance, lambdaFit, bootstrap, decoderKeys, seededRandom } from '../lambda-fit.js';
import { decoderLabel, isOurs, percent, percentRange, ratio } from '../hardware-format.js';

const RESULTS = new URL('../../data/google-results/', import.meta.url);
const EXTRACT = new URL('../../data/willow-extract/', import.meta.url);

/** Fit windows: Willow from round 10 (its first round differs from the steady state), Sycamore from 3. */
const MIN_ROUNDS = { willow: 10, sycamore: 3 };
const DRAWS = 200;

/** The lines drawn in the chart; every decoder is in the tables. */
const CHART = [
  { key: 'ours/si1000/plain', color: 'var(--ink-3)' },
  { key: 'ours/si1000/correlated', color: 'var(--x)' },
  { key: 'google/correlated_matching_decoder_with_si1000_prior', color: 'var(--z)' },
  { key: 'google/libra_decoder_with_rl_optimized_prior', color: 'var(--y)' },
];

const PUBLISHED = {
  willow: { label: 'Google: neural-network decoder, published', eps: { 7: '0.143% ± 0.003%' }, lambda: '2.14 ± 0.02' },
  sycamore: { label: 'Google: published', eps: { 3: '3.028% ± 0.023%', 5: '2.914% ± 0.016%' }, lambda: '1.04' },
};

const tick = () => new Promise((resolve) => setTimeout(resolve, 0));

async function loadRecords(name) {
  const response = await fetch(new URL(`${name}.json`, RESULTS));
  if (!response.ok) throw new Error(`${name}.json: ${response.status} ${response.statusText}`);
  return response.json();
}

/** Every decoder's fit, with its bootstrap interval, yielding between decoders. */
async function fitAll(records, minRounds, onProgress) {
  const out = new Map();
  const keys = decoderKeys(records).sort((a, b) => Number(isOurs(b)) - Number(isOurs(a)));
  for (const [i, key] of keys.entries()) {
    const byD = epsilonByDistance(records, key, { minRounds });
    const fit = lambdaFit(byD);
    const boot = bootstrap(records, key, { minRounds }, DRAWS, seededRandom(11));
    out.set(key, { byD, fit, boot });
    onProgress?.(i + 1, keys.length);
    await tick();
  }
  return out;
}

function fitRows(fits, ds, published) {
  const rows = [];
  for (const [key, { byD, fit, boot }] of fits) {
    if (byD.size < ds.length) continue;
    const ours = isOurs(key);
    rows.push(el('tr', { class: ours ? '' : 'hw__google' }, [
      el('th', { scope: 'row', text: `${decoderLabel(key)}${ours ? '' : ' †'}` }),
      ...ds.map((d) => el('td', { class: 'num' }, [
        document.createTextNode(percent(byD.get(d).eps)),
        el('span', { class: 'ci', text: percentRange(boot.eps.get(d)) }),
      ])),
      el('td', { class: 'num', text: ratio(fit.lambda, boot.lambda) }),
    ]));
  }
  rows.push(el('tr', { class: 'hw__google hw__published' }, [
    el('th', { scope: 'row', text: published.label }),
    ...ds.map((d) => el('td', { class: 'num', text: published.eps[d] ?? '—' })),
    el('td', { class: 'num', text: published.lambda }),
  ]));
  return rows;
}

function chartSeries(fits) {
  return CHART.filter((c) => fits.has(c.key)).map(({ key, color }) => {
    const { byD, fit, boot } = fits.get(key);
    const ds = [...byD.keys()];
    const points = ds.map((d) => {
      const [lo, hi] = boot.eps.get(d);
      return { x: d, y: byD.get(d).eps, lo, hi };
    });
    // The fitted line: ln ε = a − (ln Λ / 2) d, through the points' centroid.
    const b = -Math.log(fit.lambda) / 2;
    const mx = ds.reduce((s, d) => s + d, 0) / ds.length;
    const my = ds.reduce((s, d) => s + Math.log(byD.get(d).eps), 0) / ds.length;
    const line = [ds[0], ds[ds.length - 1]].map((d) => ({ x: d, y: Math.exp(my + b * (d - mx)) }));
    return { label: decoderLabel(key), color, points, line };
  });
}

/** The y axis: from the decade below the smallest value to the next 1, 2 or 5 above the largest. */
function logRange(series) {
  const ys = series.flatMap((s) => s.points.flatMap((p) => [p.lo ?? p.y, p.hi ?? p.y]));
  const lo = 10 ** Math.floor(Math.log10(Math.min(...ys)));
  let hi = lo;
  while (hi < Math.max(...ys)) hi = [2, 2.5, 2][Math.round(Math.log10(hi / lo) * 3) % 3] * hi;
  return [lo, Number(hi.toPrecision(3))];
}

function checkLine(checks, docs) {
  const recs = Object.values(checks.experiments);
  const shots = recs.reduce((s, r) => s + r.shots, 0);
  const exact = recs.filter((r) => r.m2d_detector_shots_differ === 0 && r.m2d_observable_shots_differ === 0).length;
  const stimSame = recs.filter((r) => {
    const m = r.model_vs_stim;
    return m.missing === 0 && m.extra === 0 && m.differing === 0 && m.edges_one_sided === 0 && m.splits_differ === 0;
  }).length;
  const pm = Object.values(docs.sycamore.experiments).map((r) => r.pymatching_check).filter(Boolean);
  const differ = pm.reduce((s, c) => s + c.disagree, 0);
  const notOptimal = pm.reduce((s, c) => s + c.ours_not_optimal, 0);
  return `When the counts were recorded, the detection events this engine rebuilt from the chips' raw `
    + `measurements matched Google's bit for bit in ${exact} of ${recs.length} experiments `
    + `(${shots.toLocaleString('en-US')} shots), and its model `
    + `of each noisy circuit matched Stim's in ${stimSame} of ${recs.length}. On Sycamore our plain matcher `
    + `disagreed with Google's recorded PyMatching on ${differ.toLocaleString('en-US')} shots; on `
    + `${notOptimal === 0 ? 'every one' : `all but ${notOptimal}`} of them our matching weighs exactly the `
    + 'optimum PyMatching 2.4 finds, so each is a tie.';
}

export function initHardware(root, compute) {
  const fitFig = $('[data-hw-fit]', root);
  const liveFig = $('[data-hw-live]', root);
  if (!fitFig || !liveFig) return;

  async function runFits() {
    const status = $('[data-hw-fit-status]', fitFig);
    status.textContent = 'Loading the recorded counts…';
    let docs, checks;
    try {
      const [willow, sycamore, c] = await Promise.all(['willow', 'sycamore', 'checks'].map(loadRecords));
      docs = { willow, sycamore };
      checks = c;
    } catch (error) {
      status.textContent = `Recorded counts unavailable (${error.message}).`;
      return;
    }
    const willow = Object.values(docs.willow.experiments);
    const sycamore = Object.values(docs.sycamore.experiments);
    const wFits = await fitAll(willow, MIN_ROUNDS.willow, (i, n) => {
      status.textContent = `Fitting Willow: ${i} of ${n} decoders`;
    });
    const sFits = await fitAll(sycamore, MIN_ROUNDS.sycamore, (i, n) => {
      status.textContent = `Fitting Sycamore: ${i} of ${n} decoders`;
    });

    fill($('[data-hw-willow]', fitFig), fitRows(wFits, [3, 5, 7], PUBLISHED.willow));
    fill($('[data-hw-sycamore]', fitFig), fitRows(sFits, [3, 5], PUBLISHED.sycamore));

    const series = chartSeries(wFits);
    series.push({
      label: 'Google: neural-network decoder, published',
      color: 'var(--ink)',
      points: [{ x: 7, y: 0.00143, lo: 0.0014, hi: 0.00146 }],
    });
    const plot = new Plot($('[data-hw-canvas]', fitFig), {
      xLabel: 'code distance d',
      yLabel: 'logical error per cycle, Willow',
      yLog: true,
      xTickValues: [3, 5, 7],
      formatX: (v) => `${v}`,
      formatY: (v) => percent(v, v < 0.001 ? 2 : 1).replace(/\.?0+%$/, '%'),
    });
    plot.render({ series, xRange: [2.5, 7.5], yRange: logRange(series) });
    $('[data-hw-legend]', fitFig).innerHTML = plotLegend(series);

    const draws = `${DRAWS} bootstrap draws`;
    $('[data-hw-fit-foot]', fitFig).textContent = `† Google's decoders' predictions come with the data; `
      + 'every row is fitted here the same way, ours and theirs. For each patch and basis, the logical '
      + 'fidelity 1 − 2 P_L is fitted as A (1 − 2ε)^r over the rounds from '
      + `${MIN_ROUNDS.willow} on (Willow) or ${MIN_ROUNDS.sycamore} on (Sycamore); ε at each distance is the `
      + 'mean over patches and bases, and Λ comes from a line through ln ε against d. Under each ε, the '
      + `95% interval from ${draws}, each redrawing every experiment's failures from the binomial. `
      + `Counts recorded on ${docs.willow.generated} by tools/google.py (engine ${docs.willow.engine_commit}). `
      + checkLine(checks, docs)
      + ' Data: Google Quantum AI, Zenodo records 13273331 (Willow) and 6804040 (Sycamore), CC BY 4.0.';
    status.textContent = 'Fitted.';
  }

  async function runLive() {
    const status = $('[data-hw-live-status]', liveFig);
    const body = $('[data-hw-live-rows]', liveFig);
    let manifest;
    try {
      const response = await fetch(new URL('manifest.json', EXTRACT));
      if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
      manifest = await response.json();
    } catch (error) {
      status.textContent = `Extract unavailable (${error.message}).`;
      return;
    }
    const ds = manifest.experiments.map((e) => e.d);
    const cells = new Map();
    const row = (id, label, { google = false } = {}) => {
      const tds = ds.map(() => el('td', { class: 'num', text: '—' }));
      cells.set(id, tds);
      return el('tr', { class: google ? 'hw__google' : '' }, [el('th', { scope: 'row', text: label }), ...tds]);
    };
    const pathways = manifest.experiments[0].pathways;
    fill(body, [
      row('hash', 'Detection events from the raw readouts'),
      row('si1000/plain', decoderLabel('ours/si1000/plain')),
      row('si1000/correlated', decoderLabel('ours/si1000/correlated')),
      row('ours/plain', decoderLabel('ours/ours/plain')),
      row('ours/correlated', decoderLabel('ours/ours/correlated')),
      ...pathways.map((p) => row(`google/${p}`, `${decoderLabel(`google/${p}`)} †`, { google: true })),
      row('agree', 'Same prediction as Google\'s correlated matcher'),
      row('time', 'Decode time, ours correlated, per shot'),
    ]);
    const put = (id, d, content) => {
      const td = cells.get(id)?.[ds.indexOf(d)];
      if (td) fill(td, content);
    };
    const failures = (n, shots) => [
      document.createTextNode(n.toLocaleString('en-US')),
      el('span', { class: 'ci', text: percent(n / shots, 1) }),
    ];

    try {
      await compute.call('hardware', { base: EXTRACT.href, experiments: manifest.experiments }, (p) => {
        if (p.step !== 'done') {
          status.textContent = `d = ${p.d}: ${p.step}…`;
          return;
        }
        const r = p.row;
        put('hash', r.d, el('span', {
          class: r.hashMatches && r.obsDiffer === 0 ? 'verdict-text--ok' : 'verdict-text--fail',
          text: r.hashMatches && r.obsDiffer === 0 ? 'SHA-256 matches Google\'s' : 'differ from Google\'s',
        }));
        for (const [k, v] of Object.entries(r.ours)) put(k, r.d, failures(v.failures, r.shots));
        for (const [k, v] of Object.entries(r.google)) put(`google/${k}`, r.d, failures(v.failures, r.shots));
        const corr = r.ours['si1000/correlated'];
        put('agree', r.d, `${(corr.agree / r.shots * 100).toFixed(1)}%`);
        put('time', r.d, `${corr.micros.toFixed(0)} µs`);
      });
      status.textContent = 'Done.';
    } catch (error) {
      status.textContent = `Failed: ${error.message}`;
      return;
    }
    const ex = manifest.experiments;
    $('[data-hw-live-foot]', liveFig).textContent = `Patches ${ex.map((e) => e.patch).join(', ')}: the first `
      + `${manifest.shots.toLocaleString('en-US')} shots of each, as the chip recorded them, fetched as raw `
      + 'measurements and sweep bits (the per-shot pattern the data qubits were prepared in). Everything '
      + 'else happens in this tab: the detection events, whose SHA-256 is checked against Google\'s; the '
      + 'error models, Google\'s SI1000 prior as published and ours built from the noisy circuit; and both '
      + 'matchers. † Google\'s rows are its decoders\' predictions for the same shots, published with the '
      + 'data. About 120 failures, as at d = 7, carry about ±10% of counting noise: enough to see the '
      + 'decoders\' order roughly, not to pin down Λ. Figure 9 does that from all 50,000 shots of every '
      + 'experiment.';
  }

  let started = false;
  const observer = new IntersectionObserver((entries) => {
    if (started || !entries.some((e) => e.isIntersecting)) return;
    started = true;
    observer.disconnect();
    runFits().then(runLive);
  }, { rootMargin: '200px 0px' });
  observer.observe(fitFig);
  observer.observe(liveFig);
}
