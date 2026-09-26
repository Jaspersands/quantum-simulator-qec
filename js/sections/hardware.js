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
import { decoderLabel, isOurs, percent, percentRange, ratio } from '../hardware-format.js';

const RESULTS = new URL('../../data/google-results/', import.meta.url);
const BELIEF = new URL('../../data/belief/', import.meta.url);
const EXTRACT = new URL('../../data/willow-extract/', import.meta.url);

/** Fit windows: Willow from round 10 (its first round differs from the steady state), Sycamore from 3. */
const MIN_ROUNDS = { willow: 10, sycamore: 3 };
/** As tools/lambda.mjs draws, with the same seed, so the intervals match the README's exactly. */
const DRAWS = 400;

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

async function loadJson(name) {
  const response = await fetch(new URL(name, RESULTS));
  if (!response.ok) throw new Error(`${name}: ${response.status} ${response.statusText}`);
  return response.json();
}

/** The worker's fits as a Map from decoder key to {byD, fit, boot}, ours first. */
function asFits(result) {
  const entries = result.fits.map(({ key, eps, lambda, interval }) => [key, {
    byD: new Map(eps.map((e) => [e.d, { eps: e.eps }])),
    fit: { lambda },
    boot: { eps: new Map(eps.map((e) => [e.d, e.interval])), lambda: interval },
  }]);
  entries.sort((a, b) => Number(isOurs(b[0])) - Number(isOurs(a[0])));
  return new Map(entries);
}

function fitRows(fits, ds, published = null) {
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
  if (published) {
    rows.push(el('tr', { class: 'hw__google hw__published' }, [
      el('th', { scope: 'row', text: published.label }),
      ...ds.map((d) => el('td', { class: 'num', text: published.eps[d] ?? '—' })),
      el('td', { class: 'num', text: published.lambda }),
    ]));
  }
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

function checkLine(s) {
  return `When the counts were recorded, the detection events this engine rebuilt from the chips' raw `
    + `measurements matched Google's bit for bit in ${s.m2d_exact} of ${s.experiments} experiments `
    + `(${s.shots.toLocaleString('en-US')} shots), and its model of each noisy circuit matched Stim's in `
    + `${s.models_same_as_stim} of ${s.experiments}. On Sycamore our plain matcher disagreed with Google's `
    + `recorded PyMatching on ${s.sycamore_pymatching.disagree.toLocaleString('en-US')} shots; on `
    + `${s.sycamore_pymatching.not_optimal === 0 ? 'every one' : `all but ${s.sycamore_pymatching.not_optimal}`} `
    + 'of them our matching weighs exactly the optimum PyMatching 2.4 finds, so each is a tie.';
}

/** How often our belief-matching (pij priors) and Google's agree, shot by shot, over all of Sycamore. */
async function beliefAgreement() {
  const response = await fetch(new URL('sycamore.json', BELIEF));
  if (!response.ok) throw new Error(`belief results: ${response.status}`);
  const doc = await response.json();
  let agree = 0, shots = 0;
  for (const e of Object.values(doc.experiments)) {
    if (e.agree?.['ours/pij/belief'] == null) continue;
    agree += e.agree['ours/pij/belief'];
    shots += e.shots;
  }
  return { agree, shots, maxIter: doc.max_iter };
}

function beliefLine(a) {
  return 'Belief-matching runs belief propagation over each whole error model '
    + `(${a.maxIter} iterations of product-sum BP) and matches on weights from its posteriors, exactly as the `
    + 'authors\' package does. It is costly on Willow\'s long experiments, so there it decoded the first '
    + '10,000 shots of each, with every other decoder in its table scored on the same shots. On Sycamore it '
    + `decoded every shot, with the pij priors cross-fitted as Google used them, and agrees with Google's own `
    + `belief-matching on ${(a.agree / a.shots * 100).toFixed(1)}% of ${a.shots.toLocaleString('en-US')} shots.`;
}

export function initHardware(root, compute) {
  const fitFig = $('[data-hw-fit]', root);
  const liveFig = $('[data-hw-live]', root);
  if (!fitFig || !liveFig) return;

  async function runFits() {
    const status = $('[data-hw-fit-status]', fitFig);
    status.textContent = 'Loading the recorded counts…';
    let willow, sycamore, summary, bWillow, bSycamore, agreement;
    const fitsOf = (base, name, label) => compute.call('hwfits', {
      url: new URL(`${name}.json`, base).href, minRounds: MIN_ROUNDS[name], draws: DRAWS,
    }, (p) => { status.textContent = `Fitting ${label}: ${p.done} of ${p.total} decoders`; });
    try {
      summary = await loadJson('summary.json');
      [willow, sycamore] = await Promise.all([
        fitsOf(RESULTS, 'willow', 'Willow'), fitsOf(RESULTS, 'sycamore', 'Sycamore'),
      ]);
    } catch (error) {
      status.textContent = `Recorded counts unavailable (${error.message}).`;
      return;
    }
    // Belief-matching's counts are a separate run; the tables above stand without them.
    try {
      [bWillow, bSycamore, agreement] = await Promise.all([
        fitsOf(BELIEF, 'willow', 'belief-matching on Willow'), fitsOf(BELIEF, 'sycamore', 'belief-matching on Sycamore'),
        beliefAgreement(),
      ]);
    } catch {
      bWillow = bSycamore = agreement = null;
    }
    const wFits = asFits(willow);
    const sFits = asFits(sycamore);

    fill($('[data-hw-willow]', fitFig), fitRows(wFits, [3, 5, 7], PUBLISHED.willow));
    fill($('[data-hw-sycamore]', fitFig), fitRows(sFits, [3, 5], PUBLISHED.sycamore));
    if (bWillow) fill($('[data-hw-belief-willow]', fitFig), fitRows(asFits(bWillow), [3, 5, 7]));
    if (bSycamore) fill($('[data-hw-belief-sycamore]', fitFig), fitRows(asFits(bSycamore), [3, 5]));

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

    $('[data-hw-fit-foot]', fitFig).textContent = `† Google's decoders' predictions come with the data; `
      + 'every row is fitted here the same way, ours and theirs. For each patch and basis, the logical '
      + 'fidelity 1 − 2 P_L is fitted as A (1 − 2ε)^r over the rounds from '
      + `${MIN_ROUNDS.willow} on (Willow) or ${MIN_ROUNDS.sycamore} on (Sycamore); ε at each distance is the `
      + 'mean over patches and bases, and Λ comes from a line through ln ε against d. Under each ε, the '
      + `95% interval from ${DRAWS} bootstrap draws, each redrawing every experiment's failures from the `
      + `binomial. Counts recorded on ${willow.generated} by tools/google.py (engine ${willow.engine}). `
      + checkLine(summary) + (agreement ? ` ${beliefLine(agreement)}` : '')
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
      row('belief', decoderLabel('ours/si1000/belief')),
      ...pathways.map((p) => row(`google/${p}`, `${decoderLabel(`google/${p}`)} †`, { google: true })),
      row('agree', 'Same prediction as Google\'s correlated matcher'),
      row('time', 'Decode time, ours correlated, per shot'),
      row('belief-conv', 'Belief-matching: shots BP explained alone'),
      row('belief-time', 'Decode time, belief-matching, per shot'),
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
      // One distance per worker, at once.
      const jobs = manifest.experiments.map((ex) => ({ base: EXTRACT.href, experiments: [ex] }));
      await compute.map('hardware', jobs, (_, p) => {
        if (p.step === 'belief') {
          put('belief', p.d, `${p.done.toLocaleString('en-US')} of ${p.total.toLocaleString('en-US')}…`);
          status.textContent = `Belief-matching, d = ${p.d}: ${p.done} of ${p.total} shots`;
          return;
        }
        if (p.step === 'belief-done') {
          const b = p.row.belief;
          put('belief', p.d, failures(b.failures, p.row.shots));
          put('belief-conv', p.d, `${b.converged.toLocaleString('en-US')} of ${p.row.shots.toLocaleString('en-US')}`);
          put('belief-time', p.d, `${b.micros >= 10000 ? (b.micros / 1000).toFixed(0) : (b.micros / 1000).toFixed(1)} ms`);
          return;
        }
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
      + 'error models, Google\'s SI1000 prior as published and ours built from the noisy circuit; plain and '
      + 'correlated matching; and belief-matching, which runs belief propagation over the whole error model '
      + '(20 iterations) and matches on its posteriors, last because it is the slow one. † Google\'s rows are its decoders\' predictions for the same shots, published with the '
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
