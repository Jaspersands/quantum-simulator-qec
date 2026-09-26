/**
 * Section 12 — real-time decoding.
 *
 * Figure 11 reads what tools/realtime.py measured natively: every window's
 * decode time on Willow's recorded d = 3, 5, 7 syndromes at 250 rounds,
 * scheduled at Willow's 1.1 µs cycle over 1 to 16 cores (latency.json); the
 * window decoders' accuracy against global decoding over every Willow
 * experiment (willow-windows.json, fitted here with the same fit as section
 * 11); and a million-round stream (million.json).
 *
 * Figure 12 runs window decoding here: each distance on its own worker, 64
 * streams at a time, each stream window-decoded as it is sampled and then
 * decoded globally for comparison, timed from the page.
 */

import { $, fill, el } from '../dom.js';
import { Plot, plotLegend } from '../plot.js';
import { percent, percentRange, ratio } from '../hardware-format.js';
import { cores, keepUpText, microseconds as us, windowLabel } from '../realtime-format.js';

const DATA = new URL('../../data/realtime/', import.meta.url);
const CYCLE_US = 1.1;
const GOOGLE_US = 63;
const COLORS = { 3: 'var(--d3)', 5: 'var(--d5)', 7: 'var(--d7)' };
/** SD6 p at which each distance's detectors fire as often as Willow's do (tools/realtime.py calibrate). */
const WILLOW_P = { 3: 0.00278, 5: 0.00313, 7: 0.00319 };
const LIVE = { rounds: 100, batches: 4 };

async function loadJson(name) {
  const response = await fetch(new URL(name, DATA));
  if (!response.ok) throw new Error(`${name}: ${response.status} ${response.statusText}`);
  return response.json();
}

/** The latency chart: mean latency against cores, parallel correlated windows, one line per distance. */
function chartSeries(latency) {
  const series = [];
  for (const d of [3, 5, 7]) {
    const s = latency.streams.find((x) => x.d === d && x.mode === 'parallel' && x.matcher === 'correlated');
    if (!s) continue;
    const pts = Object.entries(s.by_workers).map(([k, v]) => ({ x: Number(k), y: v.mean_us, lo: v.mean_us, hi: v.p99_us }));
    series.push({ label: `d = ${d}`, color: COLORS[d], points: pts, line: pts });
  }
  series.push({
    label: `Google's real-time decoder, d = 5 (published)`,
    color: 'var(--ink)',
    points: [],
    line: [{ x: 1, y: GOOGLE_US }, { x: 16, y: GOOGLE_US }],
  });
  return series;
}

function keepRows(latency) {
  const rows = [];
  for (const d of [3, 5, 7]) {
    for (const matcher of ['plain', 'correlated']) {
      for (const mode of ['sliding', 'parallel']) {
        const s = latency.streams.find((x) => x.d === d && x.mode === mode && x.matcher === matcher);
        if (!s) continue;
        const k = cores(s.by_workers);
        const at = k ? s.by_workers[k] : null;
        rows.push(el('tr', {}, [
          el('th', { scope: 'row', text: `d = ${d}, ${mode}, ${matcher}` }),
          el('td', { class: 'num', text: us(s.window_us.mean) }),
          el('td', { class: 'num', text: keepUpText(k, mode) }),
          el('td', { class: 'num', text: at ? us(at.mean_us) : '—' }),
          el('td', { class: 'num', text: at ? us(at.p99_us) : '—' }),
        ]));
      }
    }
  }
  return rows;
}

function accuracyRows(fits) {
  const order = ['global/plain', 'window/sliding/Bd/plain', 'window/parallel/Bd/plain',
    'window/parallel/Bhalf/plain', 'window/parallel/B2d/plain',
    'global/correlated', 'window/sliding/Bd/correlated', 'window/parallel/Bd/correlated',
    'window/parallel/Bhalf/correlated', 'window/parallel/B2d/correlated'];
  return order.filter((k) => fits.has(k)).map((key) => {
    const { eps, lambda, interval } = fits.get(key);
    const byD = new Map(eps.map((e) => [e.d, e]));
    return el('tr', {}, [
      el('th', { scope: 'row', text: windowLabel(key) }),
      ...[3, 5, 7].map((d) => {
        const e = byD.get(d);
        return el('td', { class: 'num' }, e ? [document.createTextNode(percent(e.eps)), el('span', { class: 'ci', text: percentRange(e.interval) })] : ['—']);
      }),
      el('td', { class: 'num', text: ratio(lambda, interval) }),
    ]);
  });
}

export function initRealtime(root, compute) {
  const fitFig = $('[data-rt-fit]', root);
  const liveFig = $('[data-rt-live]', root);
  if (!fitFig || !liveFig) return;

  async function runRecorded() {
    const status = $('[data-rt-fit-status]', fitFig);
    status.textContent = 'Loading the recorded measurements…';
    let latency, million, fits;
    try {
      [latency, million] = await Promise.all([loadJson('latency.json'), loadJson('million.json')]);
      const result = await compute.call('hwfits', {
        url: new URL('willow-windows.json', DATA).href, minRounds: 10, draws: 400,
      }, (p) => { status.textContent = `Fitting: ${p.done} of ${p.total} decoders`; });
      fits = new Map(result.fits.map((f) => [f.key, f]));
    } catch (error) {
      status.textContent = `Recorded measurements unavailable (${error.message}).`;
      return;
    }
    const series = chartSeries(latency);
    const plot = new Plot($('[data-rt-canvas]', fitFig), {
      xLabel: 'cores decoding one stream',
      yLabel: 'window latency, µs',
      yLog: true,
      xTickValues: [1, 2, 4, 8, 16],
      formatX: (v) => `${v}`,
      formatY: (v) => `${v >= 10 ? Math.round(v) : v.toPrecision(1)}`,
    });
    plot.render({ series, xRange: [0, 17] });
    $('[data-rt-legend]', fitFig).innerHTML = plotLegend(series);
    fill($('[data-rt-keep]', fitFig), keepRows(latency));
    fill($('[data-rt-accuracy]', fitFig), accuracyRows(fits));

    const m = million.runs.find((r) => r.mode === 'parallel' && r.matcher === 'correlated');
    const mk = m ? cores(m.by_workers) : null;
    $('[data-rt-fit-foot]', fitFig).textContent = `† Decode times measured on ${latency.machine.cpu}, one window on `
      + 'one core at a time; latency comes from scheduling those measured times with rounds arriving every '
      + `${CYCLE_US} µs, a window starting once its last round has arrived and (for layer B) its two layer-A `
      + 'neighbours are done. Latency is the time from the arrival of a window\'s last round to its commit. '
      + 'Points are the mean, bars reach the 99th percentile; "keeps up" means the latency does not grow along '
      + 'the stream. Sliding windows use one core per stream by construction. Commit and buffer are both d rounds. '
      + (m ? `A million rounds: rotated d = 5 under SD6 at p = ${(million.p * 100).toFixed(2)}%, the noise at which `
        + `detectors fire as often as Willow's (${(million.willow_detection_fraction * 100).toFixed(1)}%): `
        + `${m.streams} streams × ${million.rounds.toLocaleString('en-US')} rounds decoded with ${m.unexplained} `
        + `defects unexplained; ${keepUpText(mk, 'parallel').toLowerCase()}${mk ? `, at ${us(m.by_workers[mk].mean_us)} mean latency` : ''}. ` : '')
      + 'Google\'s 63 µs is its own real-time decoder at d = 5, on its own hardware (arXiv:2408.13687). '
      + 'The accuracy table fits every Willow experiment\'s windowed and global failures the same way as section 11; '
      + 'windows commit d rounds with a buffer of d unless the row names another, and buffers of d/2 and 2d were '
      + 'run at d = 5 only.';
    status.textContent = 'Loaded.';
  }

  async function runLive() {
    const status = $('[data-rt-live-status]', liveFig);
    const body = $('[data-rt-live-rows]', liveFig);
    const ds = [3, 5, 7];
    const cells = new Map();
    const row = (id, label) => {
      const tds = ds.map(() => el('td', { class: 'num', text: '—' }));
      cells.set(id, tds);
      return el('tr', {}, [el('th', { scope: 'row', text: label }), ...tds]);
    };
    fill(body, [
      row('cost', 'Window decoding, per round of one stream'),
      row('cores', `Cores to keep up with ${CYCLE_US} µs rounds`),
      row('agree', 'Windowed = global decoding, streams'),
      row('fail', 'Logical failures, windowed / global'),
      row('unexplained', 'Defects left unexplained'),
    ]);
    const put = (id, d, text) => { const td = cells.get(id)?.[ds.indexOf(d)]; if (td) td.textContent = text; };
    status.textContent = `Decoding on ${compute.size ?? 1} workers…`;
    try {
      const jobs = ds.map((d) => ({
        config: { d, rounds: LIVE.rounds, p: WILLOW_P[d], commit: d, buffer: d, parallel: true, correlated: true },
        batches: LIVE.batches,
      }));
      const results = await compute.map('realtime', jobs, (_, p) => {
        status.textContent = `d = ${p.d}: ${p.streams} streams`;
      });
      for (const r of results) {
        put('cost', r.d, us(r.windowMicrosPerRound));
        put('cores', r.d, `${Math.max(1, Math.ceil(r.windowMicrosPerRound / CYCLE_US))}`);
        put('agree', r.d, `${r.agree} of ${r.streams}`);
        put('fail', r.d, `${r.windowFailures} / ${r.globalFailures}`);
        put('unexplained', r.d, `${r.unexplained}`);
      }
      status.textContent = 'Done.';
    } catch (error) {
      status.textContent = `Failed: ${error.message}`;
      return;
    }
    $('[data-rt-live-foot]', liveFig).textContent = `Each distance on its own worker: ${LIVE.batches} batches of 64 `
      + `streams × ${LIVE.rounds} rounds, rotated SD6 at the p where detectors fire as often as Willow's, sampled `
      + 'here and window-decoded as they stream (parallel windows, correlated matching, commit and buffer d '
      + 'rounds), then decoded globally for comparison. The cost per round is the window decoding\'s wall time '
      + 'divided by the rounds decoded, on one core of this machine, in WebAssembly, which runs a few times '
      + 'slower than native; "cores to keep up" divides it by the 1.1 µs cycle. Parallel windows spread one '
      + 'stream over that many cores.';
  }

  let started = false;
  const observer = new IntersectionObserver((entries) => {
    if (started || !entries.some((e) => e.isIntersecting)) return;
    started = true;
    observer.disconnect();
    // The live figure runs whatever became of the recorded one.
    runRecorded()
      .catch((error) => { $('[data-rt-fit-status]', fitFig).textContent = `Failed: ${error.message}`; })
      .then(runLive);
  }, { rootMargin: '200px 0px' });
  observer.observe(fitFig);
  observer.observe(liveFig);
}
