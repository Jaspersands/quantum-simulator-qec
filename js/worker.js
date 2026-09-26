/**
 * Monte Carlo worker.
 *
 * Holds its own instance of the engine so that sweeps never block the main
 * thread. The main thread keeps a separate instance for interactive lattice
 * work; the two never share memory, and each is seeded independently from the
 * platform CSPRNG when it is instantiated.
 *
 * Protocol: the client posts {id, op, payload}. The worker replies with zero or
 * more {id, type:'progress', ...} messages and exactly one terminal
 * {id, type:'done', result} or {id, type:'error', message}. A later
 * {id, op:'cancel'} asks a streaming job with that id to stop after its
 * current chunk; it still ends with a normal 'done' carrying what it measured.
 */

import {
  instantiate, runBenchmark, estimateChannel, DEFAULT_RUN, NOISE,
  xcGenerate, xcLoadCircuit, xcCompare, xcTiming, hwM2d, hwModel, hwDecode, rtSetup, rtWindows, rtGlobal,
} from './engine.js';
import { planChunks } from './stream.js';
import { epsilonByDistance, lambdaFit, bootstrap, decoderKeys, seededRandom } from './lambda-fit.js';

let enginePromise = null;

function engine() {
  if (!enginePromise) enginePromise = instantiate();
  return enginePromise;
}

/** T = d rounds of syndrome extraction is the standard convention. */
const roundsFor = (config) => (config.noiseMode === NOISE.DATA ? 1 : config.rounds ?? config.d);

async function fetchText(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.text();
}

async function sha256Hex(text) {
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(text));
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, '0')).join('');
}

async function fetchBytes(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return new Uint8Array(await response.arrayBuffer());
}

/**
 * A text file of the extract, gzipped to a quarter of its size. Inflated here
 * unless the server already did (some send .gz with Content-Encoding: gzip),
 * which the gzip signature tells apart.
 */
async function fetchGzipText(url) {
  const bytes = await fetchBytes(url);
  if (bytes[0] !== 0x1f || bytes[1] !== 0x8b) return new TextDecoder().decode(bytes);
  if (typeof DecompressionStream === 'undefined') {
    throw new Error('this browser cannot inflate the gzipped extract (DecompressionStream is missing; '
      + 'it arrived in Safari 16.4 and Firefox 113)');
  }
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
  return new Response(stream).text();
}

async function sha256Bytes(bytes) {
  const digest = await crypto.subtle.digest('SHA-256', bytes);
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, '0')).join('');
}

/** The pathway whose predictions our correlated matcher is compared with shot by shot. */
const GOOGLE_CORRELATED = 'correlated_matching_decoder_with_si1000_prior';

/** Ids of streaming jobs asked to stop; checked between chunks. */
const cancelled = new Set();

const OPS = {
  /**
   * Warm the engine and report a measured throughput baseline.
   *
   * Takes the best of several short samples rather than timing one long run.
   * The first shots pay for JIT warm-up, and a browser that throttles the
   * worker — a background tab, a laptop in low-power mode — produces long
   * stalls that would drag a single sample down by two orders of magnitude.
   * The fastest sample is the one least contaminated by both, and this number
   * is printed on the page as a fact about the reader's machine.
   */
  async vitals(instance) {
    const best = (config, samples = 4) => {
      let fastest = 0;
      let total = 0;
      for (let i = 0; i < samples; i++) {
        const result = runBenchmark(instance, config);
        total += result.runs;
        if (result.runsPerSecond > fastest) fastest = result.runsPerSecond;
      }
      return { fastest, total };
    };

    // Discarded warm-up pass for each noise mode.
    runBenchmark(instance, { noiseMode: NOISE.DATA, d: 5, runs: 4000 });
    runBenchmark(instance, { noiseMode: NOISE.PHENOM, d: 5, rounds: 5, runs: 2000 });

    const data = best({ noiseMode: NOISE.DATA, d: 5, rounds: 1, p: 0.05, runs: 20000 });
    const phenom = best({ noiseMode: NOISE.PHENOM, d: 5, rounds: 5, p: 0.02, runs: 8000 });

    return {
      dataRunsPerSecond: data.fastest,
      phenomRunsPerSecond: phenom.fastest,
      sampleRuns: data.total + phenom.total,
    };
  },

  async benchmark(instance, config) {
    return runBenchmark(instance, { ...config, rounds: roundsFor(config) });
  },

  async channel(instance, config) {
    return estimateChannel(instance, { ...config, rounds: roundsFor(config) });
  },

  /**
   * The bench's run, in chunks, reporting the running estimate after each so
   * the page can draw it converging. Between chunks the loop yields to the
   * event loop, which is what lets a cancel message land mid-run.
   */
  async stream(instance, config, report, control) {
    const total = config.runs;
    let done = 0, failures = 0, seconds = 0, last = 0, decodeErrors = 0;
    for (const n of planChunks(total)) {
      if (control.cancelled()) break;
      const r = runBenchmark(instance, { ...config, rounds: roundsFor(config), runs: n });
      done += n; failures += Math.round(r.rate * n); seconds += r.seconds; last = r.runsPerSecond;
      decodeErrors += r.decodeErrors ?? 0;
      report({ done, total, failures, seconds, chunkRunsPerSecond: last, decodeErrors });
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
    return {
      runs: done, failures, rate: done ? failures / done : 0, seconds, decodeErrors,
      runsPerSecond: seconds > 0 ? Math.round(done / seconds) : 0, cancelled: control.cancelled(),
    };
  },

  /**
   * Section 10's comparison with Stim: for each row, fetch Stim's model, build
   * ours for the same circuit, compare. A row whose reference cannot be
   * fetched says so; nothing is ever filled in from anywhere else.
   */
  async xcheck(instance, { rows }, report) {
    const out = [];
    for (const row of rows) {
      let result;
      try {
        let demText;
        try {
          demText = await fetchText(row.demUrl);
        } catch (error) {
          result = { ok: false, unavailable: true, error: error.message };
        }
        if (!result) {
          let stale = false;
          if (row.kind === 'stim') {
            xcLoadCircuit(instance, await fetchText(row.circuitUrl));
          } else {
            const text = xcGenerate(instance, row.gen);
            if (row.circuitSha256) stale = (await sha256Hex(text)) !== row.circuitSha256;
          }
          const t0 = performance.now();
          result = { ...xcCompare(instance, demText), ms: performance.now() - t0, stale };
        }
      } catch (error) {
        result = { ok: false, error: error.message };
      }
      out.push({ key: row.key, ...result });
      report({ done: out.length, total: rows.length, row: { key: row.key, ...result } });
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
    return { rows: out };
  },

  async xctiming(instance, { cfg, runs }) {
    return xcTiming(instance, cfg, runs);
  },

  /**
   * Sweep a grid of (distance, physical error rate) points.
   * Emits a progress message per point so the caller can draw as it goes.
   */
  /**
   * Section 11's live panel. For each Willow experiment in the extract: raw
   * measurements and sweep bits to detection events (checked against the
   * SHA-256 of Google's own), two models (Google's SI1000 prior, and ours
   * built from the noisy circuit), each decoded plainly and with correlated
   * matching, and every result scored against the true observable flips.
   */
  async hardware(instance, { base, experiments }, report) {
    const rows = [];
    for (const ex of experiments) {
      const at = (name) => new URL(`${ex.dir}/${name}`, base);
      report({ d: ex.d, step: 'fetching' });
      const [ideal, noisy, dem, meas, sweeps, actualBytes, ...preds] = await Promise.all([
        fetchGzipText(at('circuit_ideal.stim.gz')), fetchGzipText(at('circuit_noisy_si1000.stim.gz')),
        fetchGzipText(at('error_model_si1000.dem.gz')), fetchBytes(at('measurements.b8')),
        fetchBytes(at('sweep_bits.b8')), fetchBytes(at('obs_flips_actual.b8')),
        ...ex.pathways.map((p) => fetchBytes(at(`pred_${p}.b8`))),
      ]);
      const shots = ex.shots;
      const actual = actualBytes.map((b) => b & 1);

      report({ d: ex.d, step: 'converting' });
      let t0 = performance.now();
      const conv = hwM2d(instance, ideal, meas, sweeps, shots);
      const m2dMs = performance.now() - t0;
      const hashMatches = (await sha256Bytes(conv.dets)) === ex.detection_events_sha256;
      let obsDiffer = 0;
      for (let i = 0; i < shots; i++) obsDiffer += (conv.obs[i] & 1) !== actual[i] ? 1 : 0;

      report({ d: ex.d, step: 'decoding' });
      const google = {};
      ex.pathways.forEach((p, k) => {
        let f = 0;
        for (let i = 0; i < shots; i++) f += (preds[k][i] & 1) !== actual[i] ? 1 : 0;
        google[p] = { failures: f, predictions: preds[k] };
      });
      const reference = google[GOOGLE_CORRELATED]?.predictions;
      const ours = {};
      const models = [['si1000', { dem }], ['ours', { circuit: noisy }]];
      for (const [slot, [prior, source]] of models.entries()) {
        hwModel(instance, slot, source);
        for (const correlated of [false, true]) {
          t0 = performance.now();
          const r = hwDecode(instance, slot, conv.dets, shots, correlated);
          const micros = ((performance.now() - t0) * 1000) / shots;
          let failures = 0, agree = 0;
          for (let i = 0; i < shots; i++) {
            const p = r.predictions[i];
            failures += p === 255 || p !== actual[i] ? 1 : 0;
            if (reference) agree += p === (reference[i] & 1) ? 1 : 0;
          }
          ours[`${prior}/${correlated ? 'correlated' : 'plain'}`] = { failures, errors: r.errors, micros, agree };
        }
      }
      for (const g of Object.values(google)) delete g.predictions;
      const row = { d: ex.d, patch: ex.patch, rounds: ex.rounds, shots, hashMatches, obsDiffer, m2dMs, ours, google };
      rows.push(row);
      report({ d: ex.d, step: 'done', row });
    }
    return rows;
  },

  /**
   * Section 11's Figure 9: every decoder's ε and Λ from the recorded counts,
   * with the same fit, draws and seed as tools/lambda.mjs, so the intervals on
   * the page are the README's to the last digit. Here rather than on the main
   * thread because the bootstrap refits every decoder hundreds of times.
   */
  async hwfits(instance, { url, minRounds, draws }, report) {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
    const doc = await response.json();
    const records = Object.values(doc.experiments);
    const keys = decoderKeys(records);
    const fits = [];
    for (const [i, key] of keys.entries()) {
      const byD = epsilonByDistance(records, key, { minRounds });
      const fit = lambdaFit(byD);
      const boot = bootstrap(records, key, { minRounds }, draws, seededRandom(11));
      fits.push({
        key,
        eps: [...byD].map(([d, v]) => ({ d, eps: v.eps, interval: boot.eps.get(d) })),
        lambda: fit.lambda,
        interval: boot.lambda,
      });
      report({ done: i + 1, total: keys.length });
    }
    return { generated: doc.generated, engine: doc.engine_commit, fits };
  },

  /**
   * Section 12's live panel: batches of 64 simulated memory streams, each
   * window-decoded as it streams and then globally, both timed from here (the
   * engine has no clock). Reports after each batch.
   */
  async realtime(instance, { config, batches }, report) {
    const setup = rtSetup(instance, config);
    let streams = 0, windowFailures = 0, globalFailures = 0, agree = 0, unexplained = 0;
    let windowSeconds = 0, globalSeconds = 0;
    for (let b = 0; b < batches; b++) {
      let t0 = performance.now();
      const w = rtWindows(instance);
      windowSeconds += (performance.now() - t0) / 1000;
      t0 = performance.now();
      const g = rtGlobal(instance);
      globalSeconds += (performance.now() - t0) / 1000;
      streams += 64;
      windowFailures += w.failures;
      globalFailures += g.failures;
      agree += g.agree;
      unexplained += w.unexplained;
      report({ d: config.d, streams, done: b + 1, total: batches });
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
    const rounds = streams * config.rounds;
    return {
      ...config, windows: setup.windows, template: setup.template, streams, rounds,
      windowFailures, globalFailures, agree, unexplained, windowSeconds, globalSeconds,
      // Per stream: the window decoding's cost of one round, and what that
      // means against Willow's cycle.
      windowMicrosPerRound: (windowSeconds * 1e6) / rounds,
      roundsPerSecond: rounds / windowSeconds,
    };
  },

  async sweep(instance, { distances, ps, base }, report) {
    const points = [];
    const total = distances.length * ps.length;
    let done = 0;

    for (const d of distances) {
      for (const p of ps) {
        const config = { ...base, d, p };
        const { rate, runs } = runBenchmark(instance, { ...config, rounds: roundsFor(config) });
        points.push({ d, p, pL: rate, runs });
        done += 1;
        report({ done, total, d, p, pL: rate, runs });
      }
    }
    return { points };
  },

  /**
   * Run an arbitrary list of configurations, emitting each as it lands so
   * tables and comparison plots can fill in progressively.
   */
  async table(instance, { cells, base }, report) {
    const results = [];
    for (let i = 0; i < cells.length; i++) {
      const cell = cells[i];
      const config = { ...base, ...cell };
      const { rate, runs } = runBenchmark(instance, { ...config, rounds: roundsFor(config) });
      const entry = { ...cell, pL: rate, runs };
      results.push(entry);
      report({ done: i + 1, total: cells.length, cell: entry });
    }
    return { results };
  },
};

self.onmessage = async (event) => {
  const { id, op, payload } = event.data;
  if (op === 'cancel') { cancelled.add(id); return; }
  const report = (progress) => self.postMessage({ id, type: 'progress', ...progress });
  const control = { cancelled: () => cancelled.has(id) };

  // The pool hands every worker the page's compiled module first, so the
  // engine is downloaded once rather than once per worker.
  if (op === 'module') {
    enginePromise = instantiate(payload.module);
    enginePromise.then(() => self.postMessage({ id, type: 'done', result: true }), (error) => {
      // Fall back to fetching the engine at the next job rather than failing every one.
      enginePromise = null;
      self.postMessage({ id, type: 'error', message: error?.message ?? String(error) });
    });
    return;
  }

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
