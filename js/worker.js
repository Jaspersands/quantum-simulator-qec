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

import { instantiate, runBenchmark, estimateChannel, DEFAULT_RUN, NOISE } from './engine.js';
import { planChunks } from './stream.js';

let enginePromise = null;

function engine() {
  if (!enginePromise) enginePromise = instantiate();
  return enginePromise;
}

/** T = d rounds of syndrome extraction is the standard convention. */
const roundsFor = (config) => (config.noiseMode === NOISE.DATA ? 1 : config.rounds ?? config.d);

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
    let done = 0, failures = 0, seconds = 0, last = 0;
    for (const n of planChunks(total)) {
      if (control.cancelled()) break;
      const r = runBenchmark(instance, { ...config, rounds: roundsFor(config), runs: n });
      done += n; failures += Math.round(r.rate * n); seconds += r.seconds; last = r.runsPerSecond;
      report({ done, total, failures, seconds, chunkRunsPerSecond: last });
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
    return {
      runs: done, failures, rate: done ? failures / done : 0, seconds,
      runsPerSecond: seconds > 0 ? Math.round(done / seconds) : 0, cancelled: control.cancelled(),
    };
  },

  /**
   * Sweep a grid of (distance, physical error rate) points.
   * Emits a progress message per point so the caller can draw as it goes.
   */
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
