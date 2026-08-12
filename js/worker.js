/**
 * Monte Carlo worker.
 *
 * Holds its own instance of the engine so that sweeps never block the main
 * thread. The main thread keeps a separate instance for interactive lattice
 * work; the two never share memory.
 *
 * Sampling goes through js/montecarlo.js rather than the engine's own
 * `wasm_run_benchmark` — see the note at the top of that file for why.
 *
 * Protocol: the client posts {id, op, payload}. The worker replies with zero or
 * more {id, type:'progress', ...} messages and exactly one terminal
 * {id, type:'done', result} or {id, type:'error', message}.
 */

import { instantiate, NOISE } from './engine.js';
import { estimateLogicalErrorRate, DEFAULTS } from './montecarlo.js';

let enginePromise = null;

function engine() {
  if (!enginePromise) enginePromise = instantiate();
  return enginePromise;
}

const OPS = {
  /** Warm the engine and report a measured throughput baseline. */
  async vitals(instance) {
    // Discard two passes: the first shots pay for JIT warm-up in both noise
    // modes, and timing them would understate the engine by an order of
    // magnitude.
    estimateLogicalErrorRate(instance, { noiseMode: NOISE.DATA, d: 5, runs: 3000 });
    estimateLogicalErrorRate(instance, { noiseMode: NOISE.PHENOM, d: 5, rounds: 5, runs: 800 });

    const data = estimateLogicalErrorRate(instance, {
      noiseMode: NOISE.DATA, d: 5, rounds: 1, p: 0.05, runs: 4000,
    });
    const phenom = estimateLogicalErrorRate(instance, {
      noiseMode: NOISE.PHENOM, d: 5, rounds: 5, p: 0.02, runs: 2000,
    });

    return {
      dataRunsPerSecond: data.runsPerSecond,
      phenomRunsPerSecond: phenom.runsPerSecond,
      sampleRuns: data.runs + phenom.runs,
    };
  },

  async benchmark(instance, config) {
    return estimateLogicalErrorRate(instance, config);
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
        // The standard convention is T = d rounds of syndrome extraction.
        const rounds = base.noiseMode === NOISE.DATA ? 1 : d;
        const { rate, runs } = estimateLogicalErrorRate(instance, { ...base, d, p, rounds });
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
      const merged = { ...base, ...cell };
      if (merged.noiseMode !== NOISE.DATA && merged.rounds == null) merged.rounds = merged.d;
      const { rate, runs } = estimateLogicalErrorRate(instance, merged);
      const entry = { ...cell, pL: rate, runs };
      results.push(entry);
      report({ done: i + 1, total: cells.length, cell: entry });
    }
    return { results };
  },
};

self.onmessage = async (event) => {
  const { id, op, payload } = event.data;
  const report = (progress) => self.postMessage({ id, type: 'progress', ...progress });

  try {
    const handler = OPS[op];
    if (!handler) throw new Error(`unknown operation "${op}"`);
    const instance = await engine();
    const result = await handler(instance, payload ?? { ...DEFAULTS }, report);
    self.postMessage({ id, type: 'done', result });
  } catch (error) {
    self.postMessage({ id, type: 'error', message: error?.message ?? String(error) });
  }
};
