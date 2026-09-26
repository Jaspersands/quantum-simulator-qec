/**
 * The resource estimator's model, kept pure so the site tests can check it.
 *
 * An algorithm is N logical qubits and D logical operations, each a lattice-
 * surgery step, within a total failure budget δ. Logical error per cycle
 * falls with distance as ε_d = ε₀ Λ^−(d − d₀)/2 from a measured ε₀ at d₀;
 * every operation takes d merged rounds (measured: the merge outcome's
 * failure rate flattens by T = d), during which every patch is exposed; so
 * one operation fails with probability about N · d · ε_d.
 */

import { epsilonByDistance, lambdaFit } from './lambda-fit.js';
import { cores as coresToKeepUp } from './realtime-format.js';

/**
 * The model's inputs from the committed data, one definition for the page
 * and tools/estimate.mjs: ε at the largest measured distance and Λ, point
 * fits from round 10 as section 11 fits them, for decoder `key` over
 * Willow-shaped `records`.
 */
export function modelFrom(records, key) {
  const byD = epsilonByDistance(records, key, { minRounds: 10 });
  const d0 = Math.max(...byD.keys());
  return { eps0: byD.get(d0).eps, d0, lambda: lambdaFit(byD).lambda };
}

/** Cores that keep one stream up at each measured distance (parallel windows, correlated), from latency.json. */
export function coresFrom(latency) {
  const out = {};
  for (const s of latency.streams) {
    if (s.mode !== 'parallel' || s.matcher !== 'correlated') continue;
    const k = coresToKeepUp(s.by_workers);
    if (k) out[s.d] = k;
  }
  return out;
}

/** ε at distance d, extrapolated from ε0 measured at d0 with suppression Λ. */
export function epsilonAt(d, { eps0, d0, lambda }) {
  return eps0 * lambda ** (-(d - d0) / 2);
}

/** Failure probability of the whole algorithm at distance d. */
export function failureAt(d, { qubits, operations }, model) {
  return operations * qubits * d * epsilonAt(d, model);
}

/**
 * The smallest odd distance (≥ 3) that meets the budget, or null if none up
 * to `maxD` does (or Λ ≤ 1, where distance does not help).
 */
export function distanceFor(algorithm, model, budget, maxD = 201) {
  if (!(model.lambda > 1)) return null;
  for (let d = 3; d <= maxD; d += 2) {
    if (failureAt(d, algorithm, model) <= budget) return d;
  }
  return null;
}

/** Physical qubits: N rotated patches of 2d² − 1, times the routing overhead. */
export function physicalQubits(d, { qubits }, overhead = 2) {
  return Math.ceil(qubits * (2 * d * d - 1) * overhead);
}

/** Wall time in seconds: D operations of d rounds each. */
export function runtimeSeconds(d, { operations }, cycleSeconds = 1.1e-6) {
  return operations * d * cycleSeconds;
}

/**
 * Decoding cores for N patches at distance d, from measured cores-to-keep-up
 * at measured distances ({d: cores}), extrapolated as a power law through the
 * two largest measured distances.
 */
export function decodingCores(d, { qubits }, measured) {
  const ds = Object.keys(measured).map(Number).sort((a, b) => a - b);
  // Two measured distances at least, or there is no law to extrapolate by.
  if (ds.length < 2) return null;
  const [a, b] = ds.slice(-2);
  const slope = Math.log(measured[b] / measured[a]) / Math.log(b / a);
  const perPatch = d <= b ? measured[ds.find((x) => x >= d) ?? b] : measured[b] * (d / b) ** slope;
  return Math.ceil(qubits * perPatch);
}

/** Everything the page shows for one algorithm and model. */
export function estimate(algorithm, model, { budget, overhead = 2, cycleSeconds = 1.1e-6, cores = null }) {
  const d = distanceFor(algorithm, model, budget);
  if (d == null) return null;
  return {
    d,
    epsilon: epsilonAt(d, model),
    failure: failureAt(d, algorithm, model),
    physical: physicalQubits(d, algorithm, overhead),
    seconds: runtimeSeconds(d, algorithm, cycleSeconds),
    cores: cores ? decodingCores(d, algorithm, cores) : null,
  };
}

/** 3.2e7 → "32 million"; 4.1e9 → "4.1 billion"; 12,345 → "12,345". */
export function bigNumber(x) {
  const scaled = (v, unit) => `${v >= 100 ? Math.round(v).toLocaleString('en-US') : v.toPrecision(2)} ${unit}`;
  if (x >= 1e9) return scaled(x / 1e9, 'billion');
  if (x >= 1e6) return scaled(x / 1e6, 'million');
  return Math.round(x).toLocaleString('en-US');
}

/** Seconds as the largest sensible unit. */
export function duration(s) {
  // Two significant figures, never in exponent form.
  const fig = (v) => (v >= 100 ? Math.round(v).toLocaleString('en-US') : v.toPrecision(2));
  if (s < 1) return `${fig(s * 1000)} ms`;
  if (s < 120) return `${fig(s)} s`;
  if (s < 7200) return `${fig(s / 60)} min`;
  if (s < 172800) return `${fig(s / 3600)} h`;
  if (s < 3.156e7) return `${fig(s / 86400)} days`;
  return `${fig(s / 3.156e7)} years`;
}
