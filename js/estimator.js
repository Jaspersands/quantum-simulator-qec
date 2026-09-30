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

/* -- The fuller model: factories, floor plans, reaction time, storage ----- */

/**
 * ε per patch per cycle at distance d under uniform circuit noise p, from the
 * law fitted to this engine's own SD6 simulations (data/estimate/noise.json):
 * ε = A (p / p_th)^((d + 1) / 2).
 */
export function epsilonUniform(d, p, { A, pth }) {
  return A * (p / pth) ** ((d + 1) / 2);
}

/**
 * Litinski's data blocks (A Game of Surface Codes, Sec. 2): tiles for n data
 * qubits, and time steps of d cycles to consume one magic state.
 */
export const BLOCKS = {
  compact: { label: 'compact', tiles: (n) => 1.5 * n + 3, steps: 9 },
  intermediate: { label: 'intermediate', tiles: (n) => 2 * n + 4, steps: 5 },
  fast: { label: 'fast', tiles: (n) => 2 * n + Math.sqrt(8 * n) + 1, steps: 1 },
};

/** Physical qubits in one tile at distance d, as Litinski counts them: 2d². */
export const tileQubits = (d) => 2 * d * d;

/**
 * The factory that supplies a Toffoli's magic states at physical error p,
 * keeping each Toffoli's share of error at most `target`, or null when none
 * applies. Cultivation is Gidney's (2025): T states cultivated to 1e-7,
 * 8T-to-CCZ at 28 p_T², a factory of 3 × 4 patches making one CCZ per 150
 * rounds at d = 25 (scaled with d), tabulated at p ≤ 0.1% only. Distillation
 * is the cheapest of Litinski's protocols (2019, Table 1), from the rows
 * tabulated at the smallest noise at least p; a CCZ row supplies a Toffoli,
 * a T row four.
 */
export function factoryFor(kind, p, d, target, sources) {
  if (kind === 'cultivation') {
    const c = sources.cultivation;
    const error = c.ccz_error_per_t2 * c.t_error ** 2;
    if (p > c.p_max * (1 + 1e-9) || error > target) return null;
    return {
      kind, name: 'cultivation, 8T-to-CCZ', qubits: c.patches * 2 * (d + 1) ** 2,
      cyclesPerState: (c.rounds_per_ccz_at_d25 * d) / 25, errorPerToffoli: error, statesPerToffoli: 1,
    };
  }
  const tabulated = [...new Set(sources.litinski_factories.map((r) => r.p_phys))].sort((a, b) => a - b);
  const at = tabulated.find((x) => x >= p * (1 - 1e-9));
  if (at == null) return null;
  const per = (r) => (r.ccz ? 1 : 4);
  const fits = sources.litinski_factories.filter((r) => r.p_phys === at && per(r) * r.p_out <= target);
  if (!fits.length) return null;
  const best = fits.sort((a, b) => a.qubitcycles * per(a) - b.qubitcycles * per(b))[0];
  return {
    kind, name: best.name, qubits: best.qubits, cyclesPerState: best.qubitcycles / best.qubits,
    errorPerToffoli: per(best) * best.p_out, statesPerToffoli: per(best),
  };
}

/** Why no factory of `kind` serves noise p at `target` error per Toffoli, in words. */
export function factoryWhyNot(kind, p, target, sources) {
  const sup = (t) => t.replace('-', '⁻').replace(/\d/g, (c) => '⁰¹²³⁴⁵⁶⁷⁸⁹'[c]);
  const sci = (x) => { const [m, e] = x.toExponential(1).split('e'); return `${m} × 10${sup(e.replace('+', ''))}`; };
  if (kind === 'cultivation') {
    const c = sources.cultivation;
    if (p > c.p_max * (1 + 1e-9)) return `cultivation is characterized at p ≤ ${c.p_max * 100}% only`;
    return `cultivation's CCZ error, ${sci(c.ccz_error_per_t2 * c.t_error ** 2)}, is more than the ${sci(target)} per Toffoli this budget allows`;
  }
  const top = Math.max(...sources.litinski_factories.map((r) => r.p_phys));
  if (p > top * (1 + 1e-9)) return `no distillation protocol is tabulated above p = ${top * 100}%`;
  return `no tabulated distillation protocol reaches ${sci(target)} per Toffoli`;
}

/**
 * Seconds a decode of one window takes at distance d: the measured p99
 * ({d: microseconds}, section 12's parallel windows with correlated matching)
 * at the smallest measured distance at least d, or beyond them a power law
 * through the two largest; `overrideUs`, when given, instead.
 */
export function latencyAt(d, measured, overrideUs = null) {
  if (overrideUs != null) return overrideUs * 1e-6;
  const ds = Object.keys(measured).map(Number).sort((a, b) => a - b);
  const at = ds.find((x) => x >= d);
  if (at != null) return measured[at] * 1e-6;
  const [a, b] = ds.slice(-2);
  const slope = Math.log(measured[b] / measured[a]) / Math.log(b / a);
  return measured[b] * (d / b) ** slope * 1e-6;
}

/**
 * The fuller estimate. Options: `noise` ({kind: 'uniform', p, fit} or
 * {kind: 'measured', p, model}), `budget`, `block`, `factory`, `storage`
 * ('surface' | 'yoked' | 'gross'), `cycleSeconds`, `controlSeconds`,
 * `latency` ({d: µs}) or `latencyOverrideUs`, `parallel` (Toffolis in
 * flight), `grossPerCycle` (a gross module's error per cycle at this p), `cores`.
 *
 * Half the budget goes to magic states, which fixes the factory. Each
 * Toffoli takes the larger of its Clifford steps (block steps × states × d
 * cycles) and its reaction (decoder latency plus control delay), divided by
 * the Toffolis in flight; factories are as many as keep up. Every tile is
 * priced as an idle patch for the whole run, which errs on the safe side
 * (section 14's CNOT fails less often than its patches held idle). The
 * distance is the smallest odd d that meets the budget.
 */
export function estimateFull(algorithm, o, sources, maxD = 201) {
  const storage = o.storage ?? 'surface';
  const cold = storage === 'surface' ? 0 : Math.max(0, algorithm.qubits - (algorithm.hot ?? algorithm.qubits));
  const hot = algorithm.qubits - cold;
  const block = BLOCKS[o.block ?? 'fast'];
  const tiles = Math.ceil(block.tiles(hot));
  const eps = (d) => (o.noise.kind === 'uniform' ? epsilonUniform(d, o.noise.p, o.noise.fit) : epsilonAt(d, o.noise.model));
  const target = o.budget / 2 / algorithm.toffolis;
  const modules = Math.ceil(cold / sources.gross_module.logical);
  if (storage === 'gross' && cold > 0 && !(o.grossPerCycle > 0)) return { error: 'the gross code is measured at p = 0.2% to 0.6% only' };
  for (let d = 3; d <= maxD; d += 2) {
    const factory = factoryFor(o.factory, o.noise.p, d, target, sources);
    if (!factory) return { error: factoryWhyNot(o.factory, o.noise.p, target, sources) };
    const clifford = block.steps * factory.statesPerToffoli * d * o.cycleSeconds;
    const reaction = latencyAt(d, o.latency ?? {}, o.latencyOverrideUs) + o.controlSeconds;
    const perToffoli = Math.max(clifford, reaction) / (o.parallel ?? 1);
    const seconds = algorithm.toffolis * perToffoli;
    const cycles = seconds / o.cycleSeconds;
    const storageFailure = storage === 'yoked' ? cold * cycles * sources.yoked.error_per_round
      : storage === 'gross' ? modules * cycles * (o.grossPerCycle ?? 0) : 0;
    const memory = tiles * cycles * eps(d);
    const magic = algorithm.toffolis * factory.errorPerToffoli;
    // Storage and magic states do not improve with d, and the run only lengthens: past the budget here, past it everywhere.
    if (storageFailure + magic > o.budget) {
      return {
        error: `${storage === 'yoked' ? `yoked storage at ${sources.yoked.error_per_round} per round` : 'gross-code storage'} would `
          + `fail ${(storageFailure * 100).toPrecision(2)}% on its own even over the shortest run it could need (${duration(seconds)}), past the `
          + `${(o.budget * 100).toPrecision(2)}% budget`,
        seconds, storageFailure,
      };
    }
    if (memory + storageFailure + magic > o.budget) continue;
    const perFactory = 1 / (factory.cyclesPerState * o.cycleSeconds);
    const factories = Math.ceil(factory.statesPerToffoli / perToffoli / perFactory - 1e-9);
    const storageQubits = storage === 'yoked' ? cold * sources.yoked.qubits_per_logical
      : storage === 'gross' ? modules * (sources.gross_module.qubits + sources.gross_module.ancillas) : 0;
    const qubits = { block: tiles * tileQubits(d), factories: factories * factory.qubits, storage: storageQubits };
    qubits.total = qubits.block + qubits.factories + qubits.storage;
    return {
      d, tiles, hot, cold, epsilon: eps(d), qubits, seconds, perToffoli, clifford, reaction,
      bound: clifford >= reaction ? 'clifford' : 'reaction', factory, factories,
      failure: { memory, storage: storageFailure, magic, total: memory + storageFailure + magic },
      cores: o.cores ? decodingCores(d, { qubits: tiles }, o.cores) : null,
    };
  }
  return { error: `no distance up to ${maxD} meets the budget` };
}

/** Section 12's measured p99 window decode per d (µs), parallel windows with correlated matching. */
export function latencyFrom(latencyDoc) {
  const out = {};
  for (const s of latencyDoc.streams) if (s.mode === 'parallel' && s.matcher === 'correlated') out[s.d] = s.window_us.p99;
  return out;
}

/**
 * The page's defaults: uniform noise p = 0.1% by this engine's fit, cultivation, the fast block,
 * surface storage, 1 µs cycles, a 10 µs control delay, this project's decoder latency.
 */
export function defaultOptions(fit, latency, cores) {
  return {
    noise: { kind: 'uniform', p: 1e-3, fit }, budget: 0.01, block: 'fast', factory: 'cultivation', storage: 'surface',
    cycleSeconds: 1e-6, controlSeconds: 10e-6, latency, latencyOverrideUs: null, parallel: 1, cores,
  };
}

/**
 * The validation against the sources: RSA-2048 entirely under Gidney's assumptions, ε and schedule
 * first, then each option switched to this project's; FeMoco under Lee et al.'s noise. Returns
 * [{group, label, algorithm, options, reference: {qubits, seconds}}].
 */
export function validationCases(sources, defaults) {
  const rsa = sources.algorithms.rsa2048;
  const perPeriod = (rsa.toffolis * 25e-6) / (rsa.reference.hours_per_shot * 3600);
  // His ε: 1e-15 per patch per round at d = 25, tenfold per two steps of d at p = 0.1%.
  const his = { kind: 'uniform', p: 1e-3, fit: { A: 0.01, pth: 0.01 } };
  const gidney = { ...defaults, noise: his, budget: 0.067, storage: 'yoked', latencyOverrideUs: 0, parallel: perPeriod };
  const rsaRef = { qubits: rsa.reference.qubits, seconds: rsa.reference.hours_per_shot * 3600 };
  const fem = sources.algorithms.femoco_reiher;
  const femRef = { qubits: fem.reference.qubits, seconds: fem.reference.days * 86400 };
  return [
    { group: 'rsa', label: "Gidney's assumptions, ε and schedule", algorithm: rsa, options: gidney, reference: rsaRef },
    { group: 'rsa', label: "this engine's measured ε instead", algorithm: rsa, options: { ...gidney, noise: defaults.noise }, reference: rsaRef },
    { group: 'rsa', label: 'one Toffoli per step instead', algorithm: rsa, options: { ...gidney, parallel: 1 }, reference: rsaRef },
    { group: 'rsa', label: 'surface-code storage instead of yoked', algorithm: rsa, options: { ...gidney, storage: 'surface' }, reference: rsaRef },
    { group: 'rsa', label: "this project's decoder latency instead", algorithm: rsa, options: { ...gidney, latencyOverrideUs: null }, reference: rsaRef },
    { group: 'femoco', label: "Lee et al.'s noise, distillation", algorithm: fem, options: { ...defaults, factory: 'distillation', latencyOverrideUs: 0 }, reference: femRef },
    { group: 'femoco', label: 'cultivation instead', algorithm: fem, options: { ...defaults, latencyOverrideUs: 0 }, reference: femRef },
  ];
}

/** Gidney's schedule: Toffolis per d = 25 lattice-surgery period of 25 µs. */
export const gidneyParallel = (sources) => {
  const rsa = sources.algorithms.rsa2048;
  return (rsa.toffolis * 25e-6) / (rsa.reference.hours_per_shot * 3600);
};

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
