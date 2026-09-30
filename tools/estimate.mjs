// The resource estimate, from the committed data, with the page's own model (js/estimator.js).
// Run: node tools/estimate.mjs [--json out.json]   (prints Markdown tables)
//
// Two parts:
// - the Λ grid: three illustrative sizes under each Λ measured on Willow (the simple model, where Λ
//   is the lever);
// - the fuller model: real algorithms and the illustrations, with magic-state factories, Litinski's
//   floor plans, reaction time from this project's decoder and storage options, at uniform noise
//   p = 0.1% (this engine's own fit), and its validation against Gidney's RSA-2048 and Lee et al.'s
//   FeMoco, each ratio's reason found by switching one option.
import { readFile, writeFile } from 'node:fs/promises';
import { estimate, distanceFor, physicalQubits, bigNumber, duration, modelFrom, coresFrom, estimateFull } from '../js/estimator.js';

const root = new URL('../data/', import.meta.url);
const load = async (path) => JSON.parse(await readFile(new URL(path, root), 'utf8'));

// -- The Λ grid -------------------------------------------------------------
const PRESETS = [
  { name: 'small', label: '100 qubits × 10⁶ operations', qubits: 100, operations: 1e6 },
  { name: 'medium', label: '1,000 qubits × 10⁹ operations', qubits: 1000, operations: 1e9 },
  { name: 'large', label: '10,000 qubits × 10¹² operations', qubits: 10000, operations: 1e12 },
];
const SOURCES = [
  { id: 'ours', file: 'google-results/willow.json', key: 'ours/si1000/correlated', label: 'ours, correlated matching' },
  { id: 'libra', file: 'google-results/willow.json', key: 'google/libra_decoder_with_rl_optimized_prior', label: "Google's Libra" },
  { id: 'belief', file: 'belief/willow.json', key: 'ours/si1000/belief', label: 'ours, belief-matching' },
];
const OPTIONS = { budget: 0.01, overhead: 2, cycleSeconds: 1.1e-6 };

const models = {};
for (const s of SOURCES) models[s.id] = modelFrom(Object.values((await load(s.file)).experiments), s.key);
const latencyDoc = await load('realtime/latency.json');
const cores = coresFrom(latencyDoc);

const rows = [];
for (const p of PRESETS) {
  for (const s of SOURCES) {
    const r = estimate(p, models[s.id], { ...OPTIONS, cores });
    rows.push({ preset: p.name, source: s.id, lambda: models[s.id].lambda, ...r,
      text: { physical: bigNumber(r.physical), seconds: duration(r.seconds), cores: r.cores == null ? '—' : bigNumber(r.cores) } });
  }
}
const medium = PRESETS[1];
const at4 = distanceFor(medium, { ...models.ours, lambda: 4 }, OPTIONS.budget);
const lever = { lambda: 4, d: at4, physical: physicalQubits(at4, medium, OPTIONS.overhead) };
lever.text = { physical: bigNumber(lever.physical) };

// -- The fuller model ---------------------------------------------------------
const sources = await load('estimate/sources.json');
const fit = (await load('estimate/noise.json')).fit;
// Section 12's measured p99 window decode, parallel windows with correlated matching, per d.
const latency = {};
for (const s of latencyDoc.streams) if (s.mode === 'parallel' && s.matcher === 'correlated') latency[s.d] = s.window_us.p99;

/** The page's defaults: uniform 0.1% (this engine's fit), cultivation, fast block, surface storage, this project's decoder. */
export const DEFAULTS = {
  noise: { kind: 'uniform', p: 1e-3, fit }, budget: 0.01, block: 'fast', factory: 'cultivation', storage: 'surface',
  cycleSeconds: 1e-6, controlSeconds: 10e-6, latency, parallel: 1, cores,
};
const text = (r) => r.error ? { error: r.error } : ({
  qubits: bigNumber(r.qubits.total), block: bigNumber(r.qubits.block), factories: bigNumber(r.qubits.factories),
  storage: bigNumber(r.qubits.storage), seconds: duration(r.seconds), perToffoli: `${(r.perToffoli * 1e6).toPrecision(3)} µs`,
  cores: r.cores == null ? '—' : bigNumber(r.cores),
});
const ALGS = ['rsa2048', 'femoco_reiher', 'femoco_li', 'small', 'medium', 'large'];
const full = [];
for (const key of ALGS) {
  const alg = sources.algorithms[key];
  for (const [variant, extra] of [['measured decoder', {}], ['decoder within the control delay', { latencyOverrideUs: 0 }]]) {
    const r = estimateFull(alg, { ...DEFAULTS, ...extra }, sources);
    full.push({ key, label: alg.label, variant, ...r, text: text(r) });
  }
}

// Validation. Gidney's assumptions: uniform 0.1%, 1 µs cycles, a 10 µs reaction that includes decoding,
// cultivation, a 6.7% failure budget per shot (93.3% succeed), cold qubits yoked, and his schedule's
// parallelism: 6.5e9 Toffolis in 12.07 h is this many per d = 25 lattice-surgery period of 25 µs.
const rsa = sources.algorithms.rsa2048;
const perPeriod = (rsa.toffolis * 25e-6) / (rsa.reference.hours_per_shot * 3600);
// His ε: 1e-15 per patch per round at d = 25, falling tenfold per two steps of distance at p = 0.1%,
// that is ε = 0.01 (p / 1%)^((d + 1)/2). The first case is entirely his; each after it switches one
// option to this project's, so each change in the ratios has one named cause.
const hisEps = { kind: 'uniform', p: 1e-3, fit: { A: 0.01, pth: 0.01 } };
const gidney = { ...DEFAULTS, noise: hisEps, budget: 0.067, storage: 'yoked', latencyOverrideUs: 0, parallel: perPeriod };
const cases = [
  ['Gidney\'s assumptions, ε and schedule', gidney],
  ['this engine\'s measured ε instead', { ...gidney, noise: DEFAULTS.noise }],
  ['one Toffoli per step instead', { ...gidney, parallel: 1 }],
  ['surface-code storage instead of yoked', { ...gidney, storage: 'surface' }],
  ['this project\'s decoder latency instead', { ...gidney, latencyOverrideUs: null }],
];
const validation = { rsa: [], femoco: [], rsaToffolisPerPeriod: perPeriod };
for (const [label, o] of cases) {
  const r = estimateFull(rsa, o, sources);
  validation.rsa.push({ label, ...r, text: text(r),
    qubitRatio: r.error ? null : r.qubits.total / rsa.reference.qubits,
    timeRatio: r.error ? null : r.seconds / 3600 / rsa.reference.hours_per_shot });
}
// Our measured ε at p = 0.1%, extrapolated to Gidney's d = 25, against his 1e-15 per round.
validation.epsAt25 = fit.A * (1e-3 / fit.pth) ** 13;
const fem = sources.algorithms.femoco_reiher;
for (const [label, o] of [['Lee et al.\'s noise, distillation', { ...DEFAULTS, factory: 'distillation', latencyOverrideUs: 0 }],
  ['cultivation instead', { ...DEFAULTS, latencyOverrideUs: 0 }]]) {
  const r = estimateFull(fem, o, sources);
  validation.femoco.push({ label, ...r, text: text(r),
    qubitRatio: r.error ? null : r.qubits.total / fem.reference.qubits,
    timeRatio: r.error ? null : r.seconds / 86400 / fem.reference.days });
}

// -- Output ---------------------------------------------------------------------
console.log('| algorithm | Λ (from) | d | physical qubits | run time | decoding cores |');
console.log('|---|---|---|---|---|---|');
for (const r of rows) {
  const p = PRESETS.find((x) => x.name === r.preset);
  const s = SOURCES.find((x) => x.id === r.source);
  console.log(`| ${p.label} | ${r.lambda.toFixed(2)} (${s.label}) | ${r.d} | ${r.text.physical} | ${r.text.seconds} | ${r.text.cores} |`);
}
console.log(`\nAt Λ = 4 the medium size would need d = ${lever.d} and ${lever.text.physical} physical qubits.\n`);
console.log('| algorithm | decoder | d | physical qubits (block + factories + storage) | per Toffoli | run time | bound |');
console.log('|---|---|---|---|---|---|---|');
for (const r of full) {
  if (r.error) { console.log(`| ${r.label} | ${r.variant} | — | ${r.error} | | | |`); continue; }
  console.log(`| ${r.label} | ${r.variant} | ${r.d} | ${r.text.qubits} (${r.text.block} + ${r.text.factories} + ${r.text.storage}) | ${r.text.perToffoli} | ${r.text.seconds} | ${r.bound} |`);
}
console.log('\n| RSA-2048 | d | physical qubits | × Gidney\'s 897,864 | time per shot | × his 12.07 h |');
console.log('|---|---|---|---|---|---|');
for (const r of validation.rsa) {
  console.log(r.error ? `| ${r.label} | — | ${r.error} | | | |`
    : `| ${r.label} | ${r.d} | ${r.text.qubits} | ${r.qubitRatio.toFixed(2)} | ${r.text.seconds} | ${r.timeRatio.toFixed(2)} |`);
}
console.log(`\nGidney's schedule runs ${perPeriod.toFixed(1)} Toffolis per 25 µs lattice-surgery period. This engine's ε at p = 0.1%, extrapolated to d = 25, is ${validation.epsAt25.toExponential(1)} per round, where Gidney assumes 1e-15.`);
console.log('\n| FeMoco (Reiher) | d | physical qubits | × Lee et al.\'s 4 million | run time | × their 3 days |');
console.log('|---|---|---|---|---|---|');
for (const r of validation.femoco) {
  console.log(r.error ? `| ${r.label} | — | ${r.error} | | | |`
    : `| ${r.label} | ${r.d} | ${r.text.qubits} | ${r.qubitRatio.toFixed(2)} | ${r.text.seconds} | ${r.timeRatio.toFixed(2)} |`);
}
const at = process.argv.indexOf('--json');
if (at >= 0) {
  await writeFile(process.argv[at + 1], `${JSON.stringify({ models, cores, options: OPTIONS, rows, lever, fit, latency, full, validation }, null, 1)}\n`);
}
