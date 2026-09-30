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
import { estimate, distanceFor, physicalQubits, bigNumber, duration, modelFrom, coresFrom, estimateFull, latencyFrom, defaultOptions,
  validationCases, gidneyParallel } from '../js/estimator.js';

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
const latency = latencyFrom(latencyDoc);
const DEFAULTS = defaultOptions(fit, latency, cores);
const text = (r) => r.error ? { error: r.error } : ({
  qubits: bigNumber(r.qubits.total), block: bigNumber(r.qubits.block), factories: bigNumber(r.qubits.factories),
  storage: bigNumber(r.qubits.storage), seconds: duration(r.seconds),
  perToffoli: r.perToffoli < 1e-3 ? `${(r.perToffoli * 1e6).toPrecision(3)} µs` : `${(r.perToffoli * 1e3).toPrecision(3)} ms`,
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

// Validation: one shared definition with the page (js/estimator.js, validationCases).
const perPeriod = gidneyParallel(sources);
const validation = { rsa: [], femoco: [], rsaToffolisPerPeriod: perPeriod, epsAt25: fit.A * (1e-3 / fit.pth) ** 13 };
for (const c of validationCases(sources, DEFAULTS)) {
  const r = estimateFull(c.algorithm, c.options, sources);
  validation[c.group].push({ label: c.label, ...r, text: text(r),
    qubitRatio: r.error ? null : r.qubits.total / c.reference.qubits, timeRatio: r.error ? null : r.seconds / c.reference.seconds });
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
