// The resource estimate for three illustrative sizes under each measured Λ,
// from the committed data, with the page's own model (js/estimator.js).
// Run: node tools/estimate.mjs [--json out.json]   (prints a Markdown table)
import { readFile, writeFile } from 'node:fs/promises';
import { estimate, distanceFor, physicalQubits, bigNumber, duration, modelFrom, coresFrom } from '../js/estimator.js';

const root = new URL('../data/', import.meta.url);
const load = async (path) => JSON.parse(await readFile(new URL(path, root), 'utf8'));
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
const cores = coresFrom(await load('realtime/latency.json'));

const rows = [];
for (const p of PRESETS) {
  for (const s of SOURCES) {
    const r = estimate(p, models[s.id], { ...OPTIONS, cores });
    // Formatted here, by the page's own formatters, so the README and report read as the page does.
    rows.push({ preset: p.name, source: s.id, lambda: models[s.id].lambda, ...r,
      text: { physical: bigNumber(r.physical), seconds: duration(r.seconds), cores: r.cores == null ? '—' : bigNumber(r.cores) } });
  }
}
// What Λ = 4 would do for the medium size: the lever.
const medium = PRESETS[1];
const at4 = distanceFor(medium, { ...models.ours, lambda: 4 }, OPTIONS.budget);
const lever = { lambda: 4, d: at4, physical: physicalQubits(at4, medium, OPTIONS.overhead) };
lever.text = { physical: bigNumber(lever.physical) };

console.log('| algorithm | Λ (from) | d | physical qubits | run time | decoding cores |');
console.log('|---|---|---|---|---|---|');
for (const r of rows) {
  const p = PRESETS.find((x) => x.name === r.preset);
  const s = SOURCES.find((x) => x.id === r.source);
  console.log(`| ${p.label} | ${r.lambda.toFixed(2)} (${s.label}) | ${r.d} | ${r.text.physical} | ${r.text.seconds} | ${r.text.cores} |`);
}
console.log(`\nAt Λ = 4 the medium size would need d = ${lever.d} and ${lever.text.physical} physical qubits.`);
const at = process.argv.indexOf('--json');
if (at >= 0) await writeFile(process.argv[at + 1], `${JSON.stringify({ models, cores, options: OPTIONS, rows, lever }, null, 1)}\n`);
