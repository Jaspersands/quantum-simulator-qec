// Threshold sweeps in Node, for the figures the README quotes.
// Run: node tools/sweep.mjs <noiseMode> <codeType> [repeats=4] [decoder=2]
//   noiseMode: 0 data, 1 phenomenological, 2 circuit-level (engine's model), 3 SD6
//   codeType:  0 rotated, 1 XZZX
// Uses the same distances, windows and shot counts as section 07, and the same
// fit (compute.js fitThreshold), so a number here is the number the page's sweep
// would report, measured several times over.
import { readFile } from 'node:fs/promises';
import { runBenchmark } from '../js/engine.js';
import { fitThreshold } from '../js/compute.js';
import { DISTANCES, SWEEP_PS, SWEEP_RUNS } from '../js/sweep-config.js';

const [noiseMode, codeType, repeats = 4, decoder = 2] = process.argv.slice(2).map(Number);
if (!(noiseMode in DISTANCES)) {
  console.error('usage: node tools/sweep.mjs <noiseMode 0-3> <codeType 0-1> [repeats] [decoder]');
  process.exit(2);
}
const bytes = await readFile(new URL('../stabilizer_qec.wasm', import.meta.url));
const pct = (x) => `${(x * 100).toFixed(3)}%`;
const fits = [];
for (let rep = 0; rep < repeats; rep++) {
  const { instance } = await WebAssembly.instantiate(bytes, {});
  const seed = new Uint32Array(2);
  crypto.getRandomValues(seed);
  instance.exports.wasm_seed(seed[0], seed[1]);
  const points = [];
  let refused = 0;
  const t0 = performance.now();
  for (const d of DISTANCES[noiseMode]) {
    for (const p of SWEEP_PS[noiseMode]) {
      const r = runBenchmark(instance, {
        noiseMode, codeType, decoder, d, p, bias: 0.5, runs: SWEEP_RUNS[noiseMode], rounds: noiseMode === 0 ? 1 : d,
      });
      refused += r.decodeErrors ?? 0;
      points.push({ d, p, pL: r.rate, runs: r.runs });
    }
  }
  const fit = fitThreshold(points, { bootstrap: 120 });
  fits.push(fit);
  const secs = ((performance.now() - t0) / 1000).toFixed(0);
  if (!fit.ok) {
    console.log(`sweep ${rep + 1}: no fit (${fit.reason}) · ${secs} s`);
    continue;
  }
  console.log(`sweep ${rep + 1}: p_th ${pct(fit.pTh)} [${pct(fit.pThLo)}, ${pct(fit.pThHi)}]`
    + ` nu ${fit.nuDetermined ? fit.nu.toFixed(2) : 'n/d'}`
    + ` omega ${fit.corrected ? fit.omega.toFixed(2) : 'n/f'}`
    + ` chi2 ${fit.reducedChi2.toFixed(2)} window ${fit.pointsUsed}/${fit.pointsTotal}`
    + ` uncorrected ${fit.leading ? pct(fit.leading.pTh) : 'n/a'}`
    + ` crossings ${fit.crossings.map((c) => `${c.small}/${c.large}@${pct(c.p)}`).join(' ')}`
    + ` refused ${refused} · ${secs} s`);
}
const ok = fits.filter((f) => f.ok);
if (ok.length) {
  const mean = ok.reduce((s, f) => s + f.pTh, 0) / ok.length;
  const sd = Math.sqrt(ok.reduce((s, f) => s + (f.pTh - mean) ** 2, 0) / Math.max(1, ok.length - 1));
  const nus = ok.filter((f) => f.nuDetermined).map((f) => f.nu);
  const omegas = ok.filter((f) => f.corrected).map((f) => f.omega);
  const lead = ok.filter((f) => f.leading).map((f) => f.leading.pTh);
  const avg = (a) => a.reduce((s, x) => s + x, 0) / a.length;
  console.log(`mean of ${ok.length}: p_th ${(mean * 100).toFixed(2)}% ± ${(sd * 100).toFixed(2)}`
    + (nus.length ? `  nu ${avg(nus).toFixed(2)} (${nus.length} determined)` : '  nu not determined')
    + (omegas.length ? `  omega ${avg(omegas).toFixed(2)}` : '')
    + (lead.length ? `  uncorrected ${(avg(lead) * 100).toFixed(2)}%` : ''));
}
