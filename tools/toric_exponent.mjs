/**
 * Measure the phenomenological critical exponent on the independent toric code,
 * with an approximate matcher and with an exact one.
 *
 * This exists to settle a disagreement. The engine reports ν ≈ 0.89 ± 0.04 for
 * phenomenological noise; an earlier run of this harness reported 1.41 ± 0.28,
 * which is about 1.8σ away — consistent with the engine's value, consistent with
 * the ~1.0 the literature gives for this universality class, and therefore
 * consistent with nothing useful. The suspected cause was this side's decoder:
 * nearest-pair-plus-2-opt is not minimum-weight, and its threshold came out
 * 1.2% against the engine's 3.4%. A decoder that weak can distort an exponent.
 *
 * So: run both matchers over the same lattices, the same rates and the same
 * seeds, and see whether ν moves.
 *
 * Usage: node tools/toric_exponent.mjs [shotsPerPoint]
 */
import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { makeToric, shot, approxMatch, makeExactMatcher } from './independent_toric.mjs';
import { fitThreshold } from '../js/compute.js';

const here = dirname(fileURLToPath(import.meta.url));
const wasm = readFileSync(join(here, '..', 'stabilizer_qec.wasm'));
const { instance } = await WebAssembly.instantiate(wasm, {});

const SHOTS = Number(process.argv[2] || 4000);
const SIZES = [4, 6, 8, 10];
const RATES = [0.010, 0.016, 0.021, 0.025, 0.029, 0.034, 0.040, 0.050];

/** Deterministic per-(L, p, seed) stream, so both matchers see identical noise. */
function makeRand(seed) {
  let s = seed >>> 0;
  return () => {
    s ^= s << 13; s >>>= 0;
    s ^= s >> 17;
    s ^= s << 5; s >>>= 0;
    return s / 4294967296;
  };
}

function measure(matcher, label) {
  const points = [];
  for (const L of SIZES) {
    const code = makeToric(L);
    for (const p of RATES) {
      let fails = 0;
      const rand = makeRand(0x9e3779b9 ^ (L * 2654435761) ^ Math.round(p * 1e6));
      for (let i = 0; i < SHOTS; i++) fails += shot(code, L, p, rand, matcher);
      points.push({ d: L, p, pL: fails / SHOTS, runs: SHOTS });
    }
  }
  const fit = fitThreshold(points, { bootstrap: 120 });
  if (!fit.ok) {
    console.log(`  ${label.padEnd(12)} REFUSED — ${fit.reason.slice(0, 70)}`);
    return null;
  }
  const ci = Number.isFinite(fit.pThLo)
    ? `${(fit.pThLo * 100).toFixed(2)}–${(fit.pThHi * 100).toFixed(2)}%`
    : 'not resolved';
  const nu = fit.nuDetermined === false ? `${fit.nu.toFixed(2)} (undetermined)` : fit.nu.toFixed(2);
  console.log(`  ${label.padEnd(12)} p_th ${(fit.pTh * 100).toFixed(3)}%  CI ${ci.padEnd(16)}`
    + `  ν ${String(nu).padEnd(18)} χ² ${fit.reducedChi2.toFixed(2)}`);
  return fit;
}

console.log(`independent toric code, phenomenological noise, T = L`);
console.log(`L = ${SIZES.join(', ')} · ${SHOTS} shots per point · ${SIZES.length * RATES.length} points\n`);

const approx = measure(approxMatch, 'approximate');
const exact = makeExactMatcher(instance.exports);
const exactFit = measure(exact, 'exact');
console.log(`\n  exact matcher: ${exact.stats.exact} solves, ${exact.stats.fellBack} fell back`);
if (approx && exactFit) {
  console.log(`\n  ν moved ${(exactFit.nu - approx.nu >= 0 ? '+' : '')}`
    + `${(exactFit.nu - approx.nu).toFixed(2)} when the decoder became exact;`
    + ` threshold moved ${((exactFit.pTh - approx.pTh) * 100).toFixed(2)} points.`);
}
