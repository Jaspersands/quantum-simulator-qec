// Node tests for the pure site modules. Run: node tools/site-tests.mjs
import assert from 'node:assert/strict';
import { poisson, footprintRate, layoutFor, chainsFromCorrection, pickPauli } from '../js/opener-math.js';
import { planChunks } from '../js/stream.js';
import { DISTANCES, SWEEP_PS, SWEEP_RUNS } from '../js/sweep-config.js';
import { verdict, formatRel, circuitLabel, microseconds, disagreementText, megabytes } from '../js/xcheck-format.js';
import { logTicks } from '../js/plot.js';
import { splitRuns, mergeStream, mergeStreamResults } from '../js/pool-merge.js';
import { poolSize } from '../js/pool.js';
import { decoderLabel, percent as pct, percentRange, ratio } from '../js/hardware-format.js';
import { fidelity, fitEpsilon, epsilonByDistance, lambdaFit, bootstrap, decoderKeys, seededRandom } from '../js/lambda-fit.js';

let passed = 0, failed = 0;
const test = (name, fn) => { try { fn(); passed++; console.log(`  ✓ ${name}`); } catch (e) { failed++; console.log(`  ✗ ${name}\n    ${e.message}`); } };
let seed = 20260918;
const rand = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296; };

test('poisson: mean within five standard errors over 50,000 draws', () => {
  for (const lambda of [0.04, 1, 4]) {
    let sum = 0; const N = 50000;
    for (let i = 0; i < N; i++) sum += poisson(lambda, rand);
    const tol = 5 * Math.sqrt(lambda / N);
    assert.ok(Math.abs(sum / N - lambda) < tol, `lambda ${lambda}: mean ${sum / N}`);
  }
  assert.equal(poisson(0, rand), 0);
});

test('footprintRate: peak at the pointer, zero beyond three sigma, monotone between', () => {
  assert.equal(footprintRate(0, 1.1, 8), 8);
  assert.equal(footprintRate(3.31, 1.1, 8), 0);
  let prev = Infinity;
  for (let r = 0; r < 3.3; r += 0.1) { const v = footprintRate(r, 1.1, 8); assert.ok(v <= prev); prev = v; }
});

test('layoutFor: the patch fills the height and is centred horizontally', () => {
  const L = layoutFor(1000, 420, 15, 20);
  assert.equal(L.size, 380);
  assert.equal(L.originX, 310);
  assert.equal(L.originY, 20);
  assert.ok(Math.abs(L.unit - 380 / 30) < 1e-9);
});

test('pickPauli follows the mix', () => {
  const mix = [['X', 0.8], ['Z', 0.15], ['Y', 0.05]];
  const n = { X: 0, Z: 0, Y: 0 };
  for (let i = 0; i < 20000; i++) n[pickPauli(rand, mix)]++;
  assert.ok(n.X > 15000 && n.Z > 2400 && n.Y > 700, JSON.stringify(n));
});

// A stand-in for engine.Session with the geometry of a d = 5 rotated patch:
// qubits at odd doubled coordinates, checks at even ones, checkerboard types.
function fakeSession(d) {
  const dataQubits = [], stabilizers = [];
  for (let r = 0; r < d; r++) for (let c = 0; c < d; c++) dataQubits.push({ idx: r * d + c, x: 2 * c + 1, y: 2 * r + 1 });
  let idx = 0;
  for (let y = 0; y <= 2 * d; y += 2) for (let x = 0; x <= 2 * d; x += 2) {
    const inside = x > 0 && x < 2 * d && y > 0 && y < 2 * d;
    const type = ((x / 2 + y / 2) % 2 === 0) ? 0 : 1;              // 0 = Z, 1 = X
    const onLR = (x === 0 || x === 2 * d) && y > 0 && y < 2 * d;    // X half-plaquettes left/right
    const onTB = (y === 0 || y === 2 * d) && x > 0 && x < 2 * d;    // Z half-plaquettes top/bottom
    if (inside || (onLR && type === 1) || (onTB && type === 0)) stabilizers.push({ idx: idx++, x, y, type });
  }
  const support = new Map(), touchedBy = new Map();
  for (const q of dataQubits) touchedBy.set(q.idx, []);
  for (const s of stabilizers) {
    const qs = dataQubits.filter((q) => Math.abs(q.x - s.x) === 1 && Math.abs(q.y - s.y) === 1).map((q) => q.idx);
    support.set(s.idx, qs);
    for (const q of qs) touchedBy.get(q).push(s.idx);
  }
  return { d, dataQubits, stabilizers, support, touchedBy, numData: d * d };
}

test('chainsFromCorrection: a row of X corrections from the left edge is one chain, boundary first', () => {
  const s = fakeSession(5);
  const cx = new Uint8Array(25), cz = new Uint8Array(25);
  cx[2 * 5 + 0] = 1; cx[2 * 5 + 1] = 1; cx[2 * 5 + 2] = 1;   // row 2, columns 0..2
  const chains = chainsFromCorrection(s, cx, cz);
  assert.equal(chains.length, 1);
  assert.equal(chains[0].type, 'X');
  assert.deepEqual(chains[0].qubits, [10, 11, 12]);
});

test('chainsFromCorrection: separate X and Z components come back as separate chains', () => {
  const s = fakeSession(5);
  const cx = new Uint8Array(25), cz = new Uint8Array(25);
  cx[12] = 1; cz[0] = 1; cz[5] = 1;                          // lone X in the middle; Z pair down column 0
  const chains = chainsFromCorrection(s, cx, cz);
  assert.equal(chains.length, 2);
  assert.deepEqual(chains.find((c) => c.type === 'X').qubits, [12]);
  assert.equal(chains.find((c) => c.type === 'Z').qubits.length, 2);
});

test('planChunks covers the request exactly with no empty chunk', () => {
  for (const total of [1, 249, 250, 251, 5000, 200000]) {
    const chunks = planChunks(total);
    assert.equal(chunks.reduce((a, b) => a + b, 0), total, `total ${total}`);
    assert.ok(chunks.every((c) => c > 0), `total ${total} has an empty chunk`);
    assert.ok(chunks.length <= 61, `total ${total}: ${chunks.length} chunks`);
  }
  assert.deepEqual(planChunks(600), [250, 250, 100]);
});

test('verdict: identical, differing, unavailable, stale, engine error, pending', () => {
  const base = { ok: true, missing: 0, extra: 0, differing: 0 };
  assert.deepEqual(verdict(base), { text: 'identical', tone: 'ok' });
  assert.deepEqual(verdict({ ...base, missing: 2, differing: 1 }), { text: '3 differ', tone: 'fail' });
  assert.equal(verdict({ ok: false, unavailable: true, error: '404' }).text, 'reference unavailable');
  assert.equal(verdict({ ...base, stale: true }).tone, 'fail');
  assert.equal(verdict({ ok: false, error: 'boom' }).text, 'engine error: boom');
  assert.equal(verdict({ pending: true }).text, 'deriving…');
  assert.equal(verdict(null).text, 'queued');
});

test('formatRel writes powers of ten with superscripts', () => {
  assert.equal(formatRel(0), '0');
  assert.equal(formatRel(2.2e-16), '2.2 × 10⁻¹⁶');
  assert.equal(formatRel(3e-10), '3.0 × 10⁻¹⁰');
  assert.equal(formatRel(NaN), '—');
});

test('circuit labels, durations, disagreements, sizes', () => {
  assert.equal(circuitLabel({ source: 'stim', d: 5 }), 'Stim’s own · d = 5');
  assert.equal(circuitLabel({ source: 'ours', code: 'xzzx', noise: 'sd6', d: 3 }), 'XZZX · SD6 · d = 3');
  assert.equal(circuitLabel({ source: 'ours', code: 'rotated', noise: 'current', d: 7 }), 'rotated · engine’s model · d = 7');
  assert.equal(microseconds(3.14), '3.1 µs');
  assert.equal(microseconds(42.4), '42 µs');
  assert.equal(microseconds(1520), '1.5 ms');
  assert.equal(disagreementText({ disagreements: 0, non_ties: 0 }), 'none');
  assert.equal(disagreementText({ disagreements: 12, non_ties: 0 }), '12 · all ties');
  assert.equal(disagreementText({ disagreements: 1200, non_ties: 2 }), '1,200 · 2 not ties');
  assert.equal(megabytes(1_400_000), '1.4 MB');
});

test('sweep config: every noise model has distances, an increasing window and a shot count', () => {
  for (const mode of [0, 1, 2, 3]) {
    assert.ok(DISTANCES[mode].length >= 4, `mode ${mode}`);
    const ps = SWEEP_PS[mode];
    assert.ok(ps.length >= 7 && ps.every((p, i) => i === 0 || p > ps[i - 1]), `mode ${mode}`);
    assert.ok(SWEEP_RUNS[mode] >= 500, `mode ${mode}`);
  }
});

// -- Section 11's fits (js/lambda-fit.js) -----------------------------------

/** Records whose fidelity is exactly A (1 − 2ε)^r, failures left fractional. */
function exactRecords(epsByD, { patches = ['a'], bases = ['X', 'Z'], A = 0.98, shots = 50000, key = 'k' } = {}) {
  const rounds = [1, 10, 30, 50, 90, 130, 170, 210, 250];
  const out = [];
  for (const [d, eps] of Object.entries(epsByD)) {
    for (const patch of patches) for (const basis of bases) for (const r of rounds) {
      const F = A * (1 - 2 * eps) ** r;
      out.push({ d: Number(d), patch, basis, rounds: r, shots, results: { [key]: { failures: shots * (1 - F) / 2 } } });
    }
  }
  return out;
}

test('fitEpsilon recovers ε from exact fidelities, whatever the weights', () => {
  const pts = exactRecords({ 5: 0.003 }).filter((r) => r.basis === 'X')
    .map((r) => ({ rounds: r.rounds, failures: r.results.k.failures, shots: r.shots }));
  const fit = fitEpsilon(pts, { minRounds: 10 });
  assert.ok(fit.ok);
  assert.ok(Math.abs(fit.eps - 0.003) < 1e-12, `${fit.eps}`);
  assert.ok(Math.abs(fit.A - 0.98) < 1e-9, `${fit.A}`);
  assert.equal(fit.n, 8);
});

test('fitEpsilon leaves out early rounds and fidelities lost in the noise', () => {
  const pts = [
    { rounds: 1, failures: 0, shots: 1000 },
    { rounds: 10, failures: 100, shots: 1000 },
    { rounds: 20, failures: 180, shots: 1000 },
    { rounds: 400, failures: 495, shots: 1000 }, // F = 0.01, σ ≈ 0.03
  ];
  const fit = fitEpsilon(pts, { minRounds: 10 });
  assert.equal(fit.n, 2);
  assert.equal(fitEpsilon(pts.slice(0, 2), { minRounds: 10 }).ok, false);
  const f = fidelity(0, 1000);
  assert.ok(f.F === 1 && f.sigma > 0);
});

test('epsilonByDistance averages patches and bases; lambdaFit gives Λ exactly', () => {
  const recs = exactRecords({ 3: 0.008, 5: 0.004, 7: 0.002 }, { patches: ['a', 'b'] });
  const byD = epsilonByDistance(recs, 'k', { minRounds: 10 });
  assert.deepEqual([...byD.keys()], [3, 5, 7]);
  assert.equal(byD.get(3).fits.length, 4);
  assert.ok(Math.abs(byD.get(5).eps - 0.004) < 1e-12);
  const fit = lambdaFit(byD);
  assert.ok(Math.abs(fit.lambda - 2) < 1e-9, `${fit.lambda}`);
  assert.ok(fit.pairwise.every((p) => Math.abs(p.lambda - 2) < 1e-9));
  // Two distances: Λ is exactly their ratio.
  const two = lambdaFit(new Map([[3, { eps: 0.03 }], [5, { eps: 0.02 }]]));
  assert.ok(Math.abs(two.lambda - 1.5) < 1e-12);
  assert.deepEqual(decoderKeys(recs), ['k']);
});

test('bootstrap intervals bracket the estimate and repeat with a seed', () => {
  const recs = exactRecords({ 3: 0.008, 5: 0.004, 7: 0.002 })
    .map((r) => ({ ...r, results: { k: { failures: Math.round(r.results.k.failures) } } }));
  const a = bootstrap(recs, 'k', { minRounds: 10 }, 200, seededRandom(7));
  const b = bootstrap(recs, 'k', { minRounds: 10 }, 200, seededRandom(7));
  assert.deepEqual(a.lambda, b.lambda);
  const est = lambdaFit(epsilonByDistance(recs, 'k', { minRounds: 10 })).lambda;
  assert.ok(a.lambda[0] < est && est < a.lambda[1], `${a.lambda} around ${est}`);
  assert.ok(a.lambda[1] - a.lambda[0] < 0.2, `${a.lambda}`);
  const [lo, hi] = a.eps.get(5);
  assert.ok(lo < 0.004 && 0.004 < hi);
});

test('logTicks: whole decades around the data, with 2x and 5x between', () => {
  const t = logTicks(0.0014, 0.008);
  assert.equal(t.lo, 0.001);
  assert.equal(t.hi, 0.01);
  assert.deepEqual(t.ticks.map((x) => x.v), [0.001, 0.002, 0.005, 0.01]);
  assert.deepEqual(t.ticks.map((x) => x.major), [true, false, false, true]);
  const one = logTicks(0.02, 0.02);
  assert.ok(one.hi > one.lo);
});

test('hardware labels and formats', () => {
  assert.equal(decoderLabel('ours/si1000/correlated'), "ours, correlated · Google's SI1000 prior");
  assert.equal(decoderLabel('google/libra_decoder_with_rl_optimized_prior'), 'Google: Libra, RL-optimised prior');
  assert.equal(decoderLabel('google/unknown'), 'Google: unknown');
  assert.equal(pct(0.00143), '0.143%');
  assert.equal(pct(NaN), '—');
  assert.equal(percentRange([0.0014, 0.00146]), '0.140–0.146%');
  assert.equal(ratio(2.1374, [2.11, 2.17]), '2.14 [2.11, 2.17]');
  assert.equal(ratio(2.1374), '2.14');
});

test('the pool splits runs without losing any, and combines progress and results', () => {
  for (const [total, n] of [[40000, 7], [5, 8], [1, 3], [64, 8]]) {
    const parts = splitRuns(total, n);
    assert.equal(parts.reduce((s, x) => s + x, 0), total);
    assert.ok(parts.every((x) => x > 0) && parts.length === Math.min(n, total));
    assert.ok(Math.max(...parts) - Math.min(...parts) <= 1);
  }
  const p = mergeStream([{ done: 100, failures: 3, seconds: 0.5 }, null, { done: 50, failures: 2, seconds: 0.8, decodeErrors: 1 }], 300);
  assert.deepEqual(p, { done: 150, total: 300, failures: 5, decodeErrors: 1, seconds: 0.8 });
  const r = mergeStreamResults([
    { runs: 100, failures: 3, seconds: 0.5, cancelled: false },
    { runs: 100, failures: 5, seconds: 1.0, cancelled: true, decodeErrors: 2 },
  ]);
  assert.deepEqual(r, { runs: 200, failures: 8, rate: 0.04, seconds: 1, decodeErrors: 2, runsPerSecond: 200, cancelled: true });
  assert.ok(poolSize() >= 1 && poolSize() <= 8);
});

console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);
