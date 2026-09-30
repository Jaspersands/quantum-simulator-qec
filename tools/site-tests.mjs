// Node tests for the pure site modules. Run: node tools/site-tests.mjs
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { poisson, footprintRate, layoutFor, chainsFromCorrection, pickPauli } from '../js/opener-math.js';
import { planChunks } from '../js/stream.js';
import { DISTANCES, SWEEP_PS, SWEEP_RUNS } from '../js/sweep-config.js';
import { verdict, formatRel, circuitLabel, microseconds, disagreementText, megabytes } from '../js/xcheck-format.js';
import { logTicks } from '../js/plot.js';
import { cores, keepUpText, microseconds as rtUs, windowLabel } from '../js/realtime-format.js';
import { splitRuns, mergeStream, mergeStreamResults } from '../js/pool-merge.js';
import { poolSize } from '../js/pool.js';
import { decoderLabel, percent as pct, percentRange, ratio } from '../js/hardware-format.js';
import { GROSS, BB72, neighbours as bbNeighbours, dataIndex, position as bbPosition, torusDelta } from '../js/bb-geometry.js';
import { surgeryLayout, cnotLayout } from '../js/surgery-geometry.js';
import { epsilonAt, failureAt, distanceFor, physicalQubits, runtimeSeconds, decodingCores, estimate, bigNumber, duration, coresFrom, epsilonUniform, BLOCKS, tileQubits, factoryFor, latencyAt, estimateFull } from '../js/estimator.js';
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
  assert.equal(decoderLabel('ours/pij/belief'), 'ours, belief-matching · the data-fitted pij models');
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

test('real-time labels: cores to keep up, latencies, window names', () => {
  assert.equal(cores({ 1: { keeps_up: false }, 2: { keeps_up: false }, 4: { keeps_up: true }, 8: { keeps_up: true } }), 4);
  assert.equal(cores({ 1: { keeps_up: false } }), null);
  assert.equal(keepUpText(1, 'sliding'), 'Keeps up on one core');
  assert.equal(keepUpText(4, 'parallel'), 'Keeps up on 4 cores');
  assert.equal(keepUpText(null, 'sliding'), 'Falls behind (one core by design)');
  assert.equal(rtUs(63.2), '63 µs');
  assert.equal(rtUs(5.71), '5.7 µs');
  assert.equal(rtUs(0.842), '0.84 µs');
  assert.equal(windowLabel('window/parallel/Bd/correlated'), 'parallel windows, correlated');
  assert.equal(windowLabel('window/sliding/Bhalf/plain'), 'sliding windows, buffer d/2, plain');
  assert.equal(windowLabel('global/plain'), 'global, plain');
});

test('bivariate bicycle geometry: the gross code as src/bb.rs builds it, commuting checks, weight six', () => {
  // X check 0 of the gross code: A = x³ + y + y² on the left, B = y³ + x + x² on the right.
  assert.deepEqual(bbNeighbours(GROSS, 0, 'X').map((q) => dataIndex(GROSS, q)), [18, 1, 2, 72 + 3, 72 + 6, 72 + 12]);
  for (const code of [GROSS, BB72]) {
    const cells = code.l * code.m;
    const rows = (type) => Array.from({ length: cells }, (_, c) => bbNeighbours(code, c, type).map((q) => dataIndex(code, q)));
    const hx = rows('X'), hz = rows('Z');
    for (const r of [...hx, ...hz]) assert.equal(new Set(r).size, 6);
    // H_X H_Zᵀ = 0: every X check and Z check overlap on an even number of qubits.
    for (const x of hx) for (const z of hz) assert.equal(x.filter((q) => z.includes(q)).length % 2, 0);
    // Every data qubit sits in three X checks and three Z checks.
    const count = new Array(2 * cells).fill(0);
    for (const r of hx) for (const q of r) count[q]++;
    assert.ok(count.every((n) => n === 3));
  }
  assert.deepEqual(bbPosition(GROSS, 'Z', 7), { col: 3, row: 3 });
  assert.equal(torusDelta(1, 23, 24), -2);
  assert.equal(torusDelta(23, 1, 24), 2);
  assert.equal(torusDelta(0, 12, 24), 12);
});

test('lattice surgery geometry: the counts src/surgery.rs asserts, and Z1Z2 as the new seam checks', () => {
  for (const d of [3, 5, 7]) {
    const g = surgeryLayout(d);
    assert.equal(g.patches.length, 2 * (d * d - 1));
    assert.equal(g.merged.length, d * (2 * d + 1) - 1);
    // The new Z checks sit in the two columns beside the seam; their supports
    // cover each seam qubit twice and patch 1's last and patch 2's first
    // columns once: their product is Z on those two columns, Z1 Z2.
    const count = new Map();
    for (const c of g.newZ) for (const [x, y] of c.support) count.set(`${x},${y}`, (count.get(`${x},${y}`) ?? 0) + 1);
    for (const [k, n] of count) {
      const x = Number(k.split(',')[0]);
      assert.equal(n % 2, x === g.seam ? 0 : 1, `qubit ${k}`);
      assert.ok([g.p1[1], g.seam, g.p2[0]].includes(x));
    }
    assert.equal([...count.values()].filter((n) => n % 2 === 1).length, 2 * d);
  }
});

test('the CNOT\'s geometry: its merges have the right checks, and X_A X_T is the new X checks on A\'s last row and T\'s first', () => {
  for (const d of [3, 5, 7]) {
    const g = cnotLayout(d);
    for (const name of ['C', 'A', 'T']) assert.equal(g.patches[name].length, d * d - 1, name);
    assert.equal(g.ca.length, d * (2 * d + 1) - 1);
    assert.equal(g.at.length, d * (2 * d + 1) - 1);
    // Parities of the new X checks' supports: each seam qubit twice, A's last
    // row and T's first row once. This is why the CNOT's X frame takes X_C X_A
    // along the last row (src/surgery.rs, cnot).
    const count = new Map();
    for (const c of g.newAT) for (const [x, y] of c.support) count.set(`${x},${y}`, (count.get(`${x},${y}`) ?? 0) + 1);
    const odd = [...count].filter(([, n]) => n % 2 === 1).map(([k]) => Number(k.split(',')[1]));
    assert.equal(odd.length, 2 * d);
    assert.deepEqual([...new Set(odd)].sort((a, b) => a - b), [g.A.y1, g.T.y0]);
    // And Z_C Z_A is the new Z checks on C's last column and A's first.
    const zc = new Map();
    for (const c of g.newCA) for (const [x, y] of c.support) zc.set(`${x},${y}`, (zc.get(`${x},${y}`) ?? 0) + 1);
    const oddX = [...zc].filter(([, n]) => n % 2 === 1).map(([k]) => Number(k.split(',')[0]));
    assert.deepEqual([...new Set(oddX)].sort((a, b) => a - b), [g.C.x1, g.A.x0]);
  }
});

test('the gauging data Figure 16 draws agree with the torus: each edge is a Z check touching exactly its ends, or an added pair', () => {
  const doc = JSON.parse(readFileSync(new URL('../data/gross/gauging.json', import.meta.url), 'utf8'));
  const h = GROSS.l * GROSS.m;
  for (const [name, systems] of Object.entries(doc.operators)) {
    for (const [construction, op] of Object.entries(systems)) {
      const label = `${name}, ${construction}`;
      const support = new Set(op.support);
      const touching = [];
      for (let c = 0; c < h; c++) {
        const on = bbNeighbours(GROSS, c, 'Z').map((nb) => dataIndex(GROSS, nb)).filter((d) => support.has(d));
        if (on.length) touching.push([c, on.sort((a, b) => a - b)]);
      }
      assert.deepEqual(touching.map(([c]) => c), op.edge_checks, `${label}: the check edges are the Z checks touching the operator`);
      touching.forEach(([, on], i) => {
        const ends = op.incidence[i].map((v) => op.support[v]).sort((a, b) => a - b);
        assert.deepEqual(ends, on, `${label}: edge ${i}'s ends`);
      });
      // Added edges follow the checks' edges, each a pair of the operator's qubits.
      assert.equal(op.incidence.length, op.edge_checks.length + op.extra_edges.length, label);
      op.extra_edges.forEach((pair, k) => assert.deepEqual(op.incidence[op.edge_checks.length + k], pair, `${label}: added edge ${k}`));
      assert.equal(op.ancillas, op.edges + op.gauss + op.flux, label);
      if (construction === 'expanded') assert.ok(op.worst_cut[0] >= op.worst_cut[1], `${label}: Cheeger constant at least 1`);
    }
  }
});

const SOURCES = JSON.parse(readFileSync(new URL('../data/estimate/sources.json', import.meta.url), 'utf8'));

test('estimator: Litinski\'s data blocks reproduce his worked examples', () => {
  // A Game of Surface Codes, Sec. 2 and 4: 100 qubits in 153, 204 and 231 tiles; 226 tiles at d = 13 are 76,400 qubits.
  assert.equal(BLOCKS.compact.tiles(100), 153);
  assert.equal(BLOCKS.intermediate.tiles(100), 204);
  // His formula gives 229.3 for the fast block; his drawn layout (Fig. 23a) has 231.
  assert.ok(Math.abs(BLOCKS.fast.tiles(100) - 231) <= 2);
  assert.ok(Math.abs((204 + 22) * tileQubits(13) - 76400) / 76400 < 0.001);
  assert.deepEqual([BLOCKS.compact.steps, BLOCKS.intermediate.steps, BLOCKS.fast.steps], [9, 5, 1]);
});

test('estimator: factories, cultivation and distillation, as the sources give them', () => {
  const c = factoryFor('cultivation', 1e-3, 25, 1e-10, SOURCES);
  assert.equal(c.qubits, 12 * 2 * 26 ** 2);
  assert.equal(c.cyclesPerState, 150);
  assert.ok(Math.abs(c.errorPerToffoli - 2.8e-13) < 1e-20);
  assert.equal(factoryFor('cultivation', 2e-3, 25, 1e-10, SOURCES), null, 'cultivation is tabulated at p <= 0.1% only');
  // Distillation at p = 0.1%: the cheapest row per Toffoli that meets the target.
  const dist = factoryFor('distillation', 1e-3, 25, 1e-10, SOURCES);
  const rows = SOURCES.litinski_factories.filter((r) => r.p_phys === 1e-3);
  const cost = (r) => r.qubitcycles * (r.ccz ? 1 : 4);
  const best = rows.filter((r) => (r.ccz ? 1 : 4) * r.p_out <= 1e-10).sort((a, b) => cost(a) - cost(b))[0];
  assert.equal(dist.name, best.name);
  assert.ok(Math.abs(dist.cyclesPerState - best.qubitcycles / best.qubits) < 1e-9);
  assert.equal(factoryFor('distillation', 2e-3, 25, 1e-10, SOURCES), null, 'no table above p = 0.1%');
  // At p = 0.05% the 0.1% rows are used (the tabulated noise at least p).
  assert.equal(factoryFor('distillation', 5e-4, 25, 1e-10, SOURCES).name, dist.name);
});

test('estimator: decoder latency is the measured p99, extrapolated as a power law', () => {
  const measured = { 3: 10, 5: 47, 7: 139 };
  assert.equal(latencyAt(5, measured), 47e-6);
  assert.equal(latencyAt(7, measured), 139e-6);
  const slope = Math.log(139 / 47) / Math.log(7 / 5);
  assert.ok(Math.abs(latencyAt(25, measured) - 139e-6 * (25 / 7) ** slope) < 1e-12);
  assert.ok(Math.abs(latencyAt(25, measured, 10) - 10e-6) < 1e-15, 'an override in microseconds');
});

test('estimator: the full model meets its budget at the smallest distance, and moves the right way', () => {
  const fit = { A: 0.03, pth: 0.0054 };
  const base = { noise: { kind: 'uniform', p: 1e-3, fit }, budget: 0.01, block: 'fast', factory: 'cultivation', storage: 'surface',
    cycleSeconds: 1e-6, controlSeconds: 10e-6, latencyOverrideUs: 0, parallel: 1 };
  const alg = { qubits: 1000, toffolis: 1e9 };
  const r = estimateFull(alg, base, SOURCES);
  assert.ok(r.failure.total <= 0.01);
  const smaller = estimateFull(alg, { ...base, budget: 1e9 }, SOURCES);
  assert.ok(smaller.d <= r.d);
  // Run time is Toffolis times the larger of the Clifford step and the reaction.
  const clifford = BLOCKS.fast.steps * r.d * 1e-6;
  assert.ok(Math.abs(r.seconds - 1e9 * Math.max(clifford, 10e-6)) / r.seconds < 1e-9);
  assert.equal(r.bound, clifford >= 10e-6 ? 'clifford' : 'reaction');
  // Qubits add up, and factories keep up with consumption.
  assert.equal(r.qubits.total, r.qubits.block + r.qubits.factories + r.qubits.storage);
  assert.ok(r.factories * (1 / (r.factory.cyclesPerState * 1e-6)) >= 1 / (r.seconds / 1e9) - 1e-9);
  // Yoked storage puts the cold qubits at 430 each and shrinks the block. (Its error per round is fixed at
  // Gidney's 1e-15, so a long enough run cannot meet the budget: 900 cold qubits for 4e10 cycles spend 3.6%.)
  assert.ok(estimateFull({ qubits: 1000, hot: 100, toffolis: 1e9 }, { ...base, storage: 'yoked' }, SOURCES).error);
  const yoked = estimateFull({ qubits: 1000, hot: 100, toffolis: 1e8 }, { ...base, storage: 'yoked' }, SOURCES);
  assert.equal(yoked.qubits.storage, 900 * 430);
  assert.ok(yoked.qubits.block < r.qubits.block);
  // Parallel Toffolis divide the run time.
  const par = estimateFull(alg, { ...base, parallel: 4 }, SOURCES);
  assert.ok(Math.abs(par.seconds * 4 - estimateFull(alg, { ...base, parallel: 1, budget: 0.01 }, SOURCES).seconds) / r.seconds < 0.5);
});

test('estimator: the model reproduces what it is given, and moves the right way', () => {
  const model = { eps0: 0.00234, d0: 7, lambda: 1.95 };
  // At the measured distance, the measured ε exactly; two distances on, Λ times smaller.
  assert.equal(epsilonAt(7, model), 0.00234);
  assert.ok(Math.abs(epsilonAt(9, model) - 0.00234 / 1.95) < 1e-15);
  const algo = { qubits: 100, operations: 1e6 };
  const d = distanceFor(algo, model, 0.01);
  assert.ok(d % 2 === 1 && d >= 3);
  assert.ok(failureAt(d, algo, model) <= 0.01 && failureAt(d - 2, algo, model) > 0.01, 'the smallest distance that meets the budget');
  // Bigger algorithms need larger distances, and more qubits.
  const bigger = distanceFor({ qubits: 100, operations: 1e9 }, model, 0.01);
  assert.ok(bigger > d);
  assert.ok(physicalQubits(bigger, algo) > physicalQubits(d, algo));
  assert.equal(physicalQubits(3, { qubits: 1 }, 1), 17);
  assert.ok(Math.abs(runtimeSeconds(5, { operations: 2 }, 1e-6) - 1e-5) < 1e-18);
  // No suppression, no distance.
  assert.equal(distanceFor(algo, { ...model, lambda: 1 }, 0.01), null);
  // Cores: measured distances read back; beyond them, a power law through the last two.
  const cores = { 3: 2, 5: 4, 7: 8 };
  assert.equal(decodingCores(5, { qubits: 1 }, cores), 4);
  assert.equal(decodingCores(4, { qubits: 1 }, cores), 4);
  // Through (5, 4) and (7, 8) the law is 8 · (d / 7)^(ln 2 / ln 1.4): 13.4 at d = 9, so 14 cores.
  assert.equal(decodingCores(9, { qubits: 1 }, cores), Math.ceil(8 * (9 / 7) ** (Math.log(2) / Math.log(1.4))));
  assert.equal(decodingCores(9, { qubits: 1 }, cores), 14);
  assert.equal(estimate(algo, model, { budget: 0.01 }).d, d);
  assert.equal(bigNumber(3.2e7), '32 million');
  assert.equal(bigNumber(12345), '12,345');
  assert.equal(bigNumber(6.7e8), '670 million');
  assert.equal(bigNumber(4.1e12), '4,100 billion');
  assert.equal(duration(90), '90 s');
  assert.equal(duration(7200 * 3), '6.0 h');
  assert.equal(duration(1.5 * 3.156e7), '1.5 years');
  assert.equal(duration(200 * 86400), '200 days');
  // Fewer than two measured distances: no law, no core count.
  assert.equal(decodingCores(9, { qubits: 1 }, { 7: 8 }), null);
  // Streams that never keep up are left out, not written as null.
  const byWorkers = (k) => ({ 1: { keeps_up: k === 1 }, 2: { keeps_up: k <= 2 }, 4: { keeps_up: k <= 4 } });
  assert.deepEqual(coresFrom({ streams: [
    { d: 3, mode: 'parallel', matcher: 'correlated', by_workers: byWorkers(2) },
    { d: 5, mode: 'parallel', matcher: 'correlated', by_workers: byWorkers(99) },
    { d: 5, mode: 'sliding', matcher: 'correlated', by_workers: byWorkers(1) },
  ] }), { 3: 2 });
});

console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);
