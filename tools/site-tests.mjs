// Node tests for the pure site modules. Run: node tools/site-tests.mjs
import assert from 'node:assert/strict';
import { poisson, footprintRate, layoutFor, chainsFromCorrection, pickPauli } from '../js/opener-math.js';

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

console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);
