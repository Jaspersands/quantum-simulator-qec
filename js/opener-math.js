/**
 * Pure helpers for the live lattice at the top of the page. No DOM, no engine:
 * everything here takes plain numbers or a Session-shaped object, so it runs
 * under Node for tests.
 */

import { STAB } from './engine.js';

/** Knuth's Poisson sampler; fine for the small means used here. */
export function poisson(lambda, rand = Math.random) {
  if (lambda <= 0) return 0;
  const L = Math.exp(-lambda);
  let k = 0, p = 1;
  do { k++; p *= rand(); } while (p > L);
  return k - 1;
}

/** Gaussian error rate under the pointer, in errors/s/qubit; dist and sigma in cells. */
export function footprintRate(dist, sigma, peak) {
  if (dist > 3 * sigma) return 0;
  return peak * Math.exp(-(dist * dist) / (2 * sigma * sigma));
}

/**
 * Square patch of distance d filling the canvas height less a pad, centred
 * horizontally. `unit` is pixels per doubled-grid step: a qubit at engine
 * (x, y) draws at (originX + x * unit, originY + y * unit).
 */
export function layoutFor(width, height, d, pad) {
  const size = Math.max(0, Math.min(height - 2 * pad, width - 2 * pad));
  return {
    size,
    originX: (width - size) / 2,
    originY: (height - size) / 2,
    unit: size / (2 * d),
  };
}

export function pickPauli(rand, mix) {
  let r = rand();
  for (const [pauli, weight] of mix) { r -= weight; if (r <= 0) return pauli; }
  return mix[mix.length - 1][0];
}

/**
 * Group the qubits a correction flips into chains for drawing, one chain per
 * connected component, ordered end to end.
 *
 * Two flipped qubits are linked when they share a check of the type that sees
 * that Pauli: X flips are seen by Z checks and Z flips by X checks. That is the
 * same adjacency the matching used, so the chains drawn are the chains the
 * decoder chose. A chain that touches the boundary starts there.
 */
export function chainsFromCorrection(session, correctionX, correctionZ) {
  const out = [];
  const maxCoord = 2 * session.d - 1;
  const seesType = { X: STAB.Z, Z: STAB.X };

  for (const [type, mask] of [['X', correctionX], ['Z', correctionZ]]) {
    const active = [];
    for (let i = 0; i < session.numData; i++) if (mask[i]) active.push(i);
    if (!active.length) continue;

    const activeSet = new Set(active);
    const byCheck = new Map();                       // check idx -> active qubits it touches
    for (const q of active) {
      for (const s of session.touchedBy.get(q) ?? []) {
        if (session.stabilizers[s].type !== seesType[type]) continue;
        if (!byCheck.has(s)) byCheck.set(s, []);
        byCheck.get(s).push(q);
      }
    }
    const adj = new Map(active.map((q) => [q, new Set()]));
    for (const qs of byCheck.values()) {
      for (const a of qs) for (const b of qs) if (a !== b) adj.get(a).add(b);
    }

    const onBoundary = (q) => {
      const { x, y } = session.dataQubits[q];
      return type === 'X' ? (x === 1 || x === maxCoord) : (y === 1 || y === maxCoord);
    };

    const seen = new Set();
    for (const start of active) {
      if (seen.has(start)) continue;
      const comp = [];
      const queue = [start];
      seen.add(start);
      while (queue.length) {
        const q = queue.shift();
        comp.push(q);
        for (const n of adj.get(q)) if (!seen.has(n) && activeSet.has(n)) { seen.add(n); queue.push(n); }
      }
      // Walk from an end: a degree-1 qubit on the boundary, else any degree-1, else any boundary, else the first.
      const degree1 = comp.filter((q) => adj.get(q).size <= 1);
      const head = degree1.find(onBoundary) ?? degree1[0] ?? comp.find(onBoundary) ?? comp[0];
      const ordered = [head];
      const used = new Set([head]);
      let cur = head;
      while (ordered.length < comp.length) {
        const next = [...adj.get(cur)].find((n) => !used.has(n)) ?? comp.find((n) => !used.has(n));
        used.add(next); ordered.push(next); cur = next;
      }
      out.push({ type, qubits: ordered });
    }
  }
  return out;
}
