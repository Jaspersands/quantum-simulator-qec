/**
 * An independent phenomenological-noise simulator: L x L toric code, pure JS,
 * sharing no code with the Rust engine.
 *
 * Different lattice (periodic, no boundaries), different stabilizers (vertex
 * stars on a square lattice of edges), different noise sampling, different
 * graph, different decoder. If the exponent measured here agrees with the one
 * the engine reports, it is a property of the physics rather than of either
 * implementation.
 *
 * Qubits live on edges: h(x,y) joins (x,y)-(x+1,y), v(x,y) joins (x,y)-(x,y+1),
 * all indices mod L. A vertex fires when an odd number of its four incident
 * edges carry an error.
 */
export function makeToric(L) {
  const H = (x, y) => ((y % L) + L) % L * L + (((x % L) + L) % L);
  const V = (x, y) => L * L + ((y % L) + L) % L * L + (((x % L) + L) % L);
  const nQ = 2 * L * L;
  const nV = L * L;

  // The four edges meeting each vertex.
  const incident = [];
  for (let y = 0; y < L; y++) {
    for (let x = 0; x < L; x++) {
      incident.push([H(x, y), H(x - 1, y), V(x, y), V(x, y - 1)]);
    }
  }
  const vIndex = (x, y) => ((y % L) + L) % L * L + (((x % L) + L) % L);

  // Cuts whose intersection parity with a cycle gives its winding number.
  const cutX = []; // horizontal edges in one column
  const cutY = []; // vertical edges in one row
  for (let y = 0; y < L; y++) cutX.push(H(0, y));
  for (let x = 0; x < L; x++) cutY.push(V(x, 0));

  return { L, nQ, nV, H, V, incident, vIndex, cutX, cutY };
}

/** Shortest separation on a ring. */
const ring = (a, b, L) => { const d = Math.abs(a - b); return Math.min(d, L - d); };

/**
 * Approximate minimum-weight matching: globally nearest pair first, then 2-opt.
 *
 * Pairing in index order instead — taking each defect in turn and giving it its
 * nearest remaining partner — is much worse, and worse in a way that grows with
 * the number of defects, so it degrades as the patch grows. That produced a
 * logical error rate that fell from L = 3 to 7 and then jumped at L = 9, which
 * reads as a threshold far below the truth rather than as a decoder failing.
 */
function match(defects, dist) {
  const n = defects.length;
  const used = new Array(n).fill(false);
  const pairs = [];
  const candidates = [];
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) candidates.push([dist(i, j), i, j]);
  }
  candidates.sort((a, b) => a[0] - b[0]);
  for (const [, i, j] of candidates) {
    if (used[i] || used[j]) continue;
    used[i] = true; used[j] = true;
    pairs.push([i, j]);
  }
  // 2-opt: swap partners between two pairs whenever it shortens the total.
  for (let pass = 0; pass < 30; pass++) {
    let improved = false;
    for (let a = 0; a < pairs.length; a++) {
      for (let b = a + 1; b < pairs.length; b++) {
        const [p, q] = pairs[a]; const [r, s] = pairs[b];
        const now = dist(p, q) + dist(r, s);
        const alt1 = dist(p, r) + dist(q, s);
        const alt2 = dist(p, s) + dist(q, r);
        if (alt1 < now && alt1 <= alt2) { pairs[a] = [p, r]; pairs[b] = [q, s]; improved = true; }
        else if (alt2 < now) { pairs[a] = [p, s]; pairs[b] = [q, r]; improved = true; }
      }
    }
    if (!improved) break;
  }
  return pairs;
}

/** One shot. Returns 1 if a logical error survived. */
export function shot(code, T, p, rand) {
  const { L, nQ, nV, H, V, incident, cutX, cutY } = code;
  const error = new Uint8Array(nQ);
  const syn = [];
  let prev = new Uint8Array(nV);

  for (let t = 0; t < T; t++) {
    if (t < T - 1) {                        // the last round is noiseless
      for (let q = 0; q < nQ; q++) if (rand() < p) error[q] ^= 1;
    }
    const s = new Uint8Array(nV);
    for (let v = 0; v < nV; v++) {
      let par = 0;
      for (const q of incident[v]) par ^= error[q];
      s[v] = par;
    }
    if (t < T - 1) {
      for (let v = 0; v < nV; v++) if (rand() < p) s[v] ^= 1;
    }
    const ev = new Uint8Array(nV);
    for (let v = 0; v < nV; v++) ev[v] = s[v] ^ prev[v];
    syn.push(ev);
    prev = s;
  }

  const defects = [];
  for (let t = 0; t < T; t++) {
    for (let v = 0; v < nV; v++) if (syn[t][v]) defects.push({ v, t });
  }
  if (defects.length % 2 === 1) defects.pop(); // cannot happen on a torus; guard

  const dist = (i, j) => {
    const a = defects[i]; const b = defects[j];
    const ax = a.v % L; const ay = (a.v - ax) / L;
    const bx = b.v % L; const by = (b.v - bx) / L;
    return ring(ax, bx, L) + ring(ay, by, L) + Math.abs(a.t - b.t);
  };

  const correction = new Uint8Array(nQ);
  for (const [i, j] of match(defects, dist)) {
    const a = defects[i]; const b = defects[j];
    let ax = a.v % L; let ay = (a.v - ax) / L;
    const bx = b.v % L; const by = (b.v - bx) / L;
    // Walk x then y, each the short way round, flipping the edges crossed.
    let stepX = ((bx - ax + L) % L <= (ax - bx + L) % L) ? 1 : -1;
    while (ax !== bx) { correction[H(stepX > 0 ? ax : ax - 1, ay)] ^= 1; ax = (ax + stepX + L) % L; }
    let stepY = ((by - ay + L) % L <= (ay - by + L) % L) ? 1 : -1;
    while (ay !== by) { correction[V(ax, stepY > 0 ? ay : ay - 1)] ^= 1; ay = (ay + stepY + L) % L; }
  }

  let wx = 0; let wy = 0;
  for (const q of cutX) wx ^= error[q] ^ correction[q];
  for (const q of cutY) wy ^= error[q] ^ correction[q];
  return (wx | wy) ? 1 : 0;
}
