/**
 * Lattice surgery's geometry, for section 14's drawing: the same layout as
 * src/surgery.rs, kept pure so the site tests can hold it to it.
 *
 * Data qubits sit at odd (x, y), checks at even (x, y), X-type where
 * ((x + y) / 2) is odd. Patch 1's data columns are x = 1 … 2d − 1, the seam
 * is x = 2d + 1, patch 2 is x = 2d + 3 … 4d + 1; rows y = 1 … 2d − 1.
 */

const mod = (a, n) => ((a % n) + n) % n;

/**
 * The checks of the rotated code on a box of data columns x0 … x1 and rows
 * y0 … y1 (odd): [{x, y, xType, support}]. X-type boundaries left and right,
 * Z-type top and bottom, as src/surgery.rs builds them.
 */
export function boxChecks(x0, x1, y0, y1) {
  const inside = (x, y) => x >= x0 && x <= x1 && y >= y0 && y <= y1;
  const checks = [];
  for (let y = y0 - 1; y <= y1 + 1; y += 2) {
    for (let x = x0 - 1; x <= x1 + 1; x += 2) {
      const xType = mod((x + y) / 2, 2) === 1;
      const allowed = xType ? y > y0 && y < y1 : x > x0 && x < x1;
      if (!allowed) continue;
      const support = [[-1, -1], [-1, 1], [1, -1], [1, 1]]
        .map(([dx, dy]) => [x + dx, y + dy])
        .filter(([qx, qy]) => inside(qx, qy));
      if (support.length >= 2) checks.push({ x, y, xType, support });
    }
  }
  return checks;
}

/** The checks of the rotated code on data columns x0 … x1 (odd), rows 1 … 2d − 1. */
export function codeChecks(d, x0, x1) {
  return boxChecks(x0, x1, 1, 2 * d - 1);
}

/** Everything the drawing needs at distance d. */
export function surgeryLayout(d) {
  const p1 = [1, 2 * d - 1], seam = 2 * d + 1, p2 = [2 * d + 3, 4 * d + 1];
  const patches = [...codeChecks(d, ...p1), ...codeChecks(d, ...p2)];
  const merged = codeChecks(d, p1[0], p2[1]);
  const key = (c) => `${c.x},${c.y}`;
  const patchKeys = new Set(patches.map(key));
  const newZ = merged.filter((c) => !c.xType && !patchKeys.has(key(c)));
  const data = [];
  for (let y = 1; y < 2 * d; y += 2) {
    for (let x = 1; x <= 4 * d + 1; x += 2) data.push({ x, y, seam: x === seam });
  }
  return { d, p1, p2, seam, patches, merged, newZ, data, width: 4 * d + 2, height: 2 * d };
}

/**
 * The logical CNOT's three patches (src/surgery.rs, `cnot`): control C at
 * tile (0, 0), ancilla A at (1, 0), target T at (1, 1), tiles 2d + 2 apart.
 * `ca` is C and A merged (Z_C Z_A, by the new Z checks `newCA`), `at` is A and
 * T merged (X_A X_T, by the new X checks `newAT`).
 */
export function cnotLayout(d) {
  const s = 2 * d + 2;
  const tile = (i, j) => ({ x0: 1 + i * s, x1: 1 + i * s + 2 * d - 2, y0: 1 + j * s, y1: 1 + j * s + 2 * d - 2 });
  const union = (a, b) => ({ x0: Math.min(a.x0, b.x0), x1: Math.max(a.x1, b.x1), y0: Math.min(a.y0, b.y0), y1: Math.max(a.y1, b.y1) });
  const code = (b) => boxChecks(b.x0, b.x1, b.y0, b.y1);
  const C = tile(0, 0), A = tile(1, 0), T = tile(1, 1);
  const patches = { C: code(C), A: code(A), T: code(T) };
  const key = (c) => `${c.x},${c.y}`;
  const own = new Set([...patches.C, ...patches.A, ...patches.T].map(key));
  const ca = code(union(C, A));
  const at = code(union(A, T));
  const newCA = ca.filter((c) => !c.xType && !own.has(key(c)));
  const newAT = at.filter((c) => c.xType && !own.has(key(c)));
  const data = [];
  const add = (b, role) => {
    for (let y = b.y0; y <= b.y1; y += 2) for (let x = b.x0; x <= b.x1; x += 2) data.push({ x, y, role });
  };
  add(C, 'C');
  add(A, 'A');
  add(T, 'T');
  add({ x0: C.x1 + 2, x1: C.x1 + 2, y0: C.y0, y1: C.y1 }, 'ca');
  add({ x0: A.x0, x1: A.x1, y0: A.y1 + 2, y1: A.y1 + 2 }, 'at');
  return { d, C, A, T, patches, ca, at, newCA, newAT, data, width: 2 * s - 2, height: 2 * s - 2 };
}

