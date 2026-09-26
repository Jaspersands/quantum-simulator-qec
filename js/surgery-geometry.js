/**
 * Lattice surgery's geometry, for section 14's drawing: the same layout as
 * src/surgery.rs, kept pure so the site tests can hold it to it.
 *
 * Data qubits sit at odd (x, y), checks at even (x, y), X-type where
 * ((x + y) / 2) is odd. Patch 1's data columns are x = 1 … 2d − 1, the seam
 * is x = 2d + 1, patch 2 is x = 2d + 3 … 4d + 1; rows y = 1 … 2d − 1.
 */

const mod = (a, n) => ((a % n) + n) % n;

/** The checks of the rotated code on data columns x0 … x1 (odd): [{x, y, xType, support}]. */
export function codeChecks(d, x0, x1) {
  const inside = (x, y) => x >= x0 && x <= x1 && y >= 1 && y < 2 * d;
  const checks = [];
  for (let y = 0; y <= 2 * d; y += 2) {
    for (let x = x0 - 1; x <= x1 + 1; x += 2) {
      const xType = mod((x + y) / 2, 2) === 1;
      const allowed = xType ? y >= 2 && y <= 2 * d - 2 : x > x0 && x < x1;
      if (!allowed) continue;
      const support = [[-1, -1], [-1, 1], [1, -1], [1, 1]]
        .map(([dx, dy]) => [x + dx, y + dy])
        .filter(([qx, qy]) => inside(qx, qy));
      if (support.length >= 2) checks.push({ x, y, xType, support });
    }
  }
  return checks;
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
