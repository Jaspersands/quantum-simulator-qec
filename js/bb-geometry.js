/**
 * Bivariate bicycle codes' geometry, for section 13's drawing: which data
 * qubits each check touches on the ℓ × m torus. The same construction as
 * src/bb.rs; kept pure so the site tests can hold it to it.
 *
 * Cells (u, v) of the torus are numbered u·m + v. Each cell holds four
 * qubits, drawn at grid position (column, row): its X check at (2u, 2v), its
 * left data qubit at (2u + 1, 2v), its right data qubit at (2u, 2v + 1) and
 * its Z check at (2u + 1, 2v + 1).
 */

/** A = x³ + y + y², B = y³ + x + x², monomials as [i, j] for x^i y^j. */
const POLYS = { a: [[3, 0], [0, 1], [0, 2]], b: [[0, 3], [1, 0], [2, 0]] };
export const GROSS = { name: 'gross', l: 12, m: 6, ...POLYS };
export const BB72 = { name: '72', l: 6, m: 6, ...POLYS };

/** The column where row `r` of the monomial's permutation matrix is set. */
export function shift(code, [i, j], r) {
  const u = Math.floor(r / code.m), v = r % code.m;
  return ((u + i) % code.l) * code.m + ((v + j) % code.m);
}

/** The row where column `c` of the monomial's permutation matrix is set. */
export function unshift(code, [i, j], c) {
  const u = Math.floor(c / code.m), v = c % code.m;
  return ((u - i + code.l) % code.l) * code.m + ((v - j + code.m) % code.m);
}

/**
 * Check `c`'s six data neighbours as {side, cell}: for an X check, A₁ A₂ A₃ on
 * the left then B₁ B₂ B₃ on the right; for a Z check, B₁ᵀ B₂ᵀ B₃ᵀ on the left
 * then A₁ᵀ A₂ᵀ A₃ᵀ on the right.
 */
export function neighbours(code, c, type) {
  if (type === 'X') {
    return [...code.a.map((mono) => ({ side: 'L', cell: shift(code, mono, c) })),
      ...code.b.map((mono) => ({ side: 'R', cell: shift(code, mono, c) }))];
  }
  return [...code.b.map((mono) => ({ side: 'L', cell: unshift(code, mono, c) })),
    ...code.a.map((mono) => ({ side: 'R', cell: unshift(code, mono, c) }))];
}

/** Grid position of a qubit: kind 'X', 'Z', 'L' or 'R', in cell `cell`. */
export function position(code, kind, cell) {
  const u = Math.floor(cell / code.m), v = cell % code.m;
  const [du, dv] = { X: [0, 0], L: [1, 0], R: [0, 1], Z: [1, 1] }[kind];
  return { col: 2 * u + du, row: 2 * v + dv };
}

/** The shortest signed displacement from a to b around a circle of `size`. */
export function torusDelta(a, b, size) {
  let d = (b - a) % size;
  if (d < 0) d += size;
  return d > size / 2 ? d - size : d;
}

/** Data qubit index (left 0..ℓm−1, then right), as src/bb.rs numbers them. */
export function dataIndex(code, { side, cell }) {
  return side === 'L' ? cell : code.l * code.m + cell;
}
