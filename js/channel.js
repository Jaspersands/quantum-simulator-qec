/**
 * The noise channel, transcribed from `sample_biased_error` in
 * `src/surface_code.rs`.
 *
 * The interactive figures scatter errors themselves rather than going through
 * the engine's batch entry points, so they need the same channel the Monte
 * Carlo uses. A figure captioned "noise at p = 8%" has to mean the same thing
 * as the p = 8% column of the results table, or the two halves of the page
 * quietly disagree — and the obvious shortcut, an independent X draw plus an
 * independent Z draw, delivers an effective rate near 2p.
 */

/** Sampled Pauli, as a bitmask: 1 = X, 2 = Z, 3 = Y. */
export const NONE = 0, X = 1, Z = 2, Y = 3;

/**
 * @param {number} p total probability of any error on this qubit
 * @param {number} eta Z-bias; 0.5 is depolarizing, higher favours Z
 */
export function samplePauli(p, eta) {
  if (Math.random() >= p) return NONE;
  const u = Math.random();
  const pz = eta / (eta + 1);
  const px = 1 / (2 * (eta + 1));
  if (u < pz) return Z;
  if (u < pz + px) return X;
  return Y;
}
