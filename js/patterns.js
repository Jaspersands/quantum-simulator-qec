/**
 * Error patterns used by the interactive figures.
 *
 * These drive the engine through its normal toggle API — they are ways of
 * choosing which qubits to click, not a second simulator.
 */

import { ERROR } from './engine.js';
import { samplePauli, NONE, X, Z } from './channel.js';

/** Data qubits sit on a square grid two units apart on the doubled lattice. */
function qubitAt(session, x, y) {
  return session.dataQubits.find((q) => q.x === x && q.y === y);
}

const pick = (list) => list[Math.floor(Math.random() * list.length)];

/**
 * Lay down a straight run of identical errors.
 *
 * The run has to be straight. Inside a straight chain, every check covering two
 * consecutive errors sees even parity and stays quiet, so exactly two defects
 * appear — one at each end. A chain that turns a corner does not have that
 * property on this lattice: consecutive qubits across a bend do not share a
 * check of the type that detects this error, nothing cancels there, and you get
 * four defects instead of two. Since the whole point of the figure is that the
 * syndrome only sees the endpoints, wandering would undercut the claim it is
 * making.
 *
 * @returns {number[]} the qubits that were flipped
 */
export function growChain(session, { length = 3, type = ERROR.X, t = 0, axis = null } = {}) {
  const AXES = [[2, 0], [0, 2]];
  const [dx, dy] = axis == null ? pick(AXES) : AXES[axis];

  // Start far enough back along the chosen axis that the run has room.
  const starts = session.dataQubits.filter((q) => {
    for (let i = 1; i < length; i++) {
      if (!qubitAt(session, q.x + dx * i, q.y + dy * i)) return false;
    }
    return true;
  });

  // Prefer a run that clears both edges along its axis. An endpoint sitting on
  // the boundary gets absorbed by it and produces no defect, which is correct
  // but makes the "one defect per end" reading harder to see.
  const interior = starts.filter((q) => {
    const before = qubitAt(session, q.x - dx, q.y - dy);
    const after = qubitAt(session, q.x + dx * length, q.y + dy * length);
    return before && after;
  });

  const pool = interior.length ? interior : starts;
  const start = pool.length ? pick(pool) : pick(session.dataQubits);
  const chain = [];
  for (let i = 0; i < length; i++) {
    const q = qubitAt(session, start.x + dx * i, start.y + dy * i);
    if (!q) break;
    chain.push(q.idx);
  }

  for (const qIdx of chain) session.toggleError(qIdx, type, t);
  return chain;
}

/**
 * Sprinkle noise at rate p, using the same channel as the Monte Carlo.
 *
 * Each qubit takes an error with probability p, and that error is X, Y or Z
 * per the bias — it is not an independent X draw plus an independent Z draw,
 * which would deliver an effective rate near 2p and silently make the figure
 * noisier than its own caption claims.
 *
 * @returns {number} how many qubits were touched
 */
export function scatter(session, { p = 0.08, t = 0, bias = 0.5 } = {}) {
  let touched = 0;
  for (const q of session.dataQubits) {
    const pauli = samplePauli(p, bias);
    if (pauli === NONE) continue;
    if (pauli & X) session.toggleError(q.idx, ERROR.X, t);
    if (pauli & Z) session.toggleError(q.idx, ERROR.Z, t);
    touched++;
  }
  return touched;
}

/** Count set bits in a slice of a state array. */
export function countSet(array) {
  let n = 0;
  for (let i = 0; i < array.length; i++) if (array[i]) n++;
  return n;
}
