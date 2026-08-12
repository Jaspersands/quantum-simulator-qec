/**
 * Monte Carlo sampling for the surface code.
 *
 * WHY THIS EXISTS
 * ---------------
 * The engine ships a `wasm_run_benchmark` that does the whole experiment in
 * Rust, and it would be the obvious thing to call. It cannot be used. In the
 * WebAssembly build its random number generator is seeded from a compile-time
 * constant:
 *
 *     #[cfg(not(feature = "python"))]
 *     let mut rng = Xorshift::new(12345);          // src/surface_code.rs
 *
 * The seed is taken fresh inside every shot, so all `num_runs` shots of a
 * benchmark are bit-identical and the reported rate collapses to a step
 * function — flat zero, then a plateau, then flat one. Under the `python`
 * feature the same lines use `rand::random()`, which is why the offline
 * benchmarks were fine and only the browser numbers were wrong.
 *
 * Rebuilding the module is not possible here (the local Rust toolchain is an
 * x86 binary on an ARM machine), so the sampling moves to JavaScript while
 * everything that constitutes the physics stays in Rust: the lattice, the
 * syndrome extraction, all three decoders, and the logical-failure verdict are
 * still the engine's, reached through the per-shot session API.
 *
 * The distributions below are transcriptions of `sample_biased_error`,
 * `sample_biased_error_with_erasure`, `get_drift_p`, and
 * `inject_correlated_noise` in `src/surface_code.rs`.
 */

import { NOISE, ERROR, CORRELATED } from './engine.js';

/** Sampled Pauli, as a bitmask: 1 = X, 2 = Z, 3 = Y. */
const NONE = 0, X = 1, Z = 2, Y = 3;

/**
 * Mirror of `sample_biased_error`.
 * eta is the Z-bias: at eta = 0.5 the channel is depolarizing.
 */
function samplePauli(p, eta) {
  if (Math.random() >= p) return NONE;
  const u = Math.random();
  const pz = eta / (eta + 1);
  const px = 1 / (2 * (eta + 1));
  if (u < pz) return Z;
  if (u < pz + px) return X;
  return Y;
}

/** Mirror of `get_drift_p` — slow 1/f-style modulation across rounds. */
function driftP(p, t, correlated) {
  if (correlated === CORRELATED.DRIFT || correlated === CORRELATED.BOTH) {
    return p * (1 + 0.5 * Math.sin((2 * Math.PI * t) / 5));
  }
  return p;
}

/**
 * A reusable code patch for repeated shots.
 *
 * Session geometry is read once; the per-shot loop only clears and toggles,
 * which keeps the JS/wasm boundary crossings proportional to the number of
 * errors rather than the number of qubits.
 */
class Patch {
  constructor(instance, d, codeType, rounds) {
    this.wasm = instance.exports;
    this.d = d;
    this.rounds = rounds;
    this.ptr = this.wasm.wasm_create_session(d, codeType);
    this.wasm.wasm_set_num_rounds(this.ptr, rounds);
    this.numData = this.wasm.wasm_get_data_qubit_count(this.ptr);
    this.numStab = this.wasm.wasm_get_stabilizer_count(this.ptr);

    // Grid positions, needed only for spatial burst noise.
    this.coords = [];
    for (let i = 0; i < this.numData; i++) {
      this.coords.push([
        (this.wasm.wasm_get_data_qubit_x(this.ptr, i) - 1) / 2,
        (this.wasm.wasm_get_data_qubit_y(this.ptr, i) - 1) / 2,
      ]);
    }
  }

  /** Apply a sampled Pauli to qubit q in round t. */
  apply(pauli, q, t) {
    if (pauli & X) this.wasm.wasm_toggle_error(this.ptr, q, ERROR.X, t);
    if (pauli & Z) this.wasm.wasm_toggle_error(this.ptr, q, ERROR.Z, t);
  }

  /** Mirror of `inject_correlated_noise` — a rare disc of correlated faults. */
  burst(t) {
    if (Math.random() >= 0.02) return;
    const cx = Math.random() * this.d;
    const cy = Math.random() * this.d;
    for (let q = 0; q < this.numData; q++) {
      const [ux, uy] = this.coords[q];
      if (Math.hypot(ux - cx, uy - cy) <= 1.5) {
        this.apply([X, Z, Y][Math.floor(Math.random() * 3)], q, t);
      }
    }
  }

  clear() { this.wasm.wasm_clear_errors(this.ptr); }
  decode(decoder) { return this.wasm.wasm_decode(this.ptr, decoder) === 1; }
  free() { this.wasm.wasm_free_session(this.ptr); this.ptr = 0; }
}

/**
 * One shot: draw a noise realisation, hand it to the engine's decoder, and
 * return the engine's verdict on whether the logical qubit survived.
 */
function shot(patch, config) {
  const { p, bias, erasure, correlated, noiseMode, decoder } = config;
  patch.clear();

  const rounds = noiseMode === NOISE.DATA ? 1 : patch.rounds;
  const burstsOn = correlated === CORRELATED.BURSTS || correlated === CORRELATED.BOTH;

  for (let t = 0; t < rounds; t++) {
    const roundP = driftP(p, t, correlated);
    const pErase = roundP * erasure;
    const pPauli = roundP * (1 - erasure);

    for (let q = 0; q < patch.numData; q++) {
      if (pErase > 0 && Math.random() < pErase) {
        // A located loss: the position is known, the Pauli is uniform.
        patch.wasm.wasm_toggle_erasure(patch.ptr, q, t);
        patch.apply([X, Z, Y][Math.floor(Math.random() * 3)], q, t);
        continue;
      }
      const pauli = samplePauli(pPauli, bias);
      if (pauli !== NONE) patch.apply(pauli, q, t);
    }

    if (burstsOn) patch.burst(t);

    // Faulty readout. The final round is taken as perfect, the usual
    // convention — otherwise the last syndrome could never be trusted.
    if (noiseMode !== NOISE.DATA && t < rounds - 1) {
      for (let s = 0; s < patch.numStab; s++) {
        if (Math.random() < roundP) patch.wasm.wasm_toggle_measurement_error(patch.ptr, s, t);
      }
    }
  }

  return patch.decode(decoder);
}

/** Defaults for a Monte Carlo run. */
export const DEFAULTS = {
  d: 5,
  codeType: 0,
  decoder: 0,
  p: 0.02,
  bias: 0.5,
  rounds: 3,
  runs: 2000,
  noiseMode: NOISE.PHENOM,
  erasure: 0,
  correlated: CORRELATED.NONE,
};

/**
 * Estimate the logical error rate.
 *
 * @param {WebAssembly.Instance} instance
 * @param {object} config see DEFAULTS
 * @returns {{rate:number, runs:number, seconds:number, runsPerSecond:number}}
 */
export function estimateLogicalErrorRate(instance, config) {
  const c = { ...DEFAULTS, ...config };
  const rounds = c.noiseMode === NOISE.DATA ? 1 : Math.max(1, c.rounds);
  const patch = new Patch(instance, c.d, c.codeType, rounds);

  // Built once — allocating this per shot dominates the loop at small d.
  const shotConfig = { ...c, rounds };

  let failures = 0;
  const start = performance.now();
  try {
    for (let i = 0; i < c.runs; i++) {
      if (shot(patch, shotConfig)) failures++;
    }
  } finally {
    patch.free();
  }
  const seconds = (performance.now() - start) / 1000;

  return {
    rate: failures / c.runs,
    runs: c.runs,
    seconds,
    runsPerSecond: seconds > 0 ? Math.round(c.runs / seconds) : 0,
  };
}
