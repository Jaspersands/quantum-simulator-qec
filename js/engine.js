/**
 * Typed wrapper around stabilizer_qec.wasm.
 *
 * This is the only module that touches raw pointers or wasm linear memory.
 * Everything else works with plain JS values.
 *
 * The module is a bare C-ABI build with no imports, so it instantiates with an
 * empty import object and can be instantiated again inside a worker.
 */

export const WASM_URL = new URL('../stabilizer_qec.wasm', import.meta.url);

/* -- Enumerations, mirroring src/lib.rs -------------------------------- */

export const CODE = { ROTATED: 0, XZZX: 1 };

export const DECODER = { UNION_FIND: 0, GREEDY: 1, MWPM: 2 };

export const DECODER_NAME = {
  0: 'Union-Find',
  1: 'Greedy',
  2: 'Exact MWPM',
};

/**
 * CIRCUIT is listed because the engine accepts it, not because the page uses
 * it: circuit-level noise runs entirely inside the Rust batch entry points,
 * which are unusable here (see the note at the foot of this file), and it
 * cannot be driven through the per-shot session API.
 */
export const NOISE = { DATA: 0, PHENOM: 1, CIRCUIT: 2 };

export const NOISE_NAME = {
  0: 'Data noise only',
  1: 'Phenomenological',
  2: 'Circuit-level',
};

/** wasm_toggle_error error_type argument. */
export const ERROR = { X: 0, Z: 1 };

/** wasm_get_stabilizer_type return value. */
export const STAB = { Z: 0, X: 1, XZZX: 2 };

export const CORRELATED = { NONE: 0, BURSTS: 1, DRIFT: 2, BOTH: 3 };

/* -- Instantiation ------------------------------------------------------ */

/**
 * Fetch and instantiate the engine.
 * @returns {Promise<WebAssembly.Instance>}
 */
export async function instantiate(url = WASM_URL) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`could not fetch engine (${response.status} ${response.statusText})`);
  }
  const bytes = await response.arrayBuffer();
  const { instance } = await WebAssembly.instantiate(bytes, {});
  return instance;
}

/* -- Session ------------------------------------------------------------ */

/**
 * One code patch: a lattice, its injected errors, its syndrome, and the last
 * correction a decoder proposed for it.
 *
 * Geometry is read once at construction. Error and syndrome arrays live in wasm
 * memory and are copied out on demand — wasm memory can grow and detach any
 * view we hold, so views are always built fresh at the moment of reading.
 */
export class Session {
  /**
   * @param {WebAssembly.Instance} instance
   * @param {{d: number, codeType: number, rounds: number}} config
   */
  constructor(instance, { d, codeType = CODE.ROTATED, rounds = 1 }) {
    this.wasm = instance.exports;
    this.d = d;
    this.codeType = codeType;
    this.rounds = rounds;

    this.ptr = this.wasm.wasm_create_session(d, codeType);
    if (!this.ptr) throw new Error('engine refused to allocate a session');
    this.wasm.wasm_set_num_rounds(this.ptr, rounds);

    this.dataQubits = this.#readDataQubits();
    this.stabilizers = this.#readStabilizers();

    /** Support map: stabilizer index -> data qubit indices it measures. */
    this.support = this.#buildSupport();
    /** Reverse map: data qubit index -> stabilizer indices touching it. */
    this.touchedBy = this.#buildTouchedBy();

    this.correction = null;
  }

  get numData() { return this.dataQubits.length; }
  get numStab() { return this.stabilizers.length; }

  #readDataQubits() {
    const n = this.wasm.wasm_get_data_qubit_count(this.ptr);
    const out = [];
    for (let i = 0; i < n; i++) {
      out.push({
        idx: i,
        x: this.wasm.wasm_get_data_qubit_x(this.ptr, i),
        y: this.wasm.wasm_get_data_qubit_y(this.ptr, i),
      });
    }
    return out;
  }

  #readStabilizers() {
    const n = this.wasm.wasm_get_stabilizer_count(this.ptr);
    const out = [];
    for (let i = 0; i < n; i++) {
      out.push({
        idx: i,
        x: this.wasm.wasm_get_stabilizer_x(this.ptr, i),
        y: this.wasm.wasm_get_stabilizer_y(this.ptr, i),
        type: this.wasm.wasm_get_stabilizer_type(this.ptr, i),
      });
    }
    return out;
  }

  /**
   * A stabilizer sits at the centre of a plaquette; the data qubits it measures
   * are its four (or two, at a boundary) diagonal neighbours on the doubled
   * coordinate grid.
   */
  #buildSupport() {
    const map = new Map();
    for (const s of this.stabilizers) {
      const qubits = [];
      for (const q of this.dataQubits) {
        if (Math.abs(s.x - q.x) === 1 && Math.abs(s.y - q.y) === 1) qubits.push(q.idx);
      }
      map.set(s.idx, qubits);
    }
    return map;
  }

  #buildTouchedBy() {
    const map = new Map();
    for (const q of this.dataQubits) map.set(q.idx, []);
    for (const [sIdx, qubits] of this.support) {
      for (const qIdx of qubits) map.get(qIdx).push(sIdx);
    }
    return map;
  }

  /** Copy a u8 array out of wasm memory. */
  #copyBytes(ptr, length) {
    return Uint8Array.prototype.slice.call(
      new Uint8Array(this.wasm.memory.buffer, ptr, length),
    );
  }

  /**
   * Detection events: where a check *changed* since the previous round.
   *
   * This is what the decoder matches, and it is not the same thing as the raw
   * syndrome. `wasm_decode` computes `outcome ^ previous_outcome` before it
   * builds its graph, and the distinction is the entire subject of section 6:
   * a data error makes a check read differently from that round onward, so it
   * produces one detection event; a lying readout makes a check disagree with
   * itself for exactly one round, so it produces two, stacked in time.
   *
   * With a single round the previous outcome is zero by definition and this
   * reduces to the raw syndrome, which is why the earlier figures can use the
   * same quantity without qualification.
   */
  #detectionEvents(syndrome) {
    const out = new Uint8Array(syndrome.length);
    for (let t = 0; t < this.rounds; t++) {
      for (let i = 0; i < this.numStab; i++) {
        const here = syndrome[i + t * this.numStab];
        const before = t === 0 ? 0 : syndrome[i + (t - 1) * this.numStab];
        out[i + t * this.numStab] = here ^ before;
      }
    }
    return out;
  }

  /**
   * Current state of the patch.
   * @returns {{errorX: Uint8Array, errorZ: Uint8Array, erased: Uint8Array,
   *            syndrome: Uint8Array, defects: Uint8Array}} each indexed `i + t * n`
   */
  read() {
    const dataLen = this.numData * this.rounds;
    const stabLen = this.numStab * this.rounds;
    const syndrome = this.#copyBytes(this.wasm.wasm_get_syndrome(this.ptr), stabLen);
    return {
      errorX: this.#copyBytes(this.wasm.wasm_get_physical_x_ptr(this.ptr), dataLen),
      errorZ: this.#copyBytes(this.wasm.wasm_get_physical_z_ptr(this.ptr), dataLen),
      erased: this.#copyBytes(this.wasm.wasm_get_physical_erased_ptr(this.ptr), dataLen),
      syndrome,
      defects: this.#detectionEvents(syndrome),
    };
  }

  /** Toggle an X or Z error on a data qubit at round t. */
  toggleError(qubit, type, t = 0) {
    this.wasm.wasm_toggle_error(this.ptr, qubit, type, t);
    this.correction = null;
  }

  /** Toggle a qubit erasure (a located loss) at round t. */
  toggleErasure(qubit, t = 0) {
    this.wasm.wasm_toggle_erasure(this.ptr, qubit, t);
    this.correction = null;
  }

  /** Toggle a lying measurement on a stabilizer at round t. */
  toggleMeasurementError(stabilizer, t = 0) {
    this.wasm.wasm_toggle_measurement_error(this.ptr, stabilizer, t);
    this.correction = null;
  }

  clearErrors() {
    this.wasm.wasm_clear_errors(this.ptr);
    this.correction = null;
  }

  /**
   * Run a decoder against the current syndrome.
   *
   * @param {number} decoder DECODER.*
   * @returns {{failed: boolean, correctionX: Uint8Array, correctionZ: Uint8Array}}
   *   `failed` is the engine's verdict: the correction combined with the true
   *   error left a logical operator behind.
   */
  decode(decoder = DECODER.UNION_FIND) {
    const failed = this.wasm.wasm_decode(this.ptr, decoder) === 1;
    const len = this.numData * this.rounds;
    this.correction = {
      failed,
      correctionX: this.#copyBytes(this.wasm.wasm_get_correction_x_ptr(this.ptr), len),
      correctionZ: this.#copyBytes(this.wasm.wasm_get_correction_z_ptr(this.ptr), len),
    };
    return this.correction;
  }

  /**
   * The residual left after applying the correction: error XOR correction.
   *
   * This is the quantity that decides success. If the residual forms closed
   * loops it acts trivially on the logical qubit and the round is a success.
   * If it forms a chain spanning the patch it is a logical operator, and the
   * information is gone.
   */
  residual() {
    const { errorX, errorZ } = this.read();
    if (!this.correction) return { x: errorX, z: errorZ, hasCorrection: false };
    const { correctionX, correctionZ } = this.correction;
    const x = new Uint8Array(errorX.length);
    const z = new Uint8Array(errorZ.length);
    for (let i = 0; i < errorX.length; i++) {
      x[i] = errorX[i] ^ correctionX[i];
      z[i] = errorZ[i] ^ correctionZ[i];
    }
    return { x, z, hasCorrection: true };
  }

  free() {
    if (this.ptr) {
      this.wasm.wasm_free_session(this.ptr);
      this.ptr = 0;
    }
  }
}

/* -- Deliberately not wrapped ------------------------------------------ */

/*
 * The engine also exports `wasm_run_benchmark` and
 * `wasm_estimate_logical_fidelity`, which run a whole Monte Carlo experiment
 * inside Rust. Neither is usable from this build: both seed their generator
 * from a compile-time constant (`Xorshift::new(12345)`) on the non-Python
 * cfg branch, so every shot in a batch is identical and the result collapses
 * to a step function. See js/montecarlo.js, which samples noise in JS and
 * drives the engine's real decoders one shot at a time instead.
 */
