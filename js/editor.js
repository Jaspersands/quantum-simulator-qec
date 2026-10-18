/**
 * The editor's model (section 17): a circuit as layers of operations between TICKs, qubits
 * with coordinates, Pauli markers that propagate forward through the Clifford gates, and the
 * detectors and observables each marker flips. Pure functions and a small class; no DOM.
 *
 * A marker is a Pauli error placed before a layer. It moves forward as Heisenberg conjugation
 * by the gates (H swaps X and Z, S turns X into Y, CX copies X down and Z up, ...), a
 * measurement that anticommutes with it records a flip, a reset clears it, a classically
 * controlled gate on a record it flipped applies its Pauli. Detectors and observables are
 * parities of measurement records, so a marker flips those whose records it flipped an odd
 * number of times — the detecting regions Crumble shows, computed the same way. The gate tables
 * are generated from the package's gate data (`js/clifford-tables.js`), every Clifford gate
 * Stim has; `SPP`/`SPP_DAG` and `MPP` follow the product rule.
 */

import { parseStim, toStim, toCrumbleUrl, parseCrumbleUrl } from './crumble-format.js';
import { ONE, TWO, MEASURE_BASIS, PAIR_MEASURE_BASIS, HERALDS, RESETS as RESET_LIST, REC_CONTROLLED, PAIRS as PAIR_LIST } from './clifford-tables.js';

export const ONE_QUBIT = Object.keys(ONE).concat(Object.keys(MEASURE_BASIS), ['R', 'RX', 'RY']);
export const TWO_QUBIT = Object.keys(TWO).concat(Object.keys(PAIR_MEASURE_BASIS));
export const PALETTE = ['H', 'S', 'S_DAG', 'SQRT_X', 'X', 'Y', 'Z', 'CX', 'CZ', 'SWAP', 'R', 'RX', 'M', 'MX', 'MR'];

const RESETS = new Set(RESET_LIST);
const HERALD = new Set(HERALDS);
/** Gates whose targets come in pairs: an edit on one qubit takes its pair with it. */
const PAIRS = new Set(PAIR_LIST);

/* -- Pauli algebra on (x, z) bits, signs ignored ------------------------------------------ */

const PAULI_BITS = { I: [0, 0], X: [1, 0], Y: [1, 1], Z: [0, 1] };
const BITS_PAULI = ['I', 'X', 'Z', 'Y']; // index x + 2z

export function pauliName(x, z) {
  return BITS_PAULI[x + 2 * z];
}

function mul(a, b) {
  const [ax, az] = PAULI_BITS[a];
  const [bx, bz] = PAULI_BITS[b];
  return pauliName(ax ^ bx, az ^ bz);
}

function applyOne(gate, p) {
  const t = ONE[gate];
  if (!t || p === 'I') return p;
  if (p === 'X') return t.X;
  if (p === 'Z') return t.Z;
  // Y = XZ up to phase: the product of the images.
  return mul(t.X, t.Z);
}

function applyTwo(gate, pa, pb) {
  const t = TWO[gate];
  if (!t) return [pa, pb];
  let out = ['I', 'I'];
  const fold = (key) => {
    out = [mul(out[0], t[key][0]), mul(out[1], t[key][1])];
  };
  const [ax, az] = PAULI_BITS[pa];
  const [bx, bz] = PAULI_BITS[pb];
  if (ax) fold('XI');
  if (az) fold('ZI');
  if (bx) fold('IX');
  if (bz) fold('IZ');
  return out;
}

function anticommutes(p, basis) {
  return p !== 'I' && p !== basis;
}

/** The Pauli products of an `MPP`/`SPP` target list: `[[['X', 0], ['Z', 1]], [['Y', 2]]]`. */
export function pauliProducts(targets) {
  const products = [];
  let current = [];
  for (let i = 0; i < targets.length; i++) {
    const t = targets[i];
    if (t === '*') continue;
    const m = String(t).match(/^!?([XYZ])(\d+)$/);
    if (m) current.push([m[1], Number(m[2])]);
    if (targets[i + 1] !== '*') {
      if (current.length) products.push(current);
      current = [];
    }
  }
  return products;
}

/** How many measurement records an operation makes. */
export function recordsMade(op) {
  const name = op.name;
  if (MEASURE_BASIS[name] || HERALD.has(name) || name === 'MPAD') return op.targets.length;
  if (PAIR_MEASURE_BASIS[name]) return Math.floor(op.targets.length / 2);
  if (name === 'MPP') return pauliProducts(op.targets).length;
  return 0;
}

/* -- Layers -------------------------------------------------------------------------------- */

/** Splits a flat circuit into layers at TICKs; coordinates and detectors are kept apart. */
export function layersOf(circuit) {
  const coords = new Map();
  const layers = [[]];
  for (const op of circuit.ops) {
    if (op.name === 'QUBIT_COORDS') {
      for (const t of op.targets) coords.set(Number(t), [...op.args]);
      continue;
    }
    if (op.name === 'TICK') {
      layers.push([]);
      continue;
    }
    layers[layers.length - 1].push({ name: op.name, args: [...op.args], targets: [...op.targets] });
  }
  if (layers.length > 1 && layers[layers.length - 1].length === 0) layers.pop();
  return { coords, layers };
}

export function circuitOf(coords, layers) {
  const ops = [];
  for (const [q, xy] of [...coords.entries()].sort((a, b) => a[0] - b[0])) ops.push({ name: 'QUBIT_COORDS', args: [...xy], targets: [String(q)] });
  layers.forEach((layer, k) => {
    if (k > 0) ops.push({ name: 'TICK', args: [], targets: [] });
    for (const op of layer) ops.push({ name: op.name, args: [...op.args], targets: [...op.targets] });
  });
  return { ops };
}

function qubitOf(target) {
  const m = String(target).match(/^!?[XYZ]?(\d+)$/);
  return m ? Number(m[1]) : null;
}

function lookbackOf(target) {
  const m = String(target).match(/^rec\[-(\d+)\]$/);
  return m ? Number(m[1]) : null;
}

/**
 * Runs the markers forward. Returns, per layer, each marker's Pauli on each qubit before that
 * layer's gates (`frames[layer][k]` a Map qubit → 'X'|'Y'|'Z'), the measurement records each
 * marker flips (`flips[k]` a Set of record indices), the records in order (`records[i]` =
 * {layer, qubit, name}), and the detectors and observables each marker flips.
 */
export function propagate(layers, markers, detectors, observables) {
  const frames = [];
  const live = markers.map(() => new Map());
  const flips = markers.map(() => new Set());
  const records = [];
  const record = (L, q, name, flipsFor) => {
    const index = records.length;
    records.push({ layer: L, qubit: q, name });
    live.forEach((f, k) => {
      if (flipsFor(f)) flips[k].add(index);
    });
  };
  layers.forEach((layer, L) => {
    // Markers placed before this layer.
    markers.forEach((mk, k) => {
      if (mk.layer === L) live[k].set(mk.qubit, mul(live[k].get(mk.qubit) ?? 'I', mk.pauli));
    });
    frames.push(live.map((f) => new Map([...f].filter(([, p]) => p !== 'I'))));
    for (const op of layer) {
      const name = op.name;
      if (ONE[name]) {
        for (const t of op.targets) {
          const q = qubitOf(t);
          if (q === null) continue;
          for (const f of live) if (f.has(q)) f.set(q, applyOne(name, f.get(q)));
        }
      } else if (TWO[name]) {
        for (let i = 0; i + 1 < op.targets.length; i += 2) {
          const a = op.targets[i];
          const b = op.targets[i + 1];
          const la = lookbackOf(a);
          const lb = lookbackOf(b);
          if (la !== null || lb !== null) {
            // Classical control: a marker that flipped the record applies the gate's Pauli.
            const pauli = REC_CONTROLLED[name];
            const q = qubitOf(la !== null ? b : a);
            const r = records.length - (la !== null ? la : lb);
            if (!pauli || q === null || r < 0) continue;
            live.forEach((f, k) => {
              if (flips[k].has(r)) f.set(q, mul(f.get(q) ?? 'I', pauli));
            });
            continue;
          }
          const qa = qubitOf(a);
          const qb = qubitOf(b);
          if (qa === null || qb === null) continue;
          for (const f of live) {
            const [pa, pb] = applyTwo(name, f.get(qa) ?? 'I', f.get(qb) ?? 'I');
            f.set(qa, pa);
            f.set(qb, pb);
          }
        }
      } else if (name === 'SPP' || name === 'SPP_DAG') {
        // exp(±iπ/4 P): a Pauli that commutes with P is unchanged, one that anticommutes is multiplied by P.
        for (const prod of pauliProducts(op.targets)) {
          for (const f of live) {
            let odd = 0;
            for (const [p, q] of prod) if (anticommutes(f.get(q) ?? 'I', p)) odd ^= 1;
            if (odd) for (const [p, q] of prod) f.set(q, mul(f.get(q) ?? 'I', p));
          }
        }
      } else if (MEASURE_BASIS[name] || RESETS.has(name)) {
        const basis = MEASURE_BASIS[name];
        for (const t of op.targets) {
          const q = qubitOf(t);
          if (q === null) continue;
          if (basis) record(L, q, name, (f) => anticommutes(f.get(q) ?? 'I', basis));
          if (RESETS.has(name)) for (const f of live) f.delete(q);
        }
      } else if (PAIR_MEASURE_BASIS[name]) {
        const basis = PAIR_MEASURE_BASIS[name];
        for (let i = 0; i + 1 < op.targets.length; i += 2) {
          const qa = qubitOf(op.targets[i]);
          const qb = qubitOf(op.targets[i + 1]);
          if (qa === null || qb === null) continue;
          record(L, qa, name, (f) => anticommutes(f.get(qa) ?? 'I', basis) !== anticommutes(f.get(qb) ?? 'I', basis));
        }
      } else if (name === 'MPP') {
        // Each product: anticommutes when an odd number of its factors do.
        for (const prod of pauliProducts(op.targets)) {
          record(L, prod[0][1], name, (f) => {
            let odd = 0;
            for (const [p, q] of prod) if (anticommutes(f.get(q) ?? 'I', p)) odd ^= 1;
            return odd === 1;
          });
        }
      } else if (HERALD.has(name) || name === 'MPAD') {
        // Heralds and padding make records no Pauli marker flips (a pad's targets are bits, not qubits).
        for (const t of op.targets) record(L, name === 'MPAD' ? null : qubitOf(t), name, () => false);
      }
    }
  });
  frames.push(live.map((f) => new Map([...f].filter(([, p]) => p !== 'I'))));
  const recordsOf = (targets, at) => targets.map((t) => {
    const k = lookbackOf(t);
    return k === null ? null : at - k;
  }).filter((i) => i !== null && i >= 0);
  const detFlips = markers.map(() => new Set());
  const obsFlips = markers.map(() => new Set());
  detectors.forEach((d, j) => {
    const recs = recordsOf(d.targets, d.at);
    markers.forEach((_, k) => {
      let odd = 0;
      for (const r of recs) if (flips[k].has(r)) odd ^= 1;
      if (odd) detFlips[k].add(j);
    });
  });
  // An observable is the parity over every OBSERVABLE_INCLUDE with its index: a marker that
  // flips two of them leaves it alone.
  observables.forEach((o) => {
    const recs = recordsOf(o.targets, o.at);
    markers.forEach((_, k) => {
      let odd = 0;
      for (const r of recs) if (flips[k].has(r)) odd ^= 1;
      if (odd) {
        if (obsFlips[k].has(o.index)) obsFlips[k].delete(o.index);
        else obsFlips[k].add(o.index);
      }
    });
  });
  return { frames, flips, records, detFlips, obsFlips };
}

/**
 * Detectors and observables of layered ops, each with the count of measurement records made
 * before it (so `rec[-k]` resolves to an absolute index), in circuit order.
 */
export function annotationsOf(layers) {
  const detectors = [];
  const observables = [];
  let at = 0;
  for (const layer of layers) {
    for (const op of layer) {
      if (op.name === 'DETECTOR') detectors.push({ targets: op.targets, args: op.args, at });
      else if (op.name === 'OBSERVABLE_INCLUDE') observables.push({ targets: op.targets, index: op.args[0] ?? 0, at });
      else at += recordsMade(op);
    }
  }
  return { detectors, observables };
}

const ANNOTATIONS = new Set(['DETECTOR', 'OBSERVABLE_INCLUDE']);

/* -- The editable circuit ------------------------------------------------------------------ */

export class EditorModel {
  constructor() {
    this.coords = new Map();
    this.layers = [[]];
    this.markers = [];
  }

  static fromStim(text) {
    const m = new EditorModel();
    const { coords, layers } = layersOf(parseStim(text));
    m.coords = coords;
    m.layers = layers.length ? layers : [[]];
    return m;
  }

  static fromCrumbleUrl(url) {
    const { circuit, markers } = parseCrumbleUrl(url);
    const m = new EditorModel();
    // A marker is a Pauli placed before the op it precedes. Stim and Crumble write marks right
    // before an instruction, mid-layer as often as not, and the model places markers before a
    // layer's gates: so a mark before an op that is not the first of its layer splits the layer
    // there (a TICK goes in), a mark before a TICK sits after that layer's gates (the next
    // layer; the same layer when it is empty), and a mark at the end is after everything.
    const marked = new Set(markers.map((mk) => mk.before));
    const ops = [];
    const layerAt = [];
    let L = 0;
    let count = 0;
    circuit.ops.forEach((op, i) => {
      if (op.name === 'QUBIT_COORDS') {
        layerAt[i] = L;
        ops.push(op);
        return;
      }
      if (op.name === 'TICK') {
        layerAt[i] = count ? L + 1 : L;
        ops.push(op);
        L++;
        count = 0;
        return;
      }
      if (marked.has(i) && count) {
        ops.push({ name: 'TICK', args: [], targets: [] });
        L++;
        count = 0;
      }
      layerAt[i] = L;
      ops.push(op);
      count++;
    });
    layerAt[circuit.ops.length] = count ? L + 1 : L;
    const split = layersOf({ ops });
    m.coords = split.coords;
    m.layers = split.layers.length ? split.layers : [[]];
    m.markers = markers.map((mk) => ({ k: mk.k, pauli: mk.pauli, qubit: mk.qubit, layer: Math.min(layerAt[mk.before] ?? L, m.layers.length - 1) }));
    return m;
  }

  toCircuit() {
    return circuitOf(this.coords, this.layers);
  }

  toStim() {
    return toStim(this.toCircuit());
  }

  toCrumbleUrl() {
    // A marker before layer L goes before the first op of that layer; in an empty layer, before
    // the TICK that closes it (or at the end of the circuit).
    const circuit = this.toCircuit();
    const anchor = [];
    let L = 0;
    circuit.ops.forEach((op, i) => {
      if (op.name === 'QUBIT_COORDS') return;
      if (anchor[L] === undefined) anchor[L] = i;
      if (op.name === 'TICK') L++;
    });
    const marks = this.markers.map((mk) => ({ ...mk, before: anchor[mk.layer] ?? circuit.ops.length }));
    return toCrumbleUrl(circuit, marks);
  }

  qubits() {
    const set = new Set(this.coords.keys());
    for (const layer of this.layers) for (const op of layer) {
      if (op.name === 'MPAD') continue; // its targets are bits
      for (const t of op.targets) {
        const q = qubitOf(t);
        if (q !== null) set.add(q);
      }
    }
    return [...set].sort((a, b) => a - b);
  }

  /** Every op in `layer` touching `qubit`. */
  opsOn(layer, qubit) {
    return this.layers[layer].filter((op) => op.name !== 'MPAD' && op.targets.some((t) => qubitOf(t) === qubit));
  }

  /** The records made before each op (`at`), in circuit order. */
  _recordStarts() {
    const starts = new Map();
    let at = 0;
    for (const layer of this.layers) for (const op of layer) {
      starts.set(op, at);
      at += recordsMade(op);
    }
    return starts;
  }

  /**
   * Every op with `rec[-k]` targets (detectors, observables, classically controlled gates),
   * each target as the absolute record index it names (a string when it is not a record or
   * reaches before the circuit). Taken before an edit; `_rewriteRecordRefs` writes the lookbacks
   * back after it, through the edit's map of old record indices to new ones.
   */
  _recordRefs() {
    const starts = this._recordStarts();
    const refs = [];
    for (const layer of this.layers) for (const op of layer) {
      if (!ANNOTATIONS.has(op.name) && !TWO[op.name]) continue;
      if (!op.targets.some((t) => lookbackOf(t) !== null)) continue;
      const at = starts.get(op);
      refs.push({ op, abs: op.targets.map((t) => {
        const k = lookbackOf(t);
        return k === null || at - k < 0 ? t : at - k;
      }) });
    }
    return refs;
  }

  /** Writes the lookbacks of `refs` back: `remap(old index)` is the record's new index, or null when the edit removed it. */
  _rewriteRecordRefs(refs, remap) {
    const starts = this._recordStarts();
    for (const { op, abs } of refs) {
      const at = starts.get(op);
      if (at === undefined) continue; // the op itself went with the edit
      let targets = abs.map((x) => {
        if (typeof x === 'string') return x;
        const n = remap(x);
        return n === null || n === undefined ? null : `rec[-${at - n}]`;
      });
      if (TWO[op.name]) {
        // A controlled gate whose record vanished loses that pair.
        const pairs = [];
        for (let j = 0; j + 1 < targets.length; j += 2) if (targets[j] !== null && targets[j + 1] !== null) pairs.push(targets[j], targets[j + 1]);
        targets = pairs;
      } else targets = targets.filter((t) => t !== null);
      op.targets = targets;
      if (!targets.length) for (const layer of this.layers) {
        const i = layer.indexOf(op);
        if (i >= 0) layer.splice(i, 1);
      }
    }
  }

  removeOn(layer, qubit) {
    const ops = this.layers[layer];
    for (let i = ops.length - 1; i >= 0; i--) {
      const op = ops[i];
      if (op.name === 'MPAD' || !op.targets.some((t) => qubitOf(t) === qubit)) continue;
      const refs = this._recordRefs();
      const base = this._recordStarts().get(op);
      const before = recordsMade(op);
      const whole = op.name === 'MPP' || op.name === 'SPP' || op.name === 'SPP_DAG' || ANNOTATIONS.has(op.name);
      // The record ordinals this op loses: all of them, those of the removed pairs, or those
      // of the removed targets.
      const removed = new Set();
      if (whole) {
        for (let j = 0; j < before; j++) removed.add(j);
        ops.splice(i, 1);
      } else if (PAIRS.has(op.name)) {
        const kept = [];
        for (let j = 0; j + 1 < op.targets.length; j += 2) {
          if (qubitOf(op.targets[j]) === qubit || qubitOf(op.targets[j + 1]) === qubit) {
            if (before) removed.add(j / 2);
          } else kept.push(op.targets[j], op.targets[j + 1]);
        }
        op.targets = kept;
        if (!op.targets.length) ops.splice(i, 1);
      } else {
        if (before) op.targets.forEach((t, j) => {
          if (qubitOf(t) === qubit) removed.add(j);
        });
        op.targets = op.targets.filter((t) => qubitOf(t) !== qubit);
        if (!op.targets.length) ops.splice(i, 1);
      }
      if (!removed.size) continue;
      const gone = [...removed].sort((a, b) => a - b);
      this._rewriteRecordRefs(refs, (a) => {
        if (a < base) return a;
        if (a < base + before) return removed.has(a - base) ? null : a - gone.filter((j) => j < a - base).length;
        return a - removed.size;
      });
    }
  }

  /** Places a gate on `qubits` in `layer`, replacing what they had there. */
  place(layer, name, qubits) {
    while (this.layers.length <= layer) this.layers.push([]);
    for (const q of qubits) this.removeOn(layer, q);
    const ops = this.layers[layer];
    const refs = this._recordRefs();
    const existing = ops.find((op) => op.name === name && !PAIRS.has(name) && !op.args.length);
    let at;
    let made;
    if (existing) {
      // New records go at the end of the existing op's run.
      at = this._recordStarts().get(existing) + recordsMade(existing);
      existing.targets.push(...qubits.map(String));
      made = recordsMade(existing) - (at - this._recordStarts().get(existing));
    } else {
      // Before the layer's annotations, so a detector written on this layer still closes it.
      let index = ops.findIndex((op) => ANNOTATIONS.has(op.name));
      if (index < 0) index = ops.length;
      const op = { name, args: [], targets: qubits.map(String) };
      ops.splice(index, 0, op);
      at = this._recordStarts().get(op);
      made = recordsMade(op);
    }
    if (made) this._rewriteRecordRefs(refs, (a) => (a < at ? a : a + made));
  }

  addLayer(after) {
    this.layers.splice(after + 1, 0, []);
    for (const mk of this.markers) if (mk.layer > after) mk.layer++;
  }

  deleteLayer(index) {
    if (this.layers.length <= 1) {
      this.layers[0] = [];
      return;
    }
    const refs = this._recordRefs();
    const ops = this.layers[index];
    const at = ops.length ? this._recordStarts().get(ops[0]) : null;
    const removed = ops.reduce((n, op) => n + recordsMade(op), 0);
    this.layers.splice(index, 1);
    if (removed) this._rewriteRecordRefs(refs, (a) => (a < at ? a : a < at + removed ? null : a - removed));
    for (const mk of this.markers) if (mk.layer > index) mk.layer--;
    this.markers = this.markers.filter((mk) => mk.layer < this.layers.length);
  }

  setCoords(qubit, x, y) {
    const rest = (this.coords.get(qubit) ?? []).slice(2);
    this.coords.set(qubit, [x, y, ...rest]);
  }

  toggleMarker(k, pauli, qubit, layer) {
    const i = this.markers.findIndex((m) => m.k === k && m.qubit === qubit && m.layer === layer);
    if (i >= 0) {
      if (this.markers[i].pauli === pauli) this.markers.splice(i, 1);
      else this.markers[i].pauli = pauli;
    } else this.markers.push({ k, pauli, qubit, layer });
  }

  /** A detector (or observable, with `index`) on the newest `count` records of `layer`. */
  addDetector(layer, lookbacks, args = [], observable = null) {
    while (this.layers.length <= layer) this.layers.push([]);
    const targets = lookbacks.map((k) => `rec[-${k}]`);
    if (observable === null) this.layers[layer].push({ name: 'DETECTOR', args, targets });
    else this.layers[layer].push({ name: 'OBSERVABLE_INCLUDE', args: [observable], targets });
  }

  /** The measurement records made in `layer`, as lookbacks counted from the end of that layer. */
  recordsIn(layer) {
    const { records } = propagate(this.layers, [], [], []);
    let through = 0;
    for (let L = 0; L <= layer; L++) through += records.filter((r) => r.layer === L).length;
    return records.map((r, i) => ({ ...r, lookback: through - i })).filter((r) => r.layer === layer);
  }

  analysis() {
    const { detectors, observables } = annotationsOf(this.layers);
    const markers = this.markers.map((m) => ({ ...m }));
    return { ...propagate(this.layers, markers, detectors, observables), detectors, observables, markers };
  }
}

export { EXAMPLES } from './editor-examples.js';
