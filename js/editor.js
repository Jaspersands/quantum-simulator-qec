/**
 * The editor's model (section 17): a circuit as layers of operations between TICKs, qubits
 * with coordinates, Pauli markers that propagate forward through the Clifford gates, and the
 * detectors and observables each marker flips. Pure functions and a small class; no DOM.
 *
 * A marker is a Pauli error placed before a layer. It moves forward as Heisenberg conjugation
 * by the gates (H swaps X and Z, S turns X into Y, CX copies X down and Z up, ...), a
 * measurement that anticommutes with it records a flip, a reset clears it. Detectors and
 * observables are parities of measurement records, so a marker flips those whose records it
 * flipped an odd number of times — the detecting regions Crumble shows, computed the same way.
 */

import { parseStim, toStim, toCrumbleUrl, parseCrumbleUrl } from './crumble-format.js';

export const ONE_QUBIT = ['H', 'S', 'S_DAG', 'SQRT_X', 'SQRT_X_DAG', 'X', 'Y', 'Z', 'I', 'R', 'RX', 'M', 'MX', 'MR', 'MRX', 'C_XYZ', 'C_ZYX'];
export const TWO_QUBIT = ['CX', 'CZ', 'CY', 'SWAP', 'ISWAP', 'XCX', 'XCZ', 'YCY'];
export const PALETTE = ['H', 'S', 'S_DAG', 'SQRT_X', 'X', 'Y', 'Z', 'CX', 'CZ', 'SWAP', 'R', 'RX', 'M', 'MX', 'MR'];

/* -- Pauli algebra on (x, z) bits, signs ignored ------------------------------------------ */

const PAULI_BITS = { I: [0, 0], X: [1, 0], Y: [1, 1], Z: [0, 1] };
const BITS_PAULI = ['I', 'X', 'Z', 'Y']; // index x + 2z

export function pauliName(x, z) {
  return BITS_PAULI[x + 2 * z];
}

/** How a one-qubit Clifford moves (x, z): a table from the gate's action on X and on Z. */
const ONE = {
  H: { X: 'Z', Z: 'X' },
  S: { X: 'Y', Z: 'Z' },
  S_DAG: { X: 'Y', Z: 'Z' },
  SQRT_X: { X: 'X', Z: 'Y' },
  SQRT_X_DAG: { X: 'X', Z: 'Y' },
  SQRT_Y: { X: 'Z', Z: 'X' },
  SQRT_Y_DAG: { X: 'Z', Z: 'X' },
  C_XYZ: { X: 'Y', Z: 'X' },
  C_ZYX: { X: 'Z', Z: 'Y' },
  H_YZ: { X: 'X', Z: 'Y' },
  H_XY: { X: 'Y', Z: 'Z' },
};

function applyOne(gate, p) {
  const t = ONE[gate];
  if (!t || p === 'I') return p;
  if (p === 'X') return t.X;
  if (p === 'Z') return t.Z;
  // Y = XZ up to phase: the product of the images.
  const [ax, az] = PAULI_BITS[t.X];
  const [bx, bz] = PAULI_BITS[t.Z];
  return pauliName(ax ^ bx, az ^ bz);
}

/** Two-qubit Cliffords as their action on X⊗I, Z⊗I, I⊗X, I⊗Z: each a pair of Paulis. */
const TWO = {
  CX: { XI: ['X', 'X'], ZI: ['Z', 'I'], IX: ['I', 'X'], IZ: ['Z', 'Z'] },
  CZ: { XI: ['X', 'Z'], ZI: ['Z', 'I'], IX: ['Z', 'X'], IZ: ['I', 'Z'] },
  CY: { XI: ['X', 'Y'], ZI: ['Z', 'I'], IX: ['Z', 'X'], IZ: ['Z', 'Z'] },
  SWAP: { XI: ['I', 'X'], ZI: ['I', 'Z'], IX: ['X', 'I'], IZ: ['Z', 'I'] },
  ISWAP: { XI: ['Z', 'Y'], ZI: ['I', 'Z'], IX: ['Y', 'Z'], IZ: ['Z', 'I'] },
  XCX: { XI: ['X', 'I'], ZI: ['Z', 'X'], IX: ['I', 'X'], IZ: ['X', 'Z'] },
  XCZ: { XI: ['X', 'I'], ZI: ['Z', 'Z'], IX: ['X', 'X'], IZ: ['I', 'Z'] },
  YCY: { XI: ['X', 'Y'], ZI: ['Z', 'Y'], IX: ['Y', 'X'], IZ: ['Y', 'Z'] },
};

function mul(a, b) {
  const [ax, az] = PAULI_BITS[a];
  const [bx, bz] = PAULI_BITS[b];
  return pauliName(ax ^ bx, az ^ bz);
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

const MEASURE_BASIS = { M: 'Z', MR: 'Z', MX: 'X', MRX: 'X', MY: 'Y', MRY: 'Y' };
const RESETS = new Set(['R', 'RX', 'RY', 'MR', 'MRX', 'MRY']);

function anticommutes(p, basis) {
  return p !== 'I' && p !== basis;
}

/* -- Layers -------------------------------------------------------------------------------- */

/** Splits a flat circuit into layers at TICKs; coordinates and detectors are kept apart. */
export function layersOf(circuit) {
  const coords = new Map();
  const layers = [[]];
  for (const op of circuit.ops) {
    if (op.name === 'QUBIT_COORDS') {
      for (const t of op.targets) coords.set(Number(t), op.args.slice(0, 2));
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
  for (const [q, xy] of [...coords.entries()].sort((a, b) => a[0] - b[0])) ops.push({ name: 'QUBIT_COORDS', args: xy, targets: [String(q)] });
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
          const a = qubitOf(op.targets[i]);
          const b = qubitOf(op.targets[i + 1]);
          if (a === null || b === null) continue;
          for (const f of live) {
            const [pa, pb] = applyTwo(name, f.get(a) ?? 'I', f.get(b) ?? 'I');
            f.set(a, pa);
            f.set(b, pb);
          }
        }
      } else if (MEASURE_BASIS[name] || RESETS.has(name)) {
        const basis = MEASURE_BASIS[name];
        for (const t of op.targets) {
          const q = qubitOf(t);
          if (q === null) continue;
          if (basis) {
            const index = records.length;
            records.push({ layer: L, qubit: q, name });
            live.forEach((f, k) => {
              if (anticommutes(f.get(q) ?? 'I', basis)) flips[k].add(index);
            });
          }
          if (RESETS.has(name)) for (const f of live) f.delete(q);
        }
      } else if (name === 'MPP') {
        // Each product: anticommutes when an odd number of its factors do.
        let group = [];
        const groups = [];
        for (const t of op.targets) {
          if (t === '*') continue;
          group.push(t);
          groups.push(group);
          group = [];
        }
        // Rebuild groups honouring the combiner.
        const products = [];
        let current = [];
        for (let i = 0; i < op.targets.length; i++) {
          const t = op.targets[i];
          if (t === '*') continue;
          current.push(t);
          if (op.targets[i + 1] !== '*') {
            products.push(current);
            current = [];
          }
        }
        for (const prod of products) {
          const index = records.length;
          records.push({ layer: L, qubit: qubitOf(prod[0]), name: 'MPP' });
          live.forEach((f, k) => {
            let odd = 0;
            for (const t of prod) {
              const m = String(t).match(/^!?([XYZ])(\d+)$/);
              if (!m) continue;
              if (anticommutes(f.get(Number(m[2])) ?? 'I', m[1])) odd ^= 1;
            }
            if (odd) flips[k].add(index);
          });
        }
      }
    }
  });
  frames.push(live.map((f) => new Map([...f].filter(([, p]) => p !== 'I'))));
  const recordsOf = (targets, at) => targets.map((t) => {
    const m = String(t).match(/^rec\[-(\d+)\]$/);
    return m ? at - Number(m[1]) : null;
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
  observables.forEach((o, j) => {
    const recs = recordsOf(o.targets, o.at);
    markers.forEach((_, k) => {
      let odd = 0;
      for (const r of recs) if (flips[k].has(r)) odd ^= 1;
      if (odd) obsFlips[k].add(o.index);
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
      if (MEASURE_BASIS[op.name]) at += op.targets.length;
      else if (op.name === 'MPP') at += op.targets.filter((t, i) => t !== '*' && op.targets[i + 1] !== '*').length;
      else if (op.name === 'DETECTOR') detectors.push({ targets: op.targets, args: op.args, at });
      else if (op.name === 'OBSERVABLE_INCLUDE') observables.push({ targets: op.targets, index: op.args[0] ?? 0, at });
    }
  }
  return { detectors, observables };
}

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
    const { coords, layers } = layersOf(circuit);
    m.coords = coords;
    m.layers = layers.length ? layers : [[]];
    // A marker before op index i lands before the layer that op begins.
    let L = 0;
    let index = 0;
    const layerOfOp = [];
    for (const op of circuit.ops) {
      if (op.name === 'TICK') L++;
      layerOfOp.push(L);
      index++;
    }
    m.markers = markers.map((mk) => ({ k: mk.k, pauli: mk.pauli, qubit: mk.qubit, layer: Math.min(layerOfOp[mk.before] ?? L, m.layers.length - 1) }));
    return m;
  }

  toCircuit() {
    return circuitOf(this.coords, this.layers);
  }

  toStim() {
    return toStim(this.toCircuit());
  }

  toCrumbleUrl() {
    // A marker before layer L goes before the first op of that layer (after its TICK).
    const circuit = this.toCircuit();
    const firstOp = [];
    let L = 0;
    circuit.ops.forEach((op, i) => {
      if (op.name === 'TICK') {
        L++;
        return;
      }
      if (op.name === 'QUBIT_COORDS') return;
      if (firstOp[L] === undefined) firstOp[L] = i;
    });
    const marks = this.markers.map((mk) => ({ ...mk, before: firstOp[mk.layer] ?? circuit.ops.length }));
    return toCrumbleUrl(circuit, marks);
  }

  qubits() {
    const set = new Set(this.coords.keys());
    for (const layer of this.layers) for (const op of layer) for (const t of op.targets) {
      const q = qubitOf(t);
      if (q !== null) set.add(q);
    }
    return [...set].sort((a, b) => a - b);
  }

  /** Every op in `layer` touching `qubit`. */
  opsOn(layer, qubit) {
    return this.layers[layer].filter((op) => op.targets.some((t) => qubitOf(t) === qubit));
  }

  removeOn(layer, qubit) {
    const ops = this.layers[layer];
    for (let i = ops.length - 1; i >= 0; i--) {
      const op = ops[i];
      if (!op.targets.some((t) => qubitOf(t) === qubit)) continue;
      if (TWO[op.name] || op.name === 'MPP' || op.name === 'DETECTOR' || op.name === 'OBSERVABLE_INCLUDE') {
        ops.splice(i, 1);
      } else {
        op.targets = op.targets.filter((t) => qubitOf(t) !== qubit);
        if (!op.targets.length) ops.splice(i, 1);
      }
    }
  }

  /** Places a gate on `qubits` in `layer`, replacing what they had there. */
  place(layer, name, qubits) {
    while (this.layers.length <= layer) this.layers.push([]);
    for (const q of qubits) this.removeOn(layer, q);
    const existing = this.layers[layer].find((op) => op.name === name && !TWO[name]);
    if (existing) existing.targets.push(...qubits.map(String));
    else this.layers[layer].push({ name, args: [], targets: qubits.map(String) });
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
    this.layers.splice(index, 1);
    for (const mk of this.markers) if (mk.layer > index) mk.layer--;
    this.markers = this.markers.filter((mk) => mk.layer < this.layers.length);
  }

  setCoords(qubit, x, y) {
    this.coords.set(qubit, [x, y]);
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
