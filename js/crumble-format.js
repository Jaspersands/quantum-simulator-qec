/**
 * Stim text and Crumble URLs, in and out, for the site's editor (section 17).
 *
 * A circuit here is `{ ops }`: a flat list of `{ name, args, targets }` with targets as Stim
 * writes them (`0`, `!1`, `X2`, `rec[-1]`, `sweep[3]`, `*` the combiner, one token each: an
 * `MPP X0*Z1 Y2` is the targets `X0`, `*`, `Z1`, `Y2`). REPEAT blocks are unrolled on the way in
 * (the editor edits a flat circuit; a million operations is the most it unrolls to); tags are
 * dropped; gate names are Stim's canonical ones (`CNOT` reads as `CX`, `MZ` as `M`).
 *
 * The Crumble URL is the one `Circuit.to_crumble_url` writes (Stim's `export_crumble_url`):
 * `#circuit=` then instructions joined by `;`, `QUBIT_COORDS` as `Q`, `DETECTOR` as `DT`,
 * `OBSERVABLE_INCLUDE` as `OI`, arguments in parentheses with integers bare and other numbers
 * as `%g`, targets joined by `_` (the first follows the arguments directly), a marker as
 * `MARKX(k)q` between TICKs, and a final `_`.
 */

import { ALIASES } from './clifford-tables.js';

export const CRUMBLE_PREFIX = 'https://algassert.com/crumble#circuit=';

/** The most operations a circuit unrolls to; past it, the parse fails rather than hangs. */
export const MAX_OPS = 1_000_000;

const SHORT = { QUBIT_COORDS: 'Q', DETECTOR: 'DT', OBSERVABLE_INCLUDE: 'OI' };
const LONG = { Q: 'QUBIT_COORDS', DT: 'DETECTOR', OI: 'OBSERVABLE_INCLUDE' };

const TARGET = /^(!?\d+|!?[XYZ]\d+|rec\[-[1-9]\d*\]|sweep\[\d+\]|\*)$/;
const NUMBER = /^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$/;

/** The most REPEAT blocks nest (the package's parser stops at the same depth). */
const MAX_DEPTH = 100;

function parseArgs(text, line) {
  const args = text.split(',').map((s) => s.trim()).filter(Boolean);
  for (const a of args) if (!NUMBER.test(a) || !Number.isFinite(Number(a))) throw new Error(`"${line}": "${a}" is not a number`);
  return args.map(Number);
}

/**
 * A number as the package prints an argument: an integer that fits an i64 bare and exact,
 * anything else as C's `%g` (six significant digits, rounded half to even on the exact binary
 * value, trailing zeros dropped, exponent form below 1e-4 and from 1e6).
 */
export function formatArg(value) {
  if (!Number.isFinite(value)) return String(value);
  if (Number.isInteger(value) && Math.abs(value) < 9223372036854775808) return BigInt(value).toString();
  if (value === 0) return '0';
  // toExponential(20) is exact for every double, so a tie at the seventh digit is seen as one.
  const [mant, expStr] = value.toExponential(20).split('e');
  const neg = mant[0] === '-';
  const digits = mant.replace('-', '').replace('.', '');
  let exp = Number(expStr);
  let head = digits.slice(0, 6);
  const tail = digits.slice(6);
  const up = tail[0] > '5' || (tail[0] === '5' && (/[1-9]/.test(tail.slice(1)) || Number(head[5]) % 2 === 1));
  if (up) {
    let n = String(Number(head) + 1);
    if (n.length > 6) {
      n = n.slice(0, 6);
      exp += 1;
    }
    head = n;
  }
  let out;
  if (exp < -4 || exp >= 6) {
    const m = (head[0] + '.' + head.slice(1)).replace(/\.?0+$/, '');
    out = `${m}e${exp < 0 ? '-' : '+'}${String(Math.abs(exp)).padStart(2, '0')}`;
  } else if (exp >= 0) {
    const intPart = head.slice(0, exp + 1);
    const frac = head.slice(exp + 1).replace(/0+$/, '');
    out = frac ? `${intPart}.${frac}` : intPart;
  } else {
    out = ('0.' + '0'.repeat(-exp - 1) + head).replace(/0+$/, '');
  }
  return (neg ? '-' : '') + out;
}

/** Stim's canonical name for a gate (`CNOT` is `CX`, `MZ` is `M`, `SQRT_Z` is `S`). */
export function canonicalName(name) {
  const upper = name.toUpperCase();
  return ALIASES[upper] ?? upper;
}

/** Stim text to a flat circuit. Unknown instructions are kept as written. */
export function parseStim(text) {
  const lines = text.split(/\r?\n/);
  let i = 0;
  const block = (depth) => {
    const body = [];
    for (; i < lines.length; i++) {
      const line = lines[i].split('#')[0].trim() === '}' ? '}' : lines[i].trim();
      if (!line || line.startsWith('#')) continue;
      if (line === '}') {
        if (depth === 0) throw new Error(`line ${i + 1}: "}" closes no REPEAT block`);
        return body;
      }
      if (line.startsWith('}')) throw new Error(`line ${i + 1}: a closing brace stands alone on its line`);
      const m = line.match(/^REPEAT\s+(\d+)\s*\{\s*(#.*)?$/i);
      if (m) {
        const count = Number(m[1]);
        if (count < 1) throw new Error(`line ${i + 1}: REPEAT ${m[1]}: a block repeats at least once`);
        if (depth >= MAX_DEPTH) throw new Error(`line ${i + 1}: REPEAT blocks nested more than ${MAX_DEPTH} deep`);
        i++;
        const inner = block(depth + 1);
        if (i >= lines.length) throw new Error('a REPEAT block is never closed');
        repeatInto(body, inner, count);
        continue;
      }
      body.push(parseInstruction(line));
      if (body.length > MAX_OPS) throw new Error(`more than ${MAX_OPS} operations`);
    }
    if (depth > 0) throw new Error('a REPEAT block is never closed');
    return body;
  };
  return { ops: block(0) };
}

/** Appends `count` copies of `inner` to `body` without spreading (a spread of a million elements overflows the stack). */
function repeatInto(body, inner, count) {
  if (!inner.length) return;
  if (body.length + inner.length * count > MAX_OPS) throw new Error(`more than ${MAX_OPS} operations after unrolling REPEAT blocks`);
  for (let k = 0; k < count; k++) for (const op of inner) body.push({ name: op.name, args: [...op.args], targets: [...op.targets] });
}

export function parseInstruction(line) {
  // The name and its tag come first (a tag may hold a '#'); the comment starts at the first '#'
  // after them.
  const head = line.match(/^\s*([A-Za-z_][A-Za-z0-9_]*)(\[[^\]]*\])?/);
  if (!head) throw new Error(`cannot read "${line}"`);
  let rest = line.slice(head[0].length);
  const hash = rest.indexOf('#');
  if (hash >= 0) rest = rest.slice(0, hash);
  rest = rest.trim();
  const m = rest.match(/^(\(([^)]*)\))?\s*(.*)$/);
  if (!m) throw new Error(`cannot read "${line}"`);
  const name = canonicalName(head[1]);
  const args = m[2] ? parseArgs(m[2], line) : [];
  const targets = splitTargets(m[3] ? m[3].split(/\s+/).filter(Boolean) : []);
  for (const t of targets) if (!TARGET.test(t)) throw new Error(`"${line}": "${t}" is not a target`);
  return { name, args, targets };
}

/** Target words to tokens: `X0*Z1` is `X0`, `*`, `Z1`; a Pauli letter in either case reads as upper. */
function splitTargets(words) {
  const out = [];
  const token = (s) => s.replace(/^(!?)([xyz])(\d+)$/, (_, bang, p, q) => `${bang}${p.toUpperCase()}${q}`);
  for (const w of words) {
    if (!w.includes('*')) {
      out.push(token(w));
      continue;
    }
    w.split('*').forEach((s, j) => {
      if (j > 0) out.push('*');
      if (s.length) out.push(token(s));
    });
  }
  return out;
}

export function toStim(circuit) {
  return circuit.ops.map((op) => formatInstruction(op)).join('\n');
}

/** Targets as Stim writes them: products joined by `*` without spaces. */
export function formatTargets(targets) {
  let text = '';
  for (let k = 0; k < targets.length; k++) {
    const t = targets[k];
    if (t === '*') {
      text += '*';
      continue;
    }
    if (k > 0 && targets[k - 1] !== '*') text += ' ';
    text += t;
  }
  return text;
}

export function formatInstruction(op) {
  let out = op.name;
  if (op.args.length) out += `(${op.args.map(formatArg).join(', ')})`;
  if (op.targets.length) out += ' ' + formatTargets(op.targets);
  return out;
}

/**
 * The Crumble URL of a circuit, with `markers` as `[{ k, pauli, qubit, before }]` placed before
 * the op at index `before` (between two TICKs, as Stim writes marks). A marker before a TICK sits
 * at the end of the layer that TICK closes.
 */
export function toCrumbleUrl(circuit, markers = []) {
  const parts = [];
  const byIndex = new Map();
  for (const mk of markers) {
    if (!byIndex.has(mk.before)) byIndex.set(mk.before, []);
    byIndex.get(mk.before).push(mk);
  }
  const emit = (marks) => {
    parts.push('TICK');
    for (const mk of marks) parts.push(`MARK${mk.pauli}(${mk.k})${mk.qubit}`);
    parts.push('TICK');
  };
  circuit.ops.forEach((op, k) => {
    const marks = byIndex.get(k);
    if (marks) emit(marks);
    parts.push(formatCrumbleOp(op));
  });
  const marks = byIndex.get(circuit.ops.length);
  if (marks) emit(marks);
  return CRUMBLE_PREFIX + parts.join(';') + '_';
}

export function formatCrumbleOp(op) {
  let out = SHORT[op.name] ?? op.name;
  if (op.args.length) out += `(${op.args.map(formatArg).join(',')})`;
  let text = '';
  for (let k = 0; k < op.targets.length; k++) {
    const t = op.targets[k];
    if (t === '*') {
      text += '*';
      continue;
    }
    if (k > 0 && op.targets[k - 1] !== '*') text += '_';
    else if (k === 0 && !op.args.length) text += '_';
    text += t;
  }
  return out + text;
}

/** A Crumble URL (or its `#circuit=` fragment) to a circuit and its markers. */
export function parseCrumbleUrl(url) {
  let body = url.includes('#circuit=') ? url.slice(url.indexOf('#circuit=') + 9) : url;
  body = decodeURIComponent(body);
  if (body.endsWith('_')) body = body.slice(0, -1);
  const ops = [];
  const markers = [];
  const items = body.split(';');
  const depth = [];
  for (const raw of items) {
    const item = raw.trim();
    if (!item.length) continue;
    const rep = item.match(/^REPEAT_(\d+)_\{$/);
    if (rep) {
      if (Number(rep[1]) < 1) throw new Error(`REPEAT ${rep[1]}: a block repeats at least once`);
      depth.push({ count: Number(rep[1]), start: ops.length });
      continue;
    }
    if (item === '}') {
      const r = depth.pop();
      if (!r) throw new Error('"}" closes no REPEAT block');
      const inner = ops.splice(r.start);
      repeatInto(ops, inner, r.count);
      continue;
    }
    const mk = item.match(/^MARK([XYZ])\((\d+)\)(\d+)$/);
    if (mk) {
      markers.push({ k: Number(mk[2]), pauli: mk[1], qubit: Number(mk[3]), before: ops.length });
      continue;
    }
    if (item.startsWith('POLYGON')) continue;
    const m = item.match(/^([A-Za-z][A-Za-z0-9_]*?)(\(([^)]*)\))(.*)$/) ?? splitBare(item);
    if (!m) throw new Error(`cannot read "${item}"`);
    const name = LONG[m[1]] ?? canonicalName(m[1]);
    const args = m[3] ? parseArgs(m[3], item) : [];
    let rest = m[4];
    if (rest.startsWith('_')) rest = rest.slice(1);
    const targets = rest.length ? splitTargets(rest.split('_').filter((s) => s.length)) : [];
    for (const t of targets) if (!TARGET.test(t)) throw new Error(`"${item}": "${t}" is not a target`);
    ops.push({ name, args, targets });
    if (ops.length > MAX_OPS) throw new Error(`more than ${MAX_OPS} operations`);
  }
  if (depth.length) throw new Error('a REPEAT block is never closed');
  // Marks between TICK;…;TICK pairs: drop the TICKs Stim adds around marks when they wrap nothing else.
  return { circuit: { ops: stripMarkTicks(ops, markers) }, markers };
}

/**
 * An item without arguments, `NAME_t1_t2`: the name is the longest run of leading underscore
 * parts made of letters only (`S_DAG`, `SQRT_X_DAG`, `H_YZ`, `X_ERROR`); a target starts with a
 * digit, `!`, a Pauli letter and a digit, `rec[`, `sweep[` or `*`.
 */
function splitBare(item) {
  const parts = item.split('_');
  let n = 1;
  while (n < parts.length && /^[A-Za-z]+$/.test(parts[n])) n++;
  return [item, parts.slice(0, n).join('_'), undefined, undefined, parts.slice(n).length ? '_' + parts.slice(n).join('_') : ''];
}

function stripMarkTicks(ops, markers) {
  // Stim writes `TICK;MARK…;TICK` around marks; a mark's `before` index then points between the
  // two TICKs. Remove such TICK pairs and re-aim the marks at the next real op (a TICK among
  // them: the mark then sits at the end of the layer that TICK closes).
  const out = [];
  const marksBefore = new Map();
  for (const m of markers) {
    if (!marksBefore.has(m.before)) marksBefore.set(m.before, []);
    marksBefore.get(m.before).push(m);
  }
  const newIndex = new Array(ops.length + 1).fill(-1);
  for (let k = 0; k < ops.length; k++) {
    const op = ops[k];
    if (op.name === 'TICK' && marksBefore.has(k + 1) && ops[k + 1]?.name === 'TICK') {
      newIndex[k] = out.length;
      newIndex[k + 1] = out.length;
      k += 1;
      continue;
    }
    newIndex[k] = out.length;
    out.push(op);
  }
  newIndex[ops.length] = out.length;
  for (const m of markers) m.before = newIndex[m.before] >= 0 ? newIndex[m.before] : out.length;
  return out;
}
