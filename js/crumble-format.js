/**
 * Stim text and Crumble URLs, in and out, for the site's editor (section 17).
 *
 * A circuit here is `{ ops }`: a flat list of `{ name, args, targets }` with targets as Stim
 * writes them (`0`, `!1`, `X2`, `rec[-1]`, `sweep[3]`, `*` the combiner). REPEAT blocks are
 * unrolled on the way in (the editor edits a flat circuit); tags are dropped.
 *
 * The Crumble URL is the one `Circuit.to_crumble_url` writes (Stim's `export_crumble_url`):
 * `#circuit=` then instructions joined by `;`, `QUBIT_COORDS` as `Q`, `DETECTOR` as `DT`,
 * `OBSERVABLE_INCLUDE` as `OI`, arguments in parentheses with integers bare and other numbers
 * as `%g`, targets joined by `_` (the first follows the arguments directly), a marker as
 * `MARKX(k)q` between TICKs, and a final `_`.
 */

export const CRUMBLE_PREFIX = 'https://algassert.com/crumble#circuit=';

const SHORT = { QUBIT_COORDS: 'Q', DETECTOR: 'DT', OBSERVABLE_INCLUDE: 'OI' };
const LONG = { Q: 'QUBIT_COORDS', DT: 'DETECTOR', OI: 'OBSERVABLE_INCLUDE' };

/** Stim's `%g`-style number, integers bare. */
export function formatArg(value) {
  if (Number.isInteger(value) && Math.abs(value) < 9.2e18) return String(value);
  // %g: six significant digits, trailing zeros dropped, exponent form outside 1e-4..1e6.
  const abs = Math.abs(value);
  if (abs !== 0 && (abs < 1e-4 || abs >= 1e6)) {
    let [m, e] = value.toExponential(5).split('e');
    m = m.replace(/\.?0+$/, '');
    const sign = e[0] === '-' ? '-' : '+';
    const digits = e.replace(/^[-+]/, '').padStart(2, '0');
    return `${m}e${sign}${digits}`;
  }
  return String(Number(value.toPrecision(6)));
}

/** Stim text to a flat circuit. Unknown instructions are kept as written. */
export function parseStim(text) {
  const ops = [];
  const lines = text.split(/\r?\n/);
  let i = 0;
  const block = (count) => {
    const body = [];
    for (; i < lines.length; i++) {
      const line = lines[i].trim();
      if (!line || line.startsWith('#')) continue;
      if (line.startsWith('}')) break;
      const m = line.match(/^REPEAT\s+(\d+)\s*\{/i);
      if (m) {
        i++;
        const inner = block(Number(m[1]));
        for (let k = 0; k < Number(m[1]); k++) body.push(...inner.map((op) => ({ ...op, targets: [...op.targets] })));
        continue;
      }
      body.push(parseInstruction(line));
    }
    return body;
  };
  const top = block(1);
  ops.push(...top);
  return { ops };
}

export function parseInstruction(line) {
  const noComment = line.split('#')[0].trim();
  const m = noComment.match(/^([A-Za-z_][A-Za-z0-9_]*)(\[[^\]]*\])?\s*(\(([^)]*)\))?\s*(.*)$/);
  if (!m) throw new Error(`cannot read "${line}"`);
  const name = m[1].toUpperCase();
  const args = m[4] ? m[4].split(',').map((s) => s.trim()).filter(Boolean).map(Number) : [];
  const targets = m[5] ? m[5].split(/\s+/).filter(Boolean) : [];
  return { name, args, targets };
}

export function toStim(circuit) {
  return circuit.ops.map((op) => formatInstruction(op)).join('\n');
}

export function formatInstruction(op) {
  let out = op.name;
  if (op.args.length) out += `(${op.args.map(formatArg).join(', ')})`;
  if (op.targets.length) out += ' ' + op.targets.join(' ');
  return out;
}

/**
 * The Crumble URL of a circuit, with `markers` as `[{ k, pauli, qubit, before }]` placed before
 * the op at index `before` (between two TICKs, as Stim writes marks).
 */
export function toCrumbleUrl(circuit, markers = []) {
  const parts = [];
  const byIndex = new Map();
  for (const mk of markers) {
    if (!byIndex.has(mk.before)) byIndex.set(mk.before, []);
    byIndex.get(mk.before).push(mk);
  }
  circuit.ops.forEach((op, k) => {
    const marks = byIndex.get(k);
    if (marks) {
      parts.push('TICK');
      for (const mk of marks) parts.push(`MARK${mk.pauli}(${mk.k})${mk.qubit}`);
      parts.push('TICK');
    }
    parts.push(formatCrumbleOp(op));
  });
  const marks = byIndex.get(circuit.ops.length);
  if (marks) {
    parts.push('TICK');
    for (const mk of marks) parts.push(`MARK${mk.pauli}(${mk.k})${mk.qubit}`);
    parts.push('TICK');
  }
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
  body = decodeURIComponent(body.replace(/\+/g, ' '));
  if (body.endsWith('_')) body = body.slice(0, -1);
  const ops = [];
  const markers = [];
  const items = body.split(';').filter((s) => s.length);
  let depth = [];
  for (const raw of items) {
    const item = raw.trim();
    const rep = item.match(/^REPEAT_(\d+)_\{$/);
    if (rep) {
      depth.push({ count: Number(rep[1]), start: ops.length });
      continue;
    }
    if (item === '}') {
      const r = depth.pop();
      if (r) {
        const body2 = ops.splice(r.start);
        for (let k = 0; k < r.count; k++) ops.push(...body2.map((op) => ({ ...op, targets: [...op.targets] })));
      }
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
    const name = LONG[m[1]] ?? m[1].toUpperCase();
    const args = m[3] ? m[3].split(',').filter(Boolean).map(Number) : [];
    let rest = m[4];
    if (rest.startsWith('_')) rest = rest.slice(1);
    const targets = [];
    if (rest.length) {
      for (const piece of rest.split('_')) {
        if (!piece.length) continue;
        const stars = piece.split('*');
        stars.forEach((s, j) => {
          if (j > 0) targets.push('*');
          if (s.length) targets.push(s);
        });
      }
    }
    ops.push({ name, args, targets });
  }
  // Marks before TICK;…;TICK pairs: drop the TICKs Stim adds around marks when they wrap nothing else.
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
  // two TICKs. Remove such empty TICK pairs and re-aim the marks at the next real op.
  const out = [];
  const marked = new Set(markers.map((m) => m.before));
  for (let k = 0; k < ops.length; k++) {
    const op = ops[k];
    if (op.name === 'TICK' && marked.has(k + 1) && ops[k + 1]?.name === 'TICK') {
      for (const m of markers) if (m.before === k + 1) m.before = out.length;
      k += 1;
      continue;
    }
    out.push(op);
  }
  // Marks that pointed past removed ticks: clamp.
  for (const m of markers) if (m.before > out.length) m.before = out.length;
  return out;
}
