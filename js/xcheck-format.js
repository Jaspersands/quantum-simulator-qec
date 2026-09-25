/**
 * Pure formatting for Figure 8. No DOM, no engine: everything here is tested
 * in Node by tools/site-tests.mjs.
 */

const SUP = { '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };

export function superscript(n) {
  return String(n).split('').map((c) => SUP[c]).join('');
}

/** A relative difference, as a power of ten. */
export function formatRel(x) {
  if (!Number.isFinite(x)) return '—';
  if (x === 0) return '0';
  const e = Math.floor(Math.log10(x));
  const m = x / 10 ** e;
  return `${m.toFixed(1)} × 10${superscript(e)}`;
}

/** What a comparison row says, and in which tone. */
export function verdict(r) {
  if (!r) return { text: 'queued', tone: 'idle' };
  if (r.pending) return { text: 'deriving…', tone: 'idle' };
  if (!r.ok) {
    return r.unavailable
      ? { text: 'reference unavailable', tone: 'idle' }
      : { text: `engine error: ${r.error}`, tone: 'fail' };
  }
  if (r.stale) return { text: 'circuit differs from the recorded one', tone: 'fail' };
  const bad = r.missing + r.extra + r.differing;
  return bad === 0 ? { text: 'identical', tone: 'ok' } : { text: `${bad} differ`, tone: 'fail' };
}

export function circuitLabel(c) {
  if (c.source === 'stim') return `Stim’s own · d = ${c.d}`;
  const code = c.code === 'xzzx' ? 'XZZX' : 'rotated';
  const noise = c.noise === 'sd6' ? 'SD6' : 'engine’s model';
  return `${code} · ${noise} · d = ${c.d}`;
}

export function microseconds(us) {
  if (!Number.isFinite(us)) return '—';
  if (us >= 1000) return `${(us / 1000).toFixed(1)} ms`;
  if (us >= 10) return `${Math.round(us)} µs`;
  return `${us.toFixed(1)} µs`;
}

export function disagreementText(rec) {
  if (rec.disagreements === 0) return 'none';
  const n = rec.disagreements.toLocaleString('en-US');
  return rec.non_ties === 0 ? `${n} · all ties` : `${n} · ${rec.non_ties} not ties`;
}

export function megabytes(bytes) {
  return `${(bytes / 1e6).toFixed(1)} MB`;
}
