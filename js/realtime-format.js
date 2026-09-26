/**
 * Names and formats for section 12, kept pure so the site tests can check them.
 */

/** The fewest cores at which a stream keeps up, from {cores: {keeps_up}}; null if none do. */
export function cores(byWorkers) {
  const ks = Object.keys(byWorkers).map(Number).sort((a, b) => a - b);
  return ks.find((k) => byWorkers[k].keeps_up) ?? null;
}

/** "keeps up on 4 cores", "keeps up on one core", or why not. */
export function keepUpText(k, mode) {
  if (k === 1) return 'Keeps up on one core';
  if (k) return `Keeps up on ${k} cores`;
  return mode === 'sliding' ? 'Falls behind (one core by design)' : 'Falls behind on up to 16';
}

/** 63.2 → "63 µs"; 5.71 → "5.7 µs"; 0.84 → "0.84 µs". */
export function microseconds(x) {
  if (!Number.isFinite(x)) return '—';
  const digits = x >= 20 ? 0 : x >= 2 ? 1 : 2;
  return `${x.toFixed(digits)} µs`;
}

/** "window/parallel/Bd/correlated" → "parallel windows, correlated"; "global/plain" → "global, plain". */
export function windowLabel(key) {
  const parts = key.split('/');
  if (parts[0] === 'global') return `global, ${parts[1]}`;
  const buffer = { Bd: '', Bhalf: ', buffer d/2', B2d: ', buffer 2d' }[parts[2]] ?? `, ${parts[2]}`;
  return `${parts[1]} windows${buffer}, ${parts[3]}`;
}
