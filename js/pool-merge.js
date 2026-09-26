/**
 * How the worker pool splits a job and puts it back together. Pure, so the
 * site tests can check it: every result the pool combines is a sum over
 * independent shots, and combining must lose nothing.
 */

/** `total` shots split into `n` near-equal parts, none empty unless total < n. */
export function splitRuns(total, n) {
  const parts = Math.max(1, Math.min(n, total));
  const base = Math.floor(total / parts);
  return Array.from({ length: parts }, (_, i) => base + (i < total % parts ? 1 : 0));
}

/**
 * The bench's progress across sub-streams, each reporting its own running
 * totals: shots and failures add; the wall time is the slowest part's, since
 * the parts run at once.
 */
export function mergeStream(parts, total) {
  const live = parts.filter(Boolean);
  const sum = (k) => live.reduce((s, p) => s + (p[k] ?? 0), 0);
  return {
    done: sum('done'),
    total,
    failures: sum('failures'),
    decodeErrors: sum('decodeErrors'),
    seconds: live.reduce((m, p) => Math.max(m, p.seconds ?? 0), 0),
  };
}

/** The bench's final result from its sub-streams' results. */
export function mergeStreamResults(results) {
  const runs = results.reduce((s, r) => s + r.runs, 0);
  const failures = results.reduce((s, r) => s + r.failures, 0);
  const seconds = results.reduce((m, r) => Math.max(m, r.seconds), 0);
  return {
    runs,
    failures,
    rate: runs ? failures / runs : 0,
    seconds,
    decodeErrors: results.reduce((s, r) => s + (r.decodeErrors ?? 0), 0),
    runsPerSecond: seconds > 0 ? Math.round(runs / seconds) : 0,
    cancelled: results.some((r) => r.cancelled),
  };
}
