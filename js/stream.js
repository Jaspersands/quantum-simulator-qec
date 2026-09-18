/**
 * Chunking for a streamed Monte Carlo run: about `target` reports over the
 * whole run, never a chunk smaller than `min` shots (a report costs a message
 * and a repaint, and a 20-shot estimate tells nobody anything), and the last
 * chunk takes whatever is left.
 */
export function planChunks(total, { target = 60, min = 250 } = {}) {
  const size = Math.max(min, Math.ceil(total / target));
  const out = [];
  for (let done = 0; done < total; done += size) out.push(Math.min(size, total - done));
  return out;
}
