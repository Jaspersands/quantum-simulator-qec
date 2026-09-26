/**
 * Names and number formats for section 11 and tools/lambda.mjs, kept pure so
 * both can use them and the tests can check them.
 */

const PRIOR = {
  si1000: "Google's SI1000 prior",
  rl: "Google's RL-optimised prior",
  ours: 'our model of the noisy circuit',
  circuit: "the circuit's model",
  pij: 'the data-fitted pij models',
};

const GOOGLE = {
  correlated_matching_decoder_with_si1000_prior: 'correlated matching, SI1000 prior',
  correlated_matching_decoder_with_rl_optimized_prior: 'correlated matching, RL-optimised prior',
  harmony_decoder_with_si1000_prior: 'Harmony, SI1000 prior',
  harmony_decoder_with_rl_optimized_prior: 'Harmony, RL-optimised prior',
  libra_decoder_with_rl_optimized_prior: 'Libra, RL-optimised prior',
  pymatching: 'PyMatching',
  correlated_matching: 'correlated matching',
  belief_matching: 'belief matching',
  tensor_network_contraction: 'tensor-network contraction',
};

/** "ours/si1000/correlated" → "ours, correlated · Google's SI1000 prior"; "google/x" → "Google: …". */
export function decoderLabel(key) {
  const [who, a, b] = key.split('/');
  if (who === 'ours') return `ours, ${b === 'belief' ? 'belief-matching' : b} · ${PRIOR[a] ?? a}`;
  return `Google: ${GOOGLE[a] ?? a}`;
}

export const isOurs = (key) => key.startsWith('ours/');

/** 0.00143 → "0.143%". */
export function percent(x, digits = 3) {
  return Number.isFinite(x) ? `${(x * 100).toFixed(digits)}%` : '—';
}

/** [0.00140, 0.00146] → "0.140–0.146%". */
export function percentRange([lo, hi], digits = 3) {
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return '—';
  return `${(lo * 100).toFixed(digits)}–${(hi * 100).toFixed(digits)}%`;
}

/** 2.1374 → "2.14"; with an interval, "2.14 [2.11, 2.17]". */
export function ratio(x, range) {
  if (!Number.isFinite(x)) return '—';
  const r = range && range.every(Number.isFinite) ? ` [${range[0].toFixed(2)}, ${range[1].toFixed(2)}]` : '';
  return `${x.toFixed(2)}${r}`;
}
