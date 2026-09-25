/**
 * What the threshold sweep measures, per noise model: distances, the window of
 * physical error rates, and shots per point. Section 07 and tools/sweep.mjs
 * (which produces the README's figures) both read it from here, so the page and
 * the quoted numbers cannot drift apart.
 */

import { NOISE } from './engine.js';

/**
 * Distances swept, per noise model.
 *
 * Three distances is not enough. The collapse ansatz is the d -> infinity limit,
 * and these patches are small, so a fit that ignores the approach to that limit
 * comes out biased — measured against synthetic data with a known threshold, by
 * +27% with three distances, and it does not improve with more shots because it
 * is bias, not noise. Separating the correction from a shift in the threshold
 * needs a fourth distance; a fifth roughly halves the residual error again. See
 * `fitThreshold`.
 */
export const DISTANCES = {
  [NOISE.DATA]: [3, 5, 7, 9, 11],
  [NOISE.PHENOM]: [3, 5, 7, 9],
  [NOISE.CIRCUIT]: [3, 5, 7, 9],
  [NOISE.SD6]: [3, 5, 7, 9],
};

/**
 * The two noise models have thresholds an order of magnitude apart, so a single
 * sweep range would waste most of its points saturated at one end or the other.
 */
export const SWEEP_PS = {
  // Reaches to 18%: the crossing sits near 12.5%, and a sweep with only one
  // rate above it brackets the threshold too thinly for the collapse to pin
  // down — the fit gets dragged toward the crowded low side.
  [NOISE.DATA]: [0.02, 0.05, 0.08, 0.10, 0.11, 0.12, 0.13, 0.15, 0.18],
  [NOISE.PHENOM]: [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.045, 0.06],
  // Circuit-level threshold sits an order of magnitude lower again: every gate
  // in the extraction circuit is a fault location, so a given per-gate rate does
  // far more damage than the same number applied once per round. The window is
  // centred on the crossing near 0.36% rather than started near zero — points
  // where every distance reads 0.00% cost as much to measure as any other and
  // tell the fit nothing.
  [NOISE.CIRCUIT]: [0.0015, 0.0022, 0.0028, 0.0032, 0.0036, 0.0040, 0.0046, 0.0055, 0.0070],
  // SD6 per basis: the distances swap order near 0.45% (measured at 2,000 shots
  // a point: at 0.4% every larger patch is still better, at 0.5% d = 3 is
  // already the best), so the window straddles that with room either side.
  [NOISE.SD6]: [0.0025, 0.0030, 0.0035, 0.0040, 0.0045, 0.0050, 0.0055, 0.0060, 0.0070],
};

/**
 * Every sweep gets the same shot count.
 *
 * Circuit-level used to be cut to 0.25x for being the slowest model, which left
 * ~300 shots on each point while the rates being separated were around 1% — three
 * or four events per point, not enough to show the distances swap order at all.
 * Now that its window no longer spends half its points where every distance
 * reads 0.00%, the full count costs about the same wall-clock as it used to and
 * actually resolves the crossing.
 */
/**
 * Shots per point, per noise model — set by what each can afford, not by taste.
 *
 * Precision here is shot-limited rather than method-limited: on synthetic data
 * the corrected fit's rms error falls from 40% at 1,200 shots to 13% at 20,000.
 * Data noise is cheap enough to buy that outright. The other two are not, and
 * their sweeps are correspondingly less precise — which the reported interval
 * shows rather than hides. Wall-clock is around 20s, 25s and 80s.
 */
export const SWEEP_RUNS = {
  [NOISE.DATA]: 20000,
  [NOISE.PHENOM]: 6000,
  [NOISE.CIRCUIT]: 1200,
  // SD6 is decoded by exact matching on the dense general-path decoder, and d = 9
  // costs 3 ms a shot at the bottom of the window and 30 ms at the top, about
  // 0.1 s a shot summed across it. 800 shots keeps the sweep near the
  // circuit-level one's wall-clock; the interval says what that buys.
  [NOISE.SD6]: 800,
};
