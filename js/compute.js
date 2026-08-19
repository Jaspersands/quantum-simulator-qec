/**
 * Client side of the Monte Carlo worker, plus the statistics that turn raw
 * sample counts into something honest to display.
 */

/* -- Worker RPC --------------------------------------------------------- */

const WORKER_URL = new URL('./worker.js', import.meta.url);

export class Compute {
  constructor() {
    this.worker = new Worker(WORKER_URL, { type: 'module' });
    this.pending = new Map();
    this.nextId = 1;
    /** Set once the worker is unusable; see #failAll. */
    this.dead = null;
    this.worker.onmessage = (event) => this.#receive(event.data);
    this.worker.onerror = (event) => this.#failAll(event.message || 'the worker crashed');
  }

  #receive({ id, type, result, message, ...progress }) {
    const job = this.pending.get(id);
    if (!job) return;
    if (type === 'progress') {
      job.onProgress?.(progress);
    } else if (type === 'done') {
      this.pending.delete(id);
      job.resolve(result);
    } else {
      this.pending.delete(id);
      job.reject(new Error(message));
    }
  }

  /**
   * The worker is gone. Reject everything outstanding, and remember why, so
   * later calls fail immediately instead of waiting on a reply that can never
   * arrive — a silent hang is the one failure mode with no visible symptom.
   */
  #failAll(message) {
    this.dead = message;
    for (const [, job] of this.pending) job.reject(new Error(message));
    this.pending.clear();
  }

  /**
   * @param {string} op one of the worker's operations
   * @param {object} [payload]
   * @param {(progress: object) => void} [onProgress]
   */
  call(op, payload, onProgress) {
    if (this.dead) return Promise.reject(new Error(this.dead));
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject, onProgress });
      this.worker.postMessage({ id, op, payload });
    });
  }
}

/* -- Statistics --------------------------------------------------------- */

/**
 * Wilson score interval for a binomial proportion.
 *
 * Preferred over the normal approximation here because logical error rates run
 * very close to 0 at low p, where the normal interval famously produces
 * negative lower bounds and zero width at p̂ = 0.
 *
 * @param {number} rate observed proportion
 * @param {number} n number of trials
 * @param {number} [z] 1.96 for 95%
 */
export function wilson(rate, n, z = 1.96) {
  if (!n) return { lo: 0, hi: 1 };
  const k = Math.round(rate * n);
  const phat = k / n;
  const z2 = z * z;
  const denom = 1 + z2 / n;
  const centre = phat + z2 / (2 * n);
  const spread = z * Math.sqrt((phat * (1 - phat) + z2 / (4 * n)) / n);
  return {
    lo: Math.max(0, (centre - spread) / denom),
    hi: Math.min(1, (centre + spread) / denom),
  };
}

/**
 * Variance of an observed rate, smoothed so that p̂ = 0 is not treated as certain.
 *
 * The naive p̂(1-p̂)/n collapses to zero when no failure is seen, which in a
 * weighted fit means infinite confidence. Flooring it at 1/(4n²) — the previous
 * approach — swaps infinity for merely enormous: at n = 1200 a zero-failure
 * point outweighed a 5% point 228 to 1, and in the phenomenological sweep two
 * points out of twenty-seven carried 84% of the total weight. Jeffreys' prior
 * adds half a success and half a failure, which is the standard fix and leaves
 * the variance finite and smooth through zero.
 */
function rateVariance(pL, runs) {
  const smoothed = (pL * runs + 0.5) / (runs + 1);
  return (smoothed * (1 - smoothed)) / (runs + 1);
}

/* -- Finite-size scaling fit -------------------------------------------- */

/** Solve a square linear system by Gaussian elimination with partial pivoting. */
function solve(matrix, rhs) {
  const n = rhs.length;
  const m = matrix.map((row, i) => [...row, rhs[i]]);
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let r = col + 1; r < n; r++) {
      if (Math.abs(m[r][col]) > Math.abs(m[pivot][col])) pivot = r;
    }
    if (Math.abs(m[pivot][col]) < 1e-14) return null;
    [m[col], m[pivot]] = [m[pivot], m[col]];
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const factor = m[r][col] / m[col][col];
      for (let c = col; c <= n; c++) m[r][c] -= factor * m[col][c];
    }
  }
  return m.map((row, i) => row[n] / row[i]);
}

/**
 * Weighted least squares of p_L against a scaling basis, for one candidate
 * (p_th, ν, ω).
 *
 * The basis is [1, x, x², d^(-ω)] with x = (p - p_th)·d^(1/ν). The first three
 * terms are the usual collapse; the fourth is the leading correction to scaling.
 * Dropping it is what biases a threshold fitted from small patches — the
 * collapse ansatz is the d → ∞ limit, and at d = 3 the approach to that limit is
 * not a small effect. Passing omega = null fits the uncorrected form, which the
 * page shows alongside so the size of the correction is visible rather than
 * asserted.
 */
function collapseAt(points, pTh, nu, omega) {
  const terms = omega === null ? 3 : 4;
  const M = Array.from({ length: terms }, () => new Array(terms).fill(0));
  const v = new Array(terms).fill(0);

  for (const pt of points) {
    const x = (pt.p - pTh) * Math.pow(pt.d, 1 / nu);
    const w = 1 / rateVariance(pt.pL, pt.runs);
    const basis = omega === null
      ? [1, x, x * x]
      : [1, x, x * x, Math.pow(pt.d, -omega)];
    for (let i = 0; i < terms; i++) {
      for (let j = 0; j < terms; j++) M[i][j] += w * basis[i] * basis[j];
      v[i] += w * basis[i] * pt.pL;
    }
  }

  const coeffs = solve(M, v);
  if (!coeffs) return null;

  let chi2 = 0;
  for (const pt of points) {
    const x = (pt.p - pTh) * Math.pow(pt.d, 1 / nu);
    const basis = omega === null
      ? [1, x, x * x]
      : [1, x, x * x, Math.pow(pt.d, -omega)];
    const model = basis.reduce((a, b, i) => a + b * coeffs[i], 0);
    chi2 += ((pt.pL - model) ** 2) / rateVariance(pt.pL, pt.runs);
  }
  return { coeffs, chi2 };
}

/**
 * Where two distances' curves cross, by linear interpolation between the
 * measured rates either side of the crossing.
 *
 * This is the threshold's definition rather than a model of it, which is what
 * makes it worth computing separately: it shares none of the collapse fit's
 * assumptions. It has its own bias — a crossing between two finite patches sits
 * away from the true threshold by an amount that shrinks as the patches grow —
 * but that bias is a known function of d, so several crossings can be
 * extrapolated to infinite size.
 */
function crossings(points) {
  const byD = new Map();
  for (const pt of points) {
    if (!byD.has(pt.d)) byD.set(pt.d, new Map());
    byD.get(pt.d).set(pt.p, pt.pL);
  }
  const ds = [...byD.keys()].sort((a, b) => a - b);
  const out = [];

  for (let i = 0; i < ds.length; i++) {
    for (let j = i + 1; j < ds.length; j++) {
      const a = byD.get(ds[i]);
      const b = byD.get(ds[j]);
      const shared = [...a.keys()].filter((p) => b.has(p)).sort((x, y) => x - y);
      for (let k = 1; k < shared.length; k++) {
        const p0 = shared[k - 1];
        const p1 = shared[k];
        // Below threshold the larger patch wins, above it loses. The crossing is
        // where that difference changes sign.
        const g0 = b.get(p0) - a.get(p0);
        const g1 = b.get(p1) - a.get(p1);
        if (g0 < 0 && g1 > 0) {
          const t = -g0 / (g1 - g0);
          out.push({ small: ds[i], large: ds[j], p: p0 + t * (p1 - p0) });
          break;
        }
      }
    }
  }
  return out;
}

/** Grid search over the non-linear parameters, then refine around the winner. */
function search(data, withCorrection, window) {
  // The threshold lies inside the swept range — the bracketing guard has already
  // established that the distances swap order somewhere within it. Letting the
  // search wander outside is how a noisy fit returns an answer nowhere near the
  // data: with the old 0.4x-to-1.6x box, a sweep topping out at 6% could report
  // a threshold of 9.6%.
  const ps = data.map((pt) => pt.p);
  const P_LO = Math.min(...ps);
  const P_HI = Math.max(...ps);
  const P_STEP = (P_HI - P_LO) / 90;
  // A correction that dies faster than about d^-6 is indistinguishable from one
  // that touches only the smallest patch, so the range stops being informative
  // rather than stopping arbitrarily. Hitting the top is reported, not hidden.
  const omegas = withCorrection
    ? [0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 6]
    : [null];

  let best = null;
  const scan = (pLo, pHi, pStep, nLo, nHi, nStep, oms) => {
    for (let pTh = pLo; pTh <= pHi; pTh += pStep) {
      for (let nu = nLo; nu <= nHi; nu += nStep) {
        for (const omega of oms) {
          const fit = collapseAt(data, pTh, nu, omega);
          if (fit && (!best || fit.chi2 < best.chi2)) best = { ...fit, pTh, nu, omega };
        }
      }
    }
  };
  scan(P_LO, P_HI, P_STEP, 0.5, 4.0, 0.2, omegas);
  if (!best) return null;
  const fine = P_STEP * 2;
  const near = best.omega === null
    ? [null]
    : omegas.filter((o) => Math.abs(o - best.omega) <= 1.0);
  scan(
    Math.max(P_LO, best.pTh - fine), best.pTh + fine, fine / 8,
    Math.max(0.3, best.nu - 0.2), best.nu + 0.2, 0.02,
    near,
  );
  best.atEdge = best.pTh <= P_LO + P_STEP || best.pTh >= P_HI - P_STEP;
  best.omegaAtEdge = best.omega !== null && best.omega >= omegas[omegas.length - 1];
  best.window = window;
  return best;
}

/**
 * A tight search around a known answer — for the bootstrap, which needs the
 * spread of the estimate rather than another exhaustive hunt for it.
 */
function localSearch(data, around, withCorrection) {
  const omega = withCorrection ? around.omega : null;
  const scan = (pLo, pHi, pStep, nLo, nHi, nStep) => {
    let best = null;
    for (let pTh = pLo; pTh <= pHi; pTh += pStep) {
      for (let nu = Math.max(0.25, nLo); nu <= nHi; nu += nStep) {
        const fit = collapseAt(data, pTh, nu, omega);
        if (fit && (!best || fit.chi2 < best.chi2)) best = { ...fit, pTh, nu, omega };
      }
    }
    return best;
  };
  // Coarse then fine. A single coarse pass quantizes the answer onto its own
  // grid, and when that grid is wider than the spread being measured, every
  // replica lands on the same cell and the interval collapses. Checked against
  // synthetic data: the one-pass version reported an interval that contained the
  // true threshold 5% of the time, which is worse than reporting none.
  const span = Math.max(1e-7, around.pTh * 0.3);
  const pStep = span / 10;
  const nStep = 0.08;
  const coarse = scan(around.pTh - span, around.pTh + span,
    pStep, around.nu - 0.8, around.nu + 0.8, nStep);
  if (!coarse) return null;
  return scan(coarse.pTh - pStep, coarse.pTh + pStep, pStep / 10,
    coarse.nu - nStep, coarse.nu + nStep, nStep / 8) ?? coarse;
}

/**
 * The collapse ansatz is an expansion about the threshold, so it only describes
 * points near it. Trim on the scaling variable itself rather than on p: |x| is
 * what the expansion is in, and it already folds in the distance.
 */
function trimToCriticalRegion(data, fit, keepFraction) {
  const scored = data
    .map((pt) => ({ pt, x: Math.abs((pt.p - fit.pTh) * Math.pow(pt.d, 1 / fit.nu)) }))
    .sort((a, b) => a.x - b.x);
  const keep = Math.max(12, Math.ceil(scored.length * keepFraction));
  const kept = scored.slice(0, keep).map((e) => e.pt);
  return new Set(kept.map((pt) => pt.d)).size >= 2 ? kept : data;
}

/** Reduced chi-squared for a fit over a given point set. */
function reduced(fit, n, withCorrection) {
  return fit.chi2 / Math.max(1, n - (withCorrection ? 6 : 5));
}

/** Fractions of the sweep to consider keeping, widest first. */
const WINDOWS = [1.0, 0.85, 0.75, 0.6, 0.5, 0.4];

/**
 * Choose how much of the sweep to fit, by whether the scaling form describes it.
 *
 * A fixed fraction cannot be right for every sweep, and picking one by hand is
 * how a number becomes an artefact of its author. Measured across the three
 * noise models, the same hard-coded 75% gave a reduced chi-squared of 2.3 for
 * data noise, 13.2 for phenomenological and 0.8 for circuit-level — that is, the
 * middle one was being fitted in a regime where the model is rejected outright,
 * and a rejected model does not have meaningful parameters. It also drives the
 * pathology this replaced: with the shape wrong, the correction exponent runs to
 * whatever extreme best absorbs the misfit, and pins at the end of its range.
 *
 * So: take the widest window the scaling form actually fits. Points far from
 * threshold are informative when the ansatz reaches them and misleading when it
 * does not, and reduced chi-squared is exactly the statistic that says which.
 */
function selectWindow(all, withCorrection, seed) {
  let fallback = null;
  let chosen = null;
  for (const fraction of WINDOWS) {
    const data = trimToCriticalRegion(all, seed, fraction);
    if (new Set(data.map((pt) => pt.d)).size < (withCorrection ? 4 : 2)) continue;
    const fit = search(data, withCorrection, data);
    if (!fit) continue;
    const red = reduced(fit, data.length, withCorrection);
    if (!fallback || red < fallback.red) fallback = { fit, data, red, fraction };
    // Widest first, so the first acceptable one wins.
    if (!chosen && red <= 2.0) chosen = { fit, data, red, fraction };
  }
  const winner = chosen ?? fallback;
  if (!winner) return null;
  return { ...winner.fit, window: winner.data, windowFraction: winner.fraction,
           windowAccepted: chosen !== null };
}

/** One binomial draw, Poisson-approximated in the small-count tail. */
function resampleRate(pL, runs) {
  const lambda = pL * runs;
  if (lambda < 25) {
    // Knuth. Exact enough where the normal approximation is worst — which is
    // exactly where these rates live below threshold.
    const limit = Math.exp(-lambda);
    let k = 0;
    let prod = Math.random();
    while (prod > limit && k < runs) { k += 1; prod *= Math.random(); }
    return Math.min(1, k / runs);
  }
  const sd = Math.sqrt(runs * pL * (1 - pL));
  let u = 0;
  for (let i = 0; i < 12; i++) u += Math.random();
  const k = Math.round(lambda + (u - 6) * sd);
  return Math.min(1, Math.max(0, k / runs));
}

/**
 * Fit the threshold by universal collapse, with the leading correction to
 * scaling.
 *
 * Near threshold every distance's logical error rate collapses onto one curve
 * when plotted against x = (p - p_th)·d^(1/ν). That statement is the d → ∞
 * limit, though, and these patches are small: at d = 3 the approach to the limit
 * is not a rounding error. Fitting the bare collapse gives a threshold pulled
 * systematically low — measured here at 5–13% below the value the crossings
 * point to, with a perfectly respectable chi-squared and no warning. Adding the
 * leading correction, `+ D·d^(-ω)`, is the standard remedy and is what Wang,
 * Harrington and Preskill used on this same code for this same reason.
 *
 * Both fits are returned. The uncorrected one is not there for decoration: the
 * gap between them is the size of the finite-size effect, which is worth seeing
 * rather than taking on trust. The independent crossing extrapolation is
 * returned too, and the three agreeing is the check that any of them is right.
 *
 * @param {Array<{d:number,p:number,pL:number,runs:number}>} points
 * @param {{bootstrap?: number}} [options] resamples for the confidence interval
 */
export function fitThreshold(points, options = {}) {
  const { bootstrap = 0 } = options;
  const reject = (reason) => ({
    pTh: NaN, nu: NaN, omega: NaN, coeffs: [0, 0, 0, 0],
    chi2: Infinity, reducedChi2: Infinity, ok: false, reason,
  });

  const usable = points.filter((pt) => Number.isFinite(pt.pL) && pt.runs > 0);
  if (usable.length < 8) return reject('not enough points to fit');

  const distances = new Set(usable.map((pt) => pt.d));
  if (distances.size < 2) {
    return reject('a collapse needs at least two code distances');
  }

  // A collapse fit is only meaningful when the data actually varies. If every
  // point sits at the same logical error rate, the model fits perfectly for
  // EVERY candidate and the search returns whichever cell it visited first —
  // a number pinned to the edge of the box, reported with a perfect goodness of
  // fit. Refuse instead.
  const rates = usable.map((pt) => pt.pL);
  const resolution = Math.min(...usable.map((pt) => 1 / pt.runs));
  if (Math.max(...rates) - Math.min(...rates) <= resolution) {
    return reject('every point returned the same logical error rate — the sweep '
      + 'carries no signal to fit. Widen the range of p, or raise the shot count.');
  }

  // A threshold is where the distances swap order, so a sweep that never shows
  // them swap has not bracketed one. The collapse will still converge on such
  // data — on a value pulled toward whichever side is populated, with a
  // respectable chi-squared and no hint anything is wrong. Measured directly: a
  // phenomenological sweep stopping at p = 3.6% fits 2.85% where the crossing is
  // near 3.1%. Repeating that measurement reproduces the bias rather than
  // exposing it, so the check has to be structural — and the honest one is the
  // physical definition, not a margin around p_th.
  const byP = new Map();
  for (const pt of usable) {
    if (!byP.has(pt.p)) byP.set(pt.p, []);
    byP.get(pt.p).push(pt);
  }
  const ordered = [...byP.entries()]
    .filter(([, pts]) => new Set(pts.map((q) => q.d)).size >= 2)
    .sort((a, b) => a[0] - b[0]);
  if (ordered.length >= 2) {
    const gaps = ordered.map(([, pts]) => {
      const lo = pts.reduce((a, q) => (q.d < a.d ? q : a));
      const hi = pts.reduce((a, q) => (q.d > a.d ? q : a));
      return hi.pL - lo.pL; // negative = the larger patch is winning
    });
    const firstAhead = gaps.findIndex((g) => g < 0);
    let lastBehind = -1;
    gaps.forEach((g, i) => { if (g > 0) lastBehind = i; });
    if (firstAhead === -1 || lastBehind === -1 || firstAhead >= lastBehind) {
      return reject('the sweep never shows the distances swap order, so it does '
        + 'not bracket a threshold: the largest patch '
        + (firstAhead === -1 ? 'is never ahead of the smallest'
           : lastBehind === -1 ? 'is ahead throughout and never falls behind'
           : 'falls behind before it is ever ahead')
        + '. Widen the range of p until the curves visibly cross.');
    }
  }

  // Locate the region, trim to it, refit. The trim is on |x| — the variable the
  // expansion is actually in — rather than on distance from p_th in p.
  const converge = (withCorrection) => {
    // Locate the region roughly, then let the data choose how much of it to fit.
    let seed = search(usable, withCorrection, usable);
    if (!seed) return null;
    let picked = selectWindow(usable, withCorrection, seed);
    // The window depends on where the threshold is and vice versa, so settle it.
    for (let pass = 0; pass < 2 && picked; pass++) {
      const again = selectWindow(usable, withCorrection, picked);
      if (!again) break;
      const settled = Math.abs(again.pTh - picked.pTh) < 1e-5;
      picked = again;
      if (settled) break;
    }
    return picked;
  };

  const leading = converge(false);
  // The correction term needs distinct values of d^(-omega) to be separable from
  // a shift in p_th. With three distances it is not identifiable and the fit is
  // worse than leaving it out: measured bias +16% against +35%, but with rms 51%
  // against 41% — trading a known bias for noise. Four is the minimum worth
  // trying.
  const corrected = distances.size >= 4 ? converge(true) : null;
  const best = corrected ?? leading;
  if (!best) return reject('the fit did not converge');

  // A winner sitting on the edge of the search box was not located by the data;
  // the search simply ran out of room.
  if (best.atEdge) {
    return reject('the best fit ran into the edge of the search range, so the '
      + 'threshold is not pinned down by this data');
  }

  // The raw crossings, for display. Extrapolating them to infinite d was tried
  // and dropped: over 40 synthetic realizations it came out biased by -35% with
  // an rms error of 49%, against -2% and 13% for the corrected fit. Three
  // crossings and three parameters is an exact fit, so the drift exponent is
  // free to run away and take the intercept with it. They are still worth
  // showing — they are where the curves visibly cross, which is the threshold's
  // definition — but they are not an estimate of it.
  const found = crossings(usable);

  // Confidence interval by resampling the shots, not by the spread of repeats:
  // repeating a measurement whose bias lives in the sweep window reproduces the
  // bias, so a spread over repeats understates the real uncertainty.
  let pThLo = NaN;
  let pThHi = NaN;
  let nuLo = NaN;
  let nuHi = NaN;
  if (bootstrap > 0) {
    // Resample every point of the full sweep, then let each replica pick its own
    // window from the neighbours of the chosen one. Holding the window fixed
    // would measure only the shot noise and call it the uncertainty, when the
    // choice of window moved the answer by more than the shots did.
    const here = WINDOWS.indexOf(best.windowFraction ?? 1);
    const nearby = WINDOWS.filter((_, i) => Math.abs(i - here) <= 1);
    const draws = [];
    for (let b = 0; b < bootstrap; b++) {
      const resampled = usable.map((pt) => ({
        ...pt, pL: resampleRate(pt.pL, pt.runs),
      }));
      let bestDraw = null;
      for (const fraction of nearby) {
        const data = trimToCriticalRegion(resampled, best, fraction);
        if (new Set(data.map((pt) => pt.d)).size < 2) continue;
        const again = localSearch(data, best, corrected !== null);
        if (!again) continue;
        const red = reduced(again, data.length, corrected !== null);
        if (!bestDraw || red < bestDraw.red) bestDraw = { pTh: again.pTh, nu: again.nu, red };
      }
      if (bestDraw) draws.push(bestDraw);
    }
    if (draws.length >= bootstrap * 0.6) {
      const ths = draws.map((d) => d.pTh).sort((a, b2) => a - b2);
      pThLo = ths[Math.floor(ths.length * 0.025)];
      pThHi = ths[Math.floor(ths.length * 0.975)];
      // nu gets an interval of its own because it is far more fragile than the
      // threshold, and fragile in a way that is invisible from a point value.
      // Checked against synthetic data with the exponent fixed in advance: for
      // data noise the fit recovers it to within 1% with a spread of 0.02, and
      // for circuit-level at the shot count that model can afford it returns
      // 0.63 for a true 1.46. A number that wrong needs to look wrong.
      const nus = draws.map((d) => d.nu).sort((a, b2) => a - b2);
      nuLo = nus[Math.floor(nus.length * 0.025)];
      nuHi = nus[Math.floor(nus.length * 0.975)];
    }
  }

  const params = corrected ? 6 : 5; // coefficients + p_th + nu (+ omega)
  const dof = Math.max(1, best.window.length - params);
  return {
    pTh: best.pTh,
    nu: best.nu,
    omega: best.omega,
    coeffs: best.coeffs,
    chi2: best.chi2,
    reducedChi2: best.chi2 / dof,
    pointsUsed: best.window.length,
    pointsTotal: usable.length,
    pThLo,
    pThHi,
    nuLo,
    nuHi,
    // Whether nu is worth quoting at all.
    //
    // It is far more fragile than the threshold, and it fails quietly. Fed
    // synthetic data with the exponent fixed at 1.46 in advance, this fit
    // returns it to within 1% from a data-noise sweep and returns 0.6 to 0.8
    // from a circuit-level one — and, importantly, reports a tight interval
    // while doing so. Its coverage of the true value was measured at 80% for
    // data noise, 85% for phenomenological and 46% for circuit-level, so the
    // interval cannot be trusted to police itself: the failure is bias, and a
    // bootstrap resamples around its own answer.
    //
    // What does predict it is how much data the sweep has. The three cases
    // above carry roughly 900k, 216k and 43k shots in total, so the gate is on
    // that, plus a sanity check that the interval is not absurd. The number is
    // not a guess dressed as a threshold — it is where the calibration above
    // stops working.
    nuDetermined: Number.isFinite(nuLo) && Number.isFinite(nuHi)
      && (nuHi - nuLo) / (2 * best.nu) < 0.5
      && usable.reduce((total, pt) => total + pt.runs, 0) >= 100000,
    corrected: corrected !== null,
    omegaAtEdge: best.omegaAtEdge === true,
    windowFraction: best.windowFraction ?? 1,
    windowAccepted: best.windowAccepted !== false,
    leading: leading ? { pTh: leading.pTh, nu: leading.nu } : null,
    crossings: found,
    ok: true,
  };
}


/** Evaluate a fitted collapse curve for one distance at one physical rate. */
export function collapseCurve(fit, d, p) {
  const x = (p - fit.pTh) * Math.pow(d, 1 / fit.nu);
  let y = fit.coeffs[0] + fit.coeffs[1] * x + fit.coeffs[2] * x * x;
  // The correction term is per-distance, which is the whole point of it: the
  // curves no longer lie on top of one another, they approach a common one.
  if (fit.corrected && fit.coeffs.length > 3) {
    y += fit.coeffs[3] * Math.pow(d, -fit.omega);
  }
  return Math.min(1, Math.max(0, y));
}

/* -- Formatting --------------------------------------------------------- */

export function percent(value, digits = 2) {
  if (!Number.isFinite(value)) return '—';
  return `${(value * 100).toFixed(digits)}%`;
}

export function count(value) {
  if (!Number.isFinite(value)) return '—';
  return value.toLocaleString('en-US');
}
