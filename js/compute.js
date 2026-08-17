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
    this.worker.onmessage = (event) => this.#receive(event.data);
    this.worker.onerror = (event) => this.#failAll(event.message || 'worker crashed');
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

  #failAll(message) {
    for (const [, job] of this.pending) job.reject(new Error(message));
    this.pending.clear();
  }

  /**
   * @param {string} op one of the worker's operations
   * @param {object} [payload]
   * @param {(progress: object) => void} [onProgress]
   */
  call(op, payload, onProgress) {
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

/** Variance of an observed rate, floored so that p̂ = 0 still carries weight. */
function rateVariance(pL, runs) {
  const v = (pL * (1 - pL)) / runs;
  return Math.max(v, 1 / (4 * runs * runs));
}

/* -- Finite-size scaling fit -------------------------------------------- */

/** Solve a 3x3 linear system by Gaussian elimination with partial pivoting. */
function solve3(matrix, rhs) {
  const m = matrix.map((row, i) => [...row, rhs[i]]);
  for (let col = 0; col < 3; col++) {
    let pivot = col;
    for (let r = col + 1; r < 3; r++) {
      if (Math.abs(m[r][col]) > Math.abs(m[pivot][col])) pivot = r;
    }
    if (Math.abs(m[pivot][col]) < 1e-14) return null;
    [m[col], m[pivot]] = [m[pivot], m[col]];
    for (let r = 0; r < 3; r++) {
      if (r === col) continue;
      const factor = m[r][col] / m[col][col];
      for (let c = col; c < 4; c++) m[r][c] -= factor * m[col][c];
    }
  }
  return [m[0][3] / m[0][0], m[1][3] / m[1][1], m[2][3] / m[2][2]];
}

/**
 * Weighted quadratic least squares of p_L against the scaling variable
 * x = (p - p_th) · d^(1/ν), for one candidate (p_th, ν).
 */
function collapseAt(points, pTh, nu) {
  const M = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  const v = [0, 0, 0];

  for (const pt of points) {
    const x = (pt.p - pTh) * Math.pow(pt.d, 1 / nu);
    const w = 1 / rateVariance(pt.pL, pt.runs);
    const basis = [1, x, x * x];
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) M[i][j] += w * basis[i] * basis[j];
      v[i] += w * basis[i] * pt.pL;
    }
  }

  const coeffs = solve3(M, v);
  if (!coeffs) return null;

  let chi2 = 0;
  for (const pt of points) {
    const x = (pt.p - pTh) * Math.pow(pt.d, 1 / nu);
    const model = coeffs[0] + coeffs[1] * x + coeffs[2] * x * x;
    chi2 += ((pt.pL - model) ** 2) / rateVariance(pt.pL, pt.runs);
  }
  return { coeffs, chi2 };
}

/**
 * Fit the threshold by universal collapse.
 *
 * Near threshold every distance's logical error rate collapses onto a single
 * curve when plotted against (p - p_th)·d^(1/ν). Grid-searching (p_th, ν) for
 * the pair that makes the collapse tightest is the standard way to read a
 * threshold off finite-size data.
 *
 * @param {Array<{d:number,p:number,pL:number,runs:number}>} points
 * @returns {{pTh:number, nu:number, coeffs:number[], chi2:number,
 *            reducedChi2:number, ok:boolean, reason?:string}}
 */
export function fitThreshold(points) {
  const reject = (reason) => ({
    pTh: NaN, nu: NaN, coeffs: [0, 0, 0],
    chi2: Infinity, reducedChi2: Infinity, ok: false, reason,
  });

  const usable = points.filter((pt) => Number.isFinite(pt.pL) && pt.runs > 0);
  if (usable.length < 6) return reject('not enough points to fit');

  if (new Set(usable.map((pt) => pt.d)).size < 2) {
    return reject('a collapse needs at least two code distances');
  }

  // A collapse fit is only meaningful when the data actually varies. If every
  // point sits at the same logical error rate — all zero below threshold, all
  // one above it, or genuinely flat — then the quadratic fits perfectly for
  // EVERY candidate (p_th, nu), chi-squared is zero everywhere, and the grid
  // search below would return whichever cell it happened to visit first. That
  // is a number pinned to the edge of the search box, reported with a perfect
  // goodness of fit. Refuse instead.
  const rates = usable.map((pt) => pt.pL);
  const spread = Math.max(...rates) - Math.min(...rates);
  const resolution = Math.min(...usable.map((pt) => 1 / pt.runs));
  if (spread <= resolution) {
    return reject('every point returned the same logical error rate — the sweep '
      + 'carries no signal to fit. Widen the range of p, or raise the shot count.');
  }

  const P_LO = 0.004, P_HI = 0.20, P_STEP = 0.002;

  /** Coarse grid then a refinement around the winner, over a given point set. */
  const fitOver = (data) => {
    let best = null;
    const search = (pLo, pHi, pStep, nLo, nHi, nStep) => {
      for (let pTh = pLo; pTh <= pHi; pTh += pStep) {
        for (let nu = nLo; nu <= nHi; nu += nStep) {
          const fit = collapseAt(data, pTh, nu);
          if (fit && (!best || fit.chi2 < best.chi2)) best = { ...fit, pTh, nu };
        }
      }
    };
    // The nu range is deliberately generous. Clipping it would report a value
    // sitting on the boundary of the search rather than a fitted one.
    search(P_LO, P_HI, P_STEP, 0.5, 5.0, 0.1);
    if (best) {
      search(
        Math.max(0.001, best.pTh - 0.004), best.pTh + 0.004, 0.0004,
        Math.max(0.3, best.nu - 0.2), best.nu + 0.2, 0.02,
      );
    }
    return best;
  };

  // The collapse ansatz only describes the critical region. Points far from
  // threshold do not obey it, and including them lets the fit buy a lower
  // chi-squared by inflating nu and dragging p_th along with it — a sweep from
  // 2% to 14% around a true threshold near 10.5% returned nu ~ 3.9, against a
  // physical value near 1.5. So: fit once to locate the region, then refit
  // using only the points inside it.
  let best = fitOver(usable);
  if (!best) return reject('the fit did not converge');

  // Keep a fixed fraction of the points, chosen as those closest to threshold
  // in relative distance. A fixed *width* window does not travel: at a data
  // noise threshold near 11% it trims sensibly, but at a phenomenological
  // threshold near 2.6% the same rule cuts down to the six-point minimum and
  // the exponent collapses. Ranking instead keeps the sample size stable
  // wherever the threshold turns out to be.
  const keep = Math.max(9, Math.ceil(usable.length * 0.6));
  let fitted = usable;
  for (let pass = 0; pass < 3 && keep < fitted.length; pass++) {
    const near = [...usable]
      .sort((a, b) => Math.abs(a.p - best.pTh) / best.pTh - Math.abs(b.p - best.pTh) / best.pTh)
      .slice(0, keep);
    if (new Set(near.map((pt) => pt.d)).size < 2) break;
    const refined = fitOver(near);
    if (!refined) break;
    const settled = Math.abs(refined.pTh - best.pTh) < 1e-4;
    best = refined;
    fitted = near;
    if (settled) break;
  }

  // A winner sitting on the edge of the search box was not located by the data;
  // the search simply ran out of room. Reporting it as a threshold would be
  // guessing.
  if (best.pTh <= P_LO + P_STEP || best.pTh >= P_HI - P_STEP) {
    return reject('the best fit ran into the edge of the search range, so the '
      + 'threshold is not pinned down by this data');
  }

  const dof = Math.max(1, fitted.length - 5); // 3 coefficients + p_th + nu
  return {
    pTh: best.pTh,
    nu: best.nu,
    coeffs: best.coeffs,
    chi2: best.chi2,
    reducedChi2: best.chi2 / dof,
    pointsUsed: fitted.length,
    pointsTotal: usable.length,
    ok: true,
  };
}

/** Evaluate a fitted collapse curve for one distance at one physical rate. */
export function collapseCurve(fit, d, p) {
  const x = (p - fit.pTh) * Math.pow(d, 1 / fit.nu);
  const y = fit.coeffs[0] + fit.coeffs[1] * x + fit.coeffs[2] * x * x;
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
