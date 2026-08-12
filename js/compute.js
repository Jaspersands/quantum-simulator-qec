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
 *            reducedChi2:number, ok:boolean}}
 */
export function fitThreshold(points) {
  const usable = points.filter((pt) => Number.isFinite(pt.pL) && pt.runs > 0);
  if (usable.length < 6) {
    return { pTh: NaN, nu: NaN, coeffs: [0, 0, 0], chi2: Infinity, reducedChi2: Infinity, ok: false };
  }

  let best = null;
  // Coarse pass, then refine around the winner.
  const search = (pLo, pHi, pStep, nLo, nHi, nStep) => {
    for (let pTh = pLo; pTh <= pHi; pTh += pStep) {
      for (let nu = nLo; nu <= nHi; nu += nStep) {
        const fit = collapseAt(usable, pTh, nu);
        if (fit && (!best || fit.chi2 < best.chi2)) best = { ...fit, pTh, nu };
      }
    }
  };

  // The nu range is deliberately generous. Clipping it would report a value
  // sitting on the boundary of the search rather than a fitted one.
  search(0.004, 0.20, 0.002, 0.5, 5.0, 0.1);
  if (best) {
    search(
      Math.max(0.001, best.pTh - 0.004), best.pTh + 0.004, 0.0004,
      Math.max(0.3, best.nu - 0.2), best.nu + 0.2, 0.02,
    );
  }

  if (!best) {
    return { pTh: NaN, nu: NaN, coeffs: [0, 0, 0], chi2: Infinity, reducedChi2: Infinity, ok: false };
  }

  const dof = Math.max(1, usable.length - 5); // 3 coefficients + p_th + nu
  return {
    pTh: best.pTh,
    nu: best.nu,
    coeffs: best.coeffs,
    chi2: best.chi2,
    reducedChi2: best.chi2 / dof,
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
