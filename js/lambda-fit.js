/**
 * Logical error per cycle, and Λ, from memory experiments run for several
 * numbers of rounds.
 *
 * A memory experiment prepares a logical state, runs r rounds of error
 * correction, measures, and asks the decoder whether the logical observable
 * flipped. Its logical fidelity F = 1 − 2 P_L decays geometrically,
 * F(r) = A (1 − 2ε)^r, and ε is the logical error per cycle. Λ is how much ε
 * falls each time the distance grows by two: ε_d ∝ Λ^(−d/2).
 *
 * One implementation serves the README (tools/lambda.mjs, in Node) and the page
 * (section 11, in the browser), and it treats this engine's predictions and
 * Google's identically, so the comparison between them is like for like.
 *
 * Records are those of data/google-results/*.json:
 * {d, patch, basis, rounds, shots, results: {key: {failures}}}.
 */

/** F = 1 − 2 P_L and its binomial standard error, kept off zero. */
export function fidelity(failures, shots) {
  const p = failures / shots;
  const q = Math.min(Math.max(p, 0.5 / shots), 1 - 0.5 / shots);
  return { F: 1 - 2 * p, sigma: 2 * Math.sqrt((q * (1 - q)) / shots) };
}

/**
 * ε for one patch and basis: a weighted least-squares line through (r, ln F),
 * each point weighted by (F / σ_F)², the inverse variance of ln F. Points
 * before `minRounds`, and points within three standard errors of F = 0 (where
 * the logarithm is noise), are left out.
 *
 * @param {Array<{rounds:number, failures:number, shots:number}>} points
 */
export function fitEpsilon(points, { minRounds = 1 } = {}) {
  const use = points
    .filter((p) => p.rounds >= minRounds)
    .map((p) => ({ r: p.rounds, ...fidelity(p.failures, p.shots) }))
    .filter((p) => p.F > 3 * p.sigma);
  if (use.length < 2) return { ok: false, n: use.length };
  let S = 0, Sx = 0, Sy = 0, Sxx = 0, Sxy = 0;
  for (const p of use) {
    const w = (p.F / p.sigma) ** 2;
    const y = Math.log(p.F);
    S += w; Sx += w * p.r; Sy += w * y; Sxx += w * p.r * p.r; Sxy += w * p.r * y;
  }
  const den = S * Sxx - Sx * Sx;
  if (!(den > 0)) return { ok: false, n: use.length };
  const slope = (S * Sxy - Sx * Sy) / den;
  const intercept = (Sy - slope * Sx) / S;
  return { ok: true, eps: (1 - Math.exp(slope)) / 2, A: Math.exp(intercept), n: use.length };
}

/** Every result key across the records, in first-seen order. */
export function decoderKeys(records) {
  const keys = new Set();
  for (const r of records) for (const k of Object.keys(r.results)) keys.add(k);
  return [...keys];
}

/**
 * ε at each distance: fitted per patch and basis, then averaged over them.
 * @returns {Map<number, {eps:number, fits:Array<{patch:string, basis:string, eps:number}>}>}
 */
export function epsilonByDistance(records, key, { minRounds = 1, failuresOf = (r) => r.results[key]?.failures } = {}) {
  const groups = new Map();
  for (const r of records) {
    const failures = failuresOf(r);
    if (failures == null) continue;
    const id = `${r.d}|${r.patch}|${r.basis}`;
    if (!groups.has(id)) groups.set(id, { d: r.d, patch: r.patch, basis: r.basis, points: [] });
    groups.get(id).points.push({ rounds: r.rounds, failures, shots: r.shots });
  }
  const byD = new Map();
  for (const g of groups.values()) {
    const fit = fitEpsilon(g.points, { minRounds });
    if (!fit.ok) continue;
    if (!byD.has(g.d)) byD.set(g.d, { eps: 0, fits: [] });
    byD.get(g.d).fits.push({ patch: g.patch, basis: g.basis, eps: fit.eps });
  }
  for (const v of byD.values()) v.eps = v.fits.reduce((s, f) => s + f.eps, 0) / v.fits.length;
  return new Map([...byD.entries()].sort((a, b) => a[0] - b[0]));
}

/**
 * Λ from ε at each distance: a least-squares line through (d, ln ε_d), and
 * Λ = exp(−2 · slope). With two distances this is exactly ε_small / ε_large.
 * Also the ratio between each consecutive pair of distances.
 */
export function lambdaFit(epsByD) {
  const ds = [...epsByD.keys()].sort((a, b) => a - b);
  const pairwise = [];
  for (let i = 0; i + 1 < ds.length; i++) {
    pairwise.push({ from: ds[i], to: ds[i + 1], lambda: epsByD.get(ds[i]).eps / epsByD.get(ds[i + 1]).eps });
  }
  if (ds.length < 2) return { lambda: NaN, pairwise };
  const n = ds.length;
  const ys = ds.map((d) => Math.log(epsByD.get(d).eps));
  const mx = ds.reduce((s, d) => s + d, 0) / n;
  const my = ys.reduce((s, y) => s + y, 0) / n;
  let sxy = 0, sxx = 0;
  ds.forEach((d, i) => { sxy += (d - mx) * (ys[i] - my); sxx += (d - mx) ** 2; });
  return { lambda: Math.exp(-2 * (sxy / sxx)), pairwise };
}

function gaussian(rng) {
  let u = 0;
  while (u === 0) u = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
}

function percentile(sorted, q) {
  if (!sorted.length) return NaN;
  const i = Math.min(sorted.length - 1, Math.max(0, Math.round(q * (sorted.length - 1))));
  return sorted[i];
}

/**
 * A parametric bootstrap: each experiment's failures redrawn from the binomial
 * its own rate implies (by the normal approximation, which at 50,000 shots is
 * exact to far better than the interval's own noise), everything refitted, and
 * the central 95% of the refits kept.
 */
export function bootstrap(records, key, { minRounds = 1 } = {}, B = 400, rng = Math.random) {
  const epsDraws = new Map();
  const lambdaDraws = [];
  const pairDraws = [];
  for (let b = 0; b < B; b++) {
    const drawn = new Map();
    for (const r of records) {
      const f = r.results[key]?.failures;
      if (f == null) continue;
      const p = f / r.shots;
      const x = Math.round(r.shots * p + Math.sqrt(r.shots * p * (1 - p)) * gaussian(rng));
      drawn.set(r, Math.min(r.shots, Math.max(0, x)));
    }
    const byD = epsilonByDistance(records, key, { minRounds, failuresOf: (r) => drawn.get(r) });
    for (const [d, v] of byD) {
      if (!epsDraws.has(d)) epsDraws.set(d, []);
      epsDraws.get(d).push(v.eps);
    }
    const fit = lambdaFit(byD);
    if (Number.isFinite(fit.lambda)) lambdaDraws.push(fit.lambda);
    fit.pairwise.forEach((p, i) => { (pairDraws[i] ??= []).push(p.lambda); });
  }
  const interval = (xs) => {
    const s = [...xs].sort((a, b) => a - b);
    return [percentile(s, 0.025), percentile(s, 0.975)];
  };
  return {
    eps: new Map([...epsDraws].map(([d, xs]) => [d, interval(xs)])),
    lambda: interval(lambdaDraws),
    pairwise: pairDraws.map(interval),
  };
}

/** A seeded generator, so a bootstrap can be repeated exactly. */
export function seededRandom(seed = 1) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
