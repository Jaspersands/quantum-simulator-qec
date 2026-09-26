// Logical error per cycle and Λ for every decoder in data/google-results/,
// with the same fit the page uses (js/lambda-fit.js).
// Run: node tools/lambda.mjs [bootstrap draws = 400] [results.json ...] [--json out.json]
// With files, fits each (Willow-shaped records, from round 10) instead of
// data/google-results/{willow,sycamore}.json. With --json, fits everything
// (Willow, Sycamore, the validation, and the files) and writes it all as JSON,
// for tools/report.py.
import { readFile, writeFile } from 'node:fs/promises';
import { epsilonByDistance, lambdaFit, bootstrap, decoderKeys, seededRandom } from '../js/lambda-fit.js';
import { decoderLabel, percent, percentRange, ratio } from '../js/hardware-format.js';
import { windowLabel } from '../js/realtime-format.js';

const args = process.argv.slice(2);
const jsonAt = args.indexOf('--json');
const jsonOut = jsonAt >= 0 ? args.splice(jsonAt, 2)[1] : null;
const B = Number(args[0] ?? 400);
const files = args.slice(1);
const load = async (name) => Object.values(JSON.parse(await readFile(new URL(`../data/google-results/${name}.json`, import.meta.url), 'utf8')).experiments);

function table(title, records, minRounds, alternatives, label = decoderLabel) {
  const ds = [...new Set(records.map((r) => r.d))].sort((a, b) => a - b);
  console.log(`\n${title}: ${records.length} experiments, fits from round ${minRounds}, ${B} bootstrap draws`);
  console.log(`  ${'decoder'.padEnd(62)} ${ds.map((d) => `eps_${d}`.padEnd(22)).join(' ')} ${ds.slice(1).map((d, i) => `L${ds[i]}/${d}`.padEnd(8)).join(' ')} Lambda [95%]`);
  const out = {};
  for (const key of decoderKeys(records)) {
    const byD = epsilonByDistance(records, key, { minRounds });
    if (byD.size < ds.length) {
      // Measured at some distances only (the buffer study is at d = 5): ε there, no Λ.
      const boot = bootstrap(records, key, { minRounds }, B, seededRandom(11));
      const eps = ds.map((d) => (byD.has(d) ? `${percent(byD.get(d).eps)} (${percentRange(boot.eps.get(d))})` : '—').padEnd(22)).join(' ');
      console.log(`  ${label(key).padEnd(62)} ${eps}`);
      out[key] = { eps: Object.fromEntries([...byD].map(([d, e]) => [d, e.eps])), epsInterval: Object.fromEntries([...byD.keys()].map((d) => [d, boot.eps.get(d)])) };
      continue;
    }
    const fit = lambdaFit(byD);
    const boot = bootstrap(records, key, { minRounds }, B, seededRandom(11));
    const eps = ds.map((d) => `${percent(byD.get(d).eps)} (${percentRange(boot.eps.get(d))})`.padEnd(22)).join(' ');
    const pairs = fit.pairwise.map((p) => p.lambda.toFixed(3).padEnd(8)).join(' ');
    const altFits = Object.fromEntries(alternatives.map((m) => [m, lambdaFit(epsilonByDistance(records, key, { minRounds: m })).lambda]));
    const alt = alternatives.map((m) => `from ${m}: ${altFits[m].toFixed(3)}`).join(', ');
    console.log(`  ${label(key).padEnd(62)} ${eps} ${pairs} ${ratio(fit.lambda, boot.lambda)}   (${alt})`);
    out[key] = {
      eps: Object.fromEntries(ds.map((d) => [d, byD.get(d).eps])),
      epsInterval: Object.fromEntries(ds.map((d) => [d, boot.eps.get(d)])),
      lambda: fit.lambda, interval: boot.lambda, pairwise: fit.pairwise.map((q) => q.lambda), fromRound: altFits,
    };
  }
  return out;
}

const all = { draws: B, files: {} };
for (const f of files) {
  const recs = Object.values(JSON.parse(await readFile(f, 'utf8')).experiments);
  // Sycamore's experiments run 1 to 25 rounds and are fitted from round 3; Willow's, from 10.
  const sycamore = /sycamore/.test(f);
  all.files[f] = table(f, recs, sycamore ? 3 : 10, sycamore ? [1, 5] : [30], (k) => (/^(global|window)\//.test(k) ? windowLabel(k) : decoderLabel(k)));
}
if (files.length && !jsonOut) process.exit(0);
const willow = await load('willow').catch(() => []);
const sycamore = await load('sycamore').catch(() => []);
if (willow.length) all.willow = table('Willow', willow, 10, [30]);
if (sycamore.length) {
  const fits = table('Sycamore', sycamore, 3, [1, 5]);
  all.sycamore = fits;
  // Validation: Google's own tensor-network predictions, fitted here, against
  // the published numbers (arXiv:2207.06431).
  const tn = fits['google/tensor_network_contraction'];
  if (tn) {
    console.log(`\nValidation, Google's tensor-network predictions fitted here: eps_3 ${percent(tn.eps[3])} `
      + `(published 3.028% ± 0.023%), eps_5 ${percent(tn.eps[5])} (published 2.914% ± 0.016%)`);
    all.validation = { eps: tn.eps, published: { 3: 0.03028, 5: 0.02914 }, byMinRounds: {} };
    for (const m of [1, 2, 3, 4, 5]) {
      const byD = epsilonByDistance(sycamore, 'google/tensor_network_contraction', { minRounds: m });
      console.log(`  from round ${m}: eps_3 ${percent(byD.get(3).eps)}, eps_5 ${percent(byD.get(5).eps)}`);
      all.validation.byMinRounds[m] = { 3: byD.get(3).eps, 5: byD.get(5).eps };
    }
  }
}
if (jsonOut) await writeFile(jsonOut, `${JSON.stringify(all, null, 1)}\n`);
