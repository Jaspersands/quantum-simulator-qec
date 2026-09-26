/**
 * Section 10, Figure 8 — checked against Stim.
 *
 * The first table derives this engine's detector error model for each circuit,
 * live in the worker, and compares it mechanism by mechanism with the model
 * Stim wrote for the same circuit. The second runs our sampler and decoder live
 * beside PyMatching's recorded numbers. Recorded values wear a dagger and the
 * secondary ink. Nothing is shown that was not computed here or read from the
 * reference file, and a reference that cannot be read says so.
 */

import { CODE, NOISE, XC_NOISE } from '../engine.js';
import { wilson, percent } from '../compute.js';
import { $, fill, el } from '../dom.js';
import { verdict, formatRel, circuitLabel, microseconds, disagreementText, megabytes } from '../xcheck-format.js';

const DATA = new URL('../../data/xcheck/', import.meta.url);

/**
 * Live shots per distance for the decoding table, set by measured cost: the
 * sparse matcher decodes d = 7 at p = 0.6% in about 0.06 ms a shot in
 * WebAssembly, so the whole table runs in roughly ten seconds.
 */
const LIVE_RUNS = { 3: 100000, 5: 100000, 7: 50000 };
const TIMING_RUNS = { 3: 20000, 5: 5000, 7: 2000 };

function job(c) {
  return {
    key: c.name,
    kind: c.source,
    demUrl: new URL(`${c.name}.dem.txt`, DATA).href,
    circuitUrl: c.source === 'stim' ? new URL(`${c.name}.stim.txt`, DATA).href : null,
    gen: c.source === 'ours'
      ? {
        codeType: c.code === 'xzzx' ? CODE.XZZX : CODE.ROTATED,
        d: c.d,
        rounds: c.rounds,
        noise: c.noise === 'sd6' ? XC_NOISE.SD6 : XC_NOISE.CURRENT,
        p: c.p,
        eta: c.eta,
        basis: 0,
      }
      : null,
    circuitSha256: c.circuit_sha256 ?? null,
  };
}

/** A Wilson interval, compactly: "1.99–2.40%". */
function interval(ci) {
  return `${(ci.lo * 100).toFixed(2)}–${percent(ci.hi)}`;
}

/** The recorded graph comparison, from the harness. */
function graphVerdict(c) {
  if (c.edges_one_sided === undefined) return { text: '—', tone: 'idle' };
  const bad = c.edges_one_sided + c.splits_differ;
  return bad === 0 && c.edges_max_rel <= 1e-9
    ? { text: 'identical †', tone: 'idle' }
    : { text: `${bad} differ †`, tone: 'fail' };
}

export function initXcheck(root, compute) {
  const figure = $('[data-xcheck]', root);
  if (!figure) return;
  const models = $('[data-xcheck-models]', figure);
  const decoding = $('[data-xcheck-decoding]', figure);
  const status = $('[data-xcheck-status]', figure);
  const foot = $('[data-xcheck-foot]', figure);
  const more = $('[data-xcheck-more]', figure);

  const modelRows = new Map();

  function modelRow(c) {
    const cells = {
      detectors: el('td', { class: 'num', text: c.detectors?.toLocaleString('en-US') ?? '—' }),
      mechanisms: el('td', { class: 'num', text: '—' }),
      rel: el('td', { class: 'num', text: '—' }),
      verdict: el('td', {}, [el('span', { class: 'verdict-text--idle', text: c.d >= 7 ? 'on request' : 'queued' })]),
    };
    const g = graphVerdict(c);
    const tr = el('tr', {}, [
      el('th', { scope: 'row', text: circuitLabel(c) }),
      cells.detectors,
      cells.mechanisms,
      cells.rel,
      cells.verdict,
      el('td', { class: 'recorded' }, [el('span', { class: `verdict-text--${g.tone}`, text: g.text })]),
    ]);
    modelRows.set(c.name, { c, cells });
    return tr;
  }

  function fillModel(result) {
    const row = modelRows.get(result.key);
    if (!row) return;
    const v = verdict(result);
    if (result.ok) {
      row.cells.detectors.textContent = result.detectors.toLocaleString('en-US');
      row.cells.mechanisms.textContent =
        `${result.ours.toLocaleString('en-US')} / ${result.theirs.toLocaleString('en-US')}`;
      row.cells.rel.textContent = formatRel(result.maxRel);
    }
    fill(row.cells.verdict, el('span', { class: `verdict-text--${v.tone}`, text: v.text }));
  }

  async function runModels(circuits) {
    for (const c of circuits) fillModel({ key: c.name, pending: true });
    await compute.call('xcheck', { rows: circuits.map(job) }, (p) => {
      fillModel(p.row);
      status.textContent = `Error models: ${p.done} of ${p.total} compared`;
    });
  }

  async function runDecoding(records) {
    // The rates run on every worker at once; the timings then run one at a
    // time, so no two timed decoders share the machine.
    let finished = 0;
    status.textContent = `Decoding ${records.length} rows on ${compute.size ?? 1} workers…`;
    await Promise.all(records.map(async (rec) => {
      const runs = LIVE_RUNS[rec.d] ?? 1000;
      const r = await compute.call('benchmark', {
        noiseMode: NOISE.SD6, codeType: CODE.ROTATED, d: rec.d, rounds: rec.d, p: rec.p, runs,
      });
      const ci = wilson(r.rate, r.runs);
      fill(rec.cells.ours, [
        document.createTextNode(percent(r.rate)),
        el('span', {
          class: 'ci',
          text: `${interval(ci)} · ${r.runs.toLocaleString('en-US')}`
            + (r.decodeErrors ? ` · ${r.decodeErrors} refused` : ''),
        }),
      ]);
      finished += 1;
      status.textContent = `Decoding: ${finished} of ${records.length} rows`;
    }));
    for (const [i, rec] of records.entries()) {
      status.textContent = `Timing the decoder: d = ${rec.d}, p = ${percent(rec.p, 1)} (${i + 1} of ${records.length})`;
      const t = await compute.call('xctiming', {
        cfg: { codeType: CODE.ROTATED, d: rec.d, rounds: rec.d, noise: XC_NOISE.SD6, p: rec.p },
        runs: TIMING_RUNS[rec.d] ?? 200,
      });
      rec.cells.oursTime.textContent = microseconds(t.decodeMicros);
    }
  }

  async function start() {
    status.textContent = 'Loading the reference…';
    let reference;
    try {
      const response = await fetch(new URL('reference.json', DATA));
      if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
      reference = await response.json();
    } catch (error) {
      status.textContent = `Reference unavailable (${error.message}); nothing to compare against.`;
      return;
    }

    fill(models, reference.circuits.map(modelRow));
    const now = reference.circuits.filter((c) => c.d < 7);
    const later = reference.circuits.filter((c) => c.d >= 7);

    const records = reference.decoding.map((rec) => {
      const cells = {
        ours: el('td', { class: 'num', text: '—' }),
        oursTime: el('td', { class: 'num', text: '—' }),
      };
      const rate = rec.pymatching_failures / rec.shots;
      const ci = wilson(rate, rec.shots);
      decoding.append(el('tr', {}, [
        el('th', { scope: 'row', text: `d = ${rec.d}, p = ${percent(rec.p, 1)}` }),
        cells.ours,
        el('td', { class: 'num recorded' }, [
          document.createTextNode(`${percent(rate)} †`),
          el('span', { class: 'ci', text: `${interval(ci)} · ${rec.shots.toLocaleString('en-US')}` }),
        ]),
        el('td', { class: 'num recorded', text: `${disagreementText(rec)} †` }),
        cells.oursTime,
        el('td', { class: 'num recorded', text: `${microseconds(rec.pymatching_us)} †` }),
      ]));
      return { ...rec, cells };
    });

    foot.textContent = `† Recorded on ${reference.generated} with Stim ${reference.stim} and PyMatching `
      + `${reference.pymatching} by tools/xcheck.py, which reproduces every one. The matching graph is the `
      + 'decomposed model a decoder actually runs on; the recorded check compares it with Stim\'s edge for '
      + 'edge. Under each rate, its 95% interval and the number of shots. "Disagree" counts shots on which '
      + 'the two decoders, given the identical detection events Stim sampled, predicted differently: two '
      + 'exact matchers may disagree only where two corrections tie in weight, and the recorded run checks '
      + 'every disagreement for that. Decode times: ours is measured now, in this tab, in WebAssembly; '
      + 'PyMatching\'s is native code on the machine that recorded it, where this decoder, also native, took '
      + reference.decoding.map((r) => `${microseconds(r.ours_us)} at d = ${r.d}, p = ${percent(r.p, 1)}`).join('; ')
      + '.';

    if (later.length) {
      const bytes = later.reduce((sum, c) => sum + (c.bytes ?? 0), 0);
      more.textContent = `Check d = 7 too (${megabytes(bytes)} of reference)`;
      more.hidden = false;
      more.addEventListener('click', async () => {
        more.disabled = true;
        try {
          await runModels(later);
          status.textContent = 'd = 7 compared.';
          more.hidden = true;
        } catch (error) {
          status.textContent = `Failed: ${error.message}. Try again.`;
          more.disabled = false;
        }
      });
    }

    try {
      await runModels(now);
      await runDecoding(records);
      status.textContent = 'Done.';
    } catch (error) {
      status.textContent = `Failed: ${error.message}`;
    }
  }

  let started = false;
  const observer = new IntersectionObserver((entries) => {
    if (started || !entries.some((e) => e.isIntersecting)) return;
    started = true;
    observer.disconnect();
    start();
  }, { rootMargin: '400px 0px' });
  observer.observe(figure);
}
