/**
 * Section 7 — the threshold.
 *
 * Two pieces of live data, no literals anywhere.
 *
 * The table asks the only question that matters about a code: does adding
 * qubits help? Each column is answered by measurement, not by a fit — if the
 * logical error rate falls from d=3 to d=7 the column is below threshold, and
 * if it rises the column is above it. That is the crossover, stated directly.
 *
 * The sweep then locates the crossing point by universal collapse and draws it.
 */

import { CODE, DECODER, NOISE } from '../engine.js';
import { Plot, plotLegend } from '../plot.js';
import { wilson, fitThreshold, collapseCurve, percent, count } from '../compute.js';
import { $, fill, el } from '../dom.js';

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
const DISTANCES = {
  [NOISE.DATA]: [3, 5, 7, 9, 11],
  [NOISE.PHENOM]: [3, 5, 7, 9],
  [NOISE.CIRCUIT]: [3, 5, 7, 9],
};

const SERIES_COLOR = {
  3: 'var(--d3)', 5: 'var(--d5)', 7: 'var(--d7)', 9: 'var(--d9)', 11: 'var(--d11)',
};

/** The load-time table is a reference, not a fit; three distances make the point. */
const TABLE_DISTANCES = [3, 5, 7];

/** Physical rates spanning both sides of the threshold for both noise models. */
const TABLE_PS = [0.005, 0.01, 0.02, 0.05, 0.10, 0.15];

const TABLE_MODELS = [
  { noiseMode: NOISE.DATA, rounds: 1, label: 'Data noise only', sub: 'perfect measurements' },
  { noiseMode: NOISE.PHENOM, rounds: null, label: 'Phenomenological', sub: 'faulty measurements, T = d' },
];

/**
 * Enough shots that the d=3 and d=7 intervals actually separate away from
 * threshold. Below about 1,500 they overlap in nearly every column and the
 * verdict row degenerates to "too close" everywhere, which tells the reader
 * nothing.
 */
const TABLE_RUNS = 2000;

/**
 * The two noise models have thresholds an order of magnitude apart, so a single
 * sweep range would waste most of its points saturated at one end or the other.
 */
const SWEEP_PS = {
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
const SWEEP_RUNS = {
  [NOISE.DATA]: 20000,
  [NOISE.PHENOM]: 6000,
  [NOISE.CIRCUIT]: 1200,
};

/* -- The results table -------------------------------------------------- */

export function initResultsTable(root, compute) {
  const body = $('[data-results-body]', root);
  const status = $('[data-results-status]', root);
  const caption = $('[data-results-caption]', root);
  if (!body) return;

  const cells = [];
  for (const model of TABLE_MODELS) {
    for (const d of TABLE_DISTANCES) {
      for (const p of TABLE_PS) {
        cells.push({ noiseMode: model.noiseMode, rounds: model.rounds, d, p });
      }
    }
  }

  /** key -> {pL, runs} */
  const results = new Map();
  let failed = false;
  const key = (c) => `${c.noiseMode}:${c.d}:${c.p}`;

  function paint() {
    const rows = [];
    for (const model of TABLE_MODELS) {
      TABLE_DISTANCES.forEach((d, i) => {
        const tds = [];
        if (i === 0) {
          tds.push(el('th', { scope: 'rowgroup', rowspan: String(TABLE_DISTANCES.length) }, [
            el('strong', { text: model.label }),
            el('span', { class: 'ci', text: model.sub }),
          ]));
        }
        tds.push(el('th', { scope: 'row', class: 'num mono', text: `d = ${d}` }));

        for (const p of TABLE_PS) {
          const entry = results.get(key({ noiseMode: model.noiseMode, d, p }));
          if (!entry) {
            tds.push(el('td', { class: 'num' }, [
              failed
                ? el('span', { class: 'muted', text: '—' })
                : el('span', { class: 'pending', text: '' }),
            ]));
          } else {
            const ci = wilson(entry.pL, entry.runs);
            tds.push(el('td', { class: 'num' }, [
              document.createTextNode(percent(entry.pL)),
              el('span', { class: 'ci', text: `${percent(ci.lo, 2)}–${percent(ci.hi, 2)}` }),
            ]));
          }
        }
        rows.push(el('tr', {}, tds));
      });

      // Verdict row: measured, not fitted — does more distance help at this p?
      const verdictCells = [
        el('th', { scope: 'row', colspan: '2' }, [
          el('span', { class: 'readout__key mono', text: 'more qubits →' }),
        ]),
      ];
      for (const p of TABLE_PS) {
        const small = results.get(key({ noiseMode: model.noiseMode, d: 3, p }));
        const large = results.get(key({ noiseMode: model.noiseMode, d: 7, p }));
        if (!small || !large) {
          verdictCells.push(el('td', { class: 'num' }, [
            failed
              ? el('span', { class: 'muted', text: '—' })
              : el('span', { class: 'pending', text: '' }),
          ]));
        } else {
          // Only claim a direction when the two intervals actually separate.
          // Right at threshold they overlap, and asserting either way would be
          // reading noise.
          const a = wilson(small.pL, small.runs);
          const b = wilson(large.pL, large.runs);
          const separated = b.hi < a.lo || a.hi < b.lo;
          const helps = large.pL < small.pL;
          verdictCells.push(el('td', { class: 'num' }, [
            el('span', {
              class: separated ? `tag tag--${helps ? 'below' : 'above'}` : 'tag tag--flat',
              title: separated ? null : 'the 95% intervals for d = 3 and d = 7 overlap',
              text: separated ? (helps ? 'helps' : 'hurts') : 'too close',
            }),
          ]));
        }
      }
      rows.push(el('tr', { class: 'row--summary' }, verdictCells));
    }
    fill(body, rows);
  }

  paint();

  compute.call('table', { cells, base: {
    codeType: CODE.ROTATED,
    decoder: DECODER.UNION_FIND,
    bias: 0.5,
    runs: TABLE_RUNS,
  } }, ({ done, total, cell }) => {
    results.set(key(cell), { pL: cell.pL, runs: cell.runs });
    status.textContent = `Measuring… ${done} / ${total} cells`;
    paint();
  }).then(() => {
    status.textContent = `Complete — ${count(cells.length * TABLE_RUNS)} Monte Carlo runs in your browser.`;
    caption.textContent = 'Union-Find decoder, rotated surface code, unbiased noise. '
      + 'Ranges are Wilson 95% intervals. The verdict row reads "too close" where those intervals '
      + 'overlap, which is what sitting on the threshold looks like. Every figure here was computed '
      + 'on this page load; reload and they will move within their intervals.';
    paint();
  }).catch((error) => {
    status.textContent = `Could not complete: ${error.message}`;
    // Repaint so cells that never landed stop showing the animated pending
    // indicator — otherwise a failed run looks identical to one still running.
    failed = true;
    paint();
  });
}

/* -- The sweep and fit -------------------------------------------------- */

export function initThresholdSweep(root, compute) {
  const canvas = $('[data-threshold-canvas]', root);
  const legend = $('[data-threshold-legend]', root);
  const status = $('[data-threshold-status]', root);
  const meter = $('[data-threshold-meter]', root);
  const runBtn = $('[data-threshold-run]', root);
  const results = $('[data-threshold-results]', root);
  const decoderSelect = $('[data-threshold-decoder]', root);
  const noiseSelect = $('[data-threshold-noise]', root);
  const codeSelect = $('[data-threshold-code]', root);
  const planNote = $('[data-threshold-plan]', root);
  if (!canvas) return;

  canvas.dataset.aspect = '0.68';
  const plot = new Plot(canvas, {
    xLabel: 'Physical error rate  p',
    yLabel: 'Logical error rate  p_L',
    formatX: (v) => `${(v * 100).toFixed(0)}%`,
    formatY: (v) => `${(v * 100).toFixed(0)}%`,
  });

  // The distances swept depend on the noise model — circuit-level cannot afford
  // what data noise can — so the legend follows the selection.
  const currentDistances = () => DISTANCES[Number(noiseSelect.value)] ?? [3, 5, 7];
  let seriesMeta = [];
  function refreshLegend() {
    seriesMeta = currentDistances().map((d) => ({ d, label: `d = ${d}`, color: SERIES_COLOR[d] }));
    legend.innerHTML = plotLegend(seriesMeta);
  }
  refreshLegend();

  /** Say up front what the sweep about to run will actually cost. */
  function describeSweep() {
    if (!planNote) return;
    const mode = Number(noiseSelect.value);
    const ds = currentDistances();
    const runs = SWEEP_RUNS[mode] ?? 4000;
    const cells = ds.length * (SWEEP_PS[mode] ?? []).length;
    planNote.textContent = `d = ${ds.join(', ')} · ${count(runs)} shots × ${cells} points`;
  }
  describeSweep();

  let points = [];
  let running = false;

  const currentPs = () => SWEEP_PS[Number(noiseSelect.value)] ?? SWEEP_PS[NOISE.PHENOM];

  function drawPlot(fit) {
    const ps = currentPs();
    const series = seriesMeta.map(({ d, label, color }) => {
      const own = points.filter((pt) => pt.d === d);
      const entry = {
        label,
        color,
        points: own.map((pt) => {
          const ci = wilson(pt.pL, pt.runs);
          return { x: pt.p, y: pt.pL, lo: ci.lo, hi: ci.hi };
        }),
      };
      if (fit?.ok) {
        entry.line = [];
        const step = (ps[ps.length - 1] - ps[0]) / 120;
        for (let p = ps[0]; p <= ps[ps.length - 1] + 1e-9; p += step) {
          entry.line.push({ x: p, y: collapseCurve(fit, d, p) });
        }
      }
      return entry;
    });

    plot.render({
      series,
      markers: fit?.ok && fit.pTh > 0 ? [{ x: fit.pTh, label: `p_th ≈ ${percent(fit.pTh)}` }] : [],
      xRange: [0, ps[ps.length - 1] * 1.05],
      empty: 'Run a sweep to plot p_L against p.',
    });
  }

  function showFit(fit, totalRuns) {
    if (!fit?.ok) {
      fill(results, el('p', {
        class: 'muted fit-note',
        text: `No threshold reported — ${(fit?.reason ?? 'the fit did not converge').replace(/\.$/, '')}.`,
      }));
      return;
    }
    const interval = Number.isFinite(fit.pThLo) && Number.isFinite(fit.pThHi)
      ? `${percent(fit.pThLo)} – ${percent(fit.pThHi)}`
      : 'not resolved';
    const crossText = fit.crossings.length
      ? fit.crossings.map((c) => `${c.small}/${c.large} at ${percent(c.p)}`).join(', ')
      : 'none in range';

    fill(results, [
      el('div', { class: 'readout__row' }, [
        el('span', { class: 'readout__key', text: 'threshold p_th' }),
        el('span', { class: 'readout__val readout__val--ink', text: percent(fit.pTh) }),
      ]),
      el('div', { class: 'readout__row' }, [
        el('span', { class: 'readout__key', text: '95% interval' }),
        el('span', { class: 'readout__val', text: interval }),
      ]),
      el('div', { class: 'readout__row' }, [
        el('span', { class: 'readout__key', text: 'exponent ν' }),
        el('span', { class: 'readout__val', text: fit.nu.toFixed(2) }),
      ]),
      el('div', { class: 'readout__row' }, [
        el('span', { class: 'readout__key', text: 'correction ω' }),
        el('span', {
          class: 'readout__val',
          text: fit.corrected
            ? (fit.omegaAtEdge ? `≥ ${fit.omega.toFixed(0)}` : fit.omega.toFixed(2))
            : 'not fitted',
        }),
      ]),
      el('div', { class: 'readout__row' }, [
        el('span', { class: 'readout__key', text: 'reduced χ²' }),
        el('span', { class: 'readout__val', text: fit.reducedChi2.toFixed(2) }),
      ]),
      el('div', { class: 'readout__row' }, [
        el('span', { class: 'readout__key', text: 'curves cross at' }),
        el('span', { class: 'readout__val', text: crossText }),
      ]),
      el('div', { class: 'readout__row' }, [
        el('span', { class: 'readout__key', text: 'points fitted' }),
        el('span', {
          class: 'readout__val',
          text: `${fit.pointsUsed} of ${fit.pointsTotal}`
            + (fit.windowAccepted ? '' : ' · form not accepted'),
        }),
      ]),
      el('div', { class: 'readout__row' }, [
        el('span', { class: 'readout__key', text: 'total runs' }),
        el('span', { class: 'readout__val', text: count(totalRuns) }),
      ]),
      el('p', {
        class: 'note',
        text: fit.corrected
          ? 'Fitted with the leading correction to scaling, which the collapse alone leaves out. '
            + `Without it the same data gives ${percent(fit.leading.pTh)} — that gap is the `
            + 'finite-size effect, not a disagreement about the physics. The interval is a '
            + 'bootstrap over the shots.'
            + (fit.omegaAtEdge
              ? ' ω ran to the top of its range, which means the correction is confined to the '
                + 'smallest patch rather than decaying gently — read it as a floor, not a value.'
              : '')
            + (fit.windowAccepted
              ? ''
              : ' No window of this sweep is described by the scaling form within its error bars, '
                + 'so these parameters are the least-bad fit rather than a measurement.')
          : 'Fitted without a correction to scaling: that needs four distances to separate from a '
            + 'shift in the threshold, and this sweep has three. Expect the value to sit high.',
      }),
    ]);
  }

  async function runSweep() {
    if (running) return;
    running = true;
    runBtn.disabled = true;
    points = [];
    drawPlot(null);
    fill(results, el('p', { class: 'muted fit-note', text: 'Sweeping…' }));
    // The worker runs one job at a time, so this may sit behind the load-time
    // table. Say so rather than leaving a stale line on screen.
    status.textContent = 'Queued…';

    const noiseMode = Number(noiseSelect.value);
    const runs = SWEEP_RUNS[noiseMode] ?? 4000;
    const distances = currentDistances();
    const ps = currentPs();
    const base = {
      codeType: Number(codeSelect.value),
      decoder: Number(decoderSelect.value),
      noiseMode,
      bias: 0.5,
      runs,
    };

    try {
      await compute.call('sweep', { distances, ps, base }, (progress) => {
        points.push({ d: progress.d, p: progress.p, pL: progress.pL, runs: progress.runs });
        status.textContent = `d = ${progress.d}, p = ${percent(progress.p, 1)} · ${progress.done}/${progress.total}`;
        meter.style.width = `${(progress.done / progress.total) * 100}%`;
        drawPlot(null);
      });

      status.textContent = 'Fitting collapse…';
      await new Promise((r) => setTimeout(r, 0));
      const fit = fitThreshold(points, { bootstrap: 120 });
      drawPlot(fit);
      showFit(fit, points.length * runs);
      status.textContent = `Done — ${count(points.length * runs)} runs across ${points.length} points.`;
    } catch (error) {
      status.textContent = `Sweep failed: ${error.message}`;
    } finally {
      meter.style.width = '0%';
      running = false;
      runBtn.disabled = false;
    }
  }

  runBtn.addEventListener('click', runSweep);
  noiseSelect.addEventListener('change', () => {
    // Each model sweeps its own range and its own set of distances, so old
    // points would be both misplaced and mislabelled.
    points = [];
    refreshLegend();

  /** Say up front what the sweep about to run will actually cost. */
  function describeSweep() {
    if (!planNote) return;
    const mode = Number(noiseSelect.value);
    const ds = currentDistances();
    const runs = SWEEP_RUNS[mode] ?? 4000;
    const cells = ds.length * (SWEEP_PS[mode] ?? []).length;
    planNote.textContent = `d = ${ds.join(', ')} · ${count(runs)} shots × ${cells} points`;
  }
  describeSweep();
    describeSweep();
    fill(results, el('p', { class: 'muted fit-note', text: 'Range changed — run the sweep.' }));
    status.textContent = 'Ready.';
    drawPlot(null);
  });
  drawPlot(null);

  return { runSweep };
}
