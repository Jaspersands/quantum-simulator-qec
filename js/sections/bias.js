/**
 * Section 8 — biased noise and the XZZX code.
 *
 * Real hardware does not produce X, Y, and Z errors in equal measure; dephasing
 * usually dominates. The rotated code is built for symmetric noise and gains
 * little from that skew. The XZZX code alternates X and Z around every
 * plaquette, which in the published result turns a strong bias into a much
 * higher threshold.
 *
 * This engine does not reproduce that, and the figure reports what it measures
 * rather than what it ought to show. The decoder's double-matching bug is
 * fixed, so XZZX no longer degrades with distance — but its curve is flat where
 * it should fall, because a lone boundary defect can be matched to the wrong
 * error family and the resulting weight-one residual is scored as a logical
 * failure. See section 10.
 */

import { CODE, DECODER, NOISE } from '../engine.js';
import { Plot, plotLegend } from '../plot.js';
import { wilson, percent, count } from '../compute.js';
import { $, fill, el } from '../dom.js';

const BIAS_VALUES = [0.5, 1, 2, 4, 8, 16, 32, 64];

const CODES = [
  { codeType: CODE.ROTATED, label: 'Rotated surface', color: 'var(--d3)' },
  { codeType: CODE.XZZX, label: 'XZZX surface', color: 'var(--d5)' },
];

export function initBias(root, compute) {
  const canvas = $('[data-bias-canvas]', root);
  const legend = $('[data-bias-legend]', root);
  const status = $('[data-bias-status]', root);
  const meter = $('[data-bias-meter]', root);
  const runBtn = $('[data-bias-run]', root);
  const summary = $('[data-bias-summary]', root);
  const pInput = $('[data-bias-p]', root);
  const dSelect = $('[data-bias-distance]', root);
  const runsInput = $('[data-bias-runs]', root);
  if (!canvas) return;

  canvas.dataset.aspect = '0.6';
  const plot = new Plot(canvas, {
    xLabel: 'Noise bias  η   (log scale)',
    yLabel: 'Logical error rate  p_L',
    formatX: (v) => {
      const eta = Math.pow(2, v);
      return eta < 1 ? eta.toFixed(1) : String(Math.round(eta));
    },
    formatY: (v) => `${(v * 100).toFixed(0)}%`,
    xTicks: 4,
  });

  legend.innerHTML = plotLegend(CODES);

  let running = false;

  function draw(results) {
    const series = CODES.map(({ codeType, label, color }) => ({
      label,
      color,
      points: results
        .filter((r) => r.codeType === codeType)
        .map((r) => {
          const ci = wilson(r.pL, r.runs);
          return { x: Math.log2(r.bias), y: r.pL, lo: ci.lo, hi: ci.hi };
        })
        .sort((a, b) => a.x - b.x),
      line: results
        .filter((r) => r.codeType === codeType)
        .map((r) => ({ x: Math.log2(r.bias), y: r.pL }))
        .sort((a, b) => a.x - b.x),
    }));

    plot.render({
      series,
      xRange: [Math.log2(BIAS_VALUES[0]), Math.log2(BIAS_VALUES[BIAS_VALUES.length - 1])],
      empty: 'Run the comparison to sweep the bias.',
    });
  }

  function summarise(results) {
    const strongest = BIAS_VALUES[BIAS_VALUES.length - 1];
    const at = (codeType, bias) => results.find((r) => r.codeType === codeType && r.bias === bias);
    const rotated = at(CODE.ROTATED, strongest);
    const xzzx = at(CODE.XZZX, strongest);
    const rotatedFlat = at(CODE.ROTATED, 0.5);
    const xzzxFlat = at(CODE.XZZX, 0.5);
    if (!rotated || !xzzx || !rotatedFlat || !xzzxFlat) return;

    const rows = [
      ['unbiased, rotated', percent(rotatedFlat.pL)],
      ['unbiased, XZZX', percent(xzzxFlat.pL)],
      [`η = ${strongest}, rotated`, percent(rotated.pL)],
      [`η = ${strongest}, XZZX`, percent(xzzx.pL)],
    ];

    fill(summary, [
      ...rows.map(([k, v]) => el('div', { class: 'readout__row' }, [
        el('span', { class: 'readout__key', text: k }),
        el('span', { class: 'readout__val', text: v }),
      ])),
      el('p', {
        class: 'note',
        text: xzzx.pL < rotated.pL
          ? 'XZZX came out ahead at the strongest bias, as the published result predicts.'
          : 'XZZX did not pull ahead at any bias, which matches the defect described above. Its '
            + 'error rate no longer runs away with distance, but it is still flat rather than '
            + 'falling, so the patch never gets far enough ahead for the bias advantage to show.',
      }),
    ]);
  }

  async function run() {
    if (running) return;
    running = true;
    runBtn.disabled = true;
    fill(summary, el('p', { class: 'muted fit-note', text: 'Sweeping…' }));
    // The worker runs one job at a time and may still be filling the section 7
    // table, so say that rather than leaving a stale line on screen.
    status.textContent = 'Queued…';

    const cells = [];
    for (const { codeType } of CODES) {
      for (const bias of BIAS_VALUES) cells.push({ codeType, bias });
    }

    const runs = Math.max(200, Number(runsInput.value) || 1000);
    const collected = [];

    try {
      await compute.call('table', {
        cells,
        base: {
          d: Number(dSelect.value),
          decoder: DECODER.UNION_FIND,
          noiseMode: NOISE.DATA,
          rounds: 1,
          p: Number(pInput.value) || 0.06,
          runs,
        },
      }, ({ done, total, cell }) => {
        collected.push(cell);
        status.textContent = `η = ${cell.bias} · ${done}/${total}`;
        meter.style.width = `${(done / total) * 100}%`;
        draw(collected);
      });

      draw(collected);
      summarise(collected);
      status.textContent = `Done — ${count(cells.length * runs)} runs.`;
    } catch (error) {
      status.textContent = `Comparison failed: ${error.message}`;
    } finally {
      meter.style.width = '0%';
      running = false;
      runBtn.disabled = false;
    }
  }

  runBtn.addEventListener('click', run);
  draw([]);
  return { run };
}
