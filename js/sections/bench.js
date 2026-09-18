/**
 * Section 9 — the bench.
 *
 * Everything the narrative did not need a control for: erasure, correlated
 * noise, arbitrary distances and shot counts. The story is over by this point;
 * this is the instrument it was describing, exposed in full.
 */

import { DECODER_NAME, NOISE_NAME } from '../engine.js';
import { ChannelView } from '../channel-view.js';
import { Meter } from '../meter.js';
import { wilson, percent, count } from '../compute.js';
import { $, fill, el } from '../dom.js';

/** Read every control in the bench panel into a run config. */
function readConfig(root) {
  const noiseMode = Number($('[data-bench-noise]', root).value);
  return {
    d: Number($('[data-bench-d]', root).value),
    codeType: Number($('[data-bench-code]', root).value),
    decoder: Number($('[data-bench-decoder]', root).value),
    noiseMode,
    p: Number($('[data-bench-p]', root).value),
    bias: Number($('[data-bench-bias]', root).value),
    rounds: Number($('[data-bench-rounds]', root).value),
    runs: Number($('[data-bench-runs]', root).value),
    erasure: Number($('[data-bench-erasure]', root).value),
    correlated: Number($('[data-bench-correlated]', root).value),
  };
}

export function initBench(root, compute) {
  const runBtn = $('[data-bench-run]', root);
  const status = $('[data-bench-status]', root);
  const output = $('[data-bench-output]', root);
  const chanBtn = $('[data-channel-run]', root);
  const chanStatus = $('[data-channel-status]', root);
  const chanOutput = $('[data-channel-output]', root);
  const chanCanvas = $('[data-channel-canvas]', root);
  const roundsField = $('[data-bench-rounds]', root);
  const noiseSelect = $('[data-bench-noise]', root);
  if (!runBtn) return;

  // Rounds are meaningless without faulty measurements.
  const syncRounds = () => {
    const dataOnly = Number(noiseSelect.value) === 0;
    roundsField.disabled = dataOnly;
    roundsField.closest('.field').classList.toggle('field--inert', dataOnly);
  };
  noiseSelect.addEventListener('change', syncRounds);
  syncRounds();

  const channel = chanCanvas ? new ChannelView(chanCanvas) : null;
  channel?.draw();

  chanBtn?.addEventListener('click', async () => {
    const config = readConfig(root);
    chanBtn.disabled = true;
    chanStatus.textContent = `Estimating over ${count(config.runs)} shots…`;
    try {
      const result = await compute.call('channel', config);
      channel?.set([result.x, result.y, result.z]);
      fill(chanOutput, [
        ...[['λ_X', result.x], ['λ_Y', result.y], ['λ_Z', result.z]].map(([k, v]) =>
          el('div', { class: 'readout__row' }, [
            el('span', { class: 'readout__key', text: k }),
            el('span', { class: 'readout__val', text: v.toFixed(4) }),
          ])),
      ]);
      chanStatus.textContent = `${count(config.runs)} shots · ${result.seconds.toFixed(2)} s`;
    } catch (error) {
      chanStatus.textContent = `Failed: ${error.message}`;
    } finally {
      chanBtn.disabled = false;
    }
  });

  // The run streams: chunks land as progress messages, and the estimate is
  // drawn converging. A second click stops after the current chunk.
  const meterCanvas = $('[data-bench-meter]', root);
  const meter = meterCanvas ? new Meter(meterCanvas) : null;
  meter?.render({ total: 1, samples: [] });
  let job = null;

  const row = (k, v) => el('div', { class: 'readout__row' }, [
    el('span', { class: 'readout__key', text: k }),
    el('span', { class: 'readout__val', text: v }),
  ]);

  runBtn.addEventListener('click', async () => {
    if (job) {
      compute.cancel(job.id);
      runBtn.disabled = true;
      status.textContent = 'Stopping after this chunk…';
      return;
    }
    const config = readConfig(root);
    const samples = [];
    let meanRate = 0;
    runBtn.textContent = 'Stop';
    status.textContent = `Running ${count(config.runs)} shots…`;

    const paint = (p, final = false) => {
      const rate = p.done ? p.failures / p.done : 0;
      const ci = wilson(rate, p.done);
      meter?.render({ total: config.runs, samples, final });
      fill(output, [
        row('logical error rate', p.done ? percent(rate) : '—'),
        row('95% interval', p.done ? `${percent(ci.lo)} – ${percent(ci.hi)}` : '—'),
        row('throughput', `${count(meanRate)} shots/s`),
        row('shots', `${count(p.done)} / ${count(config.runs)}`),
        row('wall time', `${p.seconds.toFixed(2)} s`),
      ]);
    };

    try {
      job = compute.call('stream', config, (p) => {
        const rate = p.failures / p.done;
        const ci = wilson(rate, p.done);
        samples.push({ done: p.done, rate, lo: ci.lo, hi: ci.hi });
        meanRate = p.seconds > 0 ? Math.round(p.done / p.seconds) : 0;
        paint(p);
      });
      const result = await job;
      meanRate = result.runsPerSecond;
      paint({ done: result.runs, failures: result.failures, seconds: result.seconds }, true);
      status.textContent = `${result.cancelled ? 'Stopped' : 'Done'} · ${DECODER_NAME[config.decoder]} · `
        + `${NOISE_NAME[config.noiseMode]} · d = ${config.d}`
        + (result.cancelled ? ` · ${count(result.runs)} of ${count(config.runs)} shots` : '');
    } catch (error) {
      status.textContent = `Failed: ${error.message}`;
    } finally {
      job = null;
      runBtn.textContent = 'Run Monte Carlo';
      runBtn.disabled = false;
    }
  });
}
