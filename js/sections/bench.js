/**
 * Section 9 — the bench.
 *
 * Everything the narrative did not need a control for: erasure, correlated
 * noise, arbitrary distances and shot counts. The story is over by this point;
 * this is the instrument it was describing, exposed in full.
 */

import { DECODER_NAME, NOISE_NAME } from '../engine.js';
import { ChannelView } from '../channel-view.js';
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

  runBtn.addEventListener('click', async () => {
    const config = readConfig(root);
    runBtn.disabled = true;
    status.textContent = `Running ${count(config.runs)} shots…`;

    try {
      const result = await compute.call('benchmark', config);
      const ci = wilson(result.rate, result.runs);
      fill(output, [
        el('div', { class: 'readout__row' }, [
          el('span', { class: 'readout__key', text: 'logical error rate' }),
          el('span', { class: 'readout__val', text: percent(result.rate) }),
        ]),
        el('div', { class: 'readout__row' }, [
          el('span', { class: 'readout__key', text: '95% interval' }),
          el('span', { class: 'readout__val', text: `${percent(ci.lo)} – ${percent(ci.hi)}` }),
        ]),
        el('div', { class: 'readout__row' }, [
          el('span', { class: 'readout__key', text: 'throughput' }),
          el('span', { class: 'readout__val', text: `${count(result.runsPerSecond)} shots/s` }),
        ]),
        el('div', { class: 'readout__row' }, [
          el('span', { class: 'readout__key', text: 'wall time' }),
          el('span', { class: 'readout__val', text: `${result.seconds.toFixed(2)} s` }),
        ]),
      ]);
      status.textContent = `${DECODER_NAME[config.decoder]} · ${NOISE_NAME[config.noiseMode]} · d = ${config.d}`;
    } catch (error) {
      status.textContent = `Failed: ${error.message}`;
    } finally {
      runBtn.disabled = false;
    }
  });
}
