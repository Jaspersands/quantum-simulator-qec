/**
 * The four cards under the hero.
 *
 * Throughput is measured in the reader's own browser on load rather than being
 * quoted from a benchmark run once on the author's machine. The numbers a
 * visitor sees are therefore true for the hardware they are sitting at, which
 * is the only claim this page can honestly make about speed.
 */

import { $, fill, el } from '../dom.js';
import { count } from '../compute.js';

export function initVitals(root, compute) {
  const dataRate = $('[data-vital-data]', root);
  const phenomRate = $('[data-vital-phenom]', root);
  const note = $('[data-vitals-note]', root);
  if (!dataRate) return;

  compute.call('vitals').then((result) => {
    fill(dataRate, [
      document.createTextNode(count(result.dataRunsPerSecond)),
      el('span', { class: 'vital__unit', text: ' runs/s' }),
    ]);
    fill(phenomRate, [
      document.createTextNode(count(result.phenomRunsPerSecond)),
      el('span', { class: 'vital__unit', text: ' runs/s' }),
    ]);
    if (note) {
      note.textContent = `Timed on this device just now — best of several samples over `
        + `${count(result.sampleRuns)} decoded shots at d = 5. A throttled or busy machine will read lower.`;
    }
  }).catch((error) => {
    dataRate.innerHTML = '<span class="muted">unavailable</span>';
    phenomRate.innerHTML = '<span class="muted">unavailable</span>';
    if (note) note.textContent = `Could not measure: ${error.message}`;
  });
}
