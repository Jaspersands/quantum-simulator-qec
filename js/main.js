/**
 * Wiring.
 *
 * Two engine instances run side by side. The main thread keeps one for the
 * interactive lattice figures, where every click needs an immediate answer. The
 * worker holds its own for Monte Carlo, so a long sweep never blocks a click.
 */

import { instantiate } from './engine.js';
import { Compute } from './compute.js';
import { initNav } from './nav.js';
import { $ } from './dom.js';

import { initParity } from './sections/parity.js';
import { initAnatomy } from './sections/anatomy.js';
import { initSyndrome } from './sections/syndrome.js';
import { initDecode } from './sections/decode.js';
import { initSpacetime } from './sections/spacetime.js';
import { initVitals } from './sections/vitals.js';
import { initResultsTable, initThresholdSweep } from './sections/threshold.js';
import { initBias } from './sections/bias.js';
import { initBench } from './sections/bench.js';

/**
 * Show a boot failure. Each caller supplies its own diagnosis — the two failure
 * modes have different causes and different consequences, and telling a reader
 * to re-serve the page when the real problem is a missing browser feature only
 * wastes their time.
 */
function fail(message) {
  const host = $('[data-boot-error]');
  if (!host) return;
  host.hidden = false;
  host.textContent = message;
}

const SERVE_HINT = 'ES modules and the .wasm fetch both need HTTP — run '
  + '`python3 -m http.server` from the project root rather than opening the file directly.';

async function boot() {
  initNav();

  // Section 2 is pure JS and does not wait on the engine.
  const parityRoot = $('#parity');
  if (parityRoot) initParity(parityRoot);

  let compute = null;
  try {
    compute = new Compute();
  } catch (error) {
    fail(`Could not start the simulation worker (${error.message}). `
      + 'The Monte Carlo sections — the results table, the threshold sweep, the bias comparison, '
      + 'and the bench — need module workers, which this browser appears not to support. '
      + 'The interactive lattice figures below still work.');
  }

  if (compute) {
    initVitals($('#overview') ?? document, compute);
    initResultsTable($('#threshold'), compute);
    initThresholdSweep($('#threshold'), compute);
    initBias($('#bias'), compute);
    initBench($('#bench'), compute);
  }

  let instance;
  try {
    instance = await instantiate();
  } catch (error) {
    fail(`Could not load the simulation engine (${error.message}). ${SERVE_HINT}`);
    return;
  }

  initAnatomy($('#anatomy'), instance);
  initSyndrome($('#syndrome'), instance);
  initDecode($('#decoding'), instance);
  initSpacetime($('#spacetime'), instance);
}

boot();
