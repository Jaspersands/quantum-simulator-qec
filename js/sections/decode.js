/**
 * Section 5 — decoding, and why it is a guess.
 *
 * The figure has two views of the same run. "Error + correction" shows what the
 * decoder proposed on top of what actually happened. "Residual" shows the XOR
 * of the two — what is left on the patch afterwards.
 *
 * The residual is the whole argument. If it closes into loops it acts trivially
 * and the logical qubit survived. If it runs from one side of the patch to the
 * other it is a logical operator, and the information is gone even though every
 * check reads quiet. The engine's own verdict from `wasm_decode` is what we
 * report, so the label always matches what the simulator scored.
 */

import { Session, ERROR, DECODER_NAME } from '../engine.js';
import { LatticeView, legendHTML } from '../lattice.js';
import { growChain, scatter, countSet } from '../patterns.js';
import { $, $$, fill, el } from '../dom.js';

export function initDecode(root, instance) {
  const canvas = $('[data-decode-canvas]', root);
  const legend = $('[data-decode-legend]', root);
  const verdict = $('[data-decode-verdict]', root);
  const readout = $('[data-decode-readout]', root);
  const decoderSelect = $('[data-decode-decoder]', root);
  const distanceSelect = $('[data-decode-distance]', root);
  const codeSelect = $('[data-decode-code]', root);
  if (!canvas) return;

  const view = new LatticeView(canvas, { interactive: true, showStabilizerLabels: false });
  let session = null;
  let decoded = false;

  function setLegend(mode) {
    legend.innerHTML = mode === 'residual'
      ? legendHTML(['x', 'z', 'y', 'clean'])
      : legendHTML(['x', 'z', 'correction', 'defect']);
  }

  function currentMode() {
    const checked = $$('[name="decode-view"]', root).find((r) => r.checked);
    return checked?.value ?? 'error';
  }

  function setVerdict(kind, html) {
    verdict.className = `verdict verdict--${kind}`;
    verdict.innerHTML = html;
  }

  function update() {
    // setMode redraws; calling draw() again here would paint every plaquette twice.
    const mode = currentMode();
    setLegend(mode);
    view.setMode(mode);

    const state = session.read();
    const defects = countSet(state.syndrome);
    const errors = countSet(state.errorX) + countSet(state.errorZ);
    const residual = session.residual();
    const residualWeight = decoded ? countSet(residual.x) + countSet(residual.z) : null;

    fill(readout, [
      el('div', { class: 'readout__row' }, [
        el('span', { class: 'readout__key', text: 'error weight' }),
        el('span', { class: 'readout__val', text: String(errors) }),
      ]),
      el('div', { class: 'readout__row' }, [
        el('span', { class: 'readout__key', text: 'defects' }),
        el('span', { class: 'readout__val', text: String(defects) }),
      ]),
      el('div', { class: 'readout__row' }, [
        el('span', { class: 'readout__key', text: 'residual weight' }),
        el('span', { class: 'readout__val', text: residualWeight == null ? '—' : String(residualWeight) }),
      ]),
    ]);
  }

  function runDecoder() {
    const decoder = Number(decoderSelect.value);
    const state = session.read();
    if (countSet(state.errorX) + countSet(state.errorZ) === 0) {
      setVerdict('idle', 'Nothing to decode — inject some errors first.');
      return;
    }

    const result = session.decode(decoder);
    decoded = true;

    const residual = session.residual();
    const weight = countSet(residual.x) + countSet(residual.z);
    const name = DECODER_NAME[decoder];

    if (result.failed) {
      setVerdict('fail',
        `<strong>Logical error.</strong> ${name} produced a correction that matches the syndrome, `
        + `but error ⊕ correction spans the patch. Every check now reads quiet and the encoded qubit is still wrong. `
        + `Switch to the residual view to see the chain that survived.`);
    } else if (weight === 0) {
      setVerdict('ok',
        `<strong>Recovered exactly.</strong> ${name} reconstructed the error it was shown. Residual weight 0.`);
    } else {
      setVerdict('ok',
        `<strong>Recovered.</strong> ${name} did not guess your exact error — a residual of weight ${weight} is left behind — `
        + `but it closes into loops rather than crossing the patch, and loops act trivially on the logical qubit. `
        + `A different guess, the same outcome.`);
    }
    update();
  }

  function reset(message) {
    session.clearErrors();
    decoded = false;
    setVerdict('idle', message);
    update();
  }

  function rebuild() {
    session?.free();
    session = new Session(instance, {
      d: Number(distanceSelect.value),
      codeType: Number(codeSelect.value),
      rounds: 1,
    });
    view.setSession(session);
    decoded = false;
    setVerdict('idle', 'Inject errors, then run a decoder.');
    update();
  }

  view.onPick = (target) => {
    if (target.kind !== 'qubit') return;
    const checked = $$('[name="decode-error-type"]', root).find((r) => r.checked);
    session.toggleError(target.idx, Number(checked?.value ?? ERROR.X), target.t);
    decoded = false;
    setVerdict('idle', 'Syndrome changed — run the decoder again.');
    update();
  };

  $('[data-decode-run]', root)?.addEventListener('click', runDecoder);

  $('[data-decode-chain]', root)?.addEventListener('click', () => {
    session.clearErrors();
    decoded = false;
    // A chain longer than half the patch is where failures start to appear:
    // the decoder finds the shorter way round and completes a logical operator.
    growChain(session, { length: Math.max(2, Math.ceil(session.d / 2) + 1), type: ERROR.X });
    setVerdict('idle', 'A chain reaching across part of the patch. Run the decoder.');
    update();
  });

  $('[data-decode-scatter]', root)?.addEventListener('click', () => {
    session.clearErrors();
    decoded = false;
    scatter(session, { p: 0.1 });
    setVerdict('idle', 'Random noise at p = 10%. Run the decoder.');
    update();
  });

  $('[data-decode-clear]', root)?.addEventListener('click', () => reset('Cleared.'));

  $$('[name="decode-view"]', root).forEach((radio) => radio.addEventListener('change', update));
  decoderSelect.addEventListener('change', () => {
    if (decoded) runDecoder();
  });
  distanceSelect.addEventListener('change', rebuild);
  codeSelect.addEventListener('change', rebuild);

  rebuild();
  return { rebuild };
}
