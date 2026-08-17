/**
 * Section 6 — when the measurements themselves lie.
 *
 * A faulty readout is indistinguishable from a real error in a single round.
 * Repeating the measurement turns the syndrome into a three-dimensional object:
 * a lying readout produces a defect pair separated in *time* rather than space,
 * which the decoder can tell apart from a genuine data error.
 *
 * Clicking a plaquette here toggles a measurement error rather than a data
 * error, so the vertical defect pair can be produced deliberately.
 */

import { Session, ERROR, CODE, DECODER_NAME } from '../engine.js';
import { LatticeView, legendHTML } from '../lattice.js';
import { countSet } from '../patterns.js';
import { $, $$, fill, el } from '../dom.js';

export function initSpacetime(root, instance) {
  const canvas = $('[data-spacetime-canvas]', root);
  const legend = $('[data-spacetime-legend]', root);
  const verdict = $('[data-spacetime-verdict]', root);
  const readout = $('[data-spacetime-readout]', root);
  const roundsSelect = $('[data-spacetime-rounds]', root);
  const decoderSelect = $('[data-spacetime-decoder]', root);
  const note = $('[data-spacetime-note]', root);
  if (!canvas) return;

  const view = new LatticeView(canvas, { interactive: true, showStabilizerLabels: false });
  legend.innerHTML = legendHTML(['x', 'z', 'defect', 'correction']);

  let session = null;
  let decoded = false;

  function setVerdict(kind, html) {
    verdict.className = `verdict verdict--${kind}`;
    verdict.innerHTML = html;
  }

  function update(message) {
    view.draw();
    const state = session.read();
    const defects = countSet(state.defects);
    const errors = countSet(state.errorX) + countSet(state.errorZ);

    fill(readout, [
      el('div', { class: 'readout__row' }, [
        el('span', { class: 'readout__key', text: 'rounds' }),
        el('span', { class: 'readout__val', text: String(session.rounds) }),
      ]),
      el('div', { class: 'readout__row' }, [
        el('span', { class: 'readout__key', text: 'data errors' }),
        el('span', { class: 'readout__val', text: String(errors) }),
      ]),
      el('div', { class: 'readout__row' }, [
        el('span', { class: 'readout__key', text: 'defects' }),
        el('span', { class: 'readout__val', text: String(defects) }),
      ]),
    ]);

    if (message) note.textContent = message;
  }

  function rebuild() {
    session?.free();
    const rounds = Number(roundsSelect.value);

    // The stack grows downward as rounds are added, so the canvas has to grow
    // with it or the layers get squeezed together.
    canvas.dataset.aspect = (0.5 + 0.19 * rounds).toFixed(2);

    session = new Session(instance, {
      d: 3,
      codeType: CODE.ROTATED,
      rounds,
    });
    view.setSession(session);
    decoded = false;
    setVerdict('idle', 'Click a plaquette to make its readout lie in that round.');
    update('Time runs bottom to top. Each layer is one full round of syndrome extraction.');
  }

  view.onPick = (target) => {
    if (target.kind === 'stabilizer') {
      session.toggleMeasurementError(target.idx, target.t);
      decoded = false;
      const isLastRound = target.t === session.rounds - 1;
      update(`Readout of plaquette #${target.idx} lies in round ${target.t}. `
        + (isLastRound
          ? 'This is the final round, so only one detection event appears — the partner would '
            + 'fall in a round that was never measured. An unpaired event has nothing to cancel '
            + 'against, which is why a real experiment ends with a round it can trust.'
          : 'It disagrees with its neighbours in time for exactly one round, so two detection '
            + 'events appear on the same plaquette in consecutive layers, joined by the amber '
            + 'line. A pair separated in time, not in space — the signature the decoder uses to '
            + 'discount it.'));
    } else {
      const checked = $$('[name="spacetime-error-type"]', root).find((r) => r.checked);
      session.toggleError(target.idx, Number(checked?.value ?? ERROR.X), target.t);
      decoded = false;
      update('A real data error persists from the round it appears in onwards, so it flips the '
        + 'reading once and leaves it flipped. That shows up as a pair of detection events side '
        + 'by side within a single layer — separated in space, not in time.');
    }
    setVerdict('idle', 'Run the decoder to see whether it can separate the two.');
  };

  $('[data-spacetime-run]', root)?.addEventListener('click', () => {
    const state = session.read();
    if (countSet(state.defects) === 0) {
      setVerdict('idle', 'No detection events — nothing to decode.');
      return;
    }
    const decoder = Number(decoderSelect.value);
    const result = session.decode(decoder);
    decoded = true;
    setVerdict(result.failed ? 'fail' : 'ok', result.failed
      ? `<strong>Logical error.</strong> ${DECODER_NAME[decoder]} matched the spacetime syndrome, but the residual spans the patch.`
      : `<strong>Recovered.</strong> ${DECODER_NAME[decoder]} matched defects across rounds as well as across the patch, and the logical qubit survived.`);
    update();
  });

  $('[data-spacetime-clear]', root)?.addEventListener('click', () => {
    session.clearErrors();
    decoded = false;
    setVerdict('idle', 'Cleared.');
    update('Click a plaquette to make its readout lie in that round.');
  });

  roundsSelect.addEventListener('change', rebuild);
  decoderSelect.addEventListener('change', () => { if (decoded) $('[data-spacetime-run]', root).click(); });

  rebuild();
  return { rebuild };
}
