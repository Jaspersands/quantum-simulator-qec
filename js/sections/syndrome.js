/**
 * Section 4 — errors and syndromes.
 *
 * Click qubits to inject errors and watch which checks fire. The chain button
 * is the load-bearing control: it lays down a connected run of errors so the
 * reader can see the interior stay quiet and only the two endpoints light up.
 */

import { Session, ERROR, CODE } from '../engine.js';
import { LatticeView, legendHTML } from '../lattice.js';
import { growChain, scatter, countSet } from '../patterns.js';
import { $, $$, fill, el } from '../dom.js';

export function initSyndrome(root, instance) {
  const canvas = $('[data-syndrome-canvas]', root);
  const readout = $('[data-syndrome-readout]', root);
  const legend = $('[data-syndrome-legend]', root);
  const note = $('[data-syndrome-note]', root);
  const distanceSelect = $('[data-syndrome-distance]', root);
  if (!canvas) return;

  const view = new LatticeView(canvas, { interactive: true, showStabilizerLabels: false });
  legend.innerHTML = legendHTML(['x', 'z', 'y', 'erased', 'defect', 'clean']);

  let session = null;

  const injectionType = () => {
    const checked = $$('[name="syndrome-error-type"]', root).find((r) => r.checked);
    return Number(checked?.value ?? ERROR.X);
  };

  function update(message) {
    view.draw();
    const state = session.read();
    const errors = new Set();
    for (let i = 0; i < session.numData; i++) {
      if (state.errorX[i] || state.errorZ[i]) errors.add(i);
    }
    const defects = countSet(state.syndrome);
    const erased = countSet(state.erased);

    fill(readout, [
      el('div', { class: 'readout__row' }, [
        el('span', { class: 'readout__key', text: 'qubits hit' }),
        el('span', { class: 'readout__val', text: String(errors.size) }),
      ]),
      // Only shown once there is one, so the row does not sit at zero for the
      // readers who never touch the erasure control.
      erased ? el('div', { class: 'readout__row' }, [
        el('span', { class: 'readout__key', text: 'erased' }),
        el('span', { class: 'readout__val', text: String(erased) }),
      ]) : null,
      el('div', { class: 'readout__row' }, [
        el('span', { class: 'readout__key', text: 'defects' }),
        el('span', {
          class: defects ? 'readout__val readout__val--defect' : 'readout__val',
          text: String(defects),
        }),
      ]),
    ]);

    if (message) {
      note.textContent = message;
    } else if (errors.size && defects === 0) {
      note.textContent = 'Errors are present and every check is quiet. The pattern you built is invisible to the syndrome.';
    } else if (defects) {
      note.textContent = `${defects} check${defects === 1 ? '' : 's'} fired. Notice they sit at the ends of what you drew, never along the middle.`;
    } else {
      note.textContent = 'Click a data qubit to inject an error. Click a plaquette to see which qubits it watches.';
    }
  }

  view.onPick = (target) => {
    const type = injectionType();
    if (target.kind !== 'qubit') { update(); return; }
    if (type === 2) {
      session.toggleErasure(target.idx, target.t);
      // An erasure on its own moves no check, so without a word here the click
      // looks like it did nothing.
      update('Marked as erased. A located loss changes no syndrome by itself — '
        + 'what it gives the decoder is the position, which is worth far more '
        + 'than the same error arriving unannounced.');
      return;
    }
    session.toggleError(target.idx, type, target.t);
    update();
  };

  function rebuild() {
    session?.free();
    session = new Session(instance, {
      d: Number(distanceSelect.value),
      codeType: CODE.ROTATED,
      rounds: 1,
    });
    view.setSession(session);
    update();
  }

  $('[data-syndrome-chain]', root)?.addEventListener('click', () => {
    session.clearErrors();
    // Deliberately shorter than the patch. A chain that reaches both edges has
    // its endpoints absorbed by the boundaries and shows no syndrome at all —
    // true, and the subject of section 5, but it would contradict the point
    // this figure is making.
    //
    // The two axes are not equivalent. X errors are caught by Z-plaquettes,
    // and those only line two of the four edges, so a chain running toward the
    // other pair of edges has both endpoints absorbed. Rather than hard-code
    // which axis that is, lay the chain down and count: if it does not show two
    // defects, take the other axis.
    const type = injectionType() === ERROR.Z ? ERROR.Z : ERROR.X;
    const length = Math.max(2, session.d - 2);
    let chain = growChain(session, { length, type, axis: 0 });
    if (countSet(session.read().syndrome) !== 2) {
      session.clearErrors();
      chain = growChain(session, { length, type, axis: 1 });
    }
    const defects = countSet(session.read().syndrome);
    const tail = defects === 2
      ? 'one at each end. Every check in the middle covers two of the errors, reads even parity, and stays quiet.'
      : defects === 1
        ? 'just one. The chain runs into the edge of the patch, and an endpoint that reaches a boundary is absorbed by it.'
        : 'none at all. This chain reaches the boundary at both ends, so there is nothing left to detect — which is precisely the situation section 5 is about.';
    update(`A straight chain of ${chain.length} errors, and ${defects} defect${defects === 1 ? '' : 's'} — ${tail}`);
  });

  $('[data-syndrome-scatter]', root)?.addEventListener('click', () => {
    session.clearErrors();
    const touched = scatter(session, { p: 0.08 });
    update(touched
      ? 'Depolarizing noise at p = 8% — the same channel the Monte Carlo uses. This is what the decoder actually receives.'
      : 'Nothing landed that time — noise is random. Try again.');
  });

  $('[data-syndrome-clear]', root)?.addEventListener('click', () => {
    session.clearErrors();
    update('Cleared.');
  });

  distanceSelect.addEventListener('change', rebuild);

  rebuild();
  return { rebuild };
}
