/**
 * Section 3 — anatomy of the patch.
 *
 * Read-only. Hovering a plaquette lights the data qubits it measures; hovering
 * a qubit lights the plaquettes watching it. The point is that every qubit is
 * covered by several checks and every check overlaps its neighbours, which is
 * what makes a single defect locatable at all.
 */

import { Session, CODE } from '../engine.js';
import { LatticeView, legendHTML } from '../lattice.js';
import { $, fill, el } from '../dom.js';

const STAB_DESCRIPTION = {
  0: 'measures Z⊗Z⊗Z⊗Z — it fires when an odd number of its qubits took an X error',
  1: 'measures X⊗X⊗X⊗X — it fires when an odd number of its qubits took a Z error',
  2: 'measures alternating X and Z around the plaquette',
};

export function initAnatomy(root, instance) {
  const canvas = $('[data-anatomy-canvas]', root);
  const readout = $('[data-anatomy-readout]', root);
  const legend = $('[data-anatomy-legend]', root);
  const distanceSelect = $('[data-anatomy-distance]', root);
  const codeSelect = $('[data-anatomy-code]', root);
  const counts = $('[data-anatomy-counts]', root);
  if (!canvas) return;

  const view = new LatticeView(canvas, {
    interactive: true,
    showStabilizerLabels: true,
    showQubitIndices: false,
  });

  legend.innerHTML = legendHTML(['xplaq', 'zplaq', 'clean']);

  let session = null;

  function describeIdle() {
    fill(readout, el('p', {
      class: 'muted',
      style: 'margin:0; font-size: var(--t-small)',
      text: 'Hover a plaquette or a qubit.',
    }));
  }

  function describe(target) {
    if (!target) { describeIdle(); return; }

    if (target.kind === 'stabilizer') {
      const stab = session.stabilizers[target.idx];
      const support = session.support.get(stab.idx);
      fill(readout, [
        el('div', { class: 'readout__row' }, [
          el('span', { class: 'readout__key', text: 'plaquette' }),
          el('span', { class: 'readout__val', text: `#${stab.idx}` }),
        ]),
        el('div', { class: 'readout__row' }, [
          el('span', { class: 'readout__key', text: 'measures' }),
          el('span', { class: 'readout__val', text: `${support.length} qubits` }),
        ]),
        el('p', {
          style: 'margin: var(--s2) 0 0; font-size: var(--t-micro); line-height:1.5; color: var(--ink-2)',
          text: STAB_DESCRIPTION[stab.type],
        }),
      ]);
    } else {
      const watchers = session.touchedBy.get(target.idx) ?? [];
      fill(readout, [
        el('div', { class: 'readout__row' }, [
          el('span', { class: 'readout__key', text: 'data qubit' }),
          el('span', { class: 'readout__val', text: `#${target.idx}` }),
        ]),
        el('div', { class: 'readout__row' }, [
          el('span', { class: 'readout__key', text: 'watched by' }),
          el('span', { class: 'readout__val', text: `${watchers.length} plaquettes` }),
        ]),
        el('p', {
          style: 'margin: var(--s2) 0 0; font-size: var(--t-micro); line-height:1.5; color: var(--ink-2)',
          text: watchers.length > 1
            ? 'An error here disturbs every one of them, which is how its position gets pinned down.'
            : 'Only one check covers this qubit — boundary qubits are the least constrained on the patch.',
        }),
      ]);
    }
  }

  function rebuild() {
    session?.free();
    session = new Session(instance, {
      d: Number(distanceSelect.value),
      codeType: Number(codeSelect.value),
      rounds: 1,
    });
    view.setSession(session);

    const dataCount = session.numData;
    const stabCount = session.numStab;
    fill(counts, [
      el('div', { class: 'readout__row' }, [
        el('span', { class: 'readout__key', text: 'data qubits' }),
        el('span', { class: 'readout__val', text: String(dataCount) }),
      ]),
      el('div', { class: 'readout__row' }, [
        el('span', { class: 'readout__key', text: 'checks' }),
        el('span', { class: 'readout__val', text: String(stabCount) }),
      ]),
      el('div', { class: 'readout__row' }, [
        el('span', { class: 'readout__key', text: 'encodes' }),
        el('span', { class: 'readout__val', text: `${dataCount - stabCount} logical qubit` }),
      ]),
    ]);
    describeIdle();
  }

  view.onHover = describe;

  distanceSelect.addEventListener('change', rebuild);
  codeSelect.addEventListener('change', rebuild);

  rebuild();

  return { rebuild, get session() { return session; } };
}

export { CODE };
