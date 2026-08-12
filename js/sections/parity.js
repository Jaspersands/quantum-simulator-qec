/**
 * Section 2 — a single plaquette.
 *
 * The one idea this figure has to land: a stabilizer measurement returns the
 * *parity* of four qubits and nothing else. Two different error patterns with
 * the same parity are indistinguishable to it. That is the seed of the
 * degeneracy argument in section 5, so it is worth showing before the lattice
 * arrives and makes everything busier.
 *
 * Rendered as inline SVG rather than canvas: it is small, it needs to be
 * keyboard-reachable, and it inherits the same CSS custom properties as the
 * canvas figures so the two look like one system.
 */

import { $, fill, el } from '../dom.js';

const SVG_NS = 'http://www.w3.org/2000/svg';

/** Qubit positions in SVG user units — a diamond, matching the lattice motif. */
const SITES = [
  { id: 0, x: 90,  y: 26,  label: 'a' },
  { id: 1, x: 154, y: 90,  label: 'b' },
  { id: 2, x: 90,  y: 154, label: 'c' },
  { id: 3, x: 26,  y: 90,  label: 'd' },
];

function svg(tag, attrs = {}) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  return node;
}

export function initParity(root) {
  const host = $('[data-parity-figure]', root);
  const readout = $('[data-parity-readout]', root);
  const verdict = $('[data-parity-verdict]', root);
  const clearBtn = $('[data-parity-clear]', root);
  const twinBtn = $('[data-parity-twin]', root);
  if (!host) return;

  /** Which qubits carry an error. */
  let flipped = [false, false, false, false];

  const parity = () => flipped.filter(Boolean).length % 2;

  /** Every pattern with the same parity as the current one. */
  function siblings() {
    const target = parity();
    const out = [];
    for (let mask = 0; mask < 16; mask++) {
      let bits = 0;
      for (let i = 0; i < 4; i++) if (mask & (1 << i)) bits++;
      if (bits % 2 === target) out.push(mask);
    }
    return out;
  }

  const currentMask = () => flipped.reduce((m, on, i) => (on ? m | (1 << i) : m), 0);

  function setMask(mask) {
    flipped = [0, 1, 2, 3].map((i) => Boolean(mask & (1 << i)));
    render();
  }

  function draw() {
    const canvas = svg('svg', {
      viewBox: '0 0 180 180',
      width: '100%',
      role: 'group',
      'aria-label': 'One stabilizer plaquette with four data qubits',
    });
    canvas.style.maxWidth = '210px';

    const lit = parity() === 1;

    // Plaquette face
    const face = svg('polygon', {
      points: SITES.map((s) => `${s.x},${s.y}`).join(' '),
      fill: lit ? 'var(--defect-soft)' : 'var(--z-soft)',
      stroke: lit ? 'var(--defect)' : 'var(--rule-soft)',
      'stroke-width': lit ? 2 : 1,
    });
    canvas.append(face);

    // Centre marker: the measurement outcome
    if (lit) {
      canvas.append(svg('circle', { cx: 90, cy: 90, r: 6, fill: 'var(--defect)' }));
    } else {
      const label = svg('text', {
        x: 90, y: 90,
        'text-anchor': 'middle', 'dominant-baseline': 'central',
        fill: 'var(--ink-3)',
        'font-family': 'var(--font-mono)', 'font-size': '11', 'font-weight': '600',
      });
      label.textContent = 'Z';
      canvas.append(label);
    }

    // Qubits
    SITES.forEach((site, i) => {
      const group = svg('g', {
        role: 'button',
        tabindex: '0',
        'aria-label': `Qubit ${site.label}, ${flipped[i] ? 'flipped' : 'not flipped'}`,
        'aria-pressed': String(flipped[i]),
        style: 'cursor: pointer',
      });

      group.append(svg('circle', {
        cx: site.x, cy: site.y, r: 13,
        fill: flipped[i] ? 'var(--x)' : 'var(--surface)',
        stroke: flipped[i] ? 'var(--x)' : 'var(--ink-3)',
        'stroke-width': 1.5,
      }));

      const text = svg('text', {
        x: site.x, y: site.y,
        'text-anchor': 'middle', 'dominant-baseline': 'central',
        fill: flipped[i] ? '#fff' : 'var(--ink-3)',
        'font-family': 'var(--font-mono)', 'font-size': '10', 'font-weight': '600',
        style: 'pointer-events: none; user-select: none',
      });
      text.textContent = site.label;
      group.append(text);

      const toggle = () => { flipped[i] = !flipped[i]; render(); };
      group.addEventListener('click', toggle);
      group.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); toggle(); }
      });

      canvas.append(group);
    });

    fill(host, canvas);
  }

  function render() {
    draw();

    const flippedCount = flipped.filter(Boolean).length;
    const lit = parity() === 1;
    const names = SITES.filter((_, i) => flipped[i]).map((s) => s.label);

    fill(readout, [
      el('div', { class: 'readout__row' }, [
        el('span', { class: 'readout__key', text: 'flipped' }),
        el('span', { class: 'readout__val', text: names.length ? names.join(', ') : 'none' }),
      ]),
      el('div', { class: 'readout__row' }, [
        el('span', { class: 'readout__key', text: 'count' }),
        el('span', { class: 'readout__val', text: String(flippedCount) }),
      ]),
      el('div', { class: 'readout__row' }, [
        el('span', { class: 'readout__key', text: 'measurement' }),
        el('span', {
          class: 'readout__val',
          style: `color: ${lit ? 'var(--defect)' : 'var(--ink)'}`,
          text: lit ? '−1  (defect)' : '+1  (quiet)',
        }),
      ]),
    ]);

    const alternatives = siblings().length - 1;
    verdict.innerHTML = lit
      ? `The plaquette fires. It has told you an <strong>odd</strong> number of these four qubits flipped — and nothing else. <strong>${alternatives}</strong> other patterns of errors produce this exact same outcome.`
      : `The plaquette is quiet. That does <em>not</em> mean nothing happened: an <strong>even</strong> number of flips also reads as quiet. <strong>${alternatives}</strong> other patterns look identical from here.`;

    twinBtn.disabled = alternatives === 0;
  }

  clearBtn?.addEventListener('click', () => setMask(0));

  twinBtn?.addEventListener('click', () => {
    // Jump to a different pattern the measurement cannot distinguish.
    const options = siblings().filter((m) => m !== currentMask());
    if (!options.length) return;
    setMask(options[Math.floor(Math.random() * options.length)]);
  });

  setMask(0b0001);
}
