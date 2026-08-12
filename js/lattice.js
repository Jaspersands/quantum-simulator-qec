/**
 * Canvas renderer for a surface-code patch.
 *
 * One renderer serves every figure on the page: the read-only anatomy diagram,
 * the click-to-inject syndrome figure, the residual view, and the stacked
 * spacetime view. Behaviour is chosen with options rather than by forking the
 * drawing code, so all four stay visually identical.
 *
 * Coordinates: the engine places data qubits and stabilizers on a doubled
 * integer grid spanning 0..2d. A stabilizer at (x, y) measures the data qubits
 * at (x±1, y±1) — its plaquette.
 */

import { STAB, ERROR } from './engine.js';

/** Read the palette out of CSS so the stylesheet stays the single source. */
function palette() {
  const css = getComputedStyle(document.documentElement);
  const get = (name, fallback) => (css.getPropertyValue(name).trim() || fallback);
  return {
    surface: get('--surface', '#fffefb'),
    sunk: get('--surface-sunk', '#faf8f2'),
    ink: get('--ink', '#1c1d1f'),
    ink2: get('--ink-2', '#4a4c50'),
    ink3: get('--ink-3', '#7c7e83'),
    rule: get('--rule-soft', '#ded9cd'),
    x: get('--x', '#2f5fd0'),
    xSoft: get('--x-soft', '#dfe7fa'),
    z: get('--z', '#c0392b'),
    zSoft: get('--z-soft', '#f8e2df'),
    y: get('--y', '#7b4bb5'),
    ySoft: get('--y-soft', '#ebe1f7'),
    defect: get('--defect', '#c98a06'),
    defectSoft: get('--defect-soft', '#fbeec6'),
    ok: get('--ok', '#2c7a4b'),
    fail: get('--fail', '#b23a2e'),
    erased: get('--erased', '#7c7e83'),
  };
}

const HIT_RADIUS = 17;

export class LatticeView {
  /**
   * @param {HTMLCanvasElement} canvas
   * @param {object} [options]
   * @param {boolean} [options.showQubitIndices]
   * @param {boolean} [options.showStabilizerLabels]
   * @param {boolean} [options.interactive] enable hover highlight + click
   * @param {'error'|'residual'} [options.mode] what the qubit fill represents
   */
  constructor(canvas, options = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.options = {
      showQubitIndices: false,
      showStabilizerLabels: true,
      interactive: false,
      mode: 'error',
      ...options,
    };

    this.colors = palette();
    this.session = null;
    this.hover = null;
    /** @type {(target: {kind: 'qubit'|'stabilizer', idx: number, t: number}, event: MouseEvent) => void} */
    this.onPick = null;
    this.onHover = null;

    this.#fitToDisplay();
    this.#bindEvents();

    this.resizeObserver = new ResizeObserver(() => {
      if (this.#fitToDisplay()) this.draw();
    });
    this.resizeObserver.observe(canvas);
  }

  /** Attach a session and redraw. Re-fits first — callers may have changed the
   *  requested aspect ratio along with the session (the spacetime view grows
   *  taller as rounds are added). */
  setSession(session) {
    this.session = session;
    this.hover = null;
    this.#fitToDisplay();
    this.draw();
  }

  setMode(mode) {
    this.options.mode = mode;
    this.draw();
  }

  /* -- Sizing --------------------------------------------------------- */

  #fitToDisplay() {
    const dpr = window.devicePixelRatio || 1;
    const rect = this.canvas.getBoundingClientRect();
    const cssWidth = rect.width || this.canvas.clientWidth || 460;
    const aspect = Number(this.canvas.dataset.aspect || 1);
    const cssHeight = cssWidth * aspect;

    // Setting our own height re-triggers the observer, so ignore no-op resizes.
    if (this.width === cssWidth && this.height === cssHeight) return false;

    this.canvas.style.height = `${cssHeight}px`;
    this.canvas.width = Math.round(cssWidth * dpr);
    this.canvas.height = Math.round(cssHeight * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.width = cssWidth;
    this.height = cssHeight;
    return true;
  }

  /* -- Projection ------------------------------------------------------ */

  /**
   * Grid coordinate -> canvas point.
   * With one round this is a plain square layout. With several rounds the
   * layers are stacked and skewed so time reads bottom-to-top.
   */
  project(gx, gy, t = 0) {
    const { d, rounds } = this.session;
    const span = d * 2;

    if (rounds === 1) {
      const pad = 34;
      const size = Math.min(this.width, this.height) - pad * 2;
      const originX = (this.width - size) / 2;
      const originY = (this.height - size) / 2;
      return {
        x: originX + (gx / span) * size,
        y: originY + (gy / span) * size,
      };
    }

    const layerW = Math.min(this.width * 0.62, 300);
    const layerH = layerW * 0.42;
    const centreX = this.width / 2;

    // Space the layers by a little over their own depth. Spreading them across
    // the full canvas instead leaves the stack looking like unrelated diagrams.
    const gap = layerH * 1.3;
    const stackHeight = gap * (rounds - 1);
    const firstY = (this.height + stackHeight) / 2 - 4;
    const layerY = firstY - t * gap;

    const nx = gx / d - 1;
    const ny = gy / d - 1;
    return {
      x: centreX + nx * (layerW / 2) + ny * (layerW * 0.16),
      y: layerY + ny * (layerH / 2) - nx * (layerH * 0.16),
    };
  }

  /* -- Geometry helpers ------------------------------------------------ */

  /** Polygon outlining a stabilizer's plaquette, in canvas space. */
  #plaquettePath(stab, t) {
    const support = this.session.support.get(stab.idx);
    const pts = support.map((qIdx) => {
      const q = this.session.dataQubits[qIdx];
      return this.project(q.x, q.y, t);
    });
    if (pts.length === 0) return null;

    const centre = this.project(stab.x, stab.y, t);
    // Sort by angle so the polygon never self-intersects.
    pts.sort((a, b) => Math.atan2(a.y - centre.y, a.x - centre.x)
                     - Math.atan2(b.y - centre.y, b.x - centre.x));
    return { pts, centre };
  }

  #tracePlaquette(stab, t) {
    const shape = this.#plaquettePath(stab, t);
    if (!shape) return null;
    const { pts, centre } = shape;
    const ctx = this.ctx;
    ctx.beginPath();

    if (pts.length >= 3) {
      ctx.moveTo(pts[0].x, pts[0].y);
      for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y);
      ctx.closePath();
    } else if (pts.length === 2) {
      // Boundary stabilizer: a half-disc bulging away from the patch.
      const [a, b] = pts;
      const mid = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
      const out = { x: centre.x - mid.x, y: centre.y - mid.y };
      const len = Math.hypot(out.x, out.y) || 1;
      const bulge = Math.hypot(b.x - a.x, b.y - a.y) * 0.55;
      const tip = { x: mid.x + (out.x / len) * bulge, y: mid.y + (out.y / len) * bulge };
      ctx.moveTo(a.x, a.y);
      ctx.quadraticCurveTo(tip.x, tip.y, b.x, b.y);
      ctx.closePath();
      // The stabilizer's own coordinate sits on the flat edge of the half-disc,
      // so a marker drawn there would float outside the shape. Use a point
      // inside the bulge instead.
      return { x: mid.x + (out.x / len) * bulge * 0.42, y: mid.y + (out.y / len) * bulge * 0.42 };
    } else {
      return null;
    }
    return centre;
  }

  /** Fill and stroke colours for a stabilizer given its type and defect state. */
  #stabStyle(type, triggered) {
    const c = this.colors;
    if (triggered) return { fill: c.defectSoft, stroke: c.defect, label: c.defect };
    if (type === STAB.X) return { fill: c.xSoft, stroke: c.rule, label: c.ink3 };
    if (type === STAB.Z) return { fill: c.zSoft, stroke: c.rule, label: c.ink3 };
    return { fill: c.ySoft, stroke: c.rule, label: c.ink3 };
  }

  #stabLabel(type) {
    if (type === STAB.X) return 'X';
    if (type === STAB.Z) return 'Z';
    return 'XZ';
  }

  /* -- Drawing --------------------------------------------------------- */

  draw() {
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.width, this.height);
    if (!this.session?.ptr) return;

    const session = this.session;
    const state = session.read();
    const rounds = session.rounds;
    const nData = session.numData;
    const nStab = session.numStab;

    // Which arrays drive the qubit fill.
    let fillX = state.errorX;
    let fillZ = state.errorZ;
    if (this.options.mode === 'residual') {
      const residual = session.residual();
      fillX = residual.x;
      fillZ = residual.z;
    }
    const correction = session.correction;

    const qubitRadius = rounds === 1 ? Math.max(7, Math.min(11, 120 / session.d)) : 5;

    for (let t = 0; t < rounds; t++) {
      // Earlier rounds recede slightly, but stay readable — they carry as much
      // of the story as the latest one.
      const dim = rounds === 1 ? 1 : 0.72 + 0.28 * (t / Math.max(1, rounds - 1));
      ctx.save();
      ctx.globalAlpha = dim;

      // Plaquettes
      for (let i = 0; i < nStab; i++) {
        const stab = session.stabilizers[i];
        const triggered = state.syndrome[i + t * nStab] === 1;
        const centre = this.#tracePlaquette(stab, t);
        if (!centre) continue;

        const style = this.#stabStyle(stab.type, triggered);
        const highlighted = this.#isHighlighted('stabilizer', i, t);

        ctx.fillStyle = style.fill;
        ctx.fill();
        ctx.lineWidth = triggered ? 2 : 1;
        ctx.strokeStyle = highlighted ? this.colors.ink : style.stroke;
        if (highlighted) ctx.lineWidth = 2;
        ctx.stroke();

        if (triggered) {
          ctx.beginPath();
          ctx.arc(centre.x, centre.y, rounds === 1 ? 5 : 3.5, 0, Math.PI * 2);
          ctx.fillStyle = this.colors.defect;
          ctx.fill();
        } else if (this.options.showStabilizerLabels && rounds === 1 && session.d <= 5) {
          ctx.fillStyle = style.label;
          ctx.font = `600 ${session.d <= 3 ? 10 : 9}px "JetBrains Mono", monospace`;
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(this.#stabLabel(stab.type), centre.x, centre.y);
        }
      }

      // Data qubits
      for (let i = 0; i < nData; i++) {
        const q = session.dataQubits[i];
        const at = this.project(q.x, q.y, t);
        const k = i + t * nData;

        if (state.erased[k] === 1) {
          ctx.save();
          ctx.beginPath();
          ctx.arc(at.x, at.y, qubitRadius + 4.5, 0, Math.PI * 2);
          ctx.strokeStyle = this.colors.erased;
          ctx.lineWidth = 1.5;
          ctx.setLineDash([2, 3]);
          ctx.stroke();
          ctx.restore();
        }

        if (correction) {
          const corrected = correction.correctionX[k] === 1 || correction.correctionZ[k] === 1;
          if (corrected) {
            ctx.beginPath();
            ctx.arc(at.x, at.y, qubitRadius + 3, 0, Math.PI * 2);
            ctx.strokeStyle = this.colors.ok;
            ctx.lineWidth = 2;
            ctx.stroke();
          }
        }

        const hasX = fillX[k] === 1;
        const hasZ = fillZ[k] === 1;
        ctx.beginPath();
        ctx.arc(at.x, at.y, qubitRadius, 0, Math.PI * 2);
        if (hasX && hasZ) {
          ctx.fillStyle = this.colors.y; ctx.strokeStyle = this.colors.y;
        } else if (hasX) {
          ctx.fillStyle = this.colors.x; ctx.strokeStyle = this.colors.x;
        } else if (hasZ) {
          ctx.fillStyle = this.colors.z; ctx.strokeStyle = this.colors.z;
        } else {
          ctx.fillStyle = this.colors.surface;
          ctx.strokeStyle = this.#isHighlighted('qubit', i, t) ? this.colors.ink : this.colors.ink3;
        }
        ctx.lineWidth = this.#isHighlighted('qubit', i, t) ? 2.5 : 1.5;
        ctx.fill();
        ctx.stroke();

        if (this.options.showQubitIndices && rounds === 1 && qubitRadius >= 8) {
          ctx.fillStyle = (hasX || hasZ) ? '#ffffff' : this.colors.ink3;
          ctx.font = '500 8px "JetBrains Mono", monospace';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(String(i), at.x, at.y);
        }
      }

      // Round label
      if (rounds > 1) {
        const corner = this.project(0, 0, t);
        ctx.fillStyle = this.colors.ink3;
        ctx.font = '600 9px "JetBrains Mono", monospace';
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';
        ctx.fillText(`t = ${t}`, corner.x - 12, corner.y);
      }

      ctx.restore();
    }

    // Vertical guides between rounds, drawn last so defect worldlines read on top.
    if (rounds > 1) this.#drawWorldlines(state, nStab);
  }

  #drawWorldlines(state, nStab) {
    const ctx = this.ctx;
    const session = this.session;
    for (let t = 0; t < session.rounds - 1; t++) {
      for (let i = 0; i < nStab; i++) {
        const stab = session.stabilizers[i];
        const a = this.project(stab.x, stab.y, t);
        const b = this.project(stab.x, stab.y, t + 1);
        const lit = state.syndrome[i + t * nStab] === 1 && state.syndrome[i + (t + 1) * nStab] === 1;
        ctx.save();
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        if (lit) {
          ctx.strokeStyle = this.colors.defect;
          ctx.lineWidth = 2;
        } else {
          ctx.strokeStyle = this.colors.rule;
          ctx.lineWidth = 1;
          ctx.setLineDash([2, 4]);
        }
        ctx.stroke();
        ctx.restore();
      }
    }
  }

  /* -- Hit testing and interaction ------------------------------------- */

  #isHighlighted(kind, idx, t) {
    const h = this.hover;
    if (!h || h.t !== t) return false;
    if (h.kind === kind && h.idx === idx) return true;
    // Hovering a plaquette lights the qubits it measures, and vice versa.
    if (h.kind === 'stabilizer' && kind === 'qubit') {
      return this.session.support.get(h.idx)?.includes(idx);
    }
    if (h.kind === 'qubit' && kind === 'stabilizer') {
      return this.session.touchedBy.get(h.idx)?.includes(idx);
    }
    return false;
  }

  /** Nearest pickable element to a canvas-space point, or null. */
  hitTest(px, py) {
    if (!this.session?.ptr) return null;
    const session = this.session;
    let best = null;
    let bestDist = HIT_RADIUS;

    for (let t = 0; t < session.rounds; t++) {
      for (const q of session.dataQubits) {
        const at = this.project(q.x, q.y, t);
        const dist = Math.hypot(at.x - px, at.y - py);
        if (dist < bestDist) { bestDist = dist; best = { kind: 'qubit', idx: q.idx, t }; }
      }
      for (const s of session.stabilizers) {
        const at = this.project(s.x, s.y, t);
        const dist = Math.hypot(at.x - px, at.y - py);
        if (dist < bestDist) { bestDist = dist; best = { kind: 'stabilizer', idx: s.idx, t }; }
      }
    }
    return best;
  }

  #toCanvasSpace(event) {
    const rect = this.canvas.getBoundingClientRect();
    return { x: event.clientX - rect.left, y: event.clientY - rect.top };
  }

  #bindEvents() {
    if (!this.options.interactive) return;

    this.canvas.style.cursor = 'crosshair';

    this.canvas.addEventListener('mousemove', (event) => {
      const { x, y } = this.#toCanvasSpace(event);
      const target = this.hitTest(x, y);
      const changed = JSON.stringify(target) !== JSON.stringify(this.hover);
      this.hover = target;
      this.canvas.style.cursor = target ? 'pointer' : 'crosshair';
      if (changed) {
        this.draw();
        this.onHover?.(target);
      }
    });

    this.canvas.addEventListener('mouseleave', () => {
      if (this.hover) {
        this.hover = null;
        this.draw();
        this.onHover?.(null);
      }
    });

    this.canvas.addEventListener('click', (event) => {
      const { x, y } = this.#toCanvasSpace(event);
      const target = this.hitTest(x, y);
      if (target) this.onPick?.(target, event);
    });
  }

  destroy() {
    this.resizeObserver.disconnect();
  }
}

/** Shared legend markup, so every figure labels its colours identically. */
export function legendHTML(items) {
  const swatch = {
    x: `<span class="legend__key legend__key--round" style="background: var(--x); border-color: var(--x)"></span>`,
    z: `<span class="legend__key legend__key--round" style="background: var(--z); border-color: var(--z)"></span>`,
    y: `<span class="legend__key legend__key--round" style="background: var(--y); border-color: var(--y)"></span>`,
    clean: `<span class="legend__key legend__key--round" style="background: var(--surface)"></span>`,
    defect: `<span class="legend__key" style="background: var(--defect-soft); border-color: var(--defect)"></span>`,
    xplaq: `<span class="legend__key" style="background: var(--x-soft)"></span>`,
    zplaq: `<span class="legend__key" style="background: var(--z-soft)"></span>`,
    correction: `<span class="legend__key legend__key--round" style="background: transparent; border-color: var(--ok); border-width: 2px"></span>`,
    erased: `<span class="legend__key legend__key--round" style="background: transparent; border-style: dashed; border-color: var(--erased)"></span>`,
  };
  const label = {
    x: 'X error', z: 'Z error', y: 'Y error', clean: 'no error',
    defect: 'defect', xplaq: 'X plaquette', zplaq: 'Z plaquette',
    correction: 'correction', erased: 'erased',
  };
  return items
    .map((k) => `<span class="legend__item">${swatch[k]}${label[k]}</span>`)
    .join('');
}

export { ERROR };
