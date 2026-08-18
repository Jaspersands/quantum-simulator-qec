/**
 * The logical channel, drawn as what it does to the Bloch sphere.
 *
 * Under Pauli noise and a Pauli decoder the effective logical channel is a
 * Pauli channel, which shrinks each Bloch axis independently. So the honest
 * picture is not an arrow — it is the unit sphere collapsing to an ellipsoid
 * with semi-axes (λx, λy, λz), the diagonal of the channel's Pauli transfer
 * matrix. A noiseless channel leaves the sphere alone; noise the decoder cannot
 * undo squashes it toward the centre, and a fully depolarizing channel
 * collapses it to a point.
 *
 * This replaces an earlier Bloch-vector arrow that plotted three separately
 * measured survival probabilities as if they were the components of one state.
 */

function palette() {
  const css = getComputedStyle(document.documentElement);
  const get = (n, f) => (css.getPropertyValue(n).trim() || f);
  return {
    ink: get('--ink', '#1c1d1f'),
    ink3: get('--ink-3', '#6b6d71'),
    rule: get('--rule-soft', '#ded9cd'),
    surface: get('--surface', '#fffefb'),
    sunk: get('--surface-sunk', '#faf8f2'),
    x: get('--x', '#2f5fd0'),
    y: get('--y', '#7b4bb5'),
    z: get('--z', '#c0392b'),
  };
}

export class ChannelView {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.colors = palette();
    this.factors = null;
    this.#fit();
    this.resizeObserver = new ResizeObserver(() => { if (this.#fit()) this.draw(); });
    this.resizeObserver.observe(canvas);
  }

  #fit() {
    const dpr = window.devicePixelRatio || 1;
    const rect = this.canvas.getBoundingClientRect();
    const size = Math.max(200, Math.min(rect.width || 260, 300));
    if (this.size === size) return false;
    this.canvas.style.height = `${size}px`;
    this.canvas.width = Math.round(size * dpr);
    this.canvas.height = Math.round(size * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.size = size;
    return true;
  }

  /** @param {[number,number,number]|null} factors PTM diagonal (λx, λy, λz) */
  set(factors) {
    this.factors = factors;
    this.draw();
  }

  /** Axonometric projection: z up the page, x toward the viewer, y to the right. */
  #project(x, y, z, r) {
    const c = this.size / 2;
    return { x: c + y * r - x * r * 0.45, y: c - z * r + x * r * 0.25 };
  }

  /** Trace the ellipse cut by holding one axis at zero. */
  #traceRing(axis, r, sx, sy, sz) {
    const ctx = this.ctx;
    ctx.beginPath();
    for (let i = 0; i <= 64; i++) {
      const t = (i / 64) * Math.PI * 2;
      const v = axis === 'z' ? [Math.cos(t) * sx, Math.sin(t) * sy, 0]
              : axis === 'y' ? [Math.cos(t) * sx, 0, Math.sin(t) * sz]
              :                [0, Math.cos(t) * sy, Math.sin(t) * sz];
      const p = this.#project(v[0], v[1], v[2], r);
      if (i === 0) ctx.moveTo(p.x, p.y); else ctx.lineTo(p.x, p.y);
    }
    ctx.closePath();
  }

  draw() {
    const ctx = this.ctx;
    const c = this.colors;
    const centre = this.size / 2;
    const r = this.size * 0.34;

    ctx.clearRect(0, 0, this.size, this.size);

    // The unit sphere, for reference.
    ctx.beginPath();
    ctx.arc(centre, centre, r, 0, Math.PI * 2);
    ctx.fillStyle = c.sunk;
    ctx.fill();
    ctx.strokeStyle = c.rule;
    ctx.lineWidth = 1;
    ctx.stroke();

    // Axes.
    for (const [vec, colour, label] of [
      [[0, 0, 1], c.z, 'Z'], [[0, 1, 0], c.y, 'Y'], [[1, 0, 0], c.x, 'X'],
    ]) {
      const a = this.#project(vec[0] * 1.14, vec[1] * 1.14, vec[2] * 1.14, r);
      const b = this.#project(-vec[0] * 1.14, -vec[1] * 1.14, -vec[2] * 1.14, r);
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.strokeStyle = colour;
      ctx.globalAlpha = 0.45;
      ctx.stroke();
      ctx.globalAlpha = 1;

      const l = this.#project(vec[0] * 1.3, vec[1] * 1.3, vec[2] * 1.3, r);
      ctx.fillStyle = colour;
      ctx.font = '600 9px "JetBrains Mono", monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(`${label}ₗ`, l.x, l.y);
    }

    if (!this.factors) {
      ctx.fillStyle = c.ink3;
      ctx.font = '400 10px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('no estimate yet', centre, this.size - 8);
      this.canvas.setAttribute('aria-label', 'Logical channel not yet estimated.');
      return;
    }

    // The image of the sphere: three principal rings of the ellipsoid.
    const [lx, ly, lz] = this.factors.map((v) => Math.abs(v));
    for (const axis of ['z', 'y', 'x']) {
      this.#traceRing(axis, r, lx, ly, lz);
      ctx.strokeStyle = c.ink;
      ctx.lineWidth = axis === 'z' ? 2 : 1.2;
      ctx.globalAlpha = axis === 'z' ? 0.9 : 0.5;
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    // Where |0>_L lands.
    const tip = this.#project(0, 0, this.factors[2], r);
    ctx.beginPath();
    ctx.moveTo(centre, centre);
    ctx.lineTo(tip.x, tip.y);
    ctx.strokeStyle = c.ink;
    ctx.lineWidth = 2.5;
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(tip.x, tip.y, 4.5, 0, Math.PI * 2);
    ctx.fillStyle = c.ink;
    ctx.fill();
    ctx.strokeStyle = c.surface;
    ctx.lineWidth = 1.5;
    ctx.stroke();

    ctx.fillStyle = c.ink3;
    ctx.font = '500 10px "JetBrains Mono", monospace';
    ctx.textAlign = 'center';
    ctx.fillText(`λ = (${lx.toFixed(2)}, ${ly.toFixed(2)}, ${lz.toFixed(2)})`, centre, this.size - 8);

    this.canvas.setAttribute('role', 'img');
    this.canvas.setAttribute('aria-label',
      `Logical channel shown as the Bloch sphere's image. Axis shrink factors: `
      + `X ${lx.toFixed(3)}, Y ${ly.toFixed(3)}, Z ${lz.toFixed(3)}. `
      + `A noiseless channel would leave all three at 1.`);
  }

  destroy() { this.resizeObserver.disconnect(); }
}
