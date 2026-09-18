/**
 * Strip chart for a streamed Monte Carlo run: the running logical error rate
 * against shots so far, with its Wilson band, so the reader watches the
 * estimate settle rather than waiting for a number to appear.
 */

/** Read the palette out of CSS so the stylesheet stays the single source. */
function palette() {
  const css = getComputedStyle(document.documentElement);
  const get = (name, fallback) => (css.getPropertyValue(name).trim() || fallback);
  return {
    ink: get('--ink', '#1c1d1f'), ink2: get('--ink-2', '#4a4c50'), ink3: get('--ink-3', '#6b6d71'),
    rule: get('--rule-soft', '#ded9cd'), band: get('--defect-soft', '#fbeec6'), surface: get('--surface', '#fffefb'),
  };
}

const M = { top: 14, right: 16, bottom: 30, left: 50 };

export class Meter {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.colors = palette();
    this.data = null;
    this.#fit();
    new ResizeObserver(() => { if (this.#fit() && this.data) this.render(this.data); }).observe(canvas);
  }

  #fit() {
    const dpr = window.devicePixelRatio || 1;
    const rect = this.canvas.getBoundingClientRect();
    const w = rect.width || 480;
    const h = w * Number(this.canvas.dataset.aspect || 0.42);
    if (this.width === w && this.height === h) return false;
    this.canvas.style.height = `${h}px`;
    this.canvas.width = Math.round(w * dpr);
    this.canvas.height = Math.round(h * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.width = w;
    this.height = h;
    return true;
  }

  /**
   * @param {{total: number, samples: Array<{done: number, rate: number, lo: number, hi: number}>,
   *          final?: boolean, empty?: string}} data
   */
  render(data) {
    this.data = data;
    const { ctx, width, height, colors } = this;
    ctx.clearRect(0, 0, width, height);
    const s = data.samples ?? [];
    const pw = width - M.left - M.right;
    const ph = height - M.top - M.bottom;

    if (!s.length) {
      ctx.fillStyle = colors.ink3;
      ctx.font = '400 12px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(data.empty ?? 'Run to watch the estimate settle.', width / 2, height / 2);
      this.canvas.setAttribute('aria-label', data.empty ?? 'No run yet.');
      return;
    }

    let yMax = Math.max(0.02, ...s.map((p) => p.hi));
    yMax = Math.min(1, Math.ceil(yMax * 50) / 50);
    const sx = (v) => M.left + (v / data.total) * pw;
    const sy = (v) => M.top + ph - (v / yMax) * ph;

    ctx.font = '400 10px "JetBrains Mono", monospace';
    ctx.fillStyle = colors.ink3;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (let i = 0; i <= 4; i++) {
      const v = (yMax * i) / 4;
      const y = sy(v);
      ctx.beginPath();
      ctx.moveTo(M.left, y);
      ctx.lineTo(width - M.right, y);
      ctx.strokeStyle = colors.rule;
      ctx.lineWidth = i ? 0.6 : 1;
      ctx.stroke();
      ctx.fillText(`${(v * 100).toFixed(v < 0.1 ? 1 : 0)}%`, M.left - 6, y);
    }
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    for (let i = 0; i <= 4; i++) {
      const v = (data.total * i) / 4;
      ctx.fillText(Math.round(v).toLocaleString(), sx(v), M.top + ph + 6);
    }
    ctx.fillStyle = colors.ink2;
    ctx.font = '500 11px Inter, sans-serif';
    ctx.fillText('shots', M.left + pw / 2, height - 13);

    // The interval, as a band the estimate sits inside.
    ctx.beginPath();
    s.forEach((p, i) => (i ? ctx.lineTo(sx(p.done), sy(p.hi)) : ctx.moveTo(sx(p.done), sy(p.hi))));
    for (let i = s.length - 1; i >= 0; i--) ctx.lineTo(sx(s[i].done), sy(s[i].lo));
    ctx.closePath();
    ctx.fillStyle = colors.band;
    ctx.fill();

    // The running estimate.
    ctx.beginPath();
    s.forEach((p, i) => (i ? ctx.lineTo(sx(p.done), sy(p.rate)) : ctx.moveTo(sx(p.done), sy(p.rate))));
    ctx.strokeStyle = colors.ink;
    ctx.lineWidth = 1.6;
    ctx.stroke();
    const last = s[s.length - 1];
    ctx.beginPath();
    ctx.arc(sx(last.done), sy(last.rate), 3, 0, Math.PI * 2);
    ctx.fillStyle = colors.ink;
    ctx.fill();

    if (data.final) {
      // The value beside its point, so a stopped run labels where it stopped.
      ctx.font = '600 10px "JetBrains Mono", monospace';
      ctx.textBaseline = 'middle';
      ctx.fillStyle = colors.ink;
      const label = `${(last.rate * 100).toFixed(2)}%`;
      const fits = sx(last.done) + 8 + ctx.measureText(label).width <= width - M.right;
      ctx.textAlign = fits ? 'left' : 'right';
      ctx.fillText(label, sx(last.done) + (fits ? 8 : -8), sy(last.rate));
    }

    this.canvas.setAttribute('aria-label',
      `Logical error rate estimate after ${last.done.toLocaleString()} of ${data.total.toLocaleString()} shots: `
      + `${(last.rate * 100).toFixed(2)}%, 95% interval ${(last.lo * 100).toFixed(2)}% to ${(last.hi * 100).toFixed(2)}%.`);
  }
}
