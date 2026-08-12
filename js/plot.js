/**
 * Minimal canvas plotting primitive.
 *
 * Deliberately small: axes, gridlines, point series with error bars, fitted
 * lines, and vertical markers. Every chart on the page goes through it so the
 * charts cannot drift apart stylistically.
 */

function palette() {
  const css = getComputedStyle(document.documentElement);
  const get = (name, fallback) => (css.getPropertyValue(name).trim() || fallback);
  return {
    ink: get('--ink', '#1c1d1f'),
    ink2: get('--ink-2', '#4a4c50'),
    ink3: get('--ink-3', '#7c7e83'),
    rule: get('--rule-soft', '#ded9cd'),
    surface: get('--surface', '#fffefb'),
  };
}

const MARGIN = { top: 22, right: 18, bottom: 44, left: 56 };

/**
 * Canvas has no idea what `var(--d3)` means, but the legend markup does. Series
 * colours are written in CSS custom property form so both sides name the same
 * token; this resolves them for the canvas at draw time.
 */
function resolveColor(value) {
  const match = /^var\((--[\w-]+)\)$/.exec(String(value).trim());
  if (!match) return value;
  return getComputedStyle(document.documentElement).getPropertyValue(match[1]).trim() || '#000';
}

export class Plot {
  /**
   * @param {HTMLCanvasElement} canvas
   * @param {object} options
   * @param {string} [options.xLabel]
   * @param {string} [options.yLabel]
   * @param {(v:number)=>string} [options.formatX]
   * @param {(v:number)=>string} [options.formatY]
   */
  constructor(canvas, options = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.options = {
      xLabel: '',
      yLabel: '',
      formatX: (v) => `${(v * 100).toFixed(0)}%`,
      formatY: (v) => `${(v * 100).toFixed(0)}%`,
      xTicks: 5,
      yTicks: 5,
      ...options,
    };
    this.colors = palette();
    this.data = null;

    this.#fit();
    this.resizeObserver = new ResizeObserver(() => {
      if (this.#fit() && this.data) this.render(this.data);
    });
    this.resizeObserver.observe(canvas);
  }

  #fit() {
    const dpr = window.devicePixelRatio || 1;
    const rect = this.canvas.getBoundingClientRect();
    const cssWidth = rect.width || this.canvas.clientWidth || 480;
    const cssHeight = cssWidth * Number(this.canvas.dataset.aspect || 0.66);

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

  /**
   * @param {object} data
   * @param {Array<{label:string,color:string,points?:Array<{x:number,y:number,lo?:number,hi?:number}>,line?:Array<{x:number,y:number}>}>} data.series
   * @param {Array<{x:number,label:string}>} [data.markers]
   * @param {[number,number]} [data.xRange]
   * @param {[number,number]} [data.yRange]
   * @param {string} [data.empty] message to show when there is nothing to plot
   */
  render(data) {
    this.data = data;
    const ctx = this.ctx;
    const { width, height } = this;
    ctx.clearRect(0, 0, width, height);

    const series = (data.series ?? []).map((s) => ({ ...s, color: resolveColor(s.color) }));
    const allPoints = series.flatMap((s) => [...(s.points ?? []), ...(s.line ?? [])]);

    if (!allPoints.length) {
      ctx.fillStyle = this.colors.ink3;
      ctx.font = '400 12px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(data.empty ?? 'No data yet', width / 2, height / 2);
      return;
    }

    const xs = allPoints.map((p) => p.x);
    const ys = allPoints.flatMap((p) => [p.y, p.hi ?? p.y, p.lo ?? p.y]);
    const xRange = data.xRange ?? [Math.min(...xs), Math.max(...xs)];
    let yMax = data.yRange?.[1] ?? Math.max(...ys);
    yMax = Math.min(1, Math.max(0.05, Math.ceil(yMax * 20) / 20));
    const yRange = [data.yRange?.[0] ?? 0, yMax];

    const plotW = width - MARGIN.left - MARGIN.right;
    const plotH = height - MARGIN.top - MARGIN.bottom;
    const sx = (v) => MARGIN.left + ((v - xRange[0]) / (xRange[1] - xRange[0] || 1)) * plotW;
    const sy = (v) => MARGIN.top + plotH - ((v - yRange[0]) / (yRange[1] - yRange[0] || 1)) * plotH;
    this.scaleX = sx;
    this.scaleY = sy;

    ctx.save();

    // Gridlines and y labels
    ctx.font = '400 10px "JetBrains Mono", monospace';
    ctx.fillStyle = this.colors.ink3;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (let i = 0; i <= this.options.yTicks; i++) {
      const v = yRange[0] + ((yRange[1] - yRange[0]) * i) / this.options.yTicks;
      const y = sy(v);
      ctx.beginPath();
      ctx.moveTo(MARGIN.left, y);
      ctx.lineTo(width - MARGIN.right, y);
      ctx.strokeStyle = this.colors.rule;
      ctx.lineWidth = i === 0 ? 1 : 0.6;
      ctx.stroke();
      ctx.fillText(this.options.formatY(v), MARGIN.left - 8, y);
    }

    // x labels
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    for (let i = 0; i <= this.options.xTicks; i++) {
      const v = xRange[0] + ((xRange[1] - xRange[0]) * i) / this.options.xTicks;
      ctx.fillText(this.options.formatX(v), sx(v), MARGIN.top + plotH + 9);
    }

    // Axis frame
    ctx.beginPath();
    ctx.moveTo(MARGIN.left, MARGIN.top);
    ctx.lineTo(MARGIN.left, MARGIN.top + plotH);
    ctx.lineTo(width - MARGIN.right, MARGIN.top + plotH);
    ctx.strokeStyle = this.colors.ink;
    ctx.lineWidth = 1.2;
    ctx.stroke();

    // Axis titles
    ctx.fillStyle = this.colors.ink2;
    ctx.font = '500 11px Inter, sans-serif';
    if (this.options.xLabel) {
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      ctx.fillText(this.options.xLabel, MARGIN.left + plotW / 2, height - 15);
    }
    if (this.options.yLabel) {
      ctx.save();
      ctx.translate(13, MARGIN.top + plotH / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      ctx.fillText(this.options.yLabel, 0, 0);
      ctx.restore();
    }

    // Vertical markers (threshold lines)
    for (const marker of data.markers ?? []) {
      if (marker.x < xRange[0] || marker.x > xRange[1]) continue;
      const x = sx(marker.x);
      ctx.save();
      ctx.setLineDash([4, 3]);
      ctx.strokeStyle = this.colors.ink2;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, MARGIN.top);
      ctx.lineTo(x, MARGIN.top + plotH);
      ctx.stroke();
      ctx.restore();

      ctx.fillStyle = this.colors.ink;
      ctx.font = '600 10px "JetBrains Mono", monospace';
      ctx.textBaseline = 'bottom';
      ctx.textAlign = x > MARGIN.left + plotW * 0.7 ? 'right' : 'left';
      ctx.fillText(marker.label, x + (ctx.textAlign === 'right' ? -4 : 4), MARGIN.top - 4);
    }

    // Fitted lines first, points on top
    for (const s of series) {
      if (!s.line?.length) continue;
      ctx.beginPath();
      s.line.forEach((p, i) => (i ? ctx.lineTo(sx(p.x), sy(p.y)) : ctx.moveTo(sx(p.x), sy(p.y))));
      ctx.strokeStyle = s.color;
      ctx.lineWidth = 1.8;
      ctx.globalAlpha = 0.85;
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    for (const s of series) {
      for (const p of s.points ?? []) {
        if (p.lo != null && p.hi != null && p.hi - p.lo > 1e-6) {
          ctx.beginPath();
          ctx.moveTo(sx(p.x), sy(Math.min(p.hi, yRange[1])));
          ctx.lineTo(sx(p.x), sy(Math.max(p.lo, yRange[0])));
          ctx.strokeStyle = s.color;
          ctx.lineWidth = 1;
          ctx.globalAlpha = 0.55;
          ctx.stroke();
          ctx.globalAlpha = 1;
        }
        ctx.beginPath();
        ctx.arc(sx(p.x), sy(Math.min(p.y, yRange[1])), 3.6, 0, Math.PI * 2);
        ctx.fillStyle = s.color;
        ctx.fill();
        ctx.strokeStyle = this.colors.surface;
        ctx.lineWidth = 1.2;
        ctx.stroke();
      }
    }

    ctx.restore();
  }

  destroy() { this.resizeObserver.disconnect(); }
}

/** Legend markup matching the plot series colours. */
export function plotLegend(series) {
  return series
    .map((s) => `<span class="legend__item"><span class="legend__key legend__key--line" style="border-color:${s.color}"></span>${s.label}</span>`)
    .join('');
}
