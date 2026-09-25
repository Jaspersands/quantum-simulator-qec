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

/**
 * Gridlines for a logarithmic axis: whole decades from the one at or below
 * `min` to the one at or above `max`, with 2× and 5× between them as minor
 * lines. Pure, so it can be tested without a canvas.
 */
export function logTicks(min, max) {
  const clean = (v) => Number(v.toPrecision(12));
  const lo = clean(10 ** Math.floor(Math.log10(min)));
  let hi = clean(10 ** Math.ceil(Math.log10(max)));
  if (hi <= lo) hi = clean(lo * 10);
  const ticks = [];
  for (let decade = lo; decade <= hi * (1 + 1e-9); decade = clean(decade * 10)) {
    ticks.push({ v: decade, major: true });
    for (const m of [2, 5]) if (clean(decade * m) < hi) ticks.push({ v: clean(decade * m), major: false });
  }
  return { lo, hi, ticks };
}

export class Plot {
  /**
   * @param {HTMLCanvasElement} canvas
   * @param {object} options
   * @param {string} [options.xLabel]
   * @param {string} [options.yLabel]
   * @param {(v:number)=>string} [options.formatX]
   * @param {(v:number)=>string} [options.formatY]
   * @param {boolean} [options.yLog] a logarithmic y axis, gridded by decade
   * @param {number[]} [options.xTickValues] explicit x ticks, instead of evenly spaced ones
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
      yLog: false,
      xTickValues: null,
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

    // Canvas is opaque to assistive technology; keep a text summary in sync.
    this.canvas.setAttribute('role', 'img');

    if (!allPoints.length) {
      ctx.fillStyle = this.colors.ink3;
      ctx.font = '400 12px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(data.empty ?? 'No data yet', width / 2, height / 2);
      this.canvas.setAttribute('aria-label', data.empty ?? 'Chart with no data yet.');
      return;
    }

    const xs = allPoints.map((p) => p.x);
    const ys = allPoints.flatMap((p) => [p.y, p.hi ?? p.y, p.lo ?? p.y]);
    const xRange = data.xRange ?? [Math.min(...xs), Math.max(...xs)];
    const log = this.options.yLog;
    let yRange;
    let yTicks;
    if (log && !ys.some((v) => v > 0) && !data.yRange) {
      ctx.fillStyle = this.colors.ink3;
      ctx.font = '400 12px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(data.empty ?? 'Nothing above zero to plot', width / 2, height / 2);
      this.canvas.setAttribute('aria-label', data.empty ?? 'Chart with nothing above zero to plot.');
      return;
    }
    if (log) {
      const positive = ys.filter((v) => v > 0);
      const auto = logTicks(Math.min(...positive), Math.max(...positive));
      yRange = data.yRange ?? [auto.lo, auto.hi];
      yTicks = logTicks(yRange[0], yRange[1]).ticks.filter((t) => t.v >= yRange[0] && t.v <= yRange[1]);
    } else {
      let yMax = data.yRange?.[1] ?? Math.max(...ys);
      yMax = Math.min(1, Math.max(0.05, Math.ceil(yMax * 20) / 20));
      yRange = [data.yRange?.[0] ?? 0, yMax];
      yTicks = Array.from({ length: this.options.yTicks + 1 }, (_, i) => ({
        v: yRange[0] + ((yRange[1] - yRange[0]) * i) / this.options.yTicks,
        major: true,
      }));
    }
    const ty = log ? (v) => Math.log10(Math.max(v, 1e-300)) : (v) => v;

    const plotW = width - MARGIN.left - MARGIN.right;
    const plotH = height - MARGIN.top - MARGIN.bottom;
    const sx = (v) => MARGIN.left + ((v - xRange[0]) / (xRange[1] - xRange[0] || 1)) * plotW;
    const sy = (v) => MARGIN.top + plotH - ((ty(v) - ty(yRange[0])) / (ty(yRange[1]) - ty(yRange[0]) || 1)) * plotH;
    this.scaleX = sx;
    this.scaleY = sy;

    ctx.save();

    // Gridlines and y labels
    ctx.font = '400 10px "JetBrains Mono", monospace';
    ctx.fillStyle = this.colors.ink3;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    yTicks.forEach(({ v, major }, i) => {
      const y = sy(v);
      ctx.beginPath();
      ctx.moveTo(MARGIN.left, y);
      ctx.lineTo(width - MARGIN.right, y);
      ctx.strokeStyle = this.colors.rule;
      ctx.globalAlpha = major ? 1 : 0.55;
      ctx.lineWidth = i === 0 ? 1 : 0.6;
      ctx.stroke();
      ctx.globalAlpha = 1;
      ctx.fillText(this.options.formatY(v), MARGIN.left - 8, y);
    });

    // x labels
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    const xTickValues = this.options.xTickValues
      ?? Array.from({ length: this.options.xTicks + 1 }, (_, i) => xRange[0] + ((xRange[1] - xRange[0]) * i) / this.options.xTicks);
    for (const v of xTickValues) ctx.fillText(this.options.formatX(v), sx(v), MARGIN.top + plotH + 9);

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

    const described = series
      .filter((s) => s.points?.length)
      .map((s) => {
        const ys = s.points.map((p) => p.y);
        return `${s.label} from ${this.options.formatY(Math.min(...ys))}`
          + ` to ${this.options.formatY(Math.max(...ys))}`;
      });
    const markers = (data.markers ?? []).map((m) => m.label).join(', ');
    this.canvas.setAttribute('aria-label',
      `${this.options.yLabel || 'Value'} against ${this.options.xLabel || 'x'}. `
      + (described.length ? `${described.join('; ')}.` : '')
      + (markers ? ` ${markers}.` : ''));
  }

  destroy() { this.resizeObserver.disconnect(); }
}

/**
 * Legend markup matching the plot series colours.
 *
 * Series colours are written as CSS custom property references so the canvas
 * and the legend name the same token. Here that reference becomes a modifier
 * class rather than an inline style, keeping every colour in the stylesheet.
 */
export function plotLegend(series) {
  return series.map((s) => {
    const token = /^var\(--([\w-]+)\)$/.exec(String(s.color).trim())?.[1];
    const mod = token ? ` legend__key--${token}` : '';
    return `<span class="legend__item">`
      + `<span class="legend__key legend__key--line${mod}"></span>${s.label}</span>`;
  }).join('');
}
