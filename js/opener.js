/**
 * The live lattice at the top of the page.
 *
 * A distance-d rotated patch on the main-thread engine, taking Poisson noise
 * and pointer noise, decoded by exact MWPM every round, drawn as an ink
 * diagram on the paper. It has its own renderer: the figures' LatticeView is
 * built for small patches with hover and keyboard cursors, and this one needs
 * neither, but does need per-frame animation of the decoder's chains.
 */

import { Session, DECODER, ERROR, STAB } from './engine.js';
import { poisson, footprintRate, layoutFor, chainsFromCorrection, pickPauli } from './opener-math.js';
import { $ } from './dom.js';

const C = {
  PAD: 18,
  AMBIENT: 0.8,                                           // errors / s over the patch
  CURSOR_PEAK: 6, CURSOR_SIGMA: 1.1,                      // errors / s / qubit at the pointer; cells
  PAULI: [['X', 0.45], ['Z', 0.45], ['Y', 0.10]],
  CURSOR_PAULI: [['X', 0.8], ['Z', 0.15], ['Y', 0.05]],   // bit-flip biased, so a sweep builds a chain
  MAX_PENDING: 90,
  ROUND: 1400, TICK: 50,                                  // ms
  T_ERR: 180, T_DEF: 220, T_CHAIN: 380, T_HOLD: 260, T_FADE: 320, T_FLASH: 600,
};

/** Code distance for the space the patch has: the shorter side of the canvas. */
function distanceFor(side) {
  if (side < 320) return 9;
  if (side < 420) return 11;
  return 15;
}

/** Read the palette out of CSS so the stylesheet stays the single source. */
function palette() {
  const css = getComputedStyle(document.documentElement);
  const get = (name, fallback) => (css.getPropertyValue(name).trim() || fallback);
  return {
    surface: get('--surface', '#fffefb'), ink: get('--ink', '#1c1d1f'), ink3: get('--ink-3', '#6b6d71'),
    rule: get('--rule-soft', '#ded9cd'), x: get('--x', '#2f5fd0'), xSoft: get('--x-soft', '#d3def6'),
    z: get('--z', '#c0392b'), zSoft: get('--z-soft', '#f4d6d1'), y: get('--y', '#7b4bb5'),
    defect: get('--defect', '#c98a06'), defectSoft: get('--defect-soft', '#fbeec6'),
    ok: get('--ok', '#2c7a4b'), fail: get('--fail', '#b23a2e'),
  };
}

export function initOpener(root, instance) {
  const canvas = $('[data-opener-canvas]', root);
  if (!canvas) return null;
  const ctx = canvas.getContext('2d');
  const out = {
    d: $('[data-opener-d]', root), rounds: $('[data-opener-rounds]', root), phys: $('[data-opener-phys]', root),
    logic: $('[data-opener-logic]', root), ms: $('[data-opener-ms]', root),
  };
  const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const colors = palette();
  const now = () => performance.now();

  const S = {
    session: null, L: null, width: 0, height: 0, dpr: 1, plaquettes: [],
    pending: [], lit: new Map(), anim: null, flashAt: -Infinity,
    rounds: 0, phys: 0, logical: 0, decodeMs: null,
    pointer: { x: 0, y: 0, t: -Infinity },
    running: false, tick: 0, raf: 0, dirty: true, lastTick: 0, nextRound: 0, onScreen: true,
  };

  /* -- sizing ------------------------------------------------------------ */

  function rebuild(d) {
    S.session?.free();
    S.session = new Session(instance, { d, rounds: 1 });
    S.pending = []; S.lit = new Map(); S.anim = null;
    out.d.textContent = String(d);
    canvas.setAttribute('aria-label', canvas.getAttribute('aria-label').replace(/distance-\d+/, `distance-${d}`));
  }

  function fit() {
    const rect = canvas.getBoundingClientRect();
    const w = rect.width || 800, h = rect.height || 420;
    S.dpr = Math.min(2, devicePixelRatio || 1);
    canvas.width = Math.round(w * S.dpr); canvas.height = Math.round(h * S.dpr);
    ctx.setTransform(S.dpr, 0, 0, S.dpr, 0, 0);
    S.width = w; S.height = h;
    const d = distanceFor(Math.min(w, h));
    if (!S.session || S.session.d !== d) rebuild(d);
    S.L = layoutFor(w, h, d, C.PAD);
    S.plaquettes = S.session.stabilizers.map((st) => plaquette(st));
    S.dirty = true;
  }

  const at = (gx, gy) => ({ x: S.L.originX + gx * S.L.unit, y: S.L.originY + gy * S.L.unit });

  /** A check's plaquette in grid units, corners sorted by angle so it never self-intersects. */
  function plaquette(st) {
    const pts = S.session.support.get(st.idx).map((q) => ({ x: S.session.dataQubits[q].x, y: S.session.dataQubits[q].y }));
    pts.sort((a, b) => Math.atan2(a.y - st.y, a.x - st.x) - Math.atan2(b.y - st.y, b.x - st.x));
    return { st, pts };
  }

  function tracePlaquette(p) {
    const { st, pts } = p;
    ctx.beginPath();
    if (pts.length >= 3) {
      pts.forEach((g, i) => { const q = at(g.x, g.y); if (i) ctx.lineTo(q.x, q.y); else ctx.moveTo(q.x, q.y); });
      ctx.closePath();
    } else if (pts.length === 2) {
      // Boundary check: a half-disc bulging away from the patch, as LatticeView draws it.
      const a = at(pts[0].x, pts[0].y), b = at(pts[1].x, pts[1].y), c = at(st.x, st.y);
      const mid = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
      const outv = { x: c.x - mid.x, y: c.y - mid.y }, len = Math.hypot(outv.x, outv.y) || 1;
      const bulge = Math.hypot(b.x - a.x, b.y - a.y) * 0.55;
      ctx.moveTo(a.x, a.y);
      ctx.quadraticCurveTo(mid.x + (outv.x / len) * bulge, mid.y + (outv.y / len) * bulge, b.x, b.y);
      ctx.closePath();
    }
  }

  /* -- noise ------------------------------------------------------------- */

  function inject(q, pauli, t) {
    if (S.pending.length >= C.MAX_PENDING) return 0;
    if (pauli === 'X' || pauli === 'Y') S.session.toggleError(q, ERROR.X);
    if (pauli === 'Z' || pauli === 'Y') S.session.toggleError(q, ERROR.Z);
    S.pending.push({ q, pauli, t0: t });
    return 1;
  }

  function refreshLit(t) {
    const { defects } = S.session.read();
    for (let i = 0; i < defects.length; i++) {
      if (defects[i]) { if (!S.lit.has(i)) S.lit.set(i, t); } else S.lit.delete(i);
    }
  }

  function tick() {
    if (!S.running) return;
    const t = now(), dt = Math.min(0.2, (t - S.lastTick) / 1000);
    S.lastTick = t;
    let n = 0;
    const k = poisson(C.AMBIENT * dt);
    for (let i = 0; i < k; i++) n += inject(Math.floor(Math.random() * S.session.numData), pickPauli(Math.random, C.PAULI), t);
    if (t - S.pointer.t < 400) {
      const gx = (S.pointer.x - S.L.originX) / S.L.unit, gy = (S.pointer.y - S.L.originY) / S.L.unit;
      for (const q of S.session.dataQubits) {
        const dist = Math.hypot((q.x - gx) / 2, (q.y - gy) / 2);
        const m = poisson(footprintRate(dist, C.CURSOR_SIGMA, C.CURSOR_PEAK) * dt);
        for (let j = 0; j < m; j++) n += inject(q.idx, pickPauli(Math.random, C.CURSOR_PAULI), t);
      }
      S.dirty = true;
    }
    if (n) { S.phys += n; refreshLit(t); S.dirty = true; }
    if (t >= S.nextRound) round(t);
    if (S.dirty || S.anim || t - S.flashAt < C.T_FLASH) frame();
    S.tick = setTimeout(tick, C.TICK);
  }

  function round(t) {
    S.rounds++;
    if (S.pending.length) {
      const t0 = now();
      const { failed, correctionX, correctionZ } = S.session.decode(DECODER.MWPM);
      S.decodeMs = now() - t0;
      const chains = chainsFromCorrection(S.session, correctionX, correctionZ);
      S.anim = { t0: t, chains, errors: S.pending, lit: [...S.lit.keys()], failed };
      S.session.clearErrors();
      S.pending = []; S.lit = new Map();
      if (failed) {
        S.logical++; S.flashAt = t;
        out.logic.classList.add('flare');
        setTimeout(() => out.logic.classList.remove('flare'), C.T_FLASH);
      }
      S.dirty = true;
    }
    S.nextRound = t + C.ROUND * (0.85 + 0.3 * Math.random());
    readout();
  }

  function readout() {
    out.rounds.textContent = S.rounds.toLocaleString();
    out.phys.textContent = S.phys.toLocaleString();
    out.logic.textContent = S.logical.toLocaleString();
    out.ms.textContent = S.decodeMs == null ? '·' : `${S.decodeMs < 1 ? S.decodeMs.toFixed(2) : S.decodeMs.toFixed(1)} ms`;
  }

  /* -- drawing ----------------------------------------------------------- */

  const pauliColor = (p) => (p === 'X' ? colors.x : p === 'Z' ? colors.z : colors.y);
  const clamp = (v, a, b) => Math.min(b, Math.max(a, v));

  function drawStatic() {
    for (const p of S.plaquettes) {
      tracePlaquette(p);
      ctx.fillStyle = p.st.type === STAB.X ? colors.xSoft : colors.zSoft;
      ctx.fill();
      ctx.strokeStyle = colors.rule; ctx.lineWidth = 1; ctx.stroke();
    }
    const r = Math.max(2.2, Math.min(4, S.L.unit * 0.28));
    for (const q of S.session.dataQubits) {
      const c = at(q.x, q.y);
      ctx.beginPath(); ctx.arc(c.x, c.y, r, 0, Math.PI * 2);
      ctx.fillStyle = colors.surface; ctx.fill();
      ctx.strokeStyle = colors.ink3; ctx.lineWidth = 1; ctx.stroke();
    }
  }

  function drawLit(idx, alpha) {
    const p = S.plaquettes[idx];
    ctx.globalAlpha = alpha;
    tracePlaquette(p);
    ctx.fillStyle = colors.defectSoft; ctx.fill();
    ctx.strokeStyle = colors.defect; ctx.lineWidth = 1.5; ctx.stroke();
    const c = at(p.st.x, p.st.y);
    ctx.beginPath(); ctx.arc(c.x, c.y, Math.max(2.5, S.L.unit * 0.3), 0, Math.PI * 2);
    ctx.fillStyle = colors.defect; ctx.fill();
    ctx.globalAlpha = 1;
  }

  function drawError(e, grow, alpha) {
    const q = S.session.dataQubits[e.q], c = at(q.x, q.y);
    const r = Math.max(3, S.L.unit * (0.3 + 0.28 * grow));
    ctx.globalAlpha = alpha;
    ctx.beginPath(); ctx.arc(c.x, c.y, r, 0, Math.PI * 2);
    ctx.fillStyle = pauliColor(e.pauli); ctx.fill();
    if (r >= 6) {
      // The letter is the second, non-colour channel for the Pauli type, as in the figures.
      ctx.fillStyle = colors.surface;
      ctx.font = `700 ${Math.round(r * 1.15)}px "JetBrains Mono", monospace`;
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(e.pauli, c.x, c.y);
    }
    ctx.globalAlpha = 1;
  }

  function strokePartial(pts, frac) {
    const seg = []; let total = 0;
    for (let i = 1; i < pts.length; i++) { const l = Math.hypot(pts[i].x - pts[i - 1].x, pts[i].y - pts[i - 1].y); seg.push(l); total += l; }
    let left = total * frac;
    ctx.beginPath(); ctx.moveTo(pts[0].x, pts[0].y);
    for (let i = 1; i < pts.length && left > 0; i++) {
      if (left >= seg[i - 1]) { ctx.lineTo(pts[i].x, pts[i].y); left -= seg[i - 1]; }
      else { const f = left / seg[i - 1]; ctx.lineTo(pts[i - 1].x + (pts[i].x - pts[i - 1].x) * f, pts[i - 1].y + (pts[i].y - pts[i - 1].y) * f); left = 0; }
    }
    ctx.stroke();
  }

  function drawChains(chains, frac, alpha) {
    ctx.save();
    ctx.globalAlpha = alpha; ctx.lineCap = 'round'; ctx.lineJoin = 'round';
    ctx.strokeStyle = colors.ok; ctx.lineWidth = Math.max(2, S.L.unit * 0.22);
    for (const ch of chains) {
      const pts = ch.qubits.map((q) => at(S.session.dataQubits[q].x, S.session.dataQubits[q].y));
      if (pts.length === 1) { ctx.beginPath(); ctx.arc(pts[0].x, pts[0].y, Math.max(5, S.L.unit * 0.45), 0, Math.PI * 2); ctx.stroke(); }
      else strokePartial(pts, frac);
    }
    ctx.restore();
  }

  function render(t) {
    ctx.clearRect(0, 0, S.width, S.height);
    S.dirty = false;
    drawStatic();
    for (const [idx, t0] of S.lit) drawLit(idx, clamp((t - t0) / C.T_DEF, 0, 1));
    for (const e of S.pending) drawError(e, clamp((t - e.t0) / C.T_ERR, 0, 1), 1);
    const a = S.anim;
    if (a) {
      const dt = t - a.t0, total = C.T_CHAIN + C.T_HOLD + C.T_FADE;
      if (dt >= total) S.anim = null;
      else {
        const alpha = dt < C.T_CHAIN + C.T_HOLD ? 1 : 1 - (dt - C.T_CHAIN - C.T_HOLD) / C.T_FADE;
        for (const idx of a.lit) drawLit(idx, alpha);
        for (const e of a.errors) drawError(e, 1, alpha);
        drawChains(a.chains, clamp(dt / C.T_CHAIN, 0, 1), alpha);
      }
    }
    const flash = t - S.flashAt;
    if (flash >= 0 && flash < C.T_FLASH) {
      ctx.save();
      ctx.globalAlpha = 0.5 * (1 - flash / C.T_FLASH);
      ctx.strokeStyle = colors.fail; ctx.lineWidth = 6;
      ctx.strokeRect(S.L.originX - 3, S.L.originY - 3, S.L.size + 6, S.L.size + 6);
      ctx.restore();
    }
    if (t - S.pointer.t < 400) {
      const r = 2.4 * C.CURSOR_SIGMA * 2 * S.L.unit;
      const g = ctx.createRadialGradient(S.pointer.x, S.pointer.y, 0, S.pointer.x, S.pointer.y, r);
      g.addColorStop(0, colors.defectSoft); g.addColorStop(1, 'rgba(0,0,0,0)');
      ctx.save(); ctx.globalAlpha = 0.45; ctx.fillStyle = g;
      ctx.fillRect(S.pointer.x - r, S.pointer.y - r, 2 * r, 2 * r);
      ctx.restore();
    }
  }

  function frame() {
    if (S.raf) return;
    S.raf = requestAnimationFrame((t) => {
      S.raf = 0;
      render(t);
      if (S.anim || t - S.flashAt < C.T_FLASH || t - S.pointer.t < 400) frame();
    });
  }

  /* -- reduced motion: one composed frame ---------------------------------- */

  function still() {
    root.querySelector('.opener')?.classList.add('opener--still');
    const d = S.session.d, row = Math.floor(d / 2), col = Math.floor(d / 3);
    const t = now();
    for (let c = col; c < col + 3; c++) inject(row * d + c, 'X', t);              // a short X chain
    const top = Math.max(0, row - 5);
    for (let r = top; r < top + 2; r++) inject(r * d + col + Math.floor(d / 2), 'Z', t);   // and a Z pair above it
    refreshLit(t);
    const { correctionX, correctionZ } = S.session.decode(DECODER.MWPM);
    const anim = { t0: t - C.T_CHAIN, chains: chainsFromCorrection(S.session, correctionX, correctionZ), errors: S.pending, lit: [...S.lit.keys()], failed: false };
    S.session.clearErrors();
    S.pending = []; S.lit = new Map();
    S.anim = anim;
    render(t);                                       // one frame, chains drawn in full and never advanced
    S.anim = null;
  }

  /* -- lifecycle ----------------------------------------------------------- */

  function start() {
    if (S.running || reduced || document.hidden || !S.onScreen) return;
    S.running = true; S.lastTick = now(); S.nextRound = now() + C.ROUND;
    clearTimeout(S.tick); S.tick = setTimeout(tick, C.TICK);
  }
  function stop() { S.running = false; clearTimeout(S.tick); }

  fit();
  readout();
  if (reduced) {
    still();
    let rt = 0;
    addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(() => { fit(); still(); }, 150); });
  } else {
    canvas.addEventListener('pointermove', (e) => {
      const r = canvas.getBoundingClientRect();
      S.pointer = { x: e.clientX - r.left, y: e.clientY - r.top, t: now() };
      frame();
    }, { passive: true });
    canvas.addEventListener('pointerleave', () => { S.pointer.t = -Infinity; });
    new IntersectionObserver(([entry]) => { S.onScreen = entry.isIntersecting; if (S.onScreen) start(); else stop(); }, { threshold: 0.05 }).observe(canvas);
    document.addEventListener('visibilitychange', () => { if (document.hidden) stop(); else start(); });
    let rt = 0;
    addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(() => { fit(); frame(); }, 150); });
    start();
    frame();
  }

  return { stop, state: S };
}
