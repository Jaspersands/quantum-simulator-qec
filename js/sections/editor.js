/**
 * Section 17 — the editor: a circuit on a grid of qubits, gates placed layer by layer, Pauli
 * markers that propagate live to the detectors and observables they flip, and Stim text and
 * Crumble URLs in and out. The model is js/editor.js; this file is the DOM.
 *
 * Two views share one model: the grid (the qubits at their coordinates, the current layer's
 * gates drawn on them; click a qubit to place the chosen gate, a two-qubit gate takes two
 * clicks) and the timeline (a row per qubit, a column per layer, every gate and every marker's
 * Pauli as it moves). The text panel holds the Stim text, rewritten on every edit and read
 * back on Load; the URL field the Crumble URL the same way.
 */

import { $, $$, el, fill } from '../dom.js';
import { EditorModel, PALETTE, TWO_QUBIT, EXAMPLES } from '../editor.js';

const SVG = 'http://www.w3.org/2000/svg';
const MARK_COLORS = ['var(--x)', 'var(--z)', 'var(--y)', 'var(--ok)'];
const PAULI_COLOR = { X: 'var(--x)', Z: 'var(--z)', Y: 'var(--y)' };

function svg(tag, attrs = {}, children = []) {
  const node = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  for (const c of children) node.append(c);
  return node;
}

export function initEditor(root) {
  if (!root) return;
  const state = {
    model: EditorModel.fromStim(EXAMPLES['Repetition code, d = 3, 2 rounds'] ?? Object.values(EXAMPLES)[0]),
    layer: 0,
    tool: 'H',
    marker: 0,
    pending: null, // the first qubit of a two-qubit gate
    analysis: null,
  };

  const grid = $('[data-ed-grid]', root);
  const timeline = $('[data-ed-timeline]', root);
  const text = $('[data-ed-text]', root);
  const urlField = $('[data-ed-url]', root);
  const status = $('[data-ed-status]', root);
  const layerLabel = $('[data-ed-layer]', root);
  const flips = $('[data-ed-flips]', root);
  const palette = $('[data-ed-palette]', root);
  const examples = $('[data-ed-examples]', root);

  // -- controls --
  const tools = [...PALETTE, 'MARK X', 'MARK Z', 'MARK Y', 'DETECTOR', 'OBSERVABLE', 'erase'];
  fill(palette, tools.map((t) => el('button', { class: 'btn btn--small', type: 'button', 'data-tool': t, onclick: () => setTool(t) }, [t])));
  const initial = EXAMPLES['Repetition code, d = 3, 2 rounds'] ? 'Repetition code, d = 3, 2 rounds' : Object.keys(EXAMPLES)[0];
  fill(examples, Object.keys(EXAMPLES).map((name) => el('option', { value: name, selected: name === initial }, [name])));
  examples.value = initial;
  examples.onchange = () => {
    state.model = EditorModel.fromStim(EXAMPLES[examples.value]);
    state.layer = 0;
    state.pending = null;
    render();
  };
  $('[data-ed-load]', root).onclick = () => {
    try {
      state.model = EditorModel.fromStim(text.value);
      state.layer = Math.min(state.layer, state.model.layers.length - 1);
      state.pending = null;
      say('');
      render();
    } catch (e) {
      say(`Could not read the Stim text: ${e.message}`);
    }
  };
  $('[data-ed-load-url]', root).onclick = () => {
    try {
      state.model = EditorModel.fromCrumbleUrl(urlField.value);
      state.layer = 0;
      state.pending = null;
      say('');
      render();
    } catch (e) {
      say(`Could not read the URL: ${e.message}`);
    }
  };
  $('[data-ed-open]', root).onclick = () => window.open(state.model.toCrumbleUrl(), '_blank', 'noopener');
  $('[data-ed-prev]', root).onclick = () => { state.layer = Math.max(0, state.layer - 1); state.pending = null; render(); };
  $('[data-ed-next]', root).onclick = () => { state.layer = Math.min(state.model.layers.length - 1, state.layer + 1); state.pending = null; render(); };
  $('[data-ed-add-layer]', root).onclick = () => { state.model.addLayer(state.layer); state.layer += 1; render(); };
  $('[data-ed-del-layer]', root).onclick = () => { state.model.deleteLayer(state.layer); state.layer = Math.min(state.layer, state.model.layers.length - 1); render(); };
  $('[data-ed-add-qubit]', root).onclick = () => {
    const qs = state.model.qubits();
    const q = qs.length ? Math.max(...qs) + 1 : 0;
    const xs = [...state.model.coords.values()].map((c) => c[0]);
    state.model.setCoords(q, xs.length ? Math.max(...xs) + 1 : 0, 0);
    render();
  };
  $('[data-ed-clear-marks]', root).onclick = () => { state.model.markers = []; render(); };
  for (const k of [0, 1, 2, 3]) {
    const b = $(`[data-ed-mark="${k}"]`, root);
    if (b) b.onclick = () => { state.marker = k; render(); };
  }

  function say(message) {
    status.textContent = message;
  }

  function setTool(tool) {
    state.tool = tool;
    state.pending = null;
    render();
  }

  // -- edits from the grid --
  function clickQubit(q) {
    const { model, layer, tool } = state;
    if (tool === 'erase') {
      model.removeOn(layer, q);
      model.markers = model.markers.filter((m) => !(m.qubit === q && m.layer === layer));
    } else if (tool.startsWith('MARK ')) {
      model.toggleMarker(state.marker, tool.slice(5), q, layer);
    } else if (tool === 'DETECTOR' || tool === 'OBSERVABLE') {
      // The newest record on this qubit in this layer, compared with the one before it.
      const recs = model.recordsIn(layer).filter((r) => r.qubit === q);
      if (!recs.length) {
        say('A detector compares measurements: pick a layer where this qubit is measured.');
        return;
      }
      const look = recs[recs.length - 1].lookback;
      if (tool === 'DETECTOR') model.addDetector(layer, [look], model.coords.get(q) ? [...model.coords.get(q), layer] : []);
      else model.addDetector(layer, [look], [], 0);
    } else if (TWO_QUBIT.includes(tool)) {
      if (state.pending === null) {
        state.pending = q;
        say(`${tool}: now click the second qubit (${tool === 'CX' ? 'the target' : 'the partner'}).`);
        render();
        return;
      }
      if (state.pending !== q) model.place(layer, tool, [state.pending, q]);
      state.pending = null;
      say('');
    } else {
      model.place(layer, tool, [q]);
    }
    render();
  }

  // -- drawing --
  function render() {
    const { model } = state;
    state.layer = Math.min(state.layer, model.layers.length - 1);
    state.analysis = model.analysis();
    layerLabel.textContent = `layer ${state.layer + 1} of ${model.layers.length}`;
    for (const b of $$('[data-tool]', palette)) b.classList.toggle('btn--primary', b.dataset.tool === state.tool);
    for (const k of [0, 1, 2, 3]) {
      const b = $(`[data-ed-mark="${k}"]`, root);
      if (b) b.classList.toggle('btn--primary', state.marker === k);
    }
    drawGrid();
    drawTimeline();
    drawFlips();
    text.value = model.toStim();
    urlField.value = model.toCrumbleUrl();
  }

  function layout() {
    const { model } = state;
    const qubits = model.qubits();
    const coords = new Map();
    let next = 0;
    for (const q of qubits) coords.set(q, model.coords.get(q) ?? [next++, -1]);
    const xs = [...coords.values()].map((c) => c[0]);
    const ys = [...coords.values()].map((c) => c[1]);
    const x0 = Math.min(0, ...xs);
    const y0 = Math.min(0, ...ys);
    const w = Math.max(1, ...xs) - x0;
    const h = Math.max(1, ...ys) - y0;
    return { qubits, coords, x0, y0, w, h };
  }

  function drawGrid() {
    const { model, layer, analysis } = state;
    const { qubits, coords, x0, y0, w, h } = layout();
    const cell = 48;
    const pad = 28;
    const width = (w + 1) * cell + 2 * pad;
    const height = (h + 1) * cell + 2 * pad;
    const node = svg('svg', { viewBox: `0 0 ${width} ${height}`, width: '100%', role: 'img', 'aria-label': 'The circuit\'s qubits and this layer\'s gates' });
    node.style.maxHeight = '420px';
    const at = (q) => {
      const [x, y] = coords.get(q);
      return [pad + (x - x0) * cell + cell / 2, pad + (y - y0) * cell + cell / 2];
    };
    // Two-qubit gates as links.
    for (const op of model.layers[layer]) {
      if (!TWO_QUBIT.includes(op.name)) continue;
      for (let i = 0; i + 1 < op.targets.length; i += 2) {
        const a = Number(op.targets[i]);
        const b = Number(op.targets[i + 1]);
        if (!coords.has(a) || !coords.has(b)) continue;
        const [ax, ay] = at(a);
        const [bx, by] = at(b);
        node.append(svg('line', { x1: ax, y1: ay, x2: bx, y2: by, stroke: 'var(--ink)', 'stroke-width': 3 }));
        if (op.name === 'CX') node.append(svg('circle', { cx: ax, cy: ay, r: 5, fill: 'var(--ink)' }));
      }
    }
    const frame = analysis.frames[layer] ?? [];
    for (const q of qubits) {
      const [x, y] = at(q);
      const ops = model.opsOn(layer, q).filter((op) => !TWO_QUBIT.includes(op.name) || op.name === 'CX');
      const label = ops.map((op) => (op.name === 'CX' ? (Number(op.targets[op.targets.indexOf(String(q)) ^ 1]) !== undefined && op.targets.indexOf(String(q)) % 2 === 0 ? '•' : '⊕') : op.name)).join(' ');
      const g = svg('g', { class: 'ed-qubit', tabindex: 0, role: 'button', 'aria-label': `qubit ${q}` });
      const marks = frame.map((f, k) => [k, f.get(q)]).filter(([, p]) => p);
      g.append(svg('circle', { cx: x, cy: y, r: 19, fill: state.pending === q ? 'var(--accent-soft, #ffe9b0)' : 'var(--surface)', stroke: 'var(--ink)', 'stroke-width': 1.5 }));
      marks.forEach(([k, p], i) => {
        g.append(svg('circle', { cx: x - 14 + 9 * i, cy: y - 16, r: 5, fill: PAULI_COLOR[p], stroke: MARK_COLORS[k % 4], 'stroke-width': 2 }));
      });
      g.append(svg('text', { x, y: y + 4, 'text-anchor': 'middle', 'font-size': label.length > 3 ? 9 : 12, 'font-weight': 700, fill: 'var(--ink)' }, [document.createTextNode(label || '')]));
      g.append(svg('text', { x, y: y + 31, 'text-anchor': 'middle', 'font-size': 9, fill: 'var(--ink-3)' }, [document.createTextNode(String(q))]));
      g.onclick = () => clickQubit(q);
      g.onkeydown = (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); clickQubit(q); } };
      node.append(g);
    }
    fill(grid, [node]);
  }

  function drawTimeline() {
    const { model, analysis } = state;
    const qubits = model.qubits();
    const col = 40;
    const row = 22;
    const left = 36;
    const top = 18;
    const width = left + model.layers.length * col + 10;
    const height = top + qubits.length * row + 10;
    const node = svg('svg', { viewBox: `0 0 ${width} ${height}`, width: '100%', role: 'img', 'aria-label': 'The circuit layer by layer' });
    node.style.maxHeight = '360px';
    const rowOf = new Map(qubits.map((q, i) => [q, top + i * row + row / 2]));
    model.layers.forEach((layer, L) => {
      const x = left + L * col + col / 2;
      const g = svg('g', { class: 'ed-layer' });
      if (L === state.layer) g.append(svg('rect', { x: x - col / 2, y: 2, width: col, height: height - 4, fill: 'var(--surface-sunk)', rx: 4 }));
      g.append(svg('text', { x, y: 11, 'text-anchor': 'middle', 'font-size': 9, fill: 'var(--ink-3)' }, [document.createTextNode(String(L + 1))]));
      // Marker Paulis entering the layer.
      (analysis.frames[L] ?? []).forEach((f, k) => {
        for (const [q, p] of f) {
          if (!rowOf.has(q)) continue;
          g.append(svg('rect', { x: x - col / 2 + 2, y: rowOf.get(q) - row / 2 + 1, width: col - 4, height: row - 2, fill: PAULI_COLOR[p], opacity: 0.25 + 0.15 * (k % 4), rx: 3 }));
        }
      });
      for (const op of layer) {
        if (TWO_QUBIT.includes(op.name)) {
          for (let i = 0; i + 1 < op.targets.length; i += 2) {
            const a = Number(op.targets[i]);
            const b = Number(op.targets[i + 1]);
            if (!rowOf.has(a) || !rowOf.has(b)) continue;
            g.append(svg('line', { x1: x, y1: rowOf.get(a), x2: x, y2: rowOf.get(b), stroke: 'var(--ink)', 'stroke-width': 2 }));
            g.append(svg('circle', { cx: x, cy: rowOf.get(a), r: 3, fill: 'var(--ink)' }));
            g.append(svg('circle', { cx: x, cy: rowOf.get(b), r: 5, fill: 'none', stroke: 'var(--ink)', 'stroke-width': 1.5 }));
          }
          continue;
        }
        for (const t of op.targets) {
          const q = Number(String(t).replace(/^!?[XYZ]?/, ''));
          if (!rowOf.has(q)) continue;
          const short = op.name.replace('_DAG', '†').replace('SQRT_', '√');
          g.append(svg('rect', { x: x - 14, y: rowOf.get(q) - 8, width: 28, height: 16, fill: 'var(--surface)', stroke: 'var(--ink)', rx: 2 }));
          g.append(svg('text', { x, y: rowOf.get(q) + 4, 'text-anchor': 'middle', 'font-size': short.length > 2 ? 7 : 9, 'font-weight': 600, fill: 'var(--ink)' }, [document.createTextNode(short)]));
        }
      }
      g.onclick = () => { state.layer = L; state.pending = null; render(); };
      node.append(g);
    });
    for (const q of qubits) {
      node.append(svg('line', { x1: left, y1: rowOf.get(q), x2: width - 6, y2: rowOf.get(q), stroke: 'var(--rule-soft)', 'stroke-width': 1 }));
      node.append(svg('text', { x: left - 4, y: rowOf.get(q) + 3, 'text-anchor': 'end', 'font-size': 9, fill: 'var(--ink-3)' }, [document.createTextNode(`q${q}`)]));
    }
    // Lines go behind: move them first.
    const lines = [...node.querySelectorAll(':scope > line, :scope > text')];
    for (const l of lines) node.prepend(l);
    fill(timeline, [node]);
  }

  function drawFlips() {
    const { analysis, model } = state;
    const rows = model.markers.map((mk, k) => {
      const dets = [...analysis.detFlips[k]].sort((a, b) => a - b);
      const obs = [...analysis.obsFlips[k]].sort((a, b) => a - b);
      const recs = [...analysis.flips[k]].sort((a, b) => a - b);
      return el('tr', {}, [
        el('td', {}, [el('span', { class: 'ed-swatch', style: `background:${MARK_COLORS[mk.k % 4]}` }), ` ${mk.pauli}${mk.qubit} before layer ${mk.layer + 1}`]),
        el('td', {}, [recs.length ? recs.map((r) => `rec ${r}`).join(', ') : '—']),
        el('td', {}, [dets.length ? dets.map((d) => `D${d}`).join(', ') : 'none']),
        el('td', {}, [obs.length ? obs.map((o) => `L${o}`).join(', ') : 'none']),
      ]);
    });
    fill(flips, rows.length ? rows : [el('tr', {}, [el('td', { colspan: 4, class: 'note' }, ['No markers. Choose MARK X, Y or Z and click a qubit.'])])]);
  }

  render();
}
