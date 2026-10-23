// Focused provenance/UI wiring checks: no science measurements or browser required.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { currentCoherentRows, initCoherent } from '../js/sections/coherent.js';
import { Plot } from '../js/plot.js';

const revision = 'coherent-merged-proposal-v1';
const old = {
  code: 'historical', d: 9, rounds: 9, theta: 0.2, version: '2.2.0', exact: 0.1, exact_stderr: 0,
  coherent: { rate: 0.1, stderr: 0.01, rate_merged: 0.09, ess: 0.5, exact: true },
  twirl: { rate: 0.1, stderr: 0.01, failures: 100 },
};
const current = { ...old, code: 'current', d: 3, theta: 0.3, generation_revision: revision, coherent: { ...old.coherent, exact: false } };
const unrelated = { ...old, generation_revision: 'prior-experiment' };
assert.deepEqual(currentCoherentRows([old, current, unrelated]), [current]);
assert.deepEqual(currentCoherentRows([old, unrelated]), []);
assert.equal('generation_revision' in old, false);
// Keep the three producer/consumer constants aligned without importing the native package.
for (const file of ['../tools/coherent.py', '../tools/readme_tables.py']) {
  assert.ok(readFileSync(new URL(file, import.meta.url), 'utf8').includes(`= "${revision}"`));
}

class Node {
  constructor(text = '') { this.textContent = text; this.children = []; this.dataset = {}; this.style = {}; }
  append(child) { this.children.push(child); }
  setAttribute() {}
  replaceChildren(...children) { this.children = children; }
  getContext() { return { setTransform() {} }; }
  getBoundingClientRect() { return { width: 480 }; }
}
const previousRender = Plot.prototype.render;
Plot.prototype.render = function (data) { this.canvas.plotData = data; };
globalThis.document = { documentElement: {}, createElement: () => new Node(), createTextNode: (text) => new Node(text) };
globalThis.window = { devicePixelRatio: 1 };
globalThis.getComputedStyle = () => ({ getPropertyValue: () => '' });
globalThis.ResizeObserver = class { observe() {} };
globalThis.IntersectionObserver = class {
  constructor(callback) { this.callback = callback; }
  disconnect() {}
  observe() { this.callback([{ isIntersecting: true }]); }
};
const kinds = ['coh', 'val', 'cap', 'zz'];
function section() {
  const figs = Object.fromEntries(kinds.map((kind) => {
    const nodes = new Map();
    return [kind, { nodes, querySelector(selector) {
      if (!nodes.has(selector)) nodes.set(selector, new Node());
      return nodes.get(selector);
    } }];
  }));
  return { figs, querySelector: (selector) => figs[selector.slice(6, -1)] };
}
async function render(rows) {
  globalThis.fetch = async () => ({ ok: true, json: async () => ({ rows }) });
  const root = section();
  initCoherent(root);
  await new Promise((resolve) => setImmediate(resolve));
  return root.figs;
}
try {
  // All four public lazy-loading paths must filter before rendering, including empty data.
  for (const [kind, fig] of Object.entries(await render([old, unrelated]))) {
    assert.match(fig.nodes.get(`[data-${kind}-status]`).textContent, /Awaiting measurements with the corrected proposal/);
    assert.equal(fig.nodes.size, 1, `${kind}: archival data reached plot/table rendering`);
  }
  const figs = await render([old, current, unrelated]);
  for (const [kind, fig] of Object.entries(figs)) {
    assert.match(fig.nodes.get(`[data-${kind}-status]`).textContent, /1 current measurements; 2 historical rows await remeasurement/);
    if (kind === 'val') {
      assert.equal(fig.nodes.get('[data-val-rows]').children.length, 1);
      assert.match(fig.nodes.get('[data-val-rows]').children[0].children[0].textContent, /^current, d = 3/);
    } else {
      const plot = fig.nodes.get(`[data-${kind}-plot]`).plotData;
      assert.ok(plot.series.length > 0);
      if (kind === 'cap') {
        assert.deepEqual(plot.series[0].points.map((p) => p.x), [3]);
      } else {
        assert.equal(fig.nodes.get(`[data-${kind}-rows]`).children.length, 1);
        assert.deepEqual(plot.series[0].points.map((p) => p.x), [0.3]);
        assert.equal(plot.series[0].label, 'd = 3, coherent');
      }
    }
  }
  for (const fig of Object.values(await render([]))) {
    assert.match([...fig.nodes.values()][0].textContent, /Awaiting measurements/);
  }
} finally {
  Plot.prototype.render = previousRender;
}
console.log('Coherent provenance: mixed rows, all four public rendering paths and empty states passed');
