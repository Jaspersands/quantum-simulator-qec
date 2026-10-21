// A smoke test of the page ui.serve() sends without dash (stabilizer_qec/estimate/qualtran/_ui_static/ui.html),
// run by tests/estimate_port/test_qualtran_ui_app.py where node (18 or later, for its fetch) is installed:
//
//     node qualtran_ui_page.js PAGE.html http://127.0.0.1:PORT/ ACTIONS.json
//
// It runs the page's script against the standard-library server at that address, in a small DOM of its own
// (the elements, events, styles and number-input validity the page uses) with a stand-in for plotly.js that
// keeps each figure drawn, then does ACTIONS (a JSON list) one after another, waiting after each until every
// request has been answered. It prints one JSON object: the ids drawn, the error notes drawn, and after each
// action the requests posted to update (their arguments, answers and the title while they ran), the
// callback's nine outputs as the page shows them, the graphs resized and what the page wrote to the console.
//
// Actions: {"do": "load"}; {"do": "type", "id": ID, "text": TEXT, "commit": "enter" | "blur"}, as a user
// typing TEXT into an input and pressing Enter or leaving it; {"do": "radio", "id": ID, "value": VALUE};
// {"do": "race", "first": ACTION, "second": ACTION}, where the answer to the first's request is held back
// until the second's has been answered and drawn.
"use strict";
const fs = require("fs");
const vm = require("vm");

const [pagePath, base, actionsPath] = process.argv.slice(2);
const html = fs.readFileSync(pagePath, "utf8");
const actions = JSON.parse(fs.readFileSync(actionsPath, "utf8"));
const inline = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map((m) => m[1]);
if (inline.length !== 1) throw new Error(`the page has ${inline.length} inline scripts, not 1`);
const TITLE = html.match(/<title>([^<]*)<\/title>/)[1];

function cssName(name) {
  return name.indexOf("-") >= 0 ? name : name.replace(/[A-Z]/g, (c) => "-" + c.toLowerCase());
}
// CSSStyleDeclaration: setProperty and camelCase assignment write the same map; "" removes a property.
function makeStyle() {
  const map = {};
  const methods = {
    setProperty(name, value) { if (value === "" || value === null) delete map[name]; else map[name] = String(value); },
    getPropertyValue(name) { return map[name] || ""; },
    map,
  };
  return new Proxy(methods, {
    get(target, prop) { return prop in target ? target[prop] : map[cssName(String(prop))] || ""; },
    set(target, prop, value) { target.setProperty(cssName(String(prop)), value); return true; },
  });
}

let documentBody = null;
class Node_ {
  constructor() { this.childNodes = []; this.parentNode = null; }
  appendChild(child) {
    if (child instanceof Fragment) {
      for (const c of [...child.childNodes]) this.appendChild(c);
      return child;
    }
    if (child.parentNode) child.parentNode.removeChild(child);
    child.parentNode = this;
    this.childNodes.push(child);
    return child;
  }
  removeChild(child) {
    const i = this.childNodes.indexOf(child);
    if (i >= 0) this.childNodes.splice(i, 1);
    child.parentNode = null;
    return child;
  }
  get textContent() { return this.childNodes.map((c) => c.textContent).join(""); }
  set textContent(value) {
    for (const c of this.childNodes) c.parentNode = null;
    this.childNodes = [];
    if (value !== "" && value !== null && value !== undefined) this.appendChild(new Text(String(value)));
  }
}
class Text extends Node_ {
  constructor(data) { super(); this.data = data; }
  get textContent() { return this.data; }
  set textContent(value) { this.data = String(value); }
}
class Fragment extends Node_ {}
class Element extends Node_ {
  constructor(tag) {
    super();
    this.tagName = tag.toUpperCase();
    this.attributes = {};
    this.style = makeStyle();
    this.listeners = {};
    this.id = "";
    this.className = "";
  }
  setAttribute(name, value) { this.attributes[name] = String(value); }
  removeAttribute(name) {
    if (name === "style") this.style = makeStyle();
    else delete this.attributes[name];
  }
  addEventListener(type, listener) { (this.listeners[type] = this.listeners[type] || []).push(listener); }
  dispatch(type, init) {
    const event = Object.assign({type: type, target: this}, init || {});
    for (const listener of this.listeners[type] || []) listener(event);
  }
  get classList() {
    const el = this;
    const names = () => el.className.split(/\s+/).filter(Boolean);
    return {
      contains: (c) => names().indexOf(c) >= 0,
      add: (c) => { if (names().indexOf(c) < 0) el.className = names().concat([c]).join(" "); },
    };
  }
  get offsetParent() {  // null when it or an ancestor is display: none, or it is not in the document
    for (let n = this; n; n = n.parentNode) {
      if (n.style && n.style.getPropertyValue("display") === "none") return null;
      if (n === documentBody) return n === this ? null : this.parentNode;
    }
    return null;
  }
}
// The HTML standard's valid floating-point number (what a number input's value may be).
const FLOAT = /^-?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)(?:[eE][-+]?[0-9]+)?$/;
class InputElement extends Element {
  constructor() {
    super("input");
    this.type = "text";
    this.min = "";
    this.max = "";
    this.step = "";
    this.checked = false;
    this.badInput = false;
    this._value = "";
  }
  get value() { return this._value; }
  set value(v) {  // a number input keeps only a valid number, as browsers' do
    v = String(v);
    this._value = this.type === "number" && !FLOAT.test(v) ? "" : v;
    this.badInput = false;
  }
  typed(text) {  // what a user typing text leaves: a number input's text that is not a number is a bad input
    this.value = text;
    this.badInput = this.type === "number" && !FLOAT.test(text);
  }
  checkValidity() {
    if (this.type !== "number") return true;
    if (this.badInput) return false;
    if (this._value === "") return true;
    const n = Number(this._value);
    if (this.min !== "" && n < Number(this.min)) return false;
    if (this.max !== "" && n > Number(this.max)) return false;
    if (this.step !== "any" && this.step !== "") throw new Error("a step other than 'any' is not modelled here");
    return true;
  }
}

function walk(node, visit) {
  for (const c of node.childNodes) {
    if (c instanceof Element) {
      visit(c);
      walk(c, visit);
    }
  }
}
documentBody = new Element("body");
const entry = new Element("div");
entry.id = "react-entry-point";
documentBody.appendChild(entry);
const document = {
  title: TITLE,
  body: documentBody,
  createElement: (tag) => (tag.toLowerCase() === "input" ? new InputElement() : new Element(tag)),
  createTextNode: (text) => new Text(String(text)),
  createDocumentFragment: () => new Fragment(),
  getElementById(id) {
    let found = null;
    walk(documentBody, (el) => { if (found === null && el.id === id) found = el; });
    return found;
  },
};

let resized = [];
const Plotly = {  // plotly.js's calls the page makes, keeping what they were given
  react(el, data, layout, config) {
    el.classList.add("js-plotly-plot");
    el.figure = JSON.parse(JSON.stringify({data: data, layout: layout}));
    el.config = config;
    return Promise.resolve(el);
  },
  Plots: { resize(el) { resized.push(el.id); } },
};

const posts = [];
const held = [];
let inflight = 0;
let holdNext = false;
async function pageFetch(url, options) {  // fetch, from the server at base; a POST's answer can be held
  inflight++;
  try {
    let post = null;
    let gate = null;
    if (options && options.method === "POST") {
      post = {args: JSON.parse(options.body), title: document.title};
      posts.push(post);
      if (holdNext) {
        holdNext = false;
        gate = new Promise((release) => held.push(release));
      }
    }
    const response = await fetch(new URL(url, base), options);
    const text = await response.text();
    if (post) post.status = response.status;
    if (gate) await gate;
    return {ok: response.ok, status: response.status, json: async () => JSON.parse(text), text: async () => text};
  } finally {
    inflight--;
  }
}

const logged = [];
const sandbox = {
  document: document,
  fetch: pageFetch,
  console: {
    error: (...a) => logged.push("error: " + a.map(String).join(" ")),
    warn: (...a) => logged.push("warn: " + a.map(String).join(" ")),
    log: (...a) => logged.push("log: " + a.map(String).join(" ")),
  },
  setTimeout: setTimeout,
  clearTimeout: clearTimeout,
};
sandbox.window = sandbox;
sandbox.Plotly = Plotly;
vm.createContext(sandbox);

async function settle(done) {  // until done(), every request answered and the page idle (its title back)
  const start = Date.now();
  for (;;) {
    for (let i = 0; i < 10; i++) await new Promise((r) => setImmediate(r));
    if ((done ? done() : true) && inflight === held.length && (held.length > 0 || document.title === TITLE)) return;
    if (Date.now() - start > 60000) throw new Error("the page did not settle in 60 s");
    await new Promise((r) => setTimeout(r, 5));
  }
}

function byId(id) {
  const el = document.getElementById(id);
  if (el === null) throw new Error(`no element ${id}`);
  return el;
}
function act(action) {
  if (action.do === "type") {
    const input = byId(action.id);
    input.typed(action.text);
    input.dispatch("input");
    if (action.commit === "enter") input.dispatch("keypress", {key: "Enter"});
    else if (action.commit === "blur") input.dispatch("blur");
    else throw new Error(`commit ${action.commit}`);
  } else if (action.do === "radio") {
    const group = byId(action.id);
    let options = [];
    walk(group, (el) => { if (el instanceof InputElement && el.type === "radio") options.push(el); });
    const option = options.find((o) => o.value === action.value);
    if (!option) throw new Error(`no option ${action.value} in ${action.id}`);
    options.forEach((o) => { o.checked = o === option; });
    option.dispatch("change");
  } else {
    throw new Error(`action ${action.do}`);
  }
}

function outputs() {  // the callback's outputs as the page shows them
  const style = (id) => Object.assign({}, byId(id).style.map);
  const figure = (id) => byId(id).figure || null;
  return {
    "qubit-pie-chart": figure("qubit-pie-chart"),
    "runtime-container": style("runtime-container"),
    "runtime-plot": figure("runtime-plot"),
    "magic-name": byId("magic-name").textContent,
    "total_magic": byId("total_magic").value,
    "magic-factory-container": style("magic-factory-container"),
    "min_factory_count": byId("min_factory_count").value,
    "duration-container": style("duration-container"),
    "duration": byId("duration").value,
  };
}

async function main() {
  const result = {steps: []};
  for (const action of actions) {
    const before = posts.length;
    resized = [];
    if (action.do === "load") {
      vm.runInContext(inline[0], sandbox, {filename: "ui.html"});
      await settle(() => posts.length > 0);
      const ids = [];
      const notes = [];
      walk(documentBody, (el) => {
        if (el.id && el !== entry) ids.push(el.id);
        if (el.classList.contains("ui-error")) notes.push(el.textContent);
      });
      result.ids = ids;
      result.notes = notes;
    } else if (action.do === "race") {
      holdNext = true;
      act(action.first);
      act(action.second);
      await settle(() => held.length === 1 && posts.length === before + 2 && posts[before + 1].status !== undefined);
      held.splice(0).forEach((release) => release());
      await settle();
    } else {
      act(action);
      await settle();
    }
    result.steps.push({posts: posts.slice(before), outputs: outputs(), resized: resized, title: document.title, logged: logged.splice(0)});
  }
  process.stdout.write(JSON.stringify(result));
}

main().catch((e) => {
  process.stderr.write(String(e && e.stack ? e.stack : e) + "\n");
  process.exit(1);
});
