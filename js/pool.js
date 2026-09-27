/**
 * A pool of Monte Carlo workers, each with its own engine instance.
 *
 * WHY THIS EXISTS
 * ---------------
 * One worker keeps the page responsive, but it uses one core, and the Monte
 * Carlo on this page is a sum over independent shots: nothing stops every core
 * from taking a share. WebAssembly's own threads need SharedArrayBuffer, which
 * needs cross-origin isolation headers that GitHub Pages cannot send, so the
 * parallelism is a pool of ordinary workers instead. Each instance is seeded
 * independently from the platform CSPRNG, so their shots are independent.
 *
 * The pool speaks Compute's interface (`call`, `cancel`) so every section can
 * use it unchanged, and adds `map` for jobs split into independent parts.
 *
 * Workers start when jobs need them, one per waiting job up to the pool's
 * size, not all at load: a reader who never runs the Monte Carlo pays for one
 * engine instance, the one that times the device for the overview.
 */

import { Compute } from './compute.js';
import { splitRuns, mergeStream, mergeStreamResults } from './pool-merge.js';

/** All cores but one, which the page keeps; at most eight. */
export function poolSize() {
  const cores = (typeof navigator !== 'undefined' && navigator.hardwareConcurrency) || 2;
  return Math.max(1, Math.min(8, cores - 1));
}

export class Pool {
  /**
   * @param {number} [size]
   * @param {Promise<WebAssembly.Module>} [module] the page's compiled engine,
   *   handed to every worker before its first job so none fetches its own
   */
  constructor(size = poolSize(), module = null) {
    this.size = size;
    this.module = module;
    this.workers = [];
    this.queue = [];
    this.alone = false;
    this.jobs = new Map();
    this.groups = new Map();
    this.nextId = 1;
    // The first worker starts now: the overview needs it at once, and a
    // browser without module workers says so here, where the page expects it.
    this.#spawn();
  }

  /** Start a worker. It takes no job until it has the page's module. */
  #spawn() {
    const worker = { compute: new Compute(), busy: true, ready: false };
    this.workers.push(worker);
    // Handing over the module first means no worker fetches its own copy; if
    // compiling failed, the worker fetches its own as before.
    const handed = this.module
      ? this.module.then((m) => worker.compute.call('module', { module: m })).catch(() => {})
      : Promise.resolve();
    handed.then(() => {
      worker.busy = false;
      worker.ready = true;
      this.#pump();
    });
  }

  get dead() {
    return this.workers.find((w) => w.compute.dead)?.compute.dead ?? null;
  }

  /**
   * Run one job on the next free worker. Jobs wait in the pool, not in a
   * worker's own queue, so a worker that finishes early takes the next job
   * rather than idling while another works through a backlog.
   */
  call(op, payload, onProgress, { alone = false } = {}) {
    const id = this.nextId++;
    const promise = new Promise((resolve, reject) => {
      this.queue.push({ id, op, payload, onProgress, resolve, reject, cancelled: false, alone });
    });
    promise.id = id;
    this.#pump();
    return promise;
  }

  /**
   * A job that must have the machine to itself, such as a throughput
   * measurement: it waits for every worker to be idle, and nothing else starts
   * until it is done.
   */
  callAlone(op, payload, onProgress) {
    return this.call(op, payload, onProgress, { alone: true });
  }

  #pump() {
    for (const worker of this.workers) {
      if (this.alone || !this.queue.length) return;
      if (worker.busy || !worker.ready) continue;
      const job = this.queue[0];
      if (job.alone && this.workers.some((w) => w.busy)) return;
      this.queue.shift();
      worker.busy = true;
      this.alone = job.alone;
      const inner = worker.compute.call(job.op, job.payload, job.onProgress);
      this.jobs.set(job.id, { worker, innerId: inner.id });
      if (job.cancelled) worker.compute.cancel(inner.id);
      inner.then(job.resolve, job.reject).finally(() => {
        worker.busy = false;
        if (job.alone) this.alone = false;
        this.jobs.delete(job.id);
        this.#pump();
      });
    }
    // Jobs still waiting: a worker for each, up to the pool's size. A browser
    // that will not start another worker caps the pool where it stands; the
    // workers it has carry on.
    if (this.alone) return;
    let wanted = this.queue.length - this.workers.filter((w) => !w.ready).length;
    while (wanted > 0 && this.workers.length < this.size) {
      try {
        this.#spawn();
      } catch {
        this.size = this.workers.length;
        return;
      }
      wanted -= 1;
    }
  }

  /** Ask a job, or every part of a split one, to stop after its current chunk. */
  cancel(id) {
    for (const part of this.groups.get(id) ?? [id]) {
      const running = this.jobs.get(part);
      if (running) running.worker.compute.cancel(running.innerId);
      const waiting = this.queue.find((j) => j.id === part);
      if (waiting) waiting.cancelled = true;
    }
  }

  /**
   * Independent jobs spread over the pool; resolves to their results in the
   * order given. `onProgress(index, progress)` hears from each.
   */
  map(op, payloads, onProgress) {
    return Promise.all(payloads.map((payload, i) => this.call(op, payload, (p) => onProgress?.(i, p))));
  }

  /** The worker's `table` op, one cell per job, reported in its own progress shape. */
  table(cells, base, onCell) {
    let done = 0;
    return this.map('table', cells.map((cell) => ({ cells: [cell], base })), (_, p) => {
      done += 1;
      onCell?.({ done, total: cells.length, cell: p.cell });
    }).then((parts) => ({ results: parts.flatMap((r) => r.results) }));
  }

  /**
   * The worker's `sweep` op, one point per job, reported in its own progress
   * shape. The largest distances go first: they take longest, and starting
   * them early keeps the last worker from finishing long after the rest.
   */
  sweep(distances, ps, base, onPoint) {
    const jobs = [...distances].sort((a, b) => b - a).flatMap((d) => ps.map((p) => ({ distances: [d], ps: [p], base })));
    let done = 0;
    return this.map('sweep', jobs, (_, p) => {
      done += 1;
      onPoint?.({ ...p, done, total: jobs.length });
    }).then((parts) => ({ points: parts.flatMap((r) => r.points) }));
  }

  /**
   * The bench's `stream`, its shots shared across every worker. Progress and
   * the result are the parts combined (see pool-merge.js); cancelling the
   * returned promise's id stops every part.
   */
  stream(config, onProgress) {
    const shares = splitRuns(config.runs, this.size);
    const latest = shares.map(() => null);
    const parts = shares.map((runs, i) => this.call('stream', { ...config, runs }, (p) => {
      latest[i] = p;
      onProgress?.(mergeStream(latest, config.runs));
    }));
    const id = this.nextId++;
    this.groups.set(id, parts.map((part) => part.id));
    const promise = Promise.all(parts).then(mergeStreamResults).finally(() => this.groups.delete(id));
    promise.id = id;
    return promise;
  }
}
