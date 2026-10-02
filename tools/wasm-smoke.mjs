// The WebAssembly engine, end to end in Node, through the same wrappers the
// page uses. Run: node tools/wasm-smoke.mjs [path/to/stabilizer_qec.wasm]
//
//   1. a phenomenological memory runs and fails at a sane rate;
//   2. Willow's raw measurements (data/willow-extract) convert to exactly the
//      detection events Google published, by hash, and decode;
//   3. a window-decoded stream explains every defect and agrees with global
//      decoding on nearly every stream.
import { readFile } from 'node:fs/promises';
import { gunzipSync } from 'node:zlib';
import { createHash } from 'node:crypto';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { instantiate, runBenchmark, NOISE, hwM2d, hwModel, hwDecode, rtSetup, rtWindows, rtGlobal } from '../js/engine.js';

const root = new URL('../', import.meta.url);
const wasmPath = process.argv[2] ? pathToFileURL(resolve(process.argv[2])) : new URL('stabilizer_qec.wasm', root);
const module = await WebAssembly.compile(await readFile(wasmPath));
const instance = await instantiate(module);

let failed = 0;
const check = (ok, text) => {
  console.log(`${ok ? 'ok  ' : 'FAIL'} ${text}`);
  if (!ok) failed++;
};

// 1.
const run = runBenchmark(instance, { d: 5, p: 0.02, rounds: 3, runs: 2000, noiseMode: NOISE.PHENOM });
check(run.rate > 0 && run.rate < 0.5, `phenomenological d = 5 at 2%: logical rate ${run.rate.toFixed(4)} over ${run.runs} runs`);

// 2.
const extract = new URL('data/willow-extract/', root);
const manifest = JSON.parse(await readFile(new URL('manifest.json', extract), 'utf8'));
for (const ex of manifest.experiments) {
  const dir = new URL(`${ex.dir}/`, extract);
  const read = (name) => readFile(new URL(name, dir));
  const text = async (name) => gunzipSync(await read(name)).toString('utf8');
  const [meas, sweeps, actual] = await Promise.all([read('measurements.b8'), read('sweep_bits.b8'), read('obs_flips_actual.b8')]);
  const conv = hwM2d(instance, await text('circuit_ideal.stim.gz'), meas, sweeps, ex.shots);
  const hash = createHash('sha256').update(conv.dets).digest('hex');
  check(hash === ex.detection_events_sha256, `Willow d = ${ex.d} ${ex.basis} r${ex.rounds}: detection events hash to Google's`);
  hwModel(instance, 0, { dem: await text('error_model_si1000.dem.gz') });
  const dec = hwDecode(instance, 0, conv.dets, ex.shots, true);
  let failures = 0, errors = 0;
  for (let s = 0; s < ex.shots; s++) {
    if (dec.predictions[s] === 255) errors++;
    else failures += (dec.predictions[s] ^ actual[s]) & 1;
  }
  check(errors === 0 && failures < ex.shots / 2,
    `Willow d = ${ex.d}: correlated matching, ${failures} failures in ${ex.shots} shots, ${errors} undecoded`);
}

// 3.
const setup = rtSetup(instance, { d: 3, rounds: 30, p: 0.006, commit: 3, buffer: 3, parallel: true, correlated: true });
const win = rtWindows(instance);
const glob = rtGlobal(instance);
// The shots are fresh each run, as on the page. Windows commit a little differently from a
// global decode on a few shots (about 62.5 of 64 agree, spread about one shot): 56 is six spreads
// down, a regression rather than chance.
check(win.unexplained === 0 && glob.agree >= 56,
  `stream d = 3, 30 rounds, ${setup.windows} windows: ${win.unexplained} unexplained, windowed = global on ${glob.agree} of 64`);

console.log(failed ? `${failed} failed` : 'all checks passed');
process.exit(failed ? 1 : 0);
