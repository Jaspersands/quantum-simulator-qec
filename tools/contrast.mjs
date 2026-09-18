// WCAG contrast for every text-on-wash pairing the renderers draw. Run: node tools/contrast.mjs
// Plaquette letters are --ink-2 on the washes (lattice.js); --ink-3 carries small text on the sunk surface.
import { readFileSync } from 'node:fs';
const css = readFileSync(new URL('../css/styles.css', import.meta.url), 'utf8');
const token = (name) => css.match(new RegExp(`${name}:\\s*(#[0-9a-f]{6})`, 'i'))[1];
const lum = (hex) => {
  const c = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255)
    .map((v) => (v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4));
  return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
};
const ratio = (a, b) => { const [hi, lo] = [lum(a), lum(b)].sort((x, y) => y - x); return (hi + 0.05) / (lo + 0.05); };
let failed = false;
for (const [text, on] of [['--ink-2', '--x-soft'], ['--ink-2', '--z-soft'], ['--ink-2', '--y-soft'], ['--ink-3', '--surface-sunk'], ['--ink-3', '--surface'], ['--defect-ink', '--defect-soft']]) {
  const r = ratio(token(text), token(on));
  console.log(`${text} on ${on}: ${r.toFixed(2)}:1 ${r >= 4.5 ? 'ok' : 'FAIL'}`);
  if (r < 4.5) failed = true;
}
process.exit(failed ? 1 : 0);
