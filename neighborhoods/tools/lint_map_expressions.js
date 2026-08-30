/*
 * Validate the MapLibre paint/layout expressions in index.html.
 *
 * WHY THIS EXISTS. An invalid paint expression is invisible to a reading of the file:
 * `node --check` sees valid JavaScript, and the failure is a console error in a browser
 * with the entire layer silently missing, which looks exactly like an empty map for any
 * other reason.
 *
 * `tools/screenshot.js` can now see it too, since headless Chrome does render the map
 * given --enable-unsafe-swiftshader, and it prints the page's console. This is still the
 * faster check by two orders of magnitude, needs no browser and no server, and is the one
 * to run on every edit; the screenshot is for looking at the result.
 *
 * That is not hypothetical. Sizing dots by Wikipedia sitelinks was written as
 *
 *     'circle-radius': ['*', ['interpolate', ['linear'], ['zoom'], ...], FAME]
 *
 * which is invalid: ["zoom"] may only be the DIRECT input of a top-level "step" or
 * "interpolate". The `dots` layer failed to add, and the map lost every dot and all
 * hover interaction until someone opened the console.
 *
 * Usage:  node tools/lint_map_expressions.js [path/to/index.html]
 * Exit 0 = clean, 1 = a problem OR a layer it could not check.
 */

const fs = require('fs');
const path = require('path');

const file = process.argv[2] ||
  path.join(__dirname, '..', 'index.html');
const html = fs.readFileSync(file, 'utf8');
const blocks = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]);
const src = blocks[blocks.length - 1] || '';

// Hoist top-level SCREAMING_CASE constants so layers referring to them can be evaluated.
// Without this, the layer most in need of checking is the one skipped. Two shapes, both
// of which layers actually use: array literals (FAME, CITY_FONT), possibly spanning
// lines, and plain numbers or strings (CITY_ZOOM, used as a layer's minzoom).
let prelude = '';
const CONSTS = [
  /^const\s+([A-Z_][A-Z0-9_]*)\s*=\s*(\[[\s\S]*?\]);$/gm,
  /^const\s+([A-Z_][A-Z0-9_]*)\s*=\s*(-?[0-9][0-9.eE+-]*|'[^']*'|"[^"]*");$/gm,
];
for (const re of CONSTS)
  for (const m of src.matchAll(re)) prelude += `var ${m[1]} = ${m[2]};\n`;

// Pull the object literal out of each map.addLayer({...}) by brace matching.
const layers = [];
let idx = 0;
while ((idx = src.indexOf('map.addLayer(', idx)) !== -1) {
  const start = src.indexOf('{', idx);
  let depth = 0, end = start;
  for (; end < src.length; end++) {
    if (src[end] === '{') depth++;
    else if (src[end] === '}') { depth--; if (depth === 0) break; }
  }
  layers.push(src.slice(start, end + 1));
  idx = end;
}

let bad = 0, skipped = 0;
for (const text of layers) {
  const id = (text.match(/id:\s*'([^']+)'/) || [])[1] || '?';
  let obj;
  try {
    obj = eval(prelude + '(' + text + ')');   // our own source, not untrusted input
  } catch (e) {
    console.log(`  SKIPPED ${id}: could not parse (${e.message})`);
    skipped++;
    continue;
  }
  const walk = (node, topLevel) => {
    if (!Array.isArray(node)) {
      if (node && typeof node === 'object') Object.values(node).forEach(v => walk(v, true));
      return;
    }
    const op = node[0];
    if (op === 'zoom') {
      if (!topLevel) {
        console.log(`  ERROR ${id}: ["zoom"] nested below a top-level step/interpolate`);
        bad++;
      }
      return;
    }
    if (op === 'step' || op === 'interpolate') {
      const inputIdx = op === 'step' ? 1 : 2;
      node.forEach((child, i) => walk(child, topLevel && i === inputIdx));
      return;
    }
    node.forEach(child => walk(child, false));
  };
  for (const key of ['paint', 'layout']) {
    if (obj[key]) for (const v of Object.values(obj[key])) walk(v, true);
  }
}

// A SKIPPED layer counts as failure. The layer that broke was precisely the one an
// earlier version of this script could not evaluate, so reporting "OK" while skipping
// anything is the one answer guaranteed to be wrong.
const ok = bad === 0 && skipped === 0;
console.log(ok
  ? `expression lint OK (${layers.length} layers checked)`
  : `expression lint FAILED: ${bad} problem(s), ${skipped} unchecked`);
process.exit(ok ? 0 : 1);
