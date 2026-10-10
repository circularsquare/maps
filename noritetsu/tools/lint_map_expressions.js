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

// Pull the object literal out of each map.addLayer({...}) by brace matching, and out of
// addTrackLayer({...}) (index.html's per-country layers, added without MapLibre's own check
// after the first country: this lint is then the only check they get).
const layers = [];
const CALL = /\b(?:map\.addLayer|addTrackLayer)\(\s*\{/g;
let idx = 0, m0;
while ((m0 = CALL.exec(src))) {
  idx = m0.index;
  const start = src.indexOf('{', idx);
  let depth = 0, end = start;
  for (; end < src.length; end++) {
    if (src[end] === '{') depth++;
    else if (src[end] === '}') { depth--; if (depth === 0) break; }
  }
  layers.push(src.slice(start, end + 1));
  CALL.lastIndex = end;
}

let bad = 0, skipped = 0;
for (const text of layers) {
  const id = (text.match(/id:\s*[`'"]([^`'"]+)/) || [])[1] || '?';
  /* A layer added once per country names its id and source from a variable (`track-${cc}`),
     which is not in scope here. Such names are bound to a placeholder string and the layer
     evaluated again, so its paint expressions are still checked. Only lowercase names: an
     unknown SCREAMING_CASE one is a paint constant this script failed to hoist, and that
     must stay a failure rather than be papered over. */
  let obj, local = '';
  for (let tries = 0; ; tries++) {
    try {
      obj = eval(prelude + local + '(' + text + ')');   // our own source, not untrusted input
      break;
    } catch (e) {
      const m = e instanceof ReferenceError && e.message.match(/^(\w+) is not defined/);
      if (m && /^[a-z]/.test(m[1]) && tries < 10) { local += `var ${m[1]} = '${m[1]}';\n`; continue; }
      obj = null;
      console.log(`  SKIPPED ${id}: could not parse (${e.message})`);
      break;
    }
  }
  if (!obj) { skipped++; continue; }
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

/* A NAME DECLARED TWICE. Two `function goRegion` in one script is not an error in JavaScript:
   the later one silently wins. On 2026-09-30 a leftover one that reloaded the page won over
   its replacement, and the page reloaded itself forever. Cheap to check, so it is checked
   here, on the lint that runs on every edit. */
const declared = new Map();
for (const m of src.matchAll(/^(?:async\s+)?function\s+(\w+)|^(?:const|let|var)\s+(\w+)/gm)) {
  const n = m[1] || m[2];
  declared.set(n, (declared.get(n) || 0) + 1);
}
for (const [n, c] of declared) {
  if (c > 1) { console.log(`  ERROR ${n}: declared ${c} times at top level; the last one wins`); bad++; }
}

/* THE WHOLE SCRIPT HAS TO PARSE. Each layer above is evaluated on its own, so a syntax error
   anywhere else passed: on 2026-10-03 a stray comment end in the header left the page blank
   ("Unexpected identifier 'and'") with this lint saying OK. Compiled, not run. */
try {
  new (require('vm').Script)(src, { filename: 'inline' });
} catch (e) {
  const m = /inline:(\d+)/.exec(e.stack || '');
  const line = m ? +m[1] + html.slice(0, html.lastIndexOf(src)).split('\n').length - 1 : '?';
  console.log(`  ERROR the script does not parse: ${e.message} (html line ${line})`);
  bad++;
}

// A SKIPPED layer counts as failure. The layer that broke was precisely the one an
// earlier version of this script could not evaluate, so reporting "OK" while skipping
// anything is the one answer guaranteed to be wrong.
const ok = bad === 0 && skipped === 0;
console.log(ok
  ? `expression lint OK (${layers.length} layers checked)`
  : `expression lint FAILED: ${bad} problem(s), ${skipped} unchecked`);
process.exit(ok ? 0 : 1);
