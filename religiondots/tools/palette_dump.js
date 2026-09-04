/* Dump index.html's OWN palette tables as JSON, so a checker measures the shipped allocator
   rather than a copy of it — the same rule `check_palette.py` follows for ROOT_HSL.

   The inline script cannot simply be require()d: it touches `document` and maplibre at the
   top level. So this slices out only the top-level declarations the palette needs, by name.
   Each starts at column 0, so the next column-0 const/let/function ends it.

   Usage:  node tools/palette_dump.js <lineage.json> <out.json>
   where lineage.json is branches.py's LINEAGE, which check_overview.py writes for it.  */
const fs = require('fs');
const path = require('path');
const ROOT = path.join(__dirname, '..');

const html = fs.readFileSync(path.join(ROOT, 'index.html'), 'utf8');
const script = html.match(/<script(?![^>]*\bsrc=)[^>]*>([\s\S]*?)<\/script>/)[1];

const WANT = ['hsl', 'parentOf', 'ROOT_HSL', 'PIN', 'PIN_OVERVIEW', 'TIERS',
              'ROOT_BAND', 'BAND_TIERS', 'buildPalette', 'buildOverview'];
const decls = [...script.matchAll(/^(?:const|let|function)\s+([A-Za-z_$][\w$]*)/gm)]
  .map(m => ({ name: m[1], at: m.index }));
const found = new Set();
let src = '';
for (let i = 0; i < decls.length; i++) {
  if (!WANT.includes(decls[i].name)) continue;
  found.add(decls[i].name);
  src += script.slice(decls[i].at, i + 1 < decls.length ? decls[i + 1].at : script.length) + '\n';
}
const missing = WANT.filter(w => !found.has(w));
if (missing.length) {
  console.error('index.html no longer declares: ' + missing.join(', ')
    + '\nThe slicer looks for a column-0 const/let/function of each name.');
  process.exit(1);
}

const NODES = JSON.parse(fs.readFileSync(path.join(ROOT, 'taxonomy/religions.json'), 'utf8')).nodes;
const LIN = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const LINEAGE_RANK = {}, LINEAGE_GROUP = {}, LINEAGE_PARENTS = new Set();
for (const parent in LIN) {                       // same flattening index.html does on load
  LINEAGE_PARENTS.add(parent);
  let i = 0;
  for (const g of LIN[parent])
    for (const id of g.ids) { LINEAGE_RANK[id] = i++; LINEAGE_GROUP[id] = g.label; }
}

const run = new Function('NODES', 'LINEAGE_RANK', 'LINEAGE_GROUP', 'LINEAGE_PARENTS',
  'let NODE_COLOR = {}, OVERVIEW_COLOR = {};\n' + src
  + '\nbuildPalette();\nreturn { NODE_COLOR, OVERVIEW_COLOR };');
fs.writeFileSync(process.argv[3],
  JSON.stringify(run(NODES, LINEAGE_RANK, LINEAGE_GROUP, LINEAGE_PARENTS), null, 1));
