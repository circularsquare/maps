// T-161: the real New York save's station complexes as demand gets them, for the native
// comparison (`cargo test --release --features demand complexes_real_network -- --ignored
// --nocapture`, sim/src/demand/api.rs). Loads T-007-real-nyc.save into the track model (WASM, as
// the clock worker does), groups its constructed stations with game/complexes.ts and the save's
// links, and writes x, y, name and complex per station to
// ../artifacts/trainworld/t161/nyc_real_complex.json. Run from the maps root:
//   node trainworld/notes/t161-complexes.mjs
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import { resolve } from 'node:path';
import { build } from '../app/node_modules/vite/dist/node/index.js';
import init, { TrackApi } from '../app/src/wasm/track/sim.js';

const root = resolve('trainworld/app');
const bundle = await build({ configFile: false, logLevel: 'silent', plugins: [{
  name: 'cx', resolveId(id) { if (id.endsWith('cx-entry')) return '\0cx-entry'; },
  load(id) { if (id === '\0cx-entry') return `export * from ${JSON.stringify(root + '/src/game/complexes.ts')};`; },
}], build: { write: false, minify: false, lib: { entry: 'cx-entry', formats: ['es'] } } });
const out = (Array.isArray(bundle) ? bundle[0] : bundle).output;
const { buildComplexes } = await import('data:text/javascript;base64,' + Buffer.from(out.find((o) => o.type === 'chunk').code).toString('base64'));

await init({ module_or_path: readFileSync(new URL('../app/src/wasm/track/sim_bg.wasm', import.meta.url)) });
const raw = gunzipSync(readFileSync(new URL('../T-007-real-nyc.save', import.meta.url)));
const n = raw.readUInt32LE(4);
const header = JSON.parse(raw.subarray(8, 8 + n).toString('utf8'));
const api = new TrackApi(-73.985, 40.758, true);
if (!api.load(raw.subarray(8 + n))) throw new Error('save did not load');
const ni = api.nodes_info();
const st = [];
for (let i = 0; i < ni.length; i += 10) {
  if (ni[i + 7] > 0 && ni[i + 8] === 1) st.push({ id: String(ni[i]), name: api.node_name(ni[i]), x: ni[i + 1], y: ni[i + 2], heading: ni[i + 9], length: ni[i + 7] });
}
const fl = api.station_links(), links = [];
for (let k = 0; k + 1 < fl.length; k += 2) links.push([String(fl[k]), String(fl[k + 1])]);
const cx = buildComplexes(st, links);
const dir = new URL('../../artifacts/trainworld/t161/', import.meta.url);
mkdirSync(dir, { recursive: true });
const sizes = cx.members.map((m) => m.length).sort((a, b) => b - a);
writeFileSync(new URL('nyc_real_complex.json', dir), JSON.stringify({
  frame: api.frame_kind(), origin: [...api.frame_origin()], format: header.format, links: links.length,
  stations: st.map((s) => ({ x: s.x, y: s.y, name: s.name, complex: cx.of.get(s.id) })),
}));
console.log(`${st.length} constructed stations, ${cx.members.length} complexes (largest ${sizes.slice(0, 5).join(', ')}), ${links.length} links`);
