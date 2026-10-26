// T-1168: the related-lines query (`TrackApi.line_related`) on a save, as the clock worker asks
// it: every line timed, and the groups of the lines named listed. Run from the maps root:
//   node trainworld/checks/t1168-related.mjs [save] [line name ...]
// Defaults: the real New York save, and its 2, 4, 6X, 7 and 7X. Writes every line's groups to
// ../artifacts/trainworld/t1168/related-<save>.json.
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import { basename } from 'node:path';
import init, { TrackApi } from '../app/src/wasm/track/sim.js';
import { loadApp } from './lib.mjs';

const { buildComplexes } = await loadApp('src/game/complexes.ts');
const { decodeRelated } = await loadApp('src/game/relatedLines.ts');
const { mercK } = await loadApp('src/game/frame.ts');

const savePath = process.argv[2] ?? new URL('../T-007-real-nyc.save', import.meta.url).pathname.replace(/^\/(\w:)/, '$1');
const wanted = process.argv.slice(3).length ? process.argv.slice(3) : ['subway 2', 'subway 4', 'subway 6X', 'subway 7', 'subway 7X'];

await init({ module_or_path: readFileSync(new URL('../app/src/wasm/track/sim_bg.wasm', import.meta.url)) });
const raw = gunzipSync(readFileSync(savePath));
const n = raw.readUInt32LE(4);
const api = new TrackApi(-73.985, 40.758, true);
if (!api.load(raw.subarray(8 + n))) throw new Error('save did not load');
const merc = api.frame_kind() === 2;

const ni = api.nodes_info(), geo = [];
let top = 0;
for (let i = 0; i < ni.length; i += 10) {
  top = Math.max(top, ni[i]);
  if (ni[i + 7] > 0) geo.push({ id: String(ni[i]), x: ni[i + 1], y: ni[i + 2], heading: ni[i + 9], length: ni[i + 7] });
}
const fl = api.station_links(), links = [];
for (let k = 0; k + 1 < fl.length; k += 2) links.push([String(fl[k]), String(fl[k + 1])]);
const cx = buildComplexes(geo, links, (y) => (merc ? mercK(y) : 1));
const of = new Uint32Array(top + 1).fill(0xffffffff);
for (const g of geo) of[Number(g.id)] = cx.of.get(g.id);
const cxName = (c) => api.node_name(Number(cx.members[c][0]));

const li = api.lines_info(), ids = [];
for (let i = 0; i < li.length; i += 22) ids.push(li[i]);
const name = (l) => api.line_name(l);
const all = {};
let total = 0, worst = [0, -1], groups = 0, beside = 0;
for (const l of ids) {
  const t0 = performance.now();
  const v = api.line_related(l, of);
  const dt = performance.now() - t0;
  total += dt;
  if (dt > worst[0]) worst = [dt, l];
  const g = decodeRelated(v);
  groups += g.length;
  beside += g.filter((x) => x.kind === 'beside').length;
  all[name(l)] = g;
}
console.log(`${basename(savePath)}: ${ids.length} lines, ${geo.length} stations; ${groups} groups (${beside} beside); ${(total / ids.length).toFixed(2)} ms a line, worst ${worst[0].toFixed(1)} ms (${name(worst[1])})`);

const km = (m) => (m / 1000).toFixed(1);
for (const w of wanted) {
  const l = ids.find((x) => name(x) === w);
  if (l === undefined) { console.log(`no line ${w}`); continue; }
  const stops = [...api.line_stops(l)], s = api.line_stop_s(l, 0);
  const at = (sL) => { let b = 0; for (let k = 0; k < s.length; k++) if (Math.abs(s[k] - sL) < Math.abs(s[b] - sL)) b = k; return api.node_name(stops[b]); };
  console.log(`\n== ${w} (${stops.length} stops)`);
  for (const g of all[w]) {
    const st = g.meetings.length ? `${cxName(g.meetings[0].complex)} .. ${cxName(g.meetings[g.meetings.length - 1].complex)}` : `${at(g.sL0)} .. ${at(g.sL1)}`;
    const stray = g.gaps.length ? ` stray<=${Math.round(Math.max(...g.gaps.map((x) => x.stray)))}m` : '';
    const split = g.split ? ` split(${Number.isNaN(g.split.stray) ? 'turned back' : Math.round(g.split.stray) + 'm'}/${km(g.split.run)}km)` : '';
    console.log(`  ${name(g.line).padEnd(26)} ${g.kind.padEnd(8)} ${g.shared ? 'track' : '     '} ${String(g.dir).padStart(2)} ${km(g.sL0).padStart(5)}-${km(g.sL1).padEnd(5)} meet ${String(g.meetings.length).padStart(2)} own ${String(g.own.length).padStart(2)} before ${g.stopsBefore} after ${g.stopsAfter}${stray}${split}  ${st}`);
  }
}

const dir = new URL('../../artifacts/trainworld/t1168/', import.meta.url);
mkdirSync(dir, { recursive: true });
writeFileSync(new URL(`related-${basename(savePath, '.save')}.json`, dir), JSON.stringify(all));
