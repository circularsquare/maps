// T-1374 (T-264 stage 3): a superholiday's going-out trips lean to touristy zones. The market of a
// few regions' packs (Kansai with Tokyo and Nagoya, Hangzhou with its neighbours, New York), with no
// network: for named touristy zones and for every zone of their packs, the longer trips arriving
// on a going-out day per weekday (twice a weekday's, times their lean), against ordinary zones of
// about the same people in the same place; and a weekday the same with the attractions as without.
//   node trainworld/checks/t1374-touristy.mjs      (from the maps root; under a minute)
import { readFileSync } from 'node:fs';
import { loadApp, marketRows, registry } from './lib.mjs';
import init, { MarketPacks } from '../app/src/wasm/demand/sim.js';

await init({ module_or_path: readFileSync(new URL('../app/src/wasm/demand/sim_bg.wasm', import.meta.url)) });
globalThis.location ??= { search: '', hash: '', href: 'http://localhost/' };
const { marketWorld } = await loadApp("export { marketWorld } from '@app/src/game/cities.ts';");
const reg = registry();
const PACKS = new URL('../data/packs/', import.meta.url);
const att = JSON.parse(readFileSync(new URL('../app/src/game/attractions.json', import.meta.url), 'utf8')).rows.map((r) => [r[2], r[3], r[4]]);
// the world as the demand workers give it, with the attractions they have loaded by then
const bare = JSON.parse(marketWorld(marketRows()));
const worldWith = JSON.stringify({ ...bare, attractions: att }), worldWithout = JSON.stringify({ ...bare, attractions: [] });
const OUT_DAY = 4;

let failed = 0;
const report = (name, ok, detail = '') => { if (!ok) failed++; console.log(`${ok ? 'PASS' : 'FAIL'} ${name} ${detail}`); };

const REGIONS = [
  { name: 'Kansai', packs: ['osa', 'shg', 'tyo', 'ngo', 'wky'], named: [['Kyoto (Higashiyama)', 'osa', 135.78, 35.0], ['Kyoto (Kinkaku-ji)', 'osa', 135.73, 35.04], ['Nara', 'osa', 135.83, 34.68], ['Osaka Bay (Universal Studios)', 'osa', 135.471, 34.676]] },
  { name: 'Hangzhou', packs: ['hag', ...near('hag', 250)], named: [['Hangzhou (West Lake, Lingyin)', 'hag', 120.08, 30.25], ['Hangzhou (West Lake, east shore)', 'hag', 120.17, 30.24]] },
  { name: 'New York', packs: ['nyc', ...near('nyc', 200)], named: [['Manhattan (the Met)', 'nyc', -73.96, 40.78]] },
];
function near(id, km) {
  const c = reg.cities.find((x) => x.id === id), k = Math.cos((c.center[1] * Math.PI) / 180);
  return reg.cities.filter((x) => x.id !== id && Math.hypot((x.center[0] - c.center[0]) * k, x.center[1] - c.center[1]) * 111.32 < km).map((x) => x.id);
}
const M_PER_DEG = 111320;
const load = (id) => { try { return readFileSync(new URL(`${id}.market.json`, PACKS), 'utf8'); } catch { return null; } };

for (const region of REGIONS) {
  const texts = region.packs.map((id) => [id, load(id)]).filter(([, t]) => t);
  const markets = {};
  for (const w of [worldWith, worldWithout]) {
    const mp = new MarketPacks(w);
    for (const [, t] of texts) mp.add(t);
    markets[w === worldWith ? 'with' : 'without'] = mp.market();
  }
  const zonesOf = new Map();
  for (const [id, t] of texts.filter(([id]) => region.named.some((n) => n[1] === id))) {
    const p = JSON.parse(t), nz = p.zones.length, by = Array.from({ length: nz }, () => ({ xy: [], pop: 0, x: 0, y: 0 }));
    for (const s of p.sites) { const z = by[s[4]]; z.xy.push(s[0], s[1]); z.pop += s[2]; z.x += s[0] * s[2]; z.y += s[1] * s[2]; }
    zonesOf.set(id, { place: p.place, frame: p.frame, zones: by });
  }
  // trips ending in a zone a day, all modes, with every place (and places with no zone)
  const ends = (api, place, xy, day) => {
    const v = api.selection_all_day(place, Float32Array.from(xy), new Float32Array(xy.length / 2).fill(1), day, 31);
    let s = v[v.length - 1];
    for (let p = 0; p < reg.market.places.length; p++) s += v[1 + 8 * p + 4];
    return s;
  };
  const sameWeekday = [];
  for (const [id, z] of zonesOf) {
    const k0 = Math.cos((z.frame.lat * Math.PI) / 180);
    const at = z.zones.map((zz) => (zz.pop > 0 ? [z.frame.lon + zz.x / zz.pop / (M_PER_DEG * k0), z.frame.lat + zz.y / zz.pop / M_PER_DEG] : [NaN, NaN]));
    const score = (k) => {
      const zz = z.zones[k];
      const wd = ends(markets.with, z.place, zz.xy, 0), wd0 = ends(markets.without, z.place, zz.xy, 0);
      const out = ends(markets.with, z.place, zz.xy, OUT_DAY), out0 = ends(markets.without, z.place, zz.xy, OUT_DAY);
      sameWeekday.push(wd === wd0);
      return { k, pop: zz.pop, wd, r: wd > 0 ? out / wd : NaN, r0: wd > 0 ? out0 / wd : NaN };
    };
    for (const [label, pid, lon, lat] of region.named.filter((n) => n[1] === id)) {
      const k = Math.cos((lat * Math.PI) / 180), dist = (i) => Math.hypot((at[i][0] - lon) * k, at[i][1] - lat);
      const ti = z.zones.reduce((b, zz, i) => (zz.pop > 0 && (b < 0 || dist(i) < dist(b)) ? i : b), -1);
      const t = score(ti);
      // ordinary zones of about its people (within a factor 1.5) that gain nothing from tourists
      // themselves (their trips with touristy zones lean away from them, so they lose a little)
      const cand = z.zones.map((zz, i) => i).filter((i) => i !== ti && z.zones[i].pop > t.pop / 1.5 && z.zones[i].pop < t.pop * 1.5);
      const like = [];
      for (let j = 0; j < cand.length && like.length < 25; j += Math.max(1, Math.floor(cand.length / 40))) {
        const r = score(cand[j]);
        if (r.wd > 1 && r.r <= r.r0 + 1e-9) like.push(r.r);
      }
      like.sort((a, b) => a - b);
      const med = like.length ? like[Math.floor(like.length / 2)] : NaN;
      console.log(`  ${region.name}: ${label}, zone ${t.k} of ${pid} (${Math.round(t.pop / 1000)}k people, ${Math.round(t.wd)} longer trips ending there a weekday): going out ${t.r.toFixed(2)} times a weekday's (${t.r0.toFixed(2)} without attractions); ${like.length} ordinary zones of about its people, median ${med.toFixed(2)}: ${(100 * (t.r / med - 1)).toFixed(0)}% more`);
    }
  }
  report(`${region.name}: a weekday's longer trips the same with attractions as without`, sameWeekday.every(Boolean), `(${sameWeekday.length} zones)`);
}
process.exit(failed ? 1 : 0);
