// T-3186: long ferry legs (30-150 km) find a route. Ports are clicked on land by the water as a player would;
// each case is snapped, routed in Node on the basemap's zoom 12 water (app/src/ferry/route.ts) and must give a
// route no longer than MAX_RATIO times the straight line (or `maxKm`), refused only where `expect` says so.
//   node trainworld/checks/t3186-ferry-long.mjs            all cases
//   ONLY=forth,hkmacau node trainworld/checks/t3186-ferry-long.mjs
// Tiles are kept in artifacts/trainworld/t2192/tiles (shared with t2192-ferry-routes.mjs).
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { ARTIFACTS, loadApp } from './lib.mjs';

const CACHE = join(ARTIFACTS, 't2192', 'tiles');
mkdirSync(CACHE, { recursive: true });
const UA = { 'User-Agent': 'trainworld-check' };

const CASES = {
  puget_friday: { pts: [[-122.3385, 47.6025], [-123.0140, 48.5350]], name: 'Seattle to Friday Harbor' },
  puget_townsend: { pts: [[-122.3385, 47.6025], [-122.7590, 48.1120]], name: 'Seattle to Port Townsend' },
  forth_eyemouth: { pts: [[-3.1700, 55.9810], [-2.0900, 55.8750]], name: 'Leith to Eyemouth' },
  forth_dundee: { pts: [[-3.1700, 55.9810], [-2.9650, 56.4570]], name: 'Leith to Dundee' },
  tokyo_oshima: { pts: [[139.7620, 35.6550], [139.3550, 34.7550]], name: 'Tokyo Takeshiba to Oshima, west coast' },
  georgia_nanaimo: { pts: [[-123.2730, 49.3750], [-123.9560, 49.1930]], name: 'Horseshoe Bay to Nanaimo' },
  georgia_swartz: { pts: [[-123.1320, 49.0060], [-123.4100, 48.6900]], name: 'Tsawwassen to Swartz Bay' },
  hkmacau: { pts: [[114.1600, 22.2880], [113.5795, 22.1974]], name: 'Hong Kong Central to Macau' },
  helsinki: { pts: [[24.9620, 60.1660], [24.7745, 59.4445]], name: 'Helsinki to Tallinn' },
  stockholm_sandhamn: { pts: [[18.0780, 59.3315], [18.9130, 59.2880]], name: 'Stockholm Nybroplan to Sandhamn' },
  venice_chioggia: { pts: [[12.3400, 45.4334], [12.2800, 45.2180]], name: 'Venice to Chioggia' },
  seattle_victoria: { pts: [[-122.3385, 47.6025], [-123.3680, 48.4230]], name: 'Seattle to Victoria' },
  hiroshima_matsuyama: { pts: [[132.4630, 34.3590], [132.7090, 33.8680]], name: 'Hiroshima to Matsuyama (Seto Inland Sea)' },
  kobe_takamatsu: { pts: [[135.1930, 34.6760], [134.0510, 34.3500]], name: 'Kobe to Takamatsu (Seto Inland Sea)' },
  bergen_rosendal: { pts: [[5.3210, 60.3960], [5.9610, 59.9900]], name: 'Bergen to Rosendal (Hardangerfjord)' },
  auckland_barrier: { pts: [[174.7690, -36.8430], [175.3600, -36.2000]], name: 'Auckland to Great Barrier Island' },
  piraeus_hydra: { pts: [[23.6420, 37.9400], [23.4650, 37.3500]], name: 'Piraeus to Hydra' },
  stockholm_mariehamn: { pts: [[18.1000, 59.3200], [19.9370, 60.0990]], name: 'Stockholm to Mariehamn' },
  michigan: { pts: [[-87.9040, 43.0310], [-86.2610, 43.2330]], name: 'Milwaukee to Muskegon (Lake Michigan)' },
  portsmouth_cherbourg: { pts: [[-1.1080, 50.8000], [-1.6150, 49.6500]], name: 'Portsmouth to Cherbourg (the Channel)' },
  hamburg_oslo: { pts: [[9.9680, 53.5440], [10.7350, 59.9070]], name: 'Hamburg to Oslo, over the 400 km limit', expect: 'too-long' },
  pond_berth: { pts: [[139.3655, 34.7490], [139.3550, 34.7550]], name: 'A click by a pond on Oshima is refused, not given a berth with no way out', refuse: 'far-from-water' },
  stream_berth: { pts: [[174.0030, -41.2890], [174.7770, -41.2870]], name: 'A click by a stream near Picton is refused', refuse: 'far-from-water' },
  seattle_bellingham: { pts: [[-122.3385, 47.6025], [-122.5091, 48.7204]], name: 'Seattle to Bellingham' },
  helsinki_hanko: { pts: [[24.9620, 60.1660], [22.9500, 59.8200]], name: 'Helsinki to Hanko' },
  helsinki_turku: { pts: [[24.9620, 60.1660], [22.2300, 60.4400]], name: 'Helsinki to Turku, through the archipelago' },
  copenhagen_gothenburg: { pts: [[12.5900, 55.6820], [11.9600, 57.7000]], name: 'Copenhagen to Gothenburg' },
  stockholm_helsinki: { pts: [[18.1000, 59.3200], [24.9620, 60.1660]], name: 'Stockholm to Helsinki, 395 km' },
  piraeus_santorini: { pts: [[23.6420, 37.9400], [25.4290, 36.3880]], name: 'Piraeus to Santorini' },
  wellington_picton: { pts: [[174.7770, -41.2870], [174.0070, -41.2900]], name: 'Wellington to Picton' },
  hamburg_cuxhaven: { pts: [[9.9680, 53.5440], [8.7040, 53.8700]], name: 'Hamburg to Cuxhaven (the Elbe)' },
  copenhagen_malmo: { pts: [[12.5900, 55.6820], [12.9990, 55.6130]], name: 'Copenhagen to Malmo' },
  istanbul_yalova: { pts: [[28.9745, 41.0175], [29.2650, 40.6600]], name: 'Eminonu to Yalova' },
  sydney_pittwater: { pts: [[151.2108, -33.8612], [151.3200, -33.5990]], name: 'Circular Quay to Pittwater, Palm Beach' },
};
const MAX_RATIO = 1.8;

async function tileTemplate() {
  const j = await (await fetch('https://tiles.openfreemap.org/planet', { headers: UA })).json();
  return j.tiles[0];
}

const R = await loadApp(`export * from '@app/src/ferry/route.ts'; export { decodeWater, EMPTY_TILE } from '@app/src/ferry/waterTiles.ts';`);
let TPL = null;
const decoded = new Map();
async function tile(x, y, z) {
  const key = `${z}_${x}_${y}`;
  if (decoded.has(key)) return decoded.get(key);
  const f = join(CACHE, `${key}.pbf`);
  let buf;
  if (existsSync(f)) buf = readFileSync(f);
  else {
    TPL ??= await tileTemplate();
    let r;
    for (let tries = 0; ; tries++) {
      r = await fetch(TPL.replace('{z}', z).replace('{x}', x).replace('{y}', y), { headers: UA });
      if (r.ok || r.status === 204 || r.status === 404 || tries > 3) break;
      await new Promise((ok) => setTimeout(ok, 1000));
    }
    if (r.status === 204 || r.status === 404) buf = Buffer.alloc(0);
    else if (!r.ok) throw new Error(`tile ${key}: ${r.status}`);
    else buf = Buffer.from(await r.arrayBuffer());
    writeFileSync(f, buf);
  }
  const t = buf.length ? R.decodeWater(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.length)) : R.EMPTY_TILE;
  decoded.set(key, t);
  return t;
}
const load = async (want) => {
  const got = new Map();
  for (const [x, y, z] of want) got.set(`${z}_${x}_${y}`, await tile(x, y, z));
  return (x, y, z) => got.get(`${z}_${x}_${y}`) ?? R.EMPTY_TILE;
};

let fails = 0;
const only = process.env.ONLY?.split(',');
const rows = [];
for (const [name, c] of Object.entries(CASES)) {
  if (only && !only.includes(name)) continue;
  const snaps = [];
  for (const p of c.pts) snaps.push(await R.snapPort(load, p, 800));
  if (c.refuse) {
    if (!snaps[0].ok && snaps[0].reason === c.refuse) console.log(`PASS ${name}: ${c.refuse}`);
    else { fails++; console.log(`FAIL ${name}: expected the first port refused as ${c.refuse}, got ${snaps[0].ok ? 'a berth' : snaps[0].reason}`); }
    continue;
  }
  const bad = snaps.findIndex((s) => !s.ok);
  if (bad >= 0) { fails++; console.log(`FAIL ${name}: port ${bad} does not snap (${snaps[bad].reason})`); continue; }
  const straight = R.metres(snaps[0].berth, snaps[1].berth) / 1000;
  const leg = await R.routeLeg(load, snaps[0].berth, snaps[1].berth);
  const s = leg.stats;
  const km = leg.ok ? leg.m / 1000 : 0;
  const info = s ? `cell ${s.cell.toFixed(1)}/${s.coarseCell.toFixed(0)} z${s.coarseZ} cells ${(s.cells / 1e6).toFixed(2)}M ${Math.round(s.ms)} ms` : '';
  rows.push({ name, straight: +straight.toFixed(1), km: +km.toFixed(1), ok: leg.ok, reason: leg.reason ?? '' });
  if (c.expect) {
    if (leg.reason === c.expect) console.log(`PASS ${name}: ${leg.reason}`);
    else { fails++; console.log(`FAIL ${name}: expected ${c.expect}, got ${leg.ok ? 'a route' : leg.reason}`); }
    continue;
  }
  if (!leg.ok) { fails++; console.log(`FAIL ${name}: ${leg.reason} (straight ${straight.toFixed(1)} km) ${info}`); continue; }
  if (km > MAX_RATIO * straight + 2) { fails++; console.log(`FAIL ${name}: ${km.toFixed(1)} km for ${straight.toFixed(1)} km straight ${info}`); continue; }
  console.log(`PASS ${name}: ${km.toFixed(1)} km (straight ${straight.toFixed(1)}) ${info}`);
}
console.log(fails ? `FAIL ${fails}` : 'PASS all');
process.exitCode = fails ? 1 : 0;
