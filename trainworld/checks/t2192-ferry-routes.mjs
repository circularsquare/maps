// T-2192: ferry routes over the basemap's zoom 12 water (app/src/ferry/route.ts), on the nine legs of
// notes/T-2177.md 3.3 plus a waypoint on the Grand Canal and two ports on waters that do not meet.
//   node trainworld/checks/t2192-ferry-routes.mjs            Node: snaps, routes, checks, pictures
//   node trainworld/checks/t2192-ferry-routes.mjs <dev url>  the same legs through the worker in headless Chrome: times
// Each route is checked on rasters of its own at 2.5 m (the router's finest cell, in windows along the
// route): no point under 1 m off land, at most 5% of it under 5 m (canals 10-15 m wide); Venice and Amsterdam under a bridge. Tiles are kept in artifacts/trainworld/t2192/tiles.
// Pictures: artifacts/trainworld/t2192/route-<case>.png; results: artifacts/trainworld/t2192/legs.json.
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import zlib from 'node:zlib';
import { ARTIFACTS, launchChrome, loadApp } from './lib.mjs';

const OUT = join(ARTIFACTS, 't2192');
const CACHE = join(OUT, 'tiles');
const OLD_CACHE = join(ARTIFACTS, 't2177', 'tiles');
mkdirSync(CACHE, { recursive: true });
const UA = { 'User-Agent': 'trainworld-check' };

// ports as a player would click them, on land by the water; `via` waypoints on water; real trips by boat
// (minutes, from notes/T-2177.md 3.4 and timetables; "stops" where the real boat stops on the way)
const CASES = {
  nyc: { pts: [[-74.0130, 40.7012], [-74.0715, 40.6437]], name: 'Staten Island, Whitehall to St George', real: '25' },
  bosphorus: { pts: [[28.9745, 41.0175], [29.0225, 40.9925]], name: 'Eminonu to Kadikoy', real: '20' },
  sydney: { pts: [[151.2108, -33.8612], [151.2846, -33.8004]], name: 'Circular Quay to Manly', real: '30' },
  hk: { pts: [[114.1612, 22.2878], [114.1688, 22.2938]], name: 'Star Ferry, Central to Tsim Sha Tsui', real: '9' },
  sumida: { pts: [[139.7972, 35.7108], [139.7598, 35.6515]], name: 'Sumida, Asakusa to Hinode', real: '35-40 (stops)' },
  paris: { pts: [[2.2915, 48.8605], [2.3500, 48.8535]], name: 'Seine, Eiffel Tower to Notre-Dame', real: '35 (Batobus, stops)' },
  venice: { pts: [[12.3205, 45.4392], [12.3400, 45.4334]], name: 'Venice, Piazzale Roma to San Marco', real: '25 (line 2) to 40 (line 1, stops)' },
  amsterdam: { pts: [[4.8995, 52.3785], [4.8845, 52.3620]], name: 'Amsterdam, Centraal to Rijksmuseum', real: '25-30 (canal bus)' },
  bosphorus_long: { pts: [[28.9745, 41.0175], [29.0560, 41.1640]], name: 'Eminonu to Sariyer', real: '90 (cruise, stops)' },
  rockaway: { pts: [[-74.0060, 40.7040], [-73.8215, 40.5805]], name: 'Wall Street to Rockaway Beach (a port on the ocean side)', real: '55-60 (NYC Ferry, by Jamaica Bay, stops)' },
  helsinki: { pts: [[24.9620, 60.1660], [24.7745, 59.4445]], name: 'Helsinki, Katajanokka to Tallinn', real: '120 (fast ships, 2 h)' },
  venice_wp: { pts: [[12.3205, 45.4392], [12.3400, 45.4334]], via: [[1, [12.3358, 45.4380]]], name: 'Venice with a waypoint at the Rialto', real: '' },
  apart: { pts: [[-73.9668, 40.7852], [-73.9972, 40.7642]], name: 'Central Park reservoir to the Hudson (waters apart)', real: '', expect: 'no-water-path' },
};
const NINE = ['nyc', 'bosphorus', 'sydney', 'hk', 'sumida', 'paris', 'venice', 'amsterdam', 'bosphorus_long'];

async function tileTemplate() {
  const j = await (await fetch('https://tiles.openfreemap.org/planet', { headers: UA })).json();
  return j.tiles[0];
}

const browserUrl = process.argv[2];
if (browserUrl) await browser(browserUrl);
else await node();

async function node() {
  const R = await loadApp(`export * from '@app/src/ferry/route.ts'; export { decodeWater, EMPTY_TILE } from '@app/src/ferry/waterTiles.ts';`);
  let TPL = null;
  const decoded = new Map();
  async function tile(x, y, z) {
    const key = `${z}_${x}_${y}`;
    if (decoded.has(key)) return decoded.get(key);
    let buf = null;
    for (const dir of [CACHE, OLD_CACHE]) {
      const f = join(dir, `${key}.pbf`);
      if (existsSync(f)) { buf = readFileSync(f); break; }
    }
    if (!buf) {
      TPL ??= await tileTemplate();
      const r = await fetch(TPL.replace('{z}', z).replace('{x}', x).replace('{y}', y), { headers: UA });
      if (r.status === 204 || r.status === 404) buf = Buffer.alloc(0);
      else if (!r.ok) throw new Error(`tile ${key}: ${r.status}`);
      else buf = Buffer.from(await r.arrayBuffer());
      writeFileSync(join(CACHE, `${key}.pbf`), buf);
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
  const noBridges = (get) => (x, y, z) => ({ ...get(x, y, z), bridges: [] });

  const rows = [];
  let fails = 0;
  const fail = (msg) => { fails++; console.log('FAIL ' + msg); };
  const only = process.env.ONLY?.split(',');
  for (const [name, c] of Object.entries(CASES)) {
    if (only && !only.includes(name)) continue;
    const snaps = [];
    for (const p of c.pts) snaps.push(await R.snapPort(load, p));
    const bad = snaps.findIndex((s) => !s.ok);
    if (bad >= 0) { fail(`${name}: port ${bad} does not snap (${snaps[bad].reason})`); continue; }
    const points = snaps.map((s) => ({ at: s.berth, kind: R.KIND_PORT }));
    for (const [at, ll] of (c.via ?? []).slice().reverse()) points.splice(at, 0, { at: ll, kind: R.KIND_WAYPOINT });
    const parts = [], stats = [];
    for (let k = 0; k + 1 < points.length; k++) {
      const leg = await R.routeLeg(load, points[k].at, points[k + 1].at);
      stats.push(leg.stats);
      parts.push(leg.ok ? { ok: true, pieces: leg.pieces } : { ok: false, reason: leg.reason });
    }
    const t0 = performance.now();
    const route = R.assembleLine(points, parts);
    const assembleMs = performance.now() - t0;
    if (c.expect) {
      const got = route.legs.map((l) => l.reason ?? 'ok').join(',');
      if (got !== c.expect) fail(`${name}: expected ${c.expect}, got ${got}`);
      else console.log(`PASS ${name}: ${got}`);
      rows.push({ name, title: c.name, result: got, ms: Math.round(stats.reduce((s, x) => s + (x?.ms ?? 0), 0)) });
      continue;
    }
    if (!route.ok) { fail(`${name}: ${route.legs.map((l) => l.reason ?? 'ok').join(',')}`); continue; }
    const check = await verify(R, load, noBridges, route);
    const minutes = R.runSeconds(route) / 60;
    const ms = stats.reduce((s, x) => s + x.ms, 0) + assembleMs;
    const vmin = Math.min(...route.speeds) * 3.6, vmax = Math.max(...route.speeds) * 3.6;
    const row = {
      name, title: c.name, straight_km: +(R.metres(c.pts[0], c.pts[1]) / 1000).toFixed(2), km: +(route.m / 1000).toFixed(2), minutes: +minutes.toFixed(1), real: c.real,
      ms: Math.round(ms), cell: stats.map((x) => (x.coarseZ ? `${x.cell}/${x.coarseCell.toFixed(0)}z${x.coarseZ}` : x.cell.toFixed(1))).join('+'), cells_M: +(stats.reduce((s, x) => s + x.cells, 0) / 1e6).toFixed(2), searches: stats.map((x) => x.searches).join('+'),
      k: stats.map((x) => +x.k.toFixed(2)).join('+'), plain_km: +(stats.reduce((s, x) => s + x.plainM, 0) / 1000).toFixed(2),
      fill_ms: Math.round(stats.reduce((s, x) => s + x.fill, 0)), edt_ms: Math.round(stats.reduce((s, x) => s + x.edt, 0)), search_ms: Math.round(stats.reduce((s, x) => s + x.search, 0)), assemble_ms: Math.round(assembleMs),
      points: route.kinds.length, kmh: `${vmin.toFixed(0)}-${vmax.toFixed(0)}`, clear_min_m: +check.min.toFixed(1), clear_mean_m: Math.round(check.mean), under5_m: check.under5, bridges: check.bridges, bridge_filled_m: Math.round(check.bridgeM), min_at: check.at,
      ends_m: snaps.map((s) => Math.round(R.metres(s.landing, s.berth))).join(','),
    };
    rows.push(row);
    const problems = [];
    // the search keeps HALF_M from a cell's centre to land's; a 2 m raster of its own sees piers and
    // channel edges the search's coarser cells do not, so a little under HALF_M here is the rasters
    // disagreeing; under 1 m, or much of the route under HALF_M, is the route's fault
    if (check.min < 1) problems.push(`a point ${check.min.toFixed(1)} m off land at ${check.at}`);
    if (check.under5 > 0.05 * route.m) problems.push(`${check.under5} m under ${R.HALF_M} m off land`);
    if ((name === 'venice' || name === 'amsterdam') && check.bridges <= 0) problems.push('passes under no bridge');
    if (ms > 2500) problems.push(`${Math.round(ms)} ms`);
    if (problems.length) fail(`${name}: ${problems.join(', ')}`);
    else console.log(`PASS ${name}: ${row.km} km in ${row.minutes} min (real ${c.real}), ${row.ms} ms, clear >= ${row.clear_min_m} m`);
    picture(name, parts, route, snaps, c);
  }
  writeFileSync(join(OUT, 'legs.json'), JSON.stringify(rows, null, 1));
  console.table(rows.map((r) => ({ name: r.name, km: r.km, straight: r.straight_km, min: r.minutes, real: r.real, ms: r.ms, cell: r.cell, k: r.k, clear: r.clear_min_m, under5: r.under5_m, bridges: r.bridges, kmh: r.kmh, pts: r.points, result: r.result })));
  console.log(fails ? `FAIL ${fails}` : 'PASS all');
  process.exitCode = fails ? 1 : 0;
}

/** The route's room on fine rasters of its own (2 m cells, windows along it): the least and mean room
 * every metre, and the metres under a bridge (water only because a bridge was filled). */
async function verify(R, load, noBridges, route) {
  const F = new R.Frame(route.ll[0], route.ll[1]);
  const samples = [];
  for (let i = 1; i < route.kinds.length; i++) {
    const ax = F.x(route.ll[2 * i - 2]), ay = F.y(route.ll[2 * i - 1]), bx = F.x(route.ll[2 * i]), by = F.y(route.ll[2 * i + 1]);
    const n = Math.max(1, Math.ceil(Math.hypot(bx - ax, by - ay)));
    for (let s = 0; s < n; s++) samples.push([ax + ((bx - ax) * s) / n, ay + ((by - ay) * s) / n]);
  }
  // the router's finest cell: land thinner than that (a seam between two water polygons) it cannot see
  const WIN = 400, MARGIN = 80, CELL = 2.5;
  const groups = new Map();
  for (const [x, y] of samples) {
    const k = `${Math.floor(x / WIN)},${Math.floor(y / WIN)}`;
    if (!groups.has(k)) groups.set(k, []);
    groups.get(k).push([x, y]);
  }
  let min = Infinity, sum = 0, bridgeM = 0, at = null, under5 = 0;
  for (const [k, pts] of groups) {
    const [gx, gy] = k.split(',').map(Number);
    const cx = (gx + 0.5) * WIN, cy = (gy + 0.5) * WIN;
    const frame = new R.Frame(F.lon(cx), F.lat(cy));
    const half = WIN / 2 + MARGIN;
    const box = { frame, x0: -half, y0: -half, cell: CELL, W: Math.ceil((2 * half) / CELL), H: Math.ceil((2 * half) / CELL) };
    const get = await load(R.tilesOf(box));
    const r = R.buildRaster(get, box);
    const r0 = R.buildRaster(noBridges(get), box);
    for (const [x, y] of pts) {
      const lon = F.lon(x), lat = F.lat(y);
      const c = Math.floor((frame.x(lon) - box.x0) / CELL), w = Math.floor((frame.y(lat) - box.y0) / CELL), i = w * box.W + c;
      const room = r.water[i] ? r.dist[i] - CELL / 2 : 0;
      if (room < min) { min = room; at = [+lon.toFixed(5), +lat.toFixed(5)]; }
      if (room < R.HALF_M) under5++;
      sum += Math.min(room, 1000);
      if (r.water[i] && !r0.water[i]) bridgeM++;
    }
  }
  // bridges crossed: the route's stretches against the bridge lines of the tiles under it
  const segs = [];
  const fx = (lon) => F.x(lon), fy = (lat) => F.y(lat);
  let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
  for (const [x, y] of samples) { xmin = Math.min(xmin, x); xmax = Math.max(xmax, x); ymin = Math.min(ymin, y); ymax = Math.max(ymax, y); }
  const bbox = { frame: F, x0: xmin, y0: ymin, cell: 1000, W: Math.ceil((xmax - xmin) / 1000) + 1, H: Math.ceil((ymax - ymin) / 1000) + 1 };
  const tiles = R.tilesOf(bbox);
  const get = tiles.length <= 80 ? await load(tiles) : null;
  for (const [tx, ty, tz] of get ? tiles : []) {
    const t = get(tx, ty, tz), ext = t.extent, nt = 2 ** tz;
    const lon = (px) => ((tx + px / ext) / nt) * 360 - 180;
    const lat = (py) => { const n = Math.PI - (2 * Math.PI * (ty + py / ext)) / nt; return (180 / Math.PI) * Math.atan(Math.sinh(n)); };
    for (const line of t.bridges) for (let k = 2; k < line.length; k += 2) segs.push([fx(lon(line[k - 2])), fy(lat(line[k - 1])), fx(lon(line[k])), fy(lat(line[k + 1]))]);
  }
  const cross = (a, b, c, d) => { const o = (p, q, r) => Math.sign((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])); return o(a, b, c) !== o(a, b, d) && o(c, d, a) !== o(c, d, b); };
  let bridges = 0;
  for (let i = 1; i < route.kinds.length; i++) {
    const a = [fx(route.ll[2 * i - 2]), fy(route.ll[2 * i - 1])], b = [fx(route.ll[2 * i]), fy(route.ll[2 * i + 1])];
    for (const s of segs) {
      if (Math.max(s[0], s[2]) < Math.min(a[0], b[0]) || Math.min(s[0], s[2]) > Math.max(a[0], b[0]) || Math.max(s[1], s[3]) < Math.min(a[1], b[1]) || Math.min(s[1], s[3]) > Math.max(a[1], b[1])) continue;
      if (cross(a, b, [s[0], s[1]], [s[2], s[3]])) bridges++;
    }
  }
  return { min, at, mean: sum / samples.length, bridgeM, bridges, under5 };
}

/** The first leg's raster shaded by room, the route in red (darker where slow), the clicks and berths. */
function picture(name, parts, route, snaps, c) {
  const leg = parts.find((p) => p.ok);
  if (!leg) return;
  // the piece with the widest view: the coarse middle of a leg planned coarse
  const r = leg.pieces.map((p) => p.raster).sort((p, q) => q.W * q.cell - p.W * p.cell)[0], sc = Math.max(1, Math.ceil(Math.max(r.W, r.H) / 1400)), PW = Math.floor(r.W / sc), PH = Math.floor(r.H / sc);
  const img = Buffer.alloc(PW * PH * 3);
  for (let y = 0; y < PH; y++) for (let x = 0; x < PW; x++) {
    const i = (r.H - 1 - y * sc) * r.W + x * sc, o = (y * PW + x) * 3;
    if (r.water[i]) { const t = Math.min(1, r.dist[i] / 400); img[o] = 150 - 40 * t; img[o + 1] = 200 - 30 * t; img[o + 2] = 215 - 10 * t; }
    else { img[o] = 242; img[o + 1] = 236; img[o + 2] = 222; }
  }
  const px = (lon, lat) => [(r.frame.x(lon) - r.x0) / r.cell / sc, PH - 1 - (r.frame.y(lat) - r.y0) / r.cell / sc];
  const dot = ([cx, cy], rad, col) => { for (let dy = -rad; dy <= rad; dy++) for (let dx = -rad; dx <= rad; dx++) { if (dx * dx + dy * dy > rad * rad) continue; const x = Math.round(cx + dx), y = Math.round(cy + dy); if (x < 0 || y < 0 || x >= PW || y >= PH) continue; const o = (y * PW + x) * 3; img[o] = col[0]; img[o + 1] = col[1]; img[o + 2] = col[2]; } };
  for (let i = 1; i < route.kinds.length; i++) {
    const a = px(route.ll[2 * i - 2], route.ll[2 * i - 1]), b = px(route.ll[2 * i], route.ll[2 * i + 1]);
    const v = Math.min(route.speeds[i - 1], route.speeds[i]) / (32 / 3.6), n = Math.ceil(Math.hypot(b[0] - a[0], b[1] - a[1]) * 2) + 1;
    for (let s = 0; s <= n; s++) dot([a[0] + ((b[0] - a[0]) * s) / n, a[1] + ((b[1] - a[1]) * s) / n], 1, [120 + 110 * v, 40, 60 + 20 * v]);
  }
  for (const p of c.pts) dot(px(...p), 5, [40, 40, 60]);
  for (const s of snaps) { dot(px(...s.berth), 4, [255, 255, 255]); dot(px(...s.landing), 2, [40, 40, 60]); }
  for (const [, ll] of c.via ?? []) dot(px(...ll), 4, [255, 200, 0]);
  writeFileSync(join(OUT, `route-${name}.png`), png(PW, PH, img));
}

function png(w, h, rgb) {
  const crcT = new Int32Array(256).map((_, n) => { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; return c; });
  const crc = (b) => { let c = -1; for (const x of b) c = crcT[(c ^ x) & 255] ^ (c >>> 8); return (c ^ -1) >>> 0; };
  const chunk = (t, d) => { const len = Buffer.alloc(4); len.writeUInt32BE(d.length); const td = Buffer.concat([Buffer.from(t), d]); const cr = Buffer.alloc(4); cr.writeUInt32BE(crc(td)); return Buffer.concat([len, td, cr]); };
  const raw = Buffer.alloc((w * 3 + 1) * h);
  for (let y = 0; y < h; y++) { raw[y * (w * 3 + 1)] = 0; rgb.copy(raw, y * (w * 3 + 1) + 1, y * w * 3, (y + 1) * w * 3); }
  const ihdr = Buffer.alloc(13); ihdr.writeUInt32BE(w, 0); ihdr.writeUInt32BE(h, 4); ihdr[8] = 8; ihdr[9] = 2;
  return Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), chunk('IHDR', ihdr), chunk('IDAT', zlib.deflateSync(raw)), chunk('IEND', Buffer.alloc(0))]);
}

/** The same legs through the real worker in headless Chrome, its tiles from the tile server: each leg's
 * ms after its tiles (the worker's own figure) and with the fetch. */
async function browser(url) {
  const tpl = await tileTemplate();
  const { ctx, close } = await launchChrome({ prefix: 'tw-t2192-', args: [] });
  let fails = 0;
  try {
    const page = await ctx.newPage();
    await page.goto(new URL('src/ferry/route.ts', url).href);
    const out = await page.evaluate(async ([tpl, cases, nine]) => {
      const w = new Worker('/src/workers/ferryWater.worker.ts', { type: 'module' });
      let id = 1;
      const waiting = new Map();
      w.onmessage = (e) => { waiting.get(e.data.id)?.(e.data); waiting.delete(e.data.id); };
      const ask = (m) => new Promise((res) => { m.id = id++; waiting.set(m.id, res); w.postMessage(m); });
      w.postMessage({ kind: 'template', url: tpl });
      const rows = [];
      for (const name of nine) {
        const c = cases[name];
        const snaps = [];
        for (const at of c.pts) snaps.push((await ask({ kind: 'port', at })).snap);
        if (snaps.some((s) => !s.ok)) { rows.push({ name, error: 'snap' }); continue; }
        const t0 = performance.now();
        const m = await ask({ kind: 'line', points: snaps.map((s) => ({ at: s.berth, kind: 1 })) });
        const total = performance.now() - t0;
        // again, every tile in hand: the search alone
        rows.push({ name, ok: m.route.ok, km: +(m.route.m / 1000).toFixed(2), worker_ms: Math.round(m.stats[0]?.ms ?? -1), with_fetch_ms: Math.round(total), cells_M: +((m.stats[0]?.cells ?? 0) / 1e6).toFixed(2) });
      }
      w.terminate();
      return rows;
    }, [tpl, CASES, [...NINE, 'rockaway', 'helsinki']]);
    console.table(out);
    for (const r of out) {
      if (!r.ok) { fails++; console.log(`FAIL ${r.name}: ${r.error ?? 'no route'}`); }
      else if (r.worker_ms > 2500) { fails++; console.log(`FAIL ${r.name}: ${r.worker_ms} ms after its tiles`); }
      else console.log(`PASS ${r.name}: ${r.worker_ms} ms after its tiles, ${r.with_fetch_ms} ms with them`);
    }
    writeFileSync(join(OUT, 'browser.json'), JSON.stringify(out, null, 1));
  } finally {
    await close();
  }
  console.log(fails ? `FAIL ${fails}` : 'PASS all');
  process.exitCode = fails ? 1 : 0;
}
