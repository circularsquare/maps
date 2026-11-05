// T-2169: the network diagram's water (diagram/waterReal.ts, diagram/water.ts) on the real New York save, Toronto's
// real network, Tokyo and all Japan. Each save is laid out as the page's worker does (T2464_CACHE=<folder> reads and
// writes laid out saves there, shared with t2464-borders.mjs), its zones' water masks read from the packs, the water
// found on the ground and drawn. Checked, as drawn: every hop that crosses a drawn river on the ground crosses its band
// once, between the same two stations, and no band crosses a line anywhere else; every station's pill is on land
// (outside the sea's fill). Reported: rivers and crossings, and the times (the layout's and the water's).
//   node trainworld/checks/t2169-water.mjs [nyc,tor,tyo,jpn] [--png]      (figures in artifacts/trainworld/t2169/)
import { existsSync, mkdirSync, openSync, readSync, closeSync, readFileSync, writeFileSync } from 'node:fs';
import { serialize, deserialize } from 'node:v8';
import { deflateSync } from 'node:zlib';
import { openSave, SAVES } from './diagram-saves.mjs';
import { ARTIFACTS, loadApp } from './lib.mjs';

globalThis.Path2D ??= class { moveTo() {} lineTo() {} quadraticCurveTo() {} };
const L = await loadApp(`export * from '@app/src/diagram/layout.ts'; export { drawing, pillSize } from '@app/src/diagram/draw.ts';
export { realWater } from '@app/src/diagram/waterReal.ts'; export { inRings } from '@app/src/diagram/borders.ts'; export { diagramWater } from '@app/src/diagram/water.ts'; export { toMerc01, EARTH_C } from '@app/src/map/geo.ts';
export { makeFrame } from '@app/src/game/frame.ts'; export { placeNames, splitName } from '@app/src/diagram/names.ts';`);
const TW = new URL('../', import.meta.url);
SAVES.tor ??= { file: new URL('data/packs/real/tor.save', TW), lon: -79.38, lat: 43.65 };
const args = process.argv.slice(2).filter((a) => !a.startsWith('--'));
const PNG = process.argv.includes('--png');
const which = (args[0] ?? 'nyc,tor,tyo,jpn').split(',');
const REG = JSON.parse(readFileSync(new URL('app/src/game/registry.json', TW)));
const PACKS = new URL('data/packs/', TW);
const CACHE = process.env.T2464_CACHE;
const OUT = `${ARTIFACTS}t2169/`;
mkdirSync(OUT, { recursive: true });
let failed = 0;
const say = (ok, name, info) => { if (!ok) failed++; console.log(`${ok ? 'PASS' : 'FAIL'} ${name}${info ? ': ' + info : ''}`); };

/** the save laid out (t2464-borders.mjs' cache), its graph built again for the hops' tracks, and its frame */
function laidOut(city) {
  const file = CACHE && `${CACHE}/${city}.bin`;
  const s = openSave(city), O = SAVES[city];
  const { g } = s.graph();
  const ni = s.api.nodes_info(), xy = new Map();
  for (let i = 0; i < ni.length; i += 10) xy.set(ni[i], [ni[i + 1], ni[i + 2]]);
  const names = [...g.key].map((k) => s.name(k) ?? '');
  s.api.free();
  const fr = L.makeFrame({ kind: 'equirect', lon: O.lon, lat: O.lat }), k = g.k / L.EARTH_C;
  let fx = 0, fy = 0, w = Infinity, so = Infinity, e = -Infinity, n = -Infinity;
  g.key.forEach((key, i) => {
    const [lng, lat] = fr.toLonLat(...xy.get(key)), [mx, my] = L.toMerc01(lng, lat);
    (fx += mx - g.x[i] * k), (fy += my + g.y[i] * k);
    (w = Math.min(w, lng)), (e = Math.max(e, lng)), (so = Math.min(so, lat)), (n = Math.max(n, lat));
  });
  const frame = [fx / g.key.length, fy / g.key.length, 1 / k];
  let r;
  if (file && existsSync(file)) {
    r = deserialize(readFileSync(file));
    if (r.g.key.length !== g.key.length || r.g.ea.length !== g.ea.length) throw new Error(`${city}: the cached graph is not this one`);
    (r.g.hop = g.hop), (r.g.hopAt = g.hopAt);
  } else {
    const t0 = performance.now();
    const lay = L.layoutDiagram(g);
    const layoutMs = performance.now() - t0;
    r = { g, lay, d: L.drawing(g, lay), names, layoutMs };
    if (file) writeFileSync(file, serialize({ ...r, g: { ...g, hop: undefined, hopAt: undefined } }));
  }
  return { ...r, names, frame, box: [w, so, e, n] };
}

/** the water masks of the zones whose boxes meet `box` (lng, lat), read from the packs as the page fetches them */
function masksNear(box) {
  const out = [];
  for (const c of REG.cities) {
    const b = c.box;
    if (!b || b[0] > box[2] + 0.3 || b[2] < box[0] - 0.3 || b[1] > box[3] + 0.3 || b[3] < box[1] - 0.3) continue;
    const head = JSON.parse(readFileSync(new URL(`${c.id}.json`, PACKS)));
    const row = head.arrays.find((a) => a.name === 'water_row'), xs = head.arrays.find((a) => a.name === 'water_x');
    if (!head.water || !row || !xs) continue;
    const fd = openSync(new URL(`${c.id}.bin`, PACKS), 'r');
    const rows = new Uint32Array(row.count), runs = new Uint16Array(xs.count);
    readSync(fd, new Uint8Array(rows.buffer), 0, row.count * 4, row.offset);
    readSync(fd, new Uint8Array(runs.buffer), 0, xs.count * 2, xs.offset);
    closeSync(fd);
    const w = head.water;
    out.push({ id: c.id, cell: w.cell_m, x0: w.origin_m[0], y0: w.origin_m[1], w: w.size[0], h: w.size[1], rows, runs, lon: head.origin.lon, lat: head.origin.lat });
  }
  return out;
}

const figures = {};
for (const city of which) {
  const S = laidOut(city);
  const t0 = performance.now();
  const masks = masksNear(S.box);
  const tRead = performance.now() - t0;
  const t1 = performance.now();
  const real = L.realWater(S.g, masks, S.frame);
  const realMs = performance.now() - t1;
  const crossings = real.arcs.reduce((a, r) => a + r.cross.length, 0);
  const crossed = real.arcs.filter((r) => r.cross.length).length;
  const f = { stations: S.g.key.length, masks: masks.length, arcs: real.arcs.length, crossed, crossings, evenDropped: real.evenDropped, ms: { layout: Math.round(S.layoutMs ?? 0), read: Math.round(tRead), real: Math.round(realMs), ...Object.fromEntries(Object.entries(real.ms).map(([k, v]) => [k, Math.round(v)])) } };
  const t2 = performance.now();
  const W = L.diagramWater(S.g, S.lay.x, S.lay.y, S.d, real, true);
  f.ms.drawn = Math.round(performance.now() - t2);
  f.drawn = W.stats;
  figures[city] = f;
  console.log(`     ${city}: ${f.stations} stations, ${f.masks} masks; ${f.arcs} river arcs, ${f.crossed} crossed, ${f.crossings} crossings (${f.evenDropped} hops crossing an arc evenly); ms ${JSON.stringify(f.ms)}`);
  console.log(`     drawn: ${JSON.stringify(W.stats)}`);
  const c = crossCheck(S, W, real, f);
  say(c.missing.length === 0, `${city}: every hop crossing a drawn river on the ground crosses its band`, `${c.missing.length} of ${c.expected} missing ${c.missing.slice(0, 8).join('; ')}`);
  say(c.multi.length === 0, `${city}: each line on such a hop crosses the band once`, `${c.multi.length} lines crossing it evenly ${c.multi.slice(0, 8).join('; ')}`);
  say(c.wrong.length === 0, `${city}: no band crosses a hop that does not cross that river`, `${c.wrong.length} wrong ${c.wrong.slice(0, 8).join('; ')}`);
  const wet = [];
  S.d.pills.forEach((p, q) => { const mx = (p[0] + p[2]) / 2, my = (p[1] + p[3]) / 2; if (L.inRings(W.seaRings ?? [], mx, my)) wet.push(S.names[S.d.pillOf[q]]); });
  f.seaStats = W.seaStats; f.wetPills = wet.length;
  console.log(`     sea: ${JSON.stringify(W.seaStats)}`);
  say(wet.length === 0, `${city}: every station drawn on land`, `${wet.length} pills in the sea: ${wet.slice(0, 8).join(", ")}`);
  if (PNG) groundPng(city, S, real);
  if (PNG) writeFileSync(`${OUT}render-${city}.js`, `window.TW_DATA = window.TW_DATA || {}; window.TW_DATA[${JSON.stringify(city)}] = ${JSON.stringify(renderData(S, W), (k, v) => (ArrayBuffer.isView(v) ? Array.from(v) : v))};`);
}

/** the bands' crossings with the ribbons as drawn, against the crossings on the ground */
function crossCheck(S, W, real, f) {
  const { g, d, names } = S, n = g.key.length;
  const edgeOf = new Map();
  for (let e = 0; e < g.ea.length; e++) edgeOf.set(g.ea[e] * n + g.eb[e], e);
  const hopName = (e) => `${names[g.ea[e]]} - ${names[g.eb[e]]}`;
  // ribbon segments with their graph edge
  const segs = [];
  d.ribbons.forEach((r, li) => {
    const chain = g.lines[d.lineOf[li]].chain, at = d.chainAt[li];
    for (let k = 0; k + 3 < r.length; k += 2) {
      const ci = at[k / 2], u = chain[ci], w = chain[ci + 1];
      const e = w === undefined ? -1 : edgeOf.get(Math.min(u, w) * n + Math.max(u, w)) ?? -1;
      segs.push([r[k], r[k + 1], r[k + 2], r[k + 3], li, e]);
    }
  });
  const G = 4 * d.pitch, grid = new Map();
  segs.forEach((s, i) => {
    for (let a = Math.floor(Math.min(s[0], s[2]) / G); a <= Math.floor(Math.max(s[0], s[2]) / G); a++) for (let b = Math.floor(Math.min(s[1], s[3]) / G); b <= Math.floor(Math.max(s[1], s[3]) / G); b++) {
      const k = `${a},${b}`;
      grid.has(k) ? grid.get(k).push(i) : grid.set(k, [i]);
    }
  });
  const inter = (ax, ay, bx, by, cx, cy, dx, dy) => {
    const ux = bx - ax, uy = by - ay, vx = dx - cx, vy = dy - cy, den = ux * vy - uy * vx;
    if (Math.abs(den) < 1e-12) return false;
    const r = ((cx - ax) * vy - (cy - ay) * vx) / den, t = ((cx - ax) * uy - (cy - ay) * ux) / den;
    return r >= 0 && r < 1 && t >= 0 && t < 1;
  };
  let expected = 0;
  const missing = [], multi = [], wrong = [];
  for (const run of W.runs) {
    const want = new Set(run.edges);
    expected += want.size;
    const count = new Map(); // `${li}|${e}` -> n
    for (const p of run.parts)
    for (let k = 0; k + 3 < p.length; k += 2) {
      const cand = new Set();
      for (let a = Math.floor(Math.min(p[k], p[k + 2]) / G); a <= Math.floor(Math.max(p[k], p[k + 2]) / G); a++) for (let b = Math.floor(Math.min(p[k + 1], p[k + 3]) / G); b <= Math.floor(Math.max(p[k + 1], p[k + 3]) / G); b++) for (const i of grid.get(`${a},${b}`) ?? []) cand.add(i);
      for (const i of cand) {
        const s = segs[i];
        if (!inter(p[k], p[k + 1], p[k + 2], p[k + 3], s[0], s[1], s[2], s[3])) continue;
        const key = `${s[4]}|${s[5]}`;
        count.set(key, (count.get(key) ?? 0) + 1);
      }
    }
    const crossedEdges = new Set();
    for (const [key, c] of count) {
      const [li, e] = key.split('|').map(Number);
      if (want.has(e)) {
        crossedEdges.add(e);
        if (c % 2 === 0) multi.push(`${hopName(e)} (line ${g.lines[d.lineOf[li]].id}) x${c}`);
      } else wrong.push(`arc ${run.arc}: ${e < 0 ? 'a line end' : hopName(e)} (line ${g.lines[d.lineOf[li]].id})`);
    }
    for (const e of want) if (!crossedEdges.has(e)) missing.push(`arc ${run.arc}: ${hopName(e)}`);
  }
  f.check = { expected, missing: missing.length, multi: multi.length, wrong: wrong.length };
  return { expected, missing, multi, wrong };
}

/** what t2169-shots.mjs paints: the drawing, the water and the names */
function renderData(S, W) {
  const { d, lay, g } = S;
  const names = S.names.map((t) => t.split('/')[0]);
  const fontPx = 12, wOf = (t) => t.length * 6.6;
  const width = Float32Array.from(names, wOf), width2 = Float32Array.from(names, (t) => { const p = L.splitName(t); return p ? Math.max(wOf(p[0]), wOf(p[1])) : 0; });
  const plan = L.placeNames(g, lay, d, names, width, width2, fontPx);
  return { plan, pitch: d.pitch, stroke: d.stroke, lines: d.lines.map((l) => ({ colour: l.colour, fast: l.fast, cmds: [...l.cmds] })), lineOf: d.lineOf, ribbons: d.ribbons.map((r) => [...r]), chainAt: d.chainAt.map((c) => [...c]), pills: d.pills, pillOf: d.pillOf, passages: d.passages, passageOf: d.passageOf, hooks: d.hooks, small: [...d.small], box: d.box, x: [...lay.x], y: [...lay.y], text: names, water: { rivers: [...W.rivers], sea: W.sea ? [...W.sea] : [], band: W.band }, runs: (W.runs ?? []).flatMap((r) => r.parts.map((p, i) => ({ p: [...p], marks: i ? [] : r.marks, arc: r.arc }))), failed: W.failed ?? [] };
}
writeFileSync(`${OUT}figures.json`, JSON.stringify(figures, null, 1));
process.exit(failed ? 1 : 0);

/** a picture of the ground: the sea (coarse), the rivers' arcs, the hops' tracks, crossings */
function groundPng(city, S, real) {
  const g = S.g;
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (let i = 0; i < g.key.length; i++) (x0 = Math.min(x0, g.x[i])), (x1 = Math.max(x1, g.x[i])), (y0 = Math.min(y0, g.y[i])), (y1 = Math.max(y1, g.y[i]));
  const pad = 5000;
  (x0 -= pad), (y0 -= pad), (x1 += pad), (y1 += pad);
  const W = 2400, mpp = (x1 - x0) / W, H = Math.ceil((y1 - y0) / mpp);
  const img = new Uint8Array(W * H * 3).fill(250);
  const put = (px, py, c) => { if (px >= 0 && py >= 0 && px < W && py < H) img.set(c, (py * W + px) * 3); };
  for (let py = 0; py < H; py++) for (let px = 0; px < W; px++) {
    const s = real.seaAt(x0 + (px + 0.5) * mpp, y1 - (py + 0.5) * mpp);
    if (s === 1) put(px, py, [170, 205, 225]);
    else if (s < 0) put(px, py, [235, 235, 235]);
  }
  const line = (ax, ay, bx, by, c, r = 0) => {
    const steps = Math.max(1, Math.ceil(Math.hypot(bx - ax, by - ay) / mpp));
    for (let q = 0; q <= steps; q++) {
      const px = Math.round((ax + ((bx - ax) * q) / steps - x0) / mpp), py = Math.round((y1 - (ay + ((by - ay) * q) / steps)) / mpp);
      for (let dx = -r; dx <= r; dx++) for (let dy = -r; dy <= r; dy++) put(px + dx, py + dy, c);
    }
  };
  for (let e = 0; e < g.ea.length; e++) for (let q = g.hopAt[e]; q + 1 < g.hopAt[e + 1]; q++) line(g.hop[q * 2], g.hop[q * 2 + 1], g.hop[q * 2 + 2], g.hop[q * 2 + 3], [120, 120, 120]);
  for (const a of real.arcs) for (let k = 0; k + 3 < a.pts.length; k += 2) line(a.pts[k], a.pts[k + 1], a.pts[k + 2], a.pts[k + 3], a.cross.length ? [20, 90, 200] : [120, 170, 220], 1);
  for (const a of real.arcs) for (const c of a.cross) {
    let s = 0;
    for (let k = 0; k + 3 < a.pts.length; k += 2) {
      const l = Math.hypot(a.pts[k + 2] - a.pts[k], a.pts[k + 3] - a.pts[k + 1]);
      if (s + l >= c.s) {
        const f = (c.s - s) / l, px = a.pts[k] + (a.pts[k + 2] - a.pts[k]) * f, py = a.pts[k + 1] + (a.pts[k + 3] - a.pts[k + 1]) * f;
        line(px, py, px, py, [220, 40, 40], 3);
        break;
      }
      s += l;
    }
  }
  for (const nd of real.nodes) line(nd.x, nd.y, nd.x, nd.y, nd.sea ? [0, 160, 0] : [0, 0, 0], 2);
  writeFileSync(`${OUT}${city}-ground.png`, png(W, H, img));
  console.log(`     ${OUT}${city}-ground.png`);
}

function png(w, h, rgb) {
  const raw = Buffer.alloc((w * 3 + 1) * h);
  for (let y = 0; y < h; y++) { raw[y * (w * 3 + 1)] = 0; Buffer.from(rgb.buffer, y * w * 3, w * 3).copy(raw, y * (w * 3 + 1) + 1); }
  const crcT = Array.from({ length: 256 }, (_, n) => { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; return c >>> 0; });
  const crc = (b) => { let c = 0xffffffff; for (const v of b) c = crcT[(c ^ v) & 255] ^ (c >>> 8); return (c ^ 0xffffffff) >>> 0; };
  const chunk = (type, data) => { const len = Buffer.alloc(4); len.writeUInt32BE(data.length); const td = Buffer.concat([Buffer.from(type), data]); const c = Buffer.alloc(4); c.writeUInt32BE(crc(td)); return Buffer.concat([len, td, c]); };
  const ihdr = Buffer.alloc(13); ihdr.writeUInt32BE(w, 0); ihdr.writeUInt32BE(h, 4); ihdr[8] = 8; ihdr[9] = 2;
  return Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), chunk('IHDR', ihdr), chunk('IDAT', deflateSync(raw)), chunk('IEND', Buffer.alloc(0))]);
}
