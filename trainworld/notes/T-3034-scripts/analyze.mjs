// T-3034: measure unjoined road ends in zoom-14 OFM tiles after joinRoads.
// usage: node analyze.mjs   (writes cases.json, summary.json; caches tiles in ./tiles)
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
const here = path.dirname(fileURLToPath(import.meta.url));
const SRC = "C:/Users/anita/projects/maps/trainworld/app/src/map/";
const { decodeRoads } = await import("file:///" + SRC + "roadTiles.ts");
const { joinRoads, gxOf, gyOf, metresPerUnit, EXTENT, MOTORWAY, TRUNK, LINK } = await import("file:///" + SRC + "roadGraph.ts");

export const CITIES = {
  NewYork: [-73.99, 40.74], Pittsburgh: [-79.995, 40.44], Paris: [2.35, 48.86], Prague: [14.42, 50.08],
  Tokyo: [139.70, 35.68], HoChiMinh: [106.70, 10.78], MexicoCity: [-99.13, 19.43], Lagos: [3.38, 6.52],
  Mumbai: [72.83, 19.07], SaoPaulo: [-46.64, -23.55], Melbourne: [144.96, -37.81], Istanbul: [28.98, 41.01],
};
const TILEURL = "https://tiles.openfreemap.org/planet/20261101_080013_pt/14/{x}/{y}.pbf";
fs.mkdirSync(path.join(here, "tiles"), { recursive: true });

async function tileBuf(tx, ty) {
  const f = path.join(here, "tiles", `${tx}_${ty}.pbf`);
  if (fs.existsSync(f)) return fs.readFileSync(f);
  const r = await fetch(TILEURL.replace("{x}", tx).replace("{y}", ty), { headers: { "User-Agent": "trainworld-roadgaps-measure/1.0" } });
  if (r.status !== 200) { fs.writeFileSync(f, Buffer.alloc(0)); return Buffer.alloc(0); }
  const b = Buffer.from(await r.arrayBuffer());
  fs.writeFileSync(f, b);
  return b;
}

// the game's keep
const busRoad = (cls, brunnel, service, access) =>
  cls === "trunk" ? TRUNK : ["primary", "secondary", "tertiary", "minor", "busway"].includes(cls) ? 1 : cls === "motorway" ? MOTORWAY
    : cls === "service" && !service && !access && !brunnel ? LINK : 0;
// everything else, to see what is left out; distinct weights per kind
const leftOut = (cls, brunnel, service, access) => {
  if (cls === "service") return access ? 0.25 : service === "alley" ? 0.21 : service === "driveway" ? 0.22 : service === "parking_aisle" ? 0.23 : service ? 0.24 : brunnel ? 0.26 : 0;
  if (cls === "pedestrian") return 0.31;
  if (cls === "path") return 0.32;
  if (cls === "track") return 0.33;
  return 0;
};
const LEFT_NAME = { 0.21: "service=alley", 0.22: "service=driveway", 0.23: "service=parking_aisle", 0.24: "service=other", 0.25: "service access-restricted", 0.26: "service on bridge/tunnel", 0.31: "pedestrian", 0.32: "path", 0.33: "track" };
const BANDS = [[0.0, 1, "<1"], [1, 2, "1-2"], [2, 3, "2-3"], [3, 5, "3-5"], [5, 8, "5-8"], [8, 12, "8-12"]];
const MAXM = 12;

function analyse(tx, ty, city) {
  const buf = tileBuf.cache.get(`${tx}_${ty}`);
  const out = { edgeEnds: [], cases: [], ends: 0, joinedEnds: 0, free: 0, links: null, bridging: [] };
  if (!buf.length) return out;
  const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  const raw = decodeRoads(ab, busRoad);
  const mpu = metresPerUnit((ty + 0.5) * EXTENT);
  const rl = joinRoads(raw, mpu);
  const left = decodeRoads(ab, leftOut);
  const lats = (ty + 0.5);
  // segments (graph)
  const cell = 128, cells = new Map();
  const key = (cx, cy) => (cx + 8) * 1024 + cy + 8;
  const segs = []; // [li, i]
  rl.lines.forEach((l, li) => {
    for (let i = 0; i + 3 < l.length; i += 2) {
      const s = segs.length; segs.push([li, i]);
      const cx0 = Math.floor((Math.min(l[i], l[i + 2]) - 24) / cell), cx1 = Math.floor((Math.max(l[i], l[i + 2]) + 24) / cell);
      const cy0 = Math.floor((Math.min(l[i + 1], l[i + 3]) - 24) / cell), cy1 = Math.floor((Math.max(l[i + 1], l[i + 3]) + 24) / cell);
      for (let cx = cx0; cx <= cx1; cx++) for (let cy = cy0; cy <= cy1; cy++) { const k = key(cx, cy); (cells.get(k) ?? cells.set(k, []).get(k)).push(s); }
    }
  });
  const vert = new Map(); // "x,y" -> Set of line ids
  rl.lines.forEach((l, li) => { for (let i = 0; i < l.length; i += 2) { const k = l[i] + "," + l[i + 1]; (vert.get(k) ?? vert.set(k, new Set()).get(k)).add(li); } });
  const w = (li) => rl.weights[li], lev = (li) => rl.levels[li];
  const kindOf = (li) => (w(li) >= MOTORWAY ? "motorway" : w(li) === LINK ? "link" : w(li) === TRUNK ? "trunk" : "street");
  const lengthM = (l) => { let u = 0; for (let i = 0; i + 3 < l.length; i += 2) u += Math.hypot(l[i + 2] - l[i], l[i + 3] - l[i + 1]); return u * mpu; };
  // left-out segments
  const lcells = new Map(); const lsegs = [];
  const lscale = EXTENT / left.extent;
  left.lines.forEach((l, li) => {
    for (let i = 0; i + 3 < l.length; i += 2) {
      const s = lsegs.length; lsegs.push([li, i]);
      const cx0 = Math.floor(Math.min(l[i], l[i + 2]) * lscale / cell - 0.1), cx1 = Math.floor(Math.max(l[i], l[i + 2]) * lscale / cell + 0.1);
      const cy0 = Math.floor(Math.min(l[i + 1], l[i + 3]) * lscale / cell - 0.1), cy1 = Math.floor(Math.max(l[i + 1], l[i + 3]) * lscale / cell + 0.1);
      for (let cx = cx0 - 1; cx <= cx1 + 1; cx++) for (let cy = cy0 - 1; cy <= cy1 + 1; cy++) { const k = key(cx, cy); (lcells.get(k) ?? lcells.set(k, []).get(k)).push(s); }
    }
  });
  const near = (x, y, L, segList, cellMap, sc, maxU, filter) => {
    const res = []; const seen = new Set();
    for (const s of cellMap.get(key(Math.floor(x / cell), Math.floor(y / cell))) ?? []) {
      if (seen.has(s)) continue; seen.add(s);
      const [lj, j] = segList[s]; if (!filter(lj)) continue;
      const m = L[lj]; const ax = m[j] * sc, ay = m[j + 1] * sc, dx = m[j + 2] * sc - ax, dy = m[j + 3] * sc - ay;
      const l2 = dx * dx + dy * dy; const t = l2 ? Math.max(0, Math.min(1, ((x - ax) * dx + (y - ay) * dy) / l2)) : 0;
      const px = ax + t * dx, py = ay + t * dy, d = Math.hypot(px - x, py - y);
      if (d <= maxU) res.push({ lj, j, t, px, py, d, atVertex: t === 0 || t === 1, sx: dx, sy: dy });
    }
    return res.sort((a, b) => a.d - b.d);
  };
  const maxU = MAXM / mpu;
  rl.lines.forEach((l, li) => {
    for (const endIdx of [0, l.length - 2]) {
      const x = l[endIdx], y = l[endIdx + 1];
      if (x <= 0 || y <= 0 || x >= EXTENT || y >= EXTENT) { out.edgeEnds.push({ x, y, w: w(li), lev: lev(li) }); continue; }
      out.ends++;
      const k = x + "," + y;
      const sh = vert.get(k);
      const other0 = l[0] === l[l.length - 2] && l[1] === l[l.length - 1];
      if ((sh && sh.size > 1) || other0) { out.joinedEnds++; continue; }
      const cands = near(x, y, rl.lines, segs, cells, 1, maxU, (lj) => lj !== li);
      let cat = null, cand = null;
      if (cands.length) {
        cand = cands[0];
        const sameLevel = lev(cand.lj) === lev(li);
        const kc = kindOf(cand.lj), ke = kindOf(li);
        if (kc === "motorway" && ke !== "motorway") cat = "end of street near motorway";
        else if (!sameLevel) cat = lev(li) === 0 ? "ground end near bridge/tunnel piece" : "bridge/tunnel end, other level";
        else cat = ke === "motorway" ? "motorway end near road" : (kc === "link" ? "street end near link" : "street end near street");
      } else {
        const lc = near(x, y, left.lines, lsegs, lcells, lscale, maxU, () => true);
        if (lc.length) { cand = lc[0]; cat = "end near left-out: " + LEFT_NAME[left.weights[lc[0].lj]]; cand.left = true; }
        else { out.free++; continue; }
      }
      const dm = cand.d * mpu;
      const band = BANDS.find(([a, b]) => dm >= a && dm < b);
      if (!band) continue;
      // heading of the end: from a vertex ~>=3 m back
      const dirIdx = endIdx === 0 ? 1 : -1;
      let hx = 0, hy = 0;
      for (let q = 1; q < l.length / 2; q++) {
        const vi = endIdx + dirIdx * 2 * q * (endIdx === 0 ? 1 : 1) * (endIdx === 0 ? 1 : 1);
        const bi = endIdx === 0 ? 2 * q : l.length - 2 - 2 * q;
        if (bi < 0 || bi >= l.length) break;
        hx = x - l[bi]; hy = y - l[bi + 1];
        if (Math.hypot(hx, hy) * mpu >= 3) break;
      }
      const hl = Math.hypot(hx, hy) || 1; hx /= hl; hy /= hl;
      const vx = cand.px - x, vy = cand.py - y, vl = Math.hypot(vx, vy) || 1;
      const cos = (hx * vx + hy * vy) / vl;
      const segLenM = Math.hypot(endIdx === 0 ? l[2] - l[0] : l[l.length - 2] - l[l.length - 4], endIdx === 0 ? l[3] - l[1] : l[l.length - 1] - l[l.length - 3]) * mpu;
      // spur: the line meets another road's vertex within 15 m along it of this end
      let spurM = null, along = 0;
      for (let q = 1; q < l.length / 2; q++) {
        const bi = endIdx === 0 ? 2 * q : l.length - 2 - 2 * q, pi = endIdx === 0 ? 2 * q - 2 : l.length - 2 - 2 * (q - 1);
        along += Math.hypot(l[bi] - l[pi], l[bi + 1] - l[pi + 1]) * mpu;
        if (along > 15) break;
        const sv = vert.get(l[bi] + "," + l[bi + 1]);
        if (sv && sv.size > 1) { spurM = +along.toFixed(1); break; }
      }
      const lnear = cand.left ? cand : near(x, y, left.lines, lsegs, lcells, lscale, maxU, () => true)[0];
      const kcls = cand.left ? "left" : kindOf(cand.lj);
      // lng/lat of end
      const gx = tx * EXTENT + x, gy = ty * EXTENT + y;
      const lng = (gx / (EXTENT * 2 ** 14)) * 360 - 180;
      const lat = (360 / Math.PI) * Math.atan(Math.exp(Math.PI * (1 - (2 * gy) / (EXTENT * 2 ** 14)))) - 90;
      const clsDir = cand.left ? null : [cand.sx, cand.sy];
      let par = null;
      if (clsDir) { const sl = Math.hypot(clsDir[0], clsDir[1]) || 1; par = Math.abs((hx * clsDir[0] + hy * clsDir[1]) / sl); }
      out.cases.push({ city, tx, ty, li, endIdx, x, y, lng, lat, d: +dm.toFixed(2), band: band[2], cat, cos: +cos.toFixed(3), par: par === null ? null : +par.toFixed(3), segLenM: +segLenM.toFixed(1), candKind: kcls, atVertex: cand.atVertex, candEnd: !cand.left && ((cand.t === 0 && cand.j === 0) || (cand.t === 1 && cand.j + 2 === rl.lines[cand.lj].length - 2)), selfKind: kindOf(li), lineLenM: +lengthM(l).toFixed(0), spurM, selfD: +(Math.hypot(l[0] - l[l.length - 2], l[1] - l[l.length - 1]) * mpu).toFixed(1), leftD: lnear ? +(lnear.d * mpu).toFixed(1) : null, mpu: +mpu.toFixed(4) });
    }
  });
  // short left-out/service pieces bridging two different graph streets (both ends within 2.5 m of a ground street)
  left.lines.forEach((l, li) => {
    const wgt = left.weights[li]; if (wgt >= 0.31) return; // services only
    const len = (() => { let u = 0; for (let i = 0; i + 3 < l.length; i += 2) u += Math.hypot(l[i + 2] - l[i], l[i + 3] - l[i + 1]); return u * lscale * mpu; })();
    if (len > 60) return;
    const ends = [[l[0], l[1]], [l[l.length - 2], l[l.length - 1]]].map(([a, b]) => [a * lscale, b * lscale]);
    if (ends.some(([a, b]) => a <= 0 || b <= 0 || a >= EXTENT || b >= EXTENT)) return;
    const hit = ends.map(([a, b]) => near(a, b, rl.lines, segs, cells, 1, 2.5 / mpu, (lj) => lev(lj) === 0 && kindOf(lj) !== "motorway" && kindOf(lj) !== "link")[0]);
    if (hit[0] && hit[1] && hit[0].lj !== hit[1].lj) out.bridging.push({ type: LEFT_NAME[wgt], lenM: +len.toFixed(0) });
  });
  // links kept vs dropped by the game (raw weight 0.5 lines <=60 m)
  const rawLinks = raw.lines.map((l, i) => [l, raw.weights[i]]).filter(([, wt]) => wt === LINK);
  out.links = { raw: rawLinks.length, kept: rl.lines.filter((_, i) => rl.weights[i] === LINK).length };
  out.linkEnds = [];
  for (const [l] of rawLinks) {
    const len = lengthM(l); if (len > 60) continue;
    const ds = [[l[0], l[1]], [l[l.length - 2], l[l.length - 1]]].map(([a, b]) => {
      if (a <= 0 || b <= 0 || a >= EXTENT || b >= EXTENT) return 0;
      const c = near(a, b, rl.lines, segs, cells, 1, 12 / mpu, (lj) => lev(lj) === 0 && (kindOf(lj) === "street" || kindOf(lj) === "trunk"));
      return c.length ? c[0].d * mpu : 99;
    });
    out.linkEnds.push({ len: +len.toFixed(0), worst: +Math.max(...ds).toFixed(1), best: +Math.min(...ds).toFixed(1) });
  }
  return out;
}

async function main() {
  tileBuf.cache = new Map();
  const all = { cases: [], perCity: {} };
  for (const [city, [lng, lat]] of Object.entries(CITIES)) {
    const cx = Math.floor(gxOf(lng) / EXTENT), cy = Math.floor(gyOf(lat) / EXTENT);
    const pc = { tiles: 0, ends: 0, joinedEnds: 0, free: 0, bridging: [], links: { raw: 0, kept: 0 }, centre: [cx, cy] };
    for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) {
      const tx = cx + dx, ty = cy + dy;
      tileBuf.cache.set(`${tx}_${ty}`, await tileBuf(tx, ty));
      const r = analyse(tx, ty, city);
      pc.tiles++; pc.ends += r.ends; pc.joinedEnds += r.joinedEnds; pc.free += r.free;
      pc.bridging.push(...r.bridging); (pc.linkEnds ??= []).push(...r.linkEnds);
      if (r.links) { pc.links.raw += r.links.raw; pc.links.kept += r.links.kept; }
      all.cases.push(...r.cases);
      tileBuf.edges ??= new Map(); tileBuf.edges.set(`${tx}_${ty}`, r.edgeEnds);
      tileBuf.cache.delete(`${tx}_${ty}`);
    }
    // seams: east edge of (tx,ty) vs west edge of (tx+1,ty), and south vs north
    const sm = { pairs: 0, ends: 0, matched: 0, gaps: [] };
    for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) {
      const A = tileBuf.edges.get(`${cx + dx}_${cy + dy}`);
      for (const [ox, oy, f, g, along] of [[1, 0, (e) => e.x === EXTENT, (e) => e.x === 0, "y"], [0, 1, (e) => e.y === EXTENT, (e) => e.y === 0, "x"]]) {
        const B = tileBuf.edges.get(`${cx + dx + ox}_${cy + dy + oy}`); if (!A || !B) continue;
        const ea = A.filter(f), eb = B.filter(g);
        const mpu = metresPerUnit((cy + dy + 0.5) * EXTENT);
        for (const [P, Q] of [[ea, eb], [eb, ea]]) for (const e of P) {
          sm.ends++;
          let best = Infinity; for (const q of Q) if (q.lev === e.lev) best = Math.min(best, Math.abs(q[along] - e[along]));
          if (best <= 3) sm.matched++; else sm.gaps.push(best === Infinity ? null : +(best * mpu).toFixed(1));
        }
      }
    }
    pc.seam = { ends: sm.ends, matched: sm.matched, noCounterpart: sm.gaps.filter((g) => g === null).length, gapM: sm.gaps.filter((g) => g !== null) };
    all.perCity[city] = pc;
    console.log(city, pc.centre.join(","), "ends", pc.ends, "joined", pc.joinedEnds, "free", pc.free, "cases", all.cases.filter((c) => c.city === city).length);
  }
  fs.writeFileSync(path.join(here, "cases.json"), JSON.stringify(all));
}
main();
