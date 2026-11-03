// T-2177 design measurement: ferry routing over the basemap's own water, the way the app could do it
// (a prototype for T-2192, not app code; goes once T-2192 lands).
// TZ_ZOOM=12 node trainworld/notes/T-2177-route.mjs <case> <cell_m> [min half width m, 10] [bridges 1|0]
// Prints sizes and timings as JSON; writes route-<case>-<cell>m-z<zoom>.png and dump-<case>.* to OUT.
import fs from "node:fs";
import zlib from "node:zlib";
import path from "node:path";
const NM = "C:/Users/anita/projects/maps/trainworld/app/node_modules/";
const { VectorTile } = await import("file:///" + NM + "@mapbox/vector-tile/index.js");
const pbfMod = await import("file:///" + NM + "pbf/index.js");
const Pbf = pbfMod.default ?? pbfMod.Pbf ?? pbfMod.PbfReader;

const TPL = "https://tiles.openfreemap.org/planet/20261101_080013_pt/{z}/{x}/{y}.pbf";
const OUT = "C:/Users/anita/projects/maps/artifacts/trainworld/t2177";
const CACHE = path.join(OUT, "tiles");
fs.mkdirSync(CACHE, { recursive: true });
fs.mkdirSync(OUT, { recursive: true });

// ports [lon, lat]: a point on land near the shore, as a player would click
const CASES = {
  sumida: [[139.7972, 35.7108], [139.7598, 35.6515]],        // Asakusa (Azumabashi) -> Hinode
  venice: [[12.3205, 45.4392], [12.3400, 45.4334]],          // Piazzale Roma -> San Marco (Grand Canal)
  nyc: [[-74.0130, 40.7012], [-74.0715, 40.6437]],           // Whitehall -> St George
  sydney: [[151.2108, -33.8612], [151.2846, -33.8004]],      // Circular Quay -> Manly
  hk: [[114.1612, 22.2878], [114.1688, 22.2938]],            // Central -> Tsim Sha Tsui
  paris: [[2.2915, 48.8605], [2.3500, 48.8535]],             // Eiffel -> Notre-Dame (left bank)
  bosphorus: [[28.9745, 41.0175], [29.0225, 40.9925]],       // Eminonu -> Kadikoy
  amsterdam: [[4.8995, 52.3785], [4.8845, 52.3620]],         // Centraal -> Rijksmuseum by the canals
  bosphorus_long: [[28.9745, 41.0175], [29.0560, 41.1640]],  // Eminonu -> Sariyer (up the strait)
};
const name = process.argv[2] ?? "sumida";
const CELL = Number(process.argv[3] ?? 10);
const CLEAR = 400;     // wanted distance off land, m
const K = 3;           // a metre at the shore costs 1 + K
const HALF = Number(process.argv[4] ?? 10); // a channel at least 2 x HALF wide
const [A, B] = CASES[name];

const R = 6371008.8;
const lon0 = (A[0] + B[0]) / 2, lat0 = (A[1] + B[1]) / 2, kx = Math.cos(lat0 * Math.PI / 180);
const toX = (lon) => R * (lon - lon0) * Math.PI / 180 * kx, toY = (lat) => R * (lat - lat0) * Math.PI / 180;
const straight = Math.hypot(toX(A[0]) - toX(B[0]), toY(A[1]) - toY(B[1]));
const pad = Math.max(1500, 0.35 * straight);
const x0 = Math.min(toX(A[0]), toX(B[0])) - pad, x1 = Math.max(toX(A[0]), toX(B[0])) + pad;
const y0 = Math.min(toY(A[1]), toY(B[1])) - pad, y1 = Math.max(toY(A[1]), toY(B[1])) + pad;
const W = Math.ceil((x1 - x0) / CELL), H = Math.ceil((y1 - y0) / CELL);
const lonOf = (x) => lon0 + x / (R * kx) * 180 / Math.PI, latOf = (y) => lat0 + y / R * 180 / Math.PI;

// ---- tiles
const Z = Number(process.env.TZ_ZOOM ?? 14), N = 2 ** Z;
const tx = (lon) => Math.floor((lon + 180) / 360 * N);
const ty = (lat) => { const s = Math.sin(lat * Math.PI / 180); return Math.floor((0.5 - Math.log((1 + s) / (1 - s)) / (4 * Math.PI)) * N); };
const txa = tx(lonOf(x0)), txb = tx(lonOf(x1)), tya = ty(latOf(y1)), tyb = ty(latOf(y0));
let t0 = performance.now(), bytes = 0, ntiles = 0;
const grid = new Uint8Array(W * H); // 1 water
const bridge = new Uint8Array(W * H);
const BRIDGES = (process.argv[5] ?? "1") === "1";
const jobs = [];
for (let x = txa; x <= txb; x++) for (let y = tya; y <= tyb; y++) jobs.push([x, y]);
async function getTile(x, y) {
  const f = path.join(CACHE, `${Z}_${x}_${y}.pbf`);
  if (fs.existsSync(f)) return fs.readFileSync(f);
  const r = await fetch(TPL.replace("{z}", Z).replace("{x}", x).replace("{y}", y), { headers: { "User-Agent": "trainworld-design-check" } });
  if (!r.ok) throw new Error(`tile ${x},${y}: ${r.status}`);
  const b = Buffer.from(await r.arrayBuffer());
  fs.writeFileSync(f, b);
  return b;
}
const bufs = [];
for (let i = 0; i < jobs.length; i += 8) bufs.push(...await Promise.all(jobs.slice(i, i + 8).map(([x, y]) => getTile(x, y).then((b) => [x, y, b]))));
const tFetch = performance.now() - t0;
t0 = performance.now();
let waterBytes = 0;
for (const [x, y, b] of bufs) {
  bytes += b.length; ntiles++;
  const vt = new VectorTile(new Pbf(b));
  const T = vt.layers.transportation;
  if (T && BRIDGES) for (let i = 0; i < T.length; i++) {
    const f = T.feature(i);
    if (f.properties.brunnel !== "bridge" || f.type !== 2) continue;
    const ext = T.extent;
    for (const line of f.loadGeometry()) {
      const pts = line.map((pt) => {
        const lon = (x + pt.x / ext) / N * 360 - 180;
        const n = Math.PI - 2 * Math.PI * (y + pt.y / ext) / N;
        const lat = 180 / Math.PI * Math.atan(0.5 * (Math.exp(n) - Math.exp(-n)));
        return [(toX(lon) - x0) / CELL, (toY(lat) - y0) / CELL];
      });
      const hw = Math.max(1, 8 / CELL);
      for (let k = 1; k < pts.length; k++) {
        const [xa, ya] = pts[k - 1], [xb, yb] = pts[k], n = Math.ceil(Math.hypot(xb - xa, yb - ya) * 2) + 1;
        for (let s = 0; s <= n; s++) {
          const px = xa + (xb - xa) * s / n, py = ya + (yb - ya) * s / n;
          for (let dy = -Math.ceil(hw); dy <= Math.ceil(hw); dy++) for (let dx = -Math.ceil(hw); dx <= Math.ceil(hw); dx++) {
            if (dx * dx + dy * dy > hw * hw) continue;
            const c = Math.floor(px + dx), r = Math.floor(py + dy);
            if (c >= 0 && r >= 0 && c < W && r < H) bridge[r * W + c] = 1;
          }
        }
      }
    }
  }
  const L = vt.layers.water;
  if (!L) continue;
  for (let i = 0; i < L.length; i++) {
    const f = L.feature(i);
    const p = f.properties;
    if (p.class === "swimming_pool" || p.brunnel === "tunnel" || p.intermittent === 1) continue;
    const ext = L.extent;
    const rings = f.loadGeometry().map((ring) => ring.map((pt) => {
      const lon = (x + pt.x / ext) / N * 360 - 180;
      const n = Math.PI - 2 * Math.PI * (y + pt.y / ext) / N;
      const lat = 180 / Math.PI * Math.atan(0.5 * (Math.exp(n) - Math.exp(-n)));
      return [(toX(lon) - x0) / CELL, (toY(lat) - y0) / CELL];
    }));
    fill(rings);
  }
}
function fill(rings) { // even-odd scanline at pixel centres
  let ymin = Infinity, ymax = -Infinity;
  for (const r of rings) for (const [, y] of r) { ymin = Math.min(ymin, y); ymax = Math.max(ymax, y); }
  const r0 = Math.max(0, Math.ceil(ymin - 0.5)), r1 = Math.min(H - 1, Math.floor(ymax - 0.5));
  const xs = [];
  for (let row = r0; row <= r1; row++) {
    const yc = row + 0.5; xs.length = 0;
    for (const r of rings) for (let i = 0, j = r.length - 1; i < r.length; j = i++) {
      const [xa, ya] = r[j], [xb, yb] = r[i];
      if ((ya <= yc) !== (yb <= yc)) xs.push(xa + (yc - ya) / (yb - ya) * (xb - xa));
    }
    xs.sort((a, b) => a - b);
    for (let k = 0; k + 1 < xs.length; k += 2) {
      const c0 = Math.max(0, Math.ceil(xs[k] - 0.5)), c1 = Math.min(W - 1, Math.floor(xs[k + 1] - 0.5));
      for (let c = c0; c <= c1; c++) grid[row * W + c] = 1;
    }
  }
}
const tRaster = performance.now() - t0;

// ---- distance to land (exact EDT, Felzenszwalb), metres
t0 = performance.now();
const INF = 1e20;
const dist = new Float64Array(W * H);
function edt1(f, n) {
  const d = new Float64Array(n), v = new Int32Array(n), z = new Float64Array(n + 1);
  let k = 0; v[0] = 0; z[0] = -INF; z[1] = INF;
  for (let q = 1; q < n; q++) {
    let s;
    while (true) { s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]); if (s <= z[k]) { k--; if (k < 0) { k = 0; break; } } else break; }
    k++; v[k] = q; z[k] = s; z[k + 1] = INF;
  }
  k = 0;
  for (let q = 0; q < n; q++) { while (z[k + 1] < q) k++; d[q] = (q - v[k]) ** 2 + f[v[k]]; }
  return d;
}
function edtTo(isFeature, out) { // metres from each cell to the nearest feature cell
  for (let i = 0; i < W * H; i++) out[i] = isFeature(i) ? 0 : INF;
  const col = new Float64Array(H);
  for (let c = 0; c < W; c++) { for (let r = 0; r < H; r++) col[r] = out[r * W + c]; const d = edt1(col, H); for (let r = 0; r < H; r++) out[r * W + c] = d[r]; }
  const row = new Float64Array(W);
  for (let r = 0; r < H; r++) { for (let c = 0; c < W; c++) row[c] = out[r * W + c]; const d = edt1(row, W); for (let c = 0; c < W; c++) out[r * W + c] = Math.sqrt(d[c]) * CELL; }
  return out;
}
// water polygons are often cut at bridges (Amsterdam, Venice): a bridge's cells count as water where a
// closing of the water by BRIDGE_R would fill them
let bridged = 0;
if (BRIDGES) {
  const BR = 25;
  const dW = edtTo((i) => grid[i] === 1, new Float64Array(W * H));
  const dE = edtTo((i) => dW[i] > BR, new Float64Array(W * H));
  for (let i = 0; i < W * H; i++) if (!grid[i] && bridge[i] && dE[i] > BR) { grid[i] = 1; bridged++; }
}
edtTo((i) => grid[i] === 0, dist);
const tEdt = performance.now() - t0;

// ---- ports: nearest water cell with at least 1 cell clearance to the clicked point, within 300 m
function snap([lon, lat]) {
  const cx = (toX(lon) - x0) / CELL, cy = (toY(lat) - y0) / CELL, rad = Math.ceil(300 / CELL);
  let best = -1, bd = Infinity;
  for (let dy = -rad; dy <= rad; dy++) for (let dx = -rad; dx <= rad; dx++) {
    const c = Math.floor(cx) + dx, r = Math.floor(cy) + dy;
    if (c < 0 || r < 0 || c >= W || r >= H) continue;
    const i = r * W + c; if (!grid[i] || dist[i] < Math.max(1.5 * CELL, HALF)) continue;
    const d = dx * dx + dy * dy; if (d < bd) { bd = d; best = i; }
  }
  return best;
}
const sa = snap(A), sb = snap(B);
if (sa < 0 || sb < 0) { console.log(name, "no water within 300 m of a port", sa, sb); process.exit(1); }

// ---- A* on 16 neighbours; a step's cost is its length times 1 + K * (1 - d/CLEAR)^2 below CLEAR,
// the penalty fading out within 300 m of either port so a boat may leave its pier.
t0 = performance.now();
const pc = (i) => [i % W, Math.floor(i / W)];
const [ax, ay] = pc(sa), [bx, by] = pc(sb);
let KK = K;
function weight(i) {
  const d = dist[i]; if (d < Math.max(CELL, HALF)) return -1;
  const [c, r] = pc(i);
  const near = Math.min(Math.hypot(c - ax, r - ay), Math.hypot(c - bx, r - by)) * CELL;
  const fade = Math.min(1, near / 300);
  const u = Math.max(0, 1 - d / CLEAR);
  return 1 + KK * fade * u * u;
}
function search(kv) {
KK = kv;
const NB = [[1,0],[-1,0],[0,1],[0,-1],[1,1],[1,-1],[-1,1],[-1,-1],[2,1],[2,-1],[-2,1],[-2,-1],[1,2],[1,-2],[-1,2],[-1,-2]];
const g = new Float64Array(W * H).fill(Infinity), from = new Int32Array(W * H).fill(-1), closed = new Uint8Array(W * H);
const heap = []; // [f, i]
const push = (f, i) => { heap.push([f, i]); let k = heap.length - 1; while (k > 0) { const p = (k - 1) >> 1; if (heap[p][0] <= heap[k][0]) break; [heap[p], heap[k]] = [heap[k], heap[p]]; k = p; } };
const pop = () => { const top = heap[0], last = heap.pop(); if (heap.length) { heap[0] = last; let k = 0; for (;;) { const l = 2 * k + 1, r = l + 1; let m = k; if (l < heap.length && heap[l][0] < heap[m][0]) m = l; if (r < heap.length && heap[r][0] < heap[m][0]) m = r; if (m === k) break; [heap[m], heap[k]] = [heap[k], heap[m]]; k = m; } } return top; };
const hx = (i) => { const [c, r] = pc(i); return Math.hypot(c - bx, r - by); };
g[sa] = 0; push(hx(sa), sa);
let expanded = 0;
while (heap.length) {
  const [, i] = pop();
  if (closed[i]) continue; closed[i] = 1; expanded++;
  if (i === sb) break;
  const [c, r] = pc(i); const wi = weight(i);
  for (const [dx, dy] of NB) {
    const nc = c + dx, nr = r + dy; if (nc < 0 || nr < 0 || nc >= W || nr >= H) continue;
    const j = nr * W + nc; if (closed[j]) continue;
    const wj = weight(j); if (wj < 0) continue;
    if (Math.abs(dx) + Math.abs(dy) === 3) { // knight move: both cells it crosses must be water
      const m1 = (r + Math.trunc(dy / 2)) * W + (c + Math.trunc(dx / 2)), m2 = (r + Math.round(dy / 2)) * W + (c + Math.round(dx / 2));
      if (weight(m1) < 0 || weight(m2) < 0) continue;
    }
    const ng = g[i] + Math.hypot(dx, dy) * (wi + wj) / 2;
    if (ng < g[j]) { g[j] = ng; from[j] = i; push(ng + hx(j), j); }
  }
}
if (from[sb] < 0) return null;
let raw = []; for (let i = sb; i >= 0; i = from[i]) { const [c, r] = pc(i); raw.push([c + 0.5, r + 0.5]); }
raw.reverse();
let L = 0; for (let k = 1; k < raw.length; k++) L += Math.hypot(raw[k][0] - raw[k - 1][0], raw[k][1] - raw[k - 1][1]) * CELL;
return { raw, L, expanded };
}
// the plain shortest water path first; then the penalty, eased off while its route is over DETOUR times as long
const DETOUR = 1.3;
const plain = search(0);
if (!plain) { console.log(name, "no water path"); process.exit(1); }
let best = plain, kUsed = 0, searches = 1, expanded = plain.expanded;
for (const kv of [K, K / 3, K / 10]) {
  const s = search(kv); searches++; expanded += s.expanded;
  if (s.L <= DETOUR * plain.L) { best = s; kUsed = kv; break; }
}
const tSearch = performance.now() - t0;
const raw = best.raw;

// ---- smooth: Douglas-Peucker at min(25 m, a quarter of the clearance), then a fillet arc at every
// corner, its radius at most RMAX and what the two legs and the water allow.
t0 = performance.now();
const RMAX = 250;
const isWater = (p) => { const c = Math.floor(p[0]), r = Math.floor(p[1]); return c >= 0 && r >= 0 && c < W && r < H && dist[r * W + c] >= CELL * 0.75; };
function dp(pts, tol) {
  const keep = new Uint8Array(pts.length); keep[0] = keep[pts.length - 1] = 1;
  const st = [[0, pts.length - 1]];
  while (st.length) {
    const [a, b] = st.pop(); let m = -1, md = 0;
    const [ax_, ay_] = pts[a], [bx_, by_] = pts[b], L = Math.hypot(bx_ - ax_, by_ - ay_) || 1e-9;
    for (let k = a + 1; k < b; k++) { const d = Math.abs((bx_ - ax_) * (ay_ - pts[k][1]) - (ax_ - pts[k][0]) * (by_ - ay_)) / L; if (d > md) { md = d; m = k; } }
    const c = Math.floor((pts[m]?.[0]) ?? 0), r = Math.floor((pts[m]?.[1]) ?? 0);
    const t = m >= 0 ? Math.min(25, 0.25 * dist[r * W + c]) / CELL : 0;
    if (m >= 0 && md > Math.max(t, 0.5)) { keep[m] = 1; st.push([a, m], [m, b]); }
  }
  return pts.filter((_, k) => keep[k]);
}
let poly = dp(raw, 25 / CELL);
const out = [poly[0]];
let minR = Infinity;
for (let k = 1; k + 1 < poly.length; k++) {
  const P = poly[k - 1], Q = poly[k], S = poly[k + 1];
  const u = [P[0] - Q[0], P[1] - Q[1]], v = [S[0] - Q[0], S[1] - Q[1]];
  const lu = Math.hypot(...u), lv = Math.hypot(...v);
  const cosA = (u[0] * v[0] + u[1] * v[1]) / (lu * lv), ang = Math.acos(Math.max(-1, Math.min(1, cosA)));
  const t = Math.tan(ang / 2);
  let r = Math.min(RMAX / CELL, 0.5 * Math.min(lu, lv) * t);
  let arc = null;
  for (let tries = 0; tries < 6 && r * CELL > 5; tries++, r *= 0.6) {
    const dd = r / t; // tangent distance from Q
    const T1 = [Q[0] + u[0] / lu * dd, Q[1] + u[1] / lu * dd], T2 = [Q[0] + v[0] / lv * dd, Q[1] + v[1] / lv * dd];
    const bis = [u[0] / lu + v[0] / lv, u[1] / lu + v[1] / lv], lb = Math.hypot(...bis);
    const C = [Q[0] + bis[0] / lb * r / Math.sin(ang / 2), Q[1] + bis[1] / lb * r / Math.sin(ang / 2)];
    const a1 = Math.atan2(T1[1] - C[1], T1[0] - C[0]); let a2 = Math.atan2(T2[1] - C[1], T2[0] - C[0]);
    let da = a2 - a1; while (da > Math.PI) da -= 2 * Math.PI; while (da < -Math.PI) da += 2 * Math.PI;
    const n = Math.max(2, Math.ceil(Math.abs(da) * r * CELL / 20));
    const pts = []; for (let s = 0; s <= n; s++) { const a = a1 + da * s / n; pts.push([C[0] + r * Math.cos(a), C[1] + r * Math.sin(a)]); }
    if (pts.every(isWater)) { arc = pts; break; }
  }
  if (arc) { out.push(...arc); minR = Math.min(minR, r * CELL); } else out.push(Q);
}
out.push(poly[poly.length - 1]);
const tSmooth = performance.now() - t0;
let len = 0; for (let k = 1; k < out.length; k++) len += Math.hypot(out[k][0] - out[k - 1][0], out[k][1] - out[k - 1][1]) * CELL;
let clearSum = 0, clearMin = Infinity, ns = 0;
for (let k = 1; k < out.length; k++) { const n = Math.ceil(Math.hypot(out[k][0] - out[k - 1][0], out[k][1] - out[k - 1][1])); for (let s = 0; s < n; s++) { const p = [out[k - 1][0] + (out[k][0] - out[k - 1][0]) * s / n, out[k - 1][1] + (out[k][1] - out[k - 1][1]) * s / n]; const d = dist[Math.floor(p[1]) * W + Math.floor(p[0])] ?? 0; clearSum += d; clearMin = Math.min(clearMin, d); ns++; } }

console.log(JSON.stringify({ name, cell_m: CELL, grid: [W, H], cells_M: +(W * H / 1e6).toFixed(2), tiles: ntiles, tile_MB: +(bytes / 1e6).toFixed(1),
  fetch_ms: Math.round(tFetch), raster_ms: Math.round(tRaster), edt_ms: Math.round(tEdt), bridged_cells: bridged, search_ms: Math.round(tSearch), searches, k_used: +kUsed.toFixed(2), plain_km: +(plain.L / 1000).toFixed(2), expanded, smooth_ms: Math.round(tSmooth),
  straight_km: +(straight / 1000).toFixed(2), route_km: +(len / 1000).toFixed(2), ratio: +(len / straight).toFixed(2), corners: poly.length - 2, min_radius_m: Math.round(minR), clear_mean_m: Math.round(clearSum / ns), clear_min_m: Math.round(clearMin), points: out.length }));

// ---- PNG, at most 1400 px a side
const sc = Math.max(1, Math.ceil(Math.max(W, H) / 1400)), PW = Math.floor(W / sc), PH = Math.floor(H / sc);
const img = Buffer.alloc(PW * PH * 3);
for (let r = 0; r < PH; r++) for (let c = 0; c < PW; c++) {
  const i = (H - 1 - r * sc) * W + c * sc, o = (r * PW + c) * 3;
  if (grid[i]) { const t = Math.min(1, dist[i] / CLEAR); img[o] = 150 - 40 * t; img[o + 1] = 200 - 30 * t; img[o + 2] = 215 - 10 * t; }
  else { img[o] = 242; img[o + 1] = 236; img[o + 2] = 222; }
}
{ // plain water/land mask and the route in its pixels, for the mockup
  const m = Buffer.alloc(PW * PH); for (let r = 0; r < PH; r++) for (let c = 0; c < PW; c++) m[r * PW + c] = grid[(H - 1 - r * sc) * W + c * sc] ? 1 : 0;
  const toPx = (p) => [p[0] / sc, PH - 1 - p[1] / sc];
  const portPx = (s) => { const [c, r] = pc(s); return toPx([c + 0.5, r + 0.5]); };
  const landPx = (p) => toPx([(toX(p[0]) - x0) / CELL, (toY(p[1]) - y0) / CELL]);
  fs.writeFileSync(path.join(OUT, `dump-${name}.json`), JSON.stringify({ W: PW, H: PH, m_per_px: CELL * sc, route: out.map(toPx), berth: [portPx(sa), portPx(sb)], land: [landPx(A), landPx(B)], route_km: len / 1000 }));
  fs.writeFileSync(path.join(OUT, `dump-${name}.mask`), m);
}
const dot = (p, rad, col) => { const cx = p[0] / sc, cy = PH - 1 - p[1] / sc; for (let dy = -rad; dy <= rad; dy++) for (let dx = -rad; dx <= rad; dx++) { if (dx * dx + dy * dy > rad * rad) continue; const c = Math.round(cx + dx), r = Math.round(cy + dy); if (c < 0 || r < 0 || c >= PW || r >= PH) continue; const o = (r * PW + c) * 3; img[o] = col[0]; img[o + 1] = col[1]; img[o + 2] = col[2]; } };
for (let k = 1; k < out.length; k++) { const n = Math.ceil(Math.hypot(out[k][0] - out[k - 1][0], out[k][1] - out[k - 1][1]) / sc * 2) + 1; for (let s = 0; s <= n; s++) dot([out[k - 1][0] + (out[k][0] - out[k - 1][0]) * s / n, out[k - 1][1] + (out[k][1] - out[k - 1][1]) * s / n], 1, [214, 60, 80]); }
for (const p of [A, B]) dot([(toX(p[0]) - x0) / CELL, (toY(p[1]) - y0) / CELL], 5, [40, 40, 60]);
for (const s of [sa, sb]) { const [c, r] = pc(s); dot([c + 0.5, r + 0.5], 4, [255, 255, 255]); }
function png(w, h, rgb) {
  const crcT = new Int32Array(256).map((_, n) => { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; return c; });
  const crc = (b) => { let c = -1; for (const x of b) c = crcT[(c ^ x) & 255] ^ (c >>> 8); return (c ^ -1) >>> 0; };
  const chunk = (t, d) => { const len = Buffer.alloc(4); len.writeUInt32BE(d.length); const td = Buffer.concat([Buffer.from(t), d]); const cr = Buffer.alloc(4); cr.writeUInt32BE(crc(td)); return Buffer.concat([len, td, cr]); };
  const raw = Buffer.alloc((w * 3 + 1) * h); for (let r = 0; r < h; r++) { raw[r * (w * 3 + 1)] = 0; rgb.copy(raw, r * (w * 3 + 1) + 1, r * w * 3, (r + 1) * w * 3); }
  const ihdr = Buffer.alloc(13); ihdr.writeUInt32BE(w, 0); ihdr.writeUInt32BE(h, 4); ihdr[8] = 8; ihdr[9] = 2;
  return Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), chunk("IHDR", ihdr), chunk("IDAT", zlib.deflateSync(raw)), chunk("IEND", Buffer.alloc(0))]);
}
fs.writeFileSync(path.join(OUT, `route-${name}-${CELL}m-z${Z}.png`), png(PW, PH, img));
