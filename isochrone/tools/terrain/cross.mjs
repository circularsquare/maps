// Directed move costs and L1 crossings of one tile, the same rules as tools/terrain/walk.py
// (move_costs, crossings), in node for speed. Run by walk.py's crossings_js.
//   node isochrone/tools/terrain/cross.mjs <in file> <out file>
// In: a JSON header line, then cls (P*P uint8), bits (P*P uint8), z (P*P float32 LE), the LC speed
// table (256 float32). Header: {P, halo, north, K, cap, roadKmh, and walk.py's SLOPE_RULE (toblerK,
// toblerG0, heightA, heightB, roadGrade, gMax), jobs: [{grid,
// along, m, maxSweeps, cells: n}]}, then per job its cells as int32 pairs (I, J) after the table.
// Out: per job and direction E, W, S, N, n float32 (hours per km, Infinity where not crossed),
// then per job one int32 of sweeps run (the most of its directions).
import { readFileSync, writeFileSync } from 'node:fs';

const [inPath, outPath] = process.argv.slice(2);
const buf = readFileSync(inPath);
const nl = buf.indexOf(10);
const H = JSON.parse(buf.subarray(0, nl).toString());
const { P, halo, north, K } = H;
const PP = P * P;
let off = nl + 1;
const cls = new Uint8Array(buf.buffer, buf.byteOffset + off, PP); off += PP;
const bits = new Uint8Array(buf.buffer, buf.byteOffset + off, PP); off += PP;
const zb = new Uint8Array(buf.subarray(off, off + 4 * PP)); off += 4 * PP;
const z = new Float32Array(zb.buffer);
const lcb = new Uint8Array(buf.subarray(off, off + 1024)); off += 1024;
const LC = new Float32Array(lcb.buffer);
const cellsAll = [];
for (const j of H.jobs) {
  const b = new Uint8Array(buf.subarray(off, off + 8 * j.cells)); off += 8 * j.cells;
  cellsAll.push(new Int32Array(b.buffer));
}

// the 16 moves (dc, dr) in walk.py's order, knight moves with the two cells they pass between
const MV = [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]];
const BET = [];
for (const sx of [1, -1]) for (const sy of [1, -1]) {
  MV.push([sx, 2 * sy]); BET[MV.length - 1] = [[0, sy], [sx, sy]];
  MV.push([2 * sx, sy]); BET[MV.length - 1] = [[sx, 0], [sx, sy]];
}
const moveIndex = (dc, dr) => MV.findIndex(([a, b]) => a === dc && b === dr);

const KM_DEG = 111.195, N = 1200;
const dx = new Float32Array(P);
for (let r = 0; r < P; r++) dx[r] = KM_DEG / N * Math.cos((north - (r + 0.5) / N) * Math.PI / 180);
const dy = KM_DEG / N;
const nlh = new Float32Array(PP);
for (let i = 0; i < PP; i++) nlh[i] = -Math.log(H.heightA) + H.heightB * Math.max(z[i], 0);
const TK = H.toblerK, G0 = H.toblerG0;

function costs(roads) {
  const s = new Float32Array(PP), graded = new Uint8Array(PP);
  for (let i = 0; i < PP; i++) {
    const c = cls[i], water = c === 80 || c === 200;
    let kmh = LC[c];
    let g = c === 50 ? 1 : 0;
    if (roads) {
      const walk = (bits[i] & 7) && !water, bridge = water && (bits[i] & 16);
      if (walk) { kmh = Math.max(kmh, H.roadKmh); g = 1; }
      if (bridge) { kmh = H.roadKmh; g = 1; }
      if (water && !bridge) kmh = 0;
    } else if (water) kmh = 0;
    s[i] = kmh > 0 ? 1 / Math.max(kmh, 1e-6) : Infinity;
    graded[i] = g;
  }
  const C = new Float32Array(16 * PP).fill(Infinity);
  const lim = H.cap == null ? -Infinity : -Math.log(H.cap) / TK;
  for (let k = 0; k < 16; k++) {
    const [dc, dr] = MV[k];
    const bt = BET[k];
    const r0 = Math.max(0, -dr), r1 = P - Math.max(0, dr), c0 = Math.max(0, -dc), c1 = P - Math.max(0, dc);
    const d = dr * P + dc;
    const e0 = bt ? bt[0][1] * P + bt[0][0] : 0, e1 = bt ? bt[1][1] * P + bt[1][0] : 0;
    for (let r = r0; r < r1; r++) {
      const ddx = dc * 0.5 * (dx[r] + dx[r + dr]);
      const len = Math.fround(Math.sqrt(ddx * ddx + (dr * dy) ** 2));
      for (let c = c0, u = r * P + c0; c < c1; c++, u++) {
        const v = u + d;
        let mean = bt ? (s[u] + s[v] + s[u + e0] + s[u + e1]) * 0.25 : (s[u] + s[v]) * 0.5;
        if (mean === Infinity) continue;
        let g = (z[v] - z[u]) / (len * 1000);
        if (graded[u] && graded[v]) g = g < -H.roadGrade ? -H.roadGrade : g > H.roadGrade ? H.roadGrade : g;
        if (Math.abs(g) > H.gMax) continue;
        let e = Math.abs(g + G0) - G0;
        if (e < lim) e = lim;
        C[16 * u + k] = len * mean * Math.exp(TK * e + 0.5 * (nlh[u] + nlh[v]));
      }
    }
  }
  return { C, s };
}

// a crossing block: along index a (direction of travel), across index b; grid index of (a, b) =
// origin + a * sA + b * sB. Block move q = (da, db) -> grid move by direction.
const DIRS = ['E', 'W', 'S', 'N'];
function gridMove(dir, da, db) {
  if (dir === 'E') return [da, db];
  if (dir === 'W') return [-da, db];
  if (dir === 'S') return [db, da];
  return [db, -da];
}
const BMV = MV;   // block moves are the same 16 offsets
const FWD = [], BWD = [];
for (let q = 0; q < 16; q++) (BMV[q][0] > 0 ? FWD : BMV[q][0] < 0 ? BWD : []).push(q);
const DOWN = 2, UP = 3;   // (0, 1) and (0, -1) in block terms
const FWDA = Int32Array.from(FWD), BWDA = Int32Array.from(BWD);
const QA = Int32Array.from(BMV.map((x) => x[0])), QB = Int32Array.from(BMV.map((x) => x[1]));

// Pull form of the same Gauss-Seidel sweeps: per node, the best of the 7 moves that come from the rows
// already done in this sweep, then one compare and store. D is padded by 2 rows and 2 columns of
// Infinity, so no range checks. The two across passes (DOWN, UP) run per row as before.
function crossJob(C, s, cells, along, m, maxSweeps) {
  const n = cells.length / 2;
  const A = along * K + 2 * m, Bw = K + 2 * m, lead = -Math.floor((along - 1) / 2) * K - m;
  const W = Bw + 4;
  const out = {};
  const D = new Float32Array((A + 4) * W);
  let most = 0;
  const kDown = new Int32Array(1), kUp = new Int32Array(1);
  for (const dir of DIRS) {
    const res = new Float32Array(n);
    const kq = new Int32Array(16);
    let sA, sB;
    if (dir === 'E') { sA = 1; sB = P; } else if (dir === 'W') { sA = -1; sB = P; }
    else if (dir === 'S') { sA = P; sB = 1; } else { sA = -P; sB = 1; }
    for (let q = 0; q < 16; q++) {
      const [dc, dr] = gridMove(dir, BMV[q][0], BMV[q][1]);
      kq[q] = moveIndex(dc, dr);
    }
    // per move of the forward and backward sets: D offset to the source node and C offset from the
    // destination node's own C base (16 * node) to the source's slot
    const fd = new Int32Array(7), fc = new Int32Array(7), bd = new Int32Array(7), bc = new Int32Array(7);
    for (let i = 0; i < 7; i++) {
      let q = FWDA[i];
      fd[i] = -(QA[q] * W + QB[q]); fc[i] = kq[q] - 16 * (QA[q] * sA + QB[q] * sB);
      q = BWDA[i];
      bd[i] = -(QA[q] * W + QB[q]); bc[i] = kq[q] - 16 * (QA[q] * sA + QB[q] * sB);
    }
    const kD = kq[DOWN], kU = kq[UP], sB16 = 16 * sB;
    for (let ci = 0; ci < n; ci++) {
      const I = cells[2 * ci], J = cells[2 * ci + 1];
      let origin;
      const rowTop = halo + K * I - m, colLeft = halo + K * J - m;
      const alongStartE = halo + K * J + lead, alongStartS = halo + K * I + lead;
      if (dir === 'E') origin = rowTop * P + alongStartE;
      else if (dir === 'W') origin = rowTop * P + (alongStartE + A - 1);
      else if (dir === 'S') origin = alongStartS * P + colLeft;
      else origin = (alongStartS + A - 1) * P + colLeft;
      D.fill(Infinity);
      for (let b = m; b < m + K; b++) {
        const g = origin + (m - 1) * sA + b * sB;
        if (s[g] < Infinity) D[(m - 1 + 2) * W + b + 2] = 0;
      }
      let sweeps = 0;
      for (; sweeps < maxSweeps; sweeps++) {
        const fwd = (sweeps & 1) === 0;
        let changed = false;
        const o = fwd ? fd : bd, oc = fwd ? fc : bc;
        const d0 = o[0], d1 = o[1], d2 = o[2], d3 = o[3], d4 = o[4], d5 = o[5], d6 = o[6];
        const c0 = oc[0], c1 = oc[1], c2 = oc[2], c3 = oc[3], c4 = oc[4], c5 = oc[5], c6 = oc[6];
        for (let t = 0; t < A; t++) {
          const a = fwd ? t : A - 1 - t;
          const row = (a + 2) * W + 2;
          let u16 = 16 * (origin + a * sA);
          for (let b = 0; b < Bw; b++, u16 += sB16) {
            const di = row + b;
            let v = D[di + d0] + C[u16 + c0];
            let x = D[di + d1] + C[u16 + c1]; if (x < v) v = x;
            x = D[di + d2] + C[u16 + c2]; if (x < v) v = x;
            x = D[di + d3] + C[u16 + c3]; if (x < v) v = x;
            x = D[di + d4] + C[u16 + c4]; if (x < v) v = x;
            x = D[di + d5] + C[u16 + c5]; if (x < v) v = x;
            x = D[di + d6] + C[u16 + c6]; if (x < v) v = x;
            const dv = D[di];
            if (v < dv) { if (v < dv * (1 - 1e-4)) changed = true; D[di] = v; }
          }
          const rowA16 = 16 * (origin + a * sA);
          for (let b = 1; b < Bw; b++) {
            const v = D[row + b - 1] + C[kD + rowA16 + (b - 1) * sB16];
            const dv = D[row + b];
            if (v < dv) { if (v < dv * (1 - 1e-4)) changed = true; D[row + b] = v; }
          }
          for (let b = Bw - 2; b >= 0; b--) {
            const v = D[row + b + 1] + C[kU + rowA16 + (b + 1) * sB16];
            const dv = D[row + b];
            if (v < dv) { if (v < dv * (1 - 1e-4)) changed = true; D[row + b] = v; }
          }
        }
        if (sweeps > 0 && !changed) { sweeps++; break; }
      }
      if (sweeps > most) most = sweeps;
      let far = Infinity;
      const af = m + along * K - 1;
      for (let b = m; b < m + K; b++) if (D[(af + 2) * W + b + 2] < far) far = D[(af + 2) * W + b + 2];
      const lat = north - (halo + K * I + K / 2) / N;
      const span = along * K * KM_DEG / N * (dir === 'E' || dir === 'W' ? Math.cos(lat * Math.PI / 180) : 1);
      res[ci] = far / span;
    }
    out[dir] = res;
  }
  return { out, most };
}

const t0 = performance.now();
const parts = [], sweepsRun = [];
const cache = {};
const timing = {};
for (let j = 0; j < H.jobs.length; j++) {
  const job = H.jobs[j];
  const tc = performance.now();
  if (!cache[job.grid]) {
    for (const g of Object.keys(cache)) if (g !== job.grid) delete cache[g];
    cache[job.grid] = costs(job.grid === 'roads');
    timing['costs_' + job.grid] = (performance.now() - tc) / 1000;
  }
  const { C, s } = cache[job.grid];
  const tx = performance.now();
  const { out, most } = crossJob(C, s, cellsAll[j], job.along, job.m, job.maxSweeps);
  timing['job' + j] = (performance.now() - tx) / 1000;
  for (const d of DIRS) parts.push(Buffer.from(out[d].buffer));
  sweepsRun.push(most);
}
parts.push(Buffer.from(new Int32Array(sweepsRun).buffer));
writeFileSync(outPath, Buffer.concat(parts));
process.stderr.write(JSON.stringify({ total: (performance.now() - t0) / 1000, ...timing }) + '\n');
