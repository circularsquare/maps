// classify what joinRoads(mpu) adds over the old fixed 2 units (mpu=1.5 gives gap 2 units)
import fs from "node:fs";
const SRC = "C:/Users/anita/projects/maps/trainworld/app/src/map/";
const { decodeRoads } = await import("file:///" + SRC + "roadTiles.ts");
const { metresPerUnit, EXTENT, MOTORWAY, TRUNK, LINK } = await import("file:///" + SRC + "roadGraph.ts");
const HERE = "file:///C:/Users/anita/AppData/Local/Temp/claude/dce1fefcc1d43aef/6f38e27c-5131-4ca5-81d2-160b4694f258/scratchpad/roadgaps/mods/";
const MODE = process.argv[2] ?? "new";
const joinOld = (await import(HERE + "old.ts")).joinRoads, joinNaive = (await import(HERE + "naive.ts")).joinRoads, joinCur = (await import("file:///" + SRC + "roadGraph.ts")).joinRoads;
const joinNew = MODE === "naive" ? joinNaive : joinCur;
const busRoad = (cls, brunnel, service, access) =>
  cls === "trunk" ? TRUNK : ["primary", "secondary", "tertiary", "minor", "busway"].includes(cls) ? 1 : cls === "motorway" ? MOTORWAY
    : cls === "service" && !service && !access && !brunnel ? LINK : 0;
const T = "./tiles/";
const out = { tiles: 0, endsNew: 0, spur: 0, nonSpur: 0, interiorVertsShared: 0, byCity: {}, ends: [] };
const keyOf = (x, y) => x * 8192 + y;
function sharedMap(r) { const at = new Map(); r.lines.forEach((l, li) => { for (let i = 0; i < l.length; i += 2) { const k = keyOf(l[i], l[i + 1]); (at.get(k) ?? at.set(k, new Set()).get(k)).add(li); } }); return at; }
for (const f of fs.readdirSync(T)) {
  const b = fs.readFileSync(T + f); if (!b.length) continue;
  const [tx, ty] = f.replace(".pbf", "").split("_").map(Number);
  const ab = b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
  const raw = decodeRoads(ab, busRoad), mpu = metresPerUnit((ty + 0.5) * EXTENT);
  const A = joinOld(raw, mpu), C = joinNew(raw, mpu);
  const shA = sharedMap(A), shC = sharedMap(C);
  out.tiles++;
  // interior vertices (non-end) shared with another line in C but not in A, by weight class
  C.lines.forEach((l, li) => {
    for (let i = 0; i < l.length; i += 2) {
      const x = l[i], y = l[i + 1];
      if (x <= 0 || y <= 0 || x >= EXTENT || y >= EXTENT) continue;
      const end = i === 0 || i === l.length - 2;
      const nowS = shC.get(keyOf(x, y)).size > 1, wasS = (shA.get(keyOf(x, y))?.size ?? 0) > 1;
      if (!nowS || wasS) continue;
      if (!end) { out.interiorVertsShared++; continue; }
      out.endsNew++;
      // spur: shared vertex within 15 m along line (in the OLD join)
      let along = 0, spur = false;
      for (let q = 1; q < l.length / 2; q++) {
        const bi = i === 0 ? 2 * q : l.length - 2 - 2 * q, pi = i === 0 ? 2 * q - 2 : l.length - 2 - 2 * (q - 1);
        along += Math.hypot(l[bi] - l[pi], l[bi + 1] - l[pi + 1]) * mpu; if (along > 15) break;
        const k = keyOf(l[bi], l[bi + 1]);
        if ((shA.get(k)?.size ?? 0) > 1) { spur = true; break; }
      }
      // other line's weight/level at that vertex; heading parallelism with the segment it joined
      const others = [...shC.get(keyOf(x, y))].filter((o) => o !== li);
      // parallel: end heading vs other line's local direction
      const bi = i === 0 ? 2 : l.length - 4; let hx = x - l[bi], hy = y - l[bi + 1]; const hl = Math.hypot(hx, hy) || 1; hx /= hl; hy /= hl;
      let par = 0;
      for (const o of others) { const m = C.lines[o]; for (let j = 0; j < m.length; j += 2) if (m[j] === x && m[j + 1] === y) { for (const [a, bb] of [[j - 2, j], [j, j + 2]]) if (a >= 0 && bb < m.length) { const dx = m[bb] - m[a], dy = m[bb + 1] - m[a + 1], dl = Math.hypot(dx, dy) || 1; par = Math.max(par, Math.abs((hx * dx + hy * dy) / dl)); } } }
      (spur ? out.spur++ : out.nonSpur++);
      let dd = 1e9; A.lines.forEach((m, lj) => { for (let j = 0; j + 3 < m.length; j += 2) { const ax = m[j], ay = m[j + 1], ddx = m[j + 2] - ax, ddy = m[j + 3] - ay, t = Math.max(0, Math.min(1, ((x - ax) * ddx + (y - ay) * ddy) / (ddx * ddx + ddy * ddy))); const dist = Math.hypot(ax + t * ddx - x, ay + t * ddy - y) * mpu; if (dist < dd && !(A.lines[lj] === undefined) ) { const own = lj === li; if (!own || true) dd = Math.min(dd, dist); } } });
      const W = 4096 * 2 ** 14, gx = tx * 4096 + x, gy = ty * 4096 + y;
      out.ends.push({ x, y, tx, ty, li, endIdx: i, city: "?", lng: (gx / W) * 360 - 180, lat: (360 / Math.PI) * Math.atan(Math.exp(Math.PI * (1 - (2 * gy) / W))) - 90, d: +dd.toFixed(2), tile: f, spur, par: +par.toFixed(2), w: C.weights[li], lev: C.levels[li], ow: others.map((o) => C.weights[o]), lenM: +(l.length / 2) });
    }
  });
  out.byCity[f] = (out.byCity[f] ?? 0);
}
const e = out.ends;
console.log({ tiles: out.tiles, endsNewlyJoined: out.endsNew, spur: out.spur, nonSpur: out.nonSpur, interiorVertsNewlyShared: out.interiorVertsShared });
console.log("non-spur parallel (|cos|>0.9):", e.filter((x) => !x.spur && x.par > 0.9).length, " all parallel:", e.filter((x) => x.par > 0.9).length);
const per = {}; for (const x of e) per[x.tile] = (per[x.tile] ?? 0) + 1;
console.log(Object.entries(per).sort((a, b) => b[1] - a[1]).slice(0, 6));
fs.writeFileSync("recount.json", JSON.stringify(out));
