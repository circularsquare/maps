// usage: node scene.mjs picks.json scenes.json ; picks = [{tx,ty,x,y,label,...}]
// writes, per pick, the joined graph lines, left-out roads and OSM ways within R m, in metres from the end point (x east, y north)
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
const here = path.dirname(fileURLToPath(import.meta.url));
const SRC = "C:/Users/anita/projects/maps/trainworld/app/src/map/";
const { decodeRoads } = await import("file:///" + SRC + "roadTiles.ts");
const { joinRoads, metresPerUnit, EXTENT, MOTORWAY, TRUNK, LINK } = await import("file:///" + SRC + "roadGraph.ts");
const busRoad = (cls, brunnel, service, access) =>
  cls === "trunk" ? TRUNK : ["primary", "secondary", "tertiary", "minor", "busway"].includes(cls) ? 1 : cls === "motorway" ? MOTORWAY
    : cls === "service" && !service && !access && !brunnel ? LINK : 0;
const leftOut = (cls) => (cls === "service" ? 0.2 : cls === "pedestrian" ? 0.3 : cls === "path" ? 0.31 : cls === "track" ? 0.32 : 0);
const picks = JSON.parse(fs.readFileSync(process.argv[2]));
const R = +(process.argv[4] ?? 30);
const WORLD = EXTENT * 2 ** 14;
const latOfG = (gy) => (360 / Math.PI) * Math.atan(Math.exp(Math.PI * (1 - (2 * gy) / WORLD))) - 90;
const out = [];
for (const p of picks) {
  const buf = fs.readFileSync(path.join(here, "tiles", `${p.tx}_${p.ty}.pbf`));
  const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  const mpu = metresPerUnit((p.ty + 0.5) * EXTENT);
  const rl = joinRoads(decodeRoads(ab, busRoad), mpu);
  const lf = decodeRoads(ab, leftOut);
  const U = R / mpu;
  const clip = (lines, weights, levels, sc) => {
    const res = [];
    lines.forEach((l, li) => {
      const pts = [];
      let any = false;
      for (let i = 0; i < l.length; i += 2) {
        const dx = (l[i] * sc - p.x) * mpu, dy = -(l[i + 1] * sc - p.y) * mpu;
        pts.push([+dx.toFixed(2), +dy.toFixed(2)]);
        if (i > 0) { const [px, py] = pts[pts.length - 2]; if (Math.max(px, dx) > -R * 1.3 && Math.min(px, dx) < R * 1.3 && Math.max(py, dy) > -R * 1.3 && Math.min(py, dy) < R * 1.3) any = true; }
      }
      if (any) res.push({ w: weights[li], lev: levels?.[li] ?? 0, pts });
    });
    return res;
  };
  const g = clip(rl.lines, rl.weights, rl.levels, 1);
  const o = clip(lf.lines, lf.weights, null, EXTENT / lf.extent);
  // osm
  const ow = [], oj = [];
  const oc = JSON.parse(fs.readFileSync(path.join(here, "osmcases.json")));
  const ways = oc[`${p.city}:${p.tx}:${p.ty}:${p.li}:${p.endIdx}`] ?? [];
  const kx = Math.cos((p.lat * Math.PI) / 180) * 111320, deg = new Map(), pos = new Map();
  for (const w of ways) {
    const pts = w.geom.map(([la, lo]) => [(lo - p.lng) * kx, (la - p.lat) * 111320]);
    ow.push({ hw: w.hw, pts: pts.map(([a, b]) => [+a.toFixed(2), +b.toFixed(2)]) });
    w.nodes.forEach((id, i) => { const e = i === 0 || i === w.nodes.length - 1; deg.set(id, (deg.get(id) ?? 0) + (e ? 1 : 2)); pos.set(id, pts[i]); });
  }
  for (const [id, dg] of deg) if (dg >= 3) { const q = pos.get(id); if (Math.abs(q[0]) < R && Math.abs(q[1]) < R) oj.push([+q[0].toFixed(1), +q[1].toFixed(1)]); }
  out.push({ ...p, game: g, left: o, osm: ow, junctions: oj });
}
fs.writeFileSync(process.argv[3], JSON.stringify(out));
