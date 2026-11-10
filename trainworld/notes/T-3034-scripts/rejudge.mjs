import fs from "node:fs";
const cases = JSON.parse(fs.readFileSync("recases.json")), osm = JSON.parse(fs.readFileSync("osmre.json"));
const M = 111320;
const junc = (ways) => { const deg = new Map(), pos = new Map(); for (const w of ways) w.nodes.forEach((id, i) => { deg.set(id, (deg.get(id) ?? 0) + (i === 0 || i === w.nodes.length - 1 ? 1 : 2)); pos.set(id, w.geom[i]); }); return [...deg].filter(([, g]) => g >= 3).map(([id]) => pos.get(id)); };
let n = 0;
for (const c of cases) {
  const ways = osm[c.key]; if (!ways) { c.t = null; continue; }
  const kx = Math.cos((c.lat * Math.PI) / 180) * M;
  c.t = junc(ways).some(([la, lo]) => Math.hypot((la - c.lat) * M, (lo - c.lng) * kx) <= 6);
}
const rep = (name, f) => { const x = cases.filter((c) => c.t !== null && f(c)); console.log(name.padEnd(48), `${x.filter((c) => c.t).length}/${x.length} OSM junction`); };
rep("street ends, naive 3 m, all", (c) => c.kind === "street end");
rep("street ends kept by the new rule", (c) => c.kind === "street end" && c.kept);
rep("street ends the new rule drops (parallel)", (c) => c.kind === "street end" && !c.kept);
rep("link ends (all 3 m)", (c) => c.kind === "link end");
console.log("unchecked", cases.filter((c) => c.t === null).length);
