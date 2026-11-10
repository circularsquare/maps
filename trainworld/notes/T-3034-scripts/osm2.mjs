// sample cases, fetch OSM ways near each in batches (Overpass `around`), save osmcases.json
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
const here = path.dirname(fileURLToPath(import.meta.url));
const d = JSON.parse(fs.readFileSync(path.join(here, "cases.json")));
export const keyOf = (c) => `${c.city}:${c.tx}:${c.ty}:${c.li}:${c.endIdx}`;
// deterministic shuffle
let seed = 12345; const rnd = () => ((seed = (seed * 1664525 + 1013904223) % 4294967296) / 4294967296);
const shuffle = (a) => { a = a.slice(); for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; };
const pickBand = (cases, n, perCityCap) => {
  const by = {};
  for (const c of shuffle(cases)) (by[c.city] ??= []).push(c);
  const out = []; let more = true; const cnt = {};
  while (out.length < n && more) { more = false; for (const city of Object.keys(by)) { if (out.length >= n) break; if ((cnt[city] ?? 0) >= perCityCap || !by[city].length) continue; out.push(by[city].pop()); cnt[city] = (cnt[city] ?? 0) + 1; more = true; } }
  return out;
};
const sel = [];
const street = d.cases.filter((c) => (c.cat === "street end near street" || c.cat === "street end near link") && c.spurM === null);
for (const [b, n] of [["1-2", 40], ["2-3", 45], ["3-5", 45], ["5-8", 45], ["8-12", 35]]) sel.push(...pickBand(street.filter((c) => c.band === b), n, 8));
for (const cat of ["ground end near bridge/tunnel piece", "end of street near motorway", "bridge/tunnel end, other level"]) sel.push(...pickBand(d.cases.filter((c) => c.cat === cat && c.band !== "<1"), 25, 6));
console.log("sampled", sel.length);
fs.writeFileSync(path.join(here, "sample.json"), JSON.stringify(sel));
const out = fs.existsSync(path.join(here, "osmcases.json")) ? JSON.parse(fs.readFileSync(path.join(here, "osmcases.json"))) : {};
const CL = "motorway|trunk|primary|secondary|tertiary|unclassified|residential|service|living_street|motorway_link|trunk_link|primary_link|secondary_link|tertiary_link|road";
const mirrors = ["https://overpass-api.de/api/interpreter", "https://overpass.private.coffee/api/interpreter"];
const todo = sel.filter((c) => !(keyOf(c) in out));
const segNear = (g, c, kx, R) => { for (let i = 0; i + 1 < g.length; i++) { const ax = (g[i].lon - c.lng) * kx, ay = (g[i].lat - c.lat) * M, bx = (g[i + 1].lon - c.lng) * kx, by = (g[i + 1].lat - c.lat) * M, dx = bx - ax, dy = by - ay, l2 = dx * dx + dy * dy, t = l2 ? Math.max(0, Math.min(1, -(ax * dx + ay * dy) / l2)) : 0; if (Math.hypot(ax + t * dx, ay + t * dy) < R) return true; } return false; };
const B = 12; const M = 111320;
for (let i = 0; i < todo.length; i += B) {
  const batch = todo.slice(i, i + B);
  const q = `[out:json][timeout:60];(${batch.map((c) => `way(around:42,${c.lat.toFixed(7)},${c.lng.toFixed(7)})["highway"~"^(${CL})$"];`).join("")});out geom qt;`;
  let ok = false;
  for (let a = 0; a < 5 && !ok; a++) {
    try {
      const ctl = AbortSignal.timeout(100000);
      const r = await fetch(mirrors[a % 2 === 0 ? 0 : 0], { method: "POST", signal: ctl, headers: { "User-Agent": "trainworld-roadgaps-measure/1.0", "Content-Type": "application/x-www-form-urlencoded" }, body: "data=" + encodeURIComponent(q) });
      if (r.status === 200) {
        const j = JSON.parse(await r.text());
        for (const c of batch) {
          const kx = Math.cos((c.lat * Math.PI) / 180) * M;
          out[keyOf(c)] = j.elements.filter((w) => w.type === "way" && w.geometry && segNear(w.geometry, c, kx, 45)).map((w) => ({ id: w.id, hw: w.tags?.highway, bridge: w.tags?.bridge, tunnel: w.tags?.tunnel, layer: w.tags?.layer, nodes: w.nodes, geom: w.geometry.map((p) => [p.lat, p.lon]) }));
        }
        ok = true;
      } else { console.log("status", r.status); await new Promise((r) => setTimeout(r, 10000)); }
    } catch (e) { console.log("err", String(e).slice(0, 80)); await new Promise((r) => setTimeout(r, 6000)); }
  }
  fs.writeFileSync(path.join(here, "osmcases.json"), JSON.stringify(out));
  console.log("batch", i / B + 1, "of", Math.ceil(todo.length / B), ok ? "ok" : "FAILED", Object.keys(out).length);
  await new Promise((r) => setTimeout(r, 2000));
}
