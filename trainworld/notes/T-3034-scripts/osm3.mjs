import fs from "node:fs";
const N = JSON.parse(fs.readFileSync("recount_naive.json")).ends.filter((e) => !e.spur);
const W = JSON.parse(fs.readFileSync("recount_new.json")).ends;
const inNew = new Set(W.map((e) => `${e.tile}:${e.x}:${e.y}`));
const cases = N.map((e) => ({ ...e, key: `${e.tile}:${e.x}:${e.y}`, kept: inNew.has(`${e.tile}:${e.x}:${e.y}`), kind: e.w === 0.5 ? "link end" : "street end" }));
fs.writeFileSync("recases.json", JSON.stringify(cases));
console.log("cases", cases.length, cases.filter((c) => c.kind === "street end").length, "street;", cases.filter((c) => c.kind === "street end" && c.kept).length, "kept");
const M = 111320, CL = "motorway|trunk|primary|secondary|tertiary|unclassified|residential|service|living_street|motorway_link|trunk_link|primary_link|secondary_link|tertiary_link|road";
const out = {};
const B = 12;
for (let i = 0; i < cases.length; i += B) {
  const batch = cases.slice(i, i + B);
  const q = `[out:json][timeout:60];(${batch.map((c) => `way(around:30,${c.lat.toFixed(7)},${c.lng.toFixed(7)})["highway"~"^(${CL})$"];`).join("")});out geom qt;`;
  for (let a = 0; a < 6; a++) {
    try {
      const r = await fetch("https://overpass-api.de/api/interpreter", { method: "POST", signal: AbortSignal.timeout(100000), headers: { "User-Agent": "trainworld-roadgaps-measure/1.0", "Content-Type": "application/x-www-form-urlencoded" }, body: "data=" + encodeURIComponent(q) });
      if (r.status === 200) { const j = JSON.parse(await r.text()); for (const c of batch) out[c.key] = j.elements.filter((w) => w.geometry).map((w) => ({ nodes: w.nodes, geom: w.geometry.map((p) => [p.lat, p.lon]) })); break; }
      await new Promise((r) => setTimeout(r, 10000));
    } catch (e) { await new Promise((r) => setTimeout(r, 6000)); }
  }
  fs.writeFileSync("osmre.json", JSON.stringify(out));
  console.log("batch", i / B + 1, Object.keys(out).length);
}
