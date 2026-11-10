import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
const here = path.dirname(fileURLToPath(import.meta.url));
const d = JSON.parse(fs.readFileSync(path.join(here, "cases.json")));
const sample0 = JSON.parse(fs.readFileSync(path.join(here, "sample.json")));
const byKey = new Map(d.cases.map((c) => [`${c.city}:${c.tx}:${c.ty}:${c.li}:${c.endIdx}`, c]));
const sample = sample0.map((c) => ({ ...c, ...byKey.get(`${c.city}:${c.tx}:${c.ty}:${c.li}:${c.endIdx}`) }));
const osm = JSON.parse(fs.readFileSync(path.join(here, "osmcases.json")));
const keyOf = (c) => `${c.city}:${c.tx}:${c.ty}:${c.li}:${c.endIdx}`;
const M = 111320;
function junctions(ways) {
  const deg = new Map(), pos = new Map();
  for (const w of ways) w.nodes.forEach((id, i) => { const end = i === 0 || i === w.nodes.length - 1; deg.set(id, (deg.get(id) ?? 0) + (end ? 1 : 2)); pos.set(id, w.geom[i]); });
  return [...deg].filter(([, g]) => g >= 3).map(([id]) => pos.get(id));
}
const ev = (c, slack) => {
  const ways = osm[keyOf(c)]; if (!ways) return null;
  const kx = Math.cos((c.lat * Math.PI) / 180) * M, R = Math.max(c.d + slack, 3);
  return junctions(ways).some(([la, lo]) => Math.hypot((la - c.lat) * M, (lo - c.lng) * kx) <= R);
};
for (const c of sample) { c.t15 = ev(c, 1.5); c.t3 = ev(c, 3); c.t5 = ev(c, 5); c.nways = osm[keyOf(c)]?.length ?? -1; }
fs.writeFileSync(path.join(here, "judged.json"), JSON.stringify(sample));
const bands = ["1-2", "2-3", "3-5", "5-8", "8-12"];
const isStreet = (c) => c.cat === "street end near street" || c.cat === "street end near link";
const st = sample.filter((c) => isStreet(c) && c.t3 !== null);
const rate = (xs, k) => (xs.length ? `${xs.filter((c) => c[k]).length}/${xs.length} = ${(100 * xs.filter((c) => c[k]).length / xs.length).toFixed(0)}%` : "-");
console.log("sampled with osm", st.length);
console.log("band  true@d+1.5          true@d+3            true@d+5");
for (const b of bands) { const xs = st.filter((c) => c.band === b); console.log(b.padEnd(5), rate(xs, "t15").padEnd(19), rate(xs, "t3").padEnd(19), rate(xs, "t5")); }
const cats = [...new Set(sample.map((c) => c.cat))].filter((c) => !isStreet({ cat: c }));
console.log("-- other causes true@d+3");
for (const cat of cats) { const xs = sample.filter((c) => c.cat === cat && c.t3 !== null); console.log(cat.padEnd(40), rate(xs, "t3")); }
// angle features of the true vs false street cases
console.log("-- cos of heading toward the road, street cases: mean true / false; share cos>=0.7");
for (const b of bands) { const xs = st.filter((c) => c.band === b); const T = xs.filter((c) => c.t3), F = xs.filter((c) => !c.t3); const mean = (a, k) => (a.length ? (a.reduce((s, c) => s + c[k], 0) / a.length).toFixed(2) : "-"); console.log(b.padEnd(5), "true cos", mean(T, "cos"), "false cos", mean(F, "cos"), "| true cos>=.7", T.filter((c) => c.cos >= 0.7).length + "/" + T.length, "false cos>=.7", F.filter((c) => c.cos >= 0.7).length + "/" + F.length); }
// rules
const popStreet = d.cases.filter((c) => isStreet(c) && c.spurM === null);
console.log("-- rule eval: join street ends with 1<=d<=G and cos>=C; population joins per 12 cities, and sample precision");
const rows = [];
for (const G of [2, 3, 4, 5, 6, 8, 12]) for (const C of [-2, 0.5, 0.7, 0.85, 0.93]) {
  const sel = st.filter((c) => c.d >= 1 && c.d <= G && c.cos >= C);
  const tr = sel.filter((c) => c.t3).length;
  // population estimate: per band, population passing x sample false-rate in band among those passing
  let popJoins = 0, popFalse = 0;
  const perCity = {};
  for (const b of bands) {
    const bl = [1, 2, 3, 5, 8, 12], lo = { "1-2": 1, "2-3": 2, "3-5": 3, "5-8": 5, "8-12": 8 }[b];
    const hi = { "1-2": 2, "2-3": 3, "3-5": 5, "5-8": 8, "8-12": 12 }[b];
    if (lo >= G) continue;
    const sb = st.filter((c) => c.band === b && c.cos >= C);
    const fr = sb.length ? sb.filter((c) => !c.t3).length / sb.length : 1;
    const pb = popStreet.filter((c) => c.band === b && c.d <= G && c.cos >= C);
    popJoins += pb.length; popFalse += pb.length * fr;
    for (const c of pb) { perCity[c.city] ??= [0, 0]; perCity[c.city][0]++; perCity[c.city][1] += fr; }
  }
  rows.push({ G, C, sample: sel.length, trueS: tr, popJoins, popFalse: +popFalse.toFixed(0), perCity });
  console.log(`G=${G} C=${C}`.padEnd(14), "sample", String(sel.length).padStart(3), "true", String(tr).padStart(3), "prec", ((100 * tr) / (sel.length || 1)).toFixed(0).padStart(3) + "%", "| all 12 cities: joins", String(popJoins).padStart(4), "false est", popFalse.toFixed(0).padStart(4));
}
fs.writeFileSync(path.join(here, "rules.json"), JSON.stringify(rows));
