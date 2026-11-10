import fs from "node:fs";
const d = JSON.parse(fs.readFileSync(new URL("./cases.json", import.meta.url)));
const all = Object.values(d.perCity).flatMap((p) => p.linkEnds);
const lenB = [[0, 20], [20, 40], [40, 60]];
const wB = [[0, 1.01, "<=1 (kept now)"], [1.01, 3, "1-3"], [3, 5, "3-5"], [5, 8, "5-8"], [8, 12.01, "8-12"]];
console.log("untagged service lines <=60 m, by length and by the farther end's distance to a street");
console.log("len".padEnd(8), wB.map((b) => b[2].padStart(16)).join(""), "  one end only(<=3 m) other >3");
for (const [a, b] of lenB) {
  const xs = all.filter((x) => x.len >= a && x.len < b);
  console.log(`${a}-${b}`.padEnd(8), wB.map(([lo, hi]) => String(xs.filter((x) => x.worst >= lo && x.worst < hi).length).padStart(16)).join(""), String(xs.filter((x) => x.best <= 3 && x.worst > 3).length).padStart(8), "  total", xs.length);
}
console.log("bridging pieces (excluded service, both ends on two different streets within 2.5 m), by type and length");
const br = Object.values(d.perCity).flatMap((p) => p.bridging);
const types = [...new Set(br.map((x) => x.type))];
for (const t of types) console.log(t.padEnd(28), "<=15:", br.filter((x) => x.type === t && x.lenM <= 15).length, " 16-30:", br.filter((x) => x.type === t && x.lenM > 15 && x.lenM <= 30).length, " 31-60:", br.filter((x) => x.type === t && x.lenM > 30).length);
