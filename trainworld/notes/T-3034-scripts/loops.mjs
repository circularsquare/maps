import fs from "node:fs";
const s = JSON.parse(fs.readFileSync(new URL("./judged.json", import.meta.url))).filter((c) => c.cat.startsWith("street end") && c.t3 !== null);
const d = JSON.parse(fs.readFileSync(new URL("./cases.json", import.meta.url)));
const pop = d.cases.filter((c) => c.cat.startsWith("street end") && c.spurM === null);
console.log("end within G m of its own line's other end (open loop)");
for (const G of [3, 5, 8]) {
  for (const [lo, hi] of [[1, 3], [3, 5], [5, 8]]) {
    const x = s.filter((c) => c.d >= lo && c.d < hi && c.selfD <= G), p = pop.filter((c) => c.d >= lo && c.d < hi && c.selfD <= G);
    console.log(` selfD<=${G} d ${lo}-${hi}: pop ${p.length} sample ${x.length} true ${x.filter((c) => c.t3).length}`);
  }
}
// all ends (not only the street-gap cases): open loops with selfD<=8 and the other end... also free ends
let n = 0; for (const c of d.cases) if (c.selfD <= 5 && c.lineLenM >= 15) n++;
console.log("all cases with selfD<=5 m and line >=15 m:", n);
