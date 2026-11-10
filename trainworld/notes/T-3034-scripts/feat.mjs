import fs from "node:fs";
const s = JSON.parse(fs.readFileSync(new URL("./judged.json", import.meta.url))).filter((c) => c.cat.startsWith("street end") && c.t3 !== null);
const d = JSON.parse(fs.readFileSync(new URL("./cases.json", import.meta.url)));
const pop = d.cases.filter((c) => c.cat.startsWith("street end") && c.spurM === null);
const show = (name, f) => {
  console.log(name);
  for (const [lo, hi] of [[1, 3], [3, 5], [5, 8]]) {
    const x = s.filter((c) => c.d >= lo && c.d < hi && f(c)), p = pop.filter((c) => c.d >= lo && c.d < hi && f(c));
    console.log(`  ${lo}-${hi} m: population ${p.length}, sample ${x.length}, OSM-true ${x.filter((c) => c.t3).length}`);
  }
};
show("all", () => true);
show("end within 1.5 m of a left-out road (alley, path...)", (c) => c.leftD !== null && c.leftD <= 1.5);
show("end NOT near a left-out road (>1.5 m)", (c) => c.leftD === null || c.leftD > 1.5);
show("last segment >= 15 m long and cos>=0.9 (stem aimed at road)", (c) => c.cos >= 0.9 && c.segLenM >= 15);
show("long line (>=60 m) and cos>=0.9", (c) => c.cos >= 0.9 && c.lineLenM >= 60);
show("short line (<60 m)", (c) => c.lineLenM < 60);
show("cos>=0.9 and par<0.4 (stem aimed, perpendicular)", (c) => c.cos >= 0.9 && c.par < 0.4);
show("end of candidate is end of other line (E2E)", (c) => c.candEnd);
show("candidate not E2E", (c) => !c.candEnd);
