import fs from "node:fs";
const s = JSON.parse(fs.readFileSync(new URL("./judged.json", import.meta.url))).filter((c) => (c.cat.startsWith("street end")) && c.t3 !== null);
const d = JSON.parse(fs.readFileSync(new URL("./cases.json", import.meta.url)));
const pop = d.cases.filter((c) => c.cat.startsWith("street end") && c.spurM === null);
const type = (c) => (c.candEnd && c.par > 0.8 ? "E2E collinear" : c.candEnd ? "E2E other" : c.cos >= 0.7 && c.par < 0.5 ? "T pointing" : c.cos >= 0.7 ? "ahead, oblique" : "other");
const types = ["E2E collinear", "E2E other", "T pointing", "ahead, oblique", "other"];
const bands = ["1-2", "2-3", "3-5", "5-8", "8-12"];
console.log("type".padEnd(16), "band    pop  sample  true@d+3");
for (const t of types) for (const b of bands) { const x = s.filter((c) => c.band === b && type(c) === t), p = pop.filter((c) => c.band === b && type(c) === t); if (p.length) console.log(t.padEnd(16), b.padEnd(6), String(p.length).padStart(4), String(x.length).padStart(6), `${x.filter((c) => c.t3).length}/${x.length}`.padStart(9)); }
