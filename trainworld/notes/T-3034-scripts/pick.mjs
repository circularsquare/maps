// pick.mjs band n out.json [city-cap]  : from judged.json choose n cases of a band, alternating true/false
import fs from "node:fs";
const [band, n, out, seedArg] = process.argv.slice(2);
const j = JSON.parse(fs.readFileSync(new URL("./judged.json", import.meta.url))).filter((c) => c.band === band && c.t3 !== null && (c.cat.startsWith("street end")));
let seed = +(seedArg ?? 7); const rnd = () => ((seed = (seed * 1664525 + 1013904223) % 4294967296) / 4294967296);
const T = j.filter((c) => c.t3).sort(() => rnd() - 0.5), F = j.filter((c) => !c.t3).sort(() => rnd() - 0.5);
const pick = [];
for (let i = 0; pick.length < +n && (T.length || F.length); i++) { const c = (i % 2 === 0 ? T : F).pop() ?? (T.pop() ?? F.pop()); if (c) pick.push(c); }
for (const c of pick) c.label = `${c.city} d=${c.d} m cos=${c.cos}  OSM junction: ${c.t3 ? "yes" : "no"}`;
fs.writeFileSync(out, JSON.stringify(pick));
console.log(pick.length);
