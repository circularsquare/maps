// The fare table as markdown, poorest first: city, local fare, US$, GDP per head, game default, source.
import fs from "fs";
const dir = "C:/Users/anita/AppData/Local/Temp/claude/dce1fefcc1d43aef/b35f881f-a7d6-4f9f-9c17-4b4b2ffbdf21/scratchpad/farescale/";
const T = JSON.parse(fs.readFileSync("C:/Users/anita/projects/maps/trainworld/app/src/game/countryIncome.json", "utf8"));
const lines = fs.readFileSync(dir + "fares.csv", "utf8").trim().split(/\r?\n/).slice(1);
const parse = (l) => { const out = []; let cur = "", q = false; for (const ch of l) { if (ch === '"') q = !q; else if (ch === "," && !q) { out.push(cur); cur = ""; } else cur += ch; } out.push(cur); return out; };
const tenth = (v) => Math.min(19.9, Math.max(0.1, Math.round(v * 10) / 10));
const s = (c) => tenth((T.usd[c] / T.usd.IN) ** 0.7);
const fix = { "New York": ["3.00", 3.0, "MTA OMNY from Jan 2026 (T-1521's table)"], "Paris": ["2.55", 2.88, "IDFM 2026 (T-1521's table)"] };
const rows = lines.map(parse).map(([city, iso, mode, fl, cur, covers, year, usd, fx, src, conf]) => ({ city, iso, mode, fl, cur, usd, src, conf, g: T.usd[iso] }));
rows.sort((a, b) => a.g - b.g);
const out = ["| City | Mode | Fare | US$ | GDP/head | Game default | Real / game | Source | Note |", "|---|---|---|---|---|---|---|---|---|"];
for (const r of rows) {
  const f = fix[r.city];
  const usd = f ? f[1] : r.usd ? +r.usd : null;
  const game = Math.round(2.5 * s(r.iso) / s("US") / 0.05) * 0.05;
  const link = /^https?:/.test(r.src) ? `[link](${r.src})` : f ? f[2] : r.src;
  const note = (r.conf.startsWith("weak") ? "weak" : r.conf.startsWith("not found") ? "not found" : r.conf) + (["IR", "UZ"].includes(r.iso) ? "; left out of the fit" : "") + (/^(Tijuana|Johannesburg|Mumbai .suburban rail.)$/.test(r.city) ? "; not metro, left out of the fit" : "");
  out.push(`| ${r.city} | ${r.mode} | ${f ? f[0] : r.fl} ${r.cur} | ${usd === null ? "" : usd.toFixed(2)} | ${(r.g / 1000).toFixed(1)}k | ${game.toFixed(2)} | ${usd ? (usd / game).toFixed(2) : ""} | ${link} | ${note} |`);
}
fs.writeFileSync(dir + "table.md", out.join("\n") + "\n");
console.log(out.length - 2, "rows");
