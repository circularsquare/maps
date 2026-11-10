// T-3037: the people behind the national and Europe rail-share achievements (app/src/game/achievements.ts
// NATION_PEOPLE): each country's zone populations summed from market-rows.json, and Europe's by zone
// (continentOf's rule). Prints the table to paste when the registry's zones change.
//   node trainworld/checks/nation-people.mjs
import fs from "fs";
const dir = new URL("../app/src/game/", import.meta.url);
const m = JSON.parse(fs.readFileSync(new URL("market-rows.json", dir), "utf8"));
const r = JSON.parse(fs.readFileSync(new URL("registry.json", dir), "utf8"));
const src = fs.readFileSync(new URL("continents.ts", dir), "utf8");
const europe = new Set(/Europe: "([^"]+)"/.exec(src)[1].split(" "));
const inThrace = (lng, lat) => (lat >= 40.9 ? lng < 28.99 + 0.62 * (lat - 41.01) : lat >= 40.42 ? lng < 27.3 : lng < 26.2 + 1.3 * (lat - 40.03));
const isEurope = (cc, lng, lat) => cc === "RU" ? lng <= 60 : cc === "KZ" ? lng < 51.5 : cc === "TR" ? inThrace(lng, lat) : europe.has(cc);
const by = {}; let eu = 0;
for (const c of r.cities) {
  const row = m[c.id]; if (!row) continue;
  by[c.country] = (by[c.country] ?? 0) + row.pop;
  if (isEurope(c.country, c.center[0], c.center[1])) eu += row.pop;
}
const out = { EUROPE: Math.round(eu) };
for (const k of ["JP", "US", "ES", "GB", "IT", "DE", "KR", "TW", "CN", "ID", "MX", "BR", "AU", "TR"]) out[k] = Math.round(by[k] ?? 0);
console.log(JSON.stringify(out));
