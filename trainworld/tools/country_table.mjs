// T-3116: writes shonei-server's trainworld_countries.json ({city id: lowercase ISO country}) from the
// registry, so the server can sum a game's per-city points into per-country boards; T-3132: and
// trainworld_regions.json ({city id: realm id}, game/realms.ts's zoneRealm), for Community's region filter.
// Rerun it, and redeploy the server, when a country's cities or the realms change. Usage (maps root):
//   node trainworld/tools/country_table.mjs [path to shonei-server]
import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { join } from "node:path";

const here = fileURLToPath(new URL(".", import.meta.url));
const reg = JSON.parse(readFileSync(join(here, "../app/src/game/registry.json"), "utf8"));
const { realms } = JSON.parse(readFileSync(join(here, "../app/src/game/realms.json"), "utf8"));
const server = process.argv[2] ?? "C:/Users/anita/projects/shonei-server";
const table = {};
for (const c of reg.cities) if (c.country) table[c.id] = c.country.toLowerCase();
const out = join(server, "trainworld_countries.json");
writeFileSync(out, JSON.stringify(table) + "\n");
console.log(`${Object.keys(table).length} cities, ${new Set(Object.values(table)).size} countries -> ${out}`);

const known = new Set(realms.map((r) => r.id));
const byCountry = new Map(realms.flatMap((r) => (r.countries ?? []).map((cc) => [cc, r.id])));
const regions = {};
for (const c of reg.cities) {
  const realm = c.realm && known.has(c.realm) ? c.realm : byCountry.get(c.country ?? "");
  if (realm) regions[c.id] = realm;
}
const outRegions = join(server, "trainworld_regions.json");
writeFileSync(outRegions, JSON.stringify(regions) + "\n");
console.log(`${Object.keys(regions).length} cities, ${new Set(Object.values(regions)).size} regions -> ${outRegions}`);
