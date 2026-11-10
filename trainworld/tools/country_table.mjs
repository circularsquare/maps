// T-3116: writes shonei-server's trainworld_countries.json ({city id: lowercase ISO country}) from the
// registry, so the server can sum a game's per-city points into per-country boards. Rerun it, and
// redeploy the server, when a country's cities are added. Usage (maps root):
//   node trainworld/tools/country_table.mjs [path to shonei-server]
import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { join } from "node:path";

const here = fileURLToPath(new URL(".", import.meta.url));
const reg = JSON.parse(readFileSync(join(here, "../app/src/game/registry.json"), "utf8"));
const server = process.argv[2] ?? "C:/Users/anita/projects/shonei-server";
const table = {};
for (const c of reg.cities) if (c.country) table[c.id] = c.country.toLowerCase();
const out = join(server, "trainworld_countries.json");
writeFileSync(out, JSON.stringify(table) + "\n");
console.log(`${Object.keys(table).length} cities, ${new Set(Object.values(table)).size} countries -> ${out}`);
