// T-3450: fails when a registry city's or zone's Wikidata item (i18n/place-names.json, place-names-zones.json) is not that
// place: far from it, named otherwise than the game shows it, not a place, a former country, or a town of the same name
// inside the unit the place is (Shiga's old town 志賀町 for Shiga Prefecture). The rules: tools/place_names_rules.mjs vet().
// A fault is fixed with `place_names.mjs --redo <ids>` (with --zones for zones), which applies the same rules.
//   node trainworld/checks/place_names_match.mjs [--fresh]
// Asks Wikidata's SPARQL endpoint about every item (about 12 minutes); the answers are kept a week in the temp folder.
import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { factsOf, unitLabels, vet, cleanName, PINNED, HAND_LABELS } from "../tools/place_names_rules.mjs";

const SRC = join(dirname(fileURLToPath(import.meta.url)), "..", "app", "src");
const CACHE = join(tmpdir(), "trainworld-place-facts.json");
const WEEK = 7 * 86400e3;
/** Matches a rule faults that were looked at and kept, with why. */
const ACCEPT = {};

const read = (p) => JSON.parse(readFileSync(join(SRC, p), "utf8"));
const registry = read("game/registry.json");
const hand = read("game/zone-names.json");
const byId = new Map(registry.cities.map((c) => [c.id, c]));
const files = { "place-names.json": read("i18n/place-names.json"), "place-names-zones.json": read("i18n/place-names-zones.json") };

const places = [];
for (const [file, data] of Object.entries(files))
  for (const [id, e] of Object.entries(data)) {
    const r = byId.get(id);
    if (!e.q || !r || PINNED[id] || HAND_LABELS[id]) continue;
    places.push({ file, id, q: e.q, e, c: { id, name: hand[id]?.name ?? cleanName(r.name), raw: r.name, center: r.center, box: r.box, country: r.country } });
  }

const fresh = process.argv.includes("--fresh");
const cache = !fresh && existsSync(CACHE) ? JSON.parse(readFileSync(CACHE, "utf8")) : { at: 0 };
let facts, units;
if (Date.now() - cache.at < WEEK && places.every((p) => p.q in cache.facts)) {
  facts = new Map(Object.entries(cache.facts));
  units = new Map(Object.entries(cache.units));
}
if (!facts || places.some((p) => !units.has("_" + p.id + p.q))) {
  const ids = [...new Set(places.map((p) => p.q))];
  facts = await factsOf(ids, (s) => process.stderr.write(s + "\r"));
  units = await unitLabels(facts, places.map((p) => [p.q, p.c]), (s) => process.stderr.write(s + "\r"));
  for (const p of places) units.set("_" + p.id + p.q, 1); // which place the unit labels were gathered for
  writeFileSync(CACHE, JSON.stringify({ at: Date.now(), facts: Object.fromEntries(facts), units: Object.fromEntries(units) }));
  process.stderr.write("\n");
}

let failed = 0, accepted = 0;
for (const p of places) {
  const faults = vet(p.q, p.c, p.e, facts.get(p.q), units);
  if (!faults.length) continue;
  if (ACCEPT[p.id]) { accepted++; continue; }
  failed++;
  const label = [p.e.ja, p.e["zh-Hans"]].filter(Boolean).join(" / ");
  console.log(`FAIL ${p.file} ${p.id} "${p.c.name}" ${p.q} ${label}: ${faults.map((f) => f.fault + (f.detail ? ` (${f.detail})` : "")).join("; ")}`);
}
console.log(`${places.length} matches checked: ${failed} wrong, ${accepted} accepted by hand`);
process.exit(failed ? 1 : 0);
