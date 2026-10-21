// Zone display names and search aliases (T-825, docs/zones.md "Names"). The registry's names come
// from the pipeline (GHSL urban centres, NUTS units) and are rewritten by every join, so the names
// shown are worked out here: zone-names.json (hand-edited) gives a zone's name and extra aliases,
// zone-aliases.json (pipeline/zone_aliases.py) the biggest cities and towns in its place list, and
// a zone with no hand-set name loses GHSL's bracketed second city and NUTS's suffixes. The
// registry's own name and its parts always stay aliases.

import overrides from "./zone-names.json";
import generated from "./zone-aliases.json";

interface Override { name?: string; aliases?: string[] }
const HAND = overrides as Record<string, Override>;
const AUTO = generated as Record<string, string[]>;

/** The name a zone shows for its registry name, with no hand-set name: "Rotterdam [The Hague]"
 * Rotterdam, "Devon CC" Devon, "Halle (Saale)" Halle. */
export function cleanName(raw: string): string {
  const s = raw
    .replace(/\s*\[[^\]]*\]/g, "")
    .replace(/\s*\([^)]*\)\s*$/, "")
    .replace(/ CC\b/g, "")
    .replace(/\s+/g, " ")
    .trim();
  return s || raw;
}

/** The pieces of a compound registry name, each an alias: "Grevena, Kozani", "Lancaster and
 * Wyre", "Rotterdam [The Hague]". */
function parts(raw: string): string[] {
  return raw.split(/\s*(?:\[|\]|,|\/| - | and | & | e | y )\s*/).map((s) => s.trim()).filter((s) => s.length > 1);
}

export function zoneName(id: string, raw: string): string {
  return HAND[id]?.name ?? cleanName(raw);
}

/** Every other name a zone is found by, display name excluded, duplicates dropped (by `key`). */
export function zoneAliases(id: string, raw: string, key: (s: string) => string): string[] {
  const name = zoneName(id, raw);
  const seen = new Set([key(name)]);
  const out: string[] = [];
  for (const a of [...(HAND[id]?.aliases ?? []), raw, ...parts(raw), ...(AUTO[id] ?? [])]) {
    const k = key(a);
    if (!k || seen.has(k)) continue;
    seen.add(k);
    out.push(a);
  }
  return out;
}
