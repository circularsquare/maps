// T-2681: registry city names in the languages that write them in their own script, from Wikidata
// (place-names.json, tools/place_names.mjs). T-3318: the zones' too (place-names-zones.json, `--zones`), a
// separate file fetched the first time a Japanese or Chinese name is asked for. The data's own names stay as they are.

import { signal } from "@preact/signals";
import { language } from "./index";
import names from "./place-names.json";

type Entry = { q: string | null; ja?: string; "zh-Hans"?: string; "zh-TW"?: string };
const NAMES = names as Record<string, Entry>;

const zoneNames = signal<Record<string, Entry> | null>(null);
let zonesAsked = false;
function askZones() {
  if (zonesAsked) return;
  zonesAsked = true;
  import("./place-names-zones.json").then((m) => { zoneNames.value = m.default as Record<string, Entry>; }).catch(() => { /* Latin names stay */ });
}

/** A registry city's or zone's name in the current language; `latin`, its own name, where there is none. */
export function tPlace(id: string, latin: string): string {
  const lang = language.value;
  if (lang !== "ja" && lang !== "zh-Hans" && lang !== "zh-TW") return latin;
  const own = NAMES[id];
  if (own) return own[lang] ?? latin;
  askZones();
  return zoneNames.value?.[id]?.[lang] ?? latin;
}
