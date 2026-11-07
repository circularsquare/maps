// T-2681: registry city names in the languages that write them in their own script, from Wikidata
// (place-names.json, tools/place_names.mjs). Zones and the data's own names stay as they are.

import { language } from "./index";
import names from "./place-names.json";

type Entry = { q: string | null; ja?: string; "zh-Hans"?: string; "zh-TW"?: string };
const NAMES = names as Record<string, Entry>;

/** A registry city's name in the current language; `latin`, its own name, where there is none. */
export function tPlace(id: string, latin: string): string {
  const lang = language.value;
  if (lang !== "ja" && lang !== "zh-Hans" && lang !== "zh-TW") return latin;
  return NAMES[id]?.[lang] ?? latin;
}
