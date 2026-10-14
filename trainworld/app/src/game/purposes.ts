// Trip purposes as the map's purpose filter (T-158) and the purpose pie (T-143) name and colour
// them. Plain data: the demand workers import it too.
//
// Work is every city's local commutes (`DemandApi`), the six everyday purposes are each city's
// everyday layer's (`EverydayApi`, in its purpose order; every registered city since T-142), and longer trips are the regional
// market (`IntercityApi`), which has no purposes of its own.

export type PurposeKey = "work" | "school" | "university" | "shopping" | "social" | "airport" | "medical" | "longer";
/** The map's filter: one purpose, or every purpose together. */
export type PurposeFilter = PurposeKey | "all";

/** `EverydayApi`'s purpose kinds in its own order (sim/src/demand/everyday.rs `RATE`). */
export const EVERYDAY_KEYS: PurposeKey[] = ["shopping", "social", "school", "university", "airport", "medical"];
/** The same as the selection inspector names them (demand.worker.ts `EVERYDAY_PURPOSES`); the
 * inspector shows school and university as one (`shownName`). */
export const EVERYDAY_NAMES = ["Shopping and errands", "Social and leisure", "School", "University", "Airport", "Medical"];

/** The model's purposes the player sees as another one (T-192, Anita: merge school and
 * university). The model keeps them apart, as their homes, places and timing differ; the filter,
 * the purpose pie and split, the timing charts and every tooltip show them as one, "school". */
const SHOWN_AS: Partial<Record<PurposeKey, PurposeKey>> = { university: "school" };
/** The purpose a model purpose is shown as. */
export const shownKey = (key: PurposeKey): PurposeKey => SHOWN_AS[key] ?? key;
/** The model purposes a shown purpose counts: itself and any shown as it. */
export const membersOf = (key: PurposeKey): PurposeKey[] => [key, ...(Object.keys(SHOWN_AS) as PurposeKey[]).filter((k) => SHOWN_AS[k] === key)];
/** The inspector's name for a model purpose's row ("University" is shown as "School and university"). */
export const shownName = (name: string): string => (/^(school|university)/i.test(name) ? "School and university" : name);

/** Shown purposes, in the filter's order. */
export const PURPOSES: { key: PurposeKey; name: string }[] = [
  { key: "work", name: "Work" },
  { key: "school", name: "School and university" },
  { key: "shopping", name: "Shopping and errands" },
  { key: "social", name: "Social and leisure" },
  { key: "medical", name: "Medical" },
  { key: "airport", name: "Airport" },
  { key: "longer", name: "Inter-region trips" },
];

/** The purposes the map's filter offers (T-305, Anita): the inter-region trips are left out, so
 * they show under all purposes only. */
export const FILTER_PURPOSES = PURPOSES.filter((p) => p.key !== "longer");
/** A filter value as the filter can hold it: the inter-region trips fall back to all purposes. */
export const asFilter = (v: string): PurposeFilter => (FILTER_PURPOSES.some((p) => p.key === v) ? (v as PurposeKey) : "all");

export const purposeName = (key: PurposeFilter): string => (key === "all" ? "All purposes" : PURPOSES.find((p) => p.key === key)?.name ?? key);

/** "work", "shopping and errands": the purpose inside a sentence. */
export const purposeWords = (key: PurposeKey): string => purposeName(key).toLowerCase();

// Purpose colours are independent of the UI theme. Green is reserved for future events.
// School and university share school's blue-green since T-192; with university's amber gone the
// closest pair (CIEDE2000) is medical and social at 22, then shopping and medical 24, work and
// social or school 24 (T-142 had airport 19 from university).
const COLOUR: Record<PurposeKey, string> = {
  work: "#3978c6",
  school: "#168b91",
  university: "#168b91",
  social: "#9460b4",
  shopping: "#d94a53",
  airport: "#a8a032",
  medical: "#e98abd",
  longer: "#8a817b",
};

/** A purpose's colour by its key or by a name the inspector shows ("School and university"). */
export function purposeColour(name: string): string {
  if (name in COLOUR) return COLOUR[name as PurposeKey];
  if (/^work/i.test(name)) return COLOUR.work;
  if (/school|university/i.test(name)) return COLOUR.school;
  if (/social|leisure/i.test(name)) return COLOUR.social;
  if (/shopping|errands/i.test(name)) return COLOUR.shopping;
  if (/airport/i.test(name)) return COLOUR.airport;
  if (/medical/i.test(name)) return COLOUR.medical;
  if (/event/i.test(name)) return "#439447";
  return "#8a817b";
}
