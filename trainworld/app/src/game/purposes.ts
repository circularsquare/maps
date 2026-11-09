import { t } from "../i18n";
// Trip purposes as the map's purpose filter (T-158) and the purpose pie (T-143) name and colour
// them. Plain data: the demand workers import it too.
//
// Work is every city's local commutes (`DemandApi`), the seven everyday purposes are each city's
// everyday layer's (`EverydayApi`, in its purpose order; every registered city since T-142), and longer trips are the regional
// market (`IntercityApi`), which has no purposes of its own.

export type PurposeKey = "work" | "school" | "university" | "shopping" | "social" | "airport" | "medical" | "attractions" | "longer";
/** The map's filter: one purpose, or every purpose together. */
export type PurposeFilter = PurposeKey | "all";
/** The purposes a bubble is coloured by under the Purpose scheme (T-2886): work and school, errands and leisure, and the rest. */
export const PURPOSE_CLASS: Partial<Record<PurposeFilter, "work" | "errands">> = { work: "work", school: "work", university: "work", shopping: "errands", social: "errands" };

/** `EverydayApi`'s purpose kinds in its own order (sim/src/demand/everyday.rs `RATE`). */
export const EVERYDAY_KEYS: PurposeKey[] = ["shopping", "social", "school", "university", "airport", "medical", "attractions"];
/** The same as the selection inspector names them (demand.worker.ts `EVERYDAY_PURPOSES`); the
 * inspector shows school and university as one (`shownName`). */
export const EVERYDAY_NAMES = ["Errands", "Leisure", "School", "University", "Airports", "Medical", "Attractions"]; // i18n-ignore: ids the demand worker sends; shown through shownName
/** A purpose row's name from the demand worker (an English id from EVERYDAY_NAMES, or "Work") in the current language; any other text as it is. */
export function purposeLabel(name: string): string {
  switch (name) {
    case "Errands": return t("purposes.errands");
    case "Leisure": return t("purposes.leisure");
    case "School": return t("purposes.school");
    case "University": return t("purposes.university");
    case "Airports": return t("purposes.airports");
    case "Medical": return t("purposes.medical");
    case "Attractions": return t("purposes.attractions");
    case "Work": return t("purposes.work");
    default: return name;
  }
}

/** The model's purposes the player sees as another one (T-192, Anita: merge school and
 * university). The model keeps them apart, as their homes, places and timing differ; the filter,
 * the purpose pie and split, the timing charts and every tooltip show them as one, "school". */
const SHOWN_AS: Partial<Record<PurposeKey, PurposeKey>> = { university: "school" };
/** The purpose a model purpose is shown as. */
export const shownKey = (key: PurposeKey): PurposeKey => SHOWN_AS[key] ?? key;
/** The model purposes a shown purpose counts: itself and any shown as it. */
export const membersOf = (key: PurposeKey): PurposeKey[] => [key, ...(Object.keys(SHOWN_AS) as PurposeKey[]).filter((k) => SHOWN_AS[k] === key)];
/** The inspector's name for a model purpose's row ("University" is shown as "School"). */
export const shownName = (name: string): string => (/^(school|university)/i.test(name) ? t("purposes.school") : purposeLabel(name));

/** Shown purposes, in the filter's order. */
export const PURPOSES: { key: PurposeKey; name: string }[] = [
  { key: "work", get name() { return t("purposes.work"); } },
  { key: "school", get name() { return t("purposes.school"); } },
  { key: "shopping", get name() { return t("purposes.errands"); } },
  { key: "social", get name() { return t("purposes.leisure"); } },
  { key: "medical", get name() { return t("purposes.medical"); } },
  { key: "airport", get name() { return t("purposes.airports"); } },
  { key: "attractions", get name() { return t("purposes.attractions"); } },
  { key: "longer", get name() { return t("purposes.intercity"); } },
];

/** The purposes the map's filter offers (T-305, Anita): the inter-city trips are left out, so
 * they show under all purposes only. */
export const FILTER_PURPOSES = PURPOSES.filter((p) => p.key !== "longer");
/** A filter value as the filter can hold it: the inter-city trips fall back to all purposes. */
export const asFilter = (v: string): PurposeFilter => (FILTER_PURPOSES.some((p) => p.key === v) ? (v as PurposeKey) : "all");

export const purposeName = (key: PurposeFilter): string => (key === "all" ? t("purposes.all") : PURPOSES.find((p) => p.key === key)?.name ?? key);

/** "work", "errands", "airport": the purpose inside a sentence, before "trips". */
export const purposeWords = (key: PurposeKey): string => (key === "airport" ? t("purposes.airportWord") : purposeName(key).toLowerCase());

// Purpose colours are independent of the UI theme. Green is reserved for future events.
// School and university (shown as "School") share school's blue-green since T-192; with university's amber gone the
// closest pair (CIEDE2000) is medical and social at 22, then shopping and medical 24, work and
// social or school 24 (T-142 had airport 19 from university). Attractions' orange (T-263) is 24
// from airport and inter-city trips, 28 from shopping.
const COLOUR: Record<PurposeKey, string> = {
  work: "#3978c6",
  school: "#168b91",
  university: "#168b91",
  social: "#9460b4",
  shopping: "#d94a53",
  airport: "#a8a032",
  medical: "#e98abd",
  attractions: "#d9822b",
  longer: "#8a817b",
};

/** A purpose's colour by its key or by a name the inspector shows ("School"). */
export function purposeColour(name: string): string {
  if (name in COLOUR) return COLOUR[name as PurposeKey];
  if (/^work/i.test(name)) return COLOUR.work;
  if (/school|university/i.test(name)) return COLOUR.school;
  if (/social|leisure/i.test(name)) return COLOUR.social;
  if (/shopping|errands/i.test(name)) return COLOUR.shopping;
  if (/airport/i.test(name)) return COLOUR.airport;
  if (/medical/i.test(name)) return COLOUR.medical;
  if (/attraction/i.test(name)) return COLOUR.attractions;
  if (/event/i.test(name)) return "#439447";
  return "#8a817b";
}
