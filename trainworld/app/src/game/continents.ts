// T-2124: each country's continent, for the Bicontinental achievement. The UN's M49 regions, with the
// Americas split at Panama (Central America and the Caribbean go with North America), Cyprus and the
// Caucasus in Asia as M49 has them, and the four countries that cross a line split by position:
// Russia at the Urals (60°E), Kazakhstan at the Ural river (51.5°E), Egypt at the Suez Canal and its
// gulf (Sinai is Asia), Turkey at the Bosporus, the Sea of Marmara and the Dardanelles (Thrace is
// Europe).

export type Continent = "Africa" | "Asia" | "Europe" | "North America" | "South America" | "Oceania";

const TABLE: Record<Continent, string> = {
  Africa: "DZ AO BJ BW BF BI CV CM CF TD KM CG CD CI DJ EG GQ ER SZ ET GA GM GH GN GW KE LS LR LY MG MW ML MR MU YT MA MZ NA NE NG RE RW SH ST SN SC SL SO ZA SS SD TZ TG TN UG EH ZM ZW IO",
  Asia: "AF AM AZ BH BD BT BN KH CN CY GE HK IN ID IR IQ IL JO JP KZ KW KG LA LB MO MY MV MN MM NP KP OM PK PS PH QA SA SG KR LK SY TW TJ TH TL TR TM AE UZ VN YE",
  Europe: "AX AL AD AT BY BE BA BG HR CZ DK EE FO FI FR DE GI GR GG VA HU IS IE IM IT JE XK LV LI LT LU MT MD MC ME NL MK NO PL PT RO RU SM RS SK SI ES SJ SE CH UA GB",
  "North America": "AI AG AW BS BB BZ BM BQ VG CA KY CR CU CW DM DO SV GL GD GP GT HT HN JM MQ MX MS NI PA PR BL KN LC MF PM VC SX TT TC US VI",
  "South America": "AR BO BR CL CO EC FK GF GY PY PE SR UY VE",
  Oceania: "AS AU CK FJ PF GU KI MH FM NR NC NZ NU NF MP PW PG PN WS SB TK TO TV VU WF",
};
const OF = new Map<string, Continent>();
for (const [c, list] of Object.entries(TABLE) as [Continent, string][]) for (const cc of list.split(" ")) OF.set(cc, c);

/** Thrace, west of the Bosporus (28.99°E, 41.01°N to 29.12°E, 41.22°N), the Sea of Marmara's north
 * shore and the Dardanelles (26.2°E, 40.03°N to 26.68°E, 40.4°N). */
function inThrace(lng: number, lat: number): boolean {
  if (lat >= 40.9) return lng < 28.99 + 0.62 * (lat - 41.01);
  if (lat >= 40.42) return lng < 27.3;
  return lng < 26.2 + 1.3 * (lat - 40.03);
}

/** The continent of a point in country `cc` (ISO 3166 alpha-2); null for a country not listed. */
export function continentOf(cc: string, lng: number, lat: number): Continent | null {
  if (cc === "RU") return lng > 60 ? "Asia" : "Europe";
  if (cc === "KZ") return lng < 51.5 ? "Europe" : "Asia";
  if (cc === "EG") return lng > 32.34 && (lat >= 29.95 || lng > 32.55 + 0.55 * (29.97 - lat)) ? "Asia" : "Africa";
  if (cc === "TR") return inThrace(lng, lat) ? "Europe" : "Asia";
  return OF.get(cc) ?? null;
}
