// T-3269, T-3279: the flag a player can show after their name on the leaderboards and Community cards, picked in the
// account panel (ui/Flag.tsx). The list is flags.json, which tools/flags.mjs also turns into the pictures (public/flags/,
// 4:3) and the server's list of codes it accepts. Pictures, never emoji: Windows draws flag emoji as letters.

import flagsJson from "./flags.json";
import { accountConfig, accountState, api, boardSetting, boards, type BoardSetting } from "./account";
import { language, t, tCountry } from "../i18n";

interface Place { code: string; country?: string; set?: number; name?: string; ja?: string; "zh-Hans"?: string; "zh-TW"?: string }
const LIST = flagsJson as { countries: string[]; places: Place[]; more: { code: string }[] };
export const FLAG_COUNTRIES: readonly string[] = LIST.countries;
const PLACES = new Map(LIST.places.map((p) => [p.code, p]));
const KNOWN = new Set([...LIST.countries, ...PLACES.keys(), ...LIST.more.map((m) => m.code)]);

/** A code this page has a picture for; anything else (a newer server's) shows nothing. */
export const isFlag = (code: string | undefined | null): code is string => !!code && KNOWN.has(code);

/** The flag list version the server accepts (config `trainworldFlagSet`; a server from before it, 1). */
const serverSet = () => accountConfig.value?.trainworldFlagSet ?? 1;
/** Whether the server would take this code: newer places stay out of the picker until it is deployed. */
const offered = (code: string) => (PLACES.get(code)?.set ?? 1) <= serverSet();
/** The picker's three groups, each in list order: countries, other places (cities, regions, the EU), the rest. */
export const flagGroups = () => ({
  countries: LIST.countries,
  places: LIST.places.map((p) => p.code).filter(offered),
  more: LIST.more.map((m) => m.code),
});

const MORE_NAMES: Record<string, () => string> = {
  train: () => t("flags.train"), earth: () => t("flags.earth"), rainbow: () => t("flags.rainbow"), trans: () => t("flags.trans"),
  pirate: () => t("flags.pirate"), chequered: () => t("flags.chequered"), amsterdam: () => t("flags.amsterdam"),
  chicago: () => t("flags.chicago"), washington: () => t("flags.washington"), nyc: () => t("flags.nyc"),
};

/** The flag's name in the player's language: a country's (and the EU's and UN's) from the browser, the game's few from
 * its strings, a prefecture's from flags.json (Latin, or its own in Japanese and Chinese, as the game names cities). */
export function flagName(code: string): string {
  const own = MORE_NAMES[code];
  if (own) return own();
  const p = PLACES.get(code);
  if (p?.name) { const lang = language.value; return (lang === "ja" || lang === "zh-Hans" || lang === "zh-TW" ? p[lang] : undefined) ?? p.name; }
  return tCountry(code.toUpperCase());
}

/** The country a place's flag is in (for the picker's grey note and its search), or "" */
export const flagCountry = (code: string) => { const cc = PLACES.get(code)?.country; return cc ? tCountry(cc.toUpperCase()) : ""; };

/** Relative, like the rest of public/: the game is served under a folder. */
export const flagUrl = (code: string) => `flags/${code}.svg`;

/** Whether the server keeps flags (`/noritetsu/config`'s `trainworldFlags`); off, the picker is hidden. */
export const flagsOn = () => accountState.value === "on" && accountConfig.value?.trainworldFlags === true;

/** The country the player is probably in: the browser's time zone (Asia/Tokyo is Japan), else the region of its first
 * language that has one (en-GB). Time zone first, since many people anywhere keep an en-US browser. */
export async function homeCountry(): Promise<string> {
  try {
    const zone = Intl.DateTimeFormat().resolvedOptions().timeZone;
    const table = (await import("./tzCountry.json")).default as Record<string, string>;
    if (zone && table[zone]) return table[zone];
  } catch { /* no time zone: the languages below */ }
  for (const l of navigator.languages?.length ? navigator.languages : [navigator.language ?? ""]) {
    const region = l.split(/[-_]/).slice(1).find((p) => /^[A-Za-z]{2}$/.test(p))?.toLowerCase();
    if (region && LIST.countries.includes(region)) return region;
  }
  return "";
}

export async function setFlag(code: string) {
  boardSetting.value = (await api<{ board: BoardSetting }>("PUT", "trainworld/flag", { flag: code })).board;
  boards.value = { ...boards.peek(), at: 0 }; // the boards fetch again when next shown, with the new flag
}
