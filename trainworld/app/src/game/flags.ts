// T-3269: the flag a player can show beside their name on the leaderboards and Community cards, picked in the account
// panel (ui/Flag.tsx). The list is flags.json, which tools/flags.mjs also turns into the pictures (public/flags/, from
// circle-flags, MIT) and the server's list of codes it accepts. Pictures, never emoji: Windows draws flag emoji as letters.

import flagsJson from "./flags.json";
import { accountConfig, accountState, api, boardSetting, boards, type BoardSetting } from "./account";
import { t, tCountry } from "../i18n";

const LIST = flagsJson as { more: { code: string }[]; countries: string[] };
export const FLAG_MORE: readonly string[] = LIST.more.map((m) => m.code);
export const FLAG_COUNTRIES: readonly string[] = LIST.countries;
const KNOWN = new Set([...FLAG_MORE, ...FLAG_COUNTRIES]);

/** A code this page has a picture for; anything else (a newer server's) shows nothing. */
export const isFlag = (code: string | undefined | null): code is string => !!code && KNOWN.has(code);

const MORE_NAMES: Record<string, () => string> = {
  train: () => t("flags.train"), earth: () => t("flags.earth"), rainbow: () => t("flags.rainbow"), trans: () => t("flags.trans"),
  pirate: () => t("flags.pirate"), chequered: () => t("flags.chequered"), amsterdam: () => t("flags.amsterdam"),
  chicago: () => t("flags.chicago"), washington: () => t("flags.washington"),
};

/** The flag's name in the player's language: a country's (and the EU's and UN's) from the browser, the rest from the
 * game's strings. */
export function flagName(code: string): string {
  return MORE_NAMES[code]?.() ?? tCountry(code.toUpperCase());
}

/** Relative, like the rest of public/: the game is served under a folder. */
export const flagUrl = (code: string) => `flags/${code}.svg`;

/** Whether the server keeps flags (`/noritetsu/config`'s `trainworldFlags`); off, the picker is hidden. */
export const flagsOn = () => accountState.value === "on" && accountConfig.value?.trainworldFlags === true;

export async function setFlag(code: string) {
  boardSetting.value = (await api<{ board: BoardSetting }>("PUT", "trainworld/flag", { flag: code })).board;
  boards.value = { ...boards.peek(), at: 0 }; // the boards fetch again when next shown, with the new flag
}
