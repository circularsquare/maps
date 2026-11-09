// The player-facing text by language (T-2583; conventions in docs/localization.md). Messages are ICU
// MessageFormat strings in ./messages/<lang>.json, English bundled and the others fetched when chosen.
// Read through `t` while rendering, so a panel redraws when the language or its messages arrive.

import { signal } from "@preact/signals";
import type { ComponentChild, ComponentChildren } from "preact";
import IntlMessageFormat from "intl-messageformat";
import en from "./messages/en.json";
import { countryName, formatCompact, formatDate, formatDuration, formatList, formatNumber, formatSeries, formatSpaced, isLang, langOfBrowser, OFFERED, PICKER, type Lang } from "./intl";

export { LANGS, LANG_NAMES, isLang, type Lang } from "./intl";

export type MessageKey = keyof typeof en;
type Primitive = string | number | boolean | null | undefined;
export type Values = Record<string, Primitive>;
export type RichValues = Record<string, Primitive | ComponentChild>;

/** The language shown: `?lang=`, else the player's pick on this device, else the browser's if offered. */
export const language = signal<Lang>("en");
/** `?lang=pseudo`: every message in ⟦ ⟧ over English, so text still written inline shows on screen. */
const pseudo = signal(false);

type Catalog = Partial<Record<MessageKey, string>>;
const catalogs = new Map<Lang, Catalog>([["en", en]]);
const loaders = import.meta.glob<Catalog>(["./messages/*.json", "!./messages/en.json"], { import: "default" });

/** Fetches a language's messages if needed, then shows it. */
export async function setLanguage(lang: Lang): Promise<void> {
  if (!catalogs.has(lang)) {
    const load = loaders[`./messages/${lang}.json`];
    if (!load) return;
    catalogs.set(lang, await load());
  }
  language.value = lang;
  if (typeof document !== "undefined") {
    document.documentElement.lang = lang;
    // the page font follows <html lang> (style.css), so canvas text measured in the old one is measured again
    void document.fonts?.ready.then(() => document.dispatchEvent(new Event("tw-fontchange")));
  }
}

// a message with no argument and no quote is its own output, so most never reach the parser
const compiled = new Map<string, IntlMessageFormat | string>();
function compile(lang: Lang, key: MessageKey, text: string): IntlMessageFormat | string {
  const id = lang + "\u0000" + key;
  let c = compiled.get(id);
  if (c === undefined) {
    // ignoreTag: "<" is text; rich pieces such as a chip go in as values
    c = /[{}']/.test(text) ? new IntlMessageFormat(text, lang === "en" ? "en-US" : lang, undefined, { ignoreTag: true }) : text;
    compiled.set(id, c);
  }
  return c;
}

const warned = new Set<string>();
function render(key: MessageKey, values: RichValues | undefined): string | ComponentChild | ComponentChild[] {
  const lang = language.value;
  const vals = values && Object.fromEntries(Object.entries(values).map(([k, v]) => [k, typeof v === "boolean" ? String(v) : v]));
  for (const l of lang === "en" ? (["en"] as const) : [lang, "en" as const]) {
    const text = catalogs.get(l)?.[key];
    if (text === undefined) continue;
    try {
      const c = compile(l, key, text);
      return typeof c === "string" ? c : (c.format<ComponentChild>(vals as Record<string, ComponentChild>) as string | ComponentChild | ComponentChild[]);
    } catch (e) {
      // a broken translation falls back to English; checks/i18n.mjs catches it before it lands
      if (!warned.has(l + key)) console.warn(`i18n: ${l} ${key}:`, e), warned.add(l + key);
    }
  }
  return key;
}

/** A translated word set inside a sentence, lower-cased. German keeps its capitals: every noun is written with one. */
export const lowerWord = (s: string): string => (language.value === "de" ? s : s.toLowerCase());

/** The message `key` in the current language with its placeholders filled; English where the
 * language lacks it. Values are already formatted text, except a plural's count. */
export function t(key: MessageKey, values?: Values): string {
  const out = render(key, values);
  const s = Array.isArray(out) ? out.join("") : String(out);
  return pseudo.value ? `⟦${s}⟧` : s;
}

/** As `t`, with Preact nodes (a chip, a bold figure) allowed as values: for JSX children. */
export function tRich(key: MessageKey, values: RichValues): ComponentChildren {
  const out = render(key, values);
  return pseudo.value ? ["⟦", out, "⟧"] : out;
}

// The formatting helpers in the current language (./intl.ts takes the language as its first argument).
export const tNumber = (n: number, opts?: Intl.NumberFormatOptions) => formatNumber(language.value, n, opts);
export const tList = (items: string[], type?: "and" | "or") => formatList(language.value, items, type);
export const tSeries = (items: string[]) => formatSeries(language.value, items);
export const tSpaced = (phrase: string) => formatSpaced(language.value, phrase);
export const tDate = (d: Date | number, opts?: Intl.DateTimeFormatOptions) => formatDate(language.value, d, opts);
export const tCountry = (code: string) => countryName(language.value, code);
export const tCompact = (v: number, sig: number) => formatCompact(language.value, v, sig);
export const tDuration = (s: number) => formatDuration(language.value, s);

const STORE = "trainworld-lang";

/** The Language row's choices: the offered languages, or every one being translated while `?lang=` is in
 * the address (a preview, so translations can be looked at before they are offered). */
export const pickable: readonly Lang[] = typeof location !== "undefined" && new URLSearchParams(location.search).has("lang") ? PICKER : OFFERED;

/** The player's pick in the Language row, remembered on this device. */
export function chooseLanguage(lang: Lang): Promise<void> {
  try {
    localStorage.setItem(STORE, lang);
  } catch {
    // storage refused (private mode): the pick lasts until the page closes
  }
  return setLanguage(lang);
}

/** The language a page opens in; workers (no document) are passed theirs with each job. */
function startLanguage(): Lang {
  if (typeof document === "undefined") return "en";
  const q = new URLSearchParams(location.search).get("lang");
  let saved: string | null = null;
  try {
    saved = localStorage.getItem(STORE);
  } catch {
    // no storage: the browser's language decides
  }
  if (q === "pseudo") {
    pseudo.value = true;
    return "en";
  }
  if (q && isLang(q)) return q;
  if (saved && isLang(saved)) return saved;
  return langOfBrowser(typeof navigator !== "undefined" ? (navigator.languages ?? [navigator.language]) : []);
}

// T-2672: every module that imports this one waits for the language's messages, so the first render is
// already in it; English is bundled and waits for nothing. A failed fetch opens in English.
const first = startLanguage();
if (first !== "en") await setLanguage(first).catch((e) => console.warn(`i18n: ${first} messages did not load:`, e));
