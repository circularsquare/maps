// Number, list, date and name formatting by language (T-2583). No signals or DOM, so a worker or a Node
// check may import it; the current language's versions are in ./index.ts.

/** The game's languages, as BCP 47 tags; `en` is the source and the fallback (docs/localization.md).
 * Traditional Chinese is Taiwan's (zh-TW: 捷運, 公車), Simplified the mainland's. */
export const LANGS = ["en", "ja", "zh-Hans", "zh-TW", "es", "de"] as const;
export type Lang = (typeof LANGS)[number];
export const isLang = (s: string): s is Lang => (LANGS as readonly string[]).includes(s);

/** The languages a player is offered: the Language row's list, and what a first visit may pick from the
 * browser. A language joins once its translation is complete and reviewed (docs/localization.md
 * "Releasing a language"); `?lang=<code>` opens any language and lists all of `PICKER` (T-2636). */
export const OFFERED: readonly Lang[] = ["en"];
/** The Language row's list when previewing, in this order. */
export const PICKER: readonly Lang[] = ["en", "ja", "zh-Hans", "zh-TW"];

/** Each language's name in itself, as the Language row lists it; never translated. */
export const LANG_NAMES: Record<Lang, string> = { en: "English", ja: "日本語", "zh-Hans": "简体中文", "zh-TW": "繁體中文 (台灣)", es: "Español", de: "Deutsch" };

/** The game's language for a BCP 47 tag from the browser ("zh-HK" -> zh-TW, "zh-CN" -> zh-Hans), or null. */
export function langOfTag(tag: string): Lang | null {
  const [base, ...rest] = tag.toLowerCase().split(/[-_]/);
  if (base === "zh") return rest.some((s) => s === "hant" || s === "tw" || s === "hk" || s === "mo") ? "zh-TW" : "zh-Hans";
  return base && isLang(base) ? base : null;
}

/** The first of the browser's languages that is offered, else English. An offered English ahead of
 * another language wins, as the player ranked it. */
export function langOfBrowser(tags: readonly string[], offered: readonly Lang[] = OFFERED): Lang {
  for (const tag of tags) {
    const l = langOfTag(tag);
    if (l && offered.includes(l)) return l;
  }
  return "en";
}

/** English dates and times as the browser's first English locale writes them (en-GB "6 Nov, 14:05"),
 * else en-US's (T-2655). */
const EN_DATES = (() => {
  const tags = typeof navigator !== "undefined" ? (navigator.languages ?? [navigator.language]) : [];
  try {
    return Intl.DateTimeFormat.supportedLocalesOf(tags.filter((t) => /^en(-|$)/i.test(t)))[0] ?? "en-US";
  } catch {
    return "en-US";
  }
})();

/** The Intl locale for a language. English numbers are en-US's, as the game always wrote them; English
 * lists are en-GB's, "a, b and c" with no comma before the "and". */
function locale(lang: Lang, purpose: "number" | "list" | "date"): string {
  if (lang === "en") return purpose === "list" ? "en-GB" : purpose === "date" ? EN_DATES : "en-US";
  return lang;
}

/** Whether a language puts spaces between words. Japanese and Chinese do not: the page adds the small
 * gap beside Latin letters and digits (docs/localization.md "Fonts"). */
const SPACED: Record<Lang, boolean> = { en: true, ja: false, "zh-Hans": false, "zh-TW": false, es: true, de: true };

/** A phrase a message puts straight after a word ("Trips per day{where}"): " in Osaka" in English,
 * "大阪" in Japanese; empty stays empty. */
export const formatSpaced = (lang: Lang, phrase: string): string => (phrase && SPACED[lang] ? " " + phrase : phrase);

/** Items in a row with no "and" before the last: "5 track, 2 stations", "線路5区間、2駅". */
export const formatSeries = (lang: Lang, items: string[]): string => items.join(SPACED[lang] ? ", " : "、");

// one formatter per language and options: making one costs 5-10 µs (T-1165)
const cache = new Map<string, Intl.NumberFormat | Intl.ListFormat | Intl.DateTimeFormat | Intl.DisplayNames>();
function cached<T extends Intl.NumberFormat | Intl.ListFormat | Intl.DateTimeFormat | Intl.DisplayNames>(key: string, make: () => T): T {
  let f = cache.get(key) as T | undefined;
  if (!f) cache.set(key, (f = make()));
  return f;
}

export function numberFormat(lang: Lang, opts: Intl.NumberFormatOptions = {}): Intl.NumberFormat {
  const loc = locale(lang, "number");
  return cached(`n|${loc}|${JSON.stringify(opts)}`, () => new Intl.NumberFormat(loc, opts));
}

/** 1234567.4 -> "1,234,567.4" in English, "1.234.567,4" in German; up to 3 decimals unless `opts` says. */
export const formatNumber = (lang: Lang, n: number, opts?: Intl.NumberFormatOptions): string => numberFormat(lang, opts).format(n);

/** ["a", "b", "c"] -> "a, b and c", "a、b、c", "a、b和c"; `or` for "a, b or c". */
export function formatList(lang: Lang, items: string[], type: "and" | "or" = "and"): string {
  const loc = locale(lang, "list");
  const f = cached(`l|${loc}|${type}`, () => new Intl.ListFormat(loc, { type: type === "and" ? "conjunction" : "disjunction" }));
  return f.format(items);
}

export function formatDate(lang: Lang, d: Date | number, opts: Intl.DateTimeFormatOptions = {}): string {
  const loc = locale(lang, "date");
  return cached(`d|${loc}|${JSON.stringify(opts)}`, () => new Intl.DateTimeFormat(loc, opts)).format(d);
}

/** A country's name from its ISO code ("JP" -> "Japan", "日本"); the code itself if Intl has none. */
export function countryName(lang: Lang, code: string): string {
  const f = cached(`r|${lang}`, () => new Intl.DisplayNames([locale(lang, "date")], { type: "region" }));
  try {
    return f.of(code) ?? code;
  } catch {
    return code;
  }
}
