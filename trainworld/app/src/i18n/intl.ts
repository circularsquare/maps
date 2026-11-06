// Number, list, date and name formatting by language (T-2583). No signals or DOM, so a worker or a Node
// check may import it; the current language's versions are in ./index.ts.

/** The game's languages, as BCP 47 tags; `en` is the source and the fallback (docs/localization.md).
 * Traditional Chinese is Taiwan's (zh-TW: 捷運, 公車), Simplified the mainland's. */
export const LANGS = ["en", "ja", "zh-Hans", "zh-TW", "es", "de"] as const;
export type Lang = (typeof LANGS)[number];
export const isLang = (s: string): s is Lang => (LANGS as readonly string[]).includes(s);

/** The Intl locale for a language. English numbers are en-US's, as the game always wrote them; English
 * lists are en-GB's, "a, b and c" with no comma before the "and". */
function locale(lang: Lang, purpose: "number" | "list" | "date"): string {
  if (lang === "en") return purpose === "list" ? "en-GB" : "en-US";
  return lang;
}

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

/** A language's own name for itself ("日本語", "简体中文"), for the language picker. */
export function languageName(lang: Lang): string {
  const f = cached(`g|${lang}`, () => new Intl.DisplayNames([lang], { type: "language" }));
  return f.of(lang) ?? lang;
}
