// T-3631: a new station's name in the player's script for Japanese, Chinese and Korean: a place's or
// street's own name where every letter of it is in a script the language reads (kanji and kana for
// Japanese; Han for Chinese, as the name writes it, Simplified or Traditional; Hangul or Han for Korean),
// in the form a station takes (世田谷 for 世田谷区, 浦东 for 浦东新区). Every other language, and a name
// in a script the language doesn't read, keeps the Latin name. Rules: docs/track-and-service.md "In the player's script".

import type { Lang } from "./intl";

export type NativeLang = "ja" | "zh-Hans" | "zh-TW" | "ko";
export const nativeLang = (l: Lang): NativeLang | null => (l === "ja" || l === "zh-Hans" || l === "zh-TW" || l === "ko" ? l : null);

const READS: Record<NativeLang, RegExp> = {
  ja: /^[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}ー・々〆ヶヵ0-9０-９]+$/u,
  "zh-Hans": /^[\p{Script=Han}〇·・0-9]+$/u,
  "zh-TW": /^[\p{Script=Han}〇·・0-9]+$/u,
  ko: /^[\p{Script=Hangul}\p{Script=Han}0-9]+(?: [\p{Script=Hangul}\p{Script=Han}0-9]+)*$/u,
};
const LETTER = /[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Hangul}]/u;

/** The name or the first part of it ("香港 Hong Kong", "台北縣 (Taipei County)") the language reads, or null. */
export function readable(s: string | undefined, lang: NativeLang): string | null {
  if (!s) return null;
  const t = s.replace(/\s*[(（][^)）]*[)）]/g, "").trim();
  for (const c of [t, ...t.split(/[\s/;]+/)]) if (c && LETTER.test(c) && READS[lang].test(c)) return c;
  return null;
}

/** A stop's own label rather than a place ("渋谷駅", "上海站", "강남역"; not 구역, a district). */
export const nativeStation = (s: string) => /(駅|驛|站)$/.test(s) || (/역$/.test(s) && !/(구|지|권)역$/.test(s));

const KANJI_NUM = "[一二三四五六七八九十〇0-9０-９]+";
/** Suffixes a station leaves off, by the country whose name it is, for the place kinds where it is the
 * administrative unit's (町 on a town, never on 大手町, a neighbourhood). One is taken off, and only when
 * at least two characters stay. */
const SUFFIX: Record<"ja" | "zh" | "ko", [RegExp, Set<string> | null, number?][]> = {
  ja: [
    [/[都府県]$/, new Set(["city"])],
    [/市$/, new Set(["city", "town"])],
    [/区$/, new Set(["city", "borough", "suburb"])],
    [/町$/, new Set(["town", "suburb"])],
    [/村$/, new Set(["village"])],
    [new RegExp(`${KANJI_NUM}丁目$`), null, 1],
  ],
  zh: [
    [/(街道|社区|社區)$/, null],
    [/(?<!自治)[县縣]$/, new Set(["city", "borough"])],
    [/市$/, new Set(["city", "town"])],
    [/(?<![小社园園发發业業特新景校矿礦划劃])[区區]$/, new Set(["city", "borough", "suburb", "town"])],
    [/[镇鎮乡鄉]$/, new Set(["town", "suburb"])],
  ],
  ko: [
    [/(특별자치시|특별자치도|특별시|광역시)$/, null],
    [/(?<!도)시$/, new Set(["city"])],
    [/(?<!지)[구군]$/, new Set(["city", "borough", "suburb"])],
    [/[읍면]$/, new Set(["town"])],
    [/[0-9]*동$/, new Set(["quarter", "suburb", "neighbourhood"])],
  ],
};

/** Which country's suffixes a name follows: Korea's for Hangul, else Japan's in Japan, else China's. */
const suffixSet = (name: string, country: string | undefined): "ja" | "zh" | "ko" =>
  /\p{Script=Hangul}/u.test(name) ? "ko" : country === "JP" || /[\p{Script=Hiragana}\p{Script=Katakana}]/u.test(name) ? "ja" : "zh";

/** A place's name as a station takes it: one administrative suffix off where the kind has it (above). */
export function stationForm(name: string, kind: string | undefined, country: string | undefined): string {
  const set = suffixSet(name, country);
  const n = set === "ja" ? name.replace(/^大?字(?=..)/u, "") : name;
  for (const [re, kinds, min = 2] of SUFFIX[set]) {
    if (kinds && !(kind && kinds.has(kind))) continue;
    const m = n.match(re);
    if (m && [...n.slice(0, m.index)].length >= min) return n.slice(0, m.index);
  }
  return n.length >= 2 ? n : name;
}

/** A place's name in the language's script, from its own name and other names in order (the first is
 * OSM's name where a Latin one is shown): the first the language reads, shortened to another of its names
 * that it starts with (世田谷区 and 世田谷) or else by stationForm. Null when none reads. */
export function nativePlaceName(names: readonly string[], kind: string | undefined, country: string | undefined, lang: NativeLang): string | null {
  let first: string | null = null;
  const rest: string[] = [];
  for (const raw of names) {
    const r = readable(raw, lang);
    if (!r) continue;
    if (!first) {
      if (!nativeStation(r)) first = r;
    } else rest.push(r);
  }
  if (!first) return null;
  const prefix = rest.filter((r) => r !== first && [...r].length >= 2 && first!.startsWith(r)).sort((a, b) => b.length - a.length)[0];
  return prefix ?? stationForm(first, kind, country);
}

const WAYS: Record<NativeLang, [string, string, string, string]> = {
  // east, west, north, south
  ja: ["東", "西", "北", "南"],
  "zh-Hans": ["东", "西", "北", "南"],
  "zh-TW": ["東", "西", "北", "南"],
  ko: ["동", "서", "북", "남"],
};

/** "North Williamsburg" for a native name, from the station's offset (metres east and north): 東中野 in
 * Japanese, 上海南 in Chinese, 동대구 in Korean (only for a Hangul name). Null where there is no such form. */
export function nativeCompass(name: string, dx: number, dy: number, lang: NativeLang): string | null {
  const hangul = /\p{Script=Hangul}/u.test(name);
  if ((lang === "ko") !== hangul) return null;
  const [e, w, n, s] = WAYS[lang];
  const way = Math.abs(dx) >= Math.abs(dy) ? (dx > 0 ? e : w) : dy > 0 ? n : s;
  return lang === "zh-Hans" || lang === "zh-TW" ? name + way : way + name;
}

/** "Williamsburg 2" for a native name. */
export const nativeNumbered = (name: string, k: number) => (/\p{Script=Hangul}/u.test(name) ? `${name} ${k}` : `${name}${k}`);
