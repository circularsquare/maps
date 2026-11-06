# Localization: Chinese and Japanese (then Spanish and German)

Research for Anita's ask of 2026-11-06 ("would be nice to have option for chinese and japanese on release...
what are usual workflows for this? commercial software? is there anything cheap and good"), with her answers of
the same day: nobody can review the translations, so Claude translates and reviews, aiming at correct meaning
and human-sounding text; do both Simplified and Traditional Chinese, and Japanese; Spanish and German second.
Prices were seen on 2026-11-06. Scripts and raw counts: the locale agent's scratchpad (`strings.mjs`,
`hard.mjs`, `tile_names.py`, `cjk_cover.py`, `places_sample.py`), not kept in the repo.

## In short

- **There is more text than "a few hundred strings"**: about 2,000 player-facing strings, 9,400 English
  words, spread over about 150 files, all written inline in the code today. Taking them out into a table is
  the big job; translating them is the small one.
- **No paid tool is needed.** With Claude translating and no human translators, a translation management
  system (Crowdin and the like) adds a web editor nobody would use. A plain JSON table per language in the repo,
  a small `t()` function, a glossary file and a check script are enough. Machine translation services
  (DeepL, Google) would fit their free tiers but know nothing of the game; Claude with a glossary and each
  string's context does better on UI text, for a few dollars of tokens per language.
- **Place names are mostly free.** The basemap tiles already carry Japanese and Chinese names
  (`name:ja`, `name:zh`, `name:zh-Hans`, `name:zh-Hant`) for most cities and many neighbourhoods; Japanese,
  Chinese, Taiwanese and Hong Kong place lists already keep the local-script name. Zone names (3,413, all
  English now) need a Wikidata pass.
- **Release**: the full UI can't be extracted, translated, checked and laid out safely in the few days left
  while other agents edit the same panels. I'd release in English and ship the three languages as an update
  about a week later (stages below). If she would rather hold the release, the plan takes about four working
  days with the UI frozen for the first two.

## 1. What there is to translate

Counted by parsing every `.ts`/`.tsx` file in `app/src` (the generated `wasm/` left out) and keeping the
string literals, templates and JSX text that read as English, then dropping debug pages and internal errors.
The sim (Rust) sends no text to the player; refusals arrive as codes and `game/issues.ts` words them.

| What | Count |
|---|---|
| Unique player-facing strings | about 2,000 (2,062 found, about 5% of them code or debug text) |
| English words / characters | 9,400 / 52,000 |
| Hovers (`data-tip`) | about 250 |
| Strings with values put in (`${n} trips`, `${line} stops at ${station}`) | 574, of which 270 have two or more |
| Plurals built in code (`plural(n, "car")`, `n === 1 ? "stop" : "stops"`) | 67 sites, plus 12 `andList` ("a, b and c") |
| Sentences pieced together around a chip or a value (won't translate as is) | about 120 fragments |
| Long strings (13+ words: notices, tutorial notes, explanations) | 147 |
| Biggest files | `MarketTrip.tsx` 99, `TripInspector.tsx` 85, `LineInspector.tsx` 77, `Blueprint.tsx` 74, `SavesPanel.tsx` 65, `issues.ts` 61, `MoneyTab.tsx` 61, `HelpPanel.tsx` 60 |
| The tutorial (`game/guide.ts`) | 22 steps and notes |

Data shown to the player, outside the code: holiday names (1,345, from the `holidays` package, English),
attraction names (12,641, Wikidata labels, English first), realm names (60, hand-written), zone names (3,413).
The privacy and terms pages and the Credits licence notices are best left in English (the English text is the
binding one).

The hard kinds, with examples:

- **Pieced sentences.** `map/trackHover.ts:267` builds "Fares cover " + a coloured chip + " of costs";
  `LineInspector.tsx:422` has a JSX text "-car trains." after a number. These become one message with the chip
  as a placeholder ("Fares cover {pct} of costs"), so a translator can move it.
- **Plurals and lists.** `relatedLayout.ts:308`: `${andList(stopping)} ${stopping.length === 1 ? "stops" :
  "stop"} at ${name}`. Japanese and Chinese have no plural forms (they use counters: 8両, 3本), German and
  Spanish have one/other like English. ICU plural messages cover all of them; lists go through
  `Intl.ListFormat`, which gives 、 and 和 by itself.
- **Word order.** "Reach {pct}% train share in {city}" puts the city first in Japanese and Chinese; any
  template is fine once its values are named placeholders.
- **Number and date formats.** `format.ts` hard-codes `en-US` (right for Chinese and Japanese, wrong for German
  and Spanish decimal commas). Money is "$22.4M"; in Japanese and Chinese `Intl.NumberFormat` with compact
  notation writes "$2240万", which reads naturally. `duration` writes "1h12m" (Japanese 1時間12分). Month and
  weekday names in `game/days.ts` and country names (`Intl.DisplayNames(["en"])` in three places) come free from
  `Intl` once the locale is passed. Save dates already follow the browser's locale.

## 2. Place names

| Names | Where they come from | Chinese / Japanese available? |
|---|---|---|
| Basemap labels (cities, neighbourhoods, roads, water) | OpenFreeMap tiles, shown as `name_en` then `name:latin` (`map/basemap.ts`) | Yes, in the same tiles: 82 `name:*` fields. Sampled at zoom 12: Tokyo places 41/46 have `name:ja`; London 132/153 `name:ja`, 122 `name:zh`; Paris 94 and 112 of 135; Shanghai 111/178 `name:zh-Hans`; Ohio at zoom 10, 51/74 `name:zh`. Roads mostly lack them outside their own country. A label expression change, no download. |
| Place lists (search, station names) | OSM per zone, `pipeline/places.py`; in latin-name countries the Latin name is shown and the local one kept as an alias | Japan, China, Taiwan, Hong Kong: yes, the local name is already in each place's aliases (Tokyo 28,751 of 28,839 places; Shanghai 990 of 1,463, the rest already shown in hanzi). Elsewhere: none; `name:zh`/`name:ja` would need adding to `OTHER_NAMES` and every place list rebuilt. |
| New station names | Cross street from the basemap tiles (`name:latin`, else `name_en`, else `name`), else the place list (`map/stationNames.ts`) | Street names: from the tiles, as above. Stored in the save once given, so a language change does not rename existing stations (fine; the player can rename). |
| Zone names (3,413; JP 45, CN 723, TW 13, HK 1, KR 24) | Registry from GHSL and NUTS, `zone-names.json` overrides (`game/zoneNames.ts`) | Not yet. Wikidata labels (`ja`, `zh-hans`, `zh-hant`) matched by name and position through QLever, as `attractions_json.py` already does; in CJK countries the place list's local name for the zone's main city is a ready fallback. About a day of pipeline work. |
| Realm names (60) | `realms.json`, hand-written | Translate with the UI strings. |
| City names in the diagram | Zone names (`diagram/cityNames.ts`) | Follow the zone names. |
| Attractions (12,641) | Wikidata through QLever | Yes, refetch the labels in `ja`, `zh-hans`, `zh-hant`; an hour or two. |
| The seven real-network saves | Station names baked in by the pipeline | Leave as they are. |

Default I'd take: show names in the player's language where the data has them, else in Latin as now; in
Japan, China, Taiwan and Hong Kong a Japanese or Chinese player gets the local script.

## 3. How this is usually done

- **Extraction.** Every string gets a stable id and moves to a message file, English as the source:
  `t("line.constructed", {line, cost})`. Libraries: [i18next](https://www.i18next.com/) (the most used, its own
  plural format), [FormatJS](https://formatjs.github.io/) (ICU messages; `react-intl` needs React, but its
  `intl-messageformat` works alone), [Lingui](https://lingui.dev/) (extraction by compile-time macros, more build
  setup). For one Preact app with signals, a hand-rolled `t()` over JSON files, with `intl-messageformat` for
  plurals and a helper that puts Preact nodes into placeholders (for the chips), is the least machinery. The
  language lives in a signal in `game/ui.ts`, so panels redraw on a change; the first visit picks it from
  `navigator.languages`, and Settings has a Language row.
- **ICU messages** (`{n, plural, one {# car} other {# cars}}`) are what translators and checkers expect, and
  they validate mechanically.
- **Fonts.** The UI is Nunito with Zen Maru Gothic for kana and kanji, cut into slices by code point
  (`ui/zenmaru.css`). Zen Maru is a Japanese font: it has every JIS kanji but only 63% of the 3,755 most common
  Simplified characters and 83% of the common Traditional ones (counted from the font's own table). Chinese text
  set in it would mix two fonts inside a word and show Japanese shapes for shared characters. So each language
  gets its own stack: Japanese as now; Simplified Chinese Nunito then the system font (PingFang SC, Microsoft
  YaHei, Noto Sans SC); Traditional Chinese Nunito then [Huninn](https://blog.justfont.com/2025/03/huninn-open-source-en/)
  (jf open-huninn, a rounded Traditional font built on Kosugi Maru, SIL OFL, on Google Fonts; close to Zen Maru's
  look; sliced the same way, T-1825) or PingFang TC / Microsoft JhengHei. `<html lang>` set to `ja`, `zh-Hans` or
  `zh-Hant` so the browser picks the right glyph forms and line breaking.
- **Map labels in CJK** already work: MapLibre draws kana, kanji and hanzi itself in the browser font
  (`localIdeographFontFamily`, which `pipeline/glyphs.py` relies on), so no new glyph tiles. Set that family per
  language so Chinese labels don't come out in a Japanese font.
- **Layout.** Japanese and Chinese run shorter in characters but each character is wide, and lines may break
  between any two characters; Japanese reads better with `word-break: auto-phrase` (Chrome 119 and later, with
  `lang="ja"`). German runs about a third longer than English, the main overflow risk. Canvas text (diagram,
  bullets, station labels) is measured after fonts load already.
- **Simplified to Traditional.** A converter gets most of the way. [OpenCC](https://github.com/BYVoid/OpenCC)'s
  `s2twp` converts characters and about 1,000 mainland-to-Taiwan phrases, mostly computer words (软件 to 軟體,
  信息 to 資訊, 默认 to 預設, 设置 to 設定, 加载 to 載入, 登录 to 登入). It has none of the transit words: 地铁
  (Taiwan 捷運), 公交 (公車), 站台 (月台), 换乘 (轉乘), and some of its swaps are wrong in context (保存 to
  儲存 also where it means "keep"). Since Claude does the translating, translating Traditional directly from the
  English with its own glossary column costs no more and avoids those slips; running `s2twp` over the Simplified
  file and diffing it against the direct Traditional one is then a cheap cross-check that flags terms to look at.
  Traditional here means Taiwan usage; Hong Kong differs again (港鐵, 巴士).

## 4. Tools and cost

| Option | What you get | Price (seen 2026-11-06) |
|---|---|---|
| [Crowdin](https://crowdin.com/pricing) | Web editor, MT, volunteers | Free: 60,000 hosted words, 1 private project ([CostBench, checked 2026-11-04](https://costbench.com/software/localization/crowdin/free-plan/)). Hosted words are source words times languages: 9,400 x 5 = 47,000, so it fits, just. Pro $50-59/month. Open-source plan free. |
| [Tolgee](https://tolgee.io/pricing) | Same, in-context editing | Free: 30,000 hosted words, 3 seats (too small for five languages); paid from €58/month. |
| [POEditor](https://poeditor.com/pricing/) | Same | Free: 1,000 strings (too small); 3,000 at $20/month, 10,000 at $60. |
| [Weblate](https://weblate.org/en/hosting/) | Same, open source | Hosted from €47/month for 10,000 strings; free Libre plan only for public open-source projects (trainworld isn't); self-hosted free but another server. |
| [Lokalise](https://lokalise.com/pricing), Phrase | Enterprise TMS | No free plan; Lokalise from $149/month. |
| DeepL API | Machine translation | Free 500,000 characters/month; Pro $5.49/month plus $25 per million ([G2](https://www.g2.com/products/deepl-api/pricing); DeepL's own page didn't load its API prices). Glossaries; no idea of UI context. |
| Google Cloud Translation | Machine translation | About $20 per million characters after a 500,000/month free tier; its LLM mode $10 per million in and out ([pricing page](https://cloud.google.com/translate/pricing); figures via [SimpleLocalize's 2026 comparison](https://simplelocalize.io/blog/posts/ai-machine-translation-cost-comparison/)). |
| Claude | Translation with glossary and context, review passes | Done by agents in her Claude Code sessions. Through the API it would be Opus 5.5 at $4 / $20 per million tokens in / out ([pricing](https://platform.claude.com/docs/en/about-claude/pricing)): about $3 a language for the translation, under $15 with the review passes. |
| Paid human translation | Native quality | General English to Japanese about $0.10-0.20 a word ([jobbers.io](https://www.jobbers.io/how-to-hire-a-freelance-translator-language-pair-rate-comparison/), [Translayte](https://translayte.com/blog/how-much-does-a-translator-cost-in-2024)), so $900-1,900 a language for 9,400 words; review-only editing is cheaper. Not planned. |

**Recommendation**: Claude translates and reviews; messages live in the repo as JSON; no TMS, no MT service. If
followers later offer to fix wording, Crowdin's free plan is the one that fits the word count, and the JSON
files import into it as they are.

## 5. Review without a human reviewer

Anita reads every English string before it lands, and that stays as it is for the English source. She can't do
that for Chinese and Japanese, and nobody else can, so review becomes a set of independent passes, each by an
agent that did not write the text:

1. **Context with every string.** The extraction step records, per id: the English, where it shows (button,
   hover, notice, label; file and component), what each placeholder is with an example value, and roughly how
   much room it has. Most bad UI translations are right words for the wrong context ("Track" as a verb or noun).
2. **Translation**, one agent per language, a panel at a time so related strings agree, following the glossary;
   doubtful strings get a note. Opus: a wrong meaning here is quiet.
3. **Back-translation.** A fresh agent translates the target back into English without seeing the source; a
   second compares the two and lists every change of meaning (a lost "not", "per day" become "a day ago", the
   wrong "it", a cost read as a profit). Fixed, then checked again.
4. **Naturalness.** A fresh agent reads each panel's strings together as a native player would and rewrites
   whatever sounds machine-made or stiff. AI-sounding text exists in these languages too (Chinese 赋能,
   无缝, 沉浸式; Japanese stacked katakana, "〜することができます"), and the same plain, short, a-little-cute voice
   as her English applies: Japanese labels as plain nouns and sentences in です・ます; Chinese short and plain,
   no 您 everywhere.
5. **Mechanical checks** (a script, run every time): same placeholders as the English, every ICU message
   parses, glossary terms used where the English has them, full-width punctuation (。、，), nothing left in
   English, length against the English flagged per language.
6. **Screenshots** of the main panels, the tutorial and the start menu in each language, looked at for overflow,
   missing glyphs and mixed fonts.
7. **Players.** The Feedback panel already exists; translation slips reported there get fixed in updates.

After the first pass, every English string added or changed is marked stale in the other languages (each
translation stores a hash of the English it was made from), shows in English until done, and the orchestrator
runs a translation batch before each deploy.

## 6. Glossary

`app/src/i18n/glossary.json`: each game term with what it means in the game, one fixed rendering per language,
and notes; the translator follows it, the check script enforces it. A starting draft, to be settled by the
first translation agent (who should also look at how Japanese and Chinese players of NIMBY Rails, Mini Metro and
Cities: Skylines name these):

| English | Meaning in the game | ja | zh-Hans | zh-Hant (Taiwan) |
|---|---|---|---|---|
| trip(s) | a journey people want to make (demand) | 移動 (counted 回) | 出行 (counted 人次) | 出行 (人次) |
| rides, riders | what happens on the player's lines | 乗客, 利用者 | 乘客 | 乘客 |
| boardings | people getting on at a station | 乗車人数 | 上车人次 | 上車人次 |
| line | a service the player runs | 路線 | 线路 | 路線 |
| station / platform | | 駅 / ホーム | 车站 / 站台 | 車站 / 月台 |
| transfer | changing lines | 乗り換え | 换乘 | 轉乘 |
| metro / bus | kinds of service | 地下鉄 / バス | 地铁 / 公交 | 捷運 / 公車 |
| headway | minutes between trains | 運転間隔 | 发车间隔 | 班距 |
| fare / upkeep | | 運賃 / 維持費 | 票价 / 维护费 | 票價 / 維護費 |
| zone | a demand area, a city or region | ゾーン | 区域 | 區域 |
| blueprint | planned track not yet built | 計画 (to settle) | 蓝图 | 藍圖 |
| construct | build a blueprint | 建設 | 建设 | 興建 |
| crowded | over a train's rated load | 混雑 | 拥挤 | 擁擠 |
| car (train car) | | 両 (counter) | 节 (车厢) | 節 (車廂) |

Kept as they are: "$" and the game's money, line symbols, the game's name.

## 7. Plan

| Stage | What | Effort |
|---|---|---|
| 0, now, safe | Nothing player-facing changes. Write the `t()` core, language signal, glossary file and check script on their own, so stage 1 can start at once. | an agent, half a day |
| 1, extraction | Every string to `app/src/i18n/en.json` with ids and context; pieced sentences, plurals and lists rewritten as whole ICU messages; `format.ts`, `days.ts` and the `Intl` calls take the locale; a pseudo-locale (every string wrapped in brackets) shows any missed string on screen; smoke passes. Split by folder over four agents (`ui/`, `ui/inspector/`, `game/`, `map/` with `workers/` and `diagram/`). The English must come out unchanged, so no review from Anita beyond the few strings rewritten from fragments. | 1-2 days, best with other UI work paused, since it touches every panel |
| 2, languages | Japanese, Simplified and Traditional Chinese through the review passes above; per-language fonts and `lang`; Language row in Settings (new English strings for her: "Language", and the language names written in their own scripts); screenshot pass and layout fixes. | about 1 day for all three |
| 3, names | Basemap labels in the player's language (label expression); new stations named from the same fields; attraction labels refetched; zone names from Wikidata. | half a day, plus a day for zones |
| 4, Spanish and German | The same passes, plus decimal commas and longer text. | about a day |
| Later | `name:zh` / `name:ja` in every place list (a rebuild of all zones' place lists), holiday names. | with the next place-list rebuild |

**Before release**: stage 0 only, unless she holds the release. Shipping part of the UI translated looks
worse than English, and extraction collides with every agent still editing panels this week. Stages 1 and 2
make a good first update about a week after release, announced to her Chinese and Japanese followers.

## Risks

- **Quiet meaning errors** with no native reader. The back-translation pass catches most of them, not all;
  player reports catch the rest. Wrong numbers are the worst kind: the placeholder check makes sure every value
  is still there.
- **Extraction regressions**: a string lost, a hover emptied, a chip dropped. The pseudo-locale and smoke catch
  most; extraction only moves text, so a diff that changes any English wording outside the fragment rewrites is
  a fault.
- **Upkeep**: English strings change most days. The stale check keeps that visible, and untranslated
  strings fall back to English rather than breaking.
- **Fonts**: a Chinese webfont is large; slicing by code point (as for Zen Maru) or the system font keeps the
  load small.

## Questions for Anita

- Release: English-only at release with Japanese and Chinese about a week later (my pick), or hold the release
  about four days for them?
- Traditional Chinese for readers in Taiwan (my default) or Hong Kong? The words differ (捷運 / 港鐵, 公車 / 巴士).
- Place names: in the player's language where the data has them (my default), or keep every name in Latin
  letters as now?
