# Localization

How the game's text is kept by language. The plan and its reasons are in `research/localization.md`;
this file is the rules. The languages are English (the source), Japanese (`ja`), Simplified Chinese
(`zh-Hans`), Traditional Chinese as written in Taiwan (`zh-TW`), then Spanish (`es`), Portuguese (`pt`, Brazilian), Turkish (`tr`) and German (`de`).

## Files

All in `app/src/i18n/`:

| File | Holds |
|---|---|
| `index.ts` | `t`, `tRich`, the `language` signal, `setLanguage`, and the formatting helpers in the current language (`tNumber`, `tList`, `tSeries`, `tSpaced`, `tDate`, `tCountry`, `tCompact`, `tDuration`) |
| `calendar.ts` | `weekdayLong(i)`: the full weekday name in the current language for the game's weekday index (0 is Monday); use it rather than a list of English day names |
| `intl.ts` | `LANGS`, and the same formatting with the language passed in; no signals or DOM, so workers and Node checks may import it |
| `messages/en.json` | Every English message, the source. Bundled with the game |
| `messages/<lang>.json` | The translations; a missing key shows the English. Fetched when the language is chosen |
| `context.json` | One note per key for translators: where it shows (button, hover, notice), and each placeholder with an example value |
| `glossary.json` | The game's terms, a one-line meaning and one rendering per language |
| `sources/<lang>.json` | For each translated key, the English it was translated from; a key whose English has changed since is stale. Written with the first translation |

Messages are [ICU MessageFormat](https://unicode-org.github.io/icu/userguide/format_parse/messages/),
formatted by `intl-messageformat` (about 33 KB minified, 10 KB gzipped). Tags (`<b>`) are not parsed: `<`
is plain text.

## Writing a message

- **Keys** are `<file>.<name>`, both camelCase: the file the text first lives in without its extension
  (`remove.question` for `game/remove.ts`, `lineInspector.trains` for `LineInspector.tsx`), then a short name
  for what it is. A key keeps its name when the code moves. Text used by several files goes under `common.`.
  One key per meaning, not per English wording: "Keep" on two different questions is two keys, since another
  language may word them differently.
- **Every key gets a note** in `context.json`, written with the message.
- **A sentence is one message.** Never join pieces of a sentence in code ("Fares cover " + chip + " of costs"):
  write `"Fares cover {share} of costs"` and pass the piece as a value; another language moves `{share}`
  where its grammar wants it. Joining whole sentences is fine only where each is a message of its own.
- **Choices wrap whole sentences.** `{built, select, true {Remove {what}? Nothing is refunded.} other {Remove
  {what}?}}`, not `Remove {what}?{built, select, true { Nothing is refunded.} other {}}`. Pass a boolean for a
  yes/no select; `t` turns it into `true` or `false`, so the options are `true` and `other`. A word chosen by a
  code value is a select too (`{effect, select, lose {...} stop {...} other {...}}`).
- **Plurals** are ICU plurals on the count: `{count, plural, one {# stop} other {# stops}}`, `#` being the count
  written in the language's digits. Never `n === 1 ? "stop" : "stops"` or `plural()` in new code. Japanese and
  Chinese have only `other`; German and Spanish `one` and `other` as English (Spanish also `many`, which falls
  back to `other`).
- **Lists** are made in code with `tList(names)` ("a, b and c", "a、b、c") and passed as one value; items in a row
  with no "and" ("5 track, 2 stations", "5 track、2 stations") with `tSeries`, never `.join(", ")`.
- **A phrase a message puts straight after a word** ("Trips per day{where}") is passed through `tSpaced(phrase)`:
  a space in front in English, none in Japanese and Chinese. Never `" " + phrase`.
- **Formatting happens in code**, never in the message: numbers, money, distances, durations and dates are
  formatted first (`tNumber`, `tDate`, `game/format.ts`) and passed as text. The only number a message formats
  is a plural's `#`. `{n, number}` and `{d, date}` are refused by the check. English numbers are en-US's; English
  dates and times follow the browser's first English locale (en-GB's "6 Nov, 14:05"), else en-US's.
- **Pieces that are nodes** (a chip, a coloured figure): `tRich("trackHover.fareCover", { share: <Chip .../> })`
  returns children for JSX. `t` returns a string and takes only text, numbers and booleans.
- **Apostrophes**: a lone `'` is text ("don't"). `''` is one apostrophe; `'{'` is a literal brace.
- **Read while rendering.** `t` reads the language signal, so a component or computed that calls it redraws on
  a change. Text kept in a signal or the save (a question asked, a notice) stays in the language it was made in.

## Numbers, money and durations

`game/format.ts` writes every figure in the current language. English keeps the game's own forms; Japanese and
Chinese take theirs, as Intl (ICU) writes them for the locale.

- **Short counts and money** (`count`, `money`, `moneyShort`, `moneyCash`, `millions`, `pointsShort`): English
  "763k", "1.17M", "$74.0M", "$6B", "$10-55M". Any other language gets the value English shows (after English's
  rounding) to as many significant digits, in `Intl.NumberFormat(lang, { notation: "compact" })`
  (`i18n/intl.ts` `formatCompact`, through `compactOf`): 76.3万, 117万, $7400万, $60億 in Japanese; 万 and 亿 in
  zh-Hans (1.2万亿 above 1万亿); 萬 and 億 in zh-TW. Money is "$" before the figure in every language, a minus
  "−" before that. Exact sums (`moneyExact`, `exact`) keep en-US's commas in every language. T-2798: the symbol is
  the player's currency's ("£", "€", "¥", "₩"; `game/currency.ts` has the rates), and a T (trillion, 兆) follows B
  for yen and won. T-2860: 44 currencies in all (`CURRENCIES` is the picker's order: USD EUR GBP JPY CNY KRW CAD AUD
  INR MXN, then by code); a symbol ending in a letter carries its space ("CHF 2.90", "kr 20.00"); a player who has
  not chosen starts on the currency of the first browser language with a region (`defaultCurrency`), else USD.
- **Durations** (`duration`, `i18n/intl.ts` `formatDuration`): English "4m08s", "1h01m", "42s". Japanese 4分8秒,
  1時間1分, 42秒; zh-Hans 4分8秒, 1小时1分; zh-TW 4分8秒, 1小時1分. A zero last part is left off (1時間, and a
  whole minute 2分, 2分钟, 2分鐘). A language with no words in `DURATION_UNITS` writes English's.
- **Minutes** (`headway`, `minutesText`, `ui/inspector/tripText.ts` `minText` and the like) go through the
  `inspector.min` message: "5.5 min", 5.5分, 5.5分钟, 5.5分鐘.
- **Units** km, m, mi, ft, km/h, mph and m² stay as written in every language: they are international, and
  Japanese and Chinese players read them so in games and on maps.

## What is never translated

- Save data and anything stored: ids, kinds, slot keys, the names a player gave.
- Station, line, zone and place names from the data (they are the data's, in their own language). Registry
  cities are the exception (below).
- Logs, `console.*`, thrown errors that only a developer sees, the `?debug` readouts, checks' output.
- Message keys, CSS classes, `data-*` values other than `data-tip`, URLs.
- The privacy and terms pages (the English is the binding text). The licence notices are on the about page
  (anita.garden), outside the game.

Text that looks like player text but is not, and stays inline, gets an `// i18n-ignore` comment on its line or
the line above, so the scan stops counting it.

## City names

The registry's cities (rows with no `kind`; zones keep their Latin names) show in Japanese and Chinese by their
Wikidata labels: `i18n/place-names.json`, `{ id: { q, ja, "zh-Hans", "zh-TW" } }`, read through
`tPlace(id, latinName)` (`i18n/places.ts`) where a city's name is shown. A missing label, an unmatched city
(`q: null`) and every other language show the Latin name. Labels: ja's own; zh-Hans from zh-hans, zh-cn, then zh
(through OpenCC to Simplified); zh-TW from zh-tw, zh-hant, then zh-Hans through OpenCC s2twp. A city is matched to the
item with the most sitelinks among those Wikidata's search finds for its name that have a coordinate within 60 km
of its centre.

`node trainworld/tools/place_names.mjs --opencc <opencc-js folder>` adds the cities not yet in the file (rerun it
after a join), `--relabel` fetches every known item's labels again, `--redo id,...` matches those ids again. A
wrong match is fixed by writing the right item into its `q` and running with `--relabel`. Search in the new game
sheet and the map search box find a city by its shown name as well as its Latin one (`cityMatch` in `game/search.ts`;
`isRegistryCity` stays on the Latin name). Every place a city's name is worded (hovers, the status tag, saves panel,
guide, records, bubble tags, diagram labels) reads it through `tPlace`; an achievement record keeps the Latin name and
is translated when shown.

## Workers

Workers send codes and figures, never words; the main thread words them (as `game/issues.ts` does the track
model's refusals). A worker that must write text imports `i18n/intl.ts` for formatting and is passed the
language with its job.

The clock worker's notices and save reasons are sent as message keys (`"clockWorker.newer"`, `"issues.internal"`);
`workers/clockClient.ts` `workerText` turns a key into text on the main thread. Default names written into a save
("Saved game", "Bus 3") are the save's data and stay as made; so do diagnostic details in logs.

## Names kept in lists and constants

A list read by index (`DAY_KIND_NAME`, `MONTH_LONG`, `PERIODS[i].name`, `TERRAIN_NAMES`) keeps its shape and reads its
message when read: `game/lazyText.ts` `lazyTexts([...keys])` for a plain list, a `get name()` in an object literal
for a record. A constant text that only a few files read is `export let X = ""` set by an `effect` that calls `t` (as
`heldTip`, `IDLE_TIP`). Where code compares a name with a literal, the object carries a stable `id` beside the
translated `label` (`CostTerm` in `game/trip.ts`).

## Choosing a language

`setLanguage(lang)` fetches that language's messages, then sets `language` and `<html lang>` (`en`, `ja`,
`zh-Hans`, `zh-TW`), and asks canvas text to be measured again (`tw-fontchange`). At startup `i18n/index.ts`
awaits the language's messages at its top level, so every module that imports it, `main.ts` among them, runs
and first renders in that language; English is bundled and waits for nothing, and a failed fetch opens in
English. Workers (no document) skip this and are passed their language. At startup the language is,
in order: `?lang=<code>` in the address (not remembered), the player's pick on this device (localStorage
`trainworld-lang`), the first of `navigator.languages` that is offered (`zh-TW`, `zh-HK`, `zh-MO` and any
`zh-Hant` give zh-TW, other `zh` zh-Hans; English ranked first keeps English), else English. `?lang=pseudo`
shows every message in ⟦ ⟧ over the English, so text still inline stands out on screen.

**Offered languages** are `OFFERED` in `i18n/intl.ts`: English, Japanese, Simplified and Traditional Chinese
(released 2026-11-07, T-2585); a new language joins once released (below). Settings shows the Language row
(`ui/SettingsTab.tsx`, right under the first row's Controls and account buttons), and the start menu the same picker
(`ui/LanguageSelect.tsx`) at the right of its card's first row (T-2753), while `OFFERED` has two
languages, and with `?lang=` in the address lists all of `PICKER` so a language can be seen before release.
Each language is listed by its own name (`LANG_NAMES`, never translated); the globe icon's hover names the row
and says that non-English text is translated by AI. A pick takes effect at once, without a reload (panels
redraw; text already made, such as an open question or a notice, keeps its language), and is remembered on
this device.

## Fonts

`--font` (`ui/style.css`) follows `<html lang>`: Nunito first everywhere, so Latin letters and digits, and so
tabular figures, are the same in every language; then the language's own face. Japanese: Zen Maru Gothic
(`ui/zenmaru.css`). Traditional Chinese: Huninn (jf open-huninn, rounded like Zen Maru, SIL OFL; Google Fonts'
111 woff2 slices self-hosted in `public/fonts/Huninn-400-*.woff2`, 2.5 MB in all, `i18n/huninn.css`; a page
loads only the slices its text uses), then PingFang TC, Microsoft JhengHei, Noto Sans TC. Huninn has no bold, so
zh-TW turns off the browser's made-up bold (`font-synthesis-weight: none`) and sets `.h` headings at 16px instead.
Simplified Chinese:
the system's face (PingFang SC, Microsoft YaHei, Noto Sans SC), since no rounded Simplified face is free to
host. Chinese never falls into Zen Maru, which lacks a third of common Simplified hanzi and draws Japanese
glyph shapes. In Japanese and Chinese the page sets `text-autospace` (the small gap between hanzi or kana
and Latin letters or digits, so messages carry no space for it); Japanese lines break between phrases
(`word-break: auto-phrase`). The line height is the page's fixed 1.45 in every language. MapLibre's own CJK
labels (`localIdeographFontFamily`) are not yet set per language (T-2586).

## The check

`node trainworld/checks/i18n.mjs` (smoke runs it) fails on: a key in a translation or `context.json` that
English lacks, or in `sources/` that the translation lacks; a duplicated key; a message that does not parse; a placeholder missing, added or of another kind
than the English; a select without one of the English options; a plural option the language has no category
for; a formatted argument or a tag; an English key without a note or not shaped `file.name`; a checked glossary
term the English uses whose rendering the translation lacks (a key listed in the term's `except` is let off).
It reports each language's progress (translated, stale, without a source; `--stale` lists them) and then counts
the player text still written inline in `app/src`, per folder (`--list <folder>` lists each, file:line). That
count reaching zero, `i18n-ignore` aside, is stage 1 done.

## Translating

Claude translates and reviews; nobody else reads the result, so each unit of work goes through passes by
different agents (the reasons: `research/localization.md` 5). The tools are `checks/l10n.mjs` (its header lists
the commands), `checks/i18n.mjs` (faults) and `checks/l10n-shots.mjs` (screenshots).

**Units.** One language and one scope: `--folder <dir under app/src>` (the keys of that folder's files, `.` for the
root), or `--keys <prefix,...>` to split a big folder or take the keys no file is named after (`common.`).
`l10n.mjs status` lists keys, words and progress per folder; a unit of 150-400 keys suits one translator. Each
language's files are shared: `merge` takes a lock and writes through a temp file, so translators of one language
work at once on different scopes. zh-TW is translated from the English like the others, never converted from
zh-Hans. Every agent works in its own scratch folder and never edits `messages/` or `sources/` by hand.

**Pass 1, translate** (an Opus agent per unit; a wrong meaning here is quiet):
1. `node trainworld/checks/l10n.mjs batch <lang> <scope> --out <scratch>/batch.json`: each key still to do (new,
   or `was` and `wasFrom` when the English has changed since) with its English, note and glossary terms.
2. Write `<scratch>/tr.json`, `{ "key": "translation" }`, following the style below, one panel's keys together
   so they agree. A string you doubt goes in `<scratch>/doubts.md` with why, for the review.
3. `node trainworld/checks/l10n.mjs merge <lang> <scratch>/tr.json`; fix every fault it prints and merge again.
4. `node trainworld/checks/l10n.mjs lint <lang> <scope>` (Chinese also `--opencc <dir>`); fix each warning or
   leave it with a reason in doubts.md.

**Pass 2, back-translate** (a fresh agent, Sonnet is enough): `l10n.mjs back <lang> <scope> --out
<scratch>/back-in.json` gives each key and its translation only. Prompt: "Translate each `text` into plain
English as literally as the meaning allows, keeping `{placeholders}` and ICU syntax as they are; the key names
where it shows. Do not open en.json, context.json, glossary.json or any batch file. Write `{ key: English }` to
<scratch>/back.json."

**Pass 3, review** (a fresh Opus agent that did not translate): `l10n.mjs compare <lang> <scratch>/back.json
<scope> --out <scratch>/compare.json` gives each key's English, note, translation and back-translation. Prompt:
"For each key, compare the back-translation with the English and list every change of meaning: a lost or added
not, a number or unit changed, per day read as a day ago, the wrong subject or object, a cost read as a profit,
a select option or plural branch that says something else, a glossary term rendered otherwise. Then read each
panel's translations together, as a native player of a rail game would, and rewrite whatever is stiff,
machine-made or inconsistent between neighbouring keys, following the style rules in docs/localization.md. Write
fixes as `{ key: translation }` to <scratch>/fix.json, merge them (`l10n.mjs merge`), run lint, and report each
meaning error found and each rewrite with a few words why." Meaning errors go back into the review's report for
the orchestrator; a unit with more than a few is translated again by a fresh Pass 1 agent.

**Pass 4, cross-checks** (the reviewer, at the end): `l10n.mjs lint` clean or explained; for zh-TW once the
same scope's zh-Hans is reviewed, `l10n.mjs opencc <scope> --opencc <dir>` and a look at each difference (most are
Taiwan words, 月台 against 站台; the rest is a slip in one of the two).

**Pass 5, screenshots** (the reviewer, on its own Vite after claiming cores): `node trainworld/checks/l10n-shots.mjs
http://localhost:<port>/ <artifacts>/trainworld/<agent>/ --langs en,<lang> --tabs <the scope's tabs>`, then a
look for clipped or wrapped labels, English left, mixed fonts and boxes for missing glyphs. It prints rows
taller than English's and buttons whose text is clipped. Fixes to wording go through merge; layout fixes are
the orchestrator's to hand out.

OpenCC for `--opencc`: `npm install --prefix <scratch>/opencc opencc-js`, then pass
`<scratch>/opencc/node_modules/opencc-js`.

**Style.** The English's voice in every language: plain, short, a little cute, never a pitch.
- Glossary first: the first rendering of each term, unless the term's notes say otherwise; trips and rides stay
  apart. A rendering that reads wrong in one message is a doubt for the review, not a quiet change.
- Keep every `{placeholder}`, select option and plural; Japanese and Chinese have only `other` (`{count,
  plural, other {#駅}}`). Placeholder values are already formatted (numbers, money, distances) and names come
  from the data in their own language: write around them.
- No space between hanzi or kana and Latin letters, digits or a placeholder (the page adds the gap). Full-width
  punctuation in Japanese and Chinese (、。，！？：；（）); ASCII digits. Periods only on full sentences, as in
  the English; labels and hovers that are phrases end bare.
- Japanese: labels and buttons as plain nouns or short verb phrases (撤去, 線路を削除); full sentences in
  です・ます; katakana only where Japanese players use it; never 〜することができます.
- Chinese: short and plain; 你 or no pronoun, never 您; no 赋能, 无缝, 沉浸式. zh-Hans is mainland usage, zh-TW
  Taiwan's (捷運, 公車, 月台, 轉乘, 預設, 載入, 背景執行; click 按一下, double click 按兩下, right click 按右鍵,
  key combinations with + as in Shift+按一下).
- Narrow places (the note says button, segmented, short or narrow): no longer than the English looks; lint
  flags them.

**Splitting the whole UI.** About 2,400 keys and 10,000 English words, in five scopes of 300-600 keys: `ui`
A-L and L-Z (`--keys` lists), `ui/inspector`, `game`, `map` with `workers`, `diagram` and the root, and
`common`. That is 15 units for three languages. Agents of one language translate different scopes at once;
reviewers start on a unit once its translation is merged. Back-translation, review and screenshots of a unit
can run while its language's next unit is being translated.

**Releasing a language**: once `l10n.mjs status <lang>` shows every key done and none stale, and every unit's
review is in, add the language to `OFFERED` (`i18n/intl.ts`). Every English string added or changed after that
shows as stale (`checks/i18n.mjs --stale`) and shows in English until a batch (`batch` picks the stale keys,
with their old translation and old English) goes through passes 1 to 3 again; the orchestrator runs one before
each deploy.
