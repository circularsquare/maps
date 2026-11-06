# Localization

How the game's text is kept by language. The plan and its reasons are in `research/localization.md`;
this file is the rules. The languages are English (the source), Japanese (`ja`), Simplified Chinese
(`zh-Hans`), Traditional Chinese as written in Taiwan (`zh-TW`), then Spanish (`es`) and German (`de`).

## Files

All in `app/src/i18n/`:

| File | Holds |
|---|---|
| `index.ts` | `t`, `tRich`, the `language` signal, `setLanguage`, and the formatting helpers in the current language (`tNumber`, `tList`, `tDate`, `tCountry`) |
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
- **Lists** are made in code with `tList(names)` ("a, b and c", "a、b、c") and passed as one value.
- **Formatting happens in code**, never in the message: numbers, money, distances, durations and dates are
  formatted first (`tNumber`, `tDate`, `game/format.ts`) and passed as text. The only number a message formats
  is a plural's `#`. `{n, number}` and `{d, date}` are refused by the check.
- **Pieces that are nodes** (a chip, a coloured figure): `tRich("trackHover.fareCover", { share: <Chip .../> })`
  returns children for JSX. `t` returns a string and takes only text, numbers and booleans.
- **Apostrophes**: a lone `'` is text ("don't"). `''` is one apostrophe; `'{'` is a literal brace.
- **Read while rendering.** `t` reads the language signal, so a component or computed that calls it redraws on
  a change. Text kept in a signal or the save (a question asked, a notice) stays in the language it was made in.

## What is never translated

- Save data and anything stored: ids, kinds, slot keys, the names a player gave.
- Station, line, zone and place names from the data (they are the data's, in their own language).
- Logs, `console.*`, thrown errors that only a developer sees, the `?debug` readouts, checks' output.
- Message keys, CSS classes, `data-*` values other than `data-tip`, URLs.
- The privacy and terms pages and the Credits licence notices (the English is the binding text).

Text that looks like player text but is not, and stays inline, gets an `// i18n-ignore` comment on its line or
the line above, so the scan stops counting it.

## Workers

Workers send codes and figures, never words; the main thread words them (as `game/issues.ts` does the track
model's refusals). A worker that must write text imports `i18n/intl.ts` for formatting and is passed the
language with its job.

## Choosing a language

The game shows English. `setLanguage(lang)` fetches that language's messages, then sets `language` and
`<html lang>`. The URL switch `?lang=<code>` does the same at startup, and `?lang=pseudo` shows every message
in ⟦ ⟧ over the English, so text still inline stands out on screen. A Language row in Settings and the first
visit's pick from the browser come with the first translations (stage 2).

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

A translation agent writes `messages/<lang>.json` and, for each key, the English it translated into
`sources/<lang>.json`, following `glossary.json`; the review passes are in `research/localization.md` 5.
