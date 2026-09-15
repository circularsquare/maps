# 024 — ir: Iran note: the thin Christian share reads as a Christian geography

Summary: Iran's note says the census counts Christians at 0.08-0.16% in 28 provinces, but the reviewer finds that share highest where no church is known, likely a recording floor. Add a sentence saying so? Recommended: yes.

*Filed 2026-09-14 by session `d743fc47`. Anita's call; nothing is waiting on it.*

## What I did

Left Iran's `note_public` as the builder wrote it. Iran is drawn and built; the dots follow the census
counts as printed, and nothing about them changes whichever way this goes.

## What it costs to reverse

An edit to `note_public` in `countries/ir.py`, `python tools/check_md.py`, then `tiles.py
--refresh-meta` (about a second, no retile).

## Why it is yours rather than mine

The voice and content of a `note_public` are yours (`CLAUDE.md`), and this sentence is what the map
tells readers about where a minority lives in a country held on spec §14 until ask 023.

## The detail

From the review of `ir` (`sources/ir.md` §8), on SCI's 1395 yearbook Table 3-18 as drawn:

- **The note reads:** "in the other 28 the census counts Christians at 0.08% to 0.16%", and "There is
  one Christian answer".
- **The share is highest where no church is known:** Bushehr 0.158%, Sistan and Baluchestan 0.150%,
  Ilam 0.148%. East Azerbaijan, with Tabriz's Armenians, is lower at 0.108%.
- **Christian and Zoroastrian shares rise and fall together** in the 25 provinces with no known community
  of either (Spearman +0.61, permutation p = 0.001). Two unrelated minorities moving together points to
  how answers were recorded, the same pattern as Kyrgyzstan's Buddhist cell (`sources/kg.md` §4.2).
- **Size:** that thin share is 53.7% of the Christians drawn (69,896 of 130,158). About a third of
  Zoroastrians probably sit on the same kind of floor.
- **"One Christian answer":** the 1395 form was never found, and the 1390 form offered three Christian
  answers. What is known is that the yearbook prints one Christian column.
- **Not proposed:** removing the floor from the dots, which would model over printed counts.

Options:

1. **Add one sentence** saying the thin share is highest in provinces with no known church and the census
   does not say who these people are, and change "one Christian answer" to "the yearbook prints one
   Christian column". The reviewer recommends this.
2. **Cut the "0.08% to 0.16%" clause instead**, following the cut-hard rule for public notes, and fix the
   "one Christian answer" wording.
3. **Leave the note as built.**


---

## Ruled 2026-09-15 by Anita: leave the note as built

*"i think this isnt super necessary, we're talking about like 2 placed dots in bushehr. but also this
is kinda just a minor thing and i think honestly it didn't need an ask (its really not that
catastrophic either way)."*

Option 3; no change. A note wording point this small is not an ask; `rd-super.md` step 3 now says so.
