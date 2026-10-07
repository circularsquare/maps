# Rwanda (rw): record

Drawn 2026-10-05 (session edd42a8c-mono). No language source: all 13,246,394 people on
Kinyarwanda across the 30 districts, rows `derived`, on the RPHC-5 2022 census district counts.
Placed on religiondots' Kontur hexes, which religiondots re-levelled sector by sector onto the
census's 416 sector counts (read-only), so placement inside a district follows the census down to
~15 km².

Files: `sources/rw_pop.py`, `taxonomy/rw2022.py`, `taxonomy/tree.d/rw.txt` (bare repeats),
`countries/rw.py`, `data/normalized/rw.csv`. Population: religiondots'
`data/geo/rw/rw_districts.gpkg`, read only.

## Why everyone is drawn as Kinyarwanda

- RPHC-5 (2022) and RPHC-4 (2012) ask literacy by language (Kinyarwanda, English, French,
  Swahili), not a spoken, home or first language (scout 2026-10-05; RPHC4 Education thematic
  report, statistics.gov.rw). Literacy is not a buildable question (AGENT_BRIEF §2).
- The Afrobarometer has never surveyed Rwanda (`python sources/mono_afro.py Rwanda`: 0 respondents
  in rounds 4-9), so there is no survey check on the national share.
- Kinyarwanda is the first language of practically every Rwandan. English, French and Swahili are
  official but learned (AGENT_BRIEF §2: learned second languages are not home languages). No
  cited source counts any other first language: Rufumbira on the Uganda border is a Kinyarwanda
  dialect; the Batwa speak Kinyarwanda. Congolese and Burundian refugees (~130k in camps) have no
  language count and are drawn on Kinyarwanda; said in `note_public`.

## Population and checks

`rw_pop.py` asserts 30 districts, 30 distinct units, total 13,246,394 (RPHC-5). religiondots
proved these district totals equal the thirty NISR district profiles to the person.
`check_country.py rw`: ok, 1 language, 30 units.

## Calls someone might reverse

- No minority drawn (no count exists; a guessed Swahili or French share would be invented).

## Scatter

13,246 dots over 10,845 hexes.

## Cross-border groups (2026-10-06, 5d7dac7e-xb)

Kinyarwanda is in a Rwanda-Rundi group with Kirundi, Ha and Hangaza. Their colours are still far apart. Groups only (taxonomy/regroup.txt): no label's node or count changed, each keeps its leaf and colour; dots rewritten in place. The full table of cases is in followups.md (2026-10-06, languages that stop at a border).
