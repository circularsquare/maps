# Fiji (fj): record

Drawn 2026-10-05 (session edd42a8c-last). 837,271 people (2007 census), 15 provinces, 5 nodes,
rows `derived`. 834 dots.

Files: `sources/fj_census.py`, `taxonomy/fj2007.py`, `taxonomy/tree.d/fj.txt`, `countries/fj.py`,
`data/normalized/fj.csv`. Read-only from religiondots: `data/raw/fj/fj_2007_analytical_report.pdf`,
`data/normalized/fj.csv` (province totals for the join), `data/geo/fj/fj_hexes.gpkg`,
`fj_lookup.csv`.

## Source and checks

No Fijian census has tabulated language since 1946; the 2017 census published no ethnicity by
province (its three releases in religiondots' raw folder have none). The 2007 Analytical Report,
Tables I-5a..d (pp. 31-37), prints Total / Fijian / Indian / Other per province for 1996 and
2007. The parser takes 2007 and keeps the one block whose Total equals the province total of
religiondots' 2007 religion table: all 15 match exactly once, each block's three groups sum to
its total, and the provinces sum to Table I-3's national figures (475,739 / 313,801 / 47,731).

## How (AGENT_BRIEF section 2, ethnicity read as language)

- Fijian -> Fijian; Indian -> Fiji Hindi (the Indo-Fijian home koine; no retention source
  needed in the other direction: Indo-Fijians of South Indian descent speak it too).
- Other: Rotuma's 1,893 -> Rotuman; Cakaudrove's 5,437 -> Gilbertese (Rabi Island's Banabans,
  resettled 1945; Kioa's Tuvaluans folded in); the other 40,401 -> `other` (mixed groups, no
  split by province).
- No retention source exists (no language question since 1946); recorded here.
- Fijian hand-coloured (blue, h 232) in the fragment.

## Calls someone might reverse

- 2007 vintage, 19 years old; the Indo-Fijian share has fallen since.
- Cakaudrove's "Other" all on Gilbertese, and placed over the whole province rather than Rabi.
- Western Fijian (Nadroga, western Viti Levu) not separated from Fijian.

## Room for improvement

Tikina-level ethnicity (SPC PopGIS serves 2007 tikina tables) would place Indo-Fijians in the cane
belt more exactly; a language item in a future census.
