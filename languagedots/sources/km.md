# Comoros (km): record

Drawn 2026-10-05 (session edd42a8c-mono; French added the same day, session edd42a8c-fix). No
census language question: French at Afrobarometer R10's national 9.9%, everyone else on their
island's own Comorian language, rows `modelled`, on the 2017 RGPH island counts: Ngazidja 379,367 on
Shingazidja, Ndzuwani (Anjouan) 327,382 on Shindzuani, Mwali (Mohéli) 51,567 on Shimwali;
758,316 in all. Placed on religiondots' Kontur hexes, which religiondots scaled to each island's
census count (Kontur swaps Anjouan's and Mohéli's people; religiondots' `sources/km_geo.py`).

Files: `sources/km_pop.py`, `taxonomy/km2017.py`, `taxonomy/tree.d/km.txt` (three new leaves),
`countries/km.py`, `data/normalized/km.csv`. Population: religiondots' `km_units.gpkg`, read only.

## Why one language per island

- The 2017 RGPH asks no language question. No microdata or survey with island-level language
  is open.
- Glottolog splits Comorian into Ngazidja (ngaz1238), Mwali (mwal1237), Ndzwani (ndzw1235) and
  Maore (maor1244), under Comorian Bantu (como1260); Ndzwani and Maore share a subgroup
  (shin1269). Its AES notes record each of the first three as the "statutory language of
  provincial identity" of its island (Constitution 2002, art. 1). So each island is drawn on its
  own. Movers between islands (many Anjouanais on Grande Comore) have no count and are drawn on
  their new island's language; said in `note_public`.

## The Afrobarometer check, and French

Afrobarometer R10 (2024), Comoros summary of results (religiondots'
`data/raw/km/COM_R10-Resume-des-resultats-ka-bh-23aout25-rev-1nov25.pdf`, p. 6), Q2 "Quelle est
la langue que vous parlez le plus chez vous actuellement ?": Shikomori 89.9%, French 9.9%,
Swahili 0.1%, Malagasy 0.1%; national only, no island split, R10 microdata not released.

**French is drawn at 9.9%** (Anita's ruling, 2026-10-05, reversing the first build, which folded
it into Comorian). R10 gives no island split and its microdata is not out, so the national 9.9%
is applied to each island's 2017 count: 75,073 French in all (Ngazidja 37,557, Ndzwani 32,411,
Mwali 5,105). Every row is now `modelled` (survey share x census population; the Comorian rows
too, being the 90.1% remainder). Swahili and Malagasy (0.1% each, likely a respondent or two)
stay on Comorian. Caveat kept in `note_public`: the same summary's Q96 puts 18.9% of respondents
at completed university and 3.0% post-graduate, far above the country, so the sample leans to
the people most likely to answer French and 9.9% may be high.

## New nodes

`nigercongo.bantu.comorian_ngazidja`, `_ndzwani`, `_mwali`: siblings of fr.txt's `comorian` and
`shimaore` leaves, not children of `comorian`, because France draws a named "Comorian" answer on
that node and a named label must never sit on a group node. Hand colours around Comorian's red:
Ngazidja 0.57 0.16 18, Ndzwani 0.68 0.14 42, Mwali 0.48 0.13 5.

## Checks

`km_pop.py` asserts the three island names and the 758,316 total. `check_country.py km`: ok.

## Calls someone might reverse

- French drawn at R10's national 9.9% on every island alike (no island split exists).
- Every resident of an island drawn on that island's language.

## Scatter

756 dots over 447 hexes (with French, 2026-10-05).
