# Curaçao (cw): record

Drawn 2026-10-05 (session edd42a8c-cw). Census 2023, language spoken most often at home, island
only, 147,498 people in five groups; placed inside the island per geozone.

Files: `sources/cw_census.py` (counts), `sources/cw_geo.py` (placement layer),
`taxonomy/cw2023.py`, `taxonomy/tree.d/cw.txt`, `countries/cw.py`,
`data/normalized/cw.csv`, `data/geo/cw/cw_hexes.gpkg`, `data/raw/cw/`.

## 1. What exists, and what was searched

- **Census 2023 (Senso 2023), CBS Curaçao.** The question was asked of every person ("meest
  gesproken taal of talen thuis", first, second and third). The only published figures are the
  island-wide shares in *Eerste Resultaten Census 2023* (5 June 2024), page 21: first language
  Papiamentu 78.0%, Spaans 8.4%, Nederlands 7.9%, Engels 3.8%, overig 2.0%, of 147,498 who
  answered (population 155,826). *Migranten in Curaçao* (Census 2023 series, November 2025)
  Table 18 gives the same question by country of birth, for eleven countries, in percentages.
- **Nothing by geozone or neighbourhood.** Searched: the 2023 table pages on senso.cbs.cw
  (demographic, household, migration, geozone & neighbourhood: tables G-1 to G-4 hold age, sex,
  birthplace and nationality only); the 2023 neighbourhood viewer (cbs-curacao.github.io/
  ndv-static-site; its data table has 40-odd indicators, no language); CBS's ArcGIS Online
  organisation (services5.arcgis.com/1KFGuk9LOFs0SAY7; 13 services, the 2011 neighbourhood layer
  has 30 fields, none language); the 2011 census table pages (D-6 is national only, by age);
  the government's 2019 neighbourhood profiles (gobiernu.cw "Buurtprofiel", 2001 census, no
  language section). The coverage sweep's guess of "neighbourhood (CBS neighbourhood viewer)"
  was wrong for language.
- **Census 2011, Table D-6**, "Most spoken language (in private households)", national, by age,
  ten categories, 147,862 people; asked once per household and applied to its members.

## 2. Which table, and why 2023

2023 is twelve years newer and Spanish moved most in between (5.6% to 8.4%, Venezuelan
migration); it was asked per person. 2011 names more languages (Haitian Creole, Chinese,
Portuguese, Hindi, Arabic, 3,375 people together), but the two cannot be combined: splitting 2023's
"overig" by 2011's proportions would change the counts, which is a proxy for Anita to allow. The
cost of 2023 is about 2,950 people drawn as grey "other" who in 2011 would have been named.

Counts are the shares times 147,498, scaled because the printed shares sum to 100.1%: Papiamentu
114,934, Spanish 12,377, Dutch 11,641, English 5,599, other 2,947. Every row is `derived`; the
one-decimal rounding is worth about +-75 people per language.

## 3. Checks

- `cw_census.py` stops unless the box on page 21 holds all five rows, the shares sum to 100 +-0.25,
  the answered count is 147,498, and the prose on the same page repeats the four named shares.
- 2011 against 2023 (shares of those reported): Papiamentu 80.1 / 78.0, Dutch 8.8 / 7.9,
  Spanish 5.6 / 8.4, English 3.1 / 3.8, other 2.3 / 2.0. The direction matches what the
  *Migranten* publication says (Spanish and English up); the D-6 columns sum to their totals by age.
- *Migranten* Table 18 (2023, by birthplace): Curaçao-born 92.6% Papiamentu; Colombia-born 73.4%
  and Venezuela-born 75.9% Spanish; Netherlands-born 64.9% Dutch; Jamaica-born 74.0% English. The
  Curaçao-born are about three quarters of the island, so 92.6% of them alone is about 69 points
  of the 78.0: consistent.

## 4. Mapping

Papiamentu on `creole.portuguese_based.papiamento` (pl2021's node; Glottolog papi1253 files it under
Indo-European, the project files creoles under `creole`). Spanish, Dutch, English on the shared
nodes. "overig" on `other`: in 2011 that remainder spanned four families and held no indigenous
language. No new nodes and no colours picked: Papiamento comes out pale green (#dceeb2), Spanish
ochre (#cbb76a), Dutch blue (#51c7f1), English pale blue (#c3e2fe); all four tell apart.

## 5. Placement

The island is one counted unit. Inside it (AGENT_BRIEF §4.4):

- **Geography.** religiondots' Kontur hexes for CW (503, read only). Each hex centroid goes to a
  CBS neighbourhood (ArcGIS `BuurtenCBS`, 291 polygons; 49 coastal hexes snapped, at most 412 m;
  the 2 hexes on Klein Curaçao, 2 Kontur people, left out). A neighbourhood's geozone is its
  `geocode // 100`; the numbers are named by the 2023 viewer's neighbourhood-to-geozone table (every
  number got exactly one name) and, for ten numbers the viewer does not carry, by the service's own
  "Undefined (zone X)" rows. All 60 geozones of Table G-3 matched one number each, both ways.
- **Witness.** Kontur per geozone against G-3: log r = 0.879 against a best of 0.393 over 500
  shuffles. Kontur is 1.22x the census overall and uneven (normalised p10 0.50, p90 1.65; Hato
  5.2, the airport; Christoffel 3.9; Oostpunt 0.12), so each geozone's hexes are scaled to its
  2023 population. Asiento (the refinery, 2,255 Kontur people) and Nieuwe Haven (148) are under 5
  people in G-3 and get weight 0. 1,186 people with no geozone in G-3 are not in the weights
  (only shares matter here).
- **Language weights.** Spanish, English and other: hex people times its geozone's share with a
  nationality other than Dutch (G-3; national 12.3%, from 0% in Christoffel to 29.7% in
  Scharloo). Most of these speakers were born abroad (Table 18: about 1.5k of 12.4k Spanish and
  2.2k of 5.6k English speakers are Curaçao-born, by applying its shares to roughly 117k
  Curaçao-born). Papiamentu and Dutch: hex people. Dutch was not weighted by nationality because
  Dutch-born residents hold Dutch nationality, like nearly every Curaçaoan.

## 6. Calls someone might reverse

- 2023 shares over 2011 counts (§2).
- English placed with Spanish by other-nationality share, though about 40% of English speakers
  are Curaçao-born.
- Dutch on plain population; a birthplace-by-geozone weight (G-2 foreign-born) would lean it
  towards the Netherlands-born, but G-2 lumps them with Latin American migrants.

## 7. Scatter

144 dots at 1:1000 over 121 polygons; 3,498 people fall under one dot per language and draw none.
