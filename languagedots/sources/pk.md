# Pakistan

Drawn: 156 districts, 246,500,586 people, 22 language nodes. Four provinces and Islamabad from the
2023 census's Table 11 (measured); Gilgit-Baltistan and Azad Jammu & Kashmir modelled (added
2026-10-06, session `5d7dac7e-gb`, on Anita's "would be nice to get gilgit baltistan if
possible").

| part | source | people | tier |
|---|---|---|---|
| Punjab, Sindh, KP, Balochistan, Islamabad (136 districts) | Census 2023 Table 11, mother tongue (`sources/pk_t11.py`) | 240,458,089 | measured |
| Gilgit-Baltistan (10 districts) | census 2023 GB totals x MICS district shares, raked (`sources/pk_north.py`) | 1,709,030 | modelled |
| Azad Kashmir (10 districts) | AJK Statistical Year Book 2025 Table 15.31 x census 2023 population | 4,333,467 | modelled |

## 1. Provinces and Islamabad: Table 11

`sources/pk_t11.py` reads PBS's Table 11 from the CRAN package PakPC2023 (its docstring has the
route), tehsils summed to religiondots' 2023 district ids. Fourteen named languages plus OTHERS
(3.34M: Khowar, Burushaski, Wakhi, Gujari, Persian and more, on `other`). `taxonomy/pk2023.py`.

## 2. What exists for Gilgit-Baltistan and Azad Kashmir (searched 2026-10-06)

- **PBS Table 11** covers only the four provinces and Islamabad; religiondots' `sources/pk.md`
  §10.3 lists 33 probed `table_*_{gb,ajk,...}` URLs, all 404.
- **GB at a Glance 2025** (GB P&DD Statistical & Research Cell, `pnd.gog.pk/pages/downloads`,
  saved as `data/raw/pk_north/gb_at_glance_2025.pdf`), p.9, citing "Census 2023, Pakistan Bureau
  of Statistics": **mother tongue for GB as a whole**, Shina 50.21, Balti 29.94, Pushto 0.86,
  Kohistani 0.86, Urdu 0.38, Others 17.74 (%). p.4: 2017 and 2023 census population by the ten
  districts (male, female), adding to the printed GB row (891,558 + 817,472 = 1,709,030;
  religiondots' 1,709,049 is a different print of the same total, 19 apart, likely transgender).
  No district split of mother tongue was found. The ten 2024 district brochures that might hold
  one are cut-off Wayback captures (religiondots §10.3).
- **GB MICS 2024-25** survey findings report (same downloads page,
  `gb_mics_2024_25_sfr.pdf`), Table SR.3.1, p.39: households by language of household head
  (HC1B asks the head's *mother tongue*), GB-wide: Shina 48.0, Balti 29.2, Brushaski 12.3, Khowar
  5.2, Wakhi 1.0, Other 4.2. Every other table uses language only as a row variable, never
  crossed with district.
- **GB MICS 2016-17** district x language: sampled households by language of household head for
  each of the ten districts, printed in Shah Zaman, "Treading the Sacred Linguistic Landscape of
  Gilgit-Baltistan", Pamir Times, 2023-12-23 (`pamirtimes_2023-12-23_linguistic_landscape.html`).
  6,213 households, 585-708 per district. Checks: each row adds to its printed total, the
  columns add to the printed totals row, and the districts add to the article's first table's
  three divisions (Baltistan 2,607, Diamer 1,190, Gilgit 2,416). MICS microdata (UNICEF) needs
  a registration, so the article's unweighted counts are used; MICS samples a similar number of
  households per district, and within a district the strata weights move shares little.
- **AJK Statistical Year Book 2025** (`religiondots/data/raw/pk2023/`, read in place), Table
  15.31 "Languages Spoken in AJ&K", pdf p.226: percent by district, source "Kashmir Liberation
  Cell, Muzaffarabad". Whole percents, an administrative estimate, not a census count. No census
  mother-tongue table for AJK was found (the yearbook prints the 2023 census's religion, not its
  language).

## 3. Gilgit-Baltistan method

Seed: MICS 2016-17 district shares times census 2023 district population. Targets: the census's
GB-wide shares, with its Others (17.74%) divided among Burushaski, Khowar, Wakhi and a remainder
in MICS 2024-25's proportions (12.3 : 5.2 : 1.0 : 2.1, the 2.1 being MICS "Other" 4.2 less the
census's separately named Pashto, Kohistani and Urdu). MICS 2016-17's "Other languages" seed is
split among Pashto, Urdu and the remainder in those target proportions; **Kohistani is seeded in
Diamer only** (the one GB district on the Kohistan border; Kohistani Shina of Darel and Tangir).
Then IPF to the district populations and the GB targets; integer counts by largest remainder.

The seed already lands near the census, which is the check that the two sources describe the same
place: Shina 48.2 seed against 50.2 census, Balti 30.1 against 29.9, Burushaski 11.4 against
10.6, Khowar 4.1 against 4.5, Wakhi 1.3 against 0.9.

Raked shares (%), what the map draws:

| district | Shina | Balti | Burushaski | Khowar | Wakhi | Pashto | Kohistani | Urdu | other |
|---|---|---|---|---|---|---|---|---|---|
| Astore | 99.7 | 0 | 0 | 0 | 0 | 0.1 | 0 | 0 | 0.2 |
| Diamer | 87.6 | 0.2 | 0.1 | 0 | 0 | 2.2 | 4.4 | 1.0 | 4.6 |
| Ghanche | 0 | 99.9 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Ghizer | 44.5 | 0.2 | 17.4 | 33.0 | 0.6 | 1.2 | 0 | 0.5 | 2.6 |
| Gilgit | 80.3 | 1.1 | 11.4 | 3.0 | 0.8 | 1.0 | 0 | 0.4 | 2.0 |
| Hunza | 13.9 | 0 | 69.0 | 0.2 | 16.8 | 0 | 0 | 0 | 0 |
| Kharmang | 13.8 | 83.5 | 0 | 0 | 0 | 0.8 | 0 | 0.3 | 1.6 |
| Nagar | 27.8 | 0.2 | 72.1 | 0 | 0 | 0 | 0 | 0 | 0 |
| Shigar | 0.5 | 99.4 | 0 | 0 | 0 | 0 | 0 | 0 | 0.1 |
| Skardu | 21.5 | 76.7 | 0.3 | 0.2 | 0 | 0.4 | 0 | 0.2 | 0.8 |

GB's remainder (1.8%) goes on `other`: Domaaki, Gojri, Kashmiri and whatever else MICS filed as
other, none of them named. Domaaki (Hunza and Nagar, a few hundred speakers) cannot be drawn.

## 4. Azad Kashmir method

Percent x the district's 2023 census population (religiondots' `pk.csv` AJK rows, which are the
yearbook's Table 15.24 and add to 4,333,467), largest remainder per district. Every row prints
100% except **Mirpur, 10 + 85 + 2 = 97**, scaled up. Labels, as the cells print them: Kashmiri,
Gojri, Pahari (with the variety named in seven districts), Shina (Neelum 5%), Kundal Shahi
(Neelum 2%), Dogri (Bhimber 30%, printed in the Shina column), Punjabi (Bhimber 35%, in the
Others column), Others (Sudhnoti, Kotli, Mirpur).

AJK totals drawn: Pahari-Pothwari 2,981,612 (69%), Gojri 804,438, Kashmiri 205,687, Punjabi
153,648, Dogri 131,698, other 40,878, Shina 11,076, Kundal Shahi 4,430.

Doubts, kept as printed: Kundal Shahi's 2% of Neelum is 4,430 people where linguists report a few
hundred speakers; Bhimber's 30% Dogri is high for the Pakistani side. The table is the only
district source and is drawn as the government prints it; `note_public` says it is an estimate.

## 5. Calls

- **In `pk`, not separate entries.** religiondots draws AJK inside `pk` (its `countries/pk.py`)
  and Natural Earth's Pakistan outline holds both areas, so not_drawn.py's hatching for them
  goes once `pk` covers them.
- **All Pahari varieties on one node, `pahari_pothwari`.** Table 15.31 is one Pahari column with
  the local variety written beside the figure; Glottolog files Chibhali, Punchhi, Pothwari, Mirpur
  Panjabi and Pahari as dialects of Pahari Potwari (paha1251); India's J&K Pahari is on the same
  node (in2011.py), so the colour runs across the Line of Control. Splitting would need five
  near-identical nodes from a whole-percent estimate.
- **Khowar under Dardic** as conventionally grouped (Glottolog: directly Indo-Aryan).
  **Burushaski** under `isolate`. **Kundal Shahi** under Dardic (Glottolog Shinaic).
- **GB's census totals win over MICS** where they differ; MICS only splits the census.
- **Colours** (`taxonomy/tree.d/pk.txt`): Burushaski hand-picked purple (#9f50ca), Khowar dark
  blue-teal (#116d8a), against Shina teal #33a6a0, Balti blue #488acb, Wakhi salmon #d08a7b.

## 6. Geography (`sources/pk_north_geo.py` -> `data/geo/pk/pk_hexes.gpkg`)

religiondots' pk2023 layer (146 districts, AJK included) copied, with GB appended: COD-AB v01's 14
GB districts folded to the census's 10 (Darel, Tangir into Diamer, COD's "Diamir"; Gupis-Yasin
into Ghizer; Rondu into Skardu), asserted both ways; Kontur 2023-11 hexes by centroid. None of
GB's 5,839 hexes was already in religiondots' layer.

Kontur under-reads GB: 1,150,007 against 1,709,030 (0.67), Gilgit 0.44 and Diamer 0.54, the rest
0.60-1.10; log correlation 0.92 against a best shuffle of 0.88 (ten units, a weak control). It only
moves dots inside a district, so the counts are unaffected.

**Line of Control.** India's 1:1,000 dots inside GB 2 and inside AJK 14; China's 0. But 105 of
Pakistan's AJK dots (of 4,333) fall inside India's placement polygons near Poonch and Rajouri
(lat 32.75-34.03): India's layer overruns the LoC a little there, onto AJK hexes that religiondots
already drew. It is India's layer's edge, not a gap; not changed here.

## 7. Room for improvement

- PBS's district Table 11 for GB and AJK, if it is ever printed (or the GB 2024 district
  brochures, if a full copy turns up), would replace both models with measured counts.
- GB MICS 2016-17 or 2024-25 microdata (UNICEF, registration) would give weighted district
  shares and could split "other" further.

## 8. Cut from note_public (2026-10-06 text sweep)

- Pahari varieties as Table 15.31 names them: Punchi in Poonch and Sudhnoti, Mirpuri in Mirpur
  and Bhimber, Chibali in Haveli, Dhundi-Khairali in Bagh; all on `pahari_pothwari`, the node
  India's J&K Pahari uses (§5).
- The yearbook gives the AJK table's source as the Kashmir Liberation Cell (§2).
