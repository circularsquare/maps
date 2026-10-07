# Ukraine: 2001 census, native language by raion and city

Drawn 2026-10-04 (session d9e44929-ua). 48,240,902 people counted in 672 units.

**2026-10-05, session `d9e44929-crimea`: Crimea and Sevastopol are drawn from Russia's 2021
census** (Anita: "Maybe Russia is better to use here. It has been de facto for a while."). The 26
Crimean units, the Autonomous Republic's 25 raions and cities (`UKR_01_*`) and Sevastopol
(`UKR_02_01`), 2,401,209 people in 2001, are left out of what Ukraine draws. `ua.csv` still holds
them (the normaliser's checks run over the whole census); `countries/ua.py` drops them, and
`ua_geo.py` cuts their hexes out (see Geography). Now drawn: **45,651,105 people (188,588 did
not state a language), 646 units, 26 nodes; 45,640 dots, 1 ring.** After the scatter, 0 `ua`
dots fall inside Crimea's polygons. `gap` says: "Crimea and Sevastopol (2.4 million people in
2001), drawn from Russia's 2021 census, which has counted them since 2014"; `note_public` ends
with the same fact. The national 67.5% / 29.6% in `note_public` stay: they are the census's own
figures for the whole country. Crimean Tatar drops from 231,345 drawn to 1,072 (the mainland's); Karaim from 57 to 3; Russian from 14,273,670 to
12,382,710. The figures in the sections below are the 2001 census's as first built, Crimea
included.

## Source

- **Table A, the one drawn.** State Statistics Committee of Ukraine, "Розподіл населення регіонів
  України за рідною мовою у розрізі адміністративно-територіальних одиниць" (distribution of the
  population of the regions by native language, by administrative unit),
  `http://2001.ukrcensus.gov.ua/i/u/popul_adm_00.zip`, linked from
  `/results/nationality_population/`. 27 .xls files, one per region, down to every village. Each
  row gives the share (%, two decimals) naming each of 17 languages. Shares only: no counts and
  no "other" or "not stated" column. "-" is nobody, "0.00" is under 0.005%. A few cells use
  U+05BE instead of "-" for nobody.
- **Table B, counts and boundaries.** U.S. Census Bureau, "Ukraine Subnational Population and
  Housing Data Tables with Administrative Boundaries" (HDX, CC BY, release `uscb_201905`):
  `ukraine_uscb_201905.xlsx` and `ukraine.gdb.zip`. It transcribes Ukrstat's census database
  ("Distribution of the population by nationality and native language, ... oblast"). Used for
  the 2001 population per unit, exact Ukrainian and Russian counts, not stated, the
  nationality x own-language counts, and the 2001 polygons.
- **Question.** Native language (рідна мова), one answer. In the ex-USSR tradition it leans
  towards identity rather than use. Ukrstat's own summary page
  (`/results/general/language/`) gives 14.8% of ethnic Ukrainians naming Russian.
- **Vintage.** 2001 is the only census since independence; none has been held since.
- **Grain.** 661 raions and cities of oblast significance as of 2001, Kyiv's 10 districts, and
  Sevastopol as one unit: 672 units, 72,000 people on average. Table A goes down to villages
  (about 29,000 rows), but there are no open 2001 polygons for village councils, and the shares
  would need each settlement's population as well. That is the next step if anyone wants finer.
- **Sites.** `2001.ukrcensus.gov.ua` answers. `database.ukrcensus.gov.ua` (the census database
  USCB cites) returns 404 on every path tried, and `db.ukrcensus.gov.ua` returns a blank IIS page.

## Method (`sources/ua_c01.py`, docstring has the detail)

Per unit, with T its 2001 population from B:

1. Ukrainian and Russian: B's exact counts (`measured`). One raion has Russian suppressed
   (Pidhaietskyi), so it uses A's share.
2. The other 15 languages: T x A's share (`derived`).
3. Not stated: B's `Unstated` (not drawn). Where B suppressed it (1-9 people), 0.
4. The remainder r = T minus all of the above. From it come the people of eight nationalities
   who named their own nationality's language, using B's `Nationality-Language` sheet
   (`NL_NTV_<x>`, census counts): Tatar 24,397, Azerbaijani 22,604, Georgian 10,991, Turkish
   7,756, Arabic 3,776, Vietnamese 3,490, Uzbek 2,327 and Korean 1,813. These are the
   nationalities with an unambiguous own language and at least 1,800 people. None was capped by
   the remainder. What is left, 67,956, is drawn on `other`.
5. A's two-decimal rounding left one unit's remainder short by 1 person, which was taken off
   not stated.

**Kyiv.** B has languages only for the whole city, while A has shares per district. Kyiv is
built as one unit and then each language is shared over the districts by A's share times B's
district population. The remainder categories are shared by each district's unprinted share.
District totals come out within 198 people of their census populations. All Kyiv rows are
`derived`.

**Sevastopol** is one unit: A splits it into city districts that have no polygons in B.

### Why B's other language columns are not used

B's `Language` sheet looks like a counts table, but only its Ukrainian and Russian columns are
complete. Its Belarusian, Moldovan, Romanian, Hungarian, Slovak and Crimean Tatar columns fall
short of A in 802 unit cells. For example, Biliaivskyi raion has 0 Belarusian in B against 0.19%
(199 people) in A, and B's national Belarusian is 16,921 against 56,200 from A. B is missing
part of each language wherever a component it summed was suppressed, and it does not flag those
cells with -999. Its `Other language` column is not the remainder either: it leaves out some
nationalities' own languages (the 2,231 Meskhetian Turks naming Turkish in Chaplynskyi raion
are in no column). Anyone reusing this USCB release for language should know this.

## Checks (all pass; `python sources/ua_c01.py`)

1. B's Language sheet has only the -999 sentinel. Its units sum to 48,240,902, the census total.
2. A has 27 region files. Raion/city rows per region equal B's ADM2 count in all 25 regions. The
   one extra row is Prypiat, which is all dashes.
3. The join A to B is one to one: 661 pairs, 0 duplicate keys, 0 left over. It uses a
   transliteration of A's Ukrainian name against B's `NSO_NAME`, within the region, with
   three pins:
   - `UKR_24_20` Volodymyr-Volynskyi raion. B's `NSO_NAME` repeats the city's name; its
     `AREA_NAME` and `USCBCMNT` say raion.
   - Selidove and Sloviansk. B spells them SELIDOVE and SLOVIANSK, A spells them Селидове and
     Слав'янськ.
   - Kyiv's Podilskyi district, which B spells PODOLSKYI.
4. **The independent witness.** All 1,321 of B's Ukrainian and Russian counts agree with A's
   share within 0.01 points. All but one also agree within A's rounding of 0.005: Kirovohradskyi
   raion (`UKR_12_07`) Russian is 10.42 in A and 10.4255 in B, a 2-person difference.
5. Kyiv's 10 district populations sum to the city's 2,566,953.
6. Every unit sums to its population, and the country sums to 48,240,902.
7. Printed: A's count per language against B's people of that nationality naming their own
   language. A's should be the larger and close to it. Bulgarian 134,569 against 130,066 (1.03),
   Armenian 1.04, Gagauz 1.06, Polish 1.09, Romani 1.11, Greek 1.03, Belarusian 1.03, Moldovan
   1.03, Romanian 1.03, Crimean Tatar 1.01, Hungarian 1.08 (Zakarpattia's Roma, 8,873 naming
   Hungarian, explain it), German 1.32, Karaim 57 against 42. Every ratio is at least 1.

National totals as drawn: Ukrainian 32,577,468 (67.53%) and Russian 14,273,670 (29.59%), which
are Ukrstat's headline figures. Then Crimean Tatar 231,345, Moldovan 185,259, Hungarian 161,335,
Romanian 142,296, Bulgarian 134,569, Belarusian 56,200, Armenian 51,924, Gagauz 23,346, Romani
22,875, Polish 19,139, Greek 5,589, German 4,044, Yiddish 3,096, Slovak 2,629 and Karaim 57.

## Labels and calls (`taxonomy/ua2001.py`, `taxonomy/tree.d/ua.txt`)

- "єврейську" ("Jewish") is drawn as **Yiddish**: the Soviet census term is Yiddish, and this
  table has no Hebrew line. 3,096 people.
- "грецьку" is drawn as **Greek**, though some Mariupol Greeks speak Rumeic or the Turkic Urum.
  The census printed one label.
- "циганську" (Romani) gets a new leaf `romani.romani`. Glottolog's Romani is a family and the
  label names no variety, but it is a named answer and so not on the washed-out group node.
- **Moldovan and Romanian are two nodes.** Glottolog files Moldavian as a dialect of Romanian,
  but the census asked them apart, and Chernivtsi and Odesa oblasts have both.
- New nodes, all checked against Glottolog: Crimean Tatar, Gagauz, Karaim and Tatar (Turkic,
  flat under `turkic`), Moldovan, and Romani (the leaf). Belarusian, Azerbaijani and Uzbek are
  repeated from `ca.txt`, identically, because Canada is not finished.
- **Colours.** Crimean Tatar is red and Gagauz magenta, both in Turkic's part of the wheel.
  Moldovan is amber beside Romanian's salmon. Bulgarian was generated olive, too close to
  Ukrainian's yellow-green across the Budjak, so it is hand-picked dark blue-green. Bulgarian
  was uncoloured in `us.txt`, and `ua.txt` is read first, so this is now its colour everywhere.
  **Ukrainian (#bfd869) and Russian (#54b85b) are `us.txt`'s colours**. They are told apart by
  lightness more than hue. A Ukraine render shows the Donbas clearly greener, but they are the
  pair this map is about, and changing them means editing `us.txt`.

## Geography (`sources/ua_geo.py`)

- **Units.** USCB `UA_GEOG_ADM2_2001_uscb_201905`: 672 polygons keyed by `GEO_MATCH`, the same
  key as the counts. The join is asserted both ways. These are the 2001 raions (EuroGlobalMap
  v5.1 lines), which the table was published on. The 2020 reform merged them into 136, so
  religiondots' 27-oblast COD-AB layer is not reused. Crimea and Sevastopol were drawn here at
  first, as in religiondots (spec ruling, ask 016); since 2026-10-05 they are Russia's (above).
- **The Crimea cut.** Hexes are still assigned over all 672 units and coast-snapped, so the line
  at Perekop and Chonhar is unchanged, and then split: `data/geo/ua/ua_hexes.gpkg` keeps the 646
  mainland units (285,296 hexes, 35,815,369 Kontur people) and `data/geo/ua/crimea_hexes.gpkg`
  takes the 26 Crimean units (11,767 hexes, 2,047,033 people) for `sources/ru_geo.py`. Asserted:
  the Crimean units by id are exactly the units of the two Crimean regions by name; no hex is in
  both files (and again in `ru_geo.py`, with the boundary checks).
- **Placement.** Kontur 2023-11-01 hexes are given to the unit holding their centroid, read in
  place from religiondots' copy. Kontur over the census is 0.782 nationally (Kontur is
  pre-invasion but 22 years after the census). Per unit, normalised, p10 0.80, median 0.97 and
  p90 1.19. The log correlation is 0.952 against a best of 0.119 over 500 shuffles. 7 units are
  outside a factor of 3, all small cities whose generalised polygon is too tight (Pervomaisk,
  Luhansk 0.11; Vuhledar 0.13, 1.5 km2; Yuzhne and Teplodar 0.22) or raions holding a city's
  edge (Shakhtarskyi 2.99, Kakhovskyi 2.40). Their dots are tight or spread inside the right
  unit; left as is.
- **Coast snap.** 1,670 hexes (126,337 people) had centroids outside every unit, 99% within
  1 km, on EuroGlobalMap's generalised coast and river lines (Sevastopol, Berdiansk, Yalta,
  Odesa, the Danube, the Tisza). The Kontur UA extract holds only Ukraine, so 1,644 of them
  (125,705 people) were snapped to the nearest unit within 2 km. 26 (632 people) were dropped.
- The scatter stopped on no Kontur cap block, and `kontur_cap.csv` has no rows added.
