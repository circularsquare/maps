# Russia (`ru`): 2021 census native language, by federal subject, urban and rural

Drawn 2026-10-05 by session `d9e44929-ru2`, restarting a session (`d9e44929-ru`) that was stopped
before it wrote anything to disk (nothing under `sources/ru*`, `taxonomy/ru*`, `data/*/ru*` or
`countries/ru.py` existed; the handoff said so).

Files: `sources/ru_census.py` (normaliser, `--fetch`), `sources/ru_geo.py` (placement layer),
`taxonomy/ru2021.py`, `taxonomy/tree.d/ru.txt`, `countries/ru.py`. Outputs
`data/raw/ru/` (Tom5_tab6, Tom5_tab7, the Institute of Linguistics' list, the methodology PDF),
`data/normalized/ru.csv`, `data/geo/ru/ru_grid_3km.gpkg`, `dots_ru.geojson` (130,474 dots),
`rings_ru.geojson` (64).

**2026-10-05, session `d9e44929-crimea`: Crimea and Sevastopol are now drawn from this census**
(Anita: "Maybe Russia is better to use here. It has been de facto for a while."). Before, they
were written as `excluded` and drawn from Ukraine's 2001 census. §3 "Crimea and Sevastopol" has
what changed; the figures below include them unless they say otherwise.

## 1. The table

Rosstat, *Itogi Vserossiiskoi perepisi naseleniya 2020 goda* (the census was postponed to
October-November 2021), **Volume 5 "Natsional'nyi sostav i vladenie yazykami", Table 6 "Naselenie
po rodnomu yazyku"**, `Tom5_tab6_VPN-2020.xlsx`, published 31.12.2022, from
https://rosstat.gov.ru/vpn/2020/Tom5_Nacionalnyj_sostav_i_vladenie_yazykami. (The page's file
numbers run one ahead of the table numbers: Table 6 is `tab6`, but `tab7` is Table 7 and `tab5`
is Table 4; the page's labels sit after their links.) Rosstat's certificate is from the Russian
national CA, so the fetch skips verification.

- **Question:** native language (rodnoi yazyk), ONE answer per person: in every sheet the language
  rows sum exactly to "stated a native language".
- **Grain:** 85 subjects (the census counted Crimea and Sevastopol), plus Arkhangelsk and Tyumen
  "without the AO", each split urban / rural. Nothing finer exists for the whole country. The
  coverage sweep's lead that some regional offices print native language by municipality (Perm,
  Novgorod) is true but uneven; mixing a few municipal subjects into a subject-level map was not
  worth it.
- **Labels:** 176 language groups reach the 85 subjects drawn (adding Crimea brought no new
  label), plus "other answers" (34,340 nationally) and "native language not stated". A "Maori" row (6 people) sits on the Tyumen sheet
  but in none of its three parts, and "Gabonese" (2) on the federation sheet only; neither is in a
  drawn unit.
- **Not stated:** 16,638,532 nationally (11.3%), all of it in the 85 subjects drawn (233,499 of
  it in Crimea and Sevastopol, 9.4% there). The 2021 census
  filled about 16 million people from administrative records with no questionnaire answers, so
  the gap is geographic: Khanty-Mansi 26.6%, Moscow 23.8%, Komi 21.7%, Yamal 20.3%; Tatarstan 2.9%,
  Bashkortostan 2.0%, Chechnya 1.2%, Chukotka 0.8%.

## 2. Checks (all asserted in `ru_census.py`)

1. Every one of the 88 sheets: language rows plus "other answers" = stated, for total, urban and
   rural; urban + rural = total on every row. Tyumen, the one sheet printing its whole population,
   has stated + not stated = population.
2. The 85 subjects (Arkhangelsk and Tyumen counted with their AOs, 82 sheets) sum to the Russian
   Federation sheet on 176 of 179 rows; the other three are a 6-person reshuffle (Maori 6 in the
   subjects; nationally 4 more "other answers" and 2 "Gabonese"), totals exact. Arkhangelsk without
   the AO + Nenets = Arkhangelsk; Tyumen without the AOs + Khanty-Mansi + Yamal = Tyumen (the same
   Maori reshuffle). Tolerance: 10 people per row, totals exact.
3. **A second table of the same census:** Table 7 (nationality by native language,
   `Tom5_tab7_VPN-2020.xlsx`) prints an all-population column per language per subject. It equals
   Table 6's total on all **9,244** (subject, language) cells the two share.

Drawn: 130,543,591 people with a native language in 85 subjects (168 units); Russian 111,546,569
(85.4%), Tatar 4,073,253 (3.1%), Chechen 1,644,313 (1.3%), Bashkir 1,319,650 (1.0%), Avar
907,966, Chuvash 800,100, Armenian 675,048, Kabardian 609,383, Dargwa 587,391, Kumyk 523,734.
`check_country ru` ok. Scatter: 130,474 dots, 64 rings; 69,591 people (0.05%) in languages under
one dot nationally.

## 3. Geography

religiondots' Russia layer, read-only: `religiondots/data/geo/ru/ru_grid_3km.gpkg`, Kontur H3 r6
hexes assigned and clipped to geoBoundaries' 83 ADM1 subjects (ISO 3166-2 ids), with `pop`.
religiondots' `sources/ru_geo.py` checked that join (Kontur 0.984x the census nationally, within a
factor of two per subject). The 83 are exactly the census's 85 less Crimea and Sevastopol (those two come from
Ukraine's layer, below), with
Arkhangelsk and Tyumen as their "without AO" sheets (the AOs are their own units in both). The
sheet-name to ISO table in `ru_census.py` is asserted both ways against the hex layer's units.

**Urban and rural** (`sources/ru_geo.py`): each subject's hexes are ranked by Kontur density and
the densest go urban until they hold the census's urban share of the subject (everyone, stated or
not); the crossing hex goes to whichever side is closer; at least one hex each side where the
census has people there. Result: 168 units (Moscow and St Petersburg are 100% urban in the
census, so no rural unit); mean gap between the census's urban share and the hexes' 0.2 points,
worst Altai Republic 2.5 points (two urban hexes). This is a density stand-in for Rosstat's
administrative urban/rural line, not that line; a 36 km² hex holding a small town and its
villages goes wholly one way. Worth it because the split is large for exactly the minority
languages: nationally 37% of Tatar speakers are rural, 53% of Bashkir, 61% of Chuvash, 62% of
Chechen, against 24% of Russian.

Scatter's water clip left one unit unclipped (over 95% sea); since Crimea's 400 m hexes were
added it reports 54 (Ukraine's layer, with the same coast-snapped hexes, reports 52). Not traced.

**Crimea and Sevastopol** (2,482,450 people with a row in Table 6; 2,248,951 with a native
language) are drawn from this census since 2026-10-05, on Anita's ruling. Until then they were
written as `excluded` and drawn from Ukraine's 2001 census. Now:

- `ru_census.py` writes them as geo_level `subject`, keyed by their ISO 3166-2 codes UA-43
  (Republic of Crimea sheet) and UA-40 (Sevastopol sheet), urban and rural like every subject.
  The rows were always in Table 6 and passed the three checks; only the level changed.
- **Units.** geoBoundaries' Russia, and so religiondots' layer, has no Crimea. The polygons are
  the U.S. Census Bureau's 2001 Ukrainian units that `ua` draws on: the Autonomous Republic's 25
  raions and cities merged to UA-43 (26,067 km²), Sevastopol's one polygon is UA-40 (860 km²).
  The 2001 raion lines inside the peninsula do not matter here (they are merged away); what does
  is the outer line and the Crimea/Sevastopol line, and Sevastopol as Rosstat counts it is the
  territory of the former Sevastopol city council.
- **Hexes.** Kontur UA's 400 m hexes as `ua_geo.py` assigns them over all 672 Ukrainian units
  (centroid in polygon, then the 2 km coast snap), cut out into `data/geo/ua/crimea_hexes.gpkg`
  and re-keyed: 11,767 hexes, 2,047,033 Kontur people (0.82x the census, against 0.98x for the
  rest of Russia; Kontur's 2023 Crimea figure is older input). Finer than the 3 km hexes elsewhere,
  which changes nothing but how tightly the dots sit. Crimea's urban share 0.505 in the census,
  0.505 on hexes; Sevastopol 0.923 and 0.923.
- **Asserted in `ru_geo.py`:** the merged subjects' areas equal their members' (no overlap
  inside); 0.0000 km² shared with Ukraine's mainland units and 162.7 km of boundary along them
  (no gap at Perekop, Chonhar and the Arabat Spit); UA-43 and UA-40 touch and do not overlap; no
  hex is in both the Crimean and Ukraine's mainland layer; religiondots' Russia hexes cover
  0.000 km² of Crimea's polygons (Krasnodar does not reach across the Kerch Strait).
- **After the scatter:** 2,233 `ru` dots inside Crimea's polygons and 0 `ua` dots.
- **What changed in the languages there:** Crimean Tatar 205,893 and Tatar 58,591 in 2021,
  against 230,273 Crimean Tatar and 8,880 Tatar in Ukraine's 2001 count. Russian 1,905,275,
  Ukrainian 57,321 (228,250 in 2001). The Tatar figure is the census's label as printed; it is
  widely read as many Crimean Tatars answering "Tatar", but nothing in the table says which, so
  it is drawn as Tatar like every other subject's.
- Wording: `grain` names 85 subjects "as the census counted them, Crimea and Sevastopol
  included"; `note_public` says Crimea and Sevastopol "are drawn from this census, which has
  counted them since 2014" (Rosstat's 2014 Crimean census, then 2021), with the two Tatar
  figures. Nothing more: the map takes no position beyond which census it draws.

## 4. Mapping calls (`taxonomy/ru2021.py` has each)

Rosstat publishes, beside Volume 5, the Institute of Linguistics' *Spisok yazykov Rossii v itogakh
VPN-2020* (Koryakov and Davidyuk; `data/raw/ru/Tom5_Spisok_yazykov.doc`, read with antiword),
which says what each census group holds. Used as evidence for the labels, not to redraw them:
the map draws the census's groups as printed.

- **"Mordvin" (274,876)** beside Erzya (46,241) and Moksha (23,178), and **"Mari" (318,495)**
  beside Hill Mari (18,332) and Meadow-Eastern Mari (225): the Institute makes both macrogroups
  because most speakers name only the macrolanguage. Each label is its own leaf; the unspecified
  ones are `uralic.mordvin` and the existing `uralic.mari`.
- **"Adygsky" (6,923)**: the generic Circassian self-name, split by the Institute between Adyghe
  and Kabardino-Cherkess. Leaf `abkhazadyghe.circassian`. "Kabardino-Cherkess" is Kabardian.
- **"Dagestani" (34,244)**: no such language; the Institute spreads it over Avar, Aghul, Dargwa,
  Kumyk, Lak, Lezgian, Rutul and others, Turkic and Daghestanian, so the narrowest node is the
  root: a leaf `other.dagestani` (az.txt's "Jewish" precedent).
- **"Jewish" (3,675)**: Yiddish or Juhuri; the existing `other.jewish`.
- **"Tat" (720)**: in Russia mostly Juhuri (Soviet usage called it Tat); `indoeuropean.iranian.tat`.
- **"Turkic" (946)**: the Meskhetian Turks' name for their language (Institute); `turkic.ahiska`.
  "Turkish" (115,838, many of them Meskhetian Turks) stays Turkish.
- **"Bulgar" (157)**: mostly a name for Tatar per the Institute; drawn as its own Turkic leaf.
- **"Chinese"**: `sinotibetan.sinitic`, unwashed, as au, ca and cz.
- **"Eskimo" (816) and "Yuit" (1)**: two leaves for Siberian Yupik.
- Nogai-Karagash and Yurt Tatar (Nogai dialects to the Institute), Teleut, Tofa, Chelkan
  (dialects in Glottolog), Hill and Meadow-Eastern Mari: each printed as a language, each a leaf.
- "Other answers": `other`. "Not stated": the gap.

## 5. Tree and colours (`taxonomy/tree.d/ru.txt`)

New: Andic (8) under Avar-Andic, a Tsezic branch (5), Lak, Tabasaran, Rutul, Archi; Abaza,
Circassian; 13 Turkic leaves; 12 Uralic leaves; 8 Tungusic; three new roots, one per family
(Chukotko-Kamchatkan, Yukaghir, Yeniseian), with colours; Nivkh under `isolate`; Siberian Yupik,
Yuit, Mingrelian, Church Slavonic (directly under Slavic so that South Slavic gains no uncoloured
member), Russian Sign Language, Dagestani. Families checked in Glottolog `languages.csv`.

Hand-picked colours for the Volga-Urals, Dagestan and the North Caucasus, and Siberia (the
fragment lists them). Adding uncoloured siblings moves build.py's generated colours, so the
effect was measured by colouring the tree with and without `ru.txt` on the tree as it stood
(another session edited build.py during this one): the only nodes another country draws that
moved were Azerbaijani, Karakalpak, Manchu and Karaim, now frozen at their prior values at the
bottom of the fragment, and the `other.*` greys, which have no chroma and so do not visibly
change. Not checked on the rendered map: whether the generated Andic and Tsezic colours read
apart from Avar inside western Dagestan.

## 6. What the map cannot show

Every dot in a subject's towns comes from one mixture and every dot in its countryside from
another. Nothing places a language inside a subject: Tatar villages in western Bashkortostan and
Bashkir villages in the east are drawn from the same rural mixture. tochno.st publishes the
census's nationality by settlement (CC BY), which would place languages tied to a nationality far
better, but that is an ethnicity proxy for placement, which is Anita's to allow; not done, and no
ask filed since nothing is held back.
