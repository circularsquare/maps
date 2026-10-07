# US Virgin Islands (vi): record

Drawn 2026-10-05 (session edd42a8c-vi). 2020 Island Areas Census, language spoken at home, 89
block groups with people, 80,430 people aged 5+ in households drawn (87,146 counted); placed on
Kontur hexes cut to the 2020 block groups.

Files: `sources/vi_census.py` (counts), `sources/vi_geo.py` (placement layer),
`taxonomy/vi2020.py`, `taxonomy/tree.d/vi.txt` (no new nodes; bare repeats), `countries/vi.py`,
`data/normalized/vi.csv`, `data/geo/vi/vi_bg.gpkg`, `data/raw/vi/` (DHC zip, table matrix,
geographic header, DCT list of tables, readme, TIGER block groups), Kontur VI in
`data/geo/kontur/`.

## 1. What exists

- **DHC summary file**, www2.census.gov, no key (api.census.gov wants one). Language appears in
  PBG5 (block group), PBG6 (household language, households), PCT22-PCT24 (tract and up) and the
  Detailed Cross-Tabulations CT25, CT46, CT58, CT67, CT74, CT89. All use the same four groups:
  speak only English, Spanish, "French, Haitian, or Cajun", other languages. No finer language
  list is published for the USVI at any level. The coverage sweep's lead (home language,
  estate/subdistrict) was right on the question; the finest grain is the block group, not the
  estate (the 090/091 summary levels are block-group parts, not estates).
- **Question**: does this person speak a language other than English at home; if so, which. One
  answer per person; English-only is "English", a Spanish-and-English home is "Spanish".
- **Nothing suppressed**: no "." in PBG5, PCT22 or P1 at any level (Guam had six suppressed
  tracts; the script stops if any appear).

## 2. Checks (sources/vi_census.py)

1. PBG5 and PCT22 add up in every record (groups to bands, bands to total).
2. The 92 block groups (PBG5) summed per tract equal PCT22 in all 32 tracts and all four groups,
   exactly; both tables sum to the three islands and the territory.
3. Block groups' people sum to the territory's 80,430, and their total population (P1) to the
   territory's 87,146, so no block group is missing from the header.

Territory: English only 56,173 (69.8%), Spanish 13,807 (17.2%), French/Haitian/Cajun 7,101
(8.8%), other 3,349 (4.2%). St Croix Spanish 21.7%; St Thomas French/Haitian/Cajun 11.7%.

## 3. Mapping

- **Speak only English** on English. It holds Virgin Islands Creole English (Crucian, St
  Thomian), which the questionnaire cannot tell apart; said in `note_public`.
- **Spanish** on Spanish.
- **French, Haitian, or Cajun** (7,101) on `creole.french_based`, which the viewer draws washed as
  "language not named". The Bureau codes French, Cajun, Haitian Creole and Antillean Creole
  (Kweyol) here. PCT25 (place of birth) counts 4,329 people born in Dominica, 2,881 in St Lucia,
  2,397 in Haiti and only 719 born anywhere in Europe, so the group is overwhelmingly French-based
  creoles. Strictly, French puts the narrowest containing node at the root; the creole node was
  chosen because a root-level node would draw 9% of the territory as unclassified. Not split into
  Kweyol and Haitian by birthplace: that would change counts, which is a proxy for Anita to allow.
- **Other languages** (3,349) on `other`: no breakdown anywhere (likely Arabic from St Croix's
  Palestinian community, Indian languages, Papiamento, Dutch).
- Colours: all existing nodes; nothing hand-picked.

## 4. Geography and placement (sources/vi_geo.py)

- TIGER/Line 2020 block groups (92, three of them the water tracts 9900 with no people), clipped
  to the land of the cartographic-boundary 2020 tracts (religiondots' cb_2020_us_tract_500k,
  STATEFP 78, read only). Join both ways: all 89 block groups with people have a polygon; the
  three without are the water block groups.
- Kontur VI (706 hexes, 98,734 people; densest 3,388/km2, far from the cap) cut to the block
  groups and each hex's people shared over its land pieces by area: block groups median 1.9
  km2, p10 0.5 km2, too small for whole-hex assignment. 1,219 pieces; 17 Kontur people in hexes
  touching no block group.
- Witness, Kontur per block group against census 5+ in households: ratio 1.227, normalised p10
  0.54, median 0.95, p90 1.65; 4 of 89 outside a factor of 3; log r = 0.562 against a best of
  0.290 over 500 shuffles. Highest 780109714003 (24.4, St Croix south shore, the airport and
  refinery land Kontur counts as built-up); only placement inside a unit uses Kontur.
- Plain population weighting: no birthplace table is published below tract, and the block groups
  are already finer than tracts.

## 5. Scatter

79 dots at 1:1000; 1,430 people (1.8%) under one dot per language. scatter's water step reported
one unit losing over 95% to the sea and left unclipped (a coastal sliver piece).

## 6. Calls someone might reverse

- "French, Haitian, or Cajun" on French-based creoles rather than the root (section 3).
- English-only drawn as English though much of it is Virgin Islands Creole English.
- Block groups over tracts: finer, same four groups, sums verified.
