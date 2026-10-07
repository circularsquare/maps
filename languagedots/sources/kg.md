# Kyrgyzstan (`kg`): 2022 census native language, rayons and cities

Drawn 2026-10-04 by session `d9e44929-kg`. No asks filed.

Files: `sources/kg_census.py` (normaliser), `sources/kg_geo.py` (units, lookup, placement),
`taxonomy/kg2022.py`, `taxonomy/tree.d/kg.txt`, `countries/kg.py`. Outputs
`data/raw/kg/` (nine Book III PDFs, Book II PDF and xlsx annex), `data/normalized/kg.csv`,
`data/geo/kg/kg_units.gpkg`, `kg_lookup.csv`, `kg_hexes.gpkg`, `data/geo/kontur/` (the KG Kontur
extract, copied from religiondots' raw folder), `data/processed/dots_kg.geojson` (6,927 dots),
`rings_kg.geojson` (13).

## 1. The table

*Perepis' naseleniya i zhilishchnogo fonda Kyrgyzskoy Respubliki 2022 goda*, **Book III**, "Regiony
KR": one volume for each of the seven oblasts and for Bishkek and Osh city (National Statistical
Committee, 29 December 2023 to 26 January 2024). **Table 3.4**, ethnic groups by native language.
Question 10.1 of the census form, "Vash rodnoy yazyk", one answer (the form is printed at the back
of each volume). Rows: for the oblast and for each rayon and city of oblast significance, the
whole population, then "v tom chisle" the larger ethnic groups of that unit (two to twenty rows).
Columns: the language of one's own ethnic group, Kyrgyz, Russian, and in some volumes Uzbek and
Dungan, then other. No "not stated" column: every row sums to its population.

Where the volumes are: the coverage sweep found only Naryn and Bishkek, linked from the English
Book I page. The **Russian** Book I page
(`stat.gov.kg/ru/publications/perepis-...-kniga-i-osnovnye-...`) lists the whole archive: all
nine Book III volumes, Book II (national tables, with an xlsx annex), Books V and VI. URLs are in
`kg_census.py`.

## 2. Checks (all asserted in `kg_census.py`)

1. Every row's columns sum to its whole-population figure, and each group's own-language column
   sits where the table prints "-" in that language's column (Kyrgyz row, Kyrgyz column, and so
   on). That pins each volume's column order; Osh oblast prints Uzbek before Russian.
2. Each region's units sum to the region row in every column (Osh city's urban and rural parts to
   the city); the nine regions sum to 6,936,156.
3. No unit's remainder (whole population minus the groups it lists) is negative in any column.
4. **A second publication of the same census:** Book II table 3.8 (xlsx annex) prints the same
   split for every oblast and the country. Every oblast row and every group row both print agree to
   0.0024% of the oblast: Batken 8 people between own language and other, Chui 7 between the same
   two, Issyk-Kul's Tatar row 13.

Printing faults found and handled, each asserted where handled:
- Batken's oblast row prints own 548,760 and other 954; its units sum to 548,752 and 962, which is
  also Book II's figure. Units kept.
- Naryn's oblast block prints a Dungan column and also keeps Dungan inside "other" (the oblast row
  overruns its total by the Dungan cell, 2), and its group rows are garbled beyond that (the Kazakh
  row is Naryn city's). Dungan is taken out of "other" on the oblast row, the block's group rows
  are dropped. Nothing is drawn from an oblast row.
- Stray "-" lines between labels (Osh oblast), numbers without thousands spaces (Jalal-Abad), page
  numbers at the foot of pages (Issyk-Kul).
- Sub-rows not used: Karakol without its urban-type settlements, Jalal-Abad without its villages,
  Osh city's urban and rural parts.

## 3. Result

6,936,156 people, 30 nodes, 53 placement units. Kyrgyz 5,527,168 (79.7%), Uzbek 883,838 (12.7%),
Russian 304,465 (4.4%), Dungan 61,770, Tajik 53,205, other 21,423, Uyghur 17,112, Kazakh 15,143,
Azerbaijani 12,619, Turkish 11,534, Kurdish 9,052, India and Pakistan 5,688, Tatar 4,604, Korean
2,191, Dargwa 1,527, Lezgian 1,157, Karachay-Balkar 1,015, Ukrainian 752, Chechen 555, German 512,
Avar 309, Aghul 256, Turkmen 88, Kalmyk 40, Chinese 37, Belarusian 29, Romani 23, Kumyk 21,
Bulgarian 17, Moldovan 6. Bishkek is 86.1% Kyrgyz and 10.9% Russian; Osh city 64.4% Kyrgyz and
32.9% Uzbek.

Who named another group's language (Book II, national): Uzbeks 113,385 of 986,881 Kyrgyz (11.5%);
Kazakhs 11,242 of 28,244 Kyrgyz (39.8%); Turks 8,840 of 22,074 Kyrgyz (40.0%); Uyghurs 7,276 of
31,559 Uzbek (23.1%, almost all in Osh oblast) and 3,649 Kyrgyz; Koreans 3,229 of 5,900 Russian;
Tatars 4,365 of 11,219 Russian; ethnic Kyrgyz 9,332 of 5,379,020 Russian.

The own language of groups a unit does not list is 7,071 people, 0.10%, on `other`. That was small
enough that no model was built to name it from the oblast rows, which list more groups.

## 4. Geography

COD-AB Kyrgyzstan (religiondots' `data/raw/kg/shp`, read-only): 53 admin-2 rayons and cities, plus
Bishkek and Osh city from admin 1 (no admin 2 there). Join by Russian name inside each oblast,
asserted 1:1. Two name differences: Talas's Aitmatov rayon is COD's Kara-Buura (renamed 2022), and
Kara-Kul (below). COD's Kok-Zhangak city has no census row (inside Suzak in the table) and is
dissolved into Suzak.

Kontur 2023 r8, centroid-keyed (`_grid.hex_layer`): 6,951,786 Kontur people, 282,121 of them in
hexes outside every unit (the Uzbek and Tajik enclaves in Batken and the spill over the borders).
Kontur over census, normalised: median 1.00, p10 0.47, p90 1.49; log correlation 0.870 against a
best of 0.394 over 500 shuffles.

- **Kara-Kul is drawn with Toktogul.** COD's Kara-Kul city is 1.2 km2, the dam town's core; Kontur
  put 0.02 of its census people (normalised) in it, so its dots would have stacked on one or two
  hexes. Both census rows go to Toktogul's pcode and the two polygons are dissolved.
- **Five cities sit at 0.24-0.33**: Kyzyl-Kia, Batken, Karakol, Balykchy, Naryn city. The census
  city includes the settlements under it (Karakol's sub-row "without urban-type settlements" says
  so), and COD draws the town. Their dots are denser than Kontur's there; left as is.
- **Bishkek is one unit.** Its four districts are in `kg.csv` (`geo_level` district) but not in
  COD-AB. OSM's district relations were tried through Overpass (overpass-api.de and
  overpass.kumi.systems, both 504 on 2026-10-04) and not pursued further. A later pass could add
  them; the counts are ready.
- No Kontur cap blocks stopped the scatter (none registered for KG in either project).

## 5. Calls someone might reverse

- **Each group's own language is drawn as that language** (the census's "language of one's own
  ethnic group" column resolved per group, as Azerbaijan does): Turks -> Turkish (Meskhetian Turks),
  Kurds -> Kurdish, Karachays and Balkars -> Karachay-Balkar, Moldovans -> Moldovan, Roma -> Romani,
  Chinese -> Chinese (`sinitic`).
- **"Peoples of India and Pakistan" -> `other.india_pakistan`**, a named leaf under `other`, as
  Azerbaijan's "Jewish": the census names no language and Indo-Aryan and Dravidian share no node.
  5,688 people, mostly in Bishkek, Osh and Chui.
- **"Other languages" stays on `other`** (14,352). In the four volumes without an Uzbek column
  (Issyk-Kul, Talas, Chui, Bishkek) it holds Uzbek; Book II's oblast rows put that at 837 people.
- **Dungan is a sibling of Mandarin** (Glottolog files it under Mandarin; `mandarin` is a leaf
  other countries use). Aghul under Lezgic.
- **Colours**: Kyrgyz (0.64 0.20 25, red), Uzbek (0.74 0.15 345), Kazakh (0.84 0.09 15) and Uyghur
  (0.76 0.14 55) hand-picked in `tree.d/kg.txt`; generated, all four were the same pink. These
  nodes are used elsewhere only for migrants (ca, cz, pl, ua).

## 6. Not done, and noticed

- Book III's tables 3.3 (ethnic group by age per rayon) and 3.5 (second language) are unused.
- Turkish (0.70 0.16 330) and the new Uzbek sit close; Turks are 11,534 and live among Uzbeks and
  Kyrgyz in Jalal-Abad, Osh and Chui. Not changed, since Turkish's colour belongs to cz.
