# Romania: RPL 2021, mother tongue

Built 2026-10-04 (session d9e44929-ro). Rebuild:

```
python sources/ro_census.py [--fetch]  -> data/normalized/ro.csv
python taxonomy/build.py
python sources/ro_geo.py               -> data/geo/ro/ro_hexes.gpkg
python tools/check_country.py ro
python scatter.py --country ro
```

Drawn: 16,551,437 people on 3,181 UATs (communes and towns), 22 nodes, 16,542 dots and 5 rings.

## 1. The table

INS (Statistics Romania), Recensământul Populaţiei şi Locuinţelor 2021, final results,
"caracteristici etno-culturale", table 2.3 "Populaţia rezidentă după limba maternă":

```
https://www.recensamantromania.ro/wp-content/uploads/2023/06/Tabel-2.03.1-si-Tabel-2.03.2.xlsx
```

407,822 bytes, no login, saved in `data/raw/ro/`. Sheet `Tab.2.3.1` by macroregion, development
region and judeţ; `Tab 2.3.2` by judeţ, municipiu, oraş and comună (3,181 UATs). Reference date
1 December 2021, resident population 19,053,815.

The question is mother tongue (limba maternă), one answer. The table names 22 languages, the
recognised minorities' languages plus Italian, Greek and Yiddish, then `Alta limba materna`
(other, 19,741) and `Informatie nedisponibila` (information not available, 2,502,378, 13.1%).

The layout is exactly religiondots' religion table 2.4 (religiondots/sources/ro.md), and
`ro_census.py` follows its parser: no codes (UATs keyed `JUDEŢ|NAME`), county headers told from
communes by name AND total (Călăraşi and Satu Mare are also commune names), Bucharest twice.

## 2. The 13% gap

`Informatie nedisponibila` is not a refusal. The 2021 census was built largely from
administrative registers, which hold no language, so for one person in eight the variable is
absent. Not drawn; it is the entry's `gap`. (Religion's same category in table 2.4 is 13.95%.)

Its share varies a lot by place, and with it how thinly a place is drawn: Bucharest 25.8%, Ilfov
18.1%, Iaşi 17.7%, Timiş 17.2%, Constanţa 16.1%, against Covasna 7.1% and Harghita 7.2%. The big
cities, where registers seem to miss most people, are drawn thinnest; the Székely counties,
mostly Hungarian, are drawn nearly whole. So Hungarian's 6.3% of the dots is probably a little
above its true share of Romania's people; nothing measures by how much.

## 3. Hidden cells (`*`), estimated

INS prints `*` for a confidential cell and `-` for a true zero inside the numeric columns. The
smallest published value at UAT and judeţ level is 3, so `*` hides 1-2 people, and further cells
of any size are hidden so the first cannot be worked out from the row total. One cell is `**`
(Oraş Ţăndărei, Polish), with no footnote; read as `*`.

```
uat      cells: 10,432 published, 6,096 hidden, 56,635 true zero
judet    cells:    716 published,   148 hidden,    102 true zero
uat      categories sum to 19,029,417 of 19,053,815: 24,398 people in hidden cells
judet    categories sum to 19,053,598 of 19,053,815: 217 people in hidden cells
country  categories sum to 19,053,815 exactly
```

Dropping them would cost the small languages most: Albanian 39% of its people, Macedonian 31%,
Italian 26%, Yiddish 21%, Greek 16%. So `impute()` estimates every hidden cell from the two
margins that are known exactly: the unit's total less its published cells, and the parent's
cell for that language less its published children. Judeţ cells first (from the national row),
then UATs one judeţ at a time, by iterative proportional fitting from 1 per cell. The estimated
rows are tier `derived`.

- Every language's UAT sum and judeţ sum equal the national row (asserted, to under 1e-8).
- Per judeţ and language, met exactly. Per UAT row, 362.5 people in all cannot meet both margins
  (Argeş 262, Botoşani 96, the rest under 2): the hidden-cell pattern there cannot satisfy both
  margins at once, so the estimate keeps the judeţ language totals and moves those people
  between UATs of the same judeţ. Mostly `Informatie nedisponibila`, which is not drawn.

## 4. Checks

- Units and population: 3,181 UATs, 42 judeţe, 1 country, each 19,053,815 (INS's figure).
- Categories never exceed a unit's total (0 violations).
- Second table: the UAT rows summed by judeţ never exceed `Tab.2.3.1`'s judeţ cell for the
  language (0 exceed; 1 judeţ cell, Bistriţa-Năsăud's Russian, is itself hidden and not compared).
- Geography (§6): Kontur against the census per UAT, log r = 0.982 against 0.080 for the best of
  500 shuffles.

## 5. Mapping (taxonomy/ro2021.py)

Every named label has its own node; no new nodes were needed (Rusyn from ca.txt, Crimean Tatar
from ua.txt, the rest mostly us.txt). The calls:

- **Tătară (13,805) is Crimean Tatar**, not `turkic.tatar` (Volga Tatar). The Tatars of Dobruja
  came from Crimea and the Nogai steppe; Glottolog files Dobruja Tatar (dobr1234) as a dialect
  of Crimean Tatar (crim1257), whose countries include RO.
- **Rusă (14,414) is Russian**: mostly the Lipovans of the Danube delta (Tulcea).
- **Ruteană (594) is Rusyn**, kept apart from Ukrainian (40,861), which the census prints apart.
- **Macedoneană (201) is Macedonian (South Slavic)**, as printed: Romania's recognised Macedonian
  minority is the Slavic one. Aromanian has no column; its speakers are in Romanian or `other`.
- **Romani (199,050)** on the leaf `romani.romani`, variety not stated (as Ukraine's).
- **Idiş** is Yiddish. **Alta limba maternă** is `other`; nothing says whether it is a regional
  language (Aromanian, Csángó) or a migrant one, so it is not split.

## 6. Geography (sources/ro_geo.py)

religiondots' `ro_uat.gpkg` (GISCO LAU 2021, `kod` = SIRUTA) and `ro_uat_lookup.csv`, read
only; the language table uses the same 3,181 `JUDEŢ|NAME` keys as the religion table, so the
lookup applies unchanged (asserted both ways). Kontur RO (2023) hexes keyed to the UATs by
centroid, as Poland's:

```
UAT area median 61.3 km², 83 Kontur hexes at the median
overlaps: 0 UATs cut (GISCO's Romanian UATs do not overlap)
Kontur RO: 137,471 hexes, 19,961,091 people; 766 hexes (75,318) outside every unit
Kontur / census nationally 1.044; per unit p10 0.86, median 1.01, p90 1.17
2 of 3,181 units outside a factor of 3: Oraş Bălan (Harghita) 0.10, Şelimbăr (Sibiu) 0.32
largest UAT, Bucharest: census share 0.090, Kontur 0.097
```

The low ones are new suburbs Kontur's older footprint misses (Şelimbăr by Sibiu, Moşniţa Nouă
by Timişoara, 0.38) and Bălan, a shrunken mining town. Their dot counts still come from the
census; only the placement inside them leans on fewer hexes. The 766 outside hexes are left out
(border overlap with neighbours' extracts and the coast); they carry no census people, only
weight. No Kontur cap block stopped the scatter.

Bucharest is one UAT of 1.7 million: INS publishes no language by sector.

## 7. Colour

Hungarian was generated as a mid blue (#1f99c7) beside German's #359bd9, and the two share towns
across Transylvania and the Banat, so `tree.d/ro.txt` makes Hungarian a lighter sky blue
(0.82 0.11 225, #6cd3fa).

Not fixed, because the nodes are hand-coloured in other countries' fragments: in Dobruja,
**Romanian (#ef7e80, us.txt), Turkish (#ea6878, us.txt) and Crimean Tatar (#e45268, ua.txt) are
three near-identical reds.** Turkish (17,101) and Tatar (13,805) sit among Romanian in Constanţa
and Tulcea and will not read apart. Crimean Tatar is the cheapest to move (only Ukraine and
Poland use it; in Crimea its neighbours are Russian green and Ukrainian yellow-green, so a
darker brick or an orange would also work there).

## 8. Not done

- Aromanian, Csángó Hungarian and Lipovan Russian as distinct varieties: the table has no
  column for them.
- No look at the other 2.x tables (mother tongue by ethnicity exists at some level in the
  release; not opened).
- No screenshot: the tiles are built by the supervisor's build tail. Worth a look at Dobruja
  (§7) and at Harghita/Covasna against Mureş once they are.
