# Croatia: Popis 2021, mother tongue

Built 2026-10-05 (session d9e44929-hr). Rebuild:

```
python sources/hr_census.py [--fetch]   -> data/normalized/hr.csv
python taxonomy/build.py
python sources/hr_geo.py                -> data/geo/hr/hr_hexes.gpkg
python tools/check_country.py hr
python scatter.py --country hr
```

Drawn: 3,850,993 people on 556 towns and municipalities (Zagreb one of them), 25 labels on 25
nodes, 3,840 dots, 7 rings. Not drawn: 20,840 (0.54%), "Unknown", the gap.

## 1. The table

Drzavni zavod za statistiku (DZS), Popis stanovnistva, kucanstava i stanova 2021, the release
"Stanovnistvo po gradovima/opcinama" (population by towns/municipalities), one workbook:
https://podaci.dzs.hr/media/td3jvrbu/popis_2021-stanovnistvo_po_gradovima_opcinama.xlsx (18.3 MB,
browser User-Agent, TLS does not verify from this machine, as religiondots found). The same file
religiondots reads for religion (its sheet `2.`).

Sheet `4.`, STANOVNISTVO PREMA MATERINSKOM JEZIKU PO GRADOVIMA/OPCINAMA. The methodology sheet
says Croatian comes first, then the languages of the national minorities, then other languages
and unknown. In DZS's order: Croatian, Croato-Serbian,
Albanian, Bosnian, Bulgarian, Montenegrin, Czech, Hungarian, Macedonian, German, Polish, Romani,
Romanian, Russian, Ruthenian, Slovak, Slovenian, Serbian, Serbo-Croatian, Italian, Turkish,
Ukrainian, Vlach, Hebrew, "Other languages", "Unknown". Each as a count and a percentage.

**The question** (the workbook's methodology sheet): mother tongue is the language a person
learned in early childhood, or, where the household spoke several, the one the person considers
their mother tongue. One answer. No "did not declare" column; "Unknown" is the only remainder.

**Finer than municipality**: the companion settlements workbook (religiondots' `naselja.xlsx`)
has only age and sex by settlement, no language. Not pursued further.

## 2. Reading the sheet

The layout is religiondots' religion sheet exactly (`religiondots/sources/hr.py`): no codes,
rows identified by (zupanija, name), so `geo_id` is `ZUPANIJA|NAME`, the same keys religiondots
uses. Grad Zagreb appears only as its 17 gradske cetvrti (city districts), so the cover is 555
municipalities plus 17 districts. `-` is a true zero. The parser asserts the English half of all
26 column headers and of each percentage column beside them.

## 3. Checks (all pass)

- national total 3,871,833, as published.
- the 21 county rows, and the 572-unit cover, each sum to the national row in all 27 columns.
- the 26 categories partition every one of the 594 rows (DZS neither rounds nor suppresses).
- every percentage column agrees with count / total to DZS's two decimals.
- **join**: all 572 unit ids match religiondots' `data/normalized/hr.csv` (the religion sheet of
  the same census) both ways, with identical totals in every unit.

## 4. Mappings (`taxonomy/hr2021.py`)

| label | people | node | why |
|---|---:|---|---|
| Croatian | 3,687,735 | slavic.south.croatian | |
| Serbian | 45,004 | slavic.south.serbian | |
| Bosnian | 17,531 | slavic.south.bosnian | |
| Romani | 15,269 | indoaryan.romani.romani | variety not stated; see Boyash below |
| Albanian | 13,503 | albanian.albanian | |
| Italian | 12,890 | romance.italian | Istria (Brtonigla 36%, Buje 29%) and Rijeka |
| Other languages | 9,910 | other | no breakdown at any level; mostly the big cities |
| Serbo-Croatian | 8,182 | slavic.south.serbocroatian | existing node (us, fi, cz) |
| Slovenian | 7,620 | slavic.south.slovenian | |
| Hungarian | 7,218 | uralic.hungarian | |
| Czech | 4,915 | slavic.west.czech | |
| Croato-Serbian | 4,278 | slavic.south.croatoserbian | **new leaf**, see below |
| German | 3,358 | germanic.continental.german | |
| Macedonian | 3,334 | slavic.south.macedonian | |
| Slovak | 2,859 | slavic.west.slovak | |
| Russian | 2,081 | slavic.east.russian | |
| Ukrainian | 1,198 | slavic.east.ukrainian | kept apart from Ruthenian |
| Ruthenian | 1,011 | slavic.east.rusyn | Pannonian Rusyn, Bogdanovci 18%, Tompojevci 16%; as rs2022 |
| Montenegrin | 943 | slavic.south.montenegrin | |
| Polish | 730 | slavic.west.polish | |
| Romanian | 671 | romance.romanian | as printed; likely much of it Boyash, see below |
| Turkish | 368 | turkic.turkish | |
| Bulgarian | 263 | slavic.south.bulgarian | |
| Hebrew | 82 | afroasiatic.hebrew | mostly central Zagreb |
| Vlach | 40 | romance.vlach | Serbia's node for the same word; 28 in Slavonski Brod |
| Unknown | 20,840 | not drawn | the gap |

**Serbo-Croatian and Croato-Serbian**: two word orders of one Yugoslav-era name (srpskohrvatski;
hrvatskosrpski was the official name in socialist Croatia). DZS prints them in separate columns
and both sit in the old Serb areas: Serbo-Croatian in Knin (500), Kistanje (10%), Biskupija (17%),
Erdut, Vukovar; Croato-Serbian in Dvor (15%), Donji Lapac (8%), Vukovar, Knin. The word order is
not regional; it is what people chose to write. Kept as two leaves because the census counted two
answers. Merging them would be a defensible alternative.

**Boyash**: many of Croatia's Roma, Medimurje's and Baranja's among them, speak Boyash (Bayash,
Ljimba d'bjas), a Romanian dialect (Glottolog baya1255 under Romanian roma1327). The census has no
Boyash answer. Romani's strongholds are Medimurje's Roma settlements (Orehovica 34%, Pribislavec
27%, Mala Subotica 21%, Nedelisce 15%); Romanian's are Slavonski Brod (171), Darda (53) and Sveti
Durd (11), Roma settlements rather than a Romanian community; Vlach's 40 are mostly in Slavonski
Brod too. Nothing in the table says which answers were Boyash speakers, so every label stays as
printed and the public note says so.

**Vlach is not Istro-Romanian**: none of the 40 is in Istria; the Istro-Romanian speakers of
Zejane and Susnjevica must be under Croatian or Other.

**Other languages**: highest in Split (564), Rijeka and central Zagreb, so mostly migrant. All of
Croatia's own minority languages are printed by name, and the census does not separate an
indigenous remainder, so `other` is right.

## 5. Geography

Units: religiondots' `data/geo/hr/hr_opcine.gpkg` (GISCO LAU 2021, 556 polygons, LAU code `kod`)
and `hr_lookup.csv` (572 census ids to `kod`), both read only; its `sources/hr_geo.md` has the join
(county pairing derived, en-dash vs hyphen in Istria's bilingual names, three names that repeat
across counties). Census and polygons match both ways by LAU code, 556 for 556.

**Zagreb is one unit**: 17 districts are published but no district boundaries were found by
religiondots (GISCO stops at the municipality, OSM has none). 767,131 people, 19.8%, in one unit.
The districts are 94-98% Croatian each, so the loss is small for this map (Pescenica-Zitnjak has
the most Romani and Bosnian). Not pursued: whoever finds the 17 polygons fixes both maps.

Placement (`sources/hr_geo.py`): Kontur HR 2023 (downloaded into languagedots' data/geo/kontur/,
not in religiondots), each hex to the unit its centroid falls in. 2,175 hexes (135,973 people,
3.5% of Kontur's Croatia) fell outside every unit, classed by Natural Earth 10m:

| class | hexes | people | done |
|---|---:|---:|---|
| sea (coastal cities: GISCO's coast is generalised) | 1,010 | 75,409 | snapped within 1 km |
| in Natural Earth's Croatia, over 2 km from a neighbour | 283 | 14,583 | snapped within 1 km |
| in Natural Earth's Croatia, within 2 km of a neighbour | 280 | 10,268 | dropped |
| in a neighbour country | 602 | 35,713 | dropped |

1,286 hexes (89,973 people) snapped, 889 (46,000) dropped. The dropped ones sit opposite Croatian
border towns on the Sava, Una and Drava (nearest units Drenovci, Bilje, Metkovic, Dvor, Gunja,
Stara Gradiska, Cestica, Slavonski Brod, Lukac, Hrvatska Dubica): Bosnian towns such as Gradiska,
Kozarska Dubica, Novi Grad and Brcko, and Barcs in Hungary. After this Kontur/census nationally is 1.000 (3,871,777 against 3,871,833),
against 1.011 before the land-border drop. Per unit, normalised: p10 0.87, median 1.03, p90
1.36; 2 of 556 outside a factor of 3 (Civljane 3.16, Vrhovine 3.23; Dvor 2.29 is next): Kontur models built-up area, and the Krajina's houses outlived
their people (playbook, "Kontur models built-up area, so ruins and emptied towns hold people").
The counts are the census's; only where inside those units the dots go is affected. A fix would
weight hexes by DZS's settlement populations (naselja.xlsx sheet 1 has every settlement), with a
settlement point layer; not done. Log correlation of Kontur against census per unit 0.977 against
a best of 0.154 over 500 shuffles. Median unit 67 km2, 61 hexes; no unit without a populated hex.

The scatter raised no Kontur cap block.

## 6. Colour

No hand-picks (taxonomy/tree.d/hr.txt has the distances). Croatian against Italian, the one pair
that matters in Istria, is 0.117 in the palette of 2026-10-05.
