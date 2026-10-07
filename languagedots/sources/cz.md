# Czechia: SLDB 2021, mother tongue

Built 2026-10-04 (session d9e44929-cz). Rebuild:

```
python sources/cz_sldb.py [--fetch]    -> data/normalized/cz.csv
python taxonomy/build.py
python sources/cz_geo.py               -> data/geo/cz/cz_hexes.gpkg
python tools/check_country.py cz
python scatter.py --country cz
```

Drawn: 9,764,773 people on 6,388 units (6,246 obce + 142 city districts), 56 labels on 56 nodes,
9,740 dots, 0 single-dot rings.

## 1. The tables

Český statistický úřad (ČSÚ), Sčítání lidu, domů a bytů 2021, reference date 26 March 2021,
open data in the national catalogue (data.gov.cz, publisher 00025593), the same family of files
as religiondots' religion table (religiondots/sources/cz.md):

```
https://csu.gov.cz/docs/107508/87997b45-9487-ccd0-980c-1b879f337058/sldb2021_jazyk1.csv
  "Obyvatelstvo podle mateřského jazyka (1 mateřský jazyk)"       13.9 MB
https://csu.gov.cz/docs/107508/f970b567-ffeb-4ca6-8238-41e4c4824019/sldb2021_jazyk.csv
  "Obyvatelstvo podle mateřského jazyka (1 nebo 2 mateřské jazyky)" 14.2 MB
```

Found through the catalogue's SPARQL endpoint (titles containing "jazyk" and "2021"); the same
catalogue also lists both by sex and by age and sex. Plus two ČSÚ code lists at the census date,
for the kraj of each obec and city district (`apl2.czso.cz/iSMS/do_cis_export`, KRAJ_NUTS 100 to
CISOB 43, `cisvaz=43_1250`, and to CISMC 44, `cisvaz=44_1252`, `datpohl=26.03.2021`). All in
`data/raw/cz/`, no login.

**The question**: mateřský jazyk, mother tongue, optional, one or two answers.

- `jazyk1`: persons who named ONE mother tongue, by language; plus "Osoby se dvěma mateřskými
  jazyky" (persons who named two) and "Nezjištěno" (not stated).
- `jazyk`: "<X> celkem", everyone who named X, alone or as one of two.

**The detail depends on the level.** Country and the 14 kraje carry 55 languages, sign language
and other, plus the two-answer and not-stated rows, and add up exactly. Okres, ORP, obec and city
district carry only 13 languages (Czech, Slovak, Moravian, Silesian, Polish, German, Romani,
English, Russian, Ukrainian, Vietnamese, Hungarian, Chinese) and not stated: no two-answer row,
none of the other 43 labels. Nothing is suppressed or rounded; small cells are printed down to 1.

National: 10,524,167 people; 9,504,833 named one tongue, 259,940 named two (217,512 of them Czech
and another), 759,394 did not answer (7.2%, the gap).

## 2. How a unit's people are drawn

Spec §3.6's exact split, half a person to each of two languages, with no scaling:

- **The 13 languages, per obec or city district**: `single` (tier measured) plus
  `(celkem - single) / 2` (derived). Exact: every two-answer person names exactly two.
- **Everyone else, per unit**: `pool = total - not stated - the 13 above`. Because each
  two-answer person names two, the pool is exactly the unit's people (and half-people) on the
  other 43 labels. Which of the 43 is published only per kraj, so each pool is shared out by its
  kraj's split of the 43 (single + half of pairs), tier derived. The pools of a kraj add up to
  the kraj's 43-label figure (asserted), so every language's obec shares sum to its kraj figure
  exactly. Nationally the pool is 96,410 people, 1.0% of those drawn.

Where these people sit inside a kraj follows where the census puts people with "some other mother
tongue", a published number per unit, and only the mix among them is the kraj's. Prague is its
own kraj, so its 57 city districts share Prague's exact mix.

## 3. Checks (all asserted in `cz_sldb.py` and `cz_geo.py`, all pass)

| check | result |
|---|---|
| the two files' unit totals and not-stated rows | identical in all 6,488 units |
| country and kraj: one tongue + two + not stated = total | 15 of 15 |
| country and kraj: two-answer mentions (celkem - single) = 2 x two-answer persons | 15 of 15 (519,880 nationally) |
| celkem >= single, every language, every unit | holds |
| obce summed by kraj (code list) = kraj table, 13 languages x single and celkem, totals, not stated | 14 of 14; the finest cover likewise |
| city districts against the 8 obce they replace | covered by the line above (finest cover by kraj) |
| every pool >= 0; pools per kraj = the kraj's 43-label figure | holds |
| finest cover summed per label = national | 57 labels, exact |

## 4. Placement (`cz_geo.py`)

Units are religiondots' `cz_finest.gpkg` (ČSÚ's own obce and městské části, census vintage),
read only: 6,388 units in the table, all with a polygon; the 4 polygons left over are the military
districts, which have no census rows. Kontur CZ (2023-11-01, downloaded to
`data/geo/kontur/`): 80,168 hexes, 10,548,451 people; 1,376 hexes (54,498 people) outside every
unit. Kontur / census 1.075 nationally; per unit p10 0.80, median 0.99, p90 1.40; 90 of 6,388
outside a factor of 3 (123,000 people); log correlation r = 0.97 against 0.04 for the best of 200
shuffles.

The outliers are Kontur's: it moves people out of the dense housing estates into the suburban
city districts round them (Ostrava-Jih 0.34, Praha 11 0.44, Brno-Vinohrady 0.31 against
Praha-Klánovice 4.7, Brno-jih 4.2). Each unit's dot count comes from the census, so this only
moves dots inside a district. Three units have no populated hex and get their own polygon at pop 0
(Valdice by Jičín 2,373, Adamov by České Budějovice 949, Nasavrky in South Bohemia 92: small
polygons pressed against a town, whose hex centroids fall in the town).

**Grid floor.** `scatter.py` warns: median 8 populated cells per unit, 76 units in one cell.
Accepted: units with 10 or more populated cells hold 82% of the people (5 or more: 96%), and the
ones below are villages of a few hundred, mostly under one dot. The bar Malta's record used for
the same call (per-unit Kontur / census p10 >= 0.6, p90 <= 1.6) is met: 0.80 and 1.40. The
warning asks for a line in `sources/geo_checks.csv`, which is religiondots' file and not written
from here; this paragraph is the record.

## 5. Calls

- **Hexes rather than religiondots' polygons**, so the towns over 10,000 that are not
  subdivided (Olomouc, České Budějovice, Hradec Králové, Zlín) put their dots where people live.
- **Moravian** (22,585 drawn) is its own node beside Czech, not a child: Glottolog has
  Czecho-Moravian (czec1259) and Lach (lach1246) as Czech dialects, but a child would make Czech
  a group, drawn washed out. Hand-coloured pale mint against Czech's mid teal.
- **Silesian** (1,272 drawn) goes on the node Poland uses. 800 of its 1,805 mentions are in
  Frýdek-Místek district and 549 in Karviná (Třinec, Jablunkov, Český Těšín), Těšín Silesia,
  whose speech is the Cieszyn Silesian Poland's census files under Silesian or its own gwara; 144
  are in Opava, where it would be Lach. One label for both; not split.
- **Czech's colour hand-picked** in `tree.d/cz.txt` (0.60 0.10 192). It was generated, and a
  generated colour moves with the set of fragments in the build: dark teal in one build, a
  yellow-green beside Ukrainian and Silesian in the next. Czech is also on the US, Polish and
  Canadian maps, all as a small minority, so the change is small there.
- **Chinese** on Sinitic (washed out, "language not named"), as pl2021 and us2024.
- **Moldovan and Romanian, Serbo-Croatian and Serbian/Croatian/Bosnian kept apart** as printed;
  **Montenegrin** a new South Slavic leaf (Glottolog mont1282, a dialect under
  Serbian-Croatian-Bosnian).
- **"Znakový jazyk"** (sign language, 3,117 drawn) on the `signlanguage` root, as za2011 and
  np2021: the label does not say which, though almost all will be Czech Sign Language.
- **"Jiný jazyk"** (other, 12,091) on `other`.
- **The 19 smallest labels draw nothing** (Uzbek 951 down to Turkmen 41, 7,757 people): every one
  of their obec rows is derived (the kraj split), and `scatter.py` gives the one-dot fallback only
  to a language with measured rows. Same outcome as Poland's small labels.

## 6. Second source

Not required for a mother-tongue census. Romani is low against the Roma population (16,191 drawn,
against estimates of the Roma population in the low hundreds of thousands), as in every Czech census: most Roma name Czech. That is what
the census measured, and the map draws it as measured.

## 7. Licence

ČSÚ open data via data.gov.cz, attribution to Český statistický úřad; religiondots/sources/cz.md §8
flags the same terms as not yet read against a redistribution of derived dots.
