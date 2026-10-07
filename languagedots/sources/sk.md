# Slovakia: SODB 2021, mother tongue

Built 2026-10-05 (session d9e44929-sk). Rebuild:

```
python sources/sk_sodb.py [--fetch]     -> data/normalized/sk.csv
python taxonomy/build.py
python tools/check_country.py sk
python scatter.py --country sk
```

Drawn: 5,136,906 people on 2,926 units (2,927 obce and city districts, one of them the empty
Valaškovce military district), 27 labels on 27 nodes, 5,122 dots, 8 rings. Not drawn: 312,364 whose
mother tongue was not ascertained (5.73%, the gap). Nothing is derived or modelled: every row is a
count the office published for that obec.

## 1. The table

Štatistický úrad SR, Sčítanie obyvateľov, domov a bytov 2021 (reference date 1 January 2021),
indicator Z01/14, "Structure of population by mother tongue". The results browser at
https://www.scitanie.sk/en/population/basic-results/structure-of-population-by-mother-tongue/SR/SK0/SR
loads static JSON files; its script (`disem_*.js`, `getDataByFilters` and `getDataFromParent`) names
them `Z01_14_<territory>_<spec>_<unit>.json` under `.../assets/public/disem/data/`, `?v=10`.

- `KR_<kraj>_OB`: every obec of one kraj, all 28 categories. Eight files cover the country. The
  all-country `SR_SK0_OB` exists in the browser's menu but did not answer in two minutes; the per
  kraj files answer in seconds.
- `SR_SK0_SR` (national) and `SR_SK0_KR` (8 kraje): the checks.

No login, no key, browser User-Agent; certifi's CA bundle (the Windows store fails on both hosts,
as religiondots/sources/sk.py found for the GIS server). The files open with whitespace before the
`{`. The page also offers XLSX/CSV/JSON export buttons, which build the file client-side from the
same JSON. The `other` (ostatné) entry in the JSON's `names` is typed `chart`: the pie chart's fold
of the small categories, not a table category, and dropped.

**The question**: materinský jazyk, one answer, defined on the form as the language spoken at home
in childhood. The census also asked the language used most at home and in public; not used.

**Categories**, the same 28 at every level: 27 languages (Slovak, Hungarian, Romani, Rusyn, Czech,
Ukrainian, German, Polish, Russian, Vietnamese, English, Croatian, Serbian, Italian, "Yiddish or
Hebrew", Arabic, French, Romanian, Albanian, Spanish, Slovak Sign Language, Chinese, Bulgarian,
Korean, Turkish, Persian), "iný" (other, 3,952) and "nezistený" (not ascertained, 312,364). Counts
are printed down to 1; no suppression seen.

## 2. Checks (all asserted in sk_sodb.py)

| check | result |
|---|---|
| obce in the 8 kraj files | 2,927, codes distinct |
| each obec's 28 categories sum to its total | all 2,927 |
| national total | 5,449,270 (the census's usually resident population; religiondots' total too) |
| obce summed per category = national file | all 28 exact |
| 8 kraje summed per category = national file | all 28 exact |
| GIS layer `obyv_ekchar_matjaz_vekskup/4`, same obec codes | 2,927 = 2,927 both ways |
| per obec, GIS = portal on the 10 languages it prints and the total | all 2,927 exact |
| per obec, GIS `ostatné` = portal's other 18 categories summed | all 2,927 exact |
| obec codes = religiondots' grid `unit` | 2,927 both ways, no join |

The GIS layer is the second witness: a different publication of the same table (the census's
ArcGIS server at gis.scitanie.sk, the one religiondots reads religion from), folded to 10 languages
plus `ostatné`. Its `ostatné` (333,224) is mostly not-ascertained (312,364): the portal is the only
one of the two that separates them, which is why it is the source.

Not used: the GIS server's nationality x mother tongue cross-table (`obyv_ekchar_nar_matjaz`), 15
mother-tongue categories by nationality. It sums to 5,436,701, 12,569 short of the census, and short
in every language at obec level (small cells left out), so it is not a full table.

## 3. Geography

No join and nothing built: the portal's obec codes are the LAU codes (`uzemie`, e.g.
SK0101528595) that religiondots' `data/geo/sk/sk_grid_1km.gpkg` is keyed by. That layer is the
census's own 1 km population grid (`obyv_grid_1km`, summing to 5,449,270), each cell split among the
obce it touches by area (religiondots/sources/sk_geo.py; its note explains why not by centroid).
87,240 pieces over all 2,927 obce. Read-only; `place_weight=pop_weight`, which is what religiondots'
own `_SkGridWeighter` does. Grain: obce, with Bratislava's 17 and Košice's 22 city districts
(mestské časti) as units, 1,860 people on average.

## 4. Mapping calls (taxonomy/sk2021.py, tree.d/sk.txt)

- Every label on its own node. Two new leaves: `signlanguage.spj` Slovak Sign Language (Glottolog
  slov1263), and `other.yiddish_hebrew` for "jidiš alebo hebrejský", one answer naming two
  languages of two families (273 people), a named leaf under `other` as az.txt's "Jewish" is.
- Romani on the generic `romani.romani` leaf, as Czechia. Slovakia's Romani is mostly Carpathian
  (Glottolog's West and East Slovakian Romani dialects of carp1235) with some Vlax, but the census
  does not say which.
- Chinese on Sinitic, as cz2021, pl2021, us2024, uk2021.
- Rusyn and Ukrainian apart, as printed. Rusyn is the majority in 44 obce, Ukrainian in none.
- "iný" on `other`; "nezistený" not drawn (the gap).

## 5. Colours

Slovak is hand-picked in tree.d/sk.txt, gold `0.78 0.13 70` (#eca851). Its generated colour was an
olive (#a0b747) on top of Ukrainian's yellow-green and near Rusyn's dark green, the two languages it
meets in the north-east. Gold is far from Hungarian's sky blue (the southern strip, 369 obce with a
Hungarian majority), Czech teal, Polish green and Romani purple; in Czechia it is 35 degrees and a
little darker than Silesian yellow.

Side effect: Slovak no longer takes a generated step under West Slavic, so Poland's generated
dialect leaves move one step (Cieszyn, Goral, Kociewie, Greater Poland, Kurpie and both Sorbians
each take their neighbour's old colour; Cieszyn olive against Goral dark green, still apart).

## 6. What the map shows, for the reader

- Hungarian 8.48%, the majority in 369 obce along the southern border.
- Romani 100,526 (1.84%), the majority in 22 obce (Lomnička, Bystrany, Stráne pod Tatrami at the
  top). Roma are several times that; many name Slovak or Hungarian.
- Not ascertained: median 2.3% per obec, 47.7% in Košice's Luník IX (3,357 of 7,037), 22% in Jasov,
  and 10-13% in the centres of Bratislava and Košice and in Komárno, Trebišov and Lučenec. Those
  people are not drawn, so these places look thinner than they are; `note_public` says so.

## 7. Not done

The census's "language used most at home" question would be a second view; not used, as the map
draws first languages.
