# Hungary: Népszámlálás 2022, mother tongue

Built 2026-10-04 (session d9e44929-hu). Rebuild:

```
python sources/hu_census.py [--fetch]   -> data/normalized/hu.csv
python taxonomy/build.py
python sources/hu_geo.py                -> data/geo/hu/hu_hexes.gpkg
python tools/check_country.py hu
python scatter.py --country hu
```

Drawn: 8,427,934 people on 3,177 units (3,154 settlements + Budapest's 23 districts), 16 labels on
16 nodes, 8,420 dots, 1 ring. Not drawn: 1,175,656 who did not answer (12.2%, the gap).

## 1. The table

Központi Statisztikai Hivatal (KSH), Népszámlálás 2022 (reference date 1 October 2022), the census
database at https://nepszamlalas2022.ksh.hu/adatbazis/, version V67. It is an SDMX backend behind a
JavaScript client; religiondots/sources/hu.md §2 found the routes in its app.js. The client's own
request builder (app.js, `RestSimpleDataReader`) shows the `/d/` form: each dimension as `DIM` (all
codes) or `DIM:a+b+c`, comma-separated. Naming the 20 mother-tongue codes makes WBS003 a 68,060-cell
request that returns in a second, so unlike religiondots' religion table nothing here is
hand-exported:

```
/api/dataflows/WBS003/V67/d/TIME_PERIOD:2022,TERUL_GEO5,TEL_SZ_ADAT:MT+MT_HU+MT_D+MT_GI+...+MT_NA
/api/dataflows/WBS009/V67/d/TIME_PERIOD:2022,TERUL_GEO3,TERUL_TELTIP2:HU,NEMZ:TOTAL+MT+...,TARSJELL2:NEME_SEX
/api/structure/WBS003/V67, /api/structure/WBS009/V67      (codelists: labels, geography parents)
```

Browser User-Agent, `requests` with TLS verification on (religiondots found `urllib` fails on this
machine's trust store, not the server). All in `data/raw/hu/`, no login.

**The question**: anyanyelv, mother tongue, optional, up to two answers. KSH also asked nationality
(up to two) and the language used with family and friends; those are in the same cube (`EG_*`,
`BEG_*`, `FMF_*`) and are not used.

**What a settlement carries**: the total (`MT`), Hungarian, the languages of the 13 recognised
minorities with Gipsy split into Romani (`MT_GIR`) and Boyash (`MT_GIB`), "other" (`MT_O`), "no
answer" (`MT_NA`), and two person-unions: `MT_GI` (Romani or Boyash) and `MT_D` (any minority
language). The language cells count everyone who named the language, alone or as one of two. No
combinations are published at any level of the database, and no further languages: the vármegye
table WBS009 has exactly the same codes. KSH's prose publications name more ("other" holds Russian,
Chinese, Vietnamese, English and so on) only nationally; not used, so "other" is not split.

**Suppression**: 8,886 settlement cells are `OBS_STATUS Q`, no value. Every printed settlement cell
is 0 or at least 3, so a suppressed one is 1 or 2. Hungarian and the total are never suppressed.

## 2. How a unit's people are drawn

**Suppressed cells** (`fill()`): a settlement's járás (Budapest's districts are their own) is
printed for most codes; its figure less its printed settlements is shared evenly among its
suppressed ones. Where the járás cell is suppressed as well (505 of 3,940 járás cells), the
vármegye's remainder is shared the same way. All 8,886 estimates fell inside [1, 2] without
clipping, which says the 1-or-2 reading is right. Tier `derived`, part "suppressed, estimated".
About 9,600 language mentions in all.

**Two answers** (`split()`), spec §3.6's half a person to each language named, with one inference
because KSH prints no pairs:

- E = mentions - people who answered = the people who named two (at most two allowed). Nationally
  58,266.
- Dpair = minority mentions - `MT_D` = people who named two minority languages. Nationally 876, of
  them 227 Romani and Boyash (`MT_GIR + MT_GIB - MT_GI`).
- **HX = E - Dpair = 57,390 are taken as Hungarian plus one other language.** This is the one
  assumption. What it misfiles is a person who named a minority language and an "other" one, or two
  "other" languages (if `MT_O` counts mentions; it may count persons). Hungary's minorities are old,
  bilingual communities and the "other" group is mostly migrants, so Hungarian is very likely in
  almost every pair, but nothing published checks it.
- Hungarian keeps `MT_HU - HX` as measured and `HX/2` as derived. Each other language gives up its
  share of HX/2 in proportion to its mentions, and each minority language its share of Dpair; the
  rest of it is measured.
- Dpair computed from estimated (suppressed) cells runs high: a village with German, Slovak and
  `MT_D` all suppressed shows a spurious pair. The settlement figures summed to 1,369 against the
  exact 876, so they are rescaled per vármegye to the vármegye's exact figure.

Drawn total 8,427,934 against 8,427,978 answered: the 45 come from HX clipped at a settlement's
bounds (HX >= 0, <= its Hungarian, <= its non-Hungarian mentions) where estimates disagree slightly.

## 3. Checks (asserted in `hu_census.py` and `hu_geo.py`; all pass)

| check | result |
|---|---|
| settlements | 3,177 (Budapest as 23 districts); KSH's "Budapest not divisible by district" unit (13578) has no cells |
| total = KSH's resident population | 9,603,634 |
| WBS009 (second table) = WBS003 at vármegye and country | 21 units x 20 codes, identical |
| settlements, with suppressed cells estimated, summed per vármegye = the vármegye cell | every code, gap 0.00 |
| total and Hungarian printed for every settlement, summing to the country | yes |
| suppression estimates within [1, 2] before clipping | 8,886 of 8,886 |
| drawn = people who answered | 8,427,934 vs 8,427,978 |
| join, table to religiondots' polygons, by KSH code | 3,177 = 3,177, both ways |
| Kontur against census per settlement | national ratio 0.999; median 0.99, p10 0.78, p90 1.26; 16 of 3,177 outside a factor of 3; log r 0.985 against 0.053 shuffled |

The WBS009 check is weaker than it looks: both cubes come from the same database and agree to the
person, which rules out a mis-pulled cube or a wrong code, not an error in the census itself.

National, drawn: Hungarian 8,274,110 (98.2%), other 61,963, German 23,392, Ukrainian 13,046, Romani
11,315, Romanian 9,309, Slovak 8,989, Croatian 6,838, Boyash 5,558, Serbian 3,575, Polish 2,837,
Bulgarian 2,088, Rusyn 1,651, Greek 1,474, Slovenian 1,223, Armenian 564. Mentions (KSH's own
headline figures) are higher: German 28,473, Romani 15,440, Boyash 7,979 and so on.

## 4. Labels and nodes (`taxonomy/hu2022.py`, `tree.d/hu.txt`)

Keyed by KSH's Hungarian labels from the codelist. All but one node already existed.

- **Boyash (Beás)**, a new leaf `indoeuropean.romance.boyash` beside Romanian: an archaic Romanian
  dialect spoken by Roma in Baranya, Somogy and Zala, printed apart from Romani and from Romanian.
  Glottolog has no entry for it (no glottocode given). A sibling, not a child, of Romanian, so
  Romanian is not turned into a group. Hand colour 0.74 0.14 48, a salmon-orange, apart from
  Hungarian's pale blue and Romani's pink.
- **Romani**: `romani.romani`, variety not stated (Romungro and Vlax both spoken; the census does not
  say).
- **Ruthenian**: Rusyn, as ro2021 and cz2021; Ukrainian printed apart and kept apart.
- **Croatian**: Croatian; the Bunjevac and Šokac of Baja and Mohács answer Croatian and have no label.
- **Other** (`Más anyanyelvű`): `other`, not split.
- **No answer**: not drawn; the gap.

## 5. Placement (`hu_geo.py`)

religiondots' `hu_settlements.gpkg` (GISCO LAU 2021, Budapest's districts from geoBoundaries clipped
to GISCO's Budapest; religiondots/sources/hu_geo.md), read only; its `kod` is the KSH code the
census database uses. Kontur HU (2023-11-01) downloaded into languagedots' `data/geo/kontur/`
(religiondots had no HU extract). 60,772 hexes; 602 (34,658 people) have their centre outside every
settlement, the border strip, and are dropped; every settlement has at least one populated hex.
Highest Kontur/census ratios (03188 9.3x, 13569 5.1x) are small places where Kontur sees more than
the census; they only move dots within the settlement. No Kontur cap block was hit.

## 6. Colour

Hungarian is ro.txt's pale blue (#6cd3fa) and German is the mid blue (#359bd9) from us.txt. German is
Hungary's biggest minority (Baranya and Tolna's "Swabian Turkey", the Buda hills), so these two sit
side by side; they differ by lightness more than hue. Left as is: changing either moves Romania,
Czechia and the US too. Slovak (#a0b747) and Ukrainian (#bfd869) are close yellow-greens but rarely
share a place here (Slovak villages in Békés and the Pilis; Ukrainians mostly in Budapest).

## 7. Not done

- Nationality (`EG_*`, `BEG_*`) and the family-and-friends language (`FMF_*`) at settlement level,
  same cube. FMF is the nearer thing to "home language" and could corroborate; not read.
- 2001 and 2011 are in the same cubes (TIME_PERIOD) and not read.
- KSH's national breakdown of "other" mother tongues.
