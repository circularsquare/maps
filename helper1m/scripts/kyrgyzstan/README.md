# Kyrgyzstan

Three levels, all on the units of the 2024-25 territorial reform:

| level | label | units | years |
|---|---|---|---|
| 1 | Oblast / city | 9 (7 oblasts, Bishkek, Osh) | 2024, 2025, 2026 |
| 2 | Rayon / city | 56 (40 rayons, 14 cities of oblast significance, Bishkek, Osh) | 2024, 2025, 2026 |
| 3 | Aiyl aimak / town | 301 (see below) | 2025, 2026; 2024 for 273 of them |

The 301 at level 3 are 230 aiyl aimaks, 18 towns, 14 cities and the two
capitals from the 2026 file. Three pairs are drawn as one, which leaves 261.
On top of those, 40 zero-population units ("Land outside any aiyl aimak", one
per rayon) hold the pasture, forest and reserve land that COD keeps outside
every aiyl aimak.

The current-year estimate runs off 2025 to 2026. Both years come from the
same office, are on the same units and use the same method.

Run, from the repo root:

```
C:\Python39\python.exe helper1m\scripts\kyrgyzstan\fetch.py           # downloads, crosswalk, population.csv
C:\Python39\python.exe helper1m\scripts\kyrgyzstan\prep_boundaries.py # adm{1,2,3}.gpkg
C:\Python39\python.exe helper1m\scripts\build_country.py kyrgyzstan
C:\Python39\python.exe helper1m\scripts\kyrgyzstan\check.py           # totals and Kontur
```

All of it takes about two minutes.

## Sources

- **Populations**: National Statistical Committee (NSC), "Численность
  постоянного населения областей, районов, городов, айылных аймаков и айылов
  (сел) Кыргызской Республики",
  <https://stat.gov.kg/ru/statistics/download/operational/825/>. It is one
  workbook with one sheet per oblast plus "КР,Ош+Бишкек", 14-digit SOATE codes,
  and rows down to the single village. The office replaces the file every
  spring, so:
  - 2026 (start of year): the live file, `raw/op825_2026.xls`. It is the same
    as `religiondots/data/raw/kg/kg_nsc_population_2026.xls`.
  - 2025: Wayback capture of 2025-05-25, `raw/op825_wb20250525.xls`.
  - 2024: Wayback capture of 2025-02-05, `raw/op825_wb20250205.xls`. The
    2024-06-23 capture is the same year with "постоянного" missing from the
    title.

  Older years were not available. The CDX for that URL starts in June 2024.
  `fetch.py` refuses to run if the live file stops saying "на начало 2026г"
  (it will once the 2027 file is posted). At that point, point it at a Wayback
  capture.

  Licence: CC BY-NC-SA 4.0 (the footer of every stat.gov.kg page). Article 30
  of the Law on Official Statistics also requires citing the NSC.
- **Boundaries**: COD-AB Kyrgyzstan (HDX, valid 2018-11-19),
  `data/asia1m/kyrgyzstan/kgz_admin{1,2,3}.shp`.
- **Village locations**: GeoNames `KG.zip` (CC BY), used only to place
  villages, <https://download.geonames.org/export/dump/KG.zip>.

## What the NSC file is

Rayon, city, oblast and national rows are **NSC estimates**. Aiyl aimak and
village rows are **aiyl okmotu registers** (the header says so), and they do
not add up to the rayon. Each rayon's aiyl aimaks and towns are scaled so that
they sum to the NSC rayon figure. For most rayons the factor is 0.96 to 1.04.
The outliers are Alamudun (1.25 in 2026), Sokuluk (1.32), Toktogul, Chatkal,
Chon-Alai, Kara-Kulja (1.08-1.11) and At-Bashy (0.91). The Bishkek suburbs
evidently have far more residents than registrations. So every level sums
exactly to the NSC figures in 2025 and 2026.

Code traps:

- Talas's codes are written with spaces, so the parser pulls out digits.
- Aravan rayon is coded `41706211800000`. It is renamed to `41706211000000`.
- Naryn city is `41704000000010`, with no 4xx segment. COD calls it `KG04400000010`.
- Three units changed code between the 2025 and 2026 files: Shamalduu-Sai,
  Gulcho (an aiyl aimak in 2025, a town in 2026) and Kemin. `RECODE25` maps them.
- Jalal-Abad city is "г. Манас" in the 2026 file under the same code and is
  labelled "Manas (Jalal-Abad city)".
- Some names use a Latin "c" for Cyrillic "с".

## The reform, and why every level is redrawn

Between the start of 2024 and the start of 2025:

- The aiyl aimaks were merged, 452 → 230 (231 in 2025).
- The urban-type settlement (pgt) rank was abolished.
- Bishkek took 8 aiyl aimaks from Alamudun and Sokuluk rayons (Alamudun,
  Kok-Jar, Lebedinovka, Maevka, Nizhnyaya Ala-Archa, Novopavlovka, Orok,
  Prigorodnoe). Its figure went 1,165,497 → 1,321,877.
- Osh took Kyzyl-Kyshtak and Teleyken and pieces of Papan, Narimanov and
  Shark. Its figure went 366,738 → 473,552.
- Jalal-Abad (Manas) took Yrys and other Suzak villages, and Kant and
  Kara-Balta became cities of oblast significance.
- Several towns (Kerben, Toktogul, Cholpon-Ata, Karakol, Gulcho) took
  neighbouring aiyl aimaks.

The 2024 file is still on the old units, whose codes are COD's. 2025 and 2026
are on the new ones. Since the last two years must share a basis, all three
levels use the 2026 units, and the 2018 polygons are regrouped to fit.

### How the regrouping works (`atoms.py`, `crosswalk.py`)

**Atoms.** An atom is one of:

- one of COD's 737 third-level units;
- a city of oblast significance minus its pgt (COD's third level leaves the
  cities themselves as holes);
- Bishkek or Osh whole (COD has nothing below them).

That gives 752 atoms. Lake Issyk-Kul is in none of them.

**Old unit → new unit by villages.** Each 2024 village sits under its old unit.
The same village is looked up by name among the 2026 villages:

- in the same rayon first, then in bordering rayons;
- with Kyrgyz/Russian spelling folded (ё/ө/ү, -ка/-ово endings);
- a name match is dropped when the population disagrees by more than 30%,
  which removes namesakes;
- towns can also match oblast-wide, which catches Kant and Kara-Balta becoming
  cities.

An old unit takes the new unit most of its people went to. This counts only
when at least half its people were found, or when every hit points at a unit
that kept the old code. 403 atoms are placed this way and 26 more by an
unchanged city or town code.

**New units with no atom** (aiyl aimaks created after 2018, towns COD never
drew) are placed with GeoNames points of their villages, searched within
their own rayon:

- If the points fall in an atom nobody has claimed, the unit takes that atom
  (6 cases: Kozhomkul, Gulcho, Nookat, Kadamjai, Tugol-Sai, A. Mirmakhmudov).
- If the atom is already another unit's, the two are drawn as one: Dostuk +
  S. Yusupovoy (Aravan), N. Isanov + Kok-Bel (Nookat).
- Jany-Alai (Alai) is found in neither GeoNames nor OSM. It is drawn with
  Pamir-Alai by hand (`MANUAL_MERGE`); both came out of the old Taldy-Suu
  aiyl aimak.

**COD's unnamed 9xx units are not all empty.** They include Nookat town,
Kadamjai, Osh's village belt (Japalak, Arek, Kenesh, Teeke) and an exclave of
Kyzyl-Kiya inside Nookat rayon. An unnamed unit goes to the unit whose
villages stand in it (GeoNames points, by name), when all of these hold:

- the unit is in the same rayon (or is a city);
- the unit already has ground within 10 km;
- at least 300 people are matched, at least twice the runner-up.

40 of these units are placed. One is placed by hand (`MANUAL_CLAIM`): COD's
Kara-Kul city polygon is only its Ketmen-Tebe pgt, about 1 km², and the town
itself is in unnamed unit KG03225000911. The remaining 235 unnamed pieces
become the 40 zero "land" units.

**Absorbed units.** 36 old units have their villages missing from 2026 because
a city or town now lists no villages. Each goes to the bordering unit whose
2025 figure most exceeds what its mapped 2024 units explain (2024 × 1.017),
judged on the unscaled register figures. A second pass then moves an atom
when that shrinks the two shortfalls together.

A pgt with no name match stays with its city: Pristan-Przhevalsk, Orto-Tokoy,
Vostochny, Kek-Tash, Ketmen-Tebe. Shamaldy-Sai went to Shamalduu-Sai town and
Kyzyl-Jar to Uch-Korgon (Aksy).

`data/kyrgyzstan/crosswalk.csv` has the decision for every atom, with the
reason in `how`. `crosswalk_log.txt` lists each absorbed, claimed and merged
case.

**Checked against the decree lists.**

- Bishkek: Sputnik Kyrgyzstan, 2024-03-01, names Orok, Novopavlovka, Kok-Jar,
  Maevka, Prigorodnoe, Alamudun, Nizhnyaya Ala-Archa and Lebedinovka. Those are
  exactly the 8 atoms the growth rule gave Bishkek. Mykan village went too; its
  Lenin aiyl aimak is drawn with Dostuk.
- Osh: the decree of 2023-12-29 gives Kyzyl-Kyshtak and Teleyken whole (both
  drawn in Osh). It also gives sections of Papan (Ak-Buura 1-4), Narimanov
  (Jany-Mahalla, Jim, Jiydelik) and Shark (Imam-Ata), which cannot be cut out
  of 2018 polygons. That is why Osh's 2024-on-new-boundaries figure is about
  5% low and Kara-Suu's Manas aiyl aimak about 25% high.

## 2024 (history only)

2024 is carried through the same regrouping. Old units with no atom (the
post-2018 aiyl aimaks) follow their villages, and nothing is lost: the
national sum is exactly 7,161,910.

Levels 1 and 2 have 2024 for every unit. At level 3, 2024 is left out for 28
units whose 2024 → 2025 change falls outside −6% to +12%. Most are Alamudun's
aiyl aimaks, where the register scaling changed between years (1.17 → 1.31),
and the units that gave or took the Osh and Bishkek partial pieces. So level
3 2024 does not sum to level 2 2024; levels 2 and 1 hold the full sums. The
2022 census by aiyl aimak was not used: census.stat.gov.kg times out, and it
would need the same regrouping.

## Checks (`check.py`)

- **National**: 7,161,910 / 7,281,827 / 7,404,329 for 2024 / 2025 / 2026,
  equal to the NSC rows.
- **Oblasts 2026**: all nine equal the NSC rows (Issyk-Kul 554,296,
  Jalal-Abad 1,379,916, Naryn 316,182, Batken 605,283, Osh 1,438,459, Talas
  282,931, Chui 983,664, Bishkek 1,358,661, Osh city 484,937).
  - 2025 also matches.
  - 2024 matches for the five oblasts the capitals did not touch. Bishkek,
    Osh city, Chui and Osh oblast differ from the 2024 NSC rows because those
    rows are on the old boundaries.
- **Rayons and cities 2025/2026**: level 3 sums exactly to every NSC rayon and
  city row.
- **2025 → 2026 at level 3**: median +0.95%, 8 of 261 units outside ±5%, the
  largest +13.9%. These are register changes, not crosswalk artefacts, since
  both years are on the same units.
- **Kontur Population 2023** (hex centres summed per polygon, scaled to the
  national total): agreement is poor and does not test the regrouping.
  - At level 3: 22% of populated units within 10%, 48% within 25%.
  - At level 2, which uses the NSC's own rayon totals and is untouched by the
    regrouping: 23% within 10%. So the gaps are mostly Kontur's.
  - Kontur pulls town people into the countryside: Karakol ×0.33, Balykchy
    ×0.30, Uzgen ×0.36, Kara-Balta ×0.45, while Balbay ×6.3 and Kyzyl-Oktyabr
    ×5.6.
  - It puts about 298,000 people in the 40 zero land units. Some of that will
    be real villages on unnamed COD land that no rule could place, so those
    people are drawn in the wrong polygon. Their unit's figure still counts
    them.
- **Coverage**: every level-3 unit has population, every 2025/2026 unit has a
  polygon, and every 2024 unit was carried.

## Known weaknesses

- Outlines are 2018 shapes regrouped. Where a 2024 decree moved only part of
  an aiyl aimak (around Osh, Bishkek's Mykan), the whole 2018 piece goes one
  way.
- The absorbed-unit rule is inferred from growth and borders. It matched the
  Bishkek and Osh decree lists. Saray → Kara-Suu town and Tash-Bulak → Barpy
  are unconfirmed.
- Unnamed COD land claimed through a few herders' villages makes some Naryn,
  Alai and Toktogul units much larger than their settled area, so their
  density shading reads low.
- The 2026 file is aiyl okmotu registers scaled to NSC rayon totals. Within a
  rayon, the split between units is the register's.
