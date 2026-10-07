# Russia

Built 2026-10-02. Two levels: 83 federal subjects and 2,295 municipal units.
`helper1m/data/` is gitignored, so this file is the record.

## Running it

```
C:\Python39\python.exe helper1m\scripts\russia\fetch.py           # Rosstat -> units.csv (+ population.csv once boundaries exist)
C:\Python39\python.exe helper1m\scripts\russia\fetch_osm.py       # OSM boundaries via Overpass, ~45 min when Overpass is busy
C:\Python39\python.exe helper1m\scripts\russia\fetch_wikidata.py  # OKTMO codes for the OSM relations' Wikidata items
C:\Python39\python.exe helper1m\scripts\russia\prep_boundaries.py # polygons + match -> boundaries/*.gpkg, population.csv
C:\Python39\python.exe helper1m\scripts\build_country.py russia
C:\Python39\python.exe helper1m\scripts\russia\check_kontur.py    # optional checks
C:\Python39\python.exe helper1m\scripts\russia\check_area.py
```

Everything downloaded is cached under `data/russia/raw/`; re-runs skip what is there.

## The levels

| level | label | units | years | from |
|---|---|---|---|---|
| 1 | Federal subject | 83 | 2021, 2024, 2025 | 2024/2025 = sum of level 2; 2021 = census by subject |
| 2 | Municipal district / okrug | 2,295 polygons (2,322 Rosstat units) | 2024, 2025 | Rosstat municipal table |

Level 2 is the level directly under the subject: municipal districts (raions),
municipal okrugs and urban okrugs, as they stood on 1 January 2025. Inside
Moscow and St Petersburg it is Moscow's 12 administrative okrugs and St
Petersburg's 18 districts. Rosstat prints those as subtotals of the ~250
intra-city municipalities, and they were used instead because New Moscow's
settlements were regrouped during 2024 (146 municipalities in the 2024 table,
132 in 2025), so the small units do not pair across years while the okrugs do.
They are also about 1 million people each, which suits this tool.

Level 2 is split into one file per subject (`countries/russia/adm2/<ISO>.geojson`),
11.0 MB in all, the largest file 1.0 MB. Level 1 is 1.2 MB.

## Sources

**Populations: Rosstat, all through the Wayback Machine.** rosstat.gov.ru
presents a TLS chain no Western trust store verifies (curl exits 60), so
nothing was fetched from it directly and verification was not turned off. The
Wayback CDX listing of `rosstat.gov.ru/storage/mediabank/` finds the files:

- `Сhisl_MO_01-01-2025.xlsx` (the first letter is a Cyrillic С on the server),
  "Численность постоянного населения Российской Федерации по муниципальным
  образованиям на 1 января 2025 года", snapshot 20260906222243:
  `https://web.archive.org/web/20260906222243if_/https://rosstat.gov.ru/storage/mediabank/%D0%A1hisl_MO_01-01-2025.xlsx`
  (captures before March 2026 are redirects; the file itself is only archived from then).
- `Сhisl_MO_01-01-2024.xlsx`, same table for 1 January 2024, snapshot 20240428232801.
- `Tom1_tab-5_VPN-2020.xlsx`, 2021 census volume 1 table 5, snapshot 20230725211953.
  Used for subjects only: below the subject it has 2021 municipal names and no codes.
- Checked against, not used: `PrPopul2025_Site.xlsx` and `PrPopul-2024_Site.xlsx`,
  Rosstat's preliminary national tables.

No 1 January 2026 table is archived yet (the CDX has only a broken URL for it).
The pair 2024/2025 is two estimates from the same office on the post-census
basis, so the line through them is a year's change and not a change of method.

**Boundaries: OpenStreetMap through Overpass**, downloaded 2 October 2026
(`fetch_osm.py`): every `boundary=administrative` relation at `admin_level=6`
inside Russia's OSM area (2,301), plus `admin_level=5` inside Moscow and St
Petersburg. Overpass answered 429 and 504 often (other agents were using it);
the script waits and alternates with the kumi.systems mirror. A Russian
mirror (maps.mail.ru) fails TLS like Rosstat.

**Wikidata** (`fetch_wikidata.py`): OKTMO codes (P764) for 2,560 of the 2,621
Wikidata items the OSM relations point to, plus coordinates (P625) and areas
(P2046) for checks and placing a few units.

**Subjects for placing relations:** geoBoundaries RUS ADM1 (2017 vintage, 83
subjects, no Crimea), read from `religiondots/data/geo/ru/`.

**Sea:** Natural Earth 10 m ocean (`maps/data/ne_10m_lakes/ne_10m_ocean.shp`).

What was not used, and why:
- The GADM files in `maps/data/asia1m/russia/` (adm1 = federal districts, adm2 = subjects).
- geoBoundaries RUS ADM2: it exists (2,327 units, OSM via Wambacher, data of
  March 2023) but carries only English names and no codes or parents, so it
  could only be joined by transliterated name; OSM itself has Russian names,
  Wikidata links and a 2026 vintage.
- GitHub: searched for OKTMO boundary sets; found code lists and a casualty
  study that uses OSM admin_level 6, nothing with polygons and codes.

## How it is put together

**1. Parsing the Rosstat table** (`rosstat_mo.py`). One long sheet: a subject
row, then its municipalities, each followed by its settlements and localities.
The first column is a ТЕРСОН-МО code whose first 8 digits are the OKTMO; the
cell is sometimes text with stray spaces in it (`41754000 0 0`). A municipality
is a code of 8-10 digits whose last three OKTMO digits are 000 and whose type
digit is 5 (municipal okrug), 6 (municipal district), 7 (urban okrug) or 8
(units of the former Koryak and Aga Buryat okrugs); a code of 11-15 digits is a
locality. Autonomous okrugs inside an oblast (Nenets, Khanty-Mansi,
Yamalo-Nenets: 118, 718, 719) shift everything one digit. Arkhangelsk and
Tyumen each appear twice, with and without their okrugs; the "без"/"кроме" row
is the one used.

The workbooks have typos, each repaired by rule and printed on every run:

| year | row | printed | read as |
|---|---|---|---|
| 2025 | Bikinsky okrug (Khabarovsk) | 850900001 (leading zero lost) | 08509000 |
| 2025, 2024 | Selemdzhinsky district (Amur) | 10645151 (a settlement's code) | 10645000 |
| 2025 | Irkutsk city | no code at all | 25701000 |
| 2025, 2024 | Soletsky okrug (Novgorod) | 48538000 | 49538000 |
| 2024 | Nemsky okrug (Kirov) | 335260000 (trailing zero lost) | 33526000 |
| 2024 | Irkutsk city | 12-digit code | 25701000 |

After the repairs every subject's municipalities sum exactly to its subject
row, in both years (the script stops if not).

**2. Carrying 2024 onto the 2025 units** (`crosswalk.py`). During 2024 about
180 districts and urban okrugs became municipal okrugs: usually the same land
with a new code (14710000 Alekseyevsky urban okrug became 14510000 Alekseyevsky
municipal okrug), sometimes merged (Kasimov town and Kasimovsky district into
one okrug; Polysayevo into Leninsk-Kuznetsky; Protvino and Pushchino into
Serpukhov; Chuvashia's Alatyr, Kanash and Shumerlya into the districts around
them). Same code and under 8% change is the same unit; otherwise the old unit
is linked by name (type words dropped), then by name with adjectival endings
cut, then by the unit-number digits of the code, then to the new unit most
short of people. One manual link: Pushchino into Serpukhov (the last rule
alone put it in Lyubertsy, which grew by a similar amount). Result: all 2,367
of 2024's units land on one of 2025's 2,357, and every 2025 unit is within 8%
of its 2024 parts except four, looked at one by one:

- Saratov city +3,716 and Tatishchevsky district -5,398 (-20%): a piece of the
  district moved into the city. The pair is rebased (both take the pair's
  combined change).
- Grozny +65,614 (+20%) while every Chechen district fell 3-4%: a
  re-allocation between the two estimates. All of Chechnya is rebased, so each
  unit carries Chechnya's overall +1.5% and none extrapolates the shift.
- Troitsky okrug of Moscow +16% (and Novomoskovsky +8%): New Moscow's building
  boom. Left alone.
- Svobodnensky district of Amur +46% (11,318 to 16,531): no neighbour lost
  people, so it was left as Rosstat has it; its 2026 estimate will be high.

After this only 13 of 2,295 polygons change by more than 5% in the year.

**3. Building the polygons** (`prep_boundaries.py`). Each relation's member
ways are noded together and cut into faces; a face is kept when a ray from
inside it crosses the boundary an odd number of times. That is needed because
a district that surrounds its town (an urban okrug of its own) often lists the
town's border as ordinary outer ways, and keeping every face would fill the
hole and double-count the town. Six relations have loose ends and are closed
by joining each loose end to the nearest other one (the longest join 0.04
degrees, Novosibirsky district). Nizhny Tagil's relation is missing about a
third of its border (a 0.1-degree join leaves a 21 km² sliver), so it is not
used: Nizhny Tagil is drawn as the part of Sverdlovsk Oblast no other unit
covers (4,368 km²).

Each relation goes to the subject holding its representative point; points
that miss every subject (cities on lakes or bays the 2017 subject outlines
leave out, the Kurils) go to the subject they overlap most. Inside Moscow and
St Petersburg only the level-5 relations are used; elsewhere only level 6.

**4. Pairing polygons with Rosstat units, within each subject.** First by
OKTMO from the relation's own tags or its Wikidata item (current code, or a
2024 code carried forward); then by name. Where code and name disagree, an
exact name wins over the code and a stem-only name loses to it (Spassk-Dalny's
Wikidata item carries Spassky district's code; "Maykop" stem-matches Maykopsky
district). Twins like Kemerovsky municipal okrug (the district) and Kemerovsky
urban okrug (the city) are told apart by the type words.

OSM is a year ahead of the 1 January 2025 table, so 27 Rosstat units have no
relation of their own. Each was placed by its Wikidata coordinate (or, for
Pirovsky and Nazarovo, by name, and Tyukhtetsky by area) inside the OSM
polygon that now holds it, which then carries both units. 24 polygons hold
two or three units:

- 19 in Krasnoyarsk Krai, which folded its towns into the districts around
  them and merged pairs of districts in 2025 (Achinsk town + Achinsky +
  Bolsheuluysky into Achinsky okrug; Kansk into Kansky; Lesosibirsk and
  Yeniseysk into Yeniseysky; Tyukhtetsky into Birilyussky, manual, the OSM
  area 11,744 km² being the two districts' sum).
- Kstovsky okrug (121,322 people) inside Nizhny Novgorod: OSM has no Kstovsky
  relation and the city's relation spans both (1,732 km² against the city's
  ~466). The polygon is labelled Nizhny Novgorod and holds 1,343,494.
- Dzerzhinsky inside Lyubertsy, Skopin inside Skopinsky okrug, Labytnangi
  inside Priuralsky okrug, Spassky district inside Spassk-Dalny.

`MANUAL` (one relation), `MANUAL_POINTS` (Kstovo and Dzerzhinsky, no Wikidata
coordinate) and `DROP_RELATIONS` (Nizhny Tagil) in `prep_boundaries.py` and
`MANUAL` in `crosswalk.py` (Pushchino) are the only hand edits. Every decision
is written to `data/russia/boundaries/match_report.txt`.

One OSM relation is left out: "городской округ Покров" (Vladimir), tagged as
an urban okrug in OSM though Pokrov is a town inside Petushinsky district.

**5. Sea.** OSM draws coastal units out to the 12-mile limit; Natural Earth's
ocean is cut away so coasts look like coasts. Lakes (Ladoga, Onega, Baikal)
and the Caspian stay inside their units.

**6. Level 1** is level 2 dissolved per subject, so the levels nest.

**Antimeridian.** OSM splits Chukotka's relations at 180°. Every polygon part
is asserted to span less than 180° of longitude (the widest ring is 35°);
Chukotka's subject is 81 parts with a feature bounding box of -180 to 180,
which is the split, not a ring wrapping the world.

## Checks

**National.** 2025: 143,659,377; 2024: 143,679,916; 2021 census: 144,699,673.
Each is Rosstat's own national figure (146,119,928; 146,150,789; 147,182,123)
minus Crimea and Sevastopol (2025: 1,902,249 + 558,302). The 2021 figure
equals the 83-subject total religiondots parsed from Wikipedia. Rosstat's
preliminary national tables say 146,028,325 (2025) and 146,203,613 (2024);
the municipal table is the later, revised estimate.

**Subjects.** For 2024 and 2025 every subject is the exact sum of its units and
equals the subject row of the Rosstat table. Against Rosstat's preliminary
tables, 78 subjects found by name are within 0.35% in both years (Arkhangelsk
and Tyumen are printed there with their autonomous okrugs, so not comparable;
Kemerovo, Khanty-Mansi and St Petersburg were not found by name).

**Every unit has a polygon and every polygon has population**: 2,322 of 2,322
Rosstat units are on a polygon; build_country.py reports 2,295 of 2,295
level-2 features and 83 of 83 subjects with data. No two polygons overlap by
more than 38 km² (Anadyr town and Anadyrsky district, an OSM drawing error).

**Area against Wikidata** (`check_area.py`), the test of the polygon-to-unit
join: 1,453 polygons have a Wikidata area; 92.0% are within 10% of it and
95.7% within 25%. The outliers are Wikidata's (Altai Republic's districts are
entered in thousands of km² too small; Oymyakonsky 11,863 against a real
~92,000) or disputed borders (Chechnya-Ingushetia, Dagestan), plus
Nizhneilimsky district of Irkutsk at half its Wikidata area, not chased.

**Population against Kontur** (`check_kontur.py`, Kontur r6 2023): national
0.986. 53 of 83 subjects within 10%. At level 2 only 18.6% of units are
within 10% (median Kontur/Rosstat 1.19), and it is Kontur's grain, not the
join: an r6 hex is ~36 km², so a town of 100 km² loses its people to the
district around it (Krasnodar 0.62, Novosibirsk 0.76, Novosibirsky district
2.50), and Kontur places some towns badly (it puts 177,795 of Rybinsk's people
in one hex 9 km south of the city, so Rybinsk reads 0.12 and Rybinsky
district 8.60). The area check above is the one that tests the join. A first
attempt used religiondots' `ru_grid_3km.gpkg`, which is clipped to the 2017
subject outlines and reads Zelenograd, Kronstadt and other places as empty.

## Weaknesses

- The 2026 estimate is a straight line through two consecutive 1 January
  estimates, so a unit's one-year change is extrapolated as it stands.
- 24 polygons carry two or three Rosstat units because OSM is a year newer
  than the table; the biggest is Nizhny Novgorod + Kstovsky (1.34 million).
- Nizhny Tagil's shape is the gap its neighbours leave, not a drawn boundary.
- English names are OSM's `name:en` where present (with OSM's typos: "Rtbinsk",
  "Anadtrskiy") and a transliteration otherwise; the Russian line under them is
  Rosstat's name.
- Coasts follow Natural Earth's 10 m coastline, which is coarse at small towns.
- No municipal figures before 2024: the 2021 census has municipal names but no
  codes, on 2021 boundaries, and was not joined.

## Crimea and Sevastopol

`INCLUDE_CRIMEA = False` in `fetch.py`, the internationally recognised
borders. Rosstat counts both (1,902,249 and 558,302 in 2025) and setting it
True ships them as subjects UA-43 and UA-40 with their municipal units, using
OSM's Russian-tagged relations. That path was run once, which found two
problems that are now fixed (Ukrainian-tagged raions slipping through the
filter, and Bakhchisaray district mapped twice) and one that is not:
Sevastopol's Inkerman finds no polygon and Gagarinsky is drawn from the gap,
sea included. It has not been re-run since, so look at it before shipping. The four Ukrainian oblasts
Russia has claimed since 2022 are not in Rosstat's municipal table at all.

## Ethnicity pies

`ethnicity.py` writes `countries/russia/composition.json`: nationality
(natsional'nost', i.e. ethnic group) from the 2021 census at both levels. Source:
"Settlements of Russia: population, ethnic composition, and geographic
coordinates", Rosstat data processed by To Be Precise (tochno.st/datasets/allsettlements),
CC BY, downloaded 2026-10-03 to `data/russia/raw/tochno/`. It counts all 194
census nationalities for every settlement and municipality, and is built on
the settlement database Seva Bashirov published from the census.

- Counts are the census's own 2021 municipality rows, which are complete;
  settlements of 10 people or fewer have nationality blanked.
- Those rows are on 2021 municipalities. Each is carried onto today's polygons
  through its settlements, placed by coordinates in `adm2.gpkg`. The table's
  "current" settlement codes are a year newer than Rosstat's 2025 table, and
  Altai and Krasnoyarsk renumbered in between, so codes don't work. 2,300
  municipalities land whole in one polygon, 32 (Moscow's okrugs, St
  Petersburg's districts) go by code, and 9 are split by their own settlements:
  Krasnoyarsk's two suburbs, Ramenskoye, Solnechnogorsk, and the districts
  Saratov, Tambov, Chelyabinsk, Rostov's Myasnikovsky and Grozny took land from.
  A polygon must hold 5% of a municipality's people to take a share, so a
  village geocoded over a simplified border doesn't count as a split.
- Kronstadt's census row carries its municipal code 40360000; aliased to 40280000.
- 82 groups get a colour: 50,000 people nationally, or 10% of a municipality of
  1,000+. The rest, and answers the census files under no listed nationality,
  are "Other". "Not stated" is people with no nationality on the form (16.6 M,
  mostly counted from administrative records, heaviest in big cities) plus those
  who said they had none. Census categories are kept as they are, so Erzya and
  Moksha sit beside Mordvins, Hill Mari beside Mari, Todzha beside Tuvans.
- People could give two nationalities and both are counted. Nationally that
  adds 0.26%, but where a sub-group answer is common it is larger: Mordovia
  +7.8%, Altai +3.7%, Mari El +2.3%, Tuva +2.2%, Dagestan +2.1%. Those slices
  partly hold the same people.
- Palette: `ethnicity_colors.csv`, hand-editable; the script only appends.
  Families take hue bands: Slavic blue, Turkic red-brown, Uralic green,
  Caucasian purple, Mongolic yellow, Siberian teal, other Indo-European ochre.

## Files

- `fetch.py` — Rosstat downloads, parse, crosswalk; writes `data/russia/units.csv`,
  `subjects.csv`, `links_2024.csv`, and `population.csv` once boundaries exist.
- `rosstat_mo.py` — the workbook parser and code repairs.
- `crosswalk.py` — 2024 to 2025 unit links and rebasing.
- `fetch_osm.py`, `fetch_wikidata.py` — downloads.
- `prep_boundaries.py` — polygons, matching, `data/russia/boundaries/adm1.gpkg`,
  `adm2.gpkg`, `unit_map.csv`, `match_report.txt`.
- `check_kontur.py`, `check_area.py` — the checks above.
- `boundaries.json` — build_country config (level 2 split by subject).
