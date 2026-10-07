# Tajikistan — helper1m fetcher

Two levels: region (5) and city / district (65). These are the units the
Agency on Statistics (www.stat.tj, licence CC BY 4.0) publishes every year.
Jamoats were not attempted: no jamoat population table turned up (the census
volumes only group jamoats by size), and OSM has jamoat outlines for a handful
of districts only.

Run order (from the repo root, with `C:\Python39\python.exe`):

```
helper1m/scripts/tajikistan/download.py        # raw files, ~30 MB, skips what is there
helper1m/scripts/tajikistan/prep_boundaries.py # adm1.gpkg, adm2.gpkg (~1 min, reads the pbf)
helper1m/scripts/tajikistan/fetch.py           # population.csv, prints the checks
helper1m/scripts/build_country.py tajikistan
helper1m/scripts/tajikistan/check_kontur.py    # optional, the Kontur comparison
```

`units.py` is the list of the 65 units: code, English and Tajik names, which
OSM polygon each one is, and the label each one has in the bulletin and in the
census table.

## Units and codes

Region codes are ISO 3166-2: TJ-GB Gorno-Badakhshan, TJ-SU Sughd, TJ-KT
Khatlon, TJ-DU Dushanbe, TJ-RA Districts of Republican Subordination. City and
district codes are the region code plus the unit's position in the bulletin
(TJ-KT-06 is Kushoniyon). They are our own; nothing official joins to them.

- Gorno-Badakhshan: Khorugh city and 7 districts.
- Sughd: 8 cities (Khujand, Isfara, Guliston, Konibodom, Panjakent,
  Istaravshan, Istiqlol, Buston) and 10 districts.
- Khatlon: 4 cities (Bokhtar, Kulob, Norak, Levakant) and 21 districts.
- Dushanbe: one unit.
- Districts of Republican Subordination: 4 cities (Vahdat, Tursunzoda, Hisor,
  Roghun) and 9 districts.

Most of these cities are a town plus the rural district around it. Isfara,
Konibodom, Panjakent, Istaravshan, Kulob, Norak, Vahdat, Tursunzoda, Hisor,
Roghun and Levakant (the old Sarband) each absorbed their district, and the
bulletin reports them as "city, urban settlements, rural area". So the
bulletin's Kulob city (236,500 in 2025) is the old Kulob district with the
town, and it is drawn with the OSM polygon that OSM still calls Kulob District.
Only Khujand, Guliston, Buston, Istiqlol, Bokhtar and Khorugh are towns without
a rural district attached.

## Boundaries

`prep_boundaries.py` reads the Geofabrik extract `tajikistan-261001.osm.pbf`
through GDAL's OSM driver. OSM's admin_level 6 is the current district map,
with the current names (Kushoniyon for the old Bokhtar district, Jayhun for
Qumsangir, Dusti for Jilikul, Jaloliddini Balkhi for Rumi, Shamsiddin Shohin
for Shuroobod, Levakant for Sarband), and has Khujand, Guliston and Buston as
their own level-6 units. Dushanbe is the level-4 relation; its four districts
are in OSM at level 7 but have no current population. The 62 relations tile
the country with no gap or overlap bigger than 0.5 km².

Three cities the agency reports separately have no boundary in OSM and sit
inside a district polygon: Bokhtar (in Kushoniyon), Khorugh (in Shughnon) and
Istiqlol (in Bobojon Ghafurov). Each is cut out of its district using its OSM
place outline (`CITY_OUTLINES`): Bokhtar way 38153741 (13.5 km²), Khorugh way
1367629514 (9.7 km²), Istiqlol relation 11142161 (10.3 km²). These are the
built-up towns, not legal city limits. For Khorugh and Istiqlol that is close
enough. For Bokhtar it is too tight: Kontur puts only 31,000 people inside the
outline against an official 130,800, with the rest in the edges of Kushoniyon
and Vakhsh. Shift-clicking Bokhtar with Kushoniyon and Vakhsh gives the right
total for the three.

Guliston has the same problem the other way round. Its official figure
(51,400 in 2025) includes six urban settlements: Adrasmon, Konsoy, Zarnisor and
Navgarzan in the mountains north of Khujand, and Sirdaryo and Chorruq-Dayron by
the Kayrakkum dam. Together they hold about 34,000 people, all of them outside
the 9 km² OSM outline, mostly inside the Bobojon Ghafurov polygon (Adrasmon
checked). That puts about 34,000 people on the Guliston polygon who live
roughly 10-40 km away, all inside Sughd.

Region polygons are dissolved from the 65 units, so the levels nest exactly.
Their areas match the bulletin's region areas (GBAO 62.8 vs 62.9 thousand km²,
Sughd 25.4 vs 25.2, Khatlon 24.6 vs 24.7, Dushanbe 0.198 vs 0.2, Republican
Subordination 28.5 vs 28.4).

geoBoundaries ADM2 for Tajikistan (gbOpen, OSM via Wambacher, 2017 vintage,
58 units) was checked first and not used: it has no cities at all, no Dushanbe
split, and the pre-2016 district names. HDX COD-AB (`cod-ab-tjk`) holds only a
2017 name and code list; OCHA keeps the geometry private.

## Population

### Annual bulletin, 1 January 2021-2025

"Шумораи аҳолии Ҷумҳурии Тоҷикистон то 1 январи соли 20XX / Численность
населения Республики Таджикистан на 1 января 20XX года", table "Численность
постоянного населения поселков, городов, районов и областей". Thousands to one
decimal, so each unit is rounded to the nearest 100. Each issue gives two
years:

| file | stat.tj path | years |
|---|---|---|
| bulletin_2025.pdf | wp-content/uploads/2025/12/machmuai-shumorai-aholi-to-1.01.2025.pdf | 2024, 2025 |
| bulletin_2024.pdf | wp-content/uploads/2024/09/machmuai-shumorai-aholi-to-1.01.2024.pdf | 2023, 2024 |
| bulletin_2022_corrected.pdf | wp-content/uploads/2024/08/machmuai-shumorai-aholi-to-1.01.2022-ispravlenij.pdf | 2021, 2022 |

No 1 January 2026 issue was on the site on 2026-10-02 (the 2025 issue went up
on 31 December 2025, so 2026's is likely around the end of this year). No 2023
issue was found either; 2023 comes from the 2024 issue.

The PDFs have a text layer in an old Tajik font where Ҳ Ҷ Ғ Қ Ӣ Ӯ come out as
Њ Љ Ѓ ќ ї ў, which is what the regexes in `units.py` match. `fetch.py` groups
spans into rows by their position on the page and walks the units in table
order, each search starting after the previous match, so a town row that
shares a district's name (Buston settlement in Mastchoh, Vahdat settlement in
Lakhsh) is never taken for the unit.

Typos in the bulletin, and what was done:

- Isfara, 1 January 2025: the unit row shows 56.7, which is the town alone. The
  settlements (Isfara 56.7, Shurob 3.1, Nurafshon 1.5, Neftobod 4.4) plus the
  rural area (227.6) give 293.3, and with that Sughd's units come within 0.6
  thousand of the Sughd total. The urban-settlements row repeats 2024's 65.1,
  so it is wrong as well.
- Murghob, 1 January 2025: printed "17" without a decimal; its two sub-rows give
  16.6.
- Faizobod, 1 January 2024, in the 2025 issue: the rural row is 101.5 where the
  2024 issue has 103.2; the unit row (116.7) agrees across both issues and is
  used.
- Guliston: the urban row is off by a few hundred; the unit row agrees with the
  sum of its settlements and is used.

A general rule in `fetch.py` replaces any unit row that is more than 2% away
from its own urban plus rural rows; only Isfara 2025 trips it.

### Census 2020, 2010 and 2020

`census2020_vol1_table1_ru.pdf`: volume I, table 1, "Численность постоянного
населения по областям, районам, городским поселениям, районным центрам и
сельским населенным пунктам с числом жителей 5 тысяч и более", exact counts for
the 2010 and 2020 censuses on the current list of units. Path:
wp-content/uploads/2024/05/tablicza-1.-chislennost-postoyannogo-naseleniya-po-oblastyam-rajonam-gorodskim-poseleniyam-rajonnym-czentram-i-selskim-naselennym-punktam-s-chis.pdf

Dushanbe and Rudaki are left out for both census years. Between the October
2020 census and the bulletin's first post-census year, part of Rudaki went to
Dushanbe: the census has Dushanbe 948,251 on 126.6 km² and Rudaki 603,337,
the bulletin for 1 January 2021 has 1,185,400 and 377,800. Together they move
from 1,551,588 to 1,563,200 (+0.7% in a quarter), which is a straight transfer
of about 230,000 people. Dushanbe's OSM polygon (198 km²) is the enlarged city,
which matches the bulletin's own density figure (1,267.5 thousand at 6,338 per
km² is 200 km²). The four Dushanbe districts have census 2020 figures (Sino
397,059, Firdavsi 220,757, Shohmansur 169,250, Ismoili Somoni 161,185) but only
on the old city, so Dushanbe is one unit. Region rows for Dushanbe and the
Districts of Republican Subordination have no 2010 or 2020 for the same reason.

Bokhtar's 2010 figure (75,450, +65% to 2020) is probably on a smaller city
territory; Kushoniyon grew only 17% over the same years. 2010 is context only
and was left as printed.

## Checks (fetch.py and check_kontur.py print all of these)

Sum of units against the region and national rows of the same bulletin, in
thousands:

| year | GBAO | Sughd | Khatlon | Dushanbe | Rep. Sub. | national (published) |
|---|---|---|---|---|---|---|
| 2021 | 228.5 / 228.4 | 2,783.0 / 2,783.0 | 3,459.7 / 3,459.7 | 1,185.4 | 2,060.3 / 2,060.3 | 9,716.9 (9,716.8) |
| 2022 | 230.2 / 230.1 | 2,823.9 / 2,823.9 | 3,530.1 / 3,530.0 | 1,201.8 | 2,101.0 / 2,101.0 | 9,887.0 (9,886.8) |
| 2023 | 232.0 / 232.0 | 2,870.0 / 2,870.0 | 3,611.2 / 3,611.2 | 1,221.1 | 2,144.1 / 2,144.1 | 10,078.4 (10,078.4) |
| 2024 | 233.5 / 233.6 | 2,917.3 / 2,917.3 | 3,697.5 / 3,697.8 | 1,242.6 | 2,196.9 / 2,197.0 | 10,287.8 (10,288.3) |
| 2025 | 234.7 / 234.8 | 2,965.5 / 2,966.1 | 3,790.3 / 3,790.3 | 1,267.5 | 2,249.9 / 2,249.8 | 10,507.9 (10,508.5) |

Every gap is within rounding of up to 25 units each rounded to 0.1 thousand.

Census: the 65 units sum exactly to the national count in both years
(7,564,502 in 2010, 9,657,005 in 2020) and to every region's row.

Census (1 October 2020) against the bulletin for 1 January 2021, unit by unit:
all 63 units other than Dushanbe and Rudaki are within -2% to +4%, so the
bulletin series starts on the census and on the same territory. From 2024 to
2025 every unit grows 0-4% except Murghob (16,700 to 16,600).

The 1 January 2024 unit figure is identical in the 2024 and 2025 issues for
every unit (Faizobod's disagreement is in a sub-row only).

Independent check, Kontur population (400 m hexagons, release 2023-11-01,
from HDX) summed in each polygon against the 2024 figure, as shares:

- by region, Kontur / official: Sughd 1.01, Gorno-Badakhshan 1.14, Republican
  Subordination 1.14, Dushanbe 1.19, Khatlon 0.84.
- as share of the nation: 17 of 65 units within 10%, 41 within 20%.
- as share of their own region (which takes out the regional bias): 34 of 65
  within 10%, 51 within 20%.

Worst within their region: Bokhtar 0.29 (outline too tight, see above), Khujand
0.53, Istiqlol 0.62, Shahritus 0.64, Dusti 0.64, Bobojon Ghafurov 0.71; and on
the high side Levakant 1.78, Vakhsh 1.36, Kuhistoni Mastchoh 1.35, Shamsiddin
Shohin 1.34. Levakant, Vakhsh and Kushoniyon are high because Bokhtar's people
are in them. The polygon areas of Shahritus and Dusti match the bulletin's own
areas (1.59 and 1.25 thousand km² against 1.5 and 1.2), so those gaps are not
boundary errors. Kontur is a model (GHSL and Facebook settlement layers scaled
to UN totals), and two things push it away from the official figures here: it
counts people where buildings are, while the agency's permanent population
includes people temporarily abroad, many of them labour migrants; and the
Khujand area comes out low in Kontur all round (Khujand, Ghafurov, Guliston,
Buston and Istiqlol together at 0.65). Nothing here argues for changing an
official figure.

Every boundary unit has population for 2021-2025 (63 of 65 also for 2010 and
2020), and every population row has a polygon.

## What was tried and failed

- Overpass API (overpass-api.de and overpass.kumi.systems) timed out on a
  query for Tajikistan's admin relations on 2026-10-02; the Geofabrik extract
  replaced it.
- Current Dushanbe district populations: not in the 2024 or 2025 bulletins, no
  text hit in the demographic yearbook (`demog-01.01.2024.pdf`, 15 MB, may be
  partly scanned), nothing by title in the stat.tj media library (3,569 files
  listed); one web search found only the census 2020 figures.
- data.stat.tj, old.stat.tj and nada.stat.tj were not needed and not retried;
  see `religiondots/sources/tj.md` for their state.
