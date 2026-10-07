# Uzbekistan

Two levels: 14 regions (12 viloyats, the Republic of Karakalpakstan and Tashkent
city) and 199 districts and cities (tuman and shahar), on the OCHA COD-AB 2018b
boundaries. Populations are the Statistics Committee's 1 January estimates for
2011-2026, by district, with each region scaled to its 2026 census count.

```
C:\Python39\python.exe helper1m\scripts\uzbekistan\fetch.py            # population.csv, units.csv, events.csv
C:\Python39\python.exe helper1m\scripts\uzbekistan\prep_boundaries.py  # boundaries/adm1.gpkg, adm2.gpkg
C:\Python39\python.exe helper1m\scripts\build_country.py uzbekistan
C:\Python39\python.exe helper1m\scripts\uzbekistan\check.py            # the checks below
```

`fetch.py --refresh` re-downloads the SIAT file. Everything runs in under a
minute; `check.py` takes about a minute the first time (Kontur sum), then caches.

## Sources

- **Population, every year:** National Statistics Committee (stat.uz), SIAT
  indicator 2.01.02.0001 *Permanent population - total*, thousands, 1 January,
  2010-2026, for the country, 14 regions and 206 districts and cities.
  `https://api.siat.stat.uz/media/uploads/sdmx/sdmx_data_246.json` (also `.xlsx`,
  `.csv`, `.xml`), no key. Last modified 2026-04-22; the copy fetched on
  2026-10-02 is byte-identical to `religiondots/data/raw/uz/uz_siat_sdmx_246_population.json`
  from September, so there is no fresher version. Districts sum exactly to their
  region in every year, and regions to the country. "Permanent population" is
  register-based: the 1989 census carried forward with registered births, deaths
  and moves.
- **Level, 2026:** *Preliminary Results of the Population and Agriculture Census
  of the Republic of Uzbekistan, 2026*, English edition,
  `https://stat.uz/img/news/english_natija_merged-2_p42445.pdf` (cached at
  `religiondots/data/raw/uz/uz_census2026_results_en.pdf`), printed p. 23,
  *Distribution of the population by sex, by region*. Census moment 15 January
  2026, total 39,047,321. Regions only; the 14 figures are typed into
  `fetch.py` (`CENSUS_2026`).
- **Boundaries:** OCHA COD-AB `uzb_admbnda_adm2_2018b` (199 units), from
  `maps/data/asia1m/uzbekistan/`. geoBoundaries' UZB ADM2 (labelled 2020) is the
  same geometry vertex for vertex. Regions are dissolved from the districts, so
  the levels nest exactly.

## What each level is

**District / city (199).** SIAT codes are SOATO; the COD pcode is the same
number with `UZ` for the leading `17` (SIAT `1735401` Nukus city = COD
`UZ35401`). All 199 COD units are in SIAT, so the join is by code with no name
matching. SIAT has seven more units, all created after the COD file was drawn,
and its series also record a handful of transfers of land between units that
both exist in the COD file. `fetch.py` folds both back onto the 2018 polygons.

How a transfer is undone: in the year a unit's series steps, its step is its
change that year minus its usual change (the median of its changes in the three
years either side). Units that rose hand back their rise, units that fell get it
back in proportion to their fall, and the amount follows the receiving unit's
own later growth. A unit the COD file lacks is handed back whole in every year.
`data/uzbekistan/events.csv` lists every move; the events, with the amount moved
in the year of the step (census-scaled thousands):

| year | event | moved |
|---|---|---|
| 2018 | Takhiatash district (re-formed) back into Khojeyli | 75.3 |
| 2020 | Bozatau district back into Kegeyli (18.4) and Chimbay (4.2) | 22.5 |
| 2020 | Gazgan city back into Nurota; Navoi city's gain back to Karmana | 7.7, 6.1 |
| 2020 | Bandikhan district back into Kizirik (59.4), Kumkurgan (9.3), Baysun (4.6); Termez city's gain back to Termez district | 73.6, 31.6 |
| 2021 | Tuprakkala district back into Khazarasp | 56.3 |
| 2021 | Yangihayot district (Tashkent city) back into Sergeli (111.5), Urtachirchik (11.6), Bektemir (7.0), Zangiata (4.7) | 134.8 |
| 2022 | Tashkent city's gains in Bektemir, Mirzo Ulugbek, Yashnabad and Sergeli back to Zangiata (36.7), Kibray (37.6), Urtachirchik (12.2) | 86.5 |
| 2022 | Tashkent region internal: Yangiyul city, Akhangaran district and Parkent hand back to Yangiyul district, Angren city and Yukorichirchik | 48.5 |
| 2023 | Kukdala district back into Chirakchi | 182.1 |
| 2023 | Kokand and Fergana cities' gains back to Uzbekistan, Dangara, Fergana and Uchkuprik districts | 48.0 |

The donors were read off the series (which units fall in the year another
appears) and checked against the SIAT metadata note, which dates each new unit.
Units that took in a whole new district carry it in their name, e.g. "Chirakchi
district (incl. Kukdala)", "Sergeli district (incl. most of Yangihayot)".

The consequence that matters: **Tashkent city here is its 2018 extent**, so it
reads 131,998 below its 2026 census figure and Tashkent region the same amount
above (111,323 on SIAT's own level). The city's districts are the 2018 ones.

Steps that happened before the COD vintage changed some units' territory, so
their rows start later: 2018 for Shahrisabz, Khiva, Nurafshon, Akhangaran and
Yangiyul cities and the districts they came out of, Tashkent district and
Zangiata, Khojeyli, Zomin and Zarbdar; 2017 for Namangan city, Namangan and
Uychi districts; 2012 for Samarkand city and district. 2010 is dropped
everywhere: the country rises 1.12 million from 2010 to 2011 against 0.4-0.75
million in every other year, and every district jumps with it.

Small steps left alone (under 5% of the unit): Kumkurgan +10k in 2019, Kanimekh
+5k in 2019, Almalyk +5k in 2023, and Tashkent city districts' fast growth in
2021-25 (Bektemir, Sergeli, Yakkasaray), which is building, not boundaries.

**Region (14).** The sum of its districts in every year, so the levels agree.

## The census level (`CENSUS_LEVEL` in `fetch.py`, default on)

The census counted 810,617 more people than the register estimate for 1 January
2026 (2.1%). By region, census over SIAT:

| region | SIAT 1 Jan 2026 | census 15 Jan 2026 | ratio |
|---|---:|---:|---:|
| Tashkent region | 3,160,700 | 3,763,093 | **1.191** |
| Navoi | 1,111,900 | 1,184,591 | 1.065 |
| Karakalpakstan | 2,053,200 | 2,149,932 | 1.047 |
| Khorezm | 2,069,200 | 2,140,746 | 1.035 |
| Tashkent city | 3,178,100 | 3,224,838 | 1.015 |
| Syrdarya | 946,300 | 954,361 | 1.009 |
| Samarkand | 4,379,800 | 4,404,575 | 1.006 |
| Fergana | 4,223,000 | 4,238,911 | 1.004 |
| Andijan | 3,521,800 | 3,531,777 | 1.003 |
| Bukhara | 2,107,000 | 2,104,874 | 0.999 |
| Kashkadarya | 3,717,000 | 3,692,323 | 0.993 |
| Surkhandarya | 3,011,000 | 2,984,084 | 0.991 |
| Namangan | 3,190,800 | 3,149,161 | 0.987 |
| Jizzakh | 1,566,900 | 1,524,055 | 0.973 |
| country | 38,236,700 | 39,047,321 | 1.021 |

With the switch on, every year of a region's districts is multiplied by that
region's ratio: the level is the census's, the year-to-year growth is SIAT's,
and the viewer's line through 2025-2026 lands on the census figure. The census
is not added as a separate year, which would make the line extrapolate the
jump between the two methods. With the switch off the output is SIAT as
published.

Why on: a register carried forward for 37 years against a full count with 97.3%
post-enumeration coverage, and the two agree within 3.5% in ten regions, so
the census is not off on a tangent. The one large gap, Tashkent region at
1.19, is the size that matters: switched off, the region's districts would
average 16% low, outside the 10% target almost everywhere. Its likely cause is
people living around the capital while registered elsewhere, which is also why
Jizzakh, Namangan and Surkhandarya come out slightly high in the register.

What it cannot fix: every district in a region gets the same factor. In
Tashkent region the census's extra 600,000 most likely sit in the suburbs
(Zangiata, Kibray, Tashkent district, Urtachirchik, Yukorichirchik) more than in
Bostanliq or Bekabad, so the suburbs are probably still low and the outer
districts high. No district census figures exist to say by how much.

## Checks (`check.py`)

- **National, switch off:** matches SIAT in every year 2011-2026, largest gap 2
  people (rounding).
- **Regions, switch off:** match SIAT exactly in every year except Tashkent city
  and region from 2021, which differ by the land transfer (13,967 in 2021,
  89,650 in 2022, 111,323 in 2026).
- **Regions, switch on, 2026 against the census:** within 2 people everywhere
  except Tashkent city (-131,998) and region (+131,998), the transfer again.
  National 39,047,324 against 39,047,321.
- **Coverage:** 199 of 199 districts and 14 of 14 regions have population; no
  population row lacks a polygon. Districts carry 9 to 16 years, regions 16.
- **Kontur 2023 (400 m hexes summed by centroid) against our 2023 districts, as
  shares of the national total:** no use as a check here. 17% of districts within
  10%; 22% with each city folded into its neighbouring district and Tashkent
  left out. At region level Kontur agrees within 12% except Tashkent city (0.65,
  Kontur's known false block south of the centre, see `religiondots/sources/uz.md`
  §8 and §10). Two things break it below the region. Kontur is thin in city
  cores (Samarkand city's 34 km2 core polygon holds 80,732 Kontur people against
  606,000 registered). And the COD city polygons are poor, below.

## Known problems with the COD 2018b geometry

- **The whole file is a few km off.** Against OSM's district boundaries
  (Geofabrik extract of 2026-10-02, admin_level 6), same-named districts overlap
  with a median intersection-over-union of 0.65, and OSM's centroids sit a median
  0.9 km east and 2.8 km north of COD's, varying by region from about 0 (Fergana)
  to 5-6 km (Bukhara, Navoi, Karakalpakstan). It is not one shift: moving COD by
  the median offset only lifts the overlap to 0.71. Inside Tashkent city the
  districts are small enough that COD's polygon for one district mostly covers
  another (COD's Yunusabad lies mostly on OSM's Olmazor, its Shaykhantakhur on
  Uchtepa).
- With Kontur as the yardstick, on 137 districts and cities that need no
  reconciliation (Tashkent left out), SIAT's 2023 shares match Kontur's within
  10% for 35% of units on OSM polygons against 27% on COD polygons (within 25%:
  66% against 49%). Kontur is weak here either way, but OSM's geometry is the
  better of the two.
- OSM is not a drop-in replacement: it has 202 districts (three newer than
  SIAT's list: Davlatobod and Yangi Namangan out of Namangan city, Yangi
  Toshkent in Tashkent city), lacks Jizzakh, Termez, Urgench, Khiva and Shirin
  cities and Fergana district as polygons, and leaves about 6,000 km2 uncovered
  against the COD outline. It carries no SOATO codes, so the join is by name.
  `data/uzbekistan/raw/osm_admin_4_6.gpkg` keeps the extracted boundaries.

- **Kagan city and Kagan district were swapped** in the COD file: "Kagan
  district" was a 1.9 km2 speck with 2 Kontur people, "Kagan city" a 470 km2
  district. `prep_boundaries.py` swaps the pcodes back (OSM's Kogon shahri sits at
  39.70-39.75 N inside Kogon tumani, 39.57-39.92 N).
- **City polygons are small or off target.** Navoi city's polygon is 13 km2 at
  40.02-40.07 N; the town is at 40.07-40.13 N, so the town's people sit in
  Navbahor district's polygon. Zarafshan city (2.5 km2) is south of the town;
  Shirin (0.1 km2) and Akhangaran city (0.6 km2) are specks; Almalyk, Angren,
  Chirchik and Bekabad hold under a tenth of their population in Kontur.
  The populations are right for the named unit; the city is drawn in the wrong
  place or too small, and its surroundings look emptier than they are. Click a
  city together with the district around it.
- Bo'z district is Bo'ston since 2020; the name shown is SIAT's current one.

## Tried and not used

- **District census figures.** None published. The results PDF and the
  aholi.stat.uz xlsx (item 6258, which is the agriculture volume) stop at the
  region. The kun.uz piece of 13 March 2026 ("smallest districts and cities")
  quotes SIAT's 1 January 2026 estimates (Gazgan 9,500, Tomdi 15,200, ...),
  identical to the file used here, not census counts. The Committee says the
  rest of the census comes out through July 2027 on aholi.uz; when district
  counts appear they should replace the regional scaling.
- **census.stat.uz, module.stat.uz, hudud.stat.uz** are geo-fenced (TLS reset
  on every port); **data.egov.uz** refuses connections (`religiondots/sources/uz.md` §4).
