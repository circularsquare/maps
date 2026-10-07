# Kazakhstan — helper1m fetcher

Two levels: 20 regions (17 oblasts plus the cities of Astana, Almaty and
Shymkent) and 224 rayons, cities of regional significance and city districts,
on the map of 1 January 2026. Years 2021, 2025 and 2026.

| level | label | units | years | source |
|---|---|---|---|---|
| 1 | Oblast / city | 20 | 2021, 2025, 2026 | sum of level 2 |
| 2 | Rayon / city | 224 | 2025, 2026 (all); 2021 (220 of them) | BNS bulletin, census 2021 |

Run order, all from `scripts/kazakhstan/` with `C:\Python39\python.exe`:

```
fetch_osm.py         # Overpass, cached in data/kazakhstan/raw/ (about 15 min, rate limits)
fetch.py             # population.csv (seconds)
prep_boundaries.py   # boundaries/adm1.gpkg, adm2.gpkg (about a minute)
check_kontur.py      # the independent check (about a minute)
..\build_country.py kazakhstan   (from the repo root)
```

## Sources

**1 January 2026 and 2025: the annual bulletin.** Bureau of National
Statistics, "Численность населения по полу и по типу местности", published each
March, table 2 "...в разрезе областей, городов и районов". It is linked from the
publication pages "Численность населения Республики Казахстан по полу и типу
местности (на начало 2026г.)" (stat.gov.kz/ru/industries/social-statistics/demography/publications/281559/)
and the same for 2025 (.../281558/). The files are
`https://stat.gov.kz/api/iblock/element/341322/file/ru/` (2026, .xlsx) and
`.../330829/file/ru/` (2025, .xls). Table 3 of the same file adds district
centres and urban settlements, which this does not use. No KATO codes are in
the bulletin; every row is matched to KATO by name inside its region, 1:1, and
the script stops on any miss. Two spellings needed an alias (`Саркандский`,
`Чиилийский`, both 2025).

These are register-based estimates: the 2021 census rolled forward with
births, deaths and registered moves. 2025 and 2026 are therefore on the same
footing, and the current-year figure the viewer shows is the 1 Jan 2026
estimate itself.

**2021: the census.** National Population Census 2021, table 3.1 of
"Численность населения Республики Казахстан по этносам, населенным пунктам и
возрасту" (published 28.11.2025), read from religiondots' copy at
`religiondots/data/raw/kz/kz2021_ethnos_settlement.xlsx`. It has every
settlement with its 2021 KATO code: 17 oblasts, 218 rayons, 2,332 rural
okrugs, 7,049 settlements, 19,186,015 people.

**KATO**, the administrative-territorial classifier, edition of 18.09.2026,
`КАТО_18.09.2026.xlsx` from stat.gov.kz/ru/classifiers/statistical/21/. It
gives every current unit its 9-digit code and lists every settlement under it.
Codes in the output are current KATO codes (region = first two digits +
`0000000`, e.g. `100000000` Abay; rayon = 9 digits, e.g. `103400000` Aksuat).

**Boundaries: OpenStreetMap**, admin_level 6, 226 relations, fetched through
Overpass in October 2026 (`fetch_osm.py`). OSM was the only open source with
the units created in 2022-24. OCHA's COD-AB (2023, `data/asia1m/kazakhstan/`)
has 218 rayons by coincidence of the same count as the census: it has Aksuat
and Samar (2022) but not Kosshy, Sauran, Zhanasemey, Makanshy, Markakol, Ulken
Naryn, Alatau city, or any of Astana's or Shymkent's new districts.

COD-AB is still used twice: its 20 regions place each OSM relation in its
region (which resolves the three "Abay District"s, three "Esil District"s and
so on), and its English names label units it shares with OSM. A unit counts as
shared when each holds at least 70% of the other's area. New units and renamed
ones (Taran is now Beimbet Mailin, Zelenov is Baiterek, Lebyazhye is Akkuly)
get names from a table in `prep_boundaries.py`.

## How the geography was made to agree

The 1 January 2026 map has 228 KATO units. 224 are shipped.

**Shymkent is one unit (1,294,050).** Its fifth district, Turan, dates from
2022 and the other four were redrawn around it. OSM has only the four old
districts, and they fill the whole city outline, so there is no gap to find
Turan in; and their 2025 populations no longer fit those shapes (Al-Farabi
191,578 in 2021, 266,806 in 2025). `common.MERGES` holds this.

**Astana's Saraishyk district is the city minus its other five districts.**
Saraishyk was split from the south of Almaty district on 29 January 2025. OSM
has no relation for it, but its Almaty district is already the reduced one:
93 km² by our measure against the 8,518 ha published for the remainder, and
Astana's outline minus the five OSM districts leaves 72 km² in the south-east,
against Saraishyk's published 6,953 ha (orda.kz, "Как делили Астану").
`common.CITY_GAPS` holds this.

**OSM rayons are drawn round the cities they surround, without a hole.**
Pavlodar District's polygon contains Pavlodar city, for one. Each unit loses
whatever a smaller unit covers; Pavlodar District loses 8.8% of its area,
Baydibek 1.0%, everything else under 0.3%. The remaining gaps in the coverage
are slivers along Almaty city's district borders (1.2 km² in all) and one
3 km² hole on the Akmola-Kostanay border.

**OSM name join.** 226 relations, all matched to a KATO unit; 205 by exact
Russian name, 16 by the first five letters of the name (all listed by the
script and checked), 5 by hand in `OSM_IDS` (Кызылкогинский, Костанайская
Г.А., Ойылский, Жетисайский, Улькен Нарынский).

### 2021 onto the 2026 map

Most census rayons pass whole to the current unit with the same code (or, in
the three oblasts split in 2022, the same name). Nine rayons were split or lost
land since, and those go settlement by settlement (`SPLITS` in `fetch.py`):

| census rayon | went to |
|---|---|
| Semey city admin. | Semey + Zhanasemey District |
| Urdzhar | Urzhar + Makanshy |
| Tarbagatay | Tarbagatay (East Kazakhstan) + Aksuat (Abay) |
| Kokpekty | Kokpekty (Abay) + Samar (East Kazakhstan) |
| Kurchum | Kurchum + Markakol |
| Katon-Karagay | Katon-Karagay + Ulken Naryn |
| Ili | Ili + Alatau city |
| Zhambyl District, Baizak | themselves + Taraz (2025) |

Each census settlement goes to the current unit that lists a settlement of the
same name among that rayon's successors. Twins are settled by the name of the
rural okrug, then by the tail of the code. The code is used only for a
settlement renamed since (its old name is nowhere), because codes were
reassigned inside split rayons: `635430100` was Ulken Naryn village in 2021
and is Katon-Karagay village now. The first version trusted codes first and
put Ulken Naryn's people in Katon-Karagay (2021: 16,374 against 9,218 in
2025). Settlements gone from KATO altogether went to the city that absorbed
them: 11 in Ili to Alatau (41,763 people), 11 in Zhambyl District and 3 in
Baizak to Taraz (34,897 and 10,656). What is left over: 154 people in 6
vanished villages stay with the rayon's main successor, as do 290 people in 2
villages whose name occurs on both sides of a split.
`data/kazakhstan/xwalk_2021.csv` lists every settlement and how it went.

Astana's Almaty, Esil and Nura districts and Saraishyk have no 2021 figure:
Nura was carved from Esil and Almaty in 2022 and Saraishyk from Almaty in 2025,
and the census has nothing below the city district to rebuild them from.
Shymkent gets 2021 as the city total.

### 2025 onto the 2026 map

The 2025 bulletin has 227 units, the 2026 list minus Saraishyk. Two changes
happened during 2025:

* Taraz took 14 villages from Zhambyl and Baizak districts (Zhambyl District
  went from 88,156 to 51,354). The moved share of each district's 2025 figure
  is the share of its 2021 census population living in the absorbed villages:
  40.3% of Zhambyl District (35,539) and 10.4% of Baizak (11,032).
* Astana's Almaty district (409,873) is split in its 2026 proportions: Almaty
  232,041, Saraishyk 177,832. This sets both to the same 2025-26 growth.

## Checks

* **National totals**, all exact: 2021 19,186,015 (census); 2025 20,283,399;
  2026 20,499,822 (the March annual figures; the February press release had
  20,495,975 for 2026).
* **Regions**: in 2025 and 2026 all 20 region totals in the bulletin's table 1
  equal the sum of their rayons. In 2021 the 20 current regions, folded back
  to the census's 17, equal all 17 census oblast totals.
* **Every unit has data and every row has a unit**: 20 of 20 and 224 of 224
  features carry population; no population code lacks a feature.
* **Year-to-year consistency** (catches a missed boundary change): for each
  unit, annual growth 2021-25 against 2025-26. Median -0.5% a year against
  -1.8%; 38 of 220 units differ by more than 2 points and none by more than
  3.4. Before the settlement crosswalk was fixed, Ulken Naryn and Aksuat stood
  out at -25 and -17 points.
* **Kontur 400 m hexagons (1 Nov 2023)**, summed by centroid, against 2025,
  normalised by the national ratio (0.967). This is a weak check here. Kontur
  moves city people out to the countryside on a large scale: Astana reads 0.55
  of the bulletin and Almaty city 0.67, Taraz 0.26, while Tselinograd District
  round Astana reads 5.2. Per unit, only 19.6% are within 10% (49% within
  0.8-1.25). Pooling each city with the rayons it touches (21 pools) gives 52%
  within 10% and 81% within 0.8-1.25. The 98 rayons that touch no city sit at a
  median of 1.21; against that median, 54% are within 10% and 83% within
  0.8-1.25.
  Worst pools: Taraz with Baizak and Zhambyl districts 0.33 (Kontur's densest
  Taraz hexes hold about 1,200 people; it misses most of the city); Aktau with
  Munaily and Tupkaragan 0.67; Zhanaozen with Karakiya 1.51; Baikonur with
  Karmakshy 2.43. The last may be real: Baikonur is leased to Russia, the
  bulletin counts 31,233 there, and Kontur sees far more people. Worst lone
  rayons: Taskala 2.56 and Borodulikha 2.28.

## Known weaknesses

* Shymkent is a single 1.29 M unit.
* Saraishyk's outline is inferred as a gap, not drawn from a source; its area
  agrees with the published one to 4%.
* OSM's outlines are of unknown date unit by unit. The checks above show no
  join error, but Kontur is too poor here to catch an outline that is a few
  percent off.
* The 2025 figures for Zhambyl District, Baizak, Taraz, Almaty District and
  Saraishyk are apportioned, not published.
* 2021 for the seven units created since is rebuilt from settlements, so it is
  as good as the village-name matching (444 people left unplaced).

## Not done

* **2009 census.** The religiondots Qlik engine (`qap.stat.gov.kz`, see
  `religiondots/sources/kz.md`) holds 2009 person rows with a rayon field, but
  they are on 2009 KATO, before the 2018 split of South Kazakhstan into
  Turkistan and Shymkent and the 2022 oblasts. Carrying them onto today's map
  is a second crosswalk for a context-only year, so it was left.
* **Other annual years** (2022-2024) exist in the same series; two estimate
  years plus the census are all the viewer needs.
