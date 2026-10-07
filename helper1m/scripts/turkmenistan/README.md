# Turkmenistan

Two levels: 7 top-level units (5 velayats, Ashgabat, Arkadag) and 53 etraps and
cities on the current OSM map. Years 2022 (census) and 2026 (the census grown at
the UN's national rate). Built 2026-10-06.

Run order, from the repo root, with `C:\Python39\python.exe`:

```
helper1m/scripts/turkmenistan/download.py         # raw files, skips what is there (~45 MB)
helper1m/scripts/turkmenistan/census.py           # census tables -> census_units.csv, census_settlements.csv
helper1m/scripts/turkmenistan/prep_boundaries.py  # adm1.gpkg, adm2.gpkg from the OSM extract
helper1m/scripts/turkmenistan/fetch.py            # population.csv, settlement_match.csv
helper1m/scripts/turkmenistan/check.py            # sums, census rows, Kontur comparison
helper1m/scripts/build_country.py turkmenistan
```

The OSM extract and the WPP file were deleted after the build was verified;
`download.py` gets them back (the extract by its dated Geofabrik name). The
census PDF and the Kontur boundaries are copied from religiondots' cache
(read only) when it has them. `fetch_log.txt` and `check_log.txt` in
`data/turkmenistan/` are the last run's output.

## Units and codes

Top level: ISO 3166-2 codes TM-S Ashgabat, TM-A Ahal, TM-B Balkan, TM-D Dashoguz,
TM-L Lebap, TM-M Mary, plus TM-AR for Arkadag (a city with velayat status since
2023; ISO has no code for it yet). Units below are the top-level code plus the
unit's place in English alphabetical order (TM-L-04 is Dovletli). They are our
own and join to nothing official. Names are the census's English spellings
(Vekilbazar, Charjev, Koneurgench), made from OSM's Turkmen names by a fixed
letter mapping (w>v, ş>sh, ç>ch, ý>y, ä/ö/ü/ň dropped to a/o/u/n); the Turkmen
name is kept as `name_cn` and shows in the tooltip.

- Ashgabat: 4 etraps (Bagtyyarlyk, Berkararlyk, Buzmeyin, Kopetdag).
- Ahal: 8 etraps (Altyn asyr is new since the census). Arkadag is drawn as its
  own top-level unit and is not in Ahal's outline.
- Balkan: 6 etraps and 2 cities (Balkanabat, Turkmenbashy).
- Dashoguz: 7 etraps and Dashoguz city.
- Lebap: 11 etraps and Turkmenabat city (Dovletli, Farap and Garabekevul new).
- Mary: 10 etraps and 2 cities (Mary, Bayramaly; Oguzhan etrap new).

Nothing finer was attempted. The census prints every village and the gengeshlik
(rural council) it belongs to, but OSM has gengeshlik outlines for almost none
of them.

## Boundaries

Geofabrik extract `turkmenistan-261005.osm.pbf` (OSM data to 2026-10-05). OSM
tags the etraps and the cities of velayat subordination at admin_level 5
(49 relations including Arkadag; Afghanistan's Ghormach is in the extract and
dropped), the velayats and Ashgabat at 4, and Ashgabat's four etraps at 7. The
53 units tile the country exactly (no gap or overlap) and each velayat's units
match its level-4 outline to the km2. Top-level polygons are dissolved from the
units so the levels nest.

The current map is newer than the census. Five etraps were created after
December 2022, out of parts of older ones: Altyn asyr (from Tejen, Kaka and a
little of Sarahs), Dovletli (Koytendag and Hojambaz), Farap (Charjev), Garabekevul
(Sayat) and Oguzhan (Murgap and Sakarchage). Their source etraps lose about the
area each new one has, so this is the 2025 round of reforms as OSM has it, not
redrawing by mappers. Kontur's OSM boundaries of 2023-06-28 (HDX) still have the
census's 48 units exactly, Arkadag included, and are used as the census-time map.

Not used: geoBoundaries `TKM-ADM1` has five polygons (no Ashgabat) and no ADM2;
OCHA has no COD-AB for Turkmenistan (`cod-ab-tkm` 404s), as religiondots found.

## Population

### 2022: census

State Committee of Turkmenistan on Statistics, *Results of the Complete
Population and Housing Census of Turkmenistan 2022*, section 1 "Population
number and location", English, text layer:
https://www.stat.gov.tm/population-census-pdfs/results/en/1.pdf (index at
stat.gov.tm/en/population-census). Census day 17 December 2022; 7,057,841 people.
No licence is stated.

- Table 1.3: Ashgabat and each velayat, urban and rural.
- Table 1.4: Ashgabat's four etraps.
- Tables 1.5-1.9: urban population of each velayat by etrap or city of velayat
  subordination, then by city and town.
- Tables 1.10-1.14: rural population by etrap, then by the rural area of each
  town and each gengeshlik, then by village.

`census.py` reads every row (name, both sexes, male, female). The tables do not
mark nesting, so it is rebuilt from the sums: a row whose name could be a parent
(etrap, gengeshlik, town, city) is a parent when the rows after it add up to it
exactly. Every table then adds up to its velayat row in 1.3, male plus female
equals both sexes on every row, and the 48 units sum to 7,057,841. That gives 48
census units and 1,810 leaf settlements. Quirks handled: a few figures printed
without the thousands space (`1393`) or with two spaces; "Yoloten" has one
"geneshlik" misprint; rural Mary calls Turkmengala "Tukmengala"; rural Balkan's
first row is "including Balkanabat city" (Balkanabat's 2,183 rural people, which
belong to the city's 186,246 urban to make 188,429). Bereket's village rows were
checked against the rendered page because OSM's population tags disagree with
them; the PDF is right and OSM's tags are shifted by one row.

**Carrying the census onto today's units** (`fetch.py`). Each census settlement
is looked for among OSM place nodes (village, town, city, hamlet and so on)
inside its census etrap's 2023 outline plus 5 km, by name: Turkmen and English
spellings are reduced to one key (the letter mapping above, `ňň` = census `ng`,
generic words like obasy, town, station, gengeshlik dropped). A close spelling
(difflib 0.85) is accepted when there is no exact one. Among several hits, a node
whose OSM `population` tag equals the census figure wins, then one whose
`addr:city` names the same gengeshlik. The settlement is counted in whichever
current unit its node falls in. People never leave their census velayat, so
Ashgabat, Arkadag and every velayat stay exactly on table 1.3. Then:

- a settlement without a node goes where its gengeshlik's matched villages went;
- a gengeshlik with no matched village goes where OSM's villages tagged with that
  gengeshlik (`addr:city`) or its namesake village lie, if 80% agree;
- what is left is spread, by Kontur population, over the current units that took
  this census etrap's named villages.

People placed, of 7,057,841: by name 5,222,247; close spelling 37,318; by
gengeshlik 475,715; by OSM gengeshlik tag 34,026; whole unit (Ashgabat's etraps,
Arkadag) 1,030,630; spread by Kontur 257,905 (3.7%). The match is right where it
can be tested: 860 matched nodes carry an OSM population tag (mappers have
entered census figures), and 834 equal the census row; the 26 others are
gengeshlik totals put on the main village and one block of shifted tags.

Where the census units went (only those that split):

| census etrap | went to |
|---|---|
| Tejen | Tejen 152,699, Altyn asyr 29,262 |
| Kaka | Kaka 83,708, Altyn asyr 9,944 |
| Sarahs | Sarahs 80,643, Altyn asyr 363 |
| Koytendag | Koytendag 121,003, Dovletli 78,817 |
| Hojambaz | Hojambaz 82,809, Dovletli 27,567 |
| Charjev | Charjev 150,475, Farap 80,437, Danev 2,787 |
| Sayat | Sayat 127,246, Garabekevul 61,239 |
| Murgap | Murgap 172,594, Oguzhan 17,624 |
| Sakarchage | Sakarchage 174,071, Oguzhan 22,131 |
| Vekilbazar | Vekilbazar 157,969, Mary etrap 10,765 |
| Koneurgench | Koneurgench 172,262, Boldumsaz 4,522 |
| Akdepe, Shabat, S. Turkmenbashy, Turkmengala | 100-2,600 each across a border |

The new etraps are formed of whole gengeshliks (Farap from eight Charjev
gengeshliks, Garabekevul from ten of Sayat's, and so on), which is what a
reform does, so the settlement placement agrees with itself. The small flows
between old etraps (Egriguzer, 10,557, now in Mary etrap; Koneurgench's
Galkynysh gengeshlik now in Boldumsaz) are OSM placing those villages across a
border from where the census counted them; they were not chased.

### 2026: national growth, the same for every unit

There is no second count at etrap level, nor at velayat level:

- The 2012 census was never published. A leaked national total (4,751,120,
  Chronicles of Turkmenistan, 3 February 2015, read through the Wayback Machine)
  is the only figure, with no breakdown, and 4.75 to 7.06 million in ten years is
  not a trend anyone can use.
- The 1995 census has etrap figures in places, but on a map that has been
  redrawn several times since; not used.
- stat.gov.tm publishes nothing by velayat since the census (searched the site,
  its SDG portal sdg.stat.gov.tm and the web, in English and Russian, on
  2026-10-06). The site's front page still gives only the census total.

So 2026 is the census times UN World Population Prospects 2024, medium variant,
national total: 7,291,799 interpolated to census day and 7,736,632 at 1 July
2026, a factor of 1.0610. Every unit gets the same factor, so the estimate keeps
the census's proportions and only moves the level, to 7,488,404 nationally. WPP's
own level is 3.3% above the census on census day; only its growth is used. The
viewer's current-year estimate is the 2026 figure.

## Checks (`check.py`, 2026-10-06)

- Units sum to their top-level unit in both years; the top level sums to
  7,057,841 in 2022.
- 2022 top level equals table 1.3 for all seven (Ahal 886,278 is table 1.3's
  886,845 less Arkadag's 567).
- Cities that did not change equal their census rows exactly: Dashoguz 201,142,
  Turkmenabat 230,861, Mary 167,027, Bayramaly 70,376, Turkmenbashy 91,745.
- Every census etrap total is kept through the reallocation (asserted in
  `fetch.py`).
- Kontur population (400 m hexes, 2023-11-01, religiondots'
  `data/geo/tm/tm_hexes.gpkg`, read only), summed in each current unit. By
  velayat, as share of the nation: Ashgabat 1.89, Balkan 1.87, Lebap 0.97,
  Mary 0.86, Ahal 0.82, Dashoguz 0.38, the same pattern religiondots found on
  the same hexes. Within their own velayat, 20 of 52 units are within 10% and 34
  within 25%. The cities read low against their rural neighbours throughout
  (Mary city 0.57, Dashoguz 0.62, Turkmenabat 0.62, Bayramaly 0.52), and within
  Ashgabat Buzmeyin reads 0.32 and Berkararlyk 1.38. Kontur is a model scaled to
  UN totals; at this grain it says more about Kontur than about the census.
  Arkadag reads 10,485 in Kontur against the census's 567 (see below).

## Known weaknesses

- **2026 is one national rate.** A district's real change since 2022 (Ashgabat's
  in-migration, rural emigration) is not in it. The relative picture is the
  census's.
- **Arkadag** had 567 people at the census, when it had just opened; it has filled
  up since, and Kontur already showed 10,000 in late 2023. Its 2026 figure (602)
  is certainly far too low, and the people who moved there are still counted in
  Ashgabat or wherever they came from.
- **The census total itself is disputed.** Observers (RFE/RL, Chronicles of
  Turkmenistan) doubt 7.06 million, given emigration and the leaked 4.75 million
  of 2012. There is no open alternative at this grain; the census is drawn as
  printed, as religiondots and languagedots do.
- **3.7% of people are spread by Kontur**, in 96 settlements OSM has no node for.
  For etraps that did not split this changes nothing; for those that did, it can
  shift a few thousand people between the old etrap and the new one. The biggest
  cases: 14 Vekilbazar villages (72,908 people) shared between Vekilbazar and
  Mary etrap, and 13 Koneurgench settlements (29,367) between Koneurgench and
  Boldumsaz.
- Unit outlines are OSM's; the five new etraps are a 2024-25 addition by mappers
  and their exact lines are not checked against a legal text.

## Dead ends

- `en.hronikatm.com` fails TLS from here; its 2015 article was read from the
  Wayback Machine. `chrono-tm.org` (the Russian original) is now a football
  streaming site; the Wayback CDX query for it hit an archive outage on 2026-10-06
  and was not retried, since the English article has the same figures.
- Overpass was not needed; the Geofabrik extract has everything.
