# Finland register sources (built 2026-10-01)

What the Finnish build reads, where each piece came from, and what is still wrong with it.
Finland is built with `rinf.py`; how the reader works is in its docstring, and the per-country
entry is `rinf_countries/fi.py`. Downloads live in `data/raw/rinf/fi/` (gitignored). Nothing
needed a login or a key.

## Run

```powershell
python rinf.py --fetch fi                                            # RINF, a few seconds
curl -A "noritetsu-rail-map/1.0" -o data/raw/rinf/fi/digitraffic_stations.json https://rata.digitraffic.fi/api/v1/metadata/stations
curl --compressed -A "noritetsu-rail-map/1.0" -o data/raw/rinf/fi/digitraffic_trains_2026-09-30.json https://rata.digitraffic.fi/api/v1/trains/2026-09-30   # the check only
curl -L -o data/raw/fi-finland-260930.osm.pbf https://download.geofabrik.de/europe/finland-260930.osm.pbf
$env:OSMIUM_POOL_THREADS=1; python extract.py --region fi --pbf data/raw/fi-finland-260930.osm.pbf   # 70 s; delete the .pbf after
python inspect_region.py --region fi
python build_model.py --region fi --register rinf:data/raw/rinf/fi   # 20 s
python build_tiles.py --region fi                                    # 15 s
python check_model.py --region fi
python rinf.py --dry fi          # the reader alone, with its full log
```

`finland-latest.osm.pbf` was not tried; the dated file from
https://download.geofabrik.de/europe/finland.html worked. `fi.py` has `"wikidata": None`, so
`--fetch` pulls RINF only (Wikidata's eight Finnish route-number rows are the Helsinki metro, a
light-rail line and one stray "14"; nothing for the main lines).

## Sources

- **ERA RINF**, SPARQL at `https://graph.data.era.europa.eu/repositories/rinf-plus`, fetched
  2026-10-01 (`sections.json`: 858 rows, 835 sections after versions; `points.json`: 797
  rows, 796 points, all with coordinates). 113 line ids, 5,534 km. Infrastructure managers:
  Väylävirasto (`3109_IM`, everything) and Raideinfra Oy (`5590_IM`, 46 depot-lead sections).
- **Väylävirasto's infra API** (`https://rata.digitraffic.fi/infra-api/`, open):
  `vayla_radat.json` and `vayla_tilirataosat.json`, read to see whether Väylä names its lines.
  It does not: track numbers carry no name, and the accounting sections are "(Lahti) -
  (Kouvola)". They confirmed that RINF's ids are Väylä's track numbers (ratanumero).
- **Fintraffic's station list** (`digitraffic_stations.json`, from
  `https://rata.digitraffic.fi/api/v1/metadata/stations`, open, no key): 552 Finnish rows, 209
  with `passengerTraffic`. RINF's uopid "FI00408" is its UIC code 408, an exact join.
- **Fintraffic's timetable for one day** (`digitraffic_trains_2026-09-30.json`, from
  `https://rata.digitraffic.fi/api/v1/trains/2026-09-30`): 1,211 passenger trains with every
  station they pass. Used to check which register sections a passenger train runs over (the
  scratch script is not kept; the result is the FREIGHT set in `fi.py`).
- **OpenStreetMap**, Geofabrik `finland-260930.osm.pbf`, ODbL: track, stations, 95 route
  relations (56 train, 26 tram, 6 light rail, 4 subway, 2 funicular), 47 `route=railway`
  relations. Passenger route coverage is thin: only 46% of track km lies under any route
  relation, and there is none over the Karjalan rata or Orivesi - Jyväskylä (found when
  build_model dropped their junction-ended sections, below).
- **fi.wikipedia**, raw wikitext retrieved 2026-10-01: the list of line articles in
  "Suomen rautatieliikenne" ("Artikkeleita Suomen rataverkon osuuksista"), each article's
  extent and infobox `pituus`, and that article's table of VR's distances from Helsinki (VR
  Matkahaku, cited 7.7.2025): Tampere 187, Haapamäki 301, Seinäjoki 346, Vaasa 420, Kouvola
  166, Joensuu 482 km.

## How `fi.py` reads the ids

RINF's ids are Väylä's track numbers: "001" is Pasila - Turku, "003" Helsinki - Tampere -
Seinäjoki, "006" Riihimäki - Lahti - Kouvola - Imatra - Joensuu - Nurmes - Kontiomäki (702 km),
"008" Seinäjoki - Oulu - Rovaniemi - Kemijärvi. There are also yard and depot ids ("KV225",
"PM805", "003808TRE"). A track number is the unit of Väylä's kilometre system, not a line
anyone rides by, and there is no public line number. Finnish lines are known by name, the names
of fi.wikipedia's line articles, and those cut across the track numbers. So `fi.py` files each
RINF section under its named line in rinf.py's `fix` hook, and `id_name` gives that name back:

- `WHOLE`: a track number that is one named line ("321" Turku–Toijala-rata), or a piece folded
  into one: 337 Turku asema - Turku satama into the Rantarata, OL202 (Oulu_V330 - Oulu) into
  the Oulu–Kontiomäki-rata, SK804 (Seinäjoki - Seinäjoki_V603) into the Haapamäki–Seinäjoki-rata.
  The articles' lengths include those pieces.
- `CUT`: a track number holding several named lines in a row, cut at named stations by RINF's
  own chainage. 003 is Päärata to Tampere asema, then the Tampere–Seinäjoki-rata; 006 is the
  Riihimäki–Lahti-rata, the Lahti–Kouvola-rata, the Karjalan rata, the Joensuu–Kontiomäki-rata,
  cut again at Nurmes; 008 is the Pohjanmaan rata, the Oulu–Tornio-rata from Oulu and the
  Laurila–Kelloselkä-rata from Laurila; 521 is the Oulu–Tornio-rata's Laurila - Tornio end and
  then the Kolarin rata; and so on for 005, 009, 023, 066, 731.
- Siding leads: a section with an end at a private siding's boundary (RINF op-type 140) or a
  depot (50). They are set aside. Left in, each siding point was a branch point of its line,
  the line was cut there into junction-ended sections, and build_model dropped every one no OSM
  route runs over. The Karjalan rata kept 59 of 315 km; the Orivesi–Jyväskylä-rata lost
  Jyväskylä - Jämsänkoski (53 km) at the UPM mill.
- `skip_line` leaves out everything that is not a named line (yards, depot tracks, industrial
  branches), plus `FREIGHT`, named lines or stretches with no passenger train that would
  survive because both ends are stations. That set is Siilinjärvi–Viinijärvi-rata,
  Jyväskylä–Haapajärvi-rata, Nurmes - Kontiomäki, and Olli - Sköldvik (the branch to Neste's
  refinery, cut off track number 131 in `CUT`). The Fintraffic timetable for 2026-09-30 has no
  passenger train on any of them.
- The **Porvoon rata** (Kerava - Porvoo, museum trains on summer days only) was in `FREIGHT`
  until 2026-10-01, when Anita decided to keep it. With the timetable check
  (`gtfs_served.py`, `data/raw/gtfs/fi/`) it comes back whole, Kytömaa - Nikkilä - Porvoo
  (34 km), drawn as not running, since Fintraffic's feed carries VR and HSL only and no
  museum train. Without the feed only Porvoo - Hinthaara (10 km) would return.

Päärata's own article runs Helsinki - Oulu, but its middle and north have their own articles
(Tampere–Seinäjoki-rata, Pohjanmaan rata). So the Päärata here is Helsinki - Tampere, the
stretch the article calls its most important. Luumäki - Vainikkala has no article and takes
Väylä's section name.

OSM's `route=railway` relations are ignored (`osm_rel` returns None). Some carry a track number
as `ref` (Uudenkaupungin rata 332, Rauman rata 342) or an unrelated one (Savon rata "231"),
which would have shown as line numbers, and every name they give is in `fi.py` already.

**Stations** come from Fintraffic's list through `stop_name` (a rinf.py hook added for
Finland on 2026-10-01). A point Fintraffic flags as a passenger station is a stop under VR's
Finnish name, without the "asema" suffix ("station"); one it flags as not passenger is never
a stop. That fixes RINF's typing:
- Henna, Hillosensalmi, Kempele, Nikkilä and Purola are typed junction and Härmä freight,
  though trains call at all six.
- Ilola is typed a stop, though no train calls.
- Isokyrö, Laihia, Ylistaro, Lievestuore and Purola have no OSM station and are placed at
  RINF's coordinate.

Names are Finnish where OSM has the Swedish one (Tammisaari for Ekenäs, Karjaa for Karis,
Pännäinen for Jakobstad-Pedersöre), and "Kotkan satama" is no longer matched onto OSM's
"Kotka" by name prefix. `name_m` is 1500 because RINF places Uusikylä 1.2 km from the station.
Result: 209 stops, one for each of Fintraffic's 209 passenger stations.

## What is in the register and what stays OSM

Register: 29 named lines, 3,992 km, all Väylävirasto's. Left out as unridden by build_model or
`skip_line`, which is right: Hyvinkää–Karjaa, Lahti–Loviisa, Lahti–Heinola, Uudenkaupungin
rata, Naantalin rata, Rauman rata, Valkeakosken rata, Suupohjan rata, Ilomantsin rata,
Vartiuksen rata, Niiralan rata, Luumäki - Vainikkala (the Allegro to St Petersburg is
suspended), Haminan rata, Raahen rata, Otanmäen rata, Talvivaaran rata, Ämmänsaaren rata,
Pietarsaaren rata, Mäntän rata, Kaipolan rata, Vuosaaren satamarata, Tornio–Haaparanta-rata,
Kemijärvi - Patokangas, Pori - Mäntyluoto, Vaasa - Vaskiluoto, Huutokoski - Rantasalmi.

Stay OSM lines: Helsinki metro M1 and M2, the Raide-Jokeri light rail (15), Helsinki's trams
(1-10, 1T, 8T, 13; OSM lists their stops only as platforms, which build_model's
`stop_members` falls back to), Tampere's tram (Raitiolinja 1 and 3), the Kakola funicular in Turku,
and VR's services: commuter letters (A, E, I, K, L, P, R, T, U, Y, Z, D, G, H, M, O) and the
long-distance patterns OSM maps as "Juna 3", "Juna 13: Helsinki => Oulu" and so on. HSL's
letters are services, not lines, so there is no `colours/fi.csv`; Väylä publishes no line
colours. Named trains (build_model's `fi` branch of `looks_like_service`, and its fi-only rule
that a route_master whose every route is a named train is one): "Juna 7" (OSM's route_master
for the night trains PYO 273 and PYO 276) and "Taajamajuna 751" (Parikkala - Savonlinna, mapped
by train number).

## Counts (2026-10-01)

- 29 register lines, 3,992 km.
- 76 lines in all (29 register, 47 OSM, 2 of those named trains); 448 stations; 8,778 route-km.
- 1.8 MB tile archive.

## Check

`python check_model.py --region fi`: against RINF's own section lengths, 29 lines, median
0.995, none off by more than 5%. Against published figures, 25 lines, 24 within 2%:

| line | built | published | ratio |
|---|---|---|---|
| Päärata | 186.5 | 187.0 (VR) | 1.00 |
| Rantarata | 191.9 | 195.8 | 0.98 |
| Savon rata | 351.2 | 357.8 | 0.98 |
| Karjalan rata | 313.8 | 316.0 (VR) | 0.99 |
| Pohjanmaan rata | 332.8 | 334.8 | 0.99 |
| Tampere–Seinäjoki-rata | 158.6 | 160.0 | 0.99 |
| Riihimäki–Lahti-rata | 58.4 | 59.0 | 0.99 |
| Lahti–Kouvola-rata | 61.0 | 61.4 | 0.99 |
| Lahden oikorata | 62.0 | 74.0 | 0.84 |
| Oulu–Tornio-rata | 129.1 | 131.9 | 0.98 |
| Kolarin rata | 181.0 | 182.0 | 0.99 |
| Iisalmi–Kontiomäki-rata | 106.9 | 108.4 | 0.99 |
| Oulu–Kontiomäki-rata | 165.1 | 166.0 | 0.99 |
| Iisalmi–Ylivieska-rata | 153.6 | 154.4 | 0.99 |
| Turku–Toijala-rata | 128.6 | 131.0 | 0.98 |
| Tampere–Haapamäki-rata | 112.4 | 114.0 (VR) | 0.99 |
| Haapamäki–Seinäjoki-rata | 117.3 | 117.8 | 1.00 |
| Orivesi–Jyväskylä-rata | 112.2 | 112.7 | 1.00 |
| Jyväskylä–Pieksämäki-rata | 79.2 | 79.8 | 0.99 |
| Haapamäki–Jyväskylä-rata | 77.1 | 77.2 | 1.00 |
| Pieksämäki–Joensuu-rata | 180.0 | 181.7 | 0.99 |
| Vaasan rata | 74.2 | 74.0 (VR) | 1.00 |
| Kotkan rata | 51.2 | 52.0 | 0.98 |
| Hangon rata | 49.1 | 49.3 | 1.00 |
| Kehärata | 26.6 | 27.0 | 0.99 |

"(VR)" is a difference of two VR distances from Helsinki, used where the article's figure
disagrees with RINF:
- Karjalan rata: the article says 325.8; RINF has 314.7, VR 316.
- Tampere–Haapamäki-rata: the article says 106.5; RINF 112.5.
- Vaasan rata: the article's 78 runs on to the Vaskiluoto port, which no passenger train
  serves.
- Päärata: the article gives no Helsinki - Tampere length.

- **Lahden oikorata** (0.84): RINF's track number 007 runs Kytömaa - Hakosilta (63.5 km), the
  two junctions where the line leaves the Päärata north of Kerava and joins the Riihimäki - Lahti
  track south of Lahti. The article's 74 km counts Kerava - Lahti, and those ends are built as
  part of the Päärata and the Riihimäki–Lahti-rata.
- **Rantarata** (0.98): Helsinki - Pasila (3.2 km) is built on the Päärata, where RINF files it.

Not in `REGISTER`, because the built line is a different extent from the article's:
- Joensuu–Kontiomäki-rata: the article gives 269 km to Kontiomäki; the build stops at Nurmes,
  where passenger trains end, at 159 km.
- Laurila–Kelloselkä-rata: the article's 269.3 km runs to Kelloselkä; trains end at Kemijärvi
  (built 189.5).
- Tampere–Pori-rata: the article gives 155, the route Tampere - Mäntyluoto. Built is Lielahti
  - Pori (126.1); Tampere - Lielahti is Tampere–Seinäjoki-rata track.
- Huutokoski–Parikkala-rata: built is Savonlinna - Parikkala (57.2), the part with trains.

## Still off, and why

- **Seasonal and summer services** are judged on one Wednesday's timetable plus OSM's routes.
  The Kolari night train is kept (OSM maps it). The Porvoo museum line is drawn as not running
  (above).
- **Timetable check** (gtfs_sources.md): Fintraffic's feed (24 Sep 2026 to end of 2027)
  rescues Tornio - Haparanda border track (3.5 km) and closes nothing but the Porvoon rata.
- **Kotkan satama** is placed at RINF's coordinate. OSM's "Kotka satama" station 320 m away is
  a separate station, so the O trains' OSM line ends at a stop that is not on the register
  line.
- **Lines have no English names.** Wikidata has no line items with route numbers to take them
  from, and OSM's relations carry none.
- Two named lines come in unconnected pieces in RINF (logged by `fix`); both are freight lines
  that build_model drops.
