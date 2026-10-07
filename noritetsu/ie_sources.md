# Ireland register sources (built 2026-10-03)

What the Irish build reads, where each piece came from, and what is still wrong with it.
Ireland is built with `rinf.py`; how the reader works is in its docstring, and the per-country
entry is `rinf_countries/ie.py` (its docstring explains the grouping and the repairs).
Downloads live in `data/raw/rinf/ie/`, `data/raw/ie/` and `data/raw/gtfs/ie/` (gitignored).
Nothing needed a login or a key. Northern Ireland is built in `gb`.

## Run

```powershell
python rinf.py --fetch ie                      # RINF and Wikidata, a few seconds
python extract.py --region ie --pbf data/raw/ireland-and-northern-ireland-latest.osm.pbf
python -m rinf_countries.ie --clip             # after every extract: Northern Ireland out
# the timetable: FEEDS["ie"] in gtfs_served.py (sent as a diff), then
python gtfs_served.py --fetch ie               # NTA's Irish Rail GTFS, 7.7 MB, rail only
python tools/slot.py -- python build_model.py --region ie --register rinf:data/raw/rinf/ie
python tools/slot.py -- python build_tiles.py --region ie
python check_model.py --region ie
python tools/slot.py -- python rinf.py --dry ie   # the reader alone, with its full log
```

The first build's feed was fetched with curl from the URL below (FEEDS["ie"] not yet landed);
the file is the same. The model takes about 40 s, the tiles 5 s.

## Sources

- **ERA RINF**, SPARQL at `https://graph.data.era.europa.eu/repositories/rinf-plus`, fetched
  2026-10-03: 274 sections (1,663 km), 267 points, all with coordinates. One id per section
  ("SOL1001" ... "SOL1274"), no line numbers, and every section its own infrastructure-manager
  code (1001_IM ... 1274_IM), all of them Iarnród Éireann. Licence: ERA's (EUPL 1.2 for the
  service; the dump is CC BY 4.0).
- **en.wikipedia**, raw wikitext through the API, retrieved 2026-10-03: the line names (the
  article titles: "Dublin–Cork railway line", "Limerick–Rosslare railway line", "Western
  Railway Corridor", "Ballina branch line", "Dublin–Navan railway line"...) and the infobox
  lengths, which are Iarnród Éireann's 2022 Network Statement's figures (the check). CC BY-SA
  4.0; names and lengths are facts.
- **Iarnród Éireann's Network Statement 2022** (`data/raw/ie/IE-2022-Network-Statement.pdf`,
  5.7 MB, from the Wayback Machine: the live link at irishrail.ie now answers with an HTML
  page). Its mileage and speed tables (appendix 4) are images, so the figures used are the
  ones en.wikipedia quotes from it, not read from the PDF.
- **OSM's boundary of the Republic** (relation 62273), `data/raw/ie/ie_boundary.geojson` from
  `polygons.openstreetmap.fr/get_geojson.py?id=62273&params=0`, for the clip.
- **NTA's Irish Rail GTFS**, `https://www.transportforireland.ie/transitData/Data/GTFS_Irish_Rail.zip`
  (7.7 MB, rail only, updated daily; 19 routes, 3,333 trips, calendar 1 October - 12 December
  2026, feed_info to 2 October 2027; shapes included). Licence CC BY 4.0 (TFI's open-data page,
  `transportforireland.ie/transitData/PT_Data.html`). It has the Enterprise with its Northern
  Ireland stops; it does not have the Luas (TFI's separate Luas feed), which is right: the Luas
  is OSM lines, not register lines.
- **OpenStreetMap**, Geofabrik `ireland-and-northern-ireland-latest.osm.pbf` (2026-10-03),
  ODbL: after the clip 5,710 track ways, 62 route relations (55 train, 6 tram, 1 monorail),
  95% of main-line ways (98% of main-line km) under a passenger route.

## Choosing the register

Measured both candidates:

- **OSM named track** (`probe_kr_ways.py --region ie`, before the clip): 85.6% of main and
  branch rail km carries a `name`, but the names are not a line register. The biggest is
  "IÉ Dublin – Cork" (465 track-km), then "Great Northern Railway Main Line" (183, both sides of
  the border), "Sligo Line", "Mayo Line", "Waterford and Limerick Railway", "Midland Great
  Western Railway Main Line" and "Dublin-Mullingar" (two names for parts of one route),
  "South Kerry Line", "Western Railway Corridor", "Athlone Branch"; 20 km each of "Up Line",
  "Down Line", "Down Fast", "Up Slow"; and nothing at all for most of Dublin - Rosslare. A
  Korea-style build would need a rename table for most of the network and still leave the
  Rosslare line unnamed.
- **RINF grouped into named lines** (chosen): complete (every IÉ line, freight branches
  included), stations in order, and the coordinates are good (median a few tens of metres
  from OSM's stations). Its one id per section carries no line number, so the grouping is a
  table of section numbers (`LINE_OF`); they run along each route in blocks, so the table is
  short.

## The lines

Seventeen register lines, named by en.wikipedia's article titles ("railway line" shortened to
"line"), Iarnród Éireann's own route names:

| line | from - to | built km |
|---|---|---|
| Dublin–Cork line | Heuston - Cork Kent | 265.4 |
| Dublin–Sligo line | Connolly Junction (by Newcomen Junction and Glasnevin) - Sligo, with the Docklands spur | 217.9 |
| Dublin–Rosslare line | Connolly - Rosslare Europort (the DART south of Connolly) | 167.3 |
| Dublin–Galway line | Portarlington - Athlone - Galway | 140.7 |
| Dublin–Westport line | Athlone - Westport | 132.9 |
| Dublin–Waterford line | Cherryville Junction (Kildare) - Waterford, with Kilkenny and the Lavistown triangle | 123.1 |
| Limerick–Rosslare line | Limerick - Limerick Junction - Waterford (Dunkitt Junction) | 121.6 |
| Mallow–Tralee line | Mallow - Killarney - Tralee | 99.0 |
| Western Railway Corridor | Limerick (Foynes Junction) - Ennis - Athenry | 95.7 |
| Great Northern Railway Main Line | Connolly - Dundalk - the border | 95.1 |
| Limerick–Ballybrophy line | Killonan Junction - Nenagh - Ballybrophy | 84.6 |
| Ballina branch | Manulla Junction - Ballina | 33.1 |
| Cork–Cobh line | Cork Kent - Glounthaune - Cobh | 18.4 |
| Glounthaune–Midleton line | Glounthaune (Cobh Junction) - Midleton | 10.2 |
| Phoenix Park Tunnel line | Islandbridge Junction - Glasnevin - Drumcondra - Ossory Road Junction | 7.4 |
| Dublin–Navan line | Clonsilla - M3 Parkway | 7.2 |
| Howth branch | Howth Junction and Donaghmede - Howth | 5.4 |

Decisions, mine:

- **The Belfast line is "Great Northern Railway Main Line"**, the name gb's register gives its
  half (OSM's name on both sides of the border), so the two halves read as one line. en.wikipedia's
  title is "Belfast–Dublin line"; Iarnród Éireann calls it the Belfast line and its commuter
  service the Northern Commuter.
- **Galway and Westport are two lines** from Athlone, as the article's own infobox splits them
  ("Portarlington–Athlone 63 km, Athlone–Galway 78.46, Athlone–Westport 133.374"). Athlone -
  Athlone West Junction (1 km, both lines' trains) is the Galway line's.
- **Dublin's connecting lines.** Connolly - Connolly Junction - Ossory Road - East Wall -
  Howth Junction is the Belfast line's. The Sligo line is the Midland Great Western's alignment
  (Connolly Junction - Newcomen Junction - Glasnevin) with the Docklands spur. The Great
  Southern and Western's line from Islandbridge Junction through the Phoenix Park tunnel,
  Glasnevin and Drumcondra to Ossory Road is the "Phoenix Park Tunnel line" (our name, after
  the tunnel and Iarnród Éireann's Phoenix Park Tunnel service): Maynooth trains from Connolly
  run over its Drumcondra stretch too, and the track goes to one owner.
- **Dublin–Navan line** is only Clonsilla - M3 Parkway, the part built of the line the article
  describes (Docklands - M3 Parkway, the rest shared with the Sligo line).
- **Cork–Cobh line and Glounthaune–Midleton line** are our names (Cork Suburban Rail has no
  article per line); the Midleton branch runs from Cobh Junction at Glounthaune.
- **Junctions shared by two lines**: Ballybrophy Station - Ballybrophy Junction (0.3 km) is the
  Limerick line's; Dunkitt Junction - Waterford is the Dublin–Waterford line's, so the Limerick
  - Waterford line ends at Dunkitt; Limerick Colbert - Foynes Junction is the Limerick–Rosslare
  line's, so the Western Railway Corridor starts at Foynes Junction.
- **Left out** (freight only, no passenger station): Drogheda (Navan Branch Junction) - Tara
  Mines (26.6 km), Limerick - Foynes (13.5 km, reopening for freight), Waterford - Belview
  (5.5 km), the Dublin Port and East Wall yard links, and the 0.4 km stub of the Waterford -
  Rosslare line (closed 2010). The Navan line and Foynes come back as "drop nothing, check
  the timetable" the day passenger trains run.

## RINF's Irish data, and what `fix` does with it

- **Every point is typed 30 (passenger terminal)**, the TD (train describer) boundaries and
  junctions too. 127 points are retyped junction: "TD nn", every name ending in "Junction",
  "Border", "Malahide Viaduct", and the freight terminals. That leaves 140 stops, every one an
  OSM station (139 distinct: Park West and Cherry Orchard is one station with two points).
- **Names** OSM does not share get a second name to match by (24: "Connolly Station" is
  "Dublin Connolly", "Drohgeda McBride" is "Drogheda MacBride", "Kilkenny Station" is
  "Kilkenny MacDonagh"...). Three junction names are put right ("Foyens", "Ossary Road",
  "Dunkitt Junction (No turnout here)").
- **Limerick Junction has no station point**, only five junctions round it; its "Station
  Junction" at the platforms' south end (121 m from OSM's station) is made the stop. Without
  it the timetable's Limerick Junction matched nothing and the Limerick Junction curves came
  out "ambiguous".
- **Points to merge**: Clonsilla, Manulla Junction and Athenry each have a station and a
  junction point at one coordinate; Ennis Junction is placed 7 km west of Limerick, 1.5 km
  from any track, and is 0.45 km from Foynes Junction in RINF; Athenry Junction (where the
  Western Railway Corridor meets the Galway line, 480 m west of the station) met nothing.
  Each is merged into its neighbour (`MERGE`), so the Corridor now joins the Galway line.
- **Park West and Cherry Orchard**: RINF runs Islandbridge Junction - Park West (5.9 km) - TD
  83 - TD 82 (1.5 km back towards Dublin) - Cherry Orchard (47 m from Park West). Rewired as
  Islandbridge Junction - TD 83 - TD 82 - Park West and Cherry Orchard (`REEND`).
- **Lengths.** Most sections are right to a few hundred metres; these are not (RINF km
  against the trace on OSM's track, which the end-to-end retrace confirms and which matches
  the published line lengths below):

  | section | RINF | traced |
  |---|---|---|
  | Mallow - Killarney Junction | 59.14 | 1.13 |
  | Killarney - Tralee Junction | 27.10 | 0.36 |
  | Islandbridge Junction - Park West and Cherry Orchard | 7.89 | 4.74 |
  | Waterford - Dunkitt Junction | 7.08 | 2.75 |
  | Wexford - Enniscorthy | 19.20 | 24.32 |
  | Rathdrum - Arklow | 15.88 | 18.76 |
  | Kilcock - Enfield | 10.39 | 12.53 |
  | Farranfore - Tralee | 14.83 | 17.44 |
  | Kilkenny - Lavistown West Junction | 0.39 | 3.66 |
  | Killonan Junction - Foynes Junction | 2.98 | 5.95 |
  | Glasnevin Junction North - Islandbridge Junction | 2.92 | 4.37 |
  | Lavistown South - North Junction | 2.07 | 0.54 |

  and six more under a kilometre either way (the build log's "length off" lines). Three RINF
  lengths were too short for the trace to reach the far end at all, and `LENGTH` replaces
  them: Rosslare Strand - Rosslare Europort 0.358 km (4.2 km as the crow flies; 5.23, the
  Network Statement's 3 1/4 miles), TD 66 - TD 67 between Rathmore and Killarney 0.81 km (17.2
  as the crow flies; RINF's 27.1 km for Killarney - Tralee Junction is this stretch's,
  misfiled; 19.0 as a bound), and Ennis Junction - Sixmilebridge 14.9 km (19.5 as a bound; the
  line swings west by Cratloe). Because of all this the build ships **no km_official** (a new
  rinf.py hook, `no_chain`, no-op unless set): RINF's lengths still guide the traces, and the
  published lengths below are the check.
- **The border**: RINF's "Border" point (IE+OP42) is 20 m off the track; `fix` moves it to
  where OSM's boundary of the Republic crosses the two tracks (ways 31695909 and 395764463 end
  at -6.379110, 54.069069 and -6.379074, 54.069092; their midpoint -6.379092, 54.069080).
  uopids lose their "+" ("IE+OP42" -> "IEOP42"), so the border's id is `eIEOP42`.

## The check (`check_model.py --region ie`, 2026-10-03)

Fourteen of the seventeen lines have a published length (en.wikipedia, from the Network
Statement): every one within 3%. Dublin–Cork 265.4 / 266.75 (0.99), Dublin–Sligo 217.9 /
216.05 (1.01, with the Docklands spur), Dublin–Rosslare 167.3 / 167.97 (1.00), Dublin–Galway
140.7 / 141.46, Dublin–Westport 132.9 / 133.37, Dublin–Waterford 123.1 / 122.8, Limerick–
Rosslare 121.6 / 123.1 (to Dunkitt, expect 0.98), Limerick–Ballybrophy 84.6 / 84.49,
Mallow–Tralee 99.0 / 98.97, Western Railway Corridor 95.7 / 97.16, Ballina branch 33.1 /
33.19, Great Northern Railway Main Line 95.1 / 95.76 (milepost 59 1/2 at the border),
Glounthaune–Midleton 10.2 / 10.0, Dublin–Navan 7.2 / 7.5. No figure found for Cork–Cobh, the
Howth branch or the Phoenix Park Tunnel line.

## What is in the build

48 lines (17 register, 1,626 km; 31 OSM: the InterCity routes, the Northern, Western and South
Western Commuter, the DART, the Cork commuter routes, the Enterprise, Luas Red and Green),
237 stations, 4,412 route-km. 147 register stops, which is every Irish station Iarnród
Éireann's timetable calls at (147 of the feed's 152 stops; the other five are the Enterprise's
Northern Ireland stops). `osm_stops` added the stations RINF lacks: Clontarf Road, Killester,
Harmonstown, Raheny and Kilbarrack (all between East Wall and Howth Junction, one 7 km RINF
section), Kishoge, Hansfield and Woodbrook.

**The timetable check** (the feed in `data/raw/gtfs/ie/`): 147 of 152 feed stations matched
(the five unmatched are in Northern Ireland); 1,617 of 1,626 register km served; nothing
closed, nothing rescued. Dropped as junction-ended with no OSM route and only weak timetable
evidence: Lavistown South - North Junction (0.5 km, the freight avoiding curve at Kilkenny)
and Limerick Junction North East - North West Junction (0.7 km). Dundalk - the border (8.5 km)
reads "osm" until the border point is in borders.EXTRA, then "served" (the feed calls at
Newry beyond it).

## Northern Ireland

Geofabrik has one extract for the island. `python -m rinf_countries.ie --clip` keeps what
lies in OSM's boundary of the Republic (a way if any node is inside, a stop if inside, a
route if a member stayed): 5,710 of 6,756 ways, 2,197 of 2,553 stops, 80 of 86 relations.
The only ways over the border are the Belfast line's two tracks, which OSM ends at the
boundary itself. Nothing of Northern Ireland is built twice: gb has it.

## Borders

- **Dundalk - Newry**: the only crossing. gb proposed (-6.374718, 54.064810) from the coarse
  outlines; that is 552 m south of where the track really crosses (OSM's boundary, 20 m from
  RINF's own point) and 143 m off the Irish track. The entry for `borders.EXTRA` is
  `("eIEOP42", -6.379092, 54.069080, ["gb", "ie"])`: the "e" + uopid form, because Ireland's
  register has its own point there (borders.py's rule), so the register line ends at the
  shared id with nothing else to do. gb_register picks it up (any point naming gb within 600 m
  of a line's track). Trialled on ie with the entry patched in: the Enterprise gets its 8.4 km
  Dundalk - border tail, Dundalk - border becomes "served" by the timetable, the point is
  named "Ireland – United Kingdom border". Trialled on gb the same way (trial build against
  dist/data/gb): lines, km and sections identical; the one change is that gb's Great Northern
  Railway Main Line, which already ran 15.46 km from Newry to a "Junction near Newry"
  (gj1448879420) at the very same track node, now ends at eIEOP42 (2 m away). The old junction
  id is not carried as an alias (build_model never aliases onto a junction).
- gb oddity seen on the way (gb's, not changed here): OSM tags the Irish side's track ref "B"
  and gb_register's ELR fill names those ways "Bangor Line", so gb logs the border point "on
  Bangor Line" too; it builds nothing from it. The same fill gives gb's "Bangor Line" a stray
  1.2 km section at Folkestone.

## Named trains (rules/ie.py)

None in the extract. Every route is a line, the Enterprise included (about hourly; gb's rule
keeps it a line too). Written for the day they are mapped: Belmond's Grand Hibernian, the
RPSI's steam specials, anything tagged service=night or car.

## Still off, and why

- **No colours/ie.csv**: the DART and the Luas lines carry OSM's colours; the register lines
  have none (Iarnród Éireann publishes none for its InterCity routes).
- **No English/Irish names**: line names are English; station names are OSM's (English).
- **Limerick Junction**: the station stands at RINF's "Station Junction", the south end of the
  platforms; the Limerick–Rosslare line passes it on the North West - North Junction direct
  curve and has no section through the station itself.
- **The Lartigue Monorail** (Listowel, heritage, route=monorail with no stops) builds no line;
  its 9 ways are left out of the tiles.
