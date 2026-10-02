# Germany register sources (built 2026-10-02)

What the German build reads, where each piece came from, and what is still wrong with it.
Germany is built with `rinf.py`; the per-country settings are `rinf_countries/de.py` (its
docstring explains names, versions, stations and the S-Bahn). Downloads live in
`data/raw/rinf/de/` and `data/raw/de/` (gitignored). Nothing needed a login or a key.

## Run

```powershell
python rinf.py --fetch de                                    # RINF, ~1 min (23,708 section rows)
curl -L -o data/raw/de/germany-latest.osm.pbf https://download.geofabrik.de/europe/germany-latest.osm.pbf   # 4.85 GB, ~35 min
$env:OSMIUM_POOL_THREADS=2; python extract.py --region de --pbf data/raw/de/germany-latest.osm.pbf   # 9 min; delete the .pbf after
python inspect_region.py --region de
python build_model.py --region de --register rinf:data/raw/rinf/de     # 5-6 min
python build_tiles.py --region de                                      # 3 min, de.pmtiles 18.5 MB
python check_model.py --region de
python rinf.py --dry de          # the reader alone, with its full log (~2.5 min)
```

`rinf.py --fetch de` writes no wikidata.json (`"wikidata": None`). The route-number pull used
for the check figures was made once with the Wikidata query in rinf.py (P17 Germany, P1671)
and moved to `data/raw/de/wikidata_lines.json`, where rinf.py does not read it.

The DB names file: download
`https://mobilithek.info/mdp-api/files/aux/922109165921083392/Infrastrukturdaten.zip` (28 MB;
the GovData listing gives the URL with a double slash, which returns the Mobilithek web page
instead) and unzip `M1 Streckennetz.csv` into `data/raw/de/db/`. `M1 Betriebsstellen.csv` is
there too, unused.

## Sources

- **ERA RINF**, SPARQL at `https://graph.data.era.europa.eu/repositories/rinf-plus`, fetched
  2026-10-02: 23,708 section rows (11,848 in the 2026 version), 19,213 point rows, 1,496 line
  ids, all four digits. European Union Agency for Railways, under the EU's reuse terms.
- **OpenStreetMap**, Geofabrik `germany-latest.osm.pbf` (fetched 2026-10-02), ODbL: 275,057
  track ways, 315,856 stops, 4,830 route relations (1,326 route_master), 2,925
  route=railway/route=tracks relations, 2,075 of them with a four-digit VzG `ref`. Deleted
  after the extract; `data/proc/de/` holds what was pulled out.
- **DB InfraGO, "Infrastrukturdaten der DB InfraGO"** (Mobilithek, listed on GovData,
  https://www.govdata.de/suche/daten/infrastrukturdaten-der-db-infrago, data of 2026-04-29):
  `M1 Streckennetz.csv` gives every VzG line's Streckenkurzname and per-segment length (1,549
  lines, 33,624 km). Used for line names and for two check figures. multi_sources.md records the
  dataset as CC BY 4.0; the GovData page showed no licence when fetched, so confirm before
  shipping the names anywhere public beyond attribution.
- **Wikidata** (CC0): 2,739 items with a route number (P1671) and country Germany, with
  de.wikipedia's infobox lengths (P2043). Only for `check_model.REGISTER["de"]`.
- Timetable feed, not wired yet (gtfs_served.py is shared): gtfs.de `fv_free` + `rv_free`
  (`https://download.gtfs.de/germany/fv_free/latest.zip`, `.../rv_free/latest.zip`, CC BY
  4.0, from DELFI, 30-day window, stop ids are gtfs.de's own so the join is by name). DELFI's
  official all-mode feed (470 MB, DHIDs) is the alternative.

## What RINF carries in Germany

DB InfraGO alone (`0080_IM`, "DB InfraGO" in COUNTRY), every line as VzG number, nature
"regular". No NE-Bahnen (AVG's Karlsruhe Stadtbahn track, SWEG/HzL, the Harz narrow gauge
and the like); Usedom's lines are DB InfraGO's and are in (6768, 6773, 6774), though DB's
own list lacks their names. Metros, U-Bahn, Stadtbahn and trams stay OSM lines. The ownership log's "likely register gaps" (566 km of way length in 154 places)
are mostly such other managers' lines that OSM routes run on: RB 62 Pleinfeld -
Gunzenhausen, RB 41 near Sonneberg, RB 31 to Kirchheimbolanden, RB 75 Haller Willem beyond
DB's 2950, Karlsruhe's S31/S32; not checked one by one. 64 numbers in DB's own list are not in
RINF; all are stubs under 4 km except 5623 Schaftlach - Tegernsee (12.6 km, the Tegernsee-Bahn's,
an OSM line here) and 2279 (a 14 km piece of Oberhausen - Emmerich, whose main number 2270 is
in RINF).

**Versions.** Every section is in RINF twice, valid 2026-01-01..2026-12-31 and from
2027-01-01, with different point URIs and the same uopids. rinf.py pairs 11,658 of them;
202 sections exist only in 2027 (Stuttgart 21's lines, new halts that split a section,
renamed points) and 190 only in 2026. `fix` drops whatever the label says is not valid today,
so the build is the 2026 network; from 2027-01-01 the same code builds the 2027 one.

**Station coordinates.** A big station's RINF point is one point of its whole area:
Göttingen 1,955 m from the platforms, Cuxhaven 2,327, Saarbrücken Hbf 2,303, Wiesbaden Hbf
1,630, Berlin Südkreuz 1,155. rinf.py only looks 1,000 m for the OSM station, so these were
junctions and their sections were traced from a yard. `relocate` (in `fix`) moves 115
passenger-typed points onto the OSM station of exactly their name within 3 km (72), or of
their name without the city in front within 1.6 km (43; OSM calls Hamburg's and Berlin's
S-Bahn stations "Barmbek", "Neukölln"). Points named "(Üst)", "(Awanst)", "(neu)" are left
alone ("Hamburg-Altona (neu)" is the Diebsteich station not yet open), and a two-part name
is looked up by its first part ("Berlin Hauptbahnhof - Lehrter Bahnhof" is Berlin Hbf; by its
second part, the block post "Mannheim-Waldhof - Lampertheim" had moved onto Lampertheim). The
1.6 km cap is there because "Hamburg-Eidelstedt" (2.8 km) and "Hamburg-Wilhelmsburg" (2.1
km) are yards, not the S-Bahn halts; moved, they bent 1220 Altona - Kiel 4 km long. Result:
11,860 of 13,132 passenger-typed points are an OSM station (5,727 distinct), 771 of those by
distance alone; 1,272 are not (closed halts, freight "Bahnhöfe", 2027-only halts) and are
junctions.

**S-Bahn on light_rail track.** OSM maps the Berlin and Hamburg S-Bahn as railway=light_rail
with station=light_rail stops. rinf.py's `light_rail_track` (added for Germany) puts that
track in the trace graph and those stations in the stop list, and makes a line traced mostly
on it kind light_rail, so build_model ties Hamburg's 1244 to the S-Bahn track rather than to
6100 Berlin - Hamburg 30 m away. 43 register lines are light rail: Hamburg's 12xx S-Bahn
lines and Berlin's 60xx.

## Names

"<number> <DB's Streckenkurzname>", en dashes between places: "1700 Hannover – Hamm (Westf)",
"6100 Berlin-Spandau – Hamburg-Altona", "4000 Mannheim – Basel – Konstanz" (DB writes the
via station as "--Basel--"). DB abbreviates to fit a column; `expand` writes out dotted
words from the words of the line's own RINF points ("Kaiserbr." Kaiserbrücke), city codes
from a fixed list (Bln, HH, Hmb, HL, Stg, Mü, Ffm, DO, BO, DU, GE ...), one-letter codes only
where the line's own points carry the city ("M-Gaschwitz" is Markkleeberg, "K M/Deutz"
Köln Messe/Deutz), and undotted stubs that begin a word on the line's points
("Johanngeorgenst" Johanngeorgenstadt). Short connecting numbers keep DB's switch names
("3144 Ehrang, W 3 – W 10"); a few stubs stay abbreviated ("Biesdorfer Kr Nord"). No English
names: Wikidata's English labels are article names over other extents. No `colours/de.csv`:
DB InfraGO's lines have no colours; S-Bahn and regional colours belong to OSM's lines.

Long-distance trains: OSM Germany maps DB's ICE and IC by DB's line number ("ICE 10",
"IC 26.1"; 29 ICE and 16 IC route_masters), each an hourly or two-hourly interval product,
so they are lines, with FLX 10/20/30/35 and RJ 27. Single trains are named trains: 31 in all
(EC, EuroNight, European Sleeper, Eurostar, Nightjet, TGV InOui, Leo Express, IC Łużyce).
build_model.py's `de` branch of `looks_like_service`; the managing session also stopped
EU_TRAIN flagging DB's line numbers in at/ch/nl/be, so ICE 90/91/43/12/20/78/79 agree across
countries.

## Counts (2026-10-02)

- RINF reader: 1,483 lines, 33,116 km traced against 33,061 km of RINF length; 48 merged
  sections left out for a rejected trace; 183 kept on their own relation's track with
  RINF's length off ("length off" in the log).
- After build_model: **1,097 register lines, 31,362 km** (43 of them light rail). 646
  junction-ended sections (1,754 km) no OSM passenger route runs over were dropped, emptying
  386 lines (freight curves, yards, port lines).
- 2,457 lines in all with 1,360 OSM lines (870 train, 365 tram, 53 subway, 51 light rail);
  13,729 stations; 31 named trains (10,647 km).
- Borders: 80 sections to a border point added to de's lines (2,128 km, mostly OSM lines'
  last stretch to the border), 16 sections built over a border cut there (22.5 km left to
  the neighbour), 49 border points named. 46 route ends run abroad with no border point on
  their track (every Basel Badischer Bahnhof route, Konstanz/Kreuzlingen, Schaffhausen
  line, Wasserbillig, Kehl - Strasbourg, Enschede, Hergenrath, Görlitz/Zgorzelec, Pieńsk,
  Zasieki, Grambow, Železná Ruda, Vojtanov, Dolní Žleb, Braunau and others): ch is built, so the
  Swiss crossings need `borders.EXTRA` entries.
- Ownership: 114,413 drawn ways owned by a register line; 120 single-track companions (229
  km, mostly third tracks and parallel single-track numbers); 459 lines lose a stretch to
  another register line drawn on the same rails (219 km); the fixed rule decides 1,268 places
  (3,304 km of way, mostly trams and NE-Bahnen); 11.9 km only named trains use.
- Sizes: lines.json 2.9 MB, foot.json 1.0 MB, ways.json 5.1 MB, stations.json 1.6 MB,
  de.pmtiles 18.5 MB.

## Check

`python check_model.py --region de`: against RINF's own section lengths, 865 lines of 2 km
or more, median 0.996, 75 off by more than 5% (all short connecting numbers, yard leads and
city S-Bahn stubs except the three below). Against de.wikipedia/Wikidata and DB, 46 lines,
every one within 5%, worst 0.96:

- **5919 Eltersdorf – Leipzig Hbf** (VDE 8) 0.96: Wikidata's 293.3 km against DB's own 284.0
  and RINF's 284.2; the gap is between the published figures.
- **2690 Köln – Frankfurt am Main Stadion** is checked against DB's 164.4 km; de.wikipedia's
  180 km adds the Wiesbaden and Köln/Bonn airport branches, which have numbers of their own.

## Still off, and why

- **Freight lines between two passenger stations are kept**, because build_model never
  questions a stop-to-stop section: 1280 Buchholz – Hamburg-Allermöhe (the Hamburg freight
  bypass) is 59.9 km built for 49.3, Meckelfeld - Buchholz traced over the 2200 main line
  beside it; 2315 Duisburg-Hochfeld Süd – Mannesmann (6.7 km) even came out as a tram line.
  5230 Waigolshausen – Gemünden (the Werntalbahn, excursion trains only) is 45.2 km for 40.1.
  A timetable check (gtfs.de) would grey these.
- **No OSM track to trace on**: 1100 Lübeck – Puttgarden north of Neustadt (Holst) and 6328
  Passow – Tantow have RINF points more than 1.5 km from any rail way in the extract (probably
  mapped as construction or disused while rebuilt; not checked), 3021 the Hunsrückbahn
  likewise; Frankfurt's S-Bahn tunnel Lokalbahnhof – Ostendstraße – Konstablerwache, Berlin's
  Wollankstraße – Schönholz, Oberspree – Spindlersfeld and Tegel – Schulzendorf pieces have
  no connected path; 4000's Swiss stretch Trasadingen - Bietingen lies outside the extract.
  38 sections unplaced, 20 with no path, 48 merged sections rejected; all left out, none drawn
  as a straight line (`rinf.py --dry de` lists them).
- **2414 Düsseldorf Hbf – Abstellbahnhof** traced over Stadtbahn track now that light_rail is
  in the graph (2.1 km, light rail kind); a few short Berlin numbers took light_rail kind from
  the S-Bahn beside them (6045, 6090, 6533; build_model's buffer rekind).
- **1,272 passenger-typed RINF points have no OSM station** and are junctions; most are
  freight Bahnhöfe and closed halts, but some are real stations whose RINF point is far off
  and whose name differs from OSM's (Berlin's "Hauptbahnhof-Lehrter Bf (Stadtb)", 1.2 km
  from Berlin Hbf, starts 6107 to Lehrte there). Sections between unmatched halts merge, so
  the strip shows fewer stops.
- **Names**: DB's short names, with some stubs left ("Biesdorfer Kr Nord", "Hamburg Hbf SB",
  "Rbf Einf"), and connecting numbers named by switches.
