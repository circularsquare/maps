# Canada register sources (built 2026-10-02)

What `ca_register.py` reads, where it came from, what it leaves out and why, and how the build
checks out. The downloads are in `data/raw/ca/` (gitignored); this file is the tracked record.
Nothing here needed a login, a key or an account.

## The short answer

- **Lines: FRA's North American Rail Network (NARN), by subdivision**, the same source and the
  same reader as the USA (`us_sources.md` is the method; `ca_register.py` runs
  `us_register`'s code with Canada's settings, see "How the reader is shared"). Each register
  line is one owner's subdivision: CN's Kingston Subdivision, CPKC's Galt Subdivision,
  Metrolinx's Newmarket Subdivision, VIA's own Alexandria Subdivision.
- **Stations: OpenStreetMap**, placed on a line where a passenger route that stops there runs
  along it (`us_register.place_stations`).
- **Geometry: OSM track** traced inside a corridor round NARN's line. NARN is close to OSM in
  Canada too: of 60,000 sampled points on OSM passenger route track, the distance to NARN is
  median 2.6 m, 90% 8.4 m, 95% 12.2 m, 99% 25.5 m (the USA: 1.4 / 6.0 / 8.1 / 25).
- **Lengths: NARN's own KM** per section (`chain`), so check_model compares every line.

Built: **99 register lines, 13,477 km**; 165 lines in all with OSM's, 1,494 stations (795 on
register lines, 239 of them junctions); 12 named trains (13,087 km).

## NARN Rail Lines, Canada

    https://services.arcgis.com/xOi1kZaI0eWDREZv/arcgis/rest/services/
        NTAD_North_American_Rail_Network_Lines/FeatureServer/0

`python ca_register.py --fetch` writes `data/raw/ca/narn_passenger.geojson` (9.9 MB,
`COUNTRY='CA' AND PASSNGR IS NOT NULL AND PASSNGR<>''`: **3,968 segments, 16,619 km**) and
`data/raw/ca/narn_layer.json` (the layer's own description and field domains). Fetched by the
managing session 2026-10-02; the layer was updated 2026-07-21.

**Licence.** The layer's `copyrightText`: "This NTAD dataset is a work of the United States
government as defined in 17 U.S.C. § 101 and as such are not protected by any U.S. copyrights.
This work is available for unrestricted public use." Its description says the data set "covers
all 50 States, the District of Columbia, Mexico, and Canada". Nothing in the metadata treats
the Canadian part differently or names a Canadian source for it. (Whether FRA built the
Canadian lines from Canada's NRWN, OGL-Canada, is not stated; if so, OGL-Canada asks only for
attribution.)

`PASSNGR` in Canada (km): V VIA Rail 13,503, O "Ontario Northland (Canada Network Only)" 2,470,
C commuter 197, T tourist 195, B Amtrak + commuter 134, A Amtrak 120. NARN uses O for every
non-VIA passenger operator in Canada, not only Ontario Northland: Tshiuetin over QNSL, the
ex-BC Rail line, Algoma Central. The domain lists R (rapid transit); Canada has none.
`NET`: M 15,869 km, X 580, I 68, A 55, Y 34, S 12.

### What is left out, and why

| left out | km | why |
|---|---|---|
| `NET` X, out of service | 509 | SVI's E&N (Victoria Subdivision, no trains since 2011) 213, CNTR (Cochrane - Taschereau) 134, CN North Bay - Capreol ("NEWM") 132, Deux-Montagnes (now the REM) 27 |
| `PASSNGR` T | 195 | the White Pass & Yukon Route 148, Alberta Prairie (Stettler) 35, Port Stanley 12: Anita's rule, service more often than about weekly |
| `NET` I, A, Y, S | 169 | industrial leads, abandoned, yards, sidings, though passenger-coded |
| **junction sections no scheduled passenger route runs over** (`drop_unserved_junction_sections`) | 2,373 | see below |
| junction sections build_model then drops | 10 | short tails into junctions |

**Passenger-coded track no train runs on.** NARN's Canadian passenger codes are older than its
US ones in places. Left out because no scheduled OSM passenger route runs over them (half of a
junction-ended section, as build_model's rule; stop-to-stop sections by us_register's own
25% rule):

| line | km | why |
|---|---|---|
| Algoma Central Soo Subdivision + CN Soo | 469 | Algoma Central passenger train suspended 2015; the Agawa Canyon tour train is seasonal tourism |
| ex-BC Rail: CN Squamish, Lillooet, Prince George | 579 | no scheduled train since 2002; the Rocky Mountaineer's seasonal "Rainforest to Gold Rush" is the only train (see below). 26 km of the Squamish Subdivision stay: the Tsal'alh Seton Train (Kaoham Shuttle), Lillooet - Seton Portage, runs daily |
| Ontario Northland Temagami, Ramore, Devonshire; CN Newmarket Subdivision Washago - North Bay | 609 | the Northlander (Toronto - Cochrane) stopped in 2012; its revival is announced for late 2026 with no start date. When it runs and OSM maps it, a rebuild picks these up |
| Gaspé line (CBC Cascapedia, Chandler West, CCFG Chandler East) | 325 | VIA's Montréal - Gaspé suspended since 2013 |
| Keewatin Railway Sherridon beyond Pukatawagan | 136 | the train runs The Pas - Pukatawagan only |
| CN Sussex Subdivision | 70 | Moncton - Saint John, no passenger train since 1994 |
| QNSL Northernland, CFRR Havre St-Pierre | 107 | no passenger service (Tshiuetin runs Sept-Îles - Schefferville only) |
| CPKC M&O beyond Hudson | 13 | exo1 ends at Hudson since 2010 (Rigaud dropped) |
| and short pieces | ~60 | Cartier, Victoria, Taschereau... |

**OSM routes that never vouch for track** (`NOT_A_SERVICE`): BC Rail's old passenger route
("British Columbia Railway", operator CNR, 315 ways), "CTRW Prince Albert Subdivision" (a
freight line mapped as a route), Port Stanley Terminal Rail, the heritage railways (Alberni &
Pacific, Fraser Valley, Washago-Rama, Fort Edmonton Park), and the Rocky Mountaineer. Without
them build_model's own junction rule kept 600 km of the ex-BC Rail line on the Rocky
Mountaineer's word alone (`drop_unserved_junction_sections` exists for that: build_model
measures route cover against every OSM route it has).

## Names, grouping

As the USA (`us_sources.md`): SUBDIV, written "<Name> Subdivision". `CA_OWNERS` maps the
Canadian reporting marks (VIA, GO = Metrolinx, HBRY, KRC, ONT, QNSL, TSH, ACR, SVI, BCR); CBC,
CCFG, CFRR and ARMD stay as marks (all their track is left out). NARN's Canadian file has no
unnamed track. Subdivisions keep NARN's French spellings without accents (St-Hyacinthe,
Lac St-Jean, Tete Jaune), as the railways write them.

## Holes (`--fetch-holes`)

`python ca_register.py --fetch-holes` (run by the managing session 2026-10-02: 154 envelopes,
7,456 segments, 8.1 MB, 171 s) writes `data/raw/ca/narn_holes.geojson`: NARN's uncoded
Canadian segments round the 642 km of OSM passenger route track lying more than 25 m from
passenger-coded NARN. Taken: **100 segments, 328 km**, main network only:

- **Gaps filled**: CPKC Cascade 35.8 km (the West Coast Express's Vancouver end), CPKC
  Adirondack 25.6 (exo4 Candiac), GO Lakeshore East 4.8, GO Newmarket 1.3.
- **Lines of their own**: CPKC Parry Sound Subdivision 149 km (VIA's Canadian eastbound,
  directional running, below), CPKC Parc Subdivision 54.8 (exo2 Saint-Jérôme), HBRY Thompson
  Subdivision 48.9 (the Churchill train's run into Thompson), CPKC St-Luc Branch, and short
  pieces.
- **Second tracks** folded into their line (`companion_of`): Galt 3.4 km, Thicket 0.3.

Tried and dropped: also taking uncoded yard, siding and lead track (NET Y, S, I) as holes
brought ~250 km of passing sidings in as "second tracks" and same-named stub lines (CN Yale in
35 pieces); the main file's passenger-coded Y/S/I segments took nothing.

## Directional running (`DIRECTIONAL`)

CN and CPKC run their two single-track main lines as one double track in two places, and the
Canadian takes one railway's line each way. The USA's rule (Anita, 2026-10-02: both
directions one track unless far apart; 90% within 3 km) is applied by measure:

| pair | 90% within | |
|---|---|---|
| CPKC Thompson beside CN Ashcroft (Basque - Kamloops) | 301 m | folded: Thompson is Ashcroft's companion |
| CPKC Parry Sound beside CN Bala (Parry Sound - Sudbury) | 17.6 km | kept a line of its own |
| CPKC Cascade beside CN Yale (Mission - Basque, the Fraser canyon) | 8.0 km | kept; it also carries the West Coast Express |

## Named trains and lines (the proposed `looks_like_service` "ca" branch)

Named trains (12, 13,087 km): VIA's Canadian, Ocean, Jasper - Prince Rupert, Winnipeg -
Churchill, Montréal - Jonquière/Senneterre, Sudbury - White River (all tagged
service=long_distance or night, or a VIA route outside the Corridor), the Maple Leaf, the
Adirondack (6 km in Canada as an OSM line), the Polar Bear Express, Tshiuetin, Keewatin
Railway (2-5 trains a week each), and the Rocky Mountaineer. Lines: VIA Rail Corridor, GO's
seven lines, exo's four, UP Express, West Coast Express, the Tsal'alh Seton Train (daily),
and the metros and LRTs.

Not an OSM line in Canada: the Amtrak Cascades (its relation has one stop inside the
extract). The register lines it runs on (BNSF and CN New Westminster, CN Yale) are built.

## Borders

NARN's node ids are one numbering across the border, and both builds name an end within 3 km
of a crossing "Canada – United States border" (`us_register.BORDERS`). Canada's lines end at
the very ids the US build has:

| crossing | train | id | Canadian line | US line |
|---|---|---|---|---|
| Blaine - White Rock | Amtrak Cascades | uj303674 | BNSF New Westminster Subdivision | BNSF Bellingham |
| Rouses Point - Lacolle | Adirondack | uj491551 | CN Rouses Point Subdivision | CN (unnamed, 0.14 km) |
| Niagara Falls, the Whirlpool bridge | Maple Leaf | uj467083 | CN Grimsby Subdivision | CSX Niagara Branch |

Nothing is needed in `borders.EXTRA`: register lines meet at shared ids, and no OSM line in
either build runs to the border (both extracts end their routes at the last station).

## Checks

`python check_model.py --region ca`:

- against NARN's own chainage, 88 lines of 2 km or more: **median 0.995, none off by 5%**.
- against Wikipedia (`check_model.REGISTER["ca"]`):

| line | built | published | ratio | extent |
|---|---|---|---|---|
| Wekusko Subdivision | 218.2 | 219 | 1.00 | The Pas km 0 - Wabowden km 219, Template:Hudson Bay Railway |
| Thicket Subdivision | 304.1 | 305 | 1.00 | Wabowden - Gillam km 524 |
| Herchmer Subdivision | 293.1 | 296 | 0.99 | Gillam - Churchill km 820 |
| Island Falls Subdivision | 297.1 | 299 | 0.99 | Cochrane - Moosonee, 186 mi |
| Alexandria Subdivision | 123.9 | 123 | 1.01 | Ottawa km 446 - Coteau Jct km 569, Template:Via Corridor routing |
| Kingston Subdivision | 488.0 | 483 | 1.01 | Dorval - Pickering, "approximately 300 miles" |

- **The network as a whole** (`CA_PATH_CHECKS`, two passes in the build log: "path check" over
  passenger-coded NARN alone, "built path check" over the built register geometry, holes
  filled and unserved track dropped). Built:

| path | built km | Wikipedia | ratio |
|---|---|---|---|
| Toronto - Montréal | 536.9 | 539 | 0.996 |
| Toronto - Ottawa | 442.7 | 446 | 0.993 |
| Montréal - Québec | 268.6 | 272 | 0.988 |
| Toronto - Windsor | 367.6 | 359 | 1.024 |
| Toronto - Niagara Falls | 131.9 | 132 | 0.999 |
| Montréal - Halifax (the Ocean) | 1,334.8 | 1,346 | 0.992 |
| **Toronto - Vancouver (the Canadian)** | 4,404.5 | 4,466 | 0.986 |
| Winnipeg - Churchill | 1,588.5 | 1,710 | 0.929 (the train runs into Thompson and back, ~100 km a shortest path skips) |
| The Pas - Churchill | 814.2 | 820 | 0.993 |
| Jasper - Prince Rupert | 1,149.8 | 1,160 | 0.991 |
| Sudbury - White River | 480.1 | 484 | 0.992 |
| Cochrane - Moosonee | 297.4 | 299 | 0.995 |
| Sept-Îles - Schefferville | 568.5 | 573 | 0.992 |
| Union - Barrie | 100.5 | 101.4 | 0.991 |
| Union - Milton | 50.2 | 50.2 | 1.001 |
| Union - Old Elm (Stouffville) | 45.9 | 49.6 | 0.926 (the last 3.3 km to Old Elm is not in NARN; OSM's Stouffville line owns it) |
| Waterfront - Mission City | 67.0 | 69 | 0.971 |

Before the holes, Montréal - Québec, Toronto - Vancouver, Sept-Îles and the West Coast Express
had no path over passenger-coded NARN at all.

## What is off

- **Track only named trains run over, owned by nobody** (745 km of way): the Rocky
  Mountaineer's ex-BC Rail line 718 km (intended), and ~27 km of station approaches NARN files
  as yard track or not at all: Edmonton (Blatchford, Gateway; 13 km), Sept-Îles 3, Vancouver
  Pacific Central 1.8, Montréal (Garneau) 1.7, Candiac 1.5, Halifax 1.5, Jasper, Saskatoon.
- **Operating-pattern track no register line has** (OSM lines own it; counts as their line):
  exo5 Mascouche's 2014 track 11.9 km, UP Express's airport spur 5.5, VIA near Montréal-Ouest
  6.9, Stouffville - Old Elm 3.3.
- **Station merging**: Canada builds with region "ca", so the US 150 m rule for same-named
  metro stations (`METRO_DUP_REGIONS`) does not apply. Toronto's and Montréal's metro names are
  unique; the same-named records 150-500 m apart in Canada's OSM are platform nodes of one
  station. Same-named TTC streetcar stops on two routes (Bathurst Street at Queen and at King,
  ~400 m) merge under the general 500 m rule, as tram stops do everywhere.
- The build log labels its reader lines "US:" (us_register's own log text).

## Station overrides (`CA_NOT_ON`, 2026-10-03)

The US list's method (`us_sources.md`, "Station overrides"; `us_register.NOT_ON`), Canada's
rows in `ca_register.CA_NOT_ON`, set by `adopt()`. Register sections with an end at a station
no route lists: 35, at 17 stations. Fifteen are real stops (GO's Kitchener line, VIA's Port
Hope, Gananoque, Trenton Junction, the Polar Bear Express's Gardiner, Keewatin's Pawistik,
and Arnaud Junction, which OSM tags a Tshiuetin halt; kept, it has only two junction sections
of 0.05 and 1.8 km on the flag-stop line). Two are no
stop of any scheduled train, and are now `*` rows:

- **Kamloops** (the Rocky Mountaineer's own station, 1.5 km from VIA's Kamloops North): the
  Canadian passes it; it gave CN's Clearwater Subdivision a 113 km section from it.
- **Hays**, a station record with no network 300 m from exo's Saint-Constant: no exo4
  station has that name; it gave CPKC's Adirondack Subdivision a 0.26 km Hays - Saint-Constant.

Rebuilt 2026-10-03: 99 register lines, 13,477 km; Adirondack 26.02 -> 26.06 km and
Clearwater 223.19 -> 223.25 (one section fewer each); check_model unchanged (chainage median
0.995). The two station ids carry to nothing (no station within 200 m).

**OSM lines** (`us_register.osm_extra_stops`, through `rules/ca.py` `extra_route_stops`,
waiting on the build_model hook, not in the rebuild): trial 2026-10-03, 15 unlisted stations added as stops of
19 routes; GO's Kitchener line goes from 1 section (28 km) to 8 (101 km, Union - Kitchener),
VIA's Corridor gains Port Hope, Gananoque and Trenton Junction.

## Lines in pieces (2026-10-04)

As the USA (`us_sources.md`, "Lines in pieces"): a register line whose sections do not all
connect once build_model has dropped what no train runs over becomes one line per piece,
`us_register.split_pieces`, which `ca_register` exposes as its own `split_pieces` for the
build_model hook (not landed yet: until then nothing is split). Canada's own junction drop
(`drop_unserved_junction_sections`) runs in the reader before it, so the pieces are what both
drops leave. The biggest piece keeps the id; the others take `us_register.piece_id`.

Trial 2026-10-04 (hook patched in at run time): **16 lines split into 48** (register lines 99
-> 131), every old id still ships, km unchanged but 0.13 km of stubs left out (crossovers at
Toronto Union, Charny). The biggest: CN Rivers 445.8 + 4.1 (Winnipeg apart), CN Edson 207.4 +
156.5 + 2.6 (Hinton - Jasper apart), QNSL Wacouna 345.5 + 11.6 + 1.8, CN Newcastle 277.4 + 0.4
(Campbellton), CPKC Cascade 139.4 + 65.8 + 1.1 + 0.6 (the West Coast Express's Waterfront -
Port Haney apart), CN St-Hyacinthe 59.6 + 1.4 + 0.1, HBR Thompson 48.0 + 0.4 + 0.2, CPKC Parc
32.7 + 15.5 (exo2's Parc - Bois-de-Boulogne), CPKC Galt 34.4 + 13.4, BNSF New Westminster 19.2
+ 15.9. CPKC's Adirondack Subdivision comes out in 9 pieces (8.9 km and less), the stretches
exo4 runs over between ones it does not. Penn Station's fix (`us_register.SEGMENT_NAME`) reads
US segments only; Canada's build is unchanged by it (ab.py: identical, foot.json and ways.json
too).

One-stop lines: 12 Canadian register lines have one stop and two or more junction ends (2
with a junction end that leads nowhere in the app: Brockville at "Brockville / Winchester",
and a 1.8 km Wacouna piece at Arnaud Junction); 28 have none. Not changed.

## How the reader is shared

`ca_register.adopt()` points us_register's country-specific globals at Canada's for the
process (one region per build_model run): `RAW`, `WHERE`, `HOLE_WHERE`, `PASSENGER`,
`TOURIST_OWNERS`, `OWNERS` (updated), `NAME_FIX` (updated), `PATH_CHECKS`,
`HOLE_SKIP_NETWORKS` (VIA Rail vouches for holes here), `NOT_ON` (Canada's station
overrides), and the functions `read_narn` (country
'CA'), `load_osm` (data/proc/ca; build_stations called with region "ca"), `infra_names`
(data/proc/ca), `line_id` (hashes "ca|owner|key") and `is_passenger_route` (+
`NOT_A_SERVICE`). A rename of any of these in us_register breaks Canada loudly.

## Station-twinned stubs and kinds (2026-10-09)

From us_register's changes for South Station (us_sources.md "South Station, Tucson, and
kinds"): a dead-end stub of 50 m or less past a stop is left out, so the stop ends the line.
Five in Canada: Vancouver Waterfront, Saint-Jérôme (two tracks), LaSalle, Arnaud Junction
(Adirondack 2.59 -> 2.56 km, Cascade 65.85 -> 65.84, Parc 32.68 -> 32.61, Wacouna 1.84 ->
1.79). `REGISTER_KIND_SURE` in rules/ca.py (waiting on build_model) keeps CPKC's Canpa
Subdivision "rail"; the track test called it "subway".

Open, found on the way: the Westmount Subdivision's last 3.6 km past Vendôme ends "near
Lucien-L'Allier", and the model has Lucien-L'Allier only as the Orange Line's metro station.
OSM's exo1 and exo4 routes both end at Vendôme. Either the trains still run to
Lucien-L'Allier and its train station is missing, or that stretch has no passenger trains.

## Commands

    python ca_register.py --fetch          # NARN Canada + layer metadata, under a minute
    python ca_register.py --narn           # the register alone: lines, km, path checks
    python extract.py --region ca --pbf data/raw/canada-latest.osm.pbf     # managing session
    python ca_register.py --fetch-holes    # needs data/proc/ca; ~3 min
    OMP_NUM_THREADS=2 python build_model.py --region ca --register ca_register:data/raw/ca/narn_passenger.geojson
    OMP_NUM_THREADS=2 python build_tiles.py --region ca
    python check_model.py --region ca

On 2026-10-02: build_model 1.8 min, build_tiles 41 s, 7.8 MB of tiles (30,960 tiles to z13).
