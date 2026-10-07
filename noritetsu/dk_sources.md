# Denmark register sources (built 2026-10-03)

What the Danish build reads, where each piece came from, and what is still wrong with it.
Denmark is built with `rinf.py`; how the reader works is in its docstring, and the per-country
entry is `rinf_countries/dk.py` (its docstring explains the grouping). Downloads live in
`data/raw/rinf/dk/` and `data/raw/gtfs/dk/` (gitignored). Nothing needed a login or a key.

## Run

```powershell
python rinf.py --fetch dk                      # RINF and Wikidata, a few seconds (0.7 MB)
python extract.py --region dk --pbf data/raw/denmark-latest.osm.pbf   # delete the .pbf after
# the timetable: FEEDS["dk"] in gtfs_served.py (sent as a diff), then
python gtfs_served.py --fetch dk               # Rejseplanen via Transitous, 50.7 MB -> 2.0 MB
python tools/slot.py -- python build_model.py --region dk --register rinf:data/raw/rinf/dk
python tools/slot.py -- python build_tiles.py --region dk
python check_model.py --region dk
python tools/slot.py -- python rinf.py --dry dk   # the reader alone, with its full log
```

The first build's feed was fetched by a scratch copy of `gtfs_served.fetch` (FEEDS["dk"] not
yet landed); the file is the same.

## Sources

- **ERA RINF**, SPARQL at `https://graph.data.era.europa.eu/repositories/rinf-plus`, fetched
  2026-10-03: 340 sections, 321 points, all with coordinates. Infrastructure managers 8601
  (Banedanmark, 1,652 km), 8607 (Lokaltog: Odsherredsbanen, Tølløsebanen, Østbanen,
  Lollandsbanen, 165 km), 8604 (Lokaltog: Nærumbanen, Frederiksværkbanen, Hornbækbanen,
  Gribskovbanen, 91 km), 8606 (Nordjyske Jernbaner: Hirtshals, Skagen, 49 km). RINF gives no
  names for these codes; the names are from the lines each holds. Licence: ERA's (EUPL 1.2 for
  the service; the dump is CC BY 4.0).
- **Not in RINF**: Midtjyske Jernbaner's Lemvigbanen (Vemb - Thyborøn, 56.3 km), Tønder - the
  German border, the Rødby Færge branch (closed 2021 for the Femern works), and every metro and
  light rail (Copenhagen Metro M1-M4, Hovedstadens Letbane, Odense Letbane, Aarhus Letbane with
  the Grenaa and Odder lines). They are OSM lines, as elsewhere.
- **da.wikipedia line articles**, raw wikitext through the API, retrieved 2026-10-03: the line
  names (the article titles), each infobox's `linjelængde` (the check), and the line extents
  ("Sydbanen" is Ringsted - Nykøbing F with the Rødby and Gedser branches; "Nordbanen"
  København H - Hillerød; "Lille Nord" Hillerød - Helsingør sharing Kystbanen's track from
  Snekkersten). CC BY-SA 4.0; names and lengths are facts.
- **Wikidata**, `wikidata.json` from `--fetch` (63 rows): lengths (P2043) where the article has
  none (Lille Syd 61.4, Vejle-Holstebro-banen 115, Fredericia - Padborg 110.6). CC0.
- **Rejseplanen's national timetable** (all operators: DSB, DSB S-tog, DSB Vores Tog, Lokaltog,
  Nordjyske (NT), Midtjyske (Midttrafik 92/93), Skånetrafiken's Øresundståg, Snälltåget, metro,
  light rails), through Transitous's mirror `api.transitous.org/gtfs/dk_rejseplanen.gtfs.zip`
  (the original is `rejseplanen.info/labs/GTFS.zip`, 55 MB), 50.7 MB slimmed to rail, 2.0 MB.
  Rail is route_type 2 and 109 (S-tog); the metro (1) and light rails (0) fall outside
  RAIL_TYPES, which is right: they are not register lines. Rejseplanen's terms.
- **OpenStreetMap**, Geofabrik `denmark-latest.osm.pbf` (2026-10-03), ODbL: 9,611 track ways,
  176 route relations (91 train), 95% of main-line km under a route. OSM Denmark names 91% of
  its main and branch track for its line (`probe_kr_ways.py`: Vestbanen, Den fynske hovedbane,
  Kystbanen... and "Østjyske Længdebane" for all of Padborg - Frederikshavn), and maps the S-bane
  as railway=light_rail.
- **The border crossings**: OSM's rail ways against the admin_level=2 boundary ways, read from
  Overpass in three small boxes (Tønder, Padborg, the Øresund) on 2026-10-03.

## Choosing the register

RINF's 340 ids are one per section, but they are not arbitrary: a six-digit id is Banedanmark's
line number in its first two digits and a sub-line in the third ("016080": line 01, sub-line 6,
Ringsted - Korsør; "244170": line 24, sub-line 4). So the grouping needs no chains of points as
Sweden's does: the first three digits name the line, with six whole ids where a sub-line holds
two named lines. Twelve-digit ids are two six-digit ones joined: station-internal links
(København H's S-bane and main-line points, 0.04-0.36 km) and spurs to freight terminals; all
but two are left out. The alternative, OSM's named track (the Korea recipe, 91% named), was not
needed: RINF has the stations in order and the extent of every line, and OSM's own names lump
five lines into "Østjyske Længdebane".

Names are da.wikipedia's article titles, which are also OSM's way and route=railway names
for most lines. Two are mine: "Lindholm-Aalborg Lufthavn" (the 2020 airport branch, no
article) and the unnamed main-line link Lersøen - Østerport (076), which rinf.py names by its
ends and which is dropped for want of trains anyway.

The register lines (dk.py PREFIX): Vestbanen, Storebæltsforbindelsen (Korsør - Nyborg),
København-Køge-Ringsted-banen (with its approach from København H and the Hvidovre curve RE 50
uses), Sydbanen, Gedserbanen (greyed), Lille Syd, Nordvestbanen, Lille Nord, Kystbanen,
Øresundsbanen (with Vigerslev - Kalvebod, RE 50's way from the airport to København Syd), Den
fynske hovedbane, Svendborgbanen, Fredericia-Aarhus-banen, Aarhus-Randers-banen,
Randers-Aalborg Jernbane, Vendsysselbanen, Lindholm-Aalborg Lufthavn, Fredericia-Vamdrup-banen,
Vamdrup-Padborg-banen, Snoghøj-Taulov-banen, Sønderborgbanen, Lunderskov-Esbjerg-banen,
Bramming-Tønder-banen, Den vestjyske længdebane, Langå-Struer-banen, Vejle-Holstebro-banen,
Thybanen, Skanderborg-Skjern-banen, Varde-Nørre Nebel Jernbane, Hirtshalsbanen, Skagensbanen,
the Lokaltog lines (Nærumbanen, Frederiksværkbanen, Hornbækbanen, Gribskovbanen,
Odsherredsbanen, Tølløsebanen, Østbanen, Lollandsbanen) and the S-bane (Nordbanen,
Klampenborgbanen, Høje Taastrup-banen, Frederikssundbanen, Hareskovbanen, Køge Bugt-banen,
Ringbanen). S-tog A-H, Regionaltog, InterCity, Lokaltog's numbered lines and the Øresundståg
are OSM lines (operating patterns) over them.

## RINF's Danish lengths are not track lengths

Every Danish section is shorter in RINF than the straight line between its two points: Sorø -
Slagelse 12.3 km for 13.7 km as the crow flies, København H - Østerport 1.2 km for 2.8, and the
sums are 70-90% of the published lengths (København H - Korsør 78 km for 111, Nordvestbanen 58
for 79). The coordinates are right (checked against the stations); the lengths leave out the
station areas. So they cannot be a tolerance or a chainage check. rinf.py's new `km_floor` hook
(no-op unless set) judges a trace against RINF's length as a floor (at least 0.95 of it, less
0.2 km) and an upper bound of 1.3 x crow-fly + 0.5 km or 1.15 x RINF + 3 km, and ships no
km_official, so check_model's chainage check skips Denmark; the da.wikipedia lengths are the
check.

## Borders

- **Padborg - Flensburg**: RINF's EU00059, 1 m from where OSM's track crosses the border. Both
  sides have it (border_points.json).
- **Tønder - Niebüll**: no RINF point (Germany's RINF ends at "Niebüll DB-Grenze", 0.3 km out of
  Niebüll, and Denmark's at Tønder). Needs `borders.EXTRA` ("xTonder", 8.872908, 54.899385,
  ["de", "dk"]): where NEG's track (way 223519292) crosses boundary way 1052245273. Tønder - the
  border (about 4 km) is then an OSM line's (RB 66 / R8) on the Danish side.
- **The Øresund bridge**: RINF's EU00141 "Peberholm grænse" is the boundary between Banedanmark
  and Øresundsbro Konsortiet at Peberholm's west end, 5.4 km inside Denmark; the Swedish extract's
  track stops 2.3 km short of it, so Sweden's build ends at Lernacken. dk.py moves the point to
  where the bridge crosses the border (12.808962, 55.579239: ways 1185526677/8 against boundary
  way 71417261) and lengthens Københavns Lufthavn - Peberholm grænse by 5.2 km. borders.py and
  se.py need the same move (diffs in the report of 2026-10-03; se's tried on its RINF data:
  EU00141 - SE00100 becomes 6.161 km). Then Sweden and Germany need rebuilding.
- Rødby - Puttgarden: no crossing (the ferry ended 2019; the Femern tunnel is not open).

## What is in the build (2026-10-03)

108 lines, 605 stations, 7,899 route-km. **46 register lines, 2,408 km** (RINF's own lengths
sum to 1,958). 62 OSM lines: S-tog A, B, Bx, C, E, F, H; Metro M1, M2, Cityringen,
Nordhavnsmetro; Hovedstadens Letbane, Aarhus L1/L2, Odense Letbane, Lokaltog 910; and the
train patterns (InterCity 1/4/5, InterCityLyn 3/5, IC 81, Regionaltog 40-61, DSB Vores Tog,
Lokaltog's lines, Nordjyske RE69/RE76/79, Midtjyske's Lokalbane 92 and 93 (Lemvigbanen, which
RINF lacks), Arriva's 50 and R8, the Øresundståg tables 90/95/100, SJ's Tåg 80). No named trains
in the extract.

Stations: 260 passenger-typed RINF points, 258 an OSM station (Havrebjerg and Mårsø have none).
`osm_stops` added 227 halts to register sections (some counted twice where they cut two
pieces); `osm_stop_extra` made 12 more OSM stations candidates. The timetable matches 435 of
the 532 feed stations trains call at; of the 25 left inside Denmark, 16 are Lemvigbanen's (an
OSM line, not checked), plus Bispebjerg (Ringbanen), Orehoved (RINF types it a junction and OSM
has no station of its name within 1 km), Mårsø, Helsingborg/Landskrona/Flensburg abroad.

`check_model.py --region dk`: every one of the 37 lines in REGISTER within 3% of its published
length, or off by exactly what its note says: Kystbanen 46.0/46.0, Den fynske hovedbane
88.8/88.6, Fredericia-Aarhus-banen 108.2/108.0, Den vestjyske længdebane 145.9/146.0,
Skanderborg-Skjern-banen 111.6/111.9, Langå-Struer-banen 102.0/102.4, Vestbanen 108.0/111.0,
the S-bane lines 0.99-1.01. Expected off: København-Køge-Ringsted-banen 1.07 (the approach
from København H), Sydbanen 0.59 (the article counts the Rødby and Gedser branches), Lille Nord
0.85 (Snekkersten - Helsingør is Kystbanen's), Bramming-Tønder-banen 0.94 (to Tønder, not the
border), Storebæltsforbindelsen 1.37 (station to station against the 17 km link). Unexplained:
**Gribskovbanen 42.0 against 50.6**: both branches are built, on OSM's track; the article does
not say what its 50.6 counts.

The timetable check: 2,380 of 2,408 register km served, none closed, 22 junction-ended
sections (83 km) kept. Gedserbanen (22.7 km) is greyed by `suspended`.

## Still off, and why

- **Borders not yet joined**: Tønder - the border (about 4 km) is drawn by no line until
  `borders.EXTRA` has the Tønder point; the Øresund point moves only in this build until
  borders.py and se.py move it too. Then Denmark, Sweden and Germany need rebuilding.
- **The Rødby branch** (Nykøbing F - Rødby Færge, closed 2021 for the Femern works): OSM's
  Regionaltog 54 relations still hold its track, 30.8 km of ways drawn as that line's. The
  line's stops end at Nykøbing F, so no section counts it.
- **Hovedstadens Letbane at Glostrup**: 1.0 km of its ways, beside the S-bane's, are owned by
  Høje Taastrup-banen (ownership by nearness); a Letbane ride there credits the S-bane line.
- **No colours/dk.csv**: the S-tog, metro and light-rail lines carry OSM's colours; the
  register lines have none (Banedanmark publishes none).
- **Names**: no English names; "Lindholm-Aalborg Lufthavn" is ours. The 076 link Lersøen -
  Østerport is rejected by the trace check (RINF 1.4 km, traced 4.7) and has no trains anyway.

## Named trains (rules/dk.py)

International EC/ECE/EN/NJ, ICE by a train number (an "ICE 38"-style DB line stays a line, as in
Germany), EuroCity/EuroNight/Nightjet/European Sleeper, Snälltåget, night trains, and the Tønder
- Højer museum train. None of the international ones is mapped in OSM Denmark today. DSB's
InterCity and InterCityLyn carry service=long_distance and are lines; SJ's "Tog 80: København =>
Stockholm" is a line, as Sweden's rule has it.
