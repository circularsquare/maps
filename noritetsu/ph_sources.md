# Philippines (ph): sources

Built 2026-10-08 (asia agent): see "Build (2026-10-08)" at the end. The survey below was the research for it.

## Survey (2026-10-08)

### What runs

**Metro Manila rapid transit** (all daily, every few minutes):

| line | termini | stations | km (en.wikipedia infobox, read 2026-10-08) | operator |
|---|---|---|---|---|
| LRT Line 1 | Fernando Poe Jr. - Dr. Santos | 25 | 26 (Cavite extension phase 1, Redemptorist-Aseana - Dr. Santos, opened 16 Nov 2024) | Light Rail Manila Corporation |
| LRT Line 2 | Recto - Antipolo | 13 | 17.6 (Antipolo since July 2021) | LRTA |
| MRT Line 3 | North Avenue - Taft Avenue | 13 | 16.9 | DOTr (since the BLT ended 15 July 2025) |

Not open, leave out (track in OSM is likely tagged construction): **MRT Line 7** (North Avenue -
San Jose del Monte, 22.8 km; DOTr in January 2026: stations open by Q2 2027), **LRT-1 Cavite
phases 2-3** (target 2030), **LRT-2 west extension** (approved, not started), the **Metro Manila
Subway** (from 2028+), the **North-South Commuter Railway** (Clark - Calamba, 147 km; target
2027; no section open), the **PNR South Long Haul** (stalled), the Common Station at North
Triangle (2028).

**PNR (Philippine National Railways)**, metre gauge, South Main Line only:

| service | route | status (en.wikipedia "Philippine National Railways", read 2026-10-08) |
|---|---|---|
| Inter-Provincial Commuter | Calamba - San Pablo - Lucena (and San Pablo - Lucena, Calamba - San Pablo) | running. Lucena - Calamba resumed 21 Oct 2024; stops Calamba, (Halang Dos,) Pansol, Masili, UP Los Baños, IRRI, San Pablo, Tiaong, Candelaria, Lutucan, Sariaya, Lucena. 77 km (en.wikipedia "PNR South Main Line") |
| Bicol Commuter | Lupi Viejo - Sipocot - Naga | running; extended to Lupi Viejo 5 Nov 2025 |
| Bicol Commuter | Naga - Legazpi | **suspended** since 10 Nov 2025 (Typhoon Uwan damaged the bridge at Guinobatan); still suspended in July 2026 (en.wikipedia "PNR Bicol Commuter" section). Build greyed |
| Metro North / Metro South Commuter | Tutuban - Alabang - Calamba | **closed** 27-28 March 2024 for NSCR construction "for at least five years"; much of the track is being rebuilt as the NSCR. Greyed, or left out where the NSCR has replaced the track |

PNR's active route length: 133.09 km (en.wikipedia "Philippine National Railways"); the South
Main Line infobox: 479 km, of which 255 km active.

No other rail in the country: Panay Railway and the Mindanao Railway are not running/not built.

### Sources

| source | gives | licence | where |
|---|---|---|---|
| OSM via Geofabrik `asia/philippines-latest.osm.pbf` (**580 MB**) | track, stations, route relations (counts below) | ODbL | Overpass sample `data/raw/ph/survey/osm_route_relations.json` |
| en.wikipedia "List of Philippine National Railways stations" | every PNR station, line by line and province by province, in order; italic = closed, bold = major | CC BY-SA | `data/raw/ph/survey/enwiki_list_of_pnr_stations.wikitext` |
| en.wikipedia "PNR South Main Line" | services, stops, lengths (Inter-Provincial Commuter 77 km) | CC BY-SA | `data/raw/ph/survey/enwiki_pnr_south_main_line.wikitext` |
| en.wikipedia station articles ("Lucena station" etc.) | coordinates and adjacent stations per service; no km | CC BY-SA | not saved |
| Sakay.ph's Manila GTFS (Mobility Database mdb-1106, github.com/sakayph/gtfs) | LRT-1, LRT-2, MRT-3, PNR routes and stops | no licence stated | **stale**: last commit 24 March 2015 (before LRT-2's Antipolo and LRT-1's Cavite extensions). Not useful |

OSM (Overpass, 2026-10-08, inside the Philippines): 488 `railway=rail` ways, 441 named, 394
`usage=main`; 1,764 other rail-ish ways (light_rail, subway, construction, disused, abandoned:
the NSCR, MRT-7 and Subway works and the dead northern lines); only 81 station/halt nodes, so
extract with `--station-areas`. Route relations:
- metros with colours and refs: LRT-1 (110418, 8000260, green, Fernando Poe Jr. - Dr. Santos;
  two older Monumento - Dr. Santos relations 10463194/5), LRT-2 (110410, 8000264, purple,
  tagged `route=subway`), MRT-3 (109159, 8000253, yellow, `light_rail`). No route_master.
- PNR trains: Inter-Provincial Commuter San Pablo - Lucena (13803169/70), Bicol Commuter Sipocot
  - Naga (10976231/2), Naga - Legazpi (10976183/4), Ligao - Naga (16148783/4); the closed Metro
  North Commuter and Shuttle Service (9165727/8, 8545505, 10015475): drop these, the service
  ended in March 2024.
- `route=railway`: PNR South Main Line both ways (8343623, 12520016), and about 20 historical
  PNR lines (North Main Line, Tarlac - San Jose, Tabaco, Batangas, Naic, Carmona...), the NSCR
  and Subic - Clark (not open), "Metro Manila Subway" as route=train (not open: clip).

No current rail GTFS: the Mobility Database (2026-10-08) has only Sakay's 2015 feeds (mdb-1105
P2P buses, mdb-1106) and a deprecated 2014 one (mdb-1269). No PNR chainage table found (station
articles carry no km; no route-diagram template with km).

### Recipe

- **LRT-1, LRT-2, MRT-3**: OSM lines, as metros are everywhere (vn's Hà Nội lines, th's BTS).
  Check OSM has the five Cavite-extension stations to Dr. Santos.
- **PNR**: a hand line list through rinf.py (the `nafrica_register` / `za_register` pattern),
  each line its ordered served stations from the wiki list (non-italic entries) matched to OSM
  stations, traced over OSM's track:
  - South Main Line, Calamba - Lucena (running), Lupi Viejo - Naga (running), Naga - Legazpi
    (suspended, greyed). One register line "South Main Line" cut into pieces, or three lines;
    the build agent decides (Indonesia keeps a line with a closed middle as two pieces).
  - Tutuban - Calamba (closed 2024): greyed where OSM still has the track. Lucena - Lupi Viejo
    (no train since 2006/2013, Bicol Express) is not built or greyed: the agent decides, the
    rule is "greyed if it had trains", and it did until 2013.
  - Named trains: none run now.
- `check_model` REGISTER: LRT-1 26, LRT-2 17.6, MRT-3 16.9 (OSM lines), Inter-Provincial
  Commuter 77 (Calamba - Lucena).
- No borders.

Expected: 3 metro lines (60 km) + PNR running about 130 km (+ about 100 km Naga - Legazpi and
about 55 km Tutuban - Calamba greyed). The extract is the slow part: 580 MB, a few minutes;
`--bbox 120.5,12.9,124.0,15.5` (Metro Manila to Legazpi, all of the rail) keeps it small.

### Open

- Whether Calamba - San Pablo and San Pablo - Lucena are each still several trips a day (the
  wiki lists them operational; no timetable read).
- PNR's track in Metro Manila: how much OSM still maps as rail (NSCR works).

## Build (2026-10-08)

    python tools/slot.py 2 -- python extract.py --region ph --pbf data/raw/philippines-latest.osm.pbf --bbox 120.5,12.9,124.0,15.5 --station-areas
    python asia_register.py --clip ph             # also drops the NOT_SERVICE routes
    python asia_register.py --convert ph
    python build_model.py --region ph --register asia_register:data/raw/rinf/ph
    python build_tiles.py --region ph; python check_model.py --region ph

Reader: `asia_register.py` (lk_register.py's engine), list `PH` in `asia_lines.py`; settings
`rinf_countries/ph.py`, rules `rules/ph.py` (PNR's route=train relations are named trains),
colours `colours/ph.csv` (picked).

| line | km built | status | check |
|---|---|---|---|
| South Main Line (Calamba - Lucena) | 75.6 | running (Inter-Provincial Commuter) | en.WP 77: 0.98 |
| South Main Line (Lupi - Naga) | 47.1 | running (Bicol Commuter) | none published |
| South Main Line (Naga - Legazpi) | 101.1 | greyed | ~100: 1.01 |
| South Main Line (Lucena - Lupi) | 197.3 | greyed | none |
| LRT Line 1 (OSM) | 24.6 | running | 26 / 25 stations: 0.95, all 25 stops |
| LRT Line 2 (OSM) | 16.6 | running | 17.6 / 13: 0.94, all 13 |
| MRT Line 3 (OSM) | 16.4 | running | 16.9 / 13: 0.97, all 13 |

Register 421 km: 123 running, 298 greyed. OSM metro lines 58 km.

Decisions:
- **PNR's line unit** is the South Main Line cut into four, by what runs: rinf.py greys a
  whole line or none.
- **Naga - Legazpi greyed**: suspended since Typhoon Uwan (10 Nov 2025), still in July 2026.
- **Lucena - Lupi greyed**: no regular train since the Bicol Express ended (2014), but it had
  trains and the track is mapped, so greyed (the "greyed if it had trains" rule). "Lupi" is
  OSM's Lupi (Lupi Viejo) station, where the Bicol Commuter now ends (since 5 Nov 2025).
- **Tutuban - Calamba and Tutuban - Governor Pascual are not built**: closed in March 2024 for
  the NSCR works, and OSM has no rail track left in Metro Manila to draw them on (every
  section traced nothing). They come back as NSCR lines when it opens.
- **NOT_SERVICE**: the closed PNR Metro North and Shuttle routes, and the Metro Manila Subway
  (mapped as route=train, under construction).
- Stops: `osm_stops: "all"`; OSM has the Inter-Provincial Commuter's stops except Lutucan and
  Sariaya, and the Bicol Commuter's.
