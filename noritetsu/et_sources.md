# Ethiopia (et): sources

## Survey (2026-10-08)

Research only, nothing built. By the eafrica survey agent. Djibouti's end of the same railway
is in `dj_sources.md`; the two should be built together (one line crossing at Dewele/Guelile).

### What runs (freshest evidence)

| service | status | evidence |
|---|---|---|
| Addis Ababa (Furi-Lebu) - Dire Dawa - Djibouti (Nagad), standard gauge, electrified | running: one train every second day each way, in two legs that need an overnight change at Dire Dawa (101/102 Addis - Dire Dawa overnight, K1/K2 Dire Dawa - Nagad by day) | seat61.com/Ethiopia.htm (updated 19 Nov 2025); ethiopiarailway.com/train-schedule (unofficial, "updated July 2026") gives the same pattern |
| Addis Ababa Light Rail, East-West (Ayat - Tor Hailoch, 22 stops) and North-South (Kality - Menelik II Square, 21 stops) | running, every 20 min 05:00-22:00 in the 2023 GTFS | Addis Ababa GTFS 2023 (Mobility Database tld-6782, active) |
| Awash - Weldiya (Hara Gebeya), standard gauge, 272 km of named way in OSM | **not built**: no passenger service has started (construction finished in stages, stalled by the northern war; nothing found announcing service) | |
| Old metre-gauge Dire Dawa - Dewele | closed | |

### Line list

1. **Addis Ababa - Djibouti railway, Ethiopian part**: Sebeta (km 2.0), Furi-Lebu (15.5), Bishoftu (67.3), Modjo (91.3), Adama (113.7), Wolenchiti/Feto (154.3), Metehara (217.7), Awash (248.0), Sirba Kunkur (280.2), Mieso (323.7), Bike (391.1), Dire Dawa (461.5), Aysha (622.1), Dewele (663.1), then the border. Passenger stations per en.wikipedia "Addis Ababa–Djibouti Railway" (Indode, Arawa, Adigala are freight or passing loops). The km are the line's own chainage from Sebeta, usable as `km_official`/`chain` like nz's. Whether the passenger train serves Sebeta or starts at Furi-Lebu: Furi-Lebu is the timetabled terminus; Sebeta - Furi-Lebu (13.5 km) greyed or left off.
2. **Addis LRT East-West** and **North-South**: OSM lines (route relations 5696982/3/5/6 with stops), or register lines from the GTFS stop lists; the two share Stadium - St. Lideta. OSM has 71 km of light_rail track (both directions); route length ~17 + ~17 km.

Expected: 1 register line ~650 km (plus Djibouti's ~95), 2 LRT lines ~34 km.

### Sources

- Station order and km: en.wikipedia "Addis Ababa–Djibouti Railway" station table (from the operator EDR's chainage).
- Coordinates: OSM, 88 railway=station/halt, 76 named.
- Timetable: seat61; ethiopiarailway.com (unofficial); EDR publishes no feed.
- GTFS for the LRT: `data/raw/et/survey/addis_gtfs_2023.zip` (2.7 MB, gitlab.com/digitaltransport, Mobility Database tld-6782). Stop coordinates are OSM ways' centroids; stop names English, one Amharic (Hayahulet 1).
- Licence: the digitaltransport Addis feed is published openly on GitLab (check its LICENSE before shipping; OSM-derived stop ids suggest ODbL); OSM ODbL.

### OSM quality (Overpass, 2026-10-08)

- 1,232 km of track, 1,033 named (84%): "Addis Ababa – Djibouti Railway" 688 km, "Awash–Weldiya Railway" 272 km, the LRT lines. Named track would work (th/vn recipe), but the line is one piece, so a hand list traced by rinf.py is just as short and carries the chainage.
- Relations: train route 6281977 (Addis - Djibouti, operator tagged as China Railway Group), infra 2732604 and 10693686, Awash-Weldiya 10377666, Weldiya-Mekelle 12117604 (under construction), four LRT routes.
- Geofabrik: `africa/ethiopia-latest.osm.pbf`, 133 MB.

### Recipe

Hand list traced by rinf.py with the Wikipedia chainage (the `chain` check works), shared reader with the other eafrica countries. A border point at Dewele/Guelile is needed in `borders.EXTRA` (the one passenger crossing in the region besides TAZARA). LRT as OSM lines.

### Open questions

- Is the every-second-day pattern still running after mid-2026? Only the unofficial site's July 2026 update confirms it.
- Awash - Weldiya: watch for a passenger opening.

## Build (2026-10-08)

Built with `eafrica_register.py` (see ke_sources.md "Build" for the commands; Ethiopia needs
no `--fill`). **Result**: 1 register line, Furi-Labu - Dewele - border 651.3 km, running;
Addis Ababa's light rail as 2 OSM lines (East-West 16.8, North-South 16.4). check_model: the
line 651.3 of 651.6 (en.WP chainage Furi-Lebu 15.5 to Dewele 663.1, + 4.0 traced to the
border); LRT 0.97 each.

Decisions:
- Stops: OSM's train route's stops plus Metehara (listed_only); not Addis Ababa-Kality (Indode,
  the freight terminal) or the passing loops. Sebeta - Furi-Labu left off (no passenger train).
- Dewele's stop is the customs and immigration station, mapped twice under two names
  ("Dewele customs and imigration", "Gare des douanes ... de Dewele"): both renamed "Dewele"
  (eafrica_lines.STATION_NAMES) so they merge.
- The border point XDJET1 (42.642891, 11.090904): where OSM's track (ways 967769108,
  1197034121) crosses OSM's boundary (way 31304862, read from the OSM API's map call; Overpass
  answered 504/500 all day). The app's outline crosses the track 0.8 km short of it, so within
  3 km of the point the clip sides by OSM's boundary (eafrica_lines.BORDER_LINES). Ethiopia's
  extract had been deleted by then: the 6 ways between the two crossings were copied over from
  Djibouti's data/proc (their nodes were in Ethiopia's coords); a fresh extract and
  `--clip et` gives the same. Needs `eXDJET1` in borders.EXTRA (handoff_notes/eafrica_build.md).
- The LRT masters are named "AA-LRT : Ayat <-> Tor Hailoch", read by build_model as "AA-LRT"
  for both: renamed "Addis Ababa LRT East–West" / "North–South" in the clip.
- OSM's train route 6281977 is the register line again: rules/et.py SKIP_ROUTES.
- Awash - Weldiya not built (no passenger service has started).
