# Israel sources

## Survey (2026-10-08)

Research only; nothing built. Region code `il`. The questions of who and what the map shows
(East Jerusalem, the high-speed line's West Bank stretch) are at the end, for Anita.

### What runs

**Israel Railways** (רכבת ישראל), standard gauge, about 1,500 km of route (en.wikipedia
"Israel Railways": 1,541 km; "Rail transport in Israel": 1,138 km of IR route in an older
count), now about 70 stations. The MOT feed (below) has 1,068 IR route variants with 88
terminus pairs and 57 termini for the current period. What its termini show:

- Coastal corridor Nahariya - Haifa - Tel Aviv - Ashkelon, with Binyamina, Netanya,
  Herzliya, Rehovot, Rishon LeZion Moshe Dayan, Ashdod Ad Halom turnbacks.
- Tel Aviv - Jerusalem Yitzhak Navon (the electrified A1 high-speed line), Herzliya -
  Jerusalem the main pattern; Modi'in Center branch (Modi'in - Nahariya, Modi'in -
  Jerusalem, Tel Aviv - Modi'in).
- Ben Gurion Airport (נתב"ג) calls.
- Lod - Rishonim (Rishon LeZion) shuttle; Lod - Ramla - Beit Shemesh (Netanya - Beit
  Shemesh, Tel Aviv - Beit Shemesh).
- Beersheba: via Lod and Kiryat Gat (Tel Aviv / Herzliya / Nahariya / Karmiel / Haifa -
  Beersheba Center), and via Ashkelon - Sderot - Netivot - Ofakim (Herzliya - Ofakim);
  Beersheba North - Dimona.
- Jezreel Valley: Atlit / Haifa - Afula - Beit She'an.
- Acre - Karmiel (Karmiel - Carmel Beach, Karmiel - HaMifrats).
- The Sharon line: Rosh HaAyin North - Kfar Saba - Hod HaSharon - Ra'anana South -
  Ra'anana West (Herzliya).
- **The Eastern Railway**, first phase opened June 2026: Hadera East - Shomron-Tayibe -
  Tira-Kochav Yair - Rosh HaAyin North, two trains an hour Sunday to Thursday (Globes,
  "Israel Railways opens eastern line"; Yeshiva World, June 2026). It is in the feed
  (Rosh HaAyin North - Hadera East, 28 variants).
- **Not running**: Beit Shemesh - Jerusalem Biblical Zoo - Jerusalem Malha (the old
  Jaffa - Jerusalem line's mountain section). Its stops are still in stops.txt (17078,
  17076) but no route ends there or past Beit Shemesh; en.wikipedia says closed from March
  2020. Built as track, greyed (`suspended`), or left out: the same call as the Skypark
  branch in Malaysia. Under construction, not built: Beersheba - Arad, the Eastern
  Railway's southern half (Rosh HaAyin - Lod), Eilat.

**Urban rail**:

| system | in the feed as | status |
|---|---|---|
| Tel Aviv Light Rail **Red Line** (Petah Tikva - Bat Yam, 24 km, 34 stops; opened Aug 2023) | agency 22 תבל (Tavel), route_type 0, short names 1, 2, 3 (three service patterns: Petah Tikva CBS / Kiryat Arye - Bat Yam HaKomemiyut, Kiryat Arye - Elifelet) | running. Purple and Green lines under construction (OSM: `construction`, "הקו הסגול", "הקו הירוק") |
| Jerusalem Light Rail **Red Line** (Hadassah Ein Kerem - Neve Yaakov North, extended 2024-25) | agency 21 כפיר (CFIR), route 1 (plus a Givat HaMivtar - Hadassah short working) | running |
| Jerusalem Light Rail **Green Line**, first section Malha (מנחת) - HaTurim, 12 stops | agency 21, route 3 | opened 21 August 2026 (Globes, Railway News); the rest (to Mount Scopus and Gilo) still building. OSM has 4 light_rail ways named "הקו הירוק" and 29 still `construction`: OSM lags, check before a build |
| Carmelit, Haifa (underground funicular, 6 stops, 1.8 km) | agency 20, route_type 5 (should be 7) | running |
| Haifa cable car (Rakavlit, HaMifrats - Technion - University) | agency 33, route_type 5 | an aerial lift: not rail, leave out |
| Haifa - Nazareth "Nofit" light rail | not in the feed | under construction (OSM `construction`); leave out |

### Sources

1. **MOT GTFS** (the National Public Transport Authority's single national feed, all
   operators): `https://gtfs.mot.gov.il/gtfsfiles/israel-public-transportation.zip`,
   **144.7 MB zipped, 835 MB unzipped** (stop_times.txt 609 MB, shapes.txt 225 MB). Daily.
   The host answers a non-browser User-Agent with a 3 KB HTML page instead of the zip
   (Transitous fetches it with a browser UA). **Use the Mobility Database's mirror
   instead**: `https://files.mobilitydatabase.org/mdb-2519/latest.zip` (same file, updated
   daily, 2026-10-08 01:22 UTC; supports HTTP Range). Licence: the MOT's Hebrew terms of use
   at `https://www.gov.il/he/pages/gtfs_general_transit_feed_specifications` (no standard
   licence; OSM Israel has used it with the ministry's permission; I have not read the
   terms' text, so attribution-only use is my assumption).
   - Sampled without downloading the whole file (HTTP Range on the zip's members):
     `data/raw/il/survey/il_mot_rail_routes.csv` (the 1,083 non-bus routes: IR type 2, light
     rail type 0, Carmelit and cable car type 5, plus a few demand-responsive 715s and
     share taxis) and `il_mot_rail_stops.csv` (stops with codes 17000-17999: 88, most IR
     stations with English names from translations.txt). **Not every IR station has a 17xxx
     code**: the Eastern Railway's Hadera East is stop 51798, code 2653. A reader should
     take IR stations as the stops IR trips call at (stop_times), not by code.
   - English names: translations.txt (lang EN), e.g. "Be'er Sheba - Center", "Hertsliya",
     "Yerushalayim/Yits'hak Navon": transliterations of uneven quality; OSM's `name:en` may
     be better (check per station).
   - Each route variant is one stopping pattern, so stop sequences for the line reader come
     from stop_times (the full download).
2. **OSM** (Overpass, 2026-10-08, bbox, `data/raw/il/survey/osm_routes.json`; the bbox
   reaches into Lebanon and Jordan, so some relations are theirs):
   - Track: 1,101 rail ways (not service), about 450 of them named. Named lines: מסילת החוף
     (Coastal), קו אשקלון–באר־שבע (Ashkelon - Beersheba), מסילת העמק (Jezreel Valley), המסילה
     המזרחית (Eastern), מסילת הנגב (Negev, to Dimona), מסילת השפלה (Shfela), מסילת כרמיאל
     (Karmiel), קו הרכבת המהיר לירושלים - A1 (but only 6 ways; the A1's bridges and tunnels
     carry their own names, "גשר 5", "מנהרה 3"), קשת מודיעין (Modi'in curve). 501 main-line
     ways are unnamed. So **Korea's named-track recipe does not work alone** (run
     `probe_kr_ways.py --region il` after the extract to measure).
   - Route relations are stale: 22 train relations, numbered by IR's old line scheme (0-9),
     one still Tel Aviv - Jerusalem Malha, none for the Eastern Railway's service or the
     Sharon line. Do not use them for stops. Light rail: Tel Aviv Red R1-R3, Jerusalem Red
     (L1, three relations) and a "Yellow Line" L3 relation (probably a short working;
     check), no Green Line relation yet.
3. **Wikipedia** for published lengths: en/he.wikipedia articles per line (Coastal railway,
   Tel Aviv - Jerusalem railway 56.6 km from Ben Gurion, Jezreel Valley railway 60 km Haifa -
   Beit She'an, Acre - Karmiel 23 km, Eastern Railway, Ashkelon - Beersheba railway).

### Recipe

**Malaysia's (`my_register.py`): GTFS stop lists laid on OSM track by shortest path.**

- Register lines are IR's **infrastructure lines** as he.wikipedia names them (the OSM
  track names above), each a hand list of waypoint stations (`LINES`, as in
  my_register): Coastal (Nahariya - Haifa - Tel Aviv - Ashkelon), Ayalon (Tel Aviv
  through-line, part of Coastal or its own), Tel Aviv - Jerusalem (A1, from Ben Gurion
  junction), Modi'in branch, Lod - Beersheba (via Kiryat Gat), Ashkelon - Beersheba (via
  Sderot, Netivot, Ofakim), Beersheba - Dimona, Lod - Beit Shemesh (the running part of the
  old Jerusalem line; Beit Shemesh - Malha suspended), Lod - Rishon LeZion (Shfela) /
  Rehovot, Jezreel Valley, Acre - Karmiel, Sharon (Kfar Saba - Ra'anana - Herzliya), the
  Eastern Railway (Hadera East - Rosh HaAyin North), Ashdod / airport spurs as needed.
  Expected: about 15 register lines, about 1,200-1,300 km, about 70 stations.
- Stations: every stop IR trips call at in the MOT feed (stop_times), matched to OSM by
  name within a few hundred metres; GTFS coordinates as the fallback.
- IR's service patterns (Nahariya - Beersheba and so on) are not lines: IR renumbers them
  often and the old ones in OSM are stale. Leave OSM's train relations as OSM lines or drop
  them in `--clip`.
- Light rail: Tel Aviv Red, Jerusalem Red and Green (first section), Carmelit as register
  lines from the same feed (route_type 0, 5), sections the shortest path over light_rail
  track (OSM's Jerusalem Green may still be tagged `construction`: check the extract).
- `gtfs_served` could be enabled later; the feed is complete for IR.

### Geofabrik

`asia/israel-and-palestine-latest.osm.pbf`, 114 MB (Israel, the West Bank, Gaza and the
Golan in one file). Clip to whatever outline Anita decides (below).

### Downloads needed (Anita's say-so)

- The MOT feed, 145 MB, from the Mobility Database mirror (no browser needed).
- The Geofabrik extract, 114 MB.

### Questions for Anita (who and what the map shows)

Answered 2026-10-08: "if israel administers it we can draw it under israel". Questions 1 and 2
are drawn whole in `il`; 3 is checked below ("Build", the outline); 4 is left as is (Hebrew
`name`, English `name_en`).

1. **Jerusalem's light rail in East Jerusalem.** The Red Line's northern half runs through
   areas Israel annexed in 1967 and most states treat as occupied (Shuafat, Beit Hanina,
   Pisgat Ze'ev, Neve Yaakov). The Green Line's planned extensions to Mount Scopus and Gilo
   will too; its open Malha - HaTurim section is in West Jerusalem. Draw the whole line in
   `il`, or cut it at the 1949 armistice line?
2. **The Tel Aviv - Jerusalem high-speed line** crosses into the West Bank for a few km near
   Mevo Horon / Latrun (tunnels and a bridge). Draw it whole in `il` (my default: it is one
   line with no stop there), or split the crossing out?
3. **The outline**: religiondots' `country_shapes.geojson` decides which country a section
   belongs to. Whatever it does with East Jerusalem and the Golan decides `foot.json`
   ownership and the country totals, so its `il` and `ps` shapes are worth a look before the
   build. No other rail in the West Bank, Gaza or the Golan.
4. **Station names**: Hebrew as `name`, English from the feed or OSM. Should Arabic names
   (OSM `name:ar`, on most IR stations) be shown anywhere? (Not needed for a build.)

## Build (2026-10-08)

### Run

```
python tools/slot.py 2 -- python extract.py --region il --pbf data/raw/israel-and-palestine-latest.osm.pbf --station-areas
python il_register.py --clip
python tools/slot.py 2 -- python build_model.py --region il --register il_register:data/raw/il
python tools/slot.py 1 -- python build_tiles.py --region il
python check_model.py --region il
```

The feed subset `data/raw/il/gtfs/il_rail.gtfs.zip` (0.56 MB: routes of type 2 and 0 and the
Carmelit, their trips, stop_times, stops, calendar and the stop-name translations) was cut from
the Mobility Database mirror `https://files.mobilitydatabase.org/mdb-2519/latest.zip`
(2026-10-08) and the 145 MB zip deleted. To refresh: download it again and re-cut with the
same filter (agency 20 for type 5; the cable car, agency 33, left out). The .pbf is deleted;
`data/proc/il` holds the clipped extract.

### What was built

22 register lines, 697 km, 174 stations (14 of them junctions), 172 sections. No OSM lines:
`--clip` drops every OSM route relation (below).

| line | km |
|---|---|
| Coastal Railway (Nahariya - Tel Aviv University) | 120.1 |
| Railway to Beersheba (Na'an jn - Be'er Sheva Center) | 75.9 |
| Ashkelon - Beersheba Railway | 60.9 |
| Jezreel Valley Railway | 57.0 |
| Tel Aviv - Jerusalem Railway (A1, Ganot - Navon) | 46.9 |
| Lod - Ashkelon Railway | 41.4 |
| Eastern Railway (Rosh HaAyin jn - Hadera East) | 36.3 |
| Beersheba - Dimona Railway | 34.8 |
| Jaffa - Jerusalem Railway (Lod - Beit Shemesh) | 30.4 |
| Bat Yam - Ashdod Railway | 26.5 |
| Acre - Karmiel Railway | 20.6 |
| Tel Aviv - Kfar Saba Railway | 19.6 |
| Tel Aviv - Lod Railway | 16.2 |
| Sharon Railway (Kfar Saba - Herzliya) | 11.5 |
| Anava - Modi'in Railway | 6.8 |
| Ayalon Railway | 5.9 |
| Rishonim Branch | 2.7 |
| Tel Aviv Light Rail Red Line | 24.1 |
| Jerusalem Light Rail Red Line | 19.9 |
| Jerusalem Light Rail Green Line | 6.5 |
| Carmelit | 1.8 |
| **greyed:** Beit Shemesh - Jerusalem Malha | 31.1 |

Running 666 km on 21 lines; greyed 31 km on 1. The app's country total (ownership) is
664.6 km. check_model: every line within its note's expected ratio (worst 0.83, Lod -
Ashkelon, where WP's "approximately 50 km" is loose: Lod - Ashkelon is 40 km crow-fly).
`tools/line_100_probe.js` (2026-10-08, via loadRegion before build_regions): 22/22 lines
pass, 18/18 unbranched lines whole in one pick, 9/9 country and operator groups exact.

### How the reader works (il_register.py)

- IR's lines are its infrastructure lines (en/he.wikipedia's names), each a shortest path over
  OSM's running track (railway=rail, not yard or siding, not industrial/military/tourism usage)
  through hand-listed waypoint stations. A line starting "^station" or ending "station$" is cut
  where its path leaves an earlier line's path (within 25 m, gaps under 400 m bridged) and a
  junction is made there, shared by both lines, so no track is in two lines. Where the cut is
  within 300 m of the station, the station itself is the junction (Lod, Beit Shemesh).
- IR's stations are the 70 stops IR's trips call at in the feed (stop_times), each at the
  nearest OSM railway station node within 700 m (all 70 found one), and put on every line
  whose path passes within 450 m (and within 100 m of the nearest), except next to a branch's
  cut end. HaMifrats Central's Jezreel Valley platforms (stop 17123; OSM's own node "HaMifrats
  Central Station - HaEmek Line") are on the Jezreel line only (`ONLY`): they are 400 m from
  the coastal platforms and would otherwise also be on the Coastal Railway.
- Light rail and the Carmelit: every stopping pattern in the feed (longest first), each pair of
  neighbouring stops a section, shortest path over the city's light_rail (or funicular) track.
  The Tel Aviv Red Line's Kiryat Arye branch (Aharonovich - Em HaMoshavot Bridge - Kiryat
  Arye) is only in its routes 2 and 3, so all three are read. Six stops have no OSM record of
  their name within 250 m and take the feed's point and name (JLR Central Station,
  HaDavidka; TLV Beilinson, Kiryat Arye; Carmelit Hadar City Hall, Bnei Zion hospital, the
  last with no English name in the feed).

### Decisions (made here, not put to Anita)

- **The lines.** IR's service patterns (Nahariya - Be'er Sheva, Herzliya - Jerusalem...) are
  not lines; the infrastructure lines are, as Malaysia's KTM lines are. Cuts: Ayalon is Tel
  Aviv University - HaHagana; Tel Aviv - Lod is HaHagana - Lod; the A1 starts at Ganot (WP:
  "branches off from the Tel Aviv–Lod railway at the Ganot Interchange") and includes the
  airport; the Modi'in branch leaves the A1 east of the airport (Anava); Railway to Beersheba
  leaves the Jaffa - Jerusalem line at Na'an and ends at Be'er Sheva Center (so Be'er Sheva
  North - Center is on it, and the Ashkelon and Dimona lines join it at junctions either side
  of Be'er Sheva North); the Kfar Saba line runs Tel Aviv University jn - Kfar Saba, the Sharon
  Railway Kfar Saba - Herzliya jn (the split at Kfar Saba Nordau, the old line's terminus, is
  my call); the Bat Yam line runs HaHagana jn - Ashdod jn via Rishon Moshe Dayan and Yavne
  West. Names follow the wikipedias where they have one; "Bat Yam - Ashdod Railway",
  "Rishonim Branch" and "Beersheba - Dimona Railway" are descriptive names of mine.
- **Junction names** are "<line> junction"; the real names (Ganot, Na'an, Anava) are noted
  here but not set in `JUNCTIONS`, since a junction point is where the OSM paths part, which
  can be a km from the named place.
- **Beit Shemesh - Jerusalem Malha is greyed** (`suspended`): no trains since March 2020 and IR
  "decided not to return and reactivate the line to Jerusalem as a commercial line" (en.WP
  "Jaffa–Jerusalem railway"); no trip in the feed. OSM tags most of it railway=disused, which
  the extract does not keep, so its ways come from Overpass (`data/raw/il/malha_disused.json`,
  19 ways: railway=disused with no service tag and railway=rail usage=branch, bbox
  31.70,34.98,31.78,35.20). Its stations are Biblical Zoo and Malha.
- **Beersheba - Dimona counts**: 16 trips in the feed's week, more than weekly.
- **Jerusalem's Green Line** first section (Malha - HaTurim, opened 2026-08-21) is the feed's
  route 3 of CFIR; OSM calls its relation "Yellow Line" (L3), which no source does: named
  Green. Its unopened track (Gilo, Mount Scopus) is partly tagged light_rail in OSM: drawn as
  track no line runs over (part of the tiles' 25 km of such light rail).
- **OSM route relations dropped (`--clip`)**: IR's 22 train relations use IR's old line numbers
  (one still Tel Aviv - Jerusalem Malha, none for the Eastern Railway or the Sharon line); the
  light-rail ones repeat the register lines; "Metronit" (route=tram) is Haifa's BRT bus. Also
  dropped as track: the Haifa - Nazareth Nofit light rail (80 km, being built but tagged
  light_rail, not in the feed), Tel Aviv's Orange Line stub (0.1 km), narrow gauge (10 ways)
  and monorail (2 ways), amusement and museum track. The Haifa cable car is an aerial lift and
  not in the build.
- **Colours**: none in the feed; the three light-rail lines take the colour of their name
  (`picked`, colours/il.csv). IR lines are uncoloured (the app's blue).

### The outline (who the map shows)

religiondots' `il` shape holds East Jerusalem, so the whole Jerusalem Red Line is inside it.
Three register lines have track inside its `ps` shape: the A1 6.6 km (Mevo Horon / Canada
Park), the Eastern Railway 7.2 km (it runs along the 1949 line near Tayibe and Kokhav Yair) and
the greyed Malha line 2.2 km (the 1949 line follows the railway at Battir). Each owns its ways,
and ownership.py gives a neighbour only ways no register line owns, so all three stay `il`
with no clip or outline change (handoff_notes/il_build.md, section 3).

### Check numbers (check_model.REGISTER["il"])

From en.wikipedia unless marked: A1 "about 56 km" (built 46.9 from Ganot; with the 6.8 km
Modi'in branch 53.7), Lod - Ashkelon "approximately 50", Ashkelon - Be'er Sheva "approximately
60" (its table: 70), the Lod - Be'er Sheva Center doubling project 87 (built 75.9 from Na'an
plus 10.6 Lod - Na'an on the Jaffa line), Jezreel 60 (he.WP), Acre - Karmiel 23 (he.WP; built
20.6 from its junction 2.5 km south of Acre), Beit Shemesh - Malha 36.3 (WP's station table,
the Ottoman alignment), JLR Red 22.5 (built stop to stop 19.9), JLR Green 7.0, TLV Red 24.0,
Carmelit 1.8. No published length found for the Coastal, Ayalon, Tel Aviv - Lod, Bat Yam,
Kfar Saba, Sharon, Eastern or Dimona lines; their sections are within about 1.35x crow-fly
(Lod Gane Aviv - Lod 2.9 for 2.1 round Lod's curve; Dimona's single 34.8 km section, 27.1
crow-fly).

### What is still off

- Ownership gives 0.95 km of the Sharon Railway's path near Herzliya to the Coastal Railway
  (same rails; rides credit through it).
- No English name for the Carmelit's Bnei Zion hospital stop (Hebrew only in the feed).
- The feed is a snapshot: when the Eastern Railway's southern half (Rosh HaAyin - Lod) or
  Beersheba - Arad opens, add its waypoints to `LINES` and re-cut the feed.
