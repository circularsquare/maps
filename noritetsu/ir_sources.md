# Iran sources (surveyed 2026-10-03)

Region code `ir`. Iran is in no multi-country register (not RINF, not NARN). The register is
RAI's railways as OpenStreetMap maps them, one `route=railway` relation per railway, through
kr_register's recipe the way th_register does it; passenger stops and served track come from
RAI's timetable. `ir_register.py`'s docstring is the design.

## Commands

```
# the extract (managing session): Geofabrik asia/iran-latest.osm.pbf, 0.23 GB
python extract.py --region ir --pbf data/raw/iran-latest.osm.pbf
python ir_register.py --construction data/raw/iran-latest.osm.pbf  # construction way ids (needs the .pbf)
python ir_register.py --clip          # after every extract: out abroad, non-services, construction
python ir_register.py --crawl         # iranrail.net's trains and stations (~45 min, 0.8 s apart)
python ir_register.py --parse         # data/raw/ir/iranrail/*.html -> data/raw/ir/timetable.json
python ir_register.py --timetable     # -> data/raw/gtfs/ir/ir_iranrail.gtfs.zip (needs data/proc/ir)
python ir_register.py --report [png [bbox]]   # way-to-line assignment, km per line, a plot
python build_model.py --region ir --register ir_register:data/raw/ir     # ~45 s
python build_tiles.py --region ir                                          # ~15 s
python check_model.py --region ir
```

`--clip` rewrites `data/proc/ir` and is safe to run twice; `construction.json` is kept there, so
only a new extract needs the .pbf.

## The build (2026-10-03)

- **26 register lines, 9,414 km** (check_model counts 9,398 after ownership), 196 register
  stations; 57 lines in all (31 OSM lines: 14 metro and light-rail lines in Tehran, Mashhad,
  Shiraz, Isfahan, Tabriz and Karaj, 12 RAI suburban and local trains, Tehran Metro Line 5;
  4 named trains). 8,489 register km served by the timetable.
- **Greyed** (not running): Sufian - Razi, 189.8 km (SUSPENDED: the Tehran - Van train is
  suspended since 2026-03-06, below); Fariman - Shahid Motahari, 6.3 km (no train: Mashhad -
  Sarakhs trains leave the Mashhad line at Salam).
- **Check** (`check_model.REGISTER["ir"]`, all 23 entries within 0.95-1.02): Tehran - Mashhad
  922.8 of 923.0, Tehran - Tabriz 734.5 of 735.9 (Wikidata), Tabriz - Jolfa 145.0 of 146.1,
  Bafq - Bandar Abbas 610.7 of 612.1, Kerman - Zahedan 540.5 of 539.2, Badrud - Shiraz 737.9
  of 732.2, Mashhad - Bafq 777.1 of 777.0, New Mianeh - Tabriz 172.8 of 173.6, Garmsar -
  Incheh Borun 456.8 of 457.9, Qazvin - Rasht 163.6 of 164, Yazd - Eqlid 268.8 of 271,
  Hamedan - Sanandaj 150.4 of 151, Maragheh - Urmia 181.8 of 183, Trans-Iranian 921.1 of
  938.6 (0.98), Arak - Kermanshah 254.4 of en.wikipedia's 267 (0.95: Arak - the Shazand
  junction is the Trans-Iranian's). "RAI" figures are differences of RAI's station chainage
  (fa.wikipedia's station table, from RAI's realtime system), so they are independent of OSM.
- **Metros** (`KNOWN["ir"]`): Mashhad L1 23.7 of 24, Shiraz L1 22.5 of 22.5, Isfahan L1 20.2 of
  20.2, Tehran L6 31.2 of 32; Tehran L1-L5, L7 run 6-15% under en.wikipedia's line lengths
  (published lengths take in depot tails; station counts all match). Tabriz L1: OSM lists 18
  stops, en.wikipedia says 6 of them are not open yet (not checked further).
- **Borders: none built, no `borders.EXTRA` entry.** No passenger train crosses any of Iran's
  borders now: Razi - Kapıköy (Tehran - Van suspended since 2026-03-06; Türkiye greys Van -
  Kapıköy for the same reason; Iran's greyed line ends at Razi station, 2 km short of the
  border), Jolfa - Nakhchivan, Sarakhs and Incheh Borun - Turkmenistan, Shalamcheh - Iraq (not
  joined), Mirjaveh - Pakistan (below), Khaf - Herat (freight). A trial build of tr
  (`tools/ab.py tr`) after Iran's build changed nothing.

## The line unit

| candidate | what it is | verdict |
|---|---|---|
| **OSM `route=railway` relations** (chosen) | 52 in the extract, one per railway as Wikidata and en.wikipedia name them (Wikidata's P402 points at them: Tehran - Tabriz 13974266, Garmsar - Mashhad 7369741, Badrud - Shiraz 8276443, Qazvin - Rasht 8276508...) | 88% of main-line track in a relation; with way names 91% |
| OSM way names | "راه آهن تهران - مشهد", "درود - محمدیه", "اهواز - اندیمشک" | 49% of main-line km (`probe_kr_ways.py`); used for ways outside relations and to split a relation holding two railways |
| RAI's districts (نواحی) | 22 operating districts (Tehran, Shomal 1, Zagros...) | operating areas, not lines |
| RAI's own line list | rai.ir and raja.ir answer 403 from outside Iran | not reachable |
| Wikidata | 68 railway-line items for Iran, a few with P2043 lengths, adjacency chains on 34 | glue and checks only |

Lines, with what OSM has for each (LINES, REL_LINE, NAME_LINE in ir_register.py):

- Tehran - Mashhad (Tehran - Pishva in the Transiranian relation, Pishva - Garmsar unnamed,
  Garmsar - Mashhad 7369741; the track names say Tehran - Mashhad throughout).
- Garmsar - Incheh Borun: the Trans-Iranian north of Garmsar, Gorgan, Incheh Borun (7286485,
  6646450), with a fork junction where the Gorgan spur leaves (BRANCHES).
- Trans-Iranian (Tehran - Bandar Imam Khomeini): Tehran - Eslamshahr - Robat Karim - Parand -
  Parandak - Qom - Arak - Dorud - Andimeshk - Ahvaz - Mahshahr. Tehran - Nasirshahr is one
  double-track corridor that OSM files one track under each of two relations: it is the
  Trans-Iranian's (`split`).
- Tehran - Qom (by Imam Khomeini Airport): Nasirshahr - Qomrud - Mohammadieh - Jamkaran.
- Qom - Kerman: Mohammadieh - Kashan - Badrud - Nain - Ardakan - Meybod - Yazd - Bafq - Zarand -
  Kerman; OSM has it in three relations' pieces (Qom - Meybod, the Meybod - Bafq part of "Bafq -
  Isfahan", the Bafq - Kerman part of "Bafq - Zahedan"), split by place. The curve from
  Mohammadieh onto it (no name, no relation) is WAY_LINE.
- Isfahan - Ardakan (Sistan - Varzaneh - Meybod), Badrud - Shiraz (by Isfahan; Sistan - Isfahan
  is in both relations and given to Badrud - Shiraz), Kerman - Zahedan, Bafq - Bandar Abbas,
  Mashhad - Bafq (two relations), Torbat-e Heydarieh - Khaf (no relation, unnamed: BOX_LINE),
  Yazd - Eqlid, Fariman - Sarakhs, Tehran - Tabriz (by Qazvin, Zanjan, Mianeh, Maragheh), New
  Mianeh - Tabriz (by Bostanabad), Tabriz - Jolfa, Sufian - Razi, Maragheh - Urmia, Mianeh -
  Ardabil (with OSM's "Mianeh bypass" curve, by which Ardabil trains leave Mianeh), Qazvin -
  Rasht (with Rasht - Caspian port), Tehran - Hamedan (from Parand), Hamedan - Sanandaj, Arak
  - Kermanshah, Ahvaz - Khorramshahr, Chabahar - Zahedan (open Zahedan - Khash), and the 10 km
  spur to Azarbaijan Shahid Madani University (RAI's Tabriz commuter trains).
- Left out (track stays drawn): Ardakan - Chadormalu (iron ore), the Tehran, Qom and Ahvaz
  bypasses, Rostamkola - Amirabad port, Khorramshahr - Shalamcheh, Khaf - Herat, Haft Tappeh -
  Shushtar, Zahedan - Mirjaveh (Pakistan Railways' broad gauge), industrial branches (فرعی),
  the Garmanuri - Qomrud link at Qom (the Tehran - Qom commuter route's OSM line owns it), Tehran
  Metro Line 5's track. Lines OSM maps in relations with no ways (Dorud - Khorramabad, Shiraz -
  Bushehr, Mobarakeh - Shahrekord, Malayer - Hamedan, Bojnurd - Esfarayen) are being built.

The repairs, each logged by the build: unnamed track through stations taken from the one line
round it (`propagate`, `fill_runs`), a line's two dead ends within 4 km joined over other track
(`bridge_gaps`: Qom, Rasht, Gorgan), Mianeh - Ardabil's three OSM gaps joined straight
(STRAIGHT_GAPS: OSM still has pieces of the 2026 line as construction), junction stations at
line ends on another line (`junction_ends`), runs past a fork dropped (`drop_junction_runs`,
mx_register's rule), Tehran's two records ("ایستگاه راه آهن تهران", "تهران") and Parand's
three merged (Persian "ایستگاه راه آهن" taken off for matching; one English name within 300 m).

## Passenger stops and served track: RAI's timetable

- **iranrail.net** (the unofficial Iranian railways site, data from RAI's realtime system
  pws0.rai.ir; timetable "valid until 20-10-2026", pages last updated 2025-11-08) lists 312
  trains and connecting buses: `alltrains.php`, each train's `times.php` (calls with times) and
  `info.php` (operator, days), and each station's `location.php` (a point). Crawled 0.8 s apart
  into `data/raw/ir/iranrail/`, parsed to `data/raw/ir/timetable.json` (234 trains, 78 buses).
  Its certificate chain does not verify from here (read without verification).
- **Counted**: trains that run at least weekly (daily, every second day, every four days,
  named weekdays; "not every day" read as twice a week). Left out: the 78 buses, TCDD's trains
  in Türkiye, and Tehran - Van (NOT_RUNNING). RAI's 40 local trains show no calls on iranrail
  and count as calling at their two ends. Added (EXTRA_TRAINS): Raja's Tehran - Ardabil 492/493,
  weekly since Shahrivar 1405 (late August 2026), calling at Karaj, Qazvin, Zanjan, Mianeh
  (sanatmali.ir 1405/06/02; sharghdaily.com).
- **Matching** a call to OSM (`timetable_stops`): the OSM rail station near iranrail's point
  whose English name matches (within 60 km, since some iranrail points are another station's:
  its "Saveh" is Shohada-ye Parandak, 44 km off), else the nearest within 1.2 km; TT_ALIAS for
  six names. 155 of 159 call places matched; no OSM station exists for Ahmadabad (Bafq -
  Bandar Abbas), Charizeh (Badrud - Isfahan), Robat-e Posht-e Badam (Bafq - Mashhad) and
  Chamanabad (Torbat - Khaf): those calls are stepped over.
- **Passenger stations** are those calls plus the stops of OSM's passenger routes (the Tehran,
  Tabriz and Mashhad commuter routes): 152 + 43. 312 OSM rail stations are crossing loops with
  no passenger stop and are left out. Gaps: stations only RAI's local trains call at (the
  Zagros halts between Andimeshk and Dorud) are missing, since iranrail shows no calls for
  locals: Sepiddasht - Andimeshk is one 166 km section.
- **Served track**: `--timetable` writes the counted trains as a GTFS feed,
  `data/raw/gtfs/ir/ir_iranrail.gtfs.zip`, which gtfs_served reads like every national feed (no
  FEEDS entry: it is generated here, as Ukraine's). gtfs_served's "weak" rule (a junction-ended
  section only on runs over 40 km between calls) would drop Qazvin - Rasht, Yazd - Eqlid, Arak -
  Malayer and others whole, since 100-300 km between calls is normal here; such a section is
  kept (`served_sections`) when a train calls, one after the other, at a station of the line
  and at a station of the line that meets it at the junction (`served_junction_sections`).
- **Not checked**: whether every timetabled train runs after the war of February - April 2026.
  Iran reported trains running again on the damaged routes on 2026-04-13 (Tabriz - Tehran,
  Tabriz - Mashhad, the Qom bridge); the Tehran - Van train stays suspended.

## Lines vs named trains (`rules/ir.py`)

- **Lines**: RAI's suburban and local trains (operator RAI, service=commuter): Tehran - Parand,
  Garmsar, Pishva (Emamzadeh), Firuzkuh, Hashtgerd, Qom; Tabriz - Jolfa, Tabriz - Shahid Madani
  University, Mashhad - Sarakhs, Zahedan - Khash, the Ahvaz railbus; Tehran Metro Line 5 (route
  =train); every metro and light-rail line. Each is the regional service of its corridor at
  fixed stops, from one pair a day (Tehran - Firuzkuh) to a dozen (Tehran - Parand).
- **Named trains**: the long-distance trains of RAI's passenger companies (Raja, Fadak,
  Bonrail, Noor al Reza, Rail Seir Kosar, Saba, Joopar...): each is one overnight or day train
  between two cities, daily or every second day, sold by its company as its own train (19 such
  trains a day each way between Tehran and Mashhad alone, each its own product). OSM has four
  (Raja's and Bonrail's Hamadan - Mashhad, Raja's Shiraz - Tehran, Pakistan Railways' Zahedan
  Mixed); their track counts through RAI's railways.

## Who the map depicts, and decisions recorded here

- Zahedan - Mirjaveh - Taftan (Pakistan Railways, broad gauge): the Zahedan Mixed runs about
  twice a month, under the once-a-week bar: left out of the register (track drawn, OSM's route
  a named train).
- Tehran - Van and Sufian - Razi: greyed (suspended), not dropped, as Türkiye's side is.
- Mianeh - Ardabil: built as running from the weekly Tehran - Ardabil train (late August 2026).

## Sources

| file (data/raw/ir) | what | from |
|---|---|---|
| `iranrail_alltrains.html/.json`, `iranrail/*.html`, `timetable.json` | RAI's timetable (iranrail.net copy) | `https://www.iranrail.net/alltrains.php`, `times.php`, `info.php` |
| `iranrail_stations.html/.json` | iranrail.net's 632 stations with points | `stations.php`, `location.php?id=` |
| `wd_lines.json`, `wd_adjacency.json`, `wd_stations.json` | Wikidata railway lines, station adjacency, stations in Iran | query.wikidata.org |
| `wp/*.wikitext` | en.wikipedia "Rail transport in Iran" (line table with lengths), the metro articles; fa.wikipedia "فهرست ایستگاه‌های راه‌آهن ایران" (RAI's station table with km) | wikipedia action=raw |
| `osm_route_tags.json` | Overpass: tags of Iran's rail route relations (survey only) | overpass-api.de |
| `ir_boundary.geojson` | OSM relation 304938 (Iran) | polygons.openstreetmap.fr |
| `../gtfs/ir/ir_iranrail.gtfs.zip` | the timetable as GTFS (generated) | `ir_register.py --timetable` |

Not reachable: rai.ir, raja.ir (403 outside Iran). The Mobility Database lists no Iranian feed.

## Open

- iranrail's copy dates from November 2025; recrawl (`--crawl`, `--parse`, `--timetable`) to
  pick up timetable changes (and confirm Tehran - Ardabil's frequency).
- RAI's local trains' intermediate stops (Andimeshk - Dorud's Zagros halts, Tehran - Garmsar's,
  Gorgan - Pol-e Sefid's) are only where OSM's commuter routes list them.
- Tehran - Qom: two lines (by Parandak, the Trans-Iranian; by Imam Khomeini Airport) and which
  one RAI's Tehran - Qom trains take is the timetable path's guess; both are drawn as running.
- No `colours/ir.csv` (OSM's colours on the metro routes are used).
- Tehran Metro L4's Mehrabad airport branch has no stops in the built line (22 of 25 stations).
