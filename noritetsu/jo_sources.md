# Jordan sources

## Survey (2026-10-08)

Research only; nothing built. Region code `jo`.

### What runs

Only the **Jordan Hejaz Railway (JHR)**, 1,050 mm gauge, out of Amman's Mahatta station.
Its passenger trains are public excursions, not transport, but they are scheduled and
bookable, three to four trips a week:

- **Amman - Al Jeezah** (south along the old Hejaz main line, about 35 km): Fridays and
  Saturdays, booked by phone (Jordan News, "Hejaz Railway Resumes Trips to Al-Jeezah
  Station", undated, 2025-26; a 2026 search summary says "three to four tourist trips per
  week"). Pauses one to two months in winter.
- **Amman - Mafraq** (north, about 70 km): named in older reports as a regular excursion
  ("three weekly trips from Amman to Mafraq and Amman to Al-Jeezah"); not confirmed for
  2026.
- Not passenger: the Aqaba Railway (phosphate freight, Wadi Rum's heritage train is
  charter only); the Amman - Damascus link (talks only, "could be ready by end 2026",
  Arab News); the Amman - Zarqa - airport line (a plan).

Decision: Amman - Al Jeezah is a weekly scheduled public train, so Jordan qualifies, just.
It is the same kind of service as Sabah's North Borneo heritage train, which Malaysia
counts only because the line has a daily ordinary train as well; here there is no ordinary
train at all. Built as one register line, Amman - Al Jeezah; Amman - Mafraq only if a 2026
timetable turns up.

### Line list and stations

By hand (nafrica's recipe): Amman (Mahatta) - Qasr - Al Jeezah, stations from OSM. Hejaz
station names and km are in Wikipedia's "Hejaz railway" station list (Amman km 222 from
Damascus, Al Jeezah further south). Expected: 1 line, about 35 km, 3-4 stations.

### OSM

Overpass result in `data/raw/jo/survey/osm_routes.json` (2026-10-08). Geofabrik
`asia/jordan-latest.osm.pbf`, 29.6 MB.

### Timetables

None published; JHR books by phone (0799053016, per Jordan News). No GTFS.

### Licence

OSM (ODbL).

### Open

- Whether a tourist excursion counts at all is borderline. If Anita would rather not have
  Jordan on the map for one excursion line, leave it out; nothing else depends on it.
- Amman - Mafraq's 2026 status.

## Build (2026-10-08)

The excursion counts (managing session's decision).

    python tools/slot.py 2 -- python extract.py --region jo --pbf data/raw/jordan-latest.osm.pbf --station-areas
    python mideast_register.py --clip jo
    python mideast_register.py --fill jo
    python tools/slot.py 2 -- python build_model.py --region jo --register mideast_register:data/raw/rinf/jo

**1 line, running: Hejaz Railway: Amman – Al Jeezah, 37.3 km**, against en.WP's Hejaz
railway chainage Amman km 222.4 - Al-Jizah km 259.7 = 37.3 (1.00). Two stops (`listed_only`:
the excursion runs through Qasr). OSM has no Al Jeezah station; `--fill` named the unnamed
station node 35 m from the yard at 35.9625,31.7120. Amman - Mafraq not built (no 2026
timetable). Colour picked.
