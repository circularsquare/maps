# NYC Subway Ridership Visualization

Animated map of NYC subway trains with rider counts derived from MTA origin-destination data.

## How it works

1. **O-D ridership data** (MTA, via Socrata API) gives estimated riders between every station pair by hour of day
2. **GTFS static schedules** give exact train departure/arrival times at every stop
3. **RAPTOR routing** finds the fastest journey for each O-D pair using actual timetables, including transfers between lines
4. **Rider assignment** boards riders onto the specific trains in their journey

The visualization shows each train as a dot moving along its route at real schedule times, sized by how many riders are on board.

## Pipeline

Run in order:

```bash
# 1. Download O-D data from Socrata, one file per day type. Outputs data/od_<day>_2026-05.csv
python fetch_data.py --year 2026 --month 5 --day wednesday
python fetch_data.py --year 2026 --month 5 --day saturday
python fetch_data.py --year 2026 --month 5 --day sunday

# 1c. Speed fixes to the timetable (writes data/gtfs_subway/stop_times_fixed.txt,
#     all three services at once)
python check_speeds.py --apply

# 2. Route riders via RAPTOR and assign to train runs. Outputs trains.json + stats.json
#    (--day saturday / sunday write trains-<day>.json + stats-<day>.json; ~8 min on 6 workers)
python build.py
python build.py --day saturday --workers 6
python build.py --day sunday --workers 6

# 3. Serve and open: the maps-wide server, http://localhost:8800/nycriders/
python ../../serve.py

# 4. copy to website
cp /c/Users/anita/projects/maps/riders/nycriders/index.html /c/Users/anita/projects/website/nycriders/index.html
# (and the trains*.json / stats*.json files when they change)

```

## Data sources

| File | Source | Notes |
|------|--------|-------|
| `data/od_{wednesday,saturday,sunday}_2026-05.csv` | [MTA O-D Ridership: Beginning 2026](https://data.ny.gov/d/28vm-gjqr) | May 2026, all pairs. ~1.3-1.6M rows, ~150 MB each. May was picked as the month whose weekends lose the fewest riders to closures across 2025-26 (September 2025, used before, had the 3 closed east of Franklin Av every weekend) |
| `data/gtfs_subway/` | [MTA GTFS Static](http://web.mta.info/developers/data/nyct/subway/google_transit.zip) | Weekday, Saturday and Sunday services (spring 2026 timetable). ~20k trips |
| `../../data/nystreets/MTA_Subway_Stations.csv` | MTA | 496 stations, maps GTFS Stop ID to Complex ID |

## Key config

- **build.py `SAMPLE_RATE`**: Set to `0.1` for testing, `1.0` for full data
- **build.py `TRANSFER_TIME`**: 180s default transfer penalty between routes
- **build.py `MAX_ROUNDS`**: 3 (supports up to 2 transfers per journey)
- **build.py `DEP_BIN`**: 300s (5-min bins for RAPTOR query caching)
- **index.html `SIM_SPEED`**: 60 (1 real second = 1 simulated minute)

## Known limitations

- **Straight-line geometry**: trains move in straight lines between stops (no curved track geometry)
- **5-minute departure binning**: riders within the same 5-min window share the same RAPTOR result



# anita's human todolist

copy is
cp /c/Users/anita/projects/maps/riders/nycriders/index.html /c/Users/anita/projects/website/nycriders/index.html


oh also, what are the largest individual rows of data? like lets say theres a single point where like 1600 people spawn at grand central at 8am or something. we should probably break that up instead of spawning them all at 8:16 or something and overcrowding some arbitrary train. so that we instead spawn them at like 2 random times.