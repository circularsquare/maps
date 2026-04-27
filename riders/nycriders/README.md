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
# 1. Download O-D data from Socrata (or use curl, see below). Outputs data/od_wednesday_oct.csv
python fetch_data.py

# Alternatively, one-shot curl (faster):
curl -o data/od_wednesday_oct.csv 'https://data.ny.gov/resource/jsu2-fbtj.csv?$select=hour_of_day,origin_station_complex_id,origin_station_complex_name,origin_latitude,origin_longitude,destination_station_complex_id,destination_station_complex_name,destination_latitude,destination_longitude,estimated_average_ridership&$where=month=10%20AND%20day_of_week=%27Wednesday%27&$limit=2000000&$order=hour_of_day,estimated_average_ridership%20DESC'

# 2. Route riders via RAPTOR and assign to train runs. Outputs trains.json
python build.py

# 3. Serve and open
python -m http.server 8000

# 4. copy to website
cp /c/Users/anita/projects/maps/riders/nycriders/index.html /c/Users/anita/projects/website/nycriders/index.html

```

## Data sources

| File | Source | Notes |
|------|--------|-------|
| `data/od_wednesday_oct.csv` | [MTA O-D Ridership 2024](https://data.ny.gov/Transportation/MTA-Subway-Origin-Destination-Ridership-Estimate-2/jsu2-fbtj) | Wednesdays in October, all pairs. ~1.6M rows, ~180 MB |
| `data/gtfs_subway/` | [MTA GTFS Static](http://web.mta.info/developers/data/nyct/subway/google_transit.zip) | Weekday schedule used. ~20k trips |
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