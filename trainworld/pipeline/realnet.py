"""T-007, T-509: a city's real rail network as a demand test set, and the measured numbers to check it.

    python pipeline/realnet.py net [city]          # sources -> the network JSON (CITIES[city]["out"])
    python pipeline/realnet.py fetch-osm <city>    # OSM route relations in the pack's box -> data/raw/t509/
    python pipeline/realnet.py targets    # measured ridership and mode share -> targets.json
    python pipeline/realnet.py counties   # county of every pack cell -> cell_county.u8
    python pipeline/realnet.py geom       # track geometry from GTFS shapes, for the `realsave` bin
    python pipeline/realnet.py gamesave   # the `realsave` output as a game file (T-007-real-nyc.save)
    python pipeline/realnet.py ship       # the game files a sandbox starts from -> data/packs/real/<id>.save (T-731)

The converter (T-509): each city in CITIES lists its sources; an adapter turns each source into
stations and trips (route, direction, stops, departure times), and the rest is common. Adapters:
`gtfs` (a feed's weekday, a Wednesday with the most trips), `osm` (route relations from
`fetch-osm`: stop order from the relation, trains per hour from its `interval` tag or a default
by kind of line, run times from distance at a speed by kind), `trips` (a CSV of timetabled
trips any other source can be written to: route, direction, trip, stop, name, lon, lat, time),
`n02` (Japan's whole network from MLIT's N02, T-737: pipeline/realnet_n02.py; cities `jpn`, `jpn7`).
A source's `group` table puts its stops into the stations the city's counts use (New York: the
MTA's station complexes, so in-complex transfers are free of walking, as the MTA's own counts
treat them); each station keeps that group id as `target`. Stations of different sources within
`merge_m` are merged (Penn Station, Hoboken, Newark Penn), except sources marked `merge: False`.
Stations outside the pack boundary (+1 km) are dropped.

New York's sources: the operators' GTFS (weekday schedule) for New York City Subway with the
Staten Island Railway, PATH, LIRR, Metro-North and NJ Transit rail and light rail.

Lines: per route, the weekday daytime stopping patterns. A pattern with at least `min_share` of
its route's trips (and `min_trips` a day) is its own line; a pattern that is a contiguous piece of
a kept one (a short turn) adds its trips to that line; anything else goes to the kept pattern it
overlaps most. Trains per hour per demand level: high = the mean of 7-9 and 17-19, medium =
10-16, low = 0-6, counted at each trip's stop nearest the city's count point (New York: Midtown,
the pack origin), both directions halved. Run times:
per consecutive station pair, the median of every trip of the route that runs it in the level's
hours (departure to departure, so a stop's dwell is in the hop that leaves it).

Raw GTFS: the subway feed is maps/riders/nycriders/data/gtfs_subway (MTA, Feb 2026); the others in
data/raw/gtfs/ (downloaded once, see CITIES).

Adding a city: an entry in CITIES (pack, out, sources, optionally count_point [lon, lat] and
merge_m), then `net <city>`. The calibration run on it: the `calib` bin and pipeline/calib.py.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")

import json
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAPS = os.path.dirname(ROOT)
RAW = os.path.join(ROOT, "data", "raw")
WORK = os.path.join(ROOT, "data", "work", "realnet")
PACKS = os.path.join(ROOT, "data", "packs")
R_EARTH = 6371008.8

NYCRIDERS = os.path.join(MAPS, "riders", "nycriders")
MTA_STATIONS = os.path.join(MAPS, "data", "nystreets", "MTA_Subway_Stations.csv")
OD_SUBWAY = os.path.join(NYCRIDERS, "data", "od_wednesday_oct.csv")
NYCRIDERS_STATS = os.path.join(NYCRIDERS, "stats.json")


def gtfs(name, folder, url, min_share, min_trips, train_m, kind, **kw):
    """A GTFS source. `train_m`: train length in metres per route ("_" the rest); `kind`: metro,
    commuter, light_rail, tram or monorail (`route_kind` per route overrides); `operator` (the
    source name if absent; `route_operator` per route) is what operator totals are summed by."""
    return {"name": name, "adapter": "gtfs", "folder": folder, "url": url, "min_share": min_share,
            "min_trips": min_trips, "train_m": train_m, "kind": kind, **kw}


# a group table: stop id -> the station the city's counts use (name joined, position averaged)
MTA_GROUP = {"csv": MTA_STATIONS, "stop": "GTFS Stop ID", "group": "Complex ID", "name": "Stop Name",
             "lon": "GTFS Longitude", "lat": "GTFS Latitude", "prefix": "mta"}
NJT_LIGHT = {"4": "light_rail", "13": "light_rail", "17": "light_rail"}

PAR_TRAM = ["IDFM:C02317", "IDFM:C01843", "IDFM:C01684", "IDFM:C01999", "IDFM:C01794", "IDFM:C01795", "IDFM:C01679", "IDFM:C01391",
            "IDFM:C01774", "IDFM:C01389", "IDFM:C01390", "IDFM:C02344", "IDFM:C02528", "IDFM:C02529", "IDFM:C02732"]
PAR_RER = {"IDFM:C01742": "a", "IDFM:C01743": "b", "IDFM:C01727": "c", "IDFM:C01728": "d", "IDFM:C01729": "e"}

# Train length in metres per route (cars x car length), for the game's 20 m cars. Subway: A
# division 51 ft cars (10, the 7 11), B division 60 ft (600 ft trains; J, Z, M, L 480 ft; G 300 ft),
# shuttles short; SIR 4 x 75 ft. PATH 7 x 51 ft. Commuter rail: 10 x 85 ft as a typical peak train.
# NJT's feed also has its light rail: Hudson-Bergen (route 4) 2-car, Newark (13) 2-car trains.
CITIES = {
    "nyc": {
        "pack": "nyc",
        "out": os.path.join(WORK, "nyc_real.json"),
        "what": "New York's rail network from GTFS (T-007)",
        "merge_m": 150.0,
        "sources": [
            gtfs("subway", os.path.join(NYCRIDERS, "data", "gtfs_subway"), "http://web.mta.info/developers/data/nyct/subway/google_transit.zip", 0.15, 20,
                 {"1": 155, "2": 155, "3": 155, "4": 155, "5": 155, "5X": 155, "6": 155, "6X": 155, "7": 171, "7X": 171, "GS": 93, "FS": 37, "H": 73,
                  "G": 91, "J": 146, "Z": 146, "M": 146, "L": 146, "SI": 91, "_": 183}, "metro", group=MTA_GROUP, merge=False),
            gtfs("path", os.path.join(RAW, "gtfs", "path-nj-us"), "http://data.trilliumtransit.com/gtfs/path-nj-us/path-nj-us.zip", 0.15, 20, {"_": 109}, "metro"),
            gtfs("lirr", os.path.join(RAW, "gtfs", "gtfslirr"), "https://rrgtfsfeeds.s3.amazonaws.com/gtfslirr.zip", 0.05, 4, {"_": 260}, "commuter"),
            gtfs("mnr", os.path.join(RAW, "gtfs", "gtfsmnr"), "https://rrgtfsfeeds.s3.amazonaws.com/gtfsmnr.zip", 0.05, 4, {"_": 260}, "commuter"),
            gtfs("njt", os.path.join(RAW, "gtfs", "njt_rail"), "https://www.njtransit.com/rail_data.zip", 0.05, 4, {"_": 260, "4": 60, "13": 56, "17": 60}, "commuter",
                 route_kind=NJT_LIGHT, route_operator={r: "njt_light" for r in NJT_LIGHT}),
        ],
    },
    # T-509's small city: a first rough network from OSM only (stage 3 adds the station join to
    # S12 and the city's subway counts, and checks frequencies against timetables)
    "sdj": {
        "pack": "sdj",
        "out": os.path.join(WORK, "sdj_real.json"),
        "what": "Sendai's rail network from OSM route relations (T-509, rough)",
        "count_point": [140.8823, 38.2601],  # Sendai station
        # No relation has interval tags: weekday trains an hour each way (peak, midday, first hour)
        # by line from JR East's and the city's timetables as remembered, rough (T-509 stage 3);
        # the Shinkansen and limited expresses (not tagged high speed) dropped by name. Trains: JR
        # locals 4 cars, rural lines and the private lines 2, Namboku 4 x 20 m, Tozai 4 x 15 m.
        "sources": [{"name": "osm", "adapter": "osm", "file": os.path.join(RAW, "t509", "osm_sdj.json"), "min_share": 0.05, "min_trips": 4,
                     "skip_name": r"新幹線|Shinkansen|列車|つばさ|ひたち|Hitachi|アテルイ|臨海鉄道",
                     "lines": [
                         (r"r12465875|r12465877", {"tph": (1.5, 1, 0.5), "kmh": 50, "train_m": 80, "operator": "jr"}),  # Shiroishi-Sendai locals
                         (r"南北線", {"tph": (15, 8.5, 4), "kmh": 32, "train_m": 80, "operator": "subway"}),
                         (r"東西線", {"tph": (12, 8, 4), "kmh": 32, "train_m": 60, "operator": "subway"}),
                         (r"空港|Aiport|Airport", {"tph": (4, 2.5, 1), "kmh": 42, "train_m": 80, "operator": "jr"}),
                         (r"仙石東北", {"tph": (0.5, 0.5, 0), "kmh": 50, "train_m": 80, "operator": "jr"}),  # two relations a direction
                         (r"仙石線", {"tph": (5, 3, 1), "kmh": 40, "train_m": 80, "operator": "jr"}),
                         (r"仙山線.*Local", {"tph": (2.5, 1.5, 0.5), "kmh": 43, "train_m": 80, "operator": "jr"}),
                         (r"仙山線", {"tph": (1, 1, 0), "kmh": 55, "train_m": 80, "operator": "jr"}),  # rapids
                         (r"利府", {"tph": (2, 1, 0.5), "kmh": 40, "train_m": 80, "operator": "jr"}),
                         (r"東北本線|東北線|Tohoku", {"tph": (2.5, 1.5, 0.5), "kmh": 50, "train_m": 80, "operator": "jr"}),
                         (r"常磐線", {"tph": (1.5, 1, 0.5), "kmh": 50, "train_m": 80, "operator": "jr"}),
                         (r"阿武隈急行", {"tph": (2, 1, 0.5), "kmh": 45, "train_m": 40, "operator": "abukuma"}),
                         (r"飯坂線", {"tph": (4, 2, 1), "kmh": 35, "train_m": 40, "operator": "iizaka"}),
                         (r".", {"tph": (1.5, 1, 0.5), "kmh": 45, "train_m": 40, "operator": "jr"}),  # JR rural lines
                     ]}],
    },
    # Seoul: seoulriders' weekday timetable for all its lines (pipeline/calib_seo.py trips). Stops
    # are seoulriders' station complexes, which the card counts use; the source is named like its
    # group prefix so each station's `target` is its complex id. Train length from seoulriders'
    # rated capacity at 160 a car; everything is one operator, since a journey taps in once.
    "seo": {
        "pack": "seo",
        "out": os.path.join(WORK, "seo_real.json"),
        "what": "Seoul's rail network from the seoulriders weekday timetable (T-509)",
        "count_point": [126.9707, 37.5547],  # Seoul station
        "sources": [{"name": "seo", "adapter": "trips", "file": os.path.join(WORK, "seo_trips.csv"), "group": {"prefix": "seo"},
                     "min_share": 0.05, "min_trips": 4, "operator": "rail", "kind": "metro",
                     "train_m": {"_": 200, "5": 160, "6": 160, "7": 160, "8": 120, "9": 120, "SN": 120, "AR": 120, "SH": 120, "GG": 120,
                                 "GC": 160, "I1": 160, "UI": 40, "SL": 40, "I2": 40, "GP": 40, "UJ": 40},
                     "route_kind": {**{r: "light_rail" for r in ("UI", "SL", "I2", "GP", "UJ")}, **{r: "commuter" for r in ("GJ", "GC", "GG", "SH", "AR")}}}],
    },
    # Toronto, the hold-out (pipeline/calib_tor.py): TTC rail (subway, Lines 5-6, 5xx streetcars,
    # since the census counts streetcars with rail), GO rail, UP Express. Each TTC line is its own
    # operator, as the TTC's per-line boardings count. Trains: TTC 6 x 23 m (Line 4 4 cars),
    # Flexity LRVs 2 x 30 m, streetcars 30 m, GO 12 bilevels, UP 3 DMUs.
    "tor": {
        "pack": "tor",
        "out": os.path.join(WORK, "tor_real.json"),
        "what": "Toronto's rail network from TTC, GO and UP Express GTFS (T-509)",
        "count_point": [-79.3807, 43.6453],  # Union
        "sources": [
            gtfs("ttc", os.path.join(RAW, "t509", "tor", "ttc"), "https://open.toronto.ca/dataset/ttc-routes-and-schedules/", 0.15, 20,
                 {"_": 30, "1": 138, "2": 138, "4": 92, "5": 60, "6": 60}, "tram",
                 route_kind={"1": "metro", "2": "metro", "4": "metro", "5": "light_rail", "6": "light_rail"},
                 route_operator={**{r: "ttc_l" + r for r in "12456"}, **{r: "ttc_streetcar" for r in ("501", "503", "504", "505", "506", "507", "508", "509", "510", "511", "512")}},
                 group={"csv": os.path.join(WORK, "tor_ttc_groups.csv"), "stop": "stop_id", "group": "group", "name": "name", "lon": "lon", "lat": "lat", "prefix": "ttc"}),
            gtfs("go", os.path.join(RAW, "t509", "tor", "go_rail"), "https://assets.metrolinx.com/raw/upload/Documents/Metrolinx/Open%20Data/GO-GTFS.zip", 0.05, 4, {"_": 310}, "commuter"),
            gtfs("up", os.path.join(RAW, "t509", "tor", "up"), "https://assets.metrolinx.com/raw/upload/Documents/Metrolinx/Open%20Data/UP-GTFS.zip", 0.05, 4, {"_": 78}, "commuter"),
        ],
    },
    # London: TfL rail from londonriders (pipeline/calib_lon.py trips), stops already NUMBAT
    # stations; each line is its own operator, named by NUMBAT's line code, so operator totals
    # are NUMBAT's line boarders (Circle with H&C, as NUMBAT counts it). Train lengths: stock x cars.
    "lon": {
        "pack": "lon",
        "out": os.path.join(WORK, "lon_real.json"),
        "what": "London's rail network: TfL from londonriders' timetables (T-509)",
        "sources": [{"name": "tfl", "adapter": "trips", "file": os.path.join(RAW, "t509", "lon_tfl_trips.csv"), "group": {"prefix": "numbat"},
                     "min_share": 0.05, "min_trips": 4,
                     "train_m": {"BAK": 113, "CEN": 133, "CIR": 117, "HAM": 117, "DIS": 117, "MET": 133, "JUB": 126, "NOR": 108, "PIC": 106,
                                 "VIC": 133, "WAC": 64, "DLR": 87, "EZL": 205, "WAG": 83, "GOB": 83, "WEL": 83, "URL": 83, "_": 104},
                     "route_kind": {"DLR": "light_rail", "EZL": "commuter", "WAG": "commuter", "GOB": "commuter", "WEL": "commuter", "URL": "commuter",
                                    "NLL": "commuter", "ELL": "commuter"},
                     "route_operator": {"BAK": "BAK", "CEN": "CEN", "CIR": "HAM", "HAM": "HAM", "DIS": "DIS", "MET": "MET", "JUB": "JUB", "NOR": "NOR",
                                        "PIC": "PIC", "VIC": "VIC", "WAC": "WAC", "DLR": "DLR", "EZL": "EZL", "WAG": "WAG", "GOB": "GOB", "WEL": "WEL",
                                        "URL": "URL", "NLL": "NLL", "ELL": "ELL"}},
                    # National Rail (calib_lon.py nr): one GTFS route per operator, so the share is small;
                    # stations are CRS codes (ORR's counts); 10-car trains as a typical peak suburban train
                    gtfs("nr", os.path.join(RAW, "t509", "gb", "nr_lse"), "https://beta.aubin.app/gtfs/great_britain_gtfs.zip", 0.008, 12, {"_": 200, "HX": 120, "GX": 240},
                         "commuter", group={"csv": os.path.join(RAW, "t509", "gb", "nr_lse", "crs_groups.csv"), "stop": "stop_id", "group": "crs", "name": "name",
                                            "lon": "lon", "lat": "lat", "prefix": "crs"}, operator="nr")],
        "merge_m": 200.0,
    },
    # Paris: IDFM's GTFS cut to rail (pipeline/calib_par.py gtfs); stations are the parent stops,
    # IDFM's zones de correspondance, which the validation counts use. Train lengths: Metro 5 cars
    # (75 m) except 1 and 14, 3B and 7B 3 cars; RER and Transilien double units (220 m).
    "par": {
        "pack": "par",
        "out": os.path.join(WORK, "par_real.json"),
        "what": "Paris's rail network from IDFM's GTFS (T-509)",
        "count_point": [2.3470, 48.8585],  # Chatelet
        "sources": [
            gtfs("idfm", os.path.join(RAW, "t509", "idfm", "rail"), "https://eu.ftp.opendatasoft.com/stif/GTFS/IDFM-gtfs.zip", 0.05, 4,
                 {**{f"IDFM:C0{r}": 75 for r in range(1371, 1388)}, "IDFM:C01371": 90, "IDFM:C01384": 120, "IDFM:C01386": 45, "IDFM:C01387": 45,
                  **{r: 40 for r in PAR_TRAM}, "IDFM:C01388": 34, "IDFM:C00563": 26, "_": 220}, "commuter",
                 group={"csv": os.path.join(RAW, "t509", "idfm", "rail", "zdc_groups.csv"), "stop": "stop_id", "group": "zdc", "name": "name",
                        "lon": "lon", "lat": "lat", "prefix": "zdc"},
                 route_kind={**{f"IDFM:C0{r}": "metro" for r in range(1371, 1388)}, **{r: "tram" for r in PAR_TRAM}, "IDFM:C01388": "light_rail", "IDFM:C00563": "light_rail"},
                 route_operator={**{f"IDFM:C0{r}": "metro" for r in range(1371, 1388)}, **{r: "tram" for r in PAR_TRAM}, "IDFM:C01388": "val", "IDFM:C00563": "val",
                                 **{r: "rer_" + n for r, n in PAR_RER.items()}}, operator="transilien"),
        ],
    },
}


def _tyo_routes():
    """Tokyo's lines from the 2015 transport census (pipeline/calib_tyo.py trips writes the trips
    and this per-route kind, operator and train length); stops are S12 station groups."""
    p = os.path.join(WORK, "tyo_routes.json")
    meta = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
    return {"name": "s12", "adapter": "trips", "file": os.path.join(WORK, "tyo_trips.csv"), "group": {"prefix": "s12"},
            "min_share": 0.05, "min_trips": 4, "kind": "commuter",
            "route_kind": {r: m["kind"] for r, m in meta.items()}, "route_operator": {r: m["op"] for r, m in meta.items()},
            "train_m": {"_": 200, **{r: m["train_m"] for r, m in meta.items()}}}


CITIES["tyo"] = {
    "pack": "tyo",
    "out": os.path.join(WORK, "tyo_real.json"),
    "what": "Tokyo's rail network from the 2015 transport census lines, default frequencies (T-509, rough)",
    "count_point": [139.7671, 35.6812],  # Tokyo station
    "merge_m": 0.0,  # one source; stations are already S12 groups
    "sources": [_tyo_routes()],
}
# T-737's stress test: every line in Japan from N02 (pipeline/realnet_n02.py), in Tokyo's frame
N02_DIR = os.path.join(MAPS, "riders", "tokyoriders", "data", "N02-24", "UTF-8")
CITIES["jpn"] = {
    "pack": "tyo",
    "boundary": False,
    "out": os.path.join(WORK, "jpn_real.json"),
    "what": "Japan's whole rail network from N02, frequencies from passing counts (T-737, rough)",
    "count_point": [139.7671, 35.6812],
    "merge_m": 0.0,
    "sources": [{"name": "n02", "adapter": "n02", "stations": os.path.join(N02_DIR, "N02-24_Station.geojson"),
                 "sections": os.path.join(N02_DIR, "N02-24_RailroadSection.geojson"),
                 "density": os.path.join(MAPS, "riders", "japanriders", "data", "segments.geojson"),
                 "min_share": 0.0, "min_trips": 1, "kind": "commuter"}],
}
# the same, cut to Japan's seven city packs (+10 km): seven networks no line joins, so demand solves
# each on its own (T-737's second save)
CITIES["jpn7"] = {**CITIES["jpn"], "out": os.path.join(WORK, "jpn7_real.json"), "boundary": ["tyo", "osa", "ngo", "fuk", "spk", "sdj", "hij"],
                  "boundary_km": 10.0, "what": "Japan's rail in its seven city packs from N02 (T-737, rough)",
                  "sources": [dict(CITIES["jpn"]["sources"][0])]}
CAR_M = 20.0
LEVEL_WINDOWS = {0: [(7, 9), (17, 19)], 1: [(10, 16)], 2: [(0, 6)]}
DAY = (6, 20)
MERGE_M = 150.0

# OSM sources (T-509): a weekday's trains per hour each way by kind of line where the relation has
# no `interval` tag, (peak 7-9 and 17-19, the rest of 6-22, the first hour 5-6); average speed
# between stations with the dwell, km/h; train length, metres. Judgement, to be checked per city
# against a few timetables.
OSM_TPH = {"metro": (20, 10, 4), "commuter": (8, 4, 2), "light_rail": (10, 6, 3), "tram": (10, 7, 3), "monorail": (12, 8, 3)}
OSM_KMH = {"metro": 32, "commuter": 40, "light_rail": 22, "tram": 16, "monorail": 30}
OSM_TRAIN_M = {"metro": 120, "commuter": 200, "light_rail": 60, "tram": 30, "monorail": 60}
OSM_KIND = {"subway": "metro", "train": "commuter", "light_rail": "light_rail", "tram": "tram", "monorail": "monorail"}
OSM_DETOUR = 1.15


def origin(pack="nyc"):
    h = json.load(open(os.path.join(PACKS, pack + ".json")))
    return h["origin"]["lon"], h["origin"]["lat"]


def to_xy(lon, lat, o):
    lon0, lat0 = o
    x = R_EARTH * np.radians(np.asarray(lon, float) - lon0) * np.cos(np.radians(lat0))
    y = R_EARTH * np.radians(np.asarray(lat, float) - lat0)
    return x, y


def secs(t):
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def boundary(pack="nyc"):
    from shapely.geometry import shape
    g = json.load(open(os.path.join(PACKS, pack + ".boundary.geojson")))
    geoms = [shape(f["geometry"]) for f in g["features"]] if "features" in g else [shape(g)]
    geoms = [x if x.is_valid else x.buffer(0) for x in geoms]  # London's has a self-touching ring
    from shapely import union_all
    return union_all(geoms).buffer(0.01)  # ~1 km in degrees


def read(folder, name, **kw):
    p = os.path.join(folder, name)
    return pd.read_csv(p, dtype=str, keep_default_na=False, **kw) if os.path.exists(p) else None


def service_ids(folder):
    """Service ids running on the weekday (a Wednesday) with the most trips in the feed."""
    cal = read(folder, "calendar.txt")
    cd = read(folder, "calendar_dates.txt")
    trips = read(folder, "trips.txt", usecols=["service_id"])
    per_service = trips["service_id"].value_counts().to_dict()
    dates = set()
    if cal is not None and len(cal):
        for _, r in cal.iterrows():
            d0, d1 = pd.Timestamp(r["start_date"]), pd.Timestamp(r["end_date"])
            dates.update(pd.date_range(d0, min(d1, d0 + pd.Timedelta(days=120)), freq="W-WED"))
    if cd is not None and len(cd):
        dates.update(pd.to_datetime(cd["date"].unique()))
    dates = sorted(d for d in dates if d.dayofweek == 2)
    best, best_n = None, -1
    for d in dates:
        s = set()
        if cal is not None and len(cal):
            ds = d.strftime("%Y%m%d")
            m = (cal["wednesday"] == "1") & (cal["start_date"] <= ds) & (cal["end_date"] >= ds)
            s |= set(cal.loc[m, "service_id"])
        if cd is not None and len(cd):
            ds = d.strftime("%Y%m%d")
            x = cd[cd["date"] == ds]
            s |= set(x.loc[x["exception_type"] == "1", "service_id"])
            s -= set(x.loc[x["exception_type"] == "2", "service_id"])
        n = sum(per_service.get(v, 0) for v in s)
        if n > best_n:
            best, best_n = (d, s), n
    return best


def group_table(g):
    """A source's group table: (stop -> group, group -> (name, lon, lat)). Names of a group's
    stops are joined with "/", positions averaged."""
    t = pd.read_csv(g["csv"], dtype=str)
    cx = dict(zip(t[g["stop"]], t[g["group"]]))
    cname = t.groupby(g["group"])[g["name"]].agg(lambda s: "/".join(dict.fromkeys(s)))
    cpos = t.assign(lon=t[g["lon"]].astype(float), lat=t[g["lat"]].astype(float)).groupby(g["group"])[["lon", "lat"]].mean()
    return cx, {c: (cname[c], float(cpos.loc[c, "lon"]), float(cpos.loc[c, "lat"])) for c in cname.index}


def load_gtfs(src, inside):
    """Stations and trips of one GTFS feed: (stations {key: (name, lon, lat)}, trips [(route,
    dir, [keys], [dep secs])], route names)."""
    name, folder = src["name"], src["folder"]
    date, sids = service_ids(folder)
    stops = read(folder, "stops.txt")
    parent = {}
    if "parent_station" in stops:
        parent = {r.stop_id: r.parent_station for r in stops.itertuples() if r.parent_station}
    pos = {r.stop_id: (r.stop_name, float(r.stop_lon), float(r.stop_lat)) for r in stops.itertuples()}
    cx, cst = group_table(src["group"]) if src.get("group") else ({}, {})
    prefix = src["group"]["prefix"] if src.get("group") else None

    def key(sid):
        p = parent.get(sid, sid)
        c = cx.get(p)
        return (prefix + ":" + c) if c else (name + ":" + p)
    stations = {}
    for sid in pos:
        k = key(sid)
        c = cx.get(parent.get(sid, sid))
        stations[k] = cst[c] if c else pos[parent.get(sid, sid)]
    keep = {k for k, (_, lon, lat) in stations.items() if inside(lon, lat)}
    trips = read(folder, "trips.txt")
    trips = trips[trips["service_id"].isin(sids)]
    tinfo = {r.trip_id: (r.route_id, r.direction_id) for r in trips.itertuples()}
    st = read(folder, "stop_times.txt", usecols=["trip_id", "stop_id", "arrival_time", "departure_time", "stop_sequence"])
    st = st[st["trip_id"].isin(tinfo)]
    st = st.assign(seq=st["stop_sequence"].astype(int)).sort_values(["trip_id", "seq"])
    out = []
    for tid, g in st.groupby("trip_id", sort=False):
        ks, ts = [], []
        for sid, dep, arr in zip(g["stop_id"], g["departure_time"], g["arrival_time"]):
            k = key(sid)
            if k not in keep:
                continue
            t = secs(dep or arr)
            if ks and ks[-1] == k:
                ts[-1] = t
                continue
            ks.append(k)
            ts.append(t)
        if len(ks) >= 2:
            r, d = tinfo[tid]
            out.append((r, d, ks, ts))
    print(f"{name}: {date.date()}, {len(sids)} services, {len(out)} trips, {len(keep)} of {len(stations)} stations inside")
    r = read(folder, "routes.txt")
    col = "route_short_name" if "route_short_name" in r and (r["route_short_name"] != "").all() else "route_long_name"
    return {k: v for k, v in stations.items() if k in keep}, out, dict(zip(r["route_id"], r[col]))


def osm_interval_tph(tags, kind, default=None):
    """Trains per hour each way, by hour of the day (24 values), from a route relation's
    `interval` and `interval:conditional` tags (minutes, "mm" or "hh:mm"), else `default`
    (peak, midday, first hour) or OSM_TPH."""
    def minutes(v):
        v = v.strip()
        try:
            if ":" in v:
                h, m = v.split(":")[:2]
                return int(h) * 60 + int(m)
            return float(v)
        except ValueError:
            return None
    d = default or OSM_TPH[kind]
    peak, mid, first = d
    base = minutes(tags["interval"]) if tags.get("interval") else None
    if base:
        # `interval` is the usual (midday) headway; the peak keeps the default's peak-to-midday
        # ratio unless conditional intervals say otherwise
        mid = 60.0 / base
        peak = mid if tags.get("interval:conditional") else mid * d[0] / d[1]
        first = mid / 2
    tph = [0.0] * 5 + [first] + [mid] * 16 + [mid / 2] * 2
    for h in (7, 8, 17, 18):
        tph[h] = peak
    # "00:05 @ (Mo-Fr 07:00-09:00); 00:10 @ (Mo-Fr 09:00-17:00)": weekday or undated parts
    import re
    for part in (tags.get("interval:conditional") or "").split(";"):
        if "@" not in part:
            continue
        v, cond = part.split("@", 1)
        m = minutes(v)
        if not m or (re.search(r"\b(Sa|Su|PH)\b", cond) and not re.search(r"\b(Mo|Tu|We|Th|Fr)\b", cond)):
            continue
        for a, b in re.findall(r"(\d{1,2}):\d{2}\s*-\s*(\d{1,2}):\d{2}", cond):
            for h in range(int(a), min(int(b), 24)):
                tph[h] = 60.0 / m
    return tph


def load_osm(src, inside):
    """Stations and trips from OSM route relations (`fetch-osm`'s file). Each relation is one
    direction of a route (its route_master, else itself); its stops are its stop members in order
    (platforms where it has no stops). A station is the stop_area a stop belongs to, else the
    stops of one name within `name_m`. Trips are made up hour by hour from the trains per hour,
    run times from the straight-line hop x OSM_DETOUR at the kind's speed."""
    d = json.load(open(src["file"], encoding="utf-8"))
    el = {(e["type"], e["id"]): e for e in d["elements"]}
    rels = [e for e in d["elements"] if e["type"] == "relation" and e.get("tags", {}).get("type") == "route"]
    masters = {}
    area_of = {}
    for e in d["elements"]:
        if e["type"] != "relation":
            continue
        t = e.get("tags", {})
        if t.get("type") == "route_master":
            for m in e["members"]:
                masters[m["ref"]] = e
        if t.get("public_transport") == "stop_area":
            for m in e["members"]:
                area_of.setdefault((m["type"], m["ref"]), e)

    def pos(m):
        e = el.get((m["type"], m["ref"]))
        if not e:
            return None
        if "lat" in e:
            return e["lon"], e["lat"]
        if "center" in e:
            return e["center"]["lon"], e["center"]["lat"]
        return None
    skip = set(src.get("skip_service", ["high_speed", "long_distance", "night", "tourism"]))
    import re

    def label(r):
        t = r["tags"]
        return f"{t.get('name', '')} | {t.get('name:en', '')} | {t.get('ref', '')} | r{r['id']}"

    def override(r):
        """A source's `lines`: [(regex on "name | name:en | ref | r<relation id>", {"tph": (peak,
        midday, first), "kmh", "train_m", "operator"})], the first match, for relations with no
        interval tags (Sendai)."""
        return next((v for p, v in src.get("lines", []) if re.search(p, label(r))), {})
    raw = []  # (relation, kind, [(member key, name, lon, lat)])
    for r in rels:
        t = r["tags"]
        kind = OSM_KIND.get(t.get("route"))
        if not kind or t.get("service") in skip or t.get("highspeed") == "yes":
            continue
        # relations not tagged as high speed or long distance that are (`skip_name`, a regex)
        if src.get("skip_name") and re.search(src["skip_name"], label(r)):
            continue
        stops = [m for m in r["members"] if m["role"] in ("stop", "stop_entry_only", "stop_exit_only")]
        if not stops:
            stops = [m for m in r["members"] if m["role"] in ("platform", "platform_entry_only", "platform_exit_only")]
        pts = []
        for m in stops:
            p = pos(m)
            if p is None:
                continue
            e = el.get((m["type"], m["ref"]), {})
            pts.append(((m["type"], m["ref"]), e.get("tags", {}).get("name", ""), p[0], p[1]))
        if len(pts) >= 2:
            raw.append((r, kind, pts))
    # stations: the stop_area, else one per name within name_m (a greedy cluster), else the stop
    name_m = src.get("name_m", 400.0)
    o = origin(src["pack"])
    key_of, members = {}, defaultdict(list)
    by_name = defaultdict(list)  # name -> [(key, x, y)]
    for _, _, pts in raw:
        for mk, nm, lon, lat in pts:
            if mk in key_of:
                continue
            a = area_of.get(mk)
            if a:
                k = f"osm:a{a['id']}"
                nm = a.get("tags", {}).get("name", nm)
            elif nm:
                x, y = to_xy(lon, lat, o)
                k = None
                for kk, xx, yy in by_name[nm]:
                    if (xx - x) ** 2 + (yy - y) ** 2 < name_m ** 2:
                        k = kk
                        break
                if k is None:
                    k = f"osm:{mk[0][0]}{mk[1]}"
                    by_name[nm].append((k, float(x), float(y)))
            else:
                k = f"osm:{mk[0][0]}{mk[1]}"
            key_of[mk] = k
            members[k].append((nm, lon, lat))
    stations = {k: (v[0][0], float(np.mean([p[1] for p in v])), float(np.mean([p[2] for p in v]))) for k, v in members.items()}
    keep = {k for k, (_, lon, lat) in stations.items() if inside(lon, lat)}
    xy = {k: to_xy(lon, lat, o) for k, (_, lon, lat) in stations.items()}
    trips, names = [], {}
    n_rel = Counter(f"m{masters[r['id']]['id']}" if r["id"] in masters else f"r{r['id']}" for r, _, _ in raw)
    for r, kind, pts in raw:
        t = r["tags"]
        m = masters.get(r["id"])
        route = f"m{m['id']}" if m else f"r{r['id']}"
        names[route] = (m or r).get("tags", {}).get("ref") or (m or r).get("tags", {}).get("name") or route
        ks = []
        for mk, *_ in pts:
            k = key_of[mk]
            if k in keep and (not ks or ks[-1] != k):
                ks.append(k)
        if len(ks) < 2:
            continue
        ov = override(r)
        if ov.get("operator"):
            src.setdefault("route_operator", {})[route] = ov["operator"]
        kmh = ov.get("kmh", OSM_KMH[kind])
        hop = [float(np.hypot(xy[ks[i + 1]][0] - xy[ks[i]][0], xy[ks[i + 1]][1] - xy[ks[i]][1])) * OSM_DETOUR / (kmh / 3.6) for i in range(len(ks) - 1)]
        runs = [(ks, np.concatenate([[0.0], np.cumsum(hop)]))]
        if n_rel[route] == 1:  # only one direction mapped: the trains run back the same way
            runs.append((ks[::-1], np.concatenate([[0.0], np.cumsum(hop[::-1])])))
        for ks, at in runs:
            dr = "0" if ks[0] <= ks[-1] else "1"
            acc = 0.0
            for h, n in enumerate(osm_interval_tph(t, kind, ov.get("tph"))):
                acc += n
                k = int(acc)
                acc -= k
                for j in range(k):
                    dep = h * 3600 + (j + 0.5) * 3600 / k
                    trips.append((route, dr, ks, [int(dep + v) for v in at]))
        src.setdefault("route_kind", {})[route] = kind
        src.setdefault("train_m", {"_": OSM_TRAIN_M["metro"]}).setdefault(route, ov.get("train_m", OSM_TRAIN_M[kind]))
    print(f"{src['name']}: {len(raw)} route relations, {len(trips)} trips made up, {len(keep)} of {len(stations)} stations inside")
    return {k: v for k, v in stations.items() if k in keep}, trips, names


def load_trips_csv(src, inside):
    """Stations and trips from a CSV of timetabled trips (any timetable written out): columns
    route, direction, trip, stop, name, lon, lat, time (h:mm:ss or seconds), in stop order per
    trip; optional route_name."""
    t = pd.read_csv(src["file"], dtype=str, keep_default_na=False)
    # `group: {"prefix": p}`: the CSV's stops are already the counts' stations (London's NUMBAT codes)
    pre = src["group"]["prefix"] if src.get("group") else src["name"]
    stations = {}
    for r in t.drop_duplicates("stop").itertuples():
        stations[pre + ":" + r.stop] = (r.name, float(r.lon), float(r.lat))
    keep = {k for k, (_, lon, lat) in stations.items() if inside(lon, lat)}
    trips = []
    for _, g in t.groupby("trip", sort=False):
        ks, ts = [], []
        for s, tm in zip(g["stop"], g["time"]):
            k = pre + ":" + s
            if k not in keep:
                continue
            v = secs(tm) if ":" in tm else int(float(tm))
            if ks and ks[-1] == k:
                ts[-1] = v
                continue
            ks.append(k)
            ts.append(v)
        if len(ks) >= 2:
            trips.append((g["route"].iloc[0], g["direction"].iloc[0], ks, ts))
    names = dict(zip(t["route"], t["route_name"])) if "route_name" in t else {}
    print(f"{src['name']}: {len(trips)} trips, {len(keep)} of {len(stations)} stations inside")
    return {k: v for k, v in stations.items() if k in keep}, trips, names


def load_n02(src, inside):
    import realnet_n02
    return realnet_n02.load(src, inside, osm_interval_tph)


ADAPTERS = {"gtfs": load_gtfs, "osm": load_osm, "trips": load_trips_csv, "n02": load_n02}


def is_piece(small, big):
    """`small` is a contiguous run of `big`."""
    n = len(small)
    return any(tuple(big[i:i + n]) == tuple(small) for i in range(len(big) - n + 1))


def build_lines(src, trips, xy, cxy=(0.0, 0.0)):
    """Lines of one source from its trips; `xy` station positions, `cxy` the count point."""
    name, min_share, min_trips = src["name"], src.get("min_share", 0.05), src.get("min_trips", 4)
    lines = []
    by_route = defaultdict(list)
    for t in trips:
        by_route[t[0]].append(t)
    for route, ts in sorted(by_route.items()):
        # canonical orientation: direction 0 as given, direction 1 reversed
        canon = []
        for r, d, ks, tt in ts:
            c = tuple(ks) if d != "1" else tuple(reversed(ks))
            canon.append(c)
        mid_t = []
        for (_, _, ks, tt) in ts:
            # time at the stop nearest the count point
            dist = [(xy[k][0] - cxy[0]) ** 2 + (xy[k][1] - cxy[1]) ** 2 for k in ks]
            mid_t.append(tt[int(np.argmin(dist))] / 3600.0 % 24)
        day = [DAY[0] <= h < DAY[1] for h in mid_t]
        cnt = Counter(c for c, dd in zip(canon, day) if dd)
        if not cnt:
            cnt = Counter(canon)
        total = sum(cnt.values())
        kept = []
        for c, n in cnt.most_common():
            if n < max(min_share * total, min_trips) or any(is_piece(c, k) for k in kept):
                continue
            # the same ends and nearly the same stops as a busier pattern: a variant of it (a stop
            # skipped one way for works, say), not a service of its own; splitting it off would
            # halve the frequency riders see
            if any(c[0] == k[0] and c[-1] == k[-1] and len(set(c) & set(k)) / len(set(c) | set(k)) >= 0.8 for k in kept):
                continue
            kept.append(c)
        if not kept:
            kept = [cnt.most_common(1)[0][0]]
        # every trip goes to a kept pattern: one it is a contiguous piece of (most stops first),
        # else the one it shares most stops with
        assign = {}
        for c in set(canon):
            hosts = [k for k in kept if is_piece(c, k)]
            if hosts:
                assign[c] = max(hosts, key=lambda k: cnt.get(k, 0))
                continue
            ov = [(len(set(c) & set(k)) / len(c), cnt.get(k, 0), k) for k in kept]
            best = max(ov)
            assign[c] = best[2] if best[0] >= 0.5 else None
        # hop times by station pair and level, over every trip of the route
        hop = defaultdict(lambda: defaultdict(list))
        for (_, _, ks, tt), h in zip(ts, mid_t):
            levs = [lv for lv, ws in LEVEL_WINDOWS.items() if any(a <= h < b for a, b in ws)]
            for i in range(len(ks) - 1):
                dt = tt[i + 1] - tt[i]
                if dt <= 0:
                    continue
                for lv in levs:
                    hop[(ks[i], ks[i + 1])][lv].append(dt)
                hop[(ks[i], ks[i + 1])]["all"].append(dt)
        for k in kept:
            n_lev = {lv: 0.0 for lv in LEVEL_WINDOWS}
            for c, h in zip(canon, mid_t):
                if assign.get(c) != k:
                    continue
                for lv, ws in LEVEL_WINDOWS.items():
                    if any(a <= h < b for a, b in ws):
                        n_lev[lv] += 1
            hours = {lv: sum(b - a for a, b in ws) for lv, ws in LEVEL_WINDOWS.items()}
            tph = [round(n_lev[lv] / hours[lv] / 2.0, 2) for lv in range(3)]
            if max(tph) == 0:
                continue

            def hop_t(a, b, lv):
                for src in ((a, b), (b, a)):
                    h = hop.get(src)
                    if not h:
                        continue
                    for want in (lv, 1, 0, "all"):
                        if h.get(want):
                            return float(np.median(h[want]))
                return None
            n = len(k)
            t0 = [[0.0] * n for _ in range(3)]
            t1 = [[0.0] * n for _ in range(3)]
            ok = True
            for lv in range(3):
                for i in range(n - 1):
                    f, b = hop_t(k[i], k[i + 1], lv), hop_t(k[i + 1], k[i], lv)
                    if f is None or b is None:
                        ok = False
                    t0[lv][i] = f or b or 0.0
                    t1[lv][i + 1] = b or f or 0.0
            if not ok:
                print(f"  {name} {route}: a hop has no time in one direction; mirrored")
            m = src["train_m"]
            cars = max(1, round(m.get(route, m["_"]) / CAR_M))
            lines.append({"feed": name, "route": route, "stops": list(k), "tph": tph, "t0": t0, "t1": t1, "cars": cars,
                          "trips_day": sum(1 for c in canon if assign.get(c) == k),
                          "operator": src.get("route_operator", {}).get(route, src.get("operator", name)),
                          "kind": src.get("route_kind", {}).get(route, src.get("kind", "metro"))})
    return lines


def cmd_net(city=None):
    city = city or (sys.argv[2] if len(sys.argv) > 2 else "nyc")
    cfg = CITIES[city]
    os.makedirs(os.path.dirname(cfg["out"]), exist_ok=True)
    o = origin(cfg["pack"])
    cxy = tuple(float(v) for v in to_xy(*cfg["count_point"], o)) if cfg.get("count_point") else (0.0, 0.0)
    merge_m = cfg.get("merge_m", MERGE_M)
    from shapely import prepared
    from shapely.geometry import Point
    # `boundary`: False keeps every station; a list of packs keeps those within `boundary_km` of any
    bb = cfg.get("boundary", True)
    if isinstance(bb, list):
        from shapely import union_all
        bnd = prepared.prep(union_all([boundary(p).buffer(cfg.get("boundary_km", 0) / 111.0) for p in bb]))
    else:
        bnd = prepared.prep(boundary(cfg["pack"])) if bb else None

    def inside(lon, lat):
        return bnd is None or bnd.contains(Point(lon, lat))
    stations, lines = {}, []
    no_merge = set()
    for src in cfg["sources"]:
        src = {**src, "pack": cfg["pack"]}
        name = src["name"]
        st, trips, rn = ADAPTERS[src["adapter"]](src, inside)
        group_prefix = src["group"]["prefix"] + ":" if src.get("group") else None
        for k, (nm, lon, lat) in st.items():
            x, y = to_xy(lon, lat, o)
            stations[k] = {"key": k, "name": nm, "feed": name, "lon": lon, "lat": lat, "x": float(x), "y": float(y),
                           "target": k[len(group_prefix):] if group_prefix and k.startswith(group_prefix) else None}
            if src.get("merge", True) is False:
                no_merge.add(k)
        xy = {k: (v["x"], v["y"]) for k, v in stations.items()}
        ls = build_lines(src, trips, xy, cxy)
        for l in ls:
            l["name"] = rn.get(l["route"], l["route"])
            if l["route"] in src.get("shapes", {}):  # the n02 adapter's track, for `geom`
                sx, sy = to_xy(*np.array(src["shapes"][l["route"]]).T, o)
                l["shape"] = np.stack([sx, sy], 1).round(1).tolist()
        print(f"  {name}: {len(ls)} lines from {len(set(l['route'] for l in ls))} routes")
        lines += ls
    # merge stations of different sources within merge_m (one interchange)
    keys = [k for k in stations if k not in no_merge] if merge_m > 0 else []
    alias = {}
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            sa, sb = stations[a], stations[b]
            if sa["feed"] != sb["feed"] and (sa["x"] - sb["x"]) ** 2 + (sa["y"] - sb["y"]) ** 2 < merge_m ** 2:
                ra, rb = alias.get(a, a), alias.get(b, b)
                # never chain two stations of one source together (London: Cannon Street NR
                # near both Cannon Street and Bank would make one stop of them)
                fa = {stations[k]["feed"] for k in [ra] + [k for k, v in alias.items() if v == ra]}
                fb = {stations[k]["feed"] for k in [rb] + [k for k, v in alias.items() if v == rb]}
                if ra != rb and not (fa & fb):
                    alias[rb] = ra
                    for k, v in list(alias.items()):
                        if v == rb:
                            alias[k] = ra
    for l in lines:
        l["stops"] = [alias.get(k, k) for k in l["stops"]]
    used = sorted({k for l in lines for k in l["stops"]})
    for b, a in alias.items():
        if b in stations and a in stations:
            stations[a]["name"] = stations[a]["name"] + " / " + stations[b]["name"]
    idx = {k: i for i, k in enumerate(used)}
    st_list = [stations[k] for k in used]
    for l in lines:
        l["stops"] = [idx[k] for k in l["stops"]]
    # drop lines that collapsed (merged stops next to each other)
    good = []
    for l in lines:
        s = l["stops"]
        if any(s[i] == s[i + 1] for i in range(len(s) - 1)) or len(s) < 2:
            print(f"  dropped {l['feed']} {l['name']}: repeated stop after merging")
            continue
        good.append(l)
    lines = good
    # the arrays DemandApi::set_network takes
    st_xy, line_n, stops, times, tph, cars = [], [], [], [], [], []
    for s in st_list:
        st_xy += [s["x"], s["y"]]
    for l in lines:
        n = len(l["stops"])
        line_n.append(n)
        stops += l["stops"]
        for run in (l["t0"], l["t1"]):
            for lv in range(3):
                times += run[lv]
        tph += l["tph"]
        cars.append(l["cars"])
    out = {"what": cfg.get("what", f"{city}'s rail network (T-509)"), "city": city, "pack": cfg["pack"],
           "feeds": {s["name"]: s.get("url") or s.get("file") for s in cfg["sources"]},
           "stations": st_list, "lines": lines,
           "demand": {"st_xy": st_xy, "line_n": line_n, "stops": stops, "times": times, "tph": tph, "cars": cars}}
    p = cfg["out"]
    json.dump(out, open(p, "w"))
    by = Counter(l["feed"] for l in lines)
    print(f"{len(st_list)} stations, {len(lines)} lines {dict(by)} -> {p}")


# ---------------------------------------------------------------- measured targets

ACS_B08301 = "https://www2.census.gov/programs-surveys/acs/summary_file/2023/table-based-SF/data/5YRData/acsdt5y2023-b08301.dat"
# B08301 (2019+): 001 workers 16+, 002 car/truck/van, 010 public transport (no taxi), 011 bus,
# 012 subway or elevated rail, 013 long-distance train or commuter rail, 014 light rail,
# streetcar or trolley, 015 ferry, 016 taxi, 017 motorcycle, 018 bicycle, 019 walked, 020 other,
# 021 worked from home
B08301_COLS = [f"B08301_E{i:03d}" for i in range(1, 22)]
COUNTIES = {
    "36005": "Bronx", "36047": "Kings", "36061": "New York", "36081": "Queens", "36085": "Richmond",
    "36059": "Nassau", "36103": "Suffolk", "36119": "Westchester", "36087": "Rockland", "36079": "Putnam",
    "34003": "Bergen", "34013": "Essex", "34017": "Hudson", "34019": "Hunterdon", "34023": "Middlesex",
    "34025": "Monmouth", "34027": "Morris", "34029": "Ocean", "34031": "Passaic", "34035": "Somerset",
    "34037": "Sussex", "34039": "Union", "09001": "Fairfield CT", "36027": "Dutchess", "36071": "Orange NY",
}
COUNTY_ORDER = sorted(COUNTIES)
# Measured weekday totals the subway OD cannot give (October 2024, a weekday): MTA Daily Ridership
# (data.ny.gov vxuj-8kew, the Wednesdays of October 2024, fetched by `targets`); PATH 211,525 on an
# average October 2024 weekday (Port Authority, via Progressive Railroading, 2024-11); NJ Transit
# rail has no recent published weekday figure found (FY2018: 311,250 boardings with 28,400
# rail-to-rail transfers; 2024 at roughly 80% of that is ~250k), so it is a rough number only.
PATH_WEEKDAY = 211525
NJT_WEEKDAY_ROUGH = 250000
MTA_DAILY = "https://data.ny.gov/resource/vxuj-8kew.csv?$where=date%20between%20'2024-10-01T00:00:00'%20and%20'2024-10-31T00:00:00'&$limit=100"
PERIOD_HOURS = [(6, 10), (10, 16), (16, 20), (20, 24), (0, 6)]


def period_of_hour(h):
    for q, (a, b) in enumerate(PERIOD_HOURS):
        if a <= h < b:
            return q


def cmd_counties():
    """One byte per pack cell: its county's index in COUNTY_ORDER (255 = none)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import read_pack
    import shapefile
    from shapely import contains_xy, make_valid
    from shapely.geometry import shape
    header, a = read_pack.load_pack("nyc")
    lon0, lat0 = header["origin"]["lon"], header["origin"]["lat"]
    x, y = a["x_m"].astype(float), a["y_m"].astype(float)
    lon = lon0 + np.degrees(x / (R_EARTH * np.cos(np.radians(lat0))))
    lat = lat0 + np.degrees(y / R_EARTH)
    r = shapefile.Reader(os.path.join(RAW, "cb_2020_us_county_500k.zip"))
    gi = [f[0] for f in r.fields[1:]].index("GEOID")
    out = np.full(len(x), 255, np.uint8)
    polys = {}
    for sr in r.iterShapeRecords():
        g = sr.record[gi]
        if g in COUNTIES:
            polys[g] = make_valid(shape(sr.shape.__geo_interface__))
    for i, g in enumerate(COUNTY_ORDER):
        m = contains_xy(polys[g], lon, lat)
        out[m & (out == 255)] = i
    # cells just off the generalised coast: the nearest county
    miss = np.where(out == 255)[0]
    if len(miss):
        from shapely.geometry import Point
        for c in miss:
            p = Point(lon[c], lat[c])
            out[c] = min(range(len(COUNTY_ORDER)), key=lambda i: polys[COUNTY_ORDER[i]].distance(p))
    p = os.path.join(WORK, "cell_county.u8")
    out.tofile(p)
    print(f"{len(x)} cells, {len(miss)} placed by nearest county -> {p}")


def acs_counties():
    """B08301 by our 25 counties (2020 boundaries). Connecticut's 2023 rows are planning regions,
    so Fairfield is the sum of its 2020 tracts (tract codes are unchanged; read from the 2020
    block file)."""
    cache = os.path.join(RAW, "acs2023_5y_b08301_counties.csv")
    if os.path.exists(cache):
        return pd.read_csv(cache, dtype={"fips": str}).set_index("fips")
    src = os.path.join(RAW, "acsdt5y2023-b08301.dat")
    ct_tracts = set()
    dbf = os.path.join(RAW, "tl_2020_09_tabblock20.dbf")
    import shapefile
    rd = shapefile.Reader(dbf=open(dbf, "rb"))
    names = [f[0] for f in rd.fields[1:]]
    ci, ti = names.index("COUNTYFP20"), names.index("TRACTCE20")
    for rec in rd.iterRecords(fields=["COUNTYFP20", "TRACTCE20"]):
        if rec[0] == "001":
            ct_tracts.add(rec[1])
    rows = []
    for ch in pd.read_csv(src, sep="|", usecols=["GEO_ID"] + B08301_COLS, dtype={"GEO_ID": str}, chunksize=200_000):
        cty = ch[ch["GEO_ID"].str.startswith("0500000US")]
        cty = cty.assign(fips=cty["GEO_ID"].str[9:14])
        rows.append(cty[cty["fips"].isin(COUNTIES)])
        tr = ch[ch["GEO_ID"].str.startswith("1400000US09")]
        tr = tr[tr["GEO_ID"].str[14:20].isin(ct_tracts)]
        if len(tr):
            s = tr[B08301_COLS].sum().to_frame().T
            s["fips"] = "09001"
            rows.append(s)
    d = pd.concat(rows, ignore_index=True).groupby("fips")[B08301_COLS].sum()
    assert len(d) == len(COUNTIES), d.index.tolist()
    d.to_csv(cache)
    return d


def cmd_targets():
    os.makedirs(WORK, exist_ok=True)
    net = json.load(open(os.path.join(WORK, "nyc_real.json")))
    st_idx = {s["key"]: i for i, s in enumerate(net["stations"])}
    out = {"sources": {}}
    # 1. subway entries and exits per complex per period: MTA OD estimate, Wednesdays of Oct 2024
    od = pd.read_csv(OD_SUBWAY, usecols=["hour_of_day", "origin_station_complex_id", "destination_station_complex_id", "estimated_average_ridership"],
                     dtype={"origin_station_complex_id": str, "destination_station_complex_id": str})
    od["q"] = od["hour_of_day"].map(period_of_hour)
    ent = od.groupby(["origin_station_complex_id", "q"])["estimated_average_ridership"].sum().unstack(fill_value=0)
    ext = od.groupby(["destination_station_complex_id", "q"])["estimated_average_ridership"].sum().unstack(fill_value=0)
    by_hour = od.groupby("hour_of_day")["estimated_average_ridership"].sum()
    mta = pd.read_csv(MTA_STATIONS, dtype=str)
    boro = mta.groupby("Complex ID")["Borough"].first().to_dict()
    sub = {}
    unmatched = 0.0
    for cid in sorted(set(ent.index) | set(ext.index)):
        k = "mta:" + cid
        e = [float(ent.loc[cid].get(q, 0)) if cid in ent.index else 0.0 for q in range(5)]
        x = [float(ext.loc[cid].get(q, 0)) if cid in ext.index else 0.0 for q in range(5)]
        if k not in st_idx:
            unmatched += sum(e)
            continue
        sub[k] = {"complex": cid, "borough": boro.get(cid, "?"), "entries": e, "exits": x}
    out["subway_od"] = {"stations": sub, "by_hour": by_hour.round(0).tolist(), "total": float(od["estimated_average_ridership"].sum()),
                        "unmatched_entries": unmatched}
    out["sources"]["subway_od"] = "MTA Subway Origin-Destination Ridership Estimate 2024 (data.ny.gov jsu2-fbtj), Wednesdays of October 2024, via maps/riders/nycriders"
    print(f"subway OD: {out['subway_od']['total']:,.0f} trips a day, {len(sub)} complexes matched, {unmatched:,.0f} entries unmatched")
    # 2. busiest hour per directed station pair: nycriders' RAPTOR assignment of the same OD
    stats = json.load(open(NYCRIDERS_STATS))
    mta_cx = dict(zip(mta["GTFS Stop ID"], mta["Complex ID"]))
    links = defaultdict(lambda: np.zeros(24))
    cover = defaultdict(set)
    for sgm in stats["segments"]:
        a, b = mta_cx.get(sgm["from"]), mta_cx.get(sgm["to"])
        if not a or not b or a == b:
            continue
        ia, ib = "mta:" + a, "mta:" + b
        if ia not in st_idx or ib not in st_idx:
            continue
        links[(ia, ib)] += np.array([h[0] for h in sgm["by_hour"]])
        # nycriders splits an express hop over the local stretches it runs along ("parents"),
        # so a link's riders are everyone passing there; the model's hops are spread the same way
        for pa, pb in sgm["parents"]:
            xa, xb = mta_cx.get(pa), mta_cx.get(pb)
            if xa and xb and xa != xb:
                cover[("mta:" + xa, "mta:" + xb)].add((ia, ib))
    out["subway_links"] = [{"a": a, "b": b, "by_hour": v.round(0).tolist()} for (a, b), v in links.items()]
    out["subway_cover"] = [{"hop": list(k), "links": sorted(list(v))} for k, v in cover.items()]
    out["sources"]["subway_links"] = "maps/riders/nycriders stats.json: the same OD routed over the GTFS timetable (RAPTOR), riders per segment per hour"
    print(f"subway links: {len(links)} directed station pairs")
    # 3. commute mode share by county of residence: ACS 2019-2023 B08301
    acs = acs_counties()
    out["acs"] = {f: {"name": COUNTIES[f], **{c[-3:]: float(acs.loc[f, c]) for c in B08301_COLS}} for f in COUNTY_ORDER}
    out["sources"]["acs"] = "ACS 2019-2023 5-year B08301, table-based summary file; Fairfield CT from its 2020 tracts"
    # 4. weekday totals by operator
    import io
    import requests
    resp = requests.get(MTA_DAILY, headers={"User-Agent": "trainworld-pipeline/1.0"}, timeout=60)
    resp.raise_for_status()
    md = pd.read_csv(io.StringIO(resp.text))
    md["date"] = pd.to_datetime(md["date"])
    wed = md[md["date"].dt.dayofweek == 2]
    out["operators"] = {
        "subway": float(wed["subways_total_estimated_ridership"].mean()),
        "sir": float(wed["staten_island_railway_total_estimated_ridership"].mean()),
        "lirr": float(wed["lirr_total_estimated_ridership"].mean()),
        "mnr": float(wed["metro_north_total_estimated_ridership"].mean()),
        "bus_mta": float(wed["buses_total_estimated_ridersip"].mean()),
        "path": PATH_WEEKDAY,
        "njt_rough": NJT_WEEKDAY_ROUGH,
    }
    out["sources"]["operators"] = "MTA Daily Ridership (data.ny.gov vxuj-8kew), Wednesdays of October 2024; PATH Oct 2024 weekday 211,525 (PANYNJ via Progressive Railroading); NJT rail rough"
    print("operators:", {k: round(v) for k, v in out["operators"].items()})
    p = os.path.join(WORK, "targets.json")
    json.dump(out, open(p, "w"))
    print("->", p)


def simplify(pts, tol):
    """Douglas-Peucker on an (n, 2) array; keeps the ends."""
    if len(pts) < 3:
        return pts
    keep = np.zeros(len(pts), bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = pts[i], pts[j]
        ab = b - a
        L = np.hypot(*ab)
        seg = pts[i + 1:j] - a
        d = np.abs(ab[0] * seg[:, 1] - ab[1] * seg[:, 0]) / L if L > 0 else np.hypot(seg[:, 0], seg[:, 1])
        k = int(np.argmax(d))
        if d[k] > tol:
            m = i + 1 + k
            keep[m] = True
            stack += [(i, m), (m, j)]
    return pts[keep]


def cmd_geom():
    """Track geometry for the save (T-007): per line, a node per stop on the line's own GTFS shape
    nearest its station, and the shape between consecutive stops, simplified to 15 m. A hop with no
    shape near both ends is a straight line."""
    city = sys.argv[2] if len(sys.argv) > 2 else "nyc"
    o = origin(CITIES[city]["pack"])
    net = json.load(open(CITIES[city]["out"]))
    st = net["stations"]
    shapes_by_route = {}
    for src in CITIES[city]["sources"]:
        if src.get("adapter", "gtfs") != "gtfs":  # no shapes: straight hops
            continue
        name, folder = src["name"], src["folder"]
        sh = read(folder, "shapes.txt", usecols=["shape_id", "shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"])
        if sh is None:  # London's National Rail cut has none: straight hops
            continue
        sh["seq"] = sh["shape_pt_sequence"].astype(float)
        sh = sh.sort_values(["shape_id", "seq"])
        x, y = to_xy(sh["shape_pt_lon"].astype(float).to_numpy(), sh["shape_pt_lat"].astype(float).to_numpy(), o)
        sh["x"], sh["y"] = x, y
        pts = {sid: g[["x", "y"]].to_numpy() for sid, g in sh.groupby("shape_id", sort=False)}
        tr = read(folder, "trips.txt", usecols=["route_id", "shape_id"])
        for r, g in tr.groupby("route_id"):
            ids = [s for s in g["shape_id"].unique() if s in pts]
            # longest shapes first; a short turn's shape is a piece of a longer one
            shapes_by_route[(name, r)] = sorted((pts[s] for s in ids), key=len, reverse=True)
        print(f"{name}: {len(pts)} shapes")
    out = []
    n_straight = 0
    for l in net["lines"]:
        shapes = [np.array(l["shape"])] if l.get("shape") else shapes_by_route.get((l["feed"], l["route"]), [])
        xy = [np.array([st[k]["x"], st[k]["y"]]) for k in l["stops"]]
        # the shape that passes nearest all stops of the line
        best, best_d = None, np.inf
        for s in shapes[:40]:
            d = sum(np.min(np.hypot(*(s - p).T)) for p in xy)
            if d < best_d:
                best, best_d = s, d
        nodes, hops = [], []
        idx = []
        for p in xy:
            if best is not None:
                dd = np.hypot(*(best - p).T)
                i = int(np.argmin(dd))
                idx.append(i if dd[i] < 400 else None)
                nodes.append(best[i].tolist() if dd[i] < 400 else p.tolist())
            else:
                idx.append(None)
                nodes.append(p.tolist())
        for k in range(len(xy) - 1):
            i, j = idx[k], idx[k + 1]
            if i is not None and j is not None and abs(j - i) >= 2:
                seg = best[min(i, j):max(i, j) + 1]
                if j < i:
                    seg = seg[::-1]
                seg = simplify(seg, 15.0)
                hops.append(seg[1:-1].round(1).tolist())
            else:
                hops.append([])
                if i is None or j is None:
                    n_straight += 1
        out.append({"nodes": nodes, "hops": hops})
    # one level per line, chosen so that as few line pairs as possible cross on the same level
    # (each line has its own track here; a same-level crossing would be a flat crossing)
    from shapely import STRtree
    from shapely.geometry import LineString
    geoms = []
    for g in out:
        pts = [g["nodes"][0]]
        for k, h in enumerate(g["hops"]):
            pts += h + [g["nodes"][k + 1]]
        geoms.append(LineString(pts))
    tree = STRtree(geoms)
    cross = defaultdict(int)
    for i, g in enumerate(geoms):
        for j in tree.query(g):
            j = int(j)
            if j <= i:
                continue
            x = g.intersection(geoms[j])
            n = len(getattr(x, "geoms", [x])) if not x.is_empty else 0
            if n:
                cross[(i, j)] = n
                cross[(j, i)] = n
    nbr = defaultdict(list)
    for (i, j), n in cross.items():
        nbr[i].append((j, n))
    levels = [None] * len(out)
    palette = [-1, -2, -3, 1, 2, 3]
    for i in sorted(range(len(out)), key=lambda i: -sum(n for _, n in nbr[i])):
        cost = {lv: sum(n for j, n in nbr[i] if levels[j] == lv) for lv in palette}
        levels[i] = min(palette, key=lambda lv: (cost[lv], palette.index(lv)))
    left = sum(n for (i, j), n in cross.items() if i < j and levels[i] == levels[j])
    for g, lv in zip(out, levels):
        g["level"] = lv
    print(f"{sum(1 for (i, j) in cross if i < j)} crossing line pairs; {left} crossing points left on one level")
    p = os.path.join(WORK, f"{city}_real_geom.json")
    json.dump({"lines": out}, open(p, "w"))
    print(f"{len(out)} lines, {n_straight} hops straight for want of a shape -> {p}")


def cmd_gamesave():
    """Wrap the track save (`realsave` bin) as a game file the app's "Load a file" reads (T-029's
    format: gzip of "TWG1", u32 header length, JSON header, the track save): day 1 07:00, $6B, the fleet the
    lines need, the starting fares."""
    import gzip
    import struct
    city = sys.argv[2] if len(sys.argv) > 2 else "nyc"
    track = open(os.path.join(WORK, f"{city}_real.twt2"), "rb").read()
    tj = json.load(open(os.path.join(WORK, f"{city}_real_track.json"), encoding="utf-8"))
    # a save's "city" is the world's first registry city whatever city it shows (saveFile.ts
    # isNortheastSave refuses anything else)
    reg = json.load(open(os.path.join(ROOT, "app", "src", "game", "registry.json"), encoding="utf-8"))
    header = {"game": "anitabuilder", "format": 1, "city": reg["cities"][0]["id"], "clock": 86400 + 7 * 3600, "cash": 6000.0,
              "fleet": int(tj["fleet"]), "fares": {"base": 1.5, "perKm": 0.1}, "ledger": []}
    hj = json.dumps(header).encode()
    raw = b"TWG1" + struct.pack("<I", len(hj)) + hj + track
    p = os.path.join(ROOT, "T-007-real-nyc.save") if city == "nyc" else os.path.join(WORK, f"{city}_real.save")
    with open(p, "wb") as f:
        f.write(gzip.compress(raw, mtime=0))
    print(f"{len(raw):,} bytes raw, {os.path.getsize(p):,} gzipped, fleet {header['fleet']:,} cars -> {p}")


# T-731: the cities a sandbox can start from their real network (the app's REAL_NETWORKS, ui/NewGame.tsx)
SHIPPED = ["nyc", "lon", "par", "tyo", "seo", "tor", "sdj"]


def cmd_ship():
    """Copy each shipped city's game file to data/packs/real/<id>.save, beside the packs the build
    copies. Its header's "city" is set to the registry's first id (an older gamesave wrote the whole
    registry entry there, which the app refuses)."""
    import gzip
    import struct
    reg = json.load(open(os.path.join(ROOT, "app", "src", "game", "registry.json"), encoding="utf-8"))
    out = os.path.join(ROOT, "data", "packs", "real")
    os.makedirs(out, exist_ok=True)
    for city in SHIPPED:
        src = os.path.join(ROOT, "T-007-real-nyc.save") if city == "nyc" else os.path.join(WORK, f"{city}_real.save")
        raw = gzip.decompress(open(src, "rb").read())
        n = struct.unpack("<I", raw[4:8])[0]
        header = json.loads(raw[8:8 + n])
        header["city"] = reg["cities"][0]["id"]
        hj = json.dumps(header).encode()
        p = os.path.join(out, f"{city}.save")
        with open(p, "wb") as f:
            f.write(gzip.compress(b"TWG1" + struct.pack("<I", len(hj)) + hj + raw[8 + n:], mtime=0))
        print(f"{city}: {os.path.getsize(p):,} bytes -> {p}")


OVERPASS = "https://overpass-api.de/api/interpreter"


def cmd_fetch_osm():
    """Rail route relations (train, subway, light rail, tram, monorail) in the pack's boundary box,
    with their stops, platforms (as centres), route masters and stop areas -> data/raw/t509/osm_<pack>.json."""
    import requests
    city = sys.argv[2]
    pack = CITIES[city]["pack"] if city in CITIES else city
    w, s, e, n = boundary(pack).bounds
    q = f"""[out:json][timeout:300];
rel["type"="route"]["route"~"^(train|subway|light_rail|tram|monorail)$"]({s},{w},{n},{e})->.r;
.r out body;
node(r.r)->.n;
.n out body;
way(r.r:"platform")->.w;
.w out center tags;
rel(br.r)["type"="route_master"];
out body;
rel(bn.n)["public_transport"="stop_area"];
out body;
rel(bw.w)["public_transport"="stop_area"];
out body;"""
    r = requests.post(OVERPASS, data={"data": q}, headers={"User-Agent": "trainworld-pipeline/1.0 (rail network test sets)"}, timeout=400)
    r.raise_for_status()
    p = os.path.join(RAW, "t509", f"osm_{pack}.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "wb").write(r.content)
    d = r.json()
    print(f"{len(d['elements'])} elements ({sum(1 for x in d['elements'] if x.get('tags', {}).get('type') == 'route')} routes) -> {p}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "net"
    {"net": cmd_net, "fetch-osm": cmd_fetch_osm, "counties": cmd_counties, "targets": cmd_targets, "geom": cmd_geom, "gamesave": cmd_gamesave, "ship": cmd_ship}[cmd]()
