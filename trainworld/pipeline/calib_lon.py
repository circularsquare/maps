"""T-509 stage 3: London's TfL rail as a `trips` CSV for realnet.py (`net lon`).

    python pipeline/calib_lon.py trips     # TfL timetables -> data/raw/t509/lon_tfl_trips.csv
    python pipeline/realnet.py net lon     # the network (CITIES["lon"])
    python pipeline/calib_lon.py targets [--pack dir]  # NUMBAT, census 2011 -> data/work/calib/lon/inputs.json, cell_lad.u8
    python pipeline/calib.py targets lon   # calls targets() here

Reads Anita's londonriders project (maps/riders/londonriders), which already parses a typical
weekday of every TfL rail line: the Tube and DLR from railwaydata.co.uk's working timetable PDFs
(`railwaydata.py`), the Elizabeth line and the six Overground lines from Network Rail's SCHEDULE
feed (`cif.py`, data/ezl.json.gz and og.json.gz). Stops are NaPTAN ids; each becomes its NUMBAT
station (`geometry.json` naptan_to_mnlc), so stations are the ones NUMBAT counts entries at and
the CSV's `stop` is the NUMBAT station code (realnet's `target`).

The CSV has no direction from the sources: a trip is direction 0 when the stops it shares with
the first reference pattern of its line come in that pattern's order, 1 when reversed; a trip
sharing fewer than two stops with every reference becomes a new reference.
"""
import csv
import datetime
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LR = os.path.join(os.path.dirname(ROOT), "riders", "londonriders")
OUT = os.path.join(ROOT, "data", "raw", "t509", "lon_tfl_trips.csv")
# the SCHEDULE extracts were downloaded in May 2026; a Wednesday they cover
CIF_DATE = datetime.date(2026, 5, 13)
NAMES = {"BAK": "Bakerloo", "CEN": "Central", "CIR": "Circle", "DIS": "District", "HAM": "Hammersmith & City",
         "JUB": "Jubilee", "MET": "Metropolitan", "NOR": "Northern", "PIC": "Piccadilly", "VIC": "Victoria",
         "WAC": "Waterloo & City", "DLR": "DLR", "EZL": "Elizabeth"}


def direction(line_refs, ks):
    for ref in line_refs:
        pos = {k: i for i, k in enumerate(ref)}
        shared = [pos[k] for k in ks if k in pos]
        if len(shared) >= 2:
            return "0" if shared[-1] > shared[0] else "1"
    line_refs.append(list(ks))
    return "0"


def main():
    os.chdir(LR)
    sys.path.insert(0, LR)
    import build
    import cif
    import railwaydata
    geo = build.load_geometry()
    trips = railwaydata.parse_all_pdfs()
    trips += cif.parse_cif_file(cif.CIF_EZL, ["elizabeth"], og_classify=False, target_date=CIF_DATE)
    trips += cif.parse_cif_file(cif.CIF_OG, cif.OG_LINES, og_classify=True, target_date=CIF_DATE)
    build.patch_missing_stops(trips)
    # the PDFs repeat a train as pieces and from several pages: londonriders' own clean-up
    trips = build.dedupe_same_endpoint_trips(build.dedupe_prefix_trips(trips))
    n2m = geo["naptan_to_mnlc"]
    st = geo["stations"]
    refs = {}
    rows, miss = [], set()
    for i, t in enumerate(sorted(trips, key=lambda t: -len(t["stops"]))):
        ks = []
        for nid, s in t["stops"]:
            m = n2m.get(nid)
            if m is None or str(m) not in st:
                miss.add(nid)
                continue
            if ks and ks[-1][0] == str(m):
                continue
            ks.append((str(m), s))
        if len(ks) < 2:
            continue
        d = direction(refs.setdefault(t["line"], []), [k for k, _ in ks])
        for m, s in ks:
            v = st[m]
            rows.append([t["line"], d, f"t{i}", m, v["name"], v["lon"], v["lat"], s, NAMES.get(t["line"], t.get("tfl_id", t["line"]))])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["route", "direction", "trip", "stop", "name", "lon", "lat", "time", "route_name"])
        w.writerows(rows)
    print(f"{len(trips)} trips, {len(rows)} stop rows -> {OUT}; {len(miss)} NaPTAN ids without a NUMBAT station: {sorted(miss)[:20]}")


TIER0 = os.path.join(ROOT, "data", "raw", "tier0")
CALIB = os.path.join(ROOT, "data", "work", "calib", "lon")
NUMBAT = os.path.join(LR, "data", "numbat_twt_outputs.xlsx")
QS701 = ("https://www.nomisweb.co.uk/api/v01/dataset/NM_568_1.data.csv?geography=TYPE297&rural_urban=0&cell=1...11"
         "&measures=20100&select=geography_code,cell,obs_value&recordlimit=25000&recordoffset={}")


def cmd_targets():
    """NUMBAT 2024 (Tue-Thu, autumn) station entries and line boarders; Census 2011 QS701EW method
    of travel to work by MSOA, summed to 2023 local authorities by MSOA centroid, with each pack
    cell's authority -> data/work/calib/lon/inputs.json and cell_lad.u8. `--pack <dir>` reads the
    pack from a copy (cells must match the pack the run uses)."""
    import io
    import json
    import numpy as np
    import openpyxl
    import pandas as pd
    import requests
    from shapely import contains_xy, make_valid
    from shapely.geometry import shape
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import read_pack
    import realnet as rn
    os.makedirs(CALIB, exist_ok=True)
    wb = openpyxl.load_workbook(NUMBAT, read_only=True)
    st = []
    for r in wb["Station_Entries"].iter_rows(min_row=4, values_only=True):
        if r[0] is not None and r[4]:
            st.append({"target": str(r[0]), "name": r[2], "entries": float(r[4])})
    # National Rail stations: ORR 2024-25 entries and exits, halved, over ORR_DAYS weekday equivalents
    orr = pd.read_csv(os.path.join(GB, "orr_table1410.csv"), skiprows=3, dtype=str, keep_default_na=False)
    orr.columns = [c.replace("\n", " ").strip() for c in orr.columns]
    ee = [c for c in orr.columns if c.startswith("Entries and exits") and "All tickets" in c][0]
    tlc = [c for c in orr.columns if c.startswith("Three Letter Code")][0]
    for _, r in orr.iterrows():
        try:
            v = float(r[ee].replace(",", ""))
        except ValueError:
            continue
        st.append({"target": r[tlc], "name": r["Station name"], "entries": v / 2 / ORR_DAYS, "orr": True})
    lines = {}
    for r in wb["Line_Boarders"].iter_rows(min_row=4, values_only=True):
        if r[1] and r[1] not in ("TfL", "TRM") and r[3]:
            lines[r[1]] = float(r[3])
    # census 2011 by MSOA (cells 1 home, 2 underground/metro/light rail/tram, 3 train, 4-11 the rest)
    p = os.path.join(TIER0, "qs701ew_msoa.csv")
    if not os.path.exists(p):
        parts = []
        for off in range(0, 200000, 25000):
            t = requests.get(QS701.format(off), headers={"User-Agent": "trainworld-pipeline/1.0"}, timeout=120).text
            d = pd.read_csv(io.StringIO(t))
            parts.append(d)
            if len(d) < 25000:
                break
        pd.concat(parts).to_csv(p, index=False)
    q = pd.read_csv(p).pivot_table(index="GEOGRAPHY_CODE", columns="CELL", values="OBS_VALUE", aggfunc="sum")
    pk = sys.argv[sys.argv.index("--pack") + 1] if "--pack" in sys.argv else read_pack.PACKS
    header, a = read_pack.load_pack("lon", pk)
    bnd = rn.boundary("lon")
    lads = json.load(open(os.path.join(TIER0, "lad_dec2023_uk_bgc.geojson")))["features"]
    lads = [(f["properties"]["LAD23CD"], f["properties"]["LAD23NM"], make_valid(shape(f["geometry"]))) for f in lads if f["properties"]["LAD23CD"][0] == "E"]
    lads = [l for l in lads if l[2].intersects(bnd)]
    cent = json.load(open(os.path.join(TIER0, "msoa2011_pwc_lon.json")))
    mc = np.array([f["geometry"]["coordinates"] for f in cent])
    mid = [f["properties"]["msoa11cd"] for f in cent]
    inside = contains_xy(bnd, mc[:, 0], mc[:, 1])
    lon0, lat0 = header["origin"]["lon"], header["origin"]["lat"]
    x, y = a["x_m"].astype(float), a["y_m"].astype(float)
    clon = lon0 + np.degrees(x / (rn.R_EARTH * np.cos(np.radians(lat0))))
    clat = lat0 + np.degrees(y / rn.R_EARTH)
    areas, cell = [], np.full(len(x), 255, np.uint8)
    for code, name, g in lads:
        m = contains_xy(g, mc[:, 0], mc[:, 1]) & inside
        ms = [mid[i] for i in np.where(m)[0] if mid[i] in q.index]
        if not ms:
            continue
        s = q.loc[ms].sum()
        comm = float(s[list(range(2, 12))].sum())
        if comm < 20000:
            continue
        cm = contains_xy(g, clon, clat) & (cell == 255)
        if not cm.any():
            continue
        cell[cm] = len(areas)
        areas.append({"name": name, "code": code, "rail_share": float(s[2] + s[3]) / comm, "commuters": comm,
                      "underground_share": float(s[2]) / comm, "msoas": len(ms)})
    cell.tofile(os.path.join(CALIB, "cell_lad.u8"))
    out = {"stations": st, "lines": lines, "areas": areas, "pack": pk,
           "sources": {"stations": "TfL NUMBAT 2024 (published Oct 2025), Station_Entries, typical autumn Tue-Thu, gateline entries at every LU, LO, DLR and Elizabeth line station",
                       "lines": "NUMBAT 2024 Line_Boarders, Tue-Thu (all boarders by line, interchanges included; H&C and Circle together)",
                       "areas": "Census 2011 QS701EW method of travel to work by MSOA (Nomis NM_568_1), summed to Dec 2023 local authorities by MSOA population centroid inside the pack; rail = underground, metro, light rail, tram and train over workers less those working at home (2021's TS061 was taken in lockdown)"}}
    json.dump(out, open(os.path.join(CALIB, "inputs.json"), "w"), indent=0)
    print(f"{len(st)} stations, {len(lines)} lines, {len(areas)} areas ({(cell != 255).sum()} of {len(cell)} cells) -> {CALIB}")


GB = os.path.join(ROOT, "data", "raw", "t509", "gb")
# ORR Table 1410 (https://dataportal.orr.gov.uk/media/1909/table-1410-passenger-entries-and-exits-and-interchanges-by-station.csv),
# April 2024 to March 2025; a weekday is taken as 1/305 of a year (judgement: weekdays a little
# busier than the 365-day mean, weekends much quieter)
ORR_DAYS = 305.0
GB_ZIP = os.path.join(GB, "great_britain_gtfs.zip")  # https://beta.aubin.app/gtfs/great_britain_gtfs.zip (Transitous's GB source)
NR = os.path.join(GB, "nr_lse")
# National Rail operators running suburban services in the pack (London and the South East);
# TfL's own (LO, XR) come from londonriders, intercity-only operators are left out
NR_TOCS = {"SN": "Southern", "GX": "Gatwick Express", "SE": "Southeastern", "SW": "South Western Railway", "TL": "Thameslink",
           "GN": "Great Northern", "CC": "c2c", "LE": "Greater Anglia", "CH": "Chiltern", "LM": "West Midlands / LNR",
           "GW": "GWR", "HX": "Heathrow Express"}


def cmd_nr():
    """The GB GTFS cut to NR_TOCS' rail routes with a stop inside the pack's box -> data/raw/t509/gb/nr_lse/,
    plus a group table (stop -> CRS, the ORR's station code) for realnet."""
    import zipfile
    import pandas as pd
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import realnet as rn
    z = zipfile.ZipFile(GB_ZIP)
    rd = lambda n, **kw: pd.read_csv(z.open(n), dtype=str, keep_default_na=False, **kw)
    os.makedirs(NR, exist_ok=True)
    routes = rd("routes.txt")
    agency = rd("agency.txt")
    print(agency[["agency_id", "agency_name"]].to_string()[:3000])
    keep = routes[routes["route_type"].isin(["2", "100", "101", "102", "103", "106", "109"]) & routes["agency_id"].str.lstrip("=").isin(NR_TOCS)]
    keep.to_csv(os.path.join(NR, "routes.txt"), index=False)
    agency[agency["agency_id"].str.lstrip("=").isin(NR_TOCS)].to_csv(os.path.join(NR, "agency.txt"), index=False)
    trips = rd("trips.txt")
    trips = trips[trips["route_id"].isin(keep["route_id"])]
    stops = rd("stops.txt")
    w, s, e, n = rn.boundary("lon").bounds
    lon, lat = stops["stop_lon"].astype(float), stops["stop_lat"].astype(float)
    box = set(stops.loc[(lon > w) & (lon < e) & (lat > s) & (lat < n), "stop_id"])
    tids = set(trips["trip_id"])
    out = os.path.join(NR, "stop_times.txt")
    first, used, inbox = True, set(), set()
    for ch in pd.read_csv(z.open("stop_times.txt"), dtype=str, keep_default_na=False, chunksize=2_000_000,
                          usecols=["trip_id", "arrival_time", "departure_time", "stop_id", "stop_sequence"]):
        ch = ch[ch["trip_id"].isin(tids)]
        inbox |= set(ch.loc[ch["stop_id"].isin(box), "trip_id"])
        ch.to_csv(out, index=False, mode="w" if first else "a", header=first)
        first = False
        print(".", end="", flush=True)
    st = pd.read_csv(out, dtype=str, keep_default_na=False)
    st = st[st["trip_id"].isin(inbox)]
    st.to_csv(out, index=False)
    used = set(st["stop_id"])
    trips = trips[trips["trip_id"].isin(inbox)].copy()
    # the feed has no direction_id; a trip and its reverse must share one (realnet folds by it)
    st["seq"] = st["stop_sequence"].astype(int)
    ends = st.sort_values(["trip_id", "seq"]).groupby("trip_id")["stop_id"].agg(["first", "last"])
    pstat = dict(zip(stops["stop_id"], stops["parent_station"])) if "parent_station" in stops else {}
    f = ends["first"].map(lambda s: pstat.get(s) or s)
    l = ends["last"].map(lambda s: pstat.get(s) or s)
    trips["direction_id"] = trips["trip_id"].map((f > l).map({True: "1", False: "0"}))
    trips.to_csv(os.path.join(NR, "trips.txt"), index=False)
    sids = set(trips["service_id"])
    names = {i.filename for i in z.infolist()}
    for nm in ("calendar.txt", "calendar_dates.txt"):
        if nm in names:
            c = rd(nm)
            c[c["service_id"].isin(sids)].to_csv(os.path.join(NR, nm), index=False)
    par = set(stops.loc[stops["stop_id"].isin(used), "parent_station"]) - {""} if "parent_station" in stops else set()
    stops[stops["stop_id"].isin(used | par)].to_csv(os.path.join(NR, "stops.txt"), index=False)
    print(f"\n{len(keep)} routes, {len(trips)} trips in the box, {len(used)} stops -> {NR}")
    cmd_nr_groups()


def cmd_nr_groups():
    """Parent stations (910G...) -> their CRS (`stop_code`), the code ORR's counts use."""
    import pandas as pd
    s = pd.read_csv(os.path.join(NR, "stops.txt"), dtype=str, keep_default_na=False)
    p = s[(s["location_type"] == "1") & (s["stop_code"] != "")]
    pd.DataFrame({"stop_id": p["stop_id"], "crs": p["stop_code"], "name": p["stop_name"], "lon": p["stop_lon"], "lat": p["stop_lat"]}).to_csv(
        os.path.join(NR, "crs_groups.csv"), index=False)
    print(f"{len(p)} stations with a CRS -> crs_groups.csv")


def targets():
    """calib.py's targets for London from inputs.json (`targets` above). Operators are NUMBAT's
    line codes. Hubs: the main line termini and Stratford, where riders change from National
    Rail trains (any counted station within hub_m of a station the counts miss is one anyway)."""
    import json
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import realnet as rn
    d = json.load(open(os.path.join(CALIB, "inputs.json")))
    net = rn.CITIES["lon"]["out"]
    termini = ("Waterloo", "Victoria", "London Bridge", "Liverpool Street", "King's Cross", "Euston", "Paddington",
               "Charing Cross", "Cannon Street", "Fenchurch Street", "Marylebone", "Stratford", "Clapham Junction")
    hubs = [s["target"] for s in d["stations"] if s["name"].startswith(termini) or (s.get("orr") and s["name"].startswith("London "))]
    # a TfL station a National Rail station merged into: NUMBAT counts only TfL's gates there
    if os.path.exists(net):
        hubs += [s["target"] for s in json.load(open(net))["stations"] if s.get("target") and " / " in s["name"]]
    return {"city": "lon", "net": net, "areas_file": os.path.join(CALIB, "cell_lad.u8"),
            "pack": os.path.join(d["pack"], "lon.json"), "station_min": 2000.0, "hub_m": 300.0,
            "stations": d["stations"], "hubs": hubs, "operators": d["lines"], "operators_info": {},
            "areas": d["areas"], "sources": d["sources"]}


if __name__ == "__main__":
    {"trips": main, "nr": cmd_nr, "nr-groups": cmd_nr_groups, "targets": cmd_targets}[sys.argv[1] if len(sys.argv) > 1 else "trips"]()
