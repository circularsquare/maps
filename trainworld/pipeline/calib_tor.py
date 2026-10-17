"""T-509 stage 3: Toronto (`tor`, the hold-out city): real network and calibration targets.

    python pipeline/calib_tor.py gtfs      # TTC, GO, UP feeds cut to rail -> data/raw/t509/tor/
    python pipeline/calib_tor.py groups    # TTC stops -> stations -> data/work/realnet/tor_ttc_groups.csv
    python pipeline/realnet.py net tor     # the network (CITIES["tor"])
    python pipeline/calib_tor.py areas     # pack cell -> census tract -> data/work/realnet/tor_cell_ct.u16
    python pipeline/calib.py targets tor   # calls targets() here

Sources (data/raw/t509/tor/, all open downloads):
- TTC GTFS (City of Toronto open data, "TTC Routes and Schedules", Sep-Oct 2026 service), kept to
  its rail routes: Lines 1, 2, 4, the Line 5 and 6 light rail and the 5xx streetcars (the 3xx
  night streetcars left out). The census puts streetcars with light rail and commuter trains,
  so they are in the network to keep the mode share comparable.
- GO Transit GTFS (Metrolinx, rail routes only) and UP Express GTFS (Metrolinx).
- TTC "Subway Ridership, 2023-2024": per subway line, typical weekday customer boardings; per
  station platform, customers travelling to and from it (both ways, transfers included), Sep
  2023 - Aug 2024. Transcribed below (one page).
- Census 2021 main mode of commuting by census tract (StatCan 98-10-0480-01, already on disk for
  the pack, data/raw/ca/98100480-eng.zip); tract polygons from the pack's ArcGIS copies.
  2021 is the lockdown census (May 2021): transit shares are far below 2016's and 2024's.
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import realnet as rn  # noqa: E402

RAW = os.path.join(rn.RAW, "t509", "tor")
GROUPS = os.path.join(rn.WORK, "tor_ttc_groups.csv")
CELL_CT = os.path.join(rn.WORK, "tor_cell_ct.u16")
CT_MODE = os.path.join(RAW, "ct_mode_2021.csv")
SUBWAY_ROUTES = {"1", "2", "4", "5", "6"}

# TTC Subway Ridership 2023-2024: station platform usage (to and from, typical weekday)
TTC_LINE_BOARDINGS = {"1": 625467, "2": 403582, "4": 38705}
TTC_USAGE = {
    "1": {"Bloor-Yonge": 156643, "College": 39137, "Davisville": 15903, "Downsview Park": 5618, "Dundas": 72406,
          "Dupont": 11084, "Eglinton": 60814, "Eglinton West": 13982, "Finch": 70775, "Finch West": 18345,
          "Glencairn": 5878, "Highway 407": 7649, "King": 35107, "Lawrence": 21197, "Lawrence West": 19742,
          "Museum": 9604, "North York Centre": 16699, "Osgoode": 19323, "Pioneer Village": 16570, "Queen": 36714,
          "Queen's Park": 34444, "Rosedale": 4875, "Sheppard West": 19495, "Sheppard-Yonge": 57501, "Spadina": 11479,
          "St. Andrew": 34576, "St. Clair": 27336, "St. Clair West": 21013, "St. George": 108866, "St. Patrick": 23989,
          "Summerhill": 5045, "Union": 136515, "Vaughan Metropolitan Centre": 20394, "Wellesley": 17705,
          "Wilson": 21579, "York Mills": 20498, "York University": 20447, "Yorkdale": 19725},
    "2": {"Bathurst": 30598, "Bay": 20980, "Bloor-Yonge": 121531, "Broadview": 11720, "Castle Frank": 8943,
          "Chester": 6197, "Christie": 11407, "Coxwell": 13450, "Donlands": 6317, "Dufferin": 26800,
          "Dundas West": 23861, "Greenwood": 9567, "High Park": 9173, "Islington": 25023, "Jane": 14953,
          "Keele": 16305, "Kennedy": 42881, "Kipling": 49392, "Lansdowne": 17406, "Main Street": 18682,
          "Old Mill": 6109, "Ossington": 22109, "Pape": 34506, "Royal York": 17337, "Runnymede": 15838,
          "Sherbourne": 24689, "Spadina": 27601, "St. George": 101128, "Victoria Park": 32276, "Warden": 21843,
          "Woodbine": 10803},
    "4": {"Bayview": 6205, "Bessarion": 3180, "Don Mills": 28709, "Leslie": 3988, "Sheppard-Yonge": 35327},
}
# GTFS station names (2026) that differ from the 2023-24 table's
RENAMED = {"Bloor": "Bloor-Yonge", "Yonge": "Bloor-Yonge", "TMU": "Dundas", "Cedarvale": "Eglinton West",
           "St Andrew": "St. Andrew", "St Clair": "St. Clair", "St Clair West": "St. Clair West",
           "St George": "St. George", "St Patrick": "St. Patrick", "Queens Park": "Queen's Park",
           "Sheppard-Yonge": "Sheppard-Yonge", "Sheppard": "Sheppard-Yonge", "Main": "Main Street"}
# bus terminals where riders changing from buses (counted by the TTC, not entries in the model)
# make most of the platform's use
BUS_HUBS = {"Finch", "Kipling", "Kennedy", "Don Mills", "Islington", "Warden", "Victoria Park", "Wilson",
            "Sheppard West", "York Mills", "Lawrence West", "Pioneer Village", "Vaughan Metropolitan Centre",
            "St. Clair West", "Downsview Park", "Eglinton"}


FEEDS = {"ttc": ("https://ckan0.cf.opendata.inter.prod-toronto.ca/dataset/7795b45e-e65a-4465-81fc-c36b9dfff169/resource/cfb6b2b8-6191-41e3-bda1-b175c51148cb/download/opendata_ttc_schedules.zip",
                 lambda r: r["route_id"].isin(["1", "2", "4", "5", "6"]) | r["route_short_name"].str.match(r"^5\d\d$")),
         "go_rail": ("https://assets.metrolinx.com/raw/upload/Documents/Metrolinx/Open%20Data/GO-GTFS.zip", lambda r: r["route_type"] == "2"),
         "up": ("https://assets.metrolinx.com/raw/upload/Documents/Metrolinx/Open%20Data/UP-GTFS.zip", lambda r: r["route_type"] == "2")}


def cmd_gtfs():
    """The three feeds, cut to their rail routes (with shapes, for `realnet.py geom`)."""
    import io
    import zipfile
    import requests
    for name, (url, keep) in FEEDS.items():
        z = zipfile.ZipFile(io.BytesIO(requests.get(url, headers={"User-Agent": "trainworld-pipeline/1.0"}, timeout=600).content))
        rd = lambda f: pd.read_csv(z.open(f), dtype=str, keep_default_na=False, encoding="utf-8-sig")  # noqa: E731
        dst = os.path.join(RAW, name)
        os.makedirs(dst, exist_ok=True)
        r = rd("routes.txt")
        r = r[keep(r)]
        t = rd("trips.txt")
        t = t[t["route_id"].isin(r["route_id"])]
        st = pd.concat(ch[ch["trip_id"].isin(set(t["trip_id"]))] for ch in pd.read_csv(
            z.open("stop_times.txt"), dtype=str, keep_default_na=False, chunksize=1_000_000, encoding="utf-8-sig"))
        s = rd("stops.txt")
        used = set(st["stop_id"])
        used |= set(s.loc[s["stop_id"].isin(used), "parent_station"]) - {""}
        sh = rd("shapes.txt")
        out = {"routes.txt": r, "trips.txt": t, "stop_times.txt": st, "stops.txt": s[s["stop_id"].isin(used)],
               "shapes.txt": sh[sh["shape_id"].isin(set(t["shape_id"]))]}
        for f in ("calendar.txt", "calendar_dates.txt", "agency.txt"):
            if f in z.namelist():
                out[f] = rd(f)
        for f, d in out.items():
            d.to_csv(os.path.join(dst, f), index=False)
        print(f"{name}: {len(r)} routes, {len(t)} trips -> {dst}")


def station_base(name):
    """'Bloor Station - Northbound Platform' -> 'Bloor'; None for a street stop."""
    m = re.match(r"^(.*?)\s+Station\b", name) or re.match(r"^(.*?) - \w+bound Platform$", name)
    if not m or " at " in m.group(1):
        m2 = re.search(r" - (.*?) Station$", name)
        return m2.group(1).strip() if m2 else None
    return m.group(1).strip()


def cmd_groups():
    """TTC stops into stations: a subway or light rail platform goes to its station by name (the
    two lines at Bloor-Yonge, St George, Spadina, Sheppard-Yonge as one); streetcar stops within
    80 m of each other (both sides of the street, crossing routes) as one stop."""
    f = os.path.join(RAW, "ttc")
    s = pd.read_csv(os.path.join(f, "stops.txt"), dtype=str, keep_default_na=False)
    s["lon"], s["lat"] = s["stop_lon"].astype(float), s["stop_lat"].astype(float)
    x, y = rn.to_xy(s["lon"].values, s["lat"].values, rn.origin("tor"))
    parent = list(range(len(s)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    base = [station_base(n) for n in s["stop_name"]]
    base = [RENAMED.get(b, b) if b else None for b in base]
    by_base = {}
    for i, b in enumerate(base):
        if b:
            j = by_base.setdefault(b, i)
            if np.hypot(x[i] - x[j], y[i] - y[j]) < 600:
                parent[find(i)] = find(j)
    street = [i for i, b in enumerate(base) if not b]
    for a in range(len(street)):
        for c in range(a + 1, len(street)):
            i, j = street[a], street[c]
            if (x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2 < 80 ** 2:
                parent[find(i)] = find(j)
    gid = {}
    rows = []
    for i in range(len(s)):
        r = find(i)
        b = next((base[k] for k in [r, i] if base[k]), None)
        g = gid.setdefault(r, b if b else f"st{s['stop_id'].iloc[r]}")
        rows.append({"stop_id": s["stop_id"].iloc[i], "group": g, "name": b or s["stop_name"].iloc[r],
                     "lon": s["lon"].iloc[i], "lat": s["lat"].iloc[i]})
    out = pd.DataFrame(rows)
    out.to_csv(GROUPS, index=False)
    known = set(TTC_USAGE["1"]) | set(TTC_USAGE["2"]) | set(TTC_USAGE["4"])
    named = set(out["group"]) - {g for g in out["group"] if g.startswith("st")}
    print(f"{len(s)} stops -> {out['group'].nunique()} stations ({len(named)} named); "
          f"table stations not found: {sorted(known - named)}; named not in table: {sorted(named - known)}")


def cmd_areas():
    """One u16 per pack cell: its 2021 census tract's index in the sorted tract list (65535 none)."""
    import read_pack
    from shapely import STRtree, points
    from shapely.geometry import shape
    header, a = read_pack.load_pack("tor")
    lon0, lat0 = header["origin"]["lon"], header["origin"]["lat"]
    x, y = a["x_m"].astype(float), a["y_m"].astype(float)
    lon = lon0 + np.degrees(x / (rn.R_EARTH * np.cos(np.radians(lat0))))
    lat = lat0 + np.degrees(y / rn.R_EARTH)
    polys, ids = [], []
    for fn in sorted(os.listdir(os.path.join(rn.RAW, "ca"))):
        if fn.startswith("ct_on_"):
            for ft in json.load(open(os.path.join(rn.RAW, "ca", fn)))["features"]:
                polys.append(shape(ft["geometry"]))
                ids.append(ft["properties"]["CTUID"])
    tree = STRtree(polys)
    pi, gi = tree.query(points(lon, lat), predicate="within")
    # only groups holding a cell: the calib bin counts areas up to the highest one used
    tracts = sorted({ids[g] for g in gi})
    order = sorted({tract_group(c) for c in tracts})
    idx = {c: i for i, c in enumerate(order)}
    out = np.full(len(x), 65535, np.uint16)
    out[pi] = [idx[tract_group(ids[g])] for g in gi]
    out.tofile(CELL_CT)
    json.dump({"groups": order, "tracts": tracts}, open(CELL_CT + ".ids.json", "w"))
    print(f"{len(x)} cells, {(out != 65535).sum()} in {len(tracts)} tracts, {len(order)} groups -> {CELL_CT}")


CMA_NAMES = {"535": "Toronto", "532": "Oshawa", "537": "Hamilton", "550": "Guelph", "568": "Barrie", "541": "Kitchener"}


def tract_group(ct):
    """A tract's area: its CMA and the hundreds of its tract number ('5350575.04' -> 'Toronto 05xx').
    Single tracts are too small for a share (many have no rail commuter at all, and the logit of
    zero dominates); tract numbers run by municipality, so a hundred of them is a district."""
    return f"{CMA_NAMES.get(ct[:3], ct[:3])} {ct[3:5]}xx"


def ct_mode():
    """Census 2021 commuters by main mode per census tract in the pack (cached)."""
    if os.path.exists(CT_MODE):
        return pd.read_csv(CT_MODE, dtype={"ct": str}).set_index("ct")
    want = set(json.load(open(CELL_CT + ".ids.json"))["tracts"])
    cols = ["GEO", "DGUID", "Time leaving for work (7)", "Commuting duration (7)", "Main mode of commuting (21)",
            "Place of work status (3A):Total - Place of work status[1]"]
    parts = []
    import zipfile
    with zipfile.ZipFile(os.path.join(rn.RAW, "ca", "98100480-eng.zip")) as z:
        with z.open("98100480.csv") as fh:
            for ch in pd.read_csv(fh, usecols=cols, dtype=str, chunksize=500_000, encoding="utf-8-sig"):
                ch = ch[(ch[cols[2]] == "Total - Time leaving for work") & (ch[cols[3]] == "Total - Commuting duration")
                        & ch["DGUID"].str.startswith("2021S0507")]
                ch = ch.assign(ct=ch["DGUID"].str[9:])
                parts.append(ch[ch["ct"].isin(want)])
    d = pd.concat(parts)
    d["n"] = pd.to_numeric(d[cols[5]], errors="coerce").fillna(0)
    t = d.pivot_table(index="ct", columns=cols[4], values="n", aggfunc="sum").fillna(0)
    t = t.rename(columns={"Total - Main mode of commuting": "total", "Subway or elevated rail": "subway",
                          "Light rail, streetcar or commuter train": "lrt_cr", "Bus": "bus", "Walked": "walk"})
    t = t[["total", "subway", "lrt_cr", "bus", "walk"]]
    t.to_csv(CT_MODE)
    return t


def targets():
    net = json.load(open(rn.CITIES["tor"]["out"]))
    in_net = {s["target"] for s in net["stations"] if s.get("target")}
    use = {}
    for ln, st in TTC_USAGE.items():
        for k, v in st.items():
            use[k] = use.get(k, 0) + v
    multi = {k for k in use if sum(k in TTC_USAGE[l] for l in TTC_USAGE) > 1}
    # platform use is to and from: half of it is boardings; at a two-line station half of it is
    # also changes between the lines, so those are hubs
    st = [{"target": k, "name": k, "entries": v / 2.0} for k, v in sorted(use.items()) if k in in_net]
    missing = sorted(set(use) - in_net)
    if missing:
        print("not in the network:", missing)
    hubs = sorted((multi | BUS_HUBS | {"Union"}) & in_net)
    groups = json.load(open(CELL_CT + ".ids.json"))["groups"]
    m = ct_mode()
    m = m.groupby([tract_group(c) for c in m.index]).sum()
    areas = []
    for c in groups:
        if c in m.index and m.loc[c, "total"] > 0:
            r = m.loc[c]
            areas.append({"name": c, "rail_share": float((r["subway"] + r["lrt_cr"]) / r["total"]), "commuters": float(r["total"]),
                          "walk_share": float(r["walk"] / r["total"])})
        else:
            areas.append({"name": c, "rail_share": 0.0, "commuters": 0.0})
    return {
        "city": "tor", "net": rn.CITIES["tor"]["out"], "areas_file": CELL_CT,
        "station_min": 2000.0, "hub_m": 300.0, "stations": st, "hubs": hubs,
        "operators": {"ttc_l" + k: float(v) for k, v in TTC_LINE_BOARDINGS.items()},
        # GO rail: 212,519 weekday boardings in Sep 2016 (GO ridership report); Metrolinx put GO at
        # about 100% of 2019 in July 2025, so roughly 2016-19's level. UP Express: 4,303,797 riders
        # Jun 2024 - May 2025 (Metrolinx), / 365 days; the model has no air passengers in the pack run
        "operators_info": {"go": 215000.0, "up": 11800.0},
        "areas": areas,
        "sources": {
            "stations": "TTC Subway Ridership 2023-2024 (typical weekday, Sep 2023 - Aug 2024): half of each platform's to-and-from use, platforms of a station summed; two-line stations, Union and the big bus terminals as hubs",
            "operators": "TTC 2023-24 typical weekday customer boardings per subway line (Lines 1, 2, 4), the model's trips using each line; GO and UP rough, not scored",
            "areas": "Census 2021 (May 2021, lockdown) main mode of commuting by census tract, 98-10-0480-01: subway or elevated rail + light rail, streetcar or commuter train over all commuters with a usual or no fixed place of work",
        },
    }


if __name__ == "__main__":
    {"gtfs": cmd_gtfs, "groups": cmd_groups, "areas": cmd_areas}[sys.argv[1]]()
