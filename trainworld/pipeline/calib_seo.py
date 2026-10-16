"""T-509 stage 3: Seoul's real network and calibration targets, from the seoulriders project.

    python pipeline/calib_seo.py trips     # weekday timetable -> data/work/realnet/seo_trips.csv
    python pipeline/realnet.py net seo     # the network (CITIES["seo"] reads that CSV)
    python pipeline/calib.py targets seo   # calls targets() here

Sources (maps/riders/seoulriders/data, all anonymous downloads, see its README):
- timetable: Seoul Metro's weekday (DAY) timetable for lines 1-9 (`timetable_raw.csv`) and KRIC's
  for the other lines (`timetable_extra.csv`, written by seoulriders' kric.py); every train.
  Everline (Yongin) has no stop times and is left out.
- stations: seoulriders' complexes (`stations.json`): one per station name and place, all its
  lines' platforms in it, which is how the card counts group them.
- entries: card gate boardings per station, mean of the Tue-Thu of November 2023 less 16
  November (exam day), `card_daily_202311.csv`, with KRIC's monthly figures turned into weekdays
  for the operators the card file misses (Incheon 1 and 2, Shinbundang, Gimpo Goldline,
  Uijeongbu, Jinjeop): seoulriders' `build_od.card_totals` and `outside.gate`, unchanged.
"""
import collections
import csv
import io
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import realnet as rn  # noqa: E402

SR = os.path.join(rn.MAPS, "riders", "seoulriders")
SR_DATA = os.path.join(SR, "data")
TRIPS = os.path.join(rn.WORK, "seo_trips.csv")

# rated capacity per train (seoulriders lines.py) / 160 a car = cars; the game's cars are 20 m
KIND = {"UI": "light_rail", "SL": "light_rail", "I2": "light_rail", "GP": "light_rail", "UJ": "light_rail",
        "GJ": "commuter", "GC": "commuter", "GG": "commuter", "SH": "commuter", "AR": "commuter"}
# end-to-end straight-line speed above which a "trip" is a summary row, not a train (seoulriders
# build.py: Sillim's two-stop rows); 1.5 x seoulriders' speed ceilings, km/h
MAX_KMH = {"AR": 195, **{k: 165 for k in ("1", "SB", "GJ", "GC", "SH", "GG", "SN")}, **{k: 120 for k in ("UI", "SL", "UJ")}}
MAX_KMH_DEFAULT = 135


def stations():
    return json.load(open(os.path.join(SR_DATA, "stations.json"), encoding="utf-8"))


def hms(s):
    s = (s or "").strip()
    if s.count(":") != 2:
        return None
    h, m, x = (int(v) for v in s.split(":"))
    return h * 3600 + m * 60 + x


def cmd_trips():
    sys.path.insert(0, SR)
    import lines as LR
    net = stations()
    cx = {}
    pos = {}
    for c in net["complexes"]:
        for p in c["platforms"]:
            cx[p["code"]] = c["id"]
        pos[c["id"]] = (c["name_en"] or c["name"], c["lon"], c["lat"])
    assert len(pos) == len(net["complexes"]), "complex ids not unique"
    rows = []
    for f in ("timetable_raw.csv", "timetable_extra.csv"):
        with io.open(os.path.join(SR_DATA, f), encoding="cp949", errors="replace", newline="") as fh:
            rows += [r for r in csv.DictReader(fh) if r["주중주말"].strip() == "DAY"]
    trips = collections.defaultdict(list)
    for r in rows:
        line, code = r["호선"].strip(), r["역사코드"].strip()
        if line not in LR.COLORS or code not in cx:
            continue
        arr, dep = hms(r["열차도착시간"]), hms(r["열차출발시간"])
        arr = arr or None  # 00:00:00 marks a terminus that only arrives or only departs
        dep = dep or None
        if arr is None and dep is None:
            continue
        trips[(line, r["방향"].strip(), r["열차코드"].strip())].append((dep or arr, arr or dep, code))
    out, dropped = [], collections.Counter()
    for (line, d, tid), st in trips.items():
        ts = [s[0] for s in st]
        if max(ts) - min(ts) > 12 * 3600:  # written past midnight as 00:xx
            st = [(a + 86400 if a < 4 * 3600 else a, b + 86400 if b < 4 * 3600 else b, c) for a, b, c in st]
        st.sort()
        if len(st) < 2:
            continue
        (_, lon0, lat0), (_, lon1, lat1) = pos[cx[st[0][2]]], pos[cx[st[-1][2]]]
        dist = math.hypot((lon1 - lon0) * 111320 * math.cos(math.radians(lat0)), (lat1 - lat0) * 111320)
        dt = st[-1][1] - st[0][0]
        if dt <= 0 or dist / dt * 3.6 > MAX_KMH.get(line, MAX_KMH_DEFAULT):
            dropped[line] += 1
            continue
        # DOWN and OUT (Line 2's outer loop) are direction 0, UP and IN direction 1
        dr = "1" if d in ("UP", "IN") else "0"
        for dep, _, code in st:
            k = cx[code]
            nm, lon, lat = pos[k]
            out.append((line, dr, f"{line}-{d}-{tid}", k, nm, lon, lat, dep, LR.DISPLAY_EN[line]))
    with open(TRIPS, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["route", "direction", "trip", "stop", "name", "lon", "lat", "time", "route_name"])
        w.writerows(out)
    print(f"{len(trips)} trains, {sum(dropped.values())} summary rows dropped {dict(dropped)}, {len(out)} stop times -> {TRIPS}")


def train_m():
    sys.path.insert(0, SR)
    import lines as LR
    return {"_": 200, **{k: round(v / 160) * 20 for k, v in LR.CAPACITY.items()}}


def weekday_entries():
    """Mean Tue-Thu boardings per complex, November 2023 (seoulriders' own reference days)."""
    sys.path.insert(0, SR)
    cwd = os.getcwd()
    os.chdir(SR)
    try:
        import build_od as B
        import daytype
        import outside
        net = stations()
        cs = net["complexes"]
        n = len(cs)
        by_name_label, by_name = {}, collections.defaultdict(list)
        for i, c in enumerate(cs):
            by_name[c["name"]].append(i)
            for lbl in c.get("od_labels", ()):
                by_name_label[(c["name"], lbl)] = i

        def find(name, label):
            label = B.CARD_LINE_ALIAS.get(label, label)
            i = by_name_label.get((name, label))
            if i is not None:
                return i
            cands = by_name.get(name)
            return cands[0] if cands and len(cands) == 1 else None
        day = daytype.get("weekday")
        want = B.dates_in(daytype.REF_MONTH, day["dows"], daytype.EXCLUDE_DATES)
        cb, _, have_c, got = B.card_totals(B.read_card(daytype.REF_MONTH), want, find, n)
        ob, _, have_o = outside.gate(daytype.REF_MONTH, day["dows"], find, n, [c["name"] for c in cs])
    finally:
        os.chdir(cwd)
    return [{"target": c["id"], "name": c["name_en"] or c["name"], "entries": float(cb[i] + ob[i]),
             "card": float(cb[i]), "offcard": float(ob[i])} for i, c in enumerate(cs) if have_c[i] or have_o[i]], got


# Stations where many riders arrive by modes the model has no trains for and then tap in: intercity
# rail (KTX, SRT, ITX), the big bus terminals, the airports. Reported apart, not scored.
HUBS = ("서울역", "용산", "영등포", "수서", "광명", "청량리", "고속터미널", "강변", "남부터미널", "김포공항",
        "인천공항1터미널", "인천공항2터미널", "상봉", "행신", "천안아산", "대곡")


def targets():
    net_path = rn.CITIES["seo"]["out"]
    net = json.load(open(net_path))
    st, days = weekday_entries()
    in_net = {s["target"] for s in net["stations"] if s.get("target")}
    st = [s for s in st if s["target"] in in_net]
    total = sum(s["entries"] for s in st)
    return {
        "city": "seo", "net": net_path, "areas_file": None, "station_min": 2000.0, "hub_m": 300.0,
        "stations": st, "hubs": [h for h in HUBS if h in in_net],
        # every journey taps in once (transfers inside the gates are free), so entries summed over
        # the network's stations are the trips that use it
        "operators": {"rail": total},
        "operators_info": {},
        "areas": [],
        "sources": {
            "stations": f"Seoul card gate boardings (OA-12914 CARD_SUBWAY_MONTH, {len(days)} Tue-Thu of Nov 2023 less 16 Nov) "
                        "plus KRIC monthly station boardings made weekdays for the off-card operators (seoulriders outside.gate)",
            "operators": "the same entries summed: trips using the network",
            "areas": "none: 2020 census commuting mode by si-gun-gu is KOSIS-only (browser download, not on disk)"},
    }


if __name__ == "__main__":
    {"trips": cmd_trips}[sys.argv[1]]()
