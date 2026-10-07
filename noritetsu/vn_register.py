"""Vietnam: Vietnam Railways' lines, from OpenStreetMap's named track, through kr_register's
recipe (as th_register does).

    python extract.py --region vn --pbf data/raw/vietnam-latest.osm.pbf
    python vn_register.py --clip           # after every extract: China's track and the lines
                                           # not open out of data/proc/vn
    python vn_register.py --report         # the way-to-line assignment, km per line
    python build_model.py --region vn --register vn_register:data/raw/vn

THE LINE UNIT is the national railway's own list of lines (the Vietnam Railway Authority's,
DRVN; vn_sources.md has the table): the North-South Railway Hà Nội - Sài Gòn, Hà Nội - Đồng
Đăng, Yên Viên - Lào Cai, Gia Lâm - Hải Phòng, Đông Anh - Quán Triều, Kép - Hạ Long - Cái
Lân, the branches Diêu Trì - Quy Nhơn and Bình Thuận - Phan Thiết, and what is left of the
Đà Lạt rack railway, Đà Lạt - Trại Mát. Those are exactly the extents OSM names its track
for (the shared trunk Hà Nội - Yên Viên is the Đồng Đăng line's), so no track is two lines'.
The lines keep the names everyone uses ("Đường sắt Hà Nội - Lào Cai"), which OSM's line
relations and Wikidata use too.
Freight-only lines (Bắc Hồng - Văn Điển, Yên Trạch - Na Dương, Kép - Lưu Xá, Quan Triều - Núi
Hồng, Phố Lu - Pom Hán, Chí Linh - Phả Lại) are no register lines. Kép - Cái Lân has had no
passenger train since 2020-21 and is built greyed (`suspended`). The metros (Hà Nội 2A and 3,
HCMC 1) stay OSM lines, as everywhere. The Mường Hoa mountain railway at Sa Pa (Sa Pa -
Mường Hoa, a public 2 km funicular-style line) is a register line, from its named track.

WHICH WAY IS ON WHICH LINE. OSM Vietnam names 98% of its main-line track for the line (`python
probe_kr_ways.py --region vn`). Per way, in order: its name (NAME_LINE; bridge and tunnel
names, "Cầu ...", "Hầm ...", are read as no name), else the line relation it is in (OSM maps
the lines themselves as route=train relations, "Đường sắt Hà Nội - Lào Cai", ref ĐSHN-LC, and a
few as route=railway), else the line its neighbours at both ends share (`propagate`).

WHICH STATIONS ARE ON A LINE: no open per-line list. Every OSM rail station within LIST_M of
a line's own track is listed on it (metro records left out first), as in th_register; a branch
takes the station nearest its junction end; a line left in pieces by an unnamed yard is joined
over other track (`join_pieces`). Đà Lạt's station is mapped only as a building (a multipolygon
extract.py does not read), so it is given here (EXTRA_STATIONS).

BORDERS. Đồng Đăng - Pingxiang has a daily passenger train (MR1/MR2 Gia Lâm - Nanning, since
25 May 2025); the point where the track crosses the boundary is offered as a station of the
Đồng Đăng line until it is in borders.EXTRA, and its section is kept (`served_sections`).
Lào Cai - Hekou (Hồ Kiều bridge) carries freight only: no point, the line ends at Lào Cai.

The `path` argument is data/raw/vn; the OSM half is read from data/proc/vn (extract.py).
"""
import hashlib
import json
import math
import os
import pickle
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import kr_register as kr

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
REGION = "vn"
RAW = ROOT / "data" / "raw" / "vn"
VNR = "Tổng công ty Đường sắt Việt Nam"
VNR_EN = "Vietnam Railways"
SUNWORLD = "Sun World Fansipan Legend"

# key -> (name, English name, operator, operator_en). Names as OSM's line relations and
# vi.wikipedia write them, so OSM's own line relations match them by name (merge_sources).
LINES = {
    "bacnam": ("Đường sắt Bắc Nam", "North–South Railway", VNR, VNR_EN),
    "haiphong": ("Đường sắt Hà Nội - Hải Phòng", "Hanoi–Haiphong Railway", VNR, VNR_EN),
    "laocai": ("Đường sắt Hà Nội - Lào Cai", "Hanoi–Lào Cai Railway", VNR, VNR_EN),
    "dongdang": ("Đường sắt Hà Nội - Đồng Đăng", "Hanoi–Đồng Đăng Railway", VNR, VNR_EN),
    "quantrieu": ("Đường sắt Hà Nội - Quan Triều", "Hanoi–Quán Triều Railway", VNR, VNR_EN),
    "kep": ("Đường sắt Kép - Cái Lân", "Kép–Hạ Long Railway", VNR, VNR_EN),
    "quynhon": ("Đường sắt Diêu Trì - Quy Nhơn", "Diêu Trì–Quy Nhơn Branch", VNR, VNR_EN),
    # DRVN's name; the junction station was Mường Mán until it was renamed Bình Thuận
    "phanthiet": ("Đường sắt Bình Thuận - Phan Thiết", "Bình Thuận–Phan Thiết Branch", VNR,
                  VNR_EN),
    "dalat": ("Đường sắt Đà Lạt - Trại Mát", "Đà Lạt–Trại Mát Railway", VNR, VNR_EN),
    "muonghoa": ("Tàu hỏa leo núi Mường Hoa", "Mường Hoa Mountain Railway", SUNWORLD,
                 "Sun World Fansipan Legend"),
}
# No passenger train runs (vn_sources.md): built, greyed as not running.
SUSPENDED = {"kep"}

# OSM way names -> line key; None leaves the way out of the register.
NAME_LINE = {
    "Đường sắt Bắc–Nam": "bacnam", "Đường sắt Bắc Nam": "bacnam",
    "Đường sắt Hà Nội - Hải Phòng": "haiphong",
    "Đường sắt Hà Nội - Lào Cai": "laocai",
    "Đường sắt Hà Nội - Đồng Đăng": "dongdang",
    "Đường sắt Hà Nội - Quan Triều": "quantrieu",
    "Đường sắt Kép – Cái Lân": "kep",
    "Đường sắt Diêu Trì–Quy Nhơn": "quynhon",
    "Đường sắt Bắc Nam (Phan Thiết)": "phanthiet",
    "Đường sắt Tháp Chàm - Đà Lạt": "dalat",
    "Tàu hỏa leo núi Mường Hoa": "muonghoa",
    # freight only
    "Đường sắt Bắc Hồng - Văn Điển": None, "Đường sắt Yên Trạch - Na Dương": None,
    "Đường sắt Quan Triều - Núi Hồng": None, "Đường sắt Phố Lu - Pom Hán": None,
    "Đường sắt Pom Hán - Tằng Lỏong": None, "đường sắt Pom Hán - Tằng Lỏong": None,
    "Đường sắt Phủ Lý – Thịnh Châu": None, "Đường sắt Kép – Lưu Xá": None,
    "Đường sắt nội khu KCN Dung Quất": None,
    # part of Yên Viên - Phả Lại - Hạ Long, never finished
    "Đường sắt Chí Linh - Phả Lại": None,
    # the summit funicular inside the Fansipan cable-car complex
    "Tàu hoả leo núi": None,
}
# Names on track that are a structure's, not a line's: read as no name.
JUNK = re.compile(r"^(Cầu|cầu|Hầm|hầm)\s")

# Line relations (route=train relations named for the line, and route=railway) -> line key,
# for their unnamed ways; None: never register track.
REL_LINE = {
    2709148: "bacnam", 8351998: "laocai", 9101638: "dongdang", 9101640: "quantrieu",
    9101636: "kep", 3539672: "haiphong", 3531754: "dalat", 10617633: "dalat",
    9101635: None, 9101639: None, 16331840: None, 18727155: None,
}
# An unnamed way in several line relations takes the first of their lines in this order: the
# Lào Cai and Quan Triều relations both run from Hà Nội over the Đồng Đăng line's trunk.
REL_PRIORITY = ["bacnam", "dongdang", "haiphong", "laocai", "quantrieu", "kep", "quynhon",
                "phanthiet", "dalat", "muonghoa"]

TRACK_KIND = {"rail": "rail", "narrow_gauge": "rail", "funicular": "funicular"}
# Not "tourism": the Đà Lạt - Trại Mát line is tagged so and runs 5-6 trains a day.
NOT_PASSENGER = {"industrial", "military", "test"}

# A station this close to a line's track is listed on it; the track is read with a point
# every DENSIFY_M.
LIST_M = 250
DENSIFY_M = 25
# Border points this close to a line's track become its stations.
BORDER_M = 400
# Crossings with no border point in borders.load() yet: where OSM's track crosses OSM's
# boundary of Vietnam (relation 49915, data/raw/vn/vn_boundary.geojson). To go into
# borders.EXTRA; a point already there within 50 m is used instead.
BORDERS = [
    ("xDongDang", 106.715575, 21.972213, ["cn", "vn"]),
]
# Crossings passenger trains run over: MR1/MR2 Gia Lâm - Nanning (daily since 25 May 2025).
BORDER_SERVED = {"xDongDang"}
# Stations OSM maps only as an area or a multipolygon: (name, English name, lon, lat).
EXTRA_STATIONS = [
    ("Đà Lạt", "Da Lat", 108.45448, 11.94161),     # relation 17877171, "Ga Đà Lạt"
]
# OSM stations no passenger train calls at, on track that is otherwise a passenger line's:
# Thượng Cát is "Trạm bổ trợ ga Gia Lâm", an operating post of Gia Lâm station at the
# junction of the Hải Phòng line, where the Hải Phòng line would otherwise begin.
NOT_STOPS = {"Trạm đường sắt Thượng Cát"}
# Station records of amusement-park trains: Thống Nhất park's "Tầu Hỏa mini" names its stop
# "Sai Gon", 1.1 km south of Hà Nội station and within reach of the North-South Railway.
PARK = ("Tầu Hỏa mini", "Tàu Hỏa mini", "Công viên")
# Two pieces of one line's track whose ends are this close, with no way between them (OSM's
# Lào Cai line west of Bắc Hồng stops 78 m short of its next way), are welded (`weld`).
WELD_M = 120
# A line whose track ends on another line's takes the station nearest that end within this.
JUNCTION_M = 1000

# --clip: route relations of lines that are not open, or carry no passenger train, and the
# track of lines not built (mapped as railway=rail under their future names).
NOT_OPEN_ROUTES = {
    9101636,                                   # Kép - Cái Lân: no passenger train since 2020-21
    17739982, 17739983,                        # Hà Nội line 2, under construction
    17504372, 21234053, 21239293, 21239294,    # HCMC line 2, under construction
    21285829, 21285830, 21285831, 21285832, 21285833,   # Hà Nội lines 1, 2, 8, 10, 14: planned
    19374650,                                  # unnamed, 4 ways
}
NOT_OPEN_TRACK = re.compile(
    r"^(Đường sắt đô thị [Ss][ốô] (1|2|8|10|14) Hà Nội|"
    r"HCMC Metro Tuyến 2|Đường sắt đô thị số 2 Thành phố Hồ Chí Minh|Nhánh ra vào Depot Tham Lương)")

_S = {}


def line_id(name):
    h = hashlib.blake2b(f"vn|{name}".encode("utf-8"), digest_size=5)
    return "v" + h.hexdigest()


def tidy(name):
    n = (name or "").strip()
    return "" if JUNK.match(n) else n


def way_geo(ways, coords):
    """{way: (km, centroid lon, centroid lat)}"""
    out = {}
    for wid, (tags, nodes) in ways.items():
        pos, ok = coords.many(np.asarray(nodes, dtype=np.int64))
        pos = pos[ok]
        if pos.size < 2:
            out[wid] = (0.0, 0.0, 0.0)
            continue
        x, y = coords.x[pos] / 1e7, coords.y[pos] / 1e7
        lat = np.radians((y[:-1] + y[1:]) / 2)
        km = float(np.hypot(np.diff(x) * np.cos(lat) * 111.32, np.diff(y) * 110.57).sum())
        out[wid] = (km, float(x.mean()), float(y.mean()))
    return out


def assign(ways, coords, rels, log):
    """{way: line key} for every way that is register track."""
    geo = way_geo(ways, coords)
    inrel = defaultdict(list)
    for rid, (tags, members) in rels.items():
        if rid not in REL_LINE:
            continue
        for m in members:
            if m[0] == "w" and m[1] in ways:
                inrel[m[1]].append(rid)
    out, unknown, left = {}, Counter(), Counter()
    cand = []
    for wid, (t, _n) in ways.items():
        if t.get("railway") not in TRACK_KIND or t.get("usage") in NOT_PASSENGER:
            continue
        km, lon, lat = geo[wid]
        n = tidy(t.get("name"))
        svc = t.get("service")
        if n and n in NAME_LINE:
            key = NAME_LINE[n]
            if key is None:
                left[n] += km
                continue
        elif n:
            if not svc:
                unknown[n] += km
            continue
        else:
            if svc not in (None, "crossover"):
                continue
            if any(REL_LINE.get(r, 0) is None for r in inrel[wid]):
                continue
            keys = sorted((REL_LINE[r] for r in inrel[wid] if REL_LINE.get(r)),
                          key=REL_PRIORITY.index)
            key = keys[0] if keys else None
            if key is None:
                cand.append(wid)
                continue
        out[wid] = key
    got = propagate(ways, out, cand, log)
    out.update(got)
    km_by = Counter()
    for w, k in out.items():
        if not ways[w][0].get("service"):
            km_by[k] += geo[w][0]
    log("VN: main-line track km per line (both tracks of double track): "
        + ", ".join(f"{k} {v:,.1f}" for k, v in km_by.most_common()))
    log("VN: named track left out: " + ", ".join(f"{n} {v:.1f}" for n, v in left.most_common()))
    if unknown:
        log("VN: names in no table, left out: "
            + ", ".join(f"{n} {v:.1f}" for n, v in unknown.most_common()))
    return out, geo


def propagate(ways, named, cand, log):
    """Unnamed track outside every line relation takes the line its neighbours at both ends
    share, repeated until nothing changes (th_register.propagate)."""
    node_ways = defaultdict(list)
    for w in set(named) | set(cand):
        nl = ways[w][1]
        node_ways[int(nl[0])].append(w)
        node_ways[int(nl[-1])].append(w)
    for w in set(named) | set(cand):
        for n in np.asarray(ways[w][1]).tolist()[1:-1]:
            if n in node_ways:
                node_ways[n].append(w)
    name = dict(named)
    got = {}
    for _round in range(200):
        new = {}
        for w in cand:
            if w in name:
                continue
            nl = ways[w][1]
            a = {name[o] for o in node_ways[int(nl[0])] if o != w and o in name}
            b = {name[o] for o in node_ways[int(nl[-1])] if o != w and o in name}
            if len(a & b) == 1:
                new[w] = next(iter(a & b))
        if not new:
            break
        name.update(new)
        got.update(new)
    log(f"VN: {len(got)} unnamed ways outside every line relation named from their "
        f"neighbours; {len(cand) - len(got)} left unnamed")
    return got


WELD_BASE = -7_000_000_000


def weld(ways, coords, line_of, log):
    """Where one line's track is in pieces with a dead end of one piece within WELD_M of a
    node of another piece, add a two-node way between them, of that line. OSM leaves such
    slips in the Lào Cai line west of Bắc Hồng (78 m), and a register line in two pieces loses
    the section across the slip."""
    by_line = defaultdict(list)
    for w, k in line_of.items():
        by_line[k].append(w)
    made = []
    for k, ws in by_line.items():
        parent = {}

        def find(x):
            while parent.setdefault(x, x) != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        deg = Counter()
        xy = {}
        for w in ws:
            nl = np.asarray(ways[w][1], dtype=np.int64)
            pos, ok = coords.many(nl)
            for n, p, g in zip(nl.tolist(), pos.tolist(), ok.tolist()):
                if g:
                    xy[n] = (coords.x[p] / 1e7, coords.y[p] / 1e7)
            nl = nl.tolist()
            for a, b in zip(nl[:-1], nl[1:]):
                parent[find(a)] = find(b)
                deg[a] += 1
                deg[b] += 1
        if len({find(n) for n in xy}) < 2:
            continue
        ids = np.fromiter(xy.keys(), dtype=np.int64)
        arr = np.array([xy[i] for i in ids.tolist()])
        for n in [n for n, v in deg.items() if v == 1 and n in xy]:
            x, y = xy[n]
            d = np.hypot((arr[:, 0] - x) * math.cos(math.radians(y)) * 111320,
                         (arr[:, 1] - y) * 110570)
            for j in np.argsort(d)[:50]:
                if d[j] > WELD_M:
                    break
                m = int(ids[j])
                if find(m) != find(n):
                    wid = WELD_BASE - len(made)
                    src = next(w for w in ws if n in (ways[w][1][0], ways[w][1][-1]))
                    ways[wid] = (dict(ways[src][0]), np.asarray([n, m], dtype=np.int64))
                    line_of[wid] = k
                    parent[find(n)] = find(m)
                    made.append((k, n, m, float(d[j])))
                    break
    for k, n, m, d in made:
        log(f"VN: welded {LINES[k][0]} at {xy_str(coords, n)}: {d:.0f} m to node {m}")


def xy_str(coords, n):
    p = coords.get(n)
    return f"{p[0]:.5f},{p[1]:.5f}" if p else str(n)


def load_osm(log):
    import build_model as bm
    ways, rels, stops, cid, cx, cy = bm.load(REGION, log)
    coords = bm.Coords(cid, cx, cy)
    with open(ROOT / "data" / "proc" / REGION / "infra.pkl", "rb") as f:
        infra = pickle.load(f)
    allrels = dict(infra)
    allrels.update(rels)
    line_of, geo = assign(ways, coords, allrels, log)
    weld(ways, coords, line_of, log)
    _S.update(ways=ways, rels=rels, stops=stops, coords=coords, line_of=line_of, geo=geo)
    _S["byobj"] = {id(ways[w][0]): LINES[k][0] for w, k in line_of.items()}
    return ways, stops, coords


def register_name(tags):
    return _S["byobj"].get(id(tags), "")


# ---------------------------------------------------------------- stations and borders

BORDER_BASE = -9_000_000_000
EXTRA_BASE = -8_000_000_000


def border_points(log):
    pts = []
    try:
        import borders
        pts = [p for p in borders.load(canonical_only=True) if "vn" in p["countries"]]
    except Exception as e:                          # noqa: BLE001
        log(f"VN: borders.load failed ({e})")
    served = set()
    for pid, lon, lat, ccs in BORDERS:
        have = [p for p in pts if kr.dist_m(lon, lat, p["lon"], p["lat"]) <= 50]
        if not have:
            have = [{"id": pid, "lon": lon, "lat": lat, "countries": set(ccs)}]
            pts.append(have[0])
        if pid in BORDER_SERVED:
            served.add(have[0]["id"])
    _S["border_served"] = served
    return pts


def urban_only(tags):
    """A metro station or stop record (Hà Nội 2A and 3, HCMC 1, and the planned lines' S1.24
    style records), never a VNR stop."""
    who = " ".join(tags.get(k) or "" for k in ("operator", "network"))
    return (tags.get("station") in ("subway", "light_rail", "monorail")
            or tags.get("subway") == "yes" or tags.get("light_rail") == "yes"
            or tags.get("monorail") == "yes" or any(p in who for p in PARK))


def build_stations(stops, log):
    on_urban = set()
    for t, nodes in _S["ways"].values():
        if t.get("railway") in ("subway", "monorail", "light_rail"):
            on_urban.update(np.asarray(nodes).tolist())
    drop = {n for n, (t, _x, _y) in stops.items()
            if urban_only(t) or (n in on_urban and t.get("railway") != "station")}
    stops = {n: v for n, v in stops.items() if n not in drop}
    # "Ga Vinh", "Ga Đà Lạt": the word for station is no part of the name (the tables write
    # Vinh), and a station mapped as an area often carries it
    n_ga = 0
    for n, (t, lon, lat) in list(stops.items()):
        nm = t.get("name") or ""
        if nm.startswith("Ga ") and len(nm) > 3:
            stops[n] = (dict(t, name=nm[3:].strip()), lon, lat)
            n_ga += 1
    for i, (nm, en, lon, lat) in enumerate(EXTRA_STATIONS):
        stops[EXTRA_BASE - i] = ({"name": nm, "name:en": en, "railway": "station",
                                  "train": "yes"}, lon, lat)
    log(f"VN: {len(drop)} station and stop records of the metros and park trains left out of "
        f"the register; {n_ga} names read without a leading \"Ga \"; {len(EXTRA_STATIONS)} "
        f"stations given by hand ({', '.join(s[0] for s in EXTRA_STATIONS)})")
    st, node_st, by_key, by_base = _orig["build_stations"](stops, log)
    gone = {sid for sid, s in st.items() if s["name"] in NOT_STOPS}
    for sid in gone:
        del st[sid]
    for n in [n for n, s in node_st.items() if s in gone]:
        del node_st[n]
    for d in (by_key, by_base):
        for k in list(d):
            d[k] = [s for s in d[k] if s not in gone]
    border = {}
    for i, p in enumerate(border_points(log)):
        fid = BORDER_BASE - i
        st[fid] = {"name": p["id"], "name_en": "", "lon": p["lon"], "lat": p["lat"], "rank": 0}
        by_key[kr.name_key(p["id"])].append(fid)
        by_base[kr.base_key(p["id"])].append(fid)
        border[fid] = p["id"]
    _S.update(st=st, node_st=node_st, border=border)
    log(f"VN: {len(border)} border points offered to the lines ({', '.join(border.values())})")
    return st, node_st, by_key, by_base


def load_lists(path, log):
    """{line: [[station name, ...]]}: every rail station within LIST_M of the line's track."""
    from scipy.spatial import cKDTree
    ways, coords, st = _S["ways"], _S["coords"], _S["st"]
    by_line = defaultdict(list)
    for w, k in _S["line_of"].items():
        by_line[LINES[k][0]].append(w)
    trees = {}
    kx = 111.32 * math.cos(math.radians(16.0))
    for ln, ws in by_line.items():
        pts = []
        for w in ws:
            p, ok = coords.many(np.asarray(ways[w][1], dtype=np.int64))
            p = p[ok]
            x, y = coords.x[p] / 1e7 * kx, coords.y[p] / 1e7 * 110.57
            pts.append(np.c_[x, y])
            # a point every DENSIFY_M along each segment: OSM puts few vertices on straight
            # track, and Chợ Sy and Hương Phố stand 260 m from the nearest one but beside
            # the line
            for x1, y1, x2, y2 in zip(x[:-1], y[:-1], x[1:], y[1:]):
                n = int(math.hypot(x2 - x1, y2 - y1) * 1000 // DENSIFY_M)
                if n:
                    t = np.arange(1, n + 1) / (n + 1)
                    pts.append(np.c_[x1 + (x2 - x1) * t, y1 + (y2 - y1) * t])
        trees[ln] = cKDTree(np.concatenate(pts))
    lists = defaultdict(set)
    for sid, s in st.items():
        if sid in _S["border"]:
            continue
        for ln, tr in trees.items():
            d, _i = tr.query((s["lon"] * kx, s["lat"] * 110.57))
            if d * 1000 <= LIST_M:
                lists[ln].add(s["name"])
    # junction ends: a line's track that ends on another line's takes the station nearest it
    line_of = _S["line_of"]
    inc = defaultdict(Counter)
    on = defaultdict(set)
    for w, k in line_of.items():
        nl = np.asarray(ways[w][1]).tolist()
        inc[k][nl[0]] += 1
        inc[k][nl[-1]] += 1
        for n in nl[1:-1]:
            inc[k][n] += 2
        for n in nl:
            on[n].add(k)
    real = [(sid, s) for sid, s in st.items() if sid not in _S["border"]]
    sx = np.array([s["lon"] for _i, s in real])
    sy = np.array([s["lat"] for _i, s in real])
    n_junc = 0
    for k, c in inc.items():
        ln = LINES[k][0]
        for n, v in c.items():
            if v != 1 or len(on[n]) < 2:
                continue
            p = coords.get(n)
            if p is None:
                continue
            d = np.hypot((sx - p[0]) * math.cos(math.radians(p[1])) * 111320,
                         (sy - p[1]) * 110570)
            j = int(np.argmin(d))
            if d[j] <= JUNCTION_M and real[j][1]["name"] not in lists[ln]:
                lists[ln].add(real[j][1]["name"])
                n_junc += 1
                log(f"    junction end of {ln}: {real[j][1]['name']} ({d[j]:.0f} m)")
    log(f"VN: {n_junc} junction stations listed on the line whose track ends near them")
    for fid, pid in _S["border"].items():
        p = st[fid]
        best = None
        for ln, ws in by_line.items():
            for w in ws:
                d = seg_dist(ways[w][1], coords, p["lon"], p["lat"])
                if d <= BORDER_M and (best is None or d < best[0]):
                    best = (d, ln)
        if best:
            lists[best[1]].add(pid)
            log(f"VN: border point {pid} on {best[1]} ({best[0]:.0f} m)")
        else:
            log(f"VN: border point {pid} is on no line's track")
    log(f"VN: station lists by proximity ({LIST_M} m): "
        f"{sum(len(v) for v in lists.values())} station-line pairs on {len(lists)} lines")
    out = {k: [sorted(v)] for k, v in lists.items()}
    cum = defaultdict(list)
    names = {s["name"] for s in st.values()}
    for key, fn in CHAINAGE.items():
        rows = chainage(RAW / fn)
        if not rows:
            log(f"VN: no chainage table in {fn}")
            continue
        ln = LINES[key][0]
        chain = []
        for nm, km in rows:
            nm = nm if nm in names else next((a for a in tone_variants(nm) if a in names), nm)
            chain.append(nm)
            cum[(ln, len(out.setdefault(ln, [])), kr.name_key(nm))].append(km)
        out[ln].append(chain)
        log(f"VN: {ln}: {len(rows)} stations with chainage from {fn}")
    return out, cum, {}


# Lines with a published station chainage table (vi.wikipedia, from VNR's): every section
# between two listed stations gets its published km, which check_model compares.
CHAINAGE = {"bacnam": "viwiki/Danh_sách_nhà_ga_thuộc_tuyến_đường_sắt_Thống_Nhất.wiki"}


def chainage(path):
    """[(station name, km)] from a vi.wikipedia station table ("| [[Ga X|X]] || 1.017,10 ||"),
    in table order."""
    try:
        txt = path.read_text(encoding="utf-8")
    except OSError:
        return []
    rows = []
    for m in re.finditer(r"^\|\|?\s*(?:'''|'')?\[\[([^\]|]+)(?:\|([^\]]+))?\]\](?:'''|'')?"
                         r"\s*\|\|\s*([\d.,]+)", txt, re.M):
        name = re.sub(r"^Ga ", "", (m.group(2) or m.group(1)).strip())
        rows.append((name, float(m.group(3).replace(".", "").replace(",", "."))))
    return rows


# Vietnamese writes a tone on "oa", "oe", "uy" two ways ("Hòa" and "Hoà"); OSM and the
# tables do not agree on which.
_TONES = {"̀": "", "́": "", "̉": "", "̃": "", "̣": ""}


def tone_variants(name):
    """The name with the tone mark of "oa", "oe", "uy" on the other vowel."""
    import unicodedata as ud
    d = ud.normalize("NFD", name)
    out = set()
    for pair in ("oa", "oe", "uy", "Oa", "Oe", "Uy"):
        a, b = pair
        for t in _TONES:
            out.add(ud.normalize("NFC", d.replace(a + t + b, a + b + t)))
            out.add(ud.normalize("NFC", d.replace(a + b + t, a + t + b)))
    out.discard(name)
    return sorted(out)


def seg_dist(nodes, coords, lon, lat):
    pos, ok = coords.many(np.asarray(nodes, dtype=np.int64))
    pos = pos[ok]
    if pos.size < 2:
        return 1e9
    x, y = coords.x[pos] / 1e7, coords.y[pos] / 1e7
    kx = math.cos(math.radians(lat)) * 111320
    ax, ay = (x[:-1] - lon) * kx, (y[:-1] - lat) * 110570
    bx, by = (x[1:] - lon) * kx, (y[1:] - lat) * 110570
    dx, dy = bx - ax, by - ay
    L2 = np.maximum(dx * dx + dy * dy, 1e-9)
    t = np.clip(-(ax * dx + ay * dy) / L2, 0, 1)
    return float(np.min(np.hypot(ax + t * dx, ay + t * dy)))


GAP_KM = 12.0          # two pieces of one line whose stations are this close (crow-fly)...
GAP_DETOUR = 1.5       # ... are joined over any rail track no longer than this x the crow-fly
GAP_EXTRA_KM = 1.0     # ... plus this


class Net:
    """Every rail way not left out (NAME_LINE None) as one graph, for joining the pieces of a
    line through a yard OSM leaves unnamed (th_register.Net)."""

    def __init__(self, ways, coords):
        self.adj = defaultdict(list)
        self.xy = {}
        for wid, (t, nodes) in ways.items():
            if t.get("railway") not in TRACK_KIND or t.get("usage") in NOT_PASSENGER:
                continue
            n = tidy(t.get("name"))
            if n in NAME_LINE and NAME_LINE[n] is None:
                continue
            nodes = np.asarray(nodes, dtype=np.int64)
            pos, ok = coords.many(nodes)
            prev = None
            for nd, p, good in zip(nodes.tolist(), pos.tolist(), ok.tolist()):
                if not good:
                    prev = None
                    continue
                if nd not in self.xy:
                    self.xy[nd] = (coords.x[p] / 1e7, coords.y[p] / 1e7)
                if prev is not None and prev != nd:
                    w = kr.dist_m(*self.xy[prev], *self.xy[nd]) / 1000
                    self.adj[prev].append((nd, w))
                    self.adj[nd].append((prev, w))
                prev = nd
        self.ids = np.fromiter(self.xy.keys(), dtype=np.int64)
        a = np.array([self.xy[i] for i in self.ids.tolist()])
        self.x, self.y = a[:, 0], a[:, 1]

    def node_near(self, lon, lat):
        d = np.hypot((self.x - lon) * math.cos(math.radians(lat)), self.y - lat)
        return int(self.ids[int(np.argmin(d))])

    def path(self, a, b, cap):
        import heapq
        dist, prev, heap = {a: 0.0}, {}, [(0.0, a)]
        while heap:
            d, u = heapq.heappop(heap)
            if u == b:
                break
            if d > dist.get(u, kr.INF):
                continue
            for v, w in self.adj.get(u, ()):
                nd = d + w
                if nd <= cap and nd < dist.get(v, kr.INF):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(heap, (nd, v))
        if b not in dist:
            return None
        nodes = [b]
        while nodes[-1] in prev:
            nodes.append(prev[nodes[-1]])
        return nodes[::-1], dist[b]


def join_pieces(lines, stations, geoms, log):
    """A line in several pieces is joined where two of its stations in different pieces are
    within GAP_KM and rail track joins them within GAP_DETOUR x the crow-fly + GAP_EXTRA_KM,
    nearest first (th_register.join_pieces)."""
    from n02 import walk_order
    net = Net(_S["ways"], _S["coords"])
    done = []
    for l in lines:
        parent = {}

        def find(x):
            while parent.setdefault(x, x) != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for a, b, *_ in l["sections"]:
            parent[find(a)] = find(b)
        sts = sorted({s for sec in l["sections"] for s in sec[:2]})
        if len({find(s) for s in sts}) < 2:
            continue
        cands = []
        for i, a in enumerate(sts):
            for b in sts[i + 1:]:
                if find(a) != find(b):
                    sa, sb = stations[a], stations[b]
                    crow = kr.dist_m(sa["lon"], sa["lat"], sb["lon"], sb["lat"]) / 1000
                    if crow <= GAP_KM:
                        cands.append((crow, a, b))
        for crow, a, b in sorted(cands):
            if find(a) == find(b):
                continue
            sa, sb = stations[a], stations[b]
            got = net.path(net.node_near(sa["lon"], sa["lat"]), net.node_near(sb["lon"], sb["lat"]),
                           GAP_DETOUR * crow + GAP_EXTRA_KM)
            if got is None:
                continue
            nodes, km = got
            parent[find(a)] = find(b)
            l["sections"].append([a, b, round(km, 3)])
            geoms[l["id"]][f"{a}|{b}"] = (
                [[round(sa["lon"], 5), round(sa["lat"], 5)]]
                + [[round(net.xy[n][0], 5), round(net.xy[n][1], 5)] for n in nodes]
                + [[round(sb["lon"], 5), round(sb["lat"], 5)]])
            if isinstance(l.get("highspeed_sections"), dict):
                l["highspeed_sections"][f"{a}|{b}"] = False
            done.append((l["name"], sa["name"], sb["name"], km))
        l["km"] = round(sum(s[2] for s in l["sections"]), 3)
        l["display"] = walk_order([(a, b) for a, b, *_ in l["sections"]])
        left = len({find(s) for s in sts})
        if left > 1:
            log(f"VN: {l['name']} is still in {left} pieces")
    log(f"VN: {len(done)} gaps in a line joined over other track "
        f"({sum(d[3] for d in done):.1f} km)")
    for nm, a, b, km in done:
        log(f"    {nm}: {a} - {b} {km:.1f} km")


_orig = {}


def adopt():
    if _orig:
        return
    _orig["build_stations"] = kr.build_stations
    kr.load_osm = load_osm
    kr.register_name = register_name
    kr.load_lists = load_lists
    kr.build_stations = build_stations
    kr.line_id = line_id
    kr.TRACK_KIND = TRACK_KIND
    kr.NOT_PASSENGER = NOT_PASSENGER
    kr.NAME_ALIAS = {}
    kr.STATION_ALIAS = {}
    kr.MATCH_M = JUNCTION_M


def build(path, log):
    adopt()
    import gb_register as gb
    lines, stations, geoms = kr.build(path, log)
    join_pieces(lines, stations, geoms, log)
    before = {l["id"]: {(a, b): km for a, b, km, *_ in l["sections"]} for l in lines}
    gb.drop_shortcuts(lines, geoms, log)
    for l in lines:
        after = {(a, b) for a, b, *_ in l["sections"]}
        gone = {k: v for k, v in before[l["id"]].items() if k not in after}
        for (a, b), km in gone.items():
            log(f"    shortcut dropped: {l['name']}: {stations[a]['name']} - "
                f"{stations[b]['name']} {km:.1f} km")
    # the published km again over the sections kept (kr_register summed it before the
    # shortcuts went)
    for key, fn in CHAINAGE.items():
        at = {}
        for nm, km in chainage(RAW / fn):
            for v in [nm] + tone_variants(nm):
                at.setdefault(kr.name_key(v), km)
        for l in lines:
            if l["name"] != LINES[key][0]:
                continue
            got = [abs(at[ka] - at[kb]) for a, b, *_ in l["sections"]
                   for ka, kb in [(kr.name_key(stations[a]["name"]), kr.name_key(stations[b]["name"]))]
                   if ka in at and kb in at]
            if len(got) == len(l["sections"]):
                l["km_official"] = round(sum(got), 3)
            else:
                l.pop("km_official", None)
            log(f"VN: {l['name']}: published km for {len(got)} of {len(l['sections'])} sections"
                + (f", {l['km_official']:.1f} km" if "km_official" in l else ""))
    border = {f"k{fid}": pid for fid, pid in _S.get("border", {}).items()}
    by_name = {v[0]: (k, v) for k, v in LINES.items()}

    def r(sid):
        return border.get(sid) or ("v" + sid[1:] if sid.startswith("k") else sid)

    out_st = {}
    for sid, s in stations.items():
        nid = r(sid)
        s["id"] = nid
        if sid in border:
            s["junction"] = True
            s["name"] = nid
        out_st[nid] = s
    out_geoms = {}
    for l in lines:
        key, (_n, en, op, op_en) = by_name[l["name"]]
        l["src"] = "vn"
        l["operator"], l["operator_en"] = op, op_en
        l["name_en"] = en
        if key in SUSPENDED:
            l["suspended"] = True
        l["sections"] = [[r(a), r(b), *rest] for a, b, *rest in l["sections"]]
        l["display"] = [r(x) for x in l["display"]]
        served = [f"{a}|{b}" for a, b, *_ in l["sections"]
                  if a in _S.get("border_served", ()) or b in _S.get("border_served", ())]
        if served:
            l["served_sections"] = served
        if isinstance(l.get("highspeed_sections"), dict):
            l["highspeed_sections"] = {"|".join(r(x) for x in k.split("|")): v
                                       for k, v in l["highspeed_sections"].items()}
        out_geoms[l["id"]] = {"|".join(r(x) for x in k.split("|")): v
                              for k, v in geoms[l["id"]].items()}
    for s in out_st.values():
        s["lines"] = set()
    for l in lines:
        for a, b, *_ in l["sections"]:
            out_st[a]["lines"].add(l["id"])
            out_st[b]["lines"].add(l["id"])
    out_st = {k: s for k, s in out_st.items() if s["lines"]}
    log(f"VN: {len(lines)} register lines, {sum(l['km'] for l in lines):,.1f} km, "
        f"{len(out_st)} stations")
    for l in sorted(lines, key=lambda l: -l["km"]):
        log(f"    {l['km']:8.1f} km  {len(l['sections']):3d} sections  {l['name']}"
            f"{'  (suspended)' if l.get('suspended') else ''}")
    return lines, out_st, out_geoms


# ---------------------------------------------------------------- --clip

def clip(log=print):
    """Rewrite data/proc/vn without China's track and stops (Geofabrik's cut holds Hekou's
    metre-gauge yard and the standard-gauge line towards Pingxiang) and without the route
    relations and track of lines not open (NOT_OPEN_ROUTES, NOT_OPEN_TRACK). A way goes if at
    least half its nodes are outside OSM's boundary of Vietnam (relation 49915), as in
    cn_register.clip. Run after every extract."""
    import shapely
    from shapely.geometry import shape
    d = ROOT / "data" / "proc" / REGION
    with open(d / "ways.pkl", "rb") as f:
        ways = pickle.load(f)
    with open(d / "rels.pkl", "rb") as f:
        rels = pickle.load(f)
    with open(d / "stops.pkl", "rb") as f:
        stops = pickle.load(f)
    with open(d / "infra.pkl", "rb") as f:
        infra = pickle.load(f)
    c = np.load(d / "coords.npz")
    cid, cx, cy = c["id"], c["x"], c["y"]
    g = shape(json.loads((RAW / "vn_boundary.geojson").read_text(encoding="utf-8")))
    if g.geom_type == "GeometryCollection":
        g = shapely.union_all(list(g.geoms))
    shapely.prepare(g)
    out = ~shapely.contains_xy(g, cx / 1e7, cy / 1e7)
    outside = set(cid[out].tolist())
    known = set(cid.tolist())
    keep_w, cut = {}, Counter()
    for wid, (tags, nodes) in ways.items():
        ns = [int(n) for n in nodes if int(n) in known]
        if ns and 2 * sum(n in outside for n in ns) >= len(ns):
            cut["abroad: " + (tags.get("name") or "(unnamed)")] += 1
        elif NOT_OPEN_TRACK.match(tags.get("name") or ""):
            cut["not open: " + tags["name"]] += 1
        else:
            keep_w[wid] = (tags, nodes)
    keep_s = {k: v for k, v in stops.items() if bool(shapely.contains_xy(g, v[1], v[2]))}
    kept = {("w", k) for k in keep_w} | {("n", k) for k in keep_s}
    routes = {k for k, (tags, members) in rels.items()
              if tags.get("type") == "route" and k not in NOT_OPEN_ROUTES
              and any((t, r) in kept for t, r, _ in members)}
    keep_r = {k: v for k, v in rels.items()
              if k in routes or (v[0].get("type") != "route"
                                 and any(t == "r" and r in routes for t, r, _ in v[1]))}
    gone = set(ways) - set(keep_w)
    keep_i = {k: v for k, v in infra.items()
              if not (any(t == "w" and r in gone for t, r, _ in v[1])
                      and not any(t == "w" and r in keep_w for t, r, _ in v[1]))}
    for name, v in sorted(cut.items(), key=lambda kv: -kv[1]):
        log(f"  cut {v:4d}  {name}")
    log(f"  routes left out: {sorted(set(rels) - set(keep_r))}")
    log(f"VN clip: kept {len(keep_w)}/{len(ways)} ways, {len(keep_s)}/{len(stops)} stops, "
        f"{len(keep_r)}/{len(rels)} relations, {len(keep_i)}/{len(infra)} infra relations")
    for fn, obj in (("ways.pkl", keep_w), ("rels.pkl", keep_r), ("stops.pkl", keep_s),
                    ("infra.pkl", keep_i)):
        tmp = d / (fn + ".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(obj, f, protocol=4)
        os.replace(tmp, d / fn)


def report():
    adopt()
    load_osm(print)


if __name__ == "__main__":
    if "--clip" in sys.argv:
        clip()
    elif "--report" in sys.argv:
        report()
    else:
        print(__doc__)
