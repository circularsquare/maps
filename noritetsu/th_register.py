"""Thailand: the State Railway of Thailand's lines, from OpenStreetMap's named track and its
infrastructure relations, through kr_register's recipe (as gb_register does).

    python th_register.py --report     # the way-to-line assignment: track km per line, what was
                                       # left out, stations and how far they lie from a line
    python build_model.py --region th --register th_register:data/raw/th

THE LINE UNIT is SRT's own list of lines and branches (en.wikipedia "Rail transport in
Thailand", th.wikipedia per line; th_sources.md has the table): the Northern Line Bangkok -
Chiang Mai and the Sawankhalok branch; the Northeastern Line as its three parts, Ban Phachi -
Ubon Ratchathani, Thanon Chira - Nong Khai and Kaeng Khoi - Bua Yai; the Eastern Line Bangkok -
Aranyaprathet and Chachoengsao - Sattahip (Ban Phlu Ta Luang); the Southern Line Thon Buri -
Su-ngai Kolok and its branches (Suphan Buri, Nam Tok, Khiri Rat Nikhom, Kantang, Nakhon Si
Thammarat, Padang Besar); and the Mae Klong Railway's two separate pieces. Freight-only lines
(Khlong Sip Kao - Kaeng Khoi, Map Ta Phut, Laem Chabang, Mae Nam) and the lines being built
(Den Chai - Chiang Khong, Ban Phai - Nakhon Phanom, the high-speed line) are no register
lines (FREIGHT, NOT_OPEN). Bangkok's metros, monorails, the Airport Rail Link and the SRT Red
Lines stay OSM lines, as everywhere: their track is never register track.

WHICH WAY IS ON WHICH LINE. OSM Thailand names 79% of its main-line track for the line (the
probe: `python probe_kr_ways.py --region th`), but coarsely: one "สายตะวันออกเฉียงเหนือ" for
both Isan main lines, "ทางรถไฟสายตะวันออก" for the Aranyaprathet and the Sattahip lines, and
it leaves about 1,000 km unnamed. It also has route=railway relations (infra.pkl) for most
lines and branches, which cover most unnamed track. So, per way, in order:
  1. its name, through NAME_LINE (spellings and old/new alignments of one line);
  2. else the infrastructure relation it is in, through REL_LINE;
  3. else the line its neighbours at both ends share (`propagate`).
Then the coarse names are split where SRT's lines part (`split`): the Northeastern track at
Thanon Chira Junction, the Eastern track at Chachoengsao, the Mae Klong track at the Tha Chin.

WHICH STATIONS ARE ON A LINE: no open per-line station list exists (the national timetable feed
is partial, th_sources.md). So every OSM rail station within LIST_M of a line's own track is
listed on that line (Bangkok's ARL, BTS and MRT records left out first: `urban_only`), and
kr_register then finds it on the track as it does a published list (MATCH_M), plus any stop
node lying on the track. A branch whose track ends on another line's takes the station nearest
that end within JUNCTION_M (SRT counts a branch from its junction station), and a line left in
pieces by a junction yard OSM did not name is joined over other track (`join_pieces`).

BORDERS. Border points of borders.load() naming "th", and BORDERS below until those are in
borders.EXTRA, become stations of the line whose track passes within BORDER_M (gb_register's
way), so Padang Besar, Nong Khai and Aranyaprathet run to the border.

The `path` argument is data/raw/th; the OSM half is read from data/proc/th (extract.py).
"""
import hashlib
import math
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import kr_register as kr

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
REGION = "th"
SRT = "การรถไฟแห่งประเทศไทย"
SRT_EN = "State Railway of Thailand"

# key -> (name, English name). The names are SRT's (th.wikipedia's article titles less
# "ทางรถไฟ"); English after en.wikipedia's list.
LINES = {
    "north": ("สายเหนือ", "Northern Line"),
    "sawankhalok": ("สายสวรรคโลก", "Sawankhalok Line"),
    "ubon": ("สายชุมทางบ้านภาชี–อุบลราชธานี", "Northeastern Line (Ubon Ratchathani)"),
    "nongkhai": ("สายชุมทางถนนจิระ–หนองคาย", "Northeastern Line (Nong Khai)"),
    "buayai": ("สายชุมทางแก่งคอย–ชุมทางบัวใหญ่", "Kaeng Khoi - Bua Yai Line"),
    "east": ("สายตะวันออก", "Eastern Line"),
    "sattahip": ("สายชุมทางฉะเชิงเทรา–สัตหีบ", "Eastern Seaboard Line (Chachoengsao - Sattahip)"),
    "south": ("สายใต้", "Southern Line"),
    "suphan": ("สายสุพรรณบุรี", "Suphan Buri Line"),
    "namtok": ("สายน้ำตก", "Nam Tok Line"),
    "khirirat": ("สายคีรีรัฐนิคม", "Khiri Rat Nikhom Line"),
    "kantang": ("สายกันตัง", "Kantang Line"),
    "nakhonsi": ("สายนครศรีธรรมราช", "Nakhon Si Thammarat Line"),
    "padang": ("สายชุมทางหาดใหญ่–ปาดังเบซาร์", "Padang Besar Line"),
    "mahachai": ("สายแม่กลอง (วงเวียนใหญ่–มหาชัย)", "Maeklong Railway (Wongwian Yai - Maha Chai)"),
    "maeklong": ("สายแม่กลอง (บ้านแหลม–แม่กลอง)", "Maeklong Railway (Ban Laem - Mae Klong)"),
}

# OSM way names -> line key; None leaves the way out of the register.
NAME_LINE = {
    "สายเหนือ": "north", "ทางรถไฟสายเหนือ": "north",
    # the shared trunk Bangkok - Ban Phachi Junction, counted with the Northern Line as SRT
    # counts it (its 751 km to Chiang Mai include it)
    "ทางรถไฟสายเหนือ/ตะวันออกเฉียงเหนือ": "north", "ทางรถไฟสายเหนือ/ตะวันออกฉียงเหนือ": "north",
    "ทางรถไฟสายเหนือ-ตะวันออกเฉียงเหนือ": "north",
    # the at-grade line Hua Lamphong - Bang Sue - Rangsit, which the ordinary and commuter
    # trains from Hua Lamphong still run on beside the elevated line
    "ทางรถไฟสายเหนือ/ตะวันออกเฉียงเหนือเดิม": "north",
    "สายเลี่ยงเมืองลพบุรี": "north", "ทางรถไฟเลี่ยงเมืองชุมทางบ้านภาชี": "north",
    "สายสวรรคโลก": "sawankhalok",
    "สายตะวันออกเฉียงเหนือ": "ubon",          # split: Nong Khai's part in `split`
    "ทางคู่เลี่ยงเมืองชุมทางแก่งคอย": "ubon",
    "สายแก่งคอย - บัวใหญ่": "buayai",
    "ทางรถไฟสายตะวันออก": "east",             # split: Sattahip's part in `split`
    "สายใต้": "south", "ทางรถไฟสายใต้": "south", "สายใต้เดิม": "south",
    "สายสุพรรณบุรี": "suphan",
    "สายกาญจนบุรี": "namtok",
    "สายคีรีรัฐนิคม": "khirirat",
    "สายนครศรีธรรมราช": "nakhonsi",
    "สายชุมทางหาดใหญ่–ปาดังเบซาร์": "padang", "Hat Yai-Padang Besar": "padang",
    "ทางรถไฟสายแม่กลอง": "mahachai",          # split: Ban Laem - Mae Klong in `split`
    "ทางรถไฟสายมหาชัย": "mahachai",
    # freight only (FREIGHT in th_sources.md)
    "ทางรถไฟสายชุมทางคลองสิบเก้า–ชุมทางแก่งคอย": None, "ทางรถไฟสายมาบตาพุด": None,
    # not open
    "ทางรถไฟสายเด่นชัย–เชียงราย–เชียงของ": None,
    # OSM lines: the Airport Rail Link, the Red Lines, KTM's track at Padang Besar
    "รถไฟฟ้าเอราวัน": None, "รถไฟฟ้าเอราวัน (แอร์พอร์ต เรล ลิงก์)": None,
    "รถไฟฟ้าแอร์พอร์ต เรล ลิงก์": None, "รถไฟฟ้าชานเมืองสายสีแดงเข้ม": None,
    "รถไฟฟ้าชานเมืองสายสีแดงอ่อน": None, "KTM": None,
    "ทางรถไฟสายประวัติศาสตร์": None,
}
# Names on track that are a structure's, not a line's: read as no name.
JUNK = {"สะพานดำ", "สะพานขาวทาชมภู"}

# route=railway relations -> line key, for their unnamed ways; None: never register track.
REL_LINE = {
    1820650: "north", 1820651: "ubon", 8425142: "ubon", 17458406: "nongkhai",
    13328007: "buayai", 8425166: "sawankhalok",
    1820706: "east", 9407411: "east",
    8342730: "south", 8425164: "south", 8425165: "south",
    8425163: "kantang", 13327648: "suphan", 13327676: "namtok", 13327778: "khirirat",
    13327899: "nakhonsi", 13195090: "padang",
    8425167: "mahachai", 8425168: "maeklong",
    21067324: "north",
    13196611: None, 13216187: None, 17456762: None, 20139466: None, 17056375: None,
}
# Relations the rule above overrides for their named ways: a way in the Ban Laem - Mae Klong
# relation is that line whatever its name says.
REL_FIRST = {8425168: "maeklong", 17458406: "nongkhai", 8425163: "kantang"}
# An unnamed way in several relations takes the first of their lines in this order: the
# Northern and Northeastern relations both hold the trunk Bangkok - Ban Phachi (188 km), and
# the Southern one the stretch to Bang Sue, which SRT counts with the Northern Line.
REL_PRIORITY = ["north", "south", "east", "ubon", "nongkhai", "buayai", "sawankhalok",
                "sattahip", "kantang", "suphan", "namtok", "khirirat", "nakhonsi", "padang",
                "mahachai", "maeklong"]

TRACK_KIND = {"rail": "rail", "narrow_gauge": "rail"}
NOT_PASSENGER = {"industrial", "military", "test", "tourism"}

# A station this close to a line's track is listed on it.
LIST_M = 250
# Border points this close to a line's track become its stations.
BORDER_M = 400
# Crossings with no border point in borders.load() yet: the point where the track crosses the
# national boundary in OSM (Overpass, 2026-10-03; th_sources.md "Borders"). To go into
# borders.EXTRA; a point already there within 50 m is used instead.
BORDERS = [
    ("xPadangBesar", 100.322477, 6.665252, ["my", "th"]),
    ("xNongKhaiThanaleng", 102.715092, 17.880451, ["la", "th"]),
    ("xAranyaprathetPoipet", 102.550138, 13.661698, ["kh", "th"]),
]
# Crossings passenger trains run over (th_sources.md "Borders"): SRT's 45/46 and the Hat Yai
# shuttles to Padang Besar's joint station on the Malaysian side; 133/134 and 147/148 to
# Khamsavath (Vientiane). The section from the last Thai station to the border point is kept
# though no OSM route runs over it (`served_sections`, a build_model hook). Nothing crosses at
# Khlong Luk: SRT's trains end at Ban Khlong Luk Border station, and that last 0.4 km drops.
BORDER_SERVED = {"xPadangBesar", "xNongKhaiThanaleng"}
# OSM stations no passenger train calls at, on track that is otherwise a passenger line's:
# Laem Chabang is the port's freight yard at the end of a freight-only branch from Si Racha.
NOT_STOPS = {"แหลมฉบัง"}
# A line whose track ends on another line's (a branch at its junction) takes the station
# nearest that end within this, as SRT counts its branches from the junction station.
JUNCTION_M = 1000

_S = {}


def line_id(name):
    h = hashlib.blake2b(f"th|{name}".encode("utf-8"), digest_size=5)
    return "t" + h.hexdigest()


def tidy(name):
    n = (name or "").strip()
    return "" if n in JUNK else n


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


def split(key, lon, lat):
    """Where one coarse OSM name is two of SRT's lines, away from the junction: the Nong Khai
    line north of 15.1 N (the Ubon line never leaves 14.85-15.25 N, and is east of 103 E
    where it is north of 15.05), the Sattahip line south of Chachoengsao (13.69 N; the
    Aranyaprathet line's lowest point is the border at 13.66 N, at 102.55 E), the Mae Klong
    Railway's two pieces either side of the Tha Chin. Near the junctions `split_components`
    decides."""
    if key == "ubon" and lat > 15.1 and lon < 103.0:
        return "nongkhai"
    if key == "east" and lat < 13.62 and lon < 101.5:
        return "sattahip"
    if key == "mahachai" and lon < 100.265:
        return "maeklong"
    return key


# Near the junctions: (line, junction station, the branch's first station, the branch's line).
# The branch is the track reachable from that station without passing within SPLIT_M of the
# junction station.
SPLITS = [
    ("ubon", "ชุมทางถนนจิระ", "บ้านเกาะ", "nongkhai"),
    ("east", "ชุมทางฉะเชิงเทรา", "ดอนสีนนท์", "sattahip"),
]
SPLIT_M = 400


def split_components(ways, coords, line_of, stops, log):
    pts = defaultdict(list)
    for _nid, (t, lon, lat) in stops.items():
        if t.get("railway") in ("station", "halt") and t.get("name"):
            pts[t["name"]].append((lon, lat))
    for key, jn, far, new in SPLITS:
        if not pts.get(jn) or not pts.get(far):
            log(f"TH: split {key} -> {new}: no station {jn} or {far}; not split")
            continue
        jx, jy = pts[jn][0]
        # a name can be on several lines (บ้านเกาะ): the one nearest the junction
        fx, fy = min(pts[far], key=lambda p: kr.dist_m(p[0], p[1], jx, jy))
        # over both lines' track: OSM has pieces of the coarse name inside the branch's
        # relation, which reach the branch's far station only through it
        ws = [w for w, k in line_of.items() if k in (key, new)]
        adj = defaultdict(set)
        xy = {}
        for w in ws:
            nl = np.asarray(ways[w][1], dtype=np.int64)
            pos, ok = coords.many(nl)
            for n, p, g in zip(nl.tolist(), pos.tolist(), ok.tolist()):
                if g:
                    xy[n] = (coords.x[p] / 1e7, coords.y[p] / 1e7)
            nl = [n for n in nl.tolist() if n in xy]
            for a, b in zip(nl[:-1], nl[1:]):
                adj[a].add(b)
                adj[b].add(a)
        removed = {n for n, (x, y) in xy.items() if kr.dist_m(x, y, jx, jy) <= SPLIT_M}
        start = min(xy, key=lambda n: kr.dist_m(*xy[n], fx, fy))
        seen, todo = {start}, [start]
        while todo:
            u = todo.pop()
            for v in adj[u]:
                if v not in seen and v not in removed:
                    seen.add(v)
                    todo.append(v)
        moved = 0
        for w in ws:
            if line_of[w] != key:
                continue
            nl = [n for n in np.asarray(ways[w][1]).tolist() if n in xy]
            if nl and sum(1 for n in nl if n in seen) * 2 > len(nl):
                line_of[w] = new
                moved += 1
        log(f"TH: split {key}: {moved} of {sum(1 for w in ws if line_of[w] in (key, new))} "
            f"ways beyond {jn} towards {far} are {new}")


def assign(ways, coords, infra, stops, log):
    """{way: line key} for every way that is register track."""
    geo = way_geo(ways, coords)
    inrel = defaultdict(list)
    for rid, (tags, members) in infra.items():
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
        first = [REL_FIRST[r] for r in inrel[wid] if r in REL_FIRST]
        if n and n in NAME_LINE:
            key = NAME_LINE[n]
            if key is None:
                left[n] += km
                continue
            if first and key in ("ubon", "mahachai", "south"):
                key = first[0]
        elif n:
            unknown[n] += km
            continue
        else:
            if svc not in (None, "crossover"):
                continue
            if any(r in REL_LINE and REL_LINE[r] is None for r in inrel[wid]):
                continue
            keys = sorted((REL_LINE[r] for r in inrel[wid] if REL_LINE.get(r)),
                          key=REL_PRIORITY.index)
            key = first[0] if first else (keys[0] if keys else None)
            if key is None:
                cand.append(wid)
                continue
        out[wid] = split(key, lon, lat)
    got = propagate(ways, out, cand, log)
    out.update(got)
    split_components(ways, coords, out, stops, log)
    km_by = Counter()
    for w, k in out.items():
        if not ways[w][0].get("service"):
            km_by[k] += geo[w][0]
    log("TH: main-line track km per line (both tracks of double track): "
        + ", ".join(f"{k} {v:,.0f}" for k, v in km_by.most_common()))
    log("TH: named track left out: " + ", ".join(f"{n} {v:.0f}" for n, v in left.most_common()))
    if unknown:
        log("TH: names in no table, left out: "
            + ", ".join(f"{n} {v:.1f}" for n, v in unknown.most_common()))
    return out, geo


def propagate(ways, named, cand, log):
    """Unnamed track outside every relation takes the line its neighbours at both ends share,
    repeated until nothing changes; a dead end, the one line at its joined end."""
    node_ways = defaultdict(list)
    for w in set(named) | set(cand):
        nl = ways[w][1]
        node_ways[int(nl[0])].append(w)
        node_ways[int(nl[-1])].append(w)
    # interior nodes too: a way can end in the middle of another
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
    log(f"TH: {len(got)} unnamed ways outside every relation named from their neighbours; "
        f"{len(cand) - len(got)} left unnamed")
    return got


def load_osm(log):
    import build_model as bm
    ways, rels, stops, cid, cx, cy = bm.load(REGION, log)
    coords = bm.Coords(cid, cx, cy)
    with open(ROOT / "data" / "proc" / REGION / "infra.pkl", "rb") as f:
        infra = pickle.load(f)
    line_of, geo = assign(ways, coords, infra, stops, log)
    _S.update(ways=ways, rels=rels, stops=stops, coords=coords, line_of=line_of, geo=geo)
    _S["byobj"] = {id(ways[w][0]): LINES[k][0] for w, k in line_of.items()}
    return ways, stops, coords


def register_name(tags):
    return _S["byobj"].get(id(tags), "")


# ---------------------------------------------------------------- stations and borders

BORDER_BASE = -9_000_000_000


def border_points(log):
    pts = []
    try:
        import borders
        pts = [p for p in borders.load(canonical_only=True) if "th" in p["countries"]]
    except Exception as e:                          # noqa: BLE001
        log(f"TH: borders.load failed ({e})")
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


# Station records of Bangkok's urban lines, by operator or network, never SRT stops: the
# Airport Rail Link runs beside the Eastern Line from Phaya Thai to Hua Mak, and its
# Ratchaprarop and Ramkhamhaeng (no SRT halt) were put on the Eastern Line, while its Phaya
# Thai record, merged with SRT's halt of that name, took the halt off the line. The Red
# Lines' records stay: their stations stand where the at-grade line's halts are (Bang Khen,
# Lak Si, Don Mueang, Rangsit), often the only record there.
URBAN = ("แอร์พอร์ต", "Airport Rail Link", "ARL", "เอราวัน", "เอเชียเอราวัน", "BTS", "MRT",
         "บีทีเอส", "มหานคร", "ระบบขนส่งมวลชนกรุงเทพ", "Bangkok Expressway", "โมโนเรล",
         "กรุงเทพธนาคม")


def urban_only(tags):
    tag = " ".join(tags.get(k) or "" for k in ("operator", "network"))
    return (tags.get("station") in ("subway", "monorail", "light_rail", "funicular")
            or (any(u in tag for u in URBAN) and SRT not in tag))


# Track of the urban lines whose stop nodes are never SRT's (the Red Lines' are left in, as
# their stations are).
URBAN_TRACK = {"รถไฟฟ้าเอราวัน", "รถไฟฟ้าเอราวัน (แอร์พอร์ต เรล ลิงก์)", "รถไฟฟ้าแอร์พอร์ต เรล ลิงก์"}


def build_stations(stops, log):
    on_urban = set()
    for t, nodes in _S["ways"].values():
        if t.get("railway") in ("subway", "monorail", "light_rail", "funicular") or \
                tidy(t.get("name")) in URBAN_TRACK:
            on_urban.update(np.asarray(nodes).tolist())
    drop = {n for n, (t, _x, _y) in stops.items()
            if urban_only(t) or (n in on_urban and t.get("railway") != "station")}
    stops = {n: v for n, v in stops.items() if n not in drop}
    log(f"TH: {len(drop)} station and stop records of the urban lines left out of the register")
    st, node_st, by_key, by_base = _orig["build_stations"](stops, log)
    gone = {sid for sid, s in st.items() if s["name"] in NOT_STOPS}
    for sid in gone:
        del st[sid]
    for n in [n for n, s in node_st.items() if s in gone]:
        del node_st[n]
    for d in (by_key, by_base):
        for k in list(d):
            d[k] = [s for s in d[k] if s not in gone]
    log(f"TH: {len(gone)} station records no passenger train calls at left out "
        f"({', '.join(sorted(NOT_STOPS))})")
    border = {}
    for i, p in enumerate(border_points(log)):
        fid = BORDER_BASE - i
        st[fid] = {"name": p["id"], "name_en": "", "lon": p["lon"], "lat": p["lat"], "rank": 0}
        by_key[kr.name_key(p["id"])].append(fid)
        by_base[kr.base_key(p["id"])].append(fid)
        border[fid] = p["id"]
    _S.update(st=st, node_st=node_st, border=border)
    log(f"TH: {len(border)} border points offered to the lines ({', '.join(border.values())})")
    return st, node_st, by_key, by_base


def load_lists(path, log):
    """{line: [[station name, ...]]}: every rail station within LIST_M of the line's track."""
    ways, coords, st = _S["ways"], _S["coords"], _S["st"]
    stops = _S["stops"]
    by_line = defaultdict(list)
    for w, k in _S["line_of"].items():
        by_line[LINES[k][0]].append(w)
    trees = {}
    from scipy.spatial import cKDTree
    kx = 111.32 * math.cos(math.radians(13.0))
    for ln, ws in by_line.items():
        pos = []
        for w in ws:
            p, ok = coords.many(np.asarray(ways[w][1], dtype=np.int64))
            pos.append(p[ok])
        pos = np.concatenate(pos)
        x, y = coords.x[pos] / 1e7, coords.y[pos] / 1e7
        # densify is unnecessary for 250 m: OSM's vertices on Thai track are far closer
        trees[ln] = cKDTree(np.c_[x * kx, y * 110.57])
    lists = defaultdict(set)
    for sid, s in st.items():
        if sid in _S["border"]:
            continue
        for ln, tr in trees.items():
            d, _i = tr.query((s["lon"] * kx, s["lat"] * 110.57))
            if d * 1000 <= LIST_M:
                lists[ln].add(s["name"])
    # junction ends: a line's track that ends on another line's takes the station nearest
    # that end (Kantang's track begins 370 m from Thung Song Junction, Suphan Buri's 350 m from
    # Nong Pladuk Junction, Bua Yai's 890 m from Bua Yai Junction)
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
    log(f"TH: {n_junc} junction stations listed on the line whose track ends near them")
    # border points: on the line whose track passes within BORDER_M (to the segments)
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
            log(f"TH: border point {pid} on {best[1]} ({best[0]:.0f} m)")
        else:
            log(f"TH: border point {pid} is on no line's track")
    log(f"TH: station lists by proximity ({LIST_M} m): "
        f"{sum(len(v) for v in lists.values())} station-line pairs on {len(lists)} lines")
    return ({k: [sorted(v)] for k, v in lists.items()}, defaultdict(list), {})


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
    """Every rail way that is not left out (NAME_LINE None: freight lines, the urban lines,
    track being built) as one graph, for joining the pieces of a line through a junction's
    yard that OSM leaves unnamed or names for the other line (Ban Phachi, Lop Buri)."""

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
    nearest first (cn_register's join_pieces). OSM leaves the yards of junctions unnamed or
    names them for the other line, and kr_register's search stays on a line's own track."""
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
            log(f"TH: {l['name']} is still in {left} pieces")
    log(f"TH: {len(done)} gaps in a line joined over other track "
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
    # a junction station may lie up to JUNCTION_M from its branch's track
    kr.MATCH_M = JUNCTION_M


def build(path, log):
    adopt()
    import gb_register as gb
    lines, stations, geoms = kr.build(path, log)
    join_pieces(lines, stations, geoms, log)
    gb.drop_shortcuts(lines, geoms, log)
    border = {f"k{fid}": pid for fid, pid in _S.get("border", {}).items()}
    en_of = {v[0]: v[1] for v in LINES.values()}

    def r(sid):
        return border.get(sid) or ("t" + sid[1:] if sid.startswith("k") else sid)

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
        l["src"] = "th"
        l["operator"], l["operator_en"] = SRT, SRT_EN
        l["name_en"] = en_of.get(l["name"], "")
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
    log(f"TH: {len(lines)} register lines, {sum(l['km'] for l in lines):,.0f} km, "
        f"{len(out_st)} stations")
    for l in sorted(lines, key=lambda l: -l["km"]):
        log(f"    {l['km']:8.1f} km  {len(l['sections']):3d} sections  {l['name']}")
    return lines, out_st, out_geoms


def report():
    log = print
    load_osm(log)


if __name__ == "__main__":
    if "--report" in sys.argv:
        report()
    else:
        print(__doc__)
