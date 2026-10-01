"""Lines, stations and sections for Taiwan, with OpenStreetMap's named track as the geometry.

    python build_model.py --region tw --register tw_register:data/raw/tw

THE SHAPE IS KOREA'S (kr_register.py, whose docstrings explain the three traps this reuses the
answers to). Taiwan publishes no open line-geometry register either, and 98% of its main-line
track in OSM carries the name of its line (`python probe_kr_ways.py --region tw`): 縱貫線,
臺中線, 海岸線, 宜蘭線, 台灣高速鐵路, 捷運板南線. So each register line is the graph of the OSM
ways carrying its name, and OSM's route relations (every TRA and THSR train number is one)
stay operating patterns over it.

THE REGISTER UNIT is the legal line, as TRA itself lists them: 縱貫線 (north: 基隆-竹南, south:
彰化-高雄, two lines here because they do not meet), 臺中線, 海岸線, 成追線, 屏東線, 南迴線,
宜蘭線, 北迴線, 臺東線 and the six passenger branches; THSR as one line; each metro and light
rail line as its operator names it; the Alishan Forest Railway's 阿里山線 and 祝山線.

WHICH STATIONS ARE ON A TRA LINE comes from TRA's own open data (tw_sources.md), which is
keyed not by legal line but by the ticketing trunks 西部幹線 / 東部幹線 / 南迴線 with a
cumulative km per station. LEGAL below cuts those into legal lines by station code. TRA's
station file gives every stop's position, so a listed station is matched to the OSM station
of its name NEAREST THAT POSITION -- not nearest the track, which would hand TRA's 左營 to
THSR's 左營 beside 新左營 -- and a stop OSM lacks is made from TRA's point. Codes TRA's station
file does not name (0995, 3355, 7115, ...) are signal stations and junctions, and are skipped.
An OSM station whose stop node is on the line's own track joins as well (stations newer than
the file); the build log names every one, and none did on TRA in 2026-09.

THSR AND THE METROS take their stations from OSM's own route relations (ROUTES), the only open
list that needs no TDX account: every stop any train of the line calls at, and only those.
A section survives only between two stops some relation has one after the other, which drops
track no service runs (中和新蘆線's link between its branches). The Alishan lines use the
published table in OTHER_LISTS, matched by name.

TRACK NAMED FOR A STRUCTURE. Taiwan's OSM names long tunnels and bridges for themselves
(新觀音隧道 20.6 km on 北迴線, 中央隧道 on 南迴線, 八卦山隧道 on THSR), and leaves short bits
unnamed, which would leave a hole in the line at each. Such ways inherit a line by contact:
see `assign_ways`.

WHAT THE CHECK SAYS (check_model.py --region tw): every TRA section against TRA's own km,
median 0.997, none off by 5%; every line within 5% of its published length except the two
Taipei branches (0.1-0.2 km short at the junction) and two lines whose published figure
counts tail track (安坑輕軌, 高雄捷運橘線), as the REGISTER notes say.

The `path` argument is data/raw/tw; the OSM half is read from data/proc/tw (extract.py).
"""
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from kr_register import INF, Near, between, dist_m, line_graph, neighbours

ROOT = Path(__file__).resolve().parent

# Register line -> (English name, operator, English operator).
TRA = ("國營臺灣鐵路股份有限公司", "Taiwan Railway Corporation")
TRTC = ("臺北大眾捷運股份有限公司", "Taipei Rapid Transit Corporation")
NTMC = ("新北大眾捷運股份有限公司", "New Taipei Metro Corporation")
KRTC = ("高雄捷運股份有限公司", "Kaohsiung Rapid Transit Corporation")
LINES = {
    "縱貫線北段": ("Western Trunk Line (north)", *TRA),
    "縱貫線南段": ("Western Trunk Line (south)", *TRA),
    "臺中線": ("Taichung Line", *TRA),
    "海岸線": ("West Coast Line", *TRA),
    "成追線": ("Chengzhui Line", *TRA),
    "屏東線": ("Pingtung Line", *TRA),
    "南迴線": ("South-link Line", *TRA),
    "宜蘭線": ("Yilan Line", *TRA),
    "北迴線": ("North-link Line", *TRA),
    "臺東線": ("Taitung Line", *TRA),
    "平溪線": ("Pingxi Line", *TRA),
    "深澳線": ("Shen'ao Line", *TRA),
    "內灣線": ("Neiwan Line", *TRA),
    "六家線": ("Liujia Line", *TRA),
    "集集線": ("Jiji Line", *TRA),
    "沙崙線": ("Shalun Line", *TRA),
    "台灣高速鐵路": ("Taiwan High Speed Rail", "台灣高速鐵路股份有限公司",
                 "Taiwan High Speed Rail Corporation"),
    "阿里山線": ("Alishan Line", "林業及自然保育署", "Forestry and Nature Conservation Agency"),
    "祝山線": ("Zhushan Line", "林業及自然保育署", "Forestry and Nature Conservation Agency"),
    "淡水信義線": ("Tamsui-Xinyi Line", *TRTC),
    "新北投支線": ("Xinbeitou Branch Line", *TRTC),
    "松山新店線": ("Songshan-Xindian Line", *TRTC),
    "小碧潭支線": ("Xiaobitan Branch Line", *TRTC),
    "中和新蘆線": ("Zhonghe-Xinlu Line", *TRTC),
    "板南線": ("Bannan Line", *TRTC),
    "文湖線": ("Wenhu Line", *TRTC),
    "環狀線": ("Circular Line", *NTMC),
    "三鶯線": ("Sanying Line", *NTMC),
    "安坑輕軌": ("Ankeng Light Rail", *NTMC),
    "淡海輕軌綠山線": ("Danhai LRT Green Mountain Line", *NTMC),
    "淡海輕軌藍海線": ("Danhai LRT Blue Coast Line", *NTMC),
    "桃園機場捷運": ("Taoyuan Airport MRT", "桃園大眾捷運股份有限公司",
                "Taoyuan Metro Corporation"),
    "臺中捷運綠線": ("Taichung MRT Green Line", "臺中捷運股份有限公司",
                "Taichung Mass Rapid Transit Corporation"),
    "高雄捷運紅線": ("Kaohsiung MRT Red Line", *KRTC),
    "高雄捷運橘線": ("Kaohsiung MRT Orange Line", *KRTC),
    "高雄環狀輕軌": ("Kaohsiung Circular Light Rail", *KRTC),
}

# OSM track name -> the register line(s) it is. Names not here and not caught by `track_lines`'
# prefix rules are not register track (freight branches, the 舊山線 rail bikes, sugar railways).
TRACK = {
    "台灣高速鐵路": ("台灣高速鐵路",), "高速鐵路": ("台灣高速鐵路",),
    "海岸線": ("海岸線",),
    # 成功/追分 to 彰化: legally both lines run to 彰化, so the shared stretch is on both.
    "山線、海線共用路段": ("臺中線", "海岸線"),
    "成追線": ("成追線",), "屏東線": ("屏東線",), "南迴線": ("南迴線",),
    "北迴線": ("北迴線",), "台鐵東部幹線/宜蘭線": ("宜蘭線",),
    "平溪線": ("平溪線",), "深澳線": ("深澳線",), "內灣線": ("內灣線",), "六家線": ("六家線",),
    "集集線": ("集集線",), "沙崙線": ("沙崙線",),
    "阿里山線": ("阿里山線",), "阿里山森林鐵路祝山線": ("祝山線",),
    # A 0.66 km tunnel between 奮起湖 and 多林, named "collapsed old line", is the only track in
    # OSM joining the two halves of 阿里山線; trains have run through since 2024-07-06.
    "阿里山森林鐵路登山本線 (已崩塌舊線)": ("阿里山線",),
    "捷運淡水信義線": ("淡水信義線",),
    # Opened 2026-08-30 (象山-廣慈/奉天宮); OSM's station still says "under construction".
    "信義線東延段": ("淡水信義線",),
    "捷運新北投支線": ("新北投支線",),
    "捷運松山新店線": ("松山新店線",), "捷運小碧潭支線": ("小碧潭支線",),
    "捷運中和新蘆線": ("中和新蘆線",), "中和新蘆線": ("中和新蘆線",),
    "捷運板南線": ("板南線",), "板南線": ("板南線",),
    "捷運文湖線": ("文湖線",), "捷運環狀線": ("環狀線",), "捷運三鶯線": ("三鶯線",),
    "安坑輕軌": ("安坑輕軌",),
    "淡海輕軌綠山線": ("淡海輕軌綠山線",), "淡海輕軌藍海線": ("淡海輕軌藍海線",),
    "桃園機場捷運": ("桃園機場捷運",), "桃園機場捷運中壢延伸線": ("桃園機場捷運",),
    "臺中捷運綠線": ("臺中捷運綠線",),
    # Plain 捷運紅線 / 捷運橘線 are Kaohsiung's: every way so named lies there.
    "捷運紅線": ("高雄捷運紅線",), "高雄捷運紅線": ("高雄捷運紅線",),
    "高雄捷運橘線": ("高雄捷運橘線",), "捷運橘線": ("高雄捷運橘線",),
    "高雄環狀輕軌": ("高雄環狀輕軌",), "高雄捷運環狀輕軌": ("高雄環狀輕軌",),
}

# Which OSM route relations (by name) carry each non-TRA line's stops. Their stop members are
# the published list for these lines; OSM's own relations are the only open one that needs no
# TDX account (tw_sources.md). Stop nodes on the line's track then add nothing: 新店區公所's
# stop node lies on the track OSM names 小碧潭支線, which has two stations.
ROUTES = {
    "台灣高速鐵路": r"^台灣高鐵 ",
    "淡水信義線": r"淡水信義線|淡水線-信義線",
    "新北投支線": r"新北投支線",
    "松山新店線": r"松山新店線|^松山 => 台電大樓$|^台電大樓 => 松山$",
    "小碧潭支線": r"小碧潭支線",
    "中和新蘆線": r"中和新蘆線",
    "板南線": r"板南線|南港-板橋-土城線",
    "文湖線": r"文湖線",
    "環狀線": r"^(臺北捷運|新北捷運)環狀線",
    "三鶯線": r"三鶯線",
    "安坑輕軌": r"^安坑輕軌",
    "淡海輕軌綠山線": r"^淡海輕軌 (紅樹林|崁頂)",
    "淡海輕軌藍海線": r"淡海輕軌藍海線",
    "桃園機場捷運": r"機場捷運",
    "臺中捷運綠線": r"^臺中捷運綠線",
    "高雄捷運紅線": r"^高雄捷運紅線",
    "高雄捷運橘線": r"^高雄捷運橘線",
    "高雄環狀輕軌": r"^高雄環狀輕軌",
}

# 縱貫線's two halves are one name in OSM; they are split where they would meet, between 竹南
# (24.69) and 彰化 (24.08). Its triple track 基隆-臺北 is named 縱貫線西正線/東正線/中正線.
NORTH_OF = 24.4

# Track kinds, and track whose usage says no scheduled passenger rides it. Tourism is let
# through for the Alishan lines only, which are tagged usage=tourism but run to a timetable.
TRACK_KIND = {"rail": "rail", "subway": "subway", "light_rail": "light_rail",
              "monorail": "monorail", "tram": "tram", "funicular": "funicular",
              "narrow_gauge": "narrow_gauge", "preserved": "rail"}
NOT_PASSENGER = {"industrial", "military", "test", "tourism", "freight"}
TOURISM_OK = {"阿里山線", "祝山線"}
STRUCTURE = re.compile(r"(隧道|橋|遮體)")
# Signal stations OSM maps as railway=station (古莊號誌, 中央號誌): no passenger stops there.
SIGNAL = re.compile(r"號誌[站所]?$")
BRIDGE_SHARE = 0.5         # a run of unnamed track joins a line it touches at points this share
                           # of the run's own extent apart: it spans a gap in that line
LISTED_ON_RUN_M = 300      # ... or one it touches once, with one of the line's TRA stops on it
LISTED_OUT_M = 1000        # ... at least this far from where it touches
STUB_M = 60                # a service way this short can still be part of such a run
END_M = 250               # a line without a list takes the station this near a dead end of its track

STATION_RAILWAY = {"station", "halt", "tram_stop"}
RAIL_MODES = ("train", "subway", "light_rail", "monorail", "tram", "funicular")

STOP_TO_STATION_M = 1200   # a stop_position belongs to the station of its name this close
DUP_M = 500                # two station records of one name this close are one station
TRA_MATCH_M = 1500         # an OSM station of a TRA stop's name may be this far off TRA's point
MATCH_M = 800              # a listed station may be this far off its line's track
REL_MATCH_M = 150          # a route relation's stop node this far (it is normally on the track)
FOOT_M = 150               # a station cuts every track of its line within this of its anchors
FOOT_NARROW_M = 40         # on narrow gauge, how far off its track a listed station's node may
                           # sit before it is treated as mapped away from it
FAR_FOOT_M = 600          # the most a station mapped off its track reaches from its node
MERGE_M = 100              # two stations of one line closer than this are one station

# TRA's mileage file (鐵路里程) lists ticketing trunks, not legal lines. Each legal line is a
# run of pieces (trunk, from code, to code); km continue across pieces. Codes as in TRA's
# station file: 0900 基隆, 0920 八堵, 1190 北新竹, 1210 新竹, 1250 竹南, 3360 彰化, 4400 高雄,
# 5000 屏東, 5120 枋寮, 6000 臺東, 7000 花蓮, 7120 蘇澳, 7130 蘇澳新.
LEGAL = {
    "縱貫線北段": [("西部幹線", "0900", "1250")],
    "臺中線": [("西部幹線", "1250", "3360")],
    "縱貫線南段": [("西部幹線", "3360", "4400")],
    "海岸線": [("西部幹線 (海線)", "1250", "3360")],
    "成追線": [("成追線", "3350", "2260")],
    "屏東線": [("西部幹線", "4400", "5000"), ("南迴線", "5000", "5120")],
    "南迴線": [("南迴線", "5120", "6000")],
    "宜蘭線": [("東部幹線", "0920", "7130"), ("蘇澳新-蘇澳", "7130", "7120")],
    "北迴線": [("東部幹線", "7130", "7000")],
    "臺東線": [("東部幹線", "7000", "6000")],
    "平溪線": [("平溪線", "7330", "7336")],
    "深澳線": [("深澳線", "7360", "7362")],
    # 內灣線 is 新竹-內灣 in law and in OSM's track name; TRA's file starts it at 北新竹.
    "內灣線": [("西部幹線", "1210", "1190"), ("內灣線", "1190", "1208")],
    "六家線": [("六家線", "1193", "1194")],
    "集集線": [("集集線", "3430", "3436")],
    "沙崙線": [("沙崙線", "4270", "4272")],
}
SKIP_CODES = {"1001"}      # 臺北-環島, a ticketing alias of 臺北 at the same point

# Published station lists with chainage for lines outside TRA, by name (no coordinates, so a
# name is matched to the OSM station nearest the line's track, as in Korea). The Alishan Forest
# Railway's, from zh.wikipedia 阿里山林業鐵路 (tw_sources.md). 木屐寮 has a station in OSM but
# no figure in the table, so 阿里山線 gets no km_official; 沼平 is 阿里山 + 1.3 from the
# 祝山線 table, where OSM's 阿里山線 track ends.
OTHER_LISTS = {
    "阿里山線": [("嘉義", 0), ("北門", 1.6), ("鹿麻產", 10.8), ("竹崎", 14.2), ("樟腦寮", 23.3),
             ("獨立山", 27.4), ("梨園寮", 31.4), ("交力坪", 34.9), ("水社寮", 40.5),
             ("奮起湖", 45.8), ("多林", 50.9), ("十字路", 55.3), ("屏遮那", 60.5),
             ("第一分道", 62.7), ("二萬平", 66.8), ("神木", 69.6), ("阿里山", 71.6),
             ("沼平", 72.9)],
    "祝山線": [("沼平", 0), ("十字分道", 1.6), ("對高岳", 3.6), ("祝山", 4.95)],
}


def line_id(name):
    h = hashlib.blake2b(f"tw|{name}".encode("utf-8"), digest_size=5)
    return "t" + h.hexdigest()


def name_key(name):
    """One spelling for a station name: NFKC, no whitespace, 台 as 臺, no trailing 車站/站.
    OSM writes THSR's 台北 and the metro's 台北車站 where TRA writes 臺北."""
    n = unicodedata.normalize("NFKC", name or "")
    n = re.sub(r"\s+", "", n).replace("台", "臺")
    for suf in ("火車站", "車站", "站"):
        if n.endswith(suf) and len(n) - len(suf) >= 2:
            n = n[: -len(suf)]
            break
    return n


def base_key(name):
    """Without a bracketed sub-name: 左營(舊城) -> 左營, 新城 (太魯閣) -> 新城."""
    k = name_key(name)
    b = re.sub(r"[(（].*?[)）]", "", k)
    return name_key(b) or k


def track_lines(tags, lat):
    """The register line(s) an OSM way's name makes it, or ()."""
    n = unicodedata.normalize("NFKC", tags.get("name") or "").strip()
    if not n:
        return ()
    if n in TRACK:
        return TRACK[n]
    if n.startswith(("縱貫線", "縱貫鐵路")):
        return ("縱貫線北段",) if lat >= NORTH_OF else ("縱貫線南段",)
    if n.startswith("宜蘭線"):
        return ("宜蘭線",)
    if n.startswith(("臺東線", "台東線")):
        return ("臺東線",)
    if n.startswith("臺中線"):
        return ("臺中線",)
    return ()


def along(adj, sources, limit):
    """Every vertex within `limit` km of `sources` along the track: {vertex: km}."""
    import heapq
    dist = {s: 0.0 for s in sources}
    heap = [(0.0, s) for s in sources]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist.get(u, INF):
            continue
        for v, w in adj.get(u, ()):
            nd = d + w
            if nd <= limit and nd < dist.get(v, INF):
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return dist


def load_osm(log):
    import build_model as bm
    ways, rels, stops, cid, cx, cy = bm.load("tw", log)
    return ways, rels, stops, bm.Coords(cid, cx, cy)


def route_lists(rels, stops, node_st, log):
    """{line: {station: [(lon, lat) of its stop nodes]}} from OSM's route relations."""
    out = defaultdict(lambda: defaultdict(list))
    nxt = defaultdict(set)                  # line -> {frozenset(a, b)}: consecutive stops
    for _rid, (tags, members) in rels.items():
        if tags.get("type") != "route":
            continue
        rname = unicodedata.normalize("NFKC", tags.get("name") or "")
        for line, pat in ROUTES.items():
            if not re.search(pat, rname):
                continue
            seq = []
            for ty, ref, role in members:
                if ty == "n" and ref in node_st and ref in stops:
                    _t, lon, lat = stops[ref]
                    out[line][node_st[ref]].append((lon, lat))
                    if not seq or seq[-1] != node_st[ref]:
                        seq.append(node_st[ref])
            nxt[line].update(frozenset((f"t{a}", f"t{b}")) for a, b in zip(seq[:-1], seq[1:]))
    log(f"TW: route relations give station lists for {len(out)} lines: "
        + ", ".join(f"{k} {len(v)}" for k, v in sorted(out.items())))
    return out, nxt


def load_tra(path, log):
    """TRA's legal lines as {line: [(code, name, name_en, lon, lat, km), ...]} in order."""
    p = Path(path)
    st = {s["stationCode"]: s for s in json.loads((p / "tra_stations.json").read_text("utf-8"))}
    trunks = defaultdict(list)
    for r in json.loads((p / "tra_mileage.json").read_text("utf-8")):
        trunks[r["lineName"]].append((float(r["staMil"]), r["fkSta"]))
    out = {}
    for line, pieces in LEGAL.items():
        rows, at = [], 0.0
        for trunk, a, b in pieces:
            km = {c: k for k, c in trunks[trunk]}
            ka, kb = km[a], km[b]
            lo, hi = min(ka, kb), max(ka, kb)
            run = sorted(((k, c) for k, c in trunks[trunk] if lo <= k <= hi),
                         key=lambda kc: kc[0], reverse=ka > kb)
            for k, c in run:
                if c in SKIP_CODES or c not in st:
                    continue
                cum = at + abs(k - ka)
                if rows and rows[-1][0] == c:
                    continue                   # the join between two pieces
                s = st[c]
                lat, lon = (float(v) for v in s["gps"].split())
                rows.append((c, s["stationName"], s["stationEName"], lon, lat, round(cum, 3)))
            at += abs(kb - ka)
        out[line] = rows
    for line, rows in OTHER_LISTS.items():
        out[line] = [(nm, nm, "", None, None, km) for nm, km in rows]
    log(f"TW: TRA lists for {len(out)} legal lines, {sum(len(r) for r in out.values())} "
        f"station rows, from {len(st)} stations")
    return out


def build_stations(stops, log):
    """OSM's rail stations, one per complex, and every stop node mapped onto one."""
    st = {}
    for nid, (tags, lon, lat) in stops.items():
        rail = (tags.get("railway") in STATION_RAILWAY
                or (tags.get("public_transport") == "station"
                    and any(tags.get(m) == "yes" for m in RAIL_MODES)))
        if (rail and tags.get("name") and tags.get("railway") != "disused_station"
                and not SIGNAL.search(tags["name"])):
            st[nid] = {"name": tags["name"], "name_en": tags.get("name:en") or "",
                       "lon": lon, "lat": lat,
                       "rank": 0 if tags.get("railway") == "station" else 1}
    by_key = defaultdict(list)
    for nid, s in st.items():
        by_key[base_key(s["name"])].append(nid)
    alias = {}
    for ids in by_key.values():
        ids.sort(key=lambda n: (st[n]["rank"], n))
        for i, a in enumerate(ids):
            if a in alias:
                continue
            for b in ids[i + 1:]:
                if b not in alias and dist_m(st[a]["lon"], st[a]["lat"],
                                             st[b]["lon"], st[b]["lat"]) <= DUP_M:
                    alias[b] = a
                    if not st[a]["name_en"]:
                        st[a]["name_en"] = st[b]["name_en"]
    for b in alias:
        del st[b]
    # A station mapped only as stop positions still exists (as in kr_register).
    by_key = defaultdict(list)
    for nid, s in st.items():
        by_key[base_key(s["name"])].append(nid)
    made = 0
    for nid, (tags, lon, lat) in sorted(stops.items()):
        rail_stop = (tags.get("railway") == "stop"
                     or (tags.get("public_transport") == "stop_position"
                         and any(tags.get(m) == "yes" for m in RAIL_MODES)))
        if not rail_stop or not tags.get("name") or nid in st or SIGNAL.search(tags["name"]):
            continue
        k = base_key(tags["name"])
        if any(dist_m(lon, lat, st[c]["lon"], st[c]["lat"]) <= STOP_TO_STATION_M
               for c in by_key.get(k, ())):
            continue
        st[nid] = {"name": tags["name"], "name_en": tags.get("name:en") or "",
                   "lon": lon, "lat": lat, "rank": 2}
        by_key[k].append(nid)
        made += 1
    by_key, by_base = defaultdict(list), defaultdict(list)
    for nid, s in st.items():
        by_key[name_key(s["name"])].append(nid)
        by_base[base_key(s["name"])].append(nid)
    node_st = {}
    for nid, (tags, lon, lat) in stops.items():
        if nid in st:
            node_st[nid] = nid
        elif nid in alias:
            node_st[nid] = alias[nid]
        elif tags.get("name"):
            best, bd = None, STOP_TO_STATION_M
            for c in (by_key.get(name_key(tags["name"]))
                      or by_base.get(base_key(tags["name"]), ())):
                d = dist_m(lon, lat, st[c]["lon"], st[c]["lat"])
                if d <= bd:
                    best, bd = c, d
            if best is not None:
                node_st[nid] = best
    log(f"TW: {len(st)} OSM rail stations ({len(alias)} records merged into a complex, "
        f"{made} made from stop positions alone), {len(node_st)} stop nodes placed on one")
    return st, node_st, by_key, by_base


def assign_ways(ways, coords, listed_pts, log):
    """{register line: [way id]}: by name, then structures and unnamed track by contact."""
    by_line = defaultdict(list)
    touch = defaultdict(set)                          # node -> register lines through it
    cands = {}
    for wid, (tags, nodes) in ways.items():
        if tags.get("railway") not in TRACK_KIND:
            continue
        p = coords.get(int(nodes[len(nodes) // 2]))
        lat = p[1] if p else 0.0
        lines = track_lines(tags, lat)
        if lines:
            if tags.get("usage") in NOT_PASSENGER and not set(lines) <= TOURISM_OK:
                continue
            for ln in lines:
                by_line[ln].append(wid)
            for n in nodes:
                touch[int(n)].update(lines)
            continue
        name = tags.get("name") or ""
        if tags.get("usage") in ("industrial", "military", "test"):
            continue
        if tags.get("service"):
            # Sidings and yards stay out, except a stub a few metres long: 枋野二號隧道 meets
            # the rest of 南迴線 through a 10 m way tagged service=siding.
            ps = [p for p in (coords.get(int(n)) for n in nodes) if p]
            if sum(dist_m(*a, *b) for a, b in zip(ps[:-1], ps[1:])) > STUB_M:
                continue
        if not name or STRUCTURE.search(name):
            cands[wid] = [int(n) for n in nodes]
    # Connected runs of candidate ways, and the lines each run touches, and where.
    parent = {w: w for w in cands}

    def find(w):
        while parent[w] != w:
            parent[w] = parent[parent[w]]
            w = parent[w]
        return w
    first = {}
    for w, nodes in cands.items():
        for n in nodes:
            if n in first:
                a, b = find(first[n]), find(w)
                if a != b:
                    parent[a] = b
            else:
                first[n] = w
    runs = defaultdict(list)
    for w in cands:
        runs[find(w)].append(w)
    # A run joins a line when it BRIDGES that line's track: it touches it at two places as far
    # apart as the run is long, give or take. Two touches a few metres apart at one end of a
    # 12 km run are the ends of double track, and counting those put 臺東-山里 (unnamed, and
    # 山里隧道) on 南迴線, which merely ends at 臺東.
    # A run that touches a line once joins it only if one of the line's TRA stops is on it:
    # that is how 臺東線 gets its own south end back.
    got = Counter()
    for members in runs.values():
        at = defaultdict(set)
        for w in members:
            for n in cands[w]:
                for ln in touch.get(n, ()):
                    at[ln].add(n)
        if not at:
            continue
        run_xy = np.array([p for w in members for p in (coords.get(n) for n in cands[w]) if p])
        lo, hi = run_xy.min(axis=0), run_xy.max(axis=0)
        extent = dist_m(lo[0], lo[1], hi[0], hi[1])
        for ln, ns in at.items():
            ps = [p for p in (coords.get(n) for n in ns) if p]
            span = max((dist_m(*a, *b) for a in ps for b in ps), default=0.0)
            ok = len(ps) >= 2 and span >= BRIDGE_SHARE * extent
            if not ok and listed_pts.get(ln):
                # The stop must be out along the run, not where it meets the line: 臺東 is on
                # 南迴線's list and beside the south end of the 臺東-山里 run.
                for lon, lat in listed_pts[ln]:
                    d = np.hypot((run_xy[:, 0] - lon) * np.cos(np.radians(lat)) * 111320,
                                 (run_xy[:, 1] - lat) * 110570)
                    if (d.min() <= LISTED_ON_RUN_M
                            and min(dist_m(lon, lat, *p) for p in ps) >= LISTED_OUT_M):
                        ok = True
                        break
            if ok:
                by_line[ln].extend(members)
                got[ln] += len(members)
    log(f"TW: {sum(got.values())} unnamed or structure-named ways joined a line by contact "
        f"({', '.join(f'{k} {v}' for k, v in got.most_common(8))})")
    return by_line


def build(path, log):
    ways, rels, stops, coords = load_osm(log)
    st, node_st, by_key, by_base = build_stations(stops, log)
    tra = load_tra(path, log)
    by_line = assign_ways(ways, coords,
                          {ln: [(r[3], r[4]) for r in rows if r[3] is not None]
                           for ln, rows in tra.items()}, log)
    for ln in sorted(set(LINES) - set(by_line)):
        log(f"  TW: no OSM track found for {ln}")

    # TRA's stops that OSM lacks become stations of their own, at TRA's point.
    made = {}
    stations, lines, geoms = {}, [], {}
    unmatched, dropped, missing = [], [], []
    n_vertex = n_listed = n_end = n_route = official_lines = 0
    rlists, rnext = route_lists(rels, stops, node_st, log)
    cut = []
    # Every point that stands for a station: its own record and every stop node placed on it.
    pts = [(stops[n][1], stops[n][2], s) for n, s in node_st.items()]
    pt_lon = np.array([p[0] for p in pts])
    pt_lat = np.array([p[1] for p in pts])
    pt_st = [p[2] for p in pts]
    extra = defaultdict(list)                          # line -> OSM-only stations it picked up

    def station_rec(sid):
        if sid in made:
            return made[sid]
        s = st[int(sid[1:])]
        return {"name": s["name"], "name_en": s["name_en"], "lon": s["lon"], "lat": s["lat"],
                "rank": s["rank"]}

    for name in sorted(by_line):
        if name not in LINES:
            continue
        wids = sorted(set(by_line[name]))
        adj, xy, fast = line_graph(wids, ways, coords)
        if len(xy) < 2:
            continue
        near = Near(xy)
        kinds = Counter(TRACK_KIND[ways[w][0]["railway"]] for w in wids)
        narrow = kinds.most_common(1)[0][0] == "narrow_gauge"
        foot_m = FOOT_NARROW_M if narrow else FOOT_M

        # --- which stations, and where on this line's track each one is (its anchors)
        anchors = defaultdict(set)
        if name in rlists:
            # Only the relations' stations, each where its own stop node meets this track.
            # A branch's own track can stop short of the junction platform (小碧潭支線 at
            # 七張), so a stop further off still counts if it is beyond a dead end of it.
            ends = [v for v, nb in adj.items() if len(nb) == 1]
            end_near = Near({v: xy[v] for v in ends}) if ends else None
            for cplx, pts in rlists[name].items():
                best, bd = None, REL_MATCH_M
                for lon, lat in pts:
                    v, d = near.nearest(lon, lat)
                    if d <= bd:
                        best, bd = v, d
                if best is None and end_near is not None:
                    bd = END_M
                    for lon, lat in pts:
                        v, d = end_near.nearest(lon, lat)
                        if d <= bd:
                            best, bd = v, d
                if best is not None:
                    anchors[f"t{cplx}"].add(best)
                    n_route += 1
        else:
            for n in adj:
                if n in node_st:
                    anchors[f"t{node_st[n]}"].add(n)
            n_vertex += len(anchors)
        listed = {}                                    # sid -> TRA code
        listed_order = []
        far = {}
        cum = {}
        for code, nm, nm_en, lon, lat, km in tra.get(name, ()):
            # Both spellings: TRA's 左營 is OSM's 左營(舊城), and the plain 左營 in OSM is
            # THSR's, 1.9 km away.
            cands = set(by_key.get(name_key(nm), ())) | set(by_base.get(base_key(nm), ()))
            best, bd = None, TRA_MATCH_M
            if lon is None:                   # no point of its own: nearest the track
                bd = MATCH_M
                for c in cands:
                    d = near.nearest(st[c]["lon"], st[c]["lat"])[1]
                    if d <= bd:
                        best, bd = c, d
                if best is None:
                    unmatched.append((name, nm, None))
                    continue
            for c in (cands if lon is not None else ()):
                d = dist_m(lon, lat, st[c]["lon"], st[c]["lat"])
                if d <= bd:
                    best, bd = c, d
            if best is not None:
                sid = f"t{best}"
                plon, plat = st[best]["lon"], st[best]["lat"]
            else:
                sid = f"tra{code}"
                made[sid] = {"name": nm, "name_en": nm_en, "lon": lon, "lat": lat, "rank": 0}
                plon, plat = lon, lat
            v, d = near.nearest(plon, plat)
            if d > MATCH_M:
                unmatched.append((name, nm, round(d)))
                continue
            listed[sid] = code
            cum[sid] = km
            listed_order.append(sid)
            st_en = nm_en
            if sid in made:
                pass
            elif st_en and not st[best]["name_en"]:
                st[best]["name_en"] = st_en
            if sid not in anchors:
                anchors[sid].add(v)
                n_listed += 1
            if d > foot_m:
                far[sid] = (plon, plat, d)

        # --- a line with no published list: a station just beyond a dead end of its own track
        # is its terminus. A branch's named track (新北投支線, 小碧潭支線) stops at the junction
        # platform, whose stop node lies on the trunk's track, not the branch's.
        if name not in tra and name not in rlists:
            for v, nb in adj.items():
                if len(nb) != 1 or v < 0:
                    continue
                x, y = xy[v]
                d = np.hypot((pt_lon - x) * np.cos(np.radians(y)) * 111320,
                             (pt_lat - y) * 110570)
                j = int(np.argmin(d))
                sid = f"t{pt_st[j]}"
                if d[j] <= END_M and sid not in anchors:
                    anchors[sid].add(v)
                    n_end += 1

        # --- one station per place (as kr_register)
        pos = {s: tuple(np.mean([xy[a] for a in ans], axis=0)) for s, ans in anchors.items()}
        order = sorted(anchors, key=lambda s: (s not in listed, station_rec(s)["rank"], s))
        merged = {}
        for i, s in enumerate(order):
            if s in merged:
                continue
            for t in order[i + 1:]:
                if t not in merged and dist_m(*pos[s], *pos[t]) <= MERGE_M:
                    merged[t] = s
                    anchors[s] |= anchors.pop(t)
                    if t in far and s not in far:
                        far[s] = far[t]
        if len(anchors) < 2:
            dropped.append(name)
            continue

        # --- footprints, neighbours, sections: kr_register's, except on single-track mountain
        # narrow gauge, where a station claims along its own track, not in a circle: a circle
        # reaches the next loop of a spiral (阿里山線 cut 獨立山's spiral short, 樟腦寮 to
        # 獨立山 in 0.9 km against 4.1).
        foot, foot_d = {}, {}
        if narrow:
            for sid, ans in anchors.items():
                for v, d in along(adj, ans, FOOT_M / 1000).items():
                    if d * 1000 < foot_d.get(v, INF):
                        foot[v], foot_d[v] = sid, d * 1000
        for sid, ans in anchors.items():
            if narrow:
                continue
            disks = [(*xy[a], foot_m) for a in ans]
            if sid in far:
                lon, lat, d = far[sid]
                disks.append((lon, lat, min(d + foot_m, FAR_FOOT_M)))
            for x, y, r in disks:
                ids, ds = near.within(x, y, r)
                for v, d in zip(ids, ds):
                    if d < foot_d.get(v, INF):
                        foot[v], foot_d[v] = sid, d
        for sid, ans in anchors.items():
            for a in ans:
                foot[a], foot_d[a] = sid, 0.0
        pairs = neighbours(adj, foot)
        if not pairs:
            dropped.append(name)
            continue
        foot_of = defaultdict(set)
        for v, s in foot.items():
            foot_of[s].add(v)
        centre = {s: tuple(np.mean([xy[a] for a in ans], axis=0)) for s, ans in anchors.items()}

        def offsets(s):
            c = centre[s]
            return {v: dist_m(*xy[v], *c) / 1000 for v in foot_of[s]}

        sections = {}
        for a, b in sorted(pairs):
            blocked = {v for v, s in foot.items() if s != a and s != b}
            got = between(adj, offsets(a), offsets(b), blocked)
            if got is None:
                continue
            nodes, km = got
            keep = [n for i, n in enumerate(nodes) if n > 0 or i == 0 or i == len(nodes) - 1]
            on_fast = sum(dist_m(*xy[u], *xy[v]) for u, v in zip(nodes[:-1], nodes[1:])
                          if ((u, v) if u < v else (v, u)) in fast)
            sections[(a, b)] = {
                "km": km, "geom": [centre[a]] + [xy[n] for n in keep] + [centre[b]],
                "fast": on_fast / 1000 >= 0.5 * km if km else False}
        # A line listed by its route relations keeps only sections between stops some train
        # calls at one after the other. What else the search finds is track between two
        # stations that no service runs: 中和新蘆線's link between its two branches, 台北橋 to
        # 三重國小, which made the line 31.4 km against its published 29.4.
        if name in rnext:
            for k in [k for k in sections if frozenset(k) not in rnext[name]]:
                v = sections.pop(k)
                cut.append((name, station_rec(k[0])["name"], station_rec(k[1])["name"],
                            round(v["km"], 2)))
        if not sections:
            dropped.append(name)
            continue

        if name in tra:
            seq = [s for s in listed_order if s in anchors or s in merged]
            seq = [merged.get(s, s) for s in seq]
            sadj = defaultdict(set)
            for x, y in sections:
                sadj[x].add(y)
                sadj[y].add(x)

            def joined(a, b):
                # through stations off the list only (木屐寮 between 竹崎 and 樟腦寮)
                seen, todo = {a}, [a]
                while todo:
                    u = todo.pop()
                    for w in sadj[u]:
                        if w == b:
                            return True
                        if w not in seen and w not in cum:
                            seen.add(w)
                            todo.append(w)
                return False
            for a, b in zip(seq[:-1], seq[1:]):
                if a != b and not joined(a, b):
                    missing.append((name, station_rec(a)["name"], station_rec(b)["name"],
                                    round(abs(cum.get(a, 0) - cum.get(b, 0)), 1)))
            odd = [(station_rec(a)["name"], station_rec(b)["name"], round(v["km"], 1))
                   for (a, b), v in sections.items() if a in cum and b in cum
                   and v["km"] > 1.3 * abs(cum[a] - cum[b]) + 0.5]
            for a, b, km in odd:
                log(f"  TW: {name} section {a}-{b} is {km} km, far over TRA's figure")

        chain = {f"{a}|{b}": abs(cum[a] - cum[b]) for (a, b) in sections
                 if a in cum and b in cum}
        official = bool(cum) and len(chain) == len(sections)
        if cum:
            extra[name] = sorted({station_rec(s)["name"] for k in sections for s in k
                                  if s not in listed})

        lid = line_id(name)
        kinds = Counter()
        for wid in wids:
            kinds[TRACK_KIND[ways[wid][0]["railway"]]] += 1
        for sid in {s for k in sections for s in k}:
            if sid not in stations:
                s = station_rec(sid)
                stations[sid] = {"id": sid, "name": s["name"], "name_en": s["name_en"],
                                 "lon": s["lon"], "lat": s["lat"], "lines": set()}
            stations[sid]["lines"].add(lid)
        from n02 import walk_order
        en, op, op_en = LINES[name]
        line = {
            "id": lid, "src": "tw", "service": False,
            "name": name, "name_en": en, "ref": "", "colour": "",
            "operator": op, "operator_en": op_en, "network": "",
            "kind": kinds.most_common(1)[0][0],
            # Per section, from the highspeed=yes ways it lies on (see kr_register).
            "highspeed_sections": {f"{a}|{b}": v["fast"] for (a, b), v in sections.items()},
            "km": round(sum(v["km"] for v in sections.values()), 3),
            "variants": 1, "straight_sections": 0,
            "display": walk_order(sections.keys()),
            "sections": [[a, b, round(v["km"], 3)] for (a, b), v in sections.items()],
        }
        if official:
            official_lines += 1
            line["km_official"] = round(sum(chain.values()), 3)
        lines.append(line)
        geoms[lid] = {f"{a}|{b}": [[round(x, 5), round(y, 5)] for x, y in v["geom"]]
                      for (a, b), v in sections.items()}

    total = sum(l["km"] for l in lines)
    log(f"TW: {len(lines)} register lines, {total:,.0f} km, {len(stations)} stations "
        f"({sum(1 for s in stations if s.startswith('tra'))} made from TRA's own points); "
        f"{official_lines} lines have TRA's km for every section")
    log(f"TW: stations placed by a stop node on the line's track {n_vertex}, "
        f"by TRA's list alone {n_listed}, from route relations {n_route}, beyond a dead end of the track {n_end}; dropped (under two stations): {' '.join(dropped)}")
    for name, names in sorted(extra.items()):
        if names:
            log(f"  TW: {name} also stops at OSM stations its list does not name: "
                f"{' '.join(names)}")
    if cut:
        log(f"TW: {len(cut)} sections dropped as no train's consecutive stops: "
            + "; ".join(f"{l} {a}-{b} {km}" for l, a, b, km in cut))
    if missing:
        log(f"TW: {len(missing)} pairs of TRA-listed neighbours have no section between them:")
        for line, a, b, km in missing:
            log(f"    {line}: {a}-{b} ({km} km in TRA's list)")
    if unmatched:
        log(f"TW: {len(unmatched)} listed stops not placed on their line's track:")
        for line, nm, d in unmatched:
            log(f"    {line}: {nm} " + (f"({d} m off)" if d is not None
                                        else "(no OSM station of that name near it)"))
    return lines, stations, geoms
