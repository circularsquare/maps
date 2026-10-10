"""North Korea: the Korean State Railway's lines, from OpenStreetMap's named track and its
infrastructure relations, through kr_register's recipe (as th_register does).

    python kp_register.py --clip       # after every extract: China's and Russia's track out
    python kp_register.py --report     # the way-to-line assignment and the train coverage
    python build_model.py --region kp --register kp_register:data/raw/kp

Anita, 2026-10-08: "north korea: if it runs passenger service to the best of our knowledge,
we should build it." So the whole network is built, and a line is drawn as running where the
best evidence there is says scheduled passenger trains run on it (RUNNING below, kp_sources.md
"What runs"), greyed (`suspended`) where nothing says so.

THE LINE UNIT is the legal line as OSM names it (평의선, 평부선, 평라선 ...): the Ministry of
Railways' own line names, which the South Korean literature and the Wikipedias use too. OSM
names 75% of North Korea's main-line track for its line (`python probe_kr_ways.py --region
kp`) and has a route=railway relation for every line (infra.pkl, the 46 of relation 13874309
and more). So, per way, in order:
  1. its name, through NAME_LINE;
  2. else the infrastructure relation it is in, through REL_LINE (REL_FIRST before a name);
  3. else the line its neighbours at both ends share (`propagate`).
Lines with passenger trains and no name or relation in OSM are made from the route relations
of those trains (ROUTE_LINE): the unnamed track of the train's own ways.

WHICH STATIONS ARE ON A LINE: every OSM rail station within LIST_M of the line's own track, as
in Thailand (no open per-line list; the en.wikipedia line tables were the check), plus the
station nearest a branch's junction end.

WHAT RUNS (`running`): a section runs where an OSM route=train relation (the numbered trains
compiled from the Korean State Railway's timetable, the international trains) runs over at
least half of it; RUNNING adds lines a source names a train on that OSM has no relation for,
NOT_RUNNING takes lines off. A line is cut where it changes from running to not (`cut_running`),
the greyed part a line of its own, `suspended`.

BORDERS: border points of borders.load() naming "kp", and BORDERS below until they are in
borders.EXTRA, become stations of the line whose track passes within BORDER_M.

The `path` argument is data/raw/kp; the OSM half is read from data/proc/kp (extract.py).
"""
import hashlib
import json
import math
import os
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import kr_register as kr

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
REGION = "kp"
OPERATOR = "조선민주주의인민공화국 철도성"
OPERATOR_EN = "Korean State Railway"

# key -> (name, English name). Names as OSM and the Ministry give them; English in
# McCune-Reischauer as en.wikipedia titles them.
LINES = {
    "pyongui": ("평의선", "P'yŏngŭi Line"),
    "pyongbu": ("평부선", "P'yŏngbu Line"),
    "pyongra": ("평라선", "P'yŏngra Line"),
    "kangwon": ("강원선", "Kangwŏn Line"),
    "manpo": ("만포선", "Manp'o Line"),
    "pukpu": ("북부내륙선", "Pukpu Line"),
    "paektusan": ("백두산청년선", "Paektusan Ch'ŏngnyŏn Line"),
    "hambuk": ("함북선", "Hambuk Line"),
    "musan": ("무산선", "Musan Line"),
    "pyongdok": ("평덕선", "P'yŏngdŏk Line"),
    "ichon": ("청년이천선", "Ch'ŏngnyŏn Ich'ŏn Line"),
    "kumgang": ("금강산청년선", "Kŭmgangsan Ch'ŏngnyŏn Line"),
    "hwanghae": ("황해청년선", "Hwanghae Ch'ŏngnyŏn Line"),
    "pyongbuk": ("평북선", "P'yŏngbuk Line"),
    "unnyul": ("은률선", "Ŭnnyul Line"),
    "pyongnam": ("평남선", "P'yŏngnam Line"),
    "paengmu": ("백무선", "Paengmu Line"),
    "sinhung": ("신흥선", "Sinhŭng Line"),
    "changjin": ("장진선", "Changjin Line"),
    "samjiyon": ("삼지연선", "Samjiyŏn Line"),
    "paechon": ("배천선", "Paech'ŏn Line"),
    "ongjin": ("옹진선", "Ongjin Line"),
    "kanggye": ("강계선", "Kanggye Line"),
    "kanggye_br": ("강계선 지선", "Kanggye Line branch"),
    "changyon": ("장연선", "Changyŏn Line"),
    "tokhyon": ("덕현선", "Tŏkhyŏn Line"),
    "kaechon": ("개천선", "Kaech'ŏn Line"),
    "sohae": ("서해갑문선", "Sŏhae Kammun Line"),
    "pupo": ("부포선", "Pup'o Line"),
    "ryonggang": ("룡강선", "Ryonggang Line"),
    "songrim": ("송림선", "Songrim Line"),
    "ryongsong": ("룡성선", "Ryongsŏng Line"),
    "taean": ("대안선", "Taean Line"),
    "tojiri": ("도지리선", "Tojiri Line"),
    "kumya": ("금야선", "Kŭmya Line"),
    "husan": ("후산선", "Husan Line"),
    "kumgol": ("금골선", "Kŭmgol Line"),
    "hochon": ("허천선", "Hŏch'ŏn Line"),
    "toksong": ("덕성선", "Tŏksŏng Line"),
    "taegon": ("대건선", "Taegŏn Line"),
    "mandok": ("만덕선", "Mandŏk Line"),
    "sechon": ("세천선", "Sech'ŏn Line"),
    "hongui": ("홍의선", "Hongŭi Line"),
    "tumangang": ("두만강선", "Tumangang Line"),
    "munchon": ("문천항선", "Munch'ŏn Port Line"),
    "pinallon": ("비날론선", "Pinallon Line"),
    "unha": ("운하선", "Unha Line"),
    "unsan": ("은산선", "Ŭnsan Line"),
    "anju": ("안주탄광선", "Anju Colliery Line"),
    "kowon": ("고원탄광선", "Kowŏn Colliery Line"),
    "kocham": ("고참탄광선", "Koch'am Colliery Line"),
    "songdowon": ("송도원선", "Songdowŏn Line"),
    "chonnae": ("천내선", "Ch'ŏnnae Line"),
    "chongnam": ("청남선", "Ch'ŏngnam Line"),
    "chiktong": ("직동탄광선", "Chiktong Colliery Line"),
    "posan": ("보산선", "Posan Line"),
    "chongdo": ("정도선", "Chŏngdo Line"),
    "paengma": ("백마선", "Paengma Line"),
    "chonsong": ("천성탄광선", "Ch'ŏnsŏng Colliery Line"),
    "sochang": ("서창선", "Sŏch'ang Line"),
}

# OSM way names -> line key; None leaves the way out of the register.
NAME_LINE = {
    "평의선": "pyongui", "평부선": "pyongbu", "경의선": "pyongbu",
    "평라선": "pyongra", "수성다리": "pyongra",
    "강원선": "kangwon", "만포선": "manpo", "북부내륙선": "pukpu",
    "백두산청년선": "paektusan", "함북선": "hambuk", "무산선": "musan", "평덕선": "pyongdok",
    "청년이천선": "ichon", "금강산청년선": "kumgang", "황해청년선": "hwanghae",
    "평북선": "pyongbuk", "은률선": "unnyul", "평남선": "pyongnam", "백무선": "paengmu",
    "신흥선": "sinhung", "장진선": "changjin", "삼지연선": "samjiyon", "배천선": "paechon",
    "옹진선": "ongjin", "강계선": "kanggye", "강계선 지선": "kanggye_br", "장연선": "changyon",
    "덕현선": "tokhyon", "개천선": "kaechon", "서해갑문선": "sohae", "부포선": "pupo",
    "룡강선": "ryonggang", "송림선": "songrim", "룡성선": "ryongsong", "대안선": "taean",
    "도지리선": "tojiri", "금야선": "kumya", "후산선": "husan",
    "안주탄광선": "anju", "안주탄광성": "anju", "Sohae Line Hwap'ung Branch": "anju",
    "고참탄광선": "kocham", "보산선": "posan", "정도선": "chongdo",
    # 혜산만포청년선 is 북부내륙선's older name; 0.8 km of it at Hyesan
    "혜산만포청년선": "pukpu",
    # not register track: the inter-Korean links south of the last DPRK station (no train since
    # 2008), industrial sidings, the Yangdok hot-spring resort's track, the amusement-park
    # monorails and the Paektu funicular (OSM lines where they have routes)
    "동해북부선": None, "산음인입선": None, "San'um Branch Line": None,
    "Yangdok Springs Track": None, "Yagndok Springs Rail Tunnel": None,
    "Yangdok Springs Rail Bridge": None, "Yandok Springs Track": None,
    "만경대 공중열차": None, "공중렬차": None, "향도봉호": None,
}

# route=railway relations -> line key, for their unnamed ways; None: never register track.
REL_LINE = {
    3839235: "pyongui", 3836129: "pyongbu", 8818554: "pyongbu", 3837029: "pyongra",
    3839842: "kangwon", 7783146: "manpo", 7783324: "pukpu", 6346767: "paektusan",
    6346666: "hambuk", 11528050: "musan", 6359082: "pyongdok", 7780636: "ichon",
    7780582: "kumgang", 7783535: "hwanghae", 5139896: "pyongbuk", 7788048: "unnyul",
    3839265: "pyongnam", 7786081: "paengmu", 7786095: "paengmu", 6897526: "sinhung",
    12249618: "sinhung", 6897608: "changjin", 11528037: "samjiyon", 11524971: "paechon",
    11522241: "ongjin", 13675477: "kanggye", 7063896: "kanggye_br", 11647943: "changyon",
    14166434: "tokhyon", 13675503: "kaechon", 11647790: "sohae", 12310190: "pupo",
    11647844: "ryonggang", 11647990: "songrim", 6894830: "ryongsong", 11647962: "tojiri",
    11648003: "kumya", 11647854: "husan", 6908615: "kumgol", 6908530: "hochon",
    6908462: "toksong", 6894945: "taegon", 6946822: "mandok", 6353073: "sechon",
    6346667: "hongui", 6909109: "tumangang", 8328788: "munchon", 12268995: "pinallon",
    13850303: "unha", 6894947: "unsan", 6894817: "anju", 13288831: "kowon",
    8255366: "kocham", 12479572: "songdowon", 8332275: "kangwon", 12479574: "chonnae",
    6892405: "chongnam", 6894952: "chiktong",
    # "삼지연선 구 노선" (its ways are the new line's too), the inter-Korean 동해북부선, the lines of no track (구 경원선, 남동선, 운봉선, 두언선, 회령탄광선, 토해선)
    6331371: None, 8879753: None, 13675473: None, 6892347: None, 13850291: None,
    6908491: None, 6989299: None, 6887323: None, 7609437: None,
}
# Relations that override a way's name. None so far: "삼지연선 구 노선" (6331371) holds only
# ways the rebuilt standard-gauge 삼지연선 (11528037) also holds, Wiyŏn - Kanggu, so it is no
# old alignment to leave out.
REL_FIRST = {}
REL_PRIORITY = list(LINES)

# Lines with passenger trains whose track OSM leaves unnamed and in no line relation: the
# unnamed ways of these route relations (key -> route relation ids). Names: kp_sources.md.
ROUTE_LINE = {
    # 418/419 신의주 - 석하 - 백마 - 피현 - 량책 - 염주: the 백마선, the double-track line
    # Sinŭiju - Yŏmju west of the P'yŏngŭi Line; OSM has its track, unnamed, in no relation
    "paengma": [14585126, 14585127],
    # 11/12 to 금골: the 금골선's relation leaves out 16 km of the track the train runs on
    "kumgol": [14353764],
    # 335/336 순천 - 신창 - 망일리 - 천성: the 천성탄광선 (Sinch'ang - Ch'ŏnsŏng 9.2 km)
    "chonsong": [14595422, 14595423],
    # 723/724 덕천 - 형봉: the 서창선 (Tŏkch'ŏn - Ch'ŏlgisan) and the 형봉선 (Ch'ŏlgisan -
    # Hyŏngbong 4.2 km), one line here: OSM has no station between to cut them at
    "sochang": [14585696, 14585697],
    # 710/711 고원 - 장동 and 361/362 남포 - 철광: track at the ends of 고원탄광선 and
    # 서해갑문선 the line relations leave out
    "kowon": [14351576, 14584951],
    "sohae": [14356528, 14580517],
}

TRACK_KIND = {"rail": "rail", "narrow_gauge": "rail"}
NOT_PASSENGER = {"industrial", "military", "test", "tourism"}

LIST_M = 250
BORDER_M = 400
JUNCTION_M = 1000

# Crossings with no border point in borders.load() yet: where OSM's track crosses OSM's
# boundary (admin_level 2) in the extract / Overpass (kp_sources.md "Borders").
BORDERS = [
    ("xSinuijuDandong", 124.392329, 40.115133, ["cn", "kp"]),     # way 288979790, the Friendship Bridge
    ("eXKPRUTUMANGANG", 130.641271, 42.415217, ["kp", "ru"]),     # way 42510186 / 1475326893
    ("xManpoJian", 126.273205, 41.154877, ["cn", "kp"]),          # way 839022580
    # Namyang - Tumen (way 199109778 at 129.849287, 42.949015): freight only, and OSM's track
    # to the bridge is named 남양국경선, no line's; not offered (kp_sources.md "Borders").
]
# Crossings passenger trains run over: the section from the last DPRK station to the point is
# kept though it ends at a junction (served_sections), and runs. Manp'o - Ji'an: one passenger
# car on the daily freight, for DPRK citizens and ethnic Koreans from China (en.wikipedia
# Manp'o Line).
BORDER_SERVED = {"xSinuijuDandong", "eXKPRUTUMANGANG", "xManpoJian"}

# What runs: kp_sources.md "What runs". Line keys a source names a scheduled passenger train
# on that OSM has no route relation for (the en.wikipedia line articles' Services sections,
# after Kokubu 2007 and the 2002 timetable); and keys taken off whatever OSM's relations say.
RUNNING = {
    "kumgol",     # 11/12 to Kŭmgol, local 513/516 Kŭmgol - Muhak, 913/914
    "toksong",    # 261/262 Hamhŭng - Samgi, local 866/867 over the whole line
    "hochon",     # local 551/556 Kokku - Tongdae, 925/926 Tanch'ŏn - Honggun
    "sinhung",    # local 880/881 Hamhŭng - Sinhŭng; passenger trains also beyond, unnumbered
    "sechon",     # commuter trains Hoeryŏng - Sech'ŏn via Sinhakp'o
    "tokhyon",    # three commuter pairs Sinŭiju Ch'ŏngnyŏn - Tŏkhyŏn
    "changjin",   # "significant for passenger transport in the area", no numbers
    "chonsong",   # 335/336 (OSM and the P'yŏngra Line's list), though its own article says freight
    "sochang",    # 723/724 Tŏkch'ŏn - Hyŏngbong
}
NOT_RUNNING = set()
RUN_SHARE = 0.5          # a section runs if trains' track lies beside this share of it
RUN_NEAR_M = 60          # ... within this
MIN_GREY_KM = 3.0        # a greyed stretch shorter than this, inside a running line, runs

_S = {}


def line_id(name):
    h = hashlib.blake2b(f"kp|{name}".encode("utf-8"), digest_size=5)
    return "p" + h.hexdigest()


def way_geo(ways, coords):
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


def train_routes(rels):
    return {rid: v for rid, v in rels.items()
            if v[0].get("type") == "route" and v[0].get("route") == "train"}


def assign(ways, coords, infra, rels, log):
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
        km = geo[wid][0]
        n = (t.get("name") or "").strip()
        first = [REL_FIRST[r] for r in inrel[wid] if r in REL_FIRST]
        if first:
            if first[0] is None:
                left["(relation) " + (n or "-")] += km
                continue
            out[wid] = first[0]
            continue
        if n and n in NAME_LINE:
            key = NAME_LINE[n]
            if key is None:
                left[n] += km
                continue
        elif n:
            unknown[n] += km
            continue
        else:
            if t.get("service") not in (None, "crossover"):
                continue
            if any(r in REL_LINE and REL_LINE[r] is None for r in inrel[wid]):
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
    # unnamed track of the trains that have no line of their own (ROUTE_LINE)
    for key, rids in ROUTE_LINE.items():
        n = 0
        for rid in rids:
            for m in rels.get(rid, ({}, ()))[1]:
                w = m[1]
                if m[0] == "w" and w in ways and w not in out and \
                        ways[w][0].get("railway") in TRACK_KIND and \
                        not ways[w][0].get("name") and not ways[w][0].get("service"):
                    out[w] = key
                    n += 1
        log(f"KP: {key}: {n} unnamed ways of its trains' routes")
    km_by = Counter()
    for w, k in out.items():
        if not ways[w][0].get("service"):
            km_by[k] += geo[w][0]
    log("KP: main-line track km per line (both tracks where double): "
        + ", ".join(f"{k} {v:,.0f}" for k, v in km_by.most_common()))
    log("KP: named track left out: " + ", ".join(f"{n} {v:.1f}" for n, v in left.most_common()))
    if unknown:
        log("KP: names in no table, left out: "
            + ", ".join(f"{n} {v:.1f}" for n, v in unknown.most_common()))
    # train-route track that is on no line
    orphan = Counter()
    for rid, (t, mem) in train_routes(rels).items():
        for m in mem:
            if m[0] == "w" and m[1] in ways and m[1] not in out and \
                    not ways[m[1]][0].get("service"):
                orphan[f"{rid} {t.get('name')}"] += geo[m[1]][0]
    for k, v in orphan.most_common():
        if v >= 0.5:
            log(f"    train track on no line: {k}: {v:.1f} km")
    return out, geo


def propagate(ways, named, cand, log):
    """Unnamed track outside every relation takes the line its neighbours at both ends share."""
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
    log(f"KP: {len(got)} unnamed ways outside every relation named from their neighbours; "
        f"{len(cand) - len(got)} left unnamed")
    return got


def load_osm(log):
    import build_model as bm
    ways, rels, stops, cid, cx, cy = bm.load(REGION, log)
    coords = bm.Coords(cid, cx, cy)
    with open(ROOT / "data" / "proc" / REGION / "infra.pkl", "rb") as f:
        infra = pickle.load(f)
    line_of, geo = assign(ways, coords, infra, rels, log)
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
        pts = [p for p in borders.load(canonical_only=True) if "kp" in p["countries"]]
    except Exception as e:                          # noqa: BLE001
        log(f"KP: borders.load failed ({e})")
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


URBAN_STATION = ("subway", "light_rail", "monorail", "funicular", "tram")


def urban_only(tags):
    """A station or stop of the metro, the trams, the monorails or the funicular, never of the
    Korean State Railway."""
    if tags.get("railway") == "tram_stop" or tags.get("station") in URBAN_STATION:
        return True
    if tags.get("train") == "yes":
        return False
    return any(tags.get(m) == "yes" for m in URBAN_STATION)


# Unnamed OSM station nodes a train relation stops at, named from en.wikipedia's line tables:
# 723/724's two stops past 덕천 are 서덕천 (3.7 km on, the table's 3.6) and the terminus 형봉.
NAME_STOPS = {1917275704: ("서덕천", "West Tŏkch'ŏn"), 1917308189: ("형봉", "Hyŏngbong")}

# A line whose track leaves another's at a junction with no station within JUNCTION_M: a
# section on from its nearest station to this station of the other line, over any track.
# 천성탄광선 leaves the P'yŏngra Line 2.3 km east of 수양 and 2.4 km west of 신창 (its legal
# start); 335/336 come from Sunch'ŏn, through 수양, so the trains' way in is from 수양.
# 덕현선 starts at 남신의주 (en.wikipedia), 6.1 km before OSM's first track named for it at
# 정문리; 청년이천선 ends at 세포청년 on the Kangwŏn Line, 5.9 km past 신생.
EXTEND = {"천성탄광선": "수양", "덕현선": "남신의주", "청년이천선": "세포청년"}


def extend_lines(lines, stations, geoms, log):
    from n02 import walk_order
    net = Net(_S["ways"], _S["coords"])
    for l in lines:
        target = EXTEND.get(l["name"])
        if not target:
            continue
        cand = [sid for sid, s in stations.items() if s["name"] == target]
        on = {s for sec in l["sections"] for s in sec[:2]}
        best = None
        for t in cand:
            st_t = stations[t]
            for a in on:
                d = kr.dist_m(stations[a]["lon"], stations[a]["lat"], st_t["lon"], st_t["lat"])
                if best is None or d < best[0]:
                    best = (d, a, t)
        if best is None:
            log(f"KP: extend {l['name']}: no station {target}")
            continue
        _d, a, t = best
        sa, sb = stations[a], stations[t]
        got = net.path(net.node_near(sa["lon"], sa["lat"]), net.node_near(sb["lon"], sb["lat"]),
                       3 * _d / 1000 + 2)
        if got is None:
            log(f"KP: extend {l['name']}: no track {sa['name']} - {target}")
            continue
        nodes, km = got
        l["sections"].append([a, t, round(km, 3)])
        geoms[l["id"]][f"{a}|{t}"] = (
            [[round(sa["lon"], 5), round(sa["lat"], 5)]]
            + [[round(net.xy[n][0], 5), round(net.xy[n][1], 5)] for n in nodes]
            + [[round(sb["lon"], 5), round(sb["lat"], 5)]])
        if isinstance(l.get("highspeed_sections"), dict):
            l["highspeed_sections"][f"{a}|{t}"] = False
        stations[t]["lines"].add(l["id"])
        l["km"] = round(sum(s[2] for s in l["sections"]), 3)
        l["display"] = walk_order([(x, y) for x, y, *_ in l["sections"]])
        log(f"KP: {l['name']} extended {sa['name']} - {target}, {km:.1f} km")


def build_stations(stops, log):
    for n, (nm, en) in NAME_STOPS.items():
        if n in stops and not stops[n][0].get("name"):
            t = dict(stops[n][0], name=nm)
            t.setdefault("name:en", en)
            stops[n] = (t, stops[n][1], stops[n][2])
    on_urban = set()
    for t, nodes in _S["ways"].values():
        if t.get("railway") in ("subway", "monorail", "light_rail", "funicular", "tram"):
            on_urban.update(np.asarray(nodes).tolist())
    drop = {n for n, (t, _x, _y) in stops.items()
            if urban_only(t) or (n in on_urban and t.get("railway") != "station")}
    stops = {n: v for n, v in stops.items() if n not in drop}
    log(f"KP: {len(drop)} station and stop records of the metro, trams and others left out")
    st, node_st, by_key, by_base = _orig["build_stations"](stops, log)
    border = {}
    for i, p in enumerate(border_points(log)):
        fid = BORDER_BASE - i
        st[fid] = {"name": p["id"], "name_en": "", "lon": p["lon"], "lat": p["lat"], "rank": 0}
        by_key[kr.name_key(p["id"])].append(fid)
        by_base[kr.base_key(p["id"])].append(fid)
        border[fid] = p["id"]
    _S.update(st=st, node_st=node_st, border=border)
    log(f"KP: {len(border)} border points offered to the lines ({', '.join(border.values())})")
    return st, node_st, by_key, by_base


def load_lists(path, log):
    """{line: [[station name, ...]]}: every rail station within LIST_M of the line's track,
    the station nearest a junction end, and the border points."""
    from scipy.spatial import cKDTree
    ways, coords, st = _S["ways"], _S["coords"], _S["st"]
    by_line = defaultdict(list)
    for w, k in _S["line_of"].items():
        by_line[LINES[k][0]].append(w)
    trees = {}
    kx = 111.32 * math.cos(math.radians(40.0))
    for ln, ws in by_line.items():
        pos = []
        for w in ws:
            p, ok = coords.many(np.asarray(ways[w][1], dtype=np.int64))
            pos.append(p[ok])
        pos = np.concatenate(pos)
        x, y = coords.x[pos] / 1e7, coords.y[pos] / 1e7
        trees[ln] = cKDTree(np.c_[x * kx, y * 110.57])
    lists = defaultdict(set)
    for sid, s in st.items():
        if sid in _S["border"]:
            continue
        for ln, tr in trees.items():
            d, _i = tr.query((s["lon"] * kx, s["lat"] * 110.57))
            if d * 1000 <= LIST_M:
                lists[ln].add(s["name"])
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
    log(f"KP: {n_junc} junction stations listed on the line whose track ends near them")
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
            log(f"KP: border point {pid} on {best[1]} ({best[0]:.0f} m)")
        else:
            log(f"KP: border point {pid} is on no line's track")
    log(f"KP: station lists by proximity ({LIST_M} m): "
        f"{sum(len(v) for v in lists.values())} station-line pairs on {len(lists)} lines")
    return ({k: [sorted(v)] for k, v in lists.items()}, defaultdict(list), {})


def seg_dist(nodes, coords, lon, lat):
    pos, ok = coords.many(np.asarray(nodes, dtype=np.int64))
    pos = pos[ok]
    if pos.size < 2:
        return 1e9
    x, y = coords.x[pos] / 1e7, coords.y[pos] / 1e7
    k = math.cos(math.radians(lat)) * 111320
    ax, ay = (x[:-1] - lon) * k, (y[:-1] - lat) * 110570
    bx, by = (x[1:] - lon) * k, (y[1:] - lat) * 110570
    dx, dy = bx - ax, by - ay
    L2 = np.maximum(dx * dx + dy * dy, 1e-9)
    t = np.clip(-(ax * dx + ay * dy) / L2, 0, 1)
    return float(np.min(np.hypot(ax + t * dx, ay + t * dy)))


GAP_KM = 12.0
GAP_DETOUR = 1.5
GAP_EXTRA_KM = 1.0


class Net:
    """Every rail way not left out, as one graph, for joining a line's pieces through a
    junction's yard OSM leaves unnamed or names for the other line."""

    def __init__(self, ways, coords):
        self.adj = defaultdict(list)
        self.xy = {}
        for wid, (t, nodes) in ways.items():
            if t.get("railway") not in TRACK_KIND or t.get("usage") in NOT_PASSENGER:
                continue
            n = (t.get("name") or "").strip()
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
    nearest first (th_register's join_pieces)."""
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
            log(f"KP: {l['name']} is still in {left} pieces")
    log(f"KP: {len(done)} gaps in a line joined over other track "
        f"({sum(d[3] for d in done):.1f} km)")
    for nm, a, b, km in done:
        log(f"    {nm}: {a} - {b} {km:.1f} km")


# ---------------------------------------------------------------- what runs

def train_tree():
    """A KD-tree of points every ~25 m along the ways of every OSM route=train relation."""
    from scipy.spatial import cKDTree
    ways, coords = _S["ways"], _S["coords"]
    wids = {m[1] for _t, mem in train_routes(_S["rels"]).values() for m in mem
            if m[0] == "w" and m[1] in ways}
    pts = []
    kx = 111.32 * math.cos(math.radians(40.0))
    for w in wids:
        pos, ok = coords.many(np.asarray(ways[w][1], dtype=np.int64))
        pos = pos[ok]
        x, y = coords.x[pos] / 1e7, coords.y[pos] / 1e7
        for i in range(len(x) - 1):
            n = max(1, int(kr.dist_m(x[i], y[i], x[i + 1], y[i + 1]) // 25))
            t = np.arange(n) / n
            pts.append(np.c_[(x[i] + (x[i + 1] - x[i]) * t) * kx,
                             (y[i] + (y[i + 1] - y[i]) * t) * 110.57])
        if len(x):
            pts.append(np.array([[x[-1] * kx, y[-1] * 110.57]]))
    return cKDTree(np.concatenate(pts)), kx


def section_share(tree, kx, pts):
    """The share of a section's length (its drawn points, resampled every ~50 m) lying within
    RUN_NEAR_M of a train's track."""
    a = np.asarray(pts, dtype=np.float64)
    if len(a) < 2:
        return 0.0
    seg = np.hypot(np.diff(a[:, 0]) * kx, np.diff(a[:, 1]) * 110.57)
    L = seg.sum()
    if L <= 0:
        return 0.0
    n = max(4, int(L / 0.05))
    cum = np.r_[0, np.cumsum(seg)]
    s = (np.arange(n) + 0.5) / n * L
    x = np.interp(s, cum, a[:, 0]) * kx
    y = np.interp(s, cum, a[:, 1]) * 110.57
    d, _i = tree.query(np.c_[x, y])
    return float(np.mean(d * 1000 <= RUN_NEAR_M))


def cut_running(lines, stations, geoms, log):
    """Mark each line running or not, and cut a line where it changes: the greyed sections a
    line of their own (`suspended`), named for the line and its two ends."""
    from n02 import walk_order
    tree, kx = train_tree()
    key_of = {v[0]: k for k, v in LINES.items()}
    out = []
    report = []
    for l in lines:
        key = key_of.get(l["name"])
        share = {f"{a}|{b}": section_share(tree, kx, geoms[l["id"]][f"{a}|{b}"])
                 for a, b, *_ in l["sections"]}
        if key in NOT_RUNNING:
            runs = {k: False for k in share}
        elif key in RUNNING:
            runs = {k: True for k in share}
        else:
            served = _S.get("border_served", ())
            runs = {k: v >= RUN_SHARE or any(x in served for x in k.split("|"))
                    for k, v in share.items()}
            # a short unrun stretch inside a running line runs (a junction's yard, a station
            # throat OSM's train relation skips)
            km = {f"{a}|{b}": k for a, b, k, *_ in l["sections"]}
            grey = [k for k, r in runs.items() if not r]
            if grey and any(runs.values()):
                for comp in components(grey):
                    if sum(km[k] for k in comp) < MIN_GREY_KM and touches_running(comp, runs):
                        for k in comp:
                            runs[k] = True
        run_km = sum(s[2] for s in l["sections"] if runs[f"{s[0]}|{s[1]}"])
        report.append((l["name"], l["km"], run_km))
        if all(runs.values()):
            out.append(l)
            continue
        if not any(runs.values()):
            l["suspended"] = True
            out.append(l)
            continue
        # cut: the running sections stay the line (its id), the rest a greyed line per piece
        keep = [s for s in l["sections"] if runs[f"{s[0]}|{s[1]}"]]
        rest = [s for s in l["sections"] if not runs[f"{s[0]}|{s[1]}"]]
        l["sections"] = keep
        l["km"] = round(sum(s[2] for s in keep), 3)
        l["display"] = walk_order([(a, b) for a, b, *_ in keep])
        g = geoms[l["id"]]
        out.append(l)
        for comp in components([f"{a}|{b}" for a, b, *_ in rest]):
            secs = [s for s in rest if f"{s[0]}|{s[1]}" in comp]
            order = walk_order([(a, b) for a, b, *_ in secs])
            ends = [stations[order[0]]["name"], stations[order[-1]]["name"]]
            nm = f"{l['name']} ({ends[0]} - {ends[1]})"
            en = l.get("name_en") or LINES.get(key, ("", ""))[1]
            nid = line_id(nm)
            nl = dict(l, id=nid, name=nm, sections=secs, suspended=True,
                      km=round(sum(s[2] for s in secs), 3), display=order,
                      name_en=f"{en} ({names_en(stations, order[0])} - "
                              f"{names_en(stations, order[-1])})" if en else "")
            nl.pop("km_official", None)
            if isinstance(l.get("highspeed_sections"), dict):
                nl["highspeed_sections"] = {k: False for k in comp}
                l["highspeed_sections"] = {k: v for k, v in l["highspeed_sections"].items()
                                           if k not in comp}
            geoms[nid] = {k: g.pop(k) for k in comp}
            out.append(nl)
            log(f"KP: {l['name']}: {nm} greyed, {nl['km']:.1f} km")
    lines[:] = out
    for s in stations.values():
        s["lines"] = set()
    for l in lines:
        for a, b, *_ in l["sections"]:
            stations[a]["lines"].add(l["id"])
            stations[b]["lines"].add(l["id"])
    log("KP: train coverage per line (built km, km with trains):")
    for nm, km, run in sorted(report, key=lambda r: -r[1]):
        log(f"    {km:7.1f} {run:7.1f}  {nm}")


def names_en(stations, sid):
    s = stations[sid]
    return s.get("name_en") or s["name"]


def components(keys):
    """Groups of "a|b" section keys joined by a shared station."""
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for k in keys:
        a, b = k.split("|")
        parent[find(a)] = find(b)
    out = defaultdict(list)
    for k in keys:
        out[find(k.split("|")[0])].append(k)
    return list(out.values())


def touches_running(comp, runs):
    st = {s for k in comp for s in k.split("|")}
    return any(r and (set(k.split("|")) & st) for k, r in runs.items())


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
    extend_lines(lines, stations, geoms, log)
    gb.drop_shortcuts(lines, geoms, log)
    border = {f"k{fid}": pid for fid, pid in _S.get("border", {}).items()}
    en_of = {v[0]: v[1] for v in LINES.values()}

    def r(sid):
        return border.get(sid) or ("p" + sid[1:] if sid.startswith("k") else sid)

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
        l["src"] = "kp"
        l["operator"], l["operator_en"] = OPERATOR, OPERATOR_EN
        l["name_en"] = en_of.get(l["name"], "")
        l["sections"] = [[r(a), r(b), *rest] for a, b, *rest in l["sections"]]
        l["display"] = [r(x) for x in l["display"]]
        if isinstance(l.get("highspeed_sections"), dict):
            l["highspeed_sections"] = {"|".join(r(x) for x in k.split("|")): v
                                       for k, v in l["highspeed_sections"].items()}
        out_geoms[l["id"]] = {"|".join(r(x) for x in k.split("|")): v
                              for k, v in geoms[l["id"]].items()}
    cut_running(lines, out_st, out_geoms, log)
    for l in lines:
        served = [f"{a}|{b}" for a, b, *_ in l["sections"]
                  if not l.get("suspended") and (a in _S.get("border_served", ())
                                                 or b in _S.get("border_served", ()))]
        if served:
            l["served_sections"] = served
    out_st = {k: s for k, s in out_st.items() if s["lines"]}
    run = [l for l in lines if not l.get("suspended")]
    log(f"KP: {len(lines)} register lines, {sum(l['km'] for l in lines):,.0f} km, "
        f"{len(out_st)} stations; running {len(run)} lines {sum(l['km'] for l in run):,.0f} km")
    for l in sorted(lines, key=lambda l: -l["km"]):
        log(f"    {l['km']:8.1f} km  {len(l['sections']):3d} sections  "
            f"{'grey' if l.get('suspended') else 'runs'}  {l['name']}")
    return lines, out_st, out_geoms


# ---------------------------------------------------------------- --clip

# "Abroad" is every other country's outline shrunk by this (degrees, ~1.1 km), as in
# lk_register: the outlines are simplified, and Sinuiju and Manp'o stand on the Yalu's bank.
CLIP_INSET_DEG = 0.01


# Track named for a DPRK line is never cut, whatever the outline says: 삼지연선 runs along the
# Yalu's bank above Hyesan, where the simplified outline of China reaches over it. Except
# 경의선, which is also the name of South Korea's line beyond the DMZ.
NEVER_CUT = {n for n, k in NAME_LINE.items() if k} - {"경의선"}


def hangul_only(name):
    """A name in Chosŏn'gŭl with no Chinese or Cyrillic in it: a DPRK station's. China's
    stations near the border are named in Chinese (some with Korean beside it)."""
    import re
    return bool(re.search(r"[가-힣]", name)) and \
        not re.search(r"[一-鿿Ѐ-ӿ]", name)


def clip(log=print):
    """Rewrite data/proc/kp without what lies in China, Russia or South Korea (Geofabrik's cut
    holds Dandong, Tumen, Hunchun and Ji'an's track): nafrica_register.clip's rule (a way goes
    if half its nodes are abroad) with the outlines shrunk, and NEVER_CUT."""
    import shapely
    import nafrica_register as nr
    d = ROOT / "data" / "proc" / REGION
    rd = lambda f: pickle.load(open(d / f, "rb"))  # noqa: E731
    ways, rels, stops, infra = rd("ways.pkl"), rd("rels.pkl"), rd("stops.pkl"), rd("infra.pkl")
    c = np.load(d / "coords.npz")
    cid, cx, cy = c["id"], c["x"], c["y"]
    g = nr.abroad(REGION).buffer(-CLIP_INSET_DEG)
    shapely.prepare(g)
    away = shapely.contains_xy(g, cx / 1e7, cy / 1e7)
    outside = set(cid[away].tolist())
    known = set(cid.tolist())
    keep_w, cut, spared = {}, Counter(), Counter()
    for wid, (tags, nodes) in ways.items():
        ns = [int(n) for n in nodes if int(n) in known]
        if ns and 2 * sum(n in outside for n in ns) >= len(ns):
            if (tags.get("name") or "").strip() in NEVER_CUT:
                spared[tags["name"]] += 1
                keep_w[wid] = (tags, nodes)
            else:
                cut[tags.get("name") or tags.get("railway") or "?"] += 1
        else:
            keep_w[wid] = (tags, nodes)
    keep_s = {}
    for k, v in stops.items():
        if not shapely.contains_xy(g, v[1], v[2]):
            keep_s[k] = v
        elif hangul_only(v[0].get("name") or ""):
            keep_s[k] = v
            log(f"  stop kept though abroad by the outline: {v[0].get('name')} "
                f"{v[1]:.4f},{v[2]:.4f}")
        else:
            log(f"  stop cut: {v[0].get('name')} {v[1]:.4f},{v[2]:.4f}")
    kept = {("w", k) for k in keep_w} | {("n", k) for k in keep_s}
    routes = {k for k, (tags, members) in rels.items()
              if tags.get("type") == "route" and any((t, r) in kept for t, r, _ in members)}
    keep_r = {k: v for k, v in rels.items()
              if k in routes or any(t == "r" and r in routes for t, r, _ in v[1])}
    keep_i = {k: v for k, v in infra.items()
              if any(t == "w" and r in keep_w for t, r, _ in v[1])}
    for name, v in cut.most_common(30):
        log(f"  cut {v:4d}  {name}")
    for name, v in spared.most_common():
        log(f"  kept though abroad by the outline: {v} ways of {name}")
    log(f"KP clip: kept {len(keep_w)}/{len(ways)} ways, {len(keep_s)}/{len(stops)} stops, "
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
