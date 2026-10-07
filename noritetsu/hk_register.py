"""Lines, stations and sections for Hong Kong, with OpenStreetMap's named track as the geometry.

    python extract.py --region hk --pbf data/raw/hk-260929.osm.pbf
    python hk_register.py --clip                  # cut the Shenzhen side out of data/proc/hk
    python build_model.py --region hk --register hk_register:data/raw/hk

THE SHAPE IS TAIWAN'S (tw_register.py, and kr_register.py for the traps). Hong Kong publishes no
open line-geometry file, but 99.7% of its rail track in OSM carries its line's name
(`python probe_kr_ways.py --region hk`): 港鐵東鐵綫, 港鐵屯馬綫, 輕鐵, 香港電車. So each register
line is the graph of the OSM ways carrying its name, as in Korea and Taiwan.

THE REGISTER UNIT is the line as MTR names it: the ten heavy-rail lines (東鐵綫, 屯馬綫, 觀塘綫,
荃灣綫, 港島綫, 南港島綫, 東涌綫, 機場快綫, 將軍澳綫, 迪士尼綫), MTR Light Rail as ONE line (輕鐵:
MTR draws and counts it as one 36.2 km network; its twelve numbered routes stay OSM operating
patterns over it), Hong Kong Tramways as one line, and the Peak Tram. Plus the high-speed
line's Hong Kong section, 香港西九龍 to the border, as a piece of China's 广深港高速线 under its
id (BORDER_PIECES, at the end).

WHICH STATIONS ARE ON A LINE comes from MTR's own open data (hk_sources.md): the "MTR lines and
stations" and "Light Rail routes and stops" files on DATA.GOV.HK, which give every line's
stations in order per direction and branch, with Chinese and English names, but no positions
and no distances. A listed station is found among OSM's stations by its Chinese or English
name, nearest the line's own track (as in Korea). The trams and the Peak Tram have no such
file; their stops are the stop members of OSM's own route relations (as Taiwan's metros).
Only listed stations are ever put on a line: 東涌綫 and 機場快綫 share the track OSM names
大嶼山及機場鐵路 along much of their corridor, and a stop node on it must not make 機場快綫 call
at 欣澳.

A SECTION survives only between two stations some list has one after the other (MTR's
sequences, OSM's relations), which keeps 東鐵綫's 上水-落馬洲 branch and drops track no train
runs between two stations. The trams are the exception: each direction's stops are separate,
differently named records (荷蘭街 eastbound is 堅彌地城海旁 westbound, at one kerb), so the two
records of a stop are merged into one station called "荷蘭街 / 堅彌地城海旁", and sections are
whatever the stops' order along the track gives.

OSM NAMES ARE BILINGUAL in one tag here, "羅湖 Lo Wu", "港鐵東鐵綫 MTR East Rail Line". Station
and line names written out are the Chinese half; the English half is `name_en`.

THE SHENZHEN SIDE. No extract is cut to Hong Kong's border (Geofabrik has none; openstreetmap.fr's
hong_kong.osm.pbf runs a few km past it and holds Shenzhen Metro lines 1, 2, 4, 7-10 and 13 and
the Guangshen railway at 深圳站). `--clip` rewrites data/proc/hk to OSM's own boundary of Hong
Kong (relation 913110, data/raw/hk/hk_boundary.geojson): a way stays if at least half its nodes
are inside, a stop if it is inside, a relation if a member stayed. Every HK-named way is wholly
inside; the two short ways over the Lo Wu bridge and the 5.7 km of 广深港高速线 north of the
border are the only ways it cuts, 羅湖 and 落馬洲 stay, 罗湖/深圳/福田口岸 go.

The `path` argument is data/raw/hk; the OSM half is read from data/proc/hk (extract.py).
"""
import csv
import hashlib
import json
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np

from kr_register import INF, Near, between, dist_m, line_graph, neighbours

ROOT = Path(__file__).resolve().parent

MTR = ("香港鐵路有限公司", "MTR Corporation")
# Register line -> (English name, ref, operator, English operator).
LINES = {
    "東鐵綫": ("East Rail Line", "EAL", *MTR),
    "屯馬綫": ("Tuen Ma Line", "TML", *MTR),
    "觀塘綫": ("Kwun Tong Line", "KTL", *MTR),
    "荃灣綫": ("Tsuen Wan Line", "TWL", *MTR),
    "港島綫": ("Island Line", "ISL", *MTR),
    "南港島綫": ("South Island Line", "SIL", *MTR),
    "東涌綫": ("Tung Chung Line", "TCL", *MTR),
    "機場快綫": ("Airport Express", "AEL", *MTR),
    "將軍澳綫": ("Tseung Kwan O Line", "TKL", *MTR),
    "迪士尼綫": ("Disneyland Resort Line", "DRL", *MTR),
    "輕鐵": ("Light Rail", "", *MTR),
    "香港電車": ("Hong Kong Tramways", "", "香港電車有限公司", "Hongkong Tramways Limited"),
    "山頂纜車": ("Peak Tram", "", "山頂纜車有限公司", "The Peak Tramways Company Limited"),
}
CODE = {v[1]: k for k, v in LINES.items() if v[1]}          # EAL -> 東鐵綫

# OSM track name (its Chinese half) -> the register line(s) it is. Everything else named is not
# register track: 廣深港高速鐵路 (built apart, by border_pieces), the airport's airside people mover, Ocean
# Park's 海洋列車, 藍田服務聯絡綫 (a depot link).
TRACK = {
    "港鐵東鐵綫": ("東鐵綫",), "港鐵東鐵綫馬場支綫": ("東鐵綫",),
    "港鐵東鐵綫(落馬洲支綫)": ("東鐵綫",),
    "港鐵屯馬綫": ("屯馬綫",), "港鐵屯馬綫一期": ("屯馬綫",),
    "港鐵觀塘綫": ("觀塘綫",), "港鐵荃灣綫": ("荃灣綫",), "港鐵港島綫": ("港島綫",),
    "港鐵南港島綫": ("南港島綫",), "港鐵東涌綫": ("東涌綫",), "港鐵機場快綫": ("機場快綫",),
    "港鐵將軍澳綫": ("將軍澳綫",), "港鐵迪士尼綫": ("迪士尼綫",),
    # Track 東涌綫 and 機場快綫 share (31 km of ways); both lines' relations run over it.
    "大嶼山及機場鐵路": ("東涌綫", "機場快綫"),
    "輕鐵": ("輕鐵",),
    "香港電車": ("香港電車",), "跑馬地圈": ("香港電車",),
    "山頂纜車": ("山頂纜車",),
}

TRACK_KIND = {"rail": "rail", "subway": "subway", "light_rail": "light_rail",
              "monorail": "monorail", "tram": "tram", "funicular": "funicular",
              "narrow_gauge": "narrow_gauge", "preserved": "rail"}
STATION_RAILWAY = {"station", "halt", "tram_stop"}
RAIL_MODES = ("train", "subway", "light_rail", "monorail", "tram", "funicular")

STOP_TO_STATION_M = 1200   # a stop node belongs to the station of its name this close
DUP_M = 400                # two station records of one name this close are one station
MATCH_M = 800              # a listed station may be this far off its line's track
REL_MATCH_M = 150          # a station's stop node this far off the track still anchors it
FOOT_M = {"tram": 40, "funicular": 40, "light_rail": 60}   # else FOOT_DEFAULT
FOOT_DEFAULT = 150         # a station cuts every track of its line within this of its anchors
FAR_FOOT_M = 600           # the most a station mapped off its track reaches from its node
MERGE_M = 100              # two stations of one line closer than this are one station, except
                           # on the Peak Tram, whose 花園道 and 堅尼地道 are 85 m apart
MERGE_KIND_M = {"funicular": 0}
TRAM_PAIR_M = 120          # a tram stop's two direction records are at most this far apart
STRAY_KM = 1.0             # a piece of a line's track joined to none of the rest, shorter than
                           # this, is left out (see main_track)

LATIN = re.compile(r"[A-Za-z][A-Za-z0-9'’.,&/\-]*")
HIDDEN = re.compile(r"[​-‏‪-‮⁠﻿]")
# MTR's file writes 茘景 and 茘枝角 with U+8318, OSM with the usual U+8354.
FOLD = str.maketrans({"茘": "荔", "線": "綫"})


def line_id(name):
    h = hashlib.blake2b(f"hk|{name}".encode("utf-8"), digest_size=5)
    return "h" + h.hexdigest()


def zh_part(name):
    """The Chinese half of a bilingual OSM name: "山景 (南) Shan King (South)" -> "山景 (南)".
    The whole name if it has no Chinese in it."""
    n = HIDDEN.sub("", unicodedata.normalize("NFKC", name or "")).strip()
    z = LATIN.sub("", n)
    z = re.sub(r"\(\s*\)", "", z)
    z = re.sub(r"\s+", " ", z).strip(" /")
    return z if re.search(r"[㐀-鿿]", z) else n


def zh_key(name):
    """One spelling for a Chinese station name: no whitespace, the U+8318 variant folded."""
    return re.sub(r"\s+", "", zh_part(name)).translate(FOLD)


def en_key(name):
    n = unicodedata.normalize("NFKC", name or "").casefold()
    return re.sub(r"[^a-z0-9]+", "", n)


def track_lines(tags):
    return TRACK.get(zh_key(tags.get("name") or ""), ())


def route_line(tags):
    """The register line an OSM route relation runs on, or None."""
    ref = (tags.get("ref") or "").strip()
    name = unicodedata.normalize("NFKC", tags.get("name") or "")
    if ref in CODE and tags.get("route") in ("subway", "train", "light_rail"):
        return CODE[ref]
    if tags.get("route") == "light_rail" and name.startswith(("輕鐵", "Light Rail")):
        return "輕鐵"
    if tags.get("route") == "tram" and name.startswith("香港電車"):
        return "香港電車"
    if tags.get("route") == "funicular" and name.startswith("山頂纜車"):
        return "山頂纜車"
    return None


# --------------------------------------------------------------------------- the Shenzhen cut

def clip(log=print):
    """Rewrite data/proc/hk to Hong Kong's boundary. See the module docstring."""
    import pickle
    import shapely
    from shapely.geometry import shape
    import build_model as bm
    d = ROOT / "data" / "proc" / "hk"
    ways, rels, stops, cid, cx, cy = bm.load("hk", log)
    hk = shape(json.loads((ROOT / "data" / "raw" / "hk" / "hk_boundary.geojson")
                          .read_text(encoding="utf-8")))
    shapely.prepare(hk)
    inside_ids = cid[shapely.contains_xy(hk, cx / 1e7, cy / 1e7)]
    inside = set(inside_ids.tolist())
    known = set(cid.tolist())
    keep_w, cut = {}, Counter()
    for wid, (tags, nodes) in ways.items():
        ns = [int(n) for n in nodes if int(n) in known]
        if ns and 2 * sum(n in inside for n in ns) >= len(ns):
            keep_w[wid] = (tags, nodes)
        else:
            cut[tags.get("name") or "(unnamed)"] += 1
    keep_s = {k: v for k, v in stops.items()
              if shapely.contains_xy(hk, v[1], v[2])}
    kept = {("w", k) for k in keep_w} | {("n", k) for k in keep_s}
    routes = {k for k, (tags, members) in rels.items()
              if tags.get("type") == "route" and any((t, r) in kept for t, r, _ in members)}
    keep_r = {k: v for k, v in rels.items()
              if k in routes or any(t == "r" and r in routes for t, r, _ in v[1])}
    for name, v in sorted(cut.items(), key=lambda kv: -kv[1]):
        log(f"  cut {v:4d}  {name}")
    log(f"HK clip: kept {len(keep_w)}/{len(ways)} ways, {len(keep_s)}/{len(stops)} stops, "
        f"{len(keep_r)}/{len(rels)} relations")
    for fn, obj in (("ways.pkl", keep_w), ("rels.pkl", keep_r), ("stops.pkl", keep_s)):
        tmp = d / (fn + ".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(obj, f, protocol=4)
        os.replace(tmp, d / fn)


# --------------------------------------------------------------------------- published lists

def load_lists(path, log):
    """{line: {"names": [(zh, en)], "pairs": {frozenset(zh keys)}}} from MTR's two files."""
    p = Path(path)
    out = defaultdict(lambda: {"names": [], "pairs": set()})
    seqs = defaultdict(list)
    with open(p / "mtr_lines_and_stations.csv", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if not (r.get("Line Code") or "").strip():
                continue
            seqs[(CODE[r["Line Code"].strip()], r["Direction"].strip())].append(
                (float(r["Sequence"]), r["Chinese Name"].strip(), r["English Name"].strip()))
    with open(p / "light_rail_routes_and_stops.csv", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if not (r.get("Line Code") or "").strip():
                continue
            seqs[("輕鐵", r["Line Code"].strip() + "/" + r["Direction"].strip())].append(
                (float(r["Sequence"]), r["Chinese Name"].strip(), r["English Name"].strip()))
    for (line, _dir), rows in seqs.items():
        rows.sort()
        o = out[line]
        have = {zh_key(z) for z, _e in o["names"]}
        for _s, z, e in rows:
            if zh_key(z) not in have:
                o["names"].append((z, e))
                have.add(zh_key(z))
        o["pairs"].update(frozenset((zh_key(a[1]), zh_key(b[1])))
                          for a, b in zip(rows[:-1], rows[1:]))
    log(f"HK: MTR lists for {len(out)} lines: "
        + ", ".join(f"{k} {len(v['names'])}" for k, v in out.items()))
    return out


# --------------------------------------------------------------------------- OSM stations

def build_stations(stops, log):
    """OSM's rail stations, one per complex (one Chinese name within DUP_M), and every stop
    node placed on one. Unlike Taiwan, a bracketed sub-name is part of the name: 山景(南) and
    山景(北) are two Light Rail stops 250 m apart."""
    st = {}
    for nid, (tags, lon, lat) in stops.items():
        rail = (tags.get("railway") in STATION_RAILWAY
                or (tags.get("public_transport") == "station"
                    and any(tags.get(m) == "yes" for m in RAIL_MODES)))
        if rail and tags.get("name"):
            st[nid] = {"name": zh_part(tags["name"]), "name_en": tags.get("name:en") or "",
                       "lon": lon, "lat": lat, "tram": tags.get("railway") == "tram_stop",
                       "rank": 0 if tags.get("railway") == "station" else 1}
    by_key = defaultdict(list)
    for nid, s in st.items():
        by_key[zh_key(s["name"])].append(nid)
    alias = {}
    for ids in by_key.values():
        ids.sort(key=lambda n: (st[n]["rank"], n))
        for i, a in enumerate(ids):
            if a in alias:
                continue
            for b in ids[i + 1:]:
                # Tram stops keep their own records: two stops of one name can be a block apart.
                if b in alias or st[a]["tram"] != st[b]["tram"]:
                    continue
                lim = TRAM_PAIR_M if st[a]["tram"] else DUP_M
                if dist_m(st[a]["lon"], st[a]["lat"], st[b]["lon"], st[b]["lat"]) <= lim:
                    alias[b] = a
                    if not st[a]["name_en"]:
                        st[a]["name_en"] = st[b]["name_en"]
    for b in alias:
        del st[b]
    # A station mapped only as stop positions (the Peak Tram's 花園道) still exists.
    by_key = defaultdict(list)
    for nid, s in st.items():
        by_key[zh_key(s["name"])].append(nid)
    made = 0
    for nid, (tags, lon, lat) in sorted(stops.items()):
        rail_stop = (tags.get("railway") == "stop"
                     or (tags.get("public_transport") == "stop_position"
                         and any(tags.get(m) == "yes" for m in RAIL_MODES)))
        if not rail_stop or not tags.get("name") or nid in st:
            continue
        k = zh_key(tags["name"])
        if any(dist_m(lon, lat, st[c]["lon"], st[c]["lat"]) <= STOP_TO_STATION_M
               for c in by_key.get(k, ())):
            continue
        st[nid] = {"name": zh_part(tags["name"]), "name_en": tags.get("name:en") or "",
                   "lon": lon, "lat": lat, "tram": False, "rank": 2}
        by_key[k].append(nid)
        made += 1
    by_key, by_en = defaultdict(list), defaultdict(list)
    for nid, s in st.items():
        by_key[zh_key(s["name"])].append(nid)
        if s["name_en"]:
            by_en[en_key(s["name_en"])].append(nid)
    node_st = {}
    for nid, (tags, lon, lat) in stops.items():
        if nid in st:
            node_st[nid] = nid
        elif nid in alias:
            node_st[nid] = alias[nid]
        elif tags.get("name"):
            best, bd = None, STOP_TO_STATION_M
            for c in by_key.get(zh_key(tags["name"]), ()):
                d = dist_m(lon, lat, st[c]["lon"], st[c]["lat"])
                if d <= bd:
                    best, bd = c, d
            if best is not None:
                node_st[nid] = best
    log(f"HK: {len(st)} OSM rail stations ({len(alias)} records merged into a complex, "
        f"{made} made from stop positions alone), {len(node_st)} stop nodes placed on one")
    return st, node_st, by_key, by_en


def route_stops(rels, stops, node_st):
    """{line: [[station, ...] per relation]} from the stop members of OSM's route relations."""
    out = defaultdict(list)
    for _rid, (tags, members) in rels.items():
        if tags.get("type") != "route":
            continue
        line = route_line(tags)
        if line is None:
            continue
        seq = []
        for ty, ref, _role in members:
            if ty == "n" and ref in node_st and ref in stops:
                if not seq or seq[-1] != node_st[ref]:
                    seq.append(node_st[ref])
        if seq:
            out[line].append(seq)
    return out


def assign_ways(ways, rels, log):
    """{register line: [way id]}: by name, then unnamed ways in the line's own route relations
    (東涌綫's relations run over five unnamed ways near 東涌 and 欣澳)."""
    by_line = defaultdict(set)
    for wid, (tags, nodes) in ways.items():
        if tags.get("railway") not in TRACK_KIND or tags.get("service") == "yard":
            continue
        for ln in track_lines(tags):
            by_line[ln].add(wid)
    extra = Counter()
    for _rid, (tags, members) in rels.items():
        if tags.get("type") != "route":
            continue
        line = route_line(tags)
        if line is None:
            continue
        for ty, ref, _role in members:
            if ty != "w" or ref not in ways:
                continue
            t = ways[ref][0]
            if not t.get("name") and t.get("railway") in TRACK_KIND and ref not in by_line[line]:
                by_line[line].add(ref)
                extra[line] += 1
    log(f"HK: {sum(extra.values())} unnamed ways joined a line through its route relations "
        f"({', '.join(f'{k} {v}' for k, v in extra.most_common())})")
    return {k: sorted(v) for k, v in by_line.items()}


def main_track(adj, xy, name, log):
    """The line's graph without pieces of named track joined to nothing else of it. 機場快綫's
    platform road at 香港 is a 0.13 km siding OSM connects to the line only through unnamed
    track; its stop node anchored 香港 there, and 香港-九龍 found no section."""
    comp, parts = {}, []
    for v in adj:
        if v in comp:
            continue
        comp[v] = len(parts)
        todo, members = [v], [v]
        while todo:
            u = todo.pop()
            for w, _ in adj[u]:
                if w not in comp:
                    comp[w] = len(parts)
                    members.append(w)
                    todo.append(w)
        parts.append(members)
    if len(parts) < 2:
        return adj, xy
    km = [sum(w for u in p for _, w in adj[u]) / 2 for p in parts]
    keep = {i for i, k in enumerate(km) if k >= STRAY_KM or k == max(km)}
    gone = [round(k, 2) for i, k in enumerate(km) if i not in keep]
    if gone:
        log(f"  HK: {name}: {len(gone)} stray pieces of its track left out ({gone} km)")
    adj = {v: nb for v, nb in adj.items() if comp[v] in keep}
    xy = {v: p for v, p in xy.items() if v in adj}
    return adj, xy


# --------------------------------------------------------------------------- build

def build(path, log):
    import build_model as bm
    from n02 import walk_order
    ways, rels, stops, cid, cx, cy = bm.load("hk", log)
    coords = bm.Coords(cid, cx, cy)
    st, node_st, by_key, by_en = build_stations(stops, log)
    lists = load_lists(path, log)
    rstops = route_stops(rels, stops, node_st)
    by_line = assign_ways(ways, rels, log)
    nodes_of = defaultdict(list)                     # complex -> its stop nodes
    for n, s in node_st.items():
        nodes_of[s].append(n)

    stations, lines, geoms = {}, [], {}
    unmatched, cut, dropped, missing = [], [], [], []
    en_of = {}                                      # complex -> English name from MTR's list
    tram_names = {}                                 # merged tram stop -> "A / B"

    for name in LINES:
        wids = by_line.get(name, [])
        if not wids:
            log(f"  HK: no OSM track for {name}")
            continue
        adj, xy, fast = line_graph(wids, ways, coords)
        adj, xy = main_track(adj, xy, name, log)
        if len(xy) < 2:
            continue
        near = Near(xy)
        kinds = Counter(TRACK_KIND[ways[w][0]["railway"]] for w in wids)
        kind = kinds.most_common(1)[0][0]
        foot_m = FOOT_M.get(kind, FOOT_DEFAULT)
        is_tram = name == "香港電車"

        # --- which stations: MTR's list, else OSM's relations
        members = []                                 # complexes, in list order
        if name in lists:
            for zh, en in lists[name]["names"]:
                cands = set(by_key.get(zh_key(zh), ())) | set(by_en.get(en_key(en), ()))
                best, bd = None, MATCH_M
                for c in cands:
                    d = near.nearest(st[c]["lon"], st[c]["lat"])[1]
                    # Nearest a stop node of it, too: a big complex's record can sit well off
                    # one line's platforms.
                    for n in nodes_of.get(c, ()):
                        d = min(d, near.nearest(stops[n][1], stops[n][2])[1])
                    if d <= bd:
                        best, bd = c, d
                if best is None:
                    unmatched.append((name, zh))
                    continue
                members.append(best)
                en_of.setdefault(best, en)
        else:
            for seq in rstops.get(name, ()):
                for s in seq:
                    if s not in members:
                        members.append(s)

        # --- where on this line's track each one is (its anchors)
        anchors, far = defaultdict(set), {}
        for s in members:
            own = [n for n in nodes_of.get(s, ()) if n in adj]
            if own:
                anchors[s].update(own)
                continue
            best, bd = None, REL_MATCH_M
            for n in nodes_of.get(s, ()):
                v, d = near.nearest(stops[n][1], stops[n][2])
                if d <= bd:
                    best, bd = v, d
            if best is None:
                best, bd = near.nearest(st[s]["lon"], st[s]["lat"])
                if bd > MATCH_M:
                    unmatched.append((name, st[s]["name"]))
                    continue
                if bd > foot_m:
                    far[s] = (st[s]["lon"], st[s]["lat"], bd)
            anchors[s].add(best)

        # --- one station per place. For the trams, a stop's two direction records.
        pos = {s: tuple(np.mean([xy[a] for a in ans], axis=0)) for s, ans in anchors.items()}
        order = sorted(anchors, key=lambda s: (st[s]["rank"], s))
        merged = {}
        lim = TRAM_PAIR_M if is_tram else MERGE_KIND_M.get(kind, MERGE_M)
        for i, s in enumerate(order):
            if s in merged:
                continue
            for t in order[i + 1:]:
                if t not in merged and dist_m(*pos[s], *pos[t]) <= lim:
                    if is_tram:
                        # Only records on different tracks are one stop's two directions;
                        # two stops on one track are two stops, however close.
                        if not any(dist_m(*xy[a], *xy[b]) > 3 for a in anchors[s]
                                   for b in anchors[t]):
                            continue
                        prev = tram_names.get(s, (st[s]["name"], st[s]["name_en"]))
                        if zh_key(st[t]["name"]) not in zh_key(prev[0]).split("/"):
                            tram_names[s] = (f"{prev[0]} / {st[t]['name']}",
                                             " / ".join(x for x in (prev[1], st[t]["name_en"])
                                                        if x))
                    merged[t] = s
                    anchors[s] |= anchors.pop(t)
                    if t in far and s not in far:
                        far[s] = far[t]
                    if is_tram:
                        break
        if len(anchors) < 2:
            dropped.append(name)
            continue

        # --- footprints, neighbours, sections: kr_register's
        foot, foot_d = {}, {}
        for sid, ans in anchors.items():
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
        foot_of = defaultdict(set)
        for v, s in foot.items():
            foot_of[s].add(v)
        centre = {s: tuple(np.mean([xy[a] for a in ans], axis=0)) for s, ans in anchors.items()}

        def offsets(s):
            c = centre[s]
            return {v: dist_m(*xy[v], *c) / 1000 for v in foot_of[s]}

        # Consecutive pairs, as station ids, from the list or the relations.
        ok_pairs = None
        if name in lists:
            key_st = defaultdict(set)
            for s in anchors:
                key_st[zh_key(st[s]["name"])].add(s)
            for s, m in merged.items():
                key_st[zh_key(st[s]["name"])].add(m)
            for s in members:
                key_st[zh_key(st[s]["name"])].add(merged.get(s, s))
            # a listed name whose OSM record is spelled otherwise: by English name
            for zh, en in lists[name]["names"]:
                for s in anchors:
                    if en_key(en_of.get(s, "")) == en_key(en):
                        key_st[zh_key(zh)].add(s)
            ok_pairs = set()
            for pr in lists[name]["pairs"]:
                if len(pr) < 2:
                    continue             # a circular route's last stop is its first
                a, b = tuple(pr)
                for x in key_st.get(a, ()):
                    for y in key_st.get(b, ()):
                        if x != y:
                            ok_pairs.add(frozenset((x, y)))
        elif not is_tram:
            ok_pairs = set()
            for seq in rstops.get(name, ()):
                seq = [merged.get(s, s) for s in seq]
                ok_pairs.update(frozenset(p) for p in zip(seq[:-1], seq[1:]) if p[0] != p[1])

        sections = {}
        for a, b in sorted(pairs):
            if ok_pairs is not None and frozenset((a, b)) not in ok_pairs:
                continue
            blocked = {v for v, s in foot.items() if s != a and s != b}
            got = between(adj, offsets(a), offsets(b), blocked)
            if got is None:
                continue
            path_nodes, km = got
            keep = [n for i, n in enumerate(path_nodes)
                    if n > 0 or i == 0 or i == len(path_nodes) - 1]
            on_fast = sum(dist_m(*xy[u], *xy[v]) for u, v in zip(path_nodes[:-1], path_nodes[1:])
                          if ((u, v) if u < v else (v, u)) in fast)
            sections[(f"h{a}", f"h{b}")] = {
                "km": km, "geom": [centre[a]] + [xy[n] for n in keep] + [centre[b]],
                "fast": on_fast / 1000 >= 0.5 * km if km else False}
        for a, b in sorted(pairs):
            if ok_pairs is not None and frozenset((a, b)) not in ok_pairs:
                cut.append((name, st[a]["name"], st[b]["name"]))
        if ok_pairs is not None:
            got_pairs = {frozenset((int(a[1:]), int(b[1:]))) for a, b in sections}
            for pr in sorted(ok_pairs - got_pairs, key=lambda p: sorted(p)):
                a, b = sorted(pr)
                missing.append((name, st[a]["name"], st[b]["name"]))
        if not sections:
            dropped.append(name)
            continue

        lid = line_id(name)
        for sid in {s for k in sections for s in k}:
            n = int(sid[1:])
            if sid not in stations:
                s = st[n]
                nm, ne = tram_names.get(n, (s["name"], en_of.get(n) or s["name_en"]))
                stations[sid] = {"id": sid, "name": nm, "name_en": ne,
                                 "lon": s["lon"], "lat": s["lat"], "lines": set()}
            stations[sid]["lines"].add(lid)
        en, ref, op, op_en = LINES[name]
        lines.append({
            "id": lid, "src": "hk", "service": False,
            "name": name, "name_en": en, "ref": ref, "colour": "",
            "operator": op, "operator_en": op_en, "network": "",
            "kind": kind,
            "highspeed_sections": {f"{a}|{b}": v["fast"] for (a, b), v in sections.items()},
            "km": round(sum(v["km"] for v in sections.values()), 3),
            "variants": 1, "straight_sections": 0,
            "display": walk_order(sections.keys()),
            "sections": [[a, b, round(v["km"], 3)] for (a, b), v in sections.items()],
        })
        geoms[lid] = {f"{a}|{b}": [[round(x, 5), round(y, 5)] for x, y in v["geom"]]
                      for (a, b), v in sections.items()}
        log(f"  HK: {name:<6} {lines[-1]['km']:7.2f} km  {len(sections):3d} sections  "
            f"{len({s for k in sections for s in k}):3d} stations  ({kind})")

    for line, sts, g in border_pieces(ways, st, by_key, coords, log):
        for sid, rec in sts.items():
            stations.setdefault(sid, rec)["lines"].add(line["id"])
        lines.append(line)
        geoms[line["id"]] = g

    total = sum(l["km"] for l in lines)
    log(f"HK: {len(lines)} register lines, {total:,.1f} km, {len(stations)} stations; "
        f"dropped (under two stations): {' '.join(dropped) or 'none'}")
    if cut:
        log(f"HK: {len(cut)} neighbouring pairs no list has one after the other: "
            + "; ".join(f"{l} {a}-{b}" for l, a, b in cut))
    if missing:
        log(f"HK: {len(missing)} listed consecutive pairs got no section:")
        for l, a, b in missing:
            log(f"    {l}: {a}-{b}")
    if unmatched:
        log(f"HK: {len(unmatched)} listed stops not placed on their line's track:")
        for l, nm in unmatched:
            log(f"    {l}: {nm}")
    return lines, stations, geoms


# --------------------------------------------------------------------------- over the border

# The high-speed line's Hong Kong section, 香港西九龍 to the border in the tunnel under the
# Shenzhen River (borders.EXTRA xFutian), built as Hong Kong's PIECE OF MAINLAND CHINA'S
# 广深港高速线, under cn_register's id for it. Trains run West Kowloon - 福田 - 深圳北 - 广州南
# and on into the mainland; there is no other station in Hong Kong and no OSM route relation,
# so on its own it had nothing to make a section to. The app joins lines of one id from every
# country into one line, each country's totals counting its own piece, so a ride 福田 ->
# 香港西九龍 is one ride on one line and credits both. Kept by build_model through
# `served_sections`. (Not an OSM line: none exists in either extract, and the track needs an
# owner in Hong Kong's register the way 26 km of tunnel deserves.)
#   (border point, station here (Chinese name), OSM track name (Chinese part), identity)
def _xrl():
    from cn_register import line_id as cn_line_id
    return (cn_line_id("广深港高速线"), "廣深港高速鐵路",
            "Guangzhou–Shenzhen–Hong Kong Express Rail Link", "", *MTR, "rail")


BORDER_PIECES = [("xFutian", "香港西九龍", "廣深港高速鐵路", _xrl)]
PIECE_STATION_M = 300
PIECE_BORDER_M = 60


def border_pieces(ways, st, by_key, coords, log):
    """[(line, {station id: station}, geoms)] for BORDER_PIECES (sg_register's way)."""
    import borders
    from sg_register import track_piece
    pts = {p["id"]: p for p in borders.load(canonical_only=True)}
    out = []
    for pid, stname, track, ident in BORDER_PIECES:
        p = pts.get(pid)
        cands = by_key.get(zh_key(stname), ())
        if p is None or not cands:
            log(f"  HK: border piece {stname} - {pid}: "
                + ("no border point" if p is None else "no OSM station of that name"))
            continue
        nid = min(cands, key=lambda n: st[n]["rank"])
        s = st[nid]
        wids = [w for w, (t, _n) in ways.items()
                if t.get("railway") in TRACK_KIND and t.get("service") != "yard"
                # the English half's en dashes stay in the key: "廣深港高速鐵路––"
                and zh_key(t.get("name") or "").strip("–-") == track]
        got = track_piece(wids, ways, coords, s["lon"], s["lat"], p["lon"], p["lat"])
        if got is None or got[2] > PIECE_STATION_M or got[3] > PIECE_BORDER_M:
            log(f"  HK: border piece {stname} - {pid}: no track between them "
                + ("" if got is None else f"(station {got[2]:.0f} m, point {got[3]:.0f} m off)"))
            continue
        geom, km, _da, _db = got
        geom = geom + [(p["lon"], p["lat"])]
        lid, name, name_en, ref, op, op_en, kind = ident()
        a, key = f"h{nid}", f"h{nid}|{pid}"
        line = {"id": lid, "src": "hk", "service": False, "name": name, "name_en": name_en,
                "ref": ref, "colour": "", "operator": op, "operator_en": op_en, "network": "",
                "kind": kind, "highspeed_sections": {key: True},
                "km": round(km, 3), "variants": 1, "straight_sections": 0,
                "display": [a, pid], "sections": [[a, pid, round(km, 3)]],
                "served_sections": [key]}
        sts = {a: {"id": a, "name": s["name"], "name_en": s["name_en"], "lon": s["lon"],
                   "lat": s["lat"], "lines": set()},
               pid: {"id": pid, "name": p["name"], "name_en": "", "lon": p["lon"],
                     "lat": p["lat"], "lines": set(), "junction": True}}
        log(f"  HK: {name} ({name_en}), the piece over the border: {s['name']} - {p['name']} "
            f"{km:.3f} km, under the neighbour's id {lid}")
        out.append((line, sts, {key: [[round(x, 5), round(y, 5)] for x, y in geom]}))
    return out


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if "--clip" in sys.argv:
        clip()
    else:
        sys.exit(__doc__)
