"""Lines, stations and sections for Singapore, with OpenStreetMap's named track as the geometry.

    python extract.py --region sg --pbf data/raw/sg/malaysia-singapore-brunei-260929.osm.pbf \
        --bbox 103.6,1.2,104.05,1.452
    python sg_register.py --clip          # drop the Johor track the box could not keep out
    python build_model.py --region sg --register sg_register:data/raw/sg

THE SHAPE IS TAIWAN'S METROS' (tw_register.py, which takes kr_register.py's graph helpers).
Singapore publishes no open line geometry, and 99.4% of its metro track in OSM carries its
line's name (`python probe_kr_ways.py --region sg`): "North South Line (NS)", "East West Line",
"Circle Line Extension (CE)", "Thomson–East Coast Line", "Bukit Panjang LRT", "Sentosa Express".
The exception is the Sengkang and Punggol LRTs, whose track is unnamed; for those two the
member ways of OSM's own route relations for the line are its track (ROUTE_TRACK).

THE REGISTER UNIT is the line as LTA names it on its 2026 system map: North-South, East-West
(with the Changi Airport branch, CG), North East, Circle (with the former Circle Line
Extension, now CC33-CC34, and Stage 6 to Prince Edward Road, opened 2026-07-12), Downtown,
Thomson-East Coast, and the Bukit Panjang, Sengkang and Punggol LRTs; plus the Sentosa Express
monorail, which is Sentosa Development Corporation's and not on LTA's list. The Changi Airport
Skytrain (airport people mover) is left out, as are the Jurong Region Line (under
construction; OSM already tags its stations railway=station), the RTS Link (not open), and
KTM's Shuttle Tebrau to Johor Bahru, which has one station and 1.1 km of track in Singapore
and no OSM route relation.

WHICH STATIONS ARE ON A LINE comes from LTA's own list, the DataMall "Train Station Codes and
Chinese Names" file (sg_sources.md), whose station codes (NS1, EW24, CG2, STC, PW7) give each
line's stations in order. That file is stale: it stops at TE22 and lacks Hume (DT4), Punggol
Coast (NE18), Teck Lee (PW2), TEL Stage 4 (TE23-TE29) and Circle Line Stage 6. Those are
ADDED below from LTA's 2026 system map, which also renumbers the Circle Line Extension
(CE1 Bayfront -> CC34, CE2 Marina Bay -> CC33).

Each station is placed where OSM's route relations for the line have their stop node on the
line's track (they are complete here: every open station of every line, and only those); a
listed station no relation places is matched by name to the OSM station nearest the track.
A section is kept only between two stations some relation calls at one after the other, which
drops turnback and depot links. The log compares the relations' stations against the list,
and each pair of stations adjacent in code order (plus the branch and loop joins in JOINS)
against the sections found.

There is no per-section chainage to be had (LTA publishes line lengths only, to the km), so
the check is `check_model.REGISTER["sg"]`, line by line.

The `path` argument is data/raw/sg; the OSM half is read from data/proc/sg (extract.py).
"""
import argparse
import hashlib
import os
import pickle
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np

from kr_register import INF, Near, between, dist_m, line_graph, neighbours

ROOT = Path(__file__).resolve().parent

# Register line -> (code, operator, English operator, relation name pattern).
SMRT = ("SMRT Trains", "SMRT Trains")
SBST = ("SBS Transit", "SBS Transit")
LINES = {
    "North-South Line": ("NSL", *SMRT, r"^MRT North.South Line"),
    "East-West Line": ("EWL", *SMRT, r"^MRT East.West Line"),
    "North East Line": ("NEL", *SBST, r"^MRT North.East Line"),
    "Circle Line": ("CCL", *SMRT, r"^MRT Circle Line"),
    "Downtown Line": ("DTL", *SBST, r"^MRT Downtown Line"),
    "Thomson-East Coast Line": ("TEL", *SMRT, r"^MRT Thomson.East Coast Line"),
    "Bukit Panjang LRT": ("BPLRT", *SMRT, r"^LRT Bukit Panjang Line"),
    "Sengkang LRT": ("SKLRT", *SBST, r"^LRT Sengkang Line"),
    "Punggol LRT": ("PGLRT", *SBST, r"^LRT Punggol Line"),
    "Sentosa Express": ("", "Sentosa Development Corporation",
                        "Sentosa Development Corporation", r"^Sentosa Express$"),
}

# OSM track name -> register line. Everything else is not register track: the Skytrain's S1
# and S2, KTM's Woodlands spur, and Bukit Panjang's depot tracks.
TRACK = {
    "North South Line (NS)": "North-South Line", "North South Line": "North-South Line",
    "East West Line": "East-West Line", "East-West Line": "East-West Line",
    "East West Line (EW)": "East-West Line",
    "East West Line Changi Airport Extension": "East-West Line",
    "North East Line": "North East Line",
    "Circle Line": "Circle Line", "Circle Line (CC)": "Circle Line",
    "Circle Line Extension (CE)": "Circle Line",
    # Beside Prince Edward Road; trains turn there, and sections are kept only between stops
    # some train calls at in turn, so it adds no section of its own.
    "CCL Turnback Tunnel": "Circle Line",
    "Downtown Line": "Downtown Line", "Downtown Line (DT)": "Downtown Line",
    "Thomson–East Coast Line": "Thomson-East Coast Line",
    "Thomson-East Coast Line": "Thomson-East Coast Line",
    "Bukit Panjang LRT": "Bukit Panjang LRT",
    "Sentosa Express": "Sentosa Express",
}
# Lines whose track is unnamed in OSM: their track is the member ways of their route relations.
ROUTE_TRACK = {"Sengkang LRT", "Punggol LRT"}
NOT_TRACK_SERVICE = {"yard"}           # depots; crossovers and sidings are harmless

# LTA's list names these lines; the register line each belongs to.
LTA_LINE = {
    "North-South Line": "North-South Line", "East-West Line": "East-West Line",
    "Changi Airport Branch Line": "East-West Line", "North East Line": "North East Line",
    "Circle Line": "Circle Line", "Circle Line Extension": "Circle Line",
    "Downtown Line": "Downtown Line", "Thomson-East Coast Line": "Thomson-East Coast Line",
    "Bukit Panjang LRT": "Bukit Panjang LRT", "Sengkang LRT": "Sengkang LRT",
    "Punggol LRT": "Punggol LRT",
}
# Codes LTA's 2026 system map (SM-26-01-EN) changed from the DataMall file.
RECODE = {"CE1": "CC34", "CE2": "CC33"}
# Stations LTA's DataMall file does not have, from the 2026 system map, with their opening.
ADDED = [
    ("DT4", "Hume", "Downtown Line", "2025-02-28"),
    ("NE18", "Punggol Coast", "North East Line", "2024-12-10"),
    ("PW2", "Teck Lee", "Punggol LRT", "2024-08-15"),
    ("TE23", "Tanjong Rhu", "Thomson-East Coast Line", "2024-06-23"),
    ("TE24", "Katong Park", "Thomson-East Coast Line", "2024-06-23"),
    ("TE25", "Tanjong Katong", "Thomson-East Coast Line", "2024-06-23"),
    ("TE26", "Marine Parade", "Thomson-East Coast Line", "2024-06-23"),
    ("TE27", "Marine Terrace", "Thomson-East Coast Line", "2024-06-23"),
    ("TE28", "Siglap", "Thomson-East Coast Line", "2024-06-23"),
    ("TE29", "Bayshore", "Thomson-East Coast Line", "2024-06-23"),
    ("CC30", "Keppel", "Circle Line", "2026-07-12"),
    ("CC31", "Cantonment", "Circle Line", "2026-07-12"),
    ("CC32", "Prince Edward Road", "Circle Line", "2026-07-12"),
    # Not LTA's: Sentosa Express's four stations in order (sentosa.com.sg), no codes.
    ("S1", "VivoCity", "Sentosa Express", ""),
    ("S2", "Resorts World", "Sentosa Express", ""),
    ("S3", "Imbiah", "Sentosa Express", ""),
    ("S4", "Beach", "Sentosa Express", ""),
]
# Pairs of adjacent stations the code order alone does not give: where a branch leaves (Tanah
# Merah, EW4, is also the Changi branch's first station; Promenade, CC4, meets Bayfront, CC34)
# and where a loop closes (Bukit Panjang's at BP6; each LRT loop at its town-centre station).
JOINS = [("EW4", "CG1"), ("CC4", "CC34"), ("BP13", "BP6"),
         ("STC", "SE1"), ("SE5", "STC"), ("STC", "SW1"), ("SW8", "STC"),
         ("PTC", "PE1"), ("PE7", "PTC"), ("PTC", "PW1"), ("PW7", "PTC")]

# Singapore, as a polygon a way must touch to be kept (`--clip`). The extract's box also holds
# Johor's Pasir Gudang branch and the Tanjung Pelepas line; this follows the Johor Strait
# between them, and the causeway's middle (1.4545) north of Woodlands.
SG_POLY = [(103.60, 1.20), (104.10, 1.20), (104.10, 1.40), (103.95, 1.40), (103.95, 1.43),
           (103.845, 1.43), (103.845, 1.4545), (103.70, 1.4545), (103.70, 1.40),
           (103.60, 1.40)]

TRACK_KIND = {"rail": "rail", "subway": "subway", "light_rail": "light_rail",
              "monorail": "monorail", "tram": "tram", "funicular": "funicular",
              "narrow_gauge": "narrow_gauge", "preserved": "rail"}
STATION_RAILWAY = {"station", "halt", "tram_stop"}
RAIL_MODES = ("train", "subway", "light_rail", "monorail", "tram", "funicular")

STOP_TO_STATION_M = 1200   # a stop_position belongs to the station of its name this close
DUP_M = 500                # two station records of one name this close are one station
MATCH_M = 800              # a listed station may be this far off its line's track
REL_MATCH_M = 150          # a route relation's stop node this far (it is normally on the track)
FOOT_M = 150               # a station cuts every track of its line within this of its anchors
FAR_FOOT_M = 600           # the most a station mapped off its track reaches from its node
MERGE_M = 100              # two stations of one line closer than this are one station
END_M = 250                # a relation's stop this far beyond a dead end of the track still counts


def line_id(name):
    h = hashlib.blake2b(f"sg|{name}".encode("utf-8"), digest_size=5)
    return "s" + h.hexdigest()


def name_key(name):
    """One spelling for a station name: NFKC, case-folded, dashes as hyphens, and without what
    OSM adds after " - " (the Sengkang LRT's stop positions are "Sengkang - East Loop
    Anticlockwise") or a trailing "MRT/LRT station"."""
    n = unicodedata.normalize("NFKC", name or "").replace("–", "-").casefold()
    n = re.sub(r"\s+-\s+.*$", "", n)
    n = re.sub(r"\s+((mrt|lrt)\s+)?station$", "", n)
    return re.sub(r"\s+", " ", n).strip()


def base_key(name):
    """Without a bracketed code: "Tampines (DT32)" -> tampines."""
    k = name_key(name)
    b = re.sub(r"\s*[(（].*?[)）]", "", k).strip()
    return b or k


def clean_name(name):
    """A station's name as shown: without OSM's " - East Loop Anticlockwise" platform suffix or
    a bracketed station code, "Tampines (DT32)" -> "Tampines"."""
    n = re.sub(r"\s+-\s+.*$", "", name or "")
    return re.sub(r"\s*\((?:[A-Z]{1,3}\d+[A-Z]?)\)$", "", n).strip()


def code_key(code):
    m = re.fullmatch(r"([A-Z]+)(\d*)", code)
    return (m.group(1), int(m.group(2)) if m.group(2) else -1) if m else (code, -1)


def load_list(path, log):
    """LTA's stations as {line: [(code, name, name_zh), ...]} in code order."""
    import pandas as pd
    p = Path(path) / "Train Station Codes and Chinese Names.xls"
    df = pd.read_excel(p)
    out = defaultdict(list)
    for r in df.itertuples(index=False):
        line = LTA_LINE.get(str(r.mrt_line_english).strip())
        if line is None:
            log(f"  SG: LTA list line {r.mrt_line_english!r} not a register line; skipped")
            continue
        code = RECODE.get(str(r.stn_code).strip(), str(r.stn_code).strip())
        zh = re.sub(r"\s+", "", str(r.mrt_station_chinese)).removesuffix("站")
        out[line].append((code, str(r.mrt_station_english).strip(), zh))
    n_file = sum(len(v) for v in out.values())
    for code, nm, line, _opened in ADDED:
        out[line].append((code, nm, ""))
    for line in out:
        out[line].sort(key=lambda r: code_key(r[0]))
    log(f"SG: LTA's list gives {n_file} station rows on {len(out)} lines, "
        f"{len(ADDED)} more added from the 2026 system map and Sentosa")
    return out


def load_osm(log):
    import build_model as bm
    ways, rels, stops, cid, cx, cy = bm.load("sg", log)
    return ways, rels, stops, bm.Coords(cid, cx, cy)


def build_stations(stops, log):
    """OSM's rail stations, one per complex, and every stop node mapped onto one (as
    tw_register, with this module's name keys)."""
    st = {}
    for nid, (tags, lon, lat) in stops.items():
        rail = (tags.get("railway") in STATION_RAILWAY
                or (tags.get("public_transport") == "station"
                    and any(tags.get(m) == "yes" for m in RAIL_MODES)))
        if rail and tags.get("name"):
            st[nid] = {"name": clean_name(tags["name"]), "name_en": tags.get("name:en") or "",
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
    for b in alias:
        del st[b]
    # Most Singapore stations are mapped as areas, which the extract does not read, plus stop
    # positions: the first stop position of a name becomes the station record.
    by_key = defaultdict(list)
    for nid, s in st.items():
        by_key[base_key(s["name"])].append(nid)
    made = 0
    for nid, (tags, lon, lat) in sorted(stops.items()):
        rail_stop = (tags.get("railway") == "stop"
                     or (tags.get("public_transport") == "stop_position"
                         and any(tags.get(m) == "yes" for m in RAIL_MODES)))
        if not rail_stop or not tags.get("name") or nid in st:
            continue
        k = base_key(tags["name"])
        if any(dist_m(lon, lat, st[c]["lon"], st[c]["lat"]) <= STOP_TO_STATION_M
               for c in by_key.get(k, ())):
            continue
        st[nid] = {"name": clean_name(tags["name"]),
                   "name_en": tags.get("name:en") or "", "lon": lon, "lat": lat, "rank": 2}
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
    log(f"SG: {len(st)} OSM rail stations ({len(alias)} records merged into a complex, "
        f"{made} made from stop positions alone), {len(node_st)} stop nodes placed on one")
    return st, node_st, by_key, by_base


def route_lists(rels, stops, node_st, log):
    """From OSM's route relations: {line: {station: [(lon, lat) of its stop nodes]}}, the
    pairs of stations some relation calls at in turn, and {line: {way id}}."""
    out = defaultdict(lambda: defaultdict(list))
    nxt = defaultdict(set)
    rways = defaultdict(set)
    for _rid, (tags, members) in rels.items():
        if tags.get("type") != "route":
            continue
        rname = unicodedata.normalize("NFKC", tags.get("name") or "")
        for line, (*_x, pat) in LINES.items():
            if not re.search(pat, rname):
                continue
            seq = []
            for ty, ref, role in members:
                if ty == "w":
                    rways[line].add(ref)
                if ty == "n" and ref in node_st and ref in stops:
                    _t, lon, lat = stops[ref]
                    out[line][node_st[ref]].append((lon, lat))
                    if not seq or seq[-1] != node_st[ref]:
                        seq.append(node_st[ref])
            nxt[line].update(frozenset((f"s{a}", f"s{b}")) for a, b in zip(seq[:-1], seq[1:]))
    log(f"SG: route relations give station lists for {len(out)} lines: "
        + ", ".join(f"{k} {len(v)}" for k, v in sorted(out.items())))
    return out, nxt, rways


def assign_ways(ways, rways, log):
    by_line = defaultdict(list)
    for wid, (tags, _nodes) in ways.items():
        if tags.get("railway") not in TRACK_KIND or tags.get("service") in NOT_TRACK_SERVICE:
            continue
        ln = TRACK.get(unicodedata.normalize("NFKC", tags.get("name") or "").strip())
        if ln:
            by_line[ln].append(wid)
    for ln in ROUTE_TRACK:
        got = [w for w in rways.get(ln, ()) if w in ways
               and ways[w][0].get("service") not in NOT_TRACK_SERVICE]
        by_line[ln].extend(got)
        log(f"SG: {ln}: {len(got)} ways from its route relations (its track is unnamed)")
    return by_line


def build(path, log):
    ways, rels, stops, coords = load_osm(log)
    st, node_st, by_key, by_base = build_stations(stops, log)
    lta = load_list(path, log)
    rlists, rnext, rways = route_lists(rels, stops, node_st, log)
    by_line = assign_ways(ways, rways, log)
    for ln in sorted(set(LINES) - set(by_line)):
        log(f"  SG: no OSM track found for {ln}")

    stations, lines, geoms = {}, [], {}
    dropped, cut, missing, unmatched, unlisted, unplaced = [], [], [], [], [], []
    n_route = n_listed = 0

    def station_rec(sid):
        return st[int(sid[1:])]

    for name in sorted(by_line):
        if name not in LINES:
            continue
        code, op, op_en, _pat = LINES[name]
        wids = sorted(set(by_line[name]))
        adj, xy, fast = line_graph(wids, ways, coords)
        if len(xy) < 2:
            dropped.append(name)
            continue
        near = Near(xy)

        # --- stations: the relations' stops, each where its own stop node meets this track
        anchors = defaultdict(set)
        ends = [v for v, nb in adj.items() if len(nb) == 1]
        end_near = Near({v: xy[v] for v in ends}) if ends else None
        for cplx, pts in rlists.get(name, {}).items():
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
                anchors[f"s{cplx}"].add(best)
                n_route += 1
            else:
                unplaced.append((name, st[cplx]["name"]))

        # --- LTA's list: each listed name to the OSM station nearest this track; a listed
        # station no relation placed is anchored there.
        listed, code_of, far = {}, {}, {}
        for c, nm, zh in lta.get(name, ()):
            cands = set(by_key.get(name_key(nm), ())) | set(by_base.get(base_key(nm), ()))
            best, bd, bv = None, MATCH_M, None
            for s in cands:
                v, d = near.nearest(st[s]["lon"], st[s]["lat"])
                if d <= bd:
                    best, bd, bv = s, d, v
            if best is None:
                unmatched.append((name, c, nm))
                continue
            sid = f"s{best}"
            listed[sid] = c
            code_of[c] = sid
            if sid not in anchors:
                anchors[sid].add(bv)
                n_listed += 1
                if bd > FOOT_M:
                    far[sid] = (st[best]["lon"], st[best]["lat"], bd)

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
        code_of = {c: merged.get(s, s) for c, s in code_of.items()}
        if len(anchors) < 2:
            dropped.append(name)
            continue

        # --- footprints, neighbours, sections: kr_register's
        foot, foot_d = {}, {}
        for sid, ans in anchors.items():
            disks = [(*xy[a], FOOT_M) for a in ans]
            if sid in far:
                lon, lat, d = far[sid]
                disks.append((lon, lat, min(d + FOOT_M, FAR_FOOT_M)))
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

        sections = {}
        for a, b in sorted(pairs):
            blocked = {v for v, s in foot.items() if s != a and s != b}
            got = between(adj, offsets(a), offsets(b), blocked)
            if got is None:
                continue
            nodes, km = got
            keep = [n for i, n in enumerate(nodes) if n > 0 or i == 0 or i == len(nodes) - 1]
            sections[(a, b)] = {"km": km,
                                "geom": [centre[a]] + [xy[n] for n in keep] + [centre[b]]}
        if name in rnext:
            for k in [k for k in sections if frozenset(k) not in rnext[name]]:
                v = sections.pop(k)
                cut.append((name, station_rec(k[0])["name"], station_rec(k[1])["name"],
                            round(v["km"], 2)))
        if not sections:
            dropped.append(name)
            continue

        # --- against the list: stations the relations have that it has not, and neighbours
        # in code order with no section between them
        on_line = {s for k in sections for s in k}
        for s in sorted(on_line - set(code_of.values())):
            unlisted.append((name, station_rec(s)["name"]))
        codes = [c for c, _n, _z in lta.get(name, ())]
        expect = []
        groups = defaultdict(list)
        for c in codes:
            groups[code_key(c)[0]].append(c)
        for g in groups.values():
            expect += list(zip(g[:-1], g[1:]))
        expect += [(a, b) for a, b in JOINS if a in codes and b in codes]
        have = {frozenset(k) for k in sections}
        for a, b in expect:
            sa, sb = code_of.get(a), code_of.get(b)
            if sa and sb and sa != sb and frozenset((sa, sb)) not in have:
                missing.append((name, a, b))
        extra_pairs = [k for k in sections
                       if not any(frozenset((code_of.get(a), code_of.get(b))) == frozenset(k)
                                  for a, b in expect)]

        lid = line_id(name)
        kinds = Counter(TRACK_KIND[ways[w][0]["railway"]] for w in wids)
        for sid in on_line:
            if sid not in stations:
                s = station_rec(sid)
                stations[sid] = {"id": sid, "name": s["name"], "name_en": s["name_en"],
                                 "lon": s["lon"], "lat": s["lat"], "lines": set()}
            stations[sid]["lines"].add(lid)
        from n02 import walk_order
        line = {
            "id": lid, "src": "sg", "service": False,
            "name": name, "name_en": name, "ref": code, "colour": "",
            "operator": op, "operator_en": op_en, "network": "",
            "kind": kinds.most_common(1)[0][0],
            "km": round(sum(v["km"] for v in sections.values()), 3),
            "variants": 1, "straight_sections": 0,
            "display": walk_order(sections.keys()),
            "sections": [[a, b, round(v["km"], 3)] for (a, b), v in sections.items()],
        }
        lines.append(line)
        geoms[lid] = {f"{a}|{b}": [[round(x, 5), round(y, 5)] for x, y in v["geom"]]
                      for (a, b), v in sections.items()}
        log(f"  SG: {name}: {len(on_line)} stations, {len(sections)} sections, "
            f"{line['km']:.2f} km, kind {line['kind']}"
            + (f"; sections off the code order: "
               + ", ".join(f"{station_rec(a)['name']}-{station_rec(b)['name']}"
                           for a, b in extra_pairs) if extra_pairs else ""))

    total = sum(l["km"] for l in lines)
    log(f"SG: {len(lines)} register lines, {total:,.1f} km, {len(stations)} stations; placed "
        f"by a route relation's stop node {n_route}, by LTA's list alone {n_listed}; "
        f"dropped (under two stations): {' '.join(dropped) or 'none'}")
    if cut:
        log(f"SG: {len(cut)} sections dropped as no train's consecutive stops: "
            + "; ".join(f"{l} {a}-{b} {km}" for l, a, b, km in cut))
    for label, rows in (("relation stops not on their line's track", unplaced),
                        ("stations on a line that LTA's list does not name", unlisted),
                        ("listed stations with no OSM station near the track", unmatched),
                        ("pairs adjacent in code order with no section", missing)):
        if rows:
            log(f"SG: {len(rows)} {label}:")
            for r in rows:
                log("    " + " ".join(str(x) for x in r))
        else:
            log(f"SG: no {label}")
    return lines, stations, geoms


def clip(log=print):
    """Rewrite data/proc/sg keeping only ways with a node inside SG_POLY, stops inside it,
    and relations with a member left (as extract.clip does for a box)."""
    from shapely import contains_xy
    from shapely.geometry import Polygon
    d = ROOT / "data" / "proc" / "sg"
    ways = pickle.load(open(d / "ways.pkl", "rb"))
    rels = pickle.load(open(d / "rels.pkl", "rb"))
    stops = pickle.load(open(d / "stops.pkl", "rb"))
    c = np.load(d / "coords.npz")
    nid, x, y = c["id"], c["x"] / 1e7, c["y"] / 1e7
    poly = Polygon(SG_POLY)
    inside = set(nid[contains_xy(poly, x, y)].tolist())
    n0 = (len(ways), len(stops), len(rels))
    gone = Counter(t.get("name") for t, n in ways.values()
                   if not any(int(i) in inside for i in n))
    ways = {k: v for k, v in ways.items() if any(int(i) in inside for i in v[1])}
    stops = {k: v for k, v in stops.items() if contains_xy(poly, v[1], v[2])}
    kept = {("w", k) for k in ways} | {("n", k) for k in stops}
    routes = {k for k, (tags, members) in rels.items()
              if tags.get("type") == "route" and any((t, r) in kept for t, r, _ in members)}
    rels = {k: v for k, v in rels.items()
            if k in routes or any(t == "r" and r in routes for t, r, _ in v[1])}
    for name, fn in (("ways", ways), ("rels", rels), ("stops", stops)):
        with open(d / f"{name}.pkl", "wb") as f:
            pickle.dump(fn, f, protocol=4)
    log(f"clipped to Singapore: ways {n0[0]} -> {len(ways)}, stops {n0[1]} -> {len(stops)}, "
        f"relations {n0[2]} -> {len(rels)}; ways dropped by name: {dict(gone)}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", action="store_true",
                    help="drop what lies outside Singapore from data/proc/sg")
    args = ap.parse_args()
    if args.clip:
        clip()
    else:
        ap.print_help()
