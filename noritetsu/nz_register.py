"""New Zealand: KiwiRail's own line register (open data, CC BY 4.0), written into rinf.py's input
format (as id_register.py and ua_register.py do) and traced over OSM track by rinf.py.

    python nz_register.py --fetch        # data/raw/nz/{network,locations,kmposts,metro}.geojson
    python nz_register.py --convert      # data/raw/nz/{sections,points,names}.json
    python nz_register.py --report       # the register without OSM: lines, km, checks
    python build_model.py --region nz --register nz_register:data/raw/nz

`--register nz_register:data/raw/nz` converts and then runs rinf.build on the result.
rinf_countries/nz.py holds the settings rinf.py reads, nz_sources.md the sources, the numbers
and the decisions.

SOURCE. KiwiRail's open data (data-kiwirail.opendata.arcgis.com, CC BY 4.0), three layers of
its ArcGIS feature services: NZ_Rail_Network (one feature per line, "North Island Main Trunk",
with its line code), KiwiRail_KM_Posts (4,209 posts, every half or whole km of every line, with
the line's name and abbreviation: NIMT, NAL, WRAPA...) and RailwayLocations (920 named places on
the lines: stations, yards, junctions, historic station sites, with a status, one of
Station/Passenger, Operational, Yard, Place, Historic, and the line code; their `Km_ref` is a
whole km only).

THE LINE UNIT is KiwiRail's line ("North Island Main Trunk", Wellington - Auckland, 681 km, now
on through the City Rail Link to Maungawhau; "Wairarapa Line"; "Melling Branch"). The Taieri
Gorge Railway (Dunedin City Council's, no line code, no posts) is the one line written by hand.

CHAINAGE. A location's km on its line is where it lies along the line's km posts (projected on
the polyline of posts in km order, so to the metre between posts, not KiwiRail's whole-km
`Km_ref`). A section's length is the difference of its ends' km, which is what rinf.py checks
the OSM trace against and ships as `chain`. The Taieri Gorge Railway's km is measured along
KiwiRail's own geometry of it.

WHICH STRETCHES. Only the lines and stretches with scheduled passenger trains are written
(SCOPE): Auckland's and Wellington's suburban networks, the NIMT whole (Te Huia, the Northern
Explorer, the Capital Connection), Wellington - Masterton, the Main North Line (the Coastal
Pacific, seasonal), Christchurch - Rolleston and the Midland Line (the TranzAlpine), and
Dunedin - Wingatui - Pukerangi (Dunedin Railways' Taieri Gorge train, Thursday to Monday). The
rest of KiwiRail's network is freight only and is not a register line.

POINTS. Every location in scope on its line (a location more than OFF_LINE_M from its line's
posts is left out and logged), Station/Passenger typed a stop ("10"), the rest a junction-type
point ("80") that rinf.py merges away unless lines meet there; a line's ends are the location
nearest its first or last post within REUSE_M (or ENDS), else a junction point at the post,
which also goes on the line it leaves (JOIN: Distance Junction, Melling's junction at Petone).
Points shared between lines are listed in names.json "cut", so rinf.py ends sections there.
REASSIGN moves a location filed on the wrong line (KiwiRail files Manukau under the NIMT at the
junction's km, the Strand under the NIMT, Greymouth under the Hokitika Line).

The `path` argument is data/raw/nz; the OSM half is read from data/proc/nz (extract.py).
"""
import argparse
import json
import math
import os
import pickle
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw" / "nz"
PROC = ROOT / "data" / "proc" / "nz"
USER_AGENT = "noritetsu-build/1.0 (hobby rail map)"
SERVICES = "https://services6.arcgis.com/eqX2HMCD3H8MUs6Q/ArcGIS/rest/services/"
LAYERS = {
    "network.geojson": "NZ_Rail_Network/FeatureServer/0",
    "locations.geojson": "RailwayLocations/FeatureServer/0",
    "kmposts.geojson": "KiwiRail_KM_Posts/FeatureServer/0",
    "metro.geojson": "MetroServices/FeatureServer/0",
}

OFF_LINE_M = 400       # a location further than this from its own line's posts is left out
REUSE_M = 300          # a line's end takes the location this close to its end post
JOIN_M = 150           # a junction point at a line's end goes on any line in scope this close
SAME_M = 60            # two points of a line this close are one (the passenger one kept)

# The stretches with scheduled passenger trains, per KiwiRail line abbreviation: (from, to),
# each a location NAME on the line, None for the line's own end, or "@ABBR" for where that
# line's first post meets this one. nz_sources.md has the trains behind each.
SCOPE = {
    "NIMT": [(None, None)],              # Wellington - Auckland - City Rail Link - Maungawhau
    "NAL": [(None, "SWANSON")],          # Westfield - Newmarket - Swanson; freight beyond
    "NWMKT": [(None, None)],             # Newmarket - Parnell - the Strand
    "ONHGA": [(None, None)],             # Penrose - Onehunga
    "MANUK": [(None, None)],             # Puhinui - Manukau
    "ELK": [(None, None)],               # the East Link at Maungawhau (CRL - Grafton)
    "WRAPA": [(None, "MASTERTON")],      # Wellington - Upper Hutt - Masterton; freight beyond
    "MLING": [(None, None)],             # Petone - Melling
    "JVILL": [(None, None)],             # Wellington - Johnsonville
    "MNL": [(None, None)],               # Christchurch - Picton (the Coastal Pacific)
    "MSL": [("@MNL", "@MDLND"),          # Christchurch - Rolleston (the TranzAlpine)
            ("DUNEDIN", "WINGATUI")],    # Dunedin - Wingatui (the Taieri Gorge train)
    "MDLND": [(None, "GREYMOUTH")],           # Rolleston - Greymouth (the TranzAlpine)
    "TAI": [(None, None)],               # Wingatui - Taieri (the Taieri Gorge train)
    "TGR": [(None, "Puketerangi")],      # Taieri - Pukerangi (the Taieri Gorge train)
}
# Lines with no km posts, written by hand: id -> (name, operator, the KiwiRail network
# feature whose geometry gives the km, [point names]; a name is a KiwiRail location or, failing
# that, the OSM station of that name). The Taieri Gorge Railway is Dunedin City Council's.
HAND = {
    "TGR": ("Taieri Gorge Railway", "DCC", "Taieri Gorge Railway",
            ["TAIERI", "Hindon", "Puketerangi", "Middlemarch"]),
}
# The line a line's end leaves, where that end is a junction rather than a station: the junction
# point goes on it too, so both lines' sections end there. Only these: lines run side by side
# out of Wellington (the NIMT, the Wairarapa Line and the Johnsonville Line share their first
# 1.5 km of posts), so nearness alone put the Wairarapa Line's junction on the Johnsonville Line.
JOIN = {("NAL", "start"): "NIMT", ("NWMKT", "start"): "NAL", ("MANUK", "start"): "NIMT",
        ("ELK", "start"): "NIMT", ("ELK", "end"): "NAL", ("WRAPA", "start"): "NIMT",
        ("MLING", "start"): "WRAPA", ("MDLND", "start"): "MSL", ("MNL", "start"): "MSL"}
# A line's end that is a station further than REUSE_M from its end post: the Midland Line's
# km 0 is 330 m west of Rolleston station, where the TranzAlpine leaves the Main South Line.
ENDS = {("MDLND", "start"): "ROLLESTON"}
HAND_STOPS = {"Puketerangi"}         # the train's turning point (OSM's spelling of Pukerangi)
# Locations filed on the wrong line: (name, KiwiRail's line) -> the line they lie on. Checked
# on the map: Manukau is the end of the Manukau Branch (KiwiRail files it at the junction's NIMT
# km); the Strand is the end of the Newmarket Branch at Quay Park; Greymouth is km 211 of the
# Midland Line; Christchurch station (Addington) lies on the Main North Line's first 400 m.
REASSIGN = {("MANUKAU", "NIMT"): "MANUK", ("THE STRAND", "NIMT"): "NWMKT",
            ("GREYMOUTH", "HKTKA"): "MDLND", ("CHRISTCHURCH", "MSL"): "MNL"}
# Locations KiwiRail does not flag Station/Passenger that are a line's passenger terminus: the
# Strand (Te Huia's and the Northern Explorer's Auckland station, a "Place") and Dunedin
# (Dunedin Railways' station, "Operational"). A stop in the middle of a line needs no entry:
# rinf.py's `osm_stops` puts the OSM stations trains stop at on the sections.
STOP_FIX = {"THE STRAND", "DUNEDIN"}
# KiwiRail's location names written as people do (the rest in title case).
NAME_FIX = {"WAITEMATA (BMT)": "Waitematā", "TE WAIHOROTIU (AOT)": "Te Waihorotiu",
            "KARANGA-A-HAPE (KRD)": "Karanga-a-Hape", "MAUNGAWHAU (MTD)": "Maungawhau",
            "PUKETERANGI": "Pukerangi"}


def log_print(msg, t0=time.time()):
    print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)


def dist_m(lon1, lat1, lon2, lat2):
    dx = (lon2 - lon1) * math.cos(math.radians((lat1 + lat2) / 2)) * 111320
    dy = (lat2 - lat1) * 110570
    return math.hypot(dx, dy)


def title(name):
    if name in NAME_FIX:
        return NAME_FIX[name]
    if not name.isupper():
        return name
    return " ".join("-".join(p.capitalize() for p in w.split("-"))
                    if not re.fullmatch(r"\(.*\)", w) else w for w in name.split())


# ================================================================ fetching

def _get(url):
    for k in range(5):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=180) as r:
                d = json.loads(r.read().decode("utf-8"))
            if "error" in d:
                raise RuntimeError(d["error"])
            return d
        except Exception as e:                                  # noqa: BLE001
            if k == 4:
                raise
            print(f"  retry after {e}", flush=True)
            time.sleep(5 * (k + 1))


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    for fn, layer in LAYERS.items():
        feats, off = [], 0
        while True:
            q = urllib.parse.urlencode({"where": "1=1", "outFields": "*", "f": "geojson",
                                        "outSR": 4326, "resultOffset": off,
                                        "resultRecordCount": 1000})
            page = _get(SERVICES + layer + "/query?" + q)
            got = page.get("features") or []
            feats += got
            if not got or (len(got) < 1000 and not page.get("exceededTransferLimit")):
                break
            off += len(got)
        out = RAW / fn
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps({"type": "FeatureCollection", "source": SERVICES + layer,
                                   "fetched": time.strftime("%Y-%m-%d"), "features": feats}),
                       "utf-8")
        os.replace(tmp, out)
        print(f"wrote {out} ({len(feats)} features)", flush=True)


# ================================================================ chainage

class Chain:
    """A line's km along a polyline whose vertices carry a km: the km posts in km order, or a
    line's own geometry with km measured along it."""

    def __init__(self, pts, kms):
        self.pts, self.kms = pts, kms

    @classmethod
    def from_geometry(cls, pts):
        kms, s = [0.0], 0.0
        for a, b in zip(pts[:-1], pts[1:]):
            s += dist_m(*a, *b) / 1000
            kms.append(s)
        return cls(pts, kms)

    def locate(self, lon, lat):
        """(km, metres off the polyline). Before the first vertex or past the last, the km
        runs on along the end segment (Wellington station lies 60 m short of the NIMT's
        km 0 post)."""
        k = math.cos(math.radians(lat)) * 111320
        best = None
        n = len(self.pts) - 1
        for i, (a, b) in enumerate(zip(self.pts[:-1], self.pts[1:])):
            ax, ay = (a[0] - lon) * k, (a[1] - lat) * 110570
            bx, by = (b[0] - lon) * k, (b[1] - lat) * 110570
            dx, dy = bx - ax, by - ay
            L2 = dx * dx + dy * dy
            t = 0.0 if not L2 else -(ax * dx + ay * dy) / L2
            lo = -math.inf if i == 0 else 0.0
            hi = math.inf if i == n - 1 else 1.0
            tc = min(max(t, lo), hi)
            d =math.hypot(ax + tc * dx, ay + tc * dy)
            if best is None or d < best[0]:
                best = (d, self.kms[i] + tc * (self.kms[i + 1] - self.kms[i]))
        return best[1], best[0]


def load_kiwirail():
    def rd(fn):
        return json.loads((RAW / fn).read_text("utf-8"))
    posts = defaultdict(list)
    names = {}
    for f in rd("kmposts.geojson")["features"]:
        p = f["properties"]
        posts[p["LINE_ABBRV"]].append((float(p["KM"]), tuple(f["geometry"]["coordinates"][:2])))
        names[p["LINE_ABBRV"]] = (p["LINE"], str(p["LINECODE"]))
    chains = {}
    for ab, ps in posts.items():
        ps.sort()
        # one post per km value: the one nearest the next post in km (the East Link has two
        # km-0 posts, one per leg of its junction)
        by_km = defaultdict(list)
        for km, c in ps:
            by_km[km].append(c)
        kms = sorted(by_km)
        pts = []
        for i, km in enumerate(kms):
            cs = by_km[km]
            if len(cs) > 1 and len(kms) > 1:
                nxt = by_km[kms[i + 1] if i + 1 < len(kms) else kms[i - 1]][0]
                gap = abs((kms[i + 1] if i + 1 < len(kms) else kms[i - 1]) - km) * 1000
                cs = sorted(cs, key=lambda c: abs(dist_m(*c, *nxt) - gap))
            pts.append(cs[0])
        chains[ab] = Chain(pts, kms)
    code_ab = {code: ab for ab, (_n, code) in names.items()}
    net = rd("network.geojson")["features"]
    locs = []
    for f in rd("locations.geojson")["features"]:
        p = f["properties"]
        ab = code_ab.get(str(p["LineCode"]))
        nm = p["NAME"].strip()
        locs.append({"name": nm, "km_ref": p["Km_ref"],
                     "status": "Station/Passenger" if nm in STOP_FIX else p["Status"],
                     "ab": REASSIGN.get((nm, ab), ab), "moved": (nm, ab) in REASSIGN,
                     "pt": tuple(f["geometry"]["coordinates"][:2])})
    fetched = rd("kmposts.geojson").get("fetched", "")
    return chains, names, locs, net, fetched


def osm_station_points():
    """OSM rail station points by name (for the hand line), from the extract."""
    out = defaultdict(list)
    p = PROC / "stops.pkl"
    if not p.exists():
        return out
    with open(p, "rb") as f:
        stops = pickle.load(f)
    for nid, (tags, lon, lat) in stops.items():
        if tags.get("railway") in ("station", "halt") and tags.get("name"):
            out[tags["name"]].append((lon, lat))
    return out


# ================================================================ the register

def build_register(log):
    chains, names, locs, net, fetched = load_kiwirail()
    osm_pts = osm_station_points()
    for lid, (name, im, feat, pnames) in HAND.items():
        geo = next(f["geometry"] for f in net if f["properties"].get("LINESECTIO") == feat)
        parts = [geo["coordinates"]] if geo["type"] == "LineString" else geo["coordinates"]
        pts = [tuple(c[:2]) for c in max(parts, key=len)]
        chains[lid] = Chain.from_geometry(pts)
        names[lid] = (name, "")
        for pn in pnames:
            if any(l["name"] == pn for l in locs):
                continue
            got = osm_pts.get(pn)
            if got:
                locs.append({"name": pn, "status": "Station/Passenger" if pn in HAND_STOPS
                             else "Hand", "km_ref": None, "ab": lid, "pt": got[0]})
            else:
                log(f"NZ: hand point {pn} on {lid}: no OSM station of that name")
    # every location on its own line, with its km
    ops = {}                                        # op -> point
    on_line = defaultdict(list)                     # ab -> [(km, op)]
    off = []
    by_name = defaultdict(list)
    for i, l in enumerate(locs):
        ch = chains.get(l["ab"])
        op = f"nz:{re.sub(r'[^A-Z0-9]+', '-', l['name'].upper()).strip('-')}"
        if op in ops:
            op = f"{op}-{l['ab']}"
        ops[op] = {"op": op, "name": title(l["name"]), "pt": l["pt"],
                   "stop": l["status"] == "Station/Passenger", "status": l["status"]}
        by_name[l["name"]].append(op)
        if ch is None:
            continue
        km, d = ch.locate(*l["pt"])
        if d > OFF_LINE_M and not l.get("moved"):
            off.append((l["ab"], l["name"], round(d)))
            continue
        on_line[l["ab"]].append((km, op))
    log(f"NZ: {len(locs)} KiwiRail locations; {len(off)} lie more than {OFF_LINE_M} m off "
        f"their line's km posts and are left out: "
        + ", ".join(f"{n} ({ab}, {d} m)" for ab, n, d in off[:30]))
    # a hand line's points that are another line's locations (Taieri, the end of the Taieri
    # Branch) go on the hand line too
    for lid, (_name, _im, _feat, pnames) in HAND.items():
        for pn in pnames:
            for op in by_name.get(pn, ()):
                if not any(o == op for _k, o in on_line[lid]):
                    on_line[lid].append((chains[lid].locate(*ops[op]["pt"])[0], op))

    def nearest_loc(pt, r):
        best = None
        for op, p in ops.items():
            d = dist_m(*pt, *p["pt"])
            if d <= r and (best is None or (not p["stop"], d) < (not best[1]["stop"], best[0])):
                best = (d, p, op)
        return best

    # each KiwiRail line's ends: a location within REUSE_M of the end post, else a junction
    # point (a hand line's ends are its listed points)
    ends = {}
    for ab in SCOPE:
        ch = chains.get(ab)
        if ch is None:
            log(f"NZ: no km posts or geometry for {ab}")
            continue
        if ab in HAND:
            continue
        for which, pt, km in (("start", ch.pts[0], ch.kms[0]), ("end", ch.pts[-1], ch.kms[-1])):
            got = nearest_loc(pt, REUSE_M)
            if (ab, which) in ENDS:
                op = by_name[ENDS[(ab, which)]][0]
                got = (0.0, ops[op], op)
            if got:
                op = got[2]
                kk, _d = ch.locate(*ops[op]["pt"])
            else:
                op = f"nz:j:{ab}:{km:g}"
                ops[op] = {"op": op, "name": f"{names[ab][0]} km {km:g}", "pt": pt,
                           "stop": False, "status": "Junction"}
                kk = km
            ends[(ab, which)] = op
            if not any(o == op for _k, o in on_line[ab]):
                on_line[ab].append((kk, op))
    # a junction point at a line's end goes on the line it leaves (JOIN)
    n_join = 0
    for (ab, which), op in ends.items():
        other = JOIN.get((ab, which))
        if ops[op]["status"] != "Junction" or other is None:
            continue
        km, d = chains[other].locate(*ops[op]["pt"])
        if d > JOIN_M:
            log(f"NZ: {ops[op]['name']} lies {d:.0f} m from {other}, not put on it")
            continue
        if not any(o == op for _k, o in on_line[other]):
            on_line[other].append((km, op))
            n_join += 1

    def bound(ab, b, default):
        if b is None:
            return default
        if b.startswith("@"):
            # where that line starts, which goes on this line too (Christchurch station, the
            # Main North Line's first point, is where the Main South Line's piece begins)
            op = ends[(b[1:], "start")]
            km = chains[ab].locate(*ops[op]["pt"])[0]
            if not any(o == op for _k, o in on_line[ab]):
                on_line[ab].append((km, op))
            return km
        hit = [k for k, op in on_line[ab] if op in by_name.get(b, ()) or ops[op]["name"] == b]
        if not hit:
            raise SystemExit(f"NZ: scope bound {b} not on {ab}")
        return hit[0]

    reg = {}
    shared = Counter()
    for ab, pieces in SCOPE.items():
        if ab not in chains:
            continue
        pts = sorted(on_line[ab])
        lo_all, hi_all = pts[0][0], pts[-1][0]
        rows = []
        for a, b in pieces:
            lo, hi = sorted((bound(ab, a, lo_all), bound(ab, b, hi_all)))
            pts = sorted(on_line[ab])
            run = [(k, op) for k, op in pts if lo - 0.005 <= k <= hi + 0.005]
            # two points this close are one: the stop, else a line's end, else the first
            end_ops = set(ends.values())

            def rank(op):
                return (ops[op]["stop"], op in end_ops)
            kept = []
            for k, op in run:
                if kept and (k - kept[-1][0]) * 1000 < SAME_M:
                    if rank(op) > rank(kept[-1][1]):
                        kept[-1] = (k, op)
                    continue
                kept.append((k, op))
            for (k1, a1), (k2, b2) in zip(kept[:-1], kept[1:]):
                rows.append((a1, b2, round(k2 - k1, 3)))
        if rows:
            reg[ab] = rows
            for op in {o for r in rows for o in r[:2]}:
                shared[op] += 1
    cut = sorted(op for op, n in shared.items() if n > 1)
    log(f"NZ: {len(reg)} lines in scope, {sum(len(r) for r in reg.values())} pieces, "
        f"{sum(x[2] for r in reg.values() for x in r):,.1f} km; {n_join} junction points "
        f"put on a second line; {len(cut)} points shared by two lines or more")
    return reg, ops, names, cut, fetched, chains


# ================================================================ rinf.py's input

# OSM way names (less "Up Main", "/Br 57"...) -> line id, for rinf's `way_line`.
WAY_NAMES = {
    "North Island Main Trunk": "NIMT", "North Auckland Line": "NAL",
    "Newmarket Line": "NWMKT", "Newmarket": "NWMKT", "Onehunga Branch": "ONHGA",
    "Manukau Branch": "MANUK", "East Link": "ELK", "Wairarapa Line": "WRAPA",
    "Melling Line": "MLING", "Johnsonville Line": "JVILL", "Main North Line": "MNL",
    "Main South Line": "MSL", "Midland Line": "MDLND", "Taieri Branch": "TAI",
    "Taieri Gorge Railway": "TGR",
}


def convert(log=log_print, write=True):
    reg, ops, names, cut, fetched, _chains = build_register(log)
    rows, used = [], set()
    for ab, secs in reg.items():
        im = HAND[ab][1] if ab in HAND else "KiwiRail"
        for k, (a, b, km) in enumerate(secs, 1):
            rows.append({"sol": f"{ab}:{k}", "line": ab, "a": a, "b": b, "len": f"{km:.3f}",
                         "im": im, "label": f"{ops[a]['name']} - {ops[b]['name']}"})
            used |= {a, b}
    pts_out = []
    for op in sorted(used):
        p = ops[op]
        pts_out.append({"op": op, "uopid": op.split(":", 1)[1], "name": p["name"],
                        "type": "10" if p["stop"] else "80",
                        "lon": p["pt"][0], "lat": p["pt"][1]})
    meta = {"lines": {ab: {"name": names[ab][0], "name_en": "",
                           "km": round(sum(x[2] for x in reg[ab]), 3)} for ab in reg},
            "cut": [op.split(":", 1)[1] for op in cut],
            "way_names": WAY_NAMES}
    log(f"NZ: {len(meta['lines'])} lines, {len(rows)} section rows, {len(pts_out)} points "
        f"({sum(1 for r in pts_out if r['type'] == '10')} stops)")
    if write:
        stamp = {"endpoint": "KiwiRail open data (nz_register.py)", "fetched": fetched}
        for fn, obj in (("sections.json", {**stamp, "rows": rows}),
                        ("points.json", {**stamp, "rows": pts_out}),
                        ("names.json", meta)):
            tmp = RAW / (fn + ".tmp")
            tmp.write_text(json.dumps(obj, ensure_ascii=False), "utf-8")
            os.replace(tmp, RAW / fn)
        log(f"NZ: wrote sections.json, points.json, names.json to {RAW}")
    return rows, pts_out, meta


def build(path, log):
    """build_model's register hook: convert, then rinf.py traces it over OSM track."""
    convert(log)
    import importlib
    import rinf
    import rinf_countries.nz as conf_mod
    importlib.reload(conf_mod)                 # names.json was just rewritten
    lines, stations, geoms = rinf.build(path, log)
    # A line in two pieces (the Main South Line: Christchurch - Rolleston, Dunedin - Wingatui)
    # is two lines of one name: an English name with their end stops tells them apart.
    by_name = Counter(l["name"] for l in lines)
    for l in lines:
        if by_name[l["name"]] > 1 and not l.get("name_en"):
            stops = [s for s in l["display"] if not stations[s].get("junction")]
            if len(stops) < 2:           # Dunedin - Wingatui: Wingatui is no stop
                stops = l["display"]
            l["name_en"] = (f"{l['name']} ({stations[stops[0]]['name']} – "
                            f"{stations[stops[-1]]['name']})")
    return lines, stations, geoms


# ================================================================ --report

# KiwiRail's own km between two places (from the posts), against a published figure.
CHAIN_CHECKS = [
    ("NIMT", "WELLINGTON", "WAITEMATA (BMT)", 681.0,
     "en.WP: the NIMT is 681 km Wellington - Auckland"),
    ("MNL", "CHRISTCHURCH", "PICTON", 348.0, "en.WP: Main North Line 348 km"),
    ("MDLND", None, "GREYMOUTH", 211.0, "en.WP: Midland Line 211 km Rolleston - Greymouth"),
    ("WRAPA", "WELLINGTON", "MASTERTON", 91.0, "en.WP Wairarapa Line: Masterton km 91"),
    ("JVILL", "WELLINGTON", "JOHNSONVILLE", 10.5, "en.WP: Johnsonville Branch 10.5 km"),
]


def report(log=log_print):
    reg, ops, names, cut, _f, chains = build_register(log)
    print(f"\n{'line':34} {'pieces':>6} {'km':>8}")
    for ab, rows in reg.items():
        print(f"{names[ab][0][:34]:34} {len(rows):6d} {sum(r[2] for r in rows):8.1f}  "
              f"{ops[rows[0][0]]['name']} ... {ops[rows[-1][1]]['name']}")
    print(f"all {sum(r[2] for rows in reg.values() for r in rows):,.1f} km")
    print("\nshared points:", ", ".join(ops[o]["name"] for o in cut))


def show(ab):
    reg, ops, names, cut, _f, _c = build_register(lambda m: None)
    for a, b, km in reg.get(ab, []):
        print(f"  {ops[a]['name']:<28} {'*' if ops[a]['stop'] else ' '} -> {ops[b]['name']:<28} "
              f"{'*' if ops[b]['stop'] else ' '} {km:8.3f}")


def split_pieces(lines, stations, geoms, reg_ways, state, log):
    """build_model's hook, after drop_unridden_sections: rinf.split_pieces (folds a stop's 0 km
    link junctions back into it, and bridges where the country's settings ask)."""
    import rinf
    rinf.split_pieces(lines, stations, geoms, reg_ways, state, log)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--convert", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--show", metavar="ABBR")
    args = ap.parse_args()
    if args.fetch:
        fetch()
    if args.convert:
        convert()
    if args.report:
        report()
    if args.show:
        show(args.show)
