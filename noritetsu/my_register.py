"""Lines, stations and sections for Malaysia, with OpenStreetMap's track as the geometry.

    python extract.py --region my --pbf data/raw/malaysia-singapore-brunei-latest.osm.pbf --station-areas
    python my_register.py --clip          # drop Singapore and the Thai stubs from data/proc/my
    python build_model.py --region my --register my_register:data/raw/my
    python build_tiles.py --region my
    python check_model.py --region my

WHY NOT KOREA'S NAMED-TRACK RECIPE.  Malaysia's metro track carries its line's name ("Laluan
Kajang", "Laluan Kelana Jaya", "LRT 3", "ERL"), but KTM's does not: 1,642 of its 2,850 km of
main track is named just "KTM" (`python probe_kr_ways.py --region my`), and OSM's KTM line
relations (route=railway) cover the West Coast Line for 210 km of about 900. So every line here
is laid along OSM track by SHORTEST PATH between known points, and the stations come from
the timetables:

  KTM (and Sabah's railway) are INFRASTRUCTURE LINES, as KTM and en.wikipedia describe its
  network: the West Coast Line Padang Besar - JB Sentral, the East Coast Line Gemas - Tumpat,
  and the branches to Butterworth, Batu Caves, Port Klang, Terminal Skypark and Pasir Gudang.
  Each is the shortest path over KTM's running track through the WAYPOINTS in LINES (stations
  far enough apart to pin the route). Its stations are every stop of KTM's own GTFS
  (data.gov.my, `data/raw/my/gtfs/my_ktmb.gtfs.zip`) that lies on that path, in the order the
  path passes them; a stop near two lines' paths (a junction station: KL Sentral, Gemas,
  Bukit Mertajam) is on both. KTM's services (Komuter's Seremban and Port Klang Lines, the
  northern and southern Komuter, ETS, the Intercity Shuttle Timur, Ekspres Rakyat Timuran,
  Shuttle Tebrau) are operating patterns over these lines: the Komuter and ETS ones are OSM
  lines on top (rules/my.py says which are named trains), and the Intercity ones are mapped
  in OSM only as route=railway, so they are not drawn as lines of their own.

  The Klang Valley's rapid transit lines (Kelana Jaya, Ampang, Sri Petaling, Kajang,
  Putrajaya, Shah Alam, the KL Monorail) take their stations in order from Prasarana's GTFS
  (`my_prasarana_rail.gtfs.zip`), matched to OSM by the station code OSM writes in the name
  ("KJ15 KL Sentral") or by name; each section is the shortest path between two neighbouring
  stations over that line's named track. KLIA Transit and the Penang Hill Railway take their
  stations from OSM's own route relation (neither is in an open feed).

Sections are measured along the track from station to station: on a KTM line, the distance
between the two stations' projections onto the line's path; on the others, the shortest path
between the stations' anchor points (the nearest point of the line's track to the station's
OSM node), as kr_register.py measures.

BORDERS.  The West Coast Line runs on over two borders, and each run-on is a section to a
border point (borders.py; MY_BORDER below until borders.EXTRA has them): Padang Besar to the
Thai border 0.48 km north of the station (SRT's trains from Hat Yai terminate at Padang Besar,
and KTM's ETS runs on to Hat Yai), and JB Sentral 1.27 km over the Johor Causeway to the
middle of the strait (Shuttle Tebrau to Woodlands). Those sections end at a `junction`, so
build_model keeps them only if passenger routes run over them; OSM has no route relation over
either, hence the `served_sections` list the line carries, which build_model's
drop_unridden_sections reads (landed 2026-10-03, one hook shared with th_register).

CLIP.  Geofabrik ships Malaysia with Singapore and Brunei. `--clip` rewrites data/proc/my
without what lies in Singapore (built as `sg` from the same file) or in Thailand: a way stays
if any node lies outside them (so the causeway's border-crossing way stays whole), a stop if it
lies outside, a relation if a member stayed. Where the border is, near the rail crossings, is
OSM's own admin_level=2 boundary (data/raw/my/borders_osm.geojson, fetched once from Overpass);
elsewhere Singapore is sg_register.SG_POLY.

The `path` argument is data/raw/my; the OSM half is read from data/proc/my (extract.py).
"""
import argparse
import csv
import hashlib
import io
import json
import math
import os
import pickle
import re
import sys
import unicodedata
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np

from kr_register import Near, between, dist_m, line_graph

ROOT = Path(__file__).resolve().parent
PROC = ROOT / "data" / "proc" / "my"

KTM = ("Keretapi Tanah Melayu", "Keretapi Tanah Melayu")
RAPID = ("Rapid Rail", "Rapid Rail")

# Border points the West Coast Line ends at: where OSM's track crosses OSM's admin_level=2
# boundary (Overpass, 2026-10-03). The same ids and points are proposed for borders.EXTRA;
# borders.load()'s own entry wins once it is there.
MY_BORDER = {
    # Padang Besar: way 1237531937 (KTM) meets 1419678317 (SRT's Hat Yai - Padang Besar) on
    # boundary way 206957351.
    "xPadangBesar": (100.322477, 6.665252, ["my", "th"]),
    # Johor Causeway: way 925109455 crosses boundary way 1455785333.
    "xWoodlands": (103.769336, 1.452652, ["my", "sg"]),
}

# Each line: name (Malay), English name, ref, (operator, English operator), kind, track, and
# where its stations come from:
#   ("path", [waypoint, ...])   KTM/Sabah: shortest path through these; stations = the
#                               timetable's or OSM's stops on it. A waypoint is a station name
#                               (resolved as a stop is) or a border id from MY_BORDER.
#   ("gtfs", route_id)          Prasarana's feed: its longest trip, in order
#   ("osm", relation id)        an OSM route relation's stops, in order
LINES = [
    dict(key="wcl", name="Laluan Pantai Barat", name_en="West Coast Line", ref="",
         op=KTM, kind="rail", track="ktm", stops="ktm",
         path=["xPadangBesar", "PADANG BESAR", "ARAU", "ALOR SETAR", "SUNGAI PETANI",
               "BUKIT MERTAJAM", "PARIT BUNTAR", "TAIPING", "KUALA KANGSAR", "IPOH",
               "KAMPAR", "TAPAH ROAD", "TANJONG MALIM", "RAWANG", "KEPONG SENTRAL", "PUTRA",
               "KUALA LUMPUR", "KL SENTRAL", "SEPUTEH", "BDR TASEK SELATAN", "KAJANG",
               "NILAI", "SEREMBAN", "PULAU SEBANG/TAMPIN", "GEMAS", "SEGAMAT", "LABIS",
               "PALOH", "KLUANG", "KULAI", "KEMPAS BARU", "JB SENTRAL", "xWoodlands"]),
    dict(key="but", name="Laluan Cawangan Butterworth", name_en="Butterworth Branch Line",
         ref="", op=KTM, kind="rail", track="ktm", stops="ktm",
         path=["BUKIT MERTAJAM", "BUKIT TENGAH", "BUTTERWORTH"]),
    dict(key="btc", name="Laluan Cawangan Batu Caves", name_en="Batu Caves Branch Line",
         ref="", op=KTM, kind="rail", track="ktm", stops="ktm",
         path=["PUTRA", "SENTUL", "BATU KENTOMENN", "KAMPUNG BATU", "BATU CAVES"]),
    dict(key="pkl", name="Laluan Cawangan Pelabuhan Klang", name_en="Port Klang Branch Line",
         ref="", op=KTM, kind="rail", track="ktm", stops="ktm",
         path=["KL SENTRAL", "ANGKASAPURI", "PETALING", "SUBANG JAYA", "SHAH ALAM", "KLANG",
               "PEL KLANG SEL"]),
    dict(key="sky", name="Laluan Cawangan Skypark", name_en="Skypark Branch Line",
         ref="", op=KTM, kind="rail", track="ktm", stops="ktm",
         path=["SUBANG JAYA", "TERMINAL SKYPARK"], suspended=True,
         # OSM tags the branch railway=disused since the service stopped, so extract.py does
         # not keep it; its ways come from Overpass (my_sources.md).
         extra="skypark_disused.json"),
    dict(key="pgu", name="Laluan Cawangan Pasir Gudang", name_en="Pasir Gudang Branch Line",
         ref="", op=KTM, kind="rail", track="ktm", stops="ktm",
         path=["KEMPAS BARU", "PASIR GUDANG"]),
    dict(key="ecl", name="Laluan Pantai Timur", name_en="East Coast Line", ref="",
         op=KTM, kind="rail", track="ktm", stops="ktm",
         path=["GEMAS", "BAHAU", "MENTAKAB", "JERANTUT", "KUALA LIPIS", "MERAPOH",
               "GUA MUSANG", "DABONG", "KRAI", "TANAH MERAH", "PASIR MAS", "WAKAF BHARU",
               "TUMPAT"]),
    dict(key="sab", name="Laluan Keretapi Barat Sabah", name_en="Western Sabah Railway Line",
         ref="", op=("Jabatan Keretapi Negeri Sabah", "Sabah State Railway"), kind="rail",
         track="sabah", stops="osm_near", path=["Tanjung Aru", "Papar", "Beaufort", "Halogilat",
                                                "Tenom"]),
    dict(key="kj", name="Laluan Kelana Jaya", name_en="Kelana Jaya Line", ref="KJ", op=RAPID,
         kind="light_rail", track={"Laluan Kelana Jaya", "Kelana Jaya Line"}, rel_ref="KJ",
         stops=("gtfs", "KJ")),
    dict(key="ag", name="Laluan Ampang", name_en="Ampang Line", ref="AG", op=RAPID,
         kind="light_rail", track={"Ampang Line", "Ampang and Sri Petaling Lines",
                                   "Ampang and Sri Petaling lines"}, rel_ref="AG",
         stops=("gtfs", "AG")),
    dict(key="sp", name="Laluan Sri Petaling", name_en="Sri Petaling Line", ref="SP", op=RAPID,
         kind="light_rail", track={"Sri Petaling Line", "Laluan Sri Petaling",
                                   "Ampang and Sri Petaling Lines",
                                   "Ampang and Sri Petaling lines"}, rel_ref="SP",
         stops=("gtfs", "PH")),
    dict(key="kg", name="Laluan Kajang", name_en="Kajang Line", ref="KG", op=RAPID,
         kind="subway", track={"Laluan Kajang"}, rel_ref="9", stops=("gtfs", "KGL")),
    dict(key="py", name="Laluan Putrajaya", name_en="Putrajaya Line", ref="PY", op=RAPID,
         kind="subway", track={"Laluan Putrajaya"}, rel_ref="12", stops=("gtfs", "PYL")),
    dict(key="sa", name="Laluan Shah Alam", name_en="Shah Alam Line", ref="SA", op=RAPID,
         kind="light_rail", track={"LRT 3", "LRT3"}, rel_ref="11", stops=("gtfs", "SA")),
    dict(key="mr", name="Laluan Monorel KL", name_en="KL Monorail", ref="MR", op=RAPID,
         kind="monorail", track={"Monorel KL", "KL Monorail"}, rel_ref="MR",
         stops=("gtfs", "MR")),
    dict(key="erl", name="KLIA Transit", name_en="KLIA Transit", ref="KT",
         op=("Express Rail Link", "Express Rail Link"), kind="rail",
         track={"ERL", "KLIA Transit"}, rel_ref="7", stops=("osm", 8119876)),
    dict(key="phr", name="Keretapi Bukit Bendera", name_en="Penang Hill Railway", ref="",
         op=("Perbadanan Bukit Bendera", "Penang Hill Corporation"), kind="funicular",
         track={"Penang Hill Railway", "Kereta Api Bukit Bendera"}, rel_ref=None,
         stops=("osm", 14425095)),
]
for _l in LINES:
    _l["path"] = _l.get("path", [])

# Timetable stop names (KTM writes them upper case and abbreviated) whose words differ from
# OSM's. Words are mapped one by one (WORD), whole names here.
NAME_ALIAS = {
    "batu kentomenn": "batu kentonmen",
    "perhentian midvalley": "mid valley",
    "pelabuhan klang selatan": "pelabuhan klang",
    "jb sentral": "johor bahru sentral",
    "krambit": "kerambit",
    "padang tungku": "padang tengku",
    "kampung sirian": "kampung sungai serian",
    "sungai sirian": "sungai serian",
    "krai": "kuala krai",
    # KLIA Transit's station and the MRT's terminus are one interchange, Putrajaya Sentral
    "putrajaya cyberjaya": "putrajaya sentral",
    "sungai mengkuang baru": "kampung baru sungai mengkuang",
    "kl sentral redone": "kl sentral",
    "bangsar bank rakyat": "bangsar",
    "kampung baru cbp coopbank pertama": "kampung baru",
    "bandaraya uob": "bandaraya",
    "dato menteri sa sentral": "dato menteri",
    "terminal skypark": "terminal skypark",
}
WORD = {"kg": "kampung", "bdr": "bandar", "sg": "sungai", "jln": "jalan", "telok": "teluk",
        "tanjong": "tanjung", "tasek": "tasik", "sel": "selatan", "pel": "pelabuhan"}
GENERIC = {"stesen", "station", "ktm", "komuter", "keretapi", "lrt", "mrt", "halt", "monorail",
           "erl", "departure", "arrival"}
# Stops in the feeds that are not in Malaysia.
ABROAD = {"woodlands ciq", "hat yai"}

RAIL_MODES = ("train", "subway", "light_rail", "monorail", "tram", "funicular")
STOP_RAILWAY = {"station": 0, "halt": 1, "stop": 2, "tram_stop": 3}
NOT_TRACK_SERVICE = {"yard"}
NOT_PASSENGER = {"industrial", "military", "test", "tourism"}
NOT_KTM = {"ERL", "KLIA Transit", "Laluan Keretapi Barat Sabah", "Sabah State Railway",
           "Port Klang - West Port", "Pelabuhan Klang - West Port",
           "Kerteh–Kuantan Port Railway Line", "Jejak Warisan Kereta Api"}
CODE = re.compile(r"^(?:([A-Z]{1,3})(\d{1,2})([A-Z]?))$")
CODE_PREFIX = re.compile(r"^(?:[A-Z]{1,3}\d{1,2}[A-Z]?\s+)+")

NAME_M = 2500        # a timetable stop's OSM station of the same name is within this ...
FAR_NAME_M = 80000   # ... or, if it is the only place of that name, within this
ANCHOR_M = 400       # a station's node may be this far off its line's track
FOOT_M = 120         # a station's footprint on its track: every vertex this close to the anchor
ON_PATH_M = 450      # a KTM timetable stop this close to a line's path is on that line ...
TIE_M = 100          # ... and on every other line within this of its nearest
NEAR_STOP_M = 150    # osm_near: an OSM station this close to the path is on it
BORDER_SNAP_M = 60   # a border point is on the track this close
STATION_OF_M = 300   # a route's stop node belongs to a station record of its name this close
SAME_PLACE_M = 500   # two lines' stations of one name this close are one station


def line_id(name):
    h = hashlib.blake2b(f"my|{name}".encode("utf-8"), digest_size=5)
    return "y" + h.hexdigest()


def name_key(name):
    n = unicodedata.normalize("NFKC", name or "").casefold()
    n = CODE_PREFIX.sub("", unicodedata.normalize("NFKC", name or "")).casefold() or n
    n = re.sub(r"[’'`.]", "", n)
    n = re.sub(r"[^\w/]+", " ", n).replace("/", " ")
    words = [WORD.get(w, w) for w in n.split() if w not in GENERIC]
    n = " ".join(words)
    return NAME_ALIAS.get(n, n)


def code_key(code):
    m = CODE.match(code or "")
    return f"{m.group(1)}{int(m.group(2))}{m.group(3)}" if m else None


def plain(name):
    """OSM's station name without its leading station codes: "AG7 SP7 KJ13 Masjid Jamek"."""
    return CODE_PREFIX.sub("", name or "").strip() or (name or "")


def title(name):
    """A timetable name as shown, where no OSM station gave one: "KG DATO HARUN" ->
    "Kampung Dato Harun"."""
    words = []
    for w in re.split(r"\s+", name.strip()):
        lw = w.casefold()
        w2 = WORD.get(lw, lw)
        words.append(w2.upper() if w2 in ("kl", "ukm", "upm", "uitm", "ss", "usj", "jb")
                     else w2.capitalize())
    return " ".join(words)


# --------------------------------------------------------------------------- OSM

def load_osm(log):
    import build_model as bm
    ways, rels, stops, cid, cx, cy = bm.load("my", log)
    return ways, rels, stops, bm.Coords(cid, cx, cy)


def rail_stops(stops):
    """Named rail stop records: {node: {name, name_en, lon, lat, rank, codes}}."""
    out = {}
    for nid, (t, lon, lat) in stops.items():
        rw = t.get("railway")
        modes = any(t.get(m) == "yes" for m in RAIL_MODES)
        if not (rw in STOP_RAILWAY or (t.get("public_transport") in ("station", "stop_position")
                                       and modes)):
            continue
        if rw in STOP_RAILWAY and t.get("public_transport") == "stop_position" and not modes \
                and rw not in ("station", "halt"):
            continue
        nm = t.get("name")
        if not nm:
            continue
        codes = set()
        for tok in re.split(r"[\s;,/]+", (t.get("ref") or "") + " " +
                            (CODE_PREFIX.match(nm).group(0) if CODE_PREFIX.match(nm) else "")):
            c = code_key(tok)
            if c:
                codes.add(c)
        out[nid] = {"name": plain(nm), "name_en": plain(t.get("name:en") or ""),
                    "lon": lon, "lat": lat, "rank": STOP_RAILWAY.get(rw, 2),
                    "codes": codes, "modes": {m for m in RAIL_MODES if t.get(m) == "yes"}}
    return out


def track_ways(ways, spec, rels):
    """The OSM ways a line's track is laid over."""
    tr = spec["track"]
    out = []
    if tr == "ktm":
        for wid, (t, nodes) in ways.items():
            if (t.get("railway") == "rail" and t.get("service") not in NOT_TRACK_SERVICE
                    and t.get("usage") not in NOT_PASSENGER
                    and (t.get("name") or "") not in NOT_KTM):
                out.append(wid)
        return out
    if tr == "sabah":
        return [wid for wid, (t, nodes) in ways.items()
                if t.get("railway") in ("rail", "narrow_gauge")
                and t.get("service") not in NOT_TRACK_SERVICE and t.get("usage") != "industrial"]
    names = tr
    got = {wid for wid, (t, nodes) in ways.items()
           if (t.get("name") or "").strip() in names and t.get("service") not in NOT_TRACK_SERVICE}
    # unnamed ways the line's own route relations run over
    ref = spec.get("rel_ref")
    for _rid, (t, members) in rels.items():
        if t.get("type") != "route":
            continue
        if ref is not None and (t.get("ref") or "").strip() != ref:
            continue
        if ref is None and _rid != spec["stops"][1]:
            continue
        for ty, r, _role in members:
            if ty == "w" and r in ways and not ways[r][0].get("name") \
                    and ways[r][0].get("service") not in NOT_TRACK_SERVICE:
                got.add(r)
    return sorted(got)


# --------------------------------------------------------------------------- timetables

def gtfs_rows(zpath, name):
    z = zipfile.ZipFile(zpath)
    return list(csv.DictReader(io.TextIOWrapper(z.open(name), encoding="utf-8-sig")))


def ktm_stops(path, log):
    """KTM's timetable stops trains call at: [{id, name, lon, lat}]."""
    z = Path(path) / "gtfs" / "my_ktmb.gtfs.zip"
    stops = {r["stop_id"]: r for r in gtfs_rows(z, "stops.txt")}
    called = {r["stop_id"] for r in gtfs_rows(z, "stop_times.txt")}
    out = []
    for sid in sorted(called):
        r = stops[sid]
        if name_key(r["stop_name"]) in ABROAD:
            continue
        out.append({"id": sid, "name": r["stop_name"].strip(),
                    "lon": float(r["stop_lon"]), "lat": float(r["stop_lat"])})
    log(f"MY: KTM's timetable calls at {len(out)} stops in Malaysia")
    return out


def prasarana_lines(path):
    """{route_id: [{id, name, lon, lat}, ...]} from the longest trip of each route."""
    z = Path(path) / "gtfs" / "my_prasarana_rail.gtfs.zip"
    stops = {r["stop_id"]: r for r in gtfs_rows(z, "stops.txt")}
    trips = {t["trip_id"]: t["route_id"] for t in gtfs_rows(z, "trips.txt")}
    seq = defaultdict(list)
    for r in gtfs_rows(z, "stop_times.txt"):
        seq[r["trip_id"]].append((int(r["stop_sequence"]), r["stop_id"]))
    best = {}
    for tid, s in seq.items():
        rt = trips.get(tid)
        if rt and (rt not in best or len(s) > len(best[rt])):
            best[rt] = sorted(s)
    return {rt: [{"id": sid, "name": stops[sid]["stop_name"].strip(),
                  "lon": float(stops[sid]["stop_lon"]), "lat": float(stops[sid]["stop_lat"]),
                  "code": code_key(sid)} for _i, sid in s] for rt, s in best.items()}


# --------------------------------------------------------------------------- stations

class StationIndex:
    def __init__(self, st):
        self.st = st
        self.by_name = defaultdict(list)
        self.by_code = defaultdict(list)
        for nid, s in st.items():
            for k in {name_key(s["name"]), name_key(s["name_en"])} - {""}:
                self.by_name[k].append(nid)
            for c in s["codes"]:
                self.by_code[c].append(nid)

    def find(self, name, lon, lat, code=None, near=None):
        """The OSM record of a timetable stop: by its code, else by name within NAME_M, the
        best-ranked then nearest; `near` (lon, lat -> metres) breaks ties towards a track."""
        cands = []
        if code:
            cands = [n for n in self.by_code.get(code, ())
                     if dist_m(lon, lat, self.st[n]["lon"], self.st[n]["lat"]) <= NAME_M]
        if not cands:
            cands = [n for n in self.by_name.get(name_key(name), ())
                     if dist_m(lon, lat, self.st[n]["lon"], self.st[n]["lat"]) <= NAME_M]
        if not cands:
            # KTM's timetable puts some East Coast halts kilometres off (Kuala Gris 11.5 km,
            # Sri Mahligai 26 km): one place of that name further out is taken.
            far = [n for n in self.by_name.get(name_key(name), ())
                   if dist_m(lon, lat, self.st[n]["lon"], self.st[n]["lat"]) <= FAR_NAME_M]
            if far and max(dist_m(self.st[a]["lon"], self.st[a]["lat"], self.st[b]["lon"],
                                  self.st[b]["lat"]) for a in far for b in far) <= 1000:
                cands = far
        if not cands:
            return None
        return min(cands, key=lambda n: (self.st[n]["rank"],
                                         dist_m(lon, lat, self.st[n]["lon"], self.st[n]["lat"])))


# --------------------------------------------------------------------------- geometry

class Track:
    """One line's track graph, with station anchoring and shortest paths."""

    def __init__(self, way_ids, ways, coords):
        self.adj, self.xy, _fast = line_graph(way_ids, ways, coords)
        self.near = Near(self.xy) if self.xy else None

    def add(self, extra):
        """Track the extract lacks: {"ways": {id: [node]}, "nodes": {id: [lon, lat]}}."""
        xy = {int(k): tuple(v) for k, v in extra["nodes"].items()}
        for nodes in extra["ways"].values():
            for a, b in zip(nodes[:-1], nodes[1:]):
                if a not in xy or b not in xy:
                    continue
                self.xy.setdefault(a, xy[a])
                self.xy.setdefault(b, xy[b])
                w = dist_m(*xy[a], *xy[b]) / 1000
                self.adj[a].append((b, w))
                self.adj[b].append((a, w))
        self.near = Near(self.xy)

    def anchor(self, lon, lat, lim=ANCHOR_M):
        v, d = self.near.nearest(lon, lat)
        return (v, d) if d <= lim else (None, d)

    def foot(self, v):
        x, y = self.xy[v]
        ids, ds = self.near.within(x, y, FOOT_M)
        return {i: d / 1000 for i, d in zip(ids, ds)}

    def path(self, va, vb):
        got = between(self.adj, self.foot(va), self.foot(vb), set())
        if got is None:
            return None
        nodes, km = got
        return [self.xy[va]] + [self.xy[n] for n in nodes] + [self.xy[vb]], km


def cum_m(pts):
    out = [0.0]
    for (x1, y1), (x2, y2) in zip(pts[:-1], pts[1:]):
        out.append(out[-1] + dist_m(x1, y1, x2, y2))
    return out


class Polyline:
    """A path, for projecting points onto it and slicing it."""

    def __init__(self, pts):
        from shapely.geometry import LineString
        self.pts = pts
        self.lat0 = float(np.mean([p[1] for p in pts]))
        self.kx = math.cos(math.radians(self.lat0)) * 111320
        self.ky = 110570
        self.line = LineString([(x * self.kx, y * self.ky) for x, y in pts])
        self.cum = np.array([0.0] + list(np.cumsum(
            [math.hypot((x2 - x1) * self.kx, (y2 - y1) * self.ky)
             for (x1, y1), (x2, y2) in zip(pts[:-1], pts[1:])])))

    def locate(self, lon, lat):
        """(metres along, metres off)."""
        from shapely.geometry import Point
        p = Point(lon * self.kx, lat * self.ky)
        return self.line.project(p), self.line.distance(p)

    def slice(self, a, b):
        """The path between a and b metres along, as [(lon, lat)], a <= b."""
        i = int(np.searchsorted(self.cum, a, side="right"))
        j = int(np.searchsorted(self.cum, b, side="left"))
        out = [self.at(a)] + [tuple(self.pts[k]) for k in range(i, j)] + [self.at(b)]
        return out

    def at(self, m):
        k = int(np.clip(np.searchsorted(self.cum, m) - 1, 0, len(self.pts) - 2))
        seg = self.cum[k + 1] - self.cum[k]
        t = 0.0 if seg <= 0 else (m - self.cum[k]) / seg
        (x1, y1), (x2, y2) = self.pts[k], self.pts[k + 1]
        return (x1 + (x2 - x1) * t, y1 + (y2 - y1) * t)


# --------------------------------------------------------------------------- build

def border_points(log):
    pts = {}
    try:
        import borders
        for p in borders.load(canonical_only=True):
            if "my" in p["countries"]:
                pts[p["id"]] = {"lon": p["lon"], "lat": p["lat"], "name": p["name"]}
    except Exception as e:                            # noqa: BLE001 - builds without borders
        log(f"MY: no border table ({e})")
    for pid, (lon, lat, ccs) in MY_BORDER.items():
        if pid not in pts:
            other = [c for c in ccs if c != "my"][0]
            pts[pid] = {"lon": lon, "lat": lat,
                        "name": "Malaysia – " + {"sg": "Singapore", "th": "Thailand"}[other]
                        + " border"}
    return pts


def build(path, log):
    from n02 import walk_order
    ways, rels, stops, coords = load_osm(log)
    st = rail_stops(stops)
    idx = StationIndex(st)
    log(f"MY: {len(st)} named OSM rail stop records")
    ktm = ktm_stops(path, log)
    pra = prasarana_lines(path)
    bpts = border_points(log)

    stations, lines, geoms = {}, [], {}
    problems = []

    same_place = defaultdict(list)          # name key -> station ids made so far

    def station_rec(sid, name, name_en, lon, lat, junction=False):
        """One record per station complex: KTM's, the LRT's and KLIA Transit's KL Sentral are
        one station, as are Masjid Jamek's three lines (same name within SAME_PLACE_M)."""
        if not junction and sid not in stations:
            for other in same_place[name_key(name)]:
                o = stations[other]
                if dist_m(lon, lat, o["lon"], o["lat"]) <= SAME_PLACE_M:
                    return other
            same_place[name_key(name)].append(sid)
        if sid not in stations:
            stations[sid] = {"id": sid, "name": name, "name_en": name_en, "lon": lon,
                             "lat": lat, "lines": set()}
            if junction:
                stations[sid]["junction"] = True
        return sid

    def osm_station(nid):
        s = st[nid]
        sid = f"y{nid}" if nid > 0 else f"ya{-nid}"
        en = s["name_en"] if s["name_en"] and s["name_en"] != s["name"] else ""
        return station_rec(sid, s["name"], en, s["lon"], s["lat"])

    def station_of(nid):
        """A stop node's station record, where one of a compatible name is within
        STATION_OF_M: "KL Sentral (ERL) Departure" is KL Sentral."""
        s = st[nid]
        if s["rank"] == 0:
            return nid
        words = set(name_key(s["name"]).split())
        best = None
        for n, t in st.items():
            if t["rank"] != 0:
                continue
            d = dist_m(s["lon"], s["lat"], t["lon"], t["lat"])
            w = set(name_key(t["name"]).split())
            if d <= STATION_OF_M and w and (w <= words or words <= w) \
                    and (best is None or d < best[0]):
                best = (d, n)
        return best[1] if best else nid

    def stop_station(rec, track=None, code=None):
        """A timetable stop -> station id, at the OSM record of its name if there is one."""
        nid = idx.find(rec["name"], rec["lon"], rec["lat"], code=code)
        if nid is not None:
            return osm_station(nid)
        problems.append(("no OSM station", rec["name"]))
        return station_rec(f"yg{rec['id']}", title(rec["name"]), "", rec["lon"], rec["lat"])

    # KTM's stops, each once: station id and point
    ktm_st = []
    for r in ktm:
        sid = stop_station(r)
        ktm_st.append((sid, r))
    ktm_by_key = {}
    for sid, r in ktm_st:
        ktm_by_key[name_key(r["name"])] = sid
        ktm_by_key[name_key(stations[sid]["name"])] = sid

    def waypoint(w):
        """A waypoint -> (station id, lon, lat, is border)."""
        if w in bpts:
            p = bpts[w]
            station_rec(w, p["name"], "", p["lon"], p["lat"], junction=True)
            return w, p["lon"], p["lat"], True
        sid = ktm_by_key.get(name_key(w))
        if sid is None:
            nids = idx.by_name.get(name_key(w), [])
            if nids:
                nid = min(nids, key=lambda n: st[n]["rank"])
                sid = osm_station(nid)
        if sid is None:
            raise SystemExit(f"MY: waypoint {w!r} found no station")
        s = stations[sid]
        return sid, s["lon"], s["lat"], False

    tracks = {}

    def track_for(spec):
        k = spec["track"] if isinstance(spec["track"], str) else spec["key"]
        if spec.get("extra"):
            k = f"{k}+{spec['extra']}"
        if k not in tracks:
            wids = track_ways(ways, spec, rels)
            if spec["track"] == "sabah":
                wids = [w for w in wids if coords.get(ways[w][1][0]) is not None
                        and coords.get(ways[w][1][0])[0] > 109]
            elif spec["track"] == "ktm":
                wids = [w for w in wids if coords.get(ways[w][1][0]) is None
                        or coords.get(ways[w][1][0])[0] < 109]
            tracks[k] = Track(wids, ways, coords)
            if spec.get("extra"):
                extra = json.loads((Path(path) / spec["extra"]).read_text(encoding="utf-8"))
                tracks[k].add(extra)
                log(f"MY: {spec['name_en']}: {len(extra['ways'])} ways from {spec['extra']}")
            log(f"MY: track {k}: {len(wids)} ways, {len(tracks[k].xy)} vertices")
        return tracks[k]

    # --- lines laid along a waypoint path: their path first (stations need every path)
    paths = {}
    for spec in LINES:
        if not spec["path"]:
            continue
        tr = track_for(spec)
        wps = [waypoint(w) for w in spec["path"]]
        pts, km_total = [], 0.0
        for (a, ax, ay, ab), (b, bx, by, bb) in zip(wps[:-1], wps[1:]):
            va, da = tr.anchor(ax, ay, BORDER_SNAP_M if ab else ANCHOR_M)
            vb, db = tr.anchor(bx, by, BORDER_SNAP_M if bb else ANCHOR_M)
            if va is None or vb is None:
                raise SystemExit(f"MY: {spec['name_en']}: waypoint {a if va is None else b} is "
                                 f"{da if va is None else db:.0f} m off its track")
            got = tr.path(va, vb)
            if got is None:
                raise SystemExit(f"MY: {spec['name_en']}: no track from "
                                 f"{stations[a]['name']} to {stations[b]['name']}")
            seg, km = got
            crow = dist_m(ax, ay, bx, by) / 1000
            if km > 1.6 * crow + 3:
                problems.append(("winding path", f"{spec['name_en']}: {stations[a]['name']} - "
                                 f"{stations[b]['name']} {km:.1f} km for {crow:.1f} crow-fly"))
            pts += seg if not pts else seg[1:]
            km_total += km
        paths[spec["key"]] = (Polyline(pts), wps)
        log(f"MY: {spec['name_en']}: path {km_total:.1f} km through {len(wps)} waypoints")

    # --- KTM stops onto the KTM lines whose path passes them
    on_line = defaultdict(list)                 # line key -> [(along m, station id)]
    ktm_keys = [s["key"] for s in LINES if s.get("stops") == "ktm"]
    for sid, r in ktm_st:
        s = stations[sid]
        got = []
        for k in ktm_keys:
            pl, _w = paths[k]
            along, off = pl.locate(s["lon"], s["lat"])
            # the timetable's own point too: OSM's record may be the station building
            along2, off2 = pl.locate(r["lon"], r["lat"])
            if off2 < off:
                along, off = along2, off2
            got.append((off, k, along))
        best = min(g[0] for g in got)
        if best > ON_PATH_M:
            problems.append(("stop on no line", f"{s['name']} ({best:.0f} m from the nearest)"))
            continue
        for off, k, along in got:
            if off <= ON_PATH_M and off <= best + TIE_M:
                on_line[k].append((along, sid))
    # Waypoints are on their line whatever the distance rule says (a branch's first station).
    for spec in LINES:
        if spec.get("stops") != "ktm" and spec.get("stops") != "osm_near":
            continue
        pl, wps = paths[spec["key"]]
        have = {sid for _a, sid in on_line[spec["key"]]}
        for sid, lon, lat, _b in wps:
            if sid not in have:
                on_line[spec["key"]].append((pl.locate(lon, lat)[0], sid))
                have.add(sid)

    # --- OSM stations along the Sabah line
    for spec in LINES:
        if spec.get("stops") != "osm_near":
            continue
        pl, _w = paths[spec["key"]]
        have = {sid for _a, sid in on_line[spec["key"]]}
        have_names = {name_key(stations[s]["name"]) for s in have}
        x0, y0, x1, y1 = (min(p[0] for p in pl.pts), min(p[1] for p in pl.pts),
                          max(p[0] for p in pl.pts), max(p[1] for p in pl.pts))
        for nid, s in st.items():
            if not (x0 - 0.01 <= s["lon"] <= x1 + 0.01 and y0 - 0.01 <= s["lat"] <= y1 + 0.01):
                continue
            along, off = pl.locate(s["lon"], s["lat"])
            if off > NEAR_STOP_M or along <= 1 or along >= pl.cum[-1] - 1:
                continue
            k = name_key(s["name"])
            twin = [n for n in idx.by_name.get(k, ()) if dist_m(
                s["lon"], s["lat"], st[n]["lon"], st[n]["lat"]) <= 500]
            if k in have_names or min(twin, key=lambda n: (st[n]["rank"], n)) != nid:
                continue                    # the station node and its stop position: one
            sid = osm_station(nid)
            on_line[spec["key"]].append((along, sid))
            have.add(sid)
            have_names.add(k)

    def emit(spec, sections, junction_served=()):
        lid = line_id(spec["name"])
        secs = []
        g = {}
        for a, b, km, pts in sections:
            secs.append([a, b, round(km, 3)])
            g[f"{a}|{b}"] = [[round(x, 5), round(y, 5)] for x, y in pts]
            for s in (a, b):
                stations[s]["lines"].add(lid)
        line = {"id": lid, "src": "my", "service": False,
                "name": spec["name"], "name_en": spec["name_en"], "ref": spec["ref"],
                "colour": "", "operator": spec["op"][0], "operator_en": spec["op"][1],
                "network": "", "kind": spec["kind"],
                "km": round(sum(s[2] for s in secs), 3), "variants": 1,
                "straight_sections": 0,
                "display": walk_order([(a, b) for a, b, _k in secs]),
                "sections": secs}
        if spec.get("suspended"):
            line["suspended"] = True
        if junction_served:
            line["served_sections"] = sorted(junction_served)
        lines.append(line)
        geoms[lid] = g
        log(f"  MY: {spec['name_en']:<28} {line['km']:8.2f} km {len(secs):3d} sections "
            f"{len({s for x in secs for s in x[:2]}):3d} stations ({spec['kind']})")

    # --- path lines: sections between neighbours along the path
    for spec in LINES:
        if not spec["path"]:
            continue
        pl, wps = paths[spec["key"]]
        order = sorted(on_line[spec["key"]])
        merged = []
        for along, sid in order:
            if merged and merged[-1][1] == sid:
                continue
            if merged and along - merged[-1][0] < 30:
                problems.append(("two stations at one place",
                                 f"{spec['name_en']}: {stations[merged[-1][1]]['name']} / "
                                 f"{stations[sid]['name']}"))
                continue
            merged.append((along, sid))
        secs, served = [], []
        for (a_m, a), (b_m, b) in zip(merged[:-1], merged[1:]):
            secs.append((a, b, (b_m - a_m) / 1000, pl.slice(a_m, b_m)))
            if stations[a].get("junction") or stations[b].get("junction"):
                served.append(f"{a}|{b}")
        emit(spec, secs, served)

    # --- listed lines: shortest path between neighbouring stations
    for spec in LINES:
        if spec["path"]:
            continue
        tr = track_for(spec)
        kind, src = spec["stops"]
        recs = []
        if kind == "gtfs":
            for r in pra[src]:
                nid = idx.find(r["name"], r["lon"], r["lat"], code=r["code"])
                if nid is None:
                    problems.append(("no OSM station", f"{spec['name_en']}: {r['id']} {r['name']}"))
                    sid = station_rec(f"yg{r['id']}", title(r["name"]), "", r["lon"], r["lat"])
                else:
                    sid = osm_station(nid)
                recs.append(sid)
        else:
            t, members = rels[src]
            for ty, ref, role in members:
                if ty != "n" or ref not in stops:
                    continue
                tags, lon, lat = stops[ref]
                nid = ref if ref in st else idx.find(tags.get("name") or "", lon, lat)
                if nid is None:
                    problems.append(("relation stop with no station", f"{spec['name_en']}: {ref}"))
                    continue
                nid = station_of(nid)
                sid = osm_station(nid)
                if sid not in recs:
                    recs.append(sid)
            # in the order the track passes them (Penang Hill's relation lists its stops out
            # of order): along the path between the two stops furthest apart
            if len(recs) > 2:
                a, b = max(((x, y) for x in recs for y in recs),
                           key=lambda p: dist_m(stations[p[0]]["lon"], stations[p[0]]["lat"],
                                                stations[p[1]]["lon"], stations[p[1]]["lat"]))
                va = tr.anchor(stations[a]["lon"], stations[a]["lat"])[0]
                vb = tr.anchor(stations[b]["lon"], stations[b]["lat"])[0]
                got = tr.path(va, vb) if va is not None and vb is not None else None
                if got is not None:
                    pl = Polyline(got[0])
                    recs.sort(key=lambda s: pl.locate(stations[s]["lon"], stations[s]["lat"])[0])
        secs = []
        for a, b in zip(recs[:-1], recs[1:]):
            sa, sb = stations[a], stations[b]
            va, da = tr.anchor(sa["lon"], sa["lat"])
            vb, db = tr.anchor(sb["lon"], sb["lat"])
            if va is None or vb is None:
                problems.append(("station off its track", f"{spec['name_en']}: "
                                 f"{sa['name'] if va is None else sb['name']} "
                                 f"{da if va is None else db:.0f} m"))
                continue
            got = tr.path(va, vb)
            if got is None:
                problems.append(("no track between", f"{spec['name_en']}: {sa['name']} - {sb['name']}"))
                continue
            pts, km = got
            crow = dist_m(sa["lon"], sa["lat"], sb["lon"], sb["lat"]) / 1000
            if km > 1.6 * crow + 1:
                problems.append(("winding section", f"{spec['name_en']}: {sa['name']} - "
                                 f"{sb['name']} {km:.2f} km for {crow:.2f}"))
            secs.append((a, b, km, pts))
        emit(spec, secs)

    used = {s for l in lines for sec in l["sections"] for s in sec[:2]}
    stations = {k: v for k, v in stations.items() if k in used}
    total = sum(l["km"] for l in lines)
    log(f"MY: {len(lines)} register lines, {total:,.1f} km, {len(stations)} stations")
    for kind, n in Counter(p[0] for p in problems).items():
        log(f"MY: {n} x {kind}:")
        for p in problems:
            if p[0] == kind:
                log(f"    {p[1]}")
    return lines, stations, geoms


# --------------------------------------------------------------------------- clip

def foreign_area():
    """Singapore and Thailand near the rail crossings, as one shapely geometry."""
    from shapely.geometry import Point, Polygon, box, shape
    from shapely.ops import polygonize, unary_union
    from sg_register import SG_POLY
    boxes = {"causeway": (103.60, 1.40, 104.10, 1.50),
             "padang_besar": (100.15, 6.55, 100.45, 6.75),
             "rantau_panjang": (101.90, 5.95, 102.05, 6.10)}
    # a point on each side, to tell the faces apart
    foreign_pt = {"causeway": (103.80, 1.42), "padang_besar": (100.33, 6.72),
                  "rantau_panjang": (101.966, 6.04)}
    home_pt = {"causeway": (103.76, 1.47), "padang_besar": (100.32, 6.62),
               "rantau_panjang": (101.99, 6.00)}
    fc = json.loads((ROOT / "data" / "raw" / "my" / "borders_osm.geojson")
                    .read_text(encoding="utf-8"))
    foreign, home = [Polygon(SG_POLY)], []
    for name, (w, s, e, n) in boxes.items():
        bx = box(w, s, e, n)
        lines = [shape(f["geometry"]).intersection(bx) for f in fc["features"]
                 if f["properties"]["box"] == name]
        faces = list(polygonize(unary_union(lines + [bx.exterior])))
        for f in faces:
            if f.contains(Point(*foreign_pt[name])):
                foreign.append(f)
            if f.contains(Point(*home_pt[name])):
                home.append(f)
    return unary_union(foreign).difference(unary_union(home)) if home else unary_union(foreign)


def clip(log=print):
    from shapely import contains_xy, prepare
    area = foreign_area()
    prepare(area)
    d = PROC
    ways = pickle.load(open(d / "ways.pkl", "rb"))
    rels = pickle.load(open(d / "rels.pkl", "rb"))
    stops = pickle.load(open(d / "stops.pkl", "rb"))
    infra_p = d / "infra.pkl"
    infra = pickle.load(open(infra_p, "rb")) if infra_p.exists() else None
    c = np.load(d / "coords.npz")
    nid, x, y = c["id"], c["x"] / 1e7, c["y"] / 1e7
    outside = set(nid[~contains_xy(area, x, y)].tolist())
    n0 = (len(ways), len(stops), len(rels))
    gone = Counter(t.get("name") for t, n in ways.values()
                   if not any(int(i) in outside for i in n))
    ways = {k: v for k, v in ways.items() if any(int(i) in outside for i in v[1])}
    stops = {k: v for k, v in stops.items() if not contains_xy(area, v[1], v[2])}
    kept = {("w", k) for k in ways} | {("n", k) for k in stops}
    routes = {k for k, (tags, members) in rels.items()
              if tags.get("type") == "route" and any((t, r) in kept for t, r, _ in members)}
    rels = {k: v for k, v in rels.items()
            if k in routes or any(t == "r" and r in routes for t, r, _ in v[1])}
    out = [("ways", ways), ("rels", rels), ("stops", stops)]
    if infra is not None:
        infra = {k: v for k, v in infra.items()
                 if any((t, r) in kept for t, r, _ in v[1])}
        out.append(("infra", infra))
    for name, obj in out:
        tmp = d / f"{name}.pkl.tmp"
        with open(tmp, "wb") as f:
            pickle.dump(obj, f, protocol=4)
        os.replace(tmp, d / f"{name}.pkl")
    log(f"clipped to Malaysia: ways {n0[0]} -> {len(ways)}, stops {n0[1]} -> {len(stops)}, "
        f"relations {n0[2]} -> {len(rels)}; ways dropped by name: "
        + ", ".join(f"{k} {v}" for k, v in gone.most_common(25)))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", action="store_true",
                    help="drop Singapore and Thailand from data/proc/my")
    args = ap.parse_args()
    if args.clip:
        clip()
    else:
        ap.print_help()
