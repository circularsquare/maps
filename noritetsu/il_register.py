"""Lines, stations and sections for Israel, with OpenStreetMap's track as the geometry.

    python extract.py --region il --pbf data/raw/israel-and-palestine-latest.osm.pbf --station-areas
    python il_register.py --clip          # drop OSM's stale train routes and the unopened Nofit line
    python build_model.py --region il --register il_register:data/raw/il
    python build_tiles.py --region il
    python check_model.py --region il

Malaysia's recipe (my_register.py, whose Track and Polyline this reuses): the timetable's
stops laid on OSM track by shortest path. The timetable is the Ministry of Transport's
national GTFS feed (data/raw/il/gtfs/il_rail.gtfs.zip, the rail, light-rail and Carmelit
subset of the Mobility Database's mirror mdb-2519; il_sources.md).

  ISRAEL RAILWAYS' lines are its INFRASTRUCTURE LINES as he.wikipedia and en.wikipedia name
  them (Coastal, Ayalon, Tel Aviv - Lod, Tel Aviv - Jerusalem (A1), Lod - Ashkelon...). IR's
  service patterns (Nahariya - Beersheba and the like) are renumbered often and OSM's
  numbered train relations are years out of date, so they are not lines: `--clip` drops them.
  Each line is the shortest path over IR's running track through the WAYPOINTS in LINES.
  A waypoint written "^name" (first) or "name$" (last) is a station on track an earlier line
  already has: the new line's path is cut where it leaves the earlier line's path, and that
  point becomes a junction both lines share. So no piece of track is in two lines. Stations
  are every stop IR's trips call at in the feed (stop_times; the stop code says nothing, see
  il_sources.md), each on the lines whose path passes within ON_PATH_M, at the OSM station
  node nearest it.

  LIGHT RAIL and the CARMELIT take their stations in order from the feed's longest trip per
  line; each section is the shortest path between neighbouring stations over the city's
  light-rail (or the Carmelit's funicular) track.

WHO THE MAP SHOWS (Anita, 2026-10-08): "if israel administers it we can draw it under israel".
The Jerusalem light rail's Red Line through East Jerusalem and the A1's few km through the
West Bank near Mevo Horon are drawn whole in `il`. religiondots' `il` outline already holds
East Jerusalem; the A1's stretch is in its `ps`, but ownership.py only gives a neighbour ways
no register line owns, and the A1 owns its own.

The `path` argument is data/raw/il; the OSM half is read from data/proc/il (extract.py).
"""
import argparse
import csv
import hashlib
import io
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

from kr_register import dist_m
from my_register import Polyline, Track

ROOT = Path(__file__).resolve().parent
PROC = ROOT / "data" / "proc" / "il"
GTFS = "gtfs/il_rail.gtfs.zip"

IR = ("רכבת ישראל", "Israel Railways")
CFIR = ("כפיר", "CFIR")                      # Jerusalem light rail operator
TAVEL = ("תבל", "Tavel")                      # Tel Aviv Red Line operator (Dan Tavel)
CARMELIT = ("כרמלית", "Carmelit")

# IR's infrastructure lines. Waypoints are the feed's stop names (Hebrew, as in stops.txt), or
# for stations no train calls at (the suspended Malha line), OSM's station name. Order matters:
# a "^" or "$" waypoint cuts against the paths of the lines above it.
LINES = [
    dict(key="coastal", name="מסילת החוף", name_en="Coastal Railway",
         path=["נהריה", "עכו", "קרית מוצקין", "קרית חיים", "מרכזית המפרץ/קו החוף",
               "חיפה מרכז", "בת גלים", "חוף הכרמל", "עתלית", "בנימינה", "קיסריה פרדס חנה",
               "חדרה מערב", "נתניה", "בית יהושע", "הרצליה", "תל אביב האוניברסיטה - אקספו"]),
    dict(key="ayalon", name="מסילת איילון", name_en="Ayalon Railway",
         path=["תל אביב האוניברסיטה - אקספו", "תל אביב מרכז", "השלום", "תל אביב ההגנה"]),
    dict(key="tlvlod", name="מסילת תל אביב–לוד", name_en="Tel Aviv - Lod Railway",
         path=["תל אביב ההגנה", "כפר חבד", "לוד גני אביב", "לוד"]),
    dict(key="a1", name="מסילת תל אביב–ירושלים", name_en="Tel Aviv - Jerusalem Railway (A1)",
         path=["^תל אביב ההגנה", "נתב''ג", "ירושלים/יצחק נבון"]),
    dict(key="modiin", name="מסילת ענבה–מודיעין", name_en="Anava - Modi'in Railway",
         path=["^נתב''ג", "פאתי מודיעין", "מודיעין מרכז"]),
    dict(key="jaffajlm", name="מסילת יפו–ירושלים", name_en="Jaffa - Jerusalem Railway",
         path=["^לוד", "רמלה", "בית שמש"]),
    # Beit Shemesh - Jerusalem Malha: no train since March 2020 and IR does not mean to bring
    # them back (en.wikipedia "Jaffa–Jerusalem railway"); the feed has no trip there. Greyed.
    dict(key="malha", name="מסילת יפו–ירושלים: בית שמש–ירושלים מלחה",
         name_en="Jaffa - Jerusalem Railway: Beit Shemesh - Jerusalem Malha",
         path=["^בית שמש", "גן החיות התנ\"כי", "ירושלים מלחה"], suspended=True,
         # OSM tags most of it railway=disused, which extract.py does not keep: its ways come
         # from Overpass (il_sources.md).
         extra="malha_disused.json"),
    dict(key="lodash", name="מסילת לוד–אשקלון", name_en="Lod - Ashkelon Railway",
         path=["^לוד", "באר יעקב", "רחובות", "יבנה מזרח", "אשדוד עד הלום- מטרופול",
               "אשקלון"]),
    dict(key="rishonim", name="מסילת ראשונים", name_en="Rishonim Branch",
         path=["^לוד", "ראשונים"]),
    dict(key="batyam", name="מסילת בת ים–אשדוד", name_en="Bat Yam - Ashdod Railway",
         path=["^תל אביב ההגנה", "צומת חולון", "חולון וולפסון", "בת ים אלי כהן - יוספטל",
               "בת ים קוממיות", "רשל''צ משה דיין", "יבנה מערב", "אשדוד עד הלום- מטרופול$"]),
    dict(key="south", name="המסילה לבאר שבע", name_en="Railway to Beersheba",
         path=["^רמלה", "מזכרת בתיה", "קרית מלאכי", "קרית גת", "להבים רהט", "באר שבע צפון",
               "באר שבע מרכז"]),
    dict(key="ashbs", name="מסילת אשקלון–באר שבע", name_en="Ashkelon - Beersheba Railway",
         path=["^אשקלון", "שדרות", "נתיבות", "אופקים", "באר שבע צפון$"]),
    dict(key="dimona", name="מסילת באר שבע–דימונה", name_en="Beersheba - Dimona Railway",
         path=["^באר שבע צפון", "דימונה"]),
    dict(key="kfarsaba", name="מסילת תל אביב–כפר סבא", name_en="Tel Aviv - Kfar Saba Railway",
         path=["^תל אביב האוניברסיטה - אקספו", "בני ברק", "קרית אריה", "סגולה",
               "ראש העין צפון", "כפר סבא"]),
    dict(key="sharon", name="מסילת השרון", name_en="Sharon Railway",
         path=["^כפר סבא", "הוד השרון", "רעננה דרום", "רעננה מערב", "הרצליה$"]),
    dict(key="eastern", name="המסילה המזרחית", name_en="Eastern Railway",
         path=["^ראש העין צפון", "טירה-כוכב יאיר", "שומרון-טייבה", "חדרה מזרח"]),
    dict(key="jezreel", name="מסילת העמק", name_en="Jezreel Valley Railway",
         # leaves the Coastal Railway between Haifa Center and HaMifrats Central, whose
         # Jezreel Valley platforms (the feed's stop 17123, OSM's own station node) are on it
         path=["^חיפה מרכז", "מרכזית המפרץ", "כפר יהושע", "כפר ברוך", "עפולה",
               "בית שאן - דוד לוי"]),
    dict(key="karmiel", name="מסילת עכו–כרמיאל", name_en="Acre - Karmiel Railway",
         path=["^קרית מוצקין", "אחיהוד", "כרמיאל"]),
]
for _l in LINES:
    _l.update(op=IR, kind="rail", ref="")
# Feed stops that are on one line only, whatever track passes near: HaMifrats Central's
# Jezreel Valley platforms lie 400 m from the Coastal Railway's.
ONLY = {"מרכזית המפרץ": "jezreel"}
# Junction names where the line's own article names the place: {(line key, "start"/"end"):
# (Hebrew, English)}. Others are "<line> junction".
JUNCTIONS = {}

# Light rail and the Carmelit: (route agency, route short name, route_type) in the feed.
URBAN = [
    dict(key="jlmred", name="הרכבת הקלה בירושלים – הקו האדום", name_en="Jerusalem Light Rail Red Line",
         ref="R", op=CFIR, kind="light_rail", gtfs=("21", "1", "0"), box=(35.10, 31.70, 35.30, 31.90)),
    dict(key="jlmgreen", name="הרכבת הקלה בירושלים – הקו הירוק",
         name_en="Jerusalem Light Rail Green Line", ref="G", op=CFIR, kind="light_rail",
         gtfs=("21", "3", "0"), box=(35.10, 31.70, 35.30, 31.90)),
    dict(key="tlvred", name="הקו האדום", name_en="Tel Aviv Light Rail Red Line", ref="R",
         op=TAVEL, kind="light_rail", gtfs=("22", "1", "0"), also=("2", "3"),
         box=(34.70, 31.95, 34.95, 32.15)),
    dict(key="carmelit", name="כרמלית", name_en="Carmelit", ref="", op=CARMELIT,
         kind="funicular", gtfs=("20", "1", "5"), box=(34.95, 32.78, 35.03, 32.84)),
]

GENERIC = {"תחנת", "רכבת", "station", "railway"}
NOT_TRACK_SERVICE = {"yard", "siding"}
NOT_PASSENGER = {"industrial", "military", "test", "tourism"}
NOT_LRT_NAMES = ("נופית", "Nofit", "הקו הכתום")

IR_STATION_M = 700   # an IR stop's OSM station node is the nearest within this
URBAN_STOP_M = 250   # a light-rail stop's OSM record of a matching name is within this
ANCHOR_M = 400       # a station may be this far off its line's track
ON_PATH_M = 450      # an IR stop this close to a line's path is on that line ...
TIE_M = 100          # ... and on every other line within this of its nearest
TRIM_M = 25          # a branch's path this close to an earlier line's path is on that line
END_GUARD_M = 60     # a stop is not put at a cut end of a branch unless this close to it
AT_STATION_M = 300   # a cut this close to the waypoint station is at the station
GAP_M = 400          # a branch back on an earlier line's path within this is still on it


def line_id(name):
    h = hashlib.blake2b(f"il|{name}".encode("utf-8"), digest_size=5)
    return "q" + h.hexdigest()


def name_key(name):
    n = unicodedata.normalize("NFKC", name or "").casefold()
    n = re.sub(r"[\"'`׳״’.]", "", n)
    n = re.sub(r"[^\w]+", " ", n)
    return " ".join(w for w in n.split() if w not in GENERIC)


# --------------------------------------------------------------------------- feed

def feed(path):
    z = zipfile.ZipFile(Path(path) / GTFS)

    def rows(n):
        return list(csv.DictReader(io.TextIOWrapper(z.open(n), encoding="utf-8-sig")))
    en = {r["trans_id"]: r["translation"] for r in rows("translations.txt")
          if r["lang"].upper() == "EN"}
    stops = {r["stop_id"]: r for r in rows("stops.txt")}
    routes = {r["route_id"]: r for r in rows("routes.txt")}
    trips = {r["trip_id"]: r for r in rows("trips.txt")}
    seq = defaultdict(list)
    for r in rows("stop_times.txt"):
        seq[r["trip_id"]].append((int(r["stop_sequence"]), r["stop_id"]))
    return en, stops, routes, trips, {t: [s for _i, s in sorted(v)] for t, v in seq.items()}


# --------------------------------------------------------------------------- OSM

def load_osm(log):
    import build_model as bm
    ways, rels, stops, cid, cx, cy = bm.load("il", log)
    return ways, rels, stops, bm.Coords(cid, cx, cy)


def osm_stations(stops):
    """Named OSM stop records: {node: {name, name_en, lon, lat, rail, urban}}."""
    out = {}
    for nid, (t, lon, lat) in stops.items():
        nm = t.get("name")
        if not nm:
            continue
        rw = t.get("railway")
        urban = (t.get("station") in ("light_rail", "funicular", "subway", "monorail")
                 or t.get("light_rail") == "yes" or rw == "tram_stop" or t.get("funicular") == "yes")
        rail = rw in ("station", "halt") and not urban
        if not (rail or urban or rw == "stop" or t.get("public_transport") == "station"):
            continue
        out[nid] = {"name": nm.strip(), "name_en": (t.get("name:en") or "").strip(),
                    "lon": lon, "lat": lat, "rail": rail, "urban": urban,
                    "rank": 0 if rw == "station" else 1}
    return out


def in_box(coords, nodes, box):
    p = coords.get(nodes[0])
    if p is None:
        return False
    w, s, e, n = box
    return w <= p[0] <= e and s <= p[1] <= n


def rail_ways(ways):
    return [wid for wid, (t, nodes) in ways.items()
            if t.get("railway") == "rail" and t.get("service") not in NOT_TRACK_SERVICE
            and t.get("usage") not in NOT_PASSENGER]


def urban_ways(ways, coords, spec):
    kind = "funicular" if spec["kind"] == "funicular" else "light_rail"
    return [wid for wid, (t, nodes) in ways.items()
            if t.get("railway") == kind and t.get("service") not in NOT_TRACK_SERVICE
            and not any(x in (t.get("name") or "") for x in NOT_LRT_NAMES)
            and in_box(coords, nodes, spec["box"])]


# --------------------------------------------------------------------------- build

def build(path, log):
    from n02 import walk_order
    en, fstops, routes, trips, seq = feed(path)
    ways, rels, stops, coords = load_osm(log)
    st = osm_stations(stops)
    log(f"IL: {len(st)} named OSM stop records")

    stations, lines, geoms, problems = {}, [], {}, []

    def station_rec(sid, name, name_en, lon, lat, junction=False):
        if sid not in stations:
            stations[sid] = {"id": sid, "name": name, "name_en": name_en if name_en != name else "",
                             "lon": lon, "lat": lat, "lines": set()}
            if junction:
                stations[sid]["junction"] = True
        return sid

    def osm_station(nid):
        s = st[nid]
        return station_rec(f"q{nid}", s["name"], s["name_en"], s["lon"], s["lat"])

    def feed_station(r, prefix="qg"):
        nm = r["stop_name"].strip()
        return station_rec(f"{prefix}{r['stop_id']}", nm, en.get(nm, ""),
                           float(r["stop_lon"]), float(r["stop_lat"]))

    # --- IR: every stop an IR trip calls at, at its nearest OSM railway station
    ir_called = Counter()
    for t, x in trips.items():
        if routes[x["route_id"]]["route_type"] == "2":
            for s in seq.get(t, ()):
                ir_called[s] += 1
    rail_nodes = [n for n, s in st.items() if s["rail"]]
    ir_st = {}                                   # feed stop name -> (station id, feed row)
    for sid in ir_called:
        r = fstops[sid]
        lon, lat = float(r["stop_lon"]), float(r["stop_lat"])
        near = min(rail_nodes, key=lambda n: dist_m(lon, lat, st[n]["lon"], st[n]["lat"]))
        d = dist_m(lon, lat, st[near]["lon"], st[near]["lat"])
        if d <= IR_STATION_M:
            ir_st[r["stop_name"].strip()] = (osm_station(near), r)
            if not stations[f"q{near}"]["name_en"] and en.get(r["stop_name"].strip()):
                stations[f"q{near}"]["name_en"] = en[r["stop_name"].strip()]
        else:
            problems.append(("no OSM station", f"{r['stop_name']} ({d:.0f} m to the nearest)"))
            ir_st[r["stop_name"].strip()] = (feed_station(r), r)
    log(f"IL: IR trips call at {len(ir_called)} stops, {len({v[0] for v in ir_st.values()})} "
        f"stations")

    def waypoint(w):
        if w in ir_st:
            return ir_st[w][0]
        cands = [n for n, s in st.items() if s["rail"] and name_key(s["name"]) == name_key(w)]
        if not cands:
            raise SystemExit(f"IL: waypoint {w!r} found no station")
        return osm_station(cands[0])

    # --- IR lines: paths through the waypoints, cut where they leave earlier paths
    tr = Track(rail_ways(ways), ways, coords)
    log(f"IL: IR track: {len(tr.xy)} vertices")
    paths = {}                     # key -> Polyline
    on_line = defaultdict(list)    # key -> [(along m, station id)]
    cut_ends = defaultdict(set)    # key -> {"start", "end"}
    jn = 0

    tr_main = tr
    for spec in LINES:
        tr = tr_main
        if spec.get("extra"):
            import json
            tr = Track(rail_ways(ways), ways, coords)
            extra = json.loads((Path(path) / spec["extra"]).read_text(encoding="utf-8"))
            tr.add(extra)
            log(f"IL: {spec['name_en']}: {len(extra['ways'])} ways from {spec['extra']}")
        raw = spec["path"]
        trim_start = raw[0].startswith("^")
        trim_end = raw[-1].endswith("$")
        names = [w.lstrip("^").rstrip("$") for w in raw]
        wps = [waypoint(w) for w in names]
        pts = []
        for a, b in zip(wps[:-1], wps[1:]):
            sa, sb = stations[a], stations[b]
            va, da = tr.anchor(sa["lon"], sa["lat"], ANCHOR_M)
            vb, db = tr.anchor(sb["lon"], sb["lat"], ANCHOR_M)
            if va is None or vb is None:
                raise SystemExit(f"IL: {spec['name_en']}: {sa['name'] if va is None else sb['name']}"
                                 f" is {da if va is None else db:.0f} m off the track")
            got = tr.path(va, vb)
            if got is None:
                raise SystemExit(f"IL: {spec['name_en']}: no track {sa['name']} - {sb['name']}")
            seg, km = got
            crow = dist_m(sa["lon"], sa["lat"], sb["lon"], sb["lat"]) / 1000
            if km > 1.6 * crow + 3:
                problems.append(("winding path", f"{spec['name_en']}: {sa['name']} - "
                                 f"{sb['name']} {km:.1f} km for {crow:.1f}"))
            pts += seg if not pts else seg[1:]
        # cut against earlier lines' paths
        earlier = [(k, pl) for k, pl in paths.items()]

        def on_earlier(p):
            best = None
            for k, pl in earlier:
                along, off = pl.locate(*p)
                if off <= TRIM_M and (best is None or off < best[2]):
                    best = (k, along, off)
            return best
        for end in ("start", "end"):
            if (end == "start" and not trim_start) or (end == "end" and not trim_end):
                continue
            seq_pts = pts if end == "start" else pts[::-1]
            # the last point on an earlier line before the path leaves it for good: gaps
            # shorter than GAP_M (platform loops, four-track stretches) do not end it
            i, hit, gap = 0, None, 0.0
            for k, p in enumerate(seq_pts):
                h = on_earlier(p)
                if h:
                    i, hit, gap = k + 1, h, 0.0
                    continue
                if k:
                    gap += dist_m(*seq_pts[k - 1], *p)
                if gap > GAP_M or k == 0:
                    break
            if i == 0 or hit is None:
                problems.append(("no cut", f"{spec['name_en']} {end}: path does not start on "
                                 f"an earlier line"))
                continue
            if i >= len(seq_pts):
                raise SystemExit(f"IL: {spec['name_en']} lies wholly on earlier lines")
            j = seq_pts[i - 1]
            parent, along_p, _off = hit
            ws = stations[wps[0] if end == "start" else wps[-1]]
            if dist_m(j[0], j[1], ws["lon"], ws["lat"]) <= AT_STATION_M:
                # the line leaves the earlier one at the station itself (Lod, Beit Shemesh)
                jid = ws["id"]
            else:
                jn += 1
                jname = JUNCTIONS.get((spec["key"], end), (f"צומת {spec['name']}",
                                                           f"{spec['name_en']} junction"))
                jid = station_rec(f"qj{spec['key']}{end[0]}", jname[0], jname[1], j[0], j[1],
                                  junction=True)
            on_line[parent].append((paths[parent].locate(*j)[0], jid))
            kept = seq_pts[i - 1:]
            pts = kept if end == "start" else kept[::-1]
            cut_ends[spec["key"]].add(end)
            if end == "start":
                wps = [jid] + wps[1:]
            else:
                wps = wps[:-1] + [jid]
        pl = Polyline(pts)
        paths[spec["key"]] = pl
        for w in wps:
            on_line[spec["key"]].append((pl.locate(stations[w]["lon"], stations[w]["lat"])[0], w))
        log(f"IL: {spec['name_en']}: path {pl.cum[-1] / 1000:.1f} km")

    # --- IR stops onto the lines whose path passes them
    wp_of = {spec["key"]: {s for _a, s in on_line[spec["key"]]} for spec in LINES}
    only = {ir_st[n][0]: k for n, k in ONLY.items() if n in ir_st}
    for sid in sorted({v[0] for v in ir_st.values()}):
        s = stations[sid]
        got = []
        for spec in LINES:
            k = spec["key"]
            if sid in only and only[sid] != k:
                continue
            pl = paths[k]
            along, off = pl.locate(s["lon"], s["lat"])
            ends = cut_ends[k]
            if off > END_GUARD_M and (("start" in ends and along < END_GUARD_M)
                                      or ("end" in ends and along > pl.cum[-1] - END_GUARD_M)):
                continue                      # beside a cut end: on the earlier line, not here
            got.append((off, k, along))
        if not got:
            continue
        best = min(g[0] for g in got)
        if best > ON_PATH_M:
            problems.append(("stop on no line", f"{s['name']} ({best:.0f} m)"))
            continue
        for off, k, along in got:
            if off <= ON_PATH_M and off <= best + TIE_M and sid not in wp_of[k]:
                on_line[k].append((along, sid))

    def emit(spec, sections, served=()):
        lid = line_id(spec["name"])
        secs, g = [], {}
        for a, b, km, pts in sections:
            secs.append([a, b, round(km, 3)])
            g[f"{a}|{b}"] = [[round(x, 6), round(y, 6)] for x, y in pts]
            stations[a]["lines"].add(lid)
            stations[b]["lines"].add(lid)
        line = {"id": lid, "src": "il", "service": False, "name": spec["name"],
                "name_en": spec["name_en"], "ref": spec["ref"], "colour": "",
                "operator": spec["op"][0], "operator_en": spec["op"][1], "network": "",
                "kind": spec["kind"], "km": round(sum(s[2] for s in secs), 3), "variants": 1,
                "straight_sections": 0, "display": walk_order([(a, b) for a, b, _k in secs]),
                "sections": secs}
        if spec.get("suspended"):
            line["suspended"] = True
        if served:
            line["served_sections"] = sorted(served)
        lines.append(line)
        geoms[lid] = g
        log(f"  IL: {spec['name_en']:<46} {line['km']:8.2f} km {len(secs):3d} sections "
            f"{len({s for x in secs for s in x[:2]}):3d} stations")

    for spec in LINES:
        pl = paths[spec["key"]]
        merged = []
        for along, sid in sorted(on_line[spec["key"]]):
            if merged and merged[-1][1] == sid:
                continue
            if merged and along - merged[-1][0] < 30:
                problems.append(("two stations at one place", f"{spec['name_en']}: "
                                 f"{stations[merged[-1][1]]['name']} / {stations[sid]['name']}"))
                continue
            merged.append((along, sid))
        secs, served = [], []
        for (a_m, a), (b_m, b) in zip(merged[:-1], merged[1:]):
            secs.append((a, b, (b_m - a_m) / 1000, pl.slice(a_m, b_m)))
            if not spec.get("suspended") and (stations[a].get("junction")
                                              or stations[b].get("junction")):
                served.append(f"{a}|{b}")
        emit(spec, secs, served)

    # --- light rail and the Carmelit: the feed's longest trip, shortest paths between stops
    for spec in URBAN:
        ag, short, rtype = spec["gtfs"]
        rids = {rid for rid, r in routes.items() if r["agency_id"] == ag
                and r["route_short_name"] == short and r["route_type"] == rtype}
        # every stopping pattern of the line, longest first: the Tel Aviv Red Line's Kiryat
        # Arye branch is only in its routes 2 and 3
        ag_rids = {rid for rid, r in routes.items() if r["agency_id"] == ag
                   and r["route_short_name"] in spec.get("also", ()) and r["route_type"] == rtype}
        pats = sorted({tuple(seq.get(t, ())) for t, x in trips.items()
                       if x["route_id"] in rids | ag_rids}, key=len, reverse=True)
        utr = Track(urban_ways(ways, coords, spec), ways, coords)
        cand = [n for n, s in st.items() if s["urban"] or spec["kind"] == "funicular"]
        by_name = {}                          # one record per stop name on the line

        def stop_rec(sid):
            r = fstops[sid]
            k = name_key(r["stop_name"])
            if k in by_name:
                return by_name[k]
            lon, lat = float(r["stop_lon"]), float(r["stop_lat"])
            near = [n for n in cand if dist_m(lon, lat, st[n]["lon"], st[n]["lat"]) <= URBAN_STOP_M
                    and (name_key(st[n]["name"]) == k or k in name_key(st[n]["name"])
                         or name_key(st[n]["name"]) in k)]
            if near:
                # the station node if there is one, else the nearest stop node of the name
                n = min(near, key=lambda n: (st[n]["rank"], dist_m(lon, lat, st[n]["lon"],
                                                                    st[n]["lat"])))
                x = osm_station(n)
            else:
                problems.append(("no OSM stop", f"{spec['name_en']}: {r['stop_name']}"))
                x = feed_station(r, "qu")
            by_name[k] = x
            return x
        pairs, seen = [], set()
        for pat in pats:
            recs = [stop_rec(sid) for sid in pat]
            recs = [x for i, x in enumerate(recs) if i == 0 or recs[i - 1] != x]
            for a, b in zip(recs[:-1], recs[1:]):
                if frozenset((a, b)) not in seen:
                    seen.add(frozenset((a, b)))
                    pairs.append((a, b))
        secs = []
        for a, b in pairs:
            sa, sb = stations[a], stations[b]
            va, da = utr.anchor(sa["lon"], sa["lat"], ANCHOR_M)
            vb, db = utr.anchor(sb["lon"], sb["lat"], ANCHOR_M)
            if va is None or vb is None:
                problems.append(("station off its track", f"{spec['name_en']}: "
                                 f"{sa['name'] if va is None else sb['name']}"))
                continue
            got = utr.path(va, vb)
            if got is None:
                problems.append(("no track between", f"{spec['name_en']}: {sa['name']} - "
                                 f"{sb['name']}"))
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
    log(f"IL: {len(lines)} register lines, {sum(l['km'] for l in lines):,.1f} km, "
        f"{len(stations)} stations, {jn} junctions")
    for kind, n in Counter(p[0] for p in problems).items():
        log(f"IL: {n} x {kind}:")
        for p in problems:
            if p[0] == kind:
                log(f"    {p[1]}")
    return lines, stations, geoms


# --------------------------------------------------------------------------- clip

def clip(log=print):
    """Drop from data/proc/il: OSM's Israel Railways train routes (numbered by IR's old
    scheme, one still to Jerusalem Malha: the register has IR's lines) and every other route
    relation (the light-rail ones are the register's lines again, one mislabelled "Yellow
    Line"; Haifa's Metronit is a bus); and the track of lines that do not run: the Haifa -
    Nazareth Nofit light rail (being built, tagged light_rail, not in the feed), Tel Aviv's
    Orange Line stub, narrow gauge and monorail (amusement and museum track)."""
    d = PROC
    ways = pickle.load(open(d / "ways.pkl", "rb"))
    rels = pickle.load(open(d / "rels.pkl", "rb"))
    n0 = (len(ways), len(rels))
    gone = Counter()
    keep = {}
    for k, (t, nodes) in ways.items():
        nm = t.get("name") or ""
        if t.get("railway") in ("narrow_gauge", "monorail") or any(x in nm for x in NOT_LRT_NAMES):
            gone[f"{t.get('railway')} {nm}"] += 1
            continue
        keep[k] = (t, nodes)
    rels = {k: v for k, v in rels.items()
            if v[0].get("type") not in ("route", "route_master")}
    for name, obj in (("ways", keep), ("rels", rels)):
        tmp = d / f"{name}.pkl.tmp"
        with open(tmp, "wb") as f:
            pickle.dump(obj, f, protocol=4)
        os.replace(tmp, d / f"{name}.pkl")
    log(f"IL clip: ways {n0[0]} -> {len(keep)}, relations {n0[1]} -> {len(rels)}; dropped: "
        + ", ".join(f"{k} {v}" for k, v in gone.most_common()))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", action="store_true",
                    help="drop OSM's train routes and track that does not run from data/proc/il")
    args = ap.parse_args()
    if args.clip:
        clip()
    else:
        ap.print_help()
