"""Australia: register lines from Geoscience Australia's Foundation Rail Infrastructure.

    python au_register.py --fetch     # writes data/raw/au/ga_rail_lines.geojson
    python au_register.py --ga        # the register half alone, no OSM: lines, km, path checks
    python au_register.py --ga --all  # ... and every line
    python build_model.py --region au --register au_register:data/raw/au/ga_rail_lines.geojson

au_sources.md is the record: numbers, checks, what is off.

SOURCE. Foundation Rail Infrastructure, Geoscience Australia (CC BY 4.0), layer 1
Railway_Lines of
    https://services.ga.gov.au/gis/rest/services/Foundation_Rail_Infrastructure/MapServer
a national aggregation of each state's rail centre lines (NSW Spatial Services, Queensland's
DNRME, SA's DPTI, Victoria's DELWP, Land Tasmania) with Geoscience Australia's own for WA and
the NT ("National"). Railways (featuresubtype 90015) and sidings (90016) are fetched, every
status; tramlines (90017) are not.

WHAT THE REGISTER GIVES. Line segments, each with a NAME (the state's line name: "MAIN
SOUTHERN RAILWAY", "NORTH COAST LINE", "PORT AUGUSTA - WA BORDER", "TRANS AUSTRALIAN
RAILWAY"), OPERATIONAL_STATUS, OWNER (for some states), TRACK_GAUGE, TRACKS and LENGTH_KM.
NO ROUTENAME or SECTIONNAME (multi_sources.md had those; the service has neither), NO
passenger flag, NO stations on the lines (layer 0 is a separate point list) and NO topology:
segments are not tied to numbered nodes, so the graph is made here from the geometry
(`topology`). GA draws EVERY TRACK of a multiple-track line, not a centre line.

WHAT IS KEPT. Railways whose status is Operational (or "Fully capable of operation."), except
tramways and metros mapped as railway (NOT_RAIL: Glenelg, Sydney Metro, the Epping -
Chatswood line, which stay OSM lines as metros do everywhere) and heritage railways
(HERITAGE, by name: Anita's rule counts service more often than about weekly; Puffing Billy,
daily, is kept). Sidings and non-operational track come back only where an OSM passenger route
runs over them away from kept track (`accept_holes`: Springhurst's main line is a "siding").

WHICH TRACK IS PASSENGER TRACK: OpenStreetMap's. The register has no flag, and most of
Australia's 36,000 km of operational railway is freight only (the Pilbara, Queensland's coal
systems, the grain lines). A passenger train route relation (`is_passenger_route`: a network or
operator that runs scheduled trains, not the tourist excursions) decides, as the USA's holes
are judged: stations go on the line their own routes' track lies NEAREST (`place_stations`),
sections between them prefer track a route runs over (`weighted_sections`), any section less
than UNROUTED_SHARE covered by a route is left out before build_model sees it, and dead-end
track no route runs over is peeled off (`peel`). A line no route runs over is not built.

THE LINE UNIT is the register's NAME within its state (`group`): "Main Southern Railway",
"North Coast Line". One name in two states is two lines (Sydney's and Brisbane's Airport
Line); pieces of one name more than JOIN_KM apart are lines of their own (us_register's
group_lines, with the state as the owner). South Australia's track names ("Adelaide Station
Sidetrack 31") are read as unnamed (TRACK_NAME). Unnamed track (UNKNOWN) beside a named line is
its other track (`adopt_beside`: NSW names one pair of a main line's tracks); a run between two
points of one line joins it; a run under OSM routes touching no one line is a line of its own,
named from OSM (`attach_unnamed`); the rest is left out.

GA DRAWS EVERY TRACK, so after the sections are found: junction-ended sections along a kept
section are another track and dropped (`dedup_sections`), junctions only one line uses are
fused through (`fuse_junctions`), a line lying along a longer one is its `companion_of`
(`find_companions`: South Australia names the two tracks of Adelaide's lines apart), and lines
with no stop of their own that are companions or under NO_STOP_LINE_KM are dropped.

NAMES are the register's, in title case ("Main Southern Railway", "Port Augusta - WA
Border"). Names are English, so name_en is empty unless two lines share a name (then it adds
their two furthest stops).

STATIONS, SECTIONS AND GEOMETRY are otherwise us_register's: absorbing Dijkstra between stops,
line ends and branch points (junction stations "aj<node>"), sections drawn on OSM track traced
in a corridor round GA's geometry (`trace`: tried again in WIDE_CORRIDOR_M where GA's NT and WA
geometry lies further off). GA's own LENGTH_KM over a section is its `chain`.

The `path` argument is the geojson; the OSM half is read from data/proc/au (extract.py).
"""
import hashlib
import heapq
import json
import math
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np

import us_register as U

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw" / "au"
PROC = ROOT / "data" / "proc" / "au"
GA = ("https://services.ga.gov.au/gis/rest/services/Foundation_Rail_Infrastructure/"
      "MapServer/1/query")
# Railways (90015) and rail sidings (90016), every operational status; tramlines (90017) are
# not fetched (trams and light rail stay OSM lines).
WHERE = "featuresubtype IN (90015, 90016)"
PAGE = 2000
USER_AGENT = "noritetsu-rail-map/1.0"
OUT = "ga_rail_lines.geojson"
INF = float("inf")

RAILWAY, SIDING = 90015, 90016
OPERATIONAL = {"Operational", "Fully capable of operation."}

# ---------------------------------------------------------------- what is kept

# Names that are not a railway line: tramways and metros the register files as railway (they
# stay OSM lines), and heritage railways (Anita: service more often than about weekly).
# The Epping - Chatswood railway has been Sydney Metro Northwest's track since 2019.
NOT_RAIL = re.compile(r"\bTRAM\b|\bMETRO\b|^EPPING CHATSWOOD RAILWAY$")
HERITAGE = re.compile(
    r"TOURIST|HERITAGE|HISTORIC|MUSEUM|STEAM|PICHI RICHI|ZIG ZAG|"
    r"WILDERNESS RAILWAY|GOLDFIELDS RAILWAY|SPA COUNTRY|BELLARINE|IDA BAY|MARY VALLEY|"
    r"MURRAY RIVER RAILWAY|STEAMRANGER")
NOT_A_NAME = {"", "UNKNOWN", "UNNAMED", "N/A", "NONE", "PRIVATE RAILWAY", "SPUR"}
# Names of a piece of track rather than a line: South Australia's file names every siding,
# yard road and station track ("Adelaide Station Sidetrack 31", "Dry Creek Freight Yard 05",
# "Keswick Passenger Terminal 03", "Goodwood - Adelaide 02", "Blackwood Second Track 01").
# They are read as unnamed track, so they join the line they lie between or are left out.
TRACK_NAME = re.compile(
    r"SIDE?TRACK|SIODETRACK|PASSENGERSIDETRACK|\bYARD\b|\bDEPOT\b|\bWORKSHOPS?\b|"
    r"PASSENGER TERMINAL|PASSENGER AND FREIGHT TERMINAL|FREIGHT TERMINAL|TURNSTYLE|"
    r"\bMIDDLE TRACK\b|\bSECOND TRACK\b|\bRAIL TRACK\b|\bYARD TRACK\b|\bDOCK TRACK\b|"
    r"\bFLAT TRACK\b|\bWORKSHOP TRACK\b|\bUNLOADER\b|\bCAVAN TRACK\b|\s\d{2}\.?$|\d{2}$")

# Pieces of one name in one state closer than this are one line (us_register.group_lines).
JOIN_KM = 30.0

# ---------------------------------------------------------------- topology

SNAP_M = 2.0           # segment ends this close are one node
TEE_M = 3.0            # a segment end this close to another segment's middle joins it there
BRIDGE_M = 60.0        # a loose end this close to another piece's end is joined to it (state
                       # files meet at the border without a shared vertex)

# ---------------------------------------------------------------- sections

UNROUTED_SHARE = 0.25  # a section less covered than this by OSM routes is left out
OFF_ROUTE_COST = 4.0   # sections prefer track a route runs over: other track costs this x


def dist_m(lon1, lat1, lon2, lat2):
    return U.dist_m(lon1, lat1, lon2, lat2)


# ================================================================ fetching

def _get(params):
    url = GA + "?" + urllib.parse.urlencode(params)
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=180) as r:
                d = json.loads(r.read().decode("utf-8"))
            if "error" in d:
                raise RuntimeError(d["error"])
            return d
        except Exception as e:          # a dropped page is retried, then the fetch fails
            if attempt == 4:
                raise
            print(f"  retry after {e}", flush=True)
            time.sleep(5 * (attempt + 1))


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    feats, offset = [], 0
    while True:
        page = _get({"where": WHERE, "outFields": "*", "f": "geojson", "outSR": 4326,
                     "orderByFields": "objectid", "resultOffset": offset,
                     "resultRecordCount": PAGE})
        got = page.get("features") or []
        feats += got
        print(f"  {len(feats):,} features", flush=True)
        if len(got) < PAGE and not page.get("exceededTransferLimit"):
            break
        if not got:
            break
        offset += len(got)
    out = RAW / OUT
    tmp = out.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "fetched": time.strftime("%Y-%m-%d"),
                   "source": GA, "where": WHERE, "features": feats}, f)
    tmp.replace(out)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(feats):,} features)")


# ================================================================ reading

def props(f):
    return {k.lower(): v for k, v in (f.get("properties") or {}).items()}


def clean_key(name):
    n = re.sub(r"\s+", " ", (name or "").strip().upper())
    return KEY_ALIAS.get(n, n)


# One line under two spellings in the register (cleaned upper-case keys).
KEY_ALIAS = {
    "NORTH EASTERN": "NORTH EASTERN RAILWAY",
    "MORINGTON TOURIST RAILWAY": "MORNINGTON TOURIST RAILWAY",
}
# Names the register spells as nobody would (cleaned key -> name).
NAME_FIX = {}
KEEP_UPPER = {"WA", "SA", "NSW", "QLD", "NT", "SG", "BG", "ARTC", "BHP", "CBD", "II"}
SMALL = {"AND": "and", "OF": "of", "THE": "the", "TO": "to"}


def title(word):
    if word in KEEP_UPPER:
        return word
    if word in SMALL:
        return SMALL[word]
    if re.fullmatch(r"MC[A-Z]+", word):
        return "Mc" + word[2:].capitalize()
    return "/".join("-".join(w.capitalize() for w in part.split("-"))
                    for part in word.split("/"))


def display_name(key):
    if key in NAME_FIX:
        return NAME_FIX[key]
    n = re.sub(r"\s*\(([A-Z])\)\s*$", r" (\1)", key)
    words = [title(w) for w in n.split(" ")]
    if words:
        words[0] = words[0][:1].upper() + words[0][1:]
    return " ".join(words)


def read_ga(path, log):
    """The kept segments, the candidates (sidings, non-operational track) and what was left
    out by why. A segment is one part of one feature."""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    kept, other = [], []
    left = defaultdict(float)
    left_by = defaultdict(float)
    for f in d["features"]:
        p, g = props(f), f.get("geometry")
        if not g:
            continue
        sub = int(p.get("featuresubtype") or 0)
        status = (p.get("operational_status") or "").strip()
        raw = (p.get("name") or "").strip()
        key = clean_key(raw)
        if key in NOT_A_NAME or TRACK_NAME.search(key):
            key = ""
        km = float(p.get("length_km") or 0.0)
        why = None
        if sub == SIDING:
            why = "siding"
        elif status not in OPERATIONAL:
            why = f"status {status or 'none'}"
        elif NOT_RAIL.search(key):
            why = "tram or metro"
        elif HERITAGE.search(key) or HERITAGE.search((p.get("owner") or "").upper()):
            why = "heritage"
        if g["type"] == "LineString":
            parts = [g["coordinates"]]
        elif g["type"] == "MultiLineString":
            parts = g["coordinates"]
        else:
            continue
        parts = [[tuple(c[:2]) for c in part] for part in parts if len(part) >= 2]
        geo = [U.path_m(pp) / 1000 for pp in parts]
        tot = sum(geo) or 1e-9
        oid = int(p.get("objectid"))
        for i, (pp, gk) in enumerate(zip(parts, geo)):
            s = {"id": oid * 100 + i, "oid": oid, "pts": pp, "geo_km": gk,
                 "km": km * gk / tot if km else gk,
                 "owner": (p.get("source_jurisdiction") or "").strip(),
                 "infra": (p.get("owner") or "").strip(),
                 "state": (p.get("source_jurisdiction") or "").strip(),
                 "raw": raw, "key": key, "field": "name", "code": status,
                 "tracks": p.get("tracks") or "", "gauge": p.get("track_gauge") or "",
                 "rights": [], "hole": False, "why": why}
            (other if why else kept).append(s)
        if why:
            left[why] += km
            left_by[(why, (p.get("source_jurisdiction") or ""), raw)] += km
    log(f"GA: {len(kept)} operational railway segments kept, "
        f"{sum(s['km'] for s in kept):,.0f} km; left out: "
        + ", ".join(f"{w} {km:,.0f} km" for w, km in sorted(left.items(), key=lambda x: -x[1])))
    return kept, other, left, left_by


# ================================================================ topology

class Grid:
    """Points in a hash grid of cell `r` metres, for near-point questions."""

    def __init__(self, r):
        self.r = r
        self.cells = defaultdict(list)

    def _cell(self, lon, lat):
        return (int(math.floor(lon * 111320 * math.cos(math.radians(lat)) / self.r)),
                int(math.floor(lat * 110570 / self.r)))

    def add(self, i, lon, lat):
        self.cells[self._cell(lon, lat)].append((i, lon, lat))

    def near(self, lon, lat, r):
        cx, cy = self._cell(lon, lat)
        k = int(math.ceil(r / self.r))
        for dx in range(-k, k + 1):
            for dy in range(-k, k + 1):
                for i, x, y in self.cells.get((cx + dx, cy + dy), ()):
                    if dist_m(lon, lat, x, y) <= r:
                        yield i


def topology(segs, log):
    """Node ids for every segment end ("a", "b"), made from the geometry.

    1. Ends within SNAP_M are one node.
    2. An end within TEE_M of another segment's middle (further than TEE_M from that
       segment's ends) cuts that segment there: a branch leaving a line mid-segment.
    3. A loose end (one segment there) within BRIDGE_M of an end of a piece it is not yet
       connected to is joined to it: the state files meet at the border with no shared
       vertex (Albury, Serviceton, Wallangarra) and some junctions are drawn a few metres
       apart. Logged."""
    import shapely
    from shapely import STRtree
    from shapely.geometry import LineString
    # --- 2. tees first, so a cut point becomes an end like any other
    geo = [LineString(U.merc([p[0] for p in s["pts"]], [p[1] for p in s["pts"]]))
           for s in segs]
    tree = STRtree(geo)
    cuts = defaultdict(list)
    for i, s in enumerate(segs):
        for end in (s["pts"][0], s["pts"][-1]):
            sc = U.merc_scale(end[1])
            P = shapely.points(U.merc([end[0]], [end[1]]))[0]
            for j in tree.query(P, predicate="dwithin", distance=TEE_M * sc).tolist():
                if j == i:
                    continue
                t = geo[j].project(P, normalized=True)
                L = segs[j]["geo_km"] * 1000
                if t * L <= TEE_M or (1 - t) * L <= TEE_M:
                    continue
                cuts[j].append(t)
    out = []
    n_cut = 0
    for i, s in enumerate(segs):
        ts = sorted(set(round(t, 7) for t in cuts.get(i, ())))
        if not ts:
            out.append(s)
            continue
        pieces = U.split_line(s["pts"], ts)
        n_cut += 1
        for k, pp in enumerate(pieces):
            gk = U.path_m(pp) / 1000
            q = dict(s)
            q.update({"id": s["id"] * 1000 + k, "pts": pp, "geo_km": gk,
                      "km": s["km"] * gk / (s["geo_km"] or 1e-9)})
            out.append(q)
    segs = out
    # --- 1. snap ends
    uf = U.UF()
    grid = Grid(max(SNAP_M, 1.0) * 4)
    ends = []
    for i, s in enumerate(segs):
        for e, pt in (("a", s["pts"][0]), ("b", s["pts"][-1])):
            k = len(ends)
            ends.append((i, e, pt))
            for j in grid.near(pt[0], pt[1], SNAP_M):
                uf.union(j, k)
            grid.add(k, pt[0], pt[1])
    # --- 3. bridges between loose ends of pieces not yet connected
    root = [uf.find(k) for k in range(len(ends))]
    deg = Counter(root)
    seg_uf = U.UF()
    for i, s in enumerate(segs):
        seg_uf.union(("n", root[2 * i]), ("n", root[2 * i + 1]))
    big = Grid(BRIDGE_M)
    for k, (_i, _e, pt) in enumerate(ends):
        big.add(k, pt[0], pt[1])
    bridges = []
    # Loose end to loose end first, nearest pairs first, each end once: two pieces of one
    # line that stop short of each other (Peterborough - Broken Hill at the state border, 1 m)
    # must find each other before either is tied to a siding beside them.
    piece0 = {k: seg_uf.find(("n", root[k])) for k in range(len(ends))}
    pairs = []
    for k, (i, e, pt) in enumerate(ends):
        if deg[root[k]] != 1:
            continue
        for j in big.near(pt[0], pt[1], BRIDGE_M):
            if j > k and deg[root[j]] == 1 and piece0[j] != piece0[k]:
                pairs.append((dist_m(pt[0], pt[1], *ends[j][2]), k, j))
    used = set()
    for d, k, j in sorted(pairs):
        if k in used or j in used:
            continue
        if seg_uf.find(("n", root[j])) == seg_uf.find(("n", root[k])):
            continue
        used.update((k, j))
        seg_uf.union(("n", root[j]), ("n", root[k]))
        uf.union(j, k)
        bridges.append((d, segs[ends[k][0]]["state"], segs[ends[j][0]]["state"],
                        segs[ends[k][0]]["raw"], segs[ends[j][0]]["raw"], ends[k][2]))
    # then a loose end left over to the nearest end of a piece it is not yet connected to
    for k, (i, e, pt) in enumerate(ends):
        if deg[root[k]] != 1 or k in used:
            continue
        best = None
        for j in big.near(pt[0], pt[1], BRIDGE_M):
            if seg_uf.find(("n", root[j])) == seg_uf.find(("n", root[k])):
                continue
            d = dist_m(pt[0], pt[1], *ends[j][2])
            if best is None or d < best[0]:
                best = (d, j)
        if best is not None:
            d, j = best
            seg_uf.union(("n", root[j]), ("n", root[k]))
            uf.union(j, k)
            bridges.append((d, segs[i]["state"], segs[ends[j][0]]["state"], segs[i]["raw"],
                            segs[ends[j][0]]["raw"], pt))
    node = {}
    for k, (i, e, pt) in enumerate(ends):
        r = uf.find(k)
        node.setdefault(r, len(node) + 1)
        segs[i][e] = node[r]
    log(f"topology: {len(segs)} segments ({n_cut} cut at a tee), {len(node)} nodes; "
        f"{len(bridges)} loose ends bridged to another piece within {BRIDGE_M:.0f} m")
    for d, sa, sb, na, nb, pt in sorted(bridges, key=lambda x: -x[0])[:25]:
        log(f"    bridge {d:5.1f} m  {sa} {na or '-'} / {sb} {nb or '-'}  "
            f"({pt[1]:.4f}, {pt[0]:.4f})")
    return segs


# ================================================================ lines

def attach_unnamed(lines, unnamed, log, covered=None):
    """Unnamed track joins the named line it runs between: each run of unnamed segments
    (between nodes where the run meets named track or branches) whose ends both touch one
    named line becomes part of that line (its loop or second track). A run between two
    lines, or with a loose end, is left out, unless OSM routes run over it (`covered`: seg
    id -> share), when it becomes a line of its own with no name (named from OSM later)."""
    covered = covered or {}
    at_node = defaultdict(set)
    for i, l in enumerate(lines):
        for s in l["segs"]:
            at_node[s["a"]].add(i)
            at_node[s["b"]].add(i)
    # runs of unnamed track: components after cutting at every node that touches named
    # track or where three or more unnamed segments meet
    deg = Counter()
    for s in unnamed:
        deg[s["a"]] += 1
        deg[s["b"]] += 1
    stop = {n for n in deg if n in at_node or deg[n] != 2}
    uf = U.UF()
    for s in unnamed:
        uf.find(("s", s["id"]))
    by_node = defaultdict(list)
    for s in unnamed:
        for n in (s["a"], s["b"]):
            if n not in stop:
                by_node[n].append(s["id"])
    for n, ids in by_node.items():
        for x in ids[1:]:
            uf.union(("s", ids[0]), ("s", x))
    runs = defaultdict(list)
    for s in unnamed:
        runs[uf.find(("s", s["id"]))].append(s)
    joined = own = left = 0.0
    new_lines = []
    for run in runs.values():
        ends = Counter()
        for s in run:
            ends[s["a"]] += 1
            ends[s["b"]] += 1
        tips = [n for n, c in ends.items() if c == 1]
        km = sum(s["km"] for s in run)
        common = None
        if len(tips) == 2:
            common = at_node.get(tips[0], set()) & at_node.get(tips[1], set())
        if common:
            best = max(common, key=lambda i: sum(s["km"] for s in lines[i]["segs"]))
            lines[best]["segs"].extend(run)
            joined += km
            continue
        cov = (sum(covered.get(s["id"], 0.0) * s["km"] for s in run) / km) if km else 0.0
        if cov >= 0.5 and km >= 0.3:
            new_lines.append(run)
            own += km
        else:
            left += km
    # unnamed runs that a route runs over: touching ones are one line, if it is long enough
    # to be a line rather than a connecting curve or a station throat (UNNAMED_MIN_KM)
    if new_lines:
        flat = [s for r in new_lines for s in r]
        for comp in U.components(flat):
            km = sum(s["km"] for s in comp)
            if km < UNNAMED_MIN_KM:
                own -= km
                left += km
                continue
            st = Counter(s["state"] for s in comp).most_common(1)[0][0]
            lines.append({"key": "", "field": "", "owner": st, "segs": comp, "pieces": 1,
                          "split": False})
    log(f"GA: unnamed track: {joined:,.1f} km joined the line it runs between, "
        f"{own:,.1f} km under OSM routes became lines of their own, {left:,.1f} km left out "
        f"(between two lines, or a siding)")


UNNAMED_MIN_KM = 2.0  # unnamed track under routes, touching no one line, is a line from this
BESIDE_M = 40.0      # an unnamed segment lying this close to a named line (median) is its track


def adopt_beside(segs, log):
    """Unnamed track lying beside a named line takes its name: NSW files only one pair of
    tracks of a multiple-track main line under the line's name (the Main Suburban Railway's
    six tracks out of Sydney, the Main Northern's four to Hornsby, the Illawarra's) and the
    rest as UNKNOWN, joined to each other by crossovers, so no run of it ends on the named
    line at both ends. A segment whose points lie a median BESIDE_M or less from one named
    line, and nearer it than any other, is that line's track."""
    import shapely
    from shapely import STRtree
    named = [s for s in segs if s["key"]]
    geo = [shapely.LineString(U.merc([p[0] for p in s["pts"]], [p[1] for p in s["pts"]]))
           for s in named]
    tree = STRtree(geo)
    n_took, km_took = 0, 0.0
    for s in segs:
        if s["key"]:
            continue
        a = U.densify(np.asarray(s["pts"], dtype=np.float64), 50.0)
        sc = U.merc_scale(float(a[:, 1].mean()))
        P = shapely.points(U.merc(a[:, 0], a[:, 1]))
        hit = tree.query(P, predicate="dwithin", distance=BESIDE_M * 2 * sc)
        if hit.shape[1] == 0:
            continue
        best = {}
        for pi, j in zip(hit[0].tolist(), hit[1].tolist()):
            d = float(shapely.distance(P[pi], geo[j])) / sc
            k = (named[j]["key"], named[j]["owner"])
            if d < best.get((pi, k), INF):
                best[(pi, k)] = d
        per_line = defaultdict(list)
        for (pi, k), d in best.items():
            per_line[k].append(d)
        scores = {}
        for k, ds in per_line.items():
            ds = ds + [INF] * (len(a) - len(ds))
            scores[k] = float(np.median(ds))
        k, m = min(scores.items(), key=lambda kv: kv[1])
        if m <= BESIDE_M:
            s["key"], s["owner"] = k
            s["beside"] = True
            n_took += 1
            km_took += s["km"]
    log(f"GA: {n_took} unnamed segments ({km_took:,.1f} km) lie beside a named line and take "
        f"its name (within {BESIDE_M:.0f} m)")


def group(segs, log, covered=None):
    adopt_beside(segs, log)
    named = [s for s in segs if s["key"]]
    unnamed = [s for s in segs if not s["key"]]
    saved = U.JOIN_KM
    U.JOIN_KM = JOIN_KM
    try:
        lines, _left = U.group_lines(named, lambda m: None)
    finally:
        U.JOIN_KM = saved
    attach_unnamed(lines, unnamed, log, covered)
    return lines


def line_id(state, key, segs, split):
    tag = f"au|{state}|{key}"
    if split or not key:
        tag += f"|{min(s['oid'] for s in segs)}"
    return "a" + hashlib.blake2b(tag.encode("utf-8"), digest_size=5).hexdigest()


# ================================================================ sections

def weighted_sections(adj, at, cost):
    """us_register.absorbing_sections with each edge charged `cost[seg id]` times its length,
    so a section between two stops follows track a passenger route runs over rather than a
    shorter freight cut-off of the same line. The section's km is still its real length."""
    sections = {}
    for src, sid in at.items():
        dist, prev, seen, real = {src: 0.0}, {}, set(), {src: 0.0}
        heap = [(0.0, src)]
        while heap:
            d, u = heapq.heappop(heap)
            if u in seen:
                continue
            seen.add(u)
            other = at.get(u)
            if u != src and other is not None:
                if other != sid:
                    key = (sid, other) if sid < other else (other, sid)
                    if key not in sections or d < sections[key]["cost"] - 1e-9:
                        edges, nodes, cur = [], [u], u
                        while cur != src:
                            p, e = prev[cur]
                            edges.append(e)
                            nodes.append(p)
                            cur = p
                        edges.reverse()
                        nodes.reverse()
                        if sid > other:
                            edges.reverse()
                            nodes.reverse()
                        sections[key] = {"km": real[u], "cost": d, "edges": edges,
                                         "nodes": nodes, "chain": sum(e[3] for e in edges)}
                continue                                    # absorbed
            for v, e in adj.get(u, ()):
                nd = d + e[2] * cost.get(e[5], OFF_ROUTE_COST)
                if nd < dist.get(v, INF):
                    dist[v] = nd
                    prev[v] = (u, e)
                    real[v] = real[u] + e[2]
                    heapq.heappush(heap, (nd, v))
    return sections


def line_sections(lg, stop_at, cost):
    """us_register.narn_sections over weighted_sections; junctions are "aj<node>"."""
    adj = lg.build()
    at = dict(stop_at)
    for n in U.line_ends(adj):
        if n not in at:
            at[n] = f"aj{n}"
    if len(set(at.values())) < 2:
        return {}, at
    secs = {}
    for _round in range(8):
        secs = weighted_sections(adj, at, cost)
        extra = U.branch_points(secs, at, adj, lg.pos)
        if not extra:
            break
        for n in extra:
            at[n] = f"aj{n}"
    return secs, at


# ================================================================ the OSM half

def load_osm(log):
    import build_model as bm
    ways, rels, stops, cid, cx, cy = bm.load("au", log)
    return bm, ways, rels, stops, bm.Coords(cid, cx, cy)


# Which OSM route=train relations are passenger trains that count (Anita: scheduled more
# often than about weekly; weekly-or-rarer tourist trains stay out). OSM Australia has many
# route=train relations that are none: closed and heritage railways mapped as routes
# ("Cathkin-Alexandra", "Spinifex Flyer", "Tolga-Millaa Millaa Railway", "Wilmington
# Railway"), freight ("Worsley to Hamilton", "Hobart - Boyer (Freight)"), a stabling siding,
# a proposal ("Melbourne Metro 2"). So a route counts only on a network or operator that runs
# scheduled passenger trains (2026-10-02 extract), or by its name for the long-distance trains
# mapped with neither, and never as one of the tourist trains below.
PASSENGER_NETWORKS = {
    "Sydney Trains", "NSW TrainLink", "PTV - Metropolitan Trains", "PTV - Regional Trains",
    "V/Line", "Translink", "Translink Nightlink", "Queensland Rail Travel", "Transperth",
    "Adelaide Metro"}
PASSENGER_OPERATORS = {"Journey Beyond Rail Expeditions", "Great Southern Railway", "Transwa",
                       "Countrylink", "NSW TrainLink"}
PASSENGER_NAMES = re.compile(r"^(?:The Ghan|Prospector|Indian Pacific|The Overland|"
                             r"Kuranda Scenic Railway|Puffing Billy Railway)\b")
# Weekly or rarer, or tourist excursions (the Ghan is tagged service=tourism in OSM but runs
# weekly or twice weekly all year and is one of Anita's named trains; Puffing Billy and the
# Kuranda Scenic Railway run every day). The Gulflander and the Savannahlander run weekly.
TOURIST_ROUTE = re.compile(
    r"Gulflander|Savannahlander|Rail Tour|Byron Bay Train|Heritage|Historic|Pichi Richi|"
    r"Hotham Valley|Steam|Zig Zag|Cockle Train|Great Southern\b(?! Railway)", re.IGNORECASE)


def is_passenger_route(tags):
    if tags.get("type") != "route" or tags.get("route") != "train":
        return False
    name = tags.get("name", "")
    if TOURIST_ROUTE.search(name):
        return False
    if tags.get("service") in ("industrial", "freight"):
        return False
    return (tags.get("network") in PASSENGER_NETWORKS
            or tags.get("operator") in PASSENGER_OPERATORS
            or bool(PASSENGER_NAMES.match(name)))


def route_ways(rels):
    """{rel id: [way id]}: the track members (no role, forward/backward; the Puffing Billy
    relation calls its track "line")."""
    out = {}
    for rid, (tags, members) in rels.items():
        if is_passenger_route(tags):
            out[rid] = [ref for ty, ref, role in members
                        if ty == "w" and (not role or role == "line"
                                          or role.startswith(("forward", "backward")))]
    return out


# ================================================================ outside numbers, by path

# Shortest path over the kept register track between two places, against a published length:
# the register's network as a whole. (lat, lon) pairs; Wikipedia figures.
PATH_CHECKS = [
    ("Sydney - Perth (via Broken Hill)", (-33.8832, 151.2063), (-31.9504, 115.8722), 3961.0,
     "Indian Pacific's shortest path, Sydney - Perth by rail 3,961 km, Wikipedia"),
    ("Sydney - Melbourne", (-33.8832, 151.2063), (-37.8184, 144.9524), 961.0,
     "Sydney Central - Southern Cross via Albury, NSW TrainLink XPT, Wikipedia"),
    ("Lidcombe - Albury", (-33.8642, 151.0471), (-36.0844, 146.9245), 629.6,
     "Main Southern line, Lidcombe 16.61 km to Albury 646.24 km, Wikipedia"),
    ("Brisbane - Cairns", (-27.4655, 153.0189), (-16.9265, 145.7713), 1681.0,
     "North Coast line (Queensland), Roma Street - Cairns, Wikipedia"),
    ("Port Augusta - Kalgoorlie", (-32.4936, 137.7744), (-30.7490, 121.4660), 1691.0,
     "Trans-Australian Railway, Wikipedia"),
    ("Strathfield - Armidale", (-33.8717, 151.0940), (-30.5129, 151.6649), 567.0,
     "Main North line, Strathfield km 12 - Armidale km 579, Wikipedia"),
    ("Maitland - Casino", (-32.7372, 151.5573), (-28.8660, 153.0450), 612.1,
     "North Coast line (NSW), Maitland km 193 - Casino km 805.1, Wikipedia"),
]


def path_checks(segs, log, checks=PATH_CHECKS):
    adj, pos = defaultdict(list), {}
    for s in segs:
        adj[s["a"]].append((s["b"], s["km"]))
        adj[s["b"]].append((s["a"], s["km"]))
        pos[s["a"]], pos[s["b"]] = s["pts"][0], s["pts"][-1]
    ids = list(pos)
    P = np.asarray([pos[i] for i in ids])
    out = []

    def nearest(lat, lon):
        d = np.hypot((P[:, 0] - lon) * math.cos(math.radians(lat)), P[:, 1] - lat)
        return ids[int(np.argmin(d))]
    for label, a, b, km, note in checks:
        src, dst = nearest(*a), nearest(*b)
        dist, h = {src: 0.0}, [(0.0, src)]
        while h:
            d, u = heapq.heappop(h)
            if u == dst:
                break
            if d > dist[u]:
                continue
            for v, w in adj[u]:
                if d + w < dist.get(v, INF):
                    dist[v] = d + w
                    heapq.heappush(h, (d + w, v))
        got = dist.get(dst)
        out.append((label, got, km))
        log(f"path check {label}: {'no path' if got is None else f'{got:.1f} km'} over GA "
            f"against {km:.1f} ({note})" + ("" if got is None else f", ratio {got / km:.3f}"))
    return out


# ================================================================ build

def segment_cover(segs, cover):
    """{seg id: share of it within ROUTE_NEAR_M of OSM passenger route track}."""
    return {s["id"]: cover.share(s["pts"]) for s in segs}


HOLE_FAR_M = 25        # a left-out segment's part further than this from kept track...
HOLE_AWAY = 0.5        # ...must be at least this share of it (not a track beside kept track)
HOLE_SHARE = 0.6       # ...and this much of that part must lie under an OSM passenger route


def accept_holes(segs, other, cover, log):
    """The left-out segments (sidings, track GA calls disused, closed or dismantled) that
    OSM passenger routes run over where the kept register has no track: at Springhurst the
    North Eastern standard gauge runs on what GA files as an operational siding, the line
    itself being "Dismantled" there. A tram, metro or heritage segment is never taken."""
    import shapely
    idx = U.SegIndex([{"segs": segs}])
    took, by_why = [], defaultdict(float)
    for s in other:
        if s["why"] in ("tram or metro", "heritage"):
            continue
        a = U.densify(np.asarray(s["pts"], dtype=np.float64), U.STEP_DENSE)
        near_p, _g = U.within_many(idx.tree, a, HOLE_FAR_M, shapely)
        away = np.ones(len(a), dtype=bool)
        away[near_p] = False
        if away.sum() < HOLE_AWAY * len(a) or away.sum() * U.STEP_DENSE < 50:
            continue
        A = a[away]
        on_p, _g = U.within_many(cover.tree, A, U.ROUTE_NEAR_M, shapely)
        if len(set(on_p.tolist())) / len(A) >= HOLE_SHARE:
            took.append(dict(s, hole=True))
            by_why[s["why"]] += s["km"]
    log(f"holes: {len(took)} left-out segments taken back, an OSM passenger route running "
        f"over them away from kept track ({sum(by_why.values()):,.1f} km: "
        + ", ".join(f"{w} {km:,.1f}" for w, km in sorted(by_why.items(), key=lambda x: -x[1]))
        + ")")
    agg = defaultdict(float)
    for s in took:
        agg[(s["why"], s["state"], s["raw"])] += s["km"]
    for (why, state, raw), km in sorted(agg.items(), key=lambda x: -x[1])[:20]:
        log(f"    {km:7.1f} km  {why:<22} {state:<8} {raw}")
    return took


TIE_M = 10.0          # a station goes on the line its routes' track lies nearest, and on any
                      # other line about as near (shared track): within this of the nearest


def place_stations(lines, idx, stations, served, rw, wgeo, log):
    """us_register.place_stations, but a station goes only on the line its own routes' track
    lies NEAREST (and any line within TIE_M of that): GA has no passenger flag, so a freight
    line beside the passenger line (Adelaide's standard gauge beside the Gawler line, Sydney's
    goods lines) would otherwise take the station too, and be built as a passenger line.
    Nearness is the median distance from the route's points near the stop to the line."""
    placed = defaultdict(list)
    n_on = n_off = n_beside = 0
    for nid, rids in served.items():
        s = stations[nid]
        lon, lat = s["lon"], s["lat"]
        pairs = idx.within(np.asarray([[lon, lat]]), U.STATION_M)
        if pairs.shape[1] == 0:
            continue
        cand = {}
        for j in pairs[1]:
            j = int(j)
            t, d = idx.project(j, lon, lat)
            li = int(idx.line_of[j])
            if li not in cand or d < cand[li][2]:
                cand[li] = (j, t, d)
        pts = []
        for rid in rids:
            for w in rw.get(rid, ()):
                a = wgeo.get(w)
                if a is None:
                    continue
                m = ((np.abs(a[:, 0] - lon) * math.cos(math.radians(lat)) * 111320 <= U.ALONG_R_M)
                     & (np.abs(a[:, 1] - lat) * 110570 <= U.ALONG_R_M))
                if m.any():
                    pts.append(a[m])
        if not pts:
            continue
        P = np.unique(np.round(np.vstack(pts), 6), axis=0)
        near = idx.within(P, U.NEAR_M)
        best = defaultdict(dict)                 # line -> point -> metres
        if near.shape[1]:
            PM = idx.shapely.points(U.merc(P[:, 0], P[:, 1]))
            sc = U.merc_scale(lat)
            for j in np.unique(near[1]).tolist():
                pis = near[0][near[1] == j]
                ds = idx.shapely.distance(PM[pis], idx.geo[j]) / sc
                li = int(idx.line_of[j])
                for pi, d in zip(pis.tolist(), ds.tolist()):
                    if d < best[li].get(pi, INF):
                        best[li][pi] = d
        ok = {li: float(np.median(list(v.values()))) for li, v in best.items()
              if li in cand and len(v) * U.STEP_DENSE >= U.ALONG_M}
        if not ok:
            n_off += 1
            continue
        dmin = min(ok.values())
        for li, dmed in ok.items():
            if dmed > dmin + TIE_M:
                n_beside += 1
                continue
            j, t, _d = cand[li]
            placed[li].append((idx.ref[j][1], t, nid, len(rids)))
        n_on += 1
    log(f"stations: {n_on} served rail stations placed on a register line; {n_off} near one "
        f"that none of their routes runs along; {n_beside} placements refused because another "
        f"line lies nearer the route's track (more than {TIE_M:.0f} m nearer)")
    return placed


PLATFORM_NAME = re.compile(r"^(.*?)(?:\s+(?:railway\s+)?station)?"
                           r"(?:,?\s+(?:platform|plt)\s+\w+|\s+\d{1,2}[a-z]?)$", re.IGNORECASE)
PLATFORM_M = 600.0


def fold_platform_records(st, served, log):
    """A route stop OSM names after its platform ("Broadmeadows 3", "Southern Cross 1",
    "Central, Platform 23", "Dubbo Station, Platform 1") is the station of the plain name
    within PLATFORM_M, where there is one: its routes go to that station, so no line has a
    platform for a stop."""
    by_name = defaultdict(list)
    for nid, s in st.items():
        if s["name"]:
            by_name[s["name"]].append(nid)
    moved = []
    for nid in list(served):
        s = st.get(nid)
        m = PLATFORM_NAME.match(s["name"] or "") if s else None
        if not m:
            continue
        base = m.group(1).strip().rstrip(",")
        cands = by_name.get(base, []) + by_name.get(re.sub(r"\s+Station$", "", base), [])
        cands = [c for c in cands if c != nid
                 and dist_m(s["lon"], s["lat"], st[c]["lon"], st[c]["lat"]) <= PLATFORM_M]
        if not cands:
            continue
        to = min(cands, key=lambda c: dist_m(s["lon"], s["lat"], st[c]["lon"], st[c]["lat"]))
        served[to] = set(served.get(to, set())) | served.pop(nid)
        moved.append(f"{s['name']} -> {st[to]['name']}")
    log(f"stations: {len(moved)} stops named after a platform given to their station "
        f"(e.g. {', '.join(moved[:12])})")


NO_STOP_MIN_KM = 2.0   # a line with no station needs this much track under a route
NO_STOP_LINE_KM = 3.0  # and, once built, this many km of sections
PEEL_COVER = 0.2     # a dead-end segment of a line less covered than this is peeled off


def peel(g, covered, outside):
    """Drop a line's dead-end segments no route runs over, repeatedly: sidings, freight spurs
    and yard track hanging off a passenger line, which would each make a line end, a
    junction and a section to be dropped later. Uncovered track BETWEEN covered track (a gap
    in OSM's route, a loop) stays, and so does an end where other kept track goes on
    (`outside`: node -> segments of other lines there): the Trans-Australian's last 52 km to
    the South Australian border lie 85-95 m off OSM's track, too far to count as covered, and
    the line goes on there as "Port Augusta - WA Border"."""
    segs = list(g["segs"])
    while True:
        deg = Counter()
        for s in segs:
            deg[s["a"]] += 1
            deg[s["b"]] += 1

        def loose(n):
            return deg[n] == 1 and not outside.get(n)
        drop = {id(s) for s in segs
                if (loose(s["a"]) or loose(s["b"])) and covered.get(s["id"], 0.0) < PEEL_COVER}
        if not drop:
            break
        segs = [s for s in segs if id(s) not in drop]
    g["segs"] = segs
    return g


DUP_M = 25.0           # a junction-ended section lying this close to a kept section...
DUP_SHARE = 0.7        # ...for this share of its length is another track of it, and dropped


def dedup_sections(items):
    """Keep one section per stretch of a multiple-track line. GA draws each track (NSW's main
    lines out of Sydney have four to six, joined by crossovers), so the absorbing Dijkstra
    finds sections between junctions on the second track beside the stop-to-stop section on
    the first. Stop-to-stop sections are kept first, never questioned; then junction-ended
    ones, longest first, unless DUP_SHARE of one lies within DUP_M of those kept.
    items: [(sa, sb, v, pts)]. Returns (kept, dropped)."""
    import shapely
    from shapely.geometry import LineString

    def junc(it):
        return it[0].startswith("aj") or it[1].startswith("aj")
    order = sorted(items, key=lambda it: (junc(it), -it[2]["km"]))
    kept, dropped, geo = [], [], []
    for it in order:
        a = np.asarray(it[3], dtype=np.float64)
        if len(a) < 2:
            dropped.append(it)
            continue
        bb = (a[:, 0].min(), a[:, 1].min(), a[:, 0].max(), a[:, 1].max())
        if junc(it) and geo:
            d = U.densify(a, 25.0)
            sc = U.merc_scale(float(d[:, 1].mean()))
            P = shapely.points(U.merc(d[:, 0], d[:, 1]))
            near = np.zeros(len(d), dtype=bool)
            pad = 0.001
            for gm, gb in geo:
                if (bb[2] < gb[0] - pad or gb[2] < bb[0] - pad or bb[3] < gb[1] - pad
                        or gb[3] < bb[1] - pad):
                    continue
                near |= shapely.distance(P, gm) / sc <= DUP_M
                if near.mean() >= DUP_SHARE:
                    break
            if near.mean() >= DUP_SHARE:
                dropped.append(it)
                continue
        kept.append(it)
        geo.append((LineString(U.merc(a[:, 0], a[:, 1])), bb))
    return kept, dropped


WIDE_CORRIDOR_M = 600.0   # a section no trace finds within us_register's corridor is tried
                          # again in one this wide (GA's NT and WA geometry is off by more)


def trace(track, pts, chain_km):
    """(traced (pts, km) or None, whether the wide corridor was needed)."""
    tol = U.TRACE_TOL[0] * chain_km + U.TRACE_TOL[1]
    got = track.trace(pts, chain_km)
    if got is not None and abs(got[1] - chain_km) <= tol:
        return got, False
    saved = U.CORRIDOR_M, U.SNAP_M
    U.CORRIDOR_M, U.SNAP_M = WIDE_CORRIDOR_M, WIDE_CORRIDOR_M
    try:
        got = track.trace(pts, chain_km)
    finally:
        U.CORRIDOR_M, U.SNAP_M = saved
    if got is not None and abs(got[1] - chain_km) <= tol:
        return got, True
    return None, False


def fuse_junctions(l, sgeo, fusable):
    """Join a line's two sections through a junction that only this line uses and that only
    those two sections end at: what is left of a branch point once the other track's
    sections are gone, a place nobody boards and no other line meets. Returns the count."""
    n = 0
    while True:
        ends = defaultdict(list)
        for i, sec in enumerate(l["sections"]):
            ends[sec[0]].append(i)
            ends[sec[1]].append(i)
        done = False
        for j, idx in ends.items():
            if j not in fusable or len(idx) != 2 or idx[0] == idx[1]:
                continue
            s1, s2 = l["sections"][idx[0]], l["sections"][idx[1]]
            a = s1[0] if s1[1] == j else s1[1]
            b = s2[0] if s2[1] == j else s2[1]
            if a == b or a == j or b == j:
                continue
            if f"{a}|{b}" in sgeo or f"{b}|{a}" in sgeo:
                continue
            g1 = sgeo[f"{s1[0]}|{s1[1]}"]
            g1 = g1 if s1[0] == a else g1[::-1]
            g2 = sgeo[f"{s2[0]}|{s2[1]}"]
            g2 = g2 if s2[0] == j else g2[::-1]
            c = l["chain"]
            k1, k2 = f"{s1[0]}|{s1[1]}", f"{s2[0]}|{s2[1]}"
            sgeo[f"{a}|{b}"] = g1 + g2[1:]
            c[f"{a}|{b}"] = round(c.pop(k1, 0.0) + c.pop(k2, 0.0), 3)
            del sgeo[k1], sgeo[k2]
            new = [a, b, round(s1[2] + s2[2], 3)]
            l["sections"] = [s for i, s in enumerate(l["sections"]) if i not in idx] + [new]
            n += 1
            done = True
            break
        if not done:
            return n


LINE_WORD = re.compile(r"\b(?:Line|Railway|Railroad|System)$")
COMPANION_M = 30.0    # a line whose sections lie within this of a longer line's...
COMPANION_SHARE = 0.85 # ...for this share of their length is that line's other track


def find_companions(lines, geoms, log):
    """A line drawn along another one for nearly all its length is that line's other track:
    South Australia's file names the two tracks of Adelaide's lines apart ("Gawler Line" and
    "Gawler - Adelaide", "Seaford Line" and "Noarlunga - Adelaide"). It is declared the longer
    line's companion (`companion_of`, as the USA's second tracks), and ownership.py gives its
    track to that line (Anita: both directions of a line are one track unless far apart)."""
    import shapely
    from shapely.geometry import LineString, MultiLineString
    info = {}
    for l in lines:
        parts = [np.asarray(p, dtype=np.float64) for p in geoms[l["id"]].values() if len(p) >= 2]
        if not parts:
            continue
        allp = np.vstack(parts)
        sc = U.merc_scale(float(allp[:, 1].mean()))
        dense = np.vstack([U.densify(p, 100.0) for p in parts])
        info[l["id"]] = {
            "geo": MultiLineString([LineString(U.merc(p[:, 0], p[:, 1])) for p in parts]),
            "pts": shapely.points(U.merc(dense[:, 0], dense[:, 1])), "sc": sc,
            "bb": (allp[:, 0].min(), allp[:, 1].min(), allp[:, 0].max(), allp[:, 1].max())}
    byid = {l["id"]: l for l in lines}

    def along(x, y):
        return float((shapely.distance(info[x]["pts"], info[y]["geo"]) / info[x]["sc"]
                      <= COMPANION_M).mean())

    def rank(i):          # a line's own name ("Gawler Line") before a section's ("Gawler -
        return 0 if LINE_WORD.search(byid[i]["name"]) else 1   # Adelaide")
    n = 0
    for x in sorted(info, key=lambda i: byid[i]["km"]):
        if byid[x].get("companion_of"):
            continue
        X = info[x]
        best = None
        for y, Y in info.items():
            if y == x or byid[y]["km"] <= byid[x]["km"] or byid[y].get("companion_of"):
                continue
            a, b = X["bb"], Y["bb"]
            if a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1]:
                continue
            share = along(x, y)
            if share >= COMPANION_SHARE and (best is None or share > best[0]):
                best = (share, y)
        if best is None:
            continue
        share, y = best
        comp, main = x, y
        # two tracks of one line named apart, about as long: the one named as a line is it
        if (rank(x) < rank(y) and byid[x]["km"] >= 0.85 * byid[y]["km"]
                and along(y, x) >= COMPANION_SHARE):
            comp, main = y, x
        byid[comp]["companion_of"] = main
        n += 1
        log(f"    companion: {byid[comp]['name']} ({byid[comp]['km']:.1f} km) lies along "
            f"{byid[main]['name']} for {share:.0%} of it: its other track")
    log(f"AU: {n} register lines declared the companion of a longer line they run beside")


def build(path, log):
    from n02 import walk_order
    segs, other, left, left_by = read_ga(path, log)

    bm, ways, rels, stops, coords = load_osm(log)
    st, resolved = bm.build_stations(stops, rels, coords, log, "au")
    rw = route_ways(rels)
    served = defaultdict(set)
    for rid in rw:
        for ref in bm.stop_members(rels[rid][1]):
            s = resolved.get(ref)
            if s is not None:
                served[s].add(rid)
    wraw = {}
    for w in {w for ws in rw.values() for w in ws}:
        if w in ways:
            _n, ll = U.way_lonlat(w, ways, coords)
            if ll is not None:
                wraw[w] = ll
    wgeo = {w: U.densify(a, U.STEP_DENSE) for w, a in wraw.items()}
    log(f"OSM: {len(rw)} passenger train routes over {len(wraw):,} ways; "
        f"{len(served)} stations they stop at")
    unlisted = U.unlisted_stations(st, stops, served, rw, wraw, log)
    for nid, rids in unlisted.items():
        served[nid] = rids
    fold_platform_records(st, served, log)
    cover = U.RouteCover(wraw)
    segs = segs + accept_holes(segs, other, cover, log)
    segs = topology(segs, log)
    path_checks(segs, log)
    covered = segment_cover(segs, cover)
    on_km = sum(s["km"] * covered[s["id"]] for s in segs)
    log(f"GA: {on_km:,.0f} km of the kept register lies under an OSM passenger route "
        f"(within {U.ROUTE_NEAR_M} m)")
    # how much of what the register leaves out do routes run over (a register hole)
    oc = segment_cover(other, cover)
    by_why = defaultdict(float)
    worst = []
    for s in other:
        km = s["km"] * oc[s["id"]]
        by_why[s["why"]] += km
        if oc[s["id"]] >= 0.5 and s["km"] >= 0.5:
            worst.append((s["km"], s["why"], s["state"], s["raw"]))
    log("GA left-out track under OSM passenger routes: "
        + ", ".join(f"{w} {km:,.1f} km" for w, km in sorted(by_why.items(), key=lambda x: -x[1])))
    agg = defaultdict(float)
    for km, why, state, raw in worst:
        agg[(why, state, raw)] += km
    for (why, state, raw), km in sorted(agg.items(), key=lambda x: -x[1])[:25]:
        log(f"    {km:7.1f} km  {why:<22} {state:<8} {raw}")

    groups = group(segs, log, covered)
    # a line with no track under a route is freight (or heritage) and is not built
    keep = []
    for g in groups:
        ckm = sum(s["km"] * covered.get(s["id"], 0.0) for s in g["segs"])
        if ckm >= 0.3:
            keep.append(g)
    before = sum(sum(s['km'] for s in g['segs']) for g in keep)
    at_node = defaultdict(set)
    for i, g in enumerate(keep):
        for s in g["segs"]:
            at_node[s["a"]].add(i)
            at_node[s["b"]].add(i)
    for i, g in enumerate(keep):
        peel(g, covered, {n: ls - {i} for n, ls in at_node.items() if ls - {i}})
    keep = [g for g in keep if g["segs"]]
    log(f"GA: {len(groups)} register lines, {len(keep)} with track under an OSM passenger "
        f"route ({before:,.0f} km of track; {sum(sum(s['km'] for s in g['segs']) for g in keep):,.0f}"
        f" km once dead ends no route runs over are peeled off)")
    groups = keep
    idx = U.SegIndex(groups)
    U.offsets_report(idx, wgeo, log)
    names = [display_name(g["key"]) if g["key"] else "" for g in groups]
    node_lines = defaultdict(set)
    for i, g in enumerate(groups):
        for s in g["segs"]:
            node_lines[s["a"]].add(i)
            node_lines[s["b"]].add(i)
    placed = place_stations(groups, idx, st, served, rw, wgeo, log)
    track = U.OsmTrack(ways, coords, log) if U.TRACE else None
    named = [s for s in st.values() if s["name"] and not s.get("_tram")]
    stations_any = (np.asarray([[s["lon"], s["lat"]] for s in named]).reshape(-1, 2),
                    [s["name"] for s in named])
    inames = U.infra_names(PROC, None, log)
    cost = {sid: 1.0 + (OFF_ROUTE_COST - 1.0) * (1.0 - min(1.0, c / 0.8))
            for sid, c in covered.items()}

    out_lines, out_st, geoms = [], {}, {}
    n_unrouted, km_unrouted, unrouted = 0, 0.0, []
    n_traced = n_fallback = n_wide = 0
    fallback = []
    n_merged = 0
    n_skip = 0
    n_dup = km_dup = 0.0
    t0 = time.time()
    for li, g in enumerate(groups):
        if li and li % 100 == 0:
            log(f"    ... {li} of {len(groups)} lines, {time.time() - t0:.0f} s")
        # A line no station goes on and little track of which lies under a route is a freight
        # line or a yard beside the passenger line, not a line of its own.
        if not placed.get(li) and sum(s["km"] * covered.get(s["id"], 0.0)
                                      for s in g["segs"]) < NO_STOP_MIN_KM:
            n_skip += 1
            continue
        lg = U.LineGraph(g["segs"])
        pl = []
        for si, t, nid, nr in placed.get(li, ()):
            seg = g["segs"][si]
            p = U.split_line(seg["pts"], [t])[0][-1]
            pl.append((p, si, t, nid, nr))
        pl.sort(key=lambda x: (-x[4], x[3]))
        kept_pl = []
        for x in pl:
            if any(dist_m(*x[0], *k[0]) <= U.MERGE_M for k in kept_pl):
                n_merged += 1
                continue
            kept_pl.append(x)
        stop_at = {}
        for p, si, t, nid, _nr in kept_pl:
            sid = f"n{nid}"
            for sj, seg in enumerate(g["segs"]):
                if sj != si:
                    a = np.asarray(seg["pts"])
                    if (p[0] < a[:, 0].min() - 0.002 or p[0] > a[:, 0].max() + 0.002
                            or p[1] < a[:, 1].min() - 0.002 or p[1] > a[:, 1].max() + 0.002):
                        continue
                    tj, dj = U.nearest_on(seg["pts"], p)
                    if dj > U.PARALLEL_M:
                        continue
                else:
                    tj = t
                node = U.place_node(lg, sj, tj)
                stop_at[node] = sid
        secs, at = line_sections(lg, stop_at, cost)
        if not secs:
            continue
        lid = line_id(g["owner"], g["key"], g["segs"], g["split"])
        name = names[li]
        items, pre = [], {}
        for (sa, sb), v in sorted(secs.items()):
            pts = U.section_geometry(v)
            share = cover.share(pts)
            if share < UNROUTED_SHARE and track is not None and v["km"] >= 1.0:
                # GA's own geometry can lie further from the track than ROUTE_NEAR_M (the
                # Nullarbor east of Reid): measured again on the OSM track it traces to
                got, wide = trace(track, pts, v["chain"])
                if got is not None:
                    share = max(share, cover.share(got[0]))
                    pre[(sa, sb)] = (got, wide)
            if share < UNROUTED_SHARE:
                if not (sa.startswith("aj") or sb.startswith("aj")):
                    n_unrouted += 1
                    km_unrouted += v["km"]
                    unrouted.append((v["km"], name, g["owner"], sa, sb, share))
                continue
            items.append((sa, sb, v, pts))
        items, dropped = dedup_sections(items)
        n_dup += len(dropped)
        km_dup += sum(it[2]["km"] for it in dropped)
        sections, chain, sgeo = [], {}, {}
        for sa, sb, v, pts in items:
            km = v["km"]
            if track is not None:
                got, wide = pre.get((sa, sb)) or trace(track, pts, v["chain"])
                if got is not None:
                    pts, km = got
                    n_traced += 1
                    n_wide += wide
                else:
                    n_fallback += 1
                    fallback.append((v["chain"], None, name, sa, sb))
            sections.append([sa, sb, round(km, 3)])
            chain[f"{sa}|{sb}"] = round(v["chain"], 3)
            sgeo[f"{sa}|{sb}"] = [[round(x, 5), round(y, 5)] for x, y in pts]
        if not sections:
            continue
        node_of = {sid: n for n, sid in at.items()}
        jpos = {}
        for sec in sections:
            for sid in sec[:2]:
                if sid.startswith("aj"):
                    n = node_of[sid]
                    jpos[sid] = (n, *lg.pos[n])
        infra = Counter(s["infra"] for s in g["segs"] if s["infra"]).most_common(1)
        line = {
            "id": lid, "src": "ga", "service": False,
            "name": name, "name_en": "", "ref": "", "colour": "",
            "operator": operator_name(infra[0][0]) if infra else "", "operator_en": "",
            "network": "", "kind": "rail",
            "km": 0.0, "km_official": 0.0, "chain": chain,
            "variants": 1, "straight_sections": 0, "display": [],
            "sections": sections,
            "_split": g["split"], "_state": g["owner"], "_g": g, "_li": li, "_jpos": jpos,
        }
        out_lines.append(line)
        geoms[lid] = sgeo
    log(f"AU: {n_skip} lines with no station and under {NO_STOP_MIN_KM} km under a route "
        f"skipped; {int(n_dup)} junction-ended sections ({km_dup:,.0f} km) dropped as another "
        f"track of a section kept (within {DUP_M:.0f} m for {DUP_SHARE:.0%} of them)")

    # --- a junction only one line uses, between two of its sections, is no section end
    j_lines = defaultdict(set)
    for l in out_lines:
        for sec in l["sections"]:
            for sid in sec[:2]:
                if sid.startswith("aj"):
                    j_lines[sid].add(l["id"])
    n_fused = 0
    for l in out_lines:
        n_fused += fuse_junctions(l, geoms[l["id"]], {j for j, ls in j_lines.items()
                                                      if ls == {l["id"]}})

    # --- stations
    for l in out_lines:
        li = l.pop("_li")
        jpos = l.pop("_jpos")
        for sec in l["sections"]:
            for sid in sec[:2]:
                if sid not in out_st:
                    if sid.startswith("aj"):
                        n, lon, lat = jpos[sid]
                        out_st[sid] = {"id": sid, "lon": lon, "lat": lat, "lines": set(),
                                       "name": U.junction_name(n, node_lines.get(n, {li}),
                                                              names, (lon, lat), stations_any,
                                                              border=[]),
                                       "name_en": "", "junction": True}
                    else:
                        s = st[int(sid[1:])]
                        out_st[sid] = {"id": sid, "name": s["name"], "name_en": s["name_en"],
                                       "lon": s["lon"], "lat": s["lat"], "lines": set()}
                out_st[sid]["lines"].add(l["id"])
        l["km"] = round(sum(s[2] for s in l["sections"]), 3)
        l["km_official"] = round(sum(l["chain"].values()), 3)
        l["display"] = U.display_order(l["sections"], walk_order)
    n_osm_named = 0
    for l in out_lines:
        g = l.pop("_g")
        if not g["key"]:
            # the OSM route=railway relation the track lies on, else "<first> – <last stop>"
            # (osm_line_name writes "<owner>: ..." and the owner here is a state code)
            name = U.osm_line_name(g, inames, track, out_st, l["sections"]) or "Unnamed line"
            if name.startswith(f"{g['owner']}: "):
                name = name[len(g["owner"]) + 2:]
            l["name"] = name
            n_osm_named += 1
    # One name for two lines (two states' Airport Line, or a name in pieces more than JOIN_KM
    # apart): an English name telling them apart by their end stops.
    find_companions(out_lines, geoms, log)
    # A line with no stop of its own is a loop, a spur or a connecting curve beside the lines
    # people ride (South Australia's crossing loops are named lines: "Coonalpyn Spur Line"):
    # dropped when it is another line's companion or shorter than NO_STOP_LINE_KM. The
    # Wodonga Rail Bypass (4.4 km, the XPT) and the Bethungra Spiral stay.
    gone = []
    for l in out_lines:
        stops_ = [s for sec in l["sections"] for s in sec[:2] if not s.startswith("aj")]
        if not stops_ and (l.get("companion_of") or l["km"] < NO_STOP_LINE_KM):
            gone.append(l)
    for l in gone:
        out_lines.remove(l)
        geoms.pop(l["id"], None)
        for sec in l["sections"]:
            for sid in sec[:2]:
                out_st[sid]["lines"].discard(l["id"])
    for sid in [s for s, v in out_st.items() if not v["lines"]]:
        del out_st[sid]
    left_ids = {l["id"] for l in out_lines}
    for l in out_lines:
        if l.get("companion_of") not in left_ids:
            l.pop("companion_of", None)
    log(f"AU: {len(gone)} lines with no stop of their own dropped ({sum(l['km'] for l in gone):,.1f}"
        f" km): " + ", ".join(sorted(l["name"] for l in gone))[:600])
    # One name for two lines (two states' Airport Line, or a name in pieces more than JOIN_KM
    # apart): an English name telling them apart by their two stops furthest apart.
    by_name = Counter(l["name"] for l in out_lines)
    for l in out_lines:
        split = l.pop("_split")
        l.pop("_state")
        if split or by_name[l["name"]] > 1:
            stops_ = sorted({s for sec in l["sections"] for s in sec[:2]
                             if not out_st[s].get("junction") and out_st[s]["name"]})
            if len(stops_) >= 2:
                a, b = max(((x, y) for i, x in enumerate(stops_) for y in stops_[i + 1:]),
                           key=lambda p: dist_m(out_st[p[0]]["lon"], out_st[p[0]]["lat"],
                                                out_st[p[1]]["lon"], out_st[p[1]]["lat"]))
                d = l["display"]
                if a in d and b in d and d.index(a) > d.index(b):
                    a, b = b, a
                l["name_en"] = f"{l['name']} ({out_st[a]['name']} – {out_st[b]['name']})"
    log(f"AU: {n_fused} junctions only one line used, between two of its sections, fused away; "
        f"{n_osm_named} unnamed lines named from OSM or their end stops")
    total = sum(l["km"] for l in out_lines)
    n_j = sum(1 for s in out_st.values() if s.get("junction"))
    log(f"AU: {len(out_lines)} register lines, {total:,.0f} km, {len(out_st)} stations of "
        f"which {n_j} are junctions; {n_merged} station records merged into a neighbour on a "
        f"line (within {U.MERGE_M} m)")
    log(f"AU: {n_unrouted} sections between two stops ({km_unrouted:,.0f} km) left out: no "
        f"OSM passenger route runs over {UNROUTED_SHARE:.0%} of them")
    for km, name, owner, sa, sb, share in sorted(unrouted, reverse=True)[:40]:
        log(f"    {km:7.1f} km  {name} [{owner}]  {U.out_name(st, sa)} - {U.out_name(st, sb)}"
            f"  ({share:.0%} on a route)")
    if track is not None:
        log(f"AU: {n_traced} sections drawn on OSM track ({n_wide} of them only in a "
            f"{WIDE_CORRIDOR_M:.0f} m corridor), {n_fallback} on GA's own geometry (no OSM path "
            f"within {U.TRACE_TOL[0]:.0%} + {U.TRACE_TOL[1]} km of GA's length)")
        for chain_km, got, name, sa, sb in sorted(fallback, key=lambda x: -x[0])[:30]:
            log(f"    {chain_km:7.1f} km GA  {name}  {U.out_name(st, sa, out_st)} - "
                f"{U.out_name(st, sb, out_st)}")
    for l in sorted(out_lines, key=lambda l: -l["km"]):
        log(f"    line {l['km']:8.1f} km  {len(l['sections']):3d} sections  {l['name']}"
            f"{'  / ' + l['name_en'] if l['name_en'] else ''}  [{l['operator']}]"
            f"{'  companion of ' + l['companion_of'] if l.get('companion_of') else ''}")
    return out_lines, out_st, geoms


# The register's OWNER, as the line's operator (infrastructure manager).
OPERATORS = {
    "AUSTRALIAN RAIL TRACK CORPORATION": "Australian Rail Track Corporation",
    "ARTC": "Australian Rail Track Corporation",
    "RAIL COMMISSIONER (METRO RAIL)": "Rail Commissioner (Adelaide Metro)",
    "GENESEE AND WYOMING AUSTRALIA PTY LTD": "Genesee & Wyoming Australia",
    "GENESEE AND WYOMING AUSTRALIA": "Genesee & Wyoming Australia",
    "WESTRAIL": "Westrail",
    "STATE GROWTH": "TasRail",
    "UNKNOWN": "", "PRIVATE": "",
}


def operator_name(raw):
    u = raw.strip().upper()
    if u in OPERATORS:
        return OPERATORS[u]
    return raw if raw != u else " ".join(title(w) for w in u.split())


# ================================================================ --ga: the register alone

def ga_report(path):
    t0 = time.time()

    def log(msg):
        print(f"[{time.time()-t0:6.1f}s] {msg}", flush=True)

    segs, other, left, left_by = read_ga(path, log)
    # Path checks over the operational railway plus operational sidings (GA files some main
    # track as siding: Springhurst); the build takes such track back only under OSM routes.
    log("path checks over operational railway + operational sidings:")
    path_checks(topology(segs + [s for s in other if s["why"] == "siding"
                                 and s["code"] in OPERATIONAL], lambda m: None), log)
    segs = topology(segs, log)
    groups = group(segs, log)
    log(f"GA: {len(groups)} register lines")
    rows = []
    for g in groups:
        total = sum(s["km"] for s in g["segs"])
        rows.append({"name": display_name(g["key"]) if g["key"] else "(unnamed)",
                     "owner": g["owner"], "total": total, "segs": len(g["segs"]),
                     "pieces": g["pieces"], "split": g["split"]})
    tot = sum(r["total"] for r in rows)
    log(f"GA: {len(rows)} lines, {tot:,.0f} km of track in them")
    return rows, left, left_by


def main():
    if "--fetch" in sys.argv:
        fetch()
        return
    if "--ga" in sys.argv:
        rows, left, left_by = ga_report(RAW / OUT)
        rows.sort(key=lambda r: -r["total"])
        print(f"\n{'track':>8} segs pc  state  name")
        for r in rows if "--all" in sys.argv else rows[:60]:
            print(f"{r['total']:8.1f} {r['segs']:4d} {r['pieces']:2d}  "
                  f"{r['owner']:<8} {r['name']}{'  [split]' if r['split'] else ''}")
        print("\nleft out:")
        for (why, st, name), km in sorted(left_by.items(), key=lambda kv: -kv[1])[:50]:
            print(f"  {km:8.1f}  {why:<36} {st:<8} {name}")
        return
    sys.exit(__doc__)


if __name__ == "__main__":
    main()
