"""Estimate the vertical profile of every NYC subway track, with error bars.

Model (v1). Each track is resampled to points ~STEP m apart. Unknown: rail height z at each
point (m, NAVD88). Solved as one sparse weighted least-squares problem:

  prior        z ~ ground + offset(structure, OSM level), sd by class (inflated for correlation)
  smoothness   second difference of z small (vertical curves), first difference small (grades)
  anchors      station platform depths from MTA elevator rises
  shared ends  ways that meet share one variable, so junctions agree

Error bars are the sd of 60 perturb-and-solve samples (exact posterior sd for this Gaussian
model). They are only as honest as the class sds below, which are guesses until checked
against documented depths (evidence/depths.csv).

Writes data/tracks.json for the page.
"""
import collections
import json
import math
import os
import pathlib
import re

import numpy as np
import pandas as pd
import rasterio
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from pyproj import Transformer
from scipy.spatial import cKDTree

ROOT = pathlib.Path(__file__).parent
RAW = ROOT / "raw"
OUT = ROOT / "data"
STEP = 25.0
CORR = 1500.0  # m over which prior errors are shared (a whole stretch built at one depth)
OU_ACROSS = bool(os.environ.get("TD_OU_ACROSS"))  # calibration check: prior-error chain runs through portals, as before 2026-11-02
PLATFORM_ABOVE_RAIL = 1.1
NSAMPLES = 60

to_utm = Transformer.from_crs(4326, 32618, always_xy=True)
to_ll = Transformer.from_crs(32618, 4326, always_xy=True)

# Rail depth below street by OSM level for tunnels (m), and sd. Guesses: NYC cut-and-cover puts
# top of rail ~6 m down with no mezzanine, ~10 m under a mezzanine; each lower level ~5 m more.
# The level tag is an order (which line is above which), not a depth: 191 St is -2 and ~53 m
# down. So the sds are wide; deep rock tunnels need anchors or documents.
# Calibrated 2026-11-01 with calibrate.py (2-fold hold-out over 89 elevator and 11 document
# anchors): mean = median depth of the anchors at that OSM level, sd = their standard deviation
# (levels -1 to -4 have 16, 40, 19 and 9 anchors; the old guesses had sd 3 / 6 / 9 / 12). Medians,
# not means, because each level has a long deep tail (rock stations) and the aim was no bias for the
# typical station. Level -5 has one anchor (Lexington Av/63 St, 37.9 m against a prior of 27): too
# thin, left as the guess.
TUNNEL_LEVEL = {-1: (7.0, 5.0), -2: (10.5, 10.0), -3: (15.0, 6.0), -4: (19.0, 9.0), -5: (27.0, 15.0)}
TUNNEL_DEFAULT = (9.0, 8.0)
UNDERWATER = (-25.0, 12.0)  # absolute NAVD88: river tunnels, no ground to hang off
CLASS = {  # offset above ground, sd
    "elevated": (8.0, 4.0),
    "embankment": (0.3, 1.5),  # the bare-earth ground already includes the embankment (lidar: +0.24 m median)
    "cutting": (-5.0, 3.0),
    "grade": (0.5, 2.0),
}
# absolute, and next to nothing: the approaches (and lidar, where there is any) set a bridge's height.
# Was (25, 25), which humped the NJ drawbridges without lidar (Portal Bridge 17 m between 9 m
# approaches; it is a swing bridge with 23 ft of clearance, rail about 9-10 m).
BRIDGE_OVER_WATER = (12.0, 60.0)
# grade or embankment track over water or marsh (absolute, m NAVD88, sd): a causeway, its rail clear of high
# tide (mean high water about +0.7 m). The A across Jamaica Bay sat at 0 m on the old (0 + 0.5, sd 10) rule,
# under the surface beside it; Broad Channel's at-grade rail is 2.4 m.
CAUSEWAY = (2.5, 1.5)
MARSH_M = 300.0   # m: open track on ground under 1 m this close to the water is on a causeway or trestle

# Smoothness. Held-out sweep 2026-11-02 (with riverbeds and placed rows in): grade sd 0.05 /
# curv 0.02 gave 56% / 81% inside 1 / 2 sd; 0.2 / 0.02 68 / 93; 0.2 / 0.04 67 / 95. A zero-mean
# grade term pulled long river approaches flat and dragged stations near rivers too deep; the
# hard 4% limit (GRADE_LIMIT) does the real work.
MAX_GRADE = float(os.environ.get("TD_MAX_GRADE", "0.2"))   # sd of grade
CURV_SD = float(os.environ.get("TD_CURV", "0.04"))         # sd of grade change per STEP
GRADE_LIMIT = float(os.environ.get("TD_GRADE_LIMIT", "0.04"))  # steepest grade allowed after the reweighting rounds in solve()
GRADE_ROUNDS = 30
VCURVE_R_MIN = float(os.environ.get("TD_VCURVE_R", "600"))  # m: sharpest vertical curve allowed after the rounds (inf = off)
GRADE_SD_ACTIVE = 0.005 # sd of the pull to the limit on a step that breaks it (one-sided, as the caps)
BRIDGE_APPROACH_GRADE = float(os.environ.get("TD_BRIDGE_GRADE", "0.06"))  # limit near a bridge over water
APPROACH_M = 800.0       # m along the track from a span over water counted as its approach
# Tunnel prior follows the surface's lower envelope at this grade (inf = off). Held-out sweep
# 2026-11-02: off 58%/77%, 3% 60/81, 2% 63/81, 1.2% 60/79, 0.7% 55/77 (inside 1 / 2 sd).
ENVELOPE_SLOPE = float(os.environ.get("TD_ENVELOPE_SLOPE", "0.02"))
RIVER_APPROACH_SLOPE = float(os.environ.get("TD_RIVER_SLOPE", "0.035"))  # Joralemon 3.1%, Steinway/Clark ~3-4.5%
PROFILE_RADIUS = 30.0     # m: a drawing point pins its line's tracks this close
LIDAR_SOFT_SD = 2.0      # m: sd for template-copied and interpolated lidar samples
LIDAR_OUTLIER_M = 1.5    # m off the running median of its 7 neighbours along the way -> dropped
LIDAR_SD_FLOOR = 0.4     # m: samples 30 m apart share offset errors (deck-to-rail), so no tighter
HIDDEN_DECK_R = 4.0      # m in plan: an el of a higher OSM level this close covers the track below
HIDDEN_DECK_GAP = 4.5    # m: the lower deck's rail is at least this far under the surface read above it
PORTAL_TAPER = float(os.environ.get("TD_PORTAL_TAPER", "400"))  # m: tunnel prior and cover blend to the open track's over this
PORTAL_GROUND_M = float(os.environ.get("TD_PORTAL_GROUND", "75"))  # m: open track this near a tunnel takes the approach's ground envelope
CANOPY_DZ = 1.5        # m above its station level's median reading: a platform roof, not the rail
CANOPY_GAP = 2.5         # m: the rail is at least this far under a canopy read
BED_MIN_PTS = 4           # +-100 m: deepest bed nearby
BED_MEAN_PTS = 8          # +-200 m: then averaged
BELOW_BED = (10.0, 5.0)# river tunnel rail below the riverbed (m), sd: Clark St ~11, Joralemon ~13
BED_COVER = 5.0           # rail at least this far under the riverbed (tube height plus a little cover)
# placed_rows.csv ids to leave out. TD0025+TD0155 (Canal St "lower level", 40 ft below MHW): the
# river agent couldn't tell the Nassau level from the Manhattan Bridge branch; read as the bridge
# branch it needs a 6% grade down from the bridge, so it is almost surely the Nassau level.
# TD0025+TD0155 (Canal St 1914, lower level 40 ft below MHW) was skipped for implying a 6% grade
# down from the bridge, but that grade came from roadway lidar (fix_template_upper_deck); it agrees
# with the 55 ft bridge platform (TD0086) within 3 m, so it is used again as the bridge level.
SKIP_PLACED = set()
MIN_COVER = 4.0     # m: shallowest rail below street in a tunnel (roof right under the pavement)
UNDERWATER_CAP = -12.0  # m NAVD88: river tunnels sit at least this deep (no bathymetry yet)
PRIOR_SMOOTH = 150.0  # m: the prior is diffused along the track over about this length


# ---------- geometry ----------

# Passenger rail beyond the subway, out to the map's square (fetch_osm.SQUARE; until 2026-11-02 only
# where it touched NYC + 800 m): main-line tracks only (no yards, sidings or spurs), no freight, no
# light rail but the two AirTrains. PATH is kept whole, Newark included.
RAIL_OPERATORS = ("Long Island Rail Road", "Metro-North", "Amtrak", "New Jersey Transit", "NJ Transit")
FREIGHT_OPS = ("Consolidated Rail", "Conrail", "CSX", "Norfolk Southern", "New York and Atlantic", "Canadian Pacific")
ISOLATED_M = 2000.0     # m: a piece of track this short joined to nothing ...
ISOLATED_NEAR_M = 150.0 # m: ... and with no station or stop this close is dropped


def load_tracks():
    els = json.loads((RAW / "osm_tracks.json").read_text(encoding="utf-8"))["elements"]
    keep = []
    for e in els:
        t = e["tags"]
        op = t.get("operator", "")
        rw = t.get("railway")
        if rw == "subway" and ("Port Authority" in op or t.get("operator:wikidata") == "Q908666"):
            # PATH: OSM's operator went from "Port Authority Trans-Hudson" to the Port Authority itself
            # (late 2026); keep the old name so everything keyed on it still finds PATH
            t["operator"] = op = "Port Authority Trans-Hudson"
        if rw == "subway" and ("Transit Authority" in op or "Staten Island" in op or "Trans-Hudson" in op):
            keep.append(e)
        elif rw == "rail" and "Staten Island" in (t.get("name", "") + op) \
                and not any(f in op for f in FREIGHT_OPS) and t.get("usage") != "freight":
            # (Conrail's "Staten Island Railroad" freight line in Elizabeth, NJ matched the name, 2026-11-03)
            keep.append(e)
        elif os.environ.get("TD_SUBWAY_ONLY"):   # calibration check: the network before other rail went in
            continue
        elif ((rw == "rail" and any(o in op for o in RAIL_OPERATORS) and t.get("usage") in ("main", "branch"))
              or (rw in ("light_rail", "monorail") and ("AirTrain" in t.get("name", "") + t.get("network", "") or op == "Alstom"))) \
                and t.get("service") in (None, "crossover") and len(e["geometry"]) > 1:
            # out to the map's square (raw/osm_tracks.json is already cut at its edge by fetch_osm.py)
            keep.append(e)
    if os.environ.get("TD_SUBWAY_ONLY"):
        return separate_dead_ends(drop_isolated(keep))
    # track a commuter or intercity train runs on, whoever operates it: the Raritan Valley Line runs
    # from Newark on the Lehigh Line, which OSM gives to Conrail (members of the passenger route
    # relations fetch_routes.py downloaded)
    rr = RAW / "osm_routes_rail_rels.json"
    if rr.exists():
        pax = set()
        for r in json.loads(rr.read_text(encoding="utf-8"))["elements"]:
            tg = r.get("tags", {})
            if r["type"] == "relation" and tg.get("route") == "train" and any(
                    n in tg.get("network", "") for n in ("NJ Transit", "LIRR", "Metro-North", "Amtrak")):
                pax.update(m["ref"] for m in r.get("members", []) if m["type"] == "way")
        ids = {e["id"] for e in keep}
        # Terminal platform tracks are often tagged service=yard (Long Beach, Port Washington and Far
        # Rockaway on the LIRR, like St George on the SIR), so the terminals had no track and no dot
        # (2026-11-03). A railroad's yard or siding way is kept when a passenger route lists it or it
        # carries a train stop position.
        st_els = json.loads((RAW / "osm_stations.json").read_text(encoding="utf-8"))["elements"]
        train_stops = {key(s["lon"], s["lat"]) for s in st_els if s["type"] == "node"
                       and s.get("tags", {}).get("public_transport") == "stop_position" and s["tags"].get("train") == "yes"}
        for e in els:
            t = e["tags"]
            g = e.get("geometry") or []
            if e["id"] in ids or t.get("railway") != "rail" or len(g) < 2:
                continue
            svc = t.get("service")
            if e["id"] in pax and svc in (None, "crossover", "yard", "siding"):
                keep.append(e); ids.add(e["id"])
            elif svc in ("yard", "siding") and any(o in t.get("operator", "") for o in RAIL_OPERATORS) \
                    and any(key(p["lon"], p["lat"]) in train_stops for p in g):
                keep.append(e); ids.add(e["id"])
    # main-line pieces with no operator tag that join kept passenger track at both ends (the Grand
    # Central throat, Harold interlocking, a few NJ Transit links): without them the lines had gaps
    ends = set()
    for e in keep:
        if e["tags"].get("railway") == "rail":
            g = e["geometry"]
            ends.update(key(p["lon"], p["lat"]) for p in g)
    ids = {e["id"] for e in keep}
    for e in els:
        t = e["tags"]
        g = e.get("geometry") or []
        if e["id"] in ids or t.get("railway") != "rail" or t.get("operator") or len(g) < 2 \
                or t.get("usage") not in ("main", "branch") or t.get("service") not in (None, "crossover"):
            continue
        if key(g[0]["lon"], g[0]["lat"]) in ends and key(g[-1]["lon"], g[-1]["lat"]) in ends:
            keep.append(e)
    return separate_dead_ends(drop_isolated(keep))


def drop_isolated(ways):
    """Drop connected pieces of track (ways joined at shared vertices) shorter than ISOLATED_M with no
    railway station or stop position within ISOLATED_NEAR_M: scraps the filters leave, joined to nothing
    (a freight stub of the LIRR's C Secondary, yard leads by 207 St). Pieces at a station stay: Grand
    Central Terminal's platform tracks are cut off from their throat by the service filter. Counted in
    drop_isolated.n / .m."""
    parent = list(range(len(ways)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    at = collections.defaultdict(list)
    for wi, e in enumerate(ways):
        for p in e["geometry"]:
            at[key(p["lon"], p["lat"])].append(wi)
    for lst in at.values():
        for b in lst[1:]:
            parent[find(b)] = find(lst[0])
    comp = collections.defaultdict(list)
    for wi in range(len(ways)):
        comp[find(wi)].append(wi)
    st = json.loads((RAW / "osm_stations.json").read_text(encoding="utf-8"))["elements"]
    sxy = np.array([to_utm.transform(*((e["lon"], e["lat"]) if "lon" in e else (e["center"]["lon"], e["center"]["lat"])))
                    for e in st if e.get("tags", {}).get("railway") in ("station", "halt", "stop")
                    or e.get("tags", {}).get("public_transport") == "stop_position"])
    stree = cKDTree(sxy)
    drop = set()
    for ws in comp.values():
        xy = np.array([to_utm.transform(p["lon"], p["lat"]) for wi in ws for p in ways[wi]["geometry"]])
        L = sum(float(np.hypot(*np.diff(np.array([to_utm.transform(p["lon"], p["lat"]) for p in ways[wi]["geometry"]]), axis=0).T).sum())
                for wi in ws)
        if L < ISOLATED_M and stree.query(xy)[0].min() > ISOLATED_NEAR_M:
            drop.update(ws)
    drop_isolated.n, drop_isolated.m = len(drop), len({find(w) for w in drop})
    return [e for wi, e in enumerate(ways) if wi not in drop]


TAIL_SEP = 3.6     # m: dead-end tracks of one level drawn into one bumper node are cut back to this far apart
TAIL_MAX_CUT = 80.0  # m: no more than this is cut off a track (ways drawn on top of each other longer are only parted)


def separate_dead_ends(ways):
    """Tracks that OSM draws into one end node with nothing beyond it are separate dead ends.

    Grand Central Madison's tail tracks south of the platforms come together in pairs at one bumper
    node on each level (201 with 202, 203 with 204, and the same on the 300 level), so the model drew
    them as V shapes meeting at a point; no train can use such a join, since nothing leaves the node on
    the other side (2026-11-02). Where two or more ways end at one vertex, every one of them leaves it
    on the same side and none runs through it, each way of a level shared with another is cut back
    to where it is TAIL_SEP from the others of that level (at most TAIL_MAX_CUT), so they end side by
    side as real bumpers do; ways of different levels there are parted by build_points. Counted in
    separate_dead_ends.n_nodes / n_cut."""
    to_xy = lambda p: to_utm.transform(p["lon"], p["lat"])
    at = collections.defaultdict(list)   # vertex -> [(way index, is end, which end)]
    for wi, e in enumerate(ways):
        g = e["geometry"]
        for i, p in enumerate(g):
            at[key(p["lon"], p["lat"])].append((wi, i in (0, len(g) - 1), 0 if i == 0 else -1))
    n_nodes = n_cut = 0
    nodes = []
    from shapely.geometry import LineString
    for k, lst in at.items():
        if len(lst) < 2 or not all(e for _, e, _ in lst):
            continue
        dirs = []
        for wi, _, end in lst:
            g = ways[wi]["geometry"]
            if len(g) < 2:
                break
            p0, p1 = (g[0], g[1]) if end == 0 else (g[-1], g[-2])
            v = np.array([(p1["lon"] - p0["lon"]) * 84300, (p1["lat"] - p0["lat"]) * 111000])
            dirs.append(v / (np.hypot(*v) or 1.0))
        if len(dirs) != len(lst) or not all(a @ b > 0 for a in dirs for b in dirs):
            continue
        n_nodes += 1
        nodes.append((k[0], k[1], sorted({ways[wi]["tags"].get("name", "") for wi, _, _ in lst}),
                      sorted({ways[wi]["tags"].get("service") or "" for wi, _, _ in lst}), len(lst)))
        by_lv = collections.defaultdict(list)
        for wi, _, end in lst:
            t = ways[wi]["tags"]
            by_lv[level_of(t) if classify(t) == "tunnel" else None].append((wi, end))
        for lv, grp in by_lv.items():
            if len(grp) < 2:
                continue
            lines = {wi: LineString([to_xy(p) for p in ways[wi]["geometry"]]) for wi, _ in grp}
            cuts = {}
            for wi, end in grp:
                ln = lines[wi]
                others = [lines[o] for o, _ in grp if o != wi]
                s_cut = None
                for s in np.arange(0.0, min(TAIL_MAX_CUT, ln.length - 5.0) + 0.01, 1.0):
                    q = ln.interpolate(s if end == 0 else ln.length - s)
                    if all(o.distance(q) >= TAIL_SEP for o in others):
                        s_cut = s
                        break
                cuts[wi] = s_cut
            for wi, end in grp:
                s = cuts[wi]
                if not s:
                    continue   # already apart, or drawn together too long to cut (only parted below)
                ln = lines[wi]
                c_ = np.array(ln.coords)
                d = np.r_[0, np.cumsum(np.hypot(np.diff(c_[:, 0]), np.diff(c_[:, 1])))]
                g = ways[wi]["geometry"]
                q = ln.interpolate(s if end == 0 else ln.length - s)
                lon_, lat_ = to_ll.transform(q.x, q.y)
                newp = {"lon": float(lon_), "lat": float(lat_)}
                if end == 0:
                    rest = [p for p, dd in zip(g, d) if dd > s + 0.5]
                    g2 = [newp] + rest
                else:
                    rest = [p for p, dd in zip(g, d) if dd < ln.length - s - 0.5]
                    g2 = rest + [newp]
                if len(g2) >= 2:
                    ways[wi] = {**ways[wi], "geometry": g2}
                    n_cut += 1
            # ways of this level still ending on the shared node (drawn together too long to cut): moved
            # apart by a centimetre each so each is its own dead end
            still = [(wi, end) for wi, end in grp
                     if key(ways[wi]["geometry"][end]["lon"], ways[wi]["geometry"][end]["lat"]) == k]
            for r, (wi, end) in enumerate(still[1:], 1):
                g = list(ways[wi]["geometry"])
                p = dict(g[end]); p["lon"] += 1e-7 * r
                g[end] = p
                ways[wi] = {**ways[wi], "geometry": g}
    separate_dead_ends.n_nodes, separate_dead_ends.n_cut, separate_dead_ends.nodes = n_nodes, n_cut, nodes
    return ways


def key(lon, lat):
    return (round(lon, 7), round(lat, 7))


def classify(t):
    # tunnel=building_passage is track under a building or a train shed: underground only where a
    # negative level or layer says so (Hoboken Terminal's shed tracks were pushed 14 m down as tunnel)
    bp_under = t.get("tunnel") == "building_passage" and (level_of(t) or 0) < 0
    tun = t.get("tunnel") in ("yes", "culvert") or bp_under
    br = t.get("bridge") in ("yes", "viaduct", "movable", "trestle")
    if tun:
        return "tunnel"
    if br:
        return "elevated"
    if t.get("embankment") == "yes":
        return "embankment"
    if t.get("cutting") == "yes":
        return "cutting"
    return "grade"


def level_of(t):
    for k in ("level", "layer"):
        v = t.get(k)
        if v is None:
            continue
        try:
            return int(float(v.split(";")[0]))
        except ValueError:
            pass
    return None


def build_points(ways):
    """Split ways at shared vertices, resample. Returns point arrays and per-way index lists."""
    count = collections.Counter()
    for w in ways:
        g = w["geometry"]
        ks = [key(p["lon"], p["lat"]) for p in g]
        for k in set(ks):
            count[k] += 1
        count[ks[0]] += 1
        count[ks[-1]] += 1
    # A vertex two ways share is one point of track, except where OSM draws two levels on top of
    # each other: along Central Park West the lower (southbound) tracks share nearly every vertex
    # with the upper (northbound) ones, and each shared vertex used to force the two levels to one
    # height, so the model drew them meeting at every vertex and 5 m apart between (2026-11-02).
    # Where a vertex lies inside (not at an end of) tunnel ways of two or more OSM levels, each of
    # those levels gets its own point; a way that ends there joins its own level's point, and ways
    # ending there whose level runs through no other way there share one point (a level change
    # along a track, as at 110 St). A vertex where ways only meet end to end stays one point.
    # Ends only: one point, unless every way leaves the vertex on the same side (nothing continues
    # through it). Grand Central Madison's upper and lower platform tracks all end at one bumper node
    # at 44 St, which made the two levels meet at one height there (2026-11-02).
    inc = collections.defaultdict(list)   # vertex -> [(way index, is_end, level or None)]
    away = {}                             # (vertex, way index) -> unit vector from the vertex along the way
    for wi, w in enumerate(ways):
        g = w["geometry"]
        lvw = level_of(w["tags"]) if classify(w["tags"]) == "tunnel" else None
        for i, p in enumerate(g):
            k = key(p["lon"], p["lat"])
            inc[k].append((wi, i == 0 or i == len(g) - 1, lvw))
            if (i == 0 or i == len(g) - 1) and len(g) > 1:
                q = g[1] if i == 0 else g[-2]
                v = np.array([(q["lon"] - p["lon"]) * 84300, (q["lat"] - p["lat"]) * 111000])
                away[(k, wi)] = v / (np.hypot(*v) or 1.0)
    group = {}   # (vertex, way index) -> vertex group key
    n_split = 0
    build_points.n_through = 0
    for k, lst in inc.items():
        inner = sorted({lv for _, e, lv in lst if not e and lv is not None}, reverse=True)
        levels = {lv for _, _, lv in lst if lv is not None}
        if len(levels) >= 2 and not inner:
            vs = [away.get((k, wi)) for wi, _, lv in lst if lv is not None]
            if all(v is not None for v in vs) and all(a @ b > 0 for a in vs for b in vs):
                n_split += 1
                for wi, e, lv in lst:
                    group[(k, wi)] = (k, lv if lv is not None else max(levels))
                continue
            # Ends only, and each level runs through on its own: Grand Central Madison's upper and lower
            # platform tracks (204 and 304, 203 and 303) are drawn on top of each other and their ways
            # meet end to end at shared nodes at the platforms' south end, which pinched the two levels
            # 15 m apart to one height there and sloped both platforms (2026-11-02). Where every level
            # there has two or more ways and some leave on opposite sides, each level gets its own point.
            per = collections.defaultdict(list)
            for wi, _, lv in lst:
                if lv is not None:
                    per[lv].append(away.get((k, wi)))
            if all(len(v) >= 2 and all(u is not None for u in v) and any(a @ b < 0 for a in v for b in v)
                   for v in per.values()):
                n_split += 1
                build_points.n_through += 1
                for wi, e, lv in lst:
                    group[(k, wi)] = (k, lv if lv is not None else max(levels))
            continue
        if len(levels) < 2 or not inner:
            continue
        n_split += 1
        for wi, e, lv in lst:
            if lv in inner:
                group[(k, wi)] = (k, lv)
            elif lv is None:
                group[(k, wi)] = (k, inner[0])
            else:
                group[(k, wi)] = (k, "end")
    node_var = {}
    xs, ys, way_of = [], [], []
    seqs = []

    def var_for_node(k, x, y, wi):
        k = group.get((k, wi), k)
        if k not in node_var:
            node_var[k] = len(xs)
            xs.append(x); ys.append(y); way_of.append(wi)
        return node_var[k]

    for wi, w in enumerate(ways):
        g = w["geometry"]
        lon = np.array([p["lon"] for p in g]); lat = np.array([p["lat"] for p in g])
        x, y = to_utm.transform(lon, lat)
        ks = [key(a, b) for a, b in zip(lon, lat)]
        seq = []
        cut = [i for i, k in enumerate(ks) if i == 0 or i == len(ks) - 1 or count[k] > 1]
        for a, b in zip(cut, cut[1:]):
            sx, sy = x[a:b + 1], y[a:b + 1]
            d = np.concatenate([[0], np.cumsum(np.hypot(np.diff(sx), np.diff(sy)))])
            L = d[-1]
            n = max(1, int(round(L / STEP)))
            if not seq:
                seq.append(var_for_node(ks[a], x[a], y[a], wi))
            for s in np.linspace(0, L, n + 1)[1:-1]:
                seq.append(len(xs))
                xs.append(float(np.interp(s, d, sx))); ys.append(float(np.interp(s, d, sy))); way_of.append(wi)
            seq.append(var_for_node(ks[b], x[b], y[b], wi))
        seqs.append(seq)
    build_points.n_split = n_split
    return np.array(xs), np.array(ys), np.array(way_of), seqs


# ---------- ground ----------

def sample_ground(xs, ys):
    z = np.full(len(xs), np.nan)
    for f in sorted((RAW / "dem").glob("dem_*.tif")):
        with rasterio.open(f) as ds:
            b = ds.bounds
            m = (xs >= b.left) & (xs < b.right) & (ys > b.bottom) & (ys <= b.top) & np.isnan(z)
            if not m.any():
                continue
            arr = ds.read(1)
            r, c = rasterio.transform.rowcol(ds.transform, xs[m], ys[m])
            r = np.clip(np.asarray(r), 0, arr.shape[0] - 1); c = np.clip(np.asarray(c), 0, arr.shape[1] - 1)
            v = arr[r, c].astype(float)
            v[v < -100] = np.nan
            z[m] = v
    return z


SURFACE_OFFSETS = (25.0, 40.0)   # m either side of the track where the surface beside it is read


def surface_ground(xs, ys, seqs):
    """The surface beside the track: the median of the bare-earth DEM at SURFACE_OFFSETS m to each side,
    across the track. The bare earth keeps solid structure: the Hell Gate line's masonry approach in
    Astoria read 0-1 m under its rail, so the Amtrak there showed as level with the ground where it
    crosses over the Astoria el (10 m up) at Ditmars Blvd; the floor of an open cut reads the floor; a
    causeway over Jamaica Bay reads its own top (2026-11-03). Beside a viaduct, cut or embankment the
    DEM is the street. Where fewer than two offsets have ground (water), the DEM at the track."""
    tang = np.zeros((len(xs), 2))
    for seq in seqs:
        for a, b in zip(seq, seq[1:]):
            v = np.array([xs[b] - xs[a], ys[b] - ys[a]]); v /= (np.hypot(*v) or 1.0)
            tang[a] += v; tang[b] += v
    tn = np.hypot(tang[:, 0], tang[:, 1]); tn[tn == 0] = 1.0
    nx, ny = -tang[:, 1] / tn, tang[:, 0] / tn
    samples = []
    for d in SURFACE_OFFSETS:
        for s in (1, -1):
            samples.append(sample_ground(xs + s * d * nx, ys + s * d * ny))
    S = np.array(samples)
    S[S < -0.5] = np.nan   # water (bathymetry or the DEM's fill): not a surface to stand on
    ok = np.isfinite(S).sum(axis=0) >= 2
    out = sample_ground(xs, ys)
    with np.errstate(all="ignore"):
        med = np.nanmedian(S, axis=0)
    out[ok] = med[ok]
    return out


def smooth_along(seqs, v, half=3):
    """Running median of v along each way (ground has kerbs, walls, noise)."""
    out = v.copy()
    for seq in seqs:
        a = v[seq]
        for i, idx in enumerate(seq):
            w = a[max(0, i - half):i + half + 1]
            w = w[~np.isnan(w)]
            if len(w):
                out[idx] = np.median(w)
    return out


# ---------- anchors from elevators (station_levels.py -> evidence/station_levels.csv) ----------

# MTA stations-table line -> words that appear in the OSM way names of that line's tracks.
LINE_WORDS = {
    "4th Av": ["Fourth Avenue"], "Astoria": ["Astoria"], "Broadway": ["BMT Broadway"],
    "Broadway - Brighton": ["BMT Broadway", "Brighton"], "Canarsie": ["Canarsie"],
    "Franklin Shuttle": ["Franklin"], "Jamaica": ["Jamaica Line", "Nassau", "Archer"],
    "Manhattan Bridge": ["Fourth Avenue", "BMT Broadway", "Brighton", "Sixth Avenue"],
    "Myrtle Av": ["Myrtle"], "Sea Beach": ["Sea Beach"], "West End": ["West End"],
    "Sea Beach / West End / Culver / Brighton": ["Sea Beach", "West End", "Culver", "Brighton"],
    "63rd St": ["63rd"], "6th Av - Culver": ["Sixth Avenue", "Culver", "Chrystie", "63rd"],  # OSM: 57 St (F) is on the 63rd St Line
    "8th Av - Fulton St": ["Eighth Avenue", "Fulton Street"], "Concourse": ["Concourse"],
    "Crosstown": ["Crosstown"], "Liberty Av": ["Fulton Street", "Liberty"],
    "Queens - Archer": ["Archer"], "Queens Blvd": ["Queens Boulevard"], "Rockaway": ["Rockaway"],
    "Second Av": ["Second Avenue"], "Broadway - 7Av": ["Broadway–Seventh", "Broadway-Seventh"],
    "Clark St": ["Broadway–Seventh", "Broadway-Seventh", "Clark"], "Dyre Av": ["Dyre"],
    "Eastern Pky": ["Eastern Parkway", "New Lots"],   # OSM: Sutter Av-Rutland Rd to New Lots Av is the New Lots Line
    "Flushing": ["Flushing"], "Jerome Av": ["Jerome"],
    "Lenox - White Plains Rd": ["Lenox", "White Plains"], "Lexington - Shuttle": ["42nd Street", "Lexington"],
    "Lexington Av": ["Lexington"], "Nostrand": ["Nostrand"], "Pelham": ["Pelham"],
    "Staten Island": ["Staten Island"],
}


ANCHOR_SD_FLOOR = 0.75
LEVEL_GAP = 2.0  # m: platforms of one station and line further apart than this are different levels


def station_anchors():
    """Absolute platform heights per (station, line) from evidence/station_levels.csv.

    station_levels.py builds them from the MTA elevator and escalator inventory: chains of rises
    from a street-end machine, whose street height is the DEM ground at that machine. Side
    platforms of one station and line are combined. Where one station and line has platforms on
    two levels (Lexington Av/63 St, 59 St 4/5 vs 6, Jamaica Center E vs J/Z), each level is
    kept as its own group; solve() pins a group only when the OSM tracks near the station carry
    exactly as many distinct `level` values, deeper group to deeper tracks.
    """
    lv = pd.read_csv(ROOT / "evidence" / "station_levels.csv", dtype={"station_mrn": str})
    lv = lv[lv["level_name"].str.contains("platform", na=False)]
    st = pd.read_csv(RAW / "mta_stations.csv", dtype=str)
    st["Station ID"] = st["Station ID"].str.lstrip("0")
    st = st.set_index(["Station ID", "Line"])
    rows = []
    for (mrn, line), g in lv.groupby(["station_mrn", "line"]):
        s = st.loc[(mrn, line)]
        g = g.sort_values("z_navd88_m")
        # split into levels at gaps wider than LEVEL_GAP
        groups, cur = [], [g.iloc[0]]
        for _, r in list(g.iterrows())[1:]:
            if r["z_navd88_m"] - cur[-1]["z_navd88_m"] > LEVEL_GAP:
                groups.append(cur); cur = []
            cur.append(r)
        groups.append(cur)
        for k, grp in enumerate(groups):
            z = np.array([r["z_navd88_m"] for r in grp]); sd = np.array([r["sd_m"] for r in grp])
            w = 1 / sd ** 2
            zm = float((w * z).sum() / w.sum())
            # side platforms share most of their chain, so their errors are not independent:
            # keep the best single sd, widened by any spread between them
            sdm = max(float(sd.min()), float(z.max() - z.min()) / 2, ANCHOR_SD_FLOOR)
            rows.append({
                "station": s["Stop Name"], "line": line, "station_mrn": mrn,
                "lon": float(s["GTFS Longitude"]), "lat": float(s["GTFS Latitude"]),
                "platform_z": round(zm, 2), "sd": round(sdm, 2),
                "depth": float(np.median([r["depth_below_local_street_m"] for r in grp])),
                "levels": " / ".join(r["level_name"] for r in grp),
                "level_group": k, "n_groups": len(groups),
            })
    return rows


# ---------- anchors from documents (evidence/depths.csv) ----------

DOC_MEASURES = {"platform_below_street": (1, PLATFORM_ABOVE_RAIL), "rail_below_street": (1, 0.0),
                "rail_above_street": (-1, 0.0)}   # sign of "down", metres added to reach the rail
DOC_SD = {"high": 1.5, "medium": 2.5, "low": 4.0}  # m, plus the half-width of a quoted range
DOC_RADIUS = 100.0   # m from the row's coordinates in which its track points are looked for
FT = 0.3048
ABSOLUTE_MEASURES = {"rail_elevation_datum", "tunnel_invert_below_water", "riverbed_depth"}
# Where OSM names a line differently from the row: (text in the row's line or place, words that
# appear in the OSM way names). York St (F) is on the Culver Line, 57 St (F) and Grand St (B/D)
# are on the 63rd St Line and the Chrystie St Connection in OSM, though the rows say Sixth Avenue.
DOC_LINE_WORDS = [("manhattan bridge", ["Brighton", "Fourth Avenue", "Broadway", "Sixth Avenue"]),
                  ("york street", ["Culver"]), ("57th street", ["63rd Street"]), ("grand street", ["Chrystie"])]


def norm_line(s):
    s = re.sub(r"\(.*?\)", " ", s.lower())
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(re.sub(r"\b(line|subway|shuttle)\b", " ", s).split())


def line_matches(row, osm_name):
    if not osm_name:
        return False
    for pat, words in DOC_LINE_WORDS:
        if pat in row["line"].lower() or pat in row["place"].lower():
            return any(w in osm_name for w in words)
    a, b = norm_line(row["line"]), norm_line(osm_name)
    return bool(a and b and (a in b or b in a))


def parse_value(v, unit):
    """'90-100' -> (95, 5) in metres; '~100' -> (100, 0); '>40', '<=29', '105/135' -> None."""
    v = str(v).strip()
    if v.startswith((">", "<")):
        return None
    v = v.lstrip("~")
    m = re.match(r"^([\d.]+)\s*-\s*([\d.]+)$", v)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        mid, half = (a + b) / 2, (b - a) / 2
    else:
        try:
            mid, half = float(v), 0.0
        except ValueError:
            return None
    k = FT if unit.startswith("f") else 1.0
    return mid * k, half * k


def level_hint(row):
    """'deep' / 'shallow' when the row names one level of a multi-level place, else None."""
    t = (row["place"] + " " + row["line"]).lower()
    if re.search(r"\blower\b|express|manhattan bridge", t):
        return "deep"
    if re.search(r"\bupper\b|\blocal\b|mainline", t):
        return "shallow"
    return None


def doc_rows():
    """Rows of depths.csv that say how deep a rail or platform is below (above) the street.

    Each has `depth` (m, rail below street, negative above), `half` (range half-width), `sd` (m, by
    confidence plus half), and `why` when the row is not usable as an anchor."""
    ev = pd.read_csv(ROOT / "evidence" / "depths.csv", dtype=str).fillna("")
    out = []
    for _, r in ev.iterrows():
        if r["measure"] not in DOC_MEASURES:
            continue
        sign, extra = DOC_MEASURES[r["measure"]]
        row = {"id": r["id"], "line": r["line"], "place": r["place"], "conf": r["confidence"],
               "lat": float(r["lat"]) if r["lat"] else None, "lon": float(r["lon"]) if r["lon"] else None,
               "measure": r["measure"], "depth": None, "half": 0.0, "sd": None, "why": None}
        pv = parse_value(r["value"], r["unit"] or "ft")
        if pv is not None:
            row["depth"] = sign * pv[0] + extra
            row["half"] = pv[1]
            row["sd"] = DOC_SD.get(r["confidence"], DOC_SD["low"]) + pv[1]
        row["hint"] = level_hint(row)
        row["design"] = ("Second Avenue" in r["line"] and "2004" in r["line"] + r["source_title"]
                         and "FEIS" in r["line"] + r["source_title"] and "design" in r["line"])
        if row["lat"] is None:
            row["why"] = "no coordinates"
        elif pv is None:
            row["why"] = f"value '{r['value']}' is a bound or two figures"
        out.append(row)
    # the 2004 Second Avenue design is dropped wherever an as-built row exists for the same station
    for row in out:
        if row["design"] and row["why"] is None:
            for o in out:
                if (not o["design"] and "Second Avenue" in o["line"] and o["lat"] is not None
                        and math.hypot((o["lat"] - row["lat"]) * 111000,
                                       (o["lon"] - row["lon"]) * 84000) < 150):
                    row["why"] = f"2004 design; as-built row {o['id']} exists"
                    break
    return out


def place_doc_rows(docs, xs, ys, names, levels, clss, services):
    """For each row: (point indices, OSM level, reason it could not be placed).

    Only points of the row's line, of a kind that fits (tunnel or cutting for a rail below the
    street, elevated or embankment for one above it), not yard or crossover track, within DOC_RADIUS.
    Where that line has several OSM `level` values there, the row's level is picked by order: a row
    naming the lower level or an express platform gets the lowest OSM level, one naming the upper
    level or local platform the highest, and a row naming none is not placed."""
    out = []
    for row in docs:
        if row["lat"] is None or row["depth"] is None:
            out.append(([], None, row["why"] or "no coordinates"))
            continue
        ax, ay = to_utm.transform(row["lon"], row["lat"])
        d = np.hypot(xs - ax, ys - ay)
        dep = row["depth"]
        if dep > 0:
            ok = {"tunnel", "cutting"} | ({"grade"} if dep < 8 else set())
        else:
            ok = {"elevated", "embankment"} | ({"grade"} if dep > -8 else set())
        cand = [i for i in np.where(d < DOC_RADIUS)[0]
                if clss[i] in ok and not services[i] and line_matches(row, names[i])]
        if not cand:
            out.append(([], None, f"no OSM track of this line and kind within {DOC_RADIUS:.0f} m"))
            continue
        lv = sorted({levels[i] for i in cand if levels[i] is not None})
        lvl = lv[0] if lv else None
        if len(lv) > 1:
            if row["hint"] == "deep":
                lvl = lv[0]
            elif row["hint"] == "shallow":
                lvl = lv[-1]
            else:
                out.append(([], None, f"OSM levels {lv} there and the row names no level"))
                continue
            cand = [i for i in cand if levels[i] == lvl]
        out.append((cand, lvl, None))
    return out


# ---------- solve ----------

STATION_RADIUS = 90.0   # m around a station's GTFS point counted as its platform
STATION_FLOOR = (4.6, 7.0)   # m of rail below the street at an underground station: fare control at the street / a mezzanine
# rail against the surface beside the track at MTA 'Open Cut' / 'At Grade' / 'Embankment' stations (m, sd).
# Open cut: the platform a flight of stairs under the street bridges (rail about 5.5 m under the street
# beside the cut); at grade +0.5 over the track's own ground (ballast); embankment +0.3 over the DEM at the
# track, which keeps the fill
STRUCTURE_PRIOR = {"Open Cut": (-5.5, 2.0), "At Grade": (0.5, 1.0), "Embankment": (0.3, 1.5)}
PLAT_TIE_R = 6.5        # m from an OSM platform's outline to a track beside it
PLAT_TIE_SD = float(os.environ.get("TD_PLAT_TIE_SD", "0.3"))   # m: tracks beside one platform (0 = off)
STOP_NEAR_M = 150.0    # m: an OSM stop position this near a station marks its level as a platform level
LEVEL_Z_SAME = 2.5      # m: a station's levels closer than this in rail height are one level (one dot)
STATION_TIE_SD = 0.3   # m: tracks along one platform level are at one height
STACK_RADIUS = 4.0      # m in plan: tunnel points this close on different OSM levels are stacked
STACK_GRAPH_M = 300.0   # m along the track: closer than this, two levels are a ramp, not a stack
STACK_SEP = 5.0        # m: least rail-to-rail height between stacked tunnels (box + slab)
OLD_STACK = not os.environ.get("TD_CROSS_STACK")   # crossings off by default: tried 2026-11-02, hold-out worse (see NOTES)
PAR_STACK_R = float(os.environ.get("TD_PAR_STACK_R", "8"))  # m: same-line levels side by side this close are stacked
CROSS_MIN_SIN = float(os.environ.get("TD_CROSS_MIN_SIN", "0.42"))  # crossings held only at angles over ~25 degrees
# Each track carries its own prior, but parallel tracks share one prior error (the line was
# built at one depth). Once station ties join them, n tracks would count it n times; widening
# every track's prior by about sqrt(typical track count) undoes that.
# Grade-separated tunnel crossings (crossing_pairs): the lower structure's roof and the upper
# structure's invert must fit between the two rails. From typical sections, not fitted: NYC
# cut-and-cover box, rail to the top of the roof about 4.8 m (12-13 ft clear over the rail, 0.6-1 m roof
# of steel and concrete); a bored subway tube (5.2-5.6 m inside, rail about 1 m over the invert) 5.1 m to
# the outside of its crown; a main-line railroad tunnel 5.5 m (the Penn tubes: top of rail 17 ft 1 in
# under the inside of the crown flange, outside crown 18.0 ft = 5.5 m over the rail; Hewett and Brown
# 1910, as in make_placed_rows_rail.py). Under the upper rail: 1.1 m of track bed and invert slab for the
# subway, 1.3 m for a railroad. So box over box 5.9 m, a subway box over a subway tube 6.2 m, over a
# railroad tunnel 6.6 m, a railroad over a subway box 6.1 m. A first try (2026-11-02) also kept 1 m of
# ground over every tube and took 6.3 m for the railroad roof (8.4 m under a subway box): held-out
# coverage fell from 68% / 93% to 61% / 90%, mostly the Sixth Av subway at 34 St pushed 6 m up by the
# East River tunnels, whose own height there was wrong (RL0011, RL0012 added for it).
CROSS_ROOF = {"box": 4.8, "tube": 5.1, "railroad": 5.5}
CROSS_INVERT = {"box": 1.1, "tube": 1.1, "railroad": 1.3}
CROSS_TUBE_DEPTH = 18.0   # m of rail below the street: deeper subway is a rock tunnel or tube, not a box
CROSS_SD_ORDERED = 0.8    # m: different OSM levels say which is on top (clear-cut)
CROSS_SD_UNORDERED = 1.5  # m: same OSM level, order taken from the solution without the rule
CROSS_GAP_SD = 1.2        # m: the needed gap is itself uncertain (sampled in the error bars): typical sections, +-1-1.5 m
CROSS_ON = not os.environ.get("TD_NO_CROSS")
CROSS_OPEN = not os.environ.get("TD_NO_CROSS_OPEN")   # crossings of open track (els, bridges, embankments) too
# Open track underneath: rail to the underside of what passes over it. Subway cars are 3.7 m tall
# (clearance 4.4 m); third-rail railroads (LIRR, Metro-North's Hudson and Harlem lines) about 4.9 m;
# catenary (Amtrak, NJ Transit, the New Haven line: electrified=contact_line) about 6.7 m (22 ft);
# unelectrified track 6.1 m (20 ft). Plus the upper structure: CROSS_DECK (1.5 m, rail to the underside
# of a steel el or a bridge deck), or a tunnel's invert.
CROSS_CLEAR = {"subway": 4.4, "railroad": 4.9, "catenary": 6.7, "unelectrified": 6.1}
CROSS_DECK = 1.5


def crossing_gap(lower, upper):
    """Least rail-to-rail height (m) where a tunnel of kind `upper` passes over one of kind `lower`
    (kinds: 'box', 'tube', 'railroad')."""
    return CROSS_ROOF[lower] + CROSS_INVERT[upper]


PAR_TIE_R = 6.0        # m in plan: tracks of one line and level this close run in one box or cut
PAR_TIE_SD = float(os.environ.get("TD_PAR_TIE_SD", "0.5"))  # m per 25 m point (0 = off)
PAR_TIE_CLEAR = 200.0   # m along the track from a level change: no ties (a ramp diving away)
PARALLEL = 1.0  # tried 1.7: held-out coverage got worse (53% / 77%), so it stays off


def lower_envelope(seqs, xs, ys, g, slope):
    """Lowest surface reachable along the track at a grade of `slope`: env_i = min over points j
    of g_j + slope * (track distance i..j). A tunnel can't climb a hill faster than the trains
    can, so deep rock tunnels sit near this envelope while cut-and-cover ones follow the street.
    Relaxed over the track graph until it settles (water points have g ~ 0 and pull approaches
    down toward the river)."""
    a = np.array([p for s in seqs for p in zip(s, s[1:])], dtype=int)
    if not len(a) or not np.isfinite(slope):
        return g.copy()
    i, j = a[:, 0], a[:, 1]
    cost = slope * np.hypot(xs[j] - xs[i], ys[j] - ys[i])
    env = np.where(np.isnan(g), 0.0, g)
    for _ in range(2000):
        new = env.copy()
        np.minimum.at(new, j, env[i] + cost)
        np.minimum.at(new, i, env[j] + cost)
        if np.allclose(new, env):
            break
        env = new
    return env


WATER_NEAR_M = 60.0   # m: low ground this close to the water polygon is water (shores, piers, the DEM's edge)


def near_water(xs, ys, buffer=None):
    """True where a point lies inside data/water.json's water polygon or within WATER_NEAR_M (or
    `buffer`) m of it. Without the file, every point counts (the old rule: any ground below 1 m is water)."""
    p = OUT / "water.json"
    if not p.exists() or os.environ.get("TD_OLD_WET"):
        return np.ones(len(xs), dtype=bool)
    if buffer is not None and buffer != WATER_NEAR_M:
        return _near_water(xs, ys, buffer)
    return _near_water(xs, ys, WATER_NEAR_M)


def _near_water(xs, ys, buffer):
    p = OUT / "water.json"
    from shapely.geometry import shape
    from shapely.ops import transform, unary_union
    import shapely
    gj = json.loads(p.read_text(encoding="utf-8"))
    poly = unary_union([shape(f["geometry"]) for f in gj["features"]])
    poly = transform(lambda x, y, z=None: to_utm.transform(x, y), poly).buffer(buffer)
    shapely.prepare(poly)
    return np.asarray(shapely.contains_xy(poly, xs, ys))


def load_riverbed(xs, ys):
    """Riverbed height (m NAVD88, NOAA CUDEM) at track points over water, from
    evidence/riverbed.csv (make_riverbed.py); NaN elsewhere."""
    bed = np.full(len(xs), np.nan)
    p = ROOT / "evidence" / "riverbed.csv"
    if not p.exists() or os.environ.get("TD_NO_BED"):
        return bed
    rb = pd.read_csv(p)
    rb = rb[rb["notes"].isin(["water", "shore"])]
    bx, by = to_utm.transform(rb["lon"].to_numpy(), rb["lat"].to_numpy())
    d, i = cKDTree(np.c_[xs, ys]).query(np.c_[bx, by])
    ok = d < 8
    bed[i[ok]] = rb["bed_navd88_m"].to_numpy()[ok]
    return bed


def smooth_bed(seqs, bed):
    """Riverbed as a tunnel sees it: a tube is built on long straight grades, not following every
    scour hole and shoal. Along each way, take the deepest bed within +-BED_MIN_PTS points (so the
    tube clears every hole), then a running mean over +-BED_MEAN_PTS (so it doesn't jog)."""
    out = bed.copy()
    for seq in seqs:
        a = bed[seq]
        if np.isnan(a).all():
            continue
        lo = np.array([np.nanmin(a[max(0, k - BED_MIN_PTS):k + BED_MIN_PTS + 1]) if not np.isnan(a[max(0, k - BED_MIN_PTS):k + BED_MIN_PTS + 1]).all() else np.nan
                       for k in range(len(a))])
        for k, idx in enumerate(seq):
            if np.isnan(a[k]):
                continue
            w = lo[max(0, k - BED_MEAN_PTS):k + BED_MEAN_PTS + 1]
            out[idx] = np.nanmean(w)
    return out


def load_placed(kind):
    """evidence/placed_rows.csv (kind 'rows') or placed_bounds.csv ('bounds'): published figures
    turned into rail heights NAVD88 at a point, for named OSM ways (make_placed_rows.py)."""
    files = sorted((ROOT / "evidence").glob(f"placed_{kind}*.csv"))   # placed_rows.csv, placed_rows_rail.csv
    if not files or os.environ.get("TD_NO_PLACED"):
        return pd.DataFrame(columns=["id", "bound"])
    return pd.concat([pd.read_csv(p, dtype={"osm_way_ids": str}) for p in files], ignore_index=True)


PREV_OSM = RAW / "osm_tracks_2026-11-01.json"   # the download the evidence's way ids were written against
_prev_geom = None


def prev_way_xy(wid):
    """UTM vertices of a way in the earlier OSM download, or None."""
    global _prev_geom
    if _prev_geom is None:
        _prev_geom = {}
        if PREV_OSM.exists():
            for e in json.loads(PREV_OSM.read_text(encoding="utf-8"))["elements"]:
                if e.get("geometry"):
                    _prev_geom[e["id"]] = e["geometry"]
    g = _prev_geom.get(wid)
    if not g:
        return None
    x, y = to_utm.transform([p["lon"] for p in g], [p["lat"] for p in g])
    return np.c_[x, y]


def placed_points(r, ways, xs, ys, way_of):
    """Track points of a placed row's OSM ways within its radius of its point. Ways that are gone
    from OSM since the row was written (split or redrawn) are replaced by the ways now lying along
    their old geometry (within 5 m, from raw/osm_tracks_2026-11-01.json)."""
    ids = {int(v) for v in str(r["osm_way_ids"]).split()}
    have = {w["id"] for w in ways}
    gone = [v for v in ids if v not in have]
    if gone:
        from shapely.geometry import LineString, Point
        tree = cKDTree(np.c_[xs, ys])
        for wid in gone:
            old = prev_way_xy(wid)
            if old is None or len(old) < 2:
                continue
            line = LineString(old)
            # densify the old line every 10 m and take kept points within 5 m of it
            dens = [line.interpolate(s) for s in np.arange(0, line.length + 10, 10.0)]
            for i in {i for p in dens for i in tree.query_ball_point([p.x, p.y], 5.0)}:
                ids.add(ways[way_of[i]]["id"])
    wis = [wi for wi, w in enumerate(ways) if w["id"] in ids]
    ax, ay = to_utm.transform(float(r["lon"]), float(r["lat"]))
    rad = float(r.get("radius_m", 100) or 100)
    m = np.isin(way_of, wis) & (np.hypot(xs - ax, ys - ay) < rad)
    return [int(i) for i in np.where(m)[0]]


def stop_way_set(ways):
    """Indices of ways that carry an OSM stop position (public_transport=stop_position) on one
    of their vertices: the tracks trains stop on, as opposed to express tracks passing by."""
    els = json.loads((RAW / "osm_stations.json").read_text(encoding="utf-8"))["elements"]
    stops = {key(e["lon"], e["lat"]) for e in els
             if e["type"] == "node" and e.get("tags", {}).get("public_transport") == "stop_position"}
    return {wi for wi, w in enumerate(ways) if any(key(p["lon"], p["lat"]) in stops for p in w["geometry"])}


def stacked_pairs(xs, ys, cls, ways, way_of, seqs):
    """Stacked tunnels of different OSM levels: rows (upper terms, lower terms) of the constraint
    upper - lower >= STACK_SEP, each side a point or a point interpolated along a 25 m segment.

    Point pairs of different levels within STACK_RADIUS m in plan (two levels under one street).
    With TD_CROSS_STACK=1 also two segments of different levels that cross in plan at more than ~25
    degrees, held at the crossing point: point pairs miss most crossings (points are 25 m apart, so
    the nearest points of two crossing lines are usually 5-15 m apart; the 63rd St line passes
    through the Lexington Av express level at Lex/63 in the model). Tried 2026-11-02 and left off:
    the hold-out got worse (2 sd 89% -> 86%), because a crossing passes one line's error to the
    other and the active constraint makes both sure of it (Canal St Lexington Av was pushed up by
    the Manhattan Bridge tracks, Delancey St F pinned under the J). Pairs that are also close along
    the track (within STACK_GRAPH_M) are a ramp where two levels merge, not a stack, and are left
    out."""
    lv_of = np.full(len(xs), -999)
    for wi, seq in enumerate(seqs):
        lvw = level_of(ways[wi]["tags"])
        if classify(ways[wi]["tags"]) == "tunnel" and lvw is not None:
            for i in seq:
                if cls[i] == "tunnel":
                    lv_of[i] = max(lv_of[i], lvw) if lv_of[i] != -999 else lvw
    segs = []   # (a, b, way, level)
    for wi, seq in enumerate(seqs):
        lvw = level_of(ways[wi]["tags"])
        if classify(ways[wi]["tags"]) != "tunnel" or lvw is None:
            continue
        for a, b in zip(seq, seq[1:]):
            segs.append((a, b, wi, lvw))
    if not segs:
        return []
    sa = np.array([s[0] for s in segs]); sb = np.array([s[1] for s in segs])
    sw = np.array([s[2] for s in segs]); sl = np.array([s[3] for s in segs])
    ax_, ay_, bx_, by_ = xs[sa], ys[sa], xs[sb], ys[sb]
    mid = np.c_[(ax_ + bx_) / 2, (ay_ + by_) / 2]
    half = np.hypot(bx_ - ax_, by_ - ay_) / 2
    stree = cKDTree(mid)
    reach = STACK_RADIUS + float(half.max())
    cand = {}   # key -> (upper terms, lower terms, upper ref point, lower ref points)

    def add_c(up_terms, lo_terms, key):
        if key not in cand:
            cand[key] = (up_terms, lo_terms)

    # point near a segment of another level
    pts = np.array([i for i in range(len(xs)) if lv_of[i] != -999])
    tang = np.zeros((len(xs), 2))
    for seq in seqs:
        for a, b in zip(seq, seq[1:]):
            v = np.array([xs[b] - xs[a], ys[b] - ys[a]]); v /= (np.hypot(*v) or 1.0)
            tang[a] += v; tang[b] += v
    tn = np.hypot(tang[:, 0], tang[:, 1]); tn[tn == 0] = 1.0
    tang /= tn[:, None]
    wname = [ways[wi]["tags"].get("name", "") for wi in range(len(ways))]
    reach = max(STACK_RADIUS, PAR_STACK_R) + float(half.max())
    for i, near in zip(pts, stree.query_ball_point(np.c_[xs[pts], ys[pts]], reach)):
        for k in near:
            if sl[k] == lv_of[i] or i in (sa[k], sb[k]) or sw[k] == way_of[i]:
                continue
            dx, dy = bx_[k] - ax_[k], by_[k] - ay_[k]
            L2 = dx * dx + dy * dy or 1.0
            t = float(np.clip(((xs[i] - ax_[k]) * dx + (ys[i] - ay_[k]) * dy) / L2, 0, 1))
            # Two levels of one line running side by side (a bi-level line drawn with its levels a
            # few metres apart in plan, as the Lexington Av local and express at 86 St, which came
            # out only 3 m apart): held along the segment, same line name, parallel within ~25 deg.
            # Different lines are left to the point pairs below (see Queens Plaza).
            if wname[sw[k]] and wname[sw[k]] == wname[way_of[i]] and \
                    abs(tang[i] @ np.array([dx, dy]) / math.sqrt(L2)) > 0.9 and \
                    math.hypot(xs[i] - ax_[k] - t * dx, ys[i] - ay_[k] - t * dy) < PAR_STACK_R:
                seg_terms = [(int(sa[k]), 1 - t), (int(sb[k]), t)]
                if lv_of[i] > sl[k]:
                    add_c([(int(i), 1.0)], seg_terms, ("s", int(i), int(k)))
                else:
                    add_c(seg_terms, [(int(i), 1.0)], ("s", int(i), int(k)))
                continue
            # point pairs only (a point near a segment's end): tried 2026-11-02 holding points near
            # anywhere along a segment of another level, and it pulled stations where OSM draws
            # side-by-side tracks of two levels overlapping (Queens Plaza 10 m too deep)
            t = float(round(t))
            if math.hypot(xs[i] - ax_[k] - t * dx, ys[i] - ay_[k] - t * dy) > STACK_RADIUS:
                continue
            j = int(sa[k]) if t < 0.5 else int(sb[k])
            if lv_of[i] > sl[k]:
                add_c([(int(i), 1.0)], [(j, 1.0)], ("p", int(i), j))
            else:
                add_c([(j, 1.0)], [(int(i), 1.0)], ("p", j, int(i)))
    # segments of different levels that cross in plan
    for k1, near in enumerate([] if OLD_STACK else stree.query_ball_point(mid, 2 * float(half.max()))):
        for k2 in near:
            if k2 <= k1 or sl[k1] == sl[k2]:
                continue
            p, r = np.array([ax_[k1], ay_[k1]]), np.array([bx_[k1] - ax_[k1], by_[k1] - ay_[k1]])
            q, s_ = np.array([ax_[k2], ay_[k2]]), np.array([bx_[k2] - ax_[k2], by_[k2] - ay_[k2]])
            den = r[0] * s_[1] - r[1] * s_[0]
            if abs(den) < 1e-9:
                continue
            qp = q - p
            t = (qp[0] * s_[1] - qp[1] * s_[0]) / den
            u = (qp[0] * r[1] - qp[1] * r[0]) / den
            if not (0 <= t <= 1 and 0 <= u <= 1):
                continue
            # a crossing at a real angle; near-parallel overlaps are OSM drawing, not a crossing
            sin_a = abs(den) / (math.hypot(*r) * math.hypot(*s_) or 1.0)
            if sin_a < CROSS_MIN_SIN:
                continue
            t1 = [(int(sa[k1]), 1 - t), (int(sb[k1]), t)]
            t2 = [(int(sa[k2]), 1 - u), (int(sb[k2]), u)]
            if sl[k1] > sl[k2]:
                add_c(t1, t2, ("x", k1, k2))
            else:
                add_c(t2, t1, ("x", k1, k2))
    if not cand:
        return []
    from scipy.sparse.csgraph import dijkstra
    e = np.array([p for s in seqs for p in zip(s, s[1:])], dtype=int)
    wgt = np.hypot(xs[e[:, 1]] - xs[e[:, 0]], ys[e[:, 1]] - ys[e[:, 0]]) + 1e-3
    G = sp.csr_matrix((wgt, (e[:, 0], e[:, 1])), shape=(len(xs), len(xs)))
    srcs = sorted({up[0][0] for up, _ in cand.values()})
    dist = dijkstra(G, directed=False, indices=srcs, limit=STACK_GRAPH_M)
    row = {s: k for k, s in enumerate(srcs)}
    keep = [(up, lo) for up, lo in cand.values()
            if not any(np.isfinite(dist[row[up[0][0]], j]) for j, _ in lo)]
    return keep


RAILROAD_OPS = ("Long Island Rail Road", "Metro-North", "Amtrak", "New Jersey Transit", "NJ Transit")


def crossing_pairs(xs, ys, cls, ways, seqs, z0, gsm):
    """Grade-separated tunnel crossings: two tunnel segments of different ways (no service track)
    crossing in plan at more than ~25 degrees (CROSS_MIN_SIN) and not joined along the track within
    STACK_GRAPH_M (a junction or ramp). Returns (upper terms, lower terms, needed gap, sd, ordered).

    The upper one is the higher OSM level where the levels differ (sd CROSS_SD_ORDERED); with the
    same level, the one higher in z0, the solution without this rule (sd CROSS_SD_UNORDERED). Needed
    gap from crossing_gap by kind: a railroad's tunnel (LIRR, Metro-North, Amtrak, NJ Transit) is a
    railroad tube, subway more than CROSS_TUBE_DEPTH below the street a tube, else a box."""
    lvw = [level_of(w["tags"]) for w in ways]
    rr = [any(o in w["tags"].get("operator", "") for o in RAILROAD_OPS) for w in ways]
    wcls = [classify(w["tags"]) for w in ways]
    # Above ground too (2026-11-03): the Hell Gate line's viaduct over the Astoria el at Ditmars Blvd.
    # Open track (el, bridge, embankment, grade, cutting) has OSM layer 0 when untagged.
    lv_eff = [lvw[i] if lvw[i] is not None else (0 if wcls[i] != "tunnel" else None) for i in range(len(ways))]
    segs = [(a, b, wi) for wi, seq in enumerate(seqs)
            if ways[wi]["tags"].get("service") is None and (wcls[wi] == "tunnel" or CROSS_OPEN)
            for a, b in zip(seq, seq[1:]) if cls[a] == cls[b] and (cls[a] == "tunnel") == (wcls[wi] == "tunnel")]
    if not segs:
        return []
    sa = np.array([s[0] for s in segs]); sb = np.array([s[1] for s in segs]); sw = np.array([s[2] for s in segs])
    ax_, ay_, bx_, by_ = xs[sa], ys[sa], xs[sb], ys[sb]
    mid = np.c_[(ax_ + bx_) / 2, (ay_ + by_) / 2]
    half = float(np.hypot(bx_ - ax_, by_ - ay_).max()) / 2
    cand = []
    for k1, k2 in cKDTree(mid).query_pairs(2 * half):
        if sw[k1] == sw[k2] or {sa[k1], sb[k1]} & {sa[k2], sb[k2]}:
            continue
        r = (bx_[k1] - ax_[k1], by_[k1] - ay_[k1]); s_ = (bx_[k2] - ax_[k2], by_[k2] - ay_[k2])
        den = r[0] * s_[1] - r[1] * s_[0]
        if abs(den) < 1e-9:
            continue
        qp = (ax_[k2] - ax_[k1], ay_[k2] - ay_[k1])
        t = (qp[0] * s_[1] - qp[1] * s_[0]) / den
        u = (qp[0] * r[1] - qp[1] * r[0]) / den
        if not (0 <= t <= 1 and 0 <= u <= 1):
            continue
        if abs(den) / (math.hypot(*r) * math.hypot(*s_) or 1.0) < CROSS_MIN_SIN:
            continue
        cand.append((k1, k2, t, u))
    if not cand:
        return []
    from scipy.sparse.csgraph import dijkstra
    e = np.array([p for s in seqs for p in zip(s, s[1:])], dtype=int)
    wgt = np.hypot(xs[e[:, 1]] - xs[e[:, 0]], ys[e[:, 1]] - ys[e[:, 0]]) + 1e-3
    G = sp.csr_matrix((wgt, (e[:, 0], e[:, 1])), shape=(len(xs), len(xs)))
    srcs = sorted({int(sa[c[0]]) for c in cand})
    dist = dijkstra(G, directed=False, indices=srcs, limit=STACK_GRAPH_M)
    row = {s: i for i, s in enumerate(srcs)}
    out = []
    for k1, k2, t, u in cand:
        if np.isfinite(dist[row[int(sa[k1])], sa[k2]]) or np.isfinite(dist[row[int(sa[k1])], sb[k2]]):
            continue
        t1 = [(int(sa[k1]), 1 - t), (int(sb[k1]), t)]
        t2 = [(int(sa[k2]), 1 - u), (int(sb[k2]), u)]
        z1 = sum(z0[j] * w_ for j, w_ in t1); z2 = sum(z0[j] * w_ for j, w_ in t2)
        g1 = sum(gsm[j] * w_ for j, w_ in t1); g2 = sum(gsm[j] * w_ for j, w_ in t2)
        l1, l2 = lv_eff[sw[k1]], lv_eff[sw[k2]]
        ordered = l1 is not None and l2 is not None and l1 != l2
        first_up = (l1 > l2) if ordered else (z1 >= z2)
        (up, zu, gu, wu), (lo, zl, gl, wl) = ((t1, z1, g1, sw[k1]), (t2, z2, g2, sw[k2])) if first_up \
            else ((t2, z2, g2, sw[k2]), (t1, z1, g1, sw[k1]))
        if wcls[wl] == "tunnel" and wcls[wu] != "tunnel":
            continue   # a tunnel under open track: the tunnel's cover cap holds it
        if wcls[wl] != "tunnel":
            # open track underneath: its clearance (the lower track's kind) plus the upper deck or invert
            el = ways[wl]["tags"].get("electrified", "")
            clear = CROSS_CLEAR["catenary"] if "contact_line" in el else CROSS_CLEAR["railroad" if rr[wl] else "subway"] \
                if el else CROSS_CLEAR["unelectrified"]
            need = clear + (CROSS_DECK if wcls[wu] != "tunnel" else CROSS_INVERT["railroad" if rr[wu] else "box"])
            out.append((up, lo, need, CROSS_SD_ORDERED if ordered else CROSS_SD_UNORDERED, ordered))
            continue

        def kind(wi, zz, gg):
            if rr[wi]:
                return "railroad"
            return "tube" if np.isfinite(gg) and gg - zz > CROSS_TUBE_DEPTH else "box"
        need = crossing_gap(kind(wl, zl, gl), kind(wu, zu, gu))
        out.append((up, lo, need, CROSS_SD_ORDERED if ordered else CROSS_SD_UNORDERED, ordered))
    return out


def parallel_ties(xs, ys, cls, ways, way_of, seqs):
    """Rows (point, [(point, weight), (point, weight)]) tying a tunnel or cutting point to the
    parallel track beside it: tracks of one line (same OSM name) on one OSM level, same structure,
    within PAR_TIE_R m in plan and parallel within ~14 degrees, run in one box or one cut, so they
    share a rail height. Each track had only its own prior and its own anchors, and where those
    differ two tracks of one tunnel drifted apart: the Empire Connection's two tracks under 11th Av
    by 3.5 m in one tunnel, track 3 of the Bergen Tunnels by 6 m at the portal (2026-11-02). Points
    within PAR_TIE_CLEAR m along the track of a level change (a vertex where tunnel ways of two OSM
    levels meet: ramps down to a lower level, Grand Central's upper and lower throats) are left out,
    since there one track of a level really does dive away from the others."""
    n = len(xs)
    lvw = [level_of(w["tags"]) for w in ways]
    names = [w["tags"].get("name", "") for w in ways]
    svc = [w["tags"].get("service") is not None for w in ways]
    levels_at = collections.defaultdict(set)
    for wi, seq in enumerate(seqs):
        if classify(ways[wi]["tags"]) == "tunnel":
            for i in seq:
                levels_at[i].add(lvw[wi])
    change = [i for i, s in levels_at.items() if len(s) > 1]
    clear = np.zeros(n, dtype=bool)
    if change:
        from scipy.sparse.csgraph import dijkstra
        e = np.array([p for s in seqs for p in zip(s, s[1:])], dtype=int)
        wgt = np.hypot(xs[e[:, 1]] - xs[e[:, 0]], ys[e[:, 1]] - ys[e[:, 0]]) + 1e-3
        d = dijkstra(sp.csr_matrix((wgt, (e[:, 0], e[:, 1])), shape=(n, n)), directed=False, indices=change,
                     limit=PAR_TIE_CLEAR, min_only=True)
        clear = np.isfinite(d)
    segs = [(a, b, wi) for wi, seq in enumerate(seqs)
            if not svc[wi] and names[wi] and classify(ways[wi]["tags"]) in ("tunnel", "cutting")
            for a, b in zip(seq, seq[1:]) if cls[a] == cls[b] and not clear[a] and not clear[b]]
    if not segs:
        return []
    sa = np.array([s[0] for s in segs]); sb = np.array([s[1] for s in segs]); sw = np.array([s[2] for s in segs])
    mid = np.c_[(xs[sa] + xs[sb]) / 2, (ys[sa] + ys[sb]) / 2]
    half = float(np.hypot(xs[sb] - xs[sa], ys[sb] - ys[sa]).max()) / 2
    tree = cKDTree(mid)
    out, seen = [], set()
    for k0, (a0, b0, wi) in enumerate(segs):
        for i in (a0,):
            dx0, dy0 = xs[b0] - xs[a0], ys[b0] - ys[a0]
            L0 = math.hypot(dx0, dy0) or 1.0
            for k in tree.query_ball_point([xs[i], ys[i]], PAR_TIE_R + half):
                wj = sw[k]
                if wj == wi or names[wj] != names[wi] or lvw[wj] != lvw[wi] or cls[sa[k]] != cls[i]:
                    continue
                if (i, wj) in seen or i in (sa[k], sb[k]):
                    continue
                dx, dy = xs[sb[k]] - xs[sa[k]], ys[sb[k]] - ys[sa[k]]
                L2 = dx * dx + dy * dy or 1.0
                if abs(dx0 * dx + dy0 * dy) / (L0 * math.sqrt(L2)) < 0.97:
                    continue
                t = ((xs[i] - xs[sa[k]]) * dx + (ys[i] - ys[sa[k]]) * dy) / L2
                if t < 0 or t > 1:
                    continue
                if math.hypot(xs[i] - xs[sa[k]] - t * dx, ys[i] - ys[sa[k]] - t * dy) > PAR_TIE_R:
                    continue
                seen.add((i, wj))
                out.append((int(i), [(int(sa[k]), 1 - t), (int(sb[k]), t)]))
    return out


def station_groups(ways, xs, ys, way_of):
    """Track points at each station, per (station, OSM level), split by way.

    Uses the MTA stations table (one row per station and line) and LINE_WORDS to pick that
    line's tracks. Returns a list of dicts: name, line, structure, lon, lat, level, ways
    ({way index: [point indices]}).
    """
    # one row per station and line: W 4 St, 145 St and Queensboro Plaza share a Station ID between
    # two lines, and deduplicating on the ID alone dropped the second line's ties and dot
    st = pd.read_csv(RAW / "mta_stations.csv", dtype=str).drop_duplicates(["Station ID", "Line"])
    out = []
    for _, s in st.iterrows():
        lon, lat = float(s["GTFS Longitude"]), float(s["GTFS Latitude"])
        ax, ay = to_utm.transform(lon, lat)
        words = LINE_WORDS.get(s["Line"], [])
        near = np.where(np.hypot(xs - ax, ys - ay) < STATION_RADIUS)[0]
        by_level = collections.defaultdict(lambda: collections.defaultdict(list))
        for i in near:
            t = ways[way_of[i]]["tags"]
            if t.get("service") is None and any(wd in t.get("name", "") for wd in words):
                by_level[level_of(t)][int(way_of[i])].append(int(i))
        if not by_level:   # St George: OSM tags the terminal's platform tracks service=yard
            for i in near:
                t = ways[way_of[i]]["tags"]
                if t.get("service") == "yard" and any(wd in t.get("name", "") for wd in words):
                    by_level[level_of(t)][int(way_of[i])].append(int(i))
        for lv, wmap in by_level.items():
            out.append({"name": s["Stop Name"], "line": s["Line"], "structure": s["Structure"],
                        "lon": lon, "lat": lat, "level": lv, "ways": dict(wmap)})
    return out


_surf_cache = {}


def solve(ways, xs, ys, way_of, seqs, ground, gsm, doc_ok=None, elev_ok=None, nsamples=NSAMPLES,
          quiet=False, surf=None):
    """Build and solve the system. doc_ok(group_id) / elev_ok(anchor) say whether a document or
    elevator anchor is used (default all); calibrate.py uses them to hold anchors out. Returns
    z, sd, prior_mu, prior_sd, cls, used (anchors as written to tracks.json), info (what was
    placed, held out or skipped). surf: surface_ground (computed here if not given, once per run)."""
    n = len(xs)
    if surf is None:
        k_ = (n, float(xs[0]), float(xs[-1]))
        if k_ not in _surf_cache:
            _surf_cache[k_] = surface_ground(xs, ys, seqs)
        surf = _surf_cache[k_]
    rows, cols, vals, rhs, sds = [], [], [], [], []
    eq = 0
    prior_mu = np.full(n, np.nan); prior_sd = np.full(n, np.nan); cls = [""] * n

    def add(terms, target, sd):
        nonlocal eq
        for c, v in terms:
            rows.append(eq); cols.append(c); vals.append(v)
        rhs.append(target); sds.append(sd)
        eq += 1

    pairs = []
    grade_eqs = []
    curv_eqs = []
    bed = smooth_bed(seqs, load_riverbed(xs, ys))
    # "Over water" was any ground below 1 m. Dry low ground passed too: the open cut outside the East
    # River tunnels' Queens portal at Hunterspoint Av floors at -1.5 m, so its tracks got the
    # weak over-water prior and were left out of the portal taper, and the LIRR and Amtrak tracks
    # made a 10 m V there (2026-11-02). Now low ground counts as water only within WATER_NEAR_M of
    # the water polygon (data/water.json, make_water.py), or where the DEM has no ground.
    wet_pt = np.isnan(gsm) | ((gsm < 1.0) & near_water(xs, ys))
    # open track on ground under 1 m within MARSH_M of water: marsh or a trestle over it (used for grade and
    # embankment only, in the prior)
    marsh = (gsm < 1.0) & ~wet_pt & near_water(xs, ys, MARSH_M)
    # hills at the calibrated grade; and over water the envelope hangs off the riverbed at the
    # steeper grade river approaches are built to, so they dive toward the crossing
    genv = np.minimum(lower_envelope(seqs, xs, ys, gsm, ENVELOPE_SLOPE),
                      lower_envelope(seqs, xs, ys, np.where(np.isnan(bed), np.inf, bed), RIVER_APPROACH_SLOPE))
    # Open track at a portal: the 3 m DEM reads the hill over the mouth at whichever track's portal
    # node lands on the portal face, and at a node shared with the tunnel the open prior (sd 2 m)
    # wins over the tunnel's. Track 3 of the four Bergen Tunnels tracks (Morris & Essex into Hoboken)
    # had its portal node at Tonnele Av on 33.6 m of hillside where its neighbours read the cut floor
    # at 12-13 m, so it rose 6 m over the others into the tunnel (2026-11-02). Within PORTAL_GROUND_M
    # of a tunnel, open track (grade, cutting, embankment) hangs off the lower envelope of the open
    # approach's ground at the grade limit: track at grade can't climb a portal face.
    open_c = ("grade", "cutting", "embankment")
    near_portal_g = np.zeros(n, dtype=bool)
    oenv = gsm.copy()
    tun_pts = sorted({i for wi, seq in enumerate(seqs) if classify(ways[wi]["tags"]) == "tunnel" for i in seq})
    if PORTAL_GROUND_M > 0 and tun_pts:
        from scipy.sparse.csgraph import dijkstra
        e_ = np.array([p for s in seqs for p in zip(s, s[1:])], dtype=int)
        w_ = np.hypot(xs[e_[:, 1]] - xs[e_[:, 0]], ys[e_[:, 1]] - ys[e_[:, 0]]) + 1e-3
        d_ = dijkstra(sp.csr_matrix((w_, (e_[:, 0], e_[:, 1])), shape=(n, n)), directed=False,
                      indices=tun_pts, limit=PORTAL_GROUND_M, min_only=True)
        near_portal_g = np.isfinite(d_)
        oseqs = [seq for wi, seq in enumerate(seqs) if classify(ways[wi]["tags"]) in open_c]
        g_open = np.where(np.isnan(gsm) | wet_pt, 1e4, gsm)
        oenv = lower_envelope(oseqs, xs, ys, g_open, GRADE_LIMIT)
        if not quiet:
            low = near_portal_g & (oenv < gsm - 1.0) & ~wet_pt
            print(f"  open track within {PORTAL_GROUND_M:.0f} m of a portal: {int(low.sum())} points on a portal face "
                  f"(ground over 1 m above the approach's envelope) lowered to it", flush=True)
    for wi, (w, seq) in enumerate(zip(ways, seqs)):
        t = w["tags"]
        c = classify(t)
        lv = level_of(t)
        for idx in seq:
            g = gsm[idx]
            if c in open_c and near_portal_g[idx] and oenv[idx] < g:
                g = oenv[idx]
            wet = bool(wet_pt[idx])
            if c in ("grade", "embankment") and wet and np.isfinite(ground[idx]) and ground[idx] > 1.0:
                # a causeway: track on land the DEM reads at its own point, where the running median mixes
                # in the water beside it (the A across Jamaica Bay came out at sea level, 2026-11-03)
                wet, g = False, max(g, float(ground[idx]))
            if c in ("grade", "embankment") and not wet and marsh[idx]:
                wet = True   # low ground near the water (Jamaica Bay's marsh islands): a trestle the bare earth leaves out
            if c in ("grade", "embankment") and wet:
                # open track over water or marsh is on a causeway or a low trestle: CAUSEWAY, not sea level
                mu, sd = CAUSEWAY
                if np.isnan(prior_mu[idx]) or sd < prior_sd[idx]:
                    prior_mu[idx], prior_sd[idx], cls[idx] = mu, sd, c
                continue
            if c == "tunnel":
                if wet and not np.isnan(bed[idx]):
                    mu, sd = bed[idx] - BELOW_BED[0], BELOW_BED[1]
                elif wet:
                    mu, sd = UNDERWATER
                else:
                    d, sd = TUNNEL_LEVEL.get(lv, TUNNEL_DEFAULT) if lv is not None and lv < 0 else TUNNEL_DEFAULT
                    mu = min(g, genv[idx]) - d
            elif c == "elevated" and wet:
                mu, sd = BRIDGE_OVER_WATER
            else:
                off, sd = CLASS[c]
                if c == "elevated" and np.isfinite(surf[idx]):
                    g = min(g, surf[idx])   # an el stands over the street beside it, not on a solid structure's top
                mu = (0 if wet else g) + off
                if wet:
                    sd = max(sd, 10)
            if np.isnan(prior_mu[idx]) or sd < prior_sd[idx]:
                prior_mu[idx], prior_sd[idx], cls[idx] = mu, sd, c
        pairs.extend(zip(seq, seq[1:]))
        for a, b in zip(seq, seq[1:]):
            h = math.hypot(xs[b] - xs[a], ys[b] - ys[a]) or 1.0
            grade_eqs.append((eq, a, b, h))
            add([(b, 1 / h), (a, -1 / h)], 0.0, MAX_GRADE)
        for a, b, c2 in zip(seq, seq[1:], seq[2:]):
            h1 = math.hypot(xs[b] - xs[a], ys[b] - ys[a]) or 1.0
            h2 = math.hypot(xs[c2] - xs[b], ys[c2] - ys[b]) or 1.0
            add([(c2, 1 / h2), (b, -1 / h2 - 1 / h1), (a, 1 / h1)], 0.0, CURV_SD * STEP / ((h1 + h2) / 2))
            curv_eqs.append((a, b, c2, h1, h2))
    # Portals: a tunnel's rail meets the open track's at the portal, however high the ground is over
    # the mouth. The level rule (rail 7-27 m under the street) held to the last tunnel point, so where
    # the street rises over a portal the prior dived just before it: the 1 dipped 4 m before the
    # Manhattan Valley viaduct at 122 St (2026-11-02). Within PORTAL_TAPER (400) m of open track the tunnel
    # prior blends linearly to the open track's prior at the portal, and the cover cap (below)
    # tapers to nothing the same way. River tunnels (no ground) are left alone.
    portal_s = np.full(n, np.inf)
    is_tun = np.array([c == "tunnel" for c in cls])
    if PORTAL_TAPER > 0 and (~is_tun).any() and is_tun.any():
        from scipy.sparse.csgraph import dijkstra
        e_ = np.array(pairs, dtype=int)
        w_ = np.hypot(xs[e_[:, 1]] - xs[e_[:, 0]], ys[e_[:, 1]] - ys[e_[:, 0]]) + 1e-3
        G_ = sp.csr_matrix((w_, (e_[:, 0], e_[:, 1])), shape=(n, n))
        opens = np.where(~is_tun)[0]
        d_, _, src_ = dijkstra(G_, directed=False, indices=opens, limit=PORTAL_TAPER, min_only=True,
                               return_predecessors=True)
        near_p = is_tun & np.isfinite(d_) & ~wet_pt
        portal_s[near_p] = d_[near_p]
        w_p = 1.0 - portal_s[near_p] / PORTAL_TAPER
        prior_mu[near_p] = (1 - w_p) * prior_mu[near_p] + w_p * prior_mu[src_[near_p]]
        if not quiet:
            print(f"  {int(near_p.sum())} tunnel points within {PORTAL_TAPER:.0f} m of a portal: prior and "
                  f"cover taper to the open track's", flush=True)
    # The prior jumps where a way's tags change (a portal, a level -1 way meeting a level -2 one)
    # and follows every kerb in the ground; real track ramps between them. Diffuse it along the
    # track graph over ~PRIOR_SMOOTH m so the solution isn't handed a step.
    nb = [[] for _ in range(n)]
    for a, b in pairs:
        nb[a].append(b); nb[b].append(a)
    deg = np.array([len(v) for v in nb], dtype=float)
    src = np.repeat(np.arange(n), [len(v) for v in nb]); dst = np.concatenate([v for v in nb if v] or [[]]).astype(int)
    M = sp.csr_matrix((np.ones(len(src)), (src, dst)), shape=(n, n))
    iters = int(2 * (PRIOR_SMOOTH / STEP) ** 2)
    # river crossings are short; smoothing would let the banks wash out their deep prior, so
    # they are held fixed and only spread outward into the approaches
    held = np.array([c == "tunnel" for c in cls]) & wet_pt
    mu0 = prior_mu.copy()
    # Neighbours count by how sure their prior is (1/sd^2). Until 2026-11-03 they counted equally, so a
    # bridge over water's next-to-nothing absolute prior (12 +- 60 m) pulled its approaches' priors up 4-8 m;
    # the lidar then put the approach far under that prior, and the correlated prior error carried the
    # offset into the unmeasured track beyond: the A south of Broad Channel sagged to -0.8 m between lidar
    # readings of 3.3 m on both sides.
    wpr = 1.0 / np.maximum(prior_sd, 0.1) ** 2
    wsum = M @ wpr
    for _ in range(iters):
        nbmean = np.where(wsum > 0, (M @ (wpr * prior_mu)) / np.maximum(wsum, 1e-12), prior_mu)
        prior_mu = 0.5 * prior_mu + 0.5 * nbmean
        prior_mu[held] = mu0[held]
    # Prior error d = z - mu is an Ornstein-Uhlenbeck process along the track: a stretch built
    # deeper than the rule of thumb stays deeper for ~CORR m. Pairwise terms carry the
    # correlation; the weak unary terms keep each point's marginal sd near prior_sd.
    # The chain is cut where the kind of structure changes (a portal from a bridge or el into a
    # tunnel, an embankment into a cutting): how far a viaduct sits above its rule of thumb says
    # nothing about the tunnel beyond it. Until 2026-11-02 it ran on through, and the Manhattan
    # Bridge, 8-15 m above the elevated prior, carried that into the tunnel under Canal St and
    # Chrystie St, which came out 3 m under the street where the platforms are 10-17 m down.
    for a, b in pairs:
        if cls[a] != cls[b] and not OU_ACROSS:
            continue
        h = math.hypot(xs[b] - xs[a], ys[b] - ys[a]) or 1.0
        rho = math.exp(-h / CORR)
        s = (prior_sd[a] + prior_sd[b]) / 2 * math.sqrt(1 - rho * rho) * PARALLEL
        add([(b, 1.0), (a, -rho)], prior_mu[b] - rho * prior_mu[a], s)
    if os.environ.get("TD_DUMP_PRIOR"):   # debugging: the prior as the solve sees it
        np.savez(os.environ["TD_DUMP_PRIOR"], xs=xs, ys=ys, mu=prior_mu, sd=prior_sd, cls=np.array(cls), gsm=gsm, surf=surf)
    unary = math.sqrt(20 * CORR / STEP)  # the OU terms already carry ~one sd per 2*CORR
    for i in range(n):
        add([(i, 1.0)], prior_mu[i], prior_sd[i] * unary * PARALLEL)

    # one entry per (way, point): a junction point belongs to every way that meets there, each with
    # its own name, level and kind, as in tracks.json (validate.py places rows on the same entries)
    ent_pt = np.array([i for seq in seqs for i in seq])
    ent_way = np.array([wi for wi, seq in enumerate(seqs) for _ in seq])
    ent_name = [ways[w]["tags"].get("name", "") for w in ent_way]
    ent_level = [level_of(ways[w]["tags"]) for w in ent_way]
    ent_cls = [classify(ways[w]["tags"]) for w in ent_way]
    ent_service = [ways[w]["tags"].get("service") is not None for w in ent_way]

    stop_ways = stop_way_set(ways)
    anchors = station_anchors()
    used, skipped = [], []
    elev_pts = set()    # points pinned by an elevator anchor, held out or not
    elev_info = []
    for an in anchors:
        ax, ay = to_utm.transform(an["lon"], an["lat"])
        d = np.hypot(xs - ax, ys - ay)
        # which kinds of OSM track a platform this far below (or above) the street can be on
        dep = an["depth"]
        if dep > 0:
            ok = {"tunnel", "cutting"} | ({"grade"} if dep < 8 else set())
        else:
            ok = {"elevated", "embankment"} | ({"grade"} if dep > -8 else set())
        words = LINE_WORDS.get(an["line"], [])
        near = [i for i in np.where(d < 100)[0]
                if cls[i] in ok and ways[way_of[i]]["tags"].get("service") is None
                and any(wd in ways[way_of[i]]["tags"].get("name", "") for wd in words)]
        if not near:
            skipped.append((an, "no OSM track of this line and kind within 100 m"))
            continue
        # a platform pins the tracks that stop at it, not express tracks running past on
        # another level (Central Park West: local platforms, express passing underneath)
        stopping = [i for i in near if way_of[i] in stop_ways]
        if stopping:
            near = stopping
        if an["n_groups"] > 1:
            # platforms of this line on several levels: pin a level only if the nearby tracks
            # carry exactly as many distinct OSM levels, deeper platforms to deeper tracks
            lvls = sorted({level_of(ways[way_of[i]]["tags"]) for i in near} - {None})
            if len(lvls) != an["n_groups"]:
                skipped.append((an, f"{an['n_groups']} platform levels but OSM levels nearby are {lvls}"))
                continue
            want = lvls[an["level_group"]]
            near = [i for i in near if level_of(ways[way_of[i]]["tags"]) == want]
        target = an["platform_z"] - PLATFORM_ABOVE_RAIL
        sd = max(ANCHOR_SD_FLOOR, an["sd"])
        # one equation per OSM way: the mean height of its points near the platform. The anchor's
        # error is shared by every point, so pinning each point separately would count it many times.
        by_way = collections.defaultdict(list)
        for i in near:
            by_way[way_of[i]].append(i)
        elev_pts.update(int(i) for i in near)
        held = elev_ok is not None and not elev_ok(an)
        elev_info.append({**an, "rail": float(target), "sd_used": float(sd), "idx": near, "held_out": held})
        if held:
            continue
        for w_, pts in by_way.items():
            add([(i, 1.0 / len(pts)) for i in pts], target, sd)
        used.append({**an, "source": "elevator", "rail": round(float(target), 2), "sd_used": round(sd, 2),
                     "points": len(near), "ways": len(by_way)})
    if not quiet:
        for an, why in skipped:
            print(f"  anchor not used: {an['station']} ({an['line']}) {an['levels']}: {why}")

    # ---- station ties: the tracks of one line along one platform level share a height
    # (Grand Central's 4/5 and 6 tracks came out 7 m apart without this) ----
    n_ties = 0
    for g in station_groups(ways, xs, ys, way_of):
        wl = list(g["ways"].values())
        for pts in wl[1:]:
            ref = wl[0]
            add([(i, 1.0 / len(pts)) for i in pts] + [(i, -1.0 / len(ref)) for i in ref], 0.0, STATION_TIE_SD)
            n_ties += 1
    if not quiet:
        print(f"  {n_ties} station ties between tracks of one platform level", flush=True)
    # ---- what the MTA's Structure column says, and what a station box needs (2026-11-03) ----
    # Underground ('Subway') stations: rail at least STATION_FLOOR below the street, a cut-and-cover
    # station with a mezzanine (station_levels.csv has one) 7.0 m, one with fare control at the street
    # 4.6 m; held as caps (one-sided) on the station's tunnel points. Delancey St-Essex St (J/M/Z) came
    # out 1.6 m down. 'Open Cut', 'At Grade' and 'Embankment' stations get a soft prior on the rail
    # against the surface beside the track (STRUCTURE_PRIOR), unless an elevator anchor covers that
    # station and line (also when it is held out, so the hold-out stays fair): Botanic Garden (an open
    # cut) came out 8 m down as a tunnel, the Brighton line's open-cut stations at the cut floor's DEM.
    mta_st = pd.read_csv(RAW / "mta_stations.csv", dtype=str).drop_duplicates(["Station ID", "Line"])
    struct_of = {(r["Stop Name"], r["Line"]): r["Structure"] for _, r in mta_st.iterrows()}
    slv = pd.read_csv(ROOT / "evidence" / "station_levels.csv", dtype={"station_mrn": str})
    mezz_st = set(slv.loc[slv["level_name"].str.contains("mezzanine", na=False), "station"].astype(str))
    anchored = {(a["station"], a["line"]) for a in anchors}
    floor_caps = []
    n_struct = 0
    for g in station_groups(ways, xs, ys, way_of):
        stc = struct_of.get((g["name"], g["line"]))
        if stc == "Subway":
            fl = STATION_FLOOR[1] if g["name"] in mezz_st else STATION_FLOOR[0]
            for pts in g["ways"].values():
                for i in pts:
                    if cls[i] == "tunnel" and np.isfinite(gsm[i]) and not wet_pt[i]:
                        floor_caps.append((i, float(gsm[i]) - fl))
        elif stc in STRUCTURE_PRIOR and (g["name"], g["line"]) not in anchored:
            off, sdp = STRUCTURE_PRIOR[stc]
            for pts in g["ways"].values():
                # an open cut against the street beside it; at grade and embankment against the track's
                # own bed (the DEM keeps an embankment; beside a causeway is water)
                sv = surf[pts] if stc == "Open Cut" else np.fmax(gsm[pts], ground[pts])
                if np.isfinite(sv).any():
                    add([(i, 1.0 / len(pts)) for i in pts], float(np.nanmean(sv)) + off, sdp)
                    n_struct += 1
    if not quiet:
        print(f"  {len(floor_caps)} underground station points held at least {STATION_FLOOR[0]}-{STATION_FLOOR[1]} m under "
              f"the street; {n_struct} open-cut, at-grade and embankment station tracks given the MTA structure's height", flush=True)

    # ---- tracks on the two sides of one OSM platform are at one height (cross-platform transfers:
    # Prospect Park's Franklin Av shuttle came out 4-5 m above the Brighton line it shares a platform with) ----
    n_ptie = 0
    pp = RAW / "osm_platforms.json"
    if pp.exists() and PLAT_TIE_SD > 0:
        from shapely.geometry import LineString, Point, Polygon
        ptree_ = cKDTree(np.c_[xs, ys])
        svc_pt = np.array([ways[w]["tags"].get("service") is not None for w in way_of])
        for e in json.loads(pp.read_text(encoding="utf-8"))["elements"]:
            tg = e.get("tags", {})
            if e["type"] != "way" or len(e.get("geometry") or []) < 2 or tg.get("disused") == "yes" \
                    or any(k.startswith(("disused", "abandoned", "construction")) for k in tg):
                continue
            pxy = [to_utm.transform(p["lon"], p["lat"]) for p in e["geometry"]]
            geom = Polygon(pxy).buffer(0) if len(pxy) >= 4 and pxy[0] == pxy[-1] else LineString(pxy)
            if geom.is_empty or geom.length < 20:
                continue
            c = geom.centroid
            byw = collections.defaultdict(list)
            for i in ptree_.query_ball_point([c.x, c.y], geom.length / 2 + 10):
                if not svc_pt[i] and geom.distance(Point(xs[i], ys[i])) < PLAT_TIE_R:
                    byw[int(way_of[i])].append(int(i))
            byw = {w: v for w, v in byw.items() if len(v) >= 2}
            # one OSM level, and not an el beside track on the ground (a cutting beside the tunnel it runs
            # into is fine: Prospect Park's Brighton tracks enter their tunnel at the platform's north end)
            if len(byw) < 2 or len({level_of(ways[w]["tags"]) for w in byw}) > 1 \
                    or len({cls[v[0]] == "elevated" for v in byw.values()}) > 1:
                continue
            wl = list(byw.values())
            for pts in wl[1:]:
                add([(i, 1.0 / len(pts)) for i in pts] + [(i, -1.0 / len(wl[0])) for i in wl[0]], 0.0, PLAT_TIE_SD)
                n_ptie += 1
    if not quiet:
        print(f"  {n_ptie} ties between tracks on the two sides of one OSM platform", flush=True)

    # ---- parallel tracks of one line and level in one tunnel or cut share a rail height ----
    if PAR_TIE_SD > 0:
        pt_ = parallel_ties(xs, ys, cls, ways, way_of, seqs)
        for i, seg in pt_:
            add([(i, 1.0)] + [(j, -w_) for j, w_ in seg], 0.0, PAR_TIE_SD)
        if not quiet:
            print(f"  {len(pt_)} ties between parallel tracks of one line and level in a tunnel or cut", flush=True)

    # ---- document anchors (evidence/depths.csv) ----
    docs = doc_rows()
    placed = place_doc_rows(docs, xs[ent_pt], ys[ent_pt], ent_name, ent_level, ent_cls, ent_service)
    groups = collections.defaultdict(list)   # set of entries -> rows that describe that platform
    doc_status = {}                          # row id -> (status, reason, osm level)
    for row, (ents, lvl, why) in zip(docs, placed):
        if row["why"]:
            doc_status[row["id"]] = ("skipped", row["why"], None)
        elif why:
            doc_status[row["id"]] = ("skipped", why, None)
        else:
            pts = frozenset(int(ent_pt[e]) for e in ents)
            if pts & elev_pts:
                doc_status[row["id"]] = ("elevator", "an elevator anchor already pins these tracks", lvl)
            else:
                groups[pts].append((row, ents, lvl))
    doc_groups = []
    for pts_set, members in groups.items():
        gid = min(m[0]["id"] for m in members)
        ents_all = sorted({e for m in members for e in m[1]})
        idx = sorted({int(ent_pt[e]) for e in ents_all})
        dep = np.array([m[0]["depth"] for m in members]); sdr = np.array([m[0]["sd"] for m in members])
        w = 1 / sdr ** 2
        depth = float((w * dep).sum() / w.sum())
        # rows about one platform are not independent (Wikipedia copies, one sentence given to three
        # stations): keep the best single sd, widened by half the spread between them
        sd_g = max(float(sdr.min()), float(dep.max() - dep.min()) / 2)
        held = doc_ok is not None and not doc_ok(gid)
        by_way = collections.defaultdict(list)
        for e in ents_all:
            if not np.isnan(gsm[ent_pt[e]]):
                by_way[int(ent_way[e])].append(int(ent_pt[e]))
        g = {"gid": gid, "rows": [m[0] for m in members], "idx": idx, "level": members[0][2], "depth": depth,
             "sd": sd_g, "held_out": held, "ways": len(by_way)}
        doc_groups.append(g)
        for m in members:
            doc_status[m[0]["id"]] = ("held out" if held else "anchor", f"group {gid}", m[2])
        if held or not by_way:
            continue
        for w_, pts in by_way.items():
            add([(i, 1.0 / len(pts)) for i in pts], float(np.mean(gsm[pts])) - depth, sd_g)
        r0 = members[0][0]
        used.append({"station": r0["place"][:60], "line": r0["line"][:60], "lon": r0["lon"], "lat": r0["lat"],
                     "source": "document", "rail": None, "sd_used": round(sd_g, 2), "depth": round(depth, 2),
                     "points": len(idx), "ways": len(by_way), "rows": " ".join(m[0]["id"] for m in members)})

    # ---- placed rows (evidence/placed_rows.csv): river tunnels, bridges and absolute-datum
    # figures with a point and named ways. Skipped where a row or elevator anchor already covers them.
    n_placed = 0
    for _, r in load_placed("rows").iterrows():
        rid = str(r["id"])
        if rid in SKIP_PLACED or any(doc_status.get(x, ("",))[0] in ("anchor", "elevator") for x in rid.split("+")):
            continue
        pts = placed_points(r, ways, xs, ys, way_of)
        if not pts:
            continue
        by_way = collections.defaultdict(list)
        for i in pts:
            by_way[int(way_of[i])].append(i)
        for w_, p_ in by_way.items():
            add([(i, 1.0 / len(p_)) for i in p_], float(r["rail_navd88_m"]), float(r["sd_m"]))
        used.append({"station": rid, "line": "", "lon": float(r["lon"]), "lat": float(r["lat"]), "source": "placed",
                     "rail": float(r["rail_navd88_m"]), "sd_used": float(r["sd_m"]), "points": len(pts), "ways": len(by_way)})
        n_placed += 1
    if not quiet:
        print(f"  {n_placed} placed rows used", flush=True)

    # ---- profile drawings read off documents (evidence/profiles*.csv: profiles.csv, profiles_docs.csv) ----
    n_prof = 0
    pf_files = sorted((ROOT / "evidence").glob("profiles*.csv"))
    if pf_files and not os.environ.get("TD_NO_PROFILES"):
        pf = pd.concat([pd.read_csv(p) for p in pf_files], ignore_index=True)
        for _, r in pf.iterrows():
            if not (np.isfinite(r["rail_navd88_m"]) and np.isfinite(r["lon"]) and np.isfinite(r["sd_m"])):
                continue
            base = re.sub(r"\(.*?\)", "", str(r["line"]))
            words = [re.sub(r"^(IRT|IND|BMT)\s+|\s+Line$", "", p.strip()).replace("-", "–") for p in base.split("/")]
            # "OSM level -1;-2" means both levels run there: the point's own level is picked by order,
            # as for document rows (lower / express -> lowest, upper / local -> highest). Until
            # 2026-11-02 the first number was taken for both, so the 1914 Lexington Av express-level
            # points pinned the local tracks 15-20 m too deep (59 St came out 15 m down, not 7).
            m = re.search(r"OSM level (-?\d+(?:;-?\d+)*)", str(r["notes"]))
            lvls = sorted(int(v) for v in m.group(1).split(";")) if m else []
            want_lv = lvls[0] if len(lvls) == 1 else None
            if len(lvls) > 1:
                # a point naming no level (the 1904 Times Sq shuttle) pins that line's tracks at both
                hint = level_hint({"place": str(r["place"]), "line": str(r["line"])})
                if hint is not None:
                    want_lv = lvls[0] if hint == "deep" else lvls[-1]
            ax, ay = to_utm.transform(float(r["lon"]), float(r["lat"]))
            near = np.where(np.hypot(xs - ax, ys - ay) < PROFILE_RADIUS)[0]
            pts = []
            for i in near:
                t = ways[way_of[i]]["tags"]
                nm = t.get("name", "").replace("-", "–")
                if t.get("service") is None and any(wd and wd in nm for wd in words) \
                        and (want_lv is None or level_of(t) == want_lv):
                    pts.append(int(i))
            if not pts:
                continue
            by_way = collections.defaultdict(list)
            for i in pts:
                by_way[int(way_of[i])].append(i)
            for w_, p_ in by_way.items():
                add([(i, 1.0 / len(p_)) for i in p_], float(r["rail_navd88_m"]), float(r["sd_m"]))
            n_prof += 1
    if not quiet:
        print(f"  {n_prof} profile-drawing points used", flush=True)

    # ---- elevated, bridge and embankment rail measured by lidar (structure_heights.py) ----
    n_lidar = n_lidar_out = n_hidden = 0
    lidar_eqs = []
    deck_caps = []
    sh_path = ROOT / "evidence" / "structure_heights.csv"
    if sh_path.exists() and not os.environ.get("TD_NO_LIDAR"):
        sh = pd.read_csv(sh_path)
        sh = sh[np.isfinite(sh["rail_navd88_m"]) & np.isfinite(sh["sd_m"])]
        wi_of = {w["id"]: wi for wi, w in enumerate(ways)}
        # OSM ways get split and replaced between downloads; a sample whose way is gone goes to the
        # nearest open (non-tunnel) track point within 12 m (some were redrawn several metres over), on that point's way
        lost = ~sh["way_id"].isin(list(wi_of))
        if lost.any():
            open_pts = np.array([i for i in range(n) if cls[i] != "tunnel"])
            otree = cKDTree(np.c_[xs[open_pts], ys[open_pts]])
            lx, ly = to_utm.transform(sh.loc[lost, "lon"].to_numpy(), sh.loc[lost, "lat"].to_numpy())
            dd_, kk_ = otree.query(np.c_[lx, ly])
            new_w = np.where(dd_ < 12.0, [ways[way_of[open_pts[k_]]]["id"] for k_ in kk_], -1)
            sh = sh.copy()
            sh.loc[lost, "way_id"] = new_w
            if not quiet:
                print(f"  {int(lost.sum())} lidar samples on ways no longer in OSM: {int((new_w >= 0).sum())} "
                      f"moved to the track now there", flush=True)
            sh = sh[sh["way_id"] >= 0]
        # Canopies: in an el station the tracks of one level stand at one height, so a station-zone
        # reading more than CANOPY_DZ above the median reading of that station level's tracks is a
        # roof over a platform, not the rail (Coney Island-Stillwell Av's two westmost tracks read
        # 2-4 m high under the shed roof, 2026-11-02). Those become caps, as hidden decks do.
        canopy = np.zeros(len(sh), dtype=bool)
        st_src = sh["source"].isin(["lidar_station", "lidar", "lidar_interp"]).to_numpy()
        sx_, sy_ = to_utm.transform(sh["lon"].to_numpy(), sh["lat"].to_numpy())
        sh_tree = cKDTree(np.c_[sx_, sy_])
        zz_all = sh["rail_navd88_m"].to_numpy()
        for g_ in station_groups(ways, xs, ys, way_of):
            pts_ = [i for v in g_["ways"].values() for i in v]
            if len(g_["ways"]) < 2 or not all(cls[i] == "elevated" for i in pts_):
                continue
            near_ = sorted({j for i in pts_ for j in sh_tree.query_ball_point([xs[i], ys[i]], 6.0) if st_src[j]})
            if len(near_) < 4:
                continue
            med_ = float(np.median(zz_all[near_]))
            for j in near_:
                if zz_all[j] > med_ + CANOPY_DZ:
                    canopy[j] = True
        sh = sh.assign(_canopy=canopy)
        if not quiet:
            print(f"  {int(canopy.sum())} el-station lidar samples read as a canopy over the platform", flush=True)
        trees = {}
        # Segments of elevated track by OSM level: a lower-deck track with a higher-level el within
        # HIDDEN_DECK_R m over it can't be seen from above, so its "own" lidar is the upper deck
        # (Queensboro Plaza's lower 7 track read 20.4 m, the upper deck's height, where the lower
        # N/W track beside it was at 16.4; 2026-11-02). Such samples become caps: rail at least
        # HIDDEN_DECK_GAP below the surface read.
        el_segs = [(a, b, level_of(ways[wi]["tags"])) for wi, seq in enumerate(seqs)
                   if classify(ways[wi]["tags"]) == "elevated" and level_of(ways[wi]["tags"]) is not None
                   for a, b in zip(seq, seq[1:])]
        if el_segs:
            es_a = np.array([s[0] for s in el_segs]); es_b = np.array([s[1] for s in el_segs])
            es_l = np.array([s[2] for s in el_segs])
            es_tree = cKDTree(np.c_[(xs[es_a] + xs[es_b]) / 2, (ys[es_a] + ys[es_b]) / 2])

        def under_upper_deck(px, py, lv):
            if not el_segs or lv is None:
                return False
            for k_ in es_tree.query_ball_point([px, py], HIDDEN_DECK_R + 20.0):
                if es_l[k_] <= lv:
                    continue
                ax_, ay_, bx_, by_ = xs[es_a[k_]], ys[es_a[k_]], xs[es_b[k_]], ys[es_b[k_]]
                dx_, dy_ = bx_ - ax_, by_ - ay_
                t_ = float(np.clip(((px - ax_) * dx_ + (py - ay_) * dy_) / (dx_ * dx_ + dy_ * dy_ or 1.0), 0, 1))
                if math.hypot(px - ax_ - t_ * dx_, py - ay_ - t_ * dy_) < HIDDEN_DECK_R:
                    return True
            return False
        for wid, grp in sh.groupby("way_id"):
            wi = wi_of.get(int(wid))
            if wi is None:
                continue
            pts = np.array(seqs[wi])
            if wi not in trees:
                trees[wi] = cKDTree(np.c_[xs[pts], ys[pts]])
            gx, gy = to_utm.transform(grp["lon"].to_numpy(), grp["lat"].to_numpy())
            d, k = trees[wi].query(np.c_[gx, gy])
            # Lidar sometimes reads the wrong deck (a roadway over the tracks on the Manhattan and
            # Williamsburg Bridges, a canopy). Along a way the rail is smooth, so a sample far from
            # the running median of its neighbours is dropped.
            order = np.argsort(k, kind="stable")
            zz = grp["rail_navd88_m"].to_numpy()[order]
            med = np.array([np.median(zz[max(0, j - 3):j + 4]) for j in range(len(zz))])
            keep = np.zeros(len(zz), dtype=bool)
            keep[order] = np.abs(zz - med) <= LIDAR_OUTLIER_M
            n_lidar_out += int((~keep).sum())
            # where decks stack (template-copied, interpolated or station samples) the reading is
            # less sure; let the smoothness and grade limit win there
            src = grp["source"].to_numpy()
            soft = np.isin(src, ["lidar_template", "lidar_interp"])
            sdv = np.where(soft, np.maximum(grp["sd_m"].to_numpy(), LIDAR_SOFT_SD), grp["sd_m"].to_numpy())
            lv_w = level_of(ways[wi]["tags"]) if classify(ways[wi]["tags"]) == "elevated" else None
            for dd, kk, z_, s_, ok_, src_, gx_, gy_, cn_ in zip(d, k, grp["rail_navd88_m"], sdv, keep, src, gx, gy,
                                                                grp["_canopy"]):
                if dd < 20 and cn_:
                    deck_caps.append((int(pts[kk]), float(z_) - CANOPY_GAP, 1.0))
                    continue
                if dd < 20 and lv_w is not None and src_ in ("lidar", "lidar_station", "lidar_unclean") \
                        and under_upper_deck(gx_, gy_, lv_w):
                    deck_caps.append((int(pts[kk]), float(z_) - HIDDEN_DECK_GAP, 1.0))
                    n_hidden += 1
                    continue
                if dd < 20 and src_ == "lidar_deck":
                    # under a bridge's upper roadway: the rail is at least the deck gap below it
                    # (structure_heights.fix_template_upper_deck), so a cap, applied in the rounds
                    deck_caps.append((int(pts[kk]), float(z_), max(float(s_), LIDAR_SD_FLOOR)))
                    continue
                if dd < 20 and ok_:
                    lidar_eqs.append((eq, int(pts[kk]), float(z_)))
                    add([(int(pts[kk]), 1.0)], float(z_), max(float(s_), LIDAR_SD_FLOOR))
                    n_lidar += 1
    if not quiet:
        print(f"  {n_lidar} lidar rail samples used, {n_lidar_out} dropped as off their neighbours, "
              f"{n_hidden} under a higher el deck used as caps", flush=True)

    A = sp.csr_matrix((vals, (rows, cols)), shape=(eq, n))
    sds = np.array(sds); rhs = np.array(rhs)
    # Subway grades top out near 3-4%. The grade term is a weak Gaussian, so where the solution
    # still goes steeper than GRADE_LIMIT, those pairs are held at the limit (below) and solved again.
    g_eq = np.array([e[0] for e in grade_eqs]); g_a = np.array([e[1] for e in grade_eqs])
    g_b = np.array([e[2] for e in grade_eqs]); g_h = np.array([e[3] for e in grade_eqs])
    # A tunnel can't be above its own ground: rail at least MIN_COVER below the street, and under
    # water at least UNDERWATER_CAP below sea level. One-sided, so it is switched on (a stiff pull
    # to the cap) only at points that break it, in the same rounds as the grade limit.
    tun = np.array([c == "tunnel" for c in cls])
    wet = wet_pt
    cap = np.where(wet, UNDERWATER_CAP, gsm - MIN_COVER * np.clip(portal_s / max(PORTAL_TAPER, 1e-9), 0, 1))
    capsd = np.where(wet, 4.0, 1.0)
    hasbed = wet & ~np.isnan(bed)
    cap[hasbed] = bed[hasbed] - BED_COVER       # rail under the riverbed: crown plus a little cover
    capsd[hasbed] = 1.5
    # rail under a bridge's upper roadway: at most the roadway minus the deck gap
    for i, zc, sc in deck_caps:
        cap[i] = min(cap[i], zc) if tun[i] else zc
        capsd[i] = min(capsd[i], sc) if tun[i] else sc
        tun[i] = True
    if not quiet and deck_caps:
        print(f"  {len(deck_caps)} lidar samples under a bridge's upper roadway used as caps", flush=True)
    # published limits ("at least 20 ft of water over the roof") tighten the cap on their ways
    for _, r in load_placed("bounds").iterrows():
        if r["bound"] != "max" or str(r["id"]) in SKIP_PLACED:
            continue
        for i in placed_points(r, ways, xs, ys, way_of):
            cap[i] = min(cap[i], r["rail_navd88_m"]); capsd[i] = min(capsd[i], r["sd_m"])
            tun[i] = True
    # underground stations: rail at least STATION_FLOOR under the street (see the station section above)
    for i, zc in floor_caps:
        cap[i] = min(cap[i], zc); capsd[i] = min(capsd[i], 0.7)
    wcap = np.zeros(n)
    # Stacked tunnels (two levels under one street, and line crossings): the upper rail at least
    # STACK_SEP above the lower. Same active-set treatment, as pairwise terms.
    stacks = stacked_pairs(xs, ys, cls, ways, way_of, seqs)
    s_r, s_c, s_v = [], [], []
    for k, (up, lo) in enumerate(stacks):
        for j, w_ in up:
            s_r.append(k); s_c.append(j); s_v.append(w_)
        for j, w_ in lo:
            s_r.append(k); s_c.append(j); s_v.append(-w_)
    S = sp.csr_matrix((s_v, (s_r, s_c)), shape=(len(stacks), n))
    s_up = np.arange(len(stacks))
    wsep = np.zeros(len(stacks))
    SEP_SD = 0.5
    # per row: target gap, sd of the pull, sd of the target in the error-bar samples. Stacked rows
    # are STACK_SEP / SEP_SD / 0; crossings (crossing_pairs) are added after round 0
    s_tgt = np.full(len(stacks), STACK_SEP); s_sd = np.full(len(stacks), SEP_SD); s_pert = np.zeros(len(stacks))
    n_stack = len(stacks)
    crossings = []

    # The grade limit is one-sided too: a step steeper than GRADE_LIMIT gets a stiff pull to the
    # limit itself (same sign), not to flat. Until 2026-11-02 the step's zero-mean grade term was
    # tightened instead (down to 1% sd), which held every once-steep step toward level: a ramp that
    # has to fall 20 m (the Manhattan Bridge tracks into Canal St) was pulled flat along its whole
    # length and came out 15 m too high.
    ng = len(g_a)
    Gm = sp.csr_matrix((np.r_[1 / g_h, -1 / g_h], (np.r_[np.arange(ng), np.arange(ng)], np.r_[g_b, g_a])), shape=(ng, n))
    wgr = np.zeros(ng); tgr = np.zeros(ng)
    # The approaches of the river bridges are the steepest track in the system (the Manhattan and
    # Williamsburg Bridges' are over 4%; the Williamsburg's lidar reads 4-5% straight off the
    # deck). Within APPROACH_M of a bridge span over water the limit is BRIDGE_APPROACH_GRADE.
    glim = np.full(ng, GRADE_LIMIT)
    over_water = np.array([c == "elevated" for c in cls]) & wet
    if over_water.any() and BRIDGE_APPROACH_GRADE > GRADE_LIMIT:
        from scipy.sparse.csgraph import dijkstra
        e_ = np.array(pairs, dtype=int)
        w_ = np.hypot(xs[e_[:, 1]] - xs[e_[:, 0]], ys[e_[:, 1]] - ys[e_[:, 0]]) + 1e-3
        d_ = dijkstra(sp.csr_matrix((w_, (e_[:, 0], e_[:, 1])), shape=(n, n)), directed=False,
                      indices=np.where(over_water)[0], limit=APPROACH_M, min_only=True)
        appr = np.isfinite(d_)
        glim[appr[g_a] | appr[g_b]] = BRIDGE_APPROACH_GRADE
        if not quiet:
            print(f"  {int((glim > GRADE_LIMIT).sum())} track steps within {APPROACH_M:.0f} m of a bridge over water: "
                  f"grade limit {BRIDGE_APPROACH_GRADE:.0%}", flush=True)

    # Vertical curves: a change of grade sharper than a VCURVE_R_MIN (600 m) radius is one no subway or
    # railroad builds (their vertical curves run 600-1000 m and more). The curvature term is a weak
    # Gaussian, so a bump forced by a crossing, an anchor or a tie could still come out sharp (103 St on
    # Central Park West, radius 220 m, 2026-11-03). One-sided like the grade limit: a step sharper than
    # the limit is pulled to the limit with its own sign.
    nc = len(curv_eqs)
    if nc:
        ca = np.array([c[0] for c in curv_eqs]); cb = np.array([c[1] for c in curv_eqs]); cc = np.array([c[2] for c in curv_eqs])
        h1a = np.array([c[3] for c in curv_eqs]); h2a = np.array([c[4] for c in curv_eqs])
        Cm = sp.csr_matrix((np.r_[1 / h2a, -1 / h2a - 1 / h1a, 1 / h1a],
                            (np.r_[np.arange(nc), np.arange(nc), np.arange(nc)], np.r_[cc, cb, ca])), shape=(nc, n))
        climit = (h1a + h2a) / 2 / VCURVE_R_MIN
    else:
        Cm = sp.csr_matrix((0, n)); climit = np.zeros(0)
    wcv = np.zeros(nc); tcv = np.zeros(nc)

    def system(wcap, wsep, wgr):
        W = sp.diags(1 / sds)
        Aw = W @ A
        N = (Aw.T @ Aw + sp.diags(wcap) + S.T @ sp.diags(wsep) @ S + Gm.T @ sp.diags(wgr) @ Gm
             + Cm.T @ sp.diags(wcv) @ Cm).tocsc()
        return Aw, spla.splu(N)

    for rnd in range(GRADE_ROUNDS + 1):
        Aw, lu = system(wcap, wsep, wgr)
        z = lu.solve(Aw.T @ (rhs / sds) + wcap * cap + S.T @ (wsep * s_tgt) + Gm.T @ (wgr * tgr) + Cm.T @ (wcv * tcv))
        if rnd == 0 and CROSS_ON:
            # tunnel crossings, ordered by OSM level or by this first solution
            crossings = crossing_pairs(xs, ys, cls, ways, seqs, z, gsm)
            if crossings:
                r_, c_, v_ = [], [], []
                for k, (up, lo, need, sdx, _) in enumerate(crossings):
                    for j, w_ in up:
                        r_.append(k); c_.append(j); v_.append(w_)
                    for j, w_ in lo:
                        r_.append(k); c_.append(j); v_.append(-w_)
                S = sp.vstack([S, sp.csr_matrix((v_, (r_, c_)), shape=(len(crossings), n))]).tocsr()
                s_tgt = np.r_[s_tgt, [c[2] for c in crossings]]
                s_sd = np.r_[s_sd, [c[3] for c in crossings]]
                s_pert = np.r_[s_pert, np.full(len(crossings), CROSS_GAP_SD)]
                wsep = np.r_[wsep, np.zeros(len(crossings))]
                gx = (S @ z)[n_stack:]
                if not quiet:
                    print(f"  {len(crossings)} tunnel crossings ({sum(c[4] for c in crossings)} ordered by OSM level); "
                          f"{int((gx < s_tgt[n_stack:] - 0.25).sum())} under their needed gap before the rule", flush=True)
        sg = (z[g_b] - z[g_a]) / g_h
        grade = np.abs(sg)
        steep = grade > glim + 0.002
        g_act = wgr > 0
        g_want = steep | (g_act & (grade > glim - 0.3 * GRADE_SD_ACTIVE))
        changed = (g_want != g_act).sum()
        # active set: pull points that are above their cap; let go of held points that the rest of
        # the system now puts clearly below it (a held point otherwise drags the rail up to the cap)
        active = wcap > 0
        want = tun & ((z > cap + 0.1) | (active & (z > cap - 0.3)))
        changed += (want != active).sum()
        gap = S @ z
        s_act = wsep > 0
        s_want = (gap < s_tgt - 0.1) | (s_act & (gap < s_tgt + 0.3))
        changed += (s_want != s_act).sum()
        dgc = Cm @ z
        c_act = wcv > 0
        c_want = (np.abs(dgc) > climit + 0.002) | (c_act & (np.abs(dgc) > climit - 0.3 * GRADE_SD_ACTIVE))
        changed += (c_want != c_act).sum()
        if not quiet:
            print(f"  round {rnd}: {steep.sum()} steps steeper than {GRADE_LIMIT:.1%} (max {grade.max():.1%}), "
                  f"{g_want.sum()} held at the limit, {want.sum()} tunnel points held at their cap, "
                  f"{s_want[:n_stack].sum()} of {len(s_up)} stacked pairs held {STACK_SEP:.0f} m apart, "
                  f"{s_want[n_stack:].sum()} of {len(crossings)} crossings held at their gap, "
                  f"{int(c_want.sum())} vertical curves held at {VCURVE_R_MIN:.0f} m ({changed} changed)", flush=True)
        if rnd == GRADE_ROUNDS or not changed:
            break
        tgr = np.where(g_want, np.sign(sg) * glim, 0.0)
        wgr = np.where(g_want, 1 / GRADE_SD_ACTIVE ** 2, 0.0)
        wcap = np.where(want, 1 / capsd ** 2, 0.0)
        wsep = np.where(s_want, 1 / s_sd ** 2, 0.0)
        tcv = np.where(c_want, np.sign(dgc) * climit, 0.0)
        wcv = np.where(c_want, 1 / GRADE_SD_ACTIVE ** 2, 0.0)
    if os.environ.get("TD_EXPLAIN"):
        # debugging: TD_EXPLAIN="lat,lon" lists the terms on the nearest point by how hard they pull
        la_, lo_ = (float(v) for v in os.environ["TD_EXPLAIN"].split(","))
        ex, ey = to_utm.transform(lo_, la_)
        p0 = int(np.argmin(np.hypot(xs - ex, ys - ey)))
        Ac = A.tocsc()
        r_ = (A @ z - rhs) / sds
        rows_ = Ac[:, p0].nonzero()[0]
        print(f"EXPLAIN point {p0} z {z[p0]:.2f} prior {prior_mu[p0]:.2f}+-{prior_sd[p0]:.2f} cls {cls[p0]}")
        for q in sorted(rows_, key=lambda q: -abs(r_[q] * Ac[q, p0] / sds[q]))[:15]:
            cols_ = A[q].nonzero()[1]
            print(f"  row {q}: resid/sd {r_[q]:+.2f} sd {sds[q]:.3f} coef {Ac[q, p0]:+.3f} pull {r_[q] * Ac[q, p0] / sds[q]:+.2f} "
                  f"cols {len(cols_)} target {rhs[q]:+.2f}")
        for name_, M_, w_, t_ in (("cap", None, wcap, cap), ("stack/cross", S, wsep, s_tgt), ("grade", Gm, wgr, tgr), ("curv", Cm, wcv, tcv)):
            if M_ is None:
                if w_[p0] > 0:
                    print(f"  cap active: cap {cap[p0]:.2f} z {z[p0]:.2f}")
                continue
            Mc = M_.tocsc()
            for q in Mc[:, p0].nonzero()[0]:
                if w_[q] > 0:
                    print(f"  {name_} row {q} active: value {(M_[q] @ z)[0]:+.3f} target {t_[q]:+.3f} coef {Mc[q, p0]:+.3f}")
    rng = np.random.default_rng(0)
    samples = []
    for _ in range(nsamples):
        pert = rhs + rng.normal(0, sds)
        samples.append(lu.solve(Aw.T @ (pert / sds) + wcap * (cap + rng.normal(0, capsd))
                                + S.T @ (wsep * (s_tgt + rng.normal(0, 1, len(s_tgt)) * s_pert))
                                + Gm.T @ (wgr * tgr) + Cm.T @ (wcv * tcv)))
    sd = np.std(np.array(samples), axis=0)
    info = {"docs": docs, "doc_status": doc_status, "doc_groups": doc_groups, "elev": elev_info,
            "n_elev": sum(1 for u in used if u["source"] == "elevator"),
            "n_doc": sum(1 for u in used if u["source"] == "document")}
    return z, sd, prior_mu, prior_sd, cls, used, info


# ---------- output ----------

PORTAL_MIN_TUNNEL = 60.0   # m: a tunnel run shorter than this (a street over an open cut) gets no portals
PORTAL_MIN_GAP = 50.0      # m: an open run this short between two tunnel ends is a tagging glitch
PORTAL_MERGE = 30.0        # m: portals of parallel tracks this close (and facing the same way) are one


def find_portals(ways, xs, ys, seqs, z):
    """Where track leaves a tunnel: one entry per tunnel end, parallel tracks merged.

    Runs are connected stretches of tunnel (or of non-tunnel) way edges, service tracks left out.
    A portal is a point shared by a tunnel run and an open run, unless the tunnel run is shorter
    than PORTAL_MIN_TUNNEL or the open run is shorter than PORTAL_MIN_GAP and has tunnel at two or
    more ends (a gap in the tags, not daylight). `bearing` points into the tunnel (degrees from
    north, clockwise)."""
    from scipy.sparse.csgraph import connected_components
    n = len(xs)
    edges = {True: [], False: []}
    way_at = collections.defaultdict(set)
    for wi, (w, seq) in enumerate(zip(ways, seqs)):
        if w["tags"].get("service") is not None:
            continue
        tun = classify(w["tags"]) == "tunnel"
        for a, b in zip(seq, seq[1:]):
            edges[tun].append((a, b))
        for i in seq:
            way_at[i].add((wi, tun))
    comp, length, touch = {}, {}, {}

    def components(edge_list):
        e = np.array(edge_list, dtype=int).reshape(-1, 2)
        G = sp.csr_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
        _, lab = connected_components(G, directed=False)
        L = collections.defaultdict(float)
        for a, b in e:
            L[lab[a]] += math.hypot(xs[b] - xs[a], ys[b] - ys[a])
        return lab, L
    comp[False], length[False] = components(edges[False])
    # A short open run with tunnel at two or more ends is a gap in the tags, not daylight; it is counted
    # as tunnel before tunnel runs are measured. The F's tunnel from Carroll St met a 3 m untagged way and
    # then a 20 m level -1 tunnel before the Culver Viaduct, so neither piece got a portal (2026-11-03).
    tun_nodes = {i for a, b in edges[True] for i in (a, b)}
    ends_of = collections.Counter()
    for a, b in edges[False]:
        for i in (a, b):
            if i in tun_nodes:
                ends_of[comp[False][i]] += 1
    gap_edges = [(a, b) for a, b in edges[False]
                 if length[False][comp[False][a]] < PORTAL_MIN_GAP and ends_of[comp[False][a]] >= 2]
    comp[True], length[True] = components(edges[True] + gap_edges)
    # portal nodes: on both a tunnel and an open (non-service) way
    nodes = [i for i, s in way_at.items() if any(t for _, t in s) and any(not t for _, t in s)]
    open_ends = collections.Counter(comp[False][i] for i in nodes)
    tun_nb = collections.defaultdict(list)
    for a, b in edges[True]:
        tun_nb[a].append(b); tun_nb[b].append(a)
    raw = []
    for i in nodes:
        if length[True][comp[True][i]] < PORTAL_MIN_TUNNEL:
            continue
        oc = comp[False][i]
        if length[False][oc] < PORTAL_MIN_GAP and open_ends[oc] >= 2:
            continue
        nb = tun_nb.get(i, [])
        if not nb:
            continue
        dx = float(np.mean([xs[j] - xs[i] for j in nb])); dy = float(np.mean([ys[j] - ys[i] for j in nb]))
        lines = sorted({ways[wi]["tags"].get("name", "") for wi, t in way_at[i] if t} - {""})
        raw.append({"i": i, "x": xs[i], "y": ys[i], "z": float(z[i]), "dx": dx, "dy": dy, "lines": set(lines)})
    # merge parallel tracks' portals
    used = [False] * len(raw)
    out = []
    for k, p in enumerate(raw):
        if used[k]:
            continue
        grp = [p]; used[k] = True
        h0 = math.atan2(p["dx"], p["dy"])
        for m in range(k + 1, len(raw)):
            q = raw[m]
            if used[m] or math.hypot(q["x"] - p["x"], q["y"] - p["y"]) > PORTAL_MERGE:
                continue
            if math.cos(math.atan2(q["dx"], q["dy"]) - h0) < 0.5:
                continue
            grp.append(q); used[m] = True
        x = float(np.mean([g["x"] for g in grp])); y = float(np.mean([g["y"] for g in grp]))
        dx = sum(g["dx"] / (math.hypot(g["dx"], g["dy"]) or 1) for g in grp)
        dy = sum(g["dy"] / (math.hypot(g["dx"], g["dy"]) or 1) for g in grp)
        lon, lat = to_ll.transform(x, y)
        out.append({"lon": round(float(lon), 6), "lat": round(float(lat), 6),
                    "z": round(float(np.mean([g["z"] for g in grp])), 1),
                    "lines": sorted(set().union(*[g["lines"] for g in grp])),
                    "bearing": round(math.degrees(math.atan2(dx, dy)) % 360, 0), "tracks": len(grp)})
    return out

PLATFORM_NEAR_M = 8.0   # m from an OSM platform's outline to a station level's track: that level has a platform
RAIL_KEYS = (("Long Island", ("Long Island",)), ("LIRR", ("Long Island",)), ("Metro-North", ("Metro-North",)),
             ("Amtrak", ("Amtrak",)), ("New Jersey Transit", ("New Jersey Transit", "NJ Transit")),
             ("NJ Transit", ("New Jersey Transit", "NJ Transit")), ("PATH", ("Trans-Hudson",)),
             ("Trans-Hudson", ("Trans-Hudson",)), ("Port Authority", ("Trans-Hudson",)))


def build_stations(ways, xs, ys, way_of, z, sd, gsm, surf):
    """Station dots for tracks.json: name, line, lon, lat, level, tier, routes, ground, z, sd.

    Subway and SIR: one per MTA station row (station and line) and platform level. A level is a
    platform level when an OSM platform (not disused) lies within PLATFORM_NEAR_M of its tracks (and
    its level tag, if any, agrees); where no level has one, a stop position within STOP_NEAR_M of the
    station on its tracks; where none has that either, the level with the most track. (2026-11-02 used
    stop positions only, which kept Bergen St's closed lower level; express levels with no platform,
    under Central Park West's local stations, never count.) Levels within LEVEL_Z_SAME m are one.
    `tier` ("upper level", "lower level", "middle level") only where two or more levels have platforms.
    `routes`: the services (data/services.json refs) that run on the dot's tracks and stop within
    250 m, so a label can say "72 St · B C" instead of the MTA's line name ("8th Av - Fulton St").

    Other rail: OSM stations, on tracks of the station's own railway (by its operator or network)
    within STATION_RADIUS, else 200 m (Far Rockaway's platform tracks are yard in OSM); then a dot for
    every stop of a railroad or PATH service with no dot of its railway within 300 m, one per level of
    that service's tracks there (Grand Central Madison: OSM has no station node, only Metro-North's
    Grand Central Terminal, whose dot it used to share)."""
    from shapely.geometry import LineString, Point, Polygon
    els_ = json.loads((RAW / "osm_stations.json").read_text(encoding="utf-8"))["elements"]
    stop_xy = np.array([to_utm.transform(e["lon"], e["lat"]) for e in els_
                        if e["type"] == "node" and e.get("tags", {}).get("public_transport") == "stop_position"])
    way_keys = [{key(p["lon"], p["lat"]) for p in w["geometry"]} for w in ways]
    # OSM platforms as sample points every 4 m along their outline, with their level tag
    plat_xy, plat_lv = [], []
    pp = RAW / "osm_platforms.json"
    if pp.exists():
        for e in json.loads(pp.read_text(encoding="utf-8"))["elements"]:
            tg = e.get("tags", {})
            if e["type"] != "way" or len(e.get("geometry") or []) < 2 or tg.get("disused") == "yes" \
                    or tg.get("railway") in ("disused", "abandoned") \
                    or any(k.startswith(("disused", "abandoned", "construction", "razed")) for k in tg):
                continue
            ln = LineString([to_utm.transform(p["lon"], p["lat"]) for p in e["geometry"]])
            lvt = None
            for k_ in ("level", "layer"):
                try:
                    lvt = float(str(tg[k_]).split(";")[0]); break
                except (KeyError, ValueError):
                    pass
            for s in np.arange(0, ln.length + 0.1, 4.0):
                q = ln.interpolate(s)
                plat_xy.append((q.x, q.y)); plat_lv.append(lvt)
    plat_tree = cKDTree(np.array(plat_xy)) if plat_xy else None

    def has_platform(pts, lv):
        if plat_tree is None:
            return False
        hits = 0
        for i in pts:
            for k_ in plat_tree.query_ball_point([xs[i], ys[i]], PLATFORM_NEAR_M):
                if plat_lv[k_] is None or lv is None or abs(plat_lv[k_] - lv) < 0.01:
                    hits += 1
                    break
        return hits >= 2

    # services: ways they run on, stops
    svc_ways, svc_stops, svc_refs = [], [], []
    sj = OUT / "services.json"
    if sj.exists():
        for s in json.loads(sj.read_text(encoding="utf-8")).get("services", []):
            wids, stops = set(), []
            for v in s.get("variants", []):
                if "late night" in v.get("when", ""):
                    continue   # the A and 4 run local at night: not what stops there
                wids.update(int(w[0]) for w in v.get("ways", []))
                stops.extend(v.get("stops", []))
            svc_refs.append(s["ref"]); svc_ways.append(wids)
            svc_stops.append(np.array([to_utm.transform(a, b) for a, b, *_ in stops]) if stops else np.zeros((0, 2)))

    svc_label = {}
    if sj.exists():
        svc_label = {s["ref"]: s.get("label", s["ref"]) for s in json.loads(sj.read_text(encoding="utf-8")).get("services", [])}
    daytime = {(r["Stop Name"], r["Line"]): set(str(r["Daytime Routes"]).split())
               for _, r in pd.read_csv(RAW / "mta_stations.csv", dtype=str).iterrows()}

    def routes_at(pts, x0, y0, name=None, line=None):
        wid = {ways[way_of[i]]["id"] for i in pts}
        ops = {ways[way_of[i]]["tags"].get("operator", "").split(" ")[0] for i in pts} - {""}
        out, near_only = [], []
        for r, wset, st in zip(svc_refs, svc_ways, svc_stops):
            if not len(st) or np.hypot(st[:, 0] - x0, st[:, 1] - y0).min() >= 250:
                continue
            if wid & wset:
                out.append(r)
            elif ops and any(ways_op.get(w, "").startswith(tuple(ops)) for w in wset):
                near_only.append(r)   # stops here, but on tracks the map has as yard (terminals)
        out = out or near_only
        day = daytime.get((name, line))
        if day:   # subway: only what the MTA lists as stopping here by day
            out = [r for r in out if svc_label.get(r, r) in day]
        return out

    ways_op = {w["id"]: w["tags"].get("operator", "") for w in ways}

    def dot(name, line, lon_, lat_, level, tier, pts, x0, y0):
        # ground: the track's own ground at its point nearest the dot, so the dot's depth is the track's there
        # (the mean over the platform's length differed from it by metres on slopes and at a cut's end)
        gr = gsm[pts]
        ok_ = [i for i in pts if np.isfinite(gsm[i])]
        g_at = float(gsm[min(ok_, key=lambda i: (xs[i] - x0) ** 2 + (ys[i] - y0) ** 2)]) if ok_ else None
        return {"name": name, "line": line, "lon": round(float(lon_), 6), "lat": round(float(lat_), 6),
                "level": level, "tier": tier, "routes": routes_at(pts, x0, y0, name, line),
                "ground": None if g_at is None else round(g_at, 1),
                "z": round(float(np.mean(z[pts])), 1), "sd": round(float(np.mean(sd[pts])), 1), "_x": x0, "_y": y0}

    def by_height(groups_):
        """Sort (g, pts) by rail height, merging levels within LEVEL_Z_SAME."""
        groups_ = sorted(groups_, key=lambda gp: -float(np.mean(z[gp[1]])))
        merged = []
        for g, p in groups_:
            if merged and abs(float(np.mean(z[merged[-1][1]])) - float(np.mean(z[p]))) < LEVEL_Z_SAME:
                merged[-1] = (g if len(p) > len(merged[-1][1]) else merged[-1][0], merged[-1][1] + p)
            else:
                merged.append((g, p))
        return merged

    def tiers(n_):
        return ["" if n_ == 1 else ("upper level" if r == 0 else "lower level" if r == n_ - 1 else "middle level")
                for r in range(n_)]

    by_st = collections.defaultdict(list)
    for g in station_groups(ways, xs, ys, way_of):
        pts = [i for v in g["ways"].values() for i in v]
        by_st[(g["name"], g["line"], round(g["lon"], 5))].append((g, pts))
    stations = []
    n_multi = 0
    for groups_ in by_st.values():
        g0 = groups_[0][0]
        sx_, sy_ = to_utm.transform(g0["lon"], g0["lat"])
        keep = [(g, p) for g, p in groups_ if has_platform(p, g["level"])]
        if not keep:
            near_stops = set()
            if len(stop_xy):
                for (qx, qy) in stop_xy[np.hypot(stop_xy[:, 0] - sx_, stop_xy[:, 1] - sy_) < STOP_NEAR_M]:
                    lo_, la_ = to_ll.transform(qx, qy)
                    near_stops.add(key(lo_, la_))
            keep = [(g, p) for g, p in groups_ if any(way_keys[wi] & near_stops for wi in g["ways"])]
        if not keep:
            keep = [max(groups_, key=lambda gp: len(gp[1]))]
        keep = by_height(keep)
        n_multi += len(keep) > 1
        for (g, pts), tier in zip(keep, tiers(len(keep))):
            stations.append(dot(g["name"], g["line"], g["lon"], g["lat"], g["level"], tier, pts, sx_, sy_))
    print(f"  {n_multi} station lines with platforms on more than one level: one dot per level", flush=True)

    # ---- other rail ----
    other = [wi for wi, w in enumerate(ways) if not ("Transit Authority" in w["tags"].get("operator", "")
                                                       or "Staten Island" in w["tags"].get("operator", "") + w["tags"].get("name", ""))]
    if not other:
        return stations
    om = np.isin(way_of, other)
    tree = cKDTree(np.c_[xs[om], ys[om]]); oidx = np.where(om)[0]
    seen = set()
    is_path = lambda tg: "Trans-Hudson" in tg.get("operator", "") or tg.get("network") == "PATH"

    def same_railway(s_line, ops):
        return any(o and o.split()[0] in s_line for o in ops)

    for e in els_:
        t = e.get("tags", {})
        # NYC subway stations come from the MTA table above; PATH's are station=subway in OSM too
        # (and anything NYC Transit runs: OSM's "34th Street-Herald Square" outline is train=yes)
        mta_subway = (t.get("station") == "subway" and "Trans-Hudson" not in t.get("operator", "")
                      and t.get("network") != "PATH") or "New York City Transit" in t.get("operator", "")
        if t.get("railway") != "station" or mta_subway or not t.get("name"):
            continue
        lon_, lat_ = (e["lon"], e["lat"]) if "lon" in e else (e["center"]["lon"], e["center"]["lat"])
        sx, sy = to_utm.transform(lon_, lat_)
        own = next((ks for kw, ks in RAIL_KEYS if kw in t.get("operator", "") + " " + t.get("network", "")), None)
        near = []
        for rad in (STATION_RADIUS, 200.0):
            near = [int(oidx[k]) for k in tree.query_ball_point([sx, sy], rad)]
            if own:   # the station's own railway's tracks (Grand Central Terminal is Metro-North's, not the LIRR's under it)
                mine = [i for i in near if any(o in ways[way_of[i]]["tags"].get("operator", "") for o in own)]
                near = mine
            if near:
                break
        if not near or (t["name"], round(lon_, 3)) in seen:
            continue
        # Where only PATH track is in reach, only a PATH station counts: OSM's Hudson-Bergen Light
        # Rail, NJ Transit and Amtrak stations (Exchange Place, Harsimus Cove, Hoboken Terminal,
        # Newark Penn) sit beside PATH and came out as extra PATH dots beside the real ones.
        if all(is_path(ways[way_of[i]]["tags"]) for i in near) and not is_path(t):
            continue
        ops = {ways[way_of[i]]["tags"].get("operator", "") for i in near}
        line_ = ", ".join(sorted(o for o in ops if o))[:60]
        if any(s_["line"] == line_ and (s_["name"] == t["name"] and math.hypot(s_["_x"] - sx, s_["_y"] - sy) < 300
                                        or math.hypot(s_["_x"] - sx, s_["_y"] - sy) < 200)
               for s_ in stations if "_x" in s_ and s_.get("level") is None):
            # the same station of the same railway mapped twice (node and outline; "Pennsylvania
            # Station" beside "New York Penn Station", "Broad Street Station" beside "Newark Broad Street")
            continue
        seen.add((t["name"], round(lon_, 3)))
        stations.append(dot(t["name"], line_, lon_, lat_, None, "", near, sx, sy))
    # stops of railroad, PATH and AirTrain services with no dot of their railway near
    n_svc = 0
    by_id = {w["id"]: wi for wi, w in enumerate(ways)}
    done = set()
    for r, wset, st, s in zip(svc_refs, svc_ways, svc_stops,
                              json.loads(sj.read_text(encoding="utf-8")).get("services", []) if sj.exists() else []):
        if not any(r.startswith(p) for p in ("LIRR", "MNR", "NJT", "Amtrak", "PATH", "AirTrain")):
            continue
        names = [nm for v in s.get("variants", []) if "late night" not in v.get("when", "") for *_, nm in v.get("stops", [])]
        wis = [by_id[w] for w in wset if w in by_id]
        if not wis:
            continue
        # every track of the service's railway near its stops (a relation lists one level of Grand
        # Central Madison's caverns, not both)
        sops = {ways[wi]["tags"].get("operator", "") for wi in wis} - {""}
        wis = [wi for wi, w in enumerate(ways) if w["tags"].get("operator", "") in sops] or wis
        sm = np.isin(way_of, wis)
        sidx = np.where(sm)[0]
        stree = cKDTree(np.c_[xs[sm], ys[sm]])
        for (qx, qy), nm in zip(st, names):
            ops = {ways[wi]["tags"].get("operator", "") for wi in wis}
            if any(math.hypot(s_["_x"] - qx, s_["_y"] - qy) < 300 and same_railway(s_["line"], ops) for s_ in stations) \
                    or (nm, round(qx / 100), round(qy / 100)) in done:
                continue
            done.add((nm, round(qx / 100), round(qy / 100)))
            near = [int(sidx[k]) for k in stree.query_ball_point([qx, qy], 150.0)]
            if not near:
                continue
            lv_groups = collections.defaultdict(list)
            for i in near:
                tg = ways[way_of[i]]["tags"]
                lv_groups[level_of(tg) if classify(tg) == "tunnel" else None].append(i)
            groups_ = by_height([(lv, p) for lv, p in lv_groups.items() if len(p) >= 3] or [(None, near)])
            lon_, lat_ = to_ll.transform(qx, qy)
            line_ = ", ".join(sorted({ways[way_of[i]]["tags"].get("operator", "") for i in near} - {""}))[:60]
            for (lv, pts), tier in zip(groups_, tiers(len(groups_))):
                stations.append(dot(nm, line_, lon_, lat_, lv, tier, pts, qx, qy))
                n_svc += 1
    print(f"  {n_svc} dots from railroad and PATH service stops with no OSM station near", flush=True)
    return stations


def write_doc_rows(info):
    """data/doc_rows.csv: what happened to each documented depth row (validate.py reads it), and a
    printed count of the depths.csv rows that are not used yet."""
    recs = []
    for row in info["docs"]:
        st, why, lvl = info["doc_status"].get(row["id"], ("skipped", row["why"] or "", None))
        recs.append({"id": row["id"], "status": st, "reason": why, "osm_level": lvl})
    pd.DataFrame(recs).to_csv(OUT / "doc_rows.csv", index=False)
    c = collections.Counter(r["status"] for r in recs)
    print("document rows: " + ", ".join(f"{k} {v}" for k, v in c.items()), flush=True)
    reasons = collections.Counter(re.sub(r"\d+", "#", r["reason"]) for r in recs if r["status"] == "skipped")
    for k, v in reasons.most_common():
        print(f"  skipped {v}: {k}")
    ev = pd.read_csv(ROOT / "evidence" / "depths.csv", dtype=str).fillna("")
    other = ev[~ev["measure"].isin(DOC_MEASURES)]
    absolute = other["measure"].isin(ABSOLUTE_MEASURES).sum()
    print(f"  depths.csv: {len(ev)} rows; {len(other)} not a depth below/above the street "
          f"({absolute} absolute-datum, {len(other) - absolute} relative or other)")


def main():
    OUT.mkdir(exist_ok=True)
    ways = load_tracks()
    xs, ys, way_of, seqs = build_points(ways)
    print(f"{len(ways)} ways, {len(xs)} points; {build_points.n_split} vertices shared by tunnels of "
          f"different levels drawn on top of each other, split by level ({build_points.n_through} where each "
          f"level runs through end to end)", flush=True)
    print(f"{drop_isolated.m} pieces of track under {ISOLATED_M:.0f} m joined to nothing and with no station near: "
          f"{drop_isolated.n} ways dropped", flush=True)
    print(f"{separate_dead_ends.n_nodes} end nodes where several ways end with nothing beyond: "
          f"{separate_dead_ends.n_cut} way ends cut back to end side by side", flush=True)
    ground = sample_ground(xs, ys)
    miss = np.isnan(ground)
    if miss.any() and (~miss).any():
        # track at the DEM's edge (Port Washington's terminal tracks): the nearest point that has ground
        _, kk = cKDTree(np.c_[xs[~miss], ys[~miss]]).query(np.c_[xs[miss], ys[miss]])
        ground[miss] = ground[~miss][kk]
    gsm = smooth_along(seqs, ground)
    print(f"ground missing at {np.isnan(ground).sum()} points", flush=True)
    surf = surface_ground(xs, ys, seqs)
    z, sd, pmu, psd, cls, used, info = solve(ways, xs, ys, way_of, seqs, ground, gsm, surf=surf)
    print(f"{info['n_elev']} elevator anchors, {info['n_doc']} document anchors used", flush=True)
    pd.DataFrame(used).to_csv(OUT / "anchors.csv", index=False)
    write_doc_rows(info)
    lon, lat = to_ll.transform(xs, ys)
    bed = load_riverbed(xs, ys)
    feats = []
    # `ground` is the surface a person stands on: the street over a tunnel (the DEM at the track), and for
    # open track the surface beside it (surface_ground), so a viaduct, a causeway or an embankment shows
    # its height over the street and an open cut its depth under it (2026-11-03; until then the DEM at the
    # track, which keeps solid structure: the Hell Gate approach read 0 m above the surface)
    # One ground per point, for the tracks and the station dots (and so the boxes) alike: a dot used the
    # DEM at the track while open track used the surface beside it, so New Dorp's open-cut station said
    # +1 m above the surface over track drawn 5.6 m under it (2026-11-03)
    is_tun_pt = np.array([c == "tunnel" for c in cls])
    shown = np.where(is_tun_pt | ~np.isfinite(surf), gsm, surf)
    for w, seq in zip(ways, seqs):
        t = w["tags"]
        gshow = shown
        feats.append({
            "id": w["id"],
            "name": t.get("name", ""),
            "track": t.get("railway:track_ref", ""),
            "service": t.get("service", ""),
            "cls": classify(t),
            "level": level_of(t),
            "lon": [round(float(lon[i]), 6) for i in seq],
            "lat": [round(float(lat[i]), 6) for i in seq],
            "ground": [None if np.isnan(gshow[i]) else round(float(gshow[i]), 1) for i in seq],
            "z": [round(float(z[i]), 1) for i in seq],
            "sd": [round(float(sd[i]), 1) for i in seq],
        })
        if not np.isnan(bed[seq]).all():   # riverbed under the track (m NAVD88), only where it crosses water
            feats[-1]["bed"] = [None if np.isnan(bed[i]) else round(float(bed[i]), 1) for i in seq]
    stations = build_stations(ways, xs, ys, way_of, z, sd, shown, surf)
    for s_ in stations:
        s_.pop("_x", None); s_.pop("_y", None)
    print(f"{len(stations)} station dots", flush=True)
    portals = find_portals(ways, xs, ys, seqs, z)
    print(f"{len(portals)} portals (tunnel ends, parallel tracks merged)", flush=True)
    (OUT / "tracks.json").write_text(json.dumps({"tracks": feats, "anchors": used, "stations": stations,
                                                 "portals": portals},
                                                separators=(",", ":")), encoding="utf-8")
    dep = gsm - z
    tun = np.array([c == "tunnel" for c in cls]) & ~np.isnan(gsm)
    print(f"tunnel depth below street: median {np.nanmedian(dep[tun]):.1f} m, p95 {np.nanpercentile(dep[tun], 95):.1f} m")
    print(f"sd: median {np.median(sd):.1f} m, p95 {np.percentile(sd, 95):.1f} m")
    print(f"wrote {OUT / 'tracks.json'} ({(OUT / 'tracks.json').stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
