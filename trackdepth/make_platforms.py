"""Station boxes for the 3D page: data/platforms.json from OSM platforms and data/tracks.json.

    python make_platforms.py           # uses raw/osm_platforms.json, fetching it first if missing
    python make_platforms.py --fetch   # fetch it again (Overpass, a minute when the servers are free)

Run after build.py: heights come from data/tracks.json. A few seconds.

Matching, per OSM platform (railway=platform, or public_transport=platform for trains, subway,
light rail; disused and abandoned ones skipped):
- tracks beside it: tracks.json ways (no yards or crossovers) within BESIDE_M of the platform,
  running alongside it (the stretch of track within BESIDE_M at least MIN_OVERLAP of the
  platform's length, within MAX_ANGLE degrees of its axis).
- level: where the platform's OSM `level` (else `layer`) equals the `level` of some of those
  tracks, only those tracks; else the level of the nearest track. So a platform on a stacked
  line (Central Park West, 145 St, W 4 St, the Lexington two-level stretch) takes its own level's
  rail, not the one drawn on top of it.
- height: the rail height of those tracks (tracks.json z, interpolated along the track at the
  platform's position); platform top = rail + 1.1 m (as in build.py).
- station: the nearest station dot of tracks.json (one per station, line and platform level, since
  2026-11-02; a dot of another OSM level than the box's only when its own level has none) within
  DOT_M whose line matches the tracks (build.py's LINE_WORDS for the subway, the operator for other rail);
  failing that the nearest dot of the same railway within NEAR_DOT_M (the lower levels of 145 St
  and W 4 St, whose lines have no dot of their own); failing that, if an OSM station is within
  OSM_STATION_M, the box is kept with dot null and that station's name (the New Lots line and most
  PATH stations, which have no dot on the page); else the platform is dropped (disused or yard).
  Platforms of two railways (the LIRR beside the subway at Atlantic Av) are never one box.

Boxes: platforms of one level whose tracks are shared or whose footprints come within
2 x CLOSE_M, roughly parallel and within 2.5 m of rail height, are one station box. Its
footprint is the union of the platforms and the tracks beside them (each track buffered by
TRACK_HALF m, cut to the platforms' length), with gaps up to 2 x CLOSE_M closed (express tracks
running between two side platforms). Each footprint vertex carries the rail height of the
nearest of its tracks.

Fallbacks: a station dot with no box (and not inside a box of its line given to a neighbouring
dot) gets one of a typical length centred on the dot along its own tracks (subway A division
160 m, B division 180 m, ...), marked `fallback`.

    --list-no-dot   also prints the platform groups dropped for want of a station

data/platforms.json:
  boxes:     station, line (the dot's, '' if none), dot (index into tracks.json stations, or null), lines (OSM track
             names), level, length (m), rail_z (m NAVD88, mean), fallback, ring [[lon, lat]...]
             (not closed), z [rail height per ring vertex], tracks [OSM way ids of its tracks,
             as tracks.json `id`]
  platforms: id (OSM, w123 / r123), box, station, lines, level (OSM tag), track_level (the
             tracks it took its height from), kind (area / line), length, width (m, 0 for a
             line), rail_z, platform_z, geom [[lon, lat]...], slab [[lon, lat]...] (the
             platform surface the page draws), slab_z (platform height per slab vertex)
"""
import collections
import json
import math
import pathlib
import sys

import numpy as np
import requests
from pyproj import Transformer
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.ops import polygonize, substring, unary_union
from shapely.strtree import STRtree

ROOT = pathlib.Path(__file__).parent
RAW = ROOT / "raw"
OUT = ROOT / "data"
sys.path.insert(0, str(ROOT))
from build import LINE_WORDS, PLATFORM_ABOVE_RAIL  # noqa: E402

UA = "trackdepth/0.1 (map of NYC subway track depth; personal research)"
# the map's square (fetch_osm.SQUARE), south, west, north, east. raw/osm_platforms.json was fetched
# for the older, larger terrain extent (40.508,-74.342,41.032,-73.645), which contains it.
BBOX = "40.508,-74.26,40.975,-73.645"

PLATFORMS = f"""
[out:json][timeout:300];
(
  way["railway"="platform"]({BBOX});
  relation["railway"="platform"]({BBOX});
  way["public_transport"="platform"]["train"="yes"]({BBOX});
  way["public_transport"="platform"]["subway"="yes"]({BBOX});
  way["public_transport"="platform"]["light_rail"="yes"]({BBOX});
  way["public_transport"="platform"]["monorail"="yes"]({BBOX});
);
out tags geom;
"""
ENDPOINTS = ("https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter",
             "https://overpass.private.coffee/api/interpreter")

to_utm = Transformer.from_crs(4326, 32618, always_xy=True)
to_ll = Transformer.from_crs(32618, 4326, always_xy=True)

BESIDE_M = 7.0       # m from a platform's outline to the centre of a track beside it
MIN_OVERLAP = 0.35   # of the platform's length (or 40 m, whichever is less) the track runs alongside
MAX_ANGLE = 30.0     # degrees between the track and the platform's axis
TRACK_HALF = 2.2     # m: half the width a track takes in a box (a car is ~3 m wide)
LINE_HALF = 1.6      # m: half the width of a platform drawn as a line
CLOSE_M = 6.0        # m: footprint gaps up to twice this are closed
DOT_M = 160.0        # m from a box to its station dot
NEAR_DOT_M = 120.0   # m: failing a dot of its line, a box takes the nearest dot of its railway this close
OSM_STATION_M = 200.0  # m: failing any dot, a box is kept (with no dot) if an OSM station is this close
LEVEL_Z = 2.5        # m: platforms of one box are within this of each other in rail height
SKIP_SERVICE = {"yard", "crossover"}

# fallback box lengths (m) by line; MTA lines of the A division (IRT) are 160, the rest of the
# subway 180
A_DIVISION = {"Broadway - 7Av", "Clark St", "Lexington Av", "Lexington - Shuttle", "Lenox - White Plains Rd",
              "Eastern Pky", "Nostrand", "Jerome Av", "Pelham", "Dyre Av", "Flushing"}
OTHER_LENGTH = {"Staten Island": 140, "Port Authority Trans-Hudson": 150, "Long Island Rail Road": 300,
                "Metro-North": 300, "Amtrak": 300, "Alstom": 70}


def fetch(q, out):
    """Polite: one request at a time, backing off when the servers are busy."""
    import time
    r = None
    for attempt in range(6):
        for url in ENDPOINTS:
            try:
                r = requests.post(url, data={"data": q}, headers={"User-Agent": UA}, timeout=600)
            except requests.RequestException as e:
                print(f"{url}: {e}", flush=True)
                continue
            if r.ok:
                break
            print(f"{url}: {r.status_code} {r.text[:120]!r}", flush=True)
        if r is not None and r.ok:
            break
        wait = 30 * 2 ** attempt
        print(f"all busy; waiting {wait} s", flush=True)
        time.sleep(wait)
    r.raise_for_status()
    out.write_bytes(r.content)
    n = len(json.loads(r.content)["elements"])
    print(f"{out.name}: {n} elements, {len(r.content)/1e6:.1f} MB")


# ---------- inputs ----------

def utm_line(g):
    x, y = to_utm.transform([p["lon"] for p in g], [p["lat"] for p in g])
    return list(zip(x, y))


def level_tag(t):
    """The platform's OSM level (else layer) as a list of numbers, [] if none."""
    for k in ("level", "layer"):
        v = t.get(k)
        if v is None:
            continue
        out = []
        for part in v.replace(",", ";").split(";"):
            try:
                out.append(float(part))
            except ValueError:
                pass
        if out:
            return out
    return []


def load_platforms(path):
    els = json.loads(path.read_text(encoding="utf-8"))["elements"]
    out, skipped = [], collections.Counter()
    seen = set()
    for e in els:
        t = e.get("tags", {})
        oid = ("w" if e["type"] == "way" else "r") + str(e["id"])
        if oid in seen:
            continue
        seen.add(oid)
        if any(k.startswith(("disused", "abandoned", "razed", "demolished", "construction")) for k in t) \
                or t.get("railway") in ("disused", "abandoned") or t.get("disused") == "yes":
            skipped["disused or abandoned"] += 1
            continue
        if t.get("highway") or t.get("bus") == "yes" and t.get("railway") != "platform":
            skipped["bus platform"] += 1
            continue
        geom = None
        if e["type"] == "way":
            pts = utm_line(e["geometry"])
            if len(pts) < 2:
                continue
            closed = len(pts) >= 4 and math.hypot(pts[0][0] - pts[-1][0], pts[0][1] - pts[-1][1]) < 0.01
            if closed and t.get("area") != "no":
                geom = Polygon(pts).buffer(0)
            else:
                geom = LineString(pts)
        else:
            outers = [LineString(utm_line(m["geometry"])) for m in e.get("members", [])
                      if m.get("type") == "way" and m.get("role") in ("outer", "") and len(m.get("geometry", [])) > 1]
            polys = list(polygonize(unary_union(outers))) if outers else []
            if polys:
                geom = unary_union(polys)
                if isinstance(geom, MultiPolygon):
                    geom = max(geom.geoms, key=lambda g: g.area)
        if geom is None or geom.is_empty:
            skipped["no geometry"] += 1
            continue
        out.append({"id": oid, "tags": t, "geom": geom, "kind": "area" if isinstance(geom, Polygon) else "line"})
    return out, skipped


def load_tracks():
    tj = json.loads((OUT / "tracks.json").read_text(encoding="utf-8"))
    raw = json.loads((RAW / "osm_tracks.json").read_text(encoding="utf-8"))["elements"]
    op = {e["id"]: e["tags"].get("operator", "") for e in raw}
    tracks = []
    for ti, t in enumerate(tj["tracks"]):
        if len(t["lon"]) < 2 or t["service"] in SKIP_SERVICE:
            continue
        x, y = to_utm.transform(t["lon"], t["lat"])
        line = LineString(list(zip(x, y)))
        if line.length < 1:
            continue
        cum = np.r_[0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
        tracks.append({"ti": ti, "id": t["id"], "name": t["name"], "level": t["level"], "cls": t["cls"],
                       "operator": op.get(t["id"], ""), "line": line, "cum": cum, "z": np.array(t["z"], float),
                       "ground": np.array([np.nan if g is None else g for g in t["ground"]], float),
                       "service": t["service"]})
    return tj, tracks


def z_at(tr, pt):
    """Rail height of a track at the point of it nearest pt."""
    s = tr["line"].project(pt)
    return float(np.interp(s, tr["cum"], tr["z"]))


def axis_of(geom):
    """(length, width, unit axis vector) of a platform."""
    if isinstance(geom, Polygon):
        r = geom.minimum_rotated_rectangle
        c = list(r.exterior.coords)
        a = math.hypot(c[1][0] - c[0][0], c[1][1] - c[0][1])
        b = math.hypot(c[2][0] - c[1][0], c[2][1] - c[1][1])
        if a >= b:
            v = (c[1][0] - c[0][0], c[1][1] - c[0][1]); L, W = a, b
        else:
            v = (c[2][0] - c[1][0], c[2][1] - c[1][1]); L, W = b, a
        n = math.hypot(*v) or 1
        return L, W, (v[0] / n, v[1] / n)
    c = list(geom.coords)
    v = (c[-1][0] - c[0][0], c[-1][1] - c[0][1])
    n = math.hypot(*v) or 1
    return geom.length, 0.0, (v[0] / n, v[1] / n)


def angle_between(v, w):
    d = abs(v[0] * w[0] + v[1] * w[1])
    return math.degrees(math.acos(min(1.0, d)))


def track_dir(line):
    c = list(line.coords)
    v = (c[-1][0] - c[0][0], c[-1][1] - c[0][1])
    n = math.hypot(*v) or 1
    return (v[0] / n, v[1] / n)


def cut_beside(tr, geom, pad=2.0):
    """The stretch of a track alongside a platform: between the projections of the platform's
    outline onto the track, padded a little."""
    pts = geom.exterior.coords if isinstance(geom, Polygon) else geom.coords
    s = [tr["line"].project(Point(p)) for p in pts]
    a, b = max(0.0, min(s) - pad), min(tr["line"].length, max(s) + pad)
    if b - a < 1:
        return None
    return substring(tr["line"], a, b)


def system_of(tr):
    """Which railway a track belongs to (platforms of two railways are never one box)."""
    o = tr["operator"]
    if "Transit Authority" in o:
        return "subway"
    if "Staten Island" in o + tr["name"]:
        return "sir"
    return o.split()[0] if o else "?"


def platform_system(t):
    """The railway an OSM platform says it belongs to (None: it doesn't say). PATH runs beside the
    6th Av subway at 14 St and 23 St, one level, so a platform's own tracks are told by this."""
    o, n = t.get("operator", ""), t.get("network", "")
    if "AirTrain" in n + t.get("name", "") or t.get("light_rail") == "yes":
        return "Alstom"   # the AirTrain's tracks carry its operator, Alstom
    if "PATH" in n or "Trans-Hudson" in o:
        return "Port"
    if "Staten Island" in o + n:
        return "sir"
    if "Transit Authority" in o or "NYC Subway" in n or t.get("subway") == "yes":
        return "subway"
    return None


def dot_system(dot):
    if dot["line"] == "Staten Island":
        return "sir"
    if dot["line"] in LINE_WORDS:
        return "subway"
    return dot["line"].split()[0]


LEVEL_MISS_M = 1000.0   # score added to a dot of another OSM level than the box's


def level_differs(dot, level):
    """True when the dot is of a known OSM level other than `level` (tracks.json: one dot per level)."""
    return dot.get("level") is not None and level is not None and dot["level"] != level


def dot_matches(dot, tr):
    """Does a station dot's line run on this track?"""
    words = LINE_WORDS.get(dot["line"])
    if words is not None:   # subway and SIR: words of the MTA line in the OSM track name
        if dot["line"] == "Staten Island":   # St George's terminal tracks have no name
            return system_of(tr) == "sir"
        return any(w in tr["name"] for w in words)
    # other rail: the dot's line is the operators of the tracks at it (build.py)
    return bool(tr["operator"]) and tr["operator"].split()[0] in dot["line"]


# ---------- matching ----------

def match_platform(p, tracks, tree):
    """Tracks beside a platform, on its level. Returns (tracks, why not)."""
    g = p["geom"]
    L, W, ax = axis_of(g)
    p.update(length=L, width=W, axis=ax)
    if L < 8:
        return [], "shorter than 8 m"
    cand = []
    psys = platform_system(p["tags"])
    for k in tree.query(g.buffer(BESIDE_M)):
        tr = tracks[int(k)]
        if psys is not None and system_of(tr) != psys:
            continue
        d = g.distance(tr["line"])
        if d > BESIDE_M:
            continue
        piece = tr["line"].intersection(g.buffer(BESIDE_M))
        if piece.is_empty:
            continue
        parts = list(piece.geoms) if hasattr(piece, "geoms") else [piece]
        parts = [q for q in parts if isinstance(q, LineString) and q.length > 0]
        if not parts:
            continue
        run = sum(q.length for q in parts)
        longest = max(parts, key=lambda q: q.length)
        # alongside for enough of the platform, or (a short way, at a terminal or where ways are
        # cut at switches) alongside for nearly all of its own length
        need = min(MIN_OVERLAP * L, 40.0, max(10.0, 0.8 * tr["line"].length))
        if run < need or angle_between(track_dir(longest), ax) > MAX_ANGLE:
            continue
        cand.append((d, tr))
    if not cand:
        if not len(tree.query(g.buffer(40))) or min(g.distance(tracks[int(k)]["line"]) for k in tree.query(g.buffer(40))) > 40:
            return [], "no track on the map within 40 m (a line the map leaves out: NJ Transit, LIRR and Metro-North outside NYC)"
        return [], "a track within 40 m, but none running alongside within %d m" % BESIDE_M
    want = level_tag(p["tags"])
    levels = {tr["level"] for _, tr in cand}
    same = [c for c in cand if c[1]["level"] is not None and any(abs(c[1]["level"] - w) < 0.01 for w in want)]
    if same and (len(want) == 1 or len({c[1]["level"] for c in same}) == 1):
        chosen = same
        p["level_rule"] = "tag"
    else:
        # the level of the nearest track (ties: the one with the most track alongside)
        lv = min(cand, key=lambda c: c[0])[1]["level"]
        chosen = [c for c in cand if c[1]["level"] == lv]
        p["level_rule"] = "nearest" if len(levels) > 1 else "only"
    # A platform has at most one track on each side: the nearest on each side of its axis, plus any
    # other within 1 m of that one (two tracks drawn on top of each other). Until 2026-11-03 the nearest
    # two whatever their side, so a side platform took the express track beyond its local track too, and
    # Central Park West's lower (downtown) boxes spread over the express tracks as if they served them.
    chosen.sort(key=lambda c: c[0])
    cen = g.centroid
    sides = {}
    for d, tr in chosen:
        q = tr["line"].interpolate(tr["line"].project(cen))
        sd_ = 1 if ax[0] * (q.y - cen.y) - ax[1] * (q.x - cen.x) > 0 else -1
        if d < 0.5:
            sd_ = 0   # a track under a platform drawn as a line: its own side
        sides.setdefault(sd_, []).append((d, tr))
    keep = []
    for lst in sides.values():
        keep += [c for c in lst if c[0] <= lst[0][0] + 1.0]
    keep.sort(key=lambda c: c[0])
    return [tr for _, tr in keep], None


def footprint(geoms):
    u = unary_union(geoms).buffer(CLOSE_M, join_style=2).buffer(-CLOSE_M, join_style=2)
    if isinstance(u, MultiPolygon):
        u = max(u.geoms, key=lambda g: g.area)
    if u.is_empty:
        return None
    return Polygon(u.exterior).simplify(0.4)


def _parts(members):
    """A box's pieces: its platforms, and the tracks beside each cut to the platform's length."""
    parts = []
    for p in members:
        g = p["geom"]
        parts.append(g if isinstance(g, Polygon) else g.buffer(LINE_HALF, cap_style=2))
        for tr in p["tracks"]:
            piece = cut_beside(tr, g)
            if piece is not None:
                parts.append(piece.buffer(TRACK_HALF, cap_style=2))
    return parts


def ring_ll(poly):
    c = list(poly.exterior.coords)[:-1]
    if poly.exterior.is_ccw is False:
        c = c[::-1]
    lon, lat = to_ll.transform([q[0] for q in c], [q[1] for q in c])
    return c, [[round(a, 7), round(b, 7)] for a, b in zip(lon, lat)]


def ring_z(coords, trs):
    out = []
    for q in coords:
        pt = Point(q)
        tr = min(trs, key=lambda t: t["line"].distance(pt))
        out.append(round(z_at(tr, pt), 2))
    return out


def ring_ground(coords, trs):
    """The box's ground: its nearest track's own `ground` (tracks.json) at the box's middle, so a box reports
    the depth its track does there, as a station dot does at its point (2026-11-03: boxes used their dot's
    ground; a mean along a 180 m box mixed in the river past Roosevelt Island's ends)."""
    c = Polygon(coords).centroid if len(coords) >= 3 else Point(coords[0])
    for tr in sorted(trs, key=lambda t: t["line"].distance(c)):
        g = float(np.interp(tr["line"].project(c), tr["cum"], tr["ground"]))
        if np.isfinite(g):
            return round(g, 2)
    return None


def main():
    raw = RAW / "osm_platforms.json"
    if "--raw" in sys.argv:   # another Overpass file (tests)
        raw = pathlib.Path(sys.argv[sys.argv.index("--raw") + 1])
    elif "--fetch" in sys.argv or not raw.exists():
        fetch(PLATFORMS, raw)
    tj, tracks = load_tracks()
    tree = STRtree([t["line"] for t in tracks])
    dots = tj["stations"]
    dx, dy = to_utm.transform([d["lon"] for d in dots], [d["lat"] for d in dots])
    dot_pts = [Point(a, b) for a, b in zip(dx, dy)]
    plats, skipped = load_platforms(raw)
    print(f"{len(plats)} OSM platforms read; skipped: {dict(skipped)}")

    why = collections.Counter()
    matched = []
    for p in plats:
        trs, reason = match_platform(p, tracks, tree)
        if not trs:
            why[reason] += 1
            continue
        p["tracks"] = trs
        c = p["geom"].centroid
        p["rail_z"] = float(np.mean([z_at(tr, c) for tr in trs]))
        p["track_level"] = trs[0]["level"]
        p["system"] = {system_of(tr) for tr in trs}
        matched.append(p)
    print(f"{len(matched)} platforms beside tracks")

    # ----- group platforms into boxes -----
    n = len(matched)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    ptree = STRtree([p["geom"] for p in matched])
    for i, p in enumerate(matched):
        ids_i = {tr["ti"] for tr in p["tracks"]}
        for j in ptree.query(p["geom"].buffer(2 * CLOSE_M + 2 * TRACK_HALF)):
            j = int(j)
            if j <= i:
                continue
            q = matched[j]
            if p["track_level"] != q["track_level"] or abs(p["rail_z"] - q["rail_z"]) > LEVEL_Z:
                continue
            if not p["system"] & q["system"]:   # the LIRR beside the subway at Atlantic Av
                continue
            if angle_between(p["axis"], q["axis"]) > MAX_ANGLE and min(p["length"], q["length"]) > 30:
                continue
            shared = ids_i & {tr["ti"] for tr in q["tracks"]}
            if shared or p["geom"].distance(q["geom"]) < 2 * CLOSE_M + 2 * TRACK_HALF:
                parent[find(i)] = find(j)
    groups = collections.defaultdict(list)
    for i in range(n):
        groups[find(i)].append(matched[i])

    boxes = []
    no_dot = []
    n_nodot = 0
    for members in groups.values():
        trs = {tr["ti"]: tr for p in members for tr in p["tracks"]}
        fp = footprint(_parts(members))
        if fp is None or fp.area < 50:
            why["box footprint empty"] += len(members)
            continue
        # station dot: nearest within DOT_M whose line runs on these tracks
        # (distance to the footprint, plus half the distance to its centre: where two stations of
        # a line adjoin, Chambers St and World Trade Center, the box goes to the dot near its middle)
        # tracks.json has one dot per station level (2026-11-02): a dot of another OSM level is taken
        # only when the box's own level has none (LEVEL_MISS_M added to its score)
        best, bd, cen = None, 1e9, fp.centroid
        blv = members[0]["track_level"]
        for k in range(len(dots)):
            d = fp.distance(dot_pts[k])
            if d > DOT_M:
                continue
            score = d + 0.5 * cen.distance(dot_pts[k]) + (LEVEL_MISS_M if level_differs(dots[k], blv) else 0)
            if score < bd and any(dot_matches(dots[k], tr) for tr in trs.values()):
                best, bd = k, score
        if best is None:
            # tracks.json has one dot per MTA station row, so a complex whose levels are different
            # lines (145 St's Concourse level, W 4 St's 6th Av level) has no dot of the lower line:
            # take the nearest dot of the same railway within NEAR_DOT_M
            sys_ = {system_of(tr) for tr in trs.values()}
            for k in range(len(dots)):
                d = fp.distance(dot_pts[k]) + (LEVEL_MISS_M if level_differs(dots[k], blv) else 0)
                if fp.distance(dot_pts[k]) < NEAR_DOT_M and d < bd and dot_system(dots[k]) in sys_:
                    best, bd = k, d
        if best is None:
            no_dot.append(members)
            continue
        coords, ring = ring_ll(fp)
        z = ring_z(coords, list(trs.values()))
        rr = fp.minimum_rotated_rectangle.exterior.coords
        Lb = max(math.hypot(rr[1][0] - rr[0][0], rr[1][1] - rr[0][1]), math.hypot(rr[2][0] - rr[1][0], rr[2][1] - rr[1][1]))
        b = {"station": dots[best]["name"], "line": dots[best]["line"], "dot": best,
             "lines": sorted({tr["name"] for tr in trs.values() if tr["name"]}),
             "level": members[0]["track_level"], "length": round(Lb, 1), "rail_z": round(float(np.mean(z)), 2),
             "ground": ring_ground(coords, list(trs.values())),
             "fallback": False, "ring": ring, "z": z, "_members": members, "_fp": fp, "_trs": list(trs.values())}
        boxes.append(b)
    # platforms by no dot: stations the page has no dot for (the New Lots line, most of PATH:
    # build.py's dots miss them). Kept, with no dot, when an OSM station is within OSM_STATION_M;
    # named after it
    st_els = json.loads((RAW / "osm_stations.json").read_text(encoding="utf-8"))["elements"]
    osm_st = []
    for e in st_els:
        t = e.get("tags", {})
        if t.get("railway") in ("station", "halt") and t.get("name"):
            lon_, lat_ = (e["lon"], e["lat"]) if "lon" in e else (e["center"]["lon"], e["center"]["lat"])
            osm_st.append((t["name"], Point(*to_utm.transform(lon_, lat_))))
    for members in no_dot:
        trs = {tr["ti"]: tr for p in members for tr in p["tracks"]}
        fp = footprint(_parts(members))
        near = min(osm_st, key=lambda s: fp.distance(s[1])) if fp is not None else None
        if fp is None or near is None or fp.distance(near[1]) > OSM_STATION_M:
            why["no station dot or OSM station within reach (disused or a yard platform)"] += len(members)
            if "--list-no-dot" in sys.argv:
                print("  dropped:", sorted({(m["tags"].get("name", ""), m["tags"].get("operator", ""), m["id"]) for m in members})[:3])
            continue
        coords, ring = ring_ll(fp)
        z = ring_z(coords, list(trs.values()))
        rr = fp.minimum_rotated_rectangle.exterior.coords
        Lb = max(math.hypot(rr[1][0] - rr[0][0], rr[1][1] - rr[0][1]), math.hypot(rr[2][0] - rr[1][0], rr[2][1] - rr[1][1]))
        boxes.append({"station": near[0], "line": "", "dot": None,
                      "lines": sorted({tr["name"] for tr in trs.values() if tr["name"]}),
                      "level": members[0]["track_level"], "length": round(Lb, 1), "rail_z": round(float(np.mean(z)), 2),
                      "ground": ring_ground(coords, list(trs.values())),
                      "fallback": False, "ring": ring, "z": z, "_members": members, "_fp": fp, "_trs": list(trs.values())})
        n_nodot += 1

    # ----- fallbacks for dots with no box -----
    have = {b["dot"] for b in boxes}
    fallback_fail = collections.Counter()
    covered = []
    for k, dot in enumerate(dots):
        if k in have:
            continue
        pt = dot_pts[k]
        # a dot already inside (or by) a box of its own line, given to a neighbouring dot: no second box
        cover = next((b for b in boxes if not b["fallback"] and b["_fp"].distance(pt) < 40
                      and not level_differs(dot, b["level"])
                      and any(dot_matches(dot, tr) for tr in b["_trs"])), None)
        if cover is not None:
            covered.append((dot, cover))
            continue
        near = [tracks[int(i)] for i in tree.query(pt.buffer(90))]
        near = [tr for tr in near if tr["line"].distance(pt) < 90 and dot_matches(dot, tr)]
        if not near:
            fallback_fail["no track of its line within 90 m"] += 1
            continue
        # the level whose rail is closest to the dot's height (the dot is at the level with most track)
        bylv = collections.defaultdict(list)
        for tr in near:
            bylv[tr["level"]].append(tr)
        lv = min(bylv, key=lambda l: abs(np.mean([z_at(tr, pt) for tr in bylv[l]]) - dot["z"]))
        trs = bylv[lv]
        if dot["line"] in LINE_WORDS:
            L = 160.0 if dot["line"] in A_DIVISION else OTHER_LENGTH.get(dot["line"], 180.0)
        else:
            L = next((v for o, v in OTHER_LENGTH.items() if o in dot["line"]), 200.0)
        # centre: the dot projected onto its nearest track; every track of the level cut to L there
        t0 = min(trs, key=lambda tr: tr["line"].distance(pt))
        centre = t0["line"].interpolate(t0["line"].project(pt))
        parts = []
        for tr in trs:
            if tr["line"].distance(centre) > 30:
                continue
            s = tr["line"].project(centre)
            piece = substring(tr["line"], max(0, s - L / 2), min(tr["line"].length, s + L / 2))
            if piece.length > 1:
                parts.append(piece.buffer(TRACK_HALF + 1.0, cap_style=2))
        fp = footprint(parts) if parts else None
        if fp is None:
            fallback_fail["no footprint"] += 1
            continue
        coords, ring = ring_ll(fp)
        z = ring_z(coords, trs)
        boxes.append({"station": dot["name"], "line": dot["line"], "dot": k,
                      "lines": sorted({tr["name"] for tr in trs if tr["name"]}), "level": lv, "length": L,
                      "rail_z": round(float(np.mean(z)), 2), "ground": ring_ground(coords, trs),
                      "fallback": True, "ring": ring, "z": z, "_trs": trs})

    # ----- write -----
    out_plats = []
    for bi, b in enumerate(boxes):
        for p in b.pop("_members", []):
            g = p["geom"]
            slab = g if isinstance(g, Polygon) else g.buffer(LINE_HALF, cap_style=2)
            slab = Polygon(slab.exterior).simplify(0.3)
            sc, sring = ring_ll(slab)
            sz = [round(v + PLATFORM_ABOVE_RAIL, 2) for v in ring_z(sc, p["tracks"])]
            gc = list(g.exterior.coords) if isinstance(g, Polygon) else list(g.coords)
            glon, glat = to_ll.transform([q[0] for q in gc], [q[1] for q in gc])
            out_plats.append({"id": p["id"], "box": bi, "station": b["station"],
                              "lines": sorted({tr["name"] for tr in p["tracks"] if tr["name"]}),
                              "level": p["tags"].get("level", p["tags"].get("layer")), "track_level": p["track_level"],
                              "level_rule": p["level_rule"], "kind": p["kind"],
                              "length": round(p["length"], 1), "width": round(p["width"], 1),
                              "rail_z": round(p["rail_z"], 2), "platform_z": round(p["rail_z"] + PLATFORM_ABOVE_RAIL, 2),
                              "geom": [[round(a, 7), round(c, 7)] for a, c in zip(glon, glat)],
                              "slab": sring, "slab_z": sz})
        b.pop("_fp", None)
        b["tracks"] = sorted({tr["id"] for tr in b.pop("_trs", [])})   # the page tints the box their colour
    OUT.mkdir(exist_ok=True)
    (OUT / "platforms.json").write_text(json.dumps({"boxes": boxes, "platforms": out_plats}, separators=(",", ":")),
                                        encoding="utf-8")

    # ----- counts -----
    nf = sum(b["fallback"] for b in boxes)
    print(f"{len(out_plats)} platforms matched to tracks and a station, in {len(boxes) - nf} boxes "
          f"({n_nodot} of them at stations the page has no dot for, named from OSM: dot null)")
    print(f"{nf} fallback boxes (station dots with no OSM platform matched)")
    have = {b["dot"] for b in boxes} - {None}
    print(f"{len(have)} of {len(dots)} station dots have a box of their own")
    for dot, b in covered:
        print(f"  {dot['name']} ({dot['line']}): inside the box given to {b['station']} ({b['line']})")
    rules = collections.Counter(p["level_rule"] for p in out_plats)
    print(f"level from: {dict(rules)}")
    print("platforms left unmatched:")
    for k, v in why.most_common():
        print(f"  {v}: {k}")
    if fallback_fail:
        print("station dots with neither a platform nor a fallback:")
        for k, v in fallback_fail.most_common():
            print(f"  {v}: {k}")
        for k, dot in enumerate(dots):
            if k not in have:
                print(f"    {dot['name']} ({dot['line']})")
    print(f"wrote {OUT / 'platforms.json'} ({(OUT / 'platforms.json').stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
