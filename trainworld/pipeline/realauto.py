"""T-3225: real networks with no hand work, from noritetsu's local OSM extract (notes/T-3225.md).

    python pipeline/realauto.py extract [id ...]          # the extract cut to each city -> $REALNET_OUT/osm_<id>.json
    python pipeline/realauto.py build [id ...] [--jobs N] # extract, net, geom, corridors, realsave, gamesave; a report per city
    python pipeline/realauto.py report [id ...]           # the reports' table again
    python pipeline/realauto.py ship [id ...] [--force]   # realnet.py ship for the cities that passed and lose no route or station the shipped save serves, and the app's list
    python pipeline/realauto.py drop <id>                 # (a build step) the lines in auto_<id>_drop.json out of the network JSON
    python pipeline/realauto.py keep <id> <save> [label]  # the city's keep list from a save's lines (label: where the save is from)

The cities are `realauto.json` (a city pack id each, how they were picked in its "what"); with no ids a
command takes them all. Any registry city with a pack can be built: `build <id>` takes ids not in the
list too, and `ship` ships only listed ones with "ship": true, so a city goes on the new game sheet by
an entry there. realnet.py makes each listed city a `CITIES` entry with one `osm` source, so its own
commands (net, geom, corridors, gamesave, ship) run on these as on the hand-made seven.

The extract (`noritetsu/data/proc/<cc>/`, read only): every route relation (train, subway, light
rail, tram, monorail) with two or more stops in the pack's boundary (+1 km), less:
  - `service` high_speed, long_distance, night or tourism, `highspeed=yes`, a `state` of proposed,
    construction or disused;
  - routes whose ways are mostly high-speed track, or mostly preserved or tourist railway;
  - routes mostly outside the pack: under half their stops inside, unless 8 or more are;
  - routes whose stops inside lie over 15 km apart on average (intercity trains).
A relation's stops are put right by its own ways (`tidy_stops`: sorted along the track when listed out of order,
off-track and repeated stops out, a shuttle's out-and-back relation read as its first half, and for train and
subway routes a station the track passes that the relation leaves out added between its first and last stop,
see notes/T-3782.md); the extract counts `reordered` and `stops_added`.
A station is the stops of one name within 400 m (realnet's `osm` adapter: the extract keeps no
stop areas). Names in Latin letters: OSM's `name` where it is, else `name:en`, `int_name` or
`name:latin`, else Cyrillic transliterated and (China, Hong Kong, Taiwan) Han characters in pinyin
run together; else `name` (Japanese kana and kanji). A line is its route master's `ref` (if Latin
letters or digits), else that name. Each relation's ways, chained, are its shape: where the router
finds no real-network track (trams, light rail), the track follows them. Trains an hour from
`interval` tags, else realnet.OSM_TPH by kind, shared by a route master's patterns each way
(`split_tph`); a suburban train with no route master and no ref every SUBURBAN_INTERVAL minutes.
Run times from the distance at realnet.OSM_KMH; train lengths realnet.OSM_TRAIN_M. A light-rail
route on main-line track (S-Bahn Berlin and Hamburg, OSM's `railway=light_rail` + `usage=main`;
HEAVY_*) runs commuter trains at commuter speed, through the extract's own `lines`. The extract
lists what it dropped, by reason (`dropped_routes`). Levels, shared
track, terminals and pacing as for the hand-made ones (realcorridors.py, reallevels.py, the
`realsave` bin).

`build` runs each city's steps as subprocesses, `--jobs` cities at once (default 2), each city's log
in $REALNET_OUT/auto_<id>.log. A city with a keep list (`realauto_keep/<id>.json`, written by `keep`
from a save: each line's name and stop names) is built from the lines it picks only (`kept`: each
listed line picks the line of its name sharing the most stop names), so it keeps the size it had
(T-3790: Moscow keeps v0.4.2's lines, not the extract's ~358). Lines the track model still cannot run go into auto_<id>_drop.json, `drop` takes them out of the network
JSON after `net` (with the lines the keep list does not pick), and the city is built again, until every line runs (at most BUILD_PASSES builds). It writes $REALNET_OUT/auto_<id>.json: stations,
lines, lines running, names in Latin letters, track km off the real-network graph, high-demand holds
against the free run, the network's price, the save's size, and `pass`: the save was written, 2 or
more lines, at least 90% of them running.
"""
import json
import math
import os
import pickle
import re
import subprocess
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MAPS = os.path.dirname(ROOT)


def shared_path(*parts):
    """A path under the maps root; in a worktree (<maps>/.tw-trees/<name>), which has no untracked data, the main
    tree's when the tree has none (noritetsu's extract and model, N02: all read only)."""
    p = os.path.join(MAPS, *parts)
    if not os.path.exists(p) and os.path.basename(os.path.dirname(MAPS)) == ".tw-trees":
        return os.path.join(os.path.dirname(os.path.dirname(MAPS)), *parts)
    return p


NORI_PROC = shared_path("noritetsu", "data", "proc")
LIST = os.path.join(HERE, "realauto.json")
KEEP_DIR = os.path.join(HERE, "realauto_keep")
OUT = os.path.abspath(os.environ.get("REALNET_OUT", os.path.join(ROOT, "data", "work", "realnet")))
APP_LIST = os.path.join(ROOT, "app", "src", "game", "realNetworks.json")

KINDS = ("train", "subway", "light_rail", "tram", "monorail")
SKIP_SERVICE = {"high_speed", "long_distance", "night", "tourism"}
SKIP_STATE = {"proposed", "construction", "disused", "abandoned"}
INSIDE_SHARE, INSIDE_ENOUGH = 0.5, 8
MAX_SPACING_M = 15000.0
MOSTLY = 0.5
PASS_RUNNING = 0.9
SUBURBAN_INTERVAL = "30"
# T-3790: leaving lines out changes the track model, so other lines can stop running (Saint Petersburg: 19 left out,
# then 9 more broken in the save); the city is built again until every line runs, at most this many times in all
BUILD_PASSES = 4
# T-3265: a light-rail route on main-line track (Berlin's and Hamburg's S-Bahn, Copenhagen's S-tog:
# `railway=light_rail` with `usage` main or branch, or plain rail) with stops this far apart on
# average and this many inside runs commuter trains at commuter speed; it keeps light rail's own
# track and frequency. The stop count keeps out airport people movers that OSM tags usage=main,
# and track tagged unelectrified (Chongqing's Bishan SkyShuttle) does not count: S-Bahns are electric.
HEAVY_USAGE = {"main", "branch"}
HEAVY_SPACING_M = 1000.0
HEAVY_STOPS = 5
# regions whose names in Han characters read as Mandarin (not Japan: kanji read otherwise)
HANZI_REGIONS = {"cn", "hk", "tw"}


def listed():
    return json.load(open(LIST, encoding="utf-8"))["cities"]


def registry():
    return {c["id"]: c for c in json.load(open(os.path.join(ROOT, "app", "src", "game", "registry.json"), encoding="utf-8"))["cities"]}


def region_of(pack, reg=None):
    """noritetsu's region code for a pack: its country, lower case (HK and TW are regions of their own)."""
    c = (reg or registry())[pack]
    cc = c["country"].lower()
    if not os.path.isdir(os.path.join(NORI_PROC, cc)):
        raise SystemExit(f"{pack}: no noritetsu extract for {cc}")
    return cc


def latin(s):
    """Every letter of `s` is a Latin one (digits, spaces and marks pass)."""
    letters = [ch for ch in s if ch.isalpha()]
    return bool(letters) and all("LATIN" in unicodedata.name(ch, "") for ch in letters)


CYR = dict(zip("абвгдеёжзийклмнопрстуфхцчшщъыьэюяіїєґ",
               ["a", "b", "v", "g", "d", "e", "yo", "zh", "z", "i", "y", "k", "l", "m", "n", "o", "p", "r", "s", "t", "u", "f", "kh", "ts",
                "ch", "sh", "shch", "", "y", "", "e", "yu", "ya", "i", "yi", "ye", "g"]))


def translit(s):
    """Cyrillic in Latin letters (BGN/PCGN, simplified), other letters as they are."""
    out = []
    for ch in s:
        lo = ch.lower()
        if lo in CYR:
            v = CYR[lo]
            out.append(v[:1].upper() + v[1:] if ch != lo else v)
        else:
            out.append(ch)
    return "".join(out)


def pick_name(t, cc=""):
    n = t.get("name", "")
    if not n or latin(n):
        return n
    for k in ("name:en", "int_name", "name:latin"):
        v = t.get(k, "")
        if v and latin(v):
            return v
    v = translit(n)
    if latin(v):
        return v
    if cc in HANZI_REGIONS:
        # pinyin without tones, run together as OSM's own name:en mostly has it ("Xianxinlu")
        from unidecode import unidecode
        v = "".join(unidecode(n).split()).capitalize()
        if latin(v):
            return v
    return n


def line_name(t, cc):
    """A route's or route master's name for a line with no usable ref: as `pick_name`, but Chinese in
    spaced pinyin (a line's long name run together is unreadable), and OSM's "Label: A => B"
    form shortened to the label when it holds a number ("Tramvay t2"), else to "A - B"."""
    n = pick_name(t, cc)
    own = t.get("name", "")
    if cc in HANZI_REGIONS and own and not latin(own) and not any(latin(t.get(k, "")) for k in ("name:en", "int_name", "name:latin")):
        from unidecode import unidecode
        n = " ".join(unidecode(own).split()).title()
    if ": " in n:
        label, rest = n.split(": ", 1)
        n = label if any(ch.isdigit() for ch in label) else rest
    return re.sub(r"\s*<?=>\s*", " - ", n)


def pick_ref(t):
    r = t.get("ref", "")
    return r if r and (latin(r) or r.replace(" ", "").isdigit()) else ""


def load_region(cc):
    p = os.path.join(NORI_PROC, cc)
    t0 = time.time()
    rels = pickle.load(open(os.path.join(p, "rels.pkl"), "rb"))
    stops = pickle.load(open(os.path.join(p, "stops.pkl"), "rb"))
    ways = pickle.load(open(os.path.join(p, "ways.pkl"), "rb"))
    print(f"{cc}: {len(rels):,} relations, {len(stops):,} stops, {len(ways):,} ways read in {time.time() - t0:.0f} s", flush=True)
    return rels, stops, ways


def coords(cc):
    """noritetsu's node coordinates (for stops missing from stops.pkl): sorted ids, lon, lat."""
    import numpy as np
    z = np.load(os.path.join(NORI_PROC, cc, "coords.npz"))
    ids, x, y = z["id"], z["x"], z["y"]
    o = np.argsort(ids)
    return ids[o], x[o] / 1e7, y[o] / 1e7


def cut(pack, rels, stops, ways, xy_of):
    """The pack's route relations as Overpass JSON (what realnet's `osm` adapter reads), and why the
    rest were dropped."""
    sys.path.insert(0, HERE)
    import realnet
    from shapely import prepared
    from shapely.geometry import Point
    cc = region_of(pack)
    bnd = prepared.prep(realnet.boundary(pack))
    w, s, e, n = realnet.boundary(pack).bounds
    k = math.cos(math.radians((s + n) / 2))
    masters = defaultdict(list)
    for rid, (t, ms) in rels.items():
        if t.get("type") == "route_master":
            for ty, r, _ in ms:
                if ty == "r":
                    masters[r].append(rid)
    why = Counter()
    gone = defaultdict(list)
    heavy = []

    def drop(reason, t):
        why[reason] += 1
        gone[reason].append(" ".join(x for x in (t.get("route", ""), t.get("ref", ""), pick_name(t, cc)) if x))
    keep, nodes = [], {}
    for rid, (t, ms) in rels.items():
        if t.get("type") != "route" or t.get("route") not in KINDS:
            continue
        pts = []
        for ty, r, role in ms:
            if ty != "n" or not role.startswith(("stop", "platform")):
                continue
            ll = (stops[r][1], stops[r][2]) if r in stops else xy_of(r)
            if ll is not None:
                pts.append((r, role, ll))
        # stops, else platforms (realnet's adapter does the same)
        st = [q for q in pts if q[1].startswith("stop")] or pts
        inside = [q for q in st if w <= q[2][0] <= e and s <= q[2][1] <= n and bnd.contains(Point(q[2]))]
        if len(inside) < 2:
            continue
        if t.get("service") in SKIP_SERVICE or t.get("highspeed") == "yes":
            drop("service tag", t)
            continue
        if t.get("state") in SKIP_STATE or any(t.get(f"{x}:route") for x in ("disused", "proposed")) or \
                re.search(r"out of service|disused|abandoned|closed", t.get("name:en", "") + " " + t.get("name", ""), re.I):
            drop("not running", t)
            continue
        wt = [ways[r][0] for ty, r, role in ms if ty == "w" and r in ways and role in ("", "forward", "backward")]
        if wt and sum(x.get("highspeed") == "yes" for x in wt) > MOSTLY * len(wt):
            drop("high-speed track", t)
            continue
        if wt and sum(x.get("railway") == "preserved" or x.get("usage") == "tourism" for x in wt) > MOSTLY * len(wt):
            drop("heritage track", t)
            continue
        if len(inside) < INSIDE_SHARE * len(st) and len(inside) < INSIDE_ENOUGH:
            drop("mostly outside", t)
            continue
        gap = [math.hypot((a[2][0] - b[2][0]) * k, a[2][1] - b[2][1]) * 111195 for a, b in zip(inside, inside[1:])]
        if sum(gap) / len(gap) > MAX_SPACING_M:
            drop("stops far apart", t)
            continue
        if t.get("route") == "light_rail" and wt and len(inside) >= HEAVY_STOPS and sum(gap) / len(gap) >= HEAVY_SPACING_M and \
                sum(x.get("railway") == "rail" or (x.get("usage") in HEAVY_USAGE and x.get("electrified") != "no") for x in wt) > MOSTLY * len(wt):
            heavy.append(rid)
        keep.append(rid)
        for r, _, ll in st:
            nodes[r] = ll
    kept = set(keep)
    shapes = defaultdict(list)
    reordered, added = {}, Counter()

    def in_pack(ll):
        return w <= ll[0] <= e and s <= ll[1] <= n and bnd.contains(Point(ll))
    import numpy as np
    stop_ids = np.array(sorted(stops), dtype=np.int64)

    def stop_tags(r):
        return stops[r][0] if r in stops else {}
    pending = {}
    for rid in keep:
        t, ms = rels[rid]
        wl = [ways[r][1] for ty, r, role in ms if ty == "w" and r in ways and role in ("", "forward", "backward")]
        q = chain(wl, xy_of, k)
        if len(q) >= 2:
            label = " ".join(str(x) for x in [t.get("name"), t.get("name:en"), t.get("ref"), t.get("service")] + [rels[m][0].get("name") for m in masters.get(rid, [])])
            cands = [] if SKIPS_STOPS.search(label) or t.get("route") not in ADD_KINDS else track_stops(t, masters.get(rid), rels, wl, stops, stop_ids, nodes)
            listed_stops = [(r, nodes[r]) for ty, r, role in ms if ty == "n" and r in nodes and role.startswith("stop")]
            fixed, n_add = tidy_stops(listed_stops, q, k, cands, stop_tags=stop_tags, inside=in_pack, train=t.get("route") == "train")
            pending[rid] = (q, cands, listed_stops, fixed, n_add)
            key = f"m{max(masters[rid])}" if masters.get(rid) else f"r{rid}"
            shapes[key].append([[round(a, 6), round(b, 6)] for a, b in q])
    # the other direction of a route master was given a stop the relation lacks: this one gets it too
    # (JR Tokaido Main Line's two relations each miss different stations), with no count limit
    gave = defaultdict(set)
    for rid, (q, cands, listed_stops, fixed, n_add) in pending.items():
        if fixed and n_add:
            have = {x for x, _ in listed_stops}
            for m in masters.get(rid, []):
                gave[m] |= {stop_tags(x).get("name") for x in fixed if x not in have}
    for rid, (q, cands, listed_stops, fixed, n_add) in pending.items():
        t = rels[rid][0]
        names = set().union(*[gave[m] for m in masters.get(rid, [])]) if masters.get(rid) else set()
        base = [(r, nodes.get(r) or (stops[r][1], stops[r][2])) for r in fixed] if fixed else listed_stops
        have = {stop_tags(x).get("name") for x, _ in base}
        more = [c for c in cands if stop_tags(c[0]).get("name") in names - have]
        if more:
            f2, n2 = tidy_stops(base, q, k, more, stop_tags=stop_tags, inside=in_pack, train=t.get("route") == "train", limit=False)
            if f2:
                fixed, n_add = f2, n_add + n2
        if fixed:
            reordered[rid] = fixed
            added[rid] = n_add
            for r in fixed:
                if r not in nodes:
                    nodes[r] = (stops[r][1], stops[r][2])
    el = []
    for rid in keep:
        t, ms = rels[rid]
        tags = {**t, "name": line_name(t, cc), "ref": pick_ref(t)}
        # a suburban train with no line of its own (no route master, no ref) is one of many
        # destinations out of a terminal (Russia's elektrichki): every half hour, not the
        # commuter default, which would fill the terminal's tracks many times over
        if t.get("route") == "train" and not masters.get(rid) and not t.get("ref") and not t.get("interval"):
            tags["interval"] = SUBURBAN_INTERVAL
        mem = [{"type": "node", "ref": r, "role": role} for ty, r, role in ms if ty == "n" and r in nodes and role.startswith(("stop", "platform"))]
        if rid in reordered:
            mem = [{"type": "node", "ref": r, "role": "stop"} for r in reordered[rid]]
        el.append({"type": "relation", "id": rid, "tags": tags, "members": mem})
    for mid in sorted({m for rid in keep for m in masters.get(rid, [])}):
        t, ms = rels[mid]
        el.append({"type": "relation", "id": mid, "tags": {**t, "name": line_name(t, cc), "ref": pick_ref(t)},
                   "members": [{"type": "relation", "ref": r, "role": ""} for ty, r, _ in ms if ty == "r" and r in kept]})
    for r, (lon, lat) in nodes.items():
        tg = stops[r][0] if r in stops else {}
        el.append({"type": "node", "id": r, "lon": lon, "lat": lat, "tags": {"name": pick_name(tg, cc)}})
    # T-3935: lines OSM maps only as track, added where the city's entry says `tracks` (realtracks.py)
    import realtracks
    found, refused = realtracks.lines(cc, rels, stops, ways, xy_of, (w, s, e, n), in_pack, set(nodes))
    if any(c["id"] == pack and c.get("tracks") for c in listed()):
        t_el, t_shapes, t_nodes = realtracks.elements(found, cc)
        el += t_el
        shapes.update(t_shapes)
        for r, ((lon, lat), nm) in t_nodes.items():
            if r not in nodes:
                el.append({"type": "node", "id": r, "lon": lon, "lat": lat, "tags": {"name": nm}})
    track_lines = [{x: L[x] for x in ("line", "name", "route", "from", "to", "stops", "km", "covered", "source", "id")} | {"names": L["names"]} for L in found]
    kinds = Counter(rels[r][0].get("route") for r in keep)
    # realnet's osm adapter takes these as its `lines` (a regex on "name | name:en | ref | r<id>")
    lines = [[r"\| r(" + "|".join(str(r) for r in heavy) + r")$",
              {"train_m": realnet.OSM_TRAIN_M["commuter"], "kmh": realnet.OSM_KMH["commuter"]}]] if heavy else []
    return {"elements": el, "shapes": shapes, "lines": lines,
            "realauto": {"kept": len(keep), "kinds": dict(kinds), "heavy_light_rail": len(heavy), "reordered": len(reordered), "stops_added": sum(added.values()), "dropped": dict(why),
                         "dropped_routes": {k: sorted(v) for k, v in gone.items()},
                         "track_lines": track_lines, "track_added": len(track_lines) if any(c["id"] == pack and c.get("tracks") for c in listed()) else 0,
                         "track_refused": Counter(x["why"] for x in refused), "track_refused_lines": refused,
                         "track_partly_covered": [x for x in refused if x["why"] == "routes run on it" and x["covered"] < 0.9]}}


ORDER_RATIO = 1.1
ORDER_NEAR_M = 300.0
ADD_FRAC = 0.15
ADD_KINDS = ("train", "subway")
SKIPS_STOPS = re.compile(r"express|rapid|limited|semi|\bexp\b|liner|rapide|expres|\bRE\d|\bICE?\b|快|急|特|準|экспресс|ekspres", re.I)
ADD_TRAIN_SPACING_M = 5000.0
UNOPENED = re.compile(r"planned|opening soon|under construction|proposed|zaijian|future|closed", re.I)
_ONTRACK_KINDS = ("train", "subway", "light_rail", "tram", "monorail")


def _point_at(q, k, d):
    """The (lon, lat) `d` metres along chain `q`."""
    run = 0.0
    for a, b in zip(q, q[1:]):
        seg = math.hypot((b[0] - a[0]) * k, b[1] - a[1]) * 111195.0
        if run + seg >= d and seg > 0:
            f = (d - run) / seg
            return (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f)
        run += seg
    return q[-1]


def retrace_half(q, k):
    """A relation that lists its ways out and back along the same track (a shuttle's one relation for
    both directions: Nishitetsu Kaizuka Line, JR Kagoshima Main Line's Hakata - Omuta relations) is its
    first half; None for any other chain. The chain's points a tenth to four tenths along it lie within
    200 m of the ones the same distance from its end."""
    if len(q) < 4:
        return None
    total = sum(math.hypot((b[0] - a[0]) * k, b[1] - a[1]) for a, b in zip(q, q[1:])) * 111195.0
    if total < 2000.0:
        return None
    for f in (0.1, 0.2, 0.3, 0.4):
        a, b = _point_at(q, k, f * total), _point_at(q, k, (1 - f) * total)
        if math.hypot((a[0] - b[0]) * k, a[1] - b[1]) * 111195.0 > 200.0:
            return None
    half, run = [q[0]], 0.0
    for a, b in zip(q, q[1:]):
        seg = math.hypot((b[0] - a[0]) * k, b[1] - a[1]) * 111195.0
        if run + seg >= total / 2:
            half.append(_point_at(q, k, total / 2))
            break
        half.append(b)
        run += seg
    return half


def _arcs(stops, q, k):
    """For each (node id, (lon, lat)) its distance to the chain `q` and its arc position along it, metres."""
    P = [(lon * k * 111195.0, lat * 111195.0) for lon, lat in q]
    A = P[:-1]
    sx = [(P[i + 1][0] - P[i][0], P[i + 1][1] - P[i][1]) for i in range(len(P) - 1)]
    L = [math.hypot(*d) for d in sx]
    cum = [0.0]
    for v in L:
        cum.append(cum[-1] + v)
    out = []
    for _, (lon, lat) in stops:
        cx, cy = lon * k * 111195.0, lat * 111195.0
        best = (1e18, 0.0)
        for i, (a, d) in enumerate(zip(A, sx)):
            ll = L[i] * L[i]
            t = 0.0 if ll < 1e-9 else max(0.0, min(1.0, ((cx - a[0]) * d[0] + (cy - a[1]) * d[1]) / ll))
            dist = math.hypot(cx - a[0] - d[0] * t, cy - a[1] - d[1] * t)
            if dist < best[0]:
                best = (dist, cum[i] + t * L[i])
        out.append(best)
    return out, cum[-1]


def _span(stops, idx, k):
    return sum(math.hypot((stops[a][1][0] - stops[b][1][0]) * k, stops[a][1][1] - stops[b][1][1]) for a, b in zip(idx, idx[1:])) * 111195.0


def tidy_stops(stops, q, k, cands=(), stop_tags=None, inside=lambda ll: True, train=False, limit=True):
    """T-3777, T-3782: a relation's stop members put right by its own chained ways `q`. `stops` is
    [(node id, (lon, lat))] in member order. Nothing changes unless the ways carry the stops (90% of
    them within ORDER_NEAR_M of the chain: not a branching or broken chain). Then
      - stops off the chain by more than ORDER_NEAR_M go (a depot's or yard's stop, a platform
        mapped elsewhere), which made a train run out to it and back;
      - stops listed out of line order (JR Tokaido Main Line's two relations list Tokyo, Mishima,
        Bentenjima, Araimachi, Futagawa...) are sorted along the chain, when the member order is
        ORDER_RATIO times longer than the sorted one;
      - `cands`, [(node id, (lon, lat))] passenger stops lying on the relation's ways that the relation
        does not list (JR Tokaido Main Line skips Nagoya), are added in track order when the relation
        misses at most 3 stops or 15% as many as it lists, and the stop is not within 400 m of a
        same-named listed one (150 m of any other). A train route also needs its listed stops under
        ADD_TRAIN_SPACING_M apart on average, and the relation not to name itself an express or
        rapid (SKIPS_STOPS): those skip stations on purpose. Only train and subway routes, and only
        between the relation's own first and last stop (a tram's or light rail's shared track makes
        added stops break junction turns).
    `limit` False (a stop the route master's other direction was given) skips the count and spacing limits.
    Stops count only where `inside` (the pack): the line is cut to it anyway. Returns (node ids in order
    or None when nothing changed, number added)."""
    if len(stops) < 4:
        return None, 0
    back = retrace_half(q, k)
    if back:
        q = back
    elif math.hypot((q[0][0] - q[-1][0]) * k, q[0][1] - q[-1][1]) * 111195.0 < 300.0:
        return None, 0  # a loop (a circle line, a circular light rail): its order is the ring
    ar, total = _arcs(stops, q, k)
    if sum(d <= ORDER_NEAR_M for d, _ in ar) < 0.9 * len(stops):
        return None, 0
    # a terminus a little past the track's end (its building off the last way) stays
    idx = [i for i in range(len(stops)) if ar[i][0] <= ORDER_NEAR_M or (ar[i][0] <= 1500.0 and (ar[i][1] < 1.0 or ar[i][1] > total - 1.0))]
    changed = len(idx) != len(stops) or bool(back)
    # a station listed again later (the return end of a shuttle, a depot stop, a platform listed twice) is
    # listed once, unless the line is a loop that closes on its first station
    loop = not back and len(idx) > 3 and stop_tags(stops[idx[0]][0]).get("name") and         stop_tags(stops[idx[0]][0]).get("name") == stop_tags(stops[idx[-1]][0]).get("name")
    seen, once = [], []
    for i in idx:
        nm = stop_tags(stops[i][0]).get("name")
        if nm and any(n == nm and math.hypot((stops[i][1][0] - p[0]) * k, stops[i][1][1] - p[1]) * 111195.0 < 400.0 for n, p in seen) and not (loop and i == idx[-1]):
            continue
        seen.append((nm, stops[i][1]))
        once.append(i)
    changed = changed or len(once) != len(idx)
    idx = once
    if len(idx) >= 4 and _span(stops, idx, k) >= ORDER_RATIO * _span(stops, sorted(idx, key=lambda i: ar[i][1]), k):
        idx = sorted(idx, key=lambda i: ar[i][1])
        changed = True
    ids = [stops[i][0] for i in idx]
    pos = {stops[i][0]: stops[i][1] for i in idx}
    added = 0
    if cands:
        arc = {stops[i][0]: ar[i][1] for i in idx}
        a0, a1 = arc[ids[0]], arc[ids[-1]]
        mono = all((arc[b] - arc[a]) * (1 if a1 >= a0 else -1) >= -50.0 for a, b in zip(ids, ids[1:]))
        names = [(stop_tags(r).get("name"), pos[r]) for r in ids]
        new = []
        for r, ll in cands:
            nm = stop_tags(r).get("name")
            if not nm or r in pos or not inside(ll):
                continue
            if any(math.hypot((ll[0] - p[0]) * k, ll[1] - p[1]) * 111195.0 < (400.0 if n == nm else 150.0) for n, p in names + [(stop_tags(x).get("name"), l) for x, l in new]):
                continue
            new.append((r, ll))
        served = len({stop_tags(r).get("name") for r in ids if inside(pos[r])})
        inner = [pos[r] for r in ids if inside(pos[r])]
        gap = _span([(0, p) for p in inner], range(len(inner)), k) / max(1, len(inner) - 1)
        if mono and new and (not limit or ((not train or gap <= ADD_TRAIN_SPACING_M) and len(new) <= max(3, ADD_FRAC * served))):
            ar_new, _ = _arcs(new, q, k)
            lo, hi = min(arc.values()), max(arc.values())
            ok = [(a, r, ll) for (r, ll), (d, a) in zip(new, ar_new) if d <= ORDER_NEAR_M and lo < a < hi]
            fwd = a1 >= a0
            merged = sorted([(arc[r], r) for r in ids] + [(a, r) for a, r, _ in ok], reverse=not fwd)
            if ok:
                ids = [r for _, r in merged]
                added = len(ok)
                changed = True
    return (ids if changed else None), added


def track_stops(t, master_ids, rels, wl, stops, stop_ids, listed_nodes):
    """Passenger stops (railway stop or halt) on the ways of a relation `t`, that its own network or
    operator allows: a stop tagged with a network (or operator) must carry the relation's own
    (`network`/`operator`, its name or its route master's); a stop for another kind of train
    (`subway=yes` on a train route) is not this route's. [(node id, (lon, lat))]."""
    import numpy as np
    if not wl:
        return []
    on = np.intersect1d(stop_ids, np.unique(np.concatenate([np.asarray(w, dtype=np.int64) for w in wl])))
    mine = {t.get("network"), t.get("operator"), t.get("name"), t.get("name:en")}
    for m in master_ids or []:
        mt = rels[m][0]
        mine |= {mt.get("network"), mt.get("operator"), mt.get("name"), mt.get("name:en")}
    mine.discard(None)
    mine.discard("")
    out = []
    for r in on:
        r = int(r)
        tg = stops[r][0]
        if tg.get("railway") not in ("stop", "halt") and tg.get("public_transport") != "stop_position":
            continue
        modes = {m for m in _ONTRACK_KINDS if tg.get(m) == "yes"}
        if modes and t.get("route") not in modes:
            continue
        own = {tg.get("network"), tg.get("operator")} - {None, ""}
        if own and mine and not own & mine:
            continue
        if not tg.get("name") or UNOPENED.search(tg.get("name", "") + " " + tg.get("name:en", "")) or                 any(tg.get(f"{x}:railway") or tg.get(x) for x in ("disused", "proposed", "construction", "abandoned")):
            continue
        out.append((r, (stops[r][1], stops[r][2])))
    return out


def chain(wlist, xy_of, k):
    """A relation's ways (node id lists, in member order) as one polyline of (lon, lat): each way
    turned to join the one before at a shared node, else at its nearer end across a gap."""
    out = []
    for n, nodes in enumerate(wlist):
        nodes = [int(x) for x in nodes]
        if not out:
            out = nodes
            continue
        if n == 1 and out[-1] not in (nodes[0], nodes[-1]) and out[0] in (nodes[0], nodes[-1]):
            out = out[::-1]  # the first way ran the other way
        if out[-1] == nodes[0]:
            out += nodes[1:]
        elif out[-1] == nodes[-1]:
            out += nodes[::-1][1:]
        else:
            e = xy_of(out[-1])
            a, b = xy_of(nodes[0]), xy_of(nodes[-1])
            if e and a and b:
                da = math.hypot((a[0] - e[0]) * k, a[1] - e[1])
                db = math.hypot((b[0] - e[0]) * k, b[1] - e[1])
                out += nodes if da <= db else nodes[::-1]
            else:
                out += nodes
    pts = [xy_of(x) for x in out]
    return [p for p in pts if p is not None]


def cmd_extract(ids):
    reg = registry()
    by_cc = defaultdict(list)
    for i in ids:
        by_cc[region_of(i, reg)].append(i)
    os.makedirs(OUT, exist_ok=True)
    for cc, ps in by_cc.items():
        rels, stops, ways = load_region(cc)
        lookup = {}

        def xy_of(nid):
            import numpy as np
            if "c" not in lookup:
                lookup["c"] = coords(cc)
            a, x, y = lookup["c"]
            j = int(np.searchsorted(a, nid))
            return (float(x[j]), float(y[j])) if j < len(a) and a[j] == nid else None
        for p in ps:
            d = cut(p, rels, stops, ways, xy_of)
            path = os.path.join(OUT, f"osm_{p}.json")
            json.dump(d, open(path + ".part", "w", encoding="utf-8"), ensure_ascii=False)
            os.replace(path + ".part", path)
            print(f"{p}: {d['realauto']['kept']} route relations {d['realauto']['kinds']}, dropped {d['realauto']['dropped']} -> {path}", flush=True)


def cmd_tracklines(ids):
    """T-3935's dry run: the lines realtracks finds for each city, added or not (its `tracks`), printed and written
    to $REALNET_OUT/tracklines.json. Writes nothing else."""
    reg = registry()
    by_cc = defaultdict(list)
    for i in ids:
        try:
            by_cc[region_of(i, reg)].append(i)
        except SystemExit as e:
            print(e)
    os.makedirs(OUT, exist_ok=True)
    out = {}
    for cc, ps in by_cc.items():
        rels, stops, ways = load_region(cc)
        lookup = {}

        def xy_of(nid):
            import numpy as np
            if "c" not in lookup:
                lookup["c"] = coords(cc)
            a, x, y = lookup["c"]
            j = int(np.searchsorted(a, nid))
            return (float(x[j]), float(y[j])) if j < len(a) and a[j] == nid else None
        for p in ps:
            d = cut(p, rels, stops, ways, xy_of)["realauto"]
            out[p] = {k: d[k] for k in ("track_lines", "track_added", "track_refused", "track_partly_covered", "track_refused_lines")}
            print(f"{p}: {len(d['track_lines'])} lines from track ({d['track_added']} added), refused {dict(d['track_refused'])}", flush=True)
            for L in d["track_lines"]:
                print(f"   + {L['name']} [{L['route']}, {L['source']}] {L['names'][0]} - {L['names'][-1]}, {L['stops']} stops, {L['km']} km, covered {L['covered']}", flush=True)
            for L in d["track_partly_covered"]:
                print(f"   ? {L['line']} {L['from']} - {L['to']}, {L['stops']} stops, {L['km']} km, covered {L['covered']}", flush=True)
            json.dump(out, open(os.path.join(OUT, "tracklines.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def py(*args):
    return [sys.executable, "-u", *args]


def line_key(net, l):
    """A line's key that stays put between builds: its route and its stations' keys in order."""
    return l["route"] + ":" + ",".join(net["stations"][i]["key"] for i in l["stops"])


def dropped(city):
    """Lines `build` left out after they did not run (auto_<city>_drop.json: line keys and names)."""
    p = os.path.join(OUT, f"auto_{city}_drop.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}


def keep_path(city):
    return os.path.join(KEEP_DIR, f"{city}.json")


def cmd_keep(city, save, label=None):
    """Write the city's keep list from a save: each of its lines' name and stop names in order."""
    from readsave import read
    d = read(save)
    pats = [{"name": l["name"], "stops": [d["nodes"][s].get("name", "") for s in l["stops"]]} for l in d["lines"]]
    os.makedirs(KEEP_DIR, exist_ok=True)
    rows = ",\n".join(json.dumps(p, ensure_ascii=False) for p in pats)
    body = f'{{"from": {json.dumps(label or save, ensure_ascii=False)},\n"lines": [\n{rows}\n]}}\n'
    open(keep_path(city), "w", encoding="utf-8", newline="\n").write(body)
    print(f"{city}: {len(pats)} lines, {len({p['name'] for p in pats})} routes -> {keep_path(city)}")


def kept(city, net):
    """The indices of the network JSON's lines the city's keep list picks, or None with no list. Each
    listed line picks one line of its name, the one sharing the most stop names (share of both lists' names,
    then the nearer stop count), one each where there are enough (most shared first)."""
    p = keep_path(city)
    if not os.path.exists(p):
        return None
    pats = json.load(open(p, encoding="utf-8"))["lines"]
    by_name = defaultdict(list)
    for j, l in enumerate(net["lines"]):
        by_name[l["name"]].append((j, {net["stations"][i]["name"] for i in l["stops"]}, len(l["stops"])))
    pairs = []
    for k, pat in enumerate(pats):
        S = set(pat["stops"])
        for j, T, n in by_name.get(pat["name"], []):
            share = len(S & T) / max(len(S | T), 1)
            if share > 0:
                pairs.append((-share, abs(n - len(pat["stops"])), k, j))
    pairs.sort()
    got, used = {}, set()
    for _, _, k, j in pairs:
        if k not in got and j not in used:
            got[k] = j
            used.add(j)
    for _, _, k, j in pairs:  # more listed lines of a name than the network has: they share
        got.setdefault(k, j)
    missing = sorted({pats[k]["name"] for k in range(len(pats)) if k not in got})
    if missing:
        print(f"{city}: keep list lines with no line of their name and stops: {missing}")
    return set(got.values())


def cmd_drop(city):
    """Take the dropped lines out of the city's network JSON (after `realnet.py net`), and with a keep
    list every line it does not pick; its demand arrays made again as `net` makes them, stations no
    line stops at left out."""
    drop = dropped(city)
    p = os.path.join(OUT, f"{city}_real.json")
    net = json.load(open(p, encoding="utf-8"))
    pick = kept(city, net)
    keep = [l for j, l in enumerate(net["lines"]) if line_key(net, l) not in drop and (pick is None or j in pick)]
    if pick is not None:
        print(f"{city}: the keep list picks {len(pick)} of {len(net['lines'])} lines")
    print(f"{city}: {len(net['lines']) - len(keep)} of {len(net['lines'])} lines left out ({len(drop)} listed)")
    used = sorted({i for l in keep for i in l["stops"]})
    new = {i: k for k, i in enumerate(used)}
    net["stations"] = [net["stations"][i] for i in used]
    for l in keep:
        l["stops"] = [new[i] for i in l["stops"]]
    net["lines"] = keep
    d = {"st_xy": [v for s in net["stations"] for v in (s["x"], s["y"])], "line_n": [], "stops": [], "times": [], "tph": [], "cars": []}
    for l in keep:
        d["line_n"].append(len(l["stops"]))
        d["stops"] += l["stops"]
        for run in (l["t0"], l["t1"]):
            for lv in range(3):
                d["times"] += run[lv]
        d["tph"] += l["tph"]
        d["cars"].append(l["cars"])
    net["demand"] = d
    json.dump(net, open(p + ".part", "w"))
    os.replace(p + ".part", p)


def not_running(city):
    """{line key: name} of the lines the realsave log has not running."""
    log = open(os.path.join(OUT, f"auto_{city}.log"), encoding="utf-8", errors="replace").read()
    log = log[log.rfind("== realsave"):]  # this build's, not the one before a drop
    net = json.load(open(os.path.join(OUT, f"{city}_real.json"), encoding="utf-8"))
    out = {}
    for i in re.findall(r"^line (\d+) \(.*\) not running", log, re.M):
        l = net["lines"][int(i)]
        out[line_key(net, l)] = l["name"]
    return out


def build_one(city, exe):
    """The city's steps; when some lines do not run (a terminal with more tracks than a game node
    takes), they are left out (`drop`) and the city is built again, until every line runs or
    BUILD_PASSES builds."""
    log = os.path.join(OUT, f"auto_{city}.log")
    env = {**os.environ, "REALNET_OUT": OUT, "OMP_NUM_THREADS": "1", "PYTHONIOENCODING": "utf-8"}
    steps = [("net", py(os.path.join(HERE, "realnet.py"), "net", city)),
             ("drop", py(os.path.abspath(__file__), "drop", city)),
             ("geom", py(os.path.join(HERE, "realnet.py"), "geom", city)),
             ("corridors", py(os.path.join(HERE, "realcorridors.py"), city, "--out", OUT)),
             ("realsave", [str(exe), "--dir", OUT, "--city", city, "--out", OUT, "--shared", "--slow"] + (["--left"] if any(c["id"] == city and c.get("left") for c in listed()) else [])),
             ("gamesave", py(os.path.join(HERE, "realnet.py"), "gamesave", city))]
    times = {}
    # a drop list from an earlier build would leave out lines this one could run
    if os.path.exists(os.path.join(OUT, f"auto_{city}_drop.json")):
        os.remove(os.path.join(OUT, f"auto_{city}_drop.json"))
    for attempt in range(BUILD_PASSES):
        with open(log, "a" if attempt else "w", encoding="utf-8") as f:
            for name, cmd in steps[min(attempt, 1):]:  # again from the network JSON already made
                t0 = time.time()
                f.write(f"== {name}: {' '.join(cmd)}\n")
                f.flush()
                r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=MAPS)
                times[name] = round(times.get(name, 0) + time.time() - t0, 1)
                if r.returncode:
                    f.write(f"== {name} failed ({r.returncode})\n")
                    return report(city, times, failed=name)
        bad = not_running(city)
        if not bad or attempt == BUILD_PASSES - 1:
            break
        drop = {**dropped(city), **bad}
        json.dump(drop, open(os.path.join(OUT, f"auto_{city}_drop.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        with open(log, "a", encoding="utf-8") as f:
            f.write(f"== again without {len(bad)} lines that did not run: {sorted(bad.values())}\n")
    return report(city, times)


def report(city, times=None, failed=None):
    """The city's figures from its files and log -> auto_<city>.json."""
    log = open(os.path.join(OUT, f"auto_{city}.log"), encoding="utf-8", errors="replace").read()
    if times is None and os.path.exists(os.path.join(OUT, f"auto_{city}.json")):  # `report` again: the build's own
        old = json.load(open(os.path.join(OUT, f"auto_{city}.json"), encoding="utf-8"))
        times, failed = old.get("seconds"), old.get("failed")
    r = {"city": city, "failed": failed, "seconds": times or {}}
    p = os.path.join(OUT, f"osm_{city}.json")
    if os.path.exists(p):
        r["extract"] = json.load(open(p, encoding="utf-8"))["realauto"]
    p = os.path.join(OUT, f"{city}_real.json")
    if os.path.exists(p):
        net = json.load(open(p, encoding="utf-8"))
        r["stations"] = len(net["stations"])
        r["lines"] = len(net["lines"])
        r["kinds"] = dict(Counter(l["kind"] for l in net["lines"]))
        r["coloured"] = sum(1 for l in net["lines"] if l.get("colour"))
        r["latin_names"] = round(sum(1 for s in net["stations"] if latin(s["name"])) / max(len(net["stations"]), 1), 3)
    p = os.path.join(OUT, f"{city}_real_corridors.json")
    if os.path.exists(p):
        pcs = json.load(open(p, encoding="utf-8"))["pieces"]
        # pieces the router found no real-network track for (laid on the relation's own ways instead)
        r["off_graph_km"] = round(sum(P["km"] for P in pcs if P.get("off_graph")) / max(sum(P["km"] for P in pcs), 1e-9), 3)
    m = re.findall(r"(\d+) lines running \((\d+) not\)", log)
    if m:
        r["running"], r["not_running"] = int(m[-1][0]), int(m[-1][1])
    m = re.findall(r"hops traced on real track (\d+) .*?from a shape or straight (\d+)", log)
    if m:
        a, b = int(m[-1][0]), int(m[-1][1])
        r["hops_on_track"] = round(a / max(a + b, 1), 3)
    m = re.findall(r"free run (\d+), with the capacity pass's holds (\d+) / (\d+) / (\d+)", log)
    if m:
        free, high = int(m[-1][0]), int(m[-1][1])
        r["held_high"] = round(high / max(free, 1) - 1, 3)
    m = re.findall(r"load \+ derive (\d+) ms", log)
    if m:
        r["derive_s"] = round(int(m[-1]) / 1000, 1)
    m = re.findall(r"crossings (\d+) \((\d+) flat\), total cost US\$(\d+)M", log)
    if m:
        r["crossings"], r["flat"], r["cost_busd"] = int(m[-1][0]), int(m[-1][1]), round(int(m[-1][2]) / 1000, 1)
    p = os.path.join(OUT, f"{city}_real.save")
    r["save_bytes"] = os.path.getsize(p) if os.path.exists(p) and not failed else None
    r["pass"] = bool(r["save_bytes"]) and r.get("lines", 0) >= 2 and r.get("running", 0) >= PASS_RUNNING * r.get("lines", 1)
    json.dump(r, open(os.path.join(OUT, f"auto_{city}.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    return r


def table(rows):
    """One row per city: stations, lines, lines running, share of station names in Latin letters,
    share of track km off the real-network graph, high-demand holds against the free run, the
    network's price, the save's size, build seconds."""
    print(f"{'city':5} {'pass':5} {'stations':>8} {'lines':>5} {'run':>4} {'latin':>6} {'off':>6} {'held':>6} {'US$B':>6} {'KB':>5}  seconds / failed")
    for r in rows:
        print(f"{r['city']:5} {str(r['pass']):5} {r.get('stations', '-'):>8} {r.get('lines', '-'):>5} {r.get('running', '-'):>4} "
              f"{r.get('latin_names', '-'):>6} {r.get('off_graph_km', '-'):>6} {r.get('held_high', '-'):>6} "
              f"{r.get('cost_busd', '-'):>6} {round((r.get('save_bytes') or 0) / 1e3):>5}  {sum((r['seconds'] or {}).values()):.0f} {r['failed'] or ''}")


def cmd_build(ids, jobs):
    sys.path.insert(0, HERE)
    import simbin
    exe = os.environ.get("REALSAVE_EXE") or simbin.exe(ROOT, "realsave")
    need = [i for i in ids if not os.path.exists(os.path.join(OUT, f"osm_{i}.json"))]
    if need:
        cmd_extract(need)
    with ThreadPoolExecutor(jobs) as ex:
        rows = list(ex.map(lambda c: build_one(c, exe), ids))
    table(rows)


def shipped_dir():
    """The folder of shipped saves; REALNET_PACKS_REAL points the ship gate elsewhere (a test)."""
    return os.environ.get("REALNET_PACKS_REAL") or os.path.join(ROOT, "data", "packs", "real")


def save_service(path):
    """(routes, served stations) of a save: route = a line name, station = a stop name some line stops at."""
    from readsave import read
    d = read(path)
    routes, stations = set(), set()
    for l in d["lines"]:
        routes.add(l["name"])
        for s in l["stops"]:
            n = d["nodes"][s].get("name", "")
            if n:
                stations.add(n)
    return routes, stations


def lost_service(city):
    """(routes, stations) the city's new save no longer serves against the save in the shipped folder;
    both empty when nothing went or the city has no shipped save yet."""
    import realnet
    old = os.path.join(shipped_dir(), f"{city}.save")
    if not os.path.exists(old):
        return [], []
    ro, so = save_service(old)
    rn, sn = save_service(realnet.out_or_work(f"{city}_real.save"))
    return sorted(ro - rn), sorted(so - sn)


def cmd_ship(ids, force=False):
    sys.path.insert(0, HERE)
    ok = {c["id"] for c in listed() if c.get("ship")}
    go = []
    for i in ids:
        p = os.path.join(OUT, f"auto_{i}.json")
        r = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
        if i not in ok or not r.get("pass"):
            print(f"{i}: not shipped ({'not listed to ship' if i not in ok else 'did not pass'})")
            continue
        routes, stations = lost_service(i)
        if (routes or stations) and not force:
            print(f"{i}: not shipped, the new save loses {len(routes)} routes and {len(stations)} stations the shipped one serves (--force ships it anyway)")
            print(f"   routes: {routes}")
            print(f"   stations: {stations}")
            continue
        go.append(i)
    if go:
        subprocess.run(py(os.path.join(HERE, "realnet.py"), "ship", *go), env={**os.environ, "REALNET_OUT": OUT}, check=True)
    write_app_list()


def write_app_list():
    """The app's list of cities with a real network: every realnet.SHIPPED city whose save is in
    data/packs/real."""
    import realnet
    have = [c for c in realnet.SHIPPED if os.path.exists(os.path.join(ROOT, "data", "packs", "real", f"{c}.save"))]
    body = json.dumps(have) + "\n"
    if not os.path.exists(APP_LIST) or open(APP_LIST, encoding="utf-8").read() != body:
        open(APP_LIST + ".part", "w", encoding="utf-8").write(body)
        os.replace(APP_LIST + ".part", APP_LIST)
    print(f"{len(have)} real networks -> {APP_LIST}")


if __name__ == "__main__":
    a = sys.argv[1:]
    cmd = a[0] if a else "report"
    jobs = 2
    if "--jobs" in a:
        i = a.index("--jobs")
        jobs = int(a[i + 1])
        del a[i:i + 2]
    force = "--force" in a
    if force:
        a.remove("--force")
    ids = a[1:] or [c["id"] for c in listed()]
    if cmd == "extract":
        cmd_extract(ids)
    elif cmd == "tracklines":
        cmd_tracklines(ids)
    elif cmd == "build":
        cmd_build(ids, jobs)
    elif cmd == "report":
        table([report(i) for i in ids if os.path.exists(os.path.join(OUT, f"auto_{i}.log"))])
    elif cmd == "drop":
        cmd_drop(ids[0])
    elif cmd == "keep":
        sys.path.insert(0, HERE)
        cmd_keep(a[1], a[2], a[3] if len(a) > 3 else None)
    elif cmd == "ship":
        cmd_ship(ids, force)
    else:
        sys.exit(__doc__)
