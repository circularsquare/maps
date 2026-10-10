"""East and Southern Africa: Kenya, Ethiopia, Djibouti, Mozambique, Zambia, Zimbabwe, Tanzania,
Madagascar, Malawi, Uganda (and Mauritius, which has no register: its Metro Express is OSM
lines). Each country's main-line passenger lines as a hand-written station list
(eafrica_lines.py), traced over OSM track by rinf.py with nafrica_register.py's code, as
mideast_register.py does (this module hands it the lists; nothing of nafrica's changes).

    python tools/slot.py 2 -- python extract.py --region ke --pbf data/raw/kenya-latest.osm.pbf --station-areas
    python eafrica_register.py --clip ke          # after every extract (drops other countries'
                                                  # ground and the NOT_SERVICE routes)
    python eafrica_register.py --join ke          # where a country's list needs it (see JOIN)
    python eafrica_register.py --fill ke          # where a list point names a station OSM lacks
    python build_model.py --region ke --register eafrica_register:data/raw/rinf/ke
    python eafrica_register.py --trace ke "Nairobi Terminus" "Mombasa Terminus"  # to write lists
    python eafrica_register.py --fork ke A B C    # where the trace A -> C leaves A -> B

Mauritius: `build_model.py --region mu` with no register (its only railway is light rail,
which rinf.py does not trace; its OSM route relations are complete).

WHAT IS A REGISTER LINE HERE.  The main-line railways' passenger stretches, cut so that each
line is wholly running or wholly not (`suspended=True` greys it). No operator publishes a line
register with chainage, so the km are our own traces (`no_chain`) and check_model.REGISTER's
published lengths are the outside check. Light rail and commuter routes that OSM maps as
route relations (Addis Ababa's LRT, Nairobi's commuter services, Dar es Salaam's) are OSM
lines over the register's track, as in North Africa and the Middle East.

WHAT --clip DOES BEYOND nafrica's clip.  (1) Near each BORDERS point it redoes the clip:
the way crossing the border is put back, and where BORDER_LINES gives OSM's own boundary there
the track and stations are sided by it, not by the app's coarse outline (Nakonde, Nayuchi and
2 km of the Ethiopia - Djibouti line were on the wrong side). (2) `tidy`: station names lose
a trailing "Railway Station"/"Station", same-named twins within 300 m go, STATION_NAMES
renames. (3) ROUTE_NAMES: route_masters for OSM routes whose names build_model reads as the
system's name alone ("NCR"). `build` also keeps greyed lines' junction-ended sections.

Sources, what runs and the numbers: <cc>_sources.md ("Build (2026-10-08)"); countries with
nothing running: eafrica_survey.md.
"""
import nafrica_register as nr
from eafrica_lines import (BORDER_LINES, BORDERS, JOIN, LINES, NOT_SERVICE, ROUTE_NAMES,
                           STATION_NAMES)

# nafrica_register's tables, extended in this process only.
nr.LINES.update(LINES)
nr.NOT_SERVICE.update(NOT_SERVICE)
nr.BORDERS.update(BORDERS)
nr.LANGS.update({"ke": ["en", "sw"], "et": ["am", "en"], "dj": ["fr", "ar", "en"],
                 "mz": ["pt", "en"], "zm": ["en"], "zw": ["en"], "tz": ["sw", "en"],
                 "mg": ["fr", "mg", "en"], "mw": ["en"], "ug": ["en"], "mu": ["en", "fr"]})
nr.ISO3.update({"ke": "KEN", "et": "ETH", "dj": "DJI", "mz": "MOZ", "zm": "ZMB", "zw": "ZWE",
                "tz": "TZA", "mg": "MDG", "mw": "MWI", "ug": "UGA", "mu": "MUS"})


NAME_BASE = 9_200_000_000   # synthetic route_master ids: this + the route id


def name_routes(cc, log=print):
    """OSM routes whose names say only the system ("NCR : Nairobi <-> Syokimau": build_model
    takes off the direction tail and five lines were all "NCR") get a route_master of their
    own carrying ROUTE_NAMES' name (id NAME_BASE + the route id, stable). Rewrites
    data/proc/<cc>/rels.pkl; part of --clip, so it reruns with every clip."""
    import os
    import pickle
    d = nr.ROOT / "data" / "proc" / cc
    rels = pickle.load(open(d / "rels.pkl", "rb"))
    rels = {k: v for k, v in rels.items() if not NAME_BASE <= k < NAME_BASE + 10**11}
    made = 0
    for rid, name in ROUTE_NAMES.get(cc, {}).items():
        if rid not in rels:
            log(f"  route {rid} ({name}) not in the extract")
            continue
        t = rels[rid][0]
        if t.get("type") == "route_master":
            # a master of the mapper's own ("AA-LRT : Ayat <-> Tor Hailoch"): renamed
            t.setdefault("official_name", t.get("name", ""))
            t["name"] = name
            made += 1
            continue
        rels[NAME_BASE + rid] = ({"type": "route_master", "route_master": t.get("route", ""),
                                  "name": name, "ref": t.get("ref", ""),
                                  "operator": t.get("operator", ""),
                                  "network": t.get("network", ""),
                                  "colour": t.get("colour", "")}, [("r", rid, "")])
        made += 1
    tmp = d / "rels.pkl.tmp"
    with open(tmp, "wb") as f:
        pickle.dump(rels, f, protocol=4)
    os.replace(tmp, d / "rels.pkl")
    log(f"{cc.upper()}: {made} routes named by a route_master of their own")


SUFFIX = __import__("re").compile(
    r"\s+(?:(?:Railway|Train|Rail)\s+)?(?:Station|Halt|Stop)$", __import__("re").I)
TWIN_M = 300


def tidy(cc, log=print):
    """Station names as the station is called. OSM maps many of these stations twice, a node
    and an area, as "Kibera" and "Kibera Train station", "Donholm" and "Donholm Railway
    Station", and build_model merges same-named stations only: the trailing "Railway Station",
    "Train station", "Station", "Halt" is taken off every rail station's name (not "SGR
    Station": Mtito Andei's SGR and metre-gauge stations are 180 m apart and different
    stations). Then of two stations with one name within TWIN_M, the one no route lists is
    dropped (rinf.py makes every OSM station on a line a stop, so a pair of Morendat nodes 11
    m apart became two stops). STATION_NAMES renames a station first. Rewrites
    data/proc/<cc>/stops.pkl; part of --clip."""
    import math
    import os
    import pickle
    d = nr.ROOT / "data" / "proc" / cc
    stops = pickle.load(open(d / "stops.pkl", "rb"))
    rels = pickle.load(open(d / "rels.pkl", "rb"))
    listed = {r for _t, ms in rels.values() for ty, r, _ in ms if ty == "n"}
    def rail(t):
        # main-line stations only: light rail, tram and metro stops are left as they are
        if t.get("station") in ("light_rail", "subway", "tram", "monorail") or any(
                t.get(m) == "yes" for m in ("light_rail", "subway", "tram", "monorail")):
            return False
        return (t.get("railway") in ("station", "halt", "stop")
                or (t.get("public_transport") == "station" and t.get("train") == "yes"))
    renamed = 0
    for k, name in STATION_NAMES.get(cc, {}).items():
        if k in stops:
            t = stops[k][0]
            t.setdefault("official_name", t.get("name", ""))
            t["name"] = name
            renamed += 1
    for k, (t, lon, lat) in stops.items():
        n = t.get("name") or ""
        if rail(t) and SUFFIX.search(n) and SUFFIX.sub("", n).strip():
            t.setdefault("official_name", n)
            t["name"] = SUFFIX.sub("", n).strip()
            renamed += 1
    by = {}
    for k, (t, lon, lat) in stops.items():
        if rail(t) and t.get("name"):
            by.setdefault(t["name"].casefold(), []).append(k)
    drop = set()
    for ks in by.values():
        if len(ks) < 2:
            continue
        # keep a listed one, then a railway=station node, then an OSM node over an area
        ks.sort(key=lambda k: (k not in listed, stops[k][0].get("railway") != "station", k < 0))
        for i, a in enumerate(ks):
            if a in drop:
                continue
            for b in ks[i + 1:]:
                if b in drop or b in listed:
                    continue
                (_, x1, y1), (_, x2, y2) = stops[a], stops[b]
                dm = math.hypot((x2 - x1) * math.cos(math.radians(y1)) * 111320,
                                (y2 - y1) * 110570)
                if dm <= TWIN_M:
                    drop.add(b)
    for k in drop:
        del stops[k]
    tmp = d / "stops.pkl.tmp"
    with open(tmp, "wb") as f:
        pickle.dump(stops, f, protocol=4)
    os.replace(tmp, d / "stops.pkl")
    log(f"{cc.upper()} tidy: {renamed} station names without their 'Station', {len(drop)} "
        f"same-named twins within {TWIN_M} m dropped")


BORDER_KEEP_M = 60


def keep_border_ways(cc, before, before_stops, log=print):
    """The clip's outline is coarse at the border points, so near each of this country's
    BORDERS points the clip is redone by hand.

    nafrica's clip drops a way with half its nodes abroad, so the way that crosses the border
    went from one side's extract (Ethiopia - Djibouti: a 2.3 km way with one node in Djibouti,
    which then started 321 m short of the border and its line could not reach the border
    point). Rail ways the clip dropped that pass within BORDER_KEEP_M of the point are put
    back; build_model gives the part abroad to the neighbour.

    Where BORDER_LINES gives OSM's own boundary at the point (a segment, and a point inside
    each country), everything within its `km` of the point is sided by that segment instead
    of the outline: rail ways and stations on this country's side the clip dropped are put
    back, those on the other side it kept are dropped. At Tunduma - Nakonde the outline runs
    2.5 km west of OSM's boundary and took Nakonde station and its yard out of Zambia."""
    import math
    import os
    import pickle
    import numpy as np
    from shapely.geometry import LineString, Point
    pts = [(bid, lon, lat) for bid, (lon, lat, ccs) in BORDERS.items() if cc in ccs]
    if not pts:
        return
    d = nr.ROOT / "data" / "proc" / cc
    ways = pickle.load(open(d / "ways.pkl", "rb"))
    stops = pickle.load(open(d / "stops.pkl", "rb"))
    c = np.load(d / "coords.npz")
    idx = {int(n): i for i, n in enumerate(c["id"])}
    X, Y = c["x"] / 1e7, c["y"] / 1e7

    def xy_of(ns):
        return [(X[idx[int(n)]], Y[idx[int(n)]]) for n in ns if int(n) in idx]

    def side(seg, p):
        (x1, y1), (x2, y2) = seg
        return (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1) > 0

    def km(a, b):
        return math.hypot((a[0] - b[0]) * math.cos(math.radians(a[1])) * 111.32,
                          (a[1] - b[1]) * 110.57)
    back = dropped = s_back = s_dropped = 0
    for bid, lon, lat in pts:
        bl = BORDER_LINES.get(bid)
        if bl:
            seg, mine, reach = bl["line"], side(bl["line"], bl["ref"][cc]), bl["km"]
            ours = lambda xy: all(side(seg, p) == mine for p in xy)  # noqa: E731
            near = lambda xy: min(km((lon, lat), p) for p in xy) <= reach  # noqa: E731
            for w, (t, ns) in list(ways.items()):
                xy = xy_of(ns)
                if t.get("railway") in ("rail", "narrow_gauge") and xy and near(xy) \
                        and not any(side(seg, p) == mine for p in xy):
                    del ways[w]
                    dropped += 1
            for w, (t, ns) in before.items():
                xy = xy_of(ns)
                if w not in ways and t.get("railway") in ("rail", "narrow_gauge") and xy \
                        and near(xy) and ours(xy):
                    ways[w] = (t, ns)
                    back += 1
            for k, (t, x, y) in list(stops.items()):
                if km((lon, lat), (x, y)) <= reach and side(seg, (x, y)) != mine:
                    del stops[k]
                    s_dropped += 1
            for k, (t, x, y) in before_stops.items():
                if k not in stops and km((lon, lat), (x, y)) <= reach and side(seg, (x, y)) == mine:
                    stops[k] = (t, x, y)
                    s_back += 1
        for w, (t, ns) in before.items():
            if w in ways or t.get("railway") not in ("rail", "narrow_gauge"):
                continue
            xy = xy_of(ns)
            if len(xy) < 2:
                continue
            k = np.cos(np.radians(lat))
            if LineString(xy).distance(Point(lon, lat)) * 110570 * min(1.0, k) <= BORDER_KEEP_M:
                ways[w] = (t, ns)
                back += 1
    for fn, obj in (("ways.pkl", ways), ("stops.pkl", stops)):
        tmp = d / (fn + ".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(obj, f, protocol=4)
        os.replace(tmp, d / fn)
    log(f"{cc.upper()} border: {back} rail ways put back and {dropped} dropped, {s_back} stations "
        f"put back and {s_dropped} dropped, near {len(pts)} border points")


def clip(cc, log=print):
    import pickle
    d = nr.ROOT / "data" / "proc" / cc
    before = pickle.load(open(d / "ways.pkl", "rb"))
    before_stops = pickle.load(open(d / "stops.pkl", "rb"))
    nr.clip(cc, log)
    keep_border_ways(cc, before, before_stops, log)
    tidy(cc, log)
    if ROUTE_NAMES.get(cc):
        name_routes(cc, log)


def join(cc, log=print):
    """mideast_register.join: a loose track end within 40 m of another connected piece of
    running track gets a two-node way to it. For the countries in JOIN only."""
    import mideast_register
    mideast_register.join(cc, log)


def country_conf(cc):
    return nr.country_conf(cc)


def build(path, log):
    """nafrica_register.build, and the junction-ended sections of greyed lines kept too:
    nafrica marks only running lines' as served, so build_model.drop_unridden_sections cut the
    greyed Tanga - Mruazi line at Muheza, 25 km short of the junction. A greyed line here is
    a line the operator ran passengers over, not a freight curve."""
    lines, stations, geoms = nr.build(path, log)
    n = 0
    for l in lines:
        if l.get("served_sections"):
            continue
        keys = [f"{a}|{b}" for a, b, *_ in l["sections"]
                if stations.get(a, {}).get("junction") or stations.get(b, {}).get("junction")]
        if keys:
            l["served_sections"] = keys
            n += len(keys)
    log(f"{len(lines)} lines: {n} junction-ended sections of greyed lines kept")
    return lines, stations, geoms


def split_pieces(lines, stations, geoms, reg_ways, state, log):
    nr.split_pieces(lines, stations, geoms, reg_ways, state, log)


if __name__ == "__main__":
    import argparse
    import time
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", metavar="CC")
    ap.add_argument("--fill", metavar="CC")
    ap.add_argument("--join", metavar="CC")
    ap.add_argument("--convert", metavar="CC")
    ap.add_argument("--dry", metavar="CC", help="convert without writing")
    ap.add_argument("--trace", nargs="+", metavar="ARG")
    ap.add_argument("--fork", nargs=4, metavar=("CC", "A", "B", "C"))
    a = ap.parse_args()
    t = time.time()
    lg = lambda m: print(f"[{time.time() - t:6.1f}s] {m}", flush=True)  # noqa: E731
    if a.clip:
        clip(a.clip, lg)
    if a.join:
        join(a.join, lg)
    if a.fill:
        nr.fill(a.fill, lg)
    if a.convert:
        nr.convert(a.convert, lg)
    if a.dry:
        nr.convert(a.dry, lg, write=False)
    if a.trace:
        nr.trace_cmd(a.trace[0], a.trace[1:])
    if a.fork:
        nr.fork_cmd(*a.fork)
