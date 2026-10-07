"""Canada: register lines from the FRA's North American Rail Network (NARN), by subdivision.

    python ca_register.py --fetch     # data/raw/ca/narn_passenger.geojson + narn_layer.json
    python ca_register.py --narn      # the register half alone, no OSM: lines, km, path checks
    python ca_register.py --fetch-holes   # data/raw/ca/narn_holes.geojson (needs data/proc/ca)
    python ca_register.py --holes-dry     # the envelopes --fetch-holes would query, no fetch
    python build_model.py --region ca --register ca_register:data/raw/ca/narn_passenger.geojson

The same source and the same reader as the USA (us_register.py, whose docstring is the
method): NARN covers Canada too, COUNTRY='CA', with the same fields, and its node ids are one
numbering across the border, so a Canadian line ending at the border ends at the very node the
US build's "Canada – United States border" junction (uj<node>) stands on, and the two builds
join there with no table. This module says only what differs for Canada and runs
us_register's code with it (`adopt`, which sets us_register's country-specific globals for
this process; us_register is a shared file and is not edited from here).

WHAT DIFFERS FROM THE USA
  - PASSNGR codes kept (CA_PASSENGER): V VIA Rail, O "Ontario Northland" (in Canada NARN
    uses O for every other passenger operator: Ontario Northland, Tshiuetin over QNSL, the
    old BC Rail and Algoma Central passenger lines), C commuter, B and A where Amtrak runs into
    Canada (the Cascades to Vancouver, the Adirondack to Montréal, the Maple Leaf to Toronto).
    Left out as in the USA: T tourist (Anita: service more often than about weekly), R rapid
    transit (none in Canada's file). NARN's Canadian passenger codes are older than its US
    ones in places (the E&N, the Gaspé line, the BC Rail and Algoma Central lines, Ontario
    Northland's North Bay - Cochrane, all without trains for 10-20 years): a section between
    two stops no OSM passenger route runs over is left out (us_register), and a junction-ended
    one unless a scheduled route runs over half of it (`drop_unserved_junction_sections`;
    OSM routes that are no scheduled service, the Rocky Mountaineer and BC Rail's old route
    among them, never count: NOT_A_SERVICE).
  - Directional running (DIRECTIONAL): CPKC's Thompson Subdivision is folded into CN's
    Ashcroft as a companion; CPKC's Parry Sound and Cascade lie too far from CN's Bala and Yale.
  - The commuter railways (GO, exo, West Coast Express, UP Express) run largely on track NARN
    codes V or not at all; the holes file brings in the uncoded track their OSM routes run
    over, as for the USA.
  - OWNERS: the Canadian reporting marks, added to the US map. NAME_FIX: a few spellings.
  - PATH_CHECKS: Canadian distances over NARN against Wikipedia.
  - Line ids hash "ca|<owner>|<key>" (the USA's "us|..."), "u" + 10 hex like the USA's.
  - The OSM half is data/proc/ca; infra names from data/proc/ca/infra.pkl.
  - HOLE_SKIP_NETWORKS: VIA Rail vouches for holes here (the US set leaves it out only
    because VIA's few routes mapped in the US are not US trains).

Anita's decisions (2026-10-02): register lines are subdivisions; VIA's long-distance and
once-a-day trains (the Canadian, the Ocean, the Skeena, the Hudson Bay...) are named trains;
the Québec City - Windsor corridor services, GO, exo, West Coast Express and UP Express are
lines. Both directions of a line are one track unless far apart (`companion_of`, as the USA).
"""
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np

import us_register as us

# A line whose sections do not all connect once build_model has dropped what no train runs
# over is one line per piece: us_register's build_model hook, with its record of the pieces
# (us_register.LINE_PIECES, the same dict).
split_pieces = us.split_pieces
LINE_PIECES = us.LINE_PIECES

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw" / "ca"
PROC = ROOT / "data" / "proc" / "ca"
REGION = "ca"
COUNTRY = "CA"
WHERE = "COUNTRY='CA' AND PASSNGR IS NOT NULL AND PASSNGR<>''"
HOLE_WHERE = "COUNTRY='CA' AND (PASSNGR IS NULL OR PASSNGR='')"
LAYER = us.NARN.rsplit("/query", 1)[0]

# V VIA Rail, O other passenger operators (NARN: "Ontario Northland"), C commuter, B Amtrak
# and commuter, A Amtrak. On 2026-10-02: V 13,503 km, O 2,470, T 195, C 197, B 134, A 120.
CA_PASSENGER = {"V", "O", "C", "B", "A"}

# Reporting marks of Canadian track owners (us.OWNERS has CN, CPKC, BNSF).
CA_OWNERS = {
    "VIA": "VIA Rail Canada", "GO": "Metrolinx", "HBRY": "Hudson Bay Railway",
    "KRC": "Keewatin Railway", "ONT": "Ontario Northland",
    "QNSL": "Quebec North Shore and Labrador Railway", "TSH": "Tshiuetin Rail Transportation",
    "ACR": "Algoma Central Railway", "SVI": "Southern Railway of Vancouver Island",
    "BCR": "BC Rail",
}
# Cleaned upper-case keys the register spells as no railway would.
CA_NAME_FIX = {
    "THE PAS TERMINAL": "The Pas Terminal",
}

# Canadian distances by rail, over the kept NARN track, against Wikipedia. (lat, lon).
# The corridor figures are the km posts of Wikipedia's route diagram (Template:Via Corridor
# routing, retrieved 2026-10-02), the rest the train articles' infoboxes.
CA_PATH_CHECKS = [
    ("Toronto - Montréal", (43.6453, -79.3806), (45.4999, -73.5664), 539.0,
     "VIA corridor via Kingston and Coteau, WP route diagram"),
    ("Toronto - Ottawa", (43.6453, -79.3806), (45.4166, -75.6517), 446.0,
     "VIA corridor via Brockville and Smiths Falls, WP route diagram"),
    ("Montréal - Québec", (45.4999, -73.5664), (46.8175, -71.2137), 272.0,
     "VIA corridor via Drummondville, 811 - 539, WP route diagram"),
    # Toronto - Windsor runs via Brantford; the diagram's 216 for London on that branch is
    # 185 over NARN, but its 359 to Windsor agrees. Its Sarnia (290) is via Kitchener, which
    # the shortest path over NARN does not take (it goes via Brantford, 278.8), so not used.
    ("Toronto - Windsor", (43.6453, -79.3806), (42.3270, -83.0050), 359.0,
     "VIA corridor via Brantford and London, WP route diagram"),
    ("Toronto - Niagara Falls", (43.6453, -79.3806), (43.1087, -79.0634), 132.0,
     "VIA / GO via Bayview Junction, WP route diagram"),
    ("Montréal - Halifax", (45.4999, -73.5664), (44.6406, -63.5728), 1346.0,
     "the Ocean, WP infobox 1,346 km"),
    ("Toronto - Vancouver", (43.6453, -79.3806), (49.2737, -123.0979), 4466.0,
     "the Canadian, WP infobox 4,466 km"),
    # The train runs into Thompson and back from Sipiwesk (a ~50 km branch each way), which
    # a shortest path skips: expect about 0.94.
    ("Winnipeg - Churchill", (49.8889, -97.1347), (58.7676, -94.1716), 1710.0,
     "Winnipeg - Churchill train, WP infobox 1,710 km, with the Thompson detour"),
    ("The Pas - Churchill", (53.8253, -101.2533), (58.7676, -94.1716), 820.0,
     "Hudson Bay Railway, WP route diagram: Churchill at km 820"),
    ("Jasper - Prince Rupert", (52.8770, -118.0805), (54.3121, -130.3240), 1160.0,
     "Jasper - Prince Rupert train, WP infobox 1,160 km"),
    ("Sudbury - White River", (46.4900, -80.9940), (48.5930, -85.2770), 484.0,
     "Sudbury - White River train, WP infobox 484 km"),
    ("Cochrane - Moosonee", (49.0630, -81.0190), (51.2740, -80.6430), 299.0,
     "Polar Bear Express, WP 186 mi (299 km)"),
    ("Sept-Îles - Schefferville", (50.2170, -66.3990), (54.8050, -66.8250), 573.0,
     "Tshiuetin, WP: 356 km Sept-Îles - Emeril on QNSL + 217 km Emeril - Schefferville"),
    ("Union - Barrie (Allandale Waterfront)", (43.6453, -79.3806), (44.3740, -79.6890), 101.4,
     "GO Barrie line, WP infobox 101.4 km"),
    ("Union - Milton", (43.6453, -79.3806), (43.5250, -79.8770), 50.2,
     "GO Milton line, WP infobox 50.2 km"),
    ("Union - Old Elm", (43.6453, -79.3806), (43.9870, -79.2470), 49.6,
     "GO Stouffville line, WP infobox 49.6 km"),
    ("Waterfront - Mission City", (49.2860, -123.1110), (49.1330, -122.3060), 69.0,
     "West Coast Express, WP infobox 69 km"),
]

# Networks whose OSM routes never vouch for a hole (us.HOLE_SKIP_NETWORKS less VIA Rail).
CA_HOLE_SKIP = set(us.HOLE_SKIP_NETWORKS) - {"VIA Rail"}

# Station overrides, as us_register.NOT_ON (its comment is the method). Found 2026-10-03 by
# listing register sections with an end at a station no route lists (ca_sources.md).
CA_NOT_ON = [
    # The Rocky Mountaineer's own Kamloops station, 1.5 km from VIA's Kamloops North: the
    # Canadian passes it without stopping, and the Rocky Mountaineer is no scheduled service
    # (NOT_A_SERVICE). It gave CN's Clearwater Subdivision a 113 km section from it.
    ("Kamloops", (-120.34203, 50.73155), None, "*",
     "the Rocky Mountaineer's station; VIA stops at Kamloops North"),
    # No exo4 (Candiac) station is called Hays: a station record with no network 300 m from
    # Saint-Constant, which gave CPKC's Adirondack Subdivision a 0.26 km Hays - Saint-Constant.
    ("Hays", (-73.56600, 45.37408), None, "*", "not an exo station"),
]


# ================================================================ the Canadian reader

# Holes are main-network track only (NET M), as in the USA. Tried 2026-10-02 and dropped:
# uncoded yard, siding and lead track (NET Y, S, I) as holes brought 250 km of passing sidings
# and yard tracks in as "second tracks" and same-named stub lines (CN Yale 29.5 km in 35
# pieces, a 51.7 km "Cascade Subdivision (second track)"); the main file's passenger-coded
# Y/S/I segments (108 km) as holes took nothing.


def read_narn(path, log, holes=False):
    """us_register.read_narn for COUNTRY='CA' and CA_PASSENGER (the same body; us_register's
    tests COUNTRY='US' literally)."""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    feats = d["features"]
    segs, left = [], defaultdict(float)
    left_by = defaultdict(float)
    seen = set()
    for f in feats:
        p, g = f["properties"], f.get("geometry")
        if p.get("COUNTRY") != COUNTRY or not g or p.get("FRAARCID") in seen:
            continue
        seen.add(p.get("FRAARCID"))
        km = float(p.get("KM") or 0.0)
        code = p.get("PASSNGR")
        why = None
        if holes:
            if code:
                why = "passenger-coded (in the main file)"
            elif p.get("NET") != "M":
                why = f"not main network (NET {p.get('NET')})"
        elif code == "T":
            why = "tourist (T)"
        elif code == "R":
            why = "rapid transit (R)"
        elif code not in CA_PASSENGER:
            why = f"passenger code {code}"
        elif p.get("NET") != "M":
            why = f"not main network (NET {p.get('NET')})"
        if why:
            left[why] += km
            left_by[(why, p.get("RROWNER1") or "", us.raw_name(p)[0])] += km
            continue
        if g["type"] == "LineString":
            pts = [tuple(c[:2]) for c in g["coordinates"]]
        else:
            pts = [tuple(c[:2]) for part in g["coordinates"] for c in part]
        name, field = us.raw_name(p)
        rights = [p.get(f"TRKRGHTS{i}") for i in range(1, 10)]
        segs.append({"id": int(p["FRAARCID"]), "a": int(p["FRFRANODE"]),
                     "b": int(p["TOFRANODE"]), "km": km, "pts": pts,
                     "geo_km": us.path_m(pts) / 1000,
                     "owner": p.get("RROWNER1") or "", "code": code,
                     "state": p.get("STATEAB") or "", "raw": name, "field": field,
                     "key": us.clean_name(name) if name else "",
                     "rights": [r for r in rights if r], "hole": holes})
    kept = sum(s["km"] for s in segs)
    log(f"NARN (CA){' holes file' if holes else ''}: {len(segs)} segments kept, {kept:,.0f} km; "
        f"left out: "
        + ", ".join(f"{w} {km:,.0f} km" for w, km in sorted(left.items(), key=lambda x: -x[1])))
    return segs, left, left_by


class _BmForCanada:
    """build_model as us_register.build sees it, with build_stations given region "ca" (the
    US reader passes "us" literally, which turns on rules/us.py's METRO_DUP 150 m metro rule)."""

    def __init__(self, bm):
        self._bm = bm

    def __getattr__(self, name):
        return getattr(self._bm, name)

    def build_stations(self, stops, rels, coords, log, region=None):
        return self._bm.build_stations(stops, rels, coords, log, REGION)


def load_osm(log):
    import build_model as bm
    ways, rels, stops, cid, cx, cy = bm.load(REGION, log)
    return _BmForCanada(bm), ways, rels, stops, bm.Coords(cid, cx, cy)


_us_infra_names = us.infra_names
_us_is_passenger_route = us.is_passenger_route


def infra_names(_path_proc, ways_near, log):
    return _us_infra_names(PROC, ways_near, log)


def line_id(owner, key, segs, split, hole=False):
    import hashlib
    tag = f"ca|{owner}|{key}"
    if hole:
        tag += "|hole"
    if split or (hole and not key):
        tag += f"|{min(s['id'] for s in segs)}"
    return "u" + hashlib.blake2b(tag.encode("utf-8"), digest_size=5).hexdigest()


# OSM route=train relations in Canada that are no scheduled passenger service, which would
# otherwise put stations on register lines and vouch for track (2026-10-02 extract, 98 train
# routes): BC Rail's old passenger route ("British Columbia Railway", operator CNR, 315 ways;
# no train since 2002, and its track is NARN-coded O), a freight subdivision mapped as a
# route (CTRW Prince Albert), Port Stanley Terminal Rail, the heritage railways (Alberni &
# Pacific, Fraser Valley, Washago-Rama, Fort Edmonton Park's steam train), and the Rocky
# Mountaineer (a seasonal cruise train; its Rainforest to Gold Rush route runs over the
# NARN-coded O ex-BC Rail line, where nothing else runs). The Waterloo Central and White
# Pass routes carry service=tourism and the Wheatland Express "Excursion", which us_register
# already leaves out.
NOT_A_SERVICE = ("British Columbia Railway", "CTRW", "Port Stanley", "Heritage", "Steam Train",
                 "Rocky Mountaineer", "Rainforest to Gold Rush")


def is_passenger_route(tags):
    if not _us_is_passenger_route(tags):
        return False
    if tags.get("disused") == "yes" or tags.get("state") in ("proposed", "disused"):
        return False
    nm = " ".join(tags.get(k, "") for k in ("name", "network", "operator", "ref"))
    return not any(w in nm for w in NOT_A_SERVICE)


def adopt():
    """Point us_register's country-specific globals at Canada's, for this process."""
    us.RAW = RAW
    us.WHERE = WHERE
    us.HOLE_WHERE = HOLE_WHERE
    us.PASSENGER = CA_PASSENGER
    us.TOURIST_OWNERS = set()
    us.OWNERS.update(CA_OWNERS)
    us.NAME_FIX.update(CA_NAME_FIX)
    us.PATH_CHECKS = CA_PATH_CHECKS
    us.HOLE_SKIP_NETWORKS = CA_HOLE_SKIP
    us.NOT_ON = CA_NOT_ON
    us.read_narn = read_narn
    us.load_osm = load_osm
    us.infra_names = infra_names
    us.line_id = line_id
    us.is_passenger_route = is_passenger_route


JUNCTION_SHARE = 0.5   # a junction-ended section this much under a route is kept (as build_model)


def drop_unserved_junction_sections(lines, stations, geoms, log):
    """Junction-ended sections that no scheduled passenger route (is_passenger_route, so not
    the Rocky Mountaineer or BC Rail's old route) runs over for JUNCTION_SHARE of their
    length. build_model drops junction sections by the same share, but measured against every
    OSM route it has, the Rocky Mountaineer's included, and that cruise train alone runs over
    the ex-BC Rail line North Vancouver - Lillooet - Prince George (NARN codes it O): 600 km
    no scheduled train runs over would count. The Tsal'alh Seton Train (daily) keeps Lillooet -
    Seton Portage."""
    from n02 import walk_order
    _bm, ways, rels, _stops, coords = load_osm(lambda m: None)
    rw = us.route_ways(rels)
    wraw = {}
    for w in {w for ws in rw.values() for w in ws}:
        if w in ways:
            _n, ll = us.way_lonlat(w, ways, coords)
            if ll is not None:
                wraw[w] = ll
    cover = us.RouteCover(wraw)
    n = km = 0
    gone = defaultdict(float)
    for l in lines:
        keep = []
        for sec in l["sections"]:
            a, b = sec[0], sec[1]
            key = f"{a}|{b}"
            if a.startswith("uj") or b.startswith("uj"):
                if cover.share(geoms[l["id"]][key]) < JUNCTION_SHARE:
                    n += 1
                    km += sec[2]
                    gone[l["name"]] += sec[2]
                    geoms[l["id"]].pop(key, None)
                    l["chain"].pop(key, None)
                    continue
            keep.append(sec)
        l["sections"] = keep
    out = [l for l in lines if l["sections"]]
    for l in out:
        l["km"] = round(sum(s[2] for s in l["sections"]), 3)
        l["km_official"] = round(sum(l["chain"].values()), 3)
        l["display"] = us.display_order(l["sections"], walk_order)
    used = defaultdict(set)
    for l in out:
        for s in l["sections"]:
            used[s[0]].add(l["id"])
            used[s[1]].add(l["id"])
    for sid in list(stations):
        if sid not in used:
            del stations[sid]
        else:
            stations[sid]["lines"] = used[sid]
    for l in lines:
        if not l["sections"]:
            geoms.pop(l["id"], None)
    log(f"CA: {n} junction-ended sections ({km:,.1f} km) dropped: no scheduled passenger route "
        f"runs over {JUNCTION_SHARE:.0%} of them; {len(lines) - len(out)} lines left with none")
    for name, k in sorted(gone.items(), key=lambda kv: -kv[1])[:25]:
        log(f"    {k:7.1f} km  {name}")
    return out


# Directional running: CN and CPKC run their two single-track main lines as one double track
# in two places, and VIA's Canadian takes one railway's line each way (Wikipedia, "The
# Canadian"). (line, owner) -> (its pair, owner). A pair lying within us.COMPANION_KM of its
# partner for 90% of it is folded in as the USA's second tracks are (`companion_of`, Anita
# 2026-10-02: both directions of a line one track unless far apart); the build log gives the
# measure either way.
#   - Wanup - Parry Sound: CPKC's Parry Sound Subdivision (NARN codes 14 km of it V, the rest
#     no passenger: it comes in from the holes file) beside CN's Bala Subdivision. Measured
#     2026-10-02: 90% within 17.6 km, so kept a line of its own.
#   - Mission - Kamloops: CPKC's Thompson Subdivision beside CN's Ashcroft (Basque - Kamloops,
#     the Thompson canyon): 90% within 301 m, folded. CPKC's Cascade beside CN's Yale (the
#     Fraser canyon): 90% within 8.0 km, and it carries the West Coast Express besides; only
#     measured, never folded.
DIRECTIONAL = {("Parry Sound Subdivision", "CPKC"): ("Bala Subdivision", "Canadian National"),
               ("Thompson Subdivision", "CPKC"): ("Ashcroft Subdivision", "Canadian National")}
DIRECTIONAL_MEASURED = [(("Cascade Subdivision", "CPKC"), ("Yale Subdivision", "Canadian National"))]


def fold_directional(lines, geoms, log):
    by = defaultdict(list)
    for l in lines:
        by[(l["name"], l["operator"])].append(l)
    pairs = [(a, b, True) for a, b in DIRECTIONAL.items()] + \
            [(a, b, False) for a, b in DIRECTIONAL_MEASURED]
    for a, b, fold in pairs:
        for la in by.get(a, ()):
            mains = by.get(b, ())
            if not mains:
                continue
            best = min(mains, key=lambda m: us.track_apart_m(geoms[la["id"]], geoms[m["id"]]))
            p90 = us.track_apart_m(geoms[la["id"]], geoms[best["id"]])
            ok = fold and p90 <= us.COMPANION_KM * 1000
            if ok:
                la["companion_of"] = best["id"]
            log(f"    directional pair {'folded into' if ok else 'kept apart from'} "
                f"{best['name']}: {la['name']} [{la['operator']}] {la['km']:.1f} km, 90% within "
                f"{p90:,.0f} m")


LINK_M = 60        # built path checks: register track this close is joined...
END_LINK_M = 300   # ...and a section end (a trace stops at the platform track) this close


def built_path_checks(lines, geoms, log):
    """CA_PATH_CHECKS again, over the BUILT register geometry (holes filled, unserved track
    dropped) rather than the passenger-coded NARN alone. A graph over every section's drawn
    vertices; a section end is joined to any other line's vertex within LINK_M (lines meet at
    a node where only one of them has a section end). Ends: the vertex nearest each place."""
    import heapq
    vid, xy = {}, []
    adj = defaultdict(list)

    def v(p):
        k = (round(p[0], 5), round(p[1], 5))
        if k not in vid:
            vid[k] = len(xy)
            xy.append(k)
        return vid[k]
    ends = set()
    at_station = defaultdict(set)      # the section ends at one station join, as in the app
    for l in lines:
        for key, pts in geoms.get(l["id"], {}).items():
            if len(pts) < 2:
                continue
            pts = [tuple(p) for p in us.densify(pts, LINK_M).tolist()]
            ids = [v(p) for p in pts]
            ends |= {ids[0], ids[-1]}
            sa, _, sb = key.partition("|")
            at_station[sa].add(ids[0])
            at_station[sb].add(ids[-1])
            for i, j, p, q in zip(ids[:-1], ids[1:], pts[:-1], pts[1:]):
                w = us.dist_m(*p, *q) / 1000
                adj[i].append((j, w))
                adj[j].append((i, w))
    cell = defaultdict(list)
    for i, (x, y) in enumerate(xy):
        cell[(int(x * 200), int(y * 200))].append(i)       # 0.005 degree cells
    for (cx, cy), here in cell.items():
        near = [j for dx in (-1, 0, 1) for dy in (-1, 0, 1) for j in cell.get((cx + dx, cy + dy), ())]
        for e in here:
            x, y = xy[e]
            r = END_LINK_M if e in ends else LINK_M
            for j in near:
                if j != e and (j > e or e in ends):
                    d = us.dist_m(x, y, *xy[j])
                    if d <= r:
                        adj[e].append((j, d / 1000))
                        adj[j].append((e, d / 1000))
    for vs in at_station.values():
        vs = sorted(vs)
        for i in vs:
            for j in vs:
                if i < j:
                    d = us.dist_m(*xy[i], *xy[j]) / 1000
                    adj[i].append((j, d))
                    adj[j].append((i, d))
    P = np.asarray(xy)

    def nearest(lat, lon):
        d = np.hypot((P[:, 0] - lon) * math.cos(math.radians(lat)), P[:, 1] - lat)
        k = int(np.argmin(d))
        return k, us.dist_m(lon, lat, *xy[k])
    for label, a, b, km, note in CA_PATH_CHECKS:
        (src, ds), (dst, dd) = nearest(*a), nearest(*b)
        dist, h = {src: 0.0}, [(0.0, src)]
        while h:
            d, u = heapq.heappop(h)
            if u == dst:
                break
            if d > dist[u]:
                continue
            for w_, c in adj[u]:
                if d + c < dist.get(w_, us.INF):
                    dist[w_] = d + c
                    heapq.heappush(h, (d + c, w_))
        got = dist.get(dst)
        why = ""
        if got is None:          # where the built network stops short
            far = max(dist, key=dist.get)
            close = min(dist, key=lambda k: us.dist_m(*xy[k], *xy[dst]))
            why = (f"; reached {dist[far]:.0f} km, the nearest reached point to the end is "
                   f"{xy[close]}, {us.dist_m(*xy[close], *xy[dst]) / 1000:.1f} km short")
        log(f"built path check {label}: "
            + ("no path" if got is None else f"{got:.1f} km against {km:.1f}, ratio {got / km:.3f}")
            + f" (ends {ds:.0f} m / {dd:.0f} m off; {note}{why})")


def build(path, log):
    adopt()
    lines, stations, geoms = us.build(path, log)
    lines = drop_unserved_junction_sections(lines, stations, geoms, log)
    fold_directional(lines, geoms, log)
    built_path_checks(lines, geoms, log)
    return lines, stations, geoms


# ================================================================ fetching

def fetch():
    """The passenger-coded Canadian segments, and the layer's own description (licence,
    field domains) beside them."""
    import urllib.request
    RAW.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(LAYER + "?f=json", headers={"User-Agent": us.USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as r:
        meta = json.loads(r.read().decode("utf-8"))
    with open(RAW / "narn_layer.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    print(f"wrote {RAW / 'narn_layer.json'}")
    us.fetch()          # with adopt(): WHERE and RAW are Canada's


def main():
    adopt()
    if "--fetch" in sys.argv:
        fetch()
        return
    us.main()


if __name__ == "__main__":
    main()
