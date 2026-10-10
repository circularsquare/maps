"""West and Central Africa (ng, cm, ao, ga, cg, sn, gh, bf, cd): each country's passenger line
list written into rinf.py's input format (nafrica_register.py's recipe), so rinf.py traces it
over OSM track, places the stops and names the lines.

    python extract.py --region ng --pbf data/raw/nigeria-latest.osm.pbf --station-areas
    python wafrica_register.py --clip ng          # after every extract (see CLIPPING)
    python wafrica_register.py --fill ng          # after every --clip
    python build_model.py --region ng --register wafrica_register:data/raw/rinf/ng
    python wafrica_register.py --convert ng       # the conversion alone, with its log
    python wafrica_register.py --trace ng "Iddo" "Ijoko"   # a trace, to write lists
    python wafrica_register.py --gauges ng        # track km by gauge and name, for OWN rules

`--register wafrica_register:data/raw/rinf/<cc>` converts and then runs rinf.build on the
result. The settings rinf.py reads are in rinf_countries/<cc>.py (each `country_conf` below).
Sources, what runs and the numbers: <cc>_sources.md; the lists are in wafrica_lines.py.

WHY A LIST.  None of the nine has a line register with chainage that a script can read, and
OSM names little of their track. So each line is written by hand: its stations in order (ends,
junction stations and enough stations between to keep the trace on the right track), from the
operator's timetable and OSM's track. km are traced over OSM's track, so the published lengths
in check_model.REGISTER are the outside check (`no_chain`: no km_official is shipped).

POINTS are written as in nafrica_register.py:
    "Name"                 an OSM rail station of that name (`skey`), the one nearest the
                           line's last point
    "Name@lon,lat"         that station near the coordinate (within NEAR_M); with no station of
                           the name there, the nearest OSM rail station within BLIND_M
    "~Name@lon,lat"        a junction, no stop, at the coordinate
    "#ID"                  a border point (BORDERS)
A line may have `more` pieces (branches) under the same id. `osm_stops: "all"` puts every OSM
rail station lying on a traced section on the line as a stop, unless the line is `listed_only`.

PARALLEL RAILWAYS (OWN).  Where two railways run side by side (Lagos: NRC's Cape gauge, the
Lagos - Ibadan standard gauge and LAMATA's Red Line, Ebute Metta to Agbado), a shortest-path
trace jumps between them. wafrica_lines.OWN[cc] lists (line id, rule on a way's tags): a way
the first rule matches is that line's own track, preferred in rinf.py's second pass through
its existing `way_line` hook (OFF_LINE_COST off it), and in this file's own traces.

REPAIRS IN --clip (data/proc/<cc> only): `join_gaps` joins two dead ends of one gauge within
JOIN_M that continue each other (OSM ways left unjoined), and `split_gauges` gives each gauge
its own copy of a node two gauges share, so no trace changes gauge.
"""
import argparse
import json
import math
import os
import pickle
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "raw" / "rinf"
SHAPES = ROOT.parent / "religiondots" / "data" / "processed" / "country_shapes.geojson"

NEAR_M = 3000      # "Name@lon,lat": a station of the name this close to the coordinate
BLIND_M = 400      # ... else any OSM rail station this close
MAX_JUMP_KM = 400  # "Name": a station of the name at most this far from the line's last point

SOURCE_TAG = "wafrica_register timetable"

# ============================================================== names and keys

STATION_WORDS = re.compile(r"\b(?:gare|de|du|des|d|la|le|les|l|station|halte|arret|ville|"
                           r"railway|train|estacao|apeadeiro|do|da|dos|das|e|of|the|"
                           r"terminal|sgr)\b")


def skey(s):
    """One key for a station name: case, accents, "Gare", "Station", "Estação" folded away."""
    s = unicodedata.normalize("NFKD", (s or "").casefold())
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[’'`]", " ", s)
    return re.sub(r"[^0-9a-z]", "", re.sub(r"\b(?:gare|station|halte|estacao)\b", " ", s))


def skey_short(s):
    s = unicodedata.normalize("NFKD", (s or "").casefold())
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[’'`\-]", " ", s)
    return re.sub(r"[^0-9a-z]", "", STATION_WORDS.sub(" ", s))


def dist_m(lon1, lat1, lon2, lat2):
    dx = (lon2 - lon1) * math.cos(math.radians((lat1 + lat2) / 2)) * 111320
    return math.hypot(dx, (lat2 - lat1) * 110570)


# ============================================================== the line lists

LANGS = {"ng": ["en"], "gh": ["en"], "cm": ["fr", "en"], "ga": ["fr"], "cg": ["fr"],
         "sn": ["fr"], "bf": ["fr"], "cd": ["fr"], "ao": ["pt"]}
ISO3 = {"ng": "NGA", "gh": "GHA", "cm": "CMR", "ga": "GAB", "cg": "COG", "sn": "SEN",
        "bf": "BFA", "cd": "COD", "ao": "AGO"}

from wafrica_lines import BORDERS, LINES, NOT_SERVICE, OWN  # noqa: E402


def way_owners(cc, tags):
    """Every line whose own track a way is, in OWN's order (a way may be several lines' own:
    the Lagos - Ibadan SGR's track is also the Red Line's)."""
    if tags.get("railway") not in ("rail", "narrow_gauge", "preserved", "light_rail",
                                   "subway", "construction", "disused"):
        return ()
    out = []
    for lid, rule in OWN.get(cc) or ():
        if lid not in out and rule(tags):
            out.append(lid)
    return tuple(out)


def way_owner(cc):
    """rinf.py's `way_line` hook: way tags -> the line id whose own track the way is. rinf.py
    reads one name per way, so the first OWN rule that matches wins (were rinf.py ever to set
    WAY_LINE_MULTI and take a tuple, every match would be passed)."""
    if not OWN.get(cc):
        return None
    import rinf
    multi = getattr(rinf, "WAY_LINE_MULTI", False)

    def f(tags):
        got = way_owners(cc, tags)
        if not got:
            return None
        return got if multi else got[0]
    return f


# ============================================================== conversion

def station_index(stops, cc=None):
    """rinf.osm_stations' stations, under every name tag's keys."""
    import rinf
    ost = rinf.osm_stations(stops, bool(cc and light_rail_track(cc)))
    by_key, by_short = defaultdict(list), defaultdict(list)
    for sid in ost:
        t = stops[sid][0]
        names = {t.get(k) for k in ("name", "name:fr", "name:en", "name:pt", "official_name",
                                    "alt_name", "old_name", "short_name") if t.get(k)}
        for n in list(names):
            names |= set(re.split(r"\s*[;/|]\s*", n))
        for n in names:
            if skey(n):
                by_key[skey(n)].append(sid)
            if skey_short(n):
                by_short[skey_short(n)].append(sid)
    for d in (by_key, by_short):
        for k in d:
            d[k] = sorted(set(d[k]))
    return ost, by_key, by_short


def parse(p):
    at = None
    m = re.match(r"^(.*)@([-\d.]+),([-\d.]+)$", p)
    if m:
        p, at = m.group(1), (float(m.group(2)), float(m.group(3)))
    if p.startswith("#"):
        return "border", p[1:], at or tuple(BORDERS[p[1:]][:2])
    if p.startswith("~"):
        return "junction", p[1:], at
    return "station", p, at


def own_sets(cc, ways):
    """{line id: set(way id)} of every line's own track (all OWN matches; this file's own
    traces are not limited to rinf.py's one name per way)."""
    out = defaultdict(set)
    for wid, (tags, _n) in ways.items():
        for lid in way_owners(cc, tags):
            out[lid].add(wid)
    return out


def convert(cc, log=print, write=True):
    t0 = time.time()
    import build_model as bm
    import rinf
    ways, rels, stops, cid, cx, cy = bm.load(cc, log)
    coords = bm.Coords(cid, cx, cy)
    track = rinf.Track(ways, coords, log, light_rail=bool(light_rail_track(cc)))
    ost, by_key, by_short = station_index(stops, cc)
    owns = own_sets(cc, ways)

    def cands(name):
        return by_key.get(skey(name)) or by_short.get(skey_short(name)) or []

    rows, pts_out, names = [], {}, {}
    stat = Counter()
    for l in LINES[cc]:
        pieces = [l["pts"]] + l["more"]
        traced_total, mine_total, k = 0.0, 0.0, 0
        own = owns.get(l["id"]) or None
        placed_line = {}
        for pi, seq in enumerate(pieces):
            placed = []
            for i, raw in enumerate(seq):
                kind, label, at = parse(raw)
                if kind in ("border", "junction"):
                    placed.append((kind, label, at[0], at[1], None))
                    continue
                cs = cands(label)
                hit = None
                if at:
                    near = [(dist_m(ost[c]["lon"], ost[c]["lat"], *at), c) for c in cs]
                    near = [x for x in near if x[0] <= NEAR_M]
                    if near:
                        hit = min(near)[1]
                    else:
                        blind = sorted((dist_m(s["lon"], s["lat"], *at), c)
                                       for c, s in ost.items()
                                       if abs(s["lon"] - at[0]) < 0.01 and abs(s["lat"] - at[1]) < 0.01)
                        if blind and blind[0][0] <= BLIND_M:
                            hit = blind[0][1]
                            log(f"    {l['id']}: {label} taken as OSM's {ost[hit]['name']!r} "
                                f"({blind[0][0]:.0f} m)")
                elif cs:
                    ref = (placed[-1][2:4] if placed else
                           placed_line.get(seq[i + 1]) if i + 1 < len(seq) else None)
                    if ref is None:
                        nxt = [x for x in seq[i + 1:i + 2]]
                        ncs = cands(parse(nxt[0])[1]) if nxt else []
                        if ncs:
                            hit = min(cs, key=lambda c: min(dist_m(ost[c]["lon"], ost[c]["lat"],
                                                                   ost[n]["lon"], ost[n]["lat"])
                                                            for n in ncs))
                        else:
                            hit = cs[0]
                    else:
                        d = sorted((dist_m(ost[c]["lon"], ost[c]["lat"], *ref), c) for c in cs)
                        if d[0][0] <= MAX_JUMP_KM * 1000:
                            hit = d[0][1]
                if hit is None:
                    stat["list points with no OSM station, left out"] += 1
                    log(f"    {l['id']}: no OSM station for {raw!r}")
                    continue
                s = ost[hit]
                placed.append(("station", label, s["lon"], s["lat"], hit))
                placed_line[raw] = (s["lon"], s["lat"])
            for a, b in zip(placed[:-1], placed[1:]):
                sa, sb = track.snap(a[2], a[3]), track.snap(b[2], b[3])
                crow = dist_m(a[2], a[3], b[2], b[3]) / 1000
                got = track.trace(sa, sb, crow * 2.5 + 5, own) if sa and sb else None
                if got:
                    km = got[1]
                    mine_total += got[4]
                    stat["sections traced"] += 1
                    if km > 2.0 * crow + 3:
                        log(f"    {l['id']}: {a[1]} - {b[1]} traced {km:.1f} km for "
                            f"{crow:.1f} km crow-fly: check the list")
                else:
                    km = crow * 1.2
                    stat["sections with no trace (crow-fly x 1.2)"] += 1
                    log(f"    {l['id']}: {a[1]} - {b[1]} NO TRACE ({crow:.1f} km crow-fly)")
                traced_total += km
                k += 1
                ops = []
                for q in (a, b):
                    if q[0] == "border":
                        op = f"{cc}:b:{q[1]}"
                        pts_out[op] = {"op": op, "uopid": q[1], "type": "90", "name": q[1],
                                       "lon": q[2], "lat": q[3]}
                    elif q[0] == "station":
                        op = f"{cc}:s:{q[4]}"
                        pts_out[op] = {"op": op, "uopid": f"{cc.upper()}{q[4]}", "type": "10",
                                       "name": ost[q[4]]["name"], "lon": q[2], "lat": q[3]}
                    else:
                        jk = re.sub(r"[^0-9a-z]", "", skey(q[1]))[:24] or f"{q[2]:.4f}{q[3]:.4f}"
                        op = f"{cc}:j:{jk}"
                        pts_out[op] = {"op": op, "uopid": f"{cc.upper()}J{jk}", "type": "80",
                                       "name": q[1], "lon": q[2], "lat": q[3]}
                    ops.append(op)
                rows.append({"sol": f"{cc}:{l['id']}:{k}", "line": l["id"], "a": ops[0],
                             "b": ops[1], "len": f"{km:.3f}", "im": l["im"],
                             "label": f"{a[1]} - {b[1]}"})
        names[l["id"]] = {"name": l["name"], "name_en": l["name_en"], "im": l["im"],
                          "suspended": bool(l.get("suspended")),
                          "listed_only": bool(l.get("listed_only")),
                          "traced_km": round(traced_total, 3)}
        log(f"  {l['id']:>10} {l['name'][:50]:<50} traced {traced_total:7.1f} km"
            + (f"  on own track {mine_total / traced_total:.0%}" if own and traced_total else "")
            + ("  [suspended]" if l.get("suspended") else ""))
    log(f"{cc.upper()}: {len(LINES[cc])} lines, {len(rows)} section rows, {len(pts_out)} points; "
        f"{dict(stat)}")
    if write:
        d = OUT / cc
        d.mkdir(parents=True, exist_ok=True)
        stamp = {"endpoint": "wafrica_register.py (hand-written line list)",
                 "fetched": date.today().isoformat()}
        (d / "sections.json").write_text(json.dumps({**stamp, "rows": rows}, ensure_ascii=False),
                                         "utf-8")
        (d / "points.json").write_text(json.dumps({**stamp, "rows": list(pts_out.values())},
                                                  ensure_ascii=False), "utf-8")
        (d / "names.json").write_text(json.dumps(names, ensure_ascii=False, indent=0), "utf-8")
        log(f"wrote {d} in {time.time() - t0:.0f} s")
    return rows, pts_out, names


def build(path, log):
    """build_model's register hook: convert, then rinf.py traces it over OSM track; each line
    then takes its operator, network, colour and kind from LINES.

    Every line here is written from a timetable, so a section of a running line that ends at
    a junction or a border point is one its trains run over: it goes in `served_sections`."""
    cc = Path(path).name
    convert(cc, log)
    import rinf
    lines, stations, geoms = rinf.build(path, log)
    names = _names(cc)
    by_id = {l["id"]: l for l in LINES[cc]}
    n = 0
    for ln in lines:
        lids = sorted({x.split("#")[0] for x in ln.get("rinf_ids", ())})
        info = by_id.get(lids[0]) if lids else None
        if info is None:
            log(f"{cc.upper()}: rinf.py line {ln['name']!r} is in no LINES entry")
        else:
            ln["operator"] = info["im"]
            if info.get("network"):
                ln["network"] = info["network"]
            if info.get("colour"):
                ln["colour"] = info["colour"]
            if info.get("kind") and info["kind"] != "rail":
                ln["kind"] = info["kind"]
        if ln.get("suspended") or any(names.get(x, {}).get("suspended") for x in lids):
            continue
        keys = [f"{a}|{b}" for a, b, *_ in ln["sections"]
                if stations.get(a, {}).get("junction") or stations.get(b, {}).get("junction")]
        if keys:
            ln["served_sections"] = keys
            n += len(keys)
    merge_twins(lines, stations, geoms, log)
    built = {x.split("#")[0] for ln in lines for x in ln.get("rinf_ids", ())}
    for l in LINES[cc]:
        if l["id"] not in built:
            log(f"{cc.upper()}: no line built for {l['id']} {l['name']}")
    log(f"{cc.upper()}: {n} junction- or border-ended sections of timetabled lines marked served")
    return lines, stations, geoms


TWIN_KM = 0.25


def merge_twins(lines, stations, geoms, log):
    """One station mapped twice ("Gare de Binguela" and "Binguela", "Tête d'éléphant" and
    "Tête d'Éléphant", a station and its stop positions) becomes two stops a few metres
    apart. A section shorter than TWIN_KM between two stops whose short keys (skey_short)
    are equal, or one contains the other, is one station: the one more lines use is kept,
    else the one whose name is longer (the "Gare de ..." record carries the station).
    za_register.merge_twins' rule, on this file's keys."""
    into = {}

    def root(s):
        while s in into:
            s = into[s]
        return s

    def similar(a, b):
        ka, kb = skey_short(a), skey_short(b)
        return bool(ka and kb) and (ka == kb or ka in kb or kb in ka)
    for ln in lines:
        for a, b, km, *_ in ln["sections"]:
            a, b = root(a), root(b)
            if a == b or km >= TWIN_KM or stations[a].get("junction") \
                    or stations[b].get("junction"):
                continue
            na, nb = stations[a]["name"], stations[b]["name"]
            if not similar(na, nb):
                continue
            keep, drop = a, b
            if (len(stations[b]["lines"]), len(nb)) > (len(stations[a]["lines"]), len(na)):
                keep, drop = b, a
            into[drop] = keep
            log(f"one station: {stations[drop]['name']} into {stations[keep]['name']}, "
                f"{km * 1000:.0f} m apart on {ln['name']}")
    if not into:
        return
    for ln in lines:
        g = geoms.get(ln["id"], {})
        ng, secs, nhs = {}, [], {}
        hs = ln.get("highspeed_sections") or {}
        for sec in ln["sections"]:
            a, b = root(sec[0]), root(sec[1])
            old = f"{sec[0]}|{sec[1]}"
            if a == b:
                continue
            secs.append([a, b, *sec[2:]])
            if old in g:
                ng[f"{a}|{b}"] = g[old]
            if old in hs:
                nhs[f"{a}|{b}"] = hs[old]
        ln["sections"] = secs
        geoms[ln["id"]] = ng
        if "highspeed_sections" in ln:
            ln["highspeed_sections"] = nhs
        for key in ("served_sections",):
            if key in ln:
                ln[key] = [f"{root(k.split('|')[0])}|{root(k.split('|')[1])}"
                           for k in ln[key] if root(k.split('|')[0]) != root(k.split('|')[1])]
        disp = []
        for s in ln["display"]:
            s = root(s)
            if not disp or disp[-1] != s:
                disp.append(s)
        ln["display"] = disp
        ln["km"] = round(sum(s[2] for s in secs), 3)
    for drop in into:
        keep = root(drop)
        stations[keep]["lines"] |= stations[drop]["lines"]
        del stations[drop]


# ============================================================== rinf.py settings

def _names(cc):
    f = OUT / cc / "names.json"
    return json.loads(f.read_text("utf-8")) if f.exists() else {}


def light_rail_track(cc):
    """A country whose lines include track OSM maps as railway=light_rail (Abuja's metro)."""
    return any(l.get("kind") == "light_rail" for l in LINES.get(cc, []))


def country_conf(cc):
    def id_name(lid, _uop=None):
        e = _names(cc).get(lid.split("#")[0])
        return (e["name"], e.get("name_en") or "") if e else None

    def suspended(_ref, lids):
        ns = _names(cc)
        return any(ns.get(x.split("#")[0], {}).get("suspended") for x in lids)

    def listed_only(lids):
        ns = _names(cc)
        return any(ns.get(x.split("#")[0], {}).get("listed_only") for x in lids)

    ims = {l["im"] for l in LINES.get(cc, [])}
    conf = {
        "osm_stops_skip": listed_only,
        "iso3": ISO3[cc], "langs": LANGS[cc],
        "ref": lambda _lid: None,
        "id_name": id_name,
        "suspended": suspended,
        "im": {x: x for x in ims},
        "osm_stops": "all",
        # OSM's route=railway relations here carry no line numbers to lend.
        "osm_rel": lambda _t: None,
        # the section lengths are our own traces, not a register's chainage
        "no_chain": True,
    }
    if light_rail_track(cc):
        conf["light_rail_track"] = True
    if way_owner(cc):
        conf["way_line"] = way_owner(cc)
    return conf


# ============================================================== clipping

def abroad(cc):
    """Every OTHER country's outline (nafrica_register.abroad)."""
    import shapely
    from shapely.geometry import shape
    from shapely.ops import unary_union
    gs = [shape(f["geometry"]) for f in json.loads(SHAPES.read_text("utf-8"))["features"]
          if f["properties"]["cc"] != cc]
    g = unary_union(gs)
    shapely.prepare(g)
    return g


def clip(cc, log=print):
    """Rewrite data/proc/<cc> without what lies abroad and without NOT_SERVICE routes. A way
    goes if at least half its nodes are outside, a stop if it is, a relation if no member is
    left. Master-less routes mapped one per direction get a route_master
    (nafrica_register.pair_directions)."""
    import numpy as np
    import shapely
    from nafrica_register import pair_directions
    d = ROOT / "data" / "proc" / cc
    rd = lambda f: pickle.load(open(d / f, "rb"))  # noqa: E731
    ways, rels, stops, infra = rd("ways.pkl"), rd("rels.pkl"), rd("stops.pkl"), rd("infra.pkl")
    with np.load(d / "coords.npz") as c:       # closed, so coords.npz can be replaced
        cid, cx, cy = c["id"].copy(), c["x"].copy(), c["y"].copy()
    g = abroad(cc)
    away = shapely.contains_xy(g, cx / 1e7, cy / 1e7)
    outside = set(cid[away].tolist())
    known = set(cid.tolist())
    keep_w, cut = {}, Counter()
    for wid, (tags, nodes) in ways.items():
        ns = [int(n) for n in nodes if int(n) in known]
        if ns and 2 * sum(n in outside for n in ns) >= len(ns):
            cut[tags.get("name") or tags.get("railway") or "?"] += 1
        else:
            keep_w[wid] = (tags, nodes)
    keep_s = {k: v for k, v in stops.items() if not shapely.contains_xy(g, v[1], v[2])}
    drop = set(NOT_SERVICE.get(cc, {}))
    for k in sorted(drop & set(rels)):
        log(f"  not a service: {k} {NOT_SERVICE[cc][k]}")
    pair_directions(rels, drop, log)
    kept = {("w", k) for k in keep_w} | {("n", k) for k in keep_s}
    routes = {k for k, (tags, members) in rels.items() if k not in drop
              and tags.get("type") == "route" and any((t, r) in kept for t, r, _ in members)}
    keep_r = {k: v for k, v in rels.items()
              if k in routes or (k not in drop and any(t == "r" and r in routes
                                                       for t, r, _ in v[1]))}
    keep_i = {k: v for k, v in infra.items()
              if any(t == "w" and r in keep_w for t, r, _ in v[1])}
    keep_w = {k: v for k, v in keep_w.items() if not (JOIN_BASE - 10_000_000 < k <= JOIN_BASE)}
    join_gaps(keep_w, cid, cx, cy, log)
    cid, cx, cy = split_gauges(keep_w, cid, cx, cy, log)
    tmp = d / "coords.tmp.npz"
    np.savez(tmp, id=cid, x=cx, y=cy)
    os.replace(tmp, d / "coords.npz")
    for name, v in cut.most_common(20):
        log(f"  cut {v:4d}  {name}")
    log(f"{cc.upper()} clip: kept {len(keep_w)}/{len(ways)} ways, {len(keep_s)}/{len(stops)} "
        f"stops, {len(keep_r)}/{len(rels)} relations, {len(keep_i)}/{len(infra)} infra relations")
    for fn, obj in (("ways.pkl", keep_w), ("rels.pkl", keep_r), ("stops.pkl", keep_s),
                    ("infra.pkl", keep_i)):
        tmp = d / (fn + ".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(obj, f, protocol=4)
        os.replace(tmp, d / fn)


JOIN_BASE = -8_200_000_000_000   # synthetic link way ids: this - k
JOIN_M = 60


def join_gaps(ways, cid, cx, cy, log):
    """Two dead ends of main track within JOIN_M of each other, on ways of one gauge, are one
    track OSM left unjoined (Nigeria's Lagos - Ibadan SGR north of Papalanto: a bridge's end
    and the next way's start 22 m apart). A two-node way (id JOIN_BASE - k, the first way's
    tags) joins them, so the trace runs through. Run as part of --clip."""
    pos = dict(zip(cid.tolist(), zip((cx / 1e7).tolist(), (cy / 1e7).tolist())))
    deg, end_of, inner = Counter(), {}, {}

    def heading(n, nb):
        (x1, y1), (x2, y2) = pos[nb], pos[n]
        dx, dy = (x2 - x1) * math.cos(math.radians(y1)), y2 - y1
        h = math.hypot(dx, dy) or 1.0
        return dx / h, dy / h
    for wid, (tags, nodes) in ways.items():
        if tags.get("railway") not in ("rail", "narrow_gauge", "light_rail"):
            continue
        ns = [int(n) for n in nodes]
        for a, b in zip(ns[:-1], ns[1:]):
            deg[a] += 1
            deg[b] += 1
        for n, nb in ((ns[0], ns[1]), (ns[-1], ns[-2])):
            end_of[n] = wid
            inner[n] = nb
    ends =[n for n, k in deg.items() if k == 1 and n in pos and n in end_of
            and ways[end_of[n]][0].get("service") not in ("yard", "siding", "spur")]
    grid = defaultdict(list)
    for n in ends:
        x, y = pos[n]
        grid[(int(x * 1000), int(y * 1000))].append(n)
    k, done = 0, set()
    for n in ends:
        if n in done:
            continue
        x, y = pos[n]
        best = None
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                for m in grid.get((int(x * 1000) + i, int(y * 1000) + j), ()):
                    if m == n or m in done or end_of[m] == end_of[n]:
                        continue
                    ta, tb = ways[end_of[n]][0], ways[end_of[m]][0]
                    if ta.get("gauge") and tb.get("gauge") and ta["gauge"] != tb["gauge"]:
                        continue
                    if ta.get("railway") != tb.get("railway"):
                        continue
                    dd = dist_m(x, y, *pos[m])
                    # one track continued: the two ways run on in the same direction (the
                    # buffer stops of parallel sidings face each other's way, not on)
                    if inner.get(n) not in pos or inner.get(m) not in pos:
                        continue
                    hu, hv = heading(n, inner[n]), heading(m, inner[m])
                    if -(hu[0] * hv[0] + hu[1] * hv[1]) < 0.8:
                        continue
                    if dd <= JOIN_M and (best is None or dd < best[0]):
                        best = (dd, m)
        if best:
            k += 1
            t = dict(ways[end_of[n]][0])
            t.pop("bridge", None)
            t.pop("tunnel", None)
            import numpy as np
            ways[JOIN_BASE - k] = (t, np.asarray([n, best[1]], dtype=np.int64))
            done |= {n, best[1]}
            log(f"  joined two dead ends {best[0]:.0f} m apart at {x:.5f},{y:.5f} "
                f"({t.get('name') or t.get('gauge') or t.get('railway')})")
    log(f"  {k} gaps in the track joined")


SPLIT_BASE = -8_300_000_000_000  # synthetic node ids: this - k
GAUGES_MEET = {frozenset({"1435", "1067"}), frozenset({"1000", "1067"}),
               frozenset({"1000", "1435"})}


def split_gauges(ways, cid, cx, cy, log):
    """Track of two gauges does not join: where OSM gives a 1067 way and a 1435 way one node
    (Lagos, where NRC's Cape gauge, the Lagos - Ibadan SGR and the Red Line run side by side:
    a trace from Mobolaji Johnson to Agege took 14 km of Cape gauge), the ways of every gauge
    but the first get a copy of the node (id SPLIT_BASE - k, same coordinates), so no trace
    can change gauge there. Ways with no gauge tag, or a dual gauge ("1067;1435"), stay joined
    to everything. Run as part of --clip; returns the new coordinate arrays."""
    import numpy as np
    at = defaultdict(set)
    for wid, (tags, nodes) in ways.items():
        g = tags.get("gauge")
        if not g or ";" in g or tags.get("railway") not in ("rail", "narrow_gauge", "light_rail",
                                                            "subway"):
            continue
        for n in nodes.tolist():
            at[int(n)].add(g)
    mixed = {n: gs for n, gs in at.items() if len(gs) > 1
             and any(frozenset(p) in GAUGES_MEET for p in
                     [(a, b) for a in gs for b in gs if a != b])}
    if not mixed:
        log("  0 nodes where two gauges meet")
        return cid, cx, cy
    k0 = int(np.sum(cid <= SPLIT_BASE))
    k = k0
    idx = {int(i): j for j, i in enumerate(cid.tolist()) if int(i) in mixed}
    clone = {}
    new_id, new_x, new_y = [], [], []
    for wid, (tags, nodes) in list(ways.items()):
        g = tags.get("gauge")
        if not g or ";" in g:
            continue
        ns = nodes.tolist()
        changed = False
        for i, n in enumerate(ns):
            gs = mixed.get(int(n))
            if not gs or g == sorted(gs)[0] or int(n) not in idx:
                continue
            key = (int(n), g)
            if key not in clone:
                k += 1
                clone[key] = SPLIT_BASE - k
                j = idx[int(n)]
                new_id.append(SPLIT_BASE - k)
                new_x.append(cx[j])
                new_y.append(cy[j])
            ns[i] = clone[key]
            changed = True
        if changed:
            ways[wid] = (tags, np.asarray(ns, dtype=np.int64))
    log(f"  {len(mixed)} nodes where two gauges meet, split into {k - k0} copies")
    cid = np.concatenate([cid, np.asarray(new_id, dtype=cid.dtype)])
    cx = np.concatenate([cx, np.asarray(new_x, dtype=cx.dtype)])
    cy = np.concatenate([cy, np.asarray(new_y, dtype=cy.dtype)])
    o = np.argsort(cid, kind="stable")
    return cid[o], cx[o], cy[o]


FILL_BASE = -8_400_000_000_000   # synthetic stop ids: this - k (nafrica -8.0e12, pk -8.1e12)


def fill(cc, log=print):
    """Stations the timetable names that OSM lacks or leaves unnamed (nafrica_register.fill):
    for every list point "Name@lon,lat" that finds no OSM station, an UNNAMED OSM station,
    halt or stop node within BLIND_M takes the name; with none, a halt is added at the
    coordinate (id FILL_BASE - k, tagged source=SOURCE_TAG). Rewrites
    data/proc/<cc>/stops.pkl; run after --clip."""
    d = ROOT / "data" / "proc" / cc
    stops = pickle.load(open(d / "stops.pkl", "rb"))
    # synthetic stops of an earlier run go first (any id this file made), so a rerun
    # rebuilds them from the list
    stops = {k: v for k, v in stops.items()
             if not (k < 0 and v[0].get("source") == SOURCE_TAG)}
    for k, (t, lon, lat) in stops.items():
        if t.get("source") == SOURCE_TAG and "name" in t:
            del t["name"]
            t.pop("source", None)
    ost, by_key, by_short = station_index(stops, cc)
    named, added, k = 0, 0, 0
    seen = set()
    for l in LINES[cc]:
        for seq in [l["pts"]] + l["more"]:
            for raw in seq:
                kind, label, at = parse(raw)
                if kind != "station" or not at or raw in seen:
                    continue
                seen.add(raw)
                cs = by_key.get(skey(label)) or by_short.get(skey_short(label)) or []
                if any(dist_m(ost[c]["lon"], ost[c]["lat"], *at) <= NEAR_M for c in cs):
                    continue
                if any(dist_m(s["lon"], s["lat"], *at) <= BLIND_M for s in ost.values()
                       if abs(s["lon"] - at[0]) < 0.01 and abs(s["lat"] - at[1]) < 0.01):
                    continue
                blank = sorted((dist_m(lon, lat, *at), nid) for nid, (t, lon, lat) in stops.items()
                               if not t.get("name") and abs(lon - at[0]) < 0.01
                               and abs(lat - at[1]) < 0.01
                               and (t.get("railway") in ("station", "halt", "stop")
                                    or t.get("public_transport") == "station"))
                if blank and blank[0][0] <= BLIND_M:
                    t = stops[blank[0][1]][0]
                    t["name"] = label
                    t["source"] = SOURCE_TAG
                    t["train"] = "yes"
                    named += 1
                    log(f"  {l['id']}: OSM's unnamed {t.get('railway') or 'station'} "
                        f"{blank[0][1]} ({blank[0][0]:.0f} m) named {label!r}")
                else:
                    k += 1
                    stops[FILL_BASE - k] = ({"railway": "halt", "name": label, "train": "yes",
                                             "source": SOURCE_TAG}, at[0], at[1])
                    added += 1
                    log(f"  {l['id']}: {label!r} added at the timetable's {at}")
    tmp = d / "stops.pkl.tmp"
    with open(tmp, "wb") as f:
        pickle.dump(stops, f, protocol=4)
    os.replace(tmp, d / "stops.pkl")
    log(f"{cc.upper()} fill: {named} unnamed OSM stations named, {added} stations added "
        f"from the timetable")


# ============================================================== tools for writing lists

def _track(cc):
    import build_model as bm
    import rinf
    ways, rels, stops, cid, cx, cy = bm.load(cc, lambda *_: None)
    coords = bm.Coords(cid, cx, cy)
    track = rinf.Track(ways, coords, lambda *_: None, light_rail=bool(light_rail_track(cc)))
    return ways, rels, stops, track


def trace_cmd(cc, names, own_line=None, log=print):
    """Trace station to station and print each leg's km and the stations it passes."""
    import rinf
    ways, rels, stops, track = _track(cc)
    ost, by_key, by_short = station_index(stops, cc)
    sidx = rinf.StationIndex(ost)
    own = own_sets(cc, ways).get(own_line) if own_line else None
    prev = None
    total = 0.0
    for raw in names:
        kind, label, at = parse(raw)
        if kind != "station":
            pt = at
            nm = label
        else:
            cs = by_key.get(skey(label)) or by_short.get(skey_short(label)) or []
            if at:
                cs = sorted(cs, key=lambda c: dist_m(ost[c]["lon"], ost[c]["lat"], *at))
            elif prev:
                cs = sorted(cs, key=lambda c: dist_m(ost[c]["lon"], ost[c]["lat"], *prev))
            if at and (not cs or dist_m(ost[cs[0]]["lon"], ost[cs[0]]["lat"], *at) > NEAR_M):
                near = sorted(sidx.within(*at, BLIND_M))
                cs = [near[0][1]] if near else []
                if not cs:
                    pt, nm = at, f"({label})"
            if not cs and not at:
                print(f"  ?? no station {label!r}")
                continue
            if cs:
                s = ost[cs[0]]
                pt, nm = (s["lon"], s["lat"]), s["name"]
        if prev:
            got = track.trace(track.snap(*prev), track.snap(*pt),
                              dist_m(*prev, *pt) / 1000 * 2.5 + 5, own)
            if got is None:
                print(f"  -> {nm}: NO PATH")
            else:
                pts = got[0]
                passed = []
                for x, y in pts[::3]:
                    for d, sid in sidx.within(x, y, 120):
                        if ost[sid]["name"] not in passed:
                            passed.append(ost[sid]["name"])
                total += got[1]
                g = Counter()
                for e in got[3]:
                    t = track.way_tags[int(track.ew[e])]
                    g[t.get("gauge", "?")] += float(track.elen[e]) / 1000
                print(f"  -> {nm}: {got[1]:.1f} km (total {total:.1f}); gauge "
                      f"{', '.join(f'{k} {v:.1f}' for k, v in g.most_common())}; "
                      f"via {' / '.join(passed)}")
        else:
            print(f"  {nm} {pt}")
        prev = pt


def fork_cmd(cc, a, b, c):
    """Where the trace a -> c leaves the trace a -> b: the last shared point, for a "~" junction."""
    ways, rels, stops, track = _track(cc)
    ost, by_key, by_short = station_index(stops, cc)

    def pt(n):
        kind, label, at = parse(n)
        if kind != "station" or not (by_key.get(skey(label)) or by_short.get(skey_short(label))):
            return at
        cs = by_key.get(skey(label)) or by_short.get(skey_short(label))
        if at:
            cs = sorted(cs, key=lambda x: dist_m(ost[x]["lon"], ost[x]["lat"], *at))
        return ost[cs[0]]["lon"], ost[cs[0]]["lat"]
    pa, pb, pc = pt(a), pt(b), pt(c)
    t1 = track.trace(track.snap(*pa), track.snap(*pb), dist_m(*pa, *pb) / 400 + 5)
    t2 = track.trace(track.snap(*pa), track.snap(*pc), dist_m(*pa, *pc) / 400 + 5)
    if not t1 or not t2:
        print("no trace")
        return
    from shapely.geometry import LineString, Point
    k = math.cos(math.radians(pa[1]))
    g1 = LineString([(x * k * 111320, y * 110570) for x, y in t1[0]])
    last, km, cum = None, 0.0, 0.0
    pts = t2[0]
    for i, q in enumerate(pts):
        if i:
            cum += dist_m(*pts[i - 1], *q) / 1000
        if g1.distance(Point(q[0] * k * 111320, q[1] * 110570)) <= 40:
            last, km = q, cum
    print(f"leaves at {last[0]:.5f},{last[1]:.5f} ({km:.1f} km from {a})" if last else "no shared track")


def gauges_cmd(cc):
    """Track km by railway, gauge and name, and how much of it each OWN rule claims."""
    ways, rels, stops, track = _track(cc)
    km = Counter()
    for e in range(len(track.ew)):
        t = track.way_tags[int(track.ew[e])]
        km[(t.get("railway"), t.get("gauge", "?"), t.get("name", ""))] += float(track.elen[e]) / 1000
    for (rw, g, n), v in km.most_common(60):
        print(f"  {v:8.1f} km  {rw:<12} {g:<6} {n}")
    owns = own_sets(cc, ways)
    for lid, ws in owns.items():
        print(f"  OWN {lid}: {len(ws)} ways")
    ost, _bk, _bs = station_index(stops, cc)
    print(f"  {len(ost)} OSM rail stations")


def split_pieces(lines, stations, geoms, reg_ways, state, log):
    """build_model's hook, after drop_unridden_sections: rinf.split_pieces."""
    import rinf
    rinf.split_pieces(lines, stations, geoms, reg_ways, state, log)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--convert", metavar="CC")
    ap.add_argument("--dry", metavar="CC", help="convert without writing")
    ap.add_argument("--clip", metavar="CC")
    ap.add_argument("--trace", nargs="+", metavar="ARG")
    ap.add_argument("--own", metavar="LINE", help="with --trace: prefer this line's OWN track")
    ap.add_argument("--fork", nargs=4, metavar=("CC", "A", "B", "C"))
    ap.add_argument("--fill", metavar="CC")
    ap.add_argument("--gauges", metavar="CC")
    a = ap.parse_args()
    t = time.time()
    lg = lambda m: print(f"[{time.time() - t:6.1f}s] {m}", flush=True)  # noqa: E731
    if a.clip:
        clip(a.clip, lg)
    if a.fill:
        fill(a.fill, print)
    if a.fork:
        fork_cmd(*a.fork)
    if a.gauges:
        gauges_cmd(a.gauges)
    if a.convert:
        convert(a.convert, lg)
    if a.dry:
        convert(a.dry, lg, write=False)
    if a.trace:
        trace_cmd(a.trace[0], a.trace[1:], a.own)
