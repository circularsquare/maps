"""Morocco, Algeria, Tunisia and Egypt: each country's passenger line list written into
rinf.py's input format (balkans_register.py's recipe), so rinf.py traces it over OSM track,
places the stops and names the lines.

    python nafrica_register.py --clip ma          # after every extract of ma (see CLIPPING)
    python build_model.py --region ma --register nafrica_register:data/raw/rinf/ma
    python nafrica_register.py --convert ma       # the conversion alone, with its log
    python nafrica_register.py --trace ma "Casa Voyageurs" "Kénitra"   # a trace, to write lists
    (likewise dz, tn, eg)

`--register nafrica_register:data/raw/rinf/<cc>` converts and then runs rinf.build on the
result. The settings rinf.py reads are in rinf_countries/<cc>.py (each `country_conf` below).
Sources, what runs and the numbers: nafrica_sources.md.

WHY A LIST.  None of the four is in ERA RINF, and OSM names almost none of their track (Morocco:
the LGV alone; Egypt: the Suez line; probe in nafrica_sources.md), so kr_register's named-track
recipe has nothing to read. No operator publishes a line register with chainage. So each line
is written here by hand: its stations in order (ends, junction stations and enough stations
between to keep the trace on the right track), from the operator's timetable and OSM's track.
km are traced over OSM's track, so the published lengths in check_model.REGISTER are the only
outside check (`no_chain`: no km_official is shipped, the lengths are ours).

POINTS are written as
    "Name"                 an OSM rail station of that name (`skey`: case, accents, "Gare de",
                           Arabic letters folded away), the one nearest the line's last point
    "Name@lon,lat"         that station near the coordinate (within NEAR_M); with no station of
                           the name there, the nearest OSM rail station within BLIND_M
    "~Name@lon,lat"        a junction, no stop, at the coordinate (where a line leaves another
                           on the open line)
    "#ID"                  a border point (BORDERS)
A line may have `more` pieces (branches) under the same id. rinf.py's `osm_stops: "all"` puts
every OSM rail station lying on a traced section on the line as a stop, so a list need not name
every halt.

STATIONS keep OSM's own name (which is what is written at the station: French and Arabic in the
Maghreb, Arabic in Egypt) and its name:en.

CLIPPING.  Geofabrik's extracts reach over the borders (Morocco's takes in the Mauritania
Railway and ferry terminals in Spain and France; Tunisia's and Algeria's each other's track
near Ghardimaou). `--clip <cc>` drops what lies inside ANOTHER country's outline (religiondots'
country_shapes.geojson, the outlines the app uses; not "outside this one", which cut the TGM's
causeway over the Lake of Tunis) and the route relations in NOT_SERVICE.
"""
import argparse
import hashlib
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

# ============================================================== names and keys

STATION_WORDS = re.compile(r"\b(?:gare|de|du|des|d|la|le|les|l|station|halte|arret|ville|"
                           r"railway|train|el|al|ech|es|et|en|ed|er|ez)\b")


def skey(s):
    """One key for a station name: "Gare Rabat Ville محطة الرباط المدينة" and "Rabat-Ville"
    are "rabatville"; "Fès-Médina" "fesmedina". Arabic letters fold away (rinf.norm keeps only
    [0-9a-z]); "ville" and the article are kept in the full key and dropped in the short one."""
    s = unicodedata.normalize("NFKD", (s or "").casefold())
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[’'`]", " ", s)
    return re.sub(r"[^0-9a-z]", "", re.sub(r"\b(?:gare|station|halte)\b", " ", s))


def skey_short(s):
    s = unicodedata.normalize("NFKD", (s or "").casefold())
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[’'`\-]", " ", s)
    return re.sub(r"[^0-9a-z]", "", STATION_WORDS.sub(" ", s))


def dist_m(lon1, lat1, lon2, lat2):
    dx = (lon2 - lon1) * math.cos(math.radians((lat1 + lat2) / 2)) * 111320
    return math.hypot(dx, (lat2 - lat1) * 110570)


# ============================================================== the line lists
# In nafrica_lines.py: LINES (per country), BORDERS (id -> lon, lat, countries) and
# NOT_SERVICE (route relations dropped by --clip).

LANGS = {"ma": ["fr", "ar", "en"], "dz": ["fr", "ar", "en"], "tn": ["fr", "ar", "en"],
         "eg": ["ar", "en"]}
ISO3 = {"ma": "MAR", "dz": "DZA", "tn": "TUN", "eg": "EGY"}

from nafrica_lines import BORDERS, LINES, NOT_SERVICE  # noqa: E402


# ============================================================== conversion

def station_index(stops):
    """rinf.osm_stations' stations, under every name tag's keys."""
    import rinf
    ost = rinf.osm_stations(stops)
    by_key, by_short = defaultdict(list), defaultdict(list)
    for sid in ost:
        t = stops[sid][0]
        names = {t.get(k) for k in ("name", "name:fr", "name:en", "official_name", "alt_name",
                                    "old_name", "name:ber-Latn") if t.get(k)}
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


def convert(cc, log=print, write=True):
    t0 = time.time()
    import build_model as bm
    import rinf
    ways, rels, stops, cid, cx, cy = bm.load(cc, log)
    coords = bm.Coords(cid, cx, cy)
    track = rinf.Track(ways, coords, log)
    ost, by_key, by_short = station_index(stops)

    def cands(name):
        return by_key.get(skey(name)) or by_short.get(skey_short(name)) or []

    rows, pts_out, names = [], {}, {}
    stat = Counter()
    for l in LINES[cc]:
        pieces = [l["pts"]] + l["more"]
        traced_total, k = 0.0, 0
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
                        # the first point of a piece: nearest the next named point's candidates
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
                got = track.trace(sa, sb, crow * 2.5 + 5) if sa and sb else None
                if got:
                    km = got[1]
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
            + ("  [suspended]" if l.get("suspended") else ""))
    log(f"{cc.upper()}: {len(LINES[cc])} lines, {len(rows)} section rows, {len(pts_out)} points; "
        f"{dict(stat)}")
    if write:
        d = OUT / cc
        d.mkdir(parents=True, exist_ok=True)
        stamp = {"endpoint": "nafrica_register.py (hand-written line list)",
                 "fetched": date.today().isoformat()}
        (d / "sections.json").write_text(json.dumps({**stamp, "rows": rows}, ensure_ascii=False),
                                         "utf-8")
        (d / "points.json").write_text(json.dumps({**stamp, "rows": list(pts_out.values())},
                                                  ensure_ascii=False), "utf-8")
        (d / "names.json").write_text(json.dumps(names, ensure_ascii=False, indent=0), "utf-8")
        log(f"wrote {d} in {time.time() - t0:.0f} s")
    return rows, pts_out, names


def build(path, log):
    """build_model's register hook: convert, then rinf.py traces it over OSM track.

    Every line here is written from a timetable, so a section of a running line that ends at
    a junction or a border point is one its trains run over: it goes in `served_sections`
    (build_model.drop_unridden_sections keeps it without an OSM route; the Annaba - Tunis
    train to the Tunisian border has none)."""
    cc = Path(path).name
    convert(cc, log)
    import rinf
    lines, stations, geoms = rinf.build(path, log)
    names = _names(cc)
    n = 0
    for l in lines:
        if l.get("suspended") or any(names.get(x.split("#")[0], {}).get("suspended")
                                     for x in l.get("rinf_ids", ())):
            continue
        keys = [f"{a}|{b}" for a, b, *_ in l["sections"]
                if stations.get(a, {}).get("junction") or stations.get(b, {}).get("junction")]
        if keys:
            l["served_sections"] = keys
            n += len(keys)
    log(f"{cc.upper()}: {n} junction- or border-ended sections of timetabled lines marked served")
    return lines, stations, geoms


# ============================================================== rinf.py settings

def _names(cc):
    f = OUT / cc / "names.json"
    return json.loads(f.read_text("utf-8")) if f.exists() else {}


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
    return {
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


# ============================================================== clipping

def abroad(cc):
    """Every OTHER country's outline. "Outside" is inside another country, not outside this
    one's: the outline leaves out lakes and is simplified at the coast, and clipping to it cut
    the TGM's causeway over the Lake of Tunis and Tanger Med's quay station."""
    import shapely
    from shapely.geometry import shape
    from shapely.ops import unary_union
    gs = [shape(f["geometry"]) for f in json.loads(SHAPES.read_text("utf-8"))["features"]
          if f["properties"]["cc"] != cc]
    g = unary_union(gs)
    shapely.prepare(g)
    return g


MASTER_BASE = 9_100_000_000   # synthetic route_master ids: this + the lower route id
ARROWS = re.compile(r"[→←↑↓↔⇄<>]+|︎|\b(?:aller|retour)\b", re.I)


def pair_directions(rels, drop, log):
    """One line for the two directions of a route mapped without a route_master (Algeria's
    trams "Tramway de Constantine ↑" / "↓", "Tramway d'Oran Aller" / "Retour", the SNTF's "SBA
    Railroad Line → Medrissa Railroad Line" and back): build_model groups master-less routes
    by name, so each direction became a line. Routes of one mode whose names are the same once
    arrows and Aller/Retour are taken out, the parts either side of an arrow read in either
    order, get a route_master (id MASTER_BASE + the lowest route id, stable)."""
    in_master = {r for t, ms in rels.values() if t.get("type") == "route_master"
                 for ty, r, _ in ms if ty == "r"}

    def key(t):
        n = t.get("name") or ""
        parts = [p.strip().casefold() for p in re.split(r"[→←⇄]|<>|->|=>", n) if p.strip()]
        if len(parts) == 2:
            parts = sorted(parts)
        base = " | ".join(ARROWS.sub(" ", p) for p in parts)
        return (t.get("route"), " ".join(base.split()))
    groups = defaultdict(list)
    for k, (t, ms) in rels.items():
        if t.get("type") == "route" and k not in in_master and k not in drop and t.get("name"):
            groups[key(t)].append(k)
    made = 0
    for (mode, _n), ks in groups.items():
        if len(ks) < 2:
            continue
        ks = sorted(ks)
        t0 = rels[ks[0]][0]
        name = " ".join(ARROWS.sub(" ", t0.get("name", "")).split()).strip(" -:,")
        rels[MASTER_BASE + ks[0]] = ({"type": "route_master", "route_master": mode, "name": name,
                                       "ref": t0.get("ref", ""), "operator": t0.get("operator", ""),
                                       "network": t0.get("network", ""), "colour": t0.get("colour", "")},
                                      [("r", k, "") for k in ks])
        made += 1
    log(f"  {made} routes mapped one relation per direction given a route_master")


def clip(cc, log=print):
    """Rewrite data/proc/<cc> without what lies abroad and without NOT_SERVICE routes. A way
    goes if at least half its nodes are outside, a stop if it is, a relation if no member is
    left."""
    import numpy as np
    import shapely
    d = ROOT / "data" / "proc" / cc
    rd = lambda f: pickle.load(open(d / f, "rb"))  # noqa: E731
    ways, rels, stops, infra = rd("ways.pkl"), rd("rels.pkl"), rd("stops.pkl"), rd("infra.pkl")
    c = np.load(d / "coords.npz")
    cid, cx, cy = c["id"], c["x"], c["y"]
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


FILL_BASE = -8_000_000_000_000   # synthetic stop ids: this - k


def fill(cc, log=print):
    """Stations the timetable names that OSM lacks or leaves unnamed. For every list point
    "Name@lon,lat" (a timetable stop at the timetable's own coordinate) that finds no OSM
    station as convert() looks for one: an UNNAMED OSM station, halt or stop node within
    BLIND_M takes the name (Tunisia's Benbachir and Aïn Ghelal are mapped, nameless); with
    none, a halt is added at the coordinate (id FILL_BASE - k, tagged
    `source=nafrica_register timetable`). Rewrites data/proc/<cc>/stops.pkl; run after --clip.
    Points without a coordinate are left alone: their name alone says nothing of where."""
    d = ROOT / "data" / "proc" / cc
    stops = pickle.load(open(d / "stops.pkl", "rb"))
    # synthetic stops from an earlier run go first, so a rerun rebuilds them from the list
    stops = {k: v for k, v in stops.items() if not (k <= FILL_BASE)}
    for k, (t, lon, lat) in stops.items():
        if t.get("source") == "nafrica_register timetable" and "name" in t:
            del t["name"]
            t.pop("source", None)
    ost, by_key, by_short = station_index(stops)
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
                    t["source"] = "nafrica_register timetable"
                    t["train"] = "yes"
                    named += 1
                    log(f"  {l['id']}: OSM's unnamed {t.get('railway') or 'station'} "
                        f"{blank[0][1]} ({blank[0][0]:.0f} m) named {label!r}")
                else:
                    k += 1
                    stops[FILL_BASE - k] = ({"railway": "halt", "name": label, "train": "yes",
                                             "source": "nafrica_register timetable"},
                                            at[0], at[1])
                    added += 1
                    log(f"  {l['id']}: {label!r} added at the timetable's {at}")
    tmp = d / "stops.pkl.tmp"
    with open(tmp, "wb") as f:
        pickle.dump(stops, f, protocol=4)
    os.replace(tmp, d / "stops.pkl")
    log(f"{cc.upper()} fill: {named} unnamed OSM stations named, {added} stations added "
        f"from the timetable")


# ============================================================== a trace, for writing lists

def trace_cmd(cc, names, log=print):
    """Trace station to station and print each leg's km and the stations it passes."""
    import build_model as bm
    import rinf
    ways, rels, stops, cid, cx, cy = bm.load(cc, lambda *_: None)
    coords = bm.Coords(cid, cx, cy)
    track = rinf.Track(ways, coords, lambda *_: None)
    ost, by_key, by_short = station_index(stops)
    sidx = rinf.StationIndex(ost)
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
                              dist_m(*prev, *pt) / 1000 * 2.5 + 5)
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
                print(f"  -> {nm}: {got[1]:.1f} km (total {total:.1f}); via {' / '.join(passed)}")
        else:
            print(f"  {nm} {pt}")
        prev = pt


# ============================================================== Egypt's timetable

EG_RAW = ROOT / "data" / "raw" / "eg"
ET_HOST = "https://egypttrains.com"
ET_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) noritetsu-build/1.0 (hobby rail map)"


def eg_train_list(idx_html, log=print):
    """data/raw/eg/enr_train_list.json from the site's one-page train list: every train's
    number, coach type, number of stops, ENR's km, first and last station, Active/Stopped.
    This is what the line list was checked against (the per-train pages, with their stops,
    were rate-limited, 429, after nine pages on 2026-10-03)."""
    import html
    t = re.sub(r"<script.*?</script>|<style.*?</style>", "", idx_html, flags=re.S)
    ls = [x.strip() for x in html.unescape(re.sub(r"<[^>]+>", "\n", t)).splitlines() if x.strip()]
    out = {}
    for i, x in enumerate(ls[:-7]):
        m = re.fullmatch(r"Train (\S+)", x)
        if not m:
            continue
        ty, st, km, a, b, status = ls[i + 1:i + 7]
        sm, km_m = re.fullmatch(r"(\d+) stops", st), re.fullmatch(r"([\d,]+) km", km)
        if sm and km_m:
            out[m.group(1)] = {"train": m.group(1), "type": ty, "stops": int(sm.group(1)),
                               "km": int(km_m.group(1).replace(",", "")), "from": a, "to": b,
                               "status": status}
    (EG_RAW / "enr_train_list.json").write_text(json.dumps(list(out.values()), ensure_ascii=False,
                                                           indent=0), "utf-8")
    log(f"EG: {len(out)} trains in the list "
        f"({sum(1 for r in out.values() if r['status'] == 'Active')} active)")


def crawl_eg(log=print):
    """ENR's timetable as egypttrains.com republishes it (an unofficial site built from ENR's
    published schedules; ENR's own site answers only station-pair searches): its train list,
    then each train's page, into data/raw/eg/egypttrains/ (one request a second, ~15 min),
    then data/raw/eg/enr_trains.json: [{"train", "type", "stops": [name, ...], "km"}]."""
    import html
    import urllib.request
    d = EG_RAW / "egypttrains"
    d.mkdir(parents=True, exist_ok=True)

    def get(path, f):
        if f.exists() and f.stat().st_size > 20000:
            return f.read_text("utf-8")
        # curl, not urllib: the site's front end answers Python's client 403, and Windows'
        # own curl.exe too; Git's (OpenSSL) curl is let through. It rate-limits (429) at
        # about one request a second: 4 s apart, and a 429 waits two minutes.
        import subprocess
        curl = os.environ.get("NAFRICA_CURL", r"C:\Program Files\Git\mingw64\bin\curl.exe")
        tmp = f.with_suffix(".part")
        for attempt in range(6):
            r = subprocess.run([curl, "-s", "-m", "60", "-A", ET_UA, "-o", str(tmp),
                                "-w", "%{http_code}", ET_HOST + path],
                               capture_output=True, text=True)
            time.sleep(4.0)
            if r.stdout.strip() == "200":
                os.replace(tmp, f)
                return f.read_text("utf-8")
            if r.stdout.strip() == "429":
                time.sleep(120)
                continue
            break
        raise RuntimeError(f"HTTP {r.stdout.strip()}")
    idx = get("/trains?lang=en", d / "_trains.html")
    eg_train_list(idx, log)
    nums = sorted(set(re.findall(r'href="/trains/([^"?]+)\?lang=en"', idx)))
    log(f"EG: {len(nums)} trains listed")
    if "--list-only" in sys.argv:
        return
    out = []
    for i, n in enumerate(nums):
        try:
            t = get(f"/trains/{n}?lang=en", d / f"{n}.html")
        except Exception as e:                      # noqa: BLE001
            log(f"  {n}: {e}")
            continue
        txt = html.unescape(re.sub(r"<[^>]+>", "\n", re.sub(r"<script.*?</script>|<style.*?</style>",
                                                             "", t, flags=re.S)))
        lines = [x.strip() for x in txt.splitlines() if x.strip()]
        m = re.search(r"stops at \d+ stations on the .*? route:\n(.*?)\nSee the", "\n".join(lines),
                      re.S)
        stops = [s.strip() for s in m.group(1).split(" — ")] if m else []
        km = re.search(r"covering ([\d,.]+) kilometers", " ".join(lines))
        ty = re.search(r"is a\n(.*?)\nservice", "\n".join(lines))
        st = "Active" if "\nActive\n" in "\n".join(lines) else "?"
        out.append({"train": n, "type": ty.group(1) if ty else "", "status": st, "stops": stops,
                    "km": float(km.group(1).replace(",", "")) if km else None})
        if i % 100 == 0:
            log(f"  {i}/{len(nums)}")
    (EG_RAW / "enr_trains.json").write_text(json.dumps(out, ensure_ascii=False, indent=0), "utf-8")
    log(f"EG: wrote {len(out)} trains to {EG_RAW / 'enr_trains.json'}")


def fork_cmd(cc, a, b, c):
    """Where the trace a -> c leaves the trace a -> b: the last shared point, for a "~" junction."""
    import build_model as bm
    import rinf
    ways, rels, stops, cid, cx, cy = bm.load(cc, lambda *_: None)
    coords = bm.Coords(cid, cx, cy)
    track = rinf.Track(ways, coords, lambda *_: None)
    ost, by_key, by_short = station_index(stops)

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


def split_pieces(lines, stations, geoms, reg_ways, state, log):
    """build_model's hook, after drop_unridden_sections: rinf.split_pieces (folds a stop's 0 km
    link junctions back into it, and bridges where the country's settings ask)."""
    import rinf
    rinf.split_pieces(lines, stations, geoms, reg_ways, state, log)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--convert", metavar="CC")
    ap.add_argument("--dry", metavar="CC", help="convert without writing")
    ap.add_argument("--clip", metavar="CC")
    ap.add_argument("--trace", nargs="+", metavar="ARG")
    ap.add_argument("--fork", nargs=4, metavar=("CC", "A", "B", "C"))
    ap.add_argument("--crawl-eg", action="store_true")
    ap.add_argument("--list-only", action="store_true", help="with --crawl-eg: the list page alone")
    ap.add_argument("--fill", metavar="CC")
    a = ap.parse_args()
    if a.fill:
        fill(a.fill, print)
    if a.crawl_eg:
        crawl_eg(print)
    if a.fork:
        fork_cmd(*a.fork)
    t = time.time()
    lg = lambda m: print(f"[{time.time() - t:6.1f}s] {m}", flush=True)  # noqa: E731
    if a.clip:
        clip(a.clip, lg)
    if a.convert:
        convert(a.convert, lg)
    if a.dry:
        convert(a.dry, lg, write=False)
    if a.trace:
        trace_cmd(a.trace[0], a.trace[1:])
