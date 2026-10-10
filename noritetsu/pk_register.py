"""Pakistan: a hand-written list of the passenger lines (pk_lines.py), written into rinf.py's
input format (nafrica_register.py's recipe, with Indonesia's twist of km posts), so rinf.py
traces it over OSM track, places the stops and names the lines.

    python extract.py --region pk --pbf data/raw/pakistan-latest.osm.pbf --station-areas
    python pk_register.py --fill         # after every extract: list stations OSM lacks, from
                                         # Wikidata, into data/proc/pk/stops.pkl
    python pk_register.py --convert      # data/raw/pk/{sections,points,names}.json, with a log
    python build_model.py --region pk --register pk_register:data/raw/pk
    python pk_register.py --trace "Lahore Junction" "Wagah"    # a trace, to write lists

`--register pk_register:data/raw/pk` converts and then runs rinf.build on the result. The
settings rinf.py reads are in rinf_countries/pk.py (`country_conf` below). Sources, what runs
and the numbers: pk_sources.md.

WHY A LIST.  Pakistan Railways publishes no line register (its Year Book has route totals only),
and OSM names about a third of the main-line track for its line, mostly with generic names, so
kr_register's named-track recipe has nothing to read. en.wikipedia's route diagrams (RDT) give
PR's km posts for ML-1, ML-3, ML-4 and three branches: those lines carry their posts here as
`chain`, which check_model compares every line against (`chain: True` in pk_lines.py). The
other lines are lists of stations in order, measured on OSM's track.

STOPS.  rinf.py's `osm_stops: "all"`: every OSM rail station lying on a traced section is a stop
on its line, so a list need not name every halt.
"""
import argparse
import json
import math
import os
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
CC = "pk"
OUT = ROOT / "data" / "raw" / "pk"

NEAR_M = 3000      # "Name@lon,lat": a station of the name this close to the coordinate
BLIND_M = 400      # ... else any OSM rail station this close
MAX_JUMP_KM = 250  # "Name": a station of the name at most this far from the line's last point
# A section's km posts are believed when its trace agrees this well (rinf.py's own tolerance,
# 15% + 0.3 km, is what a disagreeing section then meets: logged here first).
KM_TOL = (0.15, 1.0)

from pk_lines import BORDERS, LINES  # noqa: E402


# ============================================================== names and keys

def skey(s):
    """One key for a station name: "Sangla Hill Junction Railway Station" and "Sangla Hill
    Junction" are "sanglahilljunction"; "Kharian Cantt." and "Kharian Cantonment" the same."""
    s = unicodedata.normalize("NFKD", (s or "").casefold())
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[’'`.]", " ", s)
    s = re.sub(r"\bcantonment\b", "cantt", s)
    s = re.sub(r"\b(?:railway\s+station|train\s+station|station|rs)\b", " ", s)
    return re.sub(r"[^0-9a-z]", "", s)


def skey_short(s):
    """Shorter still: "Junction", "Jn", "Halt", "City" off ("Kundian" for "Kundian Jn")."""
    s = unicodedata.normalize("NFKD", (s or "").casefold())
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[’'`.\-]", " ", s)
    s = re.sub(r"\bcantonment\b", "cantt", s)
    s = re.sub(r"\b(?:railway|train|station|junction|jn|jct|halt|rs)\b", " ", s)
    return re.sub(r"[^0-9a-z]", "", s)


def dist_m(lon1, lat1, lon2, lat2):
    dx = (lon2 - lon1) * math.cos(math.radians((lat1 + lat2) / 2)) * 111320
    return math.hypot(dx, (lat2 - lat1) * 110570)


def station_index(stops):
    import rinf
    ost = rinf.osm_stations(stops)
    by_key, by_short = defaultdict(set), defaultdict(set)
    for sid in ost:
        t = stops[sid][0]
        names = {t.get(k) for k in ("name", "name:en", "official_name", "official_name:en",
                                    "alt_name", "alt_name:en", "old_name", "short_name",
                                    "name:ur") if t.get(k)}
        for n in list(names):
            names |= set(re.split(r"\s*[;/|]\s*", n))
        for n in names:
            if skey(n):
                by_key[skey(n)].add(sid)
            if skey_short(n):
                by_short[skey_short(n)].add(sid)
    return ost, by_key, by_short


def parse(p):
    km = None
    if "|" in p:
        p, k = p.rsplit("|", 1)
        km = float(k.replace(",", ""))
    at = None
    m = re.match(r"^(.*)@([-\d.]+),([-\d.]+)$", p)
    if m:
        p, at = m.group(1), (float(m.group(2)), float(m.group(3)))
    if p.startswith("#"):
        return "border", p[1:], at or tuple(BORDERS[p[1:]][:2]), km
    if p.startswith("~"):
        return "junction", p[1:], at, km
    return "station", p, at, km


# ============================================================== conversion

def convert(log=print, write=True):
    t0 = time.time()
    import build_model as bm
    import rinf
    ways, rels, stops, cid, cx, cy = bm.load(CC, log)
    coords = bm.Coords(cid, cx, cy)
    track = rinf.Track(ways, coords, log)
    ost, by_key, by_short = station_index(stops)

    def cands(name):
        return sorted(by_key.get(skey(name)) or by_short.get(skey_short(name)) or [])

    rows, pts_out, names = [], {}, {}
    stat = Counter()
    chosen = defaultdict(list)
    for l in LINES:
        traced_total, chain_total, k = 0.0, 0.0, 0
        for pi, seq in enumerate([l["pts"]] + l["more"]):
            placed = []
            for i, raw in enumerate(seq):
                kind, label, at, km = parse(raw)
                if kind in ("border", "junction"):
                    placed.append((kind, label, at[0], at[1], None, km))
                    continue
                cs = cands(label)
                hit = None
                if at:
                    near = sorted((dist_m(ost[c]["lon"], ost[c]["lat"], *at), c) for c in cs)
                    near = [x for x in near if x[0] <= NEAR_M]
                    if near:
                        hit = near[0][1]
                    else:
                        blind = sorted((dist_m(s["lon"], s["lat"], *at), c)
                                       for c, s in ost.items()
                                       if abs(s["lon"] - at[0]) < 0.01 and abs(s["lat"] - at[1]) < 0.01)
                        if blind and blind[0][0] <= BLIND_M:
                            hit = blind[0][1]
                            log(f"    {l['id']}: {label} taken as OSM's {ost[hit]['name']!r} "
                                f"({blind[0][0]:.0f} m)")
                elif cs:
                    ref = placed[-1][2:4] if placed else None
                    if ref is None:
                        # a line's first point: the candidate nearest the next point's
                        nxt = None
                        for raw2 in seq[i + 1:i + 4]:
                            k2, lab2, at2, _ = parse(raw2)
                            if at2:
                                nxt = [at2]
                            elif k2 == "station" and cands(lab2):
                                nxt = [(ost[c]["lon"], ost[c]["lat"]) for c in cands(lab2)]
                            if nxt:
                                break
                        if nxt:
                            hit = min(cs, key=lambda c: min(dist_m(ost[c]["lon"], ost[c]["lat"],
                                                                   *q) for q in nxt))
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
                # One node per station across lines: OSM often maps a station twice (the
                # station node and a stop position, 20-50 m apart, both named), and two lines
                # each taking the one nearest their own previous point would not meet there.
                for prev_hit in chosen[skey_short(label)]:
                    if dist_m(ost[prev_hit]["lon"], ost[prev_hit]["lat"], ost[hit]["lon"],
                              ost[hit]["lat"]) <= 2000:
                        hit = prev_hit
                        break
                else:
                    chosen[skey_short(label)].append(hit)
                s = ost[hit]
                if placed and placed[-1][4] == hit:
                    continue                      # two list names, one OSM station
                placed.append(("station", label, s["lon"], s["lat"], hit, km))
            legs = []
            for a, b in zip(placed[:-1], placed[1:]):
                sa, sb = track.snap(a[2], a[3]), track.snap(b[2], b[3])
                crow = dist_m(a[2], a[3], b[2], b[3]) / 1000
                got = track.trace(sa, sb, crow * 2.5 + 5) if sa and sb else None
                legs.append((a, b, got[1] if got else None))
            for j, new in interpolate_posts(l, legs, log).items():
                placed[j] = placed[j][:5] + (new,)
            for li, (a, b) in enumerate(zip(placed[:-1], placed[1:])):
                crow = dist_m(a[2], a[3], b[2], b[3]) / 1000
                tkm = legs[li][2]
                if tkm is None:
                    stat["sections with no trace"] += 1
                    log(f"    {l['id']}: {a[1]} - {b[1]} NO TRACE ({crow:.1f} km crow-fly)")
                elif tkm > 2.0 * crow + 3:
                    log(f"    {l['id']}: {a[1]} - {b[1]} traced {tkm:.1f} km for "
                        f"{crow:.1f} km crow-fly: check the list")
                if l["chain"] and a[5] is not None and b[5] is not None:
                    km = abs(b[5] - a[5])
                    stat["sections measured by km posts"] += 1
                    if tkm is not None and abs(tkm - km) > KM_TOL[0] * km + KM_TOL[1]:
                        stat["km posts off the trace"] += 1
                        log(f"    {l['id']}: KM {a[1]} ({a[5]:g}) - {b[1]} ({b[5]:g}): posts "
                            f"{km:.1f} km, traced {tkm:.1f} km")
                    chain_total += km
                else:
                    km = tkm if tkm is not None else crow * 1.2
                    stat["sections measured by trace"] += 1
                    if l["chain"]:
                        log(f"    {l['id']}: {a[1]} - {b[1]}: no km post on a chained line")
                traced_total += tkm if tkm is not None else crow * 1.2
                k += 1
                ops = []
                for q in (a, b):
                    if q[0] == "border":
                        op = f"{CC}:b:{q[1]}"
                        # rinf.py names the point "e" + uopid: borders.EXTRA's id
                        pts_out[op] = {"op": op, "uopid": q[1], "type": "90", "name": q[1],
                                       "lon": q[2], "lat": q[3]}
                    elif q[0] == "station":
                        op = f"{CC}:s:{q[4]}"
                        pts_out[op] = {"op": op, "uopid": f"PK{q[4]}", "type": "10",
                                       "name": ost[q[4]]["name"], "lon": q[2], "lat": q[3]}
                    else:
                        jk = re.sub(r"[^0-9a-z]", "", skey(q[1]))[:24] or f"{q[2]:.4f}{q[3]:.4f}"
                        op = f"{CC}:j:{jk}"
                        pts_out[op] = {"op": op, "uopid": f"PKJ{jk}", "type": "80",
                                       "name": q[1], "lon": q[2], "lat": q[3]}
                    ops.append(op)
                rows.append({"sol": f"{CC}:{l['id']}:{k}", "line": l["id"], "a": ops[0],
                             "b": ops[1], "len": f"{km:.3f}", "im": l["im"],
                             "label": f"{a[1]} - {b[1]}"})
        names[l["id"]] = {"name": l["name"], "name_en": l["name_en"], "im": l["im"],
                          "suspended": bool(l["suspended"]), "chain": bool(l["chain"]),
                          "traced_km": round(traced_total, 3),
                          "chain_km": round(chain_total, 3) if l["chain"] else None}
        log(f"  {l['id']:>24} {l['name'][:38]:<38} traced {traced_total:7.1f} km"
            + (f"  posts {chain_total:7.1f} km" if l["chain"] else "")
            + ("  [suspended]" if l["suspended"] else ""))
    log(f"PK: {len(LINES)} lines, {len(rows)} section rows, {len(pts_out)} points; {dict(stat)}")
    if write:
        OUT.mkdir(parents=True, exist_ok=True)
        stamp = {"endpoint": "pk_register.py (hand-written line list, pk_lines.py)",
                 "fetched": date.today().isoformat()}
        for fn, obj in (("sections.json", {**stamp, "rows": rows}),
                        ("points.json", {**stamp, "rows": list(pts_out.values())}),
                        ("names.json", names)):
            tmp = OUT / (fn + ".tmp")
            tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=0 if fn == "names.json"
                                      else None), "utf-8")
            os.replace(tmp, OUT / fn)
        log(f"wrote {OUT} in {time.time() - t0:.0f} s")
    return rows, pts_out, names


FILL_BASE = -8_100_000_000_000   # synthetic stop ids: this - k
FILL_SOURCE = "pk_register wikidata"
FILL_TRACK_M = 1500   # a Wikidata station point this close to OSM track is put on it


def load_wikidata():
    """{name key: [(label, lon, lat, qid)]} from data/raw/pk/wikidata_stations.json (QLever,
    every item in Pakistan that is a railway station by P31/P279*, with its coordinate)."""
    p = OUT / "wikidata_stations.json"
    if not p.exists():
        return {}, {}
    d = json.loads(p.read_text("utf-8"))
    by_key, by_short = defaultdict(list), defaultdict(list)
    for r in d["results"]["bindings"]:
        m = re.match(r"Point\(([-\d.]+) ([-\d.]+)\)", r["coord"]["value"], re.I)
        if not m:
            continue
        lab = re.sub(r"\s+railway station$", "", r["label"]["value"], flags=re.I)
        rec = (lab, float(m.group(1)), float(m.group(2)), r["item"]["value"].rsplit("/", 1)[-1])
        by_key[skey(lab)].append(rec)
        by_short[skey_short(lab)].append(rec)
    return by_key, by_short


def fill(log=print):
    """Stations the list names that OSM lacks (Narowal Junction, Shahinabad Junction,
    Bahawalnagar Junction...): Wikidata's point for the name, nearest the line's stations OSM
    has, if within FILL_TRACK_M of OSM track; an UNNAMED OSM station node within BLIND_M of it
    takes the name, else a halt is added on the nearest track (id FILL_BASE - k, tagged
    `source=pk_register wikidata`). Rewrites data/proc/pk/stops.pkl; run after every extract.
    A name Wikidata does not have stays out (the list's neighbours measure across it)."""
    import pickle
    import build_model as bm
    import rinf
    d = ROOT / "data" / "proc" / CC
    stops = pickle.load(open(d / "stops.pkl", "rb"))
    stops = {k: v for k, v in stops.items() if not (k <= FILL_BASE)}
    for k, (t, lon, lat) in stops.items():
        if t.get("source") == FILL_SOURCE:
            t.pop("name", None)
            t.pop("source", None)
    ways, _r, _s, cid, cx, cy = bm.load(CC, lambda m: None)
    track = rinf.Track(ways, bm.Coords(cid, cx, cy), lambda m: None)
    ost, by_key, by_short = station_index(stops)
    wk, ws = load_wikidata()
    named = added = k = 0
    seen = set()
    for l in LINES:
        for seq in [l["pts"]] + l["more"]:
            parsed = [parse(raw) for raw in seq]
            have = []
            for kind, label, at, _km in parsed:
                cs = by_key.get(skey(label)) or by_short.get(skey_short(label))
                if kind == "station" and cs:
                    have += [(ost[c]["lon"], ost[c]["lat"]) for c in cs]
            for kind, label, at, _km in parsed:
                if kind != "station" or at or label in seen:
                    continue
                if by_key.get(skey(label)) or by_short.get(skey_short(label)):
                    continue
                cands = wk.get(skey(label)) or ws.get(skey_short(label)) or []
                if not cands or not have:
                    log(f"  {l['id']}: {label!r}: in neither OSM nor Wikidata")
                    continue
                dist, rec = min((min(dist_m(r[1], r[2], *h) for h in have), r) for r in cands)
                if dist > MAX_JUMP_KM * 1000:
                    log(f"  {l['id']}: {label!r}: Wikidata's {rec[3]} is {dist / 1000:.0f} km "
                        f"from the line")
                    continue
                e, t, dd = track.near(rec[1], rec[2], FILL_TRACK_M)
                if not len(e):
                    log(f"  {l['id']}: {label!r}: Wikidata's {rec[3]} ({rec[1]:.4f}, "
                        f"{rec[2]:.4f}) is over {FILL_TRACK_M} m from any track")
                    continue
                seen.add(label)
                i = int(dd.argmin())
                ei, ti = e[i], t[i]
                lon = (track.ax[ei] + ti * (track.bx[ei] - track.ax[ei])) / track.kx
                lat = (track.ay[ei] + ti * (track.by[ei] - track.ay[ei])) / track.ky
                blank = sorted((dist_m(x, y, lon, lat), nid) for nid, (tg, x, y) in stops.items()
                               if not tg.get("name") and abs(x - lon) < 0.01
                               and abs(y - lat) < 0.01
                               and (tg.get("railway") in ("station", "halt", "stop")
                                    or tg.get("public_transport") == "station"))
                if blank and blank[0][0] <= BLIND_M:
                    tg = stops[blank[0][1]][0]
                    tg.update({"name": label, "source": FILL_SOURCE, "train": "yes"})
                    named += 1
                    log(f"  {l['id']}: OSM's unnamed station {blank[0][1]} named {label!r}")
                else:
                    k += 1
                    stops[FILL_BASE - k] = ({"railway": "station", "name": label, "train": "yes",
                                             "source": FILL_SOURCE, "wikidata": rec[3]},
                                            lon, lat)
                    added += 1
                    log(f"  {l['id']}: {label!r} added from Wikidata {rec[3]} "
                        f"({float(dd.min()):.0f} m off the track, put on it)")
    tmp = d / "stops.pkl.tmp"
    with open(tmp, "wb") as f:
        pickle.dump(stops, f, protocol=4)
    os.replace(tmp, d / "stops.pkl")
    log(f"PK fill: {named} unnamed OSM stations named, {added} stations added from Wikidata")


def interpolate_posts(l, legs, log):
    """legs: [(a, b, traced km or None)] in order, a and b placed points (6th field the post).
    A post that disagrees with the track on both sides while the span across it agrees is out
    of place (the RDT's Yousafwala at 1,066 between Sahiwal 1,056 and Okara Cantt 1,081:
    traced 17.3 and 7.3 km for posts 10 and 15): it is replaced by one interpolated along the
    traced span, so the line keeps PR's chainage end to end. Returns {index of the point in
    the run: new post}; each is logged."""
    out = {}
    if not l["chain"]:
        return out
    n = len(legs)
    for i in range(n - 1):
        (a, m, t1), (_m, b, t2) = legs[i], legs[i + 1]
        if None in (t1, t2) or None in (a[5], m[5], b[5]):
            continue
        p1, p2, span = abs(m[5] - a[5]), abs(b[5] - m[5]), abs(b[5] - a[5])

        def off(x, y):
            return abs(x - y) > KM_TOL[0] * x + KM_TOL[1]
        if off(p1, t1) and off(p2, t2) and not off(span, t1 + t2):
            new = a[5] + (b[5] - a[5]) * t1 / (t1 + t2)
            out[i + 1] = new
            log(f"    {l['id']}: post of {m[1]} {m[5]:g} out of place (traced {t1:.1f} + "
                f"{t2:.1f} for {p1:g} + {p2:g}); interpolated {new:.1f}")
    return out


def build(path, log):
    """build_model's register hook: convert, rinf.py traces it over OSM track; then lines with
    no km posts drop the chainage rinf.py ships (it would only be their own trace), and the
    junction- or border-ended sections of running lines are marked served (each line is
    written from what trains run)."""
    convert(log)
    import rinf
    lines, stations, geoms = rinf.build(path, log)
    names = _names()
    n = 0
    for l in lines:
        ids = [x.split("#")[0] for x in l.get("rinf_ids", ())]
        info = [names[x] for x in ids if x in names]
        if not all(i.get("chain") for i in info):
            l.pop("km_official", None)
            l.pop("chain", None)
        if l.get("suspended") or any(i.get("suspended") for i in info):
            continue
        keys = [f"{a}|{b}" for a, b, *_ in l["sections"]
                if stations.get(a, {}).get("junction") or stations.get(b, {}).get("junction")]
        if keys:
            l["served_sections"] = keys
            n += len(keys)
    log(f"PK: {n} junction- or border-ended sections of running lines marked served; "
        f"{sum(1 for l in lines if 'km_official' in l)} lines keep PR's chainage")
    path_checks(lines, stations, log)
    return lines, stations, geoms


# ============================================================== outside numbers, by path
#
# Few Pakistani lines have a published length of their own, but every named train's article on
# en.wikipedia gives its published distance: (label, [stations in order: the path runs through
# each], km, source). Shortest paths over all register lines between consecutive stations.
PATH_CHECKS = [
    ("Khushhal Khan Khattak Express", ["Karachi City", "Kotri Junction", "Jacobabad Junction",
                                       "Kot Adu Junction", "Attock City Junction",
                                       "Peshawar Cantonment"], 1512.0,
     "WP, Karachi City - Peshawar Cantt by ML-2"),
    ("Jaffar Express", ["Quetta", "Rohri Junction", "Multan Cantt", "Lahore Junction",
                        "Peshawar Cantonment"], 1632.0, "WP, Quetta - Peshawar Cantt via Multan"),
    ("Hazara Express", ["Karachi City", "Multan Cantt", "Khanewal Junction",
                        "Shorkot Cantonment Junction", "Lala Musa Junction", "Taxila",
                        "Havelian"], 1594.0, "WP, Karachi City - Havelian"),
    ("Fareed Express", ["Karachi City", "Lodhran Junction", "Pakpattan", "Raiwind Junction",
                        "Lahore Junction"], 1250.0, "WP, Karachi City - Lahore"),
    ("Thal Express", ["Multan Cantt", "Sher Shah Junction", "Kot Adu Junction",
                      "Kundian Junction", "Basal Junction", "Golra Sharif Junction",
                      "Rawalpindi"], 595.0, "WP, Multan Cantt - Rawalpindi"),
    ("Kohat Express", ["Rawalpindi", "Golra Sharif Junction", "Basal Junction",
                       "Jand Junction", "Kohat Cantt"], 177.0, "WP, Rawalpindi - Kohat Cantt"),
    ("Faiz Ahmed Faiz Passenger", ["Lahore Junction", "Narowal Junction"], 84.0,
     "WP, Lahore - Narowal"),
    ("Chaman Passenger", ["Quetta", "Chaman"], 142.0, "WP, Quetta - Chaman"),
    # The Badar Express (Lahore - Faisalabad, 142) and the Mianwali Express (Lahore - Mari
    # Indus by Sheikhupura, 466) cross the Qila Sattar Shah bridge, which OSM's track lacks
    # since the July 2026 floods: no path to check.
]


def path_checks(lines, stations, log):
    import heapq
    by_name = defaultdict(set)
    for sid, s in stations.items():
        for n in (s.get("name"), s.get("name_en")):
            if n:
                by_name[skey(n)].add(sid)
                by_name[skey_short(n)].add(sid)
    adj = defaultdict(list)
    for ln in lines:
        if ln.get("service"):
            continue
        for sec in ln["sections"]:
            adj[sec[0]].append((sec[1], sec[2]))
            adj[sec[1]].append((sec[0], sec[2]))

    def ids(n):
        return [s for s in (by_name.get(skey(n)) or by_name.get(skey_short(n)) or ()) if s in adj]

    def sp(src, dst):
        dist = {s: 0.0 for s in src}
        h = [(0.0, s) for s in src]
        while h:
            d, u = heapq.heappop(h)
            if u in dst:
                return d, u
            if d > dist.get(u, math.inf):
                continue
            for v, w in adj[u]:
                if d + w < dist.get(v, math.inf):
                    dist[v] = d + w
                    heapq.heappush(h, (d + w, v))
        return None, None
    out = []
    for label, via, km, note in PATH_CHECKS:
        total, cur, bad = 0.0, ids(via[0]), None
        for nxt in via[1:]:
            d, at = sp(cur, set(ids(nxt)))
            if d is None:
                bad = nxt
                break
            total += d
            cur = [at]
        out.append((label, None if bad else total, km))
        log(f"PK path check {label}: "
            + (f"no path to {bad}" if bad else f"{total:.1f} km against {km:.1f}, ratio "
                                                f"{total / km:.3f}") + f" ({note})")
    return out


def _names():
    f = OUT / "names.json"
    return json.loads(f.read_text("utf-8")) if f.exists() else {}


def country_conf():
    def id_name(lid, _uop=None):
        e = _names().get(lid.split("#")[0])
        return (e["name"], e.get("name_en") or "") if e else None

    def suspended(_ref, lids):
        ns = _names()
        return any(ns.get(x.split("#")[0], {}).get("suspended") for x in lids)

    return {
        "iso3": "PAK", "langs": ["en", "ur"],
        "ref": lambda _lid: None,
        "id_name": id_name,
        "suspended": suspended,
        "im": {"Pakistan Railways": "Pakistan Railways"},
        # every OSM rail station on a traced section is a stop: OSM has 9 PR train routes
        "osm_stops": "all",
        # OSM's route=railway relations here carry no line numbers to lend.
        "osm_rel": lambda _t: None,
    }


def split_pieces(lines, stations, geoms, reg_ways, state, log):
    """build_model's hook, after drop_unridden_sections: rinf.split_pieces."""
    import rinf
    rinf.split_pieces(lines, stations, geoms, reg_ways, state, log)


def trace(a, b):
    import build_model as bm
    import rinf
    ways, rels, stops, cid, cx, cy = bm.load(CC, print)
    coords = bm.Coords(cid, cx, cy)
    track = rinf.Track(ways, coords, print)
    ost, by_key, by_short = station_index(stops)
    ca = sorted(by_key.get(skey(a)) or by_short.get(skey_short(a)) or [])
    cb = sorted(by_key.get(skey(b)) or by_short.get(skey_short(b)) or [])
    print(f"{a}: {[(c, ost[c]['name'], ost[c]['lon'], ost[c]['lat']) for c in ca]}")
    print(f"{b}: {[(c, ost[c]['name'], ost[c]['lon'], ost[c]['lat']) for c in cb]}")
    for x in ca:
        for y in cb:
            sa = track.snap(ost[x]["lon"], ost[x]["lat"])
            sb = track.snap(ost[y]["lon"], ost[y]["lat"])
            crow = dist_m(ost[x]["lon"], ost[x]["lat"], ost[y]["lon"], ost[y]["lat"]) / 1000
            got = track.trace(sa, sb, crow * 3 + 5) if sa and sb else None
            print(f"  {x} -> {y}: crow {crow:.1f} km, traced "
                  f"{'-' if got is None else round(got[1], 1)}")


def find(pattern):
    """OSM rail stations whose any name matches a regex."""
    import build_model as bm
    import rinf
    _w, _r, stops, *_ = bm.load(CC, lambda m: None)
    ost = rinf.osm_stations(stops)
    rx = re.compile(pattern, re.I)
    for sid, s in ost.items():
        t = stops[sid][0]
        txt = " / ".join(v for k, v in t.items() if k.startswith(("name", "alt_name",
                                                                   "official_name", "old_name")))
        if rx.search(txt):
            print(f"{sid}\t{txt}\t{s['lon']:.5f},{s['lat']:.5f}\t{t.get('railway')}\t"
                  f"{t.get('ref') or ''}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fill", action="store_true", help="stations OSM lacks, from Wikidata")
    ap.add_argument("--convert", action="store_true")
    ap.add_argument("--dry", action="store_true", help="convert without writing")
    ap.add_argument("--trace", nargs=2, metavar=("FROM", "TO"))
    ap.add_argument("--find", metavar="REGEX")
    args = ap.parse_args()
    t = time.time()
    lg = lambda m: print(f"[{time.time() - t:6.1f}s] {m}", flush=True)  # noqa: E731
    if args.fill:
        fill(lg)
    if args.convert or args.dry:
        convert(lg, write=args.convert)
    elif args.trace:
        trace(*args.trace)
    elif args.find:
        find(args.find)
    elif not args.fill:
        print(__doc__)
