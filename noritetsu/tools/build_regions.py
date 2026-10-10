"""Write dist/regions.json: every built country's name, box, opening view and outline.

    python tools/build_regions.py

WHAT IT IS FOR. The app loads a country only when the map moves over it, so it has to know
where each country is BEFORE loading anything of it. This is that index: a few KB, fetched
once at startup, one entry per country that has a build under dist/data/<cc>/.

It also lists the line ids built in more than one country (`shared_lines`), the OSM line
ids to fold together across countries (`line_aliases`, see `line_aliases`), and each
country's km of line (`km`, see `owned_totals`), so the app can give a country's total
without loading its line data, which it now loads only when zoomed in on it. And it writes
`dist/data/closed.json`, every country's not-running track in one file (`write_closed`),
and `dist/data/search.json`, every line and station name for search (`write_search`).

Run it after a new country's first build, and after any rebuild (a rebuild can change line
ids, twin merges, totals and closed sections). It changes nothing else, and the app still
works without it for countries already listed.

    python tools/build_regions.py --out <dir>    # a trial: every file into <dir> instead

The outlines are religiondots' `country_shapes.geojson` (194 countries, cleaned and
simplified to ~200 m there), simplified again here to ~2 km: the only question asked of them
is "is this country on screen, and is the middle of the screen in it", and at 2 km Japan is
a few hundred points rather than tens of thousands. Names are Natural Earth's.
"""
import gzip
import json
import pathlib
import sys

from shapely.geometry import LineString, shape

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from build_model import untrained  # noqa: E402  the twin merge's own name test
# The outline sources (SHAPES; NAMES, 1:10m, not 1:110m: the small-scale file has no Singapore
# or Hong Kong at all; ISO; OUTLINE) live in borders.py, which also asks which side of a
# border a station is on.
from borders import ISO, NAME, NAMES, OUTLINE, SHAPES  # noqa: E402
DIST = HERE / "dist"

# Degrees. About 2 km at Japan's latitude; see the docstring for why that is enough.
TOLERANCE = 0.02
# Parts smaller than this (square degrees, ~25 km2) are dropped: a rock offshore does not
# decide whether a country is on screen, and there are thousands of them.
MIN_PART = 0.002
# Degrees from the largest part beyond which a part is overseas, for the opening view only.
FAR_DEG = 15

# Areas added to a country's outline. Russia: the 2022-annexed railways, built with Russia
# since 2026-10-04 (rinf_countries/ru.py ANNEX_RUNNING; running where trains run, greyed
# elsewhere): ru_register.py's clip area. religiondots' `ru` has Crimea.
EXTRA_AREAS = {"ru": [HERE / "data" / "raw" / "ru" / "annex.geojson"]}


def main():
    # --out <dir>: write regions.json and closed.json there instead (a trial run).
    out_dir = None
    if "--out" in sys.argv:
        out_dir = pathlib.Path(sys.argv[sys.argv.index("--out") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
    built = sorted(p.name for p in (DIST / "data").iterdir()
                   if p.is_dir() and (p / "lines.json").exists()
                   and (DIST / "data" / f"{p.name}.pmtiles").exists())
    if not built:
        sys.exit("no built country under dist/data/")

    # Several features can share a code (France's is also on Clipperton Island), so the most
    # populous one names the country.
    names, pop, ne_geom = {}, {}, {}
    for f in json.loads(NAMES.read_text(encoding="utf-8"))["features"]:
        q = f["properties"]
        cc, p = q.get("ISO_A2_EH"), q.get("POP_EST") or 0
        if cc not in names or p > pop[cc]:
            names[cc], pop[cc], ne_geom[cc] = q.get("NAME"), p, f["geometry"]

    by_cc = {}
    for f in json.loads(SHAPES.read_text(encoding="utf-8"))["features"]:
        by_cc.setdefault(f["properties"]["cc"], []).append(shape(f["geometry"]))

    out = {}
    for cc in built:
        if cc in OUTLINE and OUTLINE[cc].exists():
            feats = json.loads(OUTLINE[cc].read_text(encoding="utf-8"))
            feats = feats.get("features", [feats])
            by_cc[cc] = [shape(f.get("geometry", f)) for f in feats]
            print(f"  {cc}: outline from {OUTLINE[cc].name}")
        if cc not in by_cc:
            # religiondots leaves out a few small countries (Luxembourg); Natural Earth's
            # 1:10m outline does as well for "is it on screen".
            g = ne_geom.get(ISO.get(cc, cc.upper()))
            if g is None:
                print(f"  {cc}: no outline in {SHAPES.name} or {NAMES.name}; left out, so it "
                      f"will not load by panning")
                continue
            print(f"  {cc}: outline from {NAMES.name}")
            by_cc[cc] = [shape(g)]
        for extra in EXTRA_AREAS.get(cc, []):
            if extra.exists():
                feats = json.loads(extra.read_text(encoding="utf-8"))
                feats = feats.get("features", [feats])
                by_cc[cc] += [shape(f.get("geometry", f)) for f in feats]
                print(f"  {cc}: added {extra.name}")
        geom = by_cc[cc][0]
        for g in by_cc[cc][1:]:
            geom = geom.union(g)
        polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
        parts = []
        for p in polys:
            p = p.simplify(TOLERANCE, preserve_topology=True)
            if p.is_empty or p.area < MIN_PART:
                continue
            parts.append([[round(x, 3), round(y, 3)] for x, y in p.exterior.coords])
        w, s, e, n = geom.bounds
        # The opening view leaves out outlying islands: Japan's full box reaches Minami-Torishima
        # at 154 E and would open on the Pacific. Parts under 5% of the largest do not count.
        # Nor do parts far from the largest: French Guiana is big enough to pass the 5% test
        # and would open France on South America.
        main = max(polys, key=lambda p: p.area)
        mx, my = main.centroid.x, main.centroid.y
        core = [p for p in polys if p.area >= 0.05 * main.area
                and abs(p.centroid.x - mx) < FAR_DEG and abs(p.centroid.y - my) < FAR_DEG]
        vw = min(p.bounds[0] for p in core); vs = min(p.bounds[1] for p in core)
        ve = max(p.bounds[2] for p in core); vn = max(p.bounds[3] for p in core)
        out[cc] = {
            # borders.NAME for a region Natural Earth has no country for (Abkhazia, xa)
            "name": NAME.get(cc) or names.get(ISO.get(cc, cc.upper()), cc.upper()),
            "bbox": [round(w, 3), round(s, 3), round(e, 3), round(n, 3)],
            "view": [round(vw, 3), round(vs, 3), round(ve, 3), round(vn, 3)],
            "parts": parts,
        }
        print(f"  {cc}: {out[cc]['name']}, {len(parts)} parts, "
              f"{sum(len(p) for p in parts)} points")

    lines, aliases = {}, {}
    for cc in out:
        lines[cc] = {l["id"]: l for l in json.loads(
            (DIST / "data" / cc / "lines.json").read_text(encoding="utf-8"))["lines"]}
        a = DIST / "data" / cc / "aliases.json"
        aliases[cc] = (json.loads(a.read_text(encoding="utf-8")).get("lines") or {}
                       if a.exists() else {})
    fold = line_aliases(lines, aliases)

    # Each country's km of line, as the app counts it once the country is loaded, so the
    # Countries tab and the headline can give it without loading the country (`km`).
    foots = {}
    for cc in out:
        f = DIST / "data" / cc / "foot.json"
        foots[cc] = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    # The routes' countries also need who runs each route and its type (route_rows).
    opsjs, types = {}, {}
    for cc in ROUTE_CCS:
        for name, into in (("ops.json", opsjs), ("types.json", types)):
            f = DIST / "data" / cc / name
            if cc in out and f.exists():
                into[cc] = json.loads(f.read_text(encoding="utf-8"))
    for cc, km in owned_totals(lines, foots, fold, opsjs, types).items():
        out[cc]["km"] = round(km, 1)
    print("  km of line: " + ", ".join(f"{cc} {out[cc]['km']:,.0f}" for cc in out))
    write_closed(lines, fold, (out_dir or DIST / "data") / "closed.json")
    write_search(lines, fold, (out_dir or DIST / "data") / "search.json")

    # A line that crosses a border is built in every country it touches under one id (the OSM
    # route_master's). The app merges them into one line, and needs to know which countries to
    # load for it before loading any of them. Ids are folded first, as the app folds them.
    seen = {}
    for cc in out:
        for lid in lines[cc]:
            ccs = seen.setdefault(fold.get(lid, lid), [])
            if cc not in ccs:
                ccs.append(cc)
    shared = {lid: sorted(ccs) for lid, ccs in sorted(seen.items()) if len(ccs) > 1}
    print(f"  {len(shared)} line ids shared between countries")

    path = (out_dir or DIST) / "regions.json"
    tmp = path.with_suffix(".json.tmp")     # an open page never reads half a file
    tmp.write_text(json.dumps({"regions": out, "shared_lines": shared,
                               "line_aliases": dict(sorted(fold.items()))},
                              ensure_ascii=False, separators=(",", ":")),
                   encoding="utf-8")
    tmp.replace(path)
    print(f"wrote {path}, {path.stat().st_size / 1024:.1f} KB")


def merge_spans(spans, km):
    """index.html's mergeSpans: spans along one section merged, gaps under 150 m closed (at
    most 10% of the section), and the ends snapped to 0 and 1 within the same tolerance."""
    if not spans:
        return []
    tol = min(0.15 / km, 0.1) if km > 0 else 0
    out = []
    for lo, hi in sorted(spans):
        if out and lo <= out[-1][1] + tol:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    if out[0][0] <= tol:
        out[0][0] = 0
    if out[-1][1] >= 1 - tol:
        out[-1][1] = 1
    return out


OFF_REGISTER_SHARE = 0.4     # index.html's: an OSM line this far off the register is listed

# index.html's routes (ROUTE_CCS, routesOf; HANDOFF "Routes in the US, Canada and Australia"):
# the same constants, change both together.
ROUTE_CCS = ("us", "ca", "au")
ROUTE_GAP_KM, ROUTE_GAP_SHARE = 0.15, 0.01
CORRIDOR_KM, CORRIDOR_SERVICES, CORRIDOR_OPERATORS, CORRIDOR_SHARE = 20, 3, 2, 0.25
ROUTE_ON_KM, ROUTE_ON_SHARE = 0.5, 0.05
ROUTE_NETWORK = {"ARR": "Alaska Railroad", "VIA Rail": "VIA Rail Canada"}


def spans_meet(a, b):
    """index.html's spansMeet: the length two runs of merged spans share."""
    out, i, j = 0.0, 0, 0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if hi > lo:
            out += hi - lo
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def span_len(sp):
    return min(1.0, sum(hi - lo for lo, hi in sp))


def route_ops(l, opsj):
    """index.html's opKeysOf for an OSM line, with routeOpKeys for one with no operator."""
    keys = (opsj or {}).get("keys") or {}
    sw = ((opsj or {}).get("lines") or {}).get(l["id"])
    if sw:
        return {x[0] for x in sw}
    raw = (l.get("operator") or "").strip()
    if raw:
        out = set()
        for s in (x.strip() for x in raw.split(";")):
            if s:
                k = keys.get(s, s)
                out |= set(k) if isinstance(k, list) else {k}
        return out
    vals = [v for k in keys.values() for v in (k if isinstance(k, list) else [k])]
    out = set()
    for s in (x.strip() for x in (l.get("network") or "").split(";")):
        if not s:
            continue
        if s in keys:
            k = keys[s]
            out |= set(k) if isinstance(k, list) else {k}
        else:
            named = next((v for v in vals if isinstance(v, str) and v.endswith(f"({s})")), None)
            out.add(ROUTE_NETWORK.get(s) or named or s)
    return out


def route_rows(cc, piece, tracks, foot_of, cc_lines, fold, opsj, types):
    """The folded ids index.html's routesOf lists in a ROUTE_CCS country: every OSM piece with
    track (a route), and a register line the routes leave uncovered, a corridor, or one with
    no running track. Measured on footprints as the app's pieceFoot does."""
    ps, foot, sec_of, closed = piece
    by_fid = {}
    for l in cc_lines.values():
        by_fid.setdefault(fold.get(l["id"], l["id"]), []).append(l)

    def piece_foot(p):
        raw = {}
        for s in p["sections"]:
            if s[3] in closed:
                continue
            for t, fr, to, a, b in foot_of(foot, s[3]):
                if to == fr or t in closed or t not in sec_of:
                    continue
                raw.setdefault(t, []).append((min(fr, to), max(fr, to)))
        return {t: merge_spans(iv, sec_of[t][1]) for t, iv in raw.items()}

    routes = [fid for fid, p in ps.items() if p["src"] == "osm" and p["km"] > 0.05]
    rfoot = [piece_foot(ps[fid]) for fid in routes]
    on = {}
    for i, f in enumerate(rfoot):
        for t, sp in f.items():
            on.setdefault(t, []).append((i, sp))
    ops = [set().union(*[route_ops(l, opsj) for l in by_fid[fid]]) for fid in routes]
    def ltype(l):                      # types.json: a type, or {type, also}
        t = (types or {}).get(l["id"])
        return t if isinstance(t, str) else (t or {}).get("type")
    tourist = [next((ltype(l) for l in by_fid[fid] if ltype(l)), None) == "tourist"
               for fid in routes]
    rows = set(routes)
    for fid in tracks:
        km = gap = 0.0
        by = {}
        for t, sp in piece_foot(ps[fid]).items():
            k, mine = sec_of[t][1], span_len(sp)
            rs = on.get(t, [])
            km += k * mine
            allsp = merge_spans([x for _, s2 in rs for x in s2], k) if rs else []
            gap += k * max(0.0, mine - spans_meet(sp, allsp))
            for i, s2 in rs:
                m = k * spans_meet(sp, s2)
                if m > 0:
                    by[i] = by.get(i, 0.0) + m
        on2 = [(i, m) for i, m in by.items() if m >= min(ROUTE_ON_KM, ROUTE_ON_SHARE * km)]
        covered = gap <= max(ROUTE_GAP_KM, ROUTE_GAP_SHARE * km)
        many = [i for i, m in on2 if m >= CORRIDOR_SHARE * km and not tourist[i]]
        corridor = (km >= CORRIDOR_KM and len(many) >= CORRIDOR_SERVICES
                    and len(set().union(*[ops[i] for i in many])) >= CORRIDOR_OPERATORS)
        if not covered or corridor or not ps[fid]["km"] > 0.05:
            rows.add(fid)
    return rows


def owned_totals(lines, foots, fold, opsjs=None, types=None):
    """{cc: km of line}: the track the country's listed lines run over, each piece once,
    closed sections left out. The sum index.html's unionTotals makes for `c:<cc>` (Anita,
    2026-10-08: a country is the union of its lines, so riding every line reads 100%), on
    that country's build alone; the app shows this as the country's total until the country
    is loaded, and its own sum after.

    A section's footprint (foot.json; [owner section, from, to, a, b] in `scale` units, a
    short entry taking a from the last b, one of three also running to b = 1) says which owner
    track riding it covers. The total is the union of the footprints of every running section
    of every listed line's piece in the country. Which lines are listed is index.html's
    `listed`: a register line unless it owns nothing (ownTrack: the part of each of its own
    sections its own sections' footprints cover; named trains own nothing), an OSM line not a
    named train with at least OFF_REGISTER_SHARE of it on track an OSM line owns; a line over
    a border is listed if any country's piece is. Line ids are folded across countries
    (line_aliases) as the app folds them.

    ON ONE COUNTRY'S BUILD ALONE, because the app's own sum depends on which neighbours are
    loaded and in what order: a line over a border joined with a neighbour's drops the
    sections both countries built (the same two stations; a Basel tram stop is in both
    extracts) from whichever loaded second. Measured 2026-10-03 (the owned-track sum then)
    against the app with all 34 countries loaded: 30 within 0.05 km, then Czechia 6.5 km (of
    9,526), Poland 4.2, Luxembourg 2.6, France 0.6, all from border sections both countries
    built.

    In ROUTE_CCS (the US, Canada, Australia; 2026-10-09) the listed lines are index.html's
    routesOf rows instead (route_rows): every OSM line, and the register lines the routes leave
    uncovered or that are corridors. That needs each country's ops.json (`opsjs`, for the
    corridors' operators) and types.json (`types`, tourist trains aside).
    """
    opsjs, types = opsjs or {}, types or {}
    pieces, service = {}, {}          # cc -> fid -> piece; fid -> named train anywhere
    for cc, ls in lines.items():
        f = foots.get(cc) or {}
        scale = f.get("scale") or 1
        foot = {}
        for k, v in (f.get("foot") or {}).items():
            prev, out = 0, []
            for e in v:
                a = e[3] if len(e) == 5 else prev
                b = e[4] if len(e) == 5 else e[3] if len(e) == 4 else scale
                prev = b
                out.append((int(e[0]), e[1] / scale, e[2] / scale, a / scale, b / scale))
            foot[int(k)] = out
        sec_of, closed = {}, set()
        ps = {}
        for l in ls.values():
            fid = fold.get(l["id"], l["id"])
            service[fid] = service.get(fid, False) or bool(l.get("service"))
            shut = set(l.get("closed") or [])
            p = ps.setdefault(fid, {"src": l.get("src", "osm"), "km": 0.0, "sections": [],
                                    "seen": set()})
            off = 0.0
            for s in l["sections"]:
                sec_of[s[3]] = (fid, s[2])
                if f"{s[0]}|{s[1]}" in shut:
                    closed.add(s[3])
                    off += s[2]
            p["km"] += max(0.0, (l.get("km") or 0) - off)
            for s in l["sections"]:
                k = (min(s[0], s[1]), max(s[0], s[1]))
                if k in p["seen"]:          # combineParts keeps a section two lines share once
                    if s[3] not in closed:
                        p["km"] -= s[2]
                    continue
                p["seen"].add(k)
                p["sections"].append(s)
        for p in ps.values():
            p["km"] = max(0.0, p["km"])
        pieces[cc] = (ps, foot, sec_of, closed)

    def foot_of(foot, gid):
        return foot[gid] if gid in foot else [(gid, 0.0, 1.0, 0.0, 1.0)]

    # Which pieces are listed, per country, then which folded ids (any piece listed).
    listed, base_tracks = set(), {}
    for cc, (ps, foot, sec_of, closed) in pieces.items():
        raw = {}
        for fid, p in ps.items():
            if service[fid]:
                continue
            for s in p["sections"]:
                if s[3] in closed:
                    continue
                for t, fr, to, a, b in foot_of(foot, s[3]):
                    te = sec_of.get(t)
                    if te is None or te[0] != fid or t in closed or to == fr:
                        continue
                    raw.setdefault(t, []).append((min(fr, to), max(fr, to)))
        own = {t: min(1.0, sum(hi - lo for lo, hi in merge_spans(iv, sec_of[t][1])))
               for t, iv in raw.items()}
        for fid, p in ps.items():
            running = [s for s in p["sections"] if s[3] not in closed]
            if p["src"] != "osm":
                owned = sum(s[2] * own.get(s[3], 0.0) for s in running)
                ok = not (p["km"] > 0.05) or owned > 0.05
            elif service[fid]:
                ok = False
            else:
                offreg = 0.0
                for s in running:
                    for t, fr, to, a, b in foot_of(foot, s[3]):
                        te = sec_of.get(t)
                        if te is not None and ps[te[0]]["src"] == "osm":
                            offreg += s[2] * abs(b - a)
                ok = offreg > 0.05 and offreg >= OFF_REGISTER_SHARE * p["km"]
            if ok and cc not in ROUTE_CCS:
                listed.add(fid)
            if ok and cc in ROUTE_CCS and p["src"] != "osm":
                base_tracks.setdefault(cc, []).append(fid)

    # The US, Canada and Australia list routes in place of the register track they cover.
    for cc in ROUTE_CCS:
        if cc in pieces:
            listed |= route_rows(cc, pieces[cc], base_tracks.get(cc, []), foot_of,
                                 lines[cc], fold, opsjs.get(cc), types.get(cc))

    totals = {}
    for cc, (ps, foot, sec_of, closed) in pieces.items():
        raw = {}
        for fid, p in ps.items():
            if fid not in listed:
                continue
            for s in p["sections"]:
                if s[3] in closed:
                    continue
                for t, fr, to, a, b in foot_of(foot, s[3]):
                    if to == fr or t in closed or t not in sec_of:
                        continue
                    raw.setdefault(t, []).append((min(fr, to), max(fr, to)))
        total = 0.0
        for t, iv in raw.items():
            km = sec_of[t][1]
            total += km * min(1.0, sum(hi - lo for lo, hi in merge_spans(iv, km)))
        totals[cc] = total
    return totals


# Degrees: about 3 m. The not-running track is drawn alone (no tiles under it), so this only
# has to keep its shape; at 5 decimals and this tolerance it is a few hundred KB for every
# country together.
CLOSED_TOLERANCE = 0.00003


def write_closed(lines, fold, path):
    """dist/data/closed.json: every country's not-running sections (a line's `closed`), as
    [[cc, line id, [[lon, lat], ...]], ...], line ids folded as the app folds them.

    WHY. The app draws not-running track grey and dashed from each line's geometry file, so it
    had to fetch every such line's file (140 files, 2.6 MB for a view of Europe) and could draw
    it only for countries whose line data was loaded. Line data now loads only when zoomed in
    (index.html, DATA_MIN_ZOOM), and this one file draws it for every country at every zoom.
    The app falls back to the geometry files where this file is missing."""
    out, nfiles = [], 0
    for cc, ls in lines.items():
        for l in ls.values():
            if not l.get("closed"):
                continue
            g = DIST / "data" / cc / "geom" / f"{l['id']}.json"
            if not g.exists():
                continue
            nfiles += 1
            geom = json.loads(g.read_text(encoding="utf-8"))
            for k in l["closed"]:
                a, b = k.split("|")
                pts = geom.get(k) or geom.get(f"{b}|{a}")
                if not pts or len(pts) < 2:
                    continue
                ls_ = LineString(pts).simplify(CLOSED_TOLERANCE, preserve_topology=False)
                out.append([cc, fold.get(l["id"], l["id"]),
                            [[round(x, 5), round(y, 5)] for x, y in ls_.coords]])
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    tmp.replace(path)
    print(f"  wrote {path}: {len(out)} not-running sections from {nfiles} lines, "
          f"{path.stat().st_size / 1024:.0f} KB")


def op_votes(lines):
    """{raw operator: the English name its single-operator lines give most} (OP_EN)."""
    votes = {}
    for ls in lines.values():
        for l in ls.values():
            raw, en = l.get("operator") or "", l.get("operator_en") or ""
            if not raw or not en or ";" in raw or ";" in en:
                continue
            m = votes.setdefault(raw, {})
            m[en] = m.get(en, 0) + 1
    return {raw: max(m.items(), key=lambda kv: kv[1])[0] for raw, m in votes.items()}


def passenger_ops(ps, opsj, op_en):
    """The names index.html's opName gives a line (opKeysInOrder, opKeysOf, opLabel) where any
    piece's country has an ops.json (tools/operators.py), else []: a register line's passenger
    operators by share, else each part of its operator tag through that country's key table,
    else through OP_EN; each key by its short name in operators.json, deduplicated in order.
    Before 2026-10-08 search showed the track owner (DB InfraGO, Network Rail, BNSF) here."""
    if not any(cc in opsj for _, cc, _ in ps):
        return []
    keys = []
    for _, cc, l in ps:
        m = opsj.get(cc) or {}
        sw = (m.get("lines") or {}).get(l["id"])
        if sw:
            keys += [x[0] for x in sw]
            continue
        raw = (l.get("operator") or "").strip()
        for s in (x.strip() for x in raw.split(";")):
            if not s:
                continue
            k = (m.get("keys") or {}).get(s)
            keys += ([k] if isinstance(k, str) else list(k)) if k is not None else [op_en.get(s, s)]
    out = []
    for k in keys:
        name = opsj["_short"].get(k, k)
        if name not in out:
            out.append(name)
    return out


def op_display(lines):
    """{(operator, operator_en): the name index.html's opName gave a line with them before
    ops.json, still used where a country has none (passenger_ops).

    The app learns one English name per raw operator string from the single-operator lines
    that carry one (most votes, the first seen on a tie; mergeRegion's OP_EN), then names a
    line by its own operator_en, or failing that by each part of its operator translated,
    deduplicated and sorted (opKey). Learned here over every country at once."""
    op_en = op_votes(lines)
    out = {}
    for ls in lines.values():
        for l in ls.values():
            raw, en = l.get("operator") or "", l.get("operator_en") or ""
            if (raw, en) in out:
                continue
            if en and ";" not in en:
                name = en
            elif not raw.strip():
                name = "(no operator)"
            else:
                parts = [p.strip() for p in raw.strip().split(";") if p.strip()]
                name = " · ".join(sorted({op_en.get(p, p) for p in parts}))
            out[(raw, en)] = name
    return out


def write_search(lines, fold, path):
    """dist/data/search.json: every line and station name in one file, for search.

    WHY. A country's line data loads only when the map is zoomed in on it (index.html,
    DATA_MIN_ZOOM), so a search over a zoomed-out view had to load every country on screen
    before it could answer: 33 MB and 5.7 s for Europe at z4. The app loads this file instead
    on the first keystroke, searches it, and loads only the country of the hit picked.

    WHAT IS IN IT is what search matches on and what a hit shows before its country is in:
    a line's names, ref, operator (raw, English and as the app names it), kind, colour, km
    (closed sections off, as the app counts it) and whether it is a named train or "as
    operated"; a station's names, where it is (to fly there while its country loads) and how
    many lines call there (search ranks stations by it). Left out: junctions and stations with
    no name, which search never shows. Line ids are folded across countries as the app folds
    them (line_aliases), so a line over a border is one entry with the km of all its pieces, a
    section two countries both built counted once. An English name equal to the native one is
    left empty (the app falls back to the native one anyway). Each entry sits under the one
    country to load for it: for a line, one whose own build has it under the folded id (the
    app loads the rest of a line over a border itself once it is open); for a station in two
    builds, the first.

    SMALL, because it is fetched whole: columns rather than rows, per country (no country
    column), operators and kinds as indexes into a table, and station positions in thousandths
    of a degree (~100 m, close enough to fly to), each the step from the station before it in
    that country's list (the first from 0). Rows rather than columns, with 4-decimal
    positions, was 7.6 MB, 2.6 MB gzipped; this is about 2 MB gzipped.

        {"v": 2, "ops": [[operator, operator_en, shown as], ...], "kinds": [kind, ...],
         "types": [type, ...],
         "lines": {cc: [ids, names, names_en, refs, ops, kinds, colours, kms, flags, types]},
                                                 # flags: 1 named train, 2 as operated
                                                 # types: index into "types", -1 for none
                                                 # (tools/line_types.py's types.json)
         "stations": {cc: [ids, names, names_en, lon steps, lat steps, line counts]}}
    """
    ccs = sorted(lines)
    shown = op_display(lines)
    op_en = op_votes(lines)
    opsj = {}
    for cc in ccs:
        f = DIST / "data" / cc / "ops.json"
        if f.exists():
            opsj[cc] = json.loads(f.read_text(encoding="utf-8"))
    f = DIST / "data" / "operators.json"
    brands = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    opsj["_short"] = {k: v["short"] for k, v in brands.items() if v.get("short")}
    # Line types (tools/line_types.py, types.json): a line's main type, by its own id.
    tyj = {}
    for cc in ccs:
        f = DIST / "data" / cc / "types.json"
        if f.exists():
            tyj[cc] = json.loads(f.read_text(encoding="utf-8"))
    ops, opi, kinds, kindi = [], {}, [], {}
    types, typei = [], {}

    def intern(table, index, key, value):
        if key not in index:
            index[key] = len(table)
            table.append(value)
        return index[key]

    # Every piece of a folded id: those built under the id itself first (the app names a
    # joined line from them too), longest first. The longest piece's country is the one
    # loaded for the hit, and so the one whose name the line opens under: RJ 27 is "RJ 27"
    # in Germany (524 km) and "RJ 27: Hamburg <=> Dresden <=> Děčín" in Czechia (11 km).
    pieces = {}
    for cc in ccs:
        for l in lines[cc].values():
            fid = fold.get(l["id"], l["id"])
            pieces.setdefault(fid, []).append((l["id"] != fid, -(l.get("km") or 0), cc, l))
    pieces = {fid: [(p[0], p[2], p[3]) for p in sorted(ps, key=lambda p: p[:3])]
              for fid, ps in pieces.items()}
    out_lines, nlines = {}, 0
    for fid, ps in pieces.items():
        seen, km = set(), 0.0
        for _, cc, l in ps:
            shut = set(l.get("closed") or [])
            for s in l["sections"]:
                if f"{s[0]}|{s[1]}" in shut:
                    continue
                k = (min(s[0], s[1]), max(s[0], s[1]))
                if k in seen:
                    continue
                seen.add(k)
                km += s[2]
        first = lambda k: next((p[2].get(k) for p in ps if p[2].get(k)), "") or ""
        name, name_en = first("name"), first("name_en")
        # The operator pair of the piece that gave the name, so the two belong together.
        lead = next((p[2] for p in ps if p[2].get("operator")), ps[0][2])
        pair = (lead.get("operator") or "", lead.get("operator_en") or "")
        # Shown as the app's opName shows it once loaded: the passenger operators
        # (ops.json, operators.json's short names) where the country has them.
        names = passenger_ops(ps, opsj, op_en)
        show = " · ".join(names) if names else shown[pair]
        op = intern(ops, opi, (pair, show), [pair[0], pair[1], show])
        kind = intern(kinds, kindi, first("kind"), first("kind"))
        # The first piece with a type (the piece loaded for the hit comes first).
        t = next((tyj[cc][l["id"]] for _, cc, l in ps if l["id"] in tyj.get(cc, {})), None)
        t = t["type"] if isinstance(t, dict) else t
        ty = intern(types, typei, t, t) if t else -1
        flags = (1 if any(p[2].get("service") for p in ps) else 0) \
            | (2 if any(p[2].get("dup") for p in ps) else 0)
        cols = out_lines.setdefault(ps[0][1], [[] for _ in range(10)])
        row = [fid, name, "" if name_en == name else name_en, first("ref"), op, kind,
               first("colour"), round(km, 1), flags, ty]
        for col, v in zip(cols, row):
            col.append(v)
        nlines += 1

    stations = {}                       # id -> [cc, name, name_en, x, y, {lines}]
    for cc in ccs:
        sts = json.loads((DIST / "data" / cc / "stations.json").read_text(encoding="utf-8"))
        for sid, s in sts["stations"].items():
            if s.get("j"):
                continue
            calls = {fold.get(x, x) for x in s.get("l") or []}
            if sid in stations:
                stations[sid][5] |= calls
                continue
            n, e = s.get("n") or "", s.get("e") or ""
            if not n and not e:
                continue
            stations[sid] = [cc, n, "" if e == n else e,
                             round(s["x"] * 1000), round(s["y"] * 1000), calls]
    out_st, last = {}, {}
    for sid, (cc, n, e, x, y, calls) in stations.items():
        cols = out_st.setdefault(cc, [[] for _ in range(6)])
        px, py = last.get(cc, (0, 0))
        last[cc] = (x, y)
        for col, v in zip(cols, [sid, n, e, x - px, y - py, len(calls)]):
            col.append(v)

    doc = {"v": 2, "ops": ops, "kinds": kinds, "types": types, "lines": out_lines,
           "stations": out_st}
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False, separators=(",", ":")),
                   encoding="utf-8")
    tmp.replace(path)
    raw = path.read_bytes()
    print(f"  wrote {path}: {nlines} lines, {len(stations)} stations, "
          f"{len(raw) / 1024:.0f} KB ({len(gzip.compress(raw)) / 1024:.0f} KB gzipped)")


# A twin merge decided on less than this much of the kept line in that country is not taken
# as evidence for other countries: Poland's 1 km of the Zittau narrow gauge made the Jonsdorf
# and Oybin branches one line there, and they are two lines in Germany.
MIN_ALIAS_KM = 5


def brand(name):
    """A service's name without direction, train numbers or the route after a colon:
    "Eurostar: Paris - Amsterdam" -> "eurostar", "EC 212: Zagreb => Villach" -> "ec"."""
    s = untrained(name or "").split(":")[0]
    return " ".join(s.split()).casefold()


def line_aliases(lines, aliases):
    """{line id: the id it is folded into in every country}, for OSM lines over a border.

    WHY. Each country merges its OSM twins on its own (build_model.merge_osm_twins), and the
    countries can disagree. Belgium and the Netherlands merged "Eurostar: Paris - Amsterdam"
    (m5189990) into "Eurostar" (m5189989); France kept both, since its two pieces start at
    different stations. The app joins a line over a border by id, so France's m5189990 had no
    partner and a ride from Paris to Amsterdam stopped at the Belgian border. The app folds
    every id through this map as it loads a country, so France's two become one line there too.

    WHAT IS PUBLISHED. A country's alias old -> new is taken only where another country still
    ships `old` (elsewhere the country's own aliases.json already covers saved rides), `new`
    is an OpenStreetMap line (a register line is never joined over a border), the merge was
    decided on at least MIN_ALIAS_KM of it, and the two are the same service by name: the same
    name once direction, train numbers and the route after a colon are taken out, or the same
    first word and the same ref (WESTbahn). A shared ref alone is not enough: Belgium merged
    two European Sleeper trains into the Eurostar because all three are ref "ES".
    Countries can alias in a circle (Czechia keeps one D28 and Poland the other); each group
    folds into one id, the one shipped in the most countries, then the longest.
    """
    ships = {}
    for cc, ls in lines.items():
        for lid in ls:
            ships.setdefault(lid, set()).add(cc)
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    taken, refused = [], []
    for cc, m in sorted(aliases.items()):
        for old, new in sorted(m.items()):
            if not ships.get(old, set()) - {cc}:
                continue
            seen = {old}
            while new in m and new not in seen:      # a chain inside one country
                seen.add(new)
                new = m[new]
            tgt = lines[cc].get(new)
            olds = [lines[c][old] for c in sorted(ships[old]) if c != cc]
            why = None
            if tgt is None:
                why = "kept id not shipped"
            elif tgt.get("src", "osm") != "osm":
                why = "kept line is a register line"
            elif (tgt.get("km") or 0) < MIN_ALIAS_KM:
                why = f"decided on {tgt.get('km') or 0:.1f} km"
            else:
                first = lambda n: ((n or "").split() or [""])[0].rstrip(":").casefold()
                same = any(brand(o.get("name")) == brand(tgt.get("name"))
                           or (first(o.get("name")) == first(tgt.get("name"))
                               and (o.get("ref") or "") == (tgt.get("ref") or "") != "")
                           for o in olds)
                if not same:
                    why = "different service by name"
            label = (f"[{cc}] {old} {olds[0].get('name')!r} ({','.join(sorted(ships[old] - {cc}))})"
                     f" -> {new} {(tgt or {}).get('name')!r}")
            if why:
                refused.append(f"{label}: {why}")
                continue
            taken.append(label)
            parent[find(old)] = find(new)

    groups = {}
    for x in list(parent):
        groups.setdefault(find(x), []).append(x)
    fold = {}
    for members in groups.values():
        if len(members) < 2:
            continue
        keep = max(members, key=lambda i: (len(ships.get(i, ())),
                                           sum(lines[c][i].get("km") or 0
                                               for c in ships.get(i, ())), -len(i), i))
        for i in members:
            if i != keep:
                fold[i] = keep
    print(f"  line aliases over borders: {len(fold)} ids fold, from {len(taken)} country aliases")
    for t in taken:
        print(f"    folded  {t}")
    for r in refused:
        print(f"    refused {r}")
    return fold


if __name__ == "__main__":
    main()
