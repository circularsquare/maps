"""T-975: each registered zone's realm, and the realms' land, `data/packs/realms.geojson`.

    python pipeline/realms.py [--registry FILE] [--out FILE] [--dry]

A realm is a country-sized group of zones the map shows or leaves dark as one (players see only its
name; docs/zones.md "Realms"). The list is `app/src/game/realms.json`, edited by hand: each realm
an id, a name, its member countries (ISO 3166 alpha-2) and, for a country split between realms, its
first-level units (`units`: {country: [unit name, ...]} or "*" for every unit no other realm of that
country lists). A unit name matches a source unit whose name is it or starts with it and a space or
hyphen ("Altai" holds Altai Krai and Altai Republic); the US matches postal codes.

Writes `realm` on every registry row (from its country, or for a split country from where its centre
lies), and for each realm holding a zone its land: Natural Earth admin 0 (de facto borders) for whole
countries, cut along geoBoundaries ADM1 (China, Russia) or the Census states (US) for split ones;
parts farther than KEEP_KM from every one of the realm's zones left out (France's overseas
departments), rings simplified to TOL_M with borders between realms simplified once
(zone_outlines' shared runs), and a label point, the point deepest inside the largest part.
Run by world_run.py's join after the market step (which sets `country`); seconds.
"""
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIST = ROOT / "app/src/game/realms.json"
REG = ROOT / "app/src/game/registry.json"
OUT = ROOT / "data/packs/realms.geojson"
NE = ROOT / "data/raw/world_field/ne_10m_admin_0_countries.zip"
GB = ROOT / "data/raw/worldplan/geoboundaries"
US_STATES = ROOT / "data/raw/cb_2020_us_state_500k.zip"
# a split country's first-level units: the file and the field a unit is named by
UNITS = {"CN": (GB / "CHN_ADM1.geojson", "shapeName"), "RU": (GB / "RUS_ADM1.geojson", "shapeName"),
         "US": (US_STATES, "STUSPS")}
# geoBoundaries names Guangdong after its capital
SOURCE_NAMES = {"Guangzhou Province": "Guangdong Province"}
# Natural Earth parts whose alpha-2 is -99, by ADM0_A3 (as world_plan.NE_FIX)
NE_FIX = {"CYN": "CY", "KOS": "XK", "SOL": "SO", "SAH": "EH"}
TOL_M = 2000
KEEP_KM = 2000
NDP = 3  # decimals kept in the file (about 100 m)


def log(*a):
    print(*a, flush=True)


def fold(s):
    import unicodedata
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower().strip()


def matches(name, unit):
    n, u = fold(name), fold(unit)
    return n == u or n.startswith(u + " ") or n.startswith(u + "-")


def load_list(path=LIST):
    realms = json.loads(Path(path).read_text(encoding="utf-8"))["realms"]
    ids, owner = set(), {}
    for r in realms:
        if r["id"] in ids:
            raise SystemExit(f"realms.json: {r['id']} twice")
        ids.add(r["id"])
        for cc in r.get("countries", []):
            if cc in owner:
                raise SystemExit(f"realms.json: {cc} in both {owner[cc]} and {r['id']}")
            owner[cc] = r["id"]
    split = {cc for r in realms for cc in r.get("units", {})}
    for cc in split:
        if cc in owner:
            raise SystemExit(f"realms.json: {cc} is split by units and also whole in {owner[cc]}")
        if cc not in UNITS:
            raise SystemExit(f"realms.json: no first-level source for {cc} (realms.UNITS)")
    return realms, owner, split


def countries_ne(codes):
    """{alpha-2: shapely geometry} from Natural Earth admin 0, parts of one code dissolved."""
    import pyogrio
    from shapely.ops import unary_union
    df = pyogrio.read_dataframe(f"zip://{NE}!ne_10m_admin_0_countries.shp", columns=["ISO_A2_EH", "ADM0_A3"])
    df["cc"] = [NE_FIX.get(a3, a2) for a2, a3 in zip(df["ISO_A2_EH"], df["ADM0_A3"])]
    out = {}
    for cc in codes:
        g = df[df["cc"] == cc].geometry
        if len(g):
            out[cc] = unary_union(list(g)).buffer(0)
        else:
            log(f"  warning: {cc} not in Natural Earth admin 0; no land for it")
    return out


def units_of(cc):
    """[(name, geometry)] of a split country's first-level units."""
    import pyogrio
    path, field = UNITS[cc]
    df = pyogrio.read_dataframe(path if path.suffix != ".zip" else f"zip://{path}", columns=[field])
    return [(SOURCE_NAMES.get(n, n), g.buffer(0)) for n, g in zip(df[field], df.geometry)]


def split_land(realms, split, land):
    """{realm id: geometry} for the realms of split countries: their units cut to the country's
    Natural Earth land; land no unit covers goes to the nearest of that country's realms."""
    from shapely.ops import unary_union
    out = {}
    for cc in sorted(split):
        if cc not in land:
            continue
        units = units_of(cc)
        mine = [r for r in realms if cc in r.get("units", {})]
        taken, parts = set(), {}
        for r in mine:
            want = r["units"][cc]
            if want == "*":
                continue
            got = []
            for w in want:
                hit = [i for i, (n, _) in enumerate(units) if matches(n, w)]
                if not hit:
                    raise SystemExit(f"realms.json: {r['id']}: no {cc} unit matches {w!r}")
                got += hit
            taken |= set(got)
            parts[r["id"]] = got
        rest = [r for r in mine if r["units"][cc] == "*"]
        if len(rest) > 1:
            raise SystemExit(f"realms.json: two realms take the rest of {cc}")
        left = [i for i in range(len(units)) if i not in taken]
        if rest:
            parts[rest[0]["id"]] = left
        else:
            for i in left:  # a unit mostly outside the country's land (Hong Kong, Taiwan) is another country's
                if units[i][1].intersection(land[cc]).area > 0.5 * units[i][1].area:
                    log(f"  warning: {cc} unit {units[i][0]} is in no realm; its land goes to the nearest")
        country = land[cc]
        got = {rid: unary_union([units[i][1] for i in ix]).intersection(country) for rid, ix in parts.items()}
        # land the units miss (slivers along the coast and borders), each piece to the realm of the
        # nearest unit part
        import shapely
        owner_of = [rid for rid, ix in parts.items() for i in ix for _ in getattr(units[i][1], "geoms", [units[i][1]])]
        tree = shapely.STRtree([p for ix in parts.values() for i in ix for p in getattr(units[i][1], "geoms", [units[i][1]])])
        orphan = country.difference(unary_union(list(got.values())))
        extra = {}
        pieces = [p for p in getattr(orphan, "geoms", [orphan]) if not p.is_empty and p.geom_type == "Polygon"]
        if pieces:
            near = tree.query_nearest(pieces, return_distance=False, all_matches=False)
            for a, b in zip(near[0], near[1]):
                extra.setdefault(owner_of[b], []).append(pieces[a])
        for rid, ps in extra.items():
            got[rid] = unary_union([got[rid]] + ps)
        for rid, g in got.items():
            out[rid] = out[rid].union(g) if rid in out else g
    return out


def zone_realms(reg, realms, owner, split, raw):
    """{zone id: realm id or None}: its country's realm; for a split country the realm whose land
    holds its centre, else the nearest of that country's realms (in degrees: a centre off the coast)."""
    import shapely
    from shapely.geometry import Point
    by_cc = {cc: [r["id"] for r in realms if cc in r.get("units", {}) and r["id"] in raw] for cc in split}
    for cc in split:
        for rid in by_cc[cc]:
            shapely.prepare(raw[rid])
    out = {}
    for c in reg["cities"]:
        cc = c.get("country")
        if cc in owner:
            out[c["id"]] = owner[cc]
        elif cc in split and by_cc[cc]:
            p = Point(*c["center"])
            inside = [rid for rid in by_cc[cc] if raw[rid].contains(p)]
            out[c["id"]] = inside[0] if inside else min(by_cc[cc], key=lambda rid: raw[rid].distance(p))
        else:
            out[c["id"]] = None
            log(f"  warning: zone {c['id']} ({cc}) is in no realm; the map always shows it")
    return out


def near_parts(g, centres):
    """The parts of g within KEEP_KM of any centre, measured with the longitude scaled at the
    centres' mean latitude (rough, as the cut is)."""
    from shapely import affinity
    from shapely.geometry import MultiPoint, MultiPolygon
    k = math.cos(math.radians(sum(lat for _, lat in centres) / len(centres)))
    pts = MultiPoint([(lon * k, lat) for lon, lat in centres])
    parts = [p for p in getattr(g, "geoms", [g]) if p.geom_type == "Polygon"]
    keep = [p for p in parts if affinity.scale(p, xfact=k, yfact=1, origin=(0, 0)).distance(pts) * 111.32 <= KEEP_KM]
    return MultiPolygon(keep)


def rings_of(g):
    for poly in getattr(g, "geoms", [g]):
        if poly.geom_type != "Polygon":
            continue
        yield poly.exterior.coords
        for h in poly.interiors:
            yield h.coords


def simplified(raw, tol):
    """{realm id: geometry} simplified with zone_outlines' shared runs, so neighbouring realms keep
    exactly the same border; rings rebuilt into polygons by the even-odd rule."""
    import zone_outlines as zo
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    shapes = {}
    for rid, g in raw.items():
        rings = []
        for r in rings_of(g):
            pts = [zo.pt(p) for p in r]
            if len(pts) > 1 and pts[0] == pts[-1]:
                pts.pop()
            pts = [p for i, p in enumerate(pts) if p != pts[i - 1]] if len(pts) > 1 else pts
            if len(pts) >= 3:
                rings.append(pts)
        shapes[rid] = rings
    out = {}
    for rid, rings in zo.outline_rings(shapes, tol).items():
        geom = None
        for ring in sorted(rings, key=len, reverse=True):
            poly = Polygon([(x / zo.G - 180, y / zo.G - 90) for x, y in ring]).buffer(0)
            if poly.is_empty:
                continue
            geom = poly if geom is None else geom.symmetric_difference(poly)
        if geom is not None and not geom.is_empty:
            out[rid] = geom
    return out


def label_point(g):
    """The point deepest inside the largest part, found with the longitude scaled to the latitude."""
    from shapely import affinity
    from shapely.ops import polylabel
    part = max(getattr(g, "geoms", [g]), key=lambda p: p.area)
    k = math.cos(math.radians(part.centroid.y))
    p = polylabel(affinity.scale(part, xfact=k, yfact=1, origin=(0, 0)), tolerance=0.01)
    return [round(p.x / k, NDP), round(p.y, NDP)]


def feature(rid, g, zones):
    from shapely.geometry import mapping
    polys = [p for p in getattr(g, "geoms", [g]) if p.geom_type == "Polygon"]
    r = lambda ring: [[round(x, NDP), round(y, NDP)] for x, y in ring]
    coords = [[r(p.exterior.coords)] + [r(h.coords) for h in p.interiors] for p in polys]
    return {"type": "Feature", "properties": {"id": rid, "zones": zones, "label": label_point(g)},
            "geometry": {"type": "MultiPolygon", "coordinates": coords}}


def build(reg_path=REG, out=OUT, dry=False, tol=TOL_M):
    t0 = time.time()
    realms, owner, split = load_list()
    reg = json.loads(Path(reg_path).read_text(encoding="utf-8"))
    land = countries_ne(sorted(set(owner) | split))
    log(f"  countries read, {time.time() - t0:.0f} s")
    raw = split_land(realms, split, land)
    log(f"  split countries cut, {time.time() - t0:.0f} s")
    for r in realms:
        whole = [land[cc] for cc in r.get("countries", []) if cc in land]
        if whole:
            from shapely.ops import unary_union
            g = unary_union(whole + ([raw[r["id"]]] if r["id"] in raw else []))
            raw[r["id"]] = g
    zr = zone_realms(reg, realms, owner, split, raw)
    count = {}
    for rid in zr.values():
        if rid:
            count[rid] = count.get(rid, 0) + 1
    # each realm holding a zone, its parts near its zones only
    kept = {}
    for rid, n in count.items():
        if rid not in raw:
            log(f"  warning: realm {rid} has {n} zones and no land; the map always shows them")
            continue
        kept[rid] = near_parts(raw[rid], [c["center"] for c in reg["cities"] if zr[c["id"]] == rid])
    log(f"  zones placed, {time.time() - t0:.0f} s")
    shapes = simplified(kept, tol)
    order = [r["id"] for r in realms if r["id"] in shapes]
    doc = {"type": "FeatureCollection", "tol": tol, "features": [feature(rid, shapes[rid], count[rid]) for rid in order]}
    text = json.dumps(doc, separators=(",", ":"))
    import shared_lock
    out = Path(out)
    shared_lock.write_atomic(out, text)
    pts = sum(len(ring) for f in doc["features"] for poly in f["geometry"]["coordinates"] for ring in poly)
    log(f"{out.name}: {len(order)} realms, {pts:,} points, {len(text) / 1e3:.0f} KB, {time.time() - t0:.0f} s")
    for f in doc["features"]:
        log(f"  {f['properties']['id']}: {f['properties']['zones']} zones, label {f['properties']['label']}")
    if not dry:
        write_registry(reg_path, zr)
    return doc, zr


def write_registry(reg_path, zr):
    """`realm` on each registry row, read again under the registry lock so a concurrent writer's
    rows are kept; a zone registered since `zr` was worked out keeps whatever it has. A registry
    other than this tree's (a scratch copy) takes its lock beside it."""
    import shared_lock
    root = None if Path(reg_path) == REG else Path(reg_path).parent
    with shared_lock.held("registry", "realms", root=root):
        text = Path(reg_path).read_text(encoding="utf-8")
        reg = json.loads(text)
        changed = 0
        for c in reg["cities"]:
            if c["id"] not in zr:
                continue
            want = zr[c["id"]]
            if c.get("realm") != want:
                changed += 1
                if want:
                    c["realm"] = want
                else:
                    c.pop("realm", None)
        if changed:
            shared_lock.write_atomic(Path(reg_path), json.dumps(reg, indent=2, ensure_ascii=False) + "\n")
        log(f"registry: realm changed on {changed} of {len(reg['cities'])} zones")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    a = sys.argv[1:]
    build(Path(a[a.index("--registry") + 1]) if "--registry" in a else REG,
          Path(a[a.index("--out") + 1]) if "--out" in a else OUT, dry="--dry" in a)
