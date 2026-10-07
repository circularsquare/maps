"""
Where the map draws nothing because no source counted the people there, for the viewer's grey
hatching. religiondots' not_drawn.py run on languagedots' countries.

    python not_drawn.py                     # every country; reuses cached ones whose inputs are unchanged
    python not_drawn.py --countries ss,ma   # recompute just these, keep the rest from the cache
    python not_drawn.py --fresh             # ignore the cache

    -> data/processed/not_drawn.geojson   one feature per hatched area, {cc, kind, name}

Anita, 2026-10-06: hatching for regions with no data, as religiondots does (South Sudan's three
unsampled states, Western Sahara east of the berm).

THE METHOD IS religiondots'. Its docstring (../religiondots/not_drawn.py) has the reasoning; the
short version: Kontur's r6 hexes say where people live, a populated cell no counted unit's
placement polygon reaches is not drawn, every cell of a country's Natural Earth outline takes its
nearest populated cell's answer, and small areas are dropped. Territory a source names as left
out (Abkhazia, Transnistria, Karabakh) is added whole. A Natural Earth feature that no entry
here takes (Western Sahara's strip east of the berm, Northern Cyprus) is hatched whole unless a
built neighbour's dots already hold half its people.

The functions are religiondots' own, loaded by path (rdlink's rule: never by sys.path, its
module names collide with ours) and read-only. What differs is everything that names a file:
the countries are languagedots', the dots are languagedots' 1:1,000 ones (there is no 1:10,000
edition here, so a neighbour's dots are counted at 1:1,000 too), and the cache and output are
in languagedots/data.

A built entry with no Natural Earth outline of its own (`xs`, the West Bank settlements) is
drawn wherever its counted polygons are, so they count as drawn ground for whichever outline
they sit in, as religiondots' `territory=False` does.

Reads every placement layer, so a full fresh run takes a while; the cache keeps later runs to
the countries whose files changed.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "6")      # she is using the box

import shapely
import shapely.geometry

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rdlink  # noqa: E402

PROC = HERE / "data" / "processed"
OUT = PROC / "not_drawn.geojson"
CACHE = HERE / "data" / "not_drawn_cache.json"
VERSION = 1


def _rd():
    """religiondots' not_drawn and country_shapes, loaded by path. not_drawn imports
    country_shapes by name inside its functions, so that name is pointed at religiondots' copy;
    languagedots has no module of that name."""
    cs = rdlink._load("country_shapes", rdlink.RD / "country_shapes.py")
    sys.modules["country_shapes"] = cs
    nd = rdlink._load("not_drawn", rdlink.RD / "not_drawn.py")
    return nd, cs


_DOTS = {}


def dots(cc):
    if cc not in _DOTS:
        import numpy as np
        p = PROC / f"dots_{cc}.geojson"
        xy = np.zeros((0, 2))
        if p.exists():
            fs = json.loads(p.read_text(encoding="utf-8"))["features"]
            if fs:
                xy = np.array([f["geometry"]["coordinates"] for f in fs], dtype=float)
        _DOTS[cc] = xy
    return _DOTS[cc]


def _mtime(p):
    try:
        return int(os.path.getmtime(p))
    except (OSError, TypeError):
        return None


def dot_check(nd, cc, geom, kpts, outl, named=None):
    """religiondots' dot_check on 1:1,000 dots throughout: a part where this country's dots, or a
    neighbour's, stand for a quarter or more of Kontur's people is drawn after all. NAMED
    territory is never dropped."""
    lon, lat, pop = kpts
    keep, dropped = [], []
    for part in shapely.get_parts(geom):
        if named is not None and shapely.area(shapely.intersection(part, named)) > 0.5 * part.area:
            keep.append(part)
            continue
        w, s, e, n = part.bounds
        people_drawn = 0
        for c2, o in outl.items():
            ow, os_, oe, on = o.bounds
            if c2 != cc and (ow > e or oe < w or os_ > n or on < s):
                continue
            people_drawn += nd._dots_in(part, dots(c2)) * 1000
        k = (lon >= w) & (lon <= e) & (lat >= s) & (lat <= n)
        ppl = float(pop[k][shapely.contains_xy(part, lon[k], lat[k])].sum()) if k.any() else 0.0
        if people_drawn and people_drawn >= 0.25 * ppl:
            c = part.centroid
            dropped.append(f"at {c.x:.2f},{c.y:.2f}: dots for {people_drawn:,} against "
                           f"~{ppl:,.0f} by Kontur")
        else:
            keep.append(part)
    return (shapely.union_all(keep) if keep else None), dropped


def save(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
    for attempt in range(60):      # Windows refuses the swap while a reader holds the file
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 59:
                raise SystemExit(f"could not replace {path.name}; the new output is in {tmp.name}")
            time.sleep(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--countries", help="comma list to recompute; others come from the cache")
    ap.add_argument("--fresh", action="store_true", help="ignore the cache")
    args = ap.parse_args()

    nd, cs = _rd()
    from countries import COUNTRIES
    t0 = time.time()
    outl, rest = nd.outlines(COUNTRIES)
    ne_m = _mtime(cs.SRC)
    named = nd._named()
    # religiondots names Abkhazia, South Ossetia, Transnistria and Karabakh as never counted;
    # here an entry that draws one says so in `drawn_named` (2026-10-06, ge/md/az), and it leaves
    # the named list, so only the people test decides whether any of it is still hatched
    for cc, cfg in COUNTRIES.items():
        drop = set(cfg.get("drawn_named", ()))
        if drop and cc in named:
            named[cc] = [(p, k, tuple(n for n in names if n not in drop))
                         for p, k, names in named[cc]]
            named[cc] = [x for x in named[cc] if x[2]]

    cache = {}
    if CACHE.exists() and not args.fresh:
        try:
            cache = json.loads(CACHE.read_text(encoding="utf-8"))
        except ValueError:
            cache = {}
    if cache.get("version") != VERSION:
        cache = {}
    entries = cache.get("countries", {})
    force = set(args.countries.split(",")) if args.countries else set()

    def key_of(cc):
        cfg = COUNTRIES[cc]
        return [VERSION, ne_m, _mtime(HERE / "countries" / f"{cc}.py"), _mtime(cfg.get("place")),
                _mtime(PROC / f"dots_{cc}.geojson"),
                repr([(Path(p).name, k, list(v)) for p, k, v in named.get(cc, [])])]

    def store():
        save(CACHE, {"version": VERSION, "countries": entries})

    floating = [cc for cc in COUNTRIES if cc not in outl]
    todo = [cc for cc in COUNTRIES if cc in outl
            and (cc in force if force else entries.get(cc, {}).get("key") != key_of(cc))]
    kpts = nd.kontur_points() if todo else None
    if todo:
        print(f"{len(todo)} to compute: {', '.join(todo)}  (Kontur read in {time.time() - t0:.0f}s)")
    if floating:
        print(f"no outline of their own, drawn wherever they are: {', '.join(floating)}")

    def others_near(cc, o):
        b = shapely.buffer(shapely.envelope(o), 0.2)
        near = [g for c2, g in outl.items() if c2 != cc and g.intersects(b)]
        return shapely.union_all(near) if near else None

    float_geoms = []
    if todo:
        for cc in floating:
            try:
                float_geoms.extend(nd.drawn_polygons(cc, COUNTRIES[cc]))
            except (SystemExit, Exception) as e:  # noqa: BLE001
                print(f"  !! {cc}: {e}; its polygons not counted as drawn")
    for i, cc in enumerate(todo, 1):
        t = time.time()
        try:
            drawn = nd.drawn_polygons(cc, COUNTRIES[cc])
        except (SystemExit, Exception) as e:  # noqa: BLE001
            print(f"  !! {cc}: {e}; left as it was")
            continue
        o = outl[cc]
        near_float = [g for g in float_geoms if g.intersects(o)]
        geom, ppl = nd.part_of(cc, o, drawn, kpts, near_float, nd.named_geoms(cc, named),
                               others_near(cc, o))
        entries[cc] = {"key": key_of(cc), "people": round(ppl),
                       "geometry": None if geom is None else json.loads(shapely.to_geojson(geom))}
        print(f"  [{i}/{len(todo)}] {cc}: " + ("nothing" if geom is None else
              f"{shapely.area(geom):.3f} deg2, ~{ppl:,.0f} people by Kontur")
              + f"  ({time.time() - t:.0f}s)", flush=True)
        store()

    feats, dot_dropped = [], []
    for cc in COUNTRIES:
        e = entries.get(cc)
        if not e or not e.get("geometry") or cc not in outl:
            continue
        raw = shapely.geometry.shape(e["geometry"])
        w, s, ee, n = raw.bounds
        near = sorted(c2 for c2, o in outl.items()
                      if not (o.bounds[0] > ee or o.bounds[2] < w or o.bounds[1] > n or o.bounds[3] < s))
        kept_key = [e["key"]] + [(c2, _mtime(PROC / f"dots_{c2}.geojson")) for c2 in near]
        if e.get("kept_key") != json.loads(json.dumps(kept_key)):
            if kpts is None:
                kpts = nd.kontur_points()
            ng = nd.named_geoms(cc, named)
            kept, dropped = dot_check(nd, cc, raw, kpts, outl, shapely.union_all(ng) if ng else None)
            e["kept"] = None if kept is None else json.loads(shapely.to_geojson(kept))
            e["dropped"] = dropped
            e["kept_key"] = kept_key
            store()
        dot_dropped += [f"{cc} {d}" for d in e.get("dropped", [])]
        if not e.get("kept"):
            continue
        g = shapely.geometry.shape(e["kept"])
        # another entry's outline is that entry's ground
        others = [o for c2, o in outl.items() if c2 != cc and o.intersects(g)]
        if others:
            g = shapely.difference(g, shapely.union_all(others))
        g = nd._clean(g)
        if g is None:
            continue
        feats.append({"type": "Feature",
                      "properties": {"cc": cc, "kind": "part", "name": COUNTRIES[cc]["name"]},
                      "geometry": nd._round(json.loads(shapely.to_geojson(g)))})

    # A Natural Earth feature no entry takes, hatched unless the built countries around it draw
    # half its people already (Northern Cyprus, counted by `cy`).
    built_union = shapely.union_all(list(outl.values())) if outl else None

    def drawn_in(g):
        w, s, e, n = g.bounds
        hits = 0
        for c2, o in outl.items():
            ow, os_, oe, on = o.bounds
            if ow > e + 1 or oe < w - 1 or os_ > n + 1 or on < s - 1:
                continue
            hits += nd._dots_in(g, dots(c2))
        return hits * 1000

    # Natural Earth's NAME is a label abbreviation ("W. Sahara"); the card shows the full name
    admin = {f["properties"].get("ADM0_A3"): f["properties"].get("ADMIN")
             for f in json.loads(cs.SRC.read_text(encoding="utf-8"))["features"]}
    whole, by_neighbour = [], []
    for a3, name, pop, g in rest:
        name = admin.get(a3) or name
        if a3 in nd.NOBODY or not pop or pop <= 0:
            continue
        if built_union is not None:
            g = shapely.difference(g, built_union)
        g = nd._clean(shapely.simplify(g, nd.SIMPLIFY / 4), min_km2=0)
        if g is None:
            continue
        drawn_people = drawn_in(g)
        if drawn_people >= 0.5 * pop:
            by_neighbour.append(f"{name} ({drawn_people:,} drawn of {int(pop):,})")
            continue
        whole.append((name, pop))
        feats.append({"type": "Feature", "properties": {"cc": None, "kind": "whole", "name": name},
                      "geometry": nd._round(json.loads(shapely.to_geojson(g)))})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    save(OUT, {"type": "FeatureCollection", "features": feats})
    parts = [f["properties"]["cc"] for f in feats if f["properties"]["kind"] == "part"]
    print(f"\nparts of built countries ({len(parts)}): " + ", ".join(
        f"{cc} ~{entries[cc]['people']:,}" for cc in parts))
    print(f"no entry here ({len(whole)}): " + ", ".join(
        f"{n} ({int(p):,})" for n, p in sorted(whole, key=lambda x: -x[1])))
    if by_neighbour:
        print("not hatched, a built neighbour draws them: " + ", ".join(by_neighbour))
    if dot_dropped:
        print("dropped, the dots say they are drawn:\n  " + "\n  ".join(dot_dropped))
    missing = [cc for cc in COUNTRIES if cc in outl and cc not in entries]
    if missing:
        print(f"not computed yet ({len(missing)}), so not hatched: {', '.join(missing)}")
    print(f"wrote {OUT.name}  ({len(feats)} features, {OUT.stat().st_size / 1024:.0f} KB) "
          f"in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
