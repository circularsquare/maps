"""Write dist/regions.json: every built country's name, box, opening view and outline.

    python tools/build_regions.py

WHAT IT IS FOR. The app loads a country only when the map moves over it, so it has to know
where each country is BEFORE loading anything of it. This is that index: a few KB, fetched
once at startup, one entry per country that has a build under dist/data/<cc>/.

It also lists the line ids built in more than one country (`shared_lines`) and the OSM line
ids to fold together across countries (`line_aliases`, see `line_aliases`).

Run it after a new country's first build, and after any rebuild (a rebuild can change line
ids and twin merges). It changes nothing else, and the app still works without it for
countries already listed.

The outlines are religiondots' `country_shapes.geojson` (194 countries, cleaned and
simplified to ~200 m there), simplified again here to ~2 km: the only question asked of them
is "is this country on screen, and is the middle of the screen in it", and at 2 km Japan is
a few hundred points rather than tens of thousands. Names are Natural Earth's.
"""
import json
import pathlib
import sys

from shapely.geometry import shape, mapping

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from build_model import untrained  # noqa: E402  the twin merge's own name test
DIST = HERE / "dist"
MAPS = HERE.parent
SHAPES = MAPS / "religiondots" / "data" / "processed" / "country_shapes.geojson"
# 1:10m, not 1:110m: the small-scale file has no Singapore or Hong Kong at all.
NAMES = MAPS / "religiondots" / "data" / "geo" / "ne_10m_admin_0_countries.geojson"

# Degrees. About 2 km at Japan's latitude; see the docstring for why that is enough.
TOLERANCE = 0.02
# Parts smaller than this (square degrees, ~25 km2) are dropped: a rock offshore does not
# decide whether a country is on screen, and there are thousands of them.
MIN_PART = 0.002
# Degrees from the largest part beyond which a part is overseas, for the opening view only.
FAR_DEG = 15

# religiondots' codes where they differ from ISO 3166.
ISO = {"uk": "GB"}

# Areas added to a country's outline. Russia: the 2022-annexed railways are left out while no
# source says which trains run there (Anita, 2026-10-01; rinf_countries/ru.py ANNEX_RUNNING).
# When they come back, with Russia de facto, add ru_register.py's clip area here:
# {"ru": [HERE / "data" / "raw" / "ru" / "annex.geojson"]}. religiondots' `ru` has Crimea.
EXTRA_AREAS = {}


def main():
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
            "name": names.get(ISO.get(cc, cc.upper()), cc.upper()),
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

    path = DIST / "regions.json"
    path.write_text(json.dumps({"regions": out, "shared_lines": shared,
                                "line_aliases": dict(sorted(fold.items()))},
                               ensure_ascii=False, separators=(",", ":")),
                    encoding="utf-8")
    print(f"wrote {path}, {path.stat().st_size / 1024:.1f} KB")


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
