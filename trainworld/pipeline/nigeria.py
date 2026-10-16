"""T-459: Nigeria's inputs for a tier 0 city (Lagos first). Global sources only: the outline is OSM's
local government areas; population, jobs and everything else are tier 0's.

- **Outline and units**: OSM boundary=administrative relations at admin_level 6 (local government
  areas) read from the Geofabrik extract, the units whose representative point lies in one of the
  config's states (admin_level 4) plus named LGAs of neighbouring states, clipped to land by the OSM
  coastline polygons and cached as a small GeoJSON (the .pbf is deleted after routes.py).

    python pipeline/nigeria.py probe <id>    (lists the admin_level 4 and 6 relations in the config's box)
    python pipeline/nigeria.py <id>          (writes the units file, prints each unit's GHSL people)
"""
import json
import os
import sys

import numpy as np
import shapely
from shapely import make_valid, union_all
from shapely.geometry import mapping, shape

import build_city as b

# id -> (states whose LGAs are all taken (OSM admin_level 4 name), extra LGAs by OSM name, probe box)
UNITS = {
    # Lagos State whole, and the three Ogun LGAs Lagos's built-up area runs into (Africapolis 2020's
    # Lagos agglomeration: Ota, Ifo, Mowe and Ibafo)
    "lag": (["Lagos"], ["Ado Odo/Ota", "Ifo", "Obafemi Owode"], (2.6, 6.3, 4.4, 7.2)),
}


def raw(name):
    return os.path.join(b.RAW, "tier0", "nigeria", name)


def _admin(city, levels):
    import pyogrio
    import routes
    pbf = routes.download(b.CITIES[city]["extract"])
    w, s, e, n = UNITS[city][2]
    lv = ",".join(f"'{x}'" for x in levels)
    return pyogrio.read_dataframe(pbf, layer="multipolygons", columns=["osm_id", "name", "admin_level", "boundary"],
                                  where=f"boundary = 'administrative' AND admin_level IN ({lv})", bbox=(w, s, e, n))


def probe(city):
    df = _admin(city, ("4", "6"))
    for _, r in df.sort_values(["admin_level", "name"]).iterrows():
        p = r.geometry.representative_point()
        print(f"  {r.admin_level} {r['name']!r} r{r.osm_id} at {p.x:.3f},{p.y:.3f}")


def units_file(city):
    out = raw(f"{city}_units.geojson")
    if os.path.exists(out):
        return out
    import osm_water
    states, extra, _ = UNITS[city]
    df = _admin(city, ("4", "6"))
    st = union_all([make_valid(g) for g in df[(df.admin_level == "4") & df["name"].isin(states)].geometry])
    if st.is_empty:
        raise SystemExit(f"{city}: states {states} not found at admin_level 4")
    lga = df[df.admin_level == "6"]
    feats, picked = [], []
    for _, r in lga.iterrows():
        g = make_valid(r.geometry).buffer(0)
        if r["name"] in extra or st.contains(g.representative_point()):
            picked.append((r["name"], g, f"OSM r{r.osm_id}"))
    missing = set(extra) - {p[0] for p in picked}
    if missing:
        raise SystemExit(f"{city}: LGAs {sorted(missing)} not found at admin_level 6")
    w, s, e, n = union_all([g for _, g, _ in picked]).bounds
    sea = union_all(osm_water.sea((w - 0.01, s - 0.01, e + 0.01, n + 0.01)))
    for name, g, src in picked:
        g = g.difference(sea).buffer(0)
        if g.geom_type == "MultiPolygon":  # slivers left along the clipped shore
            g = shapely.MultiPolygon([p for p in g.geoms if p.area > 1e-6])
        feats.append({"type": "Feature", "properties": {"name": name, "source": src}, "geometry": mapping(g)})
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out + ".part", "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f, ensure_ascii=False)
    os.replace(out + ".part", out)
    return out


def units(city):
    with open(units_file(city), encoding="utf-8") as f:
        return [(ft["properties"], shape(ft["geometry"])) for ft in json.load(f)["features"]]


def boundary_file(city):
    """The units dissolved, for tier0's `boundary_file`; its path."""
    out = raw(f"{city}_outline.geojson")
    if not os.path.exists(out):
        g = union_all([u[1] for u in units(city)]).buffer(0)
        # holes are lagoon no LGA relation covers (Lagos: one by Ikorodu); the water mask has them
        g = shapely.MultiPolygon([shapely.Polygon(p.exterior) for p in getattr(g, "geoms", [g])])
        with open(out + ".part", "w", encoding="utf-8") as f:
            json.dump({"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {"city": city},
                                                                   "geometry": mapping(g)}]}, f)
        os.replace(out + ".part", out)
    return out


def ghsl_by_unit(city):
    """GHSL people in each unit (pixel centres), before any build: the check T-292 proposed."""
    import tier0
    us = units(city)
    g = union_all([u[1] for u in us])
    arr, X0, Y0 = tier0.ghsl_window("pop", g.bounds)
    _, inv = tier0._moll()
    iy, ix = np.nonzero(arr > 0)
    xs, ys = X0 + (ix + 0.5) * tier0.PIX_M, Y0 - (iy + 0.5) * tier0.PIX_M
    lon, lat = inv.transform(xs, ys)
    pts = shapely.points(lon, lat)
    v = arr[iy, ix]
    tot = 0.0
    for p, geom in us:
        shapely.prepare(geom)
        have = float(v[shapely.contains(geom, pts)].sum())
        tot += have
        print(f"  {p['name']}: GHSL {have:,.0f}")
    print(f"  {len(us)} units: GHSL E2020 {tot:,.0f}")


if __name__ == "__main__":
    if sys.argv[1] == "probe":
        probe(sys.argv[2])
        sys.exit()
    city = sys.argv[1]
    us = units(city)
    g = union_all([u[1] for u in us])
    km2 = 111.32 * 110.57 * 0.994  # degrees squared to km2 near 6.5 N
    print(f"{city}: {len(us)} units, outline {g.geom_type}, {len(getattr(g, 'geoms', [g]))} parts, about {g.area * km2:,.0f} km2, "
          f"bounds {tuple(round(v, 3) for v in g.bounds)}")
    for p, geom in us:
        parts = list(getattr(geom, "geoms", [geom]))
        print(f"  {p['name']}: {len(parts)} parts, {sum(len(q.interiors) for q in parts)} holes, about {geom.area * km2:,.1f} km2")
    ghsl_by_unit(city)
