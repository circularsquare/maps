"""Russia stage 1: does religiondots' country outline survive the 180th meridian, and where is
Crimea in it?

    python probe_ru_outline.py

Reads the same file tools/build_regions.py reads and repeats its steps for `ru` without writing
anything: parts, bbox, opening view. Checks every ring for an edge longer than 180 degrees of
longitude (a ring drawn across the antimeridian rather than split at it, which would make the
app's ray-cast test claim half the world), and which outline holds Simferopol and Sevastopol.
"""
import json
import pathlib
import sys

from shapely.geometry import Point, shape

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = pathlib.Path(__file__).resolve().parent
SHAPES = HERE.parent / "religiondots" / "data" / "processed" / "country_shapes.geojson"
TOLERANCE, MIN_PART, FAR_DEG = 0.02, 0.002, 15
PLACES = {"Simferopol": (34.10, 44.95), "Sevastopol (inland edge)": (33.60, 44.55), "Kerch": (36.47, 45.36),
          "Dzhankoi": (34.39, 45.71), "Donetsk": (37.80, 48.00), "Luhansk": (39.31, 48.57),
          "Melitopol": (35.37, 46.85), "Kaliningrad": (20.51, 54.71),
          "Uelen (Chukotka, 169.8W)": (-169.81, 66.16), "Anadyr": (177.51, 64.73)}


def main():
    feats = json.loads(SHAPES.read_text(encoding="utf-8"))["features"]
    print(f"{SHAPES}: {len(feats)} features")
    keys = sorted(feats[0]["properties"])
    print("  property keys:", keys)
    ru = [f for f in feats if f["properties"].get("cc") == "ru"]
    print(f"  ru features: {len(ru)}")
    geom = shape(ru[0]["geometry"])
    for f in ru[1:]:
        geom = geom.union(shape(f["geometry"]))
    polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
    print(f"  ru: {geom.geom_type}, {len(polys)} polygons, bounds {tuple(round(v, 2) for v in geom.bounds)}")
    long_edges = 0
    for p in polys:
        xs = [x for x, _ in p.exterior.coords]
        for a, b in zip(xs, xs[1:]):
            if abs(a - b) > 180:
                long_edges += 1
    print(f"  ring edges jumping more than 180 deg of longitude: {long_edges}")
    east = [p for p in polys if p.bounds[2] <= -160]
    west_edge = [p for p in polys if p.bounds[2] >= 179.9]
    print(f"  polygons wholly west of 160W (Chukotka east of the meridian): {len(east)}, "
          f"area {sum(p.area for p in east):.2f} sq deg")
    print(f"  polygons touching 180E: {len(west_edge)}")

    # build_regions.py steps
    parts = []
    for p in polys:
        s = p.simplify(TOLERANCE, preserve_topology=True)
        if s.is_empty or s.area < MIN_PART:
            continue
        parts.append(s)
    main = max(polys, key=lambda p: p.area)
    mx, my = main.centroid.x, main.centroid.y
    core = [p for p in polys if p.area >= 0.05 * main.area
            and abs(p.centroid.x - mx) < FAR_DEG and abs(p.centroid.y - my) < FAR_DEG]
    view = (min(p.bounds[0] for p in core), min(p.bounds[1] for p in core),
            max(p.bounds[2] for p in core), max(p.bounds[3] for p in core))
    print(f"  build_regions: {len(parts)} parts kept, "
          f"{sum(len(p.exterior.coords) for p in parts)} points; bbox "
          f"{tuple(round(v, 2) for v in geom.bounds)}; main part centroid ({mx:.1f}, {my:.1f}); "
          f"view {tuple(round(v, 2) for v in view)} from {len(core)} core parts")
    big = sorted(polys, key=lambda p: -p.area)[:6]
    for p in big:
        print(f"     part area {p.area:8.2f}  bounds {tuple(round(v, 1) for v in p.bounds)}")

    print("\n  which outline holds:")
    for name, (x, y) in PLACES.items():
        pt = Point(x, y)
        hit = [f["properties"].get("cc") for f in feats if shape(f["geometry"]).contains(pt)]
        print(f"    {name:<28} {hit}")


if __name__ == "__main__":
    main()
