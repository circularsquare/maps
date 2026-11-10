"""T-2964: Israel's people by CBS 2022 census statistical area for admin_totals "IL", inside the Green Line (COD-AB's
oPt line, East Jerusalem on its far side) with the Golan (world_plan.DE_FACTO keeps it with Israel, Anita 2026-10-22).

CBS publishes the census by statistical area (about 3,000 people) on data.gov.il and its own polygons on ArcGIS Online
(`Statistical__Areas_2022`, keyed SEMEL_YISHUV + STAT_2022). A `+` in the table's StatAreaCmb merges several polygons
into one published unit; localities not split into areas are drawn whole. A unit with over half its area inside COD-AB's
oPt is beyond the line and left out (religiondots/sources/il.md 7 and 8 did the same cut and checked it point by point).

    python tools/country/israel_census.py

Reads the census table and polygons from data/raw/admin_totals/IL/ or religiondots' copies (fetching the table from
data.gov.il's plain host when neither is there), and writes data/raw/admin_totals/IL/il_statareas_2022.gpkg, layer
`units` (code, name, pop, km2), and il_beyond_line_2022.gpkg, the units left out (code, name, pop, nonarab, km2), which
tools/country/palestine_census.py reads for the settlements.
"""
import csv
import json
import os
import subprocess
import sys
import zipfile

TW = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MAPS = os.path.dirname(TW)
DIR = os.path.join(TW, "data", "raw", "admin_totals", "IL")
RD = os.path.join(MAPS, "religiondots", "data", "raw")
SOURCES = {
    "table": ("sa2022.csv", os.path.join(RD, "il", "sa2022.csv"),
              "https://data.gov.il/dataset/3bd97fde-6cc3-456d-ab63-1caad16b2b6a/resource/9a9e085f-3bc8-41df-b15f-be0daaf99e30/"
              "download/selected-data-by-localities-and-statistical-areas-2022-census.csv"),
    # the ArcGIS layer saved as GeoJSON (outSR 4326, 1,000 a page): religiondots/sources/il_geo.py fetch_statareas
    "polygons": ("statareas2022.geojson", os.path.join(RD, "il", "statareas2022.geojson"),
                 "https://services2.arcgis.com/xMRYm7cNgdR5RN6F/arcgis/rest/services/Statistical__Areas_2022/FeatureServer/0"),
    "opt": ("pse_admin_boundaries.geojson.zip", os.path.join(RD, "ps", "pse_admin_boundaries.geojson.zip"),
            "https://data.humdata.org/dataset/2caf8373-816f-458c-9913-71bddb9cab7c/resource/ca372385-4c79-4378-abf1-cb506fb98023/"
            "download/pse_admin_boundaries.geojson.zip"),
}
OUT = os.path.join(DIR, "il_statareas_2022.gpkg")
BEYOND = os.path.join(DIR, "il_beyond_line_2022.gpkg")
POLYGONS = 3857
DROP_IF_INSIDE = 0.5
# points whose side of the line is not in dispute: the file must put each where expected (lon, lat, inside oPt)
PROBES = [("West Jerusalem", 35.2137, 31.7833, False), ("Shu'afat", 35.2340, 31.8080, True), ("Old City", 35.2340, 31.7767, True),
          ("Gilo", 35.1930, 31.7300, True), ("Tel Aviv", 34.7818, 32.0853, False), ("Ma'ale Adumim", 35.2980, 31.7730, True),
          ("Majdal Shams", 35.7660, 33.2680, False), ("Gaza City", 34.4667, 31.5000, True)]


def source(key):
    name, mirror, url = SOURCES[key]
    here = os.path.join(DIR, name)
    if os.path.exists(here):
        return here
    if os.path.exists(mirror):
        return mirror
    if key == "polygons":
        raise SystemExit(f"{here}: missing; save {url} as GeoJSON (religiondots/sources/il_geo.py fetch_statareas)")
    os.makedirs(DIR, exist_ok=True)
    subprocess.run(["curl", "-sS", "-L", "--fail", "-o", here, url], check=True)
    return here


def canon(cmb):
    """`2312+2311+2312` -> `2312+2311` (CBS repeats a code in some Jerusalem strings)."""
    seen, out = set(), []
    for part in str(cmb).split("+"):
        part = part.strip()
        if part and part not in seen:
            seen.add(part)
            out.append(part)
    return "+".join(out)


def table():
    """({(locality, area): unit}, {unit: (pop, nonarab)}, localities drawn whole, localities split)."""
    rows = list(csv.DictReader(open(source("table"), encoding="utf-8-sig", newline="")))
    split = {r["LocalityCode"].strip() for r in rows if r["StatArea"].strip()}
    parts, pop = {}, {}
    for r in rows:
        loc, cmb = r["LocalityCode"].strip(), canon(r["StatAreaCmb"])
        if not loc or (loc in split and not cmb):
            continue  # the national row; a split locality's own total row
        key = f"{loc}_{cmb}" if cmb else loc
        for part in cmb.split("+") if cmb else []:
            parts[(loc, part)] = key
        p = float(r["pop_approx"] or 0)
        # religion: 1 Jews, 2 Muslims, 3 Christians, 4 Druze, 5 mixed (the unit's main group); "non-Arab" as
        # religiondots' xs entry: the Jewish-majority units
        pop[key] = (p, p if r["religion"].strip() == "1" else 0.0)
    return parts, pop, split


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    import geopandas as gpd
    import shapely
    from shapely.geometry import shape, Point
    parts, pop, split = table()
    g = gpd.read_file(source("polygons"))
    if len(g) != POLYGONS:
        raise SystemExit(f"{len(g)} polygons, expected {POLYGONS}: CBS revised the layer; check the join first")
    g = g.set_crs(4326) if g.crs is None else g.to_crs(4326)

    def key(r):
        if any(v is None or v != v for v in (r["SEMEL_YISHUV"], r["STAT_2022"])):
            return None
        loc, stat = str(int(r["SEMEL_YISHUV"])), str(int(r["STAT_2022"]))
        return parts.get((loc, stat)) if loc in split else (loc if loc in pop else None)
    g["code"] = g.apply(key, axis=1)
    print(f"{len(g):,} polygons, {g.code.isna().sum()} with no census unit")
    g = g[g.code.notna()]
    names = dict(zip(g.code, g.SHEM_YISHUV_ENGLISH.fillna("").str.title()))
    u = g.dissolve(by="code", as_index=False)[["code", "geometry"]]
    u["name"] = u.code.map(names)
    u["pop"] = u.code.map(lambda c: pop[c][0])
    u["nonarab"] = u.code.map(lambda c: pop[c][1])
    z = zipfile.ZipFile(source("opt"))
    opt = shapely.union_all([shapely.make_valid(shape(f["geometry"])) for f in json.loads(z.read("pse_admin0.geojson"))["features"]])
    bad = [p for p in PROBES if opt.contains(Point(p[1], p[2])) != p[3]]
    if bad:
        raise SystemExit(f"COD-AB's oPt is not the file this was written against: {bad}")
    eq = u.to_crs(6933)
    share = eq.geometry.intersection(gpd.GeoSeries([opt], crs=4326).to_crs(6933).iloc[0]).area / eq.geometry.area
    u["km2"] = (eq.geometry.area / 1e6).round(3)
    out, beyond = u[share <= DROP_IF_INSIDE], u[share > DROP_IF_INSIDE]
    print(f"{len(u):,} units, census {u['pop'].sum():,.0f}; inside the line {len(out):,} ({out['pop'].sum():,.0f}), beyond it "
          f"{len(beyond)} ({beyond['pop'].sum():,.0f}, Jewish-majority units {beyond.nonarab.sum():,.0f}); "
          f"{int(((share > 0.01) & (share <= DROP_IF_INSIDE)).sum())} straddle and stay")
    os.makedirs(DIR, exist_ok=True)
    for frame, path, cols in ((out, OUT, ["code", "name", "pop", "km2"]), (beyond, BEYOND, ["code", "name", "pop", "nonarab", "km2"])):
        tmp = path + ".part.gpkg"
        frame[cols + ["geometry"]].to_file(tmp, layer="units", driver="GPKG")
        os.replace(tmp, path)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
