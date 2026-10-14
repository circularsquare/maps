"""T-287: build Toronto from tier 0 (pipeline/tier0.py) into a scratch folder, data/work/t287/,
as the city `t0tor`, for notes/t287_compare.py to set against the real pack. Never touches
data/packs/ (build_city.PACKS, WORK and read_pack's default are pointed at the scratch folder).

    ..\\venv\\Scripts\\python.exe -u trainworld/notes/t287_build.py [--finish]

What is not tier 0 here, and why: the water mask is Canada's (StatCan lakes, T-277), standing in
for T-041's OSM water, so both packs spread people over the same land; the admin totals are the
2021 census divisions' populations, the kind of district total every census publishes. The OSM
caches the real pack read from the same Ontario extract (job places, terminals, medical, parks)
are copied under the scratch city's name rather than read again: same source, same box.
"""
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
import build_city as b  # noqa: E402
import read_pack  # noqa: E402

SCRATCH = os.path.join(ROOT, "data", "work", "t287")
b.PACKS = os.path.join(SCRATCH, "packs")
b.WORK = os.path.join(SCRATCH, "work")
read_pack.load_pack.__defaults__ = (b.PACKS,)
import canada  # noqa: E402
import tier0  # noqa: E402

CITY = "t0tor"


def census_divisions(boundary):
    """The six census divisions (dissolved from their subdivisions) with their 2021 populations."""
    from shapely import make_valid, union_all
    from shapely.geometry import shape
    cfg = b.CITIES["tor"]
    t = canada.table("98100015", usecols=["DGUID", "Population and dwelling counts (5): Population, 2021 [1]"])
    cd = t[t.DGUID.str.startswith("2021A0003")]
    pub = {d[-4:]: v for d, v in zip(cd.DGUID, canada.num(cd.iloc[:, 1]))}
    out = []
    for c, name in cfg["counties"].items():
        g = union_all([make_valid(shape(f["geometry"])) for f in canada.csd_features() if f["properties"]["CSDUID"][:4] == c])
        out.append((name, g, float(pub[c])))
    return out


CONFIG = {
    "tier0": True,
    "origin": b.CITIES["tor"]["origin"],
    "water": {"cell_m": 25.0, "min_width_px": 2},
    "water_source": canada.water_source,
    "boundary_file": os.path.join(ROOT, "data", "packs", "tor.boundary.geojson"),
    "extract": "north-america/canada/ontario",
    "iso2": "CA",
    "admin_totals": census_divisions,
    "wfh": 0.247,                 # LFS May 2024, the one national-survey share tier 0 takes
    "od_share": {"CYYZ": 0.75},   # judgement for the hub, as the real pack
    "curve": {"median_km": 10.5},  # Toronto CMA median commuting distance, 2016 census (straight line)
    "region_name": "ON",
}


def main():
    b.CITIES[CITY] = CONFIG
    for d in (b.PACKS, b.WORK):
        os.makedirs(d, exist_ok=True)
    copies = [(os.path.join(canada.RAW_CA, "osm_jobs_tor.csv.gz"), tier0.raw(f"osm_jobs_{CITY}.csv.gz"))]
    for sub, stem in (("t142b", "osm_terminals"), ("t142b", "osm_medical"), ("t142b", "osm_hospital_buildings"), ("t196", "osm_parks_extract")):
        copies.append((os.path.join(b.RAW, sub, f"{stem}_tor.json"), os.path.join(b.RAW, sub, f"{stem}_{CITY}.json")))
    for src, dst in copies:
        if not os.path.exists(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(src, dst)
    if "--finish" in sys.argv:
        CONFIG["decay_pow"], CONFIG["decay_km"] = [float(x) for x in sys.argv[sys.argv.index("--finish") + 1:][:2]]
        tier0.finish(CITY)
    else:
        tier0.main(CITY)


if __name__ == "__main__":
    main()
