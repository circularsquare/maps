"""Saudi Arabia: boundaries and 2022 census populations for the 13 administrative regions.

Writes data/geo/sa/sa_regions.gpkg and data/geo/sa/sa_lookup.csv.

  * **boundaries**: COD-AB `cod-ab-sau` v01 (OCHA ROMENA from GADM, boundaries last edited 30
    November 2015, reviewed on HDX 19 December 2024), `sau_admin1.geojson`, 13 regions. Saudi
    Arabia's 13 regions have not changed since 1993. The pcodes skip SA13 (`'Asir` is SA14).
  * **populations**: GASTAT's 2022 census, Saudis and non-Saudis per region, as GLMM mirrors the
    census portal's table (`REGION_2022`). `sources/sa.py` reads that page and asserts it equals
    these constants, and holds every region's non-Saudi share against the census report's own
    Figure 11 (all 13 within 0.05 points).

## CHECKS

  1. COD has 13 features in EPSG:4326, and the name under every pcode is the expected one;
  2. COD's own `area_sqkm` against the polygons' equal-area area, per region (a torn or swapped
     geometry fails it);
  3. every region's `center_lat`/`center_lon` point lies inside its own polygon.

**Not checked:** area against GASTAT's own figure per region, which was not found in the census
report. The Kontur rank witness for the join is in `sources/sa_grid.py`.

Usage:
    python sources/sa_geo.py --fetch    COD-AB geojson zip (3.5 MB)
    python sources/sa_geo.py            rebuild from data/raw/sa/
"""

import io
import os
import sys
import urllib.request
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ.setdefault("OMP_NUM_THREADS", "6")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "sa")
OUT_DIR = os.path.join(ROOT, "data", "geo", "sa")
OUT_REGIONS = os.path.join(OUT_DIR, "sa_regions.gpkg")
LOOKUP = os.path.join(OUT_DIR, "sa_lookup.csv")

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/128.0 Safari/537.36"}
COD_URL = ("https://data.humdata.org/dataset/41ce9023-1d21-4549-a485-94316200aba0/resource/"
           "99111dda-821b-47ba-ac47-771c9ed5184a/download/sau_admin_boundaries.geojson.zip")
COD_ZIP = os.path.join(RAW, "sau_admin_boundaries.geojson.zip")
EQUAL_AREA = "EPSG:6933"
AREA_TOL = 0.01

# pcode -> (the name this map prints, COD adm1_name)
REGIONS = {
    "SA01": ("Riyadh", "Ar Riyad"),
    "SA02": ("Makkah", "Makkah"),
    "SA03": ("Madinah", "Al Madinah"),
    "SA04": ("Eastern Province", "Ash Sharqiyah"),
    "SA05": ("Qassim", "Al Quassim"),
    "SA06": ("Hail", "Ha'il"),
    "SA07": ("Tabuk", "Tabuk"),
    "SA08": ("Northern Borders", "Al Hudud ash Shamaliyah"),
    "SA09": ("Jazan", "Jizan"),
    "SA10": ("Najran", "Najran"),
    "SA11": ("Al Bahah", "Al Bahah"),
    "SA12": ("Al Jawf", "Al Jawf"),
    "SA14": ("Asir", "'Asir"),
}

# pcode -> (Saudis, non-Saudis), census 2022 (10 May 2022), GLMM's copy of the portal's table
REGION_2022 = {
    "SA01": (4_439_210, 4_152_538),
    "SA02": (4_153_723, 3_867_740),
    "SA03": (1_352_146, 785_837),
    "SA04": (2_949_854, 2_175_400),
    "SA05": (926_490, 409_689),
    "SA06": (527_885, 218_521),
    "SA07": (637_601, 248_435),
    "SA08": (271_358, 102_219),
    "SA09": (1_002_779, 402_218),
    "SA10": (394_976, 197_324),
    "SA11": (251_288, 87_886),
    "SA12": (440_264, 155_558),
    "SA14": (1_444_688, 579_597),
}
TOTAL = 32_175_224


def fetch():
    os.makedirs(RAW, exist_ok=True)
    if os.path.exists(COD_ZIP) and os.path.getsize(COD_ZIP) > 1_000_000:
        print(f"  have {os.path.basename(COD_ZIP)}")
        return
    print("  GET", COD_URL)
    with urllib.request.urlopen(urllib.request.Request(COD_URL, headers=UA), timeout=600) as r:
        data = r.read()
    if not data.startswith(b"PK"):
        raise SystemExit(f"{COD_URL} is not a zip")
    with open(COD_ZIP + ".part", "wb") as fh:
        fh.write(data)
    os.replace(COD_ZIP + ".part", COD_ZIP)


def main():
    import geopandas as gpd
    import pandas as pd

    if "--fetch" in sys.argv or not os.path.exists(COD_ZIP):
        fetch()
    if sum(s + n for s, n in REGION_2022.values()) != TOTAL or set(REGION_2022) != set(REGIONS):
        raise SystemExit("REGION_2022 does not sum to the census or does not match REGIONS")

    with zipfile.ZipFile(COD_ZIP) as zf:
        a1 = gpd.read_file(io.BytesIO(zf.read("sau_admin1.geojson")))
    if len(a1) != 13 or a1.crs.to_epsg() != 4326:
        raise SystemExit(f"COD-AB: {len(a1)} ADM1 features, crs {a1.crs}; expected 13 in 4326")
    got = dict(zip(a1["adm1_pcode"], a1["adm1_name"]))
    bad = [pc for pc, (_n, cod) in REGIONS.items() if got.get(pc) != cod]
    if bad or set(got) != set(REGIONS):
        raise SystemExit(f"COD names under these pcodes are not the expected ones: {bad}")
    print("  COD-AB: 13 regions, every name under its pcode as expected")

    a1 = a1.set_index("adm1_pcode")
    area = a1.to_crs(EQUAL_AREA).geometry.area / 1e6
    rel = (area / a1["area_sqkm"].astype(float))
    worst = float((rel - 1).abs().max())
    print(f"  polygon area against COD's own area_sqkm: worst {worst:.3%}")
    if worst > AREA_TOL:
        raise SystemExit("a region's polygon area disagrees with COD's area_sqkm")
    pts = gpd.GeoSeries(gpd.points_from_xy(a1["center_lon"], a1["center_lat"]), index=a1.index,
                        crs=4326)
    outside = [pc for pc in REGIONS if not a1.geometry[pc].contains(pts[pc])]
    if outside:
        raise SystemExit(f"COD centre points outside their own region: {outside}")
    print("  every region's COD centre point lies inside its polygon")

    lut = pd.DataFrame([dict(geo_id=pc, unit=pc, name=n, pop=s + ns, saudi=s, non_saudi=ns,
                             cod_area=round(float(area[pc]), 1))
                        for pc, (n, _c) in REGIONS.items() for s, ns in [REGION_2022[pc]]])
    os.makedirs(OUT_DIR, exist_ok=True)
    g = a1.reset_index()[["adm1_pcode", "geometry"]].rename(columns={"adm1_pcode": "unit"})
    g = g.merge(lut[["unit", "name", "pop"]], on="unit", how="inner", validate="1:1")
    g[["unit", "name", "pop", "geometry"]].to_file(OUT_REGIONS, layer="regions", driver="GPKG")
    lut.to_csv(LOOKUP, index=False, encoding="utf-8")
    print(f"\nwrote {OUT_REGIONS} (13) and {LOOKUP} ({int(lut['pop'].sum()):,} people)")


if __name__ == "__main__":
    main()
