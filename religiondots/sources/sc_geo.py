"""Seychelles: boundaries for the 26 districts of the 2010 census.

Writes data/geo/sc/sc_districts.gpkg (layers `districts` and `perseverance`) and
data/geo/sc/sc_lookup.csv.

OCHA COD-AB Seychelles (`cod-ab-syc`), `syc_admbnda_adm3_nbs2010`, the shapefile bundle. HDX
describes it as NBS's own 2010 census district boundaries for Mahé, Praslin and La Digue, with
an `Other Islands` feature added from GAUL. geoBoundaries' SYC ADM3 is a copy of it. It has 27
features where the census has 26 districts, and two of them need moving before they match.

**PERSEVERANCE ISLAND IS PART OF ENGLISH RIVER IN 2010.** COD draws the reclaimed island as its
own feature (`SC1127PI`), because it became a district later (Ile Perseverance, 2022 census).
The 2010 census has no such district. Table 2.3 of the 2010 report prints each district's area,
and it settles where the island belonged: every Mahé district measures within 10% of its
printed area except English River, which is **1.38 km2 against 2.3 printed**, and with the
Perseverance polygon added it is **2.32 km2**. `check_areas` asserts both halves.
The island was lived on only after the census: its first houses were built for the Indian
Ocean Games of August 2011, and by November 2011 there were fewer than 600 (a Seychelles blog
of that month, `seychellesreality.blogspot.com/2011/11/`); the 2022 census counts 5,410
people there. So its polygon is drawn as part of English River here and `sc_grid.py` keeps
the 2023 grid's Perseverance cells out of the placement weights.

**SIX INNER ISLANDS COUNT WITH LA DIGUE, NOT WITH OTHER ISLANDS.** The census's `La Digue` is
`La Digue (& Inner Islands)` in Table 2.3, and its footnote 6 lists Marianne, Félicité, Grande
Soeur, Petite Soeur, Cocos, Silhouette and North. COD's `Other Islands` feature (from GAUL)
holds all of them. `INNER` moves the parts by position, asserting that Silhouette, North,
Félicité and Marianne are each found exactly once and at the expected size; the witness is
Table 2.3's area again, **36.4 km2 printed for La Digue and its inner islands**, against 9.8
km2 for COD's La Digue on its own. Frégate, Denis and Bird are not in footnote 6 and stay in
Other Islands.

**THE NAMES AND THE CODES BOTH JOIN.** COD's ADM3 pcode carries the ISO 3166-2:SC number
(`SC1116ER` is SC-16, English River), so each district is matched on its name and on that
number, and the two keys must agree. There are no twins.

Usage:
    python sources/sc_geo.py --fetch    one ~24 MB zip from HDX
    python sources/sc_geo.py            rebuild from data/raw/sc/
"""

import os
import re
import sys
import unicodedata
import zipfile

os.environ.setdefault("OMP_NUM_THREADS", "6")          # [[feedback_cap_cpu]]
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import geopandas as gpd
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

RAW = os.path.join(ROOT, "data", "raw", "sc")
SHP_DIR = os.path.join(RAW, "cod_shp")
OUT_DIR = os.path.join(ROOT, "data", "geo", "sc")
OUT = os.path.join(OUT_DIR, "sc_districts.gpkg")
LOOKUP = os.path.join(OUT_DIR, "sc_lookup.csv")

ZIP_NAME = "syc_adm_nbs2010_shp.zip"
ZIP_URL = ("https://data.humdata.org/dataset/9ac1737f-cd33-4458-86e8-aec90eeffda2/resource/"
           "de3bca8b-f522-46ba-b4d2-e1c52034fcc0/download/syc_adm_nbs2010_shp.zip")
ZIP_SIZE = 23_768_800          # as served 2026-09-14; HDX's metadata says 23,770,193
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) religiondots/1.0"}

EXPECTED_FEATURES = 27
EXPECTED_UNITS = 26

# Africa Albers Equal Area Conic, COD's recommended projection. One UTM zone cannot measure a
# country that runs from Aldabra at 46°E to Mahé at 55.5°E.
AREA_CRS = "ESRI:102022"

# Inner islands in COD's `Other Islands` that the census counts with La Digue (report Table
# 2.3, footnote 6): (name, lat, lon, min km2, max km2). Positions from the parts COD draws; a
# part is moved when its centroid is within RADIUS_KM of one of these.
INNER_NAMED = [
    ("Silhouette", -4.487, 55.230, 15.0, 25.0),
    ("North", -4.393, 55.245, 1.5, 2.5),
    ("Félicité", -4.327, 55.875, 2.0, 3.2),
    ("Marianne", -4.342, 55.921, 0.6, 1.3),
]
# Grande Soeur, Petite Soeur and Cocos are small enough that COD may draw them as several
# slivers or none; every part whose centroid is inside this box goes with them.
INNER_BOX = (-4.31 - 0.03, 55.84, -4.27, 55.90)       # (lat_min, lon_min, lat_max, lon_max)
RADIUS_KM = 3.0

AREA_TOLERANCE = 0.12
# Printed areas this file does not assert, with the reason. Praslin's two districts print
# 50.8 km2 between them for an island group of about 42 km2 (Praslin 38.5, Curieuse 2.9,
# Cousin, Cousine and Aride); Grand Anse Praslin measures 31% under and Baie Sainte Anne 10%,
# so the printed figures are wrong and not the polygons. Other Islands prints 179.6 km2 for
# atolls whose land and lagoon edges GAUL draws differently.
AREA_UNCHECKED = {"Grand Anse Praslin", "Other Islands"}


def fold(s):
    s = unicodedata.normalize("NFKD", str(s).strip().lower())
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return " ".join(re.sub(r"[^a-z0-9]+", " ", s).split())


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    dst = os.path.join(RAW, ZIP_NAME)
    if os.path.exists(dst) and os.path.getsize(dst) == ZIP_SIZE:
        print(f"  have {ZIP_NAME}")
        return
    print("GET", ZIP_URL)
    r = requests.get(ZIP_URL, headers=UA, timeout=900)
    r.raise_for_status()
    if not r.content.startswith(b"PK\x03\x04") or len(r.content) != ZIP_SIZE:
        raise SystemExit(f"{ZIP_NAME}: {len(r.content):,} bytes, pinned at {ZIP_SIZE:,}")
    with open(dst + ".part", "wb") as fh:
        fh.write(r.content)
    os.replace(dst + ".part", dst)
    print(f"  {len(r.content):,} bytes")


def _parts(geom):
    return list(geom.geoms) if isinstance(geom, MultiPolygon) else [geom]


def split_other_islands(geom):
    """COD's Other Islands -> (inner-island parts for La Digue, the rest), asserted."""
    parts = gpd.GeoSeries(_parts(geom), crs="EPSG:4326")
    area = parts.to_crs(AREA_CRS).area / 1e6
    cen = parts.to_crs(AREA_CRS).centroid.to_crs("EPSG:4326")
    inner = [False] * len(parts)
    import math
    for name, lat, lon, lo, hi in INNER_NAMED:
        hits = []
        for i, c in enumerate(cen):
            dy = (c.y - lat) * 111.0
            dx = (c.x - lon) * 111.0 * math.cos(math.radians(lat))
            if math.hypot(dx, dy) <= RADIUS_KM and area[i] >= 0.2:
                hits.append(i)
        if len(hits) != 1:
            raise SystemExit(f"{name}: {len(hits)} parts of COD's Other Islands within "
                             f"{RADIUS_KM} km, expected one")
        i = hits[0]
        if not lo <= area[i] <= hi:
            raise SystemExit(f"{name}: the part found is {area[i]:.2f} km2, expected {lo}-{hi}")
        inner[i] = True
        print(f"      {name:<12} {area[i]:6.2f} km2 at {cen[i].y:.3f}, {cen[i].x:.3f}")
    lat0, lon0, lat1, lon1 = INNER_BOX
    boxed = [i for i, c in enumerate(cen) if lat0 <= c.y <= lat1 and lon0 <= c.x <= lon1
             and not inner[i]]
    for i in boxed:
        inner[i] = True
    print(f"      + {len(boxed)} small parts round Félicité (the Soeurs and Cocos), "
          f"{sum(area[i] for i in boxed):.2f} km2")
    keep_in = unary_union([p for p, f in zip(parts, inner) if f])
    keep_out = unary_union([p for p, f in zip(parts, inner) if not f])
    return keep_in, keep_out, sum(f for f in inner), len(parts)


def main():
    if "--fetch" in sys.argv:
        fetch()

    import sc as scmod

    zpath = os.path.join(RAW, ZIP_NAME)
    if not os.path.exists(zpath):
        raise SystemExit(f"{zpath} missing; run with --fetch")
    os.makedirs(SHP_DIR, exist_ok=True)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(SHP_DIR)
    g = gpd.read_file(os.path.join(SHP_DIR, "syc_admbnda_adm3_nbs2010.shp"))
    if len(g) != EXPECTED_FEATURES:
        raise SystemExit(f"{len(g)} ADM3 features, expected {EXPECTED_FEATURES}")
    if g.crs is None or g.crs.to_epsg() != 4326:
        raise SystemExit(f"unexpected CRS {g.crs}")
    minx, miny, maxx, maxy = g.total_bounds
    if not (45.5 < minx and maxx < 57.0 and -10.5 < miny and maxy < -3.5):
        raise SystemExit(f"bbox {g.total_bounds} is not Seychelles")

    by_name = {fold(n): i for i, n in enumerate(g["ADM3_EN"])}
    if len(by_name) != EXPECTED_FEATURES:
        raise SystemExit("two COD features fold to the same name")

    sup = scmod.read_supplement()
    pop = {d: sum(rows.values()) for d, rows in sup.items()}

    print("\n  join: name and ISO number must agree")
    rows = []
    for d, code in scmod.DISTRICTS.items():
        i = by_name.get(fold(d))
        if i is None:
            raise SystemExit(f"{d}: no COD feature of that name")
        pcode = g.loc[i, "ADM3_PCODE"]
        if code != "SC-OI" and pcode[4:6] != code[3:5]:
            raise SystemExit(f"{d}: COD pcode {pcode} carries ISO number {pcode[4:6]}, the "
                             f"census district is {code}")
        rows.append((d, code, i, pcode))
    used = {i for _, _, i, _ in rows}
    left = [g.loc[i, "ADM3_EN"] for i in range(len(g)) if i not in used]
    if left != ["Perseverance Island"]:
        raise SystemExit(f"COD features the census does not name: {left}")
    pers_i = by_name[fold("Perseverance Island")]
    pers = g.loc[pers_i, "geometry"]

    geoms = {d: g.loc[i, "geometry"] for d, _, i, _ in rows}
    print("\n  Other Islands -> La Digue (report Table 2.3, footnote 6):")
    inner, outer, n_in, n_all = split_other_islands(geoms["Other Islands"])
    print(f"      {n_in} of COD's {n_all} Other Islands parts move to La Digue")
    geoms["La Digue"] = unary_union([geoms["La Digue"], inner])
    geoms["Other Islands"] = outer
    er_alone = geoms["English River"]
    geoms["English River"] = unary_union([er_alone, pers])

    units = gpd.GeoDataFrame(
        {"unit": [c for _, c, _, _ in rows], "name": [d for d, _, _, _ in rows],
         "cod_pcode": [p for _, _, _, p in rows], "pop": [pop[d] for d, _, _, _ in rows]},
        geometry=[geoms[d] for d, _, _, _ in rows], crs="EPSG:4326")
    units.loc[units["name"] == "English River", "cod_pcode"] = "SC1116ER+SC1127PI"
    units.loc[units["name"] == "La Digue", "cod_pcode"] = "SC3615LD+inner islands of SC4726OI"
    if len(units) != EXPECTED_UNITS or units["unit"].nunique() != EXPECTED_UNITS:
        raise SystemExit(f"{len(units)} units")

    check_areas(units, er_alone, pers)

    os.makedirs(OUT_DIR, exist_ok=True)
    tmp = OUT[:-5] + ".part.gpkg"
    if os.path.exists(tmp):
        os.remove(tmp)
    units.to_file(tmp, layer="districts", driver="GPKG")
    gpd.GeoDataFrame({"name": ["Perseverance Island"], "cod_pcode": ["SC1127PI"]},
                     geometry=[pers], crs="EPSG:4326").to_file(tmp, layer="perseverance",
                                                               driver="GPKG")
    os.replace(tmp, OUT)
    print(f"\nwrote {OUT} ({len(units)} districts, {units['pop'].sum():,} people in 2010)")
    lut = units.drop(columns="geometry").sort_values("unit")
    lut.to_csv(LOOKUP, index=False, encoding="utf-8")
    print(f"wrote {LOOKUP}")


def check_areas(units, er_alone, pers):
    """Each polygon against the area the 2010 report prints for its district."""
    import sc as scmod

    m = units.to_crs(AREA_CRS)
    measured = dict(zip(units["name"], m.geometry.area / 1e6))
    units["area_km2"] = units["name"].map(measured).round(3)
    print(f"\n  witness: polygon area against the 2010 report's Table 2.3")
    worst = 0.0
    for d in units["name"]:
        printed = scmod.TABLE_2_3[d][1]
        err = measured[d] / printed - 1.0
        tag = "  (not asserted)" if d in AREA_UNCHECKED else ""
        print(f"      {d:<22}{measured[d]:>8.2f} km2 vs {printed:>6.1f} printed ({err:+.0%}){tag}")
        if d in AREA_UNCHECKED:
            continue
        worst = max(worst, abs(err))
        if abs(err) > AREA_TOLERANCE:
            raise SystemExit(f"{d} is {err:+.0%} off its printed area")
    print(f"      worst asserted residual {worst:.1%}, inside {AREA_TOLERANCE:.0%}")

    alone = gpd.GeoSeries([er_alone, pers], crs="EPSG:4326").to_crs(AREA_CRS).area / 1e6
    printed = scmod.TABLE_2_3["English River"][1]
    e_alone = alone[0] / printed - 1.0
    print(f"      English River without Perseverance Island: {alone[0]:.2f} km2 "
          f"({e_alone:+.0%}); the island is {alone[1]:.2f} km2")
    if abs(e_alone) < 0.25:
        raise SystemExit("English River matches its printed area WITHOUT Perseverance "
                         "Island, so the merge is no longer what the census says")


if __name__ == "__main__":
    main()
