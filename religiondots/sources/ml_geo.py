"""Mali — the 20 régions of the 2022 census, and the placement grid.

Writes:
    data/geo/ml/ml_regions.gpkg   the 20 counted units (`regions`)
    data/geo/ml/ml_hexes.gpkg     Kontur 400 m hexes with `unit` and `pop` (`hexes`)
    data/geo/ml/ml_lookup.csv     unit -> census population, Kontur population, cercles, area

Usage:
    python sources/ml_geo.py --fetch    COD-AB (~8 MB), geoBoundaries ADM1 + ADM2 (~4 MB), Kontur (~12 MB)
    python sources/ml_geo.py            rebuild from data/raw/ml/

THE BOUNDARIES ARE COD-AB MALI v03 (IGM Mali, boundaries created 2025-02-20, valid 2025-09-04,
OCHA FIS / HDX, `cod-ab-mli`): 20 régions and 160 cercles, which is the tier the census prints.
The census (Tableau 1.1) follows the law of 13 March 2023: 19 régions, the Bamako district, 159
cercles. geoBoundaries MLI ADM1 is the 9 units of before (2021) and no use as the drawn layer.

THE JOIN IS BY NAME, folded (`adm1_name1`, the French name), and asserted to be a bijection.
Three witnesses that the name key does not determine:

1. **Cercles.** Each COD-AB région holds the number of cercles Tableau 1.1 prints for it (Bamako,
   printed `-`, is one COD-AB cercle). Two régions with the same count can still be swapped, so:
2. **The parent région.** Each new région lies inside the old région it was cut from (loi
   n°2012-017 as brought into force by the 2023 law): geoBoundaries' 9-unit ADM1, a different
   file from a different source, must hold at least PARENT_MIN of its area.
3. **Population.** Kontur 2023 summed inside each polygon against Tableau 2.3's resident
   population: the log-log correlation over the 20 régions, against 2,000 shuffles, and a
   per-région band. Nioro and Kita (both from Kayes, 6 cercles, 678,061 and 681,671 people) pass
   all three swapped; `OLD_CERCLE` settles that pair with the pre-2023 cercle of each name.

KONTUR IS NEEDED because the régions run from Bamako (733 km2, 4.2 million people) to Taoudenni
(293,000 km2, 100,358), and the northern régions hold their people along the Niger and in a few
towns. It is a WITHIN-région weight only.
"""

import csv
import gzip
import os
import re
import shutil
import sys
import unicodedata
import zipfile

os.environ.setdefault("OMP_NUM_THREADS", "6")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
RAW = os.path.join(ROOT, "data", "raw", "ml")
GEO = os.path.join(ROOT, "data", "geo", "ml")
KONTUR = os.path.join(ROOT, "data", "geo", "kontur")
NORM = os.path.join(ROOT, "data", "normalized", "ml.csv")

COD_URL = ("https://data.humdata.org/dataset/d2ec62bb-5a93-436d-8297-88b3ee9b6818/resource/"
           "f5502554-402b-405b-8967-8182d306a4d1/download/mli_admin_boundaries.geojson.zip")
COD_ZIP = os.path.join(RAW, "mli_admin_boundaries.geojson.zip")
COD_DIR = os.path.join(RAW, "cod_ab_v03")
GB = ("https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/gbOpen/MLI/{0}/"
      "geoBoundaries-MLI-{0}.geojson")
GB_ADM1 = os.path.join(RAW, "geoBoundaries-MLI-ADM1.geojson")
GB_ADM2 = os.path.join(RAW, "geoBoundaries-MLI-ADM2.geojson")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_ML_20231101.gpkg.gz")
GZ_NAME = "kontur_population_ML_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_ML_20231101.gpkg"

OUT_UNITS = os.path.join(GEO, "ml_regions.gpkg")
OUT_HEXES = os.path.join(GEO, "ml_hexes.gpkg")
OUT_LOOKUP = os.path.join(GEO, "ml_lookup.csv")

UNITS = 20
CERCLES = 160

# The old région each 2022 région was cut from, keyed by census spelling; values are folded to
# match geoBoundaries' pre-reform ADM1 names.
PARENT = {
    "Kayes": "kayes", "Nioro": "kayes", "Kita": "kayes",
    "Koulikoro": "koulikoro", "Dioïla": "koulikoro", "Nara": "koulikoro",
    "Sikasso": "sikasso", "Bougouni": "sikasso", "Koutiala": "sikasso",
    "Ségou": "segou", "San": "segou",
    "Mopti": "mopti", "Douentza": "mopti", "Bandiagara": "mopti",
    "Tombouctou": "tombouctou", "Taoudenni": "tombouctou",
    "Gao": "gao", "Ménaka": "gao",
    "Kidal": "kidal", "Bamako": "bamako",
}
PARENT_MIN = 0.75
# Units whose 2022 polygon is bigger than the old one rather than cut from it. COD-AB's Bamako is
# 733 km2 against geoBoundaries' pre-2023 district of about 245 km2, two-thirds of it in what was
# Koulikoro; the census's Bamako is seven arrondissements (Tableau 1.1), not the old six
# communes, and 4,227,569 people. So Bamako's test is that the old district lies inside the new.
GREW = {"Bamako"}
# Pre-2023 cercles (geoBoundaries ADM2, 2017) whose representative point must fall in the
# 2022 région of the same name: the pairs witnesses 1-3 cannot tell apart.
OLD_CERCLE = ["Nioro", "Kita", "Sikasso", "Koutiala"]

# Census November 2022 (population de droit, Tableau 2.3) against Kontur 2023-11. The census
# includes 941,335 people estimated for areas it could not enumerate, so the level is
# comparable; the band is wide per région because Kontur's own inputs differ most in the north.
KONTUR_RATIO_MIN = 0.80
KONTUR_RATIO_MAX = 1.40
# Measured on the first full run: Ménaka 0.27x and Douentza 2.52x, the two ends, with Kidal 1.65x
# and Taoudenni 0.60x. Those are where the census counted least directly (Tableau 1.2 models 93,653
# of Ménaka's 318,876, and 154,979 in Douentza or 22,960 by Tableaux 2.3 minus 2.9) and where the
# nomads are, whom a building-footprint model misses. Witnesses 1, 2 and 2b pass both, so neither
# is a join error; the band is only there to catch a region that holds nobody or everybody.
REGION_RATIO_MIN = 0.20
REGION_RATIO_MAX = 3.00

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) religiondots/1.0"}


def norm(s):
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def _get(url, dst, first, min_size):
    import requests

    if os.path.exists(dst) and os.path.getsize(dst) > min_size:
        print(f"  have {os.path.basename(dst)} ({os.path.getsize(dst):,} bytes)")
        return
    print("GET", url)
    r = requests.get(url, timeout=1800, stream=True, headers=UA)
    r.raise_for_status()
    with open(dst + ".part", "wb") as fh:
        for chunk in r.iter_content(1 << 20):
            fh.write(chunk)
    with open(dst + ".part", "rb") as fh:
        head = fh.read(64).lstrip()
    if not head.startswith(first):
        raise SystemExit(f"{dst}: starts {head[:16]!r}, expected {first!r}")
    os.replace(dst + ".part", dst)
    print(f"  got {os.path.getsize(dst):,} bytes")


def fetch():
    os.makedirs(RAW, exist_ok=True)
    os.makedirs(KONTUR, exist_ok=True)
    _get(COD_URL, COD_ZIP, b"PK\x03\x04", 100_000)
    with zipfile.ZipFile(COD_ZIP) as z:
        z.extractall(COD_DIR)
    _get(GB.format("ADM1"), GB_ADM1, b"{", 100_000)
    _get(GB.format("ADM2"), GB_ADM2, b"{", 100_000)
    gz = os.path.join(KONTUR, GZ_NAME)
    gpkg = os.path.join(KONTUR, GPKG_NAME)
    _get(GZ_URL, gz, b"\x1f\x8b", 1_000_000)
    if not (os.path.exists(gpkg) and os.path.getsize(gpkg) > 1_000_000):
        with gzip.open(gz, "rb") as src, open(gpkg + ".part", "wb") as dst:
            shutil.copyfileobj(src, dst)
        os.replace(gpkg + ".part", gpkg)
    with open(gpkg, "rb") as fh:
        if fh.read(4) != b"SQLi":
            raise SystemExit(f"{gpkg} is not a GeoPackage")


def _cod_layer(level):
    names = [n for n in os.listdir(COD_DIR) if n.lower().endswith(".geojson") and f"admin{level}" in n.lower()]
    if len(names) != 1:
        raise SystemExit(f"COD-AB admin{level}: {len(names)} geojson files in {COD_DIR}: {sorted(os.listdir(COD_DIR))}")
    return os.path.join(COD_DIR, names[0])


def main():
    import geopandas as gpd
    import numpy as np
    import pandas as pd

    import ml

    if "--fetch" in sys.argv:
        fetch()
    gpkg = os.path.join(KONTUR, GPKG_NAME)
    for p in (COD_DIR, GB_ADM1, GB_ADM2, gpkg, NORM):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p} — run sources/ml.py --fetch and sources/ml_geo.py --fetch")
    os.makedirs(GEO, exist_ok=True)

    # ---- 1. the counted units
    df = pd.read_csv(NORM, keep_default_na=False, na_values=[""])
    census = df[df["source_category"] == "Total"].set_index("geo_id")["count"]
    if len(census) != UNITS or set(census.index) != set(ml.REGIONS):
        raise SystemExit(f"ml.csv units differ from ml.REGIONS: {sorted(set(census.index) ^ set(ml.REGIONS))}")

    a1 = gpd.read_file(_cod_layer(1), engine="fiona")
    a2 = gpd.read_file(_cod_layer(2), engine="fiona")
    print(f"COD-AB admin1: {len(a1)} features, version {sorted(set(a1['version']))}, crs={a1.crs}; "
          f"admin2: {len(a2)}")
    if len(a1) != UNITS or len(a2) != CERCLES:
        raise SystemExit(f"COD-AB has {len(a1)} régions and {len(a2)} cercles, expected {UNITS} and {CERCLES}")
    a1["key"] = a1["adm1_name1"].map(norm)
    if a1["key"].duplicated().any():
        raise SystemExit("repeated folded région names in COD-AB")
    lut = {}
    for name in census.index:
        hits = a1.index[a1["key"] == norm(name)].tolist()
        if len(hits) != 1:
            raise SystemExit(f"census région {name!r} matched {len(hits)} polygons: {sorted(a1['adm1_name1'])}")
        lut[name] = hits[0]
    if len(set(lut.values())) != UNITS:
        raise SystemExit("two census régions matched one polygon")
    print(f"  name join: {UNITS}/{UNITS} régions matched one polygon each, 0 polygons unused")

    units = a1.loc[list(lut.values())].copy()
    units["unit"] = list(lut.keys())
    units["census_pop"] = units["unit"].map(census).astype(int)
    units["resident_pop"] = units["unit"].map(lambda u: ml.T23[u][2])
    units["area_km2"] = units.to_crs(6933).geometry.area / 1e6

    # ---- witness 1: cercles per région against Tableau 1.1
    n2 = a2.groupby("adm1_pcode").size()
    units["cercles"] = units["adm1_pcode"].map(n2).fillna(0).astype(int)
    want = units["unit"].map(lambda u: ml.T11[u] if ml.T11[u] is not None else 1)
    off = units[units["cercles"] != want]
    if len(off):
        raise SystemExit("cercle counts differ from Tableau 1.1: " + ", ".join(
            f"{r.unit} {r.cercles}" for r in off.itertuples()))
    print(f"  witness 1, cercles: all {UNITS} régions hold Tableau 1.1's count "
          f"({CERCLES} COD-AB cercles; the census's 159 plus Bamako)")

    # ---- witness 2: each région inside its pre-2023 parent (geoBoundaries ADM1)
    g1 = gpd.read_file(GB_ADM1, engine="fiona")
    # geoBoundaries spells the old région `Koulikouro`.
    g1["key"] = g1["shapeName"].map(norm).replace({"koulikouro": "koulikoro"})
    print(f"  geoBoundaries MLI ADM1: {len(g1)} features: {sorted(g1['shapeName'])}")
    missing = sorted(set(PARENT.values()) - set(g1["key"]))
    if missing:
        raise SystemExit(f"parents not in geoBoundaries ADM1: {missing}")
    # The test that discriminates is that the parent holds the LARGEST share of each région: a
    # mislabelled polygon lands almost wholly in another old région. The share itself is below
    # 100% because geoBoundaries' 2021 lines and IGM's 2025 lines are drawn differently (Ségou
    # 89.8% on the first run), so its bar is loose.
    eq = units[["unit", "geometry"]].to_crs(6933)
    g1e = g1[["key", "geometry"]].to_crs(6933).dissolve(by="key")
    shares = []
    for r in eq.itertuples():
        if r.unit in GREW:
            # The other way round: the old unit must lie inside the new one.
            old = g1e.geometry[PARENT[r.unit]]
            share = old.intersection(r.geometry).area / old.area
            print(f"  witness 2, {r.unit}: {share:.1%} of the pre-2023 district lies inside COD-AB's "
                  f"{r.geometry.area / 1e6:,.0f} km2 (old {old.area / 1e6:,.0f} km2)")
            if share < PARENT_MIN:
                raise SystemExit(f"{r.unit}: only {share:.1%} of the old district is inside the new")
            continue
        inter = g1e.geometry.intersection(r.geometry).area / r.geometry.area
        best = inter.idxmax()
        shares.append((inter[PARENT[r.unit]], r.unit, best))
        if best != PARENT[r.unit] or inter[PARENT[r.unit]] < PARENT_MIN:
            raise SystemExit(f"{r.unit}: {inter[PARENT[r.unit]]:.1%} inside its parent "
                             f"{PARENT[r.unit]}; largest share in {best} ({inter[best]:.1%})")
    shares.sort()
    print(f"  witness 2, parent: every région's largest share is in the old région it was cut "
          f"from, at least {PARENT_MIN:.0%}; lowest " + ", ".join(
              f"{u} {s:.1%}" for s, u, _b in shares[:4]))

    # ---- witness 2b: the pairs nothing else separates, by the old cercle of the same name
    g2 = gpd.read_file(GB_ADM2, engine="fiona")
    g2["key"] = g2["shapeName"].map(norm)
    for name in OLD_CERCLE:
        hit = g2[g2["key"] == norm(name)]
        if len(hit) != 1:
            raise SystemExit(f"old cercle {name!r}: {len(hit)} in geoBoundaries ADM2")
        pt = hit.to_crs(32630).geometry.representative_point().to_crs(units.crs).iloc[0]
        inside = units.loc[units.geometry.contains(pt), "unit"].tolist()
        if inside != [name]:
            raise SystemExit(f"the pre-2023 cercle {name} falls in {inside}, not {name}")
    print(f"  witness 2b: the pre-2023 cercles {', '.join(OLD_CERCLE)} each fall in the région of "
          "their name")

    units = units[["unit", "adm1_pcode", "census_pop", "resident_pop", "cercles", "area_km2", "geometry"]]
    units.to_file(OUT_UNITS, layer="regions", driver="GPKG")
    print(f"wrote {OUT_UNITS} ({len(units)} régions)")

    # ---- 2. Kontur, joined on hex centroids
    hexes = gpd.read_file(gpkg)
    if len(hexes) == 0:
        raise SystemExit("Kontur gpkg read returned ZERO features")
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    if popcol is None:
        raise SystemExit(f"no population column in {list(hexes.columns)}")
    print(f"\nKontur hexes: {len(hexes):,}, crs={hexes.crs}, population {hexes[popcol].sum():,.0f}")
    cent = hexes.geometry.centroid
    pts = gpd.GeoDataFrame({popcol: hexes[popcol].to_numpy()}, geometry=cent, crs=hexes.crs).to_crs(units.crs)
    hexes = hexes.to_crs(units.crs)
    joined = gpd.sjoin(pts, units[["unit", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)

    # Mali is landlocked: a centroid in no région is on the national border. Snap within SNAP_M.
    SNAP_M = 2_000
    outside = joined["unit"].isna()
    print(f"  hexes whose centroid is in no région: {int(outside.sum()):,} "
          f"({float(pts.loc[outside, popcol].sum()):,.0f} people)")
    if outside.any():
        utm = units[["unit", "geometry"]].to_crs(32630)
        near = gpd.sjoin_nearest(pts.loc[outside].to_crs(32630), utm, how="left",
                                 max_distance=SNAP_M, distance_col="dist")
        near = near[~near.index.duplicated(keep="first")]
        got = near["unit"].notna()
        joined.loc[near.index[got], "unit"] = near.loc[got, "unit"].to_numpy()
        print(f"  snapped {int(got.sum()):,} within {SNAP_M:,} m ({float(near.loc[got, popcol].sum()):,.0f} people)")
    outside = joined["unit"].isna()
    lost = float(pts.loc[outside, popcol].sum())
    share_lost = lost / float(pts[popcol].sum())
    print(f"  dropped: {int(outside.sum()):,} hexes ({lost:,.0f} people, {100 * share_lost:.3f}%)")
    if share_lost > 0.02:
        raise SystemExit(f"{100 * share_lost:.2f}% of Kontur is outside every région after the snap")

    keep = ~outside
    out = gpd.GeoDataFrame({"unit": joined.loc[keep, "unit"].to_numpy(),
                            "pop": pts.loc[keep, popcol].to_numpy(dtype=float)},
                           geometry=hexes.geometry[keep.to_numpy()].to_numpy(), crs=units.crs)
    per = out.groupby("unit")["pop"].agg(["size", "sum"])
    missing = sorted(set(units["unit"]) - set(per.index))
    if missing or (per["sum"] <= 0).any():
        raise SystemExit(f"régions with no populated hex: {missing or sorted(per.index[per['sum'] <= 0])}")

    # ---- witness 3: population
    lk = units.drop(columns="geometry").merge(per.rename(columns={"size": "hexes", "sum": "kontur_pop"}),
                                              left_on="unit", right_index=True)
    lk["kontur_over_resident"] = lk["kontur_pop"] / lk["resident_pop"]
    ratio = lk["kontur_pop"].sum() / lk["resident_pop"].sum()
    print(f"\n  Kontur 2023 {lk['kontur_pop'].sum():,.0f} against the census's {lk['resident_pop'].sum():,}: {ratio:.3f}")
    if not KONTUR_RATIO_MIN <= ratio <= KONTUR_RATIO_MAX:
        raise SystemExit(f"national ratio {ratio:.3f} outside [{KONTUR_RATIO_MIN}, {KONTUR_RATIO_MAX}]")
    x, y = np.log(lk["resident_pop"].to_numpy(float)), np.log(lk["kontur_pop"].to_numpy(float))
    r_obs = np.corrcoef(x, y)[0, 1]
    rng = np.random.default_rng(20260914)
    null = np.array([np.corrcoef(x, rng.permutation(y))[0, 1] for _ in range(2000)])
    print(f"  witness 3, log-log correlation over {UNITS} régions {r_obs:.3f}; shuffled: 99th "
          f"percentile {np.quantile(null, 0.99):.3f}, max {null.max():.3f}")
    if r_obs < 0.9 or r_obs <= null.max():
        raise SystemExit("Kontur does not follow the census's régions better than a shuffle")
    band = lk[(lk["kontur_over_resident"] < REGION_RATIO_MIN) | (lk["kontur_over_resident"] > REGION_RATIO_MAX)]
    for r in lk.sort_values("kontur_over_resident").itertuples(index=False):
        print(f"    {r.unit:<11} {r.kontur_over_resident:5.2f}x  census {r.resident_pop:>9,}  "
              f"Kontur {r.kontur_pop:>11,.0f}  {r.hexes:>7,} hexes  {r.area_km2:>9,.0f} km2")
    if len(band):
        raise SystemExit(f"régions outside [{REGION_RATIO_MIN}, {REGION_RATIO_MAX}]: {sorted(band['unit'])}")

    out.to_file(OUT_HEXES, layer="hexes", driver="GPKG")
    print(f"\nwrote {OUT_HEXES} ({len(out):,} hexes)")
    with open(OUT_LOOKUP, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["unit", "adm1_pcode", "census_pop_ordinary_2022", "census_pop_resident_2022",
                    "kontur_pop_2023", "hexes", "kontur_over_resident", "cercles", "area_polygon_km2"])
        for r in lk.sort_values("unit").itertuples(index=False):
            w.writerow([r.unit, r.adm1_pcode, int(r.census_pop), int(r.resident_pop),
                        round(r.kontur_pop, 1), int(r.hexes), round(r.kontur_over_resident, 3),
                        int(r.cercles), round(r.area_km2, 1)])
    print(f"wrote {OUT_LOOKUP}")


if __name__ == "__main__":
    main()
