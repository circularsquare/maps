"""Comoros: the three islands (COD-AB adm1) and the Kontur placement layer inside them.

Writes data/geo/km/km_units.gpkg (`unit`, `pop`, geometry), data/geo/km/km_lookup.csv
(`geo_id` -> `unit`) and data/geo/km/km_hexes.gpkg (`unit`, `pop` as Kontur has it, geometry),
from data/raw/km/:

  * COD-AB Comoros (HDX `cod-ab-com`, valid on 2019-12-05; CC BY-IGO), adm1: Anjouan (Ndzouani)
    KM1, Grande Comore (Ngazidja) KM2, Mohéli (Mwali) KM3. Mayotte is not in it; the map draws
    Mayotte with France;
  * Kontur Population KM (2023-11-01), 400 m hexagons.

THE JOIN IS ON THE P-CODE (`PCODES`), witnessed by the name (each COD name must contain the census
island's name) and by Kontur's people per island against the 2017 census (`UNIT_BAND`). The
islands are 40 to 80 km apart, so the join cannot pass on a swapped label. Hex centroids are
joined to the islands; a centroid offshore is snapped to the nearest island within `SNAP_KM`
(coast), and anything further (Mayotte, if the extract holds any of it) is dropped and counted.

KONTUR KM HAS ANJOUAN'S AND MOHÉLI'S PEOPLE SWAPPED. Raw, it puts 343,284 people on Mohéli
(census 51,567; hexes of 37,211 around Fomboni, at the density cap) and 67,617 on Anjouan
(census 327,382; no hex above 1,529, Mutsamudu included); Grande Comore reads 1.05 of its share.
Each island's total is close to the other's census count, so the per-island footprint is right
and the per-island scale is not. Each island's hexes are therefore CALIBRATED to its census count
(`EXPECT_SWAPPED`, `SWAP_TOL` pin the failure), which also takes every hex far below the cap.

The shape inside each island is then witnessed against the census's préfectures
(`prefecture_witness`, `PREF_BAND`): 0.79 (Mbadjini Est) to 1.16 of each préfecture's share of its
island. COD-AB's adm2 does not match the census's 18 préfectures one to one (17 units: no Moya,
Mitsamiouli and Mboudé merged, and the uninhabited Kartala summit), so it is a witness and not a
layer to calibrate on.

Usage:
    python sources/km_geo.py --fetch    COD-AB (geojson zip, about 1 MB) and Kontur KM (gz)
    python sources/km_geo.py            rebuild from data/raw/km/
"""

import gzip
import os
import shutil
import sys
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
os.environ.setdefault("OMP_NUM_THREADS", "6")

import pandas as pd

RAW = os.path.join(ROOT, "data", "raw", "km")
GEO = os.path.join(ROOT, "data", "geo", "km")
COD_ZIP = os.path.join(RAW, "com_admin_boundaries.geojson.zip")
COD_URL = ("https://data.humdata.org/dataset/7f80a5d1-9f1a-462c-b298-338c7edc9224/resource/"
           "ce499503-58c7-4566-a578-c8738b2574d9/download/com_admin_boundaries.geojson.zip")
KONTUR_GZ = os.path.join(RAW, "kontur_population_KM_20231101.gpkg.gz")
KONTUR = os.path.join(RAW, "kontur_population_KM_20231101.gpkg")
KONTUR_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
              "kontur_population_KM_20231101.gpkg.gz")
NORMALIZED = os.path.join(ROOT, "data", "normalized", "km.csv")

PCODES = {"KM1": "Ndzuwani", "KM2": "Ngazidja", "KM3": "Mwali"}
NAME_IN = {"KM1": "Ndzouani", "KM2": "Ngazidja", "KM3": "Mwali"}   # COD spells Anjouan Ndzouani
EXPECTED_RATIO = 1.15           # Kontur 2023 over the 2017 count; about six years of growth
TOLERANCE = 0.25
UNIT_BAND = (0.80, 1.25)
SNAP_KM = 2.0
SNAP_BANDS_KM = (0.5, 1.0, 2.0, 5.0, 50.0)
DROP_MAX = 0.01
METRIC = "EPSG:32738"           # UTM 38S
EXPECT_SWAPPED = {"Ndzuwani", "Mwali"}      # Kontur KM's raw island totals, measured 2026-10-03
SWAP_TOL = 0.25
KONTUR_CAP = 46_200             # people per km2
# INSEED annex Tableau 1 (RGPH 2017), préfecture totals, transcribed 2026-10-03; sources/km.py
# asserts that each island's préfectures sum to it. Keyed by COD-AB adm2 name: COD merges
# Mitsamiouli and Mboudé, has no Moya (counted here with Sima, its western neighbour, as an
# assumption the witness prints), and draws the Kartala summit as a unit nobody lives in.
PREFECTURES_2017 = {
    "Fomboni": 30_834, "Nioumachioi": 11_384, "Djando": 9_349,
    "Mutsamudu": 63_831, "Ouani": 68_885, "Domoni": 69_904, "Mrémani": 65_448,
    "Sima": 35_174 + 24_140,
    "Moroni-Bambao": 121_236, "Hambou": 22_777, "Mbadjini Ouest": 23_599,
    "Mbadjini Est": 35_312, "Oichili-Dimani": 26_298, "Hamahamet-Mboinkou": 36_648,
    "Mitsamiouli-Mboudé": 31_821 + 24_341, "Itsandra-Hamanvou": 57_335, "Kartala": 0,
}
PREF_BAND = (0.5, 2.0)          # calibrated Kontur's share of its island over the census's
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/128.0 Safari/537.36")


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    for url, dst, minsize in ((COD_URL, COD_ZIP, 100_000), (KONTUR_URL, KONTUR_GZ, 50_000)):
        if os.path.exists(dst) and os.path.getsize(dst) > minsize:
            continue
        print("GET", url)
        r = requests.get(url, timeout=600, headers={"User-Agent": UA})
        r.raise_for_status()
        with open(dst + ".part", "wb") as fh:
            fh.write(r.content)
        os.replace(dst + ".part", dst)
    if not os.path.exists(KONTUR):
        with gzip.open(KONTUR_GZ, "rb") as src, open(KONTUR + ".part", "wb") as dst:
            shutil.copyfileobj(src, dst)
        os.replace(KONTUR + ".part", KONTUR)


def read_islands():
    import geopandas as gpd

    with zipfile.ZipFile(COD_ZIP) as z:
        name = [n for n in z.namelist() if "admin1" in n and n.endswith(".geojson")]
        if len(name) != 1:
            raise SystemExit(f"{COD_ZIP}: admin1 members {name}")
        g = gpd.read_file(io_bytes(z.read(name[0])))
    if len(g) != 3 or set(g["adm1_pcode"]) != set(PCODES):
        raise SystemExit(f"COD-AB adm1: {len(g)} units, {sorted(g['adm1_pcode'])}")
    for _, r in g.iterrows():
        if NAME_IN[r["adm1_pcode"]] not in r["adm1_name"]:
            raise SystemExit(f"{r['adm1_pcode']} is `{r['adm1_name']}`, expected "
                             f"{NAME_IN[r['adm1_pcode']]}")
    g["unit"] = g["adm1_pcode"].map(PCODES)
    return g[["unit", "geometry"]].to_crs("EPSG:4326")


def io_bytes(b):
    import io

    return io.BytesIO(b)


def prefecture_witness(out):
    """Calibrated Kontur per COD-AB adm2 against the census's préfectures, each as a share of its
    island. A witness of the shape inside each island; nothing is calibrated to it."""
    import geopandas as gpd

    with zipfile.ZipFile(COD_ZIP) as z:
        name = [n for n in z.namelist() if "admin2" in n and n.endswith(".geojson")][0]
        a2 = gpd.read_file(io_bytes(z.read(name))).to_crs(out.crs)
    if set(a2["adm2_name"]) != set(PREFECTURES_2017):
        raise SystemExit(f"COD-AB adm2 names {sorted(a2['adm2_name'])}")
    a2["unit"] = a2["adm1_pcode"].map(PCODES)
    pts = gpd.GeoDataFrame({"pop": out["pop"].to_numpy(), "unit": out["unit"].to_numpy()},
                           geometry=out.geometry.to_crs(METRIC).centroid, crs=METRIC)
    j = gpd.sjoin_nearest(pts, a2[["adm2_name", "unit", "geometry"]].to_crs(METRIC),
                          how="left", lsuffix="h", rsuffix="p")
    j = j[~j.index.duplicated(keep="first")]
    if (j["unit_h"] != j["unit_p"]).any():
        raise SystemExit("a hex's nearest préfecture is on another island")
    k = j.groupby("adm2_name")["pop"].sum()
    a2["census"] = a2["adm2_name"].map(PREFECTURES_2017)
    a2["kontur"] = a2["adm2_name"].map(k).fillna(0.0)
    print("\n  préfecture witness: calibrated Kontur's share of its island over the census's")
    bad = []
    for u, g in a2.sort_values(["unit", "adm2_name"]).groupby("unit", sort=False):
        for _, r in g.iterrows():
            if r["census"] == 0:
                print(f"      {u:<9} {r['adm2_name']:<20} census        0  Kontur {r['kontur']:>8,.0f}")
                continue
            rel = (r["kontur"] / g["kontur"].sum()) / (r["census"] / g["census"].sum())
            print(f"      {u:<9} {r['adm2_name']:<20} census {r['census']:>8,}  Kontur "
                  f"{r['kontur']:>8,.0f}  {rel:5.2f}")
            if not PREF_BAND[0] <= rel <= PREF_BAND[1]:
                bad.append(r["adm2_name"])
    if bad:
        raise SystemExit(f"préfectures outside PREF_BAND {PREF_BAND}: {bad}")


def main():
    import geopandas as gpd
    from geo_checks import read_layer

    if "--fetch" in sys.argv or not (os.path.exists(COD_ZIP) and os.path.exists(KONTUR)):
        fetch()
    if not os.path.exists(NORMALIZED):
        raise SystemExit(f"missing {NORMALIZED}; run sources/km.py first")
    units = read_islands()
    rows = pd.read_csv(NORMALIZED, usecols=["geo_id", "count"])
    pop = rows.groupby("geo_id")["count"].sum()
    if set(pop.index) != set(PCODES.values()):
        raise SystemExit(f"km.csv islands {sorted(pop.index)}")
    units["pop"] = units["unit"].map(pop).astype(int)
    a = units.to_crs("EPSG:6933").area / 1e6
    for u, km2 in zip(units["unit"], a):
        print(f"  {u:<10} {km2:7.1f} km2, census 2017 {int(pop[u]):>8,}")

    hexes = read_layer(KONTUR, "Kontur KM")
    popcol = "population"
    print(f"\nKontur hexes: {len(hexes):,}, population {hexes[popcol].sum():,.0f}")
    pts = gpd.GeoDataFrame({popcol: hexes[popcol].to_numpy()}, geometry=hexes.geometry.centroid,
                           crs=hexes.crs).to_crs(units.crs)
    hexes = hexes.to_crs(units.crs)
    joined = gpd.sjoin(pts, units[["unit", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)
    outside = joined["unit"].isna()
    print(f"  hexes whose centroid is outside every island: {int(outside.sum()):,} "
          f"({pts.loc[outside, popcol].sum():,.0f} people)")
    if outside.any():
        near = gpd.sjoin_nearest(pts.loc[outside].to_crs(METRIC),
                                 units[["unit", "geometry"]].to_crs(METRIC),
                                 how="left", distance_col="d")
        near = near[~near.index.duplicated(keep="first")]
        for b in SNAP_BANDS_KM:
            m = near["d"] <= b * 1000
            print(f"      within {b:>4g} km of an island: {int(m.sum()):>5,} hexes, "
                  f"{pts.loc[near.index[m], popcol].sum():>9,.0f} people")
        snapped = near["d"] <= SNAP_KM * 1000
        joined.loc[near.index[snapped], "unit"] = near.loc[snapped, "unit"]
        print(f"  snapped within {SNAP_KM:g} km: {int(snapped.sum()):,} hexes")
    dropped = joined["unit"].isna()
    share = pts.loc[dropped, popcol].sum() / pts[popcol].sum()
    print(f"  dropped: {int(dropped.sum()):,} hexes, {pts.loc[dropped, popcol].sum():,.0f} people "
          f"({share:.3%})")
    if share > DROP_MAX:
        raise SystemExit(f"more than {DROP_MAX:.0%} of Kontur's people dropped")

    keep = ~dropped
    out = gpd.GeoDataFrame({"unit": joined.loc[keep, "unit"].to_numpy(),
                            "pop": pts.loc[keep, popcol].to_numpy(dtype=float)},
                           geometry=hexes.geometry[keep.to_numpy()].to_numpy(), crs=units.crs)
    per = out.groupby("unit")["pop"].sum()
    drawn = dict(zip(units["unit"], units["pop"].astype(int)))
    tot = float(per.sum())
    ratio = tot / sum(drawn.values())
    print(f"\n  Kontur {tot:,.0f} vs the 2017 census {sum(drawn.values()):,}: ratio {ratio:.3f}")
    if abs(ratio - EXPECTED_RATIO) > TOLERANCE:
        raise SystemExit("Kontur and the census total disagree beyond the band")
    print("  per island: Kontur / census over the national ratio")
    bad = []
    for u in sorted(drawn, key=lambda x: -drawn[x]):
        rel = (per.get(u, 0.0) / drawn[u]) / ratio
        print(f"      {u:<10} census {drawn[u]:>9,}  Kontur {per.get(u, 0.0):>10,.0f}  {rel:5.2f}")
        if not UNIT_BAND[0] <= rel <= UNIT_BAND[1]:
            bad.append(u)
    if set(bad) != EXPECT_SWAPPED:
        raise SystemExit(f"outside UNIT_BAND {UNIT_BAND}: {bad}; pinned {sorted(EXPECT_SWAPPED)}")
    # Kontur KM holds about Anjouan's people on Mohéli and Mohéli's on Anjouan (Fomboni hexes of
    # 37,211 people; nothing on Anjouan above 1,529, Mutsamudu included). The two islands' raw
    # totals must read as that swap, within SWAP_TOL, before the per-island scale fixes it.
    for a, b in (("Mwali", "Ndzuwani"), ("Ndzuwani", "Mwali")):
        r = per[a] / ratio / drawn[b]
        print(f"      Kontur's {a} over the census's {b}: {r:.2f}")
        if abs(r - 1) > SWAP_TOL:
            raise SystemExit("the two islands' Kontur totals are not the census's swapped")

    # CALIBRATE each island's hexes to its census count; the shape inside each island stays
    # Kontur's. Done before kontur_cap.apply sees the layer, which then finds nothing at the cap.
    out["pop"] = out["pop"] * out["unit"].map({u: drawn[u] / per[u] for u in drawn})
    dens = out["pop"] / (out.to_crs("EPSG:6933").area / 1e6)
    print(f"  calibrated to the census per island; densest hex {dens.max():,.0f}/km2")
    if dens.max() > KONTUR_CAP:
        raise SystemExit("a calibrated hex is above Kontur's cap")
    prefecture_witness(out)

    os.makedirs(GEO, exist_ok=True)
    units[["unit", "pop", "geometry"]].to_file(os.path.join(GEO, "km_units.gpkg"), layer="units",
                                               driver="GPKG")
    pd.DataFrame({"geo_id": list(PCODES.values()), "unit": list(PCODES.values())}).to_csv(
        os.path.join(GEO, "km_lookup.csv"), index=False, encoding="utf-8")
    out[["unit", "pop", "geometry"]].to_file(os.path.join(GEO, "km_hexes.gpkg"), layer="hexes",
                                             driver="GPKG")
    print(f"\nwrote {GEO}/km_units.gpkg, km_lookup.csv and km_hexes.gpkg ({len(out):,} cells)")


if __name__ == "__main__":
    main()
