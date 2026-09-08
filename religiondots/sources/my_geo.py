"""Malaysia — administrative district boundaries and the placement grid.

Writes:
    data/geo/my/my_districts.gpkg   the 160 counted districts (`units`)
    data/geo/my/my_hexes.gpkg       Kontur H3 hexes with `unit` and `pop` (`place`)
    data/geo/my/my_lookup.csv       unit -> names, census population, Kontur population

Usage:
    python sources/my_geo.py --fetch    geoBoundaries ADM1+ADM2 and one Kontur MY extract
    python sources/my_geo.py            rebuild from data/raw/my/

Malaysia is not among §11h's 34 USCB countries, so unlike Ethiopia, Pakistan and
Bangladesh the boundaries do NOT come out of the same file as the counts and the
join has to be earned.  It is earned cheaply: **geoBoundaries gbOpen MYS ADM2,
vintage 2020 — the census year — matches 159 of the 160 census districts by name
once three renames are aliased.**

THE THREE RENAMES, all of which are the polygon carrying the older name:
    Kulaijaya            -> Kulai      (Johor, renamed 2015)
    Ledang               -> Tangkak    (Johor, renamed 2015)
    Nabawan / Persiangan -> Nabawan    (Sabah, alternate spelling kept as a compound)

THE NAME JOIN IS VERIFIED SPATIALLY, NOT TRUSTED.  A matching district name in the
wrong state is the silent mis-join this file keeps warning about (§9o Kenya, §9m the
Philippines), so every ADM2 polygon is given a state by point-in-polygon against
ADM1 and that state must agree with the census.  It does for all 159 — the only
disagreements are geoBoundaries writing the English **Penang** and **Malacca** where
the census writes **Pulau Pinang** and **Melaka**, which the state alias table
absorbs.  No district name collides across states.

PUTRAJAYA IS MISSING FROM ADM2 AND IS *INSIDE* SEPANG — the finding worth carrying.
geoBoundaries ships 159 ADM2 polygons, not 160: the federal territory of Putrajaya
has no ADM2 feature.  ADM1 has it, but **appending it would double-count 48.7 km²**,
because Sepang's ADM2 polygon still covers 100.0% of Putrajaya's ground — the ADM2
layer simply does not know the territory was carved out of Selangor in 2001.  So
Putrajaya is *subtracted* from Sepang before being added as its own unit.  W.P.
Kuala Lumpur and W.P. Labuan are NOT like this; both are ordinary ADM2 features.

    Check before believing a layer is a partition: it is, apart from this.  All
    159 ADM2 polygons were tested pairwise and no two overlap by more than 1 km².
    An enclave that is missing rather than overlapping is invisible to an overlap
    test and shows up only as a unit count one short.

KONTUR IS WORTH IT HERE, and the reason is Sarawak and Sabah.  The peninsular
districts are small and fairly uniform, but Malaysia's largest units are the
interior Borneo districts — and they are exactly the ones the country is worth
drawing for.  Belaga is roughly 19,000 km² with 22,502 people, Kapit similar; both
are over 85% Christian, with the population strung along the Rajang and its
tributaries and nothing at all on the ridges.  Spread those uniformly and the most
distinctive religious geography in the country is smeared across empty rainforest.
At ~2,070 km² per district against Kontur's ~0.74 km² hexes the grid is far finer
than the counting tier, so it pays (cf. the Kontur resolution floor).
"""

from __future__ import annotations

import csv
import gzip
import os
import re
import shutil
import sys
import unicodedata

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "my")
GEO = os.path.join(ROOT, "data", "geo", "my")
KONTUR = os.path.join(ROOT, "data", "geo", "kontur")
NORM = os.path.join(ROOT, "data", "normalized", "my.csv")

OUT_UNITS = os.path.join(GEO, "my_districts.gpkg")
OUT_HEXES = os.path.join(GEO, "my_hexes.gpkg")
OUT_LOOKUP = os.path.join(GEO, "my_lookup.csv")

GB_COMMIT = "9469f09"
GB = "https://github.com/wmgeolab/geoBoundaries/raw/%s/releaseData/gbOpen/MYS" % GB_COMMIT
ADM2_URL = GB + "/ADM2/geoBoundaries-MYS-ADM2.geojson"
ADM1_URL = GB + "/ADM1/geoBoundaries-MYS-ADM1.geojson"
ADM2_FILE = os.path.join(RAW, "geoBoundaries-MYS-ADM2.geojson")
ADM1_FILE = os.path.join(RAW, "geoBoundaries-MYS-ADM1.geojson")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.eu-central-1.amazonaws.com/"
          "kontur_datasets/kontur_population_MY_20231101.gpkg.gz")
GZ_NAME = "kontur_population_MY_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_MY_20231101.gpkg"

ADM2_FEATURES = 159           # 160 districts minus Putrajaya, which ADM1 supplies
DISTRICTS = 160
CENSUS_POPULATION = 32_447_385

# 2020 census against a 2023 grid: three years, 32.45M -> roughly 34M, so ~1.05 is
# expected. NOT Bangladesh's [1.00, 1.55] (twelve years) and NOT Ethiopia's
# [1.15, 2.10] (sixteen). Re-derived per country, per §12.
KONTUR_RATIO_MIN = 0.90
KONTUR_RATIO_MAX = 1.30

# geoBoundaries polygon name -> census district name.
DISTRICT_ALIAS = {
    "kulaijaya": "kulai",
    "ledang": "tangkak",
    "nabawan persiangan": "nabawan",
}
# geoBoundaries ADM1 name -> census state name.
STATE_ALIAS = {
    "penang": "pulau pinang",
    "malacca": "melaka",
    "kuala lumpur": "kuala lumpur",
    "putrajaya": "putrajaya",
    "labuan": "labuan",
}


def key(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().replace("w.p.", " ").replace("wilayah persekutuan", " ")
    s = re.sub(r"\b(daerah|district|jajahan|bahagian|division)\b", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    return re.sub(r"\s+", " ", s)


def dkey(s: str) -> str:
    k = key(s)
    return DISTRICT_ALIAS.get(k, k)


def skey(s: str) -> str:
    k = key(s)
    return STATE_ALIAS.get(k, k)


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    os.makedirs(KONTUR, exist_ok=True)
    ua = {"User-Agent": "Mozilla/5.0"}

    for url, path in ((ADM2_URL, ADM2_FILE), (ADM1_URL, ADM1_FILE)):
        if os.path.exists(path) and os.path.getsize(path) > 100_000:
            print("already have", os.path.basename(path))
            continue
        print("GET", url)
        r = requests.get(url, timeout=600, headers=ua)
        r.raise_for_status()
        # §5a: a 200 is not a download. GitHub serves an HTML page for a bad path.
        if not r.content.lstrip()[:1] == b"{":
            raise SystemExit(f"{url} did not return JSON (got {r.content[:80]!r})")
        with open(path, "wb") as fh:
            fh.write(r.content)
        print(f"  {len(r.content):,} bytes")

    gz = os.path.join(KONTUR, GZ_NAME)
    gpkg = os.path.join(KONTUR, GPKG_NAME)
    if os.path.exists(gpkg) and os.path.getsize(gpkg) > 20_000_000:
        print("already have", os.path.basename(gpkg))
        return
    if not os.path.exists(gz) or os.path.getsize(gz) < 1_000_000:
        print("GET", GZ_URL)
        r = requests.get(GZ_URL, timeout=1800, stream=True, headers=ua)
        r.raise_for_status()
        with open(gz, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
        print(f"  {os.path.getsize(gz):,} bytes")
    with gzip.open(gz, "rb") as src, open(gpkg, "wb") as dst:
        shutil.copyfileobj(src, dst)
    with open(gpkg, "rb") as fh:
        if fh.read(4) != b"SQLi":
            raise SystemExit(f"{gpkg} is not a GeoPackage")
    print(f"  unpacked {os.path.getsize(gpkg):,} bytes")


def census_districts():
    """geo_id -> (district name, state name, census population)."""
    out = {}
    with open(NORM, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["geo_level"] != "district":
                continue
            gid = r["geo_id"]
            st = r["note"].split("state=")[-1].split(";")[0].strip()
            name, pop = out.get(gid, (r["geo_name"], st, 0))[0], out.get(gid, (None, None, 0))[2]
            out[gid] = (r["geo_name"], st, pop + int(r["count"]))
    return out


def main():
    import geopandas as gpd
    import pandas as pd

    if "--fetch" in sys.argv:
        fetch()
    for p in (ADM2_FILE, ADM1_FILE):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p} — run: python sources/my_geo.py --fetch")
    gpkg = os.path.join(KONTUR, GPKG_NAME)
    if not os.path.exists(gpkg):
        raise SystemExit(f"missing {gpkg} — run: python sources/my_geo.py --fetch")
    if not os.path.exists(NORM):
        raise SystemExit(f"missing {NORM} — run: python sources/my.py")

    os.makedirs(GEO, exist_ok=True)

    adm2 = gpd.read_file(ADM2_FILE)
    adm1 = gpd.read_file(ADM1_FILE)
    if len(adm2) != ADM2_FEATURES:
        raise SystemExit(f"ADM2 has {len(adm2)} features, expected {ADM2_FEATURES}")
    print(f"geoBoundaries ADM2 {len(adm2)} polygons, ADM1 {len(adm1)}")

    # ---- 1. give every ADM2 polygon a state, by point-in-polygon against ADM1
    reps = gpd.GeoDataFrame(adm2[["shapeName"]].copy(),
                            geometry=adm2.geometry.representative_point(), crs=adm2.crs)
    sj = gpd.sjoin(reps, adm1[["shapeName", "geometry"]].rename(columns={"shapeName": "state"}),
                   how="left", predicate="within")
    sj = sj[~sj.index.duplicated(keep="first")].reindex(reps.index)
    if sj["state"].isna().any():
        raise SystemExit(f"{int(sj['state'].isna().sum())} ADM2 polygons fall in no ADM1 state")
    adm2 = adm2.copy()
    adm2["gb_state"] = sj["state"].to_numpy()

    # ---- 2. carve Putrajaya out of Sepang, then add it as its own unit
    pj = adm1[adm1["shapeName"].map(skey) == "putrajaya"]
    if len(pj) != 1:
        raise SystemExit(f"expected exactly one Putrajaya polygon in ADM1, got {len(pj)}")
    pj_geom = pj.geometry.iloc[0]

    m = adm2["shapeName"].map(dkey) == "sepang"
    if int(m.sum()) != 1:
        raise SystemExit(f"expected exactly one Sepang polygon, got {int(m.sum())}")
    sepang_before = adm2.loc[m, "geometry"].iloc[0]
    overlap = sepang_before.intersection(pj_geom).area
    if overlap <= 0:
        raise SystemExit("Putrajaya does not intersect Sepang — the layer has changed shape")
    adm2.loc[m, "geometry"] = sepang_before.difference(pj_geom)
    pj_area = gpd.GeoSeries([pj_geom], crs=adm2.crs).to_crs(3395).area.iloc[0] / 1e6
    print(f"  carved Putrajaya ({pj_area:.1f} km2) out of Sepang "
          f"({100.0 * overlap / pj_geom.area:.1f}% of it was inside)")

    adm2 = pd.concat([
        adm2[["shapeName", "gb_state", "geometry"]],
        gpd.GeoDataFrame({"shapeName": ["Putrajaya"], "gb_state": ["Putrajaya"]},
                         geometry=[pj_geom], crs=adm2.crs),
    ], ignore_index=True)
    adm2 = gpd.GeoDataFrame(adm2, geometry="geometry", crs=adm1.crs)
    if len(adm2) != DISTRICTS:
        raise SystemExit(f"{len(adm2)} polygons after adding Putrajaya, expected {DISTRICTS}")

    # ---- 3. join to the census districts on (state, district), both aliased
    cen = census_districts()
    if len(cen) != DISTRICTS:
        raise SystemExit(f"{len(cen)} census districts, expected {DISTRICTS}")
    by_key = {}
    for gid, (name, st, pop) in cen.items():
        by_key.setdefault((skey(st), dkey(name)), []).append((gid, name, st, pop))
    dup = {k: v for k, v in by_key.items() if len(v) > 1}
    if dup:
        raise SystemExit(f"census district keys collide: {dup}")

    adm2["k"] = [(skey(s), dkey(n)) for s, n in zip(adm2["gb_state"], adm2["shapeName"])]
    unmatched = [(n, s) for n, s, k in zip(adm2["shapeName"], adm2["gb_state"], adm2["k"])
                 if k not in by_key]
    if unmatched:
        raise SystemExit("polygons with no census district: " + repr(unmatched))
    used = [by_key[k][0] for k in adm2["k"]]
    if len(set(g for g, *_ in used)) != DISTRICTS:
        raise SystemExit("the join is not one-to-one")

    adm2["unit"] = [g for g, *_ in used]
    adm2["district"] = [n for _, n, _, _ in used]
    adm2["state"] = [s for _, _, s, _ in used]
    adm2["census_pop"] = [p for *_, p in used]
    total = int(adm2["census_pop"].sum())
    if total != CENSUS_POPULATION:
        raise SystemExit(f"census population {total:,}, expected {CENSUS_POPULATION:,}")
    print(f"  joined {DISTRICTS}/{DISTRICTS} districts, {total:,} people, "
          f"state verified spatially for every one")

    units = adm2[["unit", "district", "state", "census_pop", "geometry"]].copy()
    units.to_file(OUT_UNITS, layer="districts", driver="GPKG")
    print(f"\nwrote {OUT_UNITS} ({len(units)} districts)")

    # ---- 4. Kontur, joined on hex CENTROIDS so no hex is split across two districts
    hexes = gpd.read_file(gpkg)
    if len(hexes) == 0:
        raise SystemExit("Kontur gpkg read returned ZERO features")
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    if popcol is None:
        raise SystemExit(f"no population column in {list(hexes.columns)}")
    print(f"\nKontur hexes: {len(hexes):,}, crs={hexes.crs}, "
          f"population {hexes[popcol].sum():,.0f}")

    pts = gpd.GeoDataFrame({popcol: hexes[popcol].to_numpy()},
                           geometry=hexes.geometry.centroid, crs=hexes.crs).to_crs(units.crs)
    joined = gpd.sjoin(pts, units[["unit", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)

    outside = joined["unit"].isna()
    lost = float(pts.loc[outside, popcol].sum())
    tot = float(pts[popcol].sum())
    print(f"  hexes whose centroid is in no district: {int(outside.sum()):,} "
          f"({lost:,.0f} people, {100.0 * lost / tot:.3f}%)")
    print("     coastal overrun and the offshore islands; dropped. A LARGE number here "
          "\n     would mean the district cover has holes.")

    keep = ~outside
    out = gpd.GeoDataFrame(
        {"unit": joined.loc[keep, "unit"].to_numpy(),
         "pop": pts.loc[keep, popcol].to_numpy()},
        geometry=hexes.to_crs(units.crs).loc[keep.to_numpy(), "geometry"].to_numpy(),
        crs=units.crs)
    out.to_file(OUT_HEXES, layer="hexes", driver="GPKG")
    print(f"wrote {OUT_HEXES} ({len(out):,} hexes)")

    # ---- 5. the ratio band, and the lookup
    kp = out.groupby("unit")["pop"].sum()
    units = units.merge(kp.rename("kontur_pop"), left_on="unit", right_index=True, how="left")
    units["kontur_pop"] = units["kontur_pop"].fillna(0.0)

    empty = units[units["kontur_pop"] <= 0]
    if len(empty):
        print(f"\n  districts with NO Kontur population: {len(empty)}")
        for _, r in empty.iterrows():
            print(f"     {r['district']} ({r['state']}) census {r['census_pop']:,}")

    ratio = units["kontur_pop"].sum() / units["census_pop"].sum()
    print(f"\n  Kontur / census, country-wide: {ratio:.3f}  "
          f"(band {KONTUR_RATIO_MIN}-{KONTUR_RATIO_MAX}, 2020 census vs a 2023 grid)")
    if not (KONTUR_RATIO_MIN <= ratio <= KONTUR_RATIO_MAX):
        raise SystemExit("country ratio outside the band — check the join before trusting it")

    per = (units["kontur_pop"] / units["census_pop"]).replace([float("inf")], float("nan"))
    worst = units.assign(r=per).sort_values("r")
    print("\n  five lowest and five highest per-district ratios:")
    for _, r in list(worst.head(5).iterrows()) + list(worst.tail(5).iterrows()):
        print(f"     {r['district']:<22} {r['state']:<18} census {r['census_pop']:>9,}  "
              f"kontur {r['kontur_pop']:>10,.0f}  {r['r']:.2f}")

    with open(OUT_LOOKUP, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["unit", "district", "state", "census_pop", "kontur_pop"])
        for _, r in units.sort_values("unit").iterrows():
            w.writerow([r["unit"], r["district"], r["state"],
                        int(r["census_pop"]), round(float(r["kontur_pop"]), 1)])
    print(f"\nwrote {OUT_LOOKUP}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
