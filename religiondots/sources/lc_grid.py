"""Saint Lucia — the placement layer: Kontur 400 m population hexagons, keyed to district.

Writes data/geo/lc/lc_hexes.gpkg.

**THE GRID IS HERE FOR CASTRIES AND FOR THE INTERIOR.** Saint Lucia's ten districts run
19.3 km² (Canaries) to 123.3 km² (Dennery), and the island is a volcanic ridge: the centre
is forest reserve with almost nobody in it and every settlement is on the coast road.
Uniform scatter inside a district would put dots on the Barre de l'Isle and in the Edmund
Forest. **Castries alone holds 60,614 of 171,834 people on 87.4 km²**, nearly all of them in
the city and the belt from Bois d'Orange down to Cul de Sac, and the rest of the parish is
mountain.

**THE COASTLINE SNAP, AS IN BARBADOS (§9au), THE BAHAMAS (§9ar) AND CAYMAN (§9at).** A plain
`within` join leaves hexes offshore of COD's coastline; the histogram prints on every run and
anything within 1 km is snapped to the nearest district. Saint Lucia is one island with two
inhabited offshore rocks of no population, so the snap is pure recovery.

**NATIONALLY THE GRID AND THE CENSUS AGREE — 1.07x — AND PER DISTRICT THEY DO NOT, RUNNING
0.58x IN LABORIE TO 1.54x IN ANSE LA RAYE.** That is a 2.7-fold spread, as wide as the one
that turned out to be a boundary error in Cayman (§9at), so it was checked the same way and
**it is not one**: re-running the whole join against geoBoundaries' independent boundary set
moves the rms deviation from 1 by 0.003, from 0.320 to 0.317 (`sources/lc_geo.py` has the
measurement). Both boundary files place the same people. The disagreement is between the two
*sources*, and it has a direction:

  * **Kontur over-attributes to the built-up north-west** — Castries 1.44x, Anse La Raye
    1.54x, Canaries 1.33x — and **under-attributes to the dispersed rural south**, Laborie
    0.58x and Choiseul 0.67x. That is the standard bias of a building-footprint model: a
    dense roofline is easy to see and scattered rural housing is not.
  * **And CSO's own largest undercount corrections were in the same places Kontur sees
    fewest people.** Laborie was weighted 1.507 and Micoud 1.446, the two heaviest in the
    country; they come out at 0.58x and 0.80x here. Two methods that share no inputs both
    find fewer people in rural Saint Lucia than the census asserts, which is worth saying
    plainly and is not something this file can resolve.

**NONE OF IT CHANGES HOW MANY DOTS A DISTRICT GETS** (§9t): the counts are the census's and
only the shape *within* a district comes from the grid. The band below is set on the national
figure, which is what a weight can be checked on.

Usage:
    python sources/lc_grid.py --fetch    one ~55 KB gzipped gpkg from Kontur
    python sources/lc_grid.py            rebuild from data/raw/lc/
"""

import gzip
import os
import shutil
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "lc")
GEO = os.path.join(ROOT, "data", "geo", "lc")
DISTRICTS = os.path.join(GEO, "lc_districts.gpkg")
OUT = os.path.join(GEO, "lc_hexes.gpkg")
NORM = os.path.join(ROOT, "data", "normalized", "lc.csv")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_LC_20231101.gpkg.gz")
GZ_NAME = "kontur_population_LC_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_LC_20231101.gpkg"

EXPECTED_DISTRICTS = 10

SNAP_M = 1000
UTM = 32620                     # UTM 20N covers Saint Lucia

# Kontur is modelled from GHSL, HRSL and building footprints; it is not the census (§12,
# North Macedonia). The comparison is against the census's RESIDENT population, which is
# the household population plus institutions -- the closest thing the report publishes to
# what a building-footprint model is trying to estimate.
HOUSEHOLD_POPULATION = 171_834
RESIDENT_POPULATION = 172_948
MIDYEAR_ESTIMATE_2022 = 182_289
KONTUR_TOLERANCE = 0.30

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    gz = os.path.join(RAW, GZ_NAME)
    gpkg = os.path.join(RAW, GPKG_NAME)
    if os.path.exists(gpkg) and os.path.getsize(gpkg) > 50_000:
        print("already have", gpkg)
        return
    if not os.path.exists(gz):
        print("GET", GZ_URL)
        r = requests.get(GZ_URL, timeout=1800, stream=True, headers={"User-Agent": UA})
        r.raise_for_status()
        with open(gz + ".part", "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
        os.replace(gz + ".part", gz)
        print(f"  {os.path.getsize(gz):,} bytes")
    with gzip.open(gz, "rb") as src, open(gpkg + ".part", "wb") as dst:
        shutil.copyfileobj(src, dst)
    os.replace(gpkg + ".part", gpkg)
    with open(gpkg, "rb") as fh:
        magic = fh.read(16)
    if magic[:4] != b"SQLi":
        raise SystemExit(f"{gpkg} is not a GeoPackage -- starts {magic!r}")
    print(f"  unpacked {os.path.getsize(gpkg):,} bytes")


def main():
    import geopandas as gpd
    import pandas as pd

    if "--fetch" in sys.argv:
        fetch()

    gpkg = os.path.join(RAW, GPKG_NAME)
    if not os.path.exists(gpkg):
        raise SystemExit(f"missing {gpkg} -- run with --fetch first")
    if not os.path.exists(DISTRICTS):
        raise SystemExit(f"missing {DISTRICTS} -- run sources/lc_geo.py first")

    hexes = gpd.read_file(gpkg)
    if len(hexes) == 0:
        raise SystemExit("Kontur gpkg read returned ZERO features")
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    if popcol is None:
        raise SystemExit(f"no population column in {list(hexes.columns)}")
    print(f"Kontur hexes: {len(hexes):,}, crs={hexes.crs}, "
          f"population {hexes[popcol].sum():,.0f}")

    dis = gpd.read_file(DISTRICTS)
    if len(dis) != EXPECTED_DISTRICTS:
        raise SystemExit(f"{DISTRICTS} has {len(dis)} districts, "
                         f"expected {EXPECTED_DISTRICTS}")

    # §8.2: the grid has to be FINER than the tier it weights, or it is doing nothing
    # (Saint Vincent, §9ac). Measured rather than assumed.
    hex_km2 = float((hexes.to_crs(UTM).area / 1e6).median())
    unit_km2 = float((dis.to_crs(UTM).area / 1e6).median())
    print(f"  hex {hex_km2:.2f} km2 against a median district of {unit_km2:.1f} km2 "
          f"— {unit_km2 / hex_km2:.0f}x finer")
    if hex_km2 >= unit_km2:
        raise SystemExit("the placement grid is coarser than the counting tier, so it "
                         "cannot weight it")

    # Kontur ships in EPSG:3857. Take the centroid in the CRS the hexes were tiled in,
    # then reproject the POINTS -- reprojecting first and taking the centroid after
    # moves it.
    cent = hexes.geometry.centroid
    pts = gpd.GeoDataFrame({popcol: hexes[popcol].to_numpy()},
                           geometry=cent, crs=hexes.crs).to_crs(dis.crs)
    hexes = hexes.to_crs(dis.crs)

    joined = gpd.sjoin(pts, dis[["unit", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)
    unit = joined["unit"].copy()        # NaN where the hex is outside every district

    outside = joined["unit"].isna().to_numpy()
    adrift = float(pts.loc[outside, popcol].sum())
    print(f"\n  hexes whose centroid is outside every district: {int(outside.sum()):,} "
          f"({adrift:,.0f} people, {100.0 * adrift / pts[popcol].sum():.3f}%)")

    if outside.any():
        m_pts = pts[outside].to_crs(UTM)
        near = gpd.sjoin_nearest(m_pts, dis.to_crs(UTM)[["unit", "geometry"]],
                                 how="left", distance_col="dist_m")
        near = near[~near.index.duplicated(keep="first")].reindex(m_pts.index)

        print("     distance from one of those to the nearest district:")
        edges = [0, 100, 250, 500, SNAP_M, float("inf")]
        names = ["<100 m", "100-250 m", "250-500 m", f"500 m-{SNAP_M / 1000:g} km",
                 f">{SNAP_M / 1000:g} km"]
        for lo, hi, lab in zip(edges[:-1], edges[1:], names):
            sel = near[(near["dist_m"] >= lo) & (near["dist_m"] < hi)]
            if len(sel):
                print(f"       {lab:<14} {len(sel):>4} hexes  "
                      f"{sel[popcol].sum():>8,.0f} people")

        snap = near["dist_m"] <= SNAP_M
        idx = near.index[snap]
        unit.loc[idx] = near.loc[idx, "unit"]
        print(f"     snapped to the nearest district within {SNAP_M:,} m: "
              f"{int(snap.sum()):,} hexes ({float(near.loc[snap, popcol].sum()):,.0f} "
              "people).\n     Saint Lucia is one island whose only offshore land is "
              "uninhabited rock, so\n     this is COD's coastline against a 400 m hex "
              "and nothing else.")

    # NaN, not None — see sources/bs_grid.py for what `!= None` did here.
    keep = unit.notna().to_numpy()
    lost = float(pts.loc[~keep, popcol].sum())
    print(f"     still outside after the snap: {int((~keep).sum()):,} hexes "
          f"({lost:,.0f} people, {100.0 * lost / pts[popcol].sum():.3f}%)")

    out = gpd.GeoDataFrame(
        {"unit": unit[keep].to_numpy(),
         "pop": pts.loc[keep, popcol].to_numpy(dtype=float)},
        geometry=hexes.geometry[keep].to_numpy(), crs=dis.crs)

    got, want = set(out["unit"].unique()), set(dis["unit"])
    if got != want or out["unit"].isna().any():
        raise SystemExit(f"hex layer carries units {sorted(got, key=str)}, expected "
                         f"{sorted(want)}")

    per = out.groupby("unit")["pop"].agg(["size", "sum"])
    zero = sorted(per.index[per["sum"] <= 0])
    if zero:
        raise SystemExit(f"districts whose hexes sum to zero population: {zero}")
    print(f"  every one of the {EXPECTED_DISTRICTS} districts has hexes: "
          f"{per['size'].min():,}-{per['size'].max():,} each")

    tot = float(out["pop"].sum())
    ratio = tot / RESIDENT_POPULATION
    print(f"\n  Kontur {tot:,.0f} vs the census's RESIDENT population "
          f"{RESIDENT_POPULATION:,} — ratio {ratio:.3f}")
    print(f"     (against the 2022 mid-year estimate {MIDYEAR_ESTIMATE_2022:,} it is "
          f"{tot / MIDYEAR_ESTIMATE_2022:.3f})")
    if abs(ratio - 1.0) > KONTUR_TOLERANCE:
        raise SystemExit(f"Kontur and the resident population disagree by "
                         f"{abs(ratio - 1) * 100:.0f}%, which is too much for a weight")

    cen = pd.read_csv(NORM, dtype={"geo_id": str}, keep_default_na=False, na_values=[])
    cen = cen[(cen["geo_level"] == "district") & (cen["source_category"] == "Total")]
    cen = dict(zip(cen["geo_id"], pd.to_numeric(cen["count"])))
    print("\n  per-district Kontur/census ratio. NOT a boundary error — geoBoundaries' "
          "independent\n  set moves the rms deviation from 1 by 0.003 (sources/lc_geo.py) "
          "— but a real\n  disagreement between a building-footprint model and a census "
          "that weighted itself up:")
    rows = []
    for unit_id, row in per.iterrows():
        nm = dis.loc[dis["unit"] == unit_id, "name"].iloc[0]
        rows.append((row["sum"] / cen[unit_id], nm, int(row["size"]), cen[unit_id]))
    for r, nm, n, c in sorted(rows):
        print(f"      {nm:<14} {r:5.2f}x   {n:>5,} hexes   census {c:>7,}")

    os.makedirs(GEO, exist_ok=True)
    out.to_file(OUT, layer="hexes", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out):,} hexes)")


if __name__ == "__main__":
    main()
