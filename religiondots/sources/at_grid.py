"""Austria — the placement layer: Kontur 400 m population hexagons, keyed to the drawn unit.

Writes data/geo/at/at_hexes.gpkg.

**AUSTRIA IS HALF ALPS AND THE GEMEINDEN DO NOT KNOW IT.** The drawn units average 35 km²,
which sounds fine until you look at where the big ones are: Sölden is 466 km², Neustift im
Stubaital 250, Mittelberg, Kals, Heiligenblut and the rest of the Hohe Tauern fringe the same
shape — enormous polygons whose people live along one valley floor and whose remaining nine
tenths are rock, glacier and Nationalpark. An equal share per polygon would draw a
disproportionate part of Tirol, Salzburg and Kärnten onto ice.

It also removes the lakes, which in Austria sit INSIDE the Gemeinden rather than between them:
the Neusiedler See is split among Burgenland communes, and the Attersee, Traunsee, Wörthersee
and Bodensee shore are all drawn into their municipalities' polygons. A population grid has no
hexes on open water, so §8.2c's problem does not arise rather than being patched.

**THE GRID IS 2023 AND THE CENSUS IS 2001, AND THAT IS FINE FOR EXACTLY ONE REASON.** Kontur
is used only as a WITHIN-UNIT weight: how many dots a Gemeinde gets is the census's answer and
nothing here can change it. The 22-year gap therefore cannot move a single dot between
Gemeinden; it can only place a dot inside a Gemeinde according to where people live now rather
than where they lived in 2001. At a median unit of 35 km² that is a small claim, and it is a
better one than "spread evenly over the polygon", which is the alternative. Do not read the
national ratio below as an error term — Austria really did grow from 8.03M to about 9.1M in
those 22 years, and that growth is not evenly spread (Vienna and its Umland gained, much of
Kärnten and inner Styria lost).

THE JOIN IS SPATIAL, on hex CENTROIDS, so no hex is split between two units.

Usage:
    python sources/at_grid.py --fetch    one 6.0 MB gzipped gpkg from Kontur
    python sources/at_grid.py            rebuild from data/raw/at/
"""

import gzip
import math
import os
import random
import shutil
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw", "at")
GEO = os.path.join(ROOT, "data", "geo", "at")
UNITS = os.path.join(GEO, "at_units.gpkg")
OUT = os.path.join(GEO, "at_hexes.gpkg")
CSV = os.path.join(ROOT, "data", "normalized", "at.csv")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_AT_20231101.gpkg.gz")
GZ_NAME = "kontur_population_AT_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_AT_20231101.gpkg"

EXPECTED_UNITS = 2380
CENSUS_POPULATION = 8032926 - 272          # Stallehr has no polygon; see at_geo.py


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    gz = os.path.join(RAW, GZ_NAME)
    gpkg = os.path.join(RAW, GPKG_NAME)
    if os.path.exists(gpkg) and os.path.getsize(gpkg) > 5_000_000:
        print("already have", gpkg)
        return
    if not os.path.exists(gz):
        print("GET", GZ_URL)
        r = requests.get(GZ_URL, timeout=1800, stream=True,
                         headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        with open(gz, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
        print(f"  {os.path.getsize(gz):,} bytes")
    with gzip.open(gz, "rb") as src, open(gpkg, "wb") as dst:
        shutil.copyfileobj(src, dst)
    with open(gpkg, "rb") as fh:
        magic = fh.read(16)
    if magic[:4] != b"SQLi":
        raise SystemExit(f"{gpkg} is not a GeoPackage -- starts {magic!r}")
    print(f"  unpacked {os.path.getsize(gpkg):,} bytes")


def pearson(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return num / den


def main():
    import geopandas as gpd
    import pandas as pd

    if "--fetch" in sys.argv:
        fetch()

    gpkg = os.path.join(RAW, GPKG_NAME)
    for p in (gpkg, UNITS, CSV):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p} -- run at_grid.py --fetch and at_geo.py first")

    hexes = gpd.read_file(gpkg)
    if len(hexes) == 0:
        raise SystemExit("Kontur gpkg read returned ZERO features")
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    if popcol is None:
        raise SystemExit(f"no population column in {list(hexes.columns)}")
    print(f"Kontur hexes: {len(hexes):,}, crs={hexes.crs}, "
          f"population {hexes[popcol].sum():,.0f}")

    units = gpd.read_file(UNITS)
    if len(units) != EXPECTED_UNITS:
        raise SystemExit(f"{UNITS} has {len(units)} units, expected {EXPECTED_UNITS}")

    # Kontur ships in EPSG:3857.  Take the centroid in the CRS the hexes were tiled in, then
    # reproject the POINTS -- reprojecting first and taking the centroid after moves it.
    cent = hexes.geometry.centroid
    pts = gpd.GeoDataFrame({popcol: hexes[popcol].to_numpy()},
                           geometry=cent, crs=hexes.crs).to_crs(units.crs)
    hexes = hexes.to_crs(units.crs)

    joined = gpd.sjoin(pts, units[["unit", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)

    outside = joined["unit"].isna()
    lost = float(pts.loc[outside, popcol].sum())
    print(f"\n  hexes whose centroid is outside every unit: {int(outside.sum()):,} "
          f"({lost:,.0f} people, {100.0 * lost / pts[popcol].sum():.3f}%)")
    print("     Kontur's AT extract overruns the border, and GISCO's 1:1,000,000 outline is "
          "coarser\n     than the grid, so a thin rim of hexes falls outside; dropped.")

    keep = ~outside
    out = gpd.GeoDataFrame(
        {"unit": joined.loc[keep, "unit"].to_numpy(),
         "pop": pts.loc[keep, popcol].to_numpy(dtype=float)},
        geometry=hexes.geometry[keep.to_numpy()].to_numpy(), crs=units.crs)

    # A UNIT ABSENT FROM THE PLACEMENT LAYER HAS NO GEOMETRY TO DRAW INTO AND ITS PEOPLE ARE
    # DROPPED, NOT SPREAD.  Rattenberg is 0.11 km² — the smallest town in Austria — and no
    # 400 m hex has its centroid inside it, so it falls out of a centroid join entirely.
    # Mauritius's rule: give such a unit its own polygon as its placement geometry rather
    # than losing it.  Kontur's resolution floor is a fact about the grid, not about the
    # place ([[reference_kontur_resolution_floor]]).
    df0 = pd.read_csv(CSV)
    census_tot = (df0[df0["source_category"] == "Insgesamt"]
                  .set_index("geo_id")["count"].to_dict())
    per = out.groupby("unit")["pop"].agg(["size", "sum"])
    missing = sorted(set(units["unit"]) - set(per.index))
    if missing:
        if len(missing) > 5:
            raise SystemExit(f"{len(missing)} units with no populated hex, which is too many "
                             f"to be the resolution floor: {missing[:12]}")
        add = units[units["unit"].isin(missing)][["unit", "geometry"]].copy()
        add["pop"] = [float(census_tot.get(u, 1.0)) for u in add["unit"]]
        print("  %d unit(s) smaller than the grid, given their own polygon instead: %s"
              % (len(missing), ", ".join(
                  "%s (%s, %.2f km²)" % (u, units.loc[units["unit"] == u, "name"].iloc[0],
                                         units[units["unit"] == u].to_crs(3035)
                                         .geometry.area.iloc[0] / 1e6)
                  for u in missing)))
        out = gpd.GeoDataFrame(pd.concat([out, add[["unit", "pop", "geometry"]]],
                                         ignore_index=True),
                               geometry="geometry", crs=units.crs)
        per = out.groupby("unit")["pop"].agg(["size", "sum"])
    zero = sorted(per.index[per["sum"] <= 0])
    if zero:
        raise SystemExit(f"{len(zero)} units whose hexes sum to zero: {zero[:12]}")
    print(f"  every one of the {EXPECTED_UNITS} drawn units has hexes: "
          f"{per['size'].min():,}–{per['size'].max():,} each")

    tot = float(out["pop"].sum())
    print(f"\n  Kontur 2023 {tot:,.0f} vs census 2001 {CENSUS_POPULATION:,} — "
          f"ratio {tot / CENSUS_POPULATION:.3f}")
    print("     Austria grew by about a seventh between the two dates, so this ratio is a "
          "fact about\n     the country and not an error term. The grid is a WITHIN-unit "
          "weight only.")

    # ---------------------------------------------------------------- the two nulls
    df = pd.read_csv(CSV)
    tot_by_unit = (df[df["source_category"] == "Insgesamt"]
                   .set_index("geo_id")["count"].to_dict())
    rows = [(u, float(tot_by_unit[u]), float(per.loc[u, "sum"]))
            for u in units["unit"] if u in tot_by_unit and per.loc[u, "sum"] > 0]
    print(f"\n  ratio band, Kontur vs census, over {len(rows)} units:")
    ratios = sorted((k / c, u) for u, c, k in rows)
    for q, label in ((0.01, " 1st pct"), (0.25, "25th pct"), (0.50, "  median"),
                     (0.75, "75th pct"), (0.99, "99th pct")):
        print("    %s  %.2f" % (label, ratios[int(q * (len(ratios) - 1))][0]))
    print("    lowest  %.2f (%s)   highest  %.2f (%s)"
          % (ratios[0][0], ratios[0][1], ratios[-1][0], ratios[-1][1]))
    print("     A 22-year gap makes this band WIDE BY CONSTRUCTION and it is reported rather "
          "than\n     asserted: a Gemeinde that doubled since 2001 reads 2.0 and is not a "
          "join error.")

    lc = [math.log(c) for _, c, _ in rows]
    lk = [math.log(k) for _, _, k in rows]
    r_true = pearson(lc, lk)
    rng = random.Random(0)
    perm = []
    for _ in range(500):
        sh = list(lk)
        rng.shuffle(sh)
        perm.append(abs(pearson(lc, sh)))
    perm.sort()
    beat = sum(1 for x in perm if x >= r_true)
    print(f"\n  CORRELATION control: r = {r_true:.4f} against a best of {perm[-1]:.4f} over "
          f"500 shuffles ({beat} reach it).")
    if beat > 0 or r_true < 0.85:
        raise SystemExit(f"census and Kontur populations correlate at r={r_true:.4f}, which "
                         f"{beat} of 500 random pairings reach -- the join in at_geo.py is "
                         "not carrying information")
    print("     With 2,380 uneven units the correlation is the check that discriminates here; "
          "the\n     band cannot, for the vintage reason above. Zimbabwe is this the other "
          "way round.")

    os.makedirs(GEO, exist_ok=True)
    out.to_file(OUT, layer="hexes", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out):,} hexes)")


if __name__ == "__main__":
    main()
