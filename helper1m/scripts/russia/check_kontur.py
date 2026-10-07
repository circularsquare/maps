"""Independent check of level 2: Kontur's population surface summed inside
each polygon, against Rosstat's 1 January 2025 figure.

Kontur Population (H3 r6 hexes of about 36 km2, 2023-11-01) was built from
GHSL and building footprints, not from Rosstat, which is what makes it a
check. The global file sits in religiondots (read only); it is unpacked once
into helper1m/data/russia/raw/ (about 500 MB; delete when done) and read for
Russia's two bounding boxes. (religiondots' own ru_grid_3km.gpkg was tried
first and dropped: it is clipped to the 2017 subject outlines, which leave
out Zelenograd, Kotlin island and other pieces, so whole towns read zero.)
Each hex's people are split between the polygons it overlaps by area. A
36 km2 hex is coarse beside a 100 km2 town, so a town reads low and the
district around it high by construction; the test that matters is the
pair, and the national total.

Prints the share of units within 10% and the worst offenders; writes
helper1m/data/russia/kontur_check.csv.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import geopandas as gpd  # noqa: E402
import pandas as pd  # noqa: E402

HELPER = Path(__file__).resolve().parents[2]
DATA = HELPER / "data" / "russia"
KONTUR_GZ = HELPER.parent / "religiondots" / "data" / "geo" / "kontur" / "kontur_population_20231101_r6.gpkg.gz"
KONTUR = DATA / "raw" / "kontur_population_20231101_r6.gpkg"


def load_grid():
    if not KONTUR.exists():
        import gzip
        import shutil
        print("unpacking Kontur r6 (once)...", flush=True)
        tmp = KONTUR.with_suffix(".tmp")
        with gzip.open(KONTUR_GZ, "rb") as src, open(tmp, "wb") as dst:
            shutil.copyfileobj(src, dst, 1 << 24)
        tmp.replace(KONTUR)
    parts = []
    for box in ((18, 41, 180, 82), (-180, 60, -168, 72)):
        b = gpd.GeoSeries.from_xy([box[0], box[2]], [box[1], box[3]], crs="EPSG:4326").to_crs("EPSG:3857")
        parts.append(gpd.read_file(KONTUR, bbox=tuple(b.total_bounds)))
    grid = pd.concat(parts, ignore_index=True)
    grid = grid[~grid.h3.duplicated()] if "h3" in grid else grid
    grid = gpd.GeoDataFrame(grid, geometry="geometry", crs="EPSG:3857").to_crs("EPSG:4326")
    grid = grid.rename(columns={"population": "pop"})
    return grid


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    adm2 = gpd.read_file(DATA / "boundaries" / "adm2.gpkg")[["code", "name", "parent", "geometry"]]
    adm2["geometry"] = adm2.geometry.simplify(0.002)
    pop = pd.read_csv(DATA / "population.csv", dtype={"code": str})
    p25 = pop[(pop.level == 2) & (pop.year == 2025)].set_index("code")["pop"]
    grid = load_grid()
    b = grid.geometry.bounds
    grid = grid[(b.maxx - b.minx) < 90]   # the few hexes torn by the antimeridian
    grid["hex_area"] = grid.to_crs("ESRI:54009").area
    ov = gpd.overlay(grid[["pop", "hex_area", "geometry"]], adm2[["code", "geometry"]],
                     how="intersection", keep_geom_type=True)
    ov["share"] = ov.to_crs("ESRI:54009").area / ov["hex_area"]
    ov["kpop"] = ov["pop"] * ov["share"].clip(upper=1)
    k = ov.groupby("code")["kpop"].sum()
    df = adm2.drop(columns="geometry").set_index("code")
    df["rosstat2025"] = p25
    df["kontur2023"] = k.reindex(df.index).fillna(0).round()
    df["ratio"] = df["kontur2023"] / df["rosstat2025"]
    df.sort_values("ratio").to_csv(DATA / "kontur_check.csv")
    within = ((df.ratio - 1).abs() <= 0.10).mean()
    w20 = ((df.ratio - 1).abs() <= 0.20).mean()
    big = df[df.rosstat2025 >= 100_000]
    df["km2"] = adm2.set_index("code").to_crs("ESRI:54009").area / 1e6
    large = df[df.km2 >= 1500]
    subj = df.groupby("parent")[["rosstat2025", "kontur2023"]].sum()
    subj["ratio"] = subj.kontur2023 / subj.rosstat2025
    print(f"subjects: within 10% {((subj.ratio - 1).abs() <= 0.10).sum()} of {len(subj)}, "
          f"min {subj.ratio.min():.2f} ({subj.ratio.idxmin()}), max {subj.ratio.max():.2f} ({subj.ratio.idxmax()})")
    print(f"units of 1,500 km2 or more ({len(large)}): within 10% "
          f"{((large.ratio - 1).abs() <= 0.10).mean():.1%}, within 20% {((large.ratio - 1).abs() <= 0.20).mean():.1%}")
    print(f"{len(df)} units; Kontur/Rosstat national {df.kontur2023.sum() / df.rosstat2025.sum():.3f}")
    print(f"within 10%: {within:.1%}, within 20%: {w20:.1%}, median ratio {df.ratio.median():.3f}")
    print(f"units >= 100k people ({len(big)}): within 10% {((big.ratio - 1).abs() <= 0.10).mean():.1%}")
    df["gap"] = (df.kontur2023 - df.rosstat2025).abs()
    print("largest absolute gaps:")
    for code, r in df.sort_values("gap", ascending=False).head(15).iterrows():
        print(f"  {code} {r['name']} ({r['parent']}): Rosstat {r.rosstat2025:,.0f} Kontur {r.kontur2023:,.0f} ({r.ratio:.2f})")
    print("lowest / highest ratios among units >= 20k:")
    mid = df[df.rosstat2025 >= 20_000].sort_values("ratio")
    for code, r in pd.concat([mid.head(6), mid.tail(6)]).iterrows():
        print(f"  {code} {r['name']} ({r['parent']}): Rosstat {r.rosstat2025:,.0f} Kontur {r.kontur2023:,.0f} ({r.ratio:.2f})")


if __name__ == "__main__":
    main()
