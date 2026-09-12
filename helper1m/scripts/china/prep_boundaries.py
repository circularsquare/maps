"""Build China's four admin levels from the single township shapefile.

Source: data/asia1m/china/xiangzhen.shp — 43,655 township polygons carrying
Chinese names for province / prefecture / county / township and nothing else
(no codes, no population). Because every level is a name column on the same
file, dissolving it gives levels that nest exactly, which the census-derived
populations then roll up through without any reconciliation.

Codes are synthetic: a 2/4/6/9-digit positional string where each level is a
prefix of the one below. The shapefile has no official GB/T 2260 codes to use,
and nothing joins to it by code — the Chinese name tuple is the only real key,
and it is kept in the output.

Writes helper1m/data/china/boundaries/adm{1,2,3,4}.gpkg.
"""
import os

# Cap BLAS threads before numpy loads — she is using the box.
os.environ.setdefault("OMP_NUM_THREADS", "6")

import sys
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd
import shapely

sys.path.insert(0, str(Path(__file__).resolve().parent))
from names import romanise

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "data/asia1m/china/xiangzhen.shp"
OUT = REPO_ROOT / "helper1m/data/china/boundaries"

# Chinese column names in the source shapefile.
COL = {1: "省", 2: "市", 3: "县", 4: "乡"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def assign_codes(gdf):
    """Positional hierarchical codes: province 2, prefecture 4, county 6, township 9."""
    def rank(series, width):
        order = {v: i + 1 for i, v in enumerate(sorted(series.unique()))}
        return series.map(order).map(lambda i: str(i).zfill(width))

    gdf["code1"] = rank(gdf[COL[1]], 2)
    gdf["code2"] = gdf["code1"] + gdf.groupby("code1")[COL[2]].transform(
        lambda s: rank(s, 2))
    gdf["code3"] = gdf["code2"] + gdf.groupby("code2")[COL[3]].transform(
        lambda s: rank(s, 2))
    gdf["code4"] = gdf["code3"] + gdf.groupby("code3")[COL[4]].transform(
        lambda s: rank(s, 3))
    return gdf


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    log(f"reading {SRC}")
    gdf = gpd.read_file(SRC)
    log(f"  {len(gdf)} townships, crs={gdf.crs}")

    # The source carries a leftover WKT text column truncated at 254 chars; the
    # real geometry is the shapefile's own. Drop it.
    gdf = gdf.drop(columns=[c for c in gdf.columns if c == "geom"])

    invalid = ~gdf.geometry.is_valid
    if invalid.any():
        log(f"  repairing {int(invalid.sum())} invalid geometries")
        gdf.loc[invalid, "geometry"] = shapely.make_valid(gdf.loc[invalid, "geometry"])

    gdf = assign_codes(gdf)

    # Latin names — one romanisation per distinct Chinese name, not per row.
    for lvl in (1, 2, 3, 4):
        col = COL[lvl]
        uniq = {n: romanise(n, lvl) for n in gdf[col].unique()}
        gdf[f"name{lvl}"] = gdf[col].map(uniq)
    log("  romanised names, e.g. " + " / ".join(
        str(gdf.iloc[0][f"name{l}"]) for l in (1, 2, 3, 4)))

    def write(lvl, frame):
        path = OUT / f"adm{lvl}.gpkg"
        frame.to_file(path, driver="GPKG", layer=f"adm{lvl}")
        log(f"  wrote {path.name}: {len(frame)} features")

    levels = {}
    # adm4 — the townships themselves.
    levels[4] = gdf[["code4", "name4", COL[4], "code3", "code1", "geometry"]].rename(
        columns={"code4": "code", "name4": "name", COL[4]: "name_cn",
                 "code3": "parent", "code1": "group"})
    write(4, levels[4])

    # adm3/2/1 — each dissolved straight from the townships, so every level is a
    # union of the same pieces and they nest exactly. Written as they finish:
    # the dissolves take minutes and a later failure should not discard them.
    for lvl, parent_lvl in ((3, 2), (2, 1), (1, None)):
        log(f"  dissolving townships -> adm{lvl}")
        t0 = time.time()
        cols = [f"code{lvl}", f"name{lvl}", COL[lvl], "geometry"]
        renames = {f"code{lvl}": "code", f"name{lvl}": "name", COL[lvl]: "name_cn"}
        if lvl != 1:
            # adm1 is its own group — carrying code1 separately would collide with
            # the code column, both in the groupby and in this rename.
            cols.insert(-1, "code1")
            renames["code1"] = "group"
        frame = gdf[cols].copy()
        if parent_lvl:
            frame["parent"] = gdf[f"code{parent_lvl}"]
        dis = frame.dissolve(by=f"code{lvl}", aggfunc="first").reset_index()
        dis = dis.rename(columns=renames)
        if lvl == 1:
            dis["group"] = dis["code"]
        keep = ["code", "name", "name_cn", "group", "geometry"]
        if parent_lvl:
            keep.insert(3, "parent")
        levels[lvl] = dis[keep]
        log(f"    {len(dis)} features in {time.time() - t0:.0f}s")
        write(lvl, levels[lvl])

    # Flat index of every unit, for the population pipeline to key against.
    rows = []
    for lvl in (1, 2, 3, 4):
        df = levels[lvl].drop(columns="geometry").copy()
        df["level"] = lvl
        rows.append(df)
    index = pd.concat(rows, ignore_index=True)
    index.to_csv(OUT.parent / "units.csv", index=False, encoding="utf-8")
    log(f"  wrote units.csv: {len(index)} rows")


if __name__ == "__main__":
    main()
