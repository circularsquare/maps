"""Re-apply names.py to the built boundary files without re-running the dissolves.

prep_boundaries.py takes about six minutes, almost all of it dissolving, and
the Latin names are the only thing that changes when names.py is fixed. This
rewrites just the `name` column of boundaries/adm{1,2,3,4}.gpkg, plus the name
columns of units.csv and township_pop2020.csv, then build_country.py picks the
new names up. Geometry, codes and name_cn are untouched.

Each gpkg is written to a temp file and swapped in, so a failure mid-write
leaves the original intact.
"""
import os

# Cap BLAS threads before numpy loads — she is using the box.
os.environ.setdefault("OMP_NUM_THREADS", "6")

import io
import sys
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import geopandas as gpd
import pandas as pd
from names import romanise

DATA = Path(__file__).resolve().parents[2] / "data/china"
BOUNDS = DATA / "boundaries"


def main():
    new_by_code = {}
    for lvl in (1, 2, 3, 4):
        t0 = time.time()
        path = BOUNDS / f"adm{lvl}.gpkg"
        gdf = gpd.read_file(path)
        uniq = {n: romanise(n, lvl) for n in gdf["name_cn"].unique()}
        new = gdf["name_cn"].map(uniq)
        diff = new != gdf["name"]
        pairs = list(zip(gdf.loc[diff, "name"].head(4), new[diff].head(4)))
        gdf["name"] = new
        new_by_code.update(zip(gdf["code"], gdf["name"]))

        tmp = path.with_name(path.stem + ".tmp.gpkg")
        if tmp.exists():
            tmp.unlink()
        gdf.to_file(tmp, driver="GPKG", layer=f"adm{lvl}")
        os.replace(tmp, path)
        print(f"adm{lvl}: {int(diff.sum())} of {len(gdf)} names changed, "
              f"{time.time() - t0:.0f}s")
        for old, now in pairs:
            print(f"    {old}  ->  {now}")

    units = pd.read_csv(DATA / "units.csv",
                        dtype={"code": str, "parent": str, "group": str})
    units["name"] = units["code"].map(new_by_code).fillna(units["name"])
    units.to_csv(DATA / "units.csv", index=False, encoding="utf-8")
    towns = pd.read_csv(DATA / "township_pop2020.csv", dtype={"code": str})
    towns["name"] = towns["code"].map(new_by_code).fillna(towns["name"])
    towns.to_csv(DATA / "township_pop2020.csv", index=False, encoding="utf-8")
    print("units.csv and township_pop2020.csv updated")


if __name__ == "__main__":
    main()
