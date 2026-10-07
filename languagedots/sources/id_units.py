"""Indonesia: one row per placement unit of religiondots' hex layer -> data/geo/id/id_units.csv.

    python sources/id_units.py

religiondots' `data/geo/id/id_hexes.gpkg` (read-only) keys 844,370 Kontur hexes to 5,122
kecamatan, 89 regencies and one residual (`65`). This writes, per unit: its 2010 province (the
first two digits of its BPS code, `65` -> `64`, as countries/id.py), its Kontur population, and
its population-weighted centre. sources/id_shareout.py reads it to say how near each province's
people live to a language's Glottolog point; countries/id.py computes the same thing from the
layer itself when it places dots.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rdlink import RD_GEO  # noqa: E402

OUT = ROOT / "data" / "geo" / "id" / "id_units.csv"


def units(g):
    """GeoDataFrame of hexes (unit, pop, geometry) -> DataFrame unit, prov, pop, lon, lat."""
    u = g["unit"].astype(str)
    if not u.str.fullmatch(r"\d{2}|\d{4}|\d{7}").all():
        raise SystemExit("id_hexes.gpkg: a unit id is not a 2-, 4- or 7-digit BPS code")
    pt = g.geometry.representative_point()
    pop = g["pop"].to_numpy(dtype=float)
    df = pd.DataFrame({"unit": u, "pop": pop, "wx": pt.x.to_numpy() * pop,
                       "wy": pt.y.to_numpy() * pop, "x": pt.x.to_numpy(), "y": pt.y.to_numpy()})
    a = df.groupby("unit").agg(pop=("pop", "sum"), wx=("wx", "sum"), wy=("wy", "sum"),
                               x=("x", "mean"), y=("y", "mean"))
    has = a["pop"] > 0
    a["lon"] = np.where(has, a["wx"] / a["pop"].where(has, 1), a["x"])
    a["lat"] = np.where(has, a["wy"] / a["pop"].where(has, 1), a["y"])
    a = a.reset_index()
    a["prov"] = a["unit"].str[:2].replace({"65": "64"})
    return a[["unit", "prov", "pop", "lon", "lat"]]


def main():
    g = gpd.read_file(RD_GEO / "id" / "id_hexes.gpkg")
    a = units(g)
    assert a["prov"].nunique() == 33, sorted(a["prov"].unique())
    assert abs(a["pop"].sum() - g["pop"].sum()) < 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    a.to_csv(OUT, index=False, float_format="%.5f")
    print(f"wrote {OUT}: {len(a):,} units, {a['pop'].sum():,.0f} people, 33 provinces")


if __name__ == "__main__":
    main()
