"""Dissolve COD 2018 atoms into the 2026 units fetch.py decided on.

Reads data/kyrgyzstan/crosswalk.csv and units.csv (run fetch.py first) and
writes data/kyrgyzstan/boundaries/adm{1,2,3}.gpkg with columns
code, name, name_cn (Russian name, shown as the tooltip's second line),
parent, group (the oblast / capital code).

Every level is a dissolve of the same atoms, so the levels nest exactly. Lake
Issyk-Kul is in no atom (COD's rayons stop at the shore) and so in no unit.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
import sys  # noqa: E402
import warnings  # noqa: E402

import geopandas as gpd  # noqa: E402
import pandas as pd  # noqa: E402

warnings.filterwarnings("ignore", message=".*geographic CRS.*")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import atoms as atoms_mod  # noqa: E402

OUT = os.path.abspath(os.path.join(HERE, "..", "..", "data", "kyrgyzstan"))


def main():
    cw = pd.read_csv(os.path.join(OUT, "crosswalk.csv"), dtype=str)
    units = pd.read_csv(os.path.join(OUT, "units.csv"), dtype=str)
    A = atoms_mod.build_atoms()[["atom", "geometry"]].merge(cw[["atom", "adm3"]], on="atom")
    assert len(A) == len(cw), "atoms changed since fetch.py ran"
    A["geometry"] = A.geometry.make_valid()
    u3 = units[units.level == "3"].set_index("code")
    u2 = units[units.level == "2"].set_index("code")
    u1 = units[units.level == "1"].set_index("code")

    g3 = A.dissolve("adm3").reset_index().rename(columns={"adm3": "code"})
    g3["name"] = g3.code.map(u3.name)
    g3["name_cn"] = g3.code.map(u3.name_ru)
    g3["parent"] = g3.code.map(u3.adm2)
    g3["group"] = g3.code.map(u3.adm1)
    missing = set(u3.index) - set(g3.code)
    assert not missing, f"level-3 units with no polygon: {missing}"

    g2 = g3[["parent", "geometry"]].dissolve("parent").reset_index().rename(columns={"parent": "code"})
    g2["name"] = g2.code.map(u2.name)
    g2["name_cn"] = g2.code.map(u2.name_ru)
    g2["parent"] = g2.code.map(u2.adm1)
    g2["group"] = g2.parent

    g1 = g3[["group", "geometry"]].dissolve("group").reset_index().rename(columns={"group": "code"})
    g1["name"] = g1.code.map(u1.name)
    g1["name_cn"] = g1.code.map(u1.name_ru)
    g1["group"] = g1.code

    d = os.path.join(OUT, "boundaries")
    os.makedirs(d, exist_ok=True)
    for n, g in [(1, g1), (2, g2), (3, g3)]:
        g = gpd.GeoDataFrame(g, geometry="geometry", crs="EPSG:4326")
        g.to_file(os.path.join(d, f"adm{n}.gpkg"), driver="GPKG")
        print(f"adm{n}: {len(g)} units")


if __name__ == "__main__":
    main()
