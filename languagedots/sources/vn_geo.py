"""Vietnam placement layer: religiondots' 400m Kontur hexes for the 63 provinces, each province
split into an urban and a rural part -> data/geo/vn/vn_hexes.gpkg (unit = "<GSO code>-u|-r").

    python sources/vn_geo.py

WHY. The 2019 census gives every ethnic group's urban and rural count per province, and the two
differ sharply where it matters (Lam Dong: Kinh in Da Lat, Koho in the hills; Hoa: 70% urban).
This moves people only within the province the census counted them in (AGENT_BRIEF section
4.4), so it needs no ask.

HOW. Within each province, hexes are ranked by Kontur population (densest first), and the
densest hexes holding the province's census urban share of Kontur's population are the urban
part; the rest are rural. A density proxy for the urban boundary: Vietnam's official urban
areas (phuong, thi tran) are not drawn here. The hex that crosses the threshold goes urban, so
every province with any urban population has at least one urban hex.

Checks, asserted: the 63 codes are exactly vn.csv's; every unit vn.csv uses has hexes with
population; no hex is lost (urban + rural hex count = the province's).
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from rdlink import RD_GEO  # noqa: E402

SRC = RD_GEO / "vn" / "vn_grid_400m.gpkg"
NORM = HERE / "data" / "normalized" / "vn.csv"
OUT = HERE / "data" / "geo" / "vn" / "vn_hexes.gpkg"


def main():
    g = gpd.read_file(SRC)
    g["unit"] = g["unit"].astype(str).str.zfill(2)
    df = pd.read_csv(NORM, dtype={"geo_id": str})
    tot = df.groupby("geo_id")["count"].sum()
    prov = tot.index.str[:-2]
    codes = set(prov)
    if codes != set(g["unit"]):
        raise SystemExit(f"codes differ: {sorted(codes ^ set(g['unit']))}")
    if len(codes) != 63:
        raise SystemExit(f"{len(codes)} provinces")

    g["new"] = None
    rows = []
    for u, idx in g.groupby("unit").groups.items():
        sub = g.loc[idx]
        urb = tot.get(f"{u}-u", 0)
        rur = tot.get(f"{u}-r", 0)
        share = urb / (urb + rur)
        order = sub["pop"].sort_values(ascending=False, kind="mergesort")
        cum = order.cumsum().to_numpy()
        target = share * cum[-1]
        n_urb = 0 if urb == 0 else int(np.searchsorted(cum, target, side="left")) + 1
        n_urb = min(n_urb, len(order) - (1 if rur else 0))
        lab = pd.Series(f"{u}-r", index=order.index)
        lab.iloc[:n_urb] = f"{u}-u"
        g.loc[order.index, "new"] = lab
        upop = order.iloc[:n_urb].sum()
        rows.append((u, share, n_urb, len(order), upop / cum[-1]))
    if g["new"].isna().any():
        raise SystemExit("hexes left without a unit")
    g["unit"] = g["new"]
    g = g.drop(columns="new")
    used = set(tot.index)
    have = g.groupby("unit")["pop"].sum()
    missing = [x for x in used if have.get(x, 0) <= 0]
    if missing:
        raise SystemExit(f"units with no populated hexes: {missing}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    g[["unit", "pop", "geometry"]].to_file(OUT, driver="GPKG")
    r = pd.DataFrame(rows, columns=["unit", "census_urban", "urban_hexes", "hexes", "kontur_urban"])
    print(f"  wrote {OUT.name}: {len(g):,} hexes on {g['unit'].nunique()} units")
    print("  urban hexes, smallest and largest shares of the province's hexes:")
    r["hex_share"] = r["urban_hexes"] / r["hexes"]
    print(r.sort_values("hex_share").iloc[[0, 1, 2, -3, -2, -1]].to_string(index=False))


if __name__ == "__main__":
    main()
