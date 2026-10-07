"""Mozambique placement layer: religiondots' Kontur 400 m hexes, each province cut into urban and rural.

    python sources/mz_geo.py        -> data/geo/mz/mz_hexes.gpkg  (unit = "MZ01/u", "MZ01/r", ...)

INE prints mother tongue by province, for urban and rural residence separately
(sources/mz_rgph.py). The split matters here more than almost anywhere: Portuguese is 38.3% of
urban Mozambicans aged 5+ and 5.1% of rural ones. So each province is two units and its hexes
are divided between them; Maputo Cidade, which the census counts as wholly urban and prints
without a residence split, is one.

THE HEXES are religiondots' `data/geo/mz/mz_hexes.gpkg` (read-only): Kontur 400 m population
hexagons assigned to 121 districts as of 2007 and two whole provinces (Cabo Delgado, Manica),
with `unit` and `pop`. The first four characters of every unit are the INE province code
(MZ01..MZ11), which is how religiondots' own countries/mz.py checks its districts against its
province table; asserted here: eleven provinces, every hex in one.

URBAN OR RURAL, as sources/ru_geo.py does it. Within a province, hexes are ranked by population
density (Kontur people per km2 of the hex) and the densest are urban until their population
reaches the census's urban share of the province (urban / total, people aged 5+, mother tongue
known or not). The hex that crosses the line goes to whichever side leaves the share closer.
Towns are the dense hexes, so this is a stand-in for INE's administrative line (cities and
vilas); it is not that line. A city whose municipal boundary takes in farmland (Nampula, Tete,
Pemba) has rural-looking hexes that INE counts urban, and the ranking hands those people to
the densest villages instead. Printed per province: the census's urban share and the share the
hexes got.
"""
import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from rdlink import RD_GEO  # noqa: E402

SRC = RD_GEO / "mz" / "mz_hexes.gpkg"
NORM = ROOT / "data" / "normalized" / "mz.csv"
OUT = ROOT / "data" / "geo" / "mz" / "mz_hexes.gpkg"
WHOLLY_URBAN = {"MZ11"}          # Maputo Cidade


def main():
    g = gpd.read_file(SRC)
    g["prov"] = g["unit"].astype(str).str[:4]
    print(f"  religiondots hexes: {len(g):,}, {g['unit'].nunique()} units in "
          f"{g['prov'].nunique()} provinces, {g['pop'].sum():,.0f} people")
    df = pd.read_csv(NORM)
    ur = df.pivot_table(index="geo_id", columns="area", values="count", aggfunc="sum").fillna(0)
    if set(ur.index) != set(g["prov"]) or len(ur) != 11:
        raise SystemExit(f"provinces differ: census only {sorted(set(ur.index) - set(g['prov']))}, "
                         f"hexes only {sorted(set(g['prov']) - set(ur.index))}")
    if set(ur.index[ur["rural"] == 0]) != WHOLLY_URBAN:
        raise SystemExit(f"provinces with no rural rows: {sorted(ur.index[ur['rural'] == 0])}")
    share = ur["urban"] / (ur["urban"] + ur["rural"])

    area = g.geometry.to_crs(6933).area.to_numpy() / 1e6
    g["dens"] = g["pop"].to_numpy() / np.maximum(area, 1e-6)
    g["ur"] = ""
    rows = []
    for prov, idx in g.groupby("prov").groups.items():
        sub = g.loc[idx].sort_values("dens", ascending=False)
        pop = sub["pop"].to_numpy(dtype=float)
        if prov in WHOLLY_URBAN:
            g.loc[sub.index, "ur"] = "u"
            rows.append((prov, 1.0, 1.0, len(pop), len(pop)))
            continue
        target = share[prov] * pop.sum()
        cum = np.cumsum(pop)
        before = cum - pop
        k = int((before < target).sum())          # hexes [0, k) urban
        if 0 < k <= len(pop) and abs(cum[k - 1] - target) > abs(before[k - 1] - target):
            k -= 1
        k = min(max(k, 1), len(pop) - 1)
        flags = np.array(["r"] * len(pop), dtype=object)
        flags[:k] = "u"
        g.loc[sub.index, "ur"] = flags
        rows.append((prov, share[prov], pop[:k].sum() / pop.sum(), k, len(pop)))
    g["unit"] = g["prov"] + "/" + g["ur"]

    rep = pd.DataFrame(rows, columns=["prov", "census_urban", "hex_urban", "urban_hexes", "hexes"])
    off = (rep["hex_urban"] - rep["census_urban"]).abs()
    print(f"  urban share, census vs hexes: worst {rep.loc[off.idxmax(), 'prov']} {off.max():.4f}")
    for r in rep.itertuples():
        dmin = g.loc[(g["prov"] == r.prov) & (g["ur"] == "u"), "dens"].min()
        print(f"    {r.prov}  census {r.census_urban:5.3f}  hexes {r.hex_urban:5.3f}  "
              f"{r.urban_hexes:6,}/{r.hexes:,} hexes urban, the sparsest at {dmin:,.0f}/km2")
    if off.max() > 0.01:
        raise SystemExit("a province's hexes miss the census urban share by more than a point")
    if g["unit"].nunique() != 21:
        raise SystemExit(f"{g['unit'].nunique()} units, expected 21 (10 provinces x 2 + Maputo)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    g[["unit", "pop", "geometry"]].to_file(OUT, driver="GPKG")
    print(f"  wrote {OUT.relative_to(ROOT)}: {len(g):,} hexes, {g['unit'].nunique()} units")


if __name__ == "__main__":
    main()
