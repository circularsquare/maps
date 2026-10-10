"""Oman's placement layer: religiondots' om_hexes.gpkg (read-only) with Kontur's false desert
blocks lowered. Writes data/geo/om/om_hexes.gpkg. Record: sources/om.md, "Placement layer".

    python sources/om_place.py

Why. Kontur OM reads 0.889 of the 2024 register nationally, but some desert wilayat far over it:
Al Mazyunah 491,949 against 11,117 (44x), Muqshin 28x, Thumrayt, Hayma, As Sunaynah and Madha
about 5x. The dots follow the register per wilaya (countries/om.py), so this never moved people
between wilayat; it moves them inside one. Thumrayt's Kontur put 29% of the wilaya on the Fasad
oil field (five hexes, one at 44,302/km2) and only 16% on Thumrayt town; Hayma's put 14% on one
empty hex 8 km from anything GeoNames names.

Rule. Outside Mazyunah town, Kontur reads no real Omani hex above 7,712/km2 (Mutrah, the densest
part of Muscat). So every hex above CAP = 8,000/km2 is lowered to CAP, except within SEAT_KM of a
GeoNames wilaya seat (PPLA/PPLA2/PPLC), where a dense hex is the town itself. The only hexes this
spares are Al Mazyunah town's, which is the wilaya's one town; capping it would hand its weight to
the villages Kontur inflates as much (Jajwal, Shifya, Mitan). Only weights inside a wilaya change:
no count moves, and the Gulf weighter's density term is per unit.
"""
import os
import sys
import zipfile
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rdlink import RD, RD_GEO  # noqa: E402

SRC = RD_GEO / "om" / "om_hexes.gpkg"
LOOKUP = RD_GEO / "om" / "om_lookup.csv"
GEONAMES = RD / "data" / "raw" / "om" / "geonames_OM.zip"
OUT = ROOT / "data" / "geo" / "om" / "om_hexes.gpkg"
CAP = 8_000.0          # people per km2 (UTM 40N hex area)
SEAT_KM = 5.0
REAL_MAX = 7_712       # Mutrah's densest hex: the bar CAP sits above
EXPECT_CAPPED = {"Al Mazyunah", "Thumrayt", "Hayma"}   # wilayat with a hex lowered
EXPECT_SPARED = {"Al Mazyunah"}                          # seat towns above CAP


def seats():
    cols = ["id", "name", "ascii", "alt", "lat", "lon", "fclass", "fcode", "cc", "cc2", "a1",
            "a2", "a3", "a4", "pop", "elev", "dem", "tz", "mod"]
    with zipfile.ZipFile(GEONAMES) as z:
        g = pd.read_csv(z.open("OM.txt"), sep="\t", header=None, names=cols, dtype=str, quoting=3)
    g = g[g["fcode"].isin(["PPLA", "PPLA2", "PPLC"])]
    return gpd.GeoDataFrame(g[["ascii", "fcode"]], crs=4326, geometry=gpd.points_from_xy(
        g["lon"].astype(float), g["lat"].astype(float))).to_crs(32640)


def main():
    h = gpd.read_file(SRC)
    if len(h) == 0 or set(h.columns) != {"unit", "pop", "geometry"}:
        raise SystemExit(f"{SRC}: {len(h)} rows, columns {list(h.columns)}")
    m = h.to_crs(32640)
    area = m.area.to_numpy() / 1e6
    pop = h["pop"].to_numpy(dtype=float)
    dens = pop / area
    s = seats()
    near = m.geometry.centroid.distance(s.geometry.union_all()).to_numpy() <= SEAT_KM * 1000
    unit = h["unit"].astype(str).to_numpy()

    # the bar: no hex away from a seat that is a known real city reaches CAP except artefacts
    over = dens > CAP
    spared = over & near
    capped = over & ~near
    if set(unit[capped]) != EXPECT_CAPPED or set(unit[spared]) != EXPECT_SPARED:
        raise SystemExit(f"capped in {sorted(set(unit[capped]))}, spared in "
                         f"{sorted(set(unit[spared]))}; expected {sorted(EXPECT_CAPPED)} and "
                         f"{sorted(EXPECT_SPARED)}")
    ok = dens[~np.isin(unit, list(EXPECT_CAPPED))]
    if ok.max() > REAL_MAX + 1:
        raise SystemExit(f"a hex outside the flagged wilayat reads {ok.max():,.0f}/km2 > {REAL_MAX}")
    new = np.where(capped, CAP * area, pop)

    # the register check, every wilaya, before and after
    lut = pd.read_csv(LOOKUP)
    reg = lut.groupby("unit")["pop"].sum()
    if len(reg) != 61 or set(reg.index) != set(unit):
        raise SystemExit("om_lookup's units do not match the hexes'")
    t = pd.DataFrame({"register": reg,
                      "kontur": pd.Series(pop).groupby(unit).sum(),
                      "after": pd.Series(new).groupby(unit).sum(),
                      "lowered": pd.Series(capped.astype(int)).groupby(unit).sum()})
    nat = t["kontur"].sum() / t["register"].sum()
    t["x_nat"] = t["kontur"] / t["register"] / nat
    pd.set_option("display.width", 200)
    print(f"Kontur / register nationally {nat:.3f}; per wilaya over that factor:")
    print(t.sort_values("x_nat", ascending=False).round(2).to_string())
    print(f"\n{capped.sum()} hexes lowered to {CAP:,.0f}/km2 "
          f"({(pop - new).sum():,.0f} Kontur people), {spared.sum()} spared within {SEAT_KM} km "
          f"of a seat; densest hex now {np.max(new / area):,.0f}/km2")
    for u in sorted(EXPECT_CAPPED):
        k = unit == u
        print(f"  {u}: {capped[k].sum()} lowered; share of Kontur weight in lowered hexes "
              f"{pop[k & capped].sum() / pop[k].sum():.1%} -> {new[k & capped].sum() / new[k].sum():.1%}")

    out = h.copy()
    out["kontur_pop"] = pop
    out["pop"] = new
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".tmp.gpkg")
    out.to_file(tmp, driver="GPKG")
    os.replace(tmp, OUT)
    print(f"wrote {len(out):,} hexes -> {OUT}")


if __name__ == "__main__":
    main()
