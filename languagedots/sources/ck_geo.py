"""Cook Islands placement layer: religiondots' Kontur hexes re-keyed to islands
-> data/geo/ck/ck_hexes.gpkg.

    python sources/ck_geo.py

religiondots/data/geo/ck/ck_hexes.gpkg (read in place) is Kontur's 400 m grid for the country as one
unit (219 hexes, 16,958 people). The 2021 language table is by island (sources/ck_census.py), so
each hex goes to the nearest of the inhabited islands' centres. The islands are 40 km or more apart
(Mauke and Mitiaro 46, Manihiki and Rakahanga 40) and no hex centroid is farther than 20 km from
its centre (Penrhyn's reef is the largest, 17 km), so the nearest-centre rule has no close calls.
Island centres are rounded coordinates, not a boundary file; there is none in religiondots.

Suwarrow (one hex, 113 people in Kontur) has no row in the census table, which counts residents
of the 12 islands; the hex is dropped. Takutea and Manuae are uninhabited and have no hex.

CHECKS (asserted): every hex within 20 km of its island; every island of the table gets a hex;
Kontur against the census (people aged 5+ in the table, Kontur everyone) per island, normalised,
printed with the log correlation against 500 shuffles.
"""
import math
import random
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
SRC = HERE.parent / "religiondots" / "data" / "geo" / "ck" / "ck_hexes.gpkg"
OUT = HERE / "data" / "geo" / "ck" / "ck_hexes.gpkg"
CENTRES = {   # lon, lat
    "Rarotonga": (-159.78, -21.23), "Aitutaki": (-159.78, -18.86), "Mangaia": (-157.93, -21.93),
    "Atiu": (-158.12, -19.99), "Mauke": (-157.35, -20.15), "Mitiaro": (-157.70, -19.84),
    "Palmerston": (-163.17, -18.07), "Pukapuka": (-165.84, -10.88), "Nassau": (-165.43, -11.55),
    "Manihiki": (-161.00, -10.40), "Rakahanga": (-161.09, -10.03), "Penrhyn": (-158.07, -9.02),
}
NOT_IN_TABLE = {"Suwarrow": (-163.10, -13.25)}
MAX_KM = 20.0


def km(a, b):
    return math.hypot((a[0] - b[0]) * 111.2 * math.cos(math.radians((a[1] + b[1]) / 2)), (a[1] - b[1]) * 111.2)


def main():
    g = gpd.read_file(SRC)
    c = g.geometry.centroid
    names = list(CENTRES) + list(NOT_IN_TABLE)
    pos = {**CENTRES, **NOT_IN_TABLE}
    unit, dist = [], []
    for x, y in zip(c.x, c.y):
        d = [km((x, y), pos[n]) for n in names]
        j = int(np.argmin(d))
        unit.append(names[j])
        dist.append(d[j])
    g["unit"], g["km"] = unit, dist
    assert max(dist) < MAX_KM, max(dist)
    drop = g["unit"].isin(NOT_IN_TABLE)
    print(f"{len(g)} hexes; farthest from its island centre {max(dist):.1f} km; "
          f"dropping {int(drop.sum())} hex(es) at {sorted(set(g.loc[drop, 'unit']))} "
          f"({g.loc[drop, 'pop'].sum():,.0f} Kontur people, no census row)")
    g = g[~drop].copy()
    per = g.groupby("unit")["pop"].agg(["count", "sum"])
    assert set(per.index) == set(CENTRES), set(CENTRES) ^ set(per.index)
    census = pd.read_csv(HERE / "data" / "normalized" / "ck.csv").groupby("geo_id")["count"].sum()
    rows = [(u, float(census[u]), float(per.loc[u, "sum"])) for u in CENTRES]
    ratio = sum(r[2] for r in rows) / sum(r[1] for r in rows)
    print(f"  Kontur / census (people 5+) {ratio:.3f}")
    for u, ce, k in sorted(rows, key=lambda r: r[2] / r[1]):
        print(f"    {u:11s} census {ce:>8,.0f}  Kontur {k:>8,.0f}  hexes {int(per.loc[u, 'count']):>3}  "
              f"normalised {k / ce / ratio:.2f}")
    lc = [math.log(r[1]) for r in rows]
    lk = [math.log(r[2]) for r in rows]

    def pear(a, b):
        ma, mb = sum(a) / len(a), sum(b) / len(b)
        num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
        return num / math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    r = pear(lc, lk)
    rng = random.Random(0)
    best = max(abs(pear(lc, rng.sample(lk, len(lk)))) for _ in range(500))
    print(f"  log correlation r = {r:.3f} against a best of {best:.3f} over 500 shuffles")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    g[["unit", "pop", "geometry"]].to_file(OUT, layer="hexes", driver="GPKG")
    print(f"wrote {OUT} ({len(g)} hexes over {g['unit'].nunique()} islands)")


if __name__ == "__main__":
    main()
