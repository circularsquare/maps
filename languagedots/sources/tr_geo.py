"""Türkiye: placement layer for the 81 provinces (il) that sources/tr_konda.py writes counts for.

    python sources/tr_geo.py     -> data/geo/tr/tr_units.gpkg, data/geo/tr/tr_hexes.gpkg

BOUNDARIES: OCHA COD-AB Türkiye admin 1 (valid_on 2022-01-01, 81 provinces, pcodes TUR001..),
copied into data/raw/tr/tur_admin1.* from the asia1m project's download (maps/data/asia1m/turkey).
`unit` is the ASCII province name COD-AB carries in adm1_name, which is also the spelling of
the DGMM transcription (data/raw/tr/dgmm_gecici_koruma_iller_20261001.csv) and of religiondots'
İBBS-1 crosswalk. The join is asserted both ways: 81 = 81, nothing left over on either side.

PLACEMENT: Kontur 400 m hexes (the 2023-11-01 extract religiondots fetched, its .gz copied into
data/geo/kontur/), each hex to the province its centroid falls in, checked against the province
populations DGMM prints beside its counts (the address register's, 86,092,168 in all).
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import geopandas as gpd  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from _grid import hex_layer  # noqa: E402

RAW = os.path.join(ROOT, "data", "raw", "tr")
OUT = os.path.join(ROOT, "data", "geo", "tr")


def main():
    g = gpd.read_file(os.path.join(RAW, "tur_admin1.shp"))
    if len(g) != 81 or g["adm1_pcode"].nunique() != 81:
        raise SystemExit(f"COD-AB: {len(g)} provinces, expected 81")
    g["unit"] = g["adm1_name"].str.strip()
    pop = pd.read_csv(os.path.join(RAW, "dgmm_gecici_koruma_iller_20261001.csv"))
    a, b = set(g["unit"]), set(pop["province"])
    if a != b:
        raise SystemExit(f"names differ: COD-AB only {sorted(a - b)}; DGMM only {sorted(b - a)}")
    units = g[["unit", "adm1_pcode", "geometry"]].to_crs(4326)
    os.makedirs(OUT, exist_ok=True)
    units.to_file(os.path.join(OUT, "tr_units.gpkg"), layer="units", driver="GPKG")
    print(f"  {len(units)} provinces, joined to DGMM's 81 by name both ways")
    hex_layer("tr", units, census=dict(zip(pop["province"], pop["il_nufusu"])))


if __name__ == "__main__":
    main()
