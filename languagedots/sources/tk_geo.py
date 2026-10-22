"""Tokelau placement layer: religiondots' Kontur hexes for the three atolls as one unit
-> data/geo/tk/tk_hexes.gpkg.

    python sources/tk_geo.py

religiondots/data/geo/tk/tk_hexes.gpkg (read in place) keys 17 Kontur 400 m hexes to Atafu (6),
Fakaofo (5) and Nukunonu (6), 1,837 Kontur people in all. The 2016 language table is national
(sources/tk_census.py), so the hexes are re-keyed to one unit `TK`, and every language's dots
follow the hexes' Kontur population (Atafu 642, Fakaofo 662, Nukunonu 533). Against the people present
on census night (413, 399, 385) that is 1.55, 1.66 and 1.38 times, so the three atolls' shares are
close to the census's (35%, 36%, 29% in Kontur; 34%, 33%, 32% present).
"""
import sys
from pathlib import Path

import geopandas as gpd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
SRC = HERE.parent / "religiondots" / "data" / "geo" / "tk" / "tk_hexes.gpkg"
OUT = HERE / "data" / "geo" / "tk" / "tk_hexes.gpkg"


def main():
    g = gpd.read_file(SRC)
    assert sorted(g["unit"].unique()) == ["Atafu", "Fakaofo", "Nukunonu"] and len(g) == 17
    per = g.groupby("unit")["pop"].sum()
    print("Kontur people by atoll:", per.round(0).to_dict(), "total", round(per.sum()))
    g["atoll"] = g["unit"]
    g["unit"] = "TK"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    g[["unit", "atoll", "pop", "geometry"]].to_file(OUT, layer="hexes", driver="GPKG")
    print(f"wrote {OUT} ({len(g)} hexes, one unit TK)")


if __name__ == "__main__":
    main()
