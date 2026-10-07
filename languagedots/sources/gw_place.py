"""Guinea-Bissau placement layer: religiondots' gw_hexes (9 regiões) re-keyed to one national
unit, the região kept in `reg`.

    python sources/gw_place.py      -> data/geo/gw/gw_hexes.gpkg  (unit, reg, pop)

The language table is national (sources/gw_rgph.py), so every hex's `unit` is "Guinea-Bissau".
countries/gw.py weights each language's dots by data/normalized/gw_place.csv's split of that
language across the regiões (through the etnias that name it), then by Kontur population
inside a região. religiondots' layer (read-only) is COD-AB ADM1 over Kontur 2023 400 m hexes,
coast-snapped (religiondots/sources/gw.md §6).

CHECKS: 13,414 hexes, the 9 regiões, the population is religiondots' to the person.
"""
import sys
from pathlib import Path

import geopandas as gpd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from rdlink import RD_GEO  # noqa: E402

OUT = HERE / "data" / "geo" / "gw" / "gw_hexes.gpkg"
UNIT = "Guinea-Bissau"


def say(ok, msg):
    print(("  ok  " if ok else "  FAIL") + "  " + msg)
    if not ok:
        raise SystemExit("check failed: " + msg)


def main():
    hexes = gpd.read_file(RD_GEO / "gw" / "gw_hexes.gpkg")
    say(len(hexes) == 13_414 and hexes["unit"].nunique() == 9,
        f"{len(hexes):,} hexes in {hexes['unit'].nunique()} regiões")
    hexes["reg"] = hexes["unit"].astype(str)
    hexes["unit"] = UNIT
    OUT.parent.mkdir(parents=True, exist_ok=True)
    hexes[["unit", "reg", "pop", "geometry"]].to_file(OUT, driver="GPKG")
    say(abs(gpd.read_file(OUT, ignore_geometry=True)["pop"].sum() - hexes["pop"].sum()) < 1,
        f"population {hexes['pop'].sum():,.0f} = religiondots'")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
