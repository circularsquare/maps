"""United States placement layer: 2024 cartographic-boundary tracts -> data/geo/us/us_tracts.gpkg.

    python sources/us_geo.py

The ACS 2020-2024 tables are on 2020 tracts with Connecticut's 2022 planning-region codes, which
is what cb_2024_us_tract_500k carries (religiondots/data/geo/, read in place). One polygon per
tract, `unit` = GEOID, except the 14 Suffolk County tracts the C16001 tract rows leave out
(sources/us_acs.py MISSING_TRACTS): those take the unit of their PUMA's remainder.

No `pop` column: a tract is its own unit, so there is nothing to weight between, and the two
Suffolk remainder units share their dots equally between their tracts (tracts are drawn to hold
similar populations). Inside a tract a dot lands uniformly, as religiondots' US layer does; the
cartographic-boundary files are already cut back to the shoreline.

Checks: every unit in data/normalized/us.csv has a polygon, and every polygon in the 50 states
and DC either has rows or is a tract the tables count as empty (printed).
"""
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "sources"))
from us_acs import MISSING_TRACTS, RD_GEO, NORM  # noqa: E402

OUT = HERE / "data" / "geo" / "us" / "us_tracts.gpkg"
NOT_ACS = {"60", "66", "69", "72", "78"}   # territories; Puerto Rico is its own entry


def main():
    g = gpd.read_file(f"zip://{RD_GEO / 'cb_2024_us_tract_500k.zip'}")
    g = g[~g["STATEFP"].isin(NOT_ACS)].copy()
    if g["GEOID"].duplicated().any():
        raise SystemExit("duplicate tract GEOIDs in cb_2024")
    g["unit"] = g["GEOID"].map(lambda t: f"36103-rest-{MISSING_TRACTS[t]}" if t in MISSING_TRACTS else t)
    d = pd.read_csv(NORM / "us.csv", dtype={"geo_id": str}, usecols=["geo_id", "count"])
    per = d.groupby("geo_id")["count"].sum()
    want, have = set(per.index), set(g["unit"])
    if want - have:
        lost = per[sorted(want - have)]
        raise SystemExit(f"{len(lost)} units with people and no polygon ({lost.sum():,}): {list(lost.index[:6])}")
    empty = sorted(have - want)
    print(f"{len(g):,} tracts in 50 states + DC -> {len(have):,} units; all {len(want):,} units with "
          f"people have polygons; {len(empty)} polygons have no people in the tables "
          f"(e.g. {empty[:4]})")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    g[["unit", "GEOID", "geometry"]].to_crs(4326).to_file(OUT, layer="tracts", driver="GPKG")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
