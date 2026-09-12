"""United Kingdom — merge the three census geographies into one polygon layer.

The UK has no census and no census geography either. It has three, from three agencies, at
three different vintages, with three different names for the same idea:

    England and Wales  ONS 2021    Output Area   188,880   OA21CD     E00.../W00...
    Scotland           NRS 2022     Output Area    46,363   code       S00...
    Northern Ireland   NISRA 2021   Data Zone       3,780   DZ2021_cd  N20...

`countries.py` reads one `place` path, so this concatenates them. That is safe here for a
reason worth stating rather than assuming: **the three code namespaces do not collide.**
E&W codes begin E00 or W00, Scotland's S00, Northern Ireland's N20, so one `unit` column can
hold all three without a source prefix and a row can always be traced back.

All three files are ALREADY EPSG:4326 — they were reprojected when they were downloaded
(sources/uk_geo.md), so nothing here transforms geometry.

No separate placement layer, as in Czechia and Ireland: the counts are already on these
units. They are the finest on the map — an E&W Output Area is about 130 households, a
Scottish one about 120 people, an NI Data Zone about 500.

Writes:
    data/geo/uk/uk_units.gpkg   one layer, `unit` + `part`

Usage:
    python sources/uk_geo.py
"""

import os
import sys

import geopandas as gpd
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GEO = os.path.join(ROOT, "data", "geo", "uk")
OUT = os.path.join(GEO, "uk_units.gpkg")

# (file, id column, part label, expected feature count, expected code prefixes)
PARTS = [
    ("ew_oa2021_bgc_4326.gpkg", "OA21CD", "ew", 188880, ("E00", "W00")),
    ("sc_oa2022_mhw_4326.gpkg", "code", "sc", 46363, ("S00",)),
    ("ni_dz2021_4326.gpkg", "DZ2021_cd", "ni", 3780, ("N20",)),
]


def main():
    frames = []
    for fname, key, part, expect, prefixes in PARTS:
        path = os.path.join(GEO, fname)
        if not os.path.exists(path):
            raise SystemExit(f"missing {path} — see sources/uk_geo.md for the download")
        g = gpd.read_file(path)
        if key not in g.columns:
            raise SystemExit(f"{fname}: no {key} column, got {list(g.columns)}")
        g = g[[key, "geometry"]].rename(columns={key: "unit"})
        g["unit"] = g["unit"].astype(str).str.strip()
        g["part"] = part

        bad = g[~g["unit"].str.startswith(prefixes)]
        if len(bad):
            raise SystemExit(f"{fname}: {len(bad)} codes outside {prefixes}, "
                             f"e.g. {bad['unit'].head(3).tolist()}")
        note = "" if len(g) == expect else f"  !! expected {expect:,}"
        print(f"  {part}: {len(g):>7,} units  {g.crs}{note}")
        frames.append(g)

    crss = {str(f.crs) for f in frames}
    if len(crss) > 1:
        raise SystemExit(f"the three parts disagree about CRS: {crss}")

    uk = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True),
                          geometry="geometry", crs=frames[0].crs)

    dup = uk["unit"].duplicated().sum()
    if dup:
        raise SystemExit(f"{dup} duplicate unit codes across the three parts — the "
                         "namespaces were supposed to be disjoint")
    empty = uk.geometry.isna() | uk.geometry.is_empty
    if empty.any():
        print(f"  !! {int(empty.sum())} empty geometries")

    print(f"\n{len(uk):,} units total, {uk['unit'].nunique():,} distinct codes")
    uk.to_file(OUT, layer="uk_units", driver="GPKG")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
