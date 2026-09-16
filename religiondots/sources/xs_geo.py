"""Israeli settlements in the West Bank: where the dots go inside each CBS unit.

Writes:
    data/geo/xs/xs_units.gpkg    the 267 CBS 2022 units beyond the Green Line, with CBS's
                                 placeholder polygons replaced by population-sized discs
    data/geo/xs/xs_places.gpkg   `place`: each unit cut by the Kontur hexes it overlaps, with
                                 `unit` and `pop` (Kontur population in the overlap)

Usage:
    python sources/xs_geo.py [--fetch]    nothing is downloaded; needs the Israel build
                                          (il.csv, dropped_units.json, statareas2022.geojson)
                                          and Palestine's (COD-AB, both Kontur extracts)

## THE SAME GEOMETRY PALESTINE'S PLACEMENT TOOK THE SETTLEMENTS OUT OF

`sources/ps_geo.py` takes each of these units' Jews and Others off the Kontur hexes the unit
overlaps, in proportion to hex population times the overlapping share, so that Palestinian dots do
not land in the settlements. This file builds the other half from the same pieces: the units come
from `ps_geo.settlement_units()` (CBS's own polygons, with the 119 placeholder localities of
0.008 km2 replaced by discs at 4,000 people per km2), the hexes are read the way `ps_geo.main()`
reads them (Kontur PS plus IL, de-duplicated on `h3`, kept where the centroid is in a COD-AB
governorate), and each unit is cut into its overlaps with those hexes, weighted by the Kontur
population in the overlap. So a unit's dots fall where the weight was taken off Palestine.

The witness is that this reproduces `ps_lookup.csv`'s `settlements_removed` per governorate from
these pieces. If it does not, the two files have drifted and this stops.

## A UNIT WITH NO POPULATED HEX UNDER IT

Kontur models no one under some discs and small units. Those units are placed uniformly on their
own shape (one piece, `pop` 1), which is §8.2's equal share, rather than dropped.

## WHY NOT ISRAEL'S PLACEMENT

Israel's entry places on the unit polygon with no grid, because §8.2e measured the Kontur grid as
coarser than its statistical areas (`sources/il.md` §8). These are the same kind of units, so
inside a small one the weighting does little: its overlap is one or two hex pieces. It matters for
the large ones, a locality's whole jurisdiction drawn as one polygon, where Kontur puts the people
in the built-up part. And Palestine's entry already took these people's weight off exactly these
pieces, so the two entries agree on where they live.
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ.setdefault("OMP_NUM_THREADS", "6")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GEO = os.path.join(ROOT, "data", "geo", "xs")
PS_LOOKUP = os.path.join(ROOT, "data", "geo", "ps", "ps_lookup.csv")
OUT_UNITS = os.path.join(GEO, "xs_units.gpkg")
OUT_PLACES = os.path.join(GEO, "xs_places.gpkg")
sys.path.insert(0, HERE)

UNITS = 267
REMOVED_SLACK = 1.0     # ps_lookup.csv rounds to the person


def read_hexes(ps_geo):
    """Kontur PS + IL inside the COD-AB governorates, as ps_geo.main() builds `hx`, in UTM."""
    import geopandas as gpd
    import numpy as np
    import pandas as pd

    gp_ps, gp_il = ps_geo._unpack("PS"), ps_geo._unpack("IL")
    cod = ps_geo.read_cod()
    units = cod.copy()
    units["unit"] = units["adm2_pcode"].map({g[1]: g[0] for g in ps_geo.GOVERNORATES})
    if units["unit"].isna().any():
        raise SystemExit("COD-AB pcodes no longer match ps_geo.GOVERNORATES")
    units = units[["unit", "geometry"]].to_crs(4326)

    ps = gpd.read_file(gp_ps)
    il = gpd.read_file(gp_il)
    if len(ps) == 0 or len(il) == 0:
        raise SystemExit("a Kontur extract read as ZERO features")
    both = pd.concat([ps, il], ignore_index=True)
    hx = gpd.GeoDataFrame(both[~both.duplicated("h3")].reset_index(drop=True), crs=ps.crs)
    cent = gpd.GeoDataFrame({"h3": hx["h3"]}, geometry=hx.geometry.centroid,
                            crs=hx.crs).to_crs(4326)
    j = gpd.sjoin(cent, units, how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")].reindex(cent.index)
    hx["gov"] = j["unit"].to_numpy()
    hx = hx[hx["gov"].notna()].to_crs(ps_geo.UTM).reset_index(drop=True)
    hx["hid"] = np.arange(len(hx))
    hx["hex_area"] = hx.geometry.area
    print(f"Kontur PS + IL: {len(hx):,} hexes inside the governorates, "
          f"{hx['population'].sum():,.0f} people")
    return hx


def main():
    import geopandas as gpd
    import numpy as np
    import pandas as pd

    import ps_geo

    for p in (PS_LOOKUP, ps_geo.IL_CSV, ps_geo.IL_DROPPED):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p} -- run sources/il_geo.py and sources/ps_geo.py first")
    os.makedirs(GEO, exist_ok=True)

    du = ps_geo.settlement_units()                         # UTM; placeholders are discs
    if len(du) != UNITS:
        raise SystemExit(f"{len(du)} CBS units beyond the Green Line, expected {UNITS}")
    du["placeholder"] = du["km2"] < ps_geo.PLACEHOLDER_KM2
    du["unit"] = du["unit"].astype(str)

    hx = read_hexes(ps_geo)

    # ---- the pieces, with ps_geo.main()'s own formula
    ov = gpd.overlay(hx[["hid", "gov", "hex_area", "population", "geometry"]],
                     du[["unit", "nonarab", "geometry"]], how="intersection",
                     keep_geom_type=True)
    ov["w"] = ov["population"] * ov.geometry.area / ov["hex_area"]
    wsum = ov.groupby("unit")["w"].transform("sum")
    ov["take"] = np.where(wsum > 0, ov["nonarab"] * ov["w"] / wsum, 0.0)

    # ---- witness: the hex-level removal, per governorate, is ps_lookup.csv's
    take = ov.groupby("hid")["take"].sum()
    hx["take"] = hx["hid"].map(take).fillna(0.0)
    hx["removed"] = hx["population"] - (hx["population"] - hx["take"]).clip(lower=0.0)
    got = hx.groupby("gov")["removed"].sum()
    look = pd.read_csv(PS_LOOKUP).set_index("unit")["settlements_removed"]
    diff = (got.reindex(look.index).fillna(0.0) - look).abs()
    print(f"\n  witness: settlements removed per governorate, recomputed against ps_lookup.csv "
          f"(largest difference {diff.max():.1f}; total {got.sum():,.0f} against {look.sum():,})")
    if (diff > REMOVED_SLACK).any():
        raise SystemExit(f"these pieces do not reproduce Palestine's removal: "
                         f"{diff[diff > REMOVED_SLACK].round(0).to_dict()} -- rerun sources/ps_geo.py "
                         "or find what moved")

    # ---- the placement layer
    pieces = ov[ov["w"] > 0][["unit", "w", "geometry"]].rename(columns={"w": "pop"})
    have = set(pieces["unit"])
    bare = du[~du["unit"].isin(have)]
    fallback = gpd.GeoDataFrame({"unit": bare["unit"], "pop": 1.0}, geometry=bare.geometry,
                                crs=du.crs)
    places = gpd.GeoDataFrame(pd.concat([pieces, fallback], ignore_index=True), crs=du.crs)
    places = places[~places.geometry.is_empty].reset_index(drop=True)
    if set(places["unit"]) != set(du["unit"]):
        raise SystemExit(f"units with no placement piece: "
                         f"{sorted(set(du['unit']) - set(places['unit']))[:10]}")
    area_in = places.geometry.area.groupby(places["unit"]).sum()
    area_unit = du.set_index("unit").geometry.area
    over = (area_in / area_unit.reindex(area_in.index)).max()
    if over > 1.001:
        raise SystemExit(f"a unit's pieces cover {over:.3f} of its own area; the overlay doubled")

    n_ph = int(du["placeholder"].sum())
    print(f"\n  {len(du)} units: {n_ph} placeholders drawn as discs "
          f"({du.loc[du['placeholder'], 'nonarab'].sum():,.0f} Jews and Others), "
          f"{len(du) - n_ph} on CBS's polygons")
    print(f"  {len(pieces):,} weighted pieces on {len(have)} units; {len(bare)} units with no "
          f"populated hex placed uniformly on their shape ({bare['nonarab'].sum():,.0f} people)")
    cov = pieces.geometry.area.groupby(pieces["unit"]).sum() / area_unit.reindex(sorted(have))
    print(f"  share of each unit's shape under a populated hex: median {cov.median():.2f}, "
          f"p10 {cov.quantile(0.1):.2f}")

    du.to_crs(4326)[["unit", "name", "total", "nonarab", "km2", "placeholder", "geometry"]].to_file(
        OUT_UNITS, layer="units", driver="GPKG")
    places.to_crs(4326).to_file(OUT_PLACES, layer="places", driver="GPKG")
    print(f"\nwrote {OUT_UNITS} ({len(du)} units)")
    print(f"wrote {OUT_PLACES} ({len(places):,} pieces)")


if __name__ == "__main__":
    main()
