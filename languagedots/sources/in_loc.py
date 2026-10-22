"""India's placement layer with Pakistan's side of the border cut away -> data/geo/in/in_places.gpkg.

    python sources/in_loc.py

religiondots' India layer (read-only) is the 5,988 sub-districts' settlement polygons. Along the
Line of Control, and in places along the Punjab and Rajasthan border, some of those polygons run
into ground Pakistan administers, so India's dots near Poonch and Rajouri fell on the Pakistan
side while Pakistan's own layer (Azad Kashmir, Gilgit-Baltistan, the provinces) was drawn there
too. This cuts every India place polygon with Pakistan's outline (COD-AB pak_admin0, the same
boundary source Pakistan's districts, and so pk's hexes, come from), and scales the polygon's
`pop` weight by the share of its area that is left, so a cut place takes fewer dots. The census
counts per sub-district are unchanged; only where inside the unit the dots go. Nothing is written
to religiondots.

Checks printed: polygons touched, weight removed, no unit left without a polygon, and the
reverse (Pakistan's hex centroids that lie inside NE India's polygon, away from the line).
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import geopandas as gpd  # noqa: E402
import pandas as pd  # noqa: E402
from shapely.geometry import MultiPolygon  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from rdlink import RD, RD_GEO  # noqa: E402

SRC = RD_GEO / "in" / "in_places.gpkg"
COD0 = RD / "data" / "raw" / "pk2023" / "cod" / "pak_admin0.shp"
OUT = HERE / "data" / "geo" / "in" / "in_places.gpkg"
UTM = 32643


def main():
    pak = gpd.read_file(COD0).to_crs(4326)
    pak = pak.geometry.union_all().buffer(0)
    ind = gpd.read_file(SRC).to_crs(4326)
    n0, units0 = len(ind), ind["unit"].nunique()
    hit = ind.index[ind.intersects(pak)]
    print(f"{n0:,} India place polygons in {units0:,} units; {len(hit)} touch Pakistan's outline")
    sub = ind.loc[hit]
    pu = gpd.GeoSeries([pak], crs=4326).to_crs(UTM).iloc[0]
    su = sub.to_crs(UTM)
    frac_left = 1 - su.geometry.intersection(pu).area / su.geometry.area
    cut = sub.geometry.difference(pak)
    w0 = ind["pop"].sum()
    ind.loc[hit, "pop"] = sub["pop"] * frac_left
    ind.loc[hit, "geometry"] = cut
    drop = ind.geometry.is_empty | ind.geometry.isna() | (ind["pop"] <= 0) & ind.index.isin(hit)
    # a polygon wholly inside Pakistan's outline goes; a cut one keeps its polygonal parts
    ind = ind[~drop].copy()

    def poly_only(g):
        if g.geom_type == "Polygon":
            return MultiPolygon([g])
        if g.geom_type == "MultiPolygon":
            return g
        parts = [p for p in getattr(g, "geoms", []) if p.geom_type in ("Polygon", "MultiPolygon")]
        out = []
        for p in parts:
            out += list(p.geoms) if p.geom_type == "MultiPolygon" else [p]
        return MultiPolygon(out)
    ind["geometry"] = ind.geometry.map(poly_only)
    ind = ind[~ind.geometry.is_empty]
    left = set(gpd.read_file(SRC, columns=["unit"], ignore_geometry=True)["unit"]) - set(ind["unit"])
    if left:
        raise SystemExit(f"units left without a polygon: {sorted(left)[:10]}")
    print(f"  {len(ind):,} polygons kept ({n0 - len(ind)} dropped whole); weight "
          f"{w0:,.0f} -> {ind['pop'].sum():,.0f} (-{w0 - ind['pop'].sum():,.0f})")
    by_state = (sub.assign(state=sub["unit"].str[:2], gone=sub["pop"] * (1 - frac_left))
                .groupby("state")["gone"].sum().round())
    print("  weight removed by state code (01 J&K, 03 Punjab, 08 Rajasthan):", by_state.to_dict())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".tmp.gpkg")
    ind.to_file(tmp, layer="places", driver="GPKG")
    os.replace(tmp, OUT)
    print(f"  wrote {OUT}")

    # Reverse check: Pakistan's hex centroids that lie inside the Indian polygons left
    hx = gpd.read_file(HERE / "data" / "geo" / "pk" / "pk_hexes.gpkg", bbox=(68, 23.5, 77.5, 37.5))
    hx = hx.to_crs(4326)
    cen = hx.to_crs(UTM).geometry.centroid.to_crs(4326)
    pts = gpd.GeoDataFrame(hx[["unit", "pop"]], geometry=cen, crs=4326)
    j = gpd.sjoin(pts, ind[["unit", "geometry"]].rename(columns={"unit": "inunit"}), predicate="within")
    print(f"  reverse: Pakistan hex centroids inside a cut India polygon: {len(j)} "
          f"({j['pop'].sum():,.0f} people)")


if __name__ == "__main__":
    main()
