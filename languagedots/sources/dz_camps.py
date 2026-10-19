"""Placement layer for Algeria with the Sahrawi refugee camps near Tindouf added:
data/geo/dz/dz_hexes_camps.gpkg. Placement only; the people are countries/dz.py's.

    python sources/dz_camps.py

The 2008 census did not count the camps, and religiondots' Kontur hexes for Algeria hold only
the Laayoune camp (about 1,200 people a hex on its eastern side; the others are empty desert).
So this layer is religiondots' dz_hexes.gpkg (read-only) plus a patch of small hexagons around
each camp's centre, and two columns:
  camp       1.0 on the new hexagons (where the camps' people are drawn), else 0
  camp_zone  1.0 on any hexagon, new or Kontur's, within ZONE_KM of a camp centre; the census's
             own people (Tindouf town and the rest) are given no weight there
The file's name does not end in `_hexes.gpkg`, so the Kontur cap check (which expects raw Kontur)
leaves it alone; the Kontur hexes in it are unchanged and Algeria has no cap rows.

Camp centres are GeoNames' CMPRF (refugee camp) points for Awserd, Boujdour, Laayoune and Smara
(data/raw/dz geonames_DZ.zip in religiondots) and Wikipedia's coordinates for Dakhla camp, which
GeoNames lacks (26.7464 N, 7.2631 W; 170 km south-east of Tindouf). Rabouni, the administrative
centre, is left out: it is a few thousand staff, not a camp of its own.
"""
import math
import sys
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from rdlink import RD, RD_GEO  # noqa: E402

OUT = HERE / "data" / "geo" / "dz" / "dz_hexes_camps.gpkg"
GEONAMES = RD / "data" / "raw" / "dz" / "geonames_DZ.zip"
UNIT = "DZ37"                       # Tindouf wilaya
RADIUS_KM = 2.5                     # the patch each camp's people are drawn on
SIDE_KM = 0.55                      # hexagon side: 0.79 km2, about Kontur's 0.74
ZONE_KM = 3.5                       # the census people are kept off this far from a centre
DAKHLA = ("Dakhla", 26.74639, -7.26306)


def centres():
    import pandas as pd
    with zipfile.ZipFile(GEONAMES) as z, z.open("DZ.txt") as fh:
        g = pd.read_csv(fh, sep="\t", header=None, quoting=3, low_memory=False,
                        usecols=[1, 4, 5, 7], names=["name", "lat", "lon", "fcode"])
    g = g[(g["fcode"] == "CMPRF") & g["lat"].between(26, 28.5) & g["lon"].between(-9, -7)]
    out = {r["name"]: (r["lat"], r["lon"]) for _, r in g.iterrows()}
    want = {"Awserd", "Boujdour", "Laayoune", "Smara"}
    if set(out) != want:
        raise SystemExit(f"dz_camps: GeoNames camps {sorted(out)}, expected {sorted(want)}")
    out[DAKHLA[0]] = DAKHLA[1:]
    return out


def main():
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import Polygon

    hx = gpd.read_file(RD_GEO / "dz" / "dz_hexes.gpkg")
    cen = centres()
    cells, names = [], []
    for name, (lat, lon) in cen.items():
        zone = 32629                               # UTM 29N covers 12 W to 6 W
        c = gpd.GeoSeries(gpd.points_from_xy([lon], [lat]), crs=4326).to_crs(zone).iloc[0]
        s = SIDE_KM * 1000
        dx, dy = s * math.sqrt(3), s * 1.5
        n = 0
        for j in range(-8, 9):
            for i in range(-8, 9):
                x = c.x + i * dx + (dx / 2 if j % 2 else 0)
                y = c.y + j * dy
                if math.hypot(x - c.x, y - c.y) > RADIUS_KM * 1000:
                    continue
                ring = [(x + s * math.cos(math.radians(60 * k + 30)),
                         y + s * math.sin(math.radians(60 * k + 30))) for k in range(6)]
                cells.append(Polygon(ring))
                names.append(name)
                n += 1
        print(f"  {name}: {n} hexagons around {lat:.4f}, {lon:.4f}")
    camps = gpd.GeoDataFrame({"unit": UNIT, "pop": 0.0, "camp": 1.0, "camp_name": names},
                             geometry=cells, crs=zone).to_crs(4326)
    hx = hx.assign(camp=0.0, camp_name="")
    allx = pd.concat([hx, camps], ignore_index=True)
    allx = gpd.GeoDataFrame(allx, geometry="geometry", crs=4326)

    # the zone: within ZONE_KM of any camp centre, on a metric projection
    pts = gpd.GeoSeries(gpd.points_from_xy([v[1] for v in cen.values()],
                                           [v[0] for v in cen.values()]), crs=4326).to_crs(32629)
    cc = allx.geometry.to_crs(32629).centroid
    xy = np.c_[cc.x.to_numpy(), cc.y.to_numpy()]
    px = np.c_[pts.x.to_numpy(), pts.y.to_numpy()]
    dmin = np.sqrt(((xy[:, None, :] - px[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
    allx["camp_zone"] = (dmin <= ZONE_KM * 1000).astype(float)
    kontur_in = allx[(allx["camp"] == 0) & (allx["camp_zone"] == 1)]
    print(f"  {int((allx['camp'] == 1).sum())} camp hexagons; {len(kontur_in)} Kontur hexes in the "
          f"zones, {kontur_in['pop'].sum():,.0f} Kontur people (all but Laayoune's are 0-60)")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    allx.to_file(OUT, driver="GPKG")
    print(f"wrote {OUT} ({len(allx):,} polygons)")


if __name__ == "__main__":
    main()
