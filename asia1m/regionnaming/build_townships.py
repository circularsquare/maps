"""Rasterize China townships (xiangzhen 省/市/县/乡) -> adm_twn.npy for city-core detail."""
import sys, json, time, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np, geopandas as gpd
from rasterio.features import rasterize
from geo import TRANSFORM, PROJ4, W, H

SCR = HERE
XZ = os.path.join(HERE, "..", "..", "data", "asia1m", "china", "xiangzhen.shp")

def main():
    t0 = time.time()
    g = gpd.read_file(XZ)
    print(f"[{time.time()-t0:.0f}s] read {len(g)} townships")
    g = g.set_crs("EPSG:4326", allow_override=True).to_crs(PROJ4).reset_index(drop=True)
    names = [None]; shapes = []
    for _, row in g.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty: continue
        names.append(dict(prov=row['省'], city=row['市'], county=row['县'], town=row['乡']))
        shapes.append((geom, len(names)-1))
    arr = rasterize(shapes, out_shape=(H, W), transform=TRANSFORM, fill=0, dtype=np.int32)
    np.save(f"{SCR}/adm_twn.npy", arr)
    json.dump(names, open(f"{SCR}/adm_twn_names.json", "w", encoding="utf-8"), ensure_ascii=False)
    print(f"[{time.time()-t0:.0f}s] townships: {len(shapes)} units, painted {int((arr>0).sum())} px")

if __name__ == "__main__":
    main()
