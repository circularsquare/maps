"""Elevation of each Philippine barangay, for countries/ph.py's upland rule (sources/ph.md §7).

Reads religiondots' ph_barangays.gpkg (read-only) and GEBCO 2026's 15-arc-second grid (~460 m)
from ../data/ascentshed/gebco_global/ (read-only), samples it at each barangay's representative
point, and writes data/geo/ph/ph_barangay_elev.csv (bgy, elev_m; sea values clipped to 0).
GEBCO's land heights are SRTM15+; good enough to tell a lowland town from a Cordillera one.

    python sources/ph_elev.py
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import geopandas as gpd  # noqa: E402
import rasterio  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rdlink import RD_GEO  # noqa: E402

GEBCO = ROOT.parent / "data" / "ascentshed" / "gebco_global" / "gebco_2026.vrt"
OUT = ROOT / "data" / "geo" / "ph" / "ph_barangay_elev.csv"


def main():
    g = gpd.read_file(RD_GEO / "ph" / "ph_barangays.gpkg")
    pt = g.geometry.representative_point()
    with rasterio.open(GEBCO) as r:
        z = np.array([v[0] for v in r.sample(zip(pt.x, pt.y))], dtype=float)
    z = np.clip(z, 0, None)
    if len(z) != 42042 or np.isnan(z).any():
        raise SystemExit(f"ph_elev: {len(z)} barangays, {np.isnan(z).sum()} missing")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"bgy": g["bgy"].astype(str), "elev_m": z.round(0).astype(int)}).to_csv(OUT, index=False)
    q = np.percentile(z, [10, 50, 90, 99])
    print(f"wrote {OUT}: 42,042 barangays, elevation p10/50/90/99 = {q.round(0)} m; "
          f"{(z >= 300).mean():.1%} at 300 m or more, {(z >= 700).mean():.1%} at 700 m or more")
    # spot checks: Baguio's barangays sit near 1,500 m, Manila's near sea level
    u = g["unit"].astype(str)
    for name, code in [("Baguio", "1430300000"), ("Manila", "1380600000")]:
        m = (u == code).to_numpy()
        if m.any():
            print(f"  {name}: median {np.median(z[m]):.0f} m over {m.sum()} barangays")


if __name__ == "__main__":
    main()
