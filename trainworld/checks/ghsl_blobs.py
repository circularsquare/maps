"""T-2695: GHS-POP's village blobs at zone origins. For each zone row of these countries, the GHS-POP 2025 1 km people
at its origin (its densest pixel) against the next pixel of the 5x5 around it and the nearest GHSL urban centre. BLOB:
10,000 or more and 3 times the next pixel (Colombia's Tiquisio held 192,229 in one pixel against a census of about
20,000 for the whole municipio, T-2712). Compare a BLOB row's district with its census (docs/adding-a-country.md 2,
"People where GHSL has them wrong"). Reads only.

    python checks/ghsl_blobs.py ISO3[,ISO3...]        (from trainworld/, ..\\venv's Python)
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pyogrio
import rasterio
from pyproj import Transformer

TW = Path(__file__).resolve().parents[1]
TIF = TW / "data/raw/worldplan/GHS_POP_E2025_GLOBE_R2023A_54009_1000_V1_0.tif"
UCDB = TW.parent / "citybrowser/data/GHS_UCDB_THEME_GENERAL_CHARACTERISTICS_GLOBE_R2024A.gpkg"
if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
    sys.exit(__doc__)
lay = "GHS_UCDB_THEME_GENERAL_CHARACTERISTICS_GLOBE_R2024A"
uc = pyogrio.read_dataframe(UCDB, layer=lay, columns=["ID_UC_G0", "GC_UCN_MAI_2025", "GC_POP_TOT_2025"], read_geometry=False)
cen = pyogrio.read_dataframe(UCDB, layer="UC_centroids").to_crs(4326)
cxy = {i: (g.x, g.y) for i, g in zip(cen.ID_UC_G0, cen.geometry)}
ucs = [(cxy[i], n, p) for i, n, p in zip(uc.ID_UC_G0, uc.GC_UCN_MAI_2025, uc.GC_POP_TOT_2025) if i in cxy]
tr = Transformer.from_crs("EPSG:4326", "ESRI:54009", always_xy=True)
plan = json.loads((TW / "pipeline/world_plan.json").read_text(encoding="utf-8"))


def km(a, b):
    la = math.radians((a[1] + b[1]) / 2)
    return math.hypot((a[0] - b[0]) * 111.32 * math.cos(la), (a[1] - b[1]) * 110.57)


with rasterio.open(TIF) as src:
    for iso in sys.argv[1].split(","):
        for r in plan["metros"]:
            if iso not in (r.get("units") or {}) or r.get("kind") != "zone":
                continue
            lon, lat = r["origin"]
            row, col = src.index(*tr.transform(lon, lat))
            w = np.clip(src.read(1, window=((row - 2, row + 3), (col - 2, col + 3))), 0, None)
            top = float(w.max())
            ratio = top / max(float(np.sort(w.ravel())[-2]), 1.0)
            near = min(((km((lon, lat), xy), n, p) for xy, n, p in ucs if abs(xy[0] - lon) < 0.3 and abs(xy[1] - lat) < 0.3),
                       default=(99, "-", 0))
            flag = "BLOB" if top >= 10000 and ratio >= 3 else ""
            print(f"{flag:4} {iso} {r['id']} {r['name'][:28]:28} {r.get('status', 'planned'):8} pixel {top:9,.0f} x{ratio:5.1f} "
                  f"of next; 5x5 {w.sum():9,.0f}; UC {near[1]} {near[2] / 1e3:,.0f}k {near[0]:.0f} km")
