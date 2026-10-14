"""T-125: reference values for the world frame, from PROJ.

Prints the Rust constant `LCC_PROJ` (sim/src/track/frame.rs) and writes app/src/game/frame.fixture.json
(read by checks/t125-frame-check.mjs to check the TypeScript copy of the frame).
Run: ..\\venv\\Scripts\\python.exe frame_fixture.py
"""
import json
from pathlib import Path
from pyproj import CRS, Transformer

LON0, LAT0 = -73.985, 40.758
R = 6371008.8
sphere = CRS.from_proj4(f"+proj=longlat +R={R} +no_defs")
lcc = CRS.from_proj4(f"+proj=lcc +lat_1={LAT0} +lat_0={LAT0} +lon_0={LON0} +R={R} +x_0=0 +y_0=0 +units=m +no_defs")
fwd = Transformer.from_crs(sphere, lcc, always_xy=True)

POINTS = [
    (-73.985, 40.758), (-75.1636, 39.9526), (-74.7429, 40.2171), (-71.0589, 42.3601), (-77.0369, 38.9072),
    (-79.9959, 40.4406), (-87.6298, 41.8781), (-73.5673, 45.5017), (-80.8431, 35.2271), (-65.0, 44.0),
]
rows = []
for lon, lat in POINTS:
    x, y = fwd.transform(lon, lat)
    rows.append([lon, lat, round(x, 3), round(y, 3)])
for lon, lat, x, y in rows:
    print(f"        ({lon}, {lat}, {x}, {y}),")
out = Path(__file__).resolve().parents[1] / "app/src/game/frame.fixture.json"
out.write_text(json.dumps({"lon0": LON0, "lat0": LAT0, "points": rows}, indent=1), encoding="utf-8")
print("wrote", out)
