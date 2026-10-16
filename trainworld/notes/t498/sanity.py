"""Print car and rail minutes between named places (nearest zone by commuter centroid)."""
import sys
import numpy as np
C = np.load(sys.argv[1]); R = np.load(sys.argv[2])
places = [p.split("=") for p in sys.argv[3].split(";")]
idx = []
for name, ll in places:
    lon, lat = map(float, ll.split(","))
    idx.append((name, int(np.argmin((C["lon"] - lon) ** 2 + ((C["lat"] - lat) * 1.2) ** 2))))
o = idx[0]
for name, j in idx[1:]:
    km = np.hypot(C["cx"][o[1]] - C["cx"][j], C["cy"][o[1]] - C["cy"][j]) / 1000
    print(f"{o[0]} -> {name}: {km:.1f} km, car {C['car'][o[1], j]:.0f} min, rail {R['rail'][o[1], j]:.0f} min")
