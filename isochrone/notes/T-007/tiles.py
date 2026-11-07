"""The 5 degree tile ids (row * 72 + col) under checks/levels.mjs's boxes round its 25 starts
(+-3.3 degrees, the box levels.mjs solves L1 and L3 in), one line, for proto.py."""
import math

STARTS = [(49.59, 34.55), (42.03, -93.62), (-34.59, -60.95), (30.90, 75.85), (46.32, 7.99), (-13.53, -71.97),
          (28.21, 83.99), (39.63, -106.04), (62.24, 25.75), (46.49, -80.99), (47.47, -94.88), (22.79, 5.53),
          (-23.70, 133.88), (43.57, 104.42), (0.52, 25.19), (-4.08, -63.14), (58.00, 102.66), (46.77, 23.60),
          (40.758, -73.985), (51.507, -0.128), (6.52, 3.38), (28.61, 77.21), (22.82, 89.55), (52.09, 5.12),
          (61.23, 7.10)]
ids = set()
for lat, lng in STARTS:
    h = 3.3 + 0.3
    hl = h / math.cos(math.radians(lat))
    for r in range(int((85 - min(85, lat + h)) // 5), int((85 - max(-60, lat - h)) // 5) + 1):
        for c in range(int((lng - hl + 180) // 5), int((lng + hl + 180) // 5) + 1):
            ids.add(r * 72 + c % 72)
print(" ".join(str(i) for i in sorted(ids)))
