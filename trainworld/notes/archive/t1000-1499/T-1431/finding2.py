"""T-1431: Shanghai - Beijing, rail against the car (and plane) for a few lines, today and with the taper."""
import math, sys
sys.argv = [sys.argv[0], "today"]
exec(open(__file__.replace("finding2.py", "fares.py"), encoding="utf-8").read().split("only = sys.argv")[0])

A, B = rows["sha"], rows["bei"]
km = gc(A, B)
c = car(A, B, km)
beta = 0.1 * math.sqrt(min(60 / c, 1))
pt = plane_parts(A, B, km)
w = fw("CN")
t = taper(km, 50, 0.3)
print(f"km {km:.0f} car {c:.0f} beta {beta:.4f} taper {t:.3f}")
LINES = [  # name, ride minutes, fare km, extra wait (a change)
    ("direct, CR's best 4 h 30", 270, 1.05 * km, 0),
    ("direct, 250 km/h average", 1.05 * km / 250 * 60, 1.05 * km, 0),
    ("direct, 160 km/h average", 1.05 * km / 160 * 60, 1.05 * km, 0),
    ("via Wuhan, a change, 250 km/h", 1800 / 250 * 60, 1800, 15),
]
for name, ride, fkm, xw in LINES:
    for acc in (30, 130):
        for vn, f, base in (("today", 1, 100), ("taper, plane $40", t, 40)):
            rf = f * FW_IN * rail_curve(fkm)
            r = acc + WAIT + xw + ride + rf
            pl = pt + f * w * (base + 0.05 * km)
            sr, sc, sa = split(beta, [r, c, pl])
            print(f"  {name:32s} acc {acc:3d} {vn:17s} rail {r:5.0f} (fare {rf:4.0f}) car {c:5.0f} plane {pl:5.0f} -> rail {100*sr:4.0f} car {100*sc:4.0f} air {100*sa:4.0f}")

print("\nsensitivity to the plane's base, taper on: rail of rail + air (corridor = mean of centre and suburb)")
for za, zb, name, tmin, pub in PAIRS:
    A, B = rows[za], rows[zb]
    km = gc(A, B); c = car(A, B, km); beta = 0.1 * math.sqrt(min(60 / c, 1)); pt = plane_parts(A, B, km); w = fw(A["country"]); f = taper(km, 50, 0.3)
    cells = []
    for base, ff in ((30, f), (40, f), (50, f), (40, 1), (50, 1)):
        rr = []
        for acc in (30, 130):
            r = acc + WAIT + tmin + ff * FW_IN * rail_curve(1.05 * km)
            pl = pt + ff * w * (base + 0.05 * km)
            rr.append(split(beta, [r, c, pl]))
        sr = sum(x[0] for x in rr) / 2; sa = sum(x[2] for x in rr) / 2
        cells.append(f"{100*sr/(sr+sa):4.0f}")
    print(f"  {name:22s} taper $30/$40/$50: {' '.join(cells[:3])} | no taper $40/$50: {' '.join(cells[3:])}  pub {pub[2]}")
