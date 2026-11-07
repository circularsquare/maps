"""T-073: model times along recorded small-river trips, from the RiverATLAS reaches the canoe uses
(data/raw/rivers_work, as built by T-031). Each trip: snap both ends to the nearest Q >= 1 reach,
follow NEXT_DOWN from the upper end to the lower, and time it down and up under the model's rule
and candidate rules. Lakes (HydroLAKES pixels under the reach) at still water both ways.

    python reachtrips.py
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

WORK = r"C:\Users\anita\projects\maps\isochrone\data\raw\rivers_work"
P = 5.0
NORTH, WEST, L1W, L1H = 85.0, -180.0, 43200, 17400

r = dict(np.load(os.path.join(WORK, "reaches.npz")))
dz = np.load(os.path.join(WORK, "desert.npz"))
r["q"] = np.where(dz["in_hot"], dz["q_in"], r["q"])
ok = dz["ok"]
grad_ok = np.load(os.path.join(WORK, "gradient.npy"))
grad = np.full(len(ok), np.nan)
grad[ok] = grad_ok
co, off = r["co"], r["off"]
ids = r["id"]
idpos = {int(i): k for k, i in enumerate(ids)}
down = np.array([idpos.get(int(d), -1) for d in r["next_down"]], np.int64)
vreach = np.repeat(np.arange(len(off) - 1), np.diff(off))
lakes = np.load(os.path.join(WORK, "lakes_l1.npy"), mmap_mode="r")


def snap(lat, lon, box=0.15):
    m = (np.abs(co[:, 1] - lat) < box) & (np.abs(co[:, 0] - lon) < box)
    idx = np.nonzero(m)[0]
    idx = idx[ok[vreach[idx]]]
    d = (co[idx, 1] - lat) ** 2 + ((co[idx, 0] - lon) * np.cos(np.radians(lat))) ** 2
    j = idx[np.argmin(d)]
    return vreach[j], j


def lake_frac(k, a, b):
    pts = co[a:b]
    c = np.floor((pts[:, 0] - WEST) * 120).astype(np.int64) % L1W
    rr = np.clip(np.floor((NORTH - pts[:, 1]) * 120).astype(np.int64), 0, L1H - 1)
    return float((np.asarray(lakes[rr, c]) > 0).mean()) if len(pts) else 0.0


def path(a, b):
    """Pieces (reach, km, lake share) from point a (upstream) to point b."""
    ka, ja = snap(*a)
    kb, jb = snap(*b)
    out, k = [], ka
    for _ in range(100000):
        s, e = off[k], off[k + 1]
        lo = ja if k == ka else s
        hi = (jb + 1) if k == kb else e
        frac = max(hi - lo - 1, 0) / max(e - s - 1, 1)
        out.append((k, r["length_km"][k] * frac, lake_frac(k, lo, hi)))
        if k == kb:
            return out
        k = down[k]
        if k < 0:
            raise SystemExit("end not downstream of start")
    raise SystemExit("too long")


def v_kmh(q):
    return 0.51 * q ** 0.11 * 3.6


def depth(q):
    return 0.27 * q ** 0.39


RULES = {
    "now": lambda q, g: max(P - 0.4 * v_kmh(q), 2.5),
    "full": lambda q, g: max(P - v_kmh(q), 2.5),
    "bank1.5": lambda q, g: max(P - min(1, max(0.4, (1.5 / depth(q)) ** (2 / 3))) * v_kmh(q), 2.5),
    "rapids2.5": lambda q, g: 2.5 if g >= 1 else max(P - 0.4 * v_kmh(q), 2.5),
    "D<1.5:2.5": lambda q, g: 2.5 if depth(q) < 1.5 else max(P - 0.4 * v_kmh(q), 2.5),
}


def trip(name, a, b):
    pcs = path(a, b)
    for lab, pt in (("from", a), ("to", b)):
        k, j = snap(*pt)
        print(f"  {lab} {pt} -> reach {ids[k]} at {co[j, 1]:.3f},{co[j, 0]:.3f}, Q {r['q'][k]:.1f}")
    km = sum(x[1] for x in pcs)
    lake_km = sum(x[1] * x[2] for x in pcs)
    riv = [(k, l * (1 - f)) for k, l, f in pcs]
    qs = np.array([r["q"][k] for k, _ in riv])
    qm = np.array([r["qmin"][k] for k, _ in riv])
    ls = np.array([l for _, l in riv])
    gs = np.array([grad[k] for k, _ in riv])
    gs = np.where(np.isfinite(gs), gs, 0)
    closed = ((gs > 5) & (qs < 1000)) | ((qs >= 1000) & (gs > np.clip(2.5 * np.sqrt(1000 / np.maximum(qs, 1000)), 0.3, 2)))
    rap = (gs >= 1) & ~closed
    w = ls / max(ls.sum(), 1e-9)
    print(f"\n{name}: {km:.0f} km by river ({lake_km:.0f} on lakes); Q {qs.min():.1f}-{qs.max():.1f} "
          f"(length-weighted mean {np.dot(w, qs):.1f}), driest month {qm.min():.2f}-{qm.max():.1f}; "
          f"D {depth(qs.min()):.2f}-{depth(qs.max()):.2f} m; flat {ls[~rap & ~closed].sum():.0f} km, "
          f"rapids {ls[rap].sum():.0f} km, closed {ls[closed].sum():.0f} km; v {v_kmh(qs.min()):.1f}-{v_kmh(qs.max()):.1f} km/h")
    tdown = lake_km / P + sum(l / (P + v_kmh(q)) for l, q in zip(ls, qs))
    print(f"  down {tdown:.1f} h ({km / tdown:.1f} km/h)")
    for nm, f in RULES.items():
        tup = lake_km / P + sum(l / f(q, g) for l, q, g in zip(ls, qs, gs))
        print(f"  up {nm:10s} {tup:6.1f} h ({km / tup:.2f} km/h), up/down {tup / tdown:.2f}")


TRIPS = [
    # Lewis and Clark: Camp Fortunate (below Clark Canyon dam) to the Three Forks
    ("Beaverhead+Jefferson", (45.006, -112.852), (45.925, -111.515)),
    ("Beaverhead (Clark Canyon - Twin Bridges)", (45.006, -112.852), (45.545, -112.330)),
    ("Jefferson (Twin Bridges - Three Forks)", (45.555, -112.325), (45.925, -111.515)),
    # T-071's record: Three Forks to Great Falls (up in Jul 1805)
    ("Missouri Three Forks - Great Falls", (45.935, -111.500), (47.480, -111.320)),
    # Schoolcraft 1832: Itasca outlet down to Cass Lake
    ("Mississippi Itasca - Cass Lake", (47.2405, -95.2075), (47.380, -94.620)),
    # Parsnip (Mackenzie 1793), upper river
    ("Parsnip (upper)", (54.65, -122.50), (55.20, -123.00)),
    # Bad River / James Creek (Mackenzie 1793)
    ("James Creek (Bad River)", (54.58, -122.55), (54.40, -122.75)),
    # Allagash Stream: Allagash Lake to Chamberlain Lake (Smith 2006, 6 mi in 7 h up)
    ("Allagash Stream", (46.305, -69.545), (46.300, -69.390)),
    # Umbazookskus (Thoreau 1857)
    ("Umbazookskus", (46.190, -69.400), (46.120, -69.440)),
]

# stretches with a published length (km): RiverATLAS length against it
LENGTHS = [
    ("Mississippi Itasca - Knutson Dam (DNR RM 1341.6-1258.2)", (47.2405, -95.2075), (47.449, -94.482), 134.2),
    ("Beaverhead+Jefferson (69 + 83 mi)", (45.006, -112.852), (45.925, -111.515), 244.6),
    ("Jefferson from Big Hole mouth (83 mi)", (45.560, -112.335), (45.925, -111.515), 133.6),
    ("Illinois (RM 273-0)", (41.385, -88.262), (38.968, -90.440), 439.4),
    ("Ohio (RM 0-981)", (40.448, -80.030), (36.990, -89.130), 1578.7),
    ("Missouri Three Forks - mouth (2341 mi)", (45.935, -111.500), (38.815, -90.120), 3767.4),
]


def length_check(name, a, b, km_pub):
    pcs = path(a, b)
    km = sum(x[1] for x in pcs)
    qs = [r["q"][k] for k, _, _ in pcs]
    for lab, pt in (("from", a), ("to", b)):
        k, j = snap(*pt)
        print(f"  {lab} {pt} -> reach {ids[k]} at {co[j, 1]:.3f},{co[j, 0]:.3f}, Q {r['q'][k]:.1f}")
    print(f"{name}: RiverATLAS {km:.0f} km, published {km_pub:.0f} km, ratio {km_pub / km:.2f}; Q {min(qs):.0f}-{max(qs):.0f}")


# whole rivers (Wikipedia infobox length, mouth, mean discharge): RiverATLAS DIST_UP_KM at the mouth
WHOLE = [
    ("Big Hole", (45.5678, -112.3392), 246, 35),
    ("Gallatin", (45.9389, -111.4925), 193, 30),
    ("Crow Wing", (46.2711, -94.3397), 182, 44.6),
    ("Upper Iowa", (43.4661, -91.2339), 251, 19.7),
    ("Kishwaukee", (42.1861, -89.1311), 102, 30.5),
    ("Ruby", (45.5203, -112.3403), 122, 5.6),
]


def whole_check(name, mouth, km_pub, q_pub):
    lat, lon = mouth
    m = (np.abs(co[:, 1] - lat) < 0.06) & (np.abs(co[:, 0] - lon) < 0.06)
    ks = np.unique(vreach[np.nonzero(m)[0]])
    ks = [k for k in ks if 0.4 * q_pub <= r["q"][k] <= 2.5 * q_pub]
    if not ks:
        print(f"{name}: no reach near the mouth with Q near {q_pub}")
        return
    k = max(ks, key=lambda k: r["dist_up"][k])
    du = r["dist_up"][k]
    print(f"{name}: RiverATLAS reach {ids[k]} Q {r['q'][k]:.1f} (published {q_pub}), DIST_UP {du:.0f} km, "
          f"published {km_pub} km, ratio {km_pub / du:.2f}")


if __name__ == "__main__":
    sel = sys.argv[1:]
    if sel and sel[0] == "share":
        # share of open river length in the rapids class, by Q band, world and the Amazon lowland box
        q, L, g = r["q"], r["length_km"], np.nan_to_num(grad, nan=0.0)
        closed = np.where(q >= 1000, g > np.clip(2.5 * np.sqrt(1000 / np.maximum(q, 1000)), 0.3, 2), g > 5)
        rap = (g >= 1) & ~closed & ok
        opn = ~closed & ok
        st = co[off[:-1]]
        amz = (st[:, 1] > -10) & (st[:, 1] < 2) & (st[:, 0] > -70) & (st[:, 0] < -50)
        for lo, hi in ((1, 10), (10, 100), (100, 1000), (1000, 1e7)):
            b = (q >= lo) & (q < hi)
            print(f"Q {lo}-{hi:g}: open {L[b & opn].sum() / 1e3:.0f} k km, rapids {100 * L[b & rap].sum() / L[b & opn].sum():.0f}%; "
                  f"Amazon box {100 * L[b & rap & amz].sum() / max(L[b & opn & amz].sum(), 1):.0f}% of {L[b & opn & amz].sum() / 1e3:.0f} k km")
        sys.exit(0)
    if sel and sel[0] == "near":
        # near lat lon box qmin: reaches in a box, biggest first
        lat, lon, box, qlo = map(float, sel[1:5])
        m = (np.abs(co[:, 1] - lat) < box) & (np.abs(co[:, 0] - lon) < box)
        ks = np.unique(vreach[np.nonzero(m)[0]])
        ks = sorted([k for k in ks if r["q"][k] >= qlo and ok[k]], key=lambda k: -r["q"][k])
        for k in ks[:40]:
            c = co[off[k]]
            print(f"reach {ids[k]} Q {r['q'][k]:7.1f} qmin {r['qmin'][k]:6.1f} grad {grad[k]:5.2f} m/km len {r['length_km'][k]:5.1f} start {c[1]:.3f},{c[0]:.3f}")
        sys.exit(0)
    if sel and sel[0] == "whole":
        for t in WHOLE:
            whole_check(*t)
        sys.exit(0)
    if sel and sel[0] == "length":
        for t in LENGTHS:
            try:
                length_check(*t)
            except SystemExit as e:
                print(f"{t[0]}: {e}")
        sys.exit(0)
    for t in TRIPS:
        if sel and not any(s.lower() in t[0].lower() for s in sel):
            continue
        try:
            trip(*t)
        except SystemExit as e:
            print(f"\n{t[0]}: {e}")
