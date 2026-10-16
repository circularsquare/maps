"""T-453 stage 1: shared helpers. Read only on packs."""
import os, sys
os.environ["OMP_NUM_THREADS"] = "2"
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"C:\Users\anita\projects\maps\trainworld\pipeline")
import numpy as np
import h3
from read_pack import load_pack

ZONE_KEYS = ("zone", "zone_gx", "zone_gy")


def pack(city):
    h, a = load_pack(city)
    n = h["cells"]
    ll = np.array([h3.cell_to_latlng(h3.int_to_str(int(c))) for c in a["h3"]])
    f = lambda k: a[k].astype(np.float64)
    out = {"h": h, "n": n, "lat": ll[:, 0], "lon": ll[:, 1], "h3": a["h3"].astype(np.uint64),
           "zone": a["zone"].astype(np.int64), "gx": a["zone_gx"].astype(np.int64), "gy": a["zone_gy"].astype(np.int64),
           "zm": h["gravity"]["zone_m"]}
    for k in ("pop", "jobs", "errand_jobs", "leisure_jobs", "park_draw", "medical_visits"):
        out[k] = f(k) if k in a else np.zeros(n)
    return out


def signals_on(p, city):
    """tier0 signal arrays (OSM place counts) on the pack's cells, by h3 id."""
    z = np.load(rf"C:\Users\anita\projects\maps\trainworld\data\work\{city}.tier0_signals.npz")
    idx = {int(c): i for i, c in enumerate(z["h3"])}
    rows = np.array([idx.get(int(c), -1) for c in p["h3"]])
    out = {}
    for k in ("osm_errand", "osm_leisure", "osm_job", "osm_land"):
        v = np.zeros(p["n"])
        v[rows >= 0] = z[k][rows[rows >= 0]]
        out[k] = v
    print(f"  signals matched {np.mean(rows >= 0):.1%} of pack cells", flush=True)
    return out


def hav_km(lat, lon, la, lo):
    r = np.radians
    a = np.sin(r(lat - la) / 2) ** 2 + np.cos(r(lat)) * np.cos(r(la)) * np.sin(r(lon - lo) / 2) ** 2
    return 12742.0 * np.arcsin(np.sqrt(a))


def circle_shares(p, weights, centres, radius=1.5):
    """Share of each weight within radius km of each centre, and in their union."""
    rows = {}
    union = np.zeros(p["n"], bool)
    for name, (la, lo) in centres.items():
        m = hav_km(p["lat"], p["lon"], la, lo) <= radius
        union |= m
        rows[name] = {k: 100 * w[m].sum() / w.sum() for k, w in weights.items()}
    rows["all"] = {k: 100 * w[union].sum() / w.sum() for k, w in weights.items()}
    return rows


def zone_misplaced(p, w, truth):
    """Share of attraction misplaced between 2 km zones (half the L1 distance of the normalised zone sums)."""
    nz = len(p["gx"])
    a = np.bincount(p["zone"], weights=w, minlength=nz); a /= a.sum()
    b = np.bincount(p["zone"], weights=truth, minlength=nz); b /= b.sum()
    return 50 * np.abs(a - b).sum()


def social(p, leisure):
    pop, park = p["pop"], p["park_draw"]
    if park.sum() > 0:
        return 0.46 * pop / pop.sum() + 0.46 * leisure / leisure.sum() + 0.08 * park / park.sum()
    return 0.5 * pop / pop.sum() + 0.5 * leisure / leisure.sum()


def table(rows, keys):
    print("| | " + " | ".join(keys) + " |")
    for name, r in rows.items():
        print(f"| {name} | " + " | ".join(f"{r[k]:.1f}" for k in keys) + " |")
