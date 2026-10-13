"""T-162: T-159's tables before and after the region-aware gravity, from the shipped packs (each
with its own decay, border factor and zones). Read-only.

    ..\\venv\\Scripts\\python.exe trainworld/notes/t162_tables.py [BEFORE_DIR] [AFTER_DIR]

BEFORE_DIR defaults to data/work/pre-T162 (the packs as they were), AFTER_DIR to data/packs.
Philadelphia: T-159's groups (11 counties, Center City, Camden city) against LODES OD 2023 (the
PA and NJ files, as T-159). New York: county pairs against LODES OD 2023 (T-034's cache).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")

import sys

import h3
import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
sys.path.insert(0, HERE)
import county_flows  # noqa: E402
import read_pack  # noqa: E402
import realnet  # noqa: E402
import t159_flows as tf  # noqa: E402

BEFORE = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "data", "work", "pre-T162")
AFTER = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "data", "packs")


def zone_trips(header, a):
    """(start, end, T) chunks of the shipped zone gravity, with the region factor."""
    g = header["gravity"]
    gx, gy = a["zone_gx"].astype(np.int64), a["zone_gy"].astype(np.int64)
    nz = header["zones"]
    zr = a["zone_region"].astype(np.int64) if "zone_region" in a else np.zeros(nz, np.int64)
    bf = g.get("border_factor", 1.0)
    row, col = a["zone_row"].astype(np.float64), a["zone_col"].astype(np.float64)
    zk = g["zone_m"] / 1000.0
    for s in range(0, nz, 400):
        e = min(nz, s + 400)
        d = zk * np.sqrt((gx[s:e, None] - gx[None, :]) ** 2 + (gy[s:e, None] - gy[None, :]) ** 2)
        d[d == 0] = g["intrazonal_km"]
        t = row[s:e, None] * col[None, :] * np.exp(-d / g["decay_km"]) * d ** -g["decay_pow"]
        t *= np.where(zr[s:e, None] != zr[None, :], bf, 1.0)
        yield s, e, t


def phl_model(packs, b):
    header, a = read_pack.load_pack("phl", packs)
    ll = np.array([h3.cell_to_latlng(h3.int_to_str(int(i))) for i in a["h3"]])
    k = np.cos(np.radians(39.95))
    tree = cKDTree(np.c_[b["lat"].to_numpy(), b["lon"].to_numpy() * k])
    _, nb = tree.query(np.c_[ll[:, 0], ll[:, 1] * k])
    nbk = b.iloc[nb]
    cls = tf.block_class(nbk["county"].to_numpy().astype(int), tf.in_cc(ll[:, 0], ll[:, 1]), nbk["cam"].to_numpy())
    nz = header["zones"]
    zone = a["zone"].astype(np.int64)
    zh = np.zeros((nz, tf.NC * 4)); zw = np.zeros((nz, tf.NC * 4))
    np.add.at(zh, (zone, cls), a["commute_home"].astype(np.float64))
    np.add.at(zw, (zone, cls), a["commute_work"].astype(np.float64))
    zh /= np.maximum(zh.sum(1, keepdims=True), 1e-12)
    zw /= np.maximum(zw.sum(1, keepdims=True), 1e-12)
    M = np.zeros((tf.NC * 4, tf.NC * 4))
    for s, e, t in zone_trips(header, a):
        M += zh[s:e].T @ t @ zw
    return M


def phl():
    print("Philadelphia, LODES OD 2023 (PA and NJ files):")
    L, rac, b, _ = tf.lodes()
    C2G = tf.class_to_groups()
    LG = C2G.T @ L @ C2G
    homeL = LG[:, :tf.NC].sum(1)
    MB, MA = (C2G.T @ phl_model(p, b) @ C2G for p in (BEFORE, AFTER))
    hb, ha = MB[:, :tf.NC].sum(1), MA[:, :tf.NC].sum(1)
    P, CM, BU, GL, DE, MO = (tf.CIDX[x] for x in ["42101", "34007", "34005", "34015", "42045", "42091"])
    CC, CAM = tf.GCC, tf.GCAM
    rows = [(CC, CM, "Center City -> Camden County"), (CC, CAM, "Center City -> Camden city"),
            (CC, BU, "Center City -> Burlington"), (CC, GL, "Center City -> Gloucester"),
            (P, CM, "Philadelphia -> Camden County"), (P, CAM, "Philadelphia -> Camden city"),
            (P, BU, "Philadelphia -> Burlington"), (P, GL, "Philadelphia -> Gloucester"),
            (P, DE, "Philadelphia -> Delaware County"), (P, MO, "Philadelphia -> Montgomery"),
            (P, P, "Philadelphia own county"), (CM, P, "Camden County -> Philadelphia"),
            (CM, CC, "Camden County -> Center City"), (CAM, CC, "Camden city -> Center City"),
            (CAM, P, "Camden city -> Philadelphia"), (CM, CM, "Camden County own county"),
            (BU, P, "Burlington -> Philadelphia"), (GL, P, "Gloucester -> Philadelphia")]
    print("\n| home -> work | before: trips/day (share, x LODES) | after | LODES jobs (share) |")
    print("|---|---|---|---|")
    for h, w, lab in rows:
        lv = LG[h, w] / homeL[h]
        cell = lambda M, hm: f"{M[h, w]:,.0f} ({M[h, w] / hm[h]:.1%}, {M[h, w] / hm[h] / lv:.2f}x)"
        print(f"| {lab} | {cell(MB, hb)} | {cell(MA, ha)} | {LG[h, w]:,.0f} ({lv:.1%}) |")
    pa = [tf.CIDX[x] for x in tf.CF if x.startswith("42")]
    nj = [tf.CIDX[x] for x in tf.CF if x.startswith("34")]
    cross = lambda X: (X[np.ix_(pa, nj)].sum() + X[np.ix_(nj, pa)].sum()) / X[:tf.NC, :tf.NC].sum()
    own = lambda X: np.trace(X[:tf.NC, :tf.NC]) / X[:tf.NC, :tf.NC].sum()
    print(f"| PA <-> NJ share of trips | {cross(MB):.2%} | {cross(MA):.2%} | {cross(LG):.2%} |")
    print(f"| own-county share, all | {own(MB):.1%} | {own(MA):.1%} | {own(LG):.1%} |")
    print("\nown county by county (before / after / LODES; PA and NJ only, as T-159 has no DE or MD OD):")
    for i, f in enumerate(tf.CF):
        if f[:2] not in ("42", "34"):
            continue
        print(f"  {tf.COUNTIES[f]}: {MB[i, i] / hb[i]:.1%} / {MA[i, i] / ha[i]:.1%} / {LG[i, i] / homeL[i]:.1%}")


def nyc():
    print("\nNew York, LODES OD 2023 county flows:")
    L = county_flows.lodes_flows()
    MB, MA = county_flows.model_flows(BEFORE), county_flows.model_flows(AFTER)
    I = {realnet.COUNTIES[f]: i for i, f in enumerate(realnet.COUNTY_ORDER)}
    ny = [i for i, f in enumerate(realnet.COUNTY_ORDER) if f.startswith("36")]
    nj = [i for i, f in enumerate(realnet.COUNTY_ORDER) if f.startswith("34")]
    pairs = [("Hudson", "New York"), ("Bergen", "New York"), ("Essex", "New York"), ("Passaic", "New York"),
             ("Union", "New York"), ("Rockland", "Bergen"), ("Richmond", "Union"), ("Westchester", "Bergen"),
             ("Kings", "New York"), ("Queens", "New York"), ("Westchester", "New York"), ("Nassau", "New York"),
             ("Fairfield CT", "Westchester"), ("Fairfield CT", "New York")]
    print("\n| home -> work | before: trips/day (share, x LODES) | after | LODES share |")
    print("|---|---|---|---|")
    for h, w in pairs:
        i, j = I[h], I[w]
        lv = L[i, j] / L[i].sum()
        cell = lambda M: f"{M[i, j]:,.0f} ({M[i, j] / M[i].sum():.1%}, {M[i, j] / M[i].sum() / lv:.2f}x)"
        print(f"| {h} -> {w} | {cell(MB)} | {cell(MA)} | {lv:.1%} |")
    cross = lambda M: (M[np.ix_(ny, nj)].sum() + M[np.ix_(nj, ny)].sum()) / M.sum()
    print(f"| NY <-> NJ share | {cross(MB):.2%} | {cross(MA):.2%} | {cross(L):.2%} |")
    print(f"| own county | {np.trace(MB) / MB.sum():.1%} | {np.trace(MA) / MA.sum():.1%} | {np.trace(L) / L.sum():.1%} |")
    man = I["New York"]
    print(f"| to Manhattan, all | {MB[:, man].sum() / MB.sum():.1%} | {MA[:, man].sum() / MA.sum():.1%} | {L[:, man].sum() / L.sum():.1%} |")


if __name__ == "__main__":
    phl()
    nyc()
