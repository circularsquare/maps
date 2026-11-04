"""T-2424: how each zone pair's road (the market's model) fits a land-aware path across a lake.

For place pair (A, B): each zone pair's straight km, the model's road with and without the zone
detours (1.25 x straight x rk, times zone i's km detour towards B and zone j's towards A, from a
market-drives.bin with T-2424's zone section) and the land-only path on the 1/60 degree lattice scaled so the centres' land path
equals the routed road. Zone pairs weighted by people x people. As notes/T-2418.md's zone_smear.py.

    python zone_check.py BIN A B [A B ...]
Also prints, for each pair, the table's ratios and ferry.
"""
import json
import math
import struct
import sys

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

TW = r"C:\Users\anita\projects\maps\trainworld"
PER = 60
R = 6371008.8
KX, KY = 111.32, 110.57


def read(path):
    b = open(path, "rb").read()
    assert b[:4] == b"TWD1", b[:4]
    n = struct.unpack_from("<I", b, 4)[0]
    o = 8
    ids = []
    for _ in range(n):
        ln = b[o]
        ids.append(b[o + 1:o + 1 + ln].decode())
        o += 1 + ln
    cnt = struct.unpack_from(f"<{n}H", b, o)
    o += 2 * n
    pairs, order = {}, []
    for a in range(n):
        for _ in range(cnt[a]):
            bb, rt, rk = struct.unpack_from("<Hbb", b, o)
            o += 4
            pairs[(a, bb)] = [math.exp(rt / 64), math.exp(rk / 64), 0, 0]
            order.append((a, bb))
    f = struct.unpack_from("<I", b, o)[0]
    o += 4
    for _ in range(f):
        pi, km, fe = struct.unpack_from("<IHB", b, o)
        o += 7
        pairs[order[pi]][2:] = [km, fe]
    zones = {}
    if o < len(b):  # T-2424's zone section
        g = struct.unpack_from("<I", b, o)[0]
        o += 4
        for _ in range(g):
            a, ot, k = struct.unpack_from("<HHH", b, o)
            o += 6
            if k & 0x8000:
                for z in range(k & 0x7FFF):
                    qt, qk = struct.unpack_from("<bb", b, o)
                    o += 2
                    if qt or qk:
                        zones[(a, z, ot)] = (math.exp(qt / 64), math.exp(qk / 64))
                continue
            for _ in range(k):
                z, qt, qk = struct.unpack_from("<Hbb", b, o)
                o += 4
                zones[(a, z, ot)] = (math.exp(qt / 64), math.exp(qk / 64))
    return ids, pairs, zones


def zone_ll(pid):
    j = json.load(open(TW + rf"\data\packs\{pid}.market.json", encoding="utf-8"))
    lon0, lat0 = j["frame"]["lon"], j["frame"]["lat"]
    k = math.cos(math.radians(lat0))
    s = np.array([x[:5] for x in j["sites"]], float)
    zi = s[:, 4].astype(int)
    nz = len(j["zones"])
    w = s[:, 2] + s[:, 3] + 1e-9
    x = np.bincount(zi, w * s[:, 0], nz) / np.bincount(zi, w, nz)
    y = np.bincount(zi, w * s[:, 1], nz) / np.bincount(zi, w, nz)
    return lon0 + np.degrees(x / (R * k)), lat0 + np.degrees(y / R), np.bincount(zi, s[:, 2], nz)


def one(rows, ids, pairs, zones, a, b):
    idx = {s: i for i, s in enumerate(ids)}
    ia, ib = idx[a], idx[b]
    ent = pairs.get((min(ia, ib), max(ia, ib)))
    if ent is None:
        print(f"{a}-{b}: no entry (the rule)")
        return
    rt, rk, fkm, nf = ent
    la_lon, la_lat, wa = zone_ll(a)
    lb_lon, lb_lat, wb = zone_ll(b)
    allon = np.r_[la_lon, lb_lon]
    allat = np.r_[la_lat, lb_lat]
    pad = 2.0
    r0, r1 = int((allat.min() - pad + 90) * PER), int((allat.max() + pad + 90) * PER)
    c0, c1 = int((allon.min() - pad + 180) * PER), int((allon.max() + pad + 180) * PER)
    water = np.array(np.load(TW + r"\data\work\market_drives\water_burnt.npy", mmap_mode="r")[r0:r1, c0:c1])
    H, W = water.shape
    lat_mid = math.radians((r0 + r1) / 2 / PER - 90)
    dx, dy = KX * math.cos(lat_mid) / PER, KY / PER
    node = np.arange(H * W).reshape(H, W)
    land = ~water
    I, J, V = [], [], []
    for (ox, oy) in ((1, 0), (0, 1), (1, 1), (1, -1)):
        d = math.hypot(ox * dx, oy * dy)
        ys = slice(max(0, -oy), H - max(0, oy))
        yt = slice(max(0, oy), H - max(0, -oy))
        ok = land[ys, 0:W - ox] & land[yt, ox:W]
        I.append(node[ys, 0:W - ox][ok])
        J.append(node[yt, ox:W][ok])
        V.append(np.full(ok.sum(), d))
    G = coo_matrix((np.concatenate(V), (np.concatenate(I), np.concatenate(J))), shape=(H * W, H * W)).tocsr()
    from scipy.ndimage import distance_transform_edt
    _, near = distance_transform_edt(water, return_indices=True)

    def cell(lon, lat):
        r = np.clip(((np.asarray(lat) + 90) * PER).astype(int) - r0, 0, H - 1)
        c = np.clip(((np.asarray(lon) + 180) * PER).astype(int) - c0, 0, W - 1)
        return near[0][r, c] * W + near[1][r, c]
    na, nb = cell(la_lon, la_lat), cell(lb_lon, lb_lat)
    ca = cell([rows[a]["lon"]], [rows[a]["lat"]])[0]
    cb = cell([rows[b]["lon"]], [rows[b]["lat"]])[0]
    D = dijkstra(G, directed=False, indices=np.r_[na, ca])
    k = math.cos(lat_mid)

    def straight(lo1, la1, lo2, la2):
        return np.hypot((lo2 - lo1) * KX * k, (la2 - la1) * KY)
    sc = straight(rows[a]["lon"], rows[a]["lat"], rows[b]["lon"], rows[b]["lat"])
    land_c = D[-1, cb]
    c = 1.25 * sc * rk / land_c
    fa = np.array([zones.get((ia, i, ib), (1, 1))[1] for i in range(len(na))])
    fb = np.array([zones.get((ib, j, ia), (1, 1))[1] for j in range(len(nb))])
    s = straight(la_lon[:, None], la_lat[:, None], lb_lon[None, :], lb_lat[None, :])
    lp = D[:len(na)][:, nb] * c
    wt = wa[:, None] * wb[None, :]
    ok = np.isfinite(lp) & (s >= 1) & (wt > 0)
    old = (1.25 * s * rk / lp)[ok]
    new = (1.25 * s * rk * fa[:, None] * fb[None, :] / lp)[ok]
    w = wt[ok]

    def qs(x):
        o = np.argsort(x)
        cum = np.cumsum(w[o]) / w.sum()
        return " ".join(f"{x[o][np.searchsorted(cum, p)]:.2f}" for p in (0.1, 0.5, 0.9))
    za = sum(1 for (p, _, q) in zones if p == ia and q == ib)
    zb = sum(1 for (p, _, q) in zones if p == ib and q == ia)
    print(f"{a}-{b}: {sc:.0f} km, rt {rt:.3f} rk {rk:.3f} ferry {fkm} km x{nf}; land path {land_c / sc:.2f}x, road {1.25 * rk:.2f}x; "
          f"zone detours {za}/{len(na)} and {zb}/{len(nb)}; model / land-aware (10%, median, 90%): centres only {qs(old)}, with zones {qs(new)}")


def main(path, *names):
    rows = json.load(open(TW + r"\app\src\game\market-rows.json", encoding="utf-8"))
    ids, pairs, zones = read(path)
    print(f"{len(ids)} places, {len(pairs)} pairs, {len(zones)} zone detours")
    for i in range(0, len(names) - 1, 2):
        one(rows, ids, pairs, zones, names[i], names[i + 1])


if __name__ == "__main__":
    main(*sys.argv[1:])
