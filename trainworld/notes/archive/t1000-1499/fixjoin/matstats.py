"""Per pack, from its shipped factors (zone_row x curve x zone_col, as notes/t1000lib.py builds it,
streamed in row blocks): kept local commutes, mean km, share over 50 km, share in the home zone.
matstats.py <packs dir> <out.json> ids...   (ids or @file with one id a line). Read-only."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import json, sys
import numpy as np

DT = {"u64": "<u8", "f32": "<f4", "u32": "<u4", "i32": "<i4", "f64": "<f8", "u16": "<u2", "u8": "u1"}
WANT = {"zone_row", "zone_col", "zone_size", "zone_gx", "zone_gy", "zone_region"}


def load(packs, city):
    if packs == "BACKUP":  # the pack as backed up before the fix joins (data/work/pre-fixjoin/<key>/)
        import glob
        hits = glob.glob(rf"C:\Users\anita\projects\maps\trainworld\data\work\pre-fixjoin\*\data\packs\{city}.json")
        hits = [h for h in hits if "shared-" not in h]
        assert len(hits) == 1, hits
        packs = os.path.dirname(hits[0])
    with open(os.path.join(packs, f"{city}.json"), encoding="utf-8") as f:
        h = json.load(f)
    a = {}
    with open(os.path.join(packs, f"{city}.bin"), "rb") as f:
        for x in h["arrays"]:
            if x["name"] in WANT and x["dtype"] in DT:
                f.seek(x["offset"])
                dt = np.dtype(DT[x["dtype"]])
                a[x["name"]] = np.frombuffer(f.read(dt.itemsize * x["count"]), dtype=dt)
    return h, a


def stats(packs, city):
    h, a = load(packs, city)
    g = h["gravity"]
    nz = h["zones"]
    gx, gy = a["zone_gx"].astype(np.int64), a["zone_gy"].astype(np.int64)
    side = a["zone_size"].astype(np.int64) if "zone_size" in a else np.ones(nz, np.int64)
    zr = a["zone_region"].astype(np.int64) if "zone_region" in a else np.zeros(nz, np.int64)
    row, col = a["zone_row"].astype(np.float64), a["zone_col"].astype(np.float64)
    zk = g["zone_m"] / 1000.0
    bf = float(g.get("border_factor", 1.0))
    a_pow = g.get("decay_pow", 0.0) if g["decay"] == "pow_exp" else 0.0
    flat = np.float32(g.get("decay_flat_km", 0.0))
    subs = [(0, 0)] if side.max() == 1 else [(u, v) for u in (0, 1) for v in (0, 1)]
    wsub = {q: np.where(side == 2, 0.25, 1.0 if q == (0, 0) else 0.0) for q in subs}
    tot = td = t50 = home = 0.0
    B = 256
    for s in range(0, nz, B):
        e = min(nz, s + B)
        dx = gx[s:e, None] - gx[None, :]
        dy = gy[s:e, None] - gy[None, :]
        f = 0.0
        for qi in subs:
            for qj in subs:
                wq = wsub[qi][s:e, None] * wsub[qj][None, :]
                if not wq.any():
                    continue
                ex, ey = dx + qi[0] - qj[0], dy + qi[1] - qj[1]
                dq = zk * np.sqrt(ex * ex + ey * ey)
                dq[dq == 0] = g["intrazonal_km"]
                dq32 = np.maximum(dq.astype(np.float32), flat)
                f = f + wq * (np.exp(-dq32 / np.float32(g["decay_km"])) * dq32 ** np.float32(-a_pow)).astype(np.float64)
        if len(subs) == 1:
            d = dq
        else:
            hx, hy = 2 * dx + side[s:e, None] - side[None, :], 2 * dy + side[s:e, None] - side[None, :]
            d = zk * 0.5 * np.sqrt(hx * hx + hy * hy)
            d = np.where(d == 0, g["intrazonal_km"] * side[s:e, None], d)
        T = row[s:e, None] * f * np.where(zr[s:e, None] != zr[None, :], bf, 1.0) * col[None, :]
        tot += T.sum()
        td += (T * d).sum()
        t50 += T[d > 50].sum()
        home += T[np.arange(e - s), np.arange(s, e)].sum()
    return {"trips": tot, "mean": td / tot, "over50": t50 / tot, "home": home / tot}


if __name__ == "__main__":
    packs, out = sys.argv[1], sys.argv[2]
    ids = sys.argv[3:]
    if len(ids) == 1 and ids[0].startswith("@"):
        ids = open(ids[0][1:]).read().split()
    res = {}
    if os.path.exists(out):
        res = json.load(open(out))
    for k, i in enumerate(ids):
        if i in res:
            continue
        try:
            res[i] = stats(packs, i)
        except Exception as e:
            res[i] = {"error": f"{type(e).__name__}: {e}"}
        if k % 50 == 0:
            print(k, i, res[i], flush=True)
            json.dump(res, open(out, "w"), indent=0)
    json.dump(res, open(out, "w"), indent=0)
    print(len(res), "->", out)
