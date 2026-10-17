"""T-636: T-509 stage 7's test cross on every registered pack (two lines through its busiest job
square), before (the constants shipped until 2026-10-17: a calib binary built then, BEFORE_BIN) and
after (the new model), commute rail by distance ring.

    python t636_s7.py build | run [ids] | report
"""
import json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np

TW = r"C:\Users\anita\projects\maps\trainworld"
sys.path.insert(0, os.path.join(TW, "pipeline"))
import merge_cells as mc  # noqa: E402
from read_pack import load_pack, PACKS  # noqa: E402

OUT = r"C:\Users\anita\projects\maps\artifacts\trainworld\t636s7"
BIN = os.path.join(TW, "data", "work", "target", "release", "calib.exe")
BEFORE_BIN = os.environ.get("BEFORE_BIN", "")
RINGS = [3, 8, 15, 30]  # km
R_KM, SPACING, KMH, DWELL = 15.0, 1.5, 40.0, 30.0
os.makedirs(OUT, exist_ok=True)


def build(cid):
    h, a = load_pack(cid)
    x, y, jobs = a["x_m"].astype(float), a["y_m"].astype(float), a["jobs"].astype(float)
    # the 2 km square with the most jobs, smoothed over its 3x3 neighbours
    gx, gy = np.floor(x / 2000).astype(int), np.floor(y / 2000).astype(int)
    keys = {}
    for i, j, w in zip(gx, gy, jobs):
        keys[(i, j)] = keys.get((i, j), 0) + w
    best = max(keys, key=lambda k: sum(keys.get((k[0] + di, k[1] + dj), 0) for di in (-1, 0, 1) for dj in (-1, 0, 1)))
    sel = (gx == best[0]) & (gy == best[1])
    cx, cy = np.average(x[sel], weights=jobs[sel] + 1e-9), np.average(y[sel], weights=jobs[sel] + 1e-9)
    n = int(round(2 * R_KM / SPACING)) + 1
    off = np.linspace(-R_KM * 1000, R_KM * 1000, n)
    mid = n // 2
    st = [(cx + o, cy) for o in off] + [(cx, cy + o) for k, o in enumerate(off) if k != mid]
    ew = list(range(n))
    ns = [n + k if k < mid else (mid if k == mid else n + k - 1) for k in range(n)]
    hop = SPACING * 3600 / KMH + DWELL
    times = []
    for _ in range(2):
        for run in range(2):
            for lv in range(3):
                times += [0.0 if (run == 0 and k == n - 1) or (run == 1 and k == 0) else hop for k in range(n)]
    net = {"what": "T-509 stage 7 test cross", "city": cid, "pack": cid, "stations": [], "centre": [cx, cy],
           "lines": [{"operator": "ew"}, {"operator": "ns"}],
           "demand": {"st_xy": [v for p in st for v in p], "line_n": [n, n], "stops": ew + ns,
                      "times": times, "tph": [12, 8, 4, 12, 8, 4], "cars": [8, 8]}}
    json.dump(net, open(os.path.join(OUT, f"{cid}_net.json"), "w"))
    d = np.hypot(x - cx, y - cy) / 1000
    ring = np.full(len(x), 255, np.uint8)
    for k in range(len(RINGS) - 1, -1, -1):
        ring[d < RINGS[k]] = k
    ring.tofile(os.path.join(OUT, f"{cid}_rings.u8"))


def run1(job):
    cid, tag = job
    out = os.path.join(OUT, f"{cid}_{tag}.json")
    if os.path.exists(out):
        return job
    cmd = [BEFORE_BIN if tag == "before" else BIN, "--net", os.path.join(OUT, f"{cid}_net.json"), "--out", out, "--pack", os.path.join(PACKS, f"{cid}.json"),
           "--areas", os.path.join(OUT, f"{cid}_rings.u8"), "--rounds", "1"]
    with open(os.path.join(OUT, f"{cid}_{tag}.log"), "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
    return job


def summary(cid, tag):
    r = json.load(open(os.path.join(OUT, f"{cid}_{tag}.json")))
    c, e = r["day"]["commute"], r["day"]["everyday"]
    ar, at = np.array(r["areas"]["rail"]), np.array(r["areas"]["trips"])
    return {"c": c[1] / max(c[0], 1), "e": e[1] / max(e[0], 1), "rail": c[1] + e[1],
            "ring": list(ar / np.maximum(at, 1)), "board": sum(r["lines"]["board"])}


def report():
    rows = []
    for cid in mc.registered():
        try:
            b, a = summary(cid, "before"), summary(cid, "after")
        except FileNotFoundError:
            continue
        rows.append((cid, b, a))
    print(f"{'pack':5s} {'commute rail':>16s} {'everyday rail':>16s} {'rail trips x':>12s}  ring <3 km   3-8   8-15  15-30 (commute rail, after)")
    flags = []
    for cid, b, a in rows:
        x = a["rail"] / max(b["rail"], 1)
        f = "  <<" if x > 4 / 3 or x < 3 / 4 else ""
        if f:
            flags.append(cid)
        print(f"{cid:5s} {b['c']:6.1%} -> {a['c']:6.1%}  {b['e']:6.1%} -> {a['e']:6.1%}  {x:10.2f}    "
              + "  ".join(f"{v:5.1%}" for v in a["ring"]) + f)
    print("outside 3/4..4/3:", flags)
    json.dump({cid: {"before": b, "after": a} for cid, b, a in rows}, open(os.path.join(OUT, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    cmd = sys.argv[1]
    ids = sys.argv[2:] or mc.registered()
    if cmd == "build":
        for cid in ids:
            build(cid)
            print("built", cid, flush=True)
    elif cmd == "run":
        jobs = [(cid, t) for cid in ids for t in ("before", "after")]
        with ThreadPoolExecutor(3) as ex:
            for j in ex.map(run1, jobs):
                print("done", j, flush=True)
    else:
        report()
