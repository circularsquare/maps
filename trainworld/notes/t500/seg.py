"""T-500 prototype on tyo: commute gravity split by industry segment, against census 2020 table 4
(all industries) and table 10 (by industry, to the 92 big workplaces).
    python seg.py <prep.npz> <scheme> <curves>
scheme: one4 | two | ind20 | none ; curves: pack | a/L;a/L;... (one per segment) | fit
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "6")
import sys, json, itertools
import numpy as np
import pandas as pd

RAW = r"C:\Users\anita\projects\maps\trainworld\data\raw\t500"
D = np.load(sys.argv[1]); SCHEME = sys.argv[2]; CURVES = sys.argv[3]
SCHEMES = {
    "none": {k: "all" for k in "ABCDEFGHIJKLMNOPQRS"},
    "one4": {**{k: "office" for k in "GJKLR"}, **{k: "retail" for k in "IMN"},
             **{k: "industry" for k in "ABCDEFH"}, **{k: "public" for k in "OPQS"}},
    "one4s": {**{k: "office" for k in "GJKLRS"}, **{k: "retail" for k in "IMN"},
              **{k: "industry" for k in "ABCDEFH"}, **{k: "public" for k in "OPQ"}},
    "two": {**{k: "office" for k in "GJKLRS"}, **{k: "other" for k in "ABCDEFHIMNOPQ"}},
    "lodes3": {**{k: "goods" for k in "ABCDE"}, **{k: "trade" for k in "FHI"}, **{k: "services" for k in "GJKLMNOPQRS"}},
    "ind20": {k: k for k in "ABCDEFGHIJKLMNOPQRS"},
}
SEG = SCHEMES[SCHEME]
segs = sorted(set(SEG.values()))
P, Q, Cl = D["P"], D["Q"], D["Cl"]
gx, gy = D["gx"], D["gy"]
codes = list(D["codes"]); ci = {c: i for i, c in enumerate(codes)}; nm = len(codes)
g = json.loads(str(D["g"]))
d = 2.0 * np.sqrt((gx[:, None] - gx[None, :]) ** 2 + (gy[:, None] - gy[None, :]) ** 2)
d[d == 0] = g["intrazonal_km"]
ld = np.log(d)

# shares by municipality: commuters (at home out) at the home end and the work end, per segment
t8 = pd.read_csv(os.path.join(RAW, "t8.csv"), dtype={"code": str})
t8 = t8[t8.ind.isin(SEG) & t8.code.isin(ci)]
t8 = t8.assign(seg=t8.ind.map(SEG), rc=(t8.res - t8.home).clip(lower=0), wc=(t8.work - t8.home).clip(lower=0))
R = np.zeros((nm, len(segs))); W = np.zeros((nm, len(segs)))
si = {s: k for k, s in enumerate(segs)}
np.add.at(R, (t8.code.map(ci).to_numpy(), t8.seg.map(si).to_numpy()), t8.rc.to_numpy())
np.add.at(W, (t8.code.map(ci).to_numpy(), t8.seg.map(si).to_numpy()), t8.wc.to_numpy())
R /= np.maximum(R.sum(1, keepdims=True), 1e-9); W /= np.maximum(W.sum(1, keepdims=True), 1e-9)
Hs = P @ R; Js = Q @ W            # zone x segment
Jtot = Q.sum()
Hs *= Js.sum(0) / Hs.sum(0)       # each segment's homes scaled to its jobs, as the pack does in all
print(f"scheme {SCHEME}: segment shares of commuting jobs", {s: round(float(Js[:, k].sum() / Jtot), 3) for s, k in si.items()}, flush=True)

# census by segment to the listed workplaces (table 10)
t10 = pd.read_csv(os.path.join(RAW, "t10.csv"), dtype={"w": str, "h": str})
t10 = t10[t10.ind.isin(SEG) & t10.w.isin(ci) & t10.h.isin(ci)]
DW = sorted({ci[c] for c in t10.w.unique()})
C10 = np.zeros((len(segs), nm, nm))
np.add.at(C10, (t10.ind.map(SEG).map(si).to_numpy(), t10.h.map(ci).to_numpy(), t10.w.map(ci).to_numpy()), t10.n.to_numpy())

Pn = P / np.maximum(P.sum(1, keepdims=True), 1e-12); Qn = Q / np.maximum(Q.sum(1, keepdims=True), 1e-12)
ORIG = ["13203", "13204", "13115", "13120", "13208", "13206", "13202", "13201"]
C5 = [ci[c] for c in ("13101", "13102", "13103", "13104", "13113")]; C3 = C5[:3]
W8 = [ci[c] for c in ORIG]; MU = ci["13203"]
_w = np.bincount(D["li"], weights=D["ch"] + 1e-6, minlength=nm)
_r = np.hypot(np.bincount(D["li"], weights=D["x"] * (D["ch"] + 1e-6), minlength=nm),
              np.bincount(D["li"], weights=D["y"] * (D["ch"] + 1e-6), minlength=nm)) / np.maximum(_w, 1e-9) / 1000
RINGS = [(_r < 5), (_r >= 5) & (_r < 10), (_r >= 10) & (_r < 20), (_r >= 20) & (_r < 30), (_r >= 30) & (_r < 60)]


def solve(F, H, J, iters=300, tol=1e-4):
    o = H
    col = np.ones(len(J))
    for it in range(iters):
        row = np.where(o > 0, o / np.maximum(F @ col, 1e-300), 0)
        col = np.where(J > 0, J / np.maximum(row @ F, 1e-300), 0)
        if it > 5 and it % 5 == 0:
            e = np.max(np.abs((F @ col)[o > 0] * row[o > 0] / o[o > 0] - 1))
            if e < tol:
                break
    return F * row[:, None] * col[None, :]


D0 = float(os.environ.get("D0", "0"))   # softened near end: (d + D0)^-a


def decay(a, L):
    if D0:
        return np.exp(-d / L - a * np.log(d + D0))
    return np.exp(-d / L - a * ld)


def miss(a, b):
    return 0.5 * float(np.abs(a / a.sum() - b / b.sum()).sum())


def sh(M, rows, cols):
    return M[rows][:, cols].sum() / M[rows].sum()


def miss10(M, k):
    """misplaced among the listed workplaces: model and census segment flows to them, by home municipality"""
    return miss(M[:, DW], C10[k][:, DW])


def report(tag, M, Ms):
    row = dict(misplaced=miss(M, Cl), own=np.trace(M) / M.sum(), west3=sh(M, W8, C3), west5=sh(M, W8, C5),
               musa5=sh(M, [MU], C5), all3=M[:, C3].sum() / M.sum(), rings=[sh(M, np.nonzero(s)[0], C3) for s in RINGS],
               m10=[miss10(Ms[k], k) for k in range(len(segs))])
    print(f"{tag}: misplaced {row['misplaced']:.1%} own {row['own']:.1%} west->3 {row['west3']:.1%} west->5 {row['west5']:.1%} "
          f"Musashino->5 {row['musa5']:.1%} all->3 {row['all3']:.1%} rings->3 " + "/".join(f"{x:.1%}" for x in row["rings"])
          + " | by segment, misplaced to listed workplaces " + " ".join(f"{s} {row['m10'][k]:.1%}" for s, k in si.items()), flush=True)
    return row


def run(curves, tag):
    Ms = []
    for k, (a, L) in enumerate(curves):
        T = solve(decay(a, L), Hs[:, k], Js[:, k])
        Ms.append(Pn.T @ T @ Qn)
    return report(tag, sum(Ms), Ms), Ms


if CURVES == "pack":
    run([(g["decay_pow"], g["decay_km"])] * len(segs), f"{SCHEME}, pack curve d^-{g['decay_pow']:.1f} exp(-d/{g['decay_km']:g}) for all")
elif CURVES == "fit":
    GRID = [(a, L) for a in (1.2, 1.6, 2.0, 2.4, 2.8) for L in (15, 30, 80, 300)]
    best = []
    for s, k in si.items():
        res = []
        for a, L in GRID:
            T = solve(decay(a, L), Hs[:, k], Js[:, k])
            M = Pn.T @ T @ Qn
            res.append((miss10(M, k), a, L, float((T * d).sum() / T.sum())))
        res.sort()
        print(f"  {s}: best " + "; ".join(f"d^-{a} exp(-d/{L}) {m:.1%} ({km:.1f} km)" for m, a, L, km in res[:4]), flush=True)
        best.append(res[0][1:3])
    run(best, f"{SCHEME}, fitted per segment " + " ".join(f"{s} {a}/{L}" for s, (a, L) in zip(segs, best)))
else:
    cs = [tuple(map(float, c.split("/"))) for c in CURVES.split(";")]
    run(cs if len(cs) == len(segs) else cs * len(segs), f"{SCHEME}, curves {CURVES}")
cen = dict(own=np.trace(Cl) / Cl.sum(), west3=sh(Cl, W8, C3), west5=sh(Cl, W8, C5), musa5=sh(Cl, [MU], C5),
           all3=Cl[:, C3].sum() / Cl.sum(), rings=[round(sh(Cl, np.nonzero(s)[0], C3), 3) for s in RINGS])
print("census:", {k: (round(v, 3) if not isinstance(v, list) else v) for k, v in cen.items()})
