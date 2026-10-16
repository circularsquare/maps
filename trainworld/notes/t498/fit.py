"""T-498 prototype on tyo: commute gravity on d_eff = d x (t_real / t_typical(d))^w, against census table 4.
    python fit.py <prep.npz> <car.npz> <rail.npz> [w list] [curves a/L,...]
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "6")
import sys, json
import numpy as np

D = np.load(sys.argv[1]); Cz = np.load(sys.argv[2]); Rz = np.load(sys.argv[3])
WS = [float(x) for x in (sys.argv[4] if len(sys.argv) > 4 else "0,0.25,0.5,1").split(",")]
CURVES = [tuple(map(float, s.split("/"))) for s in (sys.argv[5] if len(sys.argv) > 5 else "2.2/80,2.0/60,1.85/30,2.4/80,1.6/40").split(",")]
P, Q, H, J, gx, gy, Cl = D["P"], D["Q"], D["H"], D["J"], D["gx"], D["gy"], D["Cl"]
codes = list(D["codes"]); ci = {c: i for i, c in enumerate(codes)}
g = json.loads(str(D["g"]))
d = 2.0 * np.sqrt((gx[:, None] - gx[None, :]) ** 2 + (gy[:, None] - gy[None, :]) ** 2)
d[d == 0] = g["intrazonal_km"]
car, rail = Cz["car"].astype(np.float64), Rz["rail"].astype(np.float64)
walk = d * 1000 * 1.3 / 100.0
if os.environ.get("COST") == "gen":
    # as the game prices them (kernel.rs weights): congested driving 1.33, parking at the destination
    # (3 min, plus a paid space 15 ln(d/3000) capped at 50 for the share who pay, d jobs per km² of land);
    # rail: wait 5 x 1.37, the two platform links 4 x 1.39, access and egress walk x 1.5
    ncell = np.bincount(D["zone"], minlength=len(H))
    jd = (J / 0.8) / np.maximum(ncell * 0.1053, 1e-6)
    paid = np.where(jd > 3000, np.minimum(15 * np.log(np.maximum(jd, 3000) / 3000), 50), 0)
    park = (1 - 1 / (1 + jd / 8000)) * paid
    car = 1.33 * (car - 3.0) + 3.0 + park[None, :]
    rail = rail + 5 * 0.37 + 4 * 0.39  # access walks stay at weight 1 (not split from the ride here)
    walk = 1.5 * walk
    print("COST=gen: generalised minutes as the game prices them", flush=True)
t = np.minimum(np.minimum(car, rail), walk)
# typical real minutes at each distance: the commute-weighted median in log-distance bins
wgt = H[:, None] * J[None, :]
lb = np.log(d)
edges = np.linspace(lb.min() - 1e-9, lb.max() + 1e-9, 41)
b = np.digitize(lb, edges) - 1
mid, med = [], []
for k in range(40):
    m = b == k
    if wgt[m].sum() <= 0:
        continue
    tt, ww = t[m], wgt[m]
    o = np.argsort(tt); cw = np.cumsum(ww[o])
    med.append(tt[o][np.searchsorted(cw, 0.5 * cw[-1])]); mid.append(np.average(lb[m], weights=ww))
mid, med = np.array(mid), np.array(med)
texp = np.exp(np.interp(lb, mid, np.log(med)))
r = t / texp
if os.environ.get("STORE"):
    # as the pack would store it: the zone pair within the drive window, the supercell pair beyond
    # (commute-weighted mean of log r), each a byte (q = round(64 ln r))
    sup, W = Cz["sup"], int(Cz["win"])
    ns = int(sup.max()) + 1
    lr = np.log(r)
    hs = np.bincount(sup, H, ns); js = np.bincount(sup, J, ns)
    S1 = np.zeros((ns, len(sup))); S1[sup, np.arange(len(sup))] = 1
    num = (S1 * H) @ (lr * J[None, :]) @ S1.T
    den = np.outer(hs, js)
    far = np.where(den > 0, num / np.maximum(den, 1e-30), 0.0)
    inwin = (np.abs(gx[:, None] - gx[None, :]) <= W) & (np.abs(gy[:, None] - gy[None, :]) <= W)
    lr = np.where(inwin, lr, far[sup][:, sup])
    r = np.exp(np.round(64 * lr) / 64)
    print("STORE: ratio as the pack would store it", flush=True)
share_rail = (wgt * (rail < car)).sum() / wgt.sum()
print(f"pairs where rail beats car (commute weighted): {share_rail:.1%}; ratio t/typical, commute-weighted p10/p50/p90:",
      [round(float(x), 2) for x in np.percentile(np.repeat(r.ravel()[::97], 1), [10, 50, 90])], flush=True)
print("typical minutes at 2/5/10/20/30/50 km:", [round(float(np.exp(np.interp(np.log(x), mid, np.log(med)))), 1) for x in (2, 5, 10, 20, 30, 50)], flush=True)

if "13203" in ci:
    ORIG = {"Musashino": "13203", "Mitaka": "13204", "Suginami": "13115", "Nerima": "13120", "Chofu": "13208",
            "Fuchu": "13206", "Tachikawa": "13202", "Hachioji": "13201"}
    C5 = [ci[c] for c in ("13101", "13102", "13103", "13104", "13113")]; C3 = C5[:3]
    W8 = [ci[c] for c in ORIG.values()]
    MU = ci["13203"]
else:  # London: central = City, Westminster, Camden, Tower Hamlets, Islington; "west" = every district outside Greater London
    C5 = [ci[c] for c in ("E09000001", "E09000033", "E09000007", "E09000030", "E09000019")]; C3 = C5[:2]
    W8 = [i for i, c in enumerate(codes) if not c.startswith("E09")]
    MU = ci.get("E07000241", W8[0])  # Welwyn Hatfield if inside
Pn = P / np.maximum(P.sum(1, keepdims=True), 1e-12); Qn = Q / np.maximum(Q.sum(1, keepdims=True), 1e-12)
_w = np.bincount(D["li"], weights=D["ch"] + 1e-6, minlength=len(codes))
_r = np.hypot(np.bincount(D["li"], weights=D["x"] * (D["ch"] + 1e-6), minlength=len(codes)),
              np.bincount(D["li"], weights=D["y"] * (D["ch"] + 1e-6), minlength=len(codes))) / np.maximum(_w, 1e-9) / 1000
RINGS = [(_r < 5), (_r >= 5) & (_r < 10), (_r >= 10) & (_r < 20), (_r >= 20) & (_r < 30), (_r >= 30) & (_r < 60)]


def solve(F, iters=300, tol=1e-4):
    o = H * J.sum() / H.sum()
    col = np.ones(len(J))
    for it in range(iters):
        row = np.where(o > 0, o / np.maximum(F @ col, 1e-300), 0)
        col = np.where(J > 0, J / np.maximum(row @ F, 1e-300), 0)
        if it > 5 and it % 5 == 0:
            e = np.max(np.abs((F @ col)[o > 0] * row[o > 0] / o[o > 0] - 1))
            if e < tol:
                break
    return F * row[:, None] * col[None, :]


def miss(a, b):
    return 0.5 * float(np.abs(a / a.sum() - b / b.sum()).sum())


def sh(M, rows, cols):
    return M[rows][:, cols].sum() / M[rows].sum()


res = []
for w in WS:
    deff = d * r ** w
    for a, L in CURVES:
        F = np.exp(-deff / L) * deff ** -a
        T = solve(F)
        M = Pn.T @ T @ Qn
        mu = MU
        row = dict(w=w, a=a, L=L, misplaced=miss(M, Cl), own=np.trace(M) / M.sum(),
                   west3=sh(M, W8, C3), west5=sh(M, W8, C5), musa5=sh(M, [mu], C5), all3=M[:, C3].sum() / M.sum(),
                   rings=[sh(M, np.nonzero(s)[0], C3) for s in RINGS], km=float((T * d).sum() / T.sum()))
        res.append(row)
        print(f"w {w:4.2f} d^-{a} exp(-d/{L:g}): misplaced {row['misplaced']:.1%} own {row['own']:.1%} west->3 {row['west3']:.1%} west->5 {row['west5']:.1%} "
              f"Musashino->5 {row['musa5']:.1%} all->3 {row['all3']:.1%} rings " + "/".join(f"{x:.1%}" for x in row["rings"]) + f" mean {row['km']:.1f} km", flush=True)
cen = dict(own=np.trace(Cl) / Cl.sum(), west3=sh(Cl, W8, C3), west5=sh(Cl, W8, C5), musa5=sh(Cl, [MU], C5), all3=Cl[:, C3].sum() / Cl.sum(),
           rings=[sh(Cl, np.nonzero(s)[0], C3) for s in RINGS])
print("census:", {k: (round(v, 3) if not isinstance(v, list) else [round(x, 3) for x in v]) for k, v in cen.items()})
