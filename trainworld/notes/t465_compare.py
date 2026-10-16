"""T-465: shipped tyo gravity vs census table 4, west Tokyo to the centre; plus curve trials.
    python notes/t465_prep.py                 # first: municipality shares and census flows to artifacts/
    python notes/t465_compare.py shipped      # the table for the shipped factors
    python notes/t465_compare.py inflow       # who fills the centre's jobs, by ring and prefecture
    python notes/t465_compare.py trials       # re-solve with other curves
Env: FIXWFH=0.15 (at home per municipality of work from table 4, telework 0.15 on the rest),
FIXHOME=1 (home end from table 4 by residence), CURVES=2.2/80,1.85/30 (only these), FULL=1.
"""
import os
os.environ["OMP_NUM_THREADS"] = "2"
import json, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = np.load(sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "..", "..", "artifacts", "trainworld", "T-465-prep.npz"))
P, Q, H, J, gx, gy, Cl = D["P"], D["Q"], D["H"], D["J"], D["gx"], D["gy"], D["Cl"]
codes = list(D["codes"])
g = json.loads(str(D["g"]))
ci = {c: i for i, c in enumerate(codes)}
if os.environ.get("FIXWFH"):
    # at home per municipality of work from table 4, telework 0.15 on the rest, in place of a flat 0.23
    ah = D["census_home"] / np.maximum(D["census_all"], 1)
    tele = float(os.environ["FIXWFH"])
    Q = Q * ((1 - ah) * (1 - tele) / (1 - 0.23))[None, :]
    J = Q.sum(1)
    print(f"FIXWFH: commuting jobs {J.sum():,.0f} (flat 0.23: {D['J'].sum():,.0f})")
if os.environ.get("FIXHOME"):
    # home end: the census's commuters by municipality of residence (table 4, at home out), spread by population
    P = P / np.maximum(P.sum(0, keepdims=True), 1e-12) * Cl.sum(1)[None, :]
    H = P.sum(1)
    print("FIXHOME: home end from table 4 by residence")
ORIG = {"Musashino": "13203", "Mitaka": "13204", "Suginami": "13115", "Nerima": "13120", "Chofu": "13208",
        "Fuchu": "13206", "Tachikawa": "13202", "Hachioji": "13201"}
DEST = {"Chiyoda": "13101", "Chuo": "13102", "Minato": "13103", "Shinjuku": "13104", "Shibuya": "13113"}
WARDS = [ci[f"131{k:02d}"] for k in range(1, 24)]
C5 = [ci[c] for c in DEST.values()]
C3 = C5[:3]

d = 2.0 * np.sqrt((gx[:, None] - gx[None, :]) ** 2 + (gy[:, None] - gy[None, :]) ** 2)
d[d == 0] = g["intrazonal_km"]
Pn = P / np.maximum(P.sum(1, keepdims=True), 1e-12)
Qn = Q / np.maximum(Q.sum(1, keepdims=True), 1e-12)
# same-municipality probability between zones: for an intra-municipality term
S = Pn @ Qn.T


def miss(a, b):
    return 0.5 * float(np.abs(a / a.sum() - b / b.sum()).sum())


def to_muni(T):
    return Pn.T @ T @ Qn


def solve(F, iters=400, tol=1e-5):
    o = H * J.sum() / H.sum()
    col = np.ones(len(J)); row = np.ones(len(H))
    for it in range(iters):
        row = np.where(o > 0, o / np.maximum(F @ col, 1e-300), 0)
        t = row @ F
        col = np.where(J > 0, J / np.maximum(t, 1e-300), 0)
        if it > 5 and np.max(np.abs((F @ col)[o > 0] * row[o > 0] / o[o > 0] - 1)) < tol:
            break
    return F * row[:, None] * col[None, :]


def report(C, label):
    own = np.trace(C) / C.sum()
    T_mean = None
    out = {"label": label, "misplaced": miss(C, Cl), "own": own}
    rows = []
    for n, c in ORIG.items():
        i = ci[c]
        m = C[i] / C[i].sum(); k = Cl[i] / Cl[i].sum()
        rows.append((n, m[C3].sum(), k[C3].sum(), m[C5].sum(), k[C5].sum(), m[WARDS].sum(), k[WARDS].sum(), m[i], k[i]))
    out["rows"] = rows
    w = [ci[c] for c in ORIG.values()]
    mw, kw = C[w].sum(0) / C[w].sum(), Cl[w].sum(0) / Cl[w].sum()
    out["west_c3"] = (mw[C3].sum(), kw[C3].sum())
    out["west_c5"] = (mw[C5].sum(), kw[C5].sum())
    # every origin outside the 23 wards: share to the 3 central wards
    nw = [i for i in range(len(codes)) if i not in WARDS]
    out["nonward_c3"] = ((C[nw][:, C3].sum() / C[nw].sum()), (Cl[nw][:, C3].sum() / Cl[nw].sum()))
    out["all_c3"] = (C[:, C3].sum() / C.sum(), Cl[:, C3].sum() / Cl.sum())
    out["rings"] = [(C[s][:, C3].sum() / C[s].sum(), Cl[s][:, C3].sum() / Cl[s].sum()) for s in RINGS]
    return out


_w = np.bincount(D["li"], weights=D["ch"] + 1e-6, minlength=len(codes))
_r = np.hypot(np.bincount(D["li"], weights=D["x"] * (D["ch"] + 1e-6), minlength=len(codes)),
              np.bincount(D["li"], weights=D["y"] * (D["ch"] + 1e-6), minlength=len(codes))) / np.maximum(_w, 1e-9) / 1000
RINGS = [(_r < 5), (_r >= 5) & (_r < 10), (_r >= 10) & (_r < 20), (_r >= 20) & (_r < 30), (_r >= 30) & (_r < 60)]


def show(o, full=False):
    print(f"== {o['label']}: misplaced {o['misplaced']:.1%}, own municipality {o['own']:.1%} (census {np.trace(Cl) / Cl.sum():.1%})")
    if full:
        print(f"{'from':10s} {'to Chiyoda+Chuo+Minato':>24s} {'to the 5 central':>18s} {'to the 23 wards':>18s} {'own':>14s}   (model / census)")
        for r in o["rows"]:
            print(f"{r[0]:10s} {r[1]:11.1%} / {r[2]:5.1%} {r[3]:9.1%} / {r[4]:5.1%} {r[5]:9.1%} / {r[6]:5.1%} {r[7]:6.1%} / {r[8]:5.1%}")
    print(f"   8 west origins -> Chiyoda+Chuo+Minato {o['west_c3'][0]:.1%} / {o['west_c3'][1]:.1%}; -> 5 central {o['west_c5'][0]:.1%} / {o['west_c5'][1]:.1%}; "
          f"all outside the wards -> 3 central {o['nonward_c3'][0]:.1%} / {o['nonward_c3'][1]:.1%}; everyone -> 3 central {o['all_c3'][0]:.1%} / {o['all_c3'][1]:.1%}", flush=True)
    print("   rings 0-5/5-10/10-20/20-30/30-60 km -> 3 central: " + "  ".join(f"{a:.1%}/{b:.1%}" for a, b in o["rings"]), flush=True)


def mean_km(T):
    return float((T * d).sum() / T.sum())


if sys.argv[1] == "shipped":
    T = D["row"][:, None] * D["col"][None, :] * np.exp(-d / g["decay_km"]) * d ** -g["decay_pow"]
    print(f"shipped trips {T.sum():,.0f}, mean {mean_km(T):.2f} km")
    o = report(to_muni(T), "shipped d^-2.2 exp(-d/80)")
    show(o, True)
    # absolute numbers for Musashino
    C = to_muni(T)
    for n in ("Musashino", "Mitaka"):
        i = ci[ORIG[n]]
        print(n, "model commuters", round(C[i].sum()), "census commuters", round(Cl[i].sum()),
              {k: (round(C[i, ci[c]]), round(Cl[i, ci[c]])) for k, c in DEST.items()})
    # jobs in the centre: pack vs census workers by place of work
    for k, c in DEST.items():
        i = ci[c]
        print(f"jobs {k}: pack {D['pack_jobs'][i]:,.0f}, census by place of work {D['census_all'][i]:,.0f}, commute_work {Q[:, i].sum():,.0f}, census in-flows (pack area, not at home) {Cl[:, i].sum():,.0f}")
    print(f"census flows total {Cl.sum():,.0f}; pack commuters {H.sum():,.0f}")
elif sys.argv[1] == "inflow":
    T = D["row"][:, None] * D["col"][None, :] * np.exp(-d / g["decay_km"]) * d ** -g["decay_pow"]
    C = to_muni(T)
    # municipality centres (commute-home weighted) for distance to Tokyo Station
    x, y, li, ch = D["x"], D["y"], D["li"], D["ch"]
    w = np.bincount(li, weights=ch + 1e-6, minlength=len(codes))
    mx = np.bincount(li, weights=x * (ch + 1e-6), minlength=len(codes)) / w
    my = np.bincount(li, weights=y * (ch + 1e-6), minlength=len(codes)) / w
    r = np.hypot(mx, my) / 1000
    inm, ink = C[:, C3].sum(1), Cl[:, C3].sum(1)
    print(f"into Chiyoda+Chuo+Minato: model {inm.sum():,.0f}, census {ink.sum():,.0f}")
    bands = [0, 5, 10, 15, 20, 30, 40, 60, 200]
    print("origin distance from Tokyo Station: share of the centre's commuters (model / census), and ratio of the origin's share sending to the centre")
    for lo, hi in zip(bands[:-1], bands[1:]):
        s = (r >= lo) & (r < hi)
        print(f"  {lo:3d}-{hi:3d} km: {inm[s].sum() / inm.sum():6.1%} / {ink[s].sum() / ink.sum():6.1%}   origin's commuters to centre {inm[s].sum() / C[s].sum():6.1%} / {ink[s].sum() / Cl[s].sum():6.1%}")
    pref = np.array([c[:2] for c in codes])
    wards = np.zeros(len(codes), bool); wards[WARDS] = True
    for lab, s in (("3 central wards", np.isin(np.arange(len(codes)), C3)), ("other 20 wards", wards & ~np.isin(np.arange(len(codes)), C3)),
                   ("Tokyo outside wards", (pref == "13") & ~wards), ("Kanagawa", pref == "14"), ("Saitama", pref == "11"), ("Chiba", pref == "12"), ("Ibaraki", pref == "08")):
        print(f"  {lab:20s}: {inm[s].sum() / inm.sum():6.1%} / {ink[s].sum() / ink.sum():6.1%}   to centre {inm[s].sum() / C[s].sum():6.1%} / {ink[s].sum() / Cl[s].sum():6.1%}")
    # home end check: pack commuters by residence vs census commuters by residence, share
    hm, hk = C.sum(1) / C.sum(), Cl.sum(1) / Cl.sum()
    print("home end, share of all commuters (model / census):", {lab: (round(hm[ci[c]] * 1e3, 2), round(hk[ci[c]] * 1e3, 2)) for lab, c in {**ORIG, **DEST}.items()}, "per mille")
    # work end: share of all commuters
    jm, jk = C.sum(0) / C.sum(), Cl.sum(0) / Cl.sum()
    print("work end, per mille (model / census):", {lab: (round(jm[ci[c]] * 1e3, 1), round(jk[ci[c]] * 1e3, 1)) for lab, c in {**ORIG, **DEST}.items()})
    ah = D["census_home"] / np.maximum(D["census_all"], 1)
    print("census at-home share by place of work:", {lab: f"{ah[ci[c]]:.1%}" for lab, c in {**ORIG, **DEST}.items()}, f"all {D['census_home'].sum() / D['census_all'].sum():.1%}")
elif sys.argv[1] == "trials":
    trials = []
    for pw, km in [tuple(map(float, s.split("/"))) for s in os.environ["CURVES"].split(",")] if os.environ.get("CURVES") else ():
        trials.append((f"d^-{pw} exp(-d/{km})", (np.exp(-d / km) * d ** -pw)))
    for pw, km in () if os.environ.get("CURVES") else ((2.2, 80), (2.0, 60), (1.85, 30), (1.6, 40), (1.4, 30), (1.0, 40)):
        trials.append((f"d^-{pw} exp(-d/{km})", (np.exp(-d / km) * d ** -pw)))
    # intra-municipality factor k on flatter curves (S = probability two zones' commuters share a municipality)
    for pw, km, kk in ((1.4, 40, 3), (1.6, 60, 2), (1.2, 40, 5)):
        trials.append((f"d^-{pw} exp(-d/{km}) x (1+{kk - 1}*same municipality)", np.exp(-d / km) * d ** -pw * (1 + (kk - 1) * S)))
    # intra-zone (diagonal) factor
    for pw, km, kk in ((1.4, 40, 3),):
        F = np.exp(-d / km) * d ** -pw
        F[np.diag_indices_from(F)] *= kk
        trials.append((f"d^-{pw} exp(-d/{km}), own zone x{kk}", F))
    # two-part mixture: local + long
    for a_, p1, l1, p2, l2 in ((0.7, 2.2, 15, 0.8, 60), (0.8, 2.4, 10, 1.0, 80), (0.6, 2.4, 8, 1.2, 80)):
        F1 = np.exp(-d / l1) * d ** -p1; F2 = np.exp(-d / l2) * d ** -p2
        trials.append((f"mix {a_}*d^-{p1}e^-d/{l1} + {1-a_:.1f}*d^-{p2}e^-d/{l2} (each scaled to sum 1)", a_ * F1 / F1.mean() + (1 - a_) * F2 / F2.mean()))
    if os.environ.get("CURVES"):
        trials = trials[:len(os.environ["CURVES"].split(","))]
    for label, F in trials:
        T = solve(F)
        o = report(to_muni(T), label + f" [mean {mean_km(T):.1f} km]")
        show(o, bool(os.environ.get("FULL")))
