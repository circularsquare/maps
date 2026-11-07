"""Tables from pairs.npz (analyse.py).  python report.py [len|grad|both]"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
p = np.load(os.path.join(HERE, "pairs.npz"))
box, q, ra, mk, md, gd, grad, rk = p["box"], p["q"], p["ra_km"], p["merit_km"], p["merit_drop"], p["gmted_drop"], p["grad"], p["reach_km"]
CL = [(1, 10), (10, 100), (100, 1000), (1000, 1e7)]
what = sys.argv[1] if len(sys.argv) > 1 else "both"


def closing(qq):
    return np.where(qq >= 1000, np.clip(2.5 * np.sqrt(1000 / np.maximum(qq, 1000)), 0.3, 2), 5.0)


def shares(g, qq, w):
    g = np.where(np.isfinite(g), g, 0)
    c = g > closing(qq)
    rp = (g >= 1) & ~c
    t = w.sum()
    return "%2.0f/%2.0f/%2.0f" % (100 * w[~rp & ~c].sum() / t, 100 * w[rp].sum() / t, 100 * w[c].sum() / t)


boxes = list(dict.fromkeys(box.tolist()))
ratio = mk / ra
ok = (ratio > 0.5) & (ratio < 2.5)     # drops gross mis-snaps (count shown)
if what in ("len", "both"):
    print("MERIT km / RiverATLAS km by Q class: ratio of sums (median) [pairs]; %d of %d pairs within 0.5-2.5" % (ok.sum(), len(ok)))
    print("%-20s" % "box" + "".join("%26s" % ("Q %g-%g" % c) for c in CL))
    for b in boxes + ["ALL"]:
        s = "%-20s" % b
        for lo, hi in CL:
            m = ok & (q >= lo) & (q < hi) & ((box == b) if b != "ALL" else True)
            s += "%26s" % ("%.3f (%.3f) [%d]" % (mk[m].sum() / ra[m].sum(), np.median(ratio[m]), m.sum()) if m.sum() >= 5 else "-")
        print(s)
if what in ("grad", "both"):
    print("\nshare of reach km flat/rapids/closed: built (GMTED from reach start) | GMTED midpoints | MERIT drop / RA km | MERIT drop / MERIT km")
    for lo, hi in CL[:3]:
        print("Q %g-%g" % (lo, hi))
        for b in boxes + ["ALL"]:
            m = ok & (q >= lo) & (q < hi) & ((box == b) if b != "ALL" else True)
            if m.sum() < 10:
                continue
            print("  %-20s %5d  %s | %s | %s | %s   median m/km GMTED %.2f MERIT %.2f" % (
                b, m.sum(), shares(grad[m], q[m], rk[m]), shares(gd[m] / ra[m], q[m], rk[m]),
                shares(md[m] / ra[m], q[m], rk[m]), shares(md[m] / mk[m], q[m], rk[m]),
                np.median(gd[m] / ra[m]), np.median(md[m] / ra[m])))
