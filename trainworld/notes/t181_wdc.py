"""T-181: Washington's own border factor, checked with T-162's fit (notes/t162_fit.py).

    ..\\venv\\Scripts\\python.exe trainworld/notes/t181_wdc.py

Read-only on the pack. Fits Washington's k at the shipped decay (the pack's gravity block), and
prints the state-pair commuter shares for k = 0.23 (the US value), 0.49 (T-162's Washington fit)
and the fit now, against LODES 8 OD 2023.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")

import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t162_fit as f  # noqa: E402


def pairs(c, T):
    nr = int(c.zr.max()) + 1
    R = np.zeros((nr, nr))
    for a in range(nr):
        for b in range(nr):
            R[a, b] = T[np.ix_(c.zr == a, c.zr == b)].sum(dtype=np.float64)
    return R / R.sum(1, keepdims=True)


def main():
    header, _ = f.load_pack("wdc")
    g = header["gravity"]
    pw, km = float(g.get("decay_pow", 1.0)), float(g["decay_km"])
    print(f"wdc pack gravity: d^-{pw:g} exp(-d/{km:g}), border factor {g.get('border_factor')}", flush=True)
    c = f.City("wdc")
    own = c.fit_k(pw, km)
    ids = header["regions"]["names"]
    for k in (0.23, 0.49, own):
        T = c.solve(pw, km, k)
        r = c.score(T)
        print(f.line(c, pw, km, k, r) + (" (own fit)" if k == own else ""), flush=True)
        R = pairs(c, T)
        print("    by state pair, share of the home state's commuters (rows home, columns work): " + " ".join(ids))
        for i, name in enumerate(ids):
            print(f"      {name}: " + "  ".join(f"{v:6.1%}" for v in R[i]))
    print(f"    LODES by state pair (rows home):\n{c.obs_states}")


if __name__ == "__main__":
    main()
