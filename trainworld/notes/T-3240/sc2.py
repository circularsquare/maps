"""T-3254: the pull's scenarios on base.npz (rows only, every place), with the origin damping and the window.
    python sc.py [--gamma G] [--gx G] [--cn F] [--phase lo,hi] [--damp D] [--table]
pull_ij = 1 + expm1(g_ij x_j) x exp(-D x_i) x fade(d), each origin's departures kept.
"""
import json, sys
import numpy as np
TW = r"C:\Users\anita\projects\maps\trainworld"
HERE = r"C:\Users\anita\AppData\Local\Temp\claude\dce1fefcc1d43aef\2ba1fd73-4c91-45ba-b838-fe8f908d4b21\scratchpad\pull2"
B = np.load(HERE + r"\base.npz")  # from base2.py
T, d, cc, pop, dep, Z = B["T"], B["d"], B["cc"], B["pop"], B["dep"], list(B["Z"])
ix = {z: i for i, z in enumerate(Z)}
CFG = json.load(open(TW + r"\pipeline\market_pull.json", encoding="utf-8"))
EX0 = np.array([CFG["places"].get(z, [0.])[0] for z in Z])
OFF = np.isin(cc, CFG["off"])
on = lambda x, lo, hi: np.clip((x - lo) / (hi - lo), 0, 1)

# caps: the conservative published all-mode figure, both ways a day (pull2/respairs/pairs.md)
OVER = [("ber", "ham", 20000), ("rom", "nap", 20000), ("mos", "sai", 38000), ("ist", "ank", 27000), ("mad", "bar", 30650), ("mad", "val", 15730)]
FIT = [("war", "kra", 12000), ("war", "tro+gda+stz", 10000), ("ber", "mun", 18000), ("ham", "mun", 7000), ("par", "lyo", 28000),
       ("mil", "rom", 14000), ("bei", "sha", 50000), ("jnj", "nag", 6000)]
XB = [("par", "lon", 28000), ("par", "bru", 12000), ("sin", "kua", 11000), ("mad", "lis", 5000), ("cop", "sto", 4000), ("zur", "mil", 4000),
      ("vie", "bud", 0), ("ber", "war", 0), ("ber", "pra", 0), ("mun", "vie", 0)]
OTHER = [("nyc", "bos", 53000), ("nyc", "wdc", 41000), ("lon", "man", 0), ("tyo", "osa", 0), ("fra", "col", 0), ("mun", "stu", 0),
         ("bei", "tia", 70000), ("mum", "pun", 0), ("war", "lod", 0), ("kra", "kat", 0)]
# national ceilings, one way a person a weekday, 100-300 / 300+ (pull2/resnat/national.md); None: no ceiling
NAT = {"US": (0.0268, 0.0098), "JP": (None, 0.006), "DE": (0.024, 0.008), "PL": (None, 0.003), "FR": (0.013, 0.0065),
       "ES": (None, 0.008), "CN": (None, 0.005), "RU": (None, 0.005), "TR": (None, 0.004), "IT": (None, 0.0055), "IN": (None, None), "BR": (None, None)}


def scenario(gamma=CFG["gamma"], gx=CFG["gamma_x"], cn=CFG["country_gamma"]["CN"], phase=CFG["phase_km"], damp=0.0, ex=EX0, nopull=False, cg=None, off=None):
    if nopull:
        return T
    cg = dict(cg or {}); cg["CN"] = cn
    gdom = np.array([gamma * cg.get(c, 1.0) for c in cc])
    same = cc[:, None] == cc[None, :]
    full = np.where(same, np.expm1(gdom[None, :] * ex[None, :]), np.expm1(gx * ex[None, :])).astype(np.float32)
    full *= np.exp(-damp * ex)[:, None].astype(np.float32)
    W = full * on(d, *phase).astype(np.float32) + 1
    W[OFF if off is None else np.isin(cc, off)] = 1.
    k = dep / (dep + (T * (W - 1)).sum(1))
    return T * W * k[:, None].astype(np.float32)


def pair(M, x, y):
    tot = 0.
    for yy in y.split("+"):
        i, j = ix[x], ix[yy]
        tot += M[i, j] + M[j, i]
    return float(tot)


def nat(M, c):
    m = cc == c
    sub, dd = M[np.ix_(m, m)], d[np.ix_(m, m)]
    pc = pop[m].sum()
    return float(sub[(dd >= 100) & (dd < 300)].sum() / pc), float(sub[dd >= 300].sum() / pc), float(M[m][:, ~m].sum() / pc)


if __name__ == "__main__":
    arg = lambda k, dflt: float(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else dflt
    ph = [float(v) for v in sys.argv[sys.argv.index("--phase") + 1].split(",")] if "--phase" in sys.argv else CFG["phase_km"]
    M = scenario(arg("--gamma", CFG["gamma"]), arg("--gx", CFG["gamma_x"]), arg("--cn", CFG["country_gamma"]["CN"]), ph, arg("--damp", 0.), off=sys.argv[sys.argv.index("--off") + 1].split(",") if "--off" in sys.argv else None, cg={k: float(v) for k, v in (kv.split(":") for kv in sys.argv[sys.argv.index("--cg") + 1].split(","))} if "--cg" in sys.argv else None)
    print(sys.argv[1:])
    for name, L in [("overshoot", OVER), ("fitted", FIT), ("across a border", XB), ("other", OTHER)]:
        print(name)
        for x, y, cap in L:
            k = d[ix[x], ix[y.split("+")[0]]]
            m, b = pair(M, x, y), pair(T, x, y)
            flag = " OVER" if cap and m > cap * 1.0001 and m > b * 1.0001 else ""
            print(f"  {x}-{y:12s} {k:5.0f} km {b:9.0f} -> {m:9.0f}  cap {cap:7.0f}  {m / cap if cap else 0:5.2f}{flag}")
    for c, (c1, c3) in NAT.items():
        b1, b3, ba = nat(T, c); m1, m3, ma = nat(M, c)
        print(f"  {c}: 100-300 {b1:.4f} -> {m1:.4f} (ceil {c1}), 300+ {b3:.4f} -> {m3:.4f} (ceil {c3}), abroad {ba:.4f} -> {ma:.4f}")




