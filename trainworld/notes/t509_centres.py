"""T-509 stage 4 item 4: a job-centre term in Tokyo's commute gravity, on T-465's machinery.

F_ij = d^-p exp(-d/L) x d^(gamma z_j), z_j = log of the jobs within a 3 km kernel of zone j over the
job-weighted median: big centres decay more gently (one worldwide parameter, gamma). Doubly
constrained, so a destination-only factor would be absorbed; this one varies with distance.

    python centres.py [gammas]   (FIXWFH=0.15 as shipped)
"""
import os
import sys

os.environ.setdefault("FIXWFH", "0.15")
ARGS = sys.argv[1:]
sys.argv = [sys.argv[0], "none", r"C:\Users\anita\projects\maps\artifacts\trainworld\T-465-prep.npz"]
src = open(r"C:\Users\anita\projects\maps\trainworld\notes\t465_compare.py", encoding="utf-8").read()
exec(compile(src, "t465_compare.py", "exec"))  # defines D, d, J, solve, report, show, to_muni, mean_km

gammas = [float(x) for x in ARGS[0].split(",")] if ARGS else [0.0, 0.1, 0.2, 0.3, 0.4]
pw, km = g["decay_pow"], g["decay_km"]
dz = 2.0 * np.sqrt((gx[:, None] - gx[None, :]) ** 2 + (gy[:, None] - gy[None, :]) ** 2)
M = np.exp(-dz / 3.0) @ J  # jobs within about 3 km
o = np.argsort(M)
c = np.cumsum(J[o]) / J.sum()
Mref = M[o][np.searchsorted(c, 0.5)]
z = np.log(np.maximum(M, 1.0) / Mref)
if os.environ.get("POS"):  # only centres above the median flatten
    z = np.maximum(z, 0.0)
print(f"z: job-weighted median job mass {Mref:,.0f}; z range {z.min():.2f} to {z.max():.2f}; centre zones z {np.percentile(z, 99.5):.2f}")
base = np.exp(-d / km) * d ** -pw
for gm in gammas:
    F = base * d ** (gm * z[None, :])
    T = solve(F)
    o_ = report(to_muni(T), f"gamma {gm} [mean {mean_km(T):.1f} km]")
    show(o_, bool(os.environ.get("FULL")))
