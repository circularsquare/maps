"""T-456: the proposed fallback on London's pack, against BRES sector jobs (T-453's truth), at 2 km zones."""
import os, sys, io, contextlib, runpy
os.environ["OMP_NUM_THREADS"] = "2"
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"C:\Users\anita\projects\maps\trainworld\notes")
sys.path.insert(0, r"C:\Users\anita\projects\maps\trainworld\pipeline")
import numpy as np
import t453_common, england  # noqa: imported before stdout is redirected
with contextlib.redirect_stdout(io.StringIO()):
    G = runpy.run_path(r"C:\Users\anita\projects\maps\trainworld\notes\t453_lon.py")
p, E, L, sig, zm = G["p"], G["E"], G["L"], G["sig"], G["zone_misplaced"]
rng = np.random.default_rng(456)
nz = int(p["zone"].max()) + 1
Z = lambda v: np.bincount(p["zone"], weights=v, minlength=nz)
pop, jobs = p["pop"], p["jobs"]
zj, zp = Z(jobs) / jobs.sum(), Z(pop) / pop.sum()
print("| purpose | OSM kept | OSM /1k res | variant | misplaced % |\n|---|---|---|---|---|")
for purpose, truth, osm in (("errands", E["sector"], sig["osm_errand"]), ("leisure", L["sector"], sig["osm_leisure"])):
    for frac in (1.0, 0.1, 0.01):
        o = osm if frac >= 1 else rng.binomial(np.round(osm).astype(int), frac).astype(float)
        zo = Z(o)
        r = 1000 * o.sum() / pop.sum()
        c = min(1.0, r / 5.0)
        so1 = (zo + 1) / (zo + 1).sum()
        out = {"jobs (C)": zj, "D (T-453)": np.sqrt(zj * zo / max(zo.sum(), 1))}
        for b in (0.0, 0.25, 0.4):
            base = (1 - b) * zj + b * zp
            out[f"D', b {b}"] = base ** (1 - c / 2) * so1 ** (c / 2)
        for k, wz in out.items():
            # zone weights spread by OSM inside the zone where it has any, else by jobs (as the build would)
            w = wz[p["zone"]] * np.where(Z(o)[p["zone"]] > 0, o / np.maximum(Z(o)[p["zone"]], 1e-9), jobs / np.maximum(Z(jobs)[p["zone"]], 1e-9))
            print(f"| {purpose} | {frac:.0%} | {r:.2f} | {k} | {zm(p, w, truth):.1f} |", flush=True)
