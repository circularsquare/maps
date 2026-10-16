"""T-456 part 1: OSM errand and leisure places (tier 0 signals) per 1,000 residents and per 1,000
jobs in each non-US pack, overall and by the 2 km zone's residential density. Read only."""
import os, sys, glob, io, contextlib
os.environ["OMP_NUM_THREADS"] = "2"
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"C:\Users\anita\projects\maps\trainworld\notes")
import numpy as np
from t453_common import pack, signals_on

BANDS = [(0, 1000, "<1k"), (1000, 4000, "1-4k"), (4000, 10000, "4-10k"), (10000, 1e9, "10k+")]
cities = sorted(os.path.basename(f).split(".")[0] for f in glob.glob(r"C:\Users\anita\projects\maps\trainworld\data\work\*.tier0_signals.npz"))
print("| pack | residents | jobs | errand /1k res | leisure /1k res | errand /1k jobs | leisure /1k jobs | "
      + " | ".join(f"errand /1k res, {b[2]}" for b in BANDS) + " | pop share by band |")
for c in cities:
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            p = pack(c)
            s = signals_on(p, c)
    except Exception as e:
        print(f"<!-- {c}: {type(e).__name__} {e} -->")
        continue
    pop, jobs, er, le = p["pop"], p["jobs"], s["osm_errand"], s["osm_leisure"]
    zp = np.bincount(p["zone"], weights=pop)
    zm2 = (p["zm"] / 1000.0) ** 2
    dens = zp[p["zone"]] / zm2
    cells = []
    shares = []
    for lo, hi, _ in BANDS:
        m = (dens >= lo) & (dens < hi)
        cells.append(f"{1000 * er[m].sum() / max(pop[m].sum(), 1):.1f}" if pop[m].sum() > 20000 else "")
        shares.append(f"{100 * pop[m].sum() / pop.sum():.0f}")
    print(f"| {c} | {pop.sum() / 1e6:.2f}M | {jobs.sum() / 1e6:.2f}M | {1000 * er.sum() / pop.sum():.2f} | {1000 * le.sum() / pop.sum():.2f} | "
          f"{1000 * er.sum() / jobs.sum():.1f} | {1000 * le.sum() / jobs.sum():.1f} | " + " | ".join(cells) + " | " + "/".join(shares) + " |", flush=True)
