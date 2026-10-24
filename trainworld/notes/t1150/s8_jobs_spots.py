"""T-1150: Shanghai's jobs proxy at named centres: jobs within 1 km, and what each part of the proxy
brings there; plus how even the jobs are."""
import sys
sys.stdout.reconfigure(encoding="utf-8")
from t1150lib import *
import tier0

HERE = os.path.dirname(os.path.abspath(__file__))
sig = load_sig("sha")
pop = np.load(os.path.join(HERE, "sha_pop.npy"))
jobs = np.load(os.path.join(HERE, "sha_jobs.npy"))
idx = np.load(os.path.join(HERE, "sha_idx.npy"))
s2 = dict(sig); s2["pop"] = pop
osm = sig["osm_errand"] + sig["osm_leisure"] + sig["osm_job"] + sig["osm_land"]
nres = tier0.nres_signal(s2)
parts = {"nres": 0.3 * nres / nres.sum(), "osm": 0.5 * osm / osm.sum(), "pop": 0.2 * pop / pop.sum()}
w = sum(parts.values())
# each cell's jobs split by part, in proportion to the parts' weights there
spots = {"Lujiazui": (121.5055, 31.2385), "People's Square": (121.4737, 31.2304), "Nanjing West Rd": (121.4555, 31.2290),
         "Jing'an Temple": (121.4455, 31.2235), "Xujiahui": (121.4370, 31.1950), "Hongqiao hub": (121.3200, 31.1940),
         "Zhangjiang": (121.5900, 31.2030), "Jinqiao": (121.5900, 31.2650), "Baosteel": (121.4400, 31.4100),
         "Wujiaochang": (121.5140, 31.3020), "Minhang Xinzhuang": (121.3850, 31.1110)}
R = 6371.0
print("spot | jobs within 1 km | homes within 1 km | jobs/homes | jobs by part nres/osm/pop | max cell jobs")
for name, (lo, la) in spots.items():
    d = R * np.hypot(np.radians(sig["lon"] - lo) * np.cos(np.radians(la)), np.radians(sig["lat"] - la))
    s = d < 1.0
    j = jobs[s].sum()
    byp = {k: (jobs[s] * parts[k][s] / np.maximum(w[s], 1e-30)).sum() for k in parts}
    print(f"{name} | {j:,.0f} | {pop[s].sum():,.0f} | {j / max(pop[s].sum(), 1):.2f} | "
          f"{byp['nres'] / j:.0%}/{byp['osm'] / j:.0%}/{byp['pop'] / j:.0%} | {jobs[s].max():,.0f}")
# evenness: share of jobs in the densest 1% / 5% of cells, and the share by part overall
o = np.sort(jobs)[::-1]
print(f"\njobs in the top 1% of cells {o[:len(o) // 100].sum() / o.sum():.1%}, top 5% {o[:len(o) // 20].sum() / o.sum():.1%}")
op = np.sort(pop)[::-1]
print(f"homes in the top 1% of cells {op[:len(op) // 100].sum() / op.sum():.1%}, top 5% {op[:len(op) // 20].sum() / op.sum():.1%}")
tot = {k: (jobs * parts[k] / np.maximum(w, 1e-30)).sum() / jobs.sum() for k in parts}
print("jobs by part, all Shanghai:", {k: f"{v:.1%}" for k, v in tot.items()})
print("OSM signal make-up:", {k: f"{sig['osm_' + k].sum() / osm.sum():.0%}" for k in ("errand", "leisure", "job", "land")})
# correlation of jobs with homes at 2 km zone level
