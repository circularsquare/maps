"""T-1150: the jobs proxy's height cap, height power and NRES weight, scored on Shanghai's economic
census district shares (proxy alone) and the jobs round Lujiazui once spread inside districts;
for another pack (argument), the share of its jobs near named places under each variant.
    python s16_height.py [city name=lon,lat ...]"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
from t1150lib import *
import tier0, england, china

HERE = os.path.dirname(os.path.abspath(__file__))
city = sys.argv[1] if len(sys.argv) > 1 else "sha"
sig = load_sig(city)
osm = sig["osm_errand"] + sig["osm_leisure"] + sig["osm_job"] + sig["osm_land"]
if city == "sha":
    ds = sha_units(); units = [(d[0], d[1]) for d in ds]
    idx = np.load(os.path.join(HERE, "sha_idx.npy"))
    pop = np.load(os.path.join(HERE, "sha_pop.npy"))
    J = china.JOBS["sha"]; names = [d[0] for d in ds]
    cen = np.array([J[n] for n in names]) / sum(J.values())
    c7 = [names.index(n) for n in ("黄浦区", "徐汇区", "长宁区", "静安区", "普陀区", "虹口区", "杨浦区")]
    spots = {"Lujiazui": (121.5055, 31.2385), "People's Sq": (121.4737, 31.2304), "Baosteel": (121.44, 31.41)}
else:
    pop = sig["pop"]
    spots = {a.split("=")[0]: tuple(map(float, a.split("=")[1].split(","))) for a in sys.argv[2:]}


def near(lo, la, km):
    return 6371 * np.hypot(np.radians(sig["lon"] - lo) * np.cos(np.radians(la)), np.radians(sig["lat"] - la)) < km


for cap in (40.0, 80.0, 150.0):
    for pw in (2.0, 3.0):
        nr = tier0.nres_signal(sig, pw, cap)
        for wn in (0.3, 0.5):
            blend = {"nres": wn, "osm": 0.8 - wn, "pop": 0.2}
            w = blend["nres"] * nr / nr.sum() + blend["osm"] * osm / osm.sum() + blend["pop"] * pop / pop.sum()
            if city == "sha":
                sh = np.bincount(idx[idx >= 0], weights=w[idx >= 0], minlength=len(ds)); sh /= sh.sum()
                jobs = england.spread_totals(sig["lat"], sig["lon"], units, J, w)
                sp = " ".join(f"{k} {jobs[near(*v, 1)].sum() / 1e3:,.0f}k" for k, v in spots.items())
                print(f"cap {cap:.0f} m, power {pw:g}, nres {wn}: districts misplaced {0.5 * np.abs(sh - cen).sum():.1%}, central 7 {sh[c7].sum():.1%} "
                      f"(census {cen[c7].sum():.1%}); densest cell {jobs.max():,.0f}; 1 km: {sp}")
            else:
                jobs = w / w.sum() * 1.0
                sp = " ".join(f"{k} {jobs[near(*v, 3)].sum():.1%} (1 km {jobs[near(*v, 1)].sum():.2%})" for k, v in spots.items())
                o = np.sort(jobs)[::-1]
                print(f"cap {cap:.0f} m, power {pw:g}, nres {wn}: top 1% cells {o[:len(o) // 100].sum():.1%}; within 3 km: {sp}")
