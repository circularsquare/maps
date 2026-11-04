"""T-1150: share of a tier 0 pack's proxy jobs near named places under the jobs variants (needs
s18's <city>_bi.npz).   python s19_centre.py <city> name=lon,lat ..."""
import sys
sys.stdout.reconfigure(encoding="utf-8")
from t1150lib import *
import tier0

HERE = os.path.dirname(os.path.abspath(__file__))
city = sys.argv[1]
sig = load_sig(city)
bi = dict(np.load(os.path.join(HERE, f"{city}_bi.npz")))
osm = sig["osm_errand"] + sig["osm_leisure"] + sig["osm_job"] + sig["osm_land"]
pop = sig["pop"]
nr2 = tier0.nres_signal(sig)
build = {"now": nr2, "half nres, half intensity p2": 0.5 * nr2 / nr2.sum() + 0.5 * bi["bi2"] / bi["bi2"].sum(), "intensity p2": bi["bi2"]}
spots = {a.split("=")[0]: tuple(map(float, a.split("=")[1].split(","))) for a in sys.argv[2:]}
near = lambda lo, la, km: 6371 * np.hypot(np.radians(sig["lon"] - lo) * np.cos(np.radians(la)), np.radians(sig["lat"] - la)) < km
for bn, b in build.items():
    w = 0.3 * b / b.sum() + 0.5 * osm / osm.sum() + 0.2 * pop / pop.sum()
    w = w / w.sum()
    o = np.sort(w)[::-1]
    sp = "; ".join(f"{k} {w[near(*v, 1)].sum():.2%} in 1 km, {w[near(*v, 3)].sum():.1%} in 3 km" for k, v in spots.items())
    print(f"{bn}: top 1% cells {o[:len(o) // 100].sum():.1%}; {sp}")
