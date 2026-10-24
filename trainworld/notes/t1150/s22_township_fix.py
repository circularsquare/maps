"""T-1150: Shanghai homes from the 2020 census by township (helper1m's table, ASPECT summed over each
township) spread inside each township by GHSL as now (k inf) or by the bounded fix (k 2); densest
cells, and an alt npz for s4_maps.py's cell picture."""
import sys, json
sys.stdout.reconfigure(encoding="utf-8")
from t1150lib import *
import england
from shapely.geometry import shape

HERE = os.path.dirname(os.path.abspath(__file__))
sig = load_sig("sha")
area = np.load(os.path.join(HERE, "sha_area.npy"))
var = np.load(os.path.join(HERE, "sha_bounded.npz"))
fc = json.load(open(r"C:\Users\anita\projects\maps\helper1m\countries\china\adm4\01.geojson", encoding="utf-8"))
tw = [(f["properties"]["code"], shape(f["geometry"])) for f in fc["features"]]
tot = {f["properties"]["code"]: f["properties"]["populations"]["2020"] for f in fc["features"]}
ds = sha_units()
dtot = {d[0]: d[2] for d in ds}
for nm in ("kinf", "k2", "k1.5"):
    p = england.spread_totals(sig["lat"], sig["lon"], tw, tot, var[nm])
    # back onto the 16 districts' census totals (township shapes are 2018's, district lines OSM's)
    p = england.spread_totals(sig["lat"], sig["lon"], [(d[0], d[1]) for d in ds], dtot, p)
    d = p / area
    print(f"township totals, {nm} inside: densest cell {d.max():,.0f}/km2, over 100k/km2 {(d > 1e5).sum()}, over 60k {(d > 6e4).sum()}, top 0.1% {np.quantile(d[p > 0], .999):,.0f}")
    np.savez_compressed(os.path.join(HERE, f"sha_alt_town_{nm}.npz"), pop=p)
p = england.spread_totals(sig["lat"], sig["lon"], [(d[0], d[1]) for d in ds], dtot, var["k2"])
np.savez_compressed(os.path.join(HERE, "sha_alt_k2.npz"), pop=p)
