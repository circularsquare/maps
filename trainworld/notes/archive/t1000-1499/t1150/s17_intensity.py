"""T-1150: a jobs signal from built intensity in place of GHSL's residential/non-residential split:
each cell's total built volume x (total volume a km2 within 1 km / 1 Mm3)^p, so the densest built
places (CBDs) draw more jobs per m3 whatever GHSL calls them. Scored on Shanghai's district shares
and Lujiazui as s16."""
import sys
sys.stdout.reconfigure(encoding="utf-8")
from t1150lib import *
import tier0, england, china
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sig = load_sig("sha")
ds = sha_units(); units = [(d[0], d[1]) for d in ds]
idx = np.load(os.path.join(HERE, "sha_idx.npy"))
pop = np.load(os.path.join(HERE, "sha_pop.npy"))
totv = np.load(os.path.join(HERE, "sha_totv_cells.npy"))
J = china.JOBS["sha"]; names = [d[0] for d in ds]
cen = np.array([J[n] for n in names]) / sum(J.values())
c7 = [names.index(n) for n in ("黄浦区", "徐汇区", "长宁区", "静安区", "普陀区", "虹口区", "杨浦区")]
osm = sig["osm_errand"] + sig["osm_leisure"] + sig["osm_job"] + sig["osm_land"]
lat0 = 31.23
xy = np.c_[np.radians(sig["lon"]) * np.cos(np.radians(lat0)), np.radians(sig["lat"])] * 6371.0
tree = cKDTree(xy)
nb = tree.query_ball_point(xy, 1.0)
dens = np.array([totv[n].sum() for n in nb]) / np.pi / 1e6   # Mm3 a km2 within 1 km
print(f"built intensity (Mm3 a km2 within 1 km): median {np.median(dens[totv > 0]):.2f}, q99 {np.quantile(dens, .99):.2f}, max {dens.max():.2f}")
spots = {"Lujiazui": (121.5055, 31.2385), "People's Sq": (121.4737, 31.2304), "Baosteel": (121.44, 31.41)}
near = lambda lo, la: 6371 * np.hypot(np.radians(sig["lon"] - lo) * np.cos(np.radians(la)), np.radians(sig["lat"] - la)) < 1
for nm, (lo, la) in spots.items():
    print(f"  {nm}: intensity {dens[near(lo, la)].max():.2f}")
nr = tier0.nres_signal(sig)
for p in (0.0, 1.0, 2.0):
    bi = totv * np.power(np.maximum(dens, 1e-6), p)
    for mix in (1.0, 0.5):
        b = mix * bi / bi.sum() + (1 - mix) * nr / nr.sum()
        for wn in (0.3, 0.5):
            w = wn * b + (0.8 - wn) * osm / osm.sum() + 0.2 * pop / pop.sum()
            sh = np.bincount(idx[idx >= 0], weights=w[idx >= 0], minlength=len(ds)); sh /= sh.sum()
            jobs = england.spread_totals(sig["lat"], sig["lon"], units, J, w)
            sp = " ".join(f"{k} {jobs[near(*v)].sum() / 1e3:,.0f}k" for k, v in spots.items())
            print(f"p {p:g}, intensity share of the building part {mix}, building weight {wn}: districts misplaced "
                  f"{0.5 * np.abs(sh - cen).sum():.1%}, central 7 {sh[c7].sum():.1%}; densest cell {jobs.max():,.0f}; 1 km: {sp}")
np.save(os.path.join(HERE, "sha_intensity.npy"), dens)
