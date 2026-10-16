"""T-465 prep: the shipped tyo gravity (res9 copy) and the census table 4 flows, both at municipality
level; cached to prep.npz for the curve trials."""
import os
os.environ["OMP_NUM_THREADS"] = "2"
import json, sys
import numpy as np
import h3

HERE = os.path.dirname(os.path.abspath(__file__))
TW = r"C:\Users\anita\projects\maps\trainworld"
sys.path.insert(0, os.path.join(TW, "pipeline"))
import japan, england  # noqa
from read_pack import load_pack  # noqa

PACKDIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(TW, "data", "work", "res9")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(TW, "..", "artifacts", "trainworld", "T-465-prep.npz")
prefs = ["08", "11", "12", "13", "14"]

header, a = load_pack("tyo", PACKDIR)
g = header["gravity"]
print({k: v for k, v in g.items() if not isinstance(v, (list, dict))})
lat, lon = np.array([h3.cell_to_latlng(h3.int_to_str(int(c))) if hasattr(h3, "int_to_str") else h3.h3_to_geo(hex(int(c))[2:]) for c in a["h3"]]).T
ms = japan.municipalities(prefs)
codes = [m[0] for m in ms]
names = [m[1] for m in ms]
li, near = england.cell_units(lat, lon, [(m[0], m[3]) for m in ms])
print("cells outside every municipality:", int((li < 0).sum()))
# outside cells: nearest-by-position is overkill; give them to the municipality of the nearest inside cell
if (li < 0).any():
    from scipy.spatial import cKDTree
    ok = li >= 0
    t = cKDTree(np.c_[a["x_m"][ok], a["y_m"][ok]])
    _, j = t.query(np.c_[a["x_m"][~ok], a["y_m"][~ok]])
    li[~ok] = li[ok][j]
nl, nz = len(ms), header["zones"]
zone = a["zone"].astype(np.int64)
ch, cw = a["commute_home"].astype(np.float64), a["commute_work"].astype(np.float64)
P = np.zeros((nz, nl)); Q = np.zeros((nz, nl))
np.add.at(P, (zone, li), ch); np.add.at(Q, (zone, li), cw)
H, J = P.sum(1), Q.sum(1)
fl = japan.flows(prefs)
allw = fl[fl.h == "all"].groupby("w").n.sum()
homew = fl[fl.h == "home"].groupby("w").n.sum()
ci = {c: i for i, c in enumerate(codes)}
f2 = fl[fl.h.isin(ci) & fl.w.isin(ci)]
Cl = np.zeros((nl, nl))
np.add.at(Cl, (f2.h.map(ci).to_numpy(), f2.w.map(ci).to_numpy()), f2.n.to_numpy(float))
census_all = np.array([allw.get(c, 0.0) for c in codes])
census_home = np.array([homew.get(c, 0.0) for c in codes])
pack_jobs = np.bincount(li, weights=a["jobs"].astype(np.float64), minlength=nl)
pack_pop = np.bincount(li, weights=a["pop"].astype(np.float64), minlength=nl)
np.savez(OUT, P=P, Q=Q, H=H, J=J, gx=a["zone_gx"].astype(np.float64), gy=a["zone_gy"].astype(np.float64),
         row=a["zone_row"].astype(np.float64), col=a["zone_col"].astype(np.float64), Cl=Cl,
         census_all=census_all, census_home=census_home, pack_jobs=pack_jobs, pack_pop=pack_pop,
         codes=np.array(codes), names=np.array(names), li=li, zone=zone,
         x=a["x_m"].astype(np.float64), y=a["y_m"].astype(np.float64), ch=ch, cw=cw,
         g=json.dumps(g))
print("saved", OUT, nl, "municipalities", nz, "zones")
