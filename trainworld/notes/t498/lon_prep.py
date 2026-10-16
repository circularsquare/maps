"""T-498: London's pack zones and the 2011 census flows (WU01EW) by district, in fit.py's format."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
import json, sys
import numpy as np
from shapely import make_valid, union_all
from shapely.geometry import shape
import h3

TW = r"C:\Users\anita\projects\maps\trainworld"
sys.path.insert(0, os.path.join(TW, "pipeline"))
import england  # noqa
from read_pack import load_pack  # noqa

OUT = sys.argv[1]
header, a = load_pack("lon", os.path.join(TW, "data", "work", "res9"))
g = header["gravity"]
with open(england.raw("lon_itl3.geojson"), encoding="utf-8") as f:
    boundary = union_all([make_valid(shape(ft["geometry"])) for ft in json.load(f)["features"]])
with open(england.raw("lad_dec2023_uk_bgc.geojson"), encoding="utf-8") as f:
    lads = [(ft["properties"]["LAD23CD"], ft["properties"]["LAD23NM"], make_valid(shape(ft["geometry"]))) for ft in json.load(f)["features"]]
lads = [x for x in lads if boundary.contains(x[2].representative_point())]
nl = len(lads)
lat, lon = np.array([h3.cell_to_latlng(h3.int_to_str(int(c))) for c in a["h3"]]).T
li, _ = england.cell_units(lat, lon, [(x[0], x[2]) for x in lads])
print("cells outside every district:", int((li < 0).sum()), "of", len(li))
pts = england.msoa11_points("lon", boundary)
fl = england.wu01ew()
fl = fl[fl.h.isin(pts) & fl.w.isin(pts)]
codes = sorted(pts)
mll = np.array([pts[c] for c in codes])
ml, _ = england.cell_units(mll[:, 0], mll[:, 1], [(x[0], x[2]) for x in lads])
mci = {c: ml[i] for i, c in enumerate(codes)}
h_i, w_i = fl.h.map(mci).to_numpy(), fl.w.map(mci).to_numpy()
ok = (h_i >= 0) & (w_i >= 0)
Cl = np.zeros((nl, nl)); np.add.at(Cl, (h_i[ok], w_i[ok]), fl.n.to_numpy(float)[ok])
inside = li >= 0
nz = header["zones"]
zone = a["zone"].astype(np.int64)
ch, cw = a["commute_home"].astype(np.float64), a["commute_work"].astype(np.float64)
P = np.zeros((nz, nl)); Q = np.zeros((nz, nl))
np.add.at(P, (zone[inside], li[inside]), ch[inside]); np.add.at(Q, (zone[inside], li[inside]), cw[inside])
# zones' full margins (cells outside every district still commute; they are left out of the scores)
H = np.bincount(zone, ch, nz); J = np.bincount(zone, cw, nz)
Pfull = P.copy(); Qfull = Q.copy()
np.savez(OUT, P=P, Q=Q, H=H, J=J, gx=a["zone_gx"].astype(np.float64), gy=a["zone_gy"].astype(np.float64), Cl=Cl,
         codes=np.array([x[0] for x in lads]), names=np.array([x[1] for x in lads]), li=np.where(li < 0, 0, li), zone=zone,
         x=a["x_m"].astype(np.float64), y=a["y_m"].astype(np.float64), ch=ch, cw=cw, g=json.dumps(g))
print("saved", nl, "districts", nz, "zones; census commuters", round(Cl.sum()), "own district", round(np.trace(Cl) / Cl.sum(), 3))
