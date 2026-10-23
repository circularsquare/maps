"""A built US pack's mean commute against LODES OD (JT00, home and work block both in the row's counties):
crow-fly between block internal points, and zone-centre as the game's gravity measures it.
lodes_mean.py <id> <od_main.csv.gz>"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json, sys
import numpy as np
import pandas as pd
sys.path.insert(0, r"C:\Users\anita\projects\maps\trainworld\pipeline")
import fit_decay as fd
from read_pack import load_pack

cid, odp = sys.argv[1], sys.argv[2]
plan = {r["id"]: r for r in json.load(open(r"C:\Users\anita\projects\maps\trainworld\pipeline\world_plan.json", encoding="utf-8"))["metros"] if r.get("id")}
counties = set(plan[cid]["units"]["USA"])
st = sorted({c[:2] for c in counties})
geo, lat, lon = [], [], []
for s in st:
    b = np.load(rf"C:\Users\anita\projects\maps\trainworld\data\work\usa\blocks_{s}.npz")
    geo.append(b["geoid"]); lat.append(b["lat"]); lon.append(b["lon"])
geo, lat, lon = np.concatenate(geo), np.concatenate(lat), np.concatenate(lon)
header, a = load_pack(cid)
z = fd.Zones(header, a)
bx, by = fd.local_xy(lat, lon, header["origin"]["lon"], header["origin"]["lat"])
bzone = z.of_xy(bx, by)
bi = pd.Series(np.arange(len(geo)), index=pd.Index(geo.astype(str)))
out = []
for ch in pd.read_csv(odp, usecols=["w_geocode", "h_geocode", "S000"], dtype={"w_geocode": str, "h_geocode": str}, chunksize=2_000_000):
    out.append(ch[ch["w_geocode"].str[:5].isin(counties) & ch["h_geocode"].str[:5].isin(counties)])
od = pd.concat(out, ignore_index=True)
hi, wi = od["h_geocode"].map(bi), od["w_geocode"].map(bi)
miss = hi.isna() | wi.isna()
od, hi, wi = od[~miss], hi[~miss].to_numpy(np.int64), wi[~miss].to_numpy(np.int64)
n = od["S000"].to_numpy(np.float64)
crow = np.hypot(bx[hi] - bx[wi], by[hi] - by[wi]) / 1000.0
zh, zw = bzone[hi], bzone[wi]
ok = (zh >= 0) & (zw >= 0)
dz = z.d[zh[ok], zw[ok]]
print(f"{cid}: LODES OD {n.sum():,.0f} jobs in {len(counties)} counties ({int(miss.sum())} pairs without a block); "
      f"mean crow-fly {(crow * n).sum() / n.sum():.2f} km, zone-centre {(dz * n[ok]).sum() / n[ok].sum():.2f} km; "
      f"within 10 / 25 / 50 km {n[crow < 10].sum() / n.sum():.1%} / {n[crow < 25].sum() / n.sum():.1%} / {n[crow < 50].sum() / n.sum():.1%}")
