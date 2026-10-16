"""T-551: New York's real trips for comparison (notes/T-551.md), from NHTS 2017 microdata on disk.

New York-Newark-Jersey City CBSA households, weekday legs, weighted (WTTRDFIN): mode shares by
straight-line band (road miles x 1.609 / 1.3, the game's road detour) for work legs and the rest,
and car legs' door-to-door speed (reported minutes, which include parking and the walk to the car)
by band, on road km and on straight km.

    venv\\Scripts\\python.exe trainworld/notes/t551_nhts.py > ../artifacts/trainworld/T-551-nhts.txt
"""
import zipfile

import numpy as np
import pandas as pd

Z = r"C:\Users\anita\projects\maps\trainworld\data\raw\nhts2017\csv.zip"
zf = zipfile.ZipFile(Z)
with zf.open("trippub.csv") as f:
    t = pd.read_csv(f, usecols=["WTTRDFIN", "TRAVDAY", "WHYFROM", "WHYTO", "TRPMILES", "TRVLCMIN",
                                "HH_CBSA", "TRPTRANS", "DRVR_FLG"])
t = t[t.TRAVDAY.between(2, 6) & (t.HH_CBSA.astype(str) == "35620") & (t.TRPMILES >= 0) & (t.TRVLCMIN > 0)].copy()
t["road_km"] = t.TRPMILES * 1.609
t["sl_km"] = t.road_km / 1.3
edges = [0, 2, 5, 10, 20, 50, 1e9]
names = ["0-2", "2-5", "5-10", "10-20", "20-50", "50+"]
t["band"] = pd.cut(t.sl_km, edges, right=False, labels=names)
work = np.isin(t.WHYTO, [3, 4, 5]) | np.isin(t.WHYFROM, [3, 4, 5])
t["group"] = np.where(work, "work", "everyday")
mode = np.full(len(t), "other", dtype=object)
mode[t.TRPTRANS == 1] = "walk"
mode[t.TRPTRANS.isin([15, 16])] = "rail"
mode[t.TRPTRANS.isin([3, 4, 5, 6, 18])] = "car"
mode[t.TRPTRANS.isin([10, 11, 12, 13, 14])] = "bus"
mode[t.TRPTRANS == 17] = "taxi"
mode[t.TRPTRANS == 2] = "bike"
t["mode"] = mode

print("NHTS 2017, New York CBSA households, weekday legs (weighted). Bands are straight km = road km / 1.3.")
for g in ["work", "everyday", None]:
    d = t if g is None else t[t.group == g]
    print(f"\n== {g or 'all'}: {len(d)} legs")
    tab = d.pivot_table(index="band", columns="mode", values="WTTRDFIN", aggfunc="sum", observed=False).fillna(0)
    tab.loc["all"] = tab.sum()
    share = tab.div(tab.sum(axis=1), axis=0)
    share["legs %"] = tab.sum(axis=1) / tab.sum(axis=1).iloc[:-1].sum()
    share["rail of non-walk"] = tab["rail"] / (tab.sum(axis=1) - tab["walk"])
    print((share * 100).round(1).to_string())

print("\n== car legs (car, SUV, van, pickup, rental; drivers and passengers): door-to-door speed")
c = t[t["mode"] == "car"]
rows = []
for b, d in c.groupby("band", observed=False):
    w = d.WTTRDFIN
    mins = (d.TRVLCMIN * w).sum() / w.sum()
    rows.append((b, len(d), (d.road_km * w).sum() / w.sum(), (d.sl_km * w).sum() / w.sum(), mins,
                 (d.road_km * w).sum() / (d.TRVLCMIN * w).sum() * 60, (d.sl_km * w).sum() / (d.TRVLCMIN * w).sum() * 60,
                 np.average(d.TRVLCMIN, weights=w)))
print(f"{'band':6s} {'legs':>6s} {'road km':>8s} {'str km':>7s} {'minutes':>8s} {'road km/h':>9s} {'str km/h':>8s}")
for r in rows:
    print(f"{r[0]:6s} {r[1]:6d} {r[2]:8.1f} {r[3]:7.1f} {r[4]:8.1f} {r[5]:9.1f} {r[6]:8.1f}")
