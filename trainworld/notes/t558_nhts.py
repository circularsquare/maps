"""T-558: NHTS 2017 New York CBSA walk share by fine straight-line band, work legs and the rest."""
import zipfile

import numpy as np
import pandas as pd

Z = r"C:\Users\anita\projects\maps\trainworld\data\raw\nhts2017\csv.zip"
zf = zipfile.ZipFile(Z)
with zf.open("trippub.csv") as f:
    t = pd.read_csv(f, usecols=["WTTRDFIN", "TRAVDAY", "WHYFROM", "WHYTO", "TRPMILES", "TRVLCMIN",
                                "HH_CBSA", "TRPTRANS"])
t = t[t.TRAVDAY.between(2, 6) & (t.HH_CBSA.astype(str) == "35620") & (t.TRPMILES >= 0) & (t.TRVLCMIN > 0)].copy()
t["sl_km"] = t.TRPMILES * 1.609 / 1.3
edges = [0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 1e9]
t["band"] = pd.cut(t.sl_km, edges, right=False)
work = np.isin(t.WHYTO, [3, 4, 5]) | np.isin(t.WHYFROM, [3, 4, 5])
home = (t.WHYTO.isin([1, 2])) | (t.WHYFROM.isin([1, 2]))
t["group"] = np.where(work, "work", "everyday")
t["walk"] = (t.TRPTRANS == 1).astype(float)
t["rail"] = t.TRPTRANS.isin([15, 16]).astype(float)
for g in ["work", "everyday"]:
    for hb in [False, True]:
        d = t[(t.group == g) & (home if hb else True)]
        print(f"== {g}{' home-based' if hb else ''}: {len(d)} legs")
        for b, x in d.groupby("band", observed=False):
            w = x.WTTRDFIN
            print(f"{str(b):14s} legs {len(x):5d} share {w.sum() / d.WTTRDFIN.sum() * 100:5.1f}%  walk {np.average(x.walk, weights=w) * 100:5.1f}%  rail {np.average(x.rail, weights=w) * 100:5.1f}%")
