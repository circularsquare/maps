"""T-1147: NHTS 2017 car-driver trips' door-to-door speed against the model's drive (kernel.rs
`drive_min_f`), by road distance band, for a few CBSAs. python -I nhts_speed.py <csv.zip> <out dir>"""
import os
import sys
import zipfile

import numpy as np
import pandas as pd

Z, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
cache = os.path.join(OUT, "nhts_car.pkl")
CBSA = {"35620": "New York", "26420": "Houston", "31080": "Los Angeles", "16980": "Chicago",
        "41860": "San Francisco", "47900": "Washington", "14460": "Boston", "37980": "Philadelphia"}
if not os.path.exists(cache):
    cols = ["TRVLCMIN", "TRPMILES", "TRPTRANS", "DRVR_FLG", "TDWKND", "HH_CBSA", "WTTRDFIN",
            "OTPPOPDN", "OTEEMPDN", "DTPPOPDN", "DTEEMPDN", "URBRUR"]
    with zipfile.ZipFile(Z) as z:
        df = pd.read_csv(z.open("trippub.csv"), usecols=cols, dtype=str)
    for c in ["TRVLCMIN", "TRPMILES", "WTTRDFIN", "OTPPOPDN", "OTEEMPDN", "DTPPOPDN", "DTEEMPDN"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[df.TRPTRANS.isin(["03", "04", "05", "06"]) & (df.DRVR_FLG == "01") & (df.TDWKND == "02")]
    df = df[(df.TRPMILES > 0) & (df.TRVLCMIN > 0) & (df.TRVLCMIN < 600)]
    df.to_pickle(cache)
df = pd.read_pickle(cache)
for c in ["OTPPOPDN", "OTEEMPDN", "DTPPOPDN", "DTEEMPDN"]:
    df.loc[df[c] < 0, c] = np.nan
df = df.dropna(subset=["OTPPOPDN", "OTEEMPDN", "DTPPOPDN", "DTEEMPDN"])
print("values:", sorted(df.OTPPOPDN.unique()), sorted(df.OTEEMPDN.unique()))

SQMI = 2.58999
WORKERS = 0.47  # resident workers per person (the kernel's zone `pop` counts commuters)


def dens(pop, emp):
    return (pop * WORKERS + emp) / SQMI


def place(d, s):
    f = 1.0 / (1.0 + d / s["dens50"])
    c = s["city"]
    return 60.0 / (c + (s["town"] - c) * f), 60.0 / (c + (s["far"] - c) * f)


def drive(road, a, b, s):
    e = np.minimum(0.5 * road, 2.0)
    main = road - 2 * e
    k = np.minimum(0.5 * main, 8.0)
    return e * (a[0] + b[0]) + k * (a[1] + b[1]) + (main - 2 * k) * 60.0 / s["open"]


BASE = dict(open=110.0, city=16.5, town=44.0, far=88.0, dens50=5000.0)
VARS = {
    "base": BASE,
    "x1.1": {k: (v * 1.1 if k != "dens50" else v) for k, v in BASE.items()},
    "x1.2": {k: (v * 1.2 if k != "dens50" else v) for k, v in BASE.items()},
    "x1.3": {k: (v * 1.3 if k != "dens50" else v) for k, v in BASE.items()},
    "main+open x1.2": dict(BASE, far=88 * 1.2, open=110 * 1.2),
    "main x1.3, open 120": dict(BASE, far=88 * 1.3, open=120.0),
    "town+far x1.2": dict(BASE, far=88 * 1.2, town=44 * 1.2),
}
ROUTED = 1.3 / 1.210  # the kernel's road_routed_scale: the model's road over the routed road

df["km"] = df.TRPMILES * 1.609344
od = dens(df.OTPPOPDN.values, df.OTEEMPDN.values)
dd = dens(df.DTPPOPDN.values, df.DTEEMPDN.values)
bands = [0, 2, 5, 10, 20, 40, 80, 1e9]
df["band"] = pd.cut(df.km, bands, right=False, labels=["<2", "2-5", "5-10", "10-20", "20-40", "40-80", "80+"])
for name, s in VARS.items():
    df["m_" + name] = drive(df.km.values * ROUTED, place(od, s), place(dd, s), s)

rows = []
for geo, sub in [("US", df)] + [(n, df[df.HH_CBSA == k]) for k, n in CBSA.items()]:
    for b, g in sub.groupby("band", observed=True):
        w = g.WTTRDFIN
        r = {"geo": geo, "band": str(b), "n": len(g), "share": w.sum() / sub.WTTRDFIN.sum(),
             "km": np.average(g.km, weights=w), "nhts_min": np.average(g.TRVLCMIN, weights=w),
             "nhts_kmh": (w * g.km).sum() / (w * g.TRVLCMIN).sum() * 60}
        for name in VARS:
            m = np.average(g["m_" + name], weights=w)
            r[name] = m
        rows.append(r)
    w = sub.WTTRDFIN
    r = {"geo": geo, "band": "all", "n": len(sub), "share": 1.0, "km": np.average(sub.km, weights=w),
         "nhts_min": np.average(sub.TRVLCMIN, weights=w), "nhts_kmh": (w * sub.km).sum() / (w * sub.TRVLCMIN).sum() * 60}
    for name in VARS:
        r[name] = np.average(sub["m_" + name], weights=w)
    rows.append(r)
t = pd.DataFrame(rows)
t.to_csv(os.path.join(OUT, "nhts_speed.csv"), index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 200)
print(t.round(2).to_string(index=False))
