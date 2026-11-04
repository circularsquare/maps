"""T-562: nocar.FALLBACK, INCOME_H0 and WORLD_H0 from every country with a figure (notes/T-562.md).

    python notes/t562_fit.py

Households with a car against GDP per head (World Bank NY.GDP.PCAP.PP.KD, latest), one point per
country: nocar.national's figure for each country that has one by a register or survey rule, and
the US from ACS B08201 (households, not people); Hong Kong and Singapore left out of the fit.
Ordinary least squares on the logit.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipeline"))
import nocar  # noqa: E402

gdp, income, _ = nocar.worldbank()
cars, hh = nocar.eurostat()
have = set(nocar.COUNTRIES) | {c for c, v in nocar.dhs().items() if v[1] >= 2010} | (set(cars) & set(hh))
rows = [(c, float(nocar.national(c)[0])) for c in sorted(have)]
t = pd.read_csv(os.path.join(nocar.RAW, "acsdt5y2023-b08201.dat"), sep="|", dtype={"GEO_ID": str})
us = t[t.GEO_ID == "0100000US"].iloc[0]
rows.append(("US", float(us.B08201_E002 / us.B08201_E001)))
df = pd.DataFrame(rows, columns=["iso", "h0"])
df["gdp"] = df.iso.map(lambda c: gdp.get(c, (np.nan,))[0])
df["income"] = df.iso.map(income)
fit = df.dropna(subset=["gdp"])
fit = fit[~fit.iso.isin(["HK", "SG"])]  # city states, entered because the rule cannot hold for them
x, y = np.log(fit.gdp.to_numpy(float)), np.log((1 - fit.h0) / fit.h0).to_numpy(float)
(a, b), *_ = np.linalg.lstsq(np.c_[np.ones_like(x), x], y, rcond=None)
pred = 1 / (1 + np.exp(a + b * x))
res = fit.h0.to_numpy() - pred
print(f"FALLBACK = ({a:.2f}, {b:.3f}); {len(fit)} countries, rmse {np.sqrt(np.mean(res ** 2)):.3f}, "
      f"mean |error| {np.mean(np.abs(res)):.3f}, worst {fit.iso.iloc[np.argmax(np.abs(res))]} {res[np.argmax(np.abs(res))]:+.2f}")
for lo, hi in ((0, 5000), (5000, 15000), (15000, 40000), (40000, 1e9)):
    m = (fit.gdp >= lo).to_numpy() & (fit.gdp < hi).to_numpy()
    print(f"  GDP {lo:,}-{hi:,.0f}: {m.sum()} countries, rmse {np.sqrt(np.mean(res[m] ** 2)):.3f}")
print("INCOME_H0 =", {g: round(float(v), 3) for g, v in df.dropna(subset=["income"]).groupby("income").h0.median().items()})
print("WORLD_H0 =", round(float(df.h0.median()), 3))
print("no GDP:", sorted(set(df.iso) - set(fit.iso)))
