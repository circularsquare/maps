"""Sao Paulo OD 2017, the game's outing rates by the legs convention (T-509 stage 4).

A leg's purpose is its activity end: MOTIVO_D, or MOTIVO_O when the leg goes home (8). Outings =
legs / 2 per resident (all ages, F_PESS == 1 weights). Distances straight line (DISTANCIA, m).
"""
import glob
import os

import numpy as np
import pyogrio

BASE = r"C:\Users\anita\projects\maps\trainworld\data\raw\sao\od\OD-2017"
import sys
sys.path.insert(0, r"C:\Users\anita\projects\maps\trainworld\pipeline")
import brazil  # noqa: E402
brazil._survey_dir("sao")  # extracts the microdata from the kept zip
G = {"errands": [5, 10], "social": [7, 11], "medical": [6], "school": [4], "work": [1, 2, 3], "job search": [9]}
df = pyogrio.read_dataframe(glob.glob(os.path.join(BASE, "Banco*", "OD_2017_v1.dbf"))[0], read_geometry=False,
                            columns=["F_PESS", "FE_PESS", "FE_VIA", "MOTIVO_O", "MOTIVO_D", "DISTANCIA", "TIPVG", "MODOPRIN"])
people = float(df.FE_PESS[df.F_PESS == 1].sum())
t = df[df.FE_VIA.notna() & df.MOTIVO_D.notna()].copy()
act = np.where(t.MOTIVO_D == 8, t.MOTIVO_O, t.MOTIVO_D)
print(f"people {people:,.0f}; legs {t.FE_VIA.sum():,.0f} ({t.FE_VIA.sum() / people:.3f} a resident)")
for name, codes in G.items():
    m = np.isin(act, codes)
    s = t[m]
    w = s.FE_VIA.to_numpy()
    d = s.DISTANCIA.to_numpy() / 1000.0
    ok = d <= 46
    rail = w[np.isin(s.MODOPRIN.to_numpy(), [12, 13])].sum() / w.sum()  # 12 metro, 13 train (OD 2017 codes)
    print(f"{name:10s} legs {w.sum() / people:.3f} outings {w.sum() / people / 2:.3f}  mean km (<=46) {np.average(d[ok], weights=w[ok]):.2f}"
          f"  median {np.interp(0.5, np.cumsum(w[ok][np.argsort(d[ok])]) / w[ok].sum(), np.sort(d[ok])):.2f}"
          f"  walk {w[(s.TIPVG == 3).to_numpy()].sum() / w.sum():.1%} rail {rail:.1%}")
