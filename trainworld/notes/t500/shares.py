"""Office share of commuting residents and jobs for some Tokyo municipalities, and census
segment flows to the three central wards from them (table 10)."""
import sys
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
RAW = r"C:\Users\anita\projects\maps\trainworld\data\raw\t500"
OFF = set("GJKLR"); ALL = set("ABCDEFGHIJKLMNOPQRS")
t8 = pd.read_csv(RAW + r"\t8.csv", dtype={"code": str})
t8 = t8[t8.ind.isin(ALL)].assign(rc=lambda q: q.res - q.home, wc=lambda q: q.work - q.home)
t8["off"] = t8.ind.isin(OFF)
t10 = pd.read_csv(RAW + r"\t10.csv", dtype={"w": str, "h": str})
t10 = t10[t10.ind.isin(ALL)]
C3 = {"13101", "13102", "13103"}
to3 = t10[t10.w.isin(C3) & (t10.h != "home")]
AREAS = {"Chiyoda": "13101", "Chuo": "13102", "Minato": "13103", "Bunkyo": "13105", "Shinjuku": "13104", "Koto": "13108",
         "Setagaya": "13112", "Suginami": "13115", "Nerima": "13120", "Adachi": "13121", "Edogawa": "13123",
         "Musashino": "13203", "Mitaka": "13204", "Chofu": "13208", "Fuchu": "13206", "Tachikawa": "13202", "Hachioji": "13201",
         "Kawaguchi": "11203", "Urayasu": "12227", "Funabashi": "12204", "Kawasaki Nakahara": "14135", "Yokohama Aoba": "14117"}
print("area | office share of commuting residents | office commuters to 3 central / office commuters | other to 3 / other")
for nm, c in AREAS.items():
    r = t8[t8.code == c]
    o, a = r[r.off].rc.sum(), r.rc.sum()
    x = to3[to3.h == c]
    xo, xa = x[x.ind.isin(OFF)].n.sum(), x.n.sum()
    print(f"{nm} | {o / a:.0%} | {xo / o:.1%} | {(xa - xo) / (a - o):.1%}")
w = t8[t8.code.isin(C3)]
print("3 central wards: office share of commuting jobs", round(w[w.off].wc.sum() / w.wc.sum(), 3))
