"""T-500: census 2020 tables 8 and 10 (industry) to long CSVs in data/raw/t500.
t8.csv:  code, ind, res (employed residents), home (working at home), work (employed working there)
t10.csv: w (finest municipality of work), h (finest municipality of residence, or 'home'), ind, n"""
import sys, os
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
RAW = r"C:\Users\anita\projects\maps\trainworld\data\raw\t500"
FINEST = {"0", "2", "3"}
num = lambda s: pd.to_numeric(s.astype(str).str.replace(",", ""), errors="coerce").fillna(0.0)

x = pd.read_excel(os.path.join(RAW, "e08-01.xlsx"), header=None, dtype=str)
d = x.iloc[10:]
d = d[(d.iloc[:, 3] == "0_総数") & (d.iloc[:, 4] == "0_総数") & d.iloc[:, 0].isin(FINEST)]
t8 = pd.DataFrame({"code": d.iloc[:, 2].str[:5], "ind": d.iloc[:, 6].str.split("_").str[0],
                   "res": num(d.iloc[:, 7]), "home": num(d.iloc[:, 10]), "work": num(d.iloc[:, 19])})
t8.to_csv(os.path.join(RAW, "t8.csv"), index=False)
print("t8", len(t8), t8.code.nunique())

rows = []
for pref in ("08", "11", "12", "13", "14"):
    x = pd.read_excel(os.path.join(RAW, f"e10-{pref}.xlsx"), header=None, dtype=str)
    inds = [str(c).split("_")[0] for c in x.iloc[7, 8:].tolist()]
    d = x.iloc[10:]
    d = d[d.iloc[:, 0].astype(str).isin(FINEST)]
    w = d.iloc[:, 2].str[:5]; lv = d.iloc[:, 4].astype(str); hk = d.iloc[:, 5].astype(str); h = d.iloc[:, 7].str[:5]
    for j, ind in enumerate(inds):
        n = num(d.iloc[:, 8 + j])
        part = pd.DataFrame({"w": w, "lv": lv, "hk": hk, "h": h, "n": n, "ind": ind})
        a = part[part.lv.str.startswith("011_")].assign(h="home")
        b = part[part.lv.str.startswith("012_")].assign(h=lambda q: q.w)
        c = part[part.lv.str.startswith("02_") & part.hk.isin(FINEST)]
        rows.append(pd.concat([a, b, c])[["w", "h", "ind", "n"]])
    print(pref, d.iloc[:, 2].nunique(), "work areas", flush=True)
t10 = pd.concat(rows, ignore_index=True)
t10.to_csv(os.path.join(RAW, "t10.csv"), index=False)
print("t10", len(t10), t10.w.nunique(), "work units")
