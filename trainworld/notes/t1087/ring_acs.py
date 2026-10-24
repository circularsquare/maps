"""ACS commuting of the plan's US ring zones round metros: all workers and those leaving their county."""
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.stdout.reconfigure(encoding="utf-8")
D = pickle.load(open(os.path.join(HERE, "us_data.pkl"), "rb"))
R = pickle.load(open(os.path.join(HERE, "us_out.pkl"), "rb"))
U = D["units"]
B = R["before"]
names = {r["id"]: r["name"] for r in D["rows"]}
metros = {r["id"] for r in D["rows"] if r["kind"] != "zone"} | set(D["packs"])
tot = {"within": 0, "metro": 0, "workers": 0, "xw": 0, "xm": 0, "xt": 0}
for a, b, ih, bd in B["rings"]:
    if b not in metros:
        continue
    s = B["sc"][a]
    w = s["workers"]
    xt = sum(n for h in B["groups"][a] for t, n in D["flows"].get(h, {}).items() if t != h)
    print(f"{names[a]:24s} round {names.get(b, b):14s}: of all workers inside {s['within']:.0%}, to the metro {s['to_metro']:.0%}; "
          f"of those leaving their county inside {s['x_within']:.0%}, to the metro {s['x_metro']:.0%}")
    tot["workers"] += w
    tot["within"] += w * s["within"]
    tot["metro"] += w * s["to_metro"]
    tot["xt"] += xt
    tot["xw"] += xt * s["x_within"]
    tot["xm"] += xt * s["x_metro"]
print(f"all: inside {tot['within'] / tot['workers']:.0%}, to the metro {tot['metro'] / tot['workers']:.0%}; "
      f"leaving their county: inside {tot['xw'] / tot['xt']:.0%}, to the metro {tot['xm'] / tot['xt']:.0%}")
