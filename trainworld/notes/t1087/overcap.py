"""Zones over the cap in a variant, with their biggest units and the merge log lines that made them."""
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.stdout.reconfigure(encoding="utf-8")
iso, v, cap = sys.argv[1], sys.argv[2], float(sys.argv[3])
D = pickle.load(open(os.path.join(HERE, f"{iso}_data.pkl"), "rb"))
R = pickle.load(open(os.path.join(HERE, f"{iso}_out.pkl"), "rb"))
U = D["units"]
V = R[v]
for k, ids in sorted(V["groups"].items(), key=lambda kv: -sum(U[i]["free"] for i in kv[1])):
    p = sum(U[i]["free"] for i in ids)
    if p <= cap:
        break
    big = sorted(ids, key=lambda i: -U[i]["free"])[:3]
    print(f"{p / 1e6:.2f}M {len(ids)} units {U[k]['state']}: " + ", ".join(f"{U[i]['name']} {U[i]['free'] / 1e6:.2f}" for i in big))
    for ln in V["lines"]:
        if f"joins {k} " in ln:
            print("   ", ln)
