"""List the ring pairs of a US variant with names, sizes and ACS commuting shares.

    python rings_us.py <variant> [statuses]
"""
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.stdout.reconfigure(encoding="utf-8")
D = pickle.load(open(os.path.join(HERE, "us_data.pkl"), "rb"))
R = pickle.load(open(os.path.join(HERE, "us_out.pkl"), "rb"))
U = D["units"]
v = sys.argv[1]
V = R[v]
names = {r["id"]: r["name"] for r in D["rows"]}
status = {r["id"]: r["status"] for r in D["rows"]}
names.update({k: p["name"] for k, p in D["packs"].items()})


def nm(k):
    if k in V["groups"] and k not in names:
        ids = V["groups"][k]
        return U[max(ids, key=lambda i: U[i]["free"])]["name"]
    return names.get(k, k)


print(f"{v}: {len(V['rings'])} ring pairs")
for a, b, ih, bd in sorted(V["rings"], key=lambda p: nm(p[0])):
    ids = V["groups"][a]
    ppl = sum(U[i]["free"] for i in ids)
    s = V["sc"][a]
    st = status.get(a, "") if v == "before" else ""
    print(f"  {a} {nm(a)} ({ppl / 1e6:.2f}M, {len(ids)} counties{', ' + st if st else ''}) wraps {nm(b)} "
          f"[{ih:.2f}/{bd:.2f}]; works inside {s['within']:.0%}, top metro {names.get(s['metro'], s['metro'])} {s['to_metro']:.0%}")
