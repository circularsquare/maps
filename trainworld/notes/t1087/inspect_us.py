"""A US variant's rings, zones under the floor and over the cap, and what changes for built rows.

    python inspect_us.py <variant>
"""
import collections
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
names.update({k: p["name"] for k, p in D["packs"].items()})


def zname(ids):
    return U[max(ids, key=lambda i: U[i]["free"])]["name"]


def nm(k):
    if k in V["groups"]:
        return zname(V["groups"][k])
    return names.get(k, k)


print(f"{v}: rings")
for a, b, ih, bd in V["rings"]:
    ids = V["groups"][a]
    s = V["sc"][a]
    print(f"  {nm(a)} ({sum(U[i]['free'] for i in ids) / 1e6:.2f}M, {len(ids)}) wraps {nm(b)} [{ih:.2f}/{bd:.2f}] "
          f"inside {s['within']:.0%}, cross-county inside {s['x_within']:.0%}")
print("under 0.5M:")
for k, ids in V["groups"].items():
    p = sum(U[i]["free"] for i in ids)
    if p < 500_000:
        print(f"  {zname(ids)} {p / 1e6:.2f}M {len(ids)} counties, {U[ids[0]]['state']}")
print("over 1.5M:")
for k, ids in V["groups"].items():
    p = sum(U[i]["free"] for i in ids)
    if p > 1_500_000:
        print(f"  {zname(ids)} {p / 1e6:.2f}M {len(ids)} counties, {U[ids[0]]['state']}")
# built rows kept exactly
before = R["before"]["groups"]
status = {r["id"]: r["status"] for r in D["rows"]}
after_sets = {frozenset(ids) for ids in V["groups"].values()}
c = collections.Counter()
for rid, ids in before.items():
    c[(status[rid], frozenset(ids) in after_sets)] += 1
print("plan zone rows kept exactly (status, kept): ", dict(c))
