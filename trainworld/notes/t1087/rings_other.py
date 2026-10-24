"""Rings of a country's variant, plus which plan zone rows the variant keeps exactly.

    python rings_other.py <ISO> <variant> [<variant> ...]
"""
import collections
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.stdout.reconfigure(encoding="utf-8")
iso = sys.argv[1]
D = pickle.load(open(os.path.join(HERE, f"{iso}_data.pkl"), "rb"))
R = pickle.load(open(os.path.join(HERE, f"{iso}_out.pkl"), "rb"))
U = D["units"]
names = {r["id"]: r["name"] for r in D["rows"]}
status = {r["id"]: r["status"] for r in D["rows"]}
sreg = {r["id"]: r.get("solve_region") for r in D["rows"]}
for v in sys.argv[2:]:
    V = R[v]

    def nm(k):
        if v == "before" and k in names:
            return f"{names[k]} ({k})"
        if k in V["groups"]:
            return U[max(V["groups"][k], key=lambda i: U[i]["free"])]["name"]
        return names.get(k, k) + " (metro)"
    print(f"{iso} {v}: {len(V['rings'])} ring pairs; {V['res']}")
    for a, b, ih, bd in V["rings"]:
        ids = V["groups"][a]
        print(f"  {nm(a)} {sum(U[i]['free'] for i in ids) / 1e6:.2f}M {len(ids)} units wraps {nm(b)} [{ih:.2f}/{bd:.2f}]")
    if v != "before":
        sets = {frozenset(x) for x in V["groups"].values()}
        c = collections.Counter()
        regs = collections.Counter()
        for rid, ids in R["before"]["groups"].items():
            kept = frozenset(ids) in sets
            c[(status[rid], kept)] += 1
            if not kept:
                regs[sreg[rid]] += 1
        print("  plan zone rows kept (status, kept):", dict(c))
        print("  changed rows by solve region:", dict(regs))
