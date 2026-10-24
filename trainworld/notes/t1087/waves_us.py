"""Per US wave: zones, rings, rows kept, cross-county commuters kept inside, before and after.

    python waves_us.py <variant> [<variant> ...]
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
FL = D["flows"]
W = {"pilot": "13 12", "1 South": "48 40 05 22 28 01 47 45 37",
     "2 Midwest": "39 18 17 26 55 27 19 29 20 31 46 38 21", "3 West": "06 41 53 32 04 49 08 35 16 30 56 02 15"}
wave_of = {}
for w, s in W.items():
    for st in s.split():
        wave_of[st] = w


def wv(ids):
    c = collections.Counter(wave_of.get(U[i]["state"], "4 Northeast") for i in ids)
    return c.most_common(1)[0][0]


before = R["before"]["groups"]
bsets = {frozenset(v) for v in before.values()}
for v in ["before"] + sys.argv[1:]:
    V = R[v]
    owner = {i: k for k, ids in V["groups"].items() for i in ids}
    rz = {a for a, *_ in V["rings"]}
    agg = collections.defaultdict(lambda: collections.Counter())
    for k, ids in V["groups"].items():
        w = wv(ids)
        a = agg[w]
        a["zones"] += 1
        a["rings"] += k in rz
        a["kept"] += frozenset(ids) in bsets
        for h in ids:
            for t, n in FL.get(h, {}).items():
                if t != h:
                    a["xt"] += n
                    if owner.get(t) == k:
                        a["xw"] += n
    print(v)
    for w in list(W) + ["4 Northeast"]:
        a = agg[w]
        print(f"  {w}: {a['zones']} zones, {a['rings']} rings, {a['kept']} same as the plan, "
              f"cross-county commuters kept {a['xw'] / max(a['xt'], 1):.0%}")
