"""T-1140 probe: for a run's zones under the floor, their flows (ACS, both ways) to each area, the
share of all their travel out of the zone, and the neighbours' shares. Read-only.

    python probe.py <variant> [names...]
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import common as C  # noqa: E402

D = C.load("us_data.pkl")
P = C.load("us_prep.pkl")
RUNS = C.load("us_runs.pkl")
tag = sys.argv[1]
want = sys.argv[2:]
d = D["USA"]
U = d["units"]
gr = RUNS[tag]["USA"]["groups"]
F = P["USA"]["acs"]
adj = P["USA"]["adj"]
own = {i: k for k, ids in gr.items() for i in ids}
fx = {i: f["id"] for f in d["fixed"] for i in f["units"]}
fname = {f["id"]: f["name"] for f in d["fixed"]}
for i, f in fx.items():
    own.setdefault(i, f)


def nm(a):
    return U[max(gr[a], key=lambda i: U[i].free)].name if a in gr else fname.get(a, a)


for k, ids in sorted(gr.items(), key=lambda kv: sum(U[i].free for i in kv[1])):
    n = sum(U[i].free for i in ids)
    name = nm(k)
    if want and not any(w in name for w in want):
        continue
    if not want and n >= 500_000:
        continue
    to = C.area_flows(F, ids, own)
    tot = sum(to.values())
    nb = {own.get(j) for i in ids for j in adj[i]} - {k, None}
    top = to.most_common(4)
    print(f"{name} {n / 1e6:.2f}M: out {tot:,.0f}; top " + ", ".join(f"{nm(a)}{'*' if a in fx.values() else ''} {v / tot:.0%}" for a, v in top)
          + " | zone nbrs " + ", ".join(f"{nm(a)} {to.get(a, 0) / tot:.0%} {sum(U[i].free for i in gr[a]) / 1e6:.2f}M" for a in nb if a in gr))
