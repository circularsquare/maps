"""T-1140: how far gravity stands in for the ACS flows: identical zones between two US runs, the share
of counties whose zone-mates are mostly the same, and each county's top partner county by ACS against
gravity (among its land neighbours)."""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import common as C  # noqa: E402

R = C.load("us_runs.pkl")
P = C.load("us_prep.pkl")
a, b = sys.argv[1], sys.argv[2]
ga, gb = R[a]["USA"]["groups"], R[b]["USA"]["groups"]
sa, sb = {frozenset(v) for v in ga.values()}, {frozenset(v) for v in gb.values()}
print(f"{a}: {len(sa)} zones, {b}: {len(sb)}, identical {len(sa & sb)}")
oa = {i: frozenset(v) for v in ga.values() for i in v}
ob = {i: frozenset(v) for v in gb.values() for i in v}
D = C.load("us_data.pkl")
U = D["USA"]["units"]
tot = sum(U[i].free for i in oa)
same = sum(U[i].free for i in oa if i in ob and len(oa[i] & ob[i]) / len(oa[i] | ob[i]) >= 0.5)
print(f"people whose zone overlaps half or more: {same / tot:.0%}")
acs, grav, adj = P["USA"]["acs"], P["USA"]["grav"], P["USA"]["adj"]
n = hit = 0
for i in U:
    nb = list(adj[i])
    if not nb or not acs.get(i):
        continue
    ta = max(nb, key=lambda j: acs[i].get(j, 0))
    tg = max(nb, key=lambda j: grav.get(i, {}).get(j, 0))
    n += 1
    hit += ta == tg
print(f"counties whose top neighbour by ACS is gravity's too: {hit}/{n} = {hit / n:.0%}")
