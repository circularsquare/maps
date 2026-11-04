"""T-1508: the market's longer trips by country and band, from t1508_dump's json.

    python t1508_bands.py <dump.json>
Trips are one-way trips a weekday by the origin's country (domestic only: both ends in it), pkm on
the straight line between place centres. Year = weekday x 425 (the longer trips' weekend 1.58).
"""
import json, sys, collections

d = json.load(open(sys.argv[1]))
P = d["places"]
BANDS = [100, 300, 700, 1500, 1e9]
YEAR = 425
agg = collections.defaultdict(lambda: [[0.] * 4 for _ in BANDS[:-1]])  # trips ground, tail, pkm ground, tail
abroad = collections.defaultdict(float)
pair = collections.defaultdict(float)
for i, j, km, g, t in d["pairs"]:
    ci, cj = P[i]["country"], P[j]["country"]
    if ci != cj:
        abroad[ci] += g + t
        continue
    b = max(k for k in range(len(BANDS) - 1) if km >= BANDS[k])
    a = agg[ci][b]
    a[0] += g; a[1] += t; a[2] += g * km; a[3] += t * km
    pair[(min(i, j), max(i, j))] += g + t
pop = collections.defaultdict(float)
for p in P:
    if p["has"]:
        pop[p["country"]] += p["pop"]
print("| Country | people | 100-300 km | 300-700 | 700-1500 | 1500+ | trips >=100 km a year | of them tail | pkm >=100 km a year (bn) | per head a year |")
print("|---|---|---|---|---|---|---|---|---|---|")
for c in sorted(agg, key=lambda c: -pop[c])[:16]:
    A = agg[c]
    tr = [a[0] + a[1] for a in A]
    tot = sum(tr)
    tail = sum(a[1] for a in A)
    pkm = sum(a[2] + a[3] for a in A)
    print(f"| {c} | {pop[c]/1e6:.0f}M | " + " | ".join(f"{x*YEAR/1e6:.0f}M" for x in tr) + f" | {tot*YEAR/1e6:.0f}M | {100*tail/tot:.0f}% | {pkm*YEAR/1e9:.0f} | {tot*YEAR/pop[c]:.2f} |")
names = {k: p["zone"] for k, p in enumerate(P)}
want = {("sha", "bei"), ("bei", "gua"), ("bei", "wuh"), ("gua", "she"), ("tyo", "osa"), ("tyo", "fuk"), ("nyc", "wdc"), ("nyc", "bos"), ("par", "lyo"), ("par", "mar"), ("mad", "bar"), ("ber", "ham"), ("ber", "muc")}
print()
for c in ("CN", "JP", "KR", "DE", "FR", "US", "GB", "ES", "IT"):
    top = sorted(((v, k) for k, v in pair.items() if P[k[0]]["country"] == c), reverse=True)[:6]
    print(c, "; ".join(f"{names[a]}-{names[b]} {v/1e3:.1f}k" for v, (a, b) in top))
print()
idx = {p["zone"]: k for k, p in enumerate(P)}
for a, b in sorted(want):
    if a in idx and b in idx:
        k = (min(idx[a], idx[b]), max(idx[a], idx[b]))
        print(f"{a}-{b}: {pair.get(k, 0)/1e3:.1f}k a weekday both ways")
