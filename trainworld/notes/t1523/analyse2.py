"""T-1523: level-wide alternatives to the dot, and how midday compares with the peaks on every line."""
import json
import sys

H = [4, 6, 4, 4, 6]
LEVEL = [0, 1, 0, 1, 2]
d = json.load(open(sys.argv[1]))
peak = d["peak"]

def lvl(r, L, i):
    return sum(r["per"][p]["acct"][i] for p in range(5) if LEVEL[p] == L)

n = mid_arr = mid_seg = mid_worst = 0
dots = {"busy": [0, 0], "all": [0, 0], "lost": [0, 0], "worstp": [0, 0]}
same_n = 0
for r in d["rows"]:
    if not r["per"][0].get("acct"):
        continue
    P = r["per"]
    if lvl(r, 0, 2) <= 0:
        continue
    n += 1
    a = [P[p]["acct"][2] / H[p] for p in range(5)]
    s = [max(P[p].get("maxSeg", [0, 0])) / H[p] * peak[p] for p in range(5)]
    mid_arr += a[1] >= a[0]
    mid_seg += s[1] >= s[0]
    mid_worst += s[1] >= max(s[0], s[2])
    t = r["tph"]
    if t["high"] > 0 and abs(t["high"] - t["medium"]) <= 0.02 * t["high"]:
        same_n += 1
        for L in (0, 1):
            pass

print(f"{n} lines with riders")
print(f"midday riders per hour >= morning peak's: {mid_arr}")
print(f"midday busiest-hour load on the busiest stretch >= morning peak's: {mid_seg}")
print(f"... >= both peaks': {mid_worst}")

print("\nper line, per level: dot (busy platforms), all-rider wait, share turned away, share left behind, worst period's dot")
for r in d["rows"]:
    if r["letter"] not in ("NJCL", "BNTNM", "J", "R", "H", "P", "S") or not r["per"][0].get("acct"):
        continue
    out = []
    for L, nm in ((0, "High"), (1, "Med")):
        bw, br, aw, ar, gone, left = lvl(r, L, 9), lvl(r, L, 10), lvl(r, L, 4), lvl(r, L, 2), lvl(r, L, 8), lvl(r, L, 3)
        worst = max((r["per"][p]["acct"][9] / r["per"][p]["acct"][10] / 60 if r["per"][p]["acct"][10] else 0) for p in range(5) if LEVEL[p] == L)
        out.append(f"{nm}: dot {bw/br/60 if br else 0:4.1f}  all {aw/ar/60 if ar else 0:4.2f}  away {100*gone/ar if ar else 0:4.1f}%  left {100*left/ar if ar else 0:4.1f}%  worst {worst:4.1f}")
    print(f"{r['letter']:6s} #{r['num']:<3d} " + " | ".join(out))
