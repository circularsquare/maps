"""T-1523: network-wide per-period figures in each run, and midday's busiest stretch against the evening peak's."""
import json
import statistics

ART = "C:/Users/anita/projects/maps/artifacts/trainworld/"
H = [4, 6, 4, 4, 6]
PN = ["6-10", "10-16", "16-20", "20-24", "0-6"]
for k in ("base", "medhigh", "highmed"):
    d = json.load(open(ART + f"t1523-{k}.json"))
    tot = [[0, 0, 0, 0] for _ in range(5)]
    busy_lines = [0] * 5
    for r in d["rows"]:
        for p in range(5):
            a = r["per"][p].get("acct")
            if not a:
                continue
            tot[p][0] += a[8]; tot[p][1] += a[2]; tot[p][2] += a[1]; tot[p][3] += a[0]
            busy_lines[p] += a[10] > 0
    print(k, "rail trips/day", round(d["rail"]))
    for p in range(5):
        t = tot[p]
        print(f"  {PN[p]:6s} arrivals {t[1]:10.0f}  turned away {t[0]:8.0f} ({100*t[0]/t[1]:.1f}%)  trains full {t[2]:5.0f}/{t[3]:5.0f}  lines with busy platforms {busy_lines[p]}")
d = json.load(open(ART + "t1523-base.json"))
peak = d["peak"]
rat = []
for r in d["rows"]:
    s = [max(r["per"][p].get("maxSeg", [0, 0])) / H[p] * peak[p] for p in range(5)]
    if s[2] > 100:
        rat.append(s[1] / s[2])
        rat.append(s[1] / s[2])
q = statistics.quantiles(rat, n=4)
print(f"midday busiest-stretch busiest hour / evening peak's, {len(rat)//2} lines: quartiles {q[0]:.2f} {q[1]:.2f} {q[2]:.2f}")
ev = [max(r["per"][3].get("maxSeg", [0, 0])) / 4 * peak[3] / max(1e-9, max(r["per"][2].get("maxSeg", [0, 0])) / 4 * peak[2]) for r in d["rows"] if max(r["per"][2].get("maxSeg", [0, 0])) > 400]
q = statistics.quantiles(ev, n=4)
print(f"evening 20-24 / evening peak, same measure: quartiles {q[0]:.2f} {q[1]:.2f} {q[2]:.2f}")
diff = sum(1 for r in d["rows"] if r["tph"]["high"] > 0 and abs(r["tph"]["high"] - r["tph"]["medium"]) > 0.02 * r["tph"]["high"])
print("lines with High != Medium:", diff)
