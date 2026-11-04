"""T-1523: read probe.mjs's JSON and print per-line, per-period figures."""
import json
import sys

H = [4, 6, 4, 4, 6]
NAMES = ["am peak 6-10", "midday 10-16", "pm peak 16-20", "evening 20-24", "night 0-6"]
LEVEL = [0, 1, 0, 1, 2]
CRUSH = 160

d = json.load(open(sys.argv[1]))
peak = d["peak"]
only = sys.argv[2].split(",") if len(sys.argv) > 2 else None
print("weekday view:", d["weekday"], "missing accounts:", d["missing"], "peak factors:", peak)

rows = []
for r in d["rows"]:
    tph = r["tph"]
    if not r["per"][0].get("acct"):
        continue
    same = tph["high"] > 0 and abs(tph["high"] - tph["medium"]) <= 0.02 * tph["high"]
    lv = [[0.0, 0.0, 0.0, 0.0] for _ in range(3)]  # busy wait, busy riders, all wait, all arrivals
    per = []
    for p in range(5):
        x = r["per"][p]
        a = x["acct"]
        t = [tph["high"], tph["medium"], tph["low"]][LEVEL[p]]
        cap = t * (r["cars"] or 0) * CRUSH
        seg = max(x.get("maxSeg", [0, 0])) / H[p]
        per.append({
            "p": p, "tph": t, "cap": cap, "arr_h": a[2] / H[p], "peak_arr_h": a[2] / H[p] * peak[p],
            "seg_h": seg, "seg_peak_h": seg * peak[p], "ratio": seg * peak[p] / cap if cap else 0,
            "trains": a[0], "full": a[1], "gone": a[8], "busy_min": a[9] / a[10] / 60 if a[10] else 0,
            "busy_share": a[10] / a[2] if a[2] else 0, "all_min": a[4] / a[2] / 60 if a[2] else 0,
        })
        L = lv[LEVEL[p]]
        L[0] += a[9]; L[1] += a[10]; L[2] += a[4]; L[3] += a[2]
    dot = [L[0] / L[1] / 60 if L[1] else 0 for L in lv]
    allw = [L[2] / L[3] / 60 if L[3] else 0 for L in lv]
    rows.append({"r": r, "same": same, "per": per, "dot": dot, "allw": allw})

def show(x):
    r = x["r"]
    pur = r.get("purposes", {})
    tot = sum(pur.values()) or 1
    print(f"\n{r['letter']} {r['name']} (#{r['num']}) stops {r['stops']} cars {r['cars']} tph H/M/L {r['tph']['high']:.1f}/{r['tph']['medium']:.1f}/{r['tph']['low']:.1f}"
          f"  riders/day {r['ridersPerDay']:.0f}  work {100*pur.get('work',0)/tot:.0f}%")
    print(f"  dot High {x['dot'][0]:.1f} min  Medium {x['dot'][1]:.1f} min  | all-rider avg wait High {x['allw'][0]:.2f} Medium {x['allw'][1]:.2f}")
    print("  period          tph   cap/h  arrive/h peakhr/h  seg/h  segpeak/h ratio  trains full gone  busyMin busyShare")
    for q in x["per"]:
        print(f"  {NAMES[q['p']]:14s} {q['tph']:5.1f} {q['cap']:7.0f} {q['arr_h']:8.0f} {q['peak_arr_h']:8.0f} {q['seg_h']:7.0f} {q['seg_peak_h']:8.0f} {q['ratio']:5.2f}"
              f" {q['trains']:6.0f} {q['full']:4.0f} {q['gone']:5.0f} {q['busy_min']:7.1f} {100*q['busy_share']:6.0f}%")

same = [x for x in rows if x["same"]]
print(f"\n{len(rows)} lines with an account; {len(same)} with High == Medium trains")
worse = [x for x in same if x["dot"][1] > x["dot"][0]]
print(f"of those, Medium dot worse than High on {len(worse)}; Medium dot >= 6 min (yellow+) on {sum(1 for x in same if x['dot'][1] >= 6)}, High on {sum(1 for x in same if x['dot'][0] >= 6)}")
# where the medium crowding comes from: midday or evening
for x in sorted(same, key=lambda x: -(x["dot"][1] - x["dot"][0]))[: (99 if only is None else 999)]:
    if only and x["r"]["letter"] not in only and str(x["r"]["num"]) not in only:
        continue
    show(x)
