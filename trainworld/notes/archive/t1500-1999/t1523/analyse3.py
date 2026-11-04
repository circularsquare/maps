"""T-1523 part 2: service-level options on the real save, assembled from three runs.

base: the save as it is; medHigh: every line's Medium count set to its High count; highMed: High set
to Medium. Demand solves each period on its own level's trains (morning pairs feed the evening's
return trips, and the morning is High in base and medHigh alike), so a period run at another level
is read from the run that gave it those trains.

Options, as level per clock hour, and the run each demand period is read from:
  now   : H 6-10, M 10-16, H 16-20, M 20-24, L 0-6
  day   : D 6-22, N 22-6 (Day = the High count)       periods: medHigh for all four day periods
  her   : M 5-6, H 6-17, M 17-22, L 22-5               am base, midday medHigh, pm highMed, eve base, night base
  h20   : M 5-6, H 6-20, M 20-22, L 22-5               am base, midday medHigh, pm base, eve base, night base
  three : H 6-20, M 20-24, L 0-6 (no new periods)      same periods as h20
Costs are exact by the hour (tph x 2 x km x cost a train-km); crowding per period from the run.
"""
import json
import sys

ART = "C:/Users/anita/projects/maps/artifacts/trainworld/"
H = [4, 6, 4, 4, 6]
PN = ["6-10", "10-16", "16-20", "20-24", "0-6"]
CRUSH = 160
runs = {k: json.load(open(ART + f"t1523-{k}.json")) for k in ("base", "medhigh", "highmed")}
peak = runs["base"]["peak"]

def hour_level(opt, h):
    if opt == "now":
        return "H" if 6 <= h < 10 or 16 <= h < 20 else "M" if 10 <= h < 16 or h >= 20 else "L"
    if opt == "day":
        return "H" if 6 <= h < 22 else "L"
    if opt == "her":
        return "M" if h == 5 or 17 <= h < 22 else "H" if 6 <= h < 17 else "L"
    if opt == "h20":
        return "M" if h == 5 or 20 <= h < 22 else "H" if 6 <= h < 20 else "L"
    if opt == "three":
        return "H" if 6 <= h < 20 else "M" if h >= 20 else "L"

SRC = {
    "now": ["base"] * 5,
    "day": ["medhigh"] * 4 + ["base"],
    "her": ["base", "medhigh", "highmed", "base", "base"],
    "h20": ["base", "medhigh", "base", "base", "base"],
    "three": ["base", "medhigh", "base", "base", "base"],
}
# which level's dot each period counts in, per option (a split period goes where most of its hours are)
DOT = {
    "now": "HMHML", "day": "HHHHL", "her": "HHMML", "h20": "HHHML", "three": "HHHML",
}
OPTS = ["now", "day", "her", "h20", "three"]

base_rows = {r["num"]: r for r in runs["base"]["rows"]}
idx = {k: {r["num"]: r for r in v["rows"]} for k, v in runs.items()}

def line_figs(num, opt):
    b = base_rows[num]
    tph = {"H": b["tph"]["high"], "M": b["tph"]["medium"], "L": b["tph"]["low"]}
    cost = sum(tph[hour_level(opt, h)] for h in range(24)) * 2 * (b["km"] or 0) * b["perTrainKm"]
    per, trips = [], 0.0
    lv = {}
    for p in range(5):
        r = idx[SRC[opt][p]][num]["per"][p]
        a = r.get("acct") or [0] * 11
        t = idx[SRC[opt][p]][num]["tph"]
        # the period's trains in that run: the level the run gave it
        lev_now = "HMHML"[p]
        tp = {"H": t["high"], "M": t["medium"], "L": t["low"]}[lev_now]
        cap = tp * (b["cars"] or 0) * CRUSH
        seg = max(r.get("maxSeg", [0, 0])) / H[p] * peak[p]
        per.append(dict(p=PN[p], tph=tp, cap=cap, rh=a[2] / H[p], busy=a[2] / H[p] * peak[p], seg=seg,
                        ratio=seg / cap if cap else 0, full=a[1], trains=a[0], gone=a[8],
                        dot=a[9] / a[10] / 60 if a[10] else 0))
        trips += r.get("board", 0)
        L = lv.setdefault(DOT[opt][p], [0, 0, 0, 0])
        L[0] += a[9]; L[1] += a[10]; L[2] += a[8]; L[3] += a[2]
    dots = {k: (v[0] / v[1] / 60 if v[1] else 0) for k, v in lv.items()}
    gone = sum(v[2] for v in lv.values())
    return dict(cost=cost, trips=trips, per=per, dots=dots, gone=gone)

# network totals
print("network, per weekday:")
print(f"{'option':6s} {'running cost $M':>16s} {'boardings':>12s} {'turned away':>12s}  lines with a later level's dot worse than High's")
for opt in OPTS:
    cost = trips = gone = 0
    inv = 0
    for num in base_rows:
        if not base_rows[num]["per"][0].get("acct"):
            continue
        f = line_figs(num, opt)
        cost += f["cost"]; trips += f["trips"]; gone += f["gone"]
        if f["dots"].get("M", 0) > f["dots"].get("H", 0) + 0.05:
            inv += 1
    print(f"{opt:6s} {cost/1e6:16.2f} {trips:12.0f} {gone:12.0f}  {inv}")
for opt in OPTS:
    print(opt, "rail trips a day (demand):", "n/a" if opt != "now" else f"{runs['base']['rail']:.0f}")
print("rail trips a day by run: base", round(runs["base"]["rail"]), "medHigh", round(runs["medhigh"]["rail"]), "highMed", round(runs["highmed"]["rail"]))

want = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [82, 94, 22, 46, 5]
for num in want:
    b = base_rows[num]
    print(f"\n{b['letter']} #{num}: {b['stops']} stops, {b['cars']} cars, tph H/M/L {b['tph']['high']:.1f}/{b['tph']['medium']:.1f}/{b['tph']['low']:.1f}, {b['km']:.1f} km")
    for opt in OPTS:
        f = line_figs(num, opt)
        ds = "  ".join(f"{k} {v:4.1f}" for k, v in sorted(f["dots"].items()))
        print(f"  {opt:5s} cost ${f['cost']/1e3:7.1f}k/day  boardings {f['trips']:8.0f}  turned away {f['gone']:6.0f}  dots: {ds}")
        for q in f["per"]:
            print(f"        {q['p']:6s} tph {q['tph']:4.1f} cap {q['cap']:6.0f}  riders/h {q['rh']:6.0f}  busiest hr {q['busy']:6.0f}  stretch {q['seg']:6.0f} ({q['ratio']:.2f})  full {q['full']:.0f}/{q['trains']:.0f}  away {q['gone']:5.0f}  dot {q['dot']:4.1f}")
