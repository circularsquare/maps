import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
plan = None
if len(sys.argv) > 2:
    plan = json.load(open(sys.argv[2], encoding="utf-8"))
hours = d["hours"]
levels = {"high": 0, "medium": 1, "low": 2}
lv_hours = [0, 0, 0]
for h, lv in zip(hours, d["levels"]):
    lv_hours[levels[lv]] += h
print("level hours", lv_hours)
rows = []
for l in d["lines"]:
    if l["status"] != "running":
        continue
    carkm = sum([l["tph"]["high"], l["tph"]["medium"], l["tph"]["low"]][i] * 2 * l["km"] * lv_hours[i] for i in range(3)) * l["cars"]
    pk = sum(l["pkm"]["0"])
    pk6 = sum(l["pkm"]["6"])
    rows.append(dict(name=l["name"], first=l["first"], last=l["last"], km=l["km"], cars=l["cars"], fast=l["fast"], tph=[l["tph"]["high"], l["tph"]["medium"], l["tph"]["low"]], carkm=carkm, pkm=pk, pkm_avg=pk6,
                     per=pk / carkm if carkm else None, riders=l["riders"]["0"], id=l["id"]))
tot = sum(r["pkm"] for r in rows)
tot6 = sum(r["pkm_avg"] for r in rows)
ck = sum(r["carkm"] for r in rows)
print(f"{len(rows)} running lines of {len(d['lines'])}")
print(f"weekday pkm {tot/1e6:.0f}M, average day {tot6/1e6:.0f}M, car-km {ck/1e6:.2f}M, weekday pkm/car-km {tot/ck:.1f}")
print(f"ratio to MLIT 1,100M: weekday {tot/1.1e9:.2f}, average day {tot6/1.1e9:.2f}")
print("days:", [(x["k"], round(x["railTripsPerDay"] / 1e6, 2)) for x in d["days"] if x])
under = sorted((r for r in rows if r["per"] is not None and r["per"] < 10), key=lambda r: r["per"])
print(f"lines under 10 riders/car-km: {len(under)}")
for r in under[:60]:
    print(f"  {r['per']:.1f}  {r['name']}  {r['first']}-{r['last']}  {r['km']:.1f} km  cars {r['cars']}  tph {[round(x,2) for x in r['tph']]}  riders {r['riders']:.0f}  id {r['id']}")
worst = sorted((r for r in rows if r["per"] is not None), key=lambda r: r["per"])[:5]
print("worst 5:", [(round(r["per"], 1), r["name"]) for r in worst])
nones = [r for r in rows if r["per"] is None]
print("lines with no car-km:", len(nones), [r["name"] for r in nones[:10]])
over = sorted((r for r in rows if r["per"] and r["per"] > 100), key=lambda r: -r["per"])[:10]
print("busiest:", [(round(r["per"]), r["name"]) for r in over])
json.dump([{"id": r["id"], "name": r["name"], "per": r["per"], "pkm": r["pkm"], "carkm": r["carkm"], "km": r["km"], "first": r["first"], "last": r["last"]} for r in rows], open(sys.argv[1].replace(".json", "_rows.json"), "w", encoding="utf-8"))
