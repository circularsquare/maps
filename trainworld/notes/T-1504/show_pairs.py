"""T-1504: the probe's explorer answers (`<name>.out.json`): per pair, the day's trips by mode, and
the biggest period's perceived minutes and train trip."""
import json
import sys
from pathlib import Path

f = Path(sys.argv[1])
d = json.loads(f.read_text(encoding="utf-8"))
for k, v in d.items():
    if not v:
        print(k, "no answer")
        continue
    y, b = v["day"], v["big"]
    t = y["trips"] or 1
    print(f"\n{k}: {y['trips']:,.0f} trips a day each way; rail {y['rail']:,.1f} ({100 * y['rail'] / t:.3f}%), air {y['air']:,.0f} ({100 * y['air'] / t:.1f}%), car {y['car']:,.0f} ({100 * y['car'] / t:.1f}%)"
          f"  | everyday legs {y['everydayLegs']:,.0f} rail {y['everydayRail']:,.0f}; commute legs {y['commuteLegs']:,.0f} rail {y['commuteRail']:,.0f}")
    if not b:
        continue
    print(f"   km {b['km']:.0f}, zone pairs {b['zonePairs']:.0f}, served share {b['served']:.3f}, why '{b['why']}'; rail perceived {b['railP']}, car {b['carP']}, reference {b['refP']:.0f}; base fare {b['baseFareP']:.1f} min, {b['minPerUsd']:.2f} min a $")
    for p in b["pairs"][:4]:
        print(f"   pair {p['a']} -> {p['b']}: access {p['aP']:.0f}, egress {p['bP']:.0f}, station to station {p['strat']:.0f}, share {p['share']:.3f}")
    for r in b["routes"][:2]:
        print(f"   route share {r['share']:.3f}:")
        for l in r["legs"]:
            print(f"      {l['kind']} {l.get('line')} {l.get('from')} -> {l.get('to')} stops {l.get('stops')} min {l.get('min'):.0f} perceived {l.get('perc'):.0f} fare {l.get('fare') or 0:.0f} wait {l.get('waitMin') or 0:.0f}/{l.get('waitP') or 0:.0f} change {l.get('change') or 0:.0f}")
