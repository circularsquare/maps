"""T-1504: the perceived minutes by mode for the named pairs, from the probe's explorer answers.
Rail: the busiest station pair's access and egress, its first route's ride minutes and distance
fare, the base fare, and the rest of the trip-weighted logsum (the planned wait and its tolerance
penalty, changes, and zones farther from the station than the busiest pair's). Car: the rule's
money part estimated at 1.25 x the zone distance of road, 15 cents a km, 3 minutes a dollar, two
people; the rest is time and parking."""
import json
import sys
from pathlib import Path

art = Path(__file__).resolve().parents[3] / "artifacts/trainworld/T-1504/cmd"
runs = [("base", "a_helpers.out.json", None), ("passages", "e_link.out.json", "pairs"), ("passages + 6x trains", "h_trains.out.json", "pairs"),
        ("passages + 6x trains + half fare", "i_fare.out.json", "pairs"), ("passages + half fare", "j_back.out.json", "pairs"),
        ("every pair solved: half fare", "l_fare_only.out.json", "pairs"), ("every pair solved: 6x trains", "m_trains_only.out.json", "pairs")]
want = sys.argv[1:] or ["bei-sha", "gua-shz", "wuh-chs", "sha-nan", "bei-tia", "bei-wuh"]
for label, f, key in runs:
    p = art / f
    if not p.exists():
        continue
    d = json.loads(p.read_text(encoding="utf-8"))
    d = d[key] if key else d
    print(f"\n== {label}")
    for k in want:
        v = d.get(k)
        if not v:
            continue
        y, b = v["day"], v["big"]
        t = y["trips"]
        rr = y["rail"] + y["car"] + y["air"]
        line = f"{k:8} {t:8,.0f} trips/day each way, {b['km']:5.0f} km | rail {y['rail']:8,.1f} = {100 * y['rail'] / t:5.2f}% (air {100 * y['air'] / t:4.1f}%, car {100 * y['car'] / t:5.1f}%) | served {100 * b['served']:5.1f}%"
        print(line)
        car = b["carP"] or 0
        money = b["km"] * 1.25 * 0.15 * 3 / 2
        if b["railP"] is None:
            print(f"         rail: no zone pair served | car {car:.0f} (money ~{money:.0f}, time and parking ~{car - money:.0f})")
            continue
        pr = b["pairs"][0] if b["pairs"] else None
        ride = fare = waitp = 0
        if b["routes"]:
            for l in b["routes"][0]["legs"]:
                if l["kind"] == "ride":
                    ride += l["min"]
                    fare += l["fare"] or 0
                    waitp += l["waitP"] or 0
        rest = b["railP"] - (pr["aP"] + pr["bP"] if pr else 0) - ride - fare - b["baseFareP"]
        print(f"         rail {b['railP']:.0f} = access {pr['aP']:.0f} + egress {pr['bP']:.0f} + ride {ride:.0f} + fare {fare:.0f} + base fare {b['baseFareP']:.1f} + rest {rest:.0f}"
              f"  (route's full waits {waitp:.0f}) | car {car:.0f} (money ~{money:.0f}, time and parking ~{car - money:.0f})")
