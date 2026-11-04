"""T-279: the routed drives against the market's car: road factor, time ratio, and how much rail's
odds against the car would move if the market used the routed drive (perceived minutes as
car_min: drive + 0.15 $/km x 3 min/$ / 2 occupants; scale 0.10 x sqrt(60 / car))."""
import json, math, os, statistics as st, sys
HERE = os.path.dirname(os.path.abspath(__file__))
rows = json.load(open(os.path.join(HERE, "osrm", "osrm_pairs.json")))
# the market offers no car between landmasses (ferries other than the Channel and Tsugaru are not
# modelled), and adds those two crossings itself
NO_CAR = {("Helsinki", "Tallinn"), ("London", "Dublin"), ("Athens", "Heraklion"), ("Barcelona", "Palma")}
CROSSING = {("London", "Paris"): 60 + 225, ("Tokyo", "Sapporo"): 240 + 255}


def car(minutes, road_km):
    return minutes + road_km * 0.15 * 3 / 2


print("kind | pair | km | road factor | OSRM min | model min | model / OSRM | rail odds x (routed)")
by = {}
for r in rows:
    if (r["a"], r["b"]) in NO_CAR:
        print(f"{r['kind']} | {r['a']}-{r['b']} | {r['gc_km']:.0f} | {r['factor']:.2f} | {r['osrm_min']:.0f} | no car in the market | | |")
        continue
    cm = car(r["model_min"], 1.25 * r["gc_km"]) + CROSSING.get((r["a"], r["b"]), 0)
    # OSRM's own route takes a ferry where the market adds a crossing: compare without
    if (r["a"], r["b"]) in CROSSING:
        print(f"{r['kind']} | {r['a']}-{r['b']} | {r['gc_km']:.0f} | {r['factor']:.2f} | {r['osrm_min']:.0f} | {r['model_min']:.0f} + crossing | (OSRM by ferry) | |")
        continue
    cr = car(r["osrm_min"], r["road_km"])
    b = 0.10 * math.sqrt(min(1, 60 / cm))
    odds = math.exp(b * (cr - cm))
    r["odds"] = odds
    by.setdefault(r["kind"], []).append(r)
    print(f"{r['kind']} | {r['a']}-{r['b']} | {r['gc_km']:.0f} | {r['factor']:.2f} | {r['osrm_min']:.0f} | {r['model_min']:.0f} | {r['model_over_osrm']:.2f} | {odds:.2f}")
print()
for k, v in by.items():
    print(f"{k}: n {len(v)}, road factor median {st.median(x['factor'] for x in v):.2f} (range {min(x['factor'] for x in v):.2f}-{max(x['factor'] for x in v):.2f}), "
          f"model/OSRM median {st.median(x['model_over_osrm'] for x in v):.2f}, OSRM km/h median {st.median(x['osrm_kmh'] for x in v):.0f}, rail odds median {st.median(x['odds'] for x in v):.2f}")
allv = [x for v in by.values() for x in v]
print(f"all on land: n {len(allv)}, factor median {st.median(x['factor'] for x in allv):.2f}, model/OSRM median {st.median(x['model_over_osrm'] for x in allv):.2f}")
