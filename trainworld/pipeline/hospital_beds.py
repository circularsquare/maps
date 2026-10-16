"""T-387: the pooled numbers behind tier 0's hospital sizes (`enrich_medical.py` reads them from
`hospital_beds.json`), and the held-out check.

    python pipeline/hospital_beds.py pool       pool from the packs where OSM tags beds well; write the table
    python pipeline/hospital_beds.py check <id> the rule on a pack's tagged hospitals against their tags, by 2 km zone
    python pipeline/hospital_beds.py survey     every pack's beds under the rule (nothing written)

Reads each pack's Overpass caches (data/raw/t142b/) and keeps the facilities its medical audit
lists, so nothing is fetched and no pack is touched. Pools over the packs where at least
POOL_SHARE of hospitals carry `beds` and at least POOL_MIN of those are mapped as areas, less
HELD_OUT.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import math
import sys

import numpy as np
import requests

import build_city as b
import enrich_medical as med

WB_URL = "https://api.worldbank.org/v2/country/all/indicator/SH.MED.BEDS.ZS?format=json&per_page=20000"
WB_PATH = os.path.join(b.RAW, "t387", "SH.MED.BEDS.ZS.json")
POOL_SHARE = 0.15
POOL_MIN = 5
MIN_FLOOR_M2 = 1000.0
SIZE_POWER = 0.5            # beds ~ floor^0.5: flat between 0.3 and 0.7 in the check (notes/T-387.md)
HELD_OUT = ["nyc"]          # the check's pack, kept out of the pool
CAP_SHARE = 0.05            # a site keeps at most this share of the pack's beds,
CAP_MIN_BEDS = 1500.0       # or this many where that is more (a small pack's one big hospital),
CAP_MAX_BEDS = 4000.0       # and never more than this (DECISIONS 2026-10-16, T-387)
ZONE_M = 2000.0


def packs():
    return sorted(f[:-len(".medical.audit.json")] for f in os.listdir(b.WORK) if f.endswith(".medical.audit.json"))


def hospitals(city):
    """The pack's hospitals as its last enrichment kept them, with their building floor parts."""
    with open(os.path.join(b.WORK, city + ".medical.audit.json")) as f:
        audit = json.load(f)
    keep = {r["osm"] for r in audit["facilities"] + audit.get("omitted", [])}
    kept, _ = med.facilities(city, keep=keep)
    return [r for r in kept if r["kind"] == "hospital"]


def pack_info(city):
    with open(os.path.join(b.PACKS, city + ".json")) as f:
        header = json.load(f)
    a = next(a for a in header["arrays"] if a["name"] == "pop")
    with open(os.path.join(b.PACKS, city + ".bin"), "rb") as f:
        f.seek(a["offset"])
        pop = np.frombuffer(f.read(4 * a["count"]), dtype="<f4").sum(dtype=np.float64)
    return med.country_of(city, header), float(pop), (header["origin"]["lon"], header["origin"]["lat"])


def well_tagged(hosp):
    tagged = [r for r in hosp if r["beds"]]
    return len(tagged) >= POOL_SHARE * len(hosp) and sum(1 for r in tagged if r["ha"]) >= POOL_MIN


def worldbank():
    if not os.path.exists(WB_PATH):
        r = requests.get(WB_URL, headers=b.UA, timeout=120)
        r.raise_for_status()
        os.makedirs(os.path.dirname(WB_PATH), exist_ok=True)
        with open(WB_PATH, "w") as f:
            json.dump(r.json(), f)
    with open(WB_PATH) as f:
        meta, rows = json.load(f)
    latest = {}
    for x in rows:
        if x["value"] is None:
            continue
        key = "WLD" if x["countryiso3code"] == "WLD" else x["country"]["id"]
        if key not in latest or x["date"] > latest[key][0]:
            latest[key] = [x["date"], round(float(x["value"]), 3)]
    return dict(sorted(latest.items())), meta.get("lastupdated")


def pooled(by_city):
    """The pooled numbers from these packs' hospitals. Building shape pooled over every pack instead
    (6.3 levels, 2.5 m² of floor a site m²) put New York's check at 23.2% against 20.1% (T-387)."""
    allh = [r for h in by_city.values() for r in h]
    # levels of the hospital buildings that carry them, weighted by footprint
    levels = sum(r.get("floor_tagged_m2", 0.0) for r in allh) / sum(r.get("footprint_tagged_m2", 0.0) for r in allh)
    site = [med.floor_m2(r, levels) / (r["ha"] * 1e4) for r in allh if r["ha"] and med.floor_m2(r, levels) > MIN_FLOOR_M2]
    t = dict(typical_levels=round(levels, 2), floor_per_site_m2=round(float(np.median(site)), 3), sites=len(site),
             size_power=SIZE_POWER)
    beds = [r for r in allh if r["beds"]]
    pairs = [r["beds"] / med.size_m2(r, t) ** SIZE_POWER for r in beds if med.size_m2(r, t) > MIN_FLOOR_M2]
    points = [r["beds"] for r in beds if med.size_m2(r, t) <= 0]
    t.update(beds_per_sqrt_m2=round(float(np.median(pairs)), 4), pairs=len(pairs),
             median_beds_no_site=float(np.median(points if len(points) >= 10 else [r["beds"] for r in beds])),
             points=len(points))
    return t


def gather(cities):
    out = {}
    for c in cities:
        out[c] = hospitals(c)
        print(f"  {c}: {len(out[c])} hospitals, {sum(1 for r in out[c] if r['beds'])} with beds", flush=True)
    return out


def pool():
    by_city = gather(packs())
    pool_cities = [c for c, h in by_city.items() if well_tagged(h) and c not in HELD_OUT]
    p = pooled({c: by_city[c] for c in pool_cities})
    rates, updated = worldbank()
    table = dict(
        about="T-387: tier 0 hospital sizes; written by hospital_beds.py pool, read by enrich_medical.py",
        pooled_from=pool_cities, held_out=HELD_OUT, **p, cap_share=CAP_SHARE, cap_min_beds=CAP_MIN_BEDS, cap_max_beds=CAP_MAX_BEDS,
        beds_per_1000_source=f"World Bank SH.MED.BEDS.ZS, latest year per country (updated {updated}); [year, beds per 1,000]",
        beds_per_1000=rates)
    with open(med.TABLE + ".part", "w") as f:
        json.dump(table, f, indent=1)
    os.replace(med.TABLE + ".part", med.TABLE)
    print(json.dumps({k: v for k, v in table.items() if k != "beds_per_1000"}), flush=True)


def zone_key(lon, lat, origin):
    x = (lon - origin[0]) * 111320.0 * math.cos(math.radians(origin[1]))
    y = (lat - origin[1]) * 110540.0
    return (math.floor(x / ZONE_M), math.floor(y / ZONE_M))


def check(city):
    """The rule's beds for the pack's tagged hospitals, as if untagged, against their tags: the
    share of beds in the wrong 2 km zone (half the summed absolute difference of the zone shares)
    and the total."""
    table = med.load_table()
    hosp = [r for r in hospitals(city) if r["beds"]]
    country, pop, origin = pack_info(city)
    truth, pred = {}, {}
    rows = []
    for r in hosp:
        est, src = med.raw_beds(dict(r, beds=None), table)
        z = zone_key(r["lon"], r["lat"], origin)
        truth[z] = truth.get(z, 0.0) + r["beds"]
        pred[z] = pred.get(z, 0.0) + est
        rows.append((r["beds"], est, src, r["name"]))
    T, P = sum(truth.values()), sum(pred.values())
    mis = 0.5 * sum(abs(truth.get(z, 0) / T - pred.get(z, 0) / P) for z in set(truth) | set(pred))
    by_src = {}
    for t, e, s, _ in rows:
        a = by_src.setdefault(s, [0, 0.0, 0.0])
        a[0] += 1
        a[1] += t
        a[2] += e
    print(f"{city}: {len(hosp)} tagged hospitals in {len(truth)} zones; tagged {T:,.0f} beds, rule {P:,.0f} "
          f"({P / T:.2f}); misplaced by 2 km zone {mis:.1%}", flush=True)
    for s, (n, t, e) in by_src.items():
        print(f"  from {s}: {n} hospitals, tagged {t:,.0f}, rule {e:,.0f}", flush=True)
    return mis, P / T


def survey():
    table = med.load_table()
    print(f"{'pack':5} {'cc':3} {'pop':>11} {'rate':>6} {'raw':>9} {'scale':>6} {'beds':>8} {'cap':>6} {'nc':>3} "
          f"{'top':>6} {'median':>6}  top site", flush=True)
    for c in packs():
        hosp = hospitals(c)
        country, pop, _ = pack_info(c)
        s = med.size_hospitals(hosp, pop, country, table)
        if not s:
            continue
        top = max(hosp, key=lambda r: r["beds_used"])
        print(f"{c:5} {country:3} {pop:11,.0f} {s['beds_per_1000']:6.2f} {s['raw_beds']:9,.0f} {s['scale']:6.2f} "
              f"{s['beds']:8,.0f} {s['cap_beds']:6,.0f} {s['capped_sites']:3} {top['beds_used']:6,.0f} "
              f"{np.median([r['beds_used'] for r in hosp]):6,.0f}  {top['name'][:40]}", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "pool":
        pool()
    elif cmd == "check":
        check(sys.argv[2])
    elif cmd == "survey":
        survey()
