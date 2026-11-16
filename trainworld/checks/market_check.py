r"""Verify every registered market pack (T-2775). Run before and after a join (docs/adding-a-country.md 8).

    market_check.py [--tw <trainworld folder>] [--cells] [--only id,id] [--save <file>] [--against <file>]

Reads app/src/game/registry.json and data/packs/<id>.market.json (read only) and fails (exit 1) for a pack whose
file market_pack would refuse or whose neighbours would read it wrongly:
  - a registered zone with no market file, or a file for a zone the registry no longer holds
  - parallel arrays of different lengths (zones, sites, commutes, outings)
  - a commute or outing whose home zone, place or zone index is out of range, or whose place has no file
    ("invalid commute between zones", T-2702)
  - zones that the file's own sites do not give (the 6 km squares in first-seen order)
With --cells the zones are also worked out from the pack's cells as a later run without a rewrite would do
(slower: reads every pack); a file written under older code then differs (thl: 159 zones against its file's 158).
--save writes each pack's zone count and row counts; --against compares to such a file and lists packs that changed,
so a join's effect is the list printed after it.
"""
import argparse
import json
import math
import os
import sys

TW = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZONE_M = 6000
PAIRS = (("commute", ("home", "place", "work", "trips")), ("everyday", ("home", "place", "zone", "kind", "trips", "walk")))


def zone_keys(sites):
    keys = {}
    for s in sites:
        keys.setdefault((math.floor(s[0] / ZONE_M), math.floor(s[1] / ZONE_M)), len(keys))
    return [list(k) for k in keys]


def load(tw):
    with open(os.path.join(tw, "app/src/game/registry.json"), encoding="utf8") as f:
        reg = json.load(f)
    return reg, {c["id"]: c for c in reg["cities"] if c.get("place") is not None}


def check(tw, cells=False, only=None):
    reg, live = load(tw)
    packs = os.path.join(tw, "data/packs")
    files = {}
    bad = []
    for cid in live:
        p = os.path.join(packs, f"{cid}.market.json")
        if not os.path.exists(p):
            bad.append((cid, "no market file"))
            continue
        with open(p, encoding="utf8") as f:
            files[cid] = json.load(f)
    for n in sorted(os.listdir(packs)):
        if n.endswith(".market.json") and n[:-12] not in live:
            bad.append((n[:-12], "market file of a zone not in the registry"))
    nz = {m["place"]: len(m["zones"]) for m in files.values()}
    snap = {}
    own = None
    if cells:
        sys.path.insert(0, os.path.join(tw, "pipeline"))
        import market_zones as mz
        own = mz
    for cid, m in files.items():
        if only and cid not in only:
            continue
        if m["place"] != live[cid]["place"]:
            bad.append((cid, f"place {m['place']} in the file, {live[cid]['place']} in the registry"))
        z = len(m["zones"])
        for k in ("zone_ng", "zone_nt", "zone_parts"):
            if k in m and len(m[k]) != z:
                bad.append((cid, f"{k} has {len(m[k])} entries for {z} zones"))
        ns = len(m["sites"])
        for k in ("site_park", "cell_pop", "site_part"):
            if k in m and len(m[k]) != ns:
                bad.append((cid, f"{k} has {len(m[k])} entries for {ns} sites"))
        for pre, cols in PAIRS:
            arrs = {c: m.get(f"{pre}_{c}", []) for c in cols}
            lens = {c: len(a) for c, a in arrs.items()}
            if len(set(lens.values())) > 1:
                bad.append((cid, f"{pre} arrays differ in length {lens}"))
                continue
            zc = "work" if pre == "commute" else "zone"
            n = 0
            for h, p, w in zip(arrs["home"], arrs["place"], arrs[zc]):
                if h < 0 or h >= z or p not in nz or w < 0 or w >= nz[p]:
                    n += 1
            if n:
                bad.append((cid, f"{n} of {len(arrs['home'])} {pre} rows name a zone or place that does not exist"))
        if zone_keys(m["sites"]) != [list(k) for k in m["zones"]]:
            bad.append((cid, f"zones differ from its own sites ({z} in the file, {len(zone_keys(m['sites']))} from sites)"))
        if own is not None:
            o = live[cid]["origin"]
            ks = own.zones_of(own.own_sites(cid, {"kind": "equirect", "lon": o["lon"], "lat": o["lat"]})[0])[0]
            if [list(k) for k in ks] != [list(k) for k in m["zones"]]:
                bad.append((cid, f"zones from cells {len(ks)} against the file's {z}"))
        snap[cid] = {"zones": z, "sites": ns, "commute": len(m.get("commute_home", [])), "everyday": len(m.get("everyday_home", []))}
    return bad, snap


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tw", default=TW)
    ap.add_argument("--cells", action="store_true")
    ap.add_argument("--only", help="comma-separated ids to check (the rest still serve as places)")
    ap.add_argument("--save", metavar="FILE")
    ap.add_argument("--against", metavar="FILE")
    a = ap.parse_args()
    bad, snap = check(os.path.abspath(a.tw), a.cells, set(a.only.split(",")) if a.only else None)
    for cid, why in bad[:60]:
        print("BAD", cid, why)
    if len(bad) > 60:
        print("...", len(bad) - 60, "more")
    print(len(snap), "market packs read,", len(bad), "problems")
    if a.save:
        with open(a.save, "w") as f:
            json.dump(snap, f)
        print("saved", a.save)
    if a.against:
        with open(a.against) as f:
            old = json.load(f)
        new = [k for k in snap if k not in old]
        gone = [k for k in old if k not in snap]
        diff = [k for k in snap if k in old and snap[k] != old[k]]
        print(f"against {a.against}: {len(new)} new, {len(gone)} gone, {len(diff)} changed")
        for k in diff[:30]:
            print("  changed", k, old[k], "->", snap[k])
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
