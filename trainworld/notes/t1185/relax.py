"""T-1185 rebuild: each planned Chinese zone row's 2020 township census inside its plan extent against
the row's GHSL people (pack_gate's pop_vs_plan fails outside 0.7-1.5, warns outside 0.85-1.2); with
--set, pop_vs_plan relaxed on the rows outside 0.7-1.5, as notes/t1150/prerelax.py did for the metros.
Read-only without --set; with it, run with the plan free (no runner or join writing it). About 8 s a
zone on one core (80 minutes for China's 638 on one, about 20 on --procs 4).

    python trainworld/notes/t1185/relax.py [--set] [--procs N] [--out <file.json>]
"""
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
TW = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(TW, "pipeline"))
from shapely import make_valid  # noqa: E402
from shapely.geometry import shape  # noqa: E402
import china  # noqa: E402
import shared_lock  # noqa: E402

WHY = ("T-1185: the zone holds the 2020 census by township, which GHSL's older counts miss in China "
       "(people placed in their home counties, the boom towns short); the townships agree with the county census")
PLAN = os.path.join(TW, "pipeline", "world_plan.json")
EXT = os.path.join(TW, "data", "work", "worldplan", "extents.geojson")


def census(geom):
    return sum(t[2] * t[3] for t in china.townships(make_valid(shape(geom))))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    rows = {r["id"]: r for r in shared_lock.read_json(PLAN)["metros"]
            if r.get("kind") == "zone" and "CHN" in (r.get("units") or {}) and r.get("status") == "planned"}
    feats = [f for f in shared_lock.read_json(EXT)["features"] if f["properties"].get("id") in rows]
    procs = int(sys.argv[sys.argv.index("--procs") + 1]) if "--procs" in sys.argv else 1
    if procs > 1:
        from multiprocessing import Pool
        with Pool(procs) as pool:
            counts = pool.map(census, [f["geometry"] for f in feats], chunksize=4)
    else:
        counts = [census(f["geometry"]) for f in feats]
    out = []
    for f, c in zip(feats, counts):
        r = rows[f["properties"]["id"]]
        v = c / r["people"]
        flag = "FAIL" if not 0.7 <= v <= 1.5 else "warn" if not 0.85 <= v <= 1.2 else ""
        out.append({"id": r["id"], "name": r["name"], "census": round(c), "plan": r["people"], "ratio": round(v, 3), "flag": flag})
        print(f"{r['id']} {r['name']}: census {c:,.0f}, plan {r['people']:,.0f}, {v:.2f} {flag}", flush=True)
    out.sort(key=lambda o: o["ratio"])
    c, p = sum(o["census"] for o in out), sum(o["plan"] for o in out)
    fail = [o["id"] for o in out if o["flag"] == "FAIL"]
    print(f"{len(out)} of {len(rows)} planned zone rows: census {c / 1e6:.1f}M against the plan's {p / 1e6:.1f}M ({c / max(p, 1) - 1:+.1%}); "
          f"fail {len(fail)} ({','.join(fail)}), warn {sum(o['flag'] == 'warn' for o in out)}")
    if "--out" in sys.argv:
        with open(sys.argv[sys.argv.index("--out") + 1], "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=0)
    if "--set" in sys.argv and fail:
        with shared_lock.held("plan", "t1185 relax"):
            doc = shared_lock.read_json(PLAN)
            for r in doc["metros"]:
                if r.get("id") in fail:
                    r.setdefault("gate_relax", {})["pop_vs_plan"] = WHY
            shared_lock.write_atomic(PLAN, json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
        print(f"pop_vs_plan relaxed on {len(fail)} rows")


if __name__ == "__main__":
    main()
