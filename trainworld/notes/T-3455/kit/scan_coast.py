"""T-3455: coastal land in no pack, for every registered pack, as a rebuild of that pack would add it (osm_units.reclaimed
on the pack's own outline and the OSM sea, as build() calls it), minus land other packs hold. Read-only on data/packs.

    scan_coast.py run <outdir> [--workers 3] [--limit-s 900] [--ids a,b,c]
    scan_coast.py work <W> <NW> <outdir>          (a worker; run starts them)
    scan_coast.py summary <outdir>                (flagged rows and their solve units, from the workers' files)

'run' lists the registered packs whose outline existed before it started (packs rebuilt tonight already went through
reclaimed), starts the workers (one core each), kills a worker stuck on one pack past --limit-s (recorded as
"timeout", the worker restarted past it), stops on trainworld/data/work/world_run/STOP, and ends with 'summary'.
Outlines are opened with FILE_SHARE_DELETE, so a writer's os.replace onto one never fails because the scan holds it.
Pieces that touch no sea are skipped (a hole in an outline is an enclave a rebuild keeps). Each piece gets GHSL people
and built surface (osm_units._people_many, 2020, 100 m). A plan row is flagged when its pieces hold FLAG_PEOPLE people
or FLAG_BUILT_M2 of built surface; flagged.json lists them by solve unit (world_run.groups_of), most people first.
"""
import json
import os
import subprocess
import sys
import time

os.environ["OMP_NUM_THREADS"] = "1"
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
TW = r"C:\Users\anita\projects\maps\trainworld"
MAPS = os.path.dirname(TW)
PY = os.path.join(MAPS, "venv", "Scripts", "python.exe")
PACKS = os.path.join(TW, "data", "packs")
REG = os.path.join(TW, "app", "src", "game", "registry.json")
PLAN = os.path.join(TW, "pipeline", "world_plan.json")
STOP = os.path.join(TW, "data", "work", "world_run", "STOP")
sys.path.insert(0, os.path.join(TW, "pipeline"))
FLAG_PEOPLE = 1000
FLAG_BUILT_M2 = 150_000
SIMPLIFY = 0.0003  # degrees (about 30 m), as T-3428's scan


def read_shared(path):
    """A file's text, opened so that another process may replace or delete it meanwhile (Windows)."""
    if os.name != "nt":
        return open(path, encoding="utf-8").read()
    import ctypes
    import msvcrt
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateFileW.restype = wintypes.HANDLE
    k32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD,
                                wintypes.DWORD, wintypes.HANDLE]
    h = k32.CreateFileW(path, 0x80000000, 0x1 | 0x2 | 0x4, None, 3, 0x80, None)  # GENERIC_READ, share all, OPEN_EXISTING
    if h is None or h == wintypes.HANDLE(-1).value:
        raise OSError(ctypes.get_last_error(), "CreateFileW failed", path)
    fd = msvcrt.open_osfhandle(h, os.O_RDONLY)
    with open(fd, "rb") as f:
        return f.read().decode("utf-8")


def outline(i):
    import shapely
    from shapely.geometry import shape
    fc = json.loads(read_shared(os.path.join(PACKS, f"{i}.boundary.geojson")))
    return shapely.make_valid(shapely.union_all([shapely.make_valid(shape(f["geometry"])) for f in fc["features"]]))


# ---------------------------------------------------------------- one pack

def scan_one(i, boxes, cache):
    import numpy as np
    import shapely
    import osm_units as ou
    import osm_water
    g = outline(i).simplify(SIMPLIFY)
    w, s, e, n = g.bounds
    lat0 = (s + n) / 2
    k = float(np.cos(np.radians(lat0)))
    d = ou.RECLAIM_KM / 111.0
    box = (w - d / k - 0.01, s - d - 0.01, e + d / k + 0.01, n + d + 0.01)
    sea = osm_water.sea(box)
    if not sea:
        return {"coastal": False}
    zone = shapely.transform(ou._local(g, lat0).buffer(d), lambda a: a / [k, 1.0])
    shapely.prepare(zone)
    near = [shapely.make_valid(p).intersection(zone) for p in sea if zone.intersects(p)]
    near = [p for p in near if not p.is_empty]
    if not near:
        return {"coastal": False}
    water = ou.without_islets(shapely.union_all(near), lat0)
    found = ou.reclaimed([g], water, lat0)
    out = []
    tol = 0.0005
    wb = water.buffer(tol)
    for _, p in found:
        if not p.intersects(wb):
            continue
        b = p.bounds
        others = []
        for j, (x0, y0, x1, y1) in boxes.items():
            if j != i and x0 <= b[2] and x1 >= b[0] and y0 <= b[3] and y1 >= b[1]:
                if j not in cache:
                    try:
                        cache[j] = outline(j)
                    except OSError:
                        continue
                others.append(cache[j])
        q = shapely.make_valid(p.difference(shapely.union_all(others))) if others else p
        for part in shapely.get_parts(q):
            if part.geom_type == "Polygon" and ou.km2(part, lat0) >= ou.RECLAIM_MIN_KM2:
                c = part.representative_point()
                out.append({"km2": round(ou.km2(part, lat0), 2), "lon": round(c.x, 4), "lat": round(c.y, 4), "geom": part})
    if out:
        try:
            ppl = ou._people_many([o["geom"] for o in out])
            blt = ou._people_many([o["geom"] for o in out], "tot_s")
        except Exception as x:  # a GHSL tile unreadable: the piece is still listed
            ppl = blt = [None] * len(out)
            print(f"  {i}: GHSL read failed: {x!r}"[:200], flush=True)
        for o, a, bb in zip(out, ppl, blt):
            o["people"] = None if a is None else round(float(a))
            o["built_m2"] = None if bb is None else round(float(bb))
            del o["geom"]
    return {"coastal": True, "pieces": out}


def work(W, NW, outdir):
    ids = json.load(open(os.path.join(outdir, "ids.json")))
    boxes = {k: tuple(v) for k, v in json.load(open(os.path.join(outdir, "boxes.json"))).items()}
    res = os.path.join(outdir, f"w{W}.jsonl")
    done = set()
    if os.path.exists(res):
        for line in open(res, encoding="utf-8"):
            try:
                done.add(json.loads(line)["id"])
            except ValueError:
                pass
    prog = os.path.join(outdir, f"w{W}.progress")
    cache = {}
    for c, i in enumerate(ids):
        if c % NW != W or i in done:
            continue
        if os.path.exists(STOP):
            print("STOP file: worker ends", flush=True)
            return
        with open(prog, "w") as f:
            f.write(f"{i} {time.time()}\n")
        t = time.time()
        try:
            r = scan_one(i, boxes, cache)
        except Exception as x:
            r = {"error": repr(x)[:300]}
        r.update({"id": i, "s": round(time.time() - t, 1)})
        with open(res, "a", encoding="utf-8") as f:
            f.write(json.dumps(r) + "\n")
        if len(cache) > 400:
            cache.clear()
        if r.get("pieces"):
            print(f"{c} {i}: {len(r['pieces'])} pieces, {r['s']} s", flush=True)
    with open(prog, "w") as f:
        f.write("done\n")


# ---------------------------------------------------------------- the supervisor

def run(outdir, nw, limit, only):
    os.makedirs(outdir, exist_ok=True)
    t0 = time.time()
    reg = json.load(open(REG, encoding="utf-8"))
    ids = [c["id"] for c in reg["cities"] if os.path.exists(os.path.join(PACKS, f"{c['id']}.boundary.geojson"))]
    if not os.path.exists(os.path.join(outdir, "ids.json")):
        boxes = {}
        fresh = set()
        for i in ids:
            p = os.path.join(PACKS, f"{i}.boundary.geojson")
            if os.path.getmtime(p) > t0 - 60:
                fresh.add(i)  # rebuilt tonight: its build ran reclaimed
            try:
                boxes[i] = list(outline(i).bounds)
            except Exception as x:
                print(f"{i}: outline unreadable: {x!r}"[:200], flush=True)
        scan = [i for i in ids if i in boxes and i not in fresh and (not only or i in only)]
        json.dump(boxes, open(os.path.join(outdir, "boxes.json"), "w"))
        json.dump(scan, open(os.path.join(outdir, "ids.json"), "w"))
        print(f"{len(scan)} packs to scan, outlines read in {time.time() - t0:.0f} s", flush=True)
    procs = {}

    def start(W):
        log = open(os.path.join(outdir, f"w{W}.log"), "a", encoding="utf-8")
        procs[W] = subprocess.Popen([PY, "-I", os.path.abspath(__file__), "work", str(W), str(nw), outdir],
                                    stdout=log, stderr=subprocess.STDOUT, cwd=MAPS)
    for W in range(nw):
        start(W)
    while procs:
        time.sleep(15)
        if os.path.exists(STOP):
            for p in procs.values():
                p.kill()
            print("STOP file: scan stopped; run again to go on where it was", flush=True)
            return 2
        for W, p in list(procs.items()):
            prog = os.path.join(outdir, f"w{W}.progress")
            if p.poll() is not None:
                last = open(prog).read().strip() if os.path.exists(prog) else ""
                if last == "done":
                    del procs[W]
                    print(f"worker {W} done", flush=True)
                else:
                    print(f"worker {W} exited {p.returncode} at {last!r}; restarted", flush=True)
                    if last:  # the pack it died on is recorded, so the restart goes past it
                        with open(os.path.join(outdir, f"w{W}.jsonl"), "a", encoding="utf-8") as f:
                            f.write(json.dumps({"id": last.split()[0], "error": f"worker exit {p.returncode}"}) + "\n")
                    start(W)
                continue
            try:
                i, since = open(prog).read().split()
            except (OSError, ValueError):
                continue
            if time.time() - float(since) > limit:
                p.kill()
                p.wait()
                with open(os.path.join(outdir, f"w{W}.jsonl"), "a", encoding="utf-8") as f:
                    f.write(json.dumps({"id": i, "timeout": True, "s": round(time.time() - float(since))}) + "\n")
                print(f"worker {W}: {i} over {limit} s, skipped", flush=True)
                start(W)
    print(f"scan done in {(time.time() - t0) / 60:.0f} min", flush=True)
    return summary(outdir)


def summary(outdir):
    import glob
    res = {}
    for p in sorted(glob.glob(os.path.join(outdir, "w*.jsonl"))):
        for line in open(p, encoding="utf-8"):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            res[r["id"]] = r
    plan = {r["id"]: r for r in json.load(open(PLAN, encoding="utf-8"))["metros"] if r.get("units")}
    rows = []
    for i, r in res.items():
        ps = r.get("pieces") or []
        if not ps:
            continue
        ppl = sum(p.get("people") or 0 for p in ps)
        blt = sum(p.get("built_m2") or 0 for p in ps)
        rows.append({"id": i, "km2": round(sum(p["km2"] for p in ps), 2), "people": ppl, "built_m2": blt,
                     "pieces": ps, "plan_row": i in plan and plan[i]["status"] == "shipped",
                     "iso2": plan.get(i, {}).get("iso2"), "name": plan.get(i, {}).get("name"),
                     "flag": ppl >= FLAG_PEOPLE or blt >= FLAG_BUILT_M2})
    rows.sort(key=lambda x: (-x["people"], -x["built_m2"]))
    flagged = [x for x in rows if x["flag"] and x["plan_row"]]
    hand = [x for x in rows if x["flag"] and not x["plan_row"]]
    groups = []
    if flagged:
        import world_run
        for u in world_run.groups_of([x["id"] for x in flagged]):
            key, members = u["name"], u["members"]
            mine = [x for x in flagged if x["id"] in members]
            if mine:
                cc = {c["id"]: c.get("country") for c in json.load(open(REG, encoding="utf-8"))["cities"]}
                back = sorted({cc.get(j) for j in list(members) + list(u.get("fixed") or []) if cc.get(j)})
                groups.append({"key": key, "rows": [x["id"] for x in mine], "members": len(members), "backup": back,
                               "countries": sorted({x["iso2"] for x in mine if x["iso2"]}),
                               "people": sum(x["people"] for x in mine), "built_m2": sum(x["built_m2"] for x in mine),
                               "est_join_min": round(15 + 0.8 * len(members))})
        seen = {i for g in groups for i in g["rows"]}
        for x in flagged:  # a row groups_of did not place: a group of its own
            if x["id"] not in seen:
                groups.append({"key": x["id"], "rows": [x["id"]], "members": 0, "countries": [x["iso2"]], "backup": [x["iso2"]],
                               "people": x["people"], "built_m2": x["built_m2"], "est_join_min": 30})
        groups.sort(key=lambda g: (-g["people"], -g["built_m2"]))
    timeouts = [i for i, r in res.items() if r.get("timeout")]
    errors = {i: r["error"] for i, r in res.items() if r.get("error")}
    out = {"scanned": len(res), "coastal": sum(1 for r in res.values() if r.get("coastal")), "with_pieces": len(rows),
           "flag_rule": f"people >= {FLAG_PEOPLE} or built >= {FLAG_BUILT_M2} m2", "groups": groups,
           "hand_packs_flagged": hand, "rows": rows, "timeouts": timeouts, "errors": errors}
    json.dump(out, open(os.path.join(outdir, "flagged.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"scanned {len(res)}, coastal {out['coastal']}, with pieces {len(rows)}, flagged plan rows {len(flagged)} "
          f"in {len(groups)} solve units, flagged hand packs {len(hand)}, timeouts {len(timeouts)}, errors {len(errors)}")
    for g in groups[:12]:
        print(f"  group {g['key']}: {','.join(g['rows'])} ({','.join(g['countries'])}): {g['people']:,} people, "
              f"{g['built_m2'] / 1e6:.2f} km2 built, {g['members']} packs, join about {g['est_join_min']} min")
    for x in hand[:10]:
        print(f"  hand pack {x['id']}: {x['people']:,} people, {x['km2']} km2 (not a plan row: not rebuilt)")
    return 0


def main(a):
    if a[0] == "work":
        return work(int(a[1]), int(a[2]), a[3])
    if a[0] == "summary":
        return summary(a[1])
    if a[0] == "run":
        nw = int(a[a.index("--workers") + 1]) if "--workers" in a else 3
        lim = float(a[a.index("--limit-s") + 1]) if "--limit-s" in a else 900.0
        only = set(a[a.index("--ids") + 1].split(",")) if "--ids" in a else None
        return run(a[1], nw, lim, only)
    raise SystemExit(__doc__)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]) or 0)
