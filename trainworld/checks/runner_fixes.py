"""T-1976, T-1986: the world runner's market and cache fixes, without writing anything live.

    ..\\venv\\Scripts\\python.exe checks/runner_fixes.py      (from trainworld; about a minute)

1. pair_files' index: a rewritten file is read again; market_zones.Groups for named packs gives
   those packs exactly what Groups for every pack gives them (made-up files in a temp folder).
2. clear_row_caches clears an old `data/terrain/<id>.terrain` (and data/work caches), lookalikes kept
   (a temp root).
3. heal_market picks a registered pack naming a gone place over 400 km away (a copy of the
   registry, the market step recorded, not run; the live market packs only read).
4. A run stops at its start when its gate port is taken (a dummy server), before anything runs.
One PASS or FAIL line each.
"""
import json
import math
import os
import socket
import sys
import tempfile
import time
from pathlib import Path

TW = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TW / "pipeline"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
fails = 0


def report(name, ok, detail=""):
    global fails
    fails += not ok
    print(f"{'PASS' if ok else 'FAIL'} {name}{': ' + detail if detail else ''}", flush=True)


def write(p, doc, t):
    p.write_text(json.dumps(doc), encoding="utf-8")
    os.utime(p, (t, t))


def pairs_check():
    import pair_files
    import market_zones as M
    with tempfile.TemporaryDirectory() as d:
        work = Path(d)
        ids = ["a", "b", "c", "d"]
        j = lambda solved, packs, prs, t: {"solved": solved, "packs": packs, "solved_at": t, "source": f"joint {t}",  # noqa: E731
                                           "home_pack": [packs.index(h) for h, _ in prs], "work_pack": [packs.index(w) for _, w in prs],
                                           "home_zone": [[k, 0] for k in range(len(prs))], "work_zone": [[0, k] for k in range(len(prs))],
                                           "trips": [1.5 + k for k in range(len(prs))]}
        write(work / "one.joint_commutes.json", j(["a", "b"], ["a", "b", "c"], [("a", "b"), ("b", "a"), ("a", "c"), ("c", "a"), ("a", "b")], 1), 1e9)
        write(work / "two.joint_commutes.json", j(["b"], ["b", "c", "d"], [("b", "c"), ("c", "b"), ("d", "b")], 2), 1e9 + 1)
        write(work / "three.joint_commutes.json", j(["d"], ["c", "d"], [("c", "d"), ("d", "c")], 3), 1e9 + 2)
        e = lambda solved, packs, rows, t: {"solved": solved, "packs": packs, "solved_at": t, "source": f"everyday {t}",  # noqa: E731
                                            "home_pack": [packs.index(h) for h, _, _ in rows], "act_pack": [packs.index(a) for _, a, _ in rows],
                                            "kind": [k for _, _, k in rows], "home_zone": [[k, 1] for k in range(len(rows))],
                                            "act_zone": [[1, k] for k in range(len(rows))], "trips": [2.0] * len(rows), "walk": [0.5] * len(rows)}
        write(work / "u1.cross_everyday.json", e(["a", "b", "c"], ["a", "b", "c"], [("a", "b", 0), ("b", "a", 2), ("c", "a", 3), ("a", "c", 1)], 1), 1e9)
        write(work / "u2.cross_everyday.json", e(["c", "d"], ["a", "c", "d"], [("c", "d", 0), ("a", "d", 2), ("d", "c", 0)], 2), 1e9 + 1)
        reg = {"cities": [{"id": i} for i in ids]}
        whole = M.Groups(reg, work=work)
        first = json.loads((work / pair_files.NAME).read_text(encoding="utf-8"))
        same = all(getattr(M.Groups(reg, {i}, work=work), a).get(i) == getattr(whole, a).get(i)
                   for i in ids for a in ("rows", "source", "everyday_rows", "everyday_source"))
        same &= M.Groups(reg, {"a", "d"}, work=work).rows.keys() <= {"a", "d"}
        report("Groups for named packs as for every pack", same and bool(whole.rows) and bool(whole.everyday_rows),
               f"{sum(len(v) for v in whole.rows.values())} commute and {sum(len(v) for v in whole.everyday_rows.values())} everyday rows")
        write(work / "three.joint_commutes.json", j(["d"], ["a", "c", "d"], [("d", "c"), ("d", "a")], 3), 1e9 + 5)
        again = M.Groups(reg, {"d"}, work=work)
        e3 = json.loads((work / pair_files.NAME).read_text(encoding="utf-8"))["files"]["three.joint_commutes.json"]
        report("a rewritten file is read again",
               first["files"]["three.joint_commutes.json"]["homes"] == ["c", "d"] and e3["homes"] == ["d"]
               and again.rows.get("d") == [((0, 0), "c", (0, 0), 1.5), ((1, 0), "a", (0, 1), 2.5)],
               f"d's rows now {again.rows.get('d')}")


def caches_check():
    import world_run
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "data/terrain").mkdir(parents=True)
        (root / "data/work").mkdir(parents=True)
        made = [root / "data/terrain/zzq.terrain", root / "data/work/zzq.outline.geojson", root / "data/work/zzq.units.geojson"]
        keep = [root / "data/terrain/zzqa.terrain", root / "data/terrain/world.water", root / "data/work/zzq.cells.json"]
        for p in made + keep:
            p.write_text("x")
        world_run.Run({"jobs": 1, "root": str(root)}).clear_row_caches("zzq")
        left, kept = [p.name for p in made if p.exists()], [p.name for p in keep if p.exists()]
        report("clear_row_caches clears an old terrain grid", not left and len(kept) == len(keep), f"left {left}, kept {len(kept)} of {len(keep)}")


def heal_check():
    import world_run
    run = world_run.Run({"jobs": 1})
    reg = run.registry()
    by_place = {c["place"]: c for c in reg["cities"] if "place" in c}
    km = lambda a, b: math.hypot((a[0] - b[0]) * math.cos(math.radians((a[1] + b[1]) / 2)), a[1] - b[1]) * 111.2  # noqa: E731
    far = None
    for c in reg["cities"]:
        p = run.packs / f"{c['id']}.market.json"
        if p.exists():
            m = json.loads(p.read_text(encoding="utf-8"))
            for q in m.get("commute_place", []) + m.get("everyday_place", []):
                o = by_place.get(q)
                if o and o["id"] != c["id"] and km(c["center"], o["center"]) > 400:
                    far = (c["id"], o["id"], km(c["center"], o["center"]))
                    break
        if far:
            break
    if not far:
        report("heal_market reaches past 400 km", False, "no registered pack names a place over 400 km away to test with")
        return
    reg["cities"] = [c for c in reg["cities"] if c["id"] != far[1]]
    run.registry = lambda: reg
    calls = []
    run.step = lambda name, args, log, **kw: (calls.append((name, args)), (True, 0, None))[1]
    t = time.time()
    run.heal_market()
    got = calls[0][1] if calls else []
    report("heal_market reaches past 400 km", far[0] in got,
           f"{far[1]} taken out, {far[0]} {far[2]:.0f} km away picked; {len(got)} packs to rewrite, {time.time() - t:.0f} s")


def port_check():
    import world_run
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen()
    port = s.getsockname()[1]
    run = world_run.Run({"jobs": 1, "port": port})
    ran = []
    run.select = lambda: [{"id": "zzq"}]
    run.check_sim = lambda: ran.append("check_sim")
    run.start = lambda: ran.append("start")
    try:
        run.go()
        msg = "go() went on"
    except SystemExit as x:
        msg = str(x)
    finally:
        s.close()
    report("a run refuses a busy port at its start", "already taken" in msg and not ran, f"{msg[:80]}; ran {ran}")


pairs_check()
caches_check()
heal_check()
port_check()
sys.exit(1 if fails else 0)
