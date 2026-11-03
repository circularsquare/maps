"""T-1976, T-1986: the world runner's market and cache fixes, without writing anything live.

    ..\\venv\\Scripts\\python.exe checks/runner_fixes.py      (from trainworld; about a minute)

1. pair_files' index: a rewritten file is read again; market_zones.Groups for named packs gives
   those packs exactly what Groups for every pack gives them (made-up files in a temp folder).
2. clear_row_caches clears an old `data/terrain/<id>.terrain` (and data/work caches), lookalikes kept
   (a temp root).
3. heal_market picks a registered pack naming a gone place over 400 km away (a copy of the
   registry, the market step recorded, not run; the live market packs only read).
4. A run stops at its start when its gate port is taken (a dummy server), before anything runs.
5. T-2040: pack_gate.browser_run with a stand-in for city-browser.mjs: a zone over 20M people gets
   a session of its own; a session that stops printing is stopped, its stuck zone tried once alone
   (and failed when it hangs alone too), the zones after it checked in a fresh session.
6. T-1979: netget retries a reset connection and a 503 in place; a 404 asked for comes back; the
   World Bank figures go through it (a local server, a temp folder).
7. T-1978, T-1980: the region pipeline with stand-in steps (nothing built, no lock taken): region
   k+1 reads while region k builds and not before, builds after k's builds; rows gated in chunks
   while builds go on, each row once, never two gates at once; a gate's run fault stops the run.
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


FAKE_BROWSER = r"""
// a stand-in for city-browser.mjs: each zone's result after a line; "hang*" zones print nothing and never end
import { readFileSync, writeFileSync, appendFileSync } from 'node:fs';
const rows = JSON.parse(readFileSync(process.env.ROWS, 'utf8'));
appendFileSync(process.env.SESSIONS, rows.map(r => r.id).join(',') + '\n');
for (const r of rows) {
  if (r.id.startsWith('hang')) { setInterval(() => {}, 1000); await new Promise(() => {}); }
  console.log(`.. ${r.id}: checking`);
  await new Promise(ok => setTimeout(ok, 300));
  writeFileSync(r.json, JSON.stringify({ city: r.id, checks: Array.from({ length: 8 }, (_, k) => ({ name: `c${k}`, ok: true })), tries: 1 }));
  console.log(`PASS city-browser ${r.id}`);
}
"""


def browser_check():
    import pack_gate
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        script, sessions = d / "fake-browser.mjs", d / "sessions.txt"
        script.write_text(FAKE_BROWSER, encoding="utf-8")
        os.environ["SESSIONS"] = str(sessions)
        ids = ["a1", "big", "a2", "a3", "hang1", "a4", "a5", "a6"]
        rows = [{"id": i, "row": "", "out": str(d / f"{i}-"), "json": str(d / f"{i}.browser.json"),
                 "people": 30e6 if i == "big" else 1e6} for i in ids]
        t = time.time()
        old = pack_gate.BROWSER_SESSIONS
        pack_gate.BROWSER_SESSIONS = 2
        try:
            got = pack_gate.browser_run(rows, str(d), stall_s=4, script=str(script))
        finally:
            pack_gate.BROWSER_SESSIONS = old
        secs = time.time() - t
        ran = [ln.split(",") for ln in sessions.read_text(encoding="utf-8").split()]
        ok = (["big"] in ran and ["hang1"] in ran and got["hang1"]["pass"] is False
              and all(got[i]["pass"] for i in ids if i != "hang1") and secs < 60
              and any(p and p[0] == "a4" for p in ran) and list(d.glob("browser-stalled-hang1-*.log")))
        report("a stuck browser session costs minutes, not the gate", ok,
               f"sessions {ran}; hang1 {got['hang1'].get('error', '')[:60]!r}; {secs:.0f} s")


def netget_check():
    import http.server
    import threading
    import netget
    import tier0
    hits = {"n": 0}

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            hits["n"] += 1
            if self.path.startswith("/reset") and hits["n"] <= 2:
                self.connection.shutdown(2)  # a reset, as the World Bank's under 9 builds at once
                return
            if self.path.startswith("/busy") and hits["n"] <= 1:
                self.send_response(503)
                self.end_headers()
                return
            if self.path.startswith("/none"):
                self.send_response(404)
                self.end_headers()
                return
            body = json.dumps([{"page": 1}, [{"date": "2023", "indicator": {"id": "SL.EMP.TOTL.SP.ZS"}, "value": 60.0},
                                              {"date": "2023", "indicator": {"id": "SP.POP.0014.TO.ZS"}, "value": 25.0}]]).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    old = netget.WAITS
    netget.WAITS = (0.2, 0.2, 0.2)
    oldraw, oldwb = tier0.raw, tier0.WORLDBANK
    try:
        with tempfile.TemporaryDirectory() as d:
            r1 = netget.request(base + "/reset", waits=netget.WAITS).json()
            n1, hits["n"] = hits["n"], 0
            r2 = netget.request(base + "/busy", waits=netget.WAITS).status_code
            hits["n"] = 0
            r3 = netget.request(base + "/none", ok=(404,)).status_code
            try:
                netget.request(base + "/none")
                r4 = "no error"
            except Exception as x:  # noqa: BLE001
                r4 = type(x).__name__
            # tier0's World Bank figures through it, cached once
            tier0.raw = lambda *p: os.path.join(d, *p)
            tier0.WORLDBANK = base + "/reset?{iso2}"
            hits["n"] = 0
            wb = tier0.worldbank("ZZ")
            cached = os.path.exists(os.path.join(d, "worldbank_ZZ.json"))
        ok = isinstance(r1, list) and n1 == 3 and r2 == 200 and r3 == 404 and r4 == "HTTPError" and cached and abs(wb[0] - 0.45) < 1e-9
        report("netget retries in place; the World Bank figures go through it", ok,
               f"reset then answer after {n1} asks, 503 then {r2}, 404 {r3}/{r4}, worldbank {wb}")
    finally:
        netget.WAITS = old
        tier0.raw, tier0.WORLDBANK = oldraw, oldwb
        srv.shutdown()


def pipeline_check():
    import contextlib
    import threading
    import world_run
    events, lock = [], threading.Lock()

    def ev(*e):
        with lock:
            events.append((time.time(),) + e)

    def make(regions, fault_region=None):
        with tempfile.TemporaryDirectory() as d:
            run = world_run.Run({"jobs": 4, "root": d, "no_join": True})
            rows, order = [], 0
            for k, n in enumerate(regions, 1):
                for j in range(n):
                    order += 1
                    rows.append({"id": f"r{k}x{j:02d}", "order": order, "extracts": [f"ex{k}"], "status": "planned", "people": 1e6, "name": "x"})
            run.held = lambda *a, **kw: contextlib.nullcontext()
            run.registry = lambda: {"cities": []}
            run.built_ok = lambda cid: False
            run.downloads = lambda exs, admin: (ev("download", exs[0]), time.sleep(0.2), {})[2]
            run.made = lambda *a, **kw: {}
            run.clean = lambda exs: (ev("clean", exs[0]), [])[1]

            def step(name, args, log, **kw):
                if name == "@read_extract":
                    ev("read", args[0], "start")
                    time.sleep(0.6)
                    ev("read", args[0], "end")
                return True, 0.0, None
            run.step = step

            def build_row(r):
                if run.stop_asked():  # as the real one
                    return None
                ev("build", r["id"], "start")
                time.sleep(0.25)
                ev("build", r["id"], "end")
                return True
            run.build_row = build_row
            busy = []

            def gate(ids, L):
                busy.append(1)
                ev("gate", tuple(ids), "start", len(busy))
                time.sleep(0.4)
                if fault_region and ids[0].startswith(f"r{fault_region}x"):
                    busy.pop()
                    ev("fault")
                    raise world_run.RunFault("a stand-in gate fault")
                ev("gate", tuple(ids), "end")
                busy.pop()
                return list(ids)
            run.gate = gate
            regs = run.regions(rows)
            try:
                got = run.run_regions(regs)
                return got, None, rows
            except world_run.RunFault as x:
                return None, x, rows

    old = world_run.GATE_CHUNK
    world_run.GATE_CHUNK = 5
    try:
        got, fault, rows = make([16, 8, 3])

        def when(kind, pred):
            ts = [t for t, *e in events if e[0] == kind and pred(e)]
            return (min(ts), max(ts)) if ts else (None, None)
        r1b = when("build", lambda e: e[1].startswith("r1x"))
        r2b = when("build", lambda e: e[1].startswith("r2x"))
        r3b = when("build", lambda e: e[1].startswith("r3x"))
        r2r = when("read", lambda e: e[1] == "ex2")
        r3r = when("read", lambda e: e[1] == "ex3")
        gates = [e for t, *e in events if e[0] == "gate" and e[2] == "start"]
        gated = [i for g in gates for i in g[1]]
        early = [g for g in gates if g[1][0].startswith("r1x") and when("gate", lambda e, g=g: e[1] == g[1] and e[2] == "start")[0] < r1b[1]]
        ok = (sorted(got) == sorted(r["id"] for r in rows) and sorted(gated) == sorted(got)
              and r1b[0] < r2r[0] < r1b[1] and r2b[0] >= r1b[1] - 0.05 and r3r[0] >= r2b[0] - 0.05 and r3b[0] >= r2b[1] - 0.05
              and early and max(g[3] for g in gates) == 1)
        report("regions pipelined: read ahead one, gate in chunks while building", ok,
               f"{len(gates)} gate chunks ({len(early)} of region 1 during its builds), each row gated {len(gated) / max(1, len(got)):.0f}x; "
               f"region 2 read {r2r[0] - r1b[0]:.1f} s after region 1's builds began, built {r2b[0] - r1b[1]:+.1f} s after they ended")
        events.clear()
        got, fault, rows = make([6, 6, 6], fault_region=2)
        at = min(t for t, *e in events if e[0] == "fault")
        later = [e[1] for t, *e in events if e[0] == "build" and e[2] == "start" and t > at]
        report("a gate's run fault stops the run", fault is not None and not later
               and not any(e[2] == "start" and str(e[1][0]).startswith("r3x") for t, *e in events if e[0] == "gate"),
               f"fault {fault!s:.40}; builds started after it {later}")
    finally:
        world_run.GATE_CHUNK = old


pairs_check()
caches_check()
heal_check()
port_check()
browser_check()
netget_check()
pipeline_check()
sys.exit(1 if fails else 0)
