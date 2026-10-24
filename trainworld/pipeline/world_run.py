"""T-540, pipeline stage 8: the runner. Builds world plan rows into shipped or parked packs with no
agent judgement (docs/adding-a-city.md "Running the pipeline"; design notes/T-535.md).

    python -I pipeline/world_run.py plan   [select]            what a run would do; nothing built
    python -I pipeline/world_run.py run    [select] [options]  build, gate, register, join, clean up
    python -I pipeline/world_run.py join   [--countries ..] [--jobs N]
                                                               the joint solve for every gated row
    python -I pipeline/world_run.py status                     rows by status, parked reasons, locks

    select   --rows a,b,c  (in plan order, any status but shipped or gated) | --countries GB,PT |
             --limit N (the next N planned or half-done rows in plan order; default 10)
    options  --jobs N         cores in all (default 2; 12 only on an overnight run Anita approves)
             --workers W      rows built at once (default N, each step on one thread: a build
                              step uses about 0.9 cores whatever threads it is given, T-966)
             --port P         the runner's own Vite for the gate's browser check (default 6360);
                              runners at once need a port each
             --no-browser     gate without the browser check
             --no-join        stop after the regions: gated rows stay registered, status "gated",
                              for one `join` once every parallel run is done (T-786)
             --build-only     download, read and build only: rows stay "built", for a later run
                              to gate (a rented machine's wave, T-1065; docs/aws.md)
             --keep-downloads keep the region's .osm.pbf and OSM caches
             --coarse T       4 km blocks where every 2 km zone makes under T trip ends a weekday
                              (T-852, coarse_zones.py, a step before routes); off by default
             --root DIR       a scratch tree instead of this one (below)

Several runs at once (T-786): each claims its rows (a row claimed by a live run is not selected),
holds its region's extracts, and takes the shared_lock locks for every read-modify-write of the
plan, the registry and market rows, and made.json. Gates overlap; a joint solve holds the "packs"
lock and waits for gates under way, and no gate starts while it is held: a gate's snapshot would
see the solve's changes.

A run goes a region at a time: rows sharing a Geofabrik extract, chained, in plan order. For each:
download (or reuse) and read its extracts, then per row build_city, places, routes, merge_cells,
nocar, everyday_rates (status "built"); then the gate on every built row, registered with its
market pack against a snapshot taken before the region (status "shipped", or "parked" with the
failing checks, unregistered again); then the region's downloads are deleted. After the last
region, the joint gravity groups the batch's gated packs joined, one solve a group for the whole
batch (T-723), the group re-gated after it (a FAIL restores the group and parks its new packs);
then market_zones and local trips, and a freeze of the data for publishing (pipeline/freeze.py,
T-928). Status and a record of
times live in each row of pipeline/world_plan.json; a report a batch goes to
../artifacts/trainworld/world_run/<batch>/.

Stop: ctrl-c, or create trainworld/data/work/world_run/STOP (`status` and the run's first line print
the path; with --root, that tree's). The city under way finishes its steps first; gating and joining
wait for the next run, which takes up rows where they were. A run fault (the TW_SIM_DIR copy without
its `data`/`app` junctions, a gate or re-gate that leaves no result) stops the run without parking
anything for it. A step whose log has not grown for QUIET_S dumps its stack into its log. The run
ends by printing `status`.

Each step runs in its own process (`world_run.py step ...`, `python -I`), with OMP_NUM_THREADS and
CARGO_BUILD_JOBS from --jobs; TW_SIM_DIR and TW_CARGO_TARGET_DIR pass through, for a copy of sim/
while the tree's own is mid-edit. With --root, every step's paths into data/packs, data/work,
app/ and the world plan point into that tree instead (data/raw there should be a junction to this
tree's, so caches are shared; sim/ a copy, for the gate's rates test), and the browser check is
off, since Vite serves only this tree.
"""
import ast
import contextlib
import datetime
import importlib
import importlib.abc
import importlib.machinery
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePath

HERE = Path(os.path.dirname(os.path.abspath(__file__)))
REAL = HERE.parent
MAPS = REAL.parent
OSM = REAL / "data/raw/osm"
GEOFABRIK = "https://download.geofabrik.de/{}-latest.osm.pbf"
UA = {"User-Agent": "trainworld-pipeline/0.1 (rail game data build; python-requests)"}
BUILD_STEPS = ["build", "stations", "places", "routes", "merge", "nocar", "everyday"]
VITE_FILES = (".json", ".bin", ".boundary.geojson", ".places.json", ".market.json")  # vite.config.ts copies these
CACHES = (".points.fgb", ".stations.fgb", ".areas.fgb", ".water.fgb", ".ways2.npz")  # an extract's OSM caches
QUIET_S = 1200  # a step whose log has not grown this long dumps its stack into the log (T-764)
MODULE_OUTLINES = {"england", "france", "germany", "usa"}  # plan_hooks giving a boundary_file (T-555)
# A step still running after this long is stopped and its row parks (four times as long for a row
# over BIG_PEOPLE): the longest seen, a 20M pack's build, takes under an hour on 3 cores.
STEP_LIMIT_S = {"build_city": 5400, "routes": 3600, "pack_gate": 5400, "intercity": 7200, None: 1800}
BIG_PEOPLE = 15e6


class RunFault(Exception):
    """A fault of the run, not of a row (T-764, T-773): the run stops and parks nothing for it."""


def shared():
    """pipeline/shared_lock (the runner runs with -I, so the pipeline is not on the path)."""
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import shared_lock
    return shared_lock


# ================================================================ a step, in its own process

def _norm(p):
    return os.path.normcase(os.path.normpath(str(p)))


class Redirect:
    """Points every loaded pipeline module's paths into data/packs, data/work, app/ and the world
    plan of this tree at a scratch tree instead; the tree root itself too, so paths a module joins
    at run time land there. Module constants, and function defaults, are rewritten as each module
    finishes loading."""
    KEEP = {("osm_units", "PACKS")}  # the real packs, which a new outline is cut along

    def __init__(self, root):
        self.root = os.path.normpath(str(root))
        self.rootn = _norm(root)
        r = os.path.normpath(str(REAL))
        self.real = _norm(r)
        self.prefixes = [(_norm(os.path.join(r, *p.split("/"))), os.path.join(self.root, *p.split("/")))
                         for p in ("data/packs", "data/work", "app", "pipeline/world_plan.json", "pipeline/borders.json")]
        self.code = [_norm(HERE), _norm(REAL / "checks")]

    def value(self, v):
        if isinstance(v, (str, PurePath)) and len(str(v)) >= len(self.real):
            o = os.path.normpath(str(v))
            s = os.path.normcase(o)
            new = None
            if s == self.rootn or s.startswith(self.rootn + os.sep):
                return v  # already in the scratch tree (a scratch tree inside data/work, say)
            if s == self.real:
                new = self.root
            else:
                for src, dst in self.prefixes:
                    if s == src or s.startswith(src + os.sep):
                        new = dst + o[len(src):]
                        break
            if new is not None:
                return type(v)(new) if isinstance(v, PurePath) else new
        elif isinstance(v, tuple):
            return tuple(self.value(x) for x in v)
        return v

    def module(self, m):
        for k, v in list(vars(m).items()):
            if k.startswith("__") or (m.__name__, k) in self.KEEP:
                continue
            if callable(v) and getattr(v, "__module__", None) == m.__name__ and hasattr(v, "__defaults__"):
                if v.__defaults__:
                    v.__defaults__ = self.value(v.__defaults__)
                if v.__kwdefaults__:
                    v.__kwdefaults__ = {a: self.value(x) for a, x in v.__kwdefaults__.items()}
                continue
            nv = self.value(v)
            if nv is not v:
                setattr(m, k, nv)
        if m.__name__ == "intercity" and isinstance(getattr(m, "REGISTRY", None), dict):
            m.REGISTRY = json.loads((Path(self.root) / "app/src/game/registry.json").read_text(encoding="utf-8"))

    def ours(self, origin):
        o = _norm(origin)
        return any(o.startswith(c + os.sep) for c in self.code)

    def install(self):
        red = self

        class Finder(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path, target=None):
                spec = importlib.machinery.PathFinder.find_spec(name, path)
                if spec is None or not spec.origin or not red.ours(spec.origin) or spec.loader is None:
                    return None
                run = spec.loader.exec_module

                def exec_module(module, run=run):
                    run(module)
                    red.module(module)
                spec.loader.exec_module = exec_module
                return spec
        sys.meta_path.insert(0, Finder())
        for name, m in list(sys.modules.items()):
            if name != "__main__" and getattr(m, "__file__", None) and self.ours(m.__file__):
                self.module(m)


def run_main(modname, args):
    """A pipeline script's `if __name__ == "__main__":` block, run in its already imported (and,
    with --root, redirected) module, with sys.argv as the script would see it."""
    m = importlib.import_module(modname)
    tree = ast.parse(Path(m.__file__).read_text(encoding="utf-8"))
    blocks = [n for n in tree.body if isinstance(n, ast.If) and isinstance(n.test, ast.Compare)
              and isinstance(n.test.left, ast.Name) and n.test.left.id == "__name__"]
    if not blocks:
        raise SystemExit(f"{modname}: no __main__ block")
    sys.argv = [m.__file__] + list(args)
    exec(compile(ast.Module(body=blocks[-1].body, type_ignores=[]), m.__file__, "exec"), m.__dict__)


def step_main(argv):
    """`step <script> args...` runs a pipeline or checks script; `step @<name> args...` one of
    the runner's own pieces below."""
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8")
    if os.environ.get("TW_FAULT_S"):  # a step that hangs silently: every N seconds its stack goes to its log
        import faulthandler
        faulthandler.dump_traceback_later(int(os.environ["TW_FAULT_S"]), repeat=True, file=sys.stderr)
    elif os.environ.get("TW_STEP_LOG"):
        quiet_watch(os.environ["TW_STEP_LOG"], int(os.environ.get("TW_QUIET_S", QUIET_S)))
    sys.path[:0] = [str(HERE), str(REAL / "checks")]
    root = os.environ.get("TW_RUN_ROOT")
    if root:
        Redirect(root).install()
    name, args = argv[0], argv[1:]
    if not name.startswith("@"):
        return run_main(name, args)
    out = {"@read_extract": read_extract, "@ne_gap": ne_gap, "@groups": groups_of,
           "@preview": preview}[name](args)
    if out is not None:
        print("RESULT " + json.dumps(out))


def quiet_watch(log, secs):
    """Every thread's stack into the step's log once the log has not grown for `secs`, and again
    after each further `secs` of silence. faulthandler's timer is C, so it fires even while C code
    holds the GIL; a thread re-arms it whenever the log grows."""
    import faulthandler
    faulthandler.dump_traceback_later(secs, repeat=True, file=sys.stderr)

    def watch():
        size = -1
        while True:
            time.sleep(min(30, secs / 4))
            try:
                s = os.path.getsize(log)
            except OSError:
                continue
            if s != size:
                size = s
                faulthandler.dump_traceback_later(secs, repeat=True, file=sys.stderr)
    threading.Thread(target=watch, daemon=True).start()


def read_extract(args):
    """Every cache the region's builds read from one extract, made once before the builds."""
    ex, admin = args[0], "--admin" in args
    import osm_features
    import osm_water
    import routes
    osm_features.build(ex)
    osm_water.read_inland(ex)
    routes.read_ways(ex, keep_pbf=True)
    if admin:
        import osm_units
        osm_units.admin_areas(ex)


def ne_gap(ids):
    """{id: km to the nearest pack with a Northeast market id, if within GAP_KM}: such a pack would
    stop the joint gravity (T-580)."""
    import intercity
    reg = json.loads((intercity.ROOT / "app/src/game/registry.json").read_text(encoding="utf-8"))
    ne = [c["id"] for c in reg["cities"] if c.get("market", -1) >= 0]
    shapes = {i: intercity.outline(intercity.ROOT / f"data/packs/{i}.boundary.geojson") for i in ne}
    out = {}
    for i in ids:
        g = intercity.outline(intercity.ROOT / f"data/packs/{i}.boundary.geojson")
        near = [(intercity.gap_km(g, s), j) for j, s in shapes.items()
                if s.bounds[0] - 3 < g.bounds[2] and g.bounds[0] - 3 < s.bounds[2]
                and s.bounds[1] - 3 < g.bounds[3] and g.bounds[1] - 3 < s.bounds[3]]
        near = [x for x in near if x[0] <= intercity.GROUP_GAP_KM]
        if near:
            d, j = min(near)
            out[i] = [round(d, 1), j]
    return out


def groups_of(ids):
    """The solve units (intercity.solve_units, T-786) holding any of these ids."""
    import intercity
    reg = json.loads((intercity.ROOT / "app/src/game/registry.json").read_text(encoding="utf-8"))
    return intercity.solve_units(reg, intercity.ROOT, only=ids)


def preview(ids):
    """For a dry run: each id's planned extent against the registered packs and the other ids,
    within GAP_KM (the groups it would join), and Northeast packs that near (T-580)."""
    import shapely
    from shapely.geometry import shape
    import intercity
    root = intercity.ROOT
    reg = json.loads((root / "app/src/game/registry.json").read_text(encoding="utf-8"))
    ext = json.loads((root / "data/work/worldplan/extents.geojson").read_text(encoding="utf-8"))
    plan = {f["properties"]["id"]: shapely.make_valid(shape(f["geometry"])) for f in ext["features"]}
    packs = {c["id"]: intercity.outline(root / f"data/packs/{c['id']}.boundary.geojson") for c in reg["cities"]
             if (root / f"data/packs/{c['id']}.boundary.geojson").exists()}
    ne = {c["id"] for c in reg["cities"] if c.get("market", -1) >= 0}
    out = {}
    for i in ids:
        g = plan.get(i)
        if g is None:
            out[i] = {"error": "no extent in data/work/worldplan/extents.geojson"}
            continue
        near = {}
        for j, s in list(packs.items()) + [(k, plan[k]) for k in ids if k != i and k in plan]:
            gb, sb = g.bounds, s.bounds
            if sb[0] - 2 < gb[2] and gb[0] - 2 < sb[2] and sb[1] - 2 < gb[3] and gb[1] - 2 < sb[3]:
                d = intercity.gap_km(g, s)
                if d <= intercity.GROUP_GAP_KM:
                    near[j] = round(d, 1)
        out[i] = {"near": near, "northeast": sorted(set(near) & ne)}
    return out


# ================================================================ the run, in the parent

class Run:
    def __init__(self, opts):
        self.o = opts
        self.root = Path(opts["root"]).resolve() if opts.get("root") else REAL
        self.scratch = self.root != REAL
        self.plan_path = self.root / "pipeline/world_plan.json"
        self.reg_path = self.root / "app/src/game/registry.json"
        self.packs = self.root / "data/packs"
        self.work = self.root / "data/work"
        self.stop_file = self.work / "world_run/STOP"
        self.batch = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        base = (self.root / "artifacts") if self.scratch else (MAPS / "artifacts/trainworld/world_run")
        self.logs = base / self.batch
        self.jobs = max(1, int(opts.get("jobs", 2)))
        self.workers = int(opts.get("workers") or self.jobs)
        self.threads = max(1, self.jobs // self.workers)
        self.browser = not opts.get("no_browser") and not self.scratch
        self.port = int(opts.get("port", 6360))
        self.lock = threading.Lock()
        self.streak = (None, 0)  # T-775: the last park's reason, as compared, and how many in a row
        self.stopping = False
        self.vite = None
        self.everyday = None  # the gate's rates test, built once a run (build_everyday)
        self.report = {"batch": self.batch, "root": str(self.root), "jobs": self.jobs, "rows": {}, "regions": []}

    # ---------------------------------------------------------------- files

    def plan_doc(self):
        return shared().read_json(self.plan_path)

    def rows(self):
        return [r for r in self.plan_doc()["metros"] if r.get("units")]

    def held(self, name, what, root=None):
        """A shared_lock lock of this tree (or `root`'s), held by this run."""
        return shared().held(name, f"world_run {self.batch}: {what}", root=root or self.root)

    def set_row(self, cid, status=None, **run):
        """One row's status and run record, the plan re-read first (other runs and tools write it too)."""
        with self.held("plan", f"row {cid}"):
            doc = shared().read_json(self.plan_path)
            row = next(r for r in doc["metros"] if r.get("id") == cid and r.get("units"))
            if status:
                row["status"] = status
            rec = row.setdefault("run", {})
            for k, v in run.items():
                if v is None:
                    rec.pop(k, None)
                elif k == "steps":
                    rec.setdefault("steps", {}).update(v)
                else:
                    rec[k] = v
            shared().write_atomic(self.plan_path, json.dumps(doc, indent=1, ensure_ascii=False) + "\n")

    def registry(self):
        return shared().read_json(self.reg_path)

    def unregister(self, ids):
        """Take packs out of the registry again (a parked pack): its entry and market pack go; its
        market place stays in `market.places`, unused (place ids are positions, append only)."""
        with self.held("registry", f"unregister {' '.join(ids)}"[:200]):
            reg = self.registry()
            before = len(reg["cities"])
            reg["cities"] = [c for c in reg["cities"] if c["id"] not in ids]
            if len(reg["cities"]) != before:
                shared().write_atomic(self.reg_path, json.dumps(reg, indent=2, ensure_ascii=False) + "\n")
                import market_rows  # shared() put the pipeline on the path (the runner runs with -I, T-779)
                market_rows.save(reg, market_rows.load(self.root), self.root)
            for i in ids:
                p = self.packs / f"{i}.market.json"
                if p.exists():
                    p.unlink()

    def stop_asked(self):
        return self.stopping or self.stop_file.exists()

    # ---------------------------------------------------------------- one step

    def step(self, name, args, log, threads=None, env=None, quiet=False, big=False):
        """(ok, seconds, the RESULT line's value or the log's last lines). A step outside a row's
        build gets every core of --jobs: a run's regions go one at a time, so its builds are idle."""
        t = threads or self.jobs
        e = dict(os.environ, OMP_NUM_THREADS=str(t), OPENBLAS_NUM_THREADS=str(t), MKL_NUM_THREADS=str(t),
                 RAYON_NUM_THREADS=str(t), CARGO_BUILD_JOBS=str(t), PYTHONIOENCODING="utf-8", **(env or {}))
        if self.scratch:
            e["TW_RUN_ROOT"] = str(self.root)
        log = Path(log)
        e["TW_STEP_LOG"] = str(log)
        log.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0  # ctrl-c stops the runner, not a step
        hush = int(e.get("TW_QUIET_S", QUIET_S))
        with open(log, "a", encoding="utf-8") as f:
            f.write(f"### {datetime.datetime.now().isoformat(timespec='seconds')} {name} {' '.join(args)}\n")
            f.flush()
            p = subprocess.Popen([sys.executable, "-I", "-u", os.path.abspath(__file__), "step", name, *args],
                                 stdout=f, stderr=subprocess.STDOUT, env=e, cwd=str(MAPS), creationflags=flags)
            limit = STEP_LIMIT_S.get(name, STEP_LIMIT_S[None]) * (4 if big else 1)
            size, still, said = -1, time.time(), False
            while True:
                try:
                    code = p.wait(timeout=min(30, hush / 4))
                    break
                except subprocess.TimeoutExpired:
                    pass
                now = time.time()
                if now - t0 >= limit:
                    self.kill(p)
                    code = None
                    break
                s = log.stat().st_size
                if s != size:
                    size, still, said = s, now, False
                elif not said and now - still >= hush:
                    print(f"    {name} {' '.join(args)[:60]}: no log growth for {hush / 60:.0f} min; its stack goes to {log}",
                          flush=True)
                    said = True
        secs = round(time.time() - t0, 1)
        text = log.read_text(encoding="utf-8", errors="replace")
        res = None
        for ln in text.splitlines():
            if ln.startswith("RESULT "):
                res = json.loads(ln[7:])
        if code is None:
            res = f"stopped after {limit / 60:.0f} minutes (STEP_LIMIT_S); last lines: " + " | ".join(text.rstrip().splitlines()[-3:])
        elif res is None:
            res = "\n".join(text.rstrip().splitlines()[-6:])
        if not quiet:
            print(f"    {name} {' '.join(a for a in args if len(a) < 60)[:80]}: "
                  f"{'ok' if code == 0 else 'TIMED OUT' if code is None else 'FAILED'} {secs:.0f} s", flush=True)
        return code == 0, secs, res

    @staticmethod
    def kill(p):
        """One step's process and its children only."""
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], capture_output=True)
        else:
            p.kill()
        p.wait(timeout=60)

    # ---------------------------------------------------------------- selection

    def claimed(self, r):
        """The batch of another run that is live and has claimed this row, else None (T-786)."""
        b = (r.get("run") or {}).get("claim")
        return b if b and b != self.batch and shared().is_held(f"batch-{b}", self.root) else None

    def select(self):
        rows = sorted(self.rows(), key=lambda r: r["order"])
        busy = {r["id"]: self.claimed(r) for r in rows if self.claimed(r)}
        if self.o.get("rows"):
            want = self.o["rows"].split(",")
            missing = [w for w in want if w not in {r["id"] for r in rows}]
            if missing:
                raise SystemExit(f"not plan rows: {missing}")
            gated = [r["id"] for r in rows if r["id"] in want and r["status"] == "gated"]
            if gated:
                print(f"gated, for `join`: {' '.join(gated)}")
            pick = [r for r in rows if r["id"] in want and r["status"] not in ("shipped", "gated")]
        else:
            pick = [r for r in rows if r["status"] in ("planned", "built")]
            if self.o.get("countries"):
                cs = set(self.o["countries"].split(","))
                pick = [r for r in pick if r.get("iso2") in cs]
        if busy:
            taken = [f"{r['id']} ({busy[r['id']]})" for r in pick if r["id"] in busy]
            if taken:
                print(f"claimed by a live run, left out: {' '.join(taken)}")
            pick = [r for r in pick if r["id"] not in busy]
        return pick if self.o.get("rows") else pick[:int(self.o.get("limit") or 10)]

    def claim(self, rows):
        """Mark the rows as this run's, under the plan lock; rows another live run claimed in the
        meantime are dropped. Returns the rows kept."""
        with self.held("plan", "claim rows"):
            doc = shared().read_json(self.plan_path)
            by = {r["id"]: r for r in doc["metros"] if r.get("units")}
            keep = []
            for r in rows:
                cur = by[r["id"]]
                if self.claimed(cur):
                    print(f"  {r['id']}: claimed by {self.claimed(cur)} meanwhile, left out", flush=True)
                    continue
                cur.setdefault("run", {})["claim"] = self.batch
                keep.append(r)
            shared().write_atomic(self.plan_path, json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
        return keep

    def unclaim(self):
        with self.held("plan", "release claims"):
            doc = shared().read_json(self.plan_path)
            for r in doc["metros"]:
                if (r.get("run") or {}).get("claim") == self.batch:
                    del r["run"]["claim"]
            shared().write_atomic(self.plan_path, json.dumps(doc, indent=1, ensure_ascii=False) + "\n")

    @staticmethod
    def regions(rows):
        """Rows sharing an extract, chained, each region in the plan order of its first row."""
        parent = list(range(len(rows)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        by = {}
        for i, r in enumerate(rows):
            for e in r["extracts"]:
                if e in by:
                    parent[find(i)] = find(by[e])
                by.setdefault(e, i)
        out = {}
        for i, r in enumerate(rows):
            out.setdefault(find(i), []).append(r)
        return sorted(out.values(), key=lambda g: g[0]["order"])

    # ---------------------------------------------------------------- dry run

    def dry(self):
        rows = self.select()
        regs = self.regions(rows)
        print(f"{len(rows)} rows in {len(regs)} regions; jobs {self.jobs}: {self.workers} at a time, {self.threads} threads each; "
              f"browser check {'on, Vite on ' + str(self.port) if self.browser else 'off'}; root {self.root}")
        ok, _, prev = self.step("@preview", [r["id"] for r in rows], self.work / "world_run/preview.log", threads=1, quiet=True)
        prev = prev if ok and isinstance(prev, dict) else {}
        for k, reg in enumerate(regs, 1):
            exs = sorted({e for r in reg for e in r["extracts"]})
            print(f"region {k}: {', '.join(exs)}")
            for e in exs:
                stem = OSM / e.replace("/", "_")
                have = [c for c in CACHES if Path(str(stem) + c).exists()]
                pbf = Path(str(stem) + ".osm.pbf").exists()
                print(f"  {e}: {'cached' if len(have) == len(CACHES) else ('pbf on disk' if pbf else 'to download')}"
                      f"{'; caches ' + ' '.join(have) if have and len(have) < len(CACHES) else ''}")
            for r in reg:
                p = prev.get(r["id"], {})
                near = ", ".join(f"{j} {d:g} km" for j, d in sorted(p.get("near", {}).items(), key=lambda x: x[1]))
                print(f"  {r['order']:>4} {r['id']} {r['name']} ({r.get('iso2')}, {r['people'] / 1e6:.2f}M, "
                      f"{sum(len(v) for v in r['units'].values())} units, {r['archetype']}{', ' + r['module'] if r.get('module') else ''})"
                      f" status {r['status']}{'; joins ' + near if near else ''}"
                      f"{'; PARKS (T-580): near the Northeast ' + ','.join(p['northeast']) if p.get('northeast') else ''}")

    # ---------------------------------------------------------------- the run

    def start(self):
        """The batch's log folder (a second run started in the same second gets its pid added) and
        its lock, held for the run's life: its row claims are live while it is held."""
        try:
            self.logs.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            self.batch += f"-{os.getpid()}"
            self.report["batch"] = self.batch
            self.logs = self.logs.parent / self.batch
            self.logs.mkdir(parents=True, exist_ok=True)
        return self.held(f"batch-{self.batch}", "the run")

    def go(self):
        if not self.select():
            print("nothing to build")
            return
        if self.stop_file.exists():
            raise SystemExit(f"{self.stop_file} exists: delete it to run")
        try:
            self.check_sim()
        except RunFault as x:
            raise SystemExit(f"RUN FAULT, nothing run: {x}")
        with self.start():
            rows = self.claim(self.select())
            if not rows:
                print("nothing to build: every row is claimed by another live run")
                return
            try:
                self.run_rows(rows)
            finally:
                self.unclaim()
        print()
        status(self.root)
        if "fault" in self.report:
            sys.exit(2)

    def run_rows(self, rows):
        signal.signal(signal.SIGINT, self.on_ctrl_c)
        t0 = time.time()
        regs = self.regions(rows)
        print(f"batch {self.batch}: {len(rows)} rows in {len(regs)} regions; jobs {self.jobs} ({self.workers} x {self.threads}); "
              f"{'no joint solve (--no-join); ' if self.o.get('no_join') else ''}"
              f"logs {self.logs}; to stop, create {self.stop_file}", flush=True)
        self.sim_vs_head()
        gated = []
        try:
            self.everyday = None if self.o.get("build_only") else self.build_everyday()
            for k, reg in enumerate(regs, 1):
                if self.stop_asked():
                    break
                gated += self.region(k, reg)
            if self.o.get("build_only"):
                built = [i for i in self.report["rows"] if self.status(i) == "built"]
                print(f"  built, for a run to gate: {' '.join(built) or 'none'}", flush=True)
            elif self.o.get("no_join"):
                # T-786: the regions of several runs at once, then one `join` for them all
                for i in gated:
                    self.set_row(i, "gated", gated=datetime.datetime.now().isoformat(timespec="seconds"))
                    self.rec(i, result="gated")
                if gated:
                    print(f"  gated, for `world_run.py join`: {' '.join(gated)}", flush=True)
            # one joint solve a group for the whole batch, after every region (T-723: a covered country's
            # group re-solved after each of its regions was most of the join's time)
            elif not self.stop_asked() and (gated or any(self.report["rows"].get(i, {}).get("result") == "parked"
                                                         for i in self.report["rows"])):
                t = time.time()
                with self.held("packs", "joint solve"):
                    self.wait_gates()
                    self.join(gated, self.logs / "join")
                    self.freeze()
                self.report["join_seconds"] = round(time.time() - t)
            elif gated:
                print(f"  stopped before the joint solve: {' '.join(gated)} stay registered and built; the next run gates them again",
                      flush=True)
        except RunFault as x:
            self.stop_vite()
            self.report["fault"] = str(x)
            self.stopping = True
            print(f"\nRUN FAULT, the run stops (no row parked for it): {x}", flush=True)
        finally:
            if self.everyday:
                self.everyday.unlink(missing_ok=True)
        self.report["seconds"] = round(time.time() - t0)
        self.report["stopped"] = self.stop_asked()
        self.write_report()

    def join_only(self):
        """T-786: the joint solve for every gated row (of --countries), once, after the parallel
        runs' regions; one at a time (the packs lock), on every core of --jobs."""
        rows = [r for r in self.rows() if r["status"] == "gated"]
        if self.o.get("countries"):
            cs = set(self.o["countries"].split(","))
            rows = [r for r in rows if r.get("iso2") in cs]
        if not rows:
            print("nothing gated to join")
            return
        if self.stop_file.exists():
            raise SystemExit(f"{self.stop_file} exists: delete it to run")
        try:
            self.check_sim()
        except RunFault as x:
            raise SystemExit(f"RUN FAULT, nothing run: {x}")
        with self.start():
            rows = self.claim(rows)
            ids = [r["id"] for r in rows]
            t0 = time.time()
            print(f"batch {self.batch}: the joint solve for {len(ids)} gated rows ({' '.join(ids)}); jobs {self.jobs}; logs {self.logs}",
                  flush=True)
            self.sim_vs_head()
            self.report["regions"].append({"n": 0, "extracts": [], "rows": ids, "seconds": {}})
            try:
                with self.held("packs", "joint solve"):
                    self.wait_gates()
                    have = {c["id"] for c in self.registry()["cities"]}
                    gone = [i for i in ids if i not in have]
                    for i in gone:  # gated but no longer registered: gate it again
                        self.set_row(i, "built")
                        print(f"  {i}: gated but not registered; back to built for the next run", flush=True)
                    ids = [i for i in ids if i in have]
                    if ids:
                        self.join(ids, self.logs / "join")
                        self.freeze()
            except RunFault as x:
                self.report["fault"] = str(x)
                print(f"\nRUN FAULT, the join stops (no row parked for it): {x}", flush=True)
            finally:
                self.unclaim()
            self.report["join_seconds"] = self.report["seconds"] = round(time.time() - t0)
            self.report["stopped"] = False
            self.write_report()
            print()
            status(self.root)
        if "fault" in self.report:
            sys.exit(2)

    def sim_vs_head(self):
        """T-775: the TW_SIM_DIR copy against HEAD's sim/ (pipeline scripts run from this tree, the
        sim from the copy; a stale copy broke T-613's merges). Differences are printed and go in
        the report's problems; each says whether the tree's own sim/ has the copy's version (a
        change not yet committed, carried on purpose)."""
        sim = os.environ.get("TW_SIM_DIR")
        if not sim:
            return
        import hashlib
        sim = Path(sim)
        out = subprocess.run(["git", "-C", str(REAL), "ls-tree", "-r", "HEAD", "sim"], capture_output=True, text=True)
        if out.returncode != 0:
            self.report.setdefault("problems", []).append(f"sim copy not checked: git ls-tree failed: {out.stderr.strip()[:200]}")
            return

        def blob(data):
            return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
        head, diff = {}, []
        for ln in out.stdout.splitlines():
            meta, rel = ln.split("\t", 1)
            head[rel] = meta.split()[2]
        for rel, sha in head.items():
            p = sim / rel[len("sim/"):]
            if not p.exists():
                diff.append((rel, "missing in the copy"))
                continue
            data = p.read_bytes()
            if blob(data) != sha and blob(data.replace(b"\r\n", b"\n")) != sha:
                diff.append((rel, "differs"))
        for top, dirs, files in os.walk(sim):
            if Path(top) == sim:
                dirs[:] = [d for d in dirs if d not in ("target", "pkg")]  # build output
            for f in files:
                rel = "sim/" + (Path(top) / f).relative_to(sim).as_posix()
                if rel not in head:
                    diff.append((rel, "not in HEAD"))
        if not diff:
            print(f"sim copy {sim}: same as HEAD's sim/", flush=True)
            return
        notes = []
        for rel, what in diff:
            tree = REAL / rel
            same = tree.exists() and (sim / rel[len("sim/"):]).exists() and tree.read_bytes() == (sim / rel[len("sim/"):]).read_bytes()
            notes.append(f"{rel} {what}{' (as in the tree: not committed yet)' if same else ''}")
        msg = f"sim copy {sim} is not HEAD's sim/ in {len(diff)} files: " + "; ".join(notes[:10])
        print("  " + msg, flush=True)
        self.report.setdefault("problems", []).append(msg)

    def check_sim(self):
        """The TW_SIM_DIR copy's `data` and `app` junctions beside it, which the gate's rates test and
        the joint solve read (T-616): missing, they fail every row, so the run stops instead (T-764)."""
        sim = os.environ.get("TW_SIM_DIR")
        if not sim:
            return
        sim, bad = Path(sim), []
        if not (sim / "Cargo.toml").exists():
            bad.append(f"{sim / 'Cargo.toml'} missing")
        for rel, want in (("data/packs", self.packs), ("app/src/game/registry.json", self.reg_path)):
            p = sim.parent / rel
            top, target = sim.parent / rel.split("/")[0], want.parents[len(rel.split("/")) - 2]
            if not p.exists() and not os.path.lexists(top) and os.name == "nt" and (sim / "Cargo.toml").exists():
                # junctions in a scratch folder have vanished mid-run twice, cause unknown (T-613, T-779):
                # put the same junction back and say so rather than stop a long run
                subprocess.run(["cmd", "/c", "mklink", "/J", str(top), str(target)], capture_output=True)
                print(f"  sim copy: junction {top} was missing, made again to {target}", flush=True)
                self.report.setdefault("problems", []).append(f"junction {top} made again at {datetime.datetime.now().isoformat(timespec='seconds')}")
            if not p.exists():
                bad.append(f"{p} missing")
            elif not os.path.samefile(p, want):
                bad.append(f"{p} is not {want}")
        if bad:
            raise RunFault("the TW_SIM_DIR copy cannot see this tree's packs: " + "; ".join(bad) +
                           " (junctions: docs/adding-a-city.md, Running the pipeline)")

    def build_everyday(self):
        """T-931: the gate's `every_pack_everyday` test binary, built once at the run's start from
        the sim the gate reads (TW_SIM_DIR, else the tree's sim/) and copied beside the run's locks,
        so every region's gate runs the same binary however sim/ changes meanwhile (a recompile
        took 10-65 minutes). None if it does not build: each gate then builds the test itself."""
        sim = Path(os.environ.get("TW_SIM_DIR") or self.root / "sim")
        log = self.logs / "everyday-test.log"
        e = dict(os.environ, CARGO_BUILD_JOBS=str(self.jobs))
        e["PATH"] = os.path.join(os.path.expanduser("~"), ".cargo", "bin") + os.pathsep + e.get("PATH", "")
        t0 = time.time()
        cargo = shutil.which("cargo", path=e["PATH"])
        if not cargo:
            print("  cargo not found: each gate builds the rates test itself", flush=True)
            return None
        with open(log, "w", encoding="utf-8") as f:
            p = subprocess.run([cargo, "test", "--release", "--features", "demand", "--lib", "--no-run",
                                "--message-format", "json-render-diagnostics", "--manifest-path", str(sim / "Cargo.toml")],
                               env=e, stdout=subprocess.PIPE, stderr=f, text=True, encoding="utf-8", errors="replace")
        exe = None
        for ln in p.stdout.splitlines():
            if ln.startswith("{"):
                m = json.loads(ln)
                if m.get("reason") == "compiler-artifact" and m.get("executable") and m.get("profile", {}).get("test"):
                    exe = m["executable"]
        if p.returncode != 0 or not exe:
            print(f"  the gate's rates test did not build ({log}); each gate builds it", flush=True)
            return None
        out = self.work / "world_run" / f"everyday-{self.batch}{Path(exe).suffix}"
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(exe, out)
        print(f"  the gate's rates test built from {sim} in {time.time() - t0:.0f} s", flush=True)
        return out

    def on_ctrl_c(self, *_):
        if not self.stopping:
            print("\nctrl-c: stopping once the city under way is done (the steps keep running)", flush=True)
        self.stopping = True

    def rec(self, cid, **kv):
        self.report["rows"].setdefault(cid, {}).update(kv)

    def region(self, k, reg):
        """One region, holding its extracts (T-786: another run sharing one waits, since a region
        deletes its downloads and caches at the end; data/raw is this tree's in a scratch run too)."""
        with contextlib.ExitStack() as st:
            for e in sorted({e for r in reg for e in r["extracts"]}):
                st.enter_context(self.held("extract-" + e.replace("/", "_"), f"region {k}", root=REAL))
            return self.region_held(k, reg)

    def region_held(self, k, reg):
        ids = [r["id"] for r in reg]
        exs = sorted({e for r in reg for e in r["extracts"]})
        rr = {"n": k, "extracts": exs, "rows": ids, "seconds": {}}
        self.report["regions"].append(rr)
        print(f"region {k}: {' '.join(ids)} ({', '.join(exs)})", flush=True)
        L = self.logs / f"region{k}"
        # a row left registered by a run that died mid-gate is taken out first
        stale = [i for i in ids if i in {c["id"] for c in self.registry()["cities"]}]
        if stale:
            print(f"  unregistering {stale}: not shipped", flush=True)
            self.unregister(stale)
        # downloads and reads, once a region
        todo = [r for r in reg if not (r["status"] == "built" and self.built_ok(r["id"]))]
        if todo:
            t = time.time()
            # osm_units' outline reads the admin relations, unless the country module draws its own
            admin = {e for r in todo if r.get("module") not in MODULE_OUTLINES for e in r["extracts"]}
            before = {e: self.caches(e) for e in exs}
            failed = self.downloads(exs, admin)
            rr["seconds"]["download"] = round(time.time() - t)
            t = time.time()
            for e in exs:
                if e in failed:
                    continue
                ok, _, tail = self.step("@read_extract", [e] + (["--admin"] if e in admin else []), f"{L}.read.log")
                if not ok:
                    failed[e] = tail
            self.made({e: sorted(set(self.caches(e)) - set(before[e])) for e in exs})
            rr["seconds"]["read"] = round(time.time() - t)
            for r in todo:
                bad = [e for e in r["extracts"] if e in failed]
                if bad:
                    self.park(r["id"], f"extract {bad[0]}: {str(failed[bad[0]])[-300:]}")
            todo = [r for r in todo if not any(e in failed for e in r["extracts"])]
            # builds, `workers` at a time
            with ThreadPoolExecutor(self.workers) as pool:
                list(pool.map(self.build_row, todo))
        if self.stop_asked():
            print("  stopped: the built rows are gated by the next run", flush=True)
            return []
        built = [r["id"] for r in reg if self.status(r["id"]) == "built"]
        shipped = []
        if built and not self.o.get("build_only"):
            t = time.time()
            shipped = self.gate(built, L)
            rr["seconds"]["gate"] = round(time.time() - t)
        if not self.o.get("keep_downloads"):
            rr["deleted"] = self.clean(exs)
        return shipped

    @staticmethod
    def caches(e):
        stem = str(OSM / e.replace("/", "_"))
        return [c for c in CACHES if Path(stem + c).exists()]

    def made(self, add=None, drop=()):
        """The OSM caches runs have made and not yet deleted, by extract (those already on disk
        before a run stay when it cleans up)."""
        p = REAL / "data/work/world_run/made.json"  # this tree's, as the caches are (data/raw is shared)
        with self.held("made", "made.json", root=REAL):
            made = shared().read_json(p) if p.exists() else {}
            for e, cs in (add or {}).items():
                made[e] = sorted(set(made.get(e, [])) | set(cs))
            for e in drop:
                made.pop(e, None)
            made = {e: cs for e, cs in made.items() if cs}
            p.parent.mkdir(parents=True, exist_ok=True)
            shared().write_atomic(p, json.dumps(made, indent=1))
        return made

    def status(self, cid):
        return next(r["status"] for r in self.rows() if r["id"] == cid)

    def built_ok(self, cid):
        return all((self.packs / f"{cid}{x}").exists() for x in (".json", ".bin", ".boundary.geojson", ".places.json"))

    def downloads(self, exs, admin):
        """Each extract whose caches are not all on disk, unless its .osm.pbf is: four tries."""
        import requests
        failed = {}
        for e in exs:
            stem = str(OSM / e.replace("/", "_"))
            need = CACHES + ((".admin.fgb",) if e in admin else ())
            if all(Path(stem + c).exists() for c in need) or Path(stem + ".osm.pbf").exists():
                continue
            path, url = Path(stem + ".osm.pbf"), GEOFABRIK.format(e)
            for attempt in range(4):
                t = time.time()
                try:
                    with requests.get(url, headers=UA, timeout=600, stream=True) as r:
                        r.raise_for_status()
                        with open(str(path) + ".part", "wb") as f:
                            for chunk in r.iter_content(1 << 20):
                                f.write(chunk)
                    os.replace(str(path) + ".part", path)
                    print(f"    downloaded {e}: {path.stat().st_size / 1e6:.0f} MB in {time.time() - t:.0f} s", flush=True)
                    break
                except (requests.RequestException, OSError) as x:
                    print(f"    download {e} try {attempt + 1}: {type(x).__name__} {x}", flush=True)
                    if attempt == 3:
                        failed[e] = f"download failed: {type(x).__name__} {x}"
                    else:
                        time.sleep(30 * 2 ** attempt)
        return failed

    def build_row(self, r):
        cid = r["id"]
        if self.stop_asked():
            return
        print(f"  {cid} {r['name']}: building", flush=True)
        self.set_row(cid, run_batch=self.batch, reason=None, steps=None, gate=None, parked=None)
        L = self.logs / cid
        steps, t0 = {}, time.time()
        ex = ",".join(r["extracts"])
        cmds = {"build": ("build_city", [cid] + (["--coarse-next"] if self.o.get("coarse") else [])), "stations": ("pack_gate", ["stations", cid, "--extracts", ex]),
                "places": ("places", [cid]),
                "routes": ("routes", [cid, "--extracts", ex, "--workers", str(self.threads), "--keep-pbf"]),
                "merge": ("merge_cells", [cid]), "nocar": ("nocar", [cid]), "everyday": ("everyday_rates", [cid]),
                "coarse": ("coarse_zones", [cid, "--under", str(self.o.get("coarse"))])}
        # T-852: the coarse grid re-solves the gravity before the routes are made for its zones
        order = BUILD_STEPS if not self.o.get("coarse") else [x for s in BUILD_STEPS for x in (["coarse", s] if s == "routes" else [s])]
        for s in order:
            mod, args = cmds[s]
            big = r["people"] > BIG_PEOPLE
            ok, secs, tail = self.step(mod, args, f"{L}.{s}.log", threads=self.threads, big=big)
            if not ok and not str(tail).startswith("stopped after"):
                time.sleep(60)  # once more: most failures here are a download or a server's timeout
                ok, secs2, tail = self.step(mod, args, f"{L}.{s}.log", threads=self.threads, big=big)
                secs += secs2
            steps[s] = secs
            if not ok:
                self.set_row(cid, steps=steps)
                self.park(cid, f"{s} failed: {str(tail)[-400:]}")
                return
        self.set_row(cid, "built", steps=steps, built=datetime.datetime.now().isoformat(timespec="seconds"))
        with self.lock:
            self.streak = (None, 0)
        self.rec(cid, name=r["name"], build_s=round(time.time() - t0), steps=steps)
        print(f"  {cid}: built in {time.time() - t0:.0f} s", flush=True)

    STREAK = 3

    def park(self, cid, reason, gate=None, streak=True):
        self.set_row(cid, "parked", reason=reason, gate=gate, parked=datetime.datetime.now().isoformat(timespec="seconds"))
        self.rec(cid, result="parked", reason=reason)
        print(f"  {cid}: PARKED: {reason[:300]}", flush=True)
        if not streak:
            return
        # T-775: three rows in a row parking on one reason is a fault of the run (T-613: 15 rows on one
        # merge error), so the run stops. The row's id is taken out; a failed step's log tail also
        # loses its numbers (times), while a gate's figures stay, so rows failing one check on their
        # own values (Liguria's drive factors) do not stop it
        key = reason.replace(cid, "<id>")
        if not key.startswith("gate:"):
            key = re.sub(r"\d+(\.\d+)?", "#", key)
        with self.lock:
            n = self.streak[1] + 1 if self.streak[0] == key else 1
            self.streak = (key, n)
        if n >= self.STREAK and not self.stopping:
            self.stopping = True
            msg = f"{n} rows in a row parked on one reason, so the run stops (T-775): {reason[:300]}"
            self.report.setdefault("problems", []).append(msg)
            print(f"  {msg}", flush=True)

    # ---------------------------------------------------------------- gate

    def gate(self, ids, L):
        """Gate the region's built rows, each registered with its market pack; returns those that
        ship (registered), the rest parked and unregistered again. Gates of several runs overlap, but
        never a joint solve (T-786), which would change packs between the snapshot and the verdicts:
        the gate holds its own gate lock and backs off while a solve holds (or waits for) packs."""
        sl = shared()
        while True:
            with self.held(f"gate-{self.batch}", f"gate {L.name} {' '.join(ids)}"[:200]):
                if not sl.is_held("packs", self.root):
                    return self.gate_held(ids, L)
            with self.held("packs", "a gate waiting for a joint solve to end"):
                pass

    def wait_gates(self):
        """A joint solve, holding packs (so no gate starts), waits for gates under way (T-786)."""
        sl, said = shared(), False
        while True:
            busy = [p.stem for p in (self.work / "world_run/locks").glob("gate-*.lock")
                    if p.stem != f"gate-{self.batch}" and sl.is_held(p.stem, self.root)]
            if not busy:
                return
            if not said:
                print(f"    the joint solve waits for gates under way: {' '.join(busy)}", flush=True)
                said = True
            time.sleep(5)

    def gate_held(self, ids, L):
        gd = L.parent / "gate"
        ok, _, near = self.step("@ne_gap", ids, f"{L}.ne_gap.log", threads=1)
        near = near if ok and isinstance(near, dict) else {}
        for cid, (d, j) in near.items():
            self.park(cid, f"T-580: {d:g} km from {j}, a Northeast pack solved on its own sites; a pack this near "
                           "would need a Northeast market place (the Northeast moved onto the --packs solve first)")
        ids = [i for i in ids if i not in near]
        if not ids:
            return []
        self.check_sim()
        snap = L.parent / f"{L.name}.snapshot.json"
        ok, _, tail = self.step("pack_gate", ["snapshot", str(snap)], f"{L}.snapshot.log", threads=1)
        if not ok:
            for i in ids:
                self.park(i, f"gate snapshot failed: {str(tail)[-300:]}")
            return []
        ok, _, tail = self.step("register", ids + ["--market"], f"{L}.register.log")
        if not ok:
            self.unregister(ids)
            for i in ids:
                self.park(i, f"register failed: {str(tail)[-300:]}")
            return []
        env = {"TW_EVERYDAY_EXE": str(self.everyday)} if self.everyday else {}
        if self.browser:
            env.update(BASE=self.start_vite(L), WORKERS="1")
        args = ids + ["--since", str(snap), "--out", str(gd)] + (["--browser"] if self.browser else [])
        rows = {r["id"]: r for r in self.rows()}
        for i in ids:
            log = self.logs / f"{i}.build.log"
            made = (rows.get(i, {}).get("run") or {}).get("run_batch")
            if not log.exists() and made:  # a row built by an earlier batch, gated by this one (T-725)
                log = self.logs.parent / made / f"{i}.build.log"
            args += ["--log", f"{i}={log}"]
            (gd / f"{i}.json").unlink(missing_ok=True)
        try:
            ok, _, tail = self.step("pack_gate", args, f"{L}.gate.log", env=env)
        finally:
            self.stop_vite()
        try:
            self.check_sim()  # the junctions went during the gate (T-613): its rates test read nothing
        except RunFault:
            self.unregister(ids)
            raise
        # a pack's own error is a FAIL in its result: no result means the gate itself died (T-773)
        none = [i for i in ids if not (gd / f"{i}.json").exists()]
        if none:
            self.unregister(ids)
            raise RunFault(f"the gate {'crashed' if not ok else 'ended'} with no result for {' '.join(none)} "
                           f"({str(tail)[-300:]}); the region's rows stay built for the next run (see {L}.gate.log)")
        shipped = []
        for i in ids:
            p = gd / f"{i}.json"
            res = json.loads(p.read_text(encoding="utf-8"))
            fails = [f"{c['name']} {c.get('value')} {c.get('detail') or ''}".strip() for c in res["checks"] if c["status"] == "FAIL"]
            warns = [f"{c['name']} {c.get('value')}" for c in res["checks"] if c["status"] == "WARN"]
            self.rec(i, warns=warns)
            if res["verdict"] == "ship":
                shipped.append(i)
                self.set_row(i, gate={"warn": warns})
                self.streak = (None, 0)
            else:
                self.park(i, "gate: " + "; ".join(fails)[:600], gate={"fail": fails, "warn": warns})
        parked = [i for i in ids if i not in shipped]
        if parked:
            self.unregister(parked)
        return shipped

    def start_vite(self, L):
        import urllib.request
        log = open(f"{L}.vite.log", "a", encoding="utf-8")
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        self.vite = subprocess.Popen(["node", str(REAL / "checks/dev-nowatch.mjs"), str(self.port)], cwd=str(MAPS),
                                     stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
        base = f"http://localhost:{self.port}"
        for _ in range(120):
            try:
                urllib.request.urlopen(base + "/", timeout=5)
                return base
            except Exception:
                if self.vite.poll() is not None:
                    raise SystemExit(f"Vite on {self.port} did not start (see {L}.vite.log)")
                time.sleep(1)
        raise SystemExit(f"Vite on {self.port} did not answer in 2 minutes")

    def stop_vite(self):
        """The runner's own Vite only: its PID and its children."""
        if self.vite and self.vite.poll() is None:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(self.vite.pid), "/T", "/F"], capture_output=True)
            else:
                self.vite.terminate()
            self.vite.wait(timeout=30)
        self.vite = None

    # ---------------------------------------------------------------- joint gravity and the market

    def join(self, shipped, L):
        changed = set(shipped)
        if shipped:
            ok, _, groups = self.step("@groups", shipped, f"{L}.groups.log", threads=1)
            if not ok or not isinstance(groups, list):
                for i in shipped:
                    self.park(i, f"joint gravity groups failed: {str(groups)[-300:]}", streak=False)
                groups = []
                self.unregister(shipped)
                shipped, changed = [], set()
            # T-786: a solve unit is one country's packs of a group, its neighbours within the cut fixed
            # (intercity.solve_units); units solve in turn, so a later one sees an earlier one's factors
            for u in groups:
                g = u["members"]
                new = [i for i in g if i in shipped]
                if not new:
                    continue
                u = self.live_unit(u)
                parked, rest = self.solve(u, new, L)
                if rest:  # only some new members failed: the group again without them, once (T-773)
                    ok, _, again = self.step("@groups", rest, f"{L}.groups-again.log", threads=1)
                    if not ok or not isinstance(again, list):
                        raise RunFault(f"joint gravity groups failed for {' '.join(rest)}: {str(again)[-300:]}")
                    for u2 in again:
                        p2, _ = self.solve(self.live_unit(u2), [i for i in u2["members"] if i in rest], L, again=True)
                        parked += p2
                        if not p2:
                            changed |= set(u2["members"])
                elif not parked:
                    # the fixed neighbours too: their commutes into the members come from this solve now
                    changed |= set(g) | set(u.get("fixed", []))
                shipped = [i for i in shipped if i not in parked]
                changed -= set(parked)
        order = [c["id"] for c in self.registry()["cities"]]
        ids = sorted((i for i in changed if i in order), key=order.index)  # a unit that parked later took its packs out
        if ids:
            ok, _, tail = self.step("register", ids + ["--market", "--trips"], f"{L}.market.log")
        else:
            ok, _, tail = self.step("market_zones", ["--rows"], f"{L}.market.log")
        if not ok:
            # the packs are registered and gated; a market failure leaves them so and says it
            self.report.setdefault("problems", []).append(f"market step failed: {str(tail)[-300:]}")
        for i in shipped:
            missing = [x for x in VITE_FILES if not (self.packs / f"{i}{x}").exists()]
            if missing:
                self.unregister([i])
                self.park(i, f"files a production build copies are missing: {', '.join(missing)}", streak=False)
                continue
            self.set_row(i, "shipped", shipped=datetime.datetime.now().isoformat(timespec="seconds"))
            self.rec(i, result="shipped")
            print(f"  {i}: SHIPPED", flush=True)
        try:  # T-800: the coarse outline of every registered zone; a failure only leaves the file as it was
            ok, _, tail = self.step("zone_outlines", [], f"{L}.outlines.log")
            if not ok:
                print(f"  warning: coarse zone outlines not rebuilt: {str(tail)[-200:]}", flush=True)
        except Exception as e:
            print(f"  warning: coarse zone outlines not rebuilt: {e}", flush=True)
        try:  # T-975: each zone's realm and the realms' land; a failure only leaves both as they were
            ok, _, tail = self.step("realms", [], f"{L}.realms.log")
            if not ok:
                print(f"  warning: realms not rebuilt: {str(tail)[-200:]}", flush=True)
        except Exception as e:
            print(f"  warning: realms not rebuilt: {e}", flush=True)
        try:  # T-836: search aliases for the new zones; a failure only leaves zone-aliases.json as it was
            ok, _, tail = self.step("zone_aliases", [], f"{L}.aliases.log")
            if not ok:
                print(f"  warning: zone search aliases not rebuilt: {str(tail)[-200:]}", flush=True)
        except Exception as e:
            print(f"  warning: zone search aliases not rebuilt: {e}", flush=True)

    def freeze(self):
        """T-928: the state this join left, frozen for publishing (pipeline/freeze.py), unless its
        market step failed or another run is live (its rows may be registered and not yet joined)."""
        others = [p.stem for p in (self.work / "world_run/locks").glob("batch-*.lock")
                  if p.stem != f"batch-{self.batch}" and shared().is_held(p.stem, self.root)]
        why = ("the market step failed" if any(p.startswith("market step failed") for p in self.report.get("problems", [])) else
               f"{' '.join(others)} live" if others else None)
        if why:
            print(f"  not frozen for publishing: {why}", flush=True)
            return
        ok, _, tail = self.step("freeze", ["--label", self.batch, "--locked"], self.logs / "freeze.log", threads=1)
        if not ok:
            self.report.setdefault("problems", []).append(f"freeze failed: {str(tail)[-300:]}")

    def live_unit(self, u):
        """T-997: the unit with its fixed list cut to the packs registered now. The lists are made
        before the first solve, and a unit that parked since has unregistered its packs, which
        joint_gravity refuses as fixed."""
        live = {c["id"] for c in self.registry()["cities"]}
        fixed = [i for i in u.get("fixed", []) if i in live]
        if len(fixed) != len(u.get("fixed", [])):
            left = [i for i in u["fixed"] if i not in live]
            print(f"  joint group {u['name']}: fixed without {' '.join(left)} (parked, no longer registered)", flush=True)
        return {**u, "fixed": fixed}

    def solve(self, u, new, L, again=False):
        """One solve unit's joint solve (`u`: name, members, fixed): members back to res-9, solved
        with the fixed packs held at their factors, merged again, re-gated (gravity may change in
        the members only). Returns (parked, rest). A FAIL restores every member; when only new
        members FAIL, those park and `rest` (the other new ones) go to a second solve without them;
        when an older member FAILs, or on the second solve, every new member parks. A re-gate that
        leaves a member without a result is a run fault (T-773)."""
        g, fixed, name = u["members"], u.get("fixed", []), u["name"]
        tag = f"{L}.group-{name}" + ("-again" if again else "")
        print(f"  joint group {name}{' again' if again else ''}: {' '.join(g)} (new: {' '.join(new)})"
              f"{'; fixed: ' + ' '.join(fixed) if fixed else ''}", flush=True)
        self.check_sim()
        bak = Path(f"{tag}-backup")
        self.backup(g, bak)
        # the re-gate compares against the packs just before this solve (an earlier group's solve changed its own)
        snap = Path(f"{tag}.snapshot.json")
        self.step("pack_gate", ["snapshot", str(snap)], f"{tag}.log", threads=1)
        steps = [("merge_cells", g + ["--undo"]),
                 ("intercity", ["--solve", ",".join(g), "--name", name] + (["--fixed", ",".join(fixed)] if fixed else [])),
                 ("merge_cells", g)]
        fail, bad = None, {}
        for mod, args in steps:
            # nothing else runs during the join: each step takes every core, and a big unit's solve the long limit
            solo = mod == "intercity"
            ok, _, tail = self.step(mod, args, f"{tag}.log", big=solo and len(g) > 100)
            if not ok:
                try:
                    self.check_sim()  # intercity builds from the sim copy
                except RunFault:
                    self.restore(g, bak)
                    raise
                fail = f"{mod} failed: {str(tail)[-300:]}"
                break
        if fail is None:
            gd = Path(f"{tag}-gate")
            ok, _, tail = self.step("pack_gate", g + ["--no-rust", "--since", str(snap), "--out", str(gd)], f"{tag}.gate.log")
            none = [i for i in g if not (gd / f"{i}.json").exists()]
            if none:
                self.restore(g, bak)
                raise RunFault(f"joint group {g[0]}: the re-gate {'crashed' if not ok else 'ended'} with no result for "
                               f"{' '.join(none)} ({str(tail)[-300:]}); the group is restored and its new packs "
                               f"({' '.join(new)}) stay built and registered for the next run (see {tag}.gate.log)")
            for i in g:
                res = json.loads((gd / f"{i}.json").read_text(encoding="utf-8"))
                if res["verdict"] != "ship":
                    bad[i] = ", ".join(f"{c['name']} {c.get('value')}" for c in res["checks"] if c["status"] == "FAIL")
            if bad:
                fail = "after the joint solve " + "; ".join(f"{i}: {v}" for i, v in bad.items())
        if fail is None:
            return [], []
        self.restore(g, bak)
        only_new = bool(bad) and set(bad) <= set(new) and not again
        parked = [i for i in new if i in bad] if only_new else list(new)
        self.unregister(parked)
        for i in parked:
            why = f"after the joint solve: {bad[i]}" if only_new else fail
            self.park(i, f"joint group {g[0]} ({' '.join(g)}): {why}"[:600])
        return parked, ([i for i in new if i not in parked] if only_new else [])

    def group_files(self, g):
        out = []
        for i in g:
            out += [self.packs / f"{i}.json", self.packs / f"{i}.bin", self.work / f"res9/{i}.json",
                    self.work / f"res9/{i}.bin", self.work / f"res9/{i}.merged.json"]
        for p in self.work.glob("*.joint_commutes.json"):
            if set(json.loads(p.read_text(encoding="utf-8")).get("packs", [])) & set(g):
                out.append(p)
        return out

    def backup(self, g, bak):
        bak.mkdir(parents=True, exist_ok=True)
        manifest = []
        for p in self.group_files(g):
            if p.exists():
                rel = p.relative_to(self.root)
                (bak / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, bak / rel)
                manifest.append(str(rel))
        (bak / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    def restore(self, g, bak):
        manifest = json.loads((bak / "manifest.json").read_text(encoding="utf-8"))
        for p in self.group_files(g):  # a joint file the solve wrote
            if str(p.relative_to(self.root)) not in manifest and p.name.endswith(".joint_commutes.json"):
                p.unlink()
        for rel in manifest:
            part = self.root / (rel + ".part")
            shutil.copy2(bak / rel, part)
            os.replace(part, self.root / rel)
        print(f"  joint group {g[0]}: restored from {bak}", flush=True)

    # ---------------------------------------------------------------- clean up and report

    def clean(self, exs):
        """The region's .osm.pbf files, and the OSM caches runs made (those there before stay; the
        small .admin.fgb stays too)."""
        made, gone = self.made(), []
        for e in exs:
            stem = str(OSM / e.replace("/", "_"))
            for c in [".osm.pbf"] + made.get(e, []):
                p = Path(stem + c)
                if p.exists():
                    mb = p.stat().st_size / 1e6
                    p.unlink()
                    gone.append(f"{p.name} {mb:.0f} MB")
        self.made(drop=exs)
        if gone:
            print(f"  deleted {', '.join(gone)}", flush=True)
        return gone

    def write_report(self):
        rows = {r["id"]: r for r in self.rows()}
        lines = [f"world_run batch {self.batch}: {self.report['seconds'] / 60:.0f} minutes, jobs {self.jobs}"
                 f"{', STOPPED early' if self.report['stopped'] else ''}"]
        for res in ("shipped", "gated", "parked", "built", "planned"):
            ids = [i for i in self.report["rows"] if rows.get(i, {}).get("status") == res]
            ids += [i for rr in self.report["regions"] for i in rr["rows"] if rows.get(i, {}).get("status") == res and i not in ids]
            if not ids:
                continue
            lines.append(f"{res}: {len(ids)}")
            for i in ids:
                r, run = rows[i], rows[i].get("run", {})
                st = run.get("steps", {})
                t = f"build {sum(st.values()) / 60:.1f} min ({', '.join(f'{k} {v:.0f}s' for k, v in st.items())})" if st else ""
                extra = ""
                if res == "parked":
                    extra = "; " + run.get("reason", "")
                elif run.get("gate", {}).get("warn"):
                    extra = "; WARN " + ", ".join(run["gate"]["warn"])
                lines.append(f"  {i} {r['name']} ({r.get('iso2')}, {r['people'] / 1e6:.2f}M): {t}{extra}")
        for rr in self.report["regions"]:
            lines.append(f"region {rr['n']} ({', '.join(rr['extracts'])}): " +
                         ", ".join(f"{k} {v / 60:.1f} min" for k, v in rr["seconds"].items()) +
                         (f"; deleted {len(rr.get('deleted', []))} files" if rr.get("deleted") else ""))
        if "join_seconds" in self.report:
            lines.append(f"joint solve and market, once for the batch: {self.report['join_seconds'] / 60:.1f} min")
        for p in self.report.get("problems", []):
            lines.append("problem: " + p)
        text = "\n".join(lines)
        print("\n" + text, flush=True)
        self.logs.mkdir(parents=True, exist_ok=True)
        (self.logs / "report.txt").write_text(text + "\n", encoding="utf-8")
        (self.logs / "report.json").write_text(json.dumps(self.report, indent=1, ensure_ascii=False), encoding="utf-8")


def status(root):
    import collections
    plan = Path(root or REAL) / "pipeline/world_plan.json"
    rows = [r for r in json.loads(plan.read_text(encoding="utf-8"))["metros"] if r.get("units")]
    stop = Path(root or REAL) / "data/work/world_run/STOP"
    print(f"STOP file: {stop}{' (present: the next run will not start)' if stop.exists() else ''}")
    print(dict(collections.Counter(r["status"] for r in rows)))
    sl = shared()
    for r in sorted(rows, key=lambda r: r["order"]):
        if r["status"] in ("built", "gated", "shipped", "parked"):
            claim = (r.get("run") or {}).get("claim")
            live = claim and sl.is_held(f"batch-{claim}", root or REAL)
            print(f"  {r['order']:>4} {r['id']} {r['name']}: {r['status']}"
                  f"{'; ' + r.get('run', {}).get('reason', '') if r['status'] == 'parked' else ''}"
                  f"{'; claimed by run ' + claim if live else ''}")
    # T-786: locks held now (runs at once): which run holds what
    locks = sorted((Path(root or REAL) / "data/work/world_run/locks").glob("*.lock"))
    if root:
        locks += sorted((REAL / "data/work/world_run/locks").glob("extract-*.lock"))
    for p in locks:
        if sl.is_held(p.stem, p.parents[4]):
            w = sl.who(p.stem, p.parents[4]) or {}
            print(f"lock {p.stem}: held by pid {w.get('pid')} ({w.get('what')}) since {w.get('since')}")


def parse(argv):
    o = {}
    flags = {"--no-browser": "no_browser", "--keep-downloads": "keep_downloads", "--no-join": "no_join",
             "--build-only": "build_only"}
    valued = {"--rows": "rows", "--countries": "countries", "--limit": "limit", "--jobs": "jobs",
              "--port": "port", "--root": "root", "--workers": "workers", "--coarse": "coarse"}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in flags:
            o[flags[a]] = True
        elif a in valued:
            o[valued[a]] = argv[i + 1]
            i += 1
        else:
            raise SystemExit(f"unknown argument {a}\n{__doc__}")
        i += 1
    return o


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "step":
        step_main(sys.argv[2:])
        sys.exit()
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    opts = parse(sys.argv[2:])
    if cmd == "plan":
        Run(opts).dry()
    elif cmd == "run":
        Run(opts).go()
    elif cmd == "join":
        Run(opts).join_only()
    elif cmd == "status":
        status(opts.get("root"))
    else:
        raise SystemExit(__doc__)
