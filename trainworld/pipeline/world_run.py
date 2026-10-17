"""T-540, pipeline stage 8: the runner. Builds world plan rows into shipped or parked packs with no
agent judgement (docs/adding-a-city.md "Running the pipeline"; design notes/T-535.md).

    python -I pipeline/world_run.py plan   [select]            what a run would do; nothing built
    python -I pipeline/world_run.py run    [select] [options]  build, gate, register, join, clean up
    python -I pipeline/world_run.py status                     rows by status, parked reasons

    select   --rows a,b,c  (in plan order, any status but shipped) | --countries GB,PT | --limit N
             (the next N planned or half-done rows in plan order; default 10)
    options  --jobs N         cores in all (default 2; 12 only on an overnight run Anita approves)
             --port P         the runner's own Vite for the gate's browser check (default 6360)
             --no-browser     gate without the browser check
             --keep-downloads keep the region's .osm.pbf and OSM caches
             --root DIR       a scratch tree instead of this one (below)

A run goes a region at a time: rows sharing a Geofabrik extract, chained, in plan order. For each:
download (or reuse) and read its extracts, then per row build_city, places, routes, merge_cells,
nocar, everyday_rates (status "built"); then the gate on every built row, registered with its
market pack against a snapshot taken before the region (status "shipped", or "parked" with the
failing checks, unregistered again). Then the joint gravity groups the shipped packs joined, one
solve a group, the group re-gated after it (a FAIL restores the group and parks its new packs);
then market_zones and local trips; then the region's downloads are deleted. Status and a record of
times live in each row of pipeline/world_plan.json; a report a batch goes to
../artifacts/trainworld/world_run/<batch>/.

Stop: ctrl-c, or create data/work/world_run/STOP. The city under way finishes its steps first;
gating and joining wait for the next run, which takes up rows where they were.

Each step runs in its own process (`world_run.py step ...`, `python -I`), with OMP_NUM_THREADS and
CARGO_BUILD_JOBS from --jobs; TW_SIM_DIR and TW_CARGO_TARGET_DIR pass through, for a copy of sim/
while the tree's own is mid-edit. With --root, every step's paths into data/packs, data/work,
app/ and the world plan point into that tree instead (data/raw there should be a junction to this
tree's, so caches are shared; sim/ a copy, for the gate's rates test), and the browser check is
off, since Vite serves only this tree.
"""
import ast
import datetime
import importlib
import importlib.abc
import importlib.machinery
import json
import os
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
BUILD_STEPS = ["build", "places", "routes", "merge", "nocar", "everyday"]
VITE_FILES = (".json", ".bin", ".boundary.geojson", ".places.json", ".market.json")  # vite.config.ts copies these
CACHES = (".points.fgb", ".areas.fgb", ".water.fgb", ".ways2.npz")  # an extract's big OSM caches
MODULE_OUTLINES = {"england", "france", "germany"}  # plan_hooks giving a boundary_file (T-555)
# A step still running after this long is stopped and its row parks (four times as long for a row
# over BIG_PEOPLE): the longest seen, a 20M pack's build, takes under an hour on 3 cores.
STEP_LIMIT_S = {"build_city": 5400, "routes": 3600, "pack_gate": 5400, "intercity": 7200, None: 1800}
BIG_PEOPLE = 15e6


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
    """The registered packs' joint gravity groups holding any of these ids."""
    import intercity
    reg = json.loads((intercity.ROOT / "app/src/game/registry.json").read_text(encoding="utf-8"))
    return [g for g in intercity.auto_groups(reg, intercity.ROOT) if set(g) & set(ids)]


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
        self.workers = int(opts.get("workers") or max(1, self.jobs // 3))
        self.threads = max(1, self.jobs // self.workers)
        self.browser = not opts.get("no_browser") and not self.scratch
        self.port = int(opts.get("port", 6360))
        self.lock = threading.Lock()
        self.stopping = False
        self.vite = None
        self.report = {"batch": self.batch, "root": str(self.root), "jobs": self.jobs, "rows": {}, "regions": []}

    # ---------------------------------------------------------------- files

    def plan_doc(self):
        return json.loads(self.plan_path.read_text(encoding="utf-8"))

    def rows(self):
        return [r for r in self.plan_doc()["metros"] if r.get("units")]

    def set_row(self, cid, status=None, **run):
        """One row's status and run record, the plan re-read first (other tools write it too)."""
        with self.lock:
            doc = self.plan_doc()
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
            tmp = self.plan_path.with_suffix(".json.part")
            tmp.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
            os.replace(tmp, self.plan_path)

    def registry(self):
        return json.loads(self.reg_path.read_text(encoding="utf-8"))

    def unregister(self, ids):
        """Take packs out of the registry again (a parked pack): its entry and market pack go; its
        market place stays in `market.places`, unused (place ids are positions, append only)."""
        with self.lock:
            reg = self.registry()
            before = len(reg["cities"])
            reg["cities"] = [c for c in reg["cities"] if c["id"] not in ids]
            if len(reg["cities"]) != before:
                self.reg_path.write_text(json.dumps(reg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            for i in ids:
                p = self.packs / f"{i}.market.json"
                if p.exists():
                    p.unlink()

    def stop_asked(self):
        return self.stopping or self.stop_file.exists()

    # ---------------------------------------------------------------- one step

    def step(self, name, args, log, threads=None, env=None, quiet=False, big=False):
        """(ok, seconds, the RESULT line's value or the log's last lines)."""
        t = threads or self.threads
        e = dict(os.environ, OMP_NUM_THREADS=str(t), OPENBLAS_NUM_THREADS=str(t), MKL_NUM_THREADS=str(t),
                 RAYON_NUM_THREADS=str(t), CARGO_BUILD_JOBS=str(t), PYTHONIOENCODING="utf-8", **(env or {}))
        if self.scratch:
            e["TW_RUN_ROOT"] = str(self.root)
        log = Path(log)
        log.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0  # ctrl-c stops the runner, not a step
        with open(log, "a", encoding="utf-8") as f:
            f.write(f"### {datetime.datetime.now().isoformat(timespec='seconds')} {name} {' '.join(args)}\n")
            f.flush()
            p = subprocess.Popen([sys.executable, "-I", "-u", os.path.abspath(__file__), "step", name, *args],
                                 stdout=f, stderr=subprocess.STDOUT, env=e, cwd=str(MAPS), creationflags=flags)
            limit = STEP_LIMIT_S.get(name, STEP_LIMIT_S[None]) * (4 if big else 1)
            try:
                code = p.wait(timeout=limit)
            except subprocess.TimeoutExpired:
                self.kill(p)
                code = None
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

    def select(self):
        rows = sorted(self.rows(), key=lambda r: r["order"])
        if self.o.get("rows"):
            want = self.o["rows"].split(",")
            missing = [w for w in want if w not in {r["id"] for r in rows}]
            if missing:
                raise SystemExit(f"not plan rows: {missing}")
            return [r for r in rows if r["id"] in want and r["status"] != "shipped"]
        pick = [r for r in rows if r["status"] in ("planned", "built")]
        if self.o.get("countries"):
            cs = set(self.o["countries"].split(","))
            pick = [r for r in pick if r.get("iso2") in cs]
        return pick[:int(self.o.get("limit") or 10)]

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

    def go(self):
        rows = self.select()
        if not rows:
            print("nothing to build")
            return
        self.logs.mkdir(parents=True, exist_ok=True)
        if self.stop_file.exists():
            raise SystemExit(f"{self.stop_file} exists: delete it to run")
        signal.signal(signal.SIGINT, self.on_ctrl_c)
        t0 = time.time()
        regs = self.regions(rows)
        print(f"batch {self.batch}: {len(rows)} rows in {len(regs)} regions; jobs {self.jobs} ({self.workers} x {self.threads}); "
              f"logs {self.logs}", flush=True)
        for k, reg in enumerate(regs, 1):
            if self.stop_asked():
                break
            self.region(k, reg)
        self.report["seconds"] = round(time.time() - t0)
        self.report["stopped"] = self.stop_asked()
        self.write_report()

    def on_ctrl_c(self, *_):
        if not self.stopping:
            print("\nctrl-c: stopping once the city under way is done (the steps keep running)", flush=True)
        self.stopping = True

    def rec(self, cid, **kv):
        self.report["rows"].setdefault(cid, {}).update(kv)

    def region(self, k, reg):
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
            return
        built = [r["id"] for r in reg if self.status(r["id"]) == "built"]
        if built:
            t = time.time()
            shipped = self.gate(built, L)
            rr["seconds"]["gate"] = round(time.time() - t)
            t = time.time()
            self.join(shipped, L)
            rr["seconds"]["join"] = round(time.time() - t)
        if not self.o.get("keep_downloads"):
            rr["deleted"] = self.clean(exs)

    @staticmethod
    def caches(e):
        stem = str(OSM / e.replace("/", "_"))
        return [c for c in CACHES if Path(stem + c).exists()]

    def made(self, add=None, drop=()):
        """The OSM caches runs have made and not yet deleted, by extract (those already on disk
        before a run stay when it cleans up)."""
        p = REAL / "data/work/world_run/made.json"  # this tree's, as the caches are (data/raw is shared)
        with self.lock:
            made = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
            for e, cs in (add or {}).items():
                made[e] = sorted(set(made.get(e, [])) | set(cs))
            for e in drop:
                made.pop(e, None)
            made = {e: cs for e, cs in made.items() if cs}
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(made, indent=1), encoding="utf-8")
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
        cmds = {"build": ("build_city", [cid]), "places": ("places", [cid]),
                "routes": ("routes", [cid, "--extracts", ex, "--workers", str(self.threads), "--keep-pbf"]),
                "merge": ("merge_cells", [cid]), "nocar": ("nocar", [cid]), "everyday": ("everyday_rates", [cid])}
        for s in BUILD_STEPS:
            mod, args = cmds[s]
            big = r["people"] > BIG_PEOPLE
            ok, secs, tail = self.step(mod, args, f"{L}.{s}.log", big=big)
            if not ok and not str(tail).startswith("stopped after"):
                time.sleep(60)  # once more: most failures here are a download or a server's timeout
                ok, secs2, tail = self.step(mod, args, f"{L}.{s}.log", big=big)
                secs += secs2
            steps[s] = secs
            if not ok:
                self.set_row(cid, steps=steps)
                self.park(cid, f"{s} failed: {str(tail)[-400:]}")
                return
        self.set_row(cid, "built", steps=steps, built=datetime.datetime.now().isoformat(timespec="seconds"))
        self.rec(cid, name=r["name"], build_s=round(time.time() - t0), steps=steps)
        print(f"  {cid}: built in {time.time() - t0:.0f} s", flush=True)

    def park(self, cid, reason, gate=None):
        self.set_row(cid, "parked", reason=reason, gate=gate, parked=datetime.datetime.now().isoformat(timespec="seconds"))
        self.rec(cid, result="parked", reason=reason)
        print(f"  {cid}: PARKED: {reason[:300]}", flush=True)

    # ---------------------------------------------------------------- gate

    def gate(self, ids, L):
        """Gate the region's built rows, each registered with its market pack; returns those that
        ship (registered), the rest parked and unregistered again."""
        gd = L.parent / "gate"
        ok, _, near = self.step("@ne_gap", ids, f"{L}.ne_gap.log", threads=1)
        near = near if ok and isinstance(near, dict) else {}
        for cid, (d, j) in near.items():
            self.park(cid, f"T-580: {d:g} km from {j}, a Northeast pack solved on its own sites; a pack this near "
                           "would need a Northeast market place (the Northeast moved onto the --packs solve first)")
        ids = [i for i in ids if i not in near]
        if not ids:
            return []
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
        env = {}
        if self.browser:
            env = {"BASE": self.start_vite(L), "WORKERS": "1"}
        args = ids + ["--since", str(snap), "--out", str(gd)] + (["--browser"] if self.browser else [])
        for i in ids:
            args += ["--log", f"{i}={self.logs / i}.build.log"]
        try:
            self.step("pack_gate", args, f"{L}.gate.log", env=env)
        finally:
            self.stop_vite()
        shipped = []
        for i in ids:
            p = gd / f"{i}.json"
            if not p.exists():
                self.park(i, "gate: no result (see the gate log)")
                continue
            res = json.loads(p.read_text(encoding="utf-8"))
            fails = [f"{c['name']} {c.get('value')} {c.get('detail') or ''}".strip() for c in res["checks"] if c["status"] == "FAIL"]
            warns = [f"{c['name']} {c.get('value')}" for c in res["checks"] if c["status"] == "WARN"]
            self.rec(i, warns=warns)
            if res["verdict"] == "ship":
                shipped.append(i)
                self.set_row(i, gate={"warn": warns})
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
                groups = []
                for i in shipped:
                    self.park(i, f"joint gravity groups failed: {str(groups)[-300:]}")
                self.unregister(shipped)
                shipped, changed = [], set()
            for g in groups:
                new = [i for i in g if i in shipped]
                if self.solve(g, new, L):
                    changed |= set(g)
                else:
                    shipped = [i for i in shipped if i not in new]
                    changed -= set(new)
        ids = sorted(changed, key=lambda i: [c["id"] for c in self.registry()["cities"]].index(i)) if changed else []
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
                self.park(i, f"files a production build copies are missing: {', '.join(missing)}")
                continue
            self.set_row(i, "shipped", shipped=datetime.datetime.now().isoformat(timespec="seconds"))
            self.rec(i, result="shipped")
            print(f"  {i}: SHIPPED", flush=True)

    def solve(self, g, new, L):
        """One group's joint solve: members back to res-9, solved, merged again, re-gated (gravity
        may change in the group only). A FAIL restores every member and parks the group's new packs."""
        tag = f"{L}.group-{g[0]}"
        print(f"  joint group {g[0]}: {' '.join(g)} (new: {' '.join(new)})", flush=True)
        bak = Path(f"{tag}-backup")
        self.backup(g, bak)
        steps = [("merge_cells", g + ["--undo"]), ("intercity", ["--group", "auto"] + new + ["--no-market"]),
                 ("merge_cells", g)]
        fail = None
        for mod, args in steps:
            ok, _, tail = self.step(mod, args, f"{tag}.log")
            if not ok:
                fail = f"{mod} failed: {str(tail)[-300:]}"
                break
        if fail is None:
            gd = Path(f"{tag}-gate")
            snap = L.parent / f"{L.name}.snapshot.json"
            self.step("pack_gate", g + ["--no-rust", "--since", str(snap), "--out", str(gd)], f"{tag}.gate.log")
            bad = []
            for i in g:
                p = gd / f"{i}.json"
                res = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"verdict": "park", "checks": []}
                if res["verdict"] != "ship":
                    bad.append(f"{i}: " + ", ".join(c["name"] for c in res["checks"] if c["status"] == "FAIL"))
            if bad:
                fail = "after the joint solve " + "; ".join(bad)
        if fail is None:
            return True
        self.restore(g, bak)
        self.unregister(new)
        for i in new:
            self.park(i, f"joint group {g[0]} ({' '.join(g)}): {fail}"[:600])
        return False

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
        for res in ("shipped", "parked", "built", "planned"):
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
    print(dict(collections.Counter(r["status"] for r in rows)))
    for r in sorted(rows, key=lambda r: r["order"]):
        if r["status"] in ("built", "shipped", "parked"):
            print(f"  {r['order']:>4} {r['id']} {r['name']}: {r['status']}"
                  f"{'; ' + r.get('run', {}).get('reason', '') if r['status'] == 'parked' else ''}")


def parse(argv):
    o = {}
    flags = {"--no-browser": "no_browser", "--keep-downloads": "keep_downloads"}
    valued = {"--rows": "rows", "--countries": "countries", "--limit": "limit", "--jobs": "jobs",
              "--port": "port", "--root": "root", "--workers": "workers"}
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
    elif cmd == "status":
        status(opts.get("root"))
    else:
        raise SystemExit(__doc__)
