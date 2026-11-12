"""T-3244: the light pass, every demand change that needs no extract read and no joint solve, over every
shipped tier 0 pack (docs/world-run.md "The light pass").

    python -I pipeline/world_run.py step light_pass run [--jobs N] [--from STEP] [--batch NAME]
    python -I pipeline/world_run.py step light_pass plan          the id lists and what would run, nothing written

Steps, in order (each its own process and log in ../artifacts/trainworld/light_pass/<batch>/):
  backup     data/packs and data/work/res9 into data/work/pre-T3244/ as hard links (every writer replaces a
             pack file, so a link keeps the old file at no space; the small merge stamps are copied)
  campus     campus_outlines.py split: each pack's universities and colleges with outlines (the bands must
             be fetched: `campus_outlines.py fetch`)
  wikidata   wikidata_prefill.py: student counts, parents, labels and airport figures, 200 ids a query,
             then campus_pools.py build (each institution's count shared over its campuses in the world)
  through    merge_cells --through, --keep-going, N workers: "tier0 {id} --education --airports" for the
             packs whose airport audit names an airport, "tier0 {id} --education" for the rest
  cross      cross_everyday.py --group auto with N threads (universities cross zone lines)
  market     market_zones.py --workers N --full: every zone file and every market row
  trips      zones.local_trips: every registered zone's localTrips in the registry
  gate       checks/pack_gate.py --no-browser over every pack, in N processes, figures and rates
  load       checks/t2213_packs_load.py after.json --against the baseline
  freeze     freeze.py
`--from STEP` starts there (the steps before it done); `--steps a,b` runs those alone; `--ids FILE` only
the packs listed (a proof on a scratch tree: TW_RUN_ROOT, docs/world-run.md "A scratch tree"). The run holds the `packs` lock throughout, so no
world_run gate or join starts meanwhile; it claims N cores in cores.py. A through failure leaves that pack
res-9 in data/packs: the run stops after `through` with the failed ids in failed.txt; rerun them with
`--from through --ids <file>`, then `--from cross`.
"""
import datetime
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TW = HERE.parent
MAPS = TW.parent
SIM = HERE.parent / "sim"  # the real tree's, also in a scratch run (Redirect moves TW)
PACKS = TW / "data/packs"
WORK = TW / "data/work"
REG = TW / "app/src/game/registry.json"
PLAN = HERE / "world_plan.json"
STEPS = ["backup", "campus", "wikidata", "through", "cross", "market", "trips", "gate", "load", "freeze"]
BASELINE = MAPS / "artifacts/trainworld/t3244/baseline/packs_load_before.json"
BASELINE_GATE = MAPS / "artifacts/trainworld/light_pass/baseline-head/gate"  # the gate on the packs before the pass
PY = sys.executable


def opt(args, flag, default=None):
    return args[args.index(flag) + 1] if flag in args else default


def id_lists(only=None):
    """(airport packs, the other packs): shipped tier 0 plan rows, registered, with a res-9 copy."""
    reg = {c["id"] for c in json.loads(REG.read_text(encoding="utf-8"))["cities"]}
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    ids = [r["id"] for r in plan["metros"] if r.get("units") and r.get("status") == "shipped"
           and r["id"] in reg and (WORK / "res9" / f"{r['id']}.bin").exists()]
    if only is not None:
        ids = [i for i in ids if i in only]
    air, rest = [], []
    for cid in ids:
        p = WORK / f"{cid}.airports.audit.json"
        a = json.loads(p.read_text(encoding="utf-8")) if p.exists() else []
        (air if isinstance(a, list) and a else rest).append(cid)
    return air, rest


class Run:
    def __init__(self, args):
        self.jobs = int(opt(args, "--jobs", 2))
        self.batch = opt(args, "--batch") or datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.logs = MAPS / "artifacts/trainworld/light_pass" / self.batch
        self.logs.mkdir(parents=True, exist_ok=True)
        f = opt(args, "--ids")
        self.only = set(Path(f).read_text().split()) if f else None
        self.air, self.rest = id_lists(self.only)
        self.steps = opt(args, "--steps", "").split(",") if "--steps" in args else None  # these alone, in this order
        self.times = {}

    def step(self, name, cmd, env=None, check=True):
        log = self.logs / f"{name}.log"
        t = time.time()
        print(f"[{datetime.datetime.now():%H:%M}] {name}: {' '.join(str(c) for c in cmd)[:300]}  (log {log})", flush=True)
        with open(log, "a", encoding="utf-8") as f:
            r = subprocess.run([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT, cwd=str(MAPS),
                               env=dict(os.environ, **(env or {})))
        self.times[name] = self.times.get(name, 0) + round(time.time() - t)
        print(f"  {name}: exit {r.returncode} in {time.time() - t:.0f} s", flush=True)
        if check and r.returncode:
            raise SystemExit(f"{name} failed (exit {r.returncode}); see {log}")
        return r.returncode

    def tw(self, *a):
        return [PY, "-I", HERE / "world_run.py", "step", *a]

    def threads(self):
        n = str(self.jobs)
        return {"OMP_NUM_THREADS": n, "OPENBLAS_NUM_THREADS": n, "MKL_NUM_THREADS": n}

    # ---------------------------------------------------------------- the steps
    def backup(self):
        dst = WORK / "pre-T3244"
        n = 0
        for src_dir, sub in ((PACKS, "packs"), (WORK / "res9", "res9")):
            (dst / sub).mkdir(parents=True, exist_ok=True)
            for p in src_dir.iterdir():
                if not p.is_file():
                    continue
                q = dst / sub / p.name
                if q.exists():
                    continue  # a backup made by an earlier start of this pass is the one to keep
                if p.name.endswith(".merged.json") or p.suffix not in (".json", ".bin", ".geojson"):
                    shutil.copy2(p, q)  # written in place by its writer: a copy
                else:
                    os.link(p, q)
                n += 1
        for f in ("registry.json", "market-rows.json"):
            if not (dst / f).exists():
                shutil.copy2(TW / "app/src/game" / f, dst / f)
        print(f"  backup: {n} files into {dst}", flush=True)

    def campus(self):
        self.step("campus", self.tw("campus_outlines", "split", *self.air, *self.rest))

    def wikidata(self):
        self.step("wikidata", self.tw("wikidata_prefill"))

    def through(self):
        failed = []
        for ids, tmpl, name in ((self.air, "tier0 {id} --education --airports", "through-airports"),
                                (self.rest, "tier0 {id} --education", "through-education")):
            if not ids:
                continue
            lst = self.logs / f"{name}.ids"
            lst.write_text(" ".join(ids))
            self.step(name, self.tw("merge_cells", *ids, "--through", tmpl, "--workers", str(self.jobs), "--keep-going"),
                      check=False)
            log = (self.logs / f"{name}.log").read_text(encoding="utf-8", errors="replace")
            # the log is appended across reruns of a batch: a pack's last line decides
            last = {ln.split()[1]: ln.split()[0] for ln in log.splitlines() if ln.startswith(("FAILED ", "DONE "))}
            failed += [i for i in ids if last.get(i) == "FAILED"]
        (self.logs / "failed.txt").write_text(" ".join(failed))
        if failed:
            raise SystemExit(f"through: {len(failed)} packs failed, left res-9 in data/packs: {self.logs / 'failed.txt'}; "
                             f"rerun with --from through --ids {self.logs / 'failed.txt'}, then --from cross")

    def cross(self):
        some = sorted(self.only) if self.only is not None else []  # --ids: the units holding them
        self.step("cross", self.tw("cross_everyday", "--group", "auto", *some), env=self.threads())

    def market(self):
        some = sorted(self.only) if self.only is not None else ["--full"]
        self.step("market", self.tw("market_zones", *some, "--workers", str(self.jobs)))

    def trips(self):
        self.step("trips", self.tw("light_pass", "trips"))

    def gate(self):
        exe = self.exe
        ids = self.air + self.rest
        out = self.logs / "gate"
        out.mkdir(exist_ok=True)
        procs = []
        for k in range(self.jobs):
            part = ids[k::self.jobs]
            log = open(self.logs / f"gate-{k}.log", "w", encoding="utf-8")
            env = dict(os.environ, OMP_NUM_THREADS="1", **({"TW_EVERYDAY_EXE": str(exe)} if exe else {}))
            procs.append((subprocess.Popen([*self.tw("pack_gate"), *part, "--no-browser", "--out", out],
                                           stdout=log, stderr=subprocess.STDOUT, cwd=str(MAPS), env=env), log))
        t = time.time()
        for p, log in procs:
            p.wait()
            log.close()
        self.times["gate"] = round(time.time() - t)
        def failing(d):
            got = {}
            for p in Path(d).glob("*.json"):
                r = json.loads(p.read_text(encoding="utf-8"))
                got[p.stem] = {c["name"] for c in r.get("checks", []) if c.get("status") == "FAIL"}
            return got
        now, before = failing(out), failing(BASELINE_GATE)
        # today's packs FAIL some checks already (joint_borders on a pack gated alone): only new FAILs count
        new = {c: sorted(f - before.get(c, set())) for c, f in now.items() if f - before.get(c, set())}
        (self.logs / "gate-fails.txt").write_text("\n".join(f"{c}: {', '.join(sorted(f))}" for c, f in sorted(now.items()) if f))
        (self.logs / "gate-new-fails.txt").write_text("\n".join(f"{c}: {', '.join(f)}" for c, f in sorted(new.items())))
        print(f"  gate: {len(ids)} packs in {time.time() - t:.0f} s, {sum(1 for f in now.values() if f)} with a FAIL, "
              f"{len(new)} with a FAIL the baseline lacks ({self.logs / 'gate-new-fails.txt'})", flush=True)

    def everyday_exe(self):
        sim = Path(os.environ.get("TW_SIM_DIR") or SIM)
        e = {"CARGO_BUILD_JOBS": str(self.jobs),
             "PATH": os.path.join(os.path.expanduser("~"), ".cargo", "bin") + os.pathsep + os.environ.get("PATH", "")}
        cargo = shutil.which("cargo", path=e["PATH"])
        p = subprocess.run([cargo, "test", "--release", "--features", "demand", "--lib", "--no-run", "--message-format", "json",
                            "--manifest-path", str(sim / "Cargo.toml")], env=dict(os.environ, **e),
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        exe = None
        for ln in p.stdout.splitlines():
            if ln.startswith("{"):
                m = json.loads(ln)
                if m.get("reason") == "compiler-artifact" and m.get("executable") and m.get("profile", {}).get("test"):
                    exe = m["executable"]
        if not exe:
            print("  the rates test did not build; each gate process builds it", flush=True)
            return None
        out = self.logs / f"everyday{Path(exe).suffix}"
        shutil.copy2(exe, out)
        return out

    def load(self):
        self.step("load", [*self.tw("t2213_packs_load"), self.logs / "packs_load_after.json",
                           "--against", BASELINE, "--jobs", str(self.jobs)], check=False)

    def freeze(self):
        self.step("freeze", self.tw("freeze", "--locked"))  # this run holds the packs lock

    def go(self, start):
        sys.path.insert(0, str(HERE))
        import shared_lock
        subprocess.run([PY, MAPS / "cores.py", "claim", str(self.jobs), "light-pass", "--pid", str(os.getpid()),
                        f"T-3244 light pass {self.batch}"], cwd=str(MAPS))
        print(f"light pass {self.batch}: {len(self.air)} packs with airports, {len(self.rest)} without; jobs {self.jobs}; "
              f"logs {self.logs}", flush=True)
        t0 = time.time()
        todo = self.steps or STEPS[STEPS.index(start):]
        # the rates test from sim/ as it is at the start, so a sim edit meanwhile changes nothing (T-931)
        self.exe = self.everyday_exe() if "gate" in todo else None
        try:
            with shared_lock.held("packs", f"light_pass {self.batch}"):
                for s in todo:
                    getattr(self, s)()
                    (self.logs / "done.json").write_text(json.dumps({"last": s, "times": self.times}, indent=1))
        finally:
            subprocess.run([PY, MAPS / "cores.py", "release", "light-pass"], cwd=str(MAPS))
            print(f"light pass: {time.time() - t0:.0f} s; times {self.times}", flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] == ["trips"]:  # the trips step, in its own process
        import shared_lock
        import zones
        with shared_lock.held("registry", "light_pass trips"):
            reg = shared_lock.read_json(REG)
            t = zones.local_trips(reg)
            shared_lock.write_atomic(REG, json.dumps(reg, indent=2, ensure_ascii=False) + "\n")
        print(f"{len(t)} zones' localTrips, {sum(t.values()):,.0f} in all")
        sys.exit(0)
    if not args or args[0] not in ("run", "plan"):
        sys.exit(__doc__)
    r = Run(args)
    if args[0] == "plan":
        print(f"{len(r.air)} packs with airports, {len(r.rest)} without; steps {' '.join(STEPS)}; logs {r.logs}")
        shutil.rmtree(r.logs)
    else:
        r.go(opt(args, "--from", STEPS[0]))
