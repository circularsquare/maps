"""The fix batch's joins (T-1000, T-1091): rerun shipped solve units with world_run's own steps, no
parking. From the maps root:

    venv\\Scripts\\python.exe fixjoin.py --order KEY,KEY,.. --jobs N [--no-market] [--dry]

KEY is a solve key (intercity.solve_key: "JP", "GB", "CN-qinghai"); units go in --order's order and a
key may repeat (solved again). Each key's files (members' packs, res-9 files, its units' joint files,
market packs) are backed up to data/work/pre-fixjoin/<key>/ before its first unit, unless that folder
exists; the shared files (registry, market rows, world plan, every joint file, the fixed neighbours'
market packs) once a run to data/work/pre-fixjoin/shared-<batch>/. Per unit, as world_run.Run.solve:
snapshot, merge_cells --undo, intercity --solve (with --hold-report when it has fixed packs),
merge_cells, pack_gate --no-rust --since. A failed step or a gate FAIL restores that key from its
backup and skips its later units; nothing parks, nothing is unregistered. Then one register --market
--trips on every changed pack (members and fixed of the units that held, `--also KEY,..` units an
earlier stopped run solved, `--extra id,..`). Before each unit: cores from data/work/pre-fixjoin/JOBS
if present (else 6 at most from 07:45), the STOP file, and under 2 GB free it stops. Holds the packs
lock and a batch lock throughout. Written for T-1000's fix batch (notes/T-1000.md "Fix batch: joins").
"""
import json
import os
import shutil
import sys
import time
from pathlib import Path

MAPS = Path(r"C:\Users\anita\projects\maps")
TW = MAPS / "trainworld"
sys.path[:0] = [str(TW / "pipeline")]
import world_run  # noqa: E402

world_run.STEP_LIMIT_S["register"] = 4 * 3600
world_run.STEP_LIMIT_S["intercity"] = 4 * 3600
CHUNK = 100000  # one register call: its market rows rebuild is a fixed cost a call


def cc_of(key):
    """The backup and failure unit: the solve key itself (a country, or one Chinese province)."""
    return key


class Fix(world_run.Run):
    def go_fix(self, order, market=True, dry=False):
        import intercity
        reg = self.registry()
        units = intercity.solve_units(reg, self.root)
        by = {}
        for u in units:
            by.setdefault(u["country"], []).append(u)
        missing = [k for k in order if k not in by]
        if missing:
            raise SystemExit(f"no solve unit for {missing}")
        seq = [u for k in order for u in by[k]]
        print(f"batch {self.batch}: {len(seq)} unit solves, jobs {self.jobs}, logs {self.logs}", flush=True)
        for u in seq:
            print(f"  {u['country']:>18} {u['name']}: {len(u['members'])} packs, {len(u['fixed'])} fixed", flush=True)
        if dry:
            return
        bak_root = self.work / "pre-fixjoin"
        shared_bak = bak_root / f"shared-{self.batch}"
        self.backup_shared(shared_bak, seq)
        failed, done, changed = {}, [], set()
        t0 = time.time()
        with self.held("packs", "fix batch joins"):
            self.wait_gates()
            for k, u in enumerate(seq):
                jf = bak_root / "JOBS"  # cores for the next unit, changeable mid-run (Anita's hours)
                try:
                    j = int(jf.read_text().strip())
                    if j != self.jobs:
                        print(f"  jobs {self.jobs} -> {j} ({jf})", flush=True)
                        self.jobs = j
                except (OSError, ValueError):
                    pass
                if not jf.exists() and time.localtime().tm_hour * 60 + time.localtime().tm_min >= 7 * 60 + 45 and self.jobs > 6:
                    print(f"  jobs {self.jobs} -> 6 (Anita on the machine from 08:00)", flush=True)
                    self.jobs = 6
                free = shutil.disk_usage(self.root).free / 1e9
                if free < 2.0:  # other agents share the disk; a full one stopped the first Europe run mid-backup
                    print(f"  {free:.1f} GB free: units from {u['name']} ({k + 1}/{len(seq)}) not run", flush=True)
                    self.report["stopped_at"] = k
                    self.report.setdefault("problems", []).append(f"disk: {free:.1f} GB free")
                    break
                if self.stop_file.exists():
                    print(f"  STOP file: units from {u['name']} ({k + 1}/{len(seq)}) not run", flush=True)
                    self.report["stopped_at"] = k
                    break
                cc = cc_of(u["country"])
                if cc in failed:
                    print(f"  skip {u['name']}: {cc} failed earlier", flush=True)
                    continue
                bak = bak_root / cc
                if not (bak / "manifest.json").exists():
                    self.backup_country(cc, [x for x in units if cc_of(x["country"]) == cc], bak)
                u = self.live_unit(u)
                why = self.solve_fix(u, f"{self.logs / 'join'}.{k:03d}-{u['name']}")
                if why:
                    print(f"  FAIL {u['name']}: {why}", flush=True)
                    self.restore_country(bak, [x["name"] for x in units if cc_of(x["country"]) == cc])
                    failed[cc] = f"{u['name']}: {why}"
                    changed -= {i for x in units if cc_of(x["country"]) == cc for i in x["members"]}
                    continue
                done.append(u["name"])
                changed |= set(u["members"]) | set(u.get("fixed", []))
                print(f"  {u['name']} done ({k + 1}/{len(seq)}, {(time.time() - t0) / 60:.0f} min)", flush=True)
            # a failed country's own packs are as before; its neighbours' market packs still take the
            # commutes the units that held wrote
            changed |= set(self.o.get("extra", []))  # packs re-solved alone beside the run (T-1091's unjoined)
            for k in self.o.get("also", []):  # units an earlier, stopped run solved: to the market step too
                for x in by[k]:
                    changed |= set(x["members"]) | set(x.get("fixed", []))
            order_ids = [c["id"] for c in self.registry()["cities"]]
            ids = sorted((i for i in changed if i in order_ids), key=order_ids.index)
            if market and ids:
                for s in range(0, len(ids), CHUNK):
                    part = ids[s:s + CHUNK]
                    ok, secs, tail = self.step("register", part + ["--market", "--trips"],
                                               self.logs / f"market-{s // CHUNK:02d}.log")
                    print(f"  register {s}-{s + len(part)}: {'ok' if ok else 'FAILED'} {secs:.0f} s", flush=True)
                    if not ok:
                        self.report.setdefault("problems", []).append(f"market step failed at {part[0]}: {str(tail)[-300:]}")
                        break
        self.report.update({"done": done, "failed": failed, "changed": len(ids), "seconds": round(time.time() - t0)})
        (self.logs / "fixjoin.json").write_text(json.dumps(self.report, indent=1), encoding="utf-8")
        print(f"\nfix joins: {len(done)} units held, failed {failed or 'none'}, {len(ids)} packs to the market step, "
              f"{(time.time() - t0) / 60:.0f} min; problems {self.report.get('problems')}", flush=True)

    def solve_fix(self, u, tag):
        g, fixed, name = u["members"], u.get("fixed", []), u["name"]
        print(f"  unit {name}: {' '.join(g)}{'; fixed: ' + ' '.join(fixed) if fixed else ''}", flush=True)
        self.check_sim()
        snap = Path(f"{tag}.snapshot.json")
        self.step("pack_gate", ["snapshot", str(snap)], f"{tag}.log", threads=1)
        solve = ["--solve", ",".join(g), "--name", name]
        if fixed:
            solve += ["--fixed", ",".join(fixed), "--hold-report", f"{tag}.hold.json"]
        for mod, args in (("merge_cells", g + ["--undo"]), ("intercity", solve), ("merge_cells", g)):
            ok, _, tail = self.step(mod, args, f"{tag}.log", big=mod == "intercity" and len(g) > 100)
            if not ok:
                return f"{mod} failed: {str(tail)[-300:]}"
        gd = Path(f"{tag}-gate")
        ok, _, tail = self.step("pack_gate", g + ["--no-rust", "--since", str(snap), "--out", str(gd)], f"{tag}.gate.log")
        bad = {}
        for i in g:
            p = gd / f"{i}.json"
            if not p.exists():
                bad[i] = "no gate result"
                continue
            res = json.loads(p.read_text(encoding="utf-8"))
            if res["verdict"] != "ship":
                bad[i] = ", ".join(f"{c['name']} {c.get('value')}" for c in res["checks"] if c["status"] == "FAIL")
        snap.unlink(missing_ok=True)
        if bad:
            return "gate: " + "; ".join(f"{i}: {v}" for i, v in bad.items())
        return None

    def copy_in(self, files, bak):
        bak.mkdir(parents=True, exist_ok=True)
        manifest, size = [], 0
        for p in files:
            if p.exists():
                rel = p.relative_to(self.root)
                (bak / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, bak / rel)
                manifest.append(str(rel))
                size += p.stat().st_size
        (bak / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
        print(f"  backup {bak}: {len(manifest)} files, {size / 1e9:.2f} GB", flush=True)

    def backup_country(self, cc, cunits, bak):
        members = [i for u in cunits for i in u["members"]]
        files = [p for p in self.group_files(members) if not p.name.endswith(".joint_commutes.json")]
        files += [self.work / f"{u['name']}.joint_commutes.json" for u in cunits]
        files += [self.packs / f"{i}.market.json" for i in members]
        self.copy_in(list(dict.fromkeys(files)), bak)

    def backup_shared(self, bak, seq):
        mine = {i for u in seq for i in u["members"]}
        fixed = {i for u in seq for i in u.get("fixed", [])} - mine
        files = [self.reg_path, self.root / "app/src/game/market-rows.json", self.work / "market_rows.json",
                 self.plan_path] + [self.packs / f"{i}.market.json" for i in sorted(fixed)]
        files += sorted(self.work.glob("*.joint_commutes.json"))
        self.copy_in(files, bak)

    def restore_country(self, bak, names):
        """The country's packs and its own units' joint files (named after them) as backed up; joint
        files of other countries' units, which the backup also holds where they name its packs, stay."""
        manifest = json.loads((bak / "manifest.json").read_text(encoding="utf-8"))
        own = {f"{n}.joint_commutes.json" for n in names}
        manifest = [r for r in manifest if not r.endswith(".joint_commutes.json") or Path(r).name in own]
        held = {Path(r).name for r in manifest}
        for n in own - held:  # a unit's joint file the backup lacks: written by this run
            p = self.work / n
            if p.exists():
                print(f"  remove {p.name}", flush=True)
                p.unlink()
        for rel in manifest:
            part = self.root / (rel + ".part")
            shutil.copy2(bak / rel, part)
            os.replace(part, self.root / rel)
        print(f"  restored {len(manifest)} files from {bak}", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    a = sys.argv[1:]
    order = a[a.index("--order") + 1].split(",")
    jobs = a[a.index("--jobs") + 1] if "--jobs" in a else "2"
    f = Fix({"jobs": jobs, "extra": a[a.index("--extra") + 1].split(",") if "--extra" in a else [],
             "also": a[a.index("--also") + 1].split(",") if "--also" in a else []})
    if "--dry" in a:
        f.go_fix(order, dry=True)
        sys.exit()
    if f.stop_file.exists():
        raise SystemExit(f"{f.stop_file} exists")
    f.check_sim()
    with f.start():
        f.go_fix(order, market="--no-market" not in a)
