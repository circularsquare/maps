"""T-4364: world_run's own gate for a hand pack (not a plan row: Boston, bos), as T-3688's Hokkaido driver
(notes/T-3688.md): register --market, snapshot, pack_gate --browser on the run's own Vite. A plan row passed with it
gets world_run's own row records; a hand pack's are printed only. Its join is world_run's (`join --countries US`
solves the units holding the gated rows, the hand pack a member of its unit).

    hand_run.py heal <ids>   before the build: world_run's unregister (as it takes a registered row out before
                             rebuilding it: the market pack goes, and the neighbours' market packs drop their commutes
                             into its old zones, which a smaller rebuild leaves out of range: "invalid commute between
                             zones" at the gate's register, bos 11-26), then the registry entry back as it was, since
                             register.py cannot make a hand pack's (no name in build_city.CITIES); the entry is kept in
                             data/work/hand_run/, and `gate` puts it back after a park took it out
    hand_run.py gate <ids> [--jobs N] [--port P] [--logs-from <dir with <id>.build.log>]

Exit 0 when every id shipped (passed the gate and registered); else 2.
"""
import json
import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
TW = Path(r"C:\Users\anita\projects\maps\trainworld")
stage, ids = sys.argv[1], sys.argv[2].split(",")
argv = sys.argv[3:]


def opt(name, default=None):
    return argv[argv.index(name) + 1] if name in argv else default


sys.path.insert(0, str(TW / "pipeline"))
import world_run as W  # noqa: E402

PLAN_IDS = {r["id"] for r in json.loads((TW / "pipeline" / "world_plan.json").read_text(encoding="utf-8"))["metros"]
            if r.get("units")}


class Hand(W.Run):
    def set_row(self, cid, status=None, **run):
        if cid in PLAN_IDS:
            return super().set_row(cid, status, **run)
        rec = {k: v for k, v in run.items() if k != "steps" and v is not None}
        print(f"  [{cid}] hand pack: {status or ''} {json.dumps(rec, ensure_ascii=False)[:600]}", flush=True)
        self.report.setdefault("hand_rows", {}).setdefault(cid, []).append({"status": status, **rec})


import shared_lock  # noqa: E402

REG = TW / "app" / "src" / "game" / "registry.json"
ENTRIES = TW / "data" / "work" / "hand_run"  # <id>.entry.json: a hand pack's registry entry and its place in the list


def put_back(ids):
    """The saved entries back where they stood, for hand packs the registry lacks: world_run's unregister (at a heal,
    or a gate's park) takes the entry out, and register.py cannot make a hand pack's (no name in build_city.CITIES)."""
    with shared_lock.held("registry", f"hand_run entries {' '.join(ids)}"[:200]):
        reg = shared_lock.read_json(REG)
        back = []
        for cid in ids:
            p = ENTRIES / f"{cid}.entry.json"
            if p.exists() and not any(c["id"] == cid for c in reg["cities"]):
                s = json.loads(p.read_text(encoding="utf-8"))
                reg["cities"].insert(min(s["index"], len(reg["cities"])), s["entry"])
                back.append(cid)
        if back:
            shared_lock.write_atomic(REG, json.dumps(reg, indent=2, ensure_ascii=False) + "\n")
    return back


if stage == "heal":
    ENTRIES.mkdir(parents=True, exist_ok=True)
    for i, c in enumerate(shared_lock.read_json(REG)["cities"]):
        if c["id"] in ids and c["id"] not in PLAN_IDS:
            shared_lock.write_atomic(ENTRIES / f"{c['id']}.entry.json", json.dumps({"index": i, "entry": c}, ensure_ascii=False))
    u = Hand({"jobs": opt("--jobs", "4")})
    with u.start():
        u.unregister(ids)
    print(f"healed the market packs naming {' '.join(ids)}; registry entries back: {' '.join(put_back(ids)) or 'none'}",
          flush=True)
    sys.exit(0)
if stage != "gate":
    raise SystemExit(f"unknown stage {stage}")
r = Hand({"jobs": opt("--jobs", "4"), "port": opt("--port", "6378")})
try:
    r.check_sim()
except W.RunFault as x:
    raise SystemExit(f"RUN FAULT, nothing run: {x}")
ok = False
with r.start():
    print(f"batch {r.batch}: gate {' '.join(ids)}; jobs {r.jobs}; logs {r.logs}", flush=True)
    r.sim_vs_head()
    try:
        src = opt("--logs-from")
        if src:
            for i in ids:
                p = Path(src) / f"{i}.build.log"
                if p.exists():
                    shutil.copy2(p, r.logs / f"{i}.build.log")
        back = put_back(ids)
        if back:
            print(f"  registry entries back (taken out by an earlier park): {' '.join(back)}", flush=True)
        r.check_port()
        r.everyday = r.build_everyday()
        shipped = r.gate(ids, r.logs / "gate")
        print(f"GATE passed: {' '.join(shipped) or 'none'}; parked: {' '.join(i for i in ids if i not in shipped) or 'none'}", flush=True)
        ok = set(shipped) == set(ids)
    except W.RunFault as x:
        r.report["fault"] = str(x)
        print(f"RUN FAULT: {x}", flush=True)
    finally:
        r.stop_vite()
        r.logs.mkdir(parents=True, exist_ok=True)
        (r.logs / "report.json").write_text(json.dumps(r.report, indent=1, ensure_ascii=False), encoding="utf-8")
print("HAND_RUN", "OK" if ok else "FAILED", flush=True)
sys.exit(0 if ok else 2)
