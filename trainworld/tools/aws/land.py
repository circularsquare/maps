"""T-1065: a wave's results from the inbox into the tree. docs/aws.md rule 9. Makes no AWS call.

    land.py <run> [--only name,name] [--dry]

For each artifacts/trainworld/aws/inbox/<run>/results_<name> (bucket.py get), unpacked beside it:
every row the machine left "built" with its four pack files whose plan row here is planned, parked
or built is landed: its files (manifest.json's list, each checked against its SHA-256) copied into
data/ through a temporary name, the batch logs into artifacts/trainworld/world_run/<batch>/, then,
under the plan lock, its status set to "built" with the machine's run record. A state cache
(data/work/usa/blocks_<fips>.npz) already here stays as it is. Rows shipped or gated here (the check
row) and rows the machine parked are listed, never landed. Refuses while a gate or joint solve is
under way, or a live run claims one of the rows: they would see half a region.

Then `world_run.py run --rows <ids> --no-join` gates the landed rows (a built row with its packs
skips the download and build).
"""
import hashlib
import json
import os
import shutil
import sys
import tarfile
import time
from pathlib import Path

import awslib as A

TW = A.TW
sys.path.insert(0, str(TW / "pipeline"))
import shared_lock  # noqa: E402

PACK = (".json", ".bin", ".boundary.geojson", ".places.json")


def busy(ids):
    """What landing these rows now would cut into: a gate or a joint solve under way (they snapshot
    and re-gate the registered packs), or a live run that claims one of the rows. Another run's
    builds of other rows are no obstacle (T-1078 built wave 2 while wave 1 landed)."""
    locks = TW / "data/work/world_run/locks"
    out = [p.stem for p in locks.glob("gate-*.lock") if shared_lock.is_held(p.stem)]
    out += ["packs (a joint solve)"] if shared_lock.is_held("packs") else []
    plan = shared_lock.read_json(TW / "pipeline/world_plan.json")
    for r in plan["metros"]:
        claim = (r.get("run") or {}).get("claim")
        if r.get("id") in ids and claim and shared_lock.is_held(f"batch-{claim}"):
            out.append(f"{r['id']} claimed by run {claim}")
    return out


def unpack(tgz, dest):
    """The tar into dest; every member a plain file or folder under it."""
    if (dest / "manifest.json").exists():
        return
    part = dest.with_name(dest.name + ".part")
    shutil.rmtree(part, ignore_errors=True)
    part.mkdir(parents=True)
    with tarfile.open(tgz) as t:
        for m in t.getmembers():
            p = Path(m.name)
            if p.is_absolute() or ".." in p.parts or not (m.isfile() or m.isdir()):
                raise SystemExit(f"{tgz}: member {m.name!r} is not a plain file under the tar; nothing landed")
        t.extractall(part)
    os.replace(part, dest)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def place(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".landing")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def main(argv):
    if not argv or argv[0].startswith("-"):
        raise SystemExit(__doc__)
    run, dry = argv[0], "--dry" in argv
    only = set(argv[argv.index("--only") + 1].split(",")) if "--only" in argv else None
    box = A.INBOX / run
    tgzs = sorted(p for p in box.glob("results_*") if p.is_file() and (only is None or p.name[8:] in only))
    if not tgzs:
        raise SystemExit(f"no results_* in {box}: bucket.py get <run>/results_<name> first")
    plan_path = TW / "pipeline/world_plan.json"
    home = {r["id"]: r for r in shared_lock.read_json(plan_path)["metros"] if r.get("units")}
    landed, skipped, records = [], [], {}
    for tgz in tgzs:
        name = tgz.name[len("results_"):]
        dest = box / f"x_{name}"
        unpack(tgz, dest)
        man = json.loads((dest / "manifest.json").read_text(encoding="utf-8"))
        held = busy(set(man["ids"]))
        if held:
            raise A.Refused(f"{', '.join(held)}: land between gates and joins, never mid-batch")
        bad =[f for f, v in man["files"].items() if not (dest / f).is_file() or os.path.getsize(dest / f) != v["size"]
               or sha(dest / f) != v["sha256"]]
        if bad:
            raise SystemExit(f"{tgz}: {len(bad)} files differ from its manifest ({bad[:3]}); nothing of it landed")
        for cid in man["ids"]:
            row = man["rows"].get(cid) or {}
            files = man["row_files"].get(cid, [])
            why = None
            if row.get("status") != "built":
                why = f"the machine left it {row.get('status')}: {(row.get('run') or {}).get('reason', '')[:300]}"
            elif any(f"trainworld/data/packs/{cid}{x}" not in files for x in PACK):
                why = "pack files missing: " + ", ".join(x for x in PACK if f"trainworld/data/packs/{cid}{x}" not in files)
            elif cid not in home:
                why = "not a plan row here"
            elif home[cid]["status"] not in ("planned", "parked", "built"):
                why = f"{home[cid]['status']} here: not replaced"
            if why:
                skipped.append((name, cid, why))
                continue
            kept = []
            for f in files:
                dst = A.MAPS / f
                if f.startswith("trainworld/data/work/usa/") and dst.exists():
                    if sha(dst) != man["files"][f]["sha256"]:
                        kept.append(f.rsplit("/", 1)[1])
                    continue
                if not dry:
                    place(dest / f, dst)
            rec = {k: v for k, v in (row.get("run") or {}).items() if k != "claim"}
            rec["landed"] = {"run": run, "name": name, "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
            records[cid] = rec
            landed.append(cid)
            print(f"  {name} {cid}: {len(files)} files{'; state caches kept as here (differ): ' + ' '.join(kept) if kept else ''}")
        for f in man["files"]:
            if f.startswith("artifacts/") and not dry:
                dst = A.MAPS / f
                if not dst.exists():
                    place(dest / f, dst)
    if landed and not dry:
        with shared_lock.held("plan", f"land.py {run}: {' '.join(landed)}"[:200]):
            doc = shared_lock.read_json(plan_path)
            for r in doc["metros"]:
                if r.get("units") and r["id"] in records:
                    r["status"] = "built"
                    r["run"] = records[r["id"]]
            shared_lock.write_atomic(plan_path, json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
        log = box / "landed.json"
        old = json.loads(log.read_text(encoding="utf-8")) if log.exists() else []
        old.append({"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "landed": landed, "skipped": skipped})
        log.write_text(json.dumps(old, indent=1), encoding="utf-8")
    print(f"{'would land' if dry else 'landed'} {len(landed)} rows: {','.join(landed)}")
    for name, cid, why in skipped:
        print(f"  not landed: {name} {cid}: {why}")


if __name__ == "__main__":
    main(sys.argv[1:])
