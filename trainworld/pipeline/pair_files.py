"""T-1976: what each joint commutes file (`*.joint_commutes.json`, intercity.py) and cross-everyday
file (`*.cross_everyday.json`, cross_everyday.py) in a data/work holds, kept in
`<work>/pair_files.index.json` so a reader parses only the files touching the packs it wants.

Per file: its solved packs (in the file's order), its packs, its time (`solved_at`, else the file's
modified time) and the packs its pairs start from (`homes`). An entry holds while the file's size and
modified time do; a new or rewritten file is read again on the next call.

    python pipeline/pair_files.py unused [--move]

lists the files that give market_zones.py no pair for any registered pack or world plan row (every
pair replaced by a newer file's, or naming a pack that is gone: a replan's old units), and with
`--move` moves them to data/work/retired/ (never deleted; under the registry and packs locks).
"""
import json
import os
import sys
from pathlib import Path

GLOB = {"joint": "*.joint_commutes.json", "everyday": "*.cross_everyday.json"}
# The everyday kinds an everyday file's activity end's pack draws (school, university and, from
# T-2896, airport): the file's own `site_kinds` where it has them.
SITE_KINDS = (2, 3)


def site_kinds(g):
    return tuple(g.get("site_kinds", SITE_KINDS))
NAME = "pair_files.index.json"
FORMAT = 1
INDEX = None  # another index file than <work>/NAME (a test's)
ROOT = Path(__file__).resolve().parents[1]


def _entry(p, st, kind):
    g = json.loads(p.read_text(encoding="utf-8"))
    ids = g["packs"]
    # a joint file from before T-786 solved every pack it lists
    solved = (g.get("solved") or ids) if kind == "joint" else g["solved"]
    return {"stat": [st.st_size, st.st_mtime_ns], "solved": list(solved), "packs": list(ids),
            "t": float(g.get("solved_at") or st.st_mtime), "homes": sorted({ids[k] for k in set(g["home_pack"])})}


def files(work, kind, index=None):
    """{path: entry} for every `kind` ("joint" or "everyday") file in `work`, oldest first (by time,
    then name), as intercity.joint_files and cross_everyday.unit_files list them."""
    work = Path(work)
    index = Path(index or INDEX or work / NAME)
    try:
        doc = json.loads(index.read_text(encoding="utf-8"))
        if doc.get("format") != FORMAT:
            doc = None
    except (OSError, ValueError):
        doc = None
    doc = doc or {"format": FORMAT, "files": {}}
    known, changed, out = doc["files"], False, {}
    for p in sorted(work.glob(GLOB[kind])):
        try:
            st = p.stat()
        except FileNotFoundError:  # removed by another writer since the listing
            continue
        e = known.get(p.name)
        if not e or e["stat"] != [st.st_size, st.st_mtime_ns]:
            e = known[p.name] = _entry(p, st, kind)
            changed = True
        out[p] = e
    for name in [n for n in known if n.endswith(GLOB[kind][1:]) and not (work / n).exists()]:
        del known[name]
        changed = True
    if changed:
        # a lost race with another writer only means a file is read again next time
        part = index.with_name(f"{index.name}.{os.getpid()}.part")
        part.write_text(json.dumps(doc, separators=(",", ":")), encoding="utf-8")
        try:
            os.replace(part, index)
        except PermissionError:
            part.unlink()
    return dict(sorted(out.items(), key=lambda kv: kv[1]["t"]))


def unused(work, known):
    """The joint and everyday files of `work` from which market_zones.Groups takes no pair while the
    packs `known` are registered: {path: MB}."""
    out = {}
    joint = list(files(work, "joint").items())
    for i, (p, f) in enumerate(joint):
        later = set().union(*(g["solved"] for _, g in joint[i + 1:]))
        if not (set(f["homes"]) & known) - later:
            out[p] = f
            continue
        g = json.loads(p.read_text(encoding="utf-8"))
        ids = g["packs"]
        if not any(h in known and w in known and h not in later and w not in later
                   for h, w in ((ids[a], ids[b]) for a, b in zip(g["home_pack"], g["work_pack"]))):
            out[p] = f
    every = files(work, "everyday")
    last = {}
    for p, f in every.items():
        for o in f["solved"]:
            last[o] = p
    for p, f in every.items():
        mine = {o for o, q in last.items() if q == p}
        if mine and set(f["homes"]) & known:
            g = json.loads(p.read_text(encoding="utf-8"))
            ids, site = g["packs"], site_kinds(g)
            if any((ids[a] if k in site else ids[h]) in mine and ids[h] in known and ids[a] in known
                   for h, a, k in zip(g["home_pack"], g["act_pack"], g["kind"])):
                continue
        out[p] = f
    return {p: round(p.stat().st_size / 1e6, 1) for p in out}


def main(argv):
    sys.path.insert(0, str(Path(__file__).parent))
    import shared_lock
    work = ROOT / "data/work"

    def known():
        reg = shared_lock.read_json(ROOT / "app/src/game/registry.json")
        plan = shared_lock.read_json(ROOT / "pipeline/world_plan.json")
        # a plan row's id counts too: a parked row registered again would take its pairs back
        return {c["id"] for c in reg["cities"]} | {r["id"] for r in plan["metros"] if r.get("id")}

    def state():
        return known(), {p.name: p.stat().st_mtime_ns for g in GLOB.values() for p in work.glob(g)}
    # worked out outside the locks (minutes: most files are read whole), then checked unchanged under them
    before = state()
    dead = unused(work, before[0])
    for p, mb in dead.items():
        print(f"{p.name} {mb} MB")
    print(f"{len(dead)} files, {sum(dead.values()):.0f} MB, give no registered pack or plan row a pair")
    if "--move" not in argv or not dead:
        return
    with shared_lock.held("registry", "pair_files unused"), shared_lock.held("packs", "pair_files unused"):
        if state() != before:
            sys.exit("the registry, the plan or a file changed while listing: nothing moved; run it again")
        if dead:
            to = work / "retired"
            to.mkdir(exist_ok=True)
            for p in dead:
                q = to / p.name
                if q.exists():  # an older file of the same name moved before: both kept
                    q = to / f"{p.name[:-5]}.{p.stat().st_mtime_ns}.json"
                os.replace(p, q)
            files(work, "joint")
            files(work, "everyday")
            print(f"moved to {to}")


if __name__ == "__main__":
    main(sys.argv[1:])
