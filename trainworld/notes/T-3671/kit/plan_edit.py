"""Hand edits to built world plan rows, for a targeted rebuild (nightprep, 2026-11-17): units moved between rows,
rows retired into a neighbour, row fields set (module, archetype, gate_relax), every touched row set back to
planned, and zone-names.json entries. One JSON spec a task:

    {"task": "T-3488", "iso": "TUR",
     "moves": [["<unit id>", "<unit name>", "<from row>", "<to row>"], ...],
     "retire": {"<row>": "<row taking all its units>"},
     "set": {"<row>": {"module": "admin_totals", "archetype": "C", "gate_relax": {"pop_vs_plan": "<why>"}}},
     "replan": ["<row>", ...],            rows rebuilt with no unit change
     "names": {"drop": ["<id>"], "aliases": {"<id>": ["..."]}},
     "why": "..."}

    plan_edit.py <spec.json>               dry: what changes, each touched row's pieces before and after, its
                                           neighbours, and the build waves (written to <spec>.waves.json)
    plan_edit.py <spec.json> --go          apply under the plan lock (world_plan.json, extents.geojson), then names
    plan_edit.py <spec.json> retire [--go] unregister each retired row (world_run's unregister, market heal
                                           included) and move its data/packs/<id>.* files to data/work/pre-<task>/retired/

Waves: a row that takes units from another row is built after it (a build cuts every pack outline on disk out
of its own, osm_units.cut_packs, so a taker built first would lose the moved units to the giver's old outline);
retired rows give nothing, since their files leave data/packs before the builds. People are the plan state's
unit people (data/work/worldplan/state.pkl), bbox and extracts as world_plan's own joined rows (T-3287).
"""
import collections
import glob
import json
import os
import pickle
import shutil
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
TW = r"C:\Users\anita\projects\maps\trainworld"
sys.path.insert(0, os.path.join(TW, "pipeline"))
import shared_lock  # noqa: E402

PLAN = os.path.join(TW, "pipeline", "world_plan.json")
EXT = os.path.join(TW, "data", "work", "worldplan", "extents.geojson")
STATE = os.path.join(TW, "data", "work", "worldplan", "state.pkl")
NAMES = os.path.join(TW, "app", "src", "game", "zone-names.json")
PACKS = os.path.join(TW, "data", "packs")
TOUCH_M = 500.0  # units this close are one piece (world_plan.TOUCH_M)


def load_units(iso):
    import world_plan
    import __main__
    __main__.Unit = world_plan.Unit  # the pickle may name the class from a run as a script
    st = pickle.loads(open(STATE, "rb").read())
    return {u.id: u for u in st["units"] if u.iso == iso}


def pieces(ids, by):
    """Connected pieces of a unit list (touching within TOUCH_M), as lists of names."""
    import shapely
    us = [by[i] for i in ids]
    geoms = [u.geom.buffer(TOUCH_M / 2) for u in us]
    tree = shapely.STRtree(geoms)
    parent = list(range(len(us)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for k, g in enumerate(geoms):
        for j in tree.query(g, predicate="intersects"):
            parent[find(int(j))] = find(k)
    out = collections.defaultdict(list)
    for k, u in enumerate(us):
        out[find(k)].append(u.name)
    return sorted(out.values(), key=len, reverse=True)


def neighbours(ids, by, owner):
    """Rows sharing this unit list's outline, with their share of it."""
    import shapely
    g = shapely.union_all([by[i].geom for i in ids])
    edge = g.boundary
    L = edge.length or 1.0
    share = collections.Counter()
    gb = g.buffer(TOUCH_M)
    for u in by.values():
        if u.id in ids or not u.geom.intersects(gb):
            continue
        share[owner.get(u.id, "(no row)")] += edge.intersection(u.geom.buffer(TOUCH_M)).length / L
    return {k: round(v, 2) for k, v in share.most_common()}


def plan_change(spec, doc, by):
    """(new rows by id, removed ids, waves, report lines) for the spec against plan doc; SystemExit on any fault."""
    iso = spec["iso"]
    rows = {r["id"]: r for r in doc["metros"] if r.get("units")}
    owner = {u: r["id"] for r in rows.values() for u in r["units"].get(iso, [])}
    new = {}
    lines = []

    def row(k):
        if k not in rows:
            raise SystemExit(f"{k}: no such row")
        if k not in new:
            r = rows[k]
            if r["status"] not in ("shipped", "planned", "parked"):
                raise SystemExit(f"{k}: status {r['status']} (a run has it); not edited")
            new[k] = json.loads(json.dumps(r))
        return new[k]
    gives = collections.defaultdict(set)  # taker -> givers
    for uid, name, a, b in spec.get("moves", []):
        if owner.get(uid) != a:
            raise SystemExit(f"{name} ({uid}) is in {owner.get(uid)}, not {a}")
        if uid not in by:
            raise SystemExit(f"{name} ({uid}) is not in the plan state's units")
        rb = row(b)
        if a is not None:  # a is None: a free unit (in no row) joins b (T-3671)
            row(a)["units"][iso].remove(uid)
            gives[b].add(a)
        rb["units"][iso].append(uid)
        lines.append(f"move {name} ({by[uid].pop:,.0f}) {a} -> {b}")
    gone = []
    for k, into in spec.get("retire", {}).items():
        r, ri = row(k), row(into)
        for uid in list(r["units"][iso]):
            ri["units"][iso].append(uid)
            lines.append(f"retire {k}: {by[uid].name} ({by[uid].pop:,.0f}) -> {into}")
        gone.append(k)
    for k, fields in spec.get("set", {}).items():
        r = row(k)
        for f, v in fields.items():
            if f == "gate_relax":
                r.setdefault("gate_relax", {}).update(v)
            else:
                r[f] = v
        lines.append(f"set {k}: {fields}")
    for k in spec.get("replan", []):
        row(k)
    # units of rows that are not retired are recomputed; the people as the plan state has them
    import shapely
    import world_plan as wp
    ex, kids = wp.geofabrik()
    others = {}
    for k, r in new.items():
        if k in gone:
            continue
        us = r["units"][iso]
        if set(us) != set(rows[k]["units"][iso]):
            # T-3671: people by delta, as world_plan grows a row (a row's people need not be its state units' sum:
            # cai's was 1.09M over it), and the bbox only grows when units only come in
            old_us = set(rows[k]["units"][iso])
            added, lost = [u for u in us if u not in old_us], [u for u in old_us if u not in set(us)]
            r["people"] = round(rows[k]["people"] + sum(by[u].pop for u in added) - sum(by[u].pop for u in lost))
            g = wp.polygonal(shapely.union_all([by[u].geom for u in us]))
            ll = wp.to_ll(g)
            bb = [round(v, 3) for v in ll.bounds]
            if not lost:
                ob = rows[k]["bbox"]
                bb = [min(ob[0], bb[0]), min(ob[1], bb[1]), max(ob[2], bb[2]), max(ob[3], bb[3])]
            lines.append(f"{k}: bbox {rows[k]['bbox']} -> {bb}")
            r["bbox"] = bb
            exs = wp.extracts(ll, ex, kids)
            if exs != r["extracts"]:
                lines.append(f"{k}: extracts {r['extracts']} -> {exs}")
                r["extracts"] = exs
            names = [by[u].name for u in us]
            src = r.setdefault("source", {})
            if "unit_names" in src:
                src["unit_names"] = names
            src["hand_edit"] = {"task": spec["task"], "why": spec.get("why", "")}
            others[k] = shapely.set_precision(ll, 1e-5)
        r["status"] = "planned"
    owner2 = dict(owner)
    for k, r in new.items():
        if k not in gone:
            for u in r["units"][iso]:
                owner2[u] = k
    for k, r in sorted(new.items()):
        if k in gone:
            continue
        before = pieces(rows[k]["units"][iso], by)
        after = pieces(r["units"][iso], by)
        lines.append(f"{k} {r['name']}: people {rows[k]['people']:,} -> {r['people']:,}; units {len(rows[k]['units'][iso])} -> "
                     f"{len(r['units'][iso])}; pieces {len(before)} -> {len(after)}"
                     + ("" if len(after) <= 1 else f" (smaller: {after[1:]})"))
        if len(after) > len(before) and k not in spec.get("water_ok", []):  # water_ok: an island joins on purpose
            raise SystemExit(f"{k}: the edit splits it ({len(before)} -> {len(after)} pieces: {after}); not applied")
        if set(r["units"][iso]) != set(rows[k]["units"][iso]):
            lines.append(f"   neighbours before {neighbours(set(rows[k]['units'][iso]), by, owner)}")
            lines.append(f"   neighbours after  {neighbours(set(r['units'][iso]), by, owner2)}")
    # build waves: takers after their (non-retired) givers
    todo = [k for k in new if k not in gone]
    waves, done = [], set()
    while todo:
        w = [k for k in todo if not (gives.get(k, set()) - set(gone) - done)]
        if not w:
            raise SystemExit(f"rows give units to each other: {todo}; no build order")
        waves.append(sorted(w))
        done |= set(w)
        todo = [k for k in todo if k not in done]
    return new, gone, waves, others, lines


def apply_names(spec, go):
    n = spec.get("names")
    if not n:
        return
    text = open(NAMES, encoding="utf-8").read()
    have = json.loads(text)
    out = text.splitlines()
    for k in n.get("drop", []):
        if k in have:
            out = [l for l in out if not l.startswith(f'"{k}": ')]
            print(f"names: drop {k} {have[k]}")
    for k, al in n.get("aliases", {}).items():
        e = dict(have.get(k, {}))
        e["aliases"] = list(dict.fromkeys(e.get("aliases", []) + al))
        line = f'"{k}": {json.dumps(e, ensure_ascii=False)}'
        if k in have:
            out = [line + ("," if l.rstrip().endswith(",") else "") if l.startswith(f'"{k}": ') else l for l in out]
        else:
            out.insert(1, line + ",")
        print(f"names: {k} {e}")
    body = "\n".join(out) + "\n"
    # a dropped last entry leaves a trailing comma before the closing brace
    body = body.replace(",\n}", "\n}")
    json.loads(body)
    if go:
        tmp = NAMES + ".part"
        open(tmp, "w", encoding="utf-8").write(body)
        os.replace(tmp, NAMES)
        print("names written")


def retire(spec, go):
    sys.path.insert(0, os.path.join(TW, "pipeline"))
    import world_run
    doc = shared_lock.read_json(PLAN)
    left = {r["id"] for r in doc["metros"] if r.get("units")}
    dest = os.path.join(TW, "data", "work", f"pre-{spec['task'].replace('-', '')}", "retired")
    for k in spec.get("retire", {}):
        if k in left and go:
            raise SystemExit(f"{k} is still a plan row: apply the edit (--go) first")
        files = sorted(glob.glob(os.path.join(PACKS, f"{k}.*")))
        print(f"{k}: unregister; move {len(files)} files to {dest}: {[os.path.basename(f) for f in files]}")
        if go:
            world_run.Run({"jobs": os.environ.get("NQ_JOBS", "4")}).unregister([k])
            os.makedirs(dest, exist_ok=True)
            for f in files:
                if os.path.exists(f):  # unregister deletes the market pack itself
                    shutil.move(f, os.path.join(dest, os.path.basename(f)))
            print(f"{k}: retired")


def main():
    path = sys.argv[1]
    spec = json.load(open(path, encoding="utf-8"))
    go = "--go" in sys.argv
    if "retire" in sys.argv[2:]:
        return retire(spec, go)
    by = load_units(spec["iso"])
    doc = shared_lock.read_json(PLAN)
    new, gone, waves, others, lines = plan_change(spec, doc, by)
    print("\n".join(lines))
    print("waves:", " | ".join(",".join(w) for w in waves))
    print("retired:", gone)
    json.dump({"waves": waves, "retire": gone, "rows": sorted(k for k in new if k not in gone)},
              open(os.path.splitext(path)[0] + ".waves.json", "w"), indent=1)
    if not go:
        apply_names(spec, False)
        print("(dry: nothing written)")
        return
    with shared_lock.held("plan", f"plan_edit {spec['task']}"):
        doc = shared_lock.read_json(PLAN)
        new2, gone2, _, others2, _ = plan_change(spec, doc, by)  # again on the plan as it is now
        doc["metros"] = [new2.get(r["id"], r) for r in doc["metros"] if r.get("id") not in gone2 or not r.get("units")]
        ext = shared_lock.read_json(EXT)
        from shapely.geometry import mapping
        for f in ext["features"]:
            k = f["properties"].get("id")
            if k in others2:
                f["geometry"] = mapping(others2[k])
                f["properties"]["people"] = new2[k]["people"]
        ext["features"] = [f for f in ext["features"] if f["properties"].get("id") not in gone2]
        shared_lock.write_atomic(EXT, json.dumps(ext))
        shared_lock.write_atomic(PLAN, json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    print("plan written:", sorted(new2), "removed:", gone2)
    apply_names(spec, True)


if __name__ == "__main__":
    main()
