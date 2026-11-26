"""Write a plan_edit spec from unit names: each row's wanted units by name; moves for every unit whose row changes.
usage: from_names.py <draft.json> <spec.json>
draft: the spec's own fields (task, iso, label, view, new, set, retire, names, why, note) plus
  "assign": {"<row>": ["<unit name>", ...]}   names matched among units of the rows named in assign (and their
                                               current owners), exact; "Name@lon,lat" picks among same-named units."""
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plan_edit as pe  # noqa: E402
import shared_lock  # noqa: E402
import shapely  # noqa: E402
import world_plan as wp  # noqa: E402


def main():
    draft = json.load(open(sys.argv[1], encoding="utf-8"))
    iso = draft["iso"]
    by = pe.load_units(iso)
    doc = shared_lock.read_json(pe.PLAN)
    owner = {u: r["id"] for r in doc["metros"] for u in (r.get("units") or {}).get(iso, [])}
    rows = set(draft["assign"])
    pool = list(by.values())
    moves = []
    for row, names in draft["assign"].items():
        for nm in names:
            at = None
            if "@" in nm:
                nm, xy = nm.split("@")
                at = shapely.Point(*map(float, xy.split(",")))
            cands = [u for u in pool if u.name == nm]
            if len(cands) > 1 and "view" in draft:  # same-named units elsewhere in the country: those in view
                vb = shapely.box(*draft["view"])
                cands = [u for u in cands if wp.to_ll(u.geom).intersects(vb)]
            if at is not None and cands:
                cands = [min(cands, key=lambda u: wp.to_ll(u.geom).distance(at))]
            if len(cands) != 1:
                raise SystemExit(f"{nm}: {len(cands)} units match ({[(u.id, owner.get(u.id)) for u in cands]})")
            u = cands[0]
            if owner.get(u.id) != row:
                moves.append([u.id, u.name, owner.get(u.id), row])
    spec = {k: v for k, v in draft.items() if k != "assign"}
    spec["moves"] = moves
    json.dump(spec, open(sys.argv[2], "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"{len(moves)} moves -> {sys.argv[2]}")


if __name__ == "__main__":
    main()
