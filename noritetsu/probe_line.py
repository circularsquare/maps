"""Print one line's relations as they actually are, before trusting any of it.

    python probe_line.py --region jp --name 山手線
    python probe_line.py --region jp --ref JY --members

Used to check, on a line whose shape is known, that route_master grouping, stop roles and
way member order are what build_model.py assumes.
"""
import argparse
import pickle
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--members", action="store_true")
    ap.add_argument("--ways", action="store_true",
                    help="also report track WAYS carrying this name, for lines OSM maps as "
                         "named track with no route relation over them")
    args = ap.parse_args()
    d = ROOT / "data" / "proc" / args.region

    if args.ways:
        import numpy as np
        with open(d / "ways.pkl", "rb") as f:
            ways = pickle.load(f)
        c = np.load(d / "coords.npz")
        cid, cx, cy = c["id"], c["x"], c["y"]
        hits = {w: v for w, v in ways.items()
                if args.name and args.name in (v[0].get("name") or "")}
        km = 0.0
        ops, usages = {}, {}
        for w, (tags, nodes) in hits.items():
            pos = np.searchsorted(cid, nodes)
            np.clip(pos, 0, cid.size - 1, out=pos)
            ok = cid[pos] == nodes
            pos = pos[ok]
            if pos.size >= 2:
                lon, lat = cx[pos] / 1e7, cy[pos] / 1e7
                dx = np.diff(lon) * np.cos(np.radians(lat[:-1])) * 111.32
                dy = np.diff(lat) * 110.57
                km += float(np.hypot(dx, dy).sum())
            o = tags.get("operator", "(none)")
            ops[o] = ops.get(o, 0) + 1
            u = tags.get("usage", "(none)")
            usages[u] = usages.get(u, 0) + 1
        print(f"{len(hits)} track ways named like {args.name!r}, {km:.0f} km")
        for o, n in sorted(ops.items(), key=lambda kv: -kv[1])[:6]:
            print(f"    operator {o!r}: {n}")
        for u, n in sorted(usages.items(), key=lambda kv: -kv[1])[:6]:
            print(f"    usage {u!r}: {n}")
        print()

    with open(d / "rels.pkl", "rb") as f:
        rels = pickle.load(f)
    with open(d / "stops.pkl", "rb") as f:
        stops = pickle.load(f)

    def matches(t):
        if args.name and args.name not in (t.get("name") or ""):
            return False
        if args.ref and args.ref != t.get("ref"):
            return False
        return True

    hits = {i: v for i, v in rels.items() if matches(v[0])}
    masters = {i: v for i, v in hits.items() if v[0].get("type") == "route_master"}
    print(f"{len(hits)} relations match, {len(masters)} of them route_master\n")

    for mid, (mt, mm) in masters.items():
        kids = [r for ty, r, _ in mm if ty == "r"]
        print(f"route_master {mid}  {mt.get('name')}  ref={mt.get('ref')} "
              f"operator={mt.get('operator')} colour={mt.get('colour')}")
        print(f"  {len(kids)} member routes")
        for k in kids:
            if k not in rels:
                print(f"    r{k}  (not in this extract)")
                continue
            t, members = rels[k]
            nodes = [(r, role) for ty, r, role in members if ty == "n"]
            wroles = {}
            for ty, r, role in members:
                if ty == "w":
                    wroles[role or "(none)"] = wroles.get(role or "(none)", 0) + 1
            print(f"    r{k}  {t.get('name')}")
            print(f"        colour={t.get('colour')} ptv2={t.get('public_transport:version')}"
                  f"  {len(nodes)} node members, way roles {wroles}")
            roles = {}
            for _r, role in nodes:
                roles[role or "(none)"] = roles.get(role or "(none)", 0) + 1
            print(f"        node roles {roles}")
            named = [stops[r][0].get("name") for r, role in nodes
                     if role.startswith("stop") and r in stops]
            print(f"        first stops: {named[:6]}")
            print(f"        last stops:  {named[-3:]}")
            if args.members:
                for ty, r, role in members[:40]:
                    nm = stops[r][0].get("name") if ty == "n" and r in stops else ""
                    print(f"          {ty}{r} role={role!r} {nm}")
        print()


if __name__ == "__main__":
    main()
