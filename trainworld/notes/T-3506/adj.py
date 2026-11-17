"""T-3506: unit adjacency for a few rows: each unit of the rows given, its people, its row, and every unit it touches
(within 500 m) with the shared border in km and that unit's row ('-' no row). Read-only.
usage: adj.py ISO row,row,... [ISO row,...]"""
import json, os, pickle, sys, ast
sys.stdout.reconfigure(encoding="utf-8")
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, r"C:/Users/anita/projects/maps/trainworld/pipeline")
import world_plan
import __main__
__main__.Unit = world_plan.Unit
import shapely
st = pickle.loads(open(r"C:/Users/anita/projects/maps/trainworld/data/work/worldplan/state.pkl", "rb").read())
plan = json.load(open(r"C:/Users/anita/projects/maps/trainworld/pipeline/world_plan.json", encoding="utf8"))


def U(r):
    u = r.get("units")
    return ast.literal_eval(u) if isinstance(u, str) else (u or {})


args = sys.argv[1:]
for iso, rl in zip(args[0::2], args[1::2]):
    want = rl.split(",")
    owner = {}
    for r in plan["metros"]:
        for u in U(r).get(iso, []):
            owner[u] = r["id"]
    by = {u.id: u for u in st["units"] if u.iso == iso}
    tree_ids = list(by)
    tree = shapely.STRtree([by[i].geom for i in tree_ids])
    print(f"\n######## {iso} {rl}")
    for rid in want:
        r = next(x for x in plan["metros"] if x["id"] == rid and U(x))
        ids = U(r).get(iso, [])
        print(f"\n== {rid} {r['name']} people {r.get('people'):,} status {r.get('status')}")
        for i in sorted(ids, key=lambda i: -by[i].pop):
            u = by[i]
            gb = u.geom.buffer(500)
            nbs = []
            for k in tree.query(gb):
                j = tree_ids[k]
                if j == i:
                    continue
                v = by[j]
                s = u.geom.boundary.intersection(v.geom.buffer(500)).length / 1000
                if s > 0.3:
                    nbs.append((s, v.name, owner.get(j, "-"), v.pop))
            nbs.sort(reverse=True)
            print(f"  {i} {u.name} {u.pop:,.0f}: " + "; ".join(f"{n} [{o}] {s:.0f}km" for s, n, o, p in nbs))
