"""T-2341, Jakarta's plan patched by hand (after T-1857's patch.py). The over-40M split of GHSL's Jakarta gave jak
DKI Jakarta, Bogor, Depok, Bekasi, Karawang, Purwakarta, Sukabumi and Cianjur, and tan Tangerang with Serang and
Cilegon. Now:
- jak Jakarta = Jabodetabek: DKI's five kota, Kab and Kota Bogor, Kota Depok, Kab and Kota Bekasi, Kab and Kota
  Tangerang, Kota Tangerang Selatan (id, origin and name kept);
- a zone Sukabumi: Kab Sukabumi, Kota Sukabumi, Cianjur (Cipanas is in Cianjur), as Anita asked: one zone;
- a zone Karawang: Karawang, Purwakarta (not Jabodetabek; they would take jak over 40M);
- a zone Serang: Kab and Kota Serang, Kota Cilegon (tan's rest; tan's id is retired, its successor jak, since its
  old centre is in Tangerang).
The changed rows go to status planned; people from GHS-POP over the geoBoundaries units; extents redrawn from them.

    python jak_patch.py          dry: prints the rows, writes nothing
    python jak_patch.py --go     under the plan lock: writes world_plan.json and extents.geojson
"""
import json
import sys
sys.path.insert(0, r"C:\Users\anita\projects\maps\trainworld\pipeline")
sys.stdout.reconfigure(encoding="utf-8")
import shapely  # noqa: E402
from shapely.geometry import shape, mapping  # noqa: E402
from shapely.ops import unary_union  # noqa: E402
import shared_lock  # noqa: E402
import world_plan as wp  # noqa: E402

TW = r"C:\Users\anita\projects\maps\trainworld"
PLAN = TW + r"\pipeline\world_plan.json"
EXT = TW + r"\data\work\worldplan\extents.geojson"
GB = TW + r"\data\raw\worldplan\geoboundaries\IDN_ADM2.geojson"
GO = "--go" in sys.argv

gb = json.load(open(GB, encoding="utf-8"))["features"]
by_name, geom = {}, {}
for f in gb:
    by_name.setdefault(f["properties"]["shapeName"], []).append(f["properties"]["shapeID"])
    geom[f["properties"]["shapeID"]] = shapely.make_valid(shape(f["geometry"]))
name_of = {u: n for n, ids in by_name.items() for u in ids}

JAK = ["Kota Jakarta Barat", "Kota Jakarta Pusat", "Kota Jakarta Selatan", "Kota Jakarta Timur", "Kota Jakarta Utara",
       "Bogor", "Kota Bogor", "Kota Depok", "Bekasi", "Kota Bekasi", "Tangerang", "Kota Tangerang", "Kota Tangerang Selatan"]
NEW = {  # name: (units, origin at the town's centre: T-2299, origins off their towns)
    "Sukabumi": (["Sukabumi", "Kota Sukabumi", "Cianjur"], [106.9270, -6.9210]),
    "Karawang": (["Karawang", "Purwakarta"], [107.3050, -6.3050]),
    "Serang": (["Serang", "Kota Serang", "Kota Cilegon"], [106.1500, -6.1200]),
}
pop = wp.Pop()


def ids(names):
    out = []
    for n in names:
        cands = [u for u in by_name[n] if u in pool]
        assert len(cands) == 1, (n, by_name[n])
        out.append(cands[0])
    return out


def count(units):
    return round(float(pop.inside(wp.polygonal(wp.to_moll(unary_union([geom[u] for u in units]))))[2].sum()))


def bbox(units):
    return [round(v, 3) for v in unary_union([geom[u] for u in units]).bounds]


import contextlib  # noqa: E402
with shared_lock.held("plan", "T-2341 Jakarta by hand") if GO else contextlib.nullcontext():
    doc = shared_lock.read_json(PLAN)
    ext = shared_lock.read_json(EXT)
    rows = {r["id"]: r for r in doc["metros"]}
    jak, tan = rows["jak"], rows["tan"]
    pool = set(jak["units"]["IDN"]) | set(tan["units"]["IDN"])
    assert len(pool) == 21
    used = set(rows) | {c["id"] for c in json.load(open(TW + r"\app\src\game\registry.json", encoding="utf-8"))["cities"]}
    jak_units = ids(JAK)
    new_rows = []
    taken = set(jak_units)
    order = max(r.get("order", 0) for r in doc["metros"])
    for k, (name, (names, origin)) in enumerate(NEW.items()):
        us = ids(names)
        taken |= set(us)
        rid = wp.make_id(name, used)
        order += 1
        new_rows.append({"order": order, "id": rid, "name": name, "kind": "zone", "status": "planned", "region": "world",
                         "countries": ["IDN"], "iso2": "ID", "people": count(us), "origin": origin, "tz": "Asia/Jakarta",
                         "utc_offset": 7, "extracts": ["asia/indonesia/java"], "archetype": "D", "module": None,
                         "bbox": bbox(us), "units": {"IDN": us},
                         "source": {"zone": "T-723", "unit_names": names, "hand": "T-2341"}})
    assert taken == pool, pool - taken
    old = (jak["people"], len(jak["units"]["IDN"]))
    jak["units"]["IDN"] = jak_units
    jak["people"] = count(jak_units)
    jak["bbox"] = bbox(jak_units)
    jak["status"] = "planned"
    jak.setdefault("source", {})["hand"] = {"task": "T-2341", "why": "Jabodetabek: Tangerang in, Sukabumi, Cianjur, Karawang and Purwakarta out (Anita 2026-11-04)"}
    print("jak", old, "->", jak["people"], len(jak_units))
    for r in new_rows:
        print(r["id"], r["name"], r["people"], r["origin"], r["source"]["unit_names"])
    doc["metros"] = [r for r in doc["metros"] if r["id"] != "tan"] + new_rows
    feats = {f["properties"]["id"]: f for f in ext["features"]}
    proto = feats.pop("tan")
    for r in [jak] + new_rows:
        g = unary_union([geom[u] for u in r["units"]["IDN"]])
        f = feats.get(r["id"]) or json.loads(json.dumps(proto))
        f["properties"]["id"] = r["id"]
        if "order" in f["properties"]:
            f["properties"]["order"] = r["order"]
        if "name" in f["properties"]:
            f["properties"]["name"] = r["name"]
        if "people" in f["properties"]:
            f["properties"]["people"] = r["people"]
        f["geometry"] = mapping(g)
        feats[r["id"]] = f
        print("  extent", r["id"], round(g.area, 3), "props", sorted(f["properties"]))
    ext["features"] = list(feats.values())
    if GO:
        shared_lock.write_atomic(EXT, json.dumps(ext))
        shared_lock.write_atomic(PLAN, json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
        print("written")
    else:
        print("dry: nothing written")
