"""T-456: OSM places of an extract as tier 0 classes them (canada.place_kind), with a finer class:
shop (shop=*), eam (errand amenity), food (restaurant, cafe, fast food, bar, pub...), oleis (other
leisure), job. Read only on the cached extract files. Usage: osm_pois.py <extract> [<extract>...]"""
import os, sys, time
os.environ["OMP_NUM_THREADS"] = "2"
sys.path.insert(0, r"C:\Users\anita\projects\maps\trainworld\pipeline")
import pandas as pd
import osm_features, routes
from canada import ERRAND_AMENITY, LEISURE_AMENITY, OTHER_JOB_AMENITY, LEISURE_TOURISM, LEISURE_LEISURE

FOOD = {"restaurant", "cafe", "fast_food", "bar", "pub", "food_court", "ice_cream", "biergarten"}
OUT = r"C:\Users\anita\projects\maps\trainworld\data\raw\t456"


def cls(shop, am, tour, lei):
    if shop and shop not in ("vacant", "no"):
        return "shop"
    if am in ERRAND_AMENITY:
        return "eam"
    if am in FOOD:
        return "food"
    if am in LEISURE_AMENITY or tour in LEISURE_TOURISM or lei in LEISURE_LEISURE:
        return "oleis"
    return "job"


def run(extract):
    out = os.path.join(OUT, f"pois_{extract.replace('/', '_')}.csv.gz")
    if os.path.exists(out):
        return
    t0 = time.time()
    keys = ("shop", "amenity", "office", "craft", "tourism", "leisure")
    pts = osm_features.read(extract, "points", columns=["other_tags"],
                            where=" OR ".join(f"other_tags LIKE '%\"{k}\"=>%'" for k in keys))
    rows = []
    for o, g in zip(pts.other_tags.fillna("").to_numpy(), pts.geometry.to_numpy()):
        t = {k: routes.tag(o, k) for k in keys}
        am = t["amenity"] if t["amenity"] in ERRAND_AMENITY | LEISURE_AMENITY | OTHER_JOB_AMENITY else ""
        tour = t["tourism"] if t["tourism"] in LEISURE_TOURISM else ""
        lei = t["leisure"] if t["leisure"] in LEISURE_LEISURE else ""
        if not (t["shop"] or am or tour or lei or t["office"] or t["craft"]):
            continue
        rows.append((cls(t["shop"], am, tour, lei), round(g.y, 6), round(g.x, 6), 0))
    npts = len(rows)
    cols = ["shop", "amenity", "office", "craft", "tourism", "leisure"]
    mp = osm_features.read(extract, "multipolygons", columns=cols,
                           where=" OR ".join(f"{k} IS NOT NULL" for k in cols))
    for r in mp.itertuples():
        shop, am, tour, lei = (v if isinstance(v, str) else "" for v in (r.shop, r.amenity, r.tourism, r.leisure))
        work = isinstance(r.office, str) or isinstance(r.craft, str)
        am = am if am in ERRAND_AMENITY | LEISURE_AMENITY | OTHER_JOB_AMENITY else ""
        tour = tour if tour in LEISURE_TOURISM else ""
        lei = lei if lei in LEISURE_LEISURE else ""
        if not (shop or am or tour or lei or work):
            continue
        c = r.geometry.representative_point()
        rows.append((cls(shop, am, tour, lei), round(c.y, 6), round(c.x, 6), 1))
    df = pd.DataFrame(rows, columns=["cls", "lat", "lon", "area"])
    df.to_csv(out, index=False)
    print(f"{extract}: {npts:,} points, {len(df) - npts:,} areas, {time.time() - t0:.0f} s; "
          + ", ".join(f"{k} {v:,}" for k, v in df.cls.value_counts().items()), flush=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for e in sys.argv[1:]:
        run(e)
