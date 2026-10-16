"""T-456: Geofabrik US state extracts -> OSM places classified as osm_pois.py does (csv per state)."""
import os, sys, time, requests
os.environ["OMP_NUM_THREADS"] = "2"
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"C:\Users\anita\projects\maps\trainworld\pipeline")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd, pyogrio
import routes
from osm_pois import cls, ERRAND_AMENITY, LEISURE_AMENITY, OTHER_JOB_AMENITY, LEISURE_TOURISM, LEISURE_LEISURE

D = r"C:\Users\anita\projects\maps\trainworld\data\raw\t456\us"
UA = {"User-Agent": "trainworld-research/0.1 (open data analysis)"}
pyogrio.set_gdal_config_options({"OSM_MAX_TMPFILE_SIZE": "4000"})
os.makedirs(D, exist_ok=True)
KEEP = ERRAND_AMENITY | LEISURE_AMENITY | OTHER_JOB_AMENITY

for st in sys.argv[1:]:
    out = os.path.join(D, f"pois_{st}.csv.gz")
    if os.path.exists(out):
        continue
    pbf = os.path.join(D, f"{st}-latest.osm.pbf")
    if not os.path.exists(pbf):
        r = requests.get(f"https://download.geofabrik.de/north-america/us/{st}-latest.osm.pbf", headers=UA, timeout=1800, stream=True)
        r.raise_for_status()
        with open(pbf + ".part", "wb") as f:
            for ch in r.iter_content(1 << 20):
                f.write(ch)
        os.replace(pbf + ".part", pbf)
    t0 = time.time()
    keys = ("shop", "amenity", "office", "craft", "tourism", "leisure")
    pts = pyogrio.read_dataframe(pbf, layer="points", columns=["other_tags"],
                                 where=" OR ".join(f"other_tags LIKE '%\"{k}\"=>%'" for k in keys))
    rows = []
    for o, g in zip(pts.other_tags.fillna("").to_numpy(), pts.geometry.to_numpy()):
        t = {k: routes.tag(o, k) for k in keys}
        am = t["amenity"] if t["amenity"] in KEEP else ""
        tour = t["tourism"] if t["tourism"] in LEISURE_TOURISM else ""
        lei = t["leisure"] if t["leisure"] in LEISURE_LEISURE else ""
        if t["shop"] or am or tour or lei or t["office"] or t["craft"]:
            rows.append((cls(t["shop"], am, tour, lei), g.y, g.x, 0))
    cols = ["shop", "amenity", "office", "craft", "tourism", "leisure"]
    mp = pyogrio.read_dataframe(pbf, layer="multipolygons", columns=cols, where=" OR ".join(f"{k} IS NOT NULL" for k in cols))
    for r in mp.itertuples():
        shop, am, tour, lei = (v if isinstance(v, str) else "" for v in (r.shop, r.amenity, r.tourism, r.leisure))
        am = am if am in KEEP else ""
        tour = tour if tour in LEISURE_TOURISM else ""
        lei = lei if lei in LEISURE_LEISURE else ""
        if not (shop or am or tour or lei or isinstance(r.office, str) or isinstance(r.craft, str)) or r.geometry is None:
            continue
        c = r.geometry.representative_point()
        rows.append((cls(shop, am, tour, lei), c.y, c.x, 1))
    df = pd.DataFrame(rows, columns=["cls", "lat", "lon", "area"])
    df.to_csv(out, index=False)
    print(f"{st}: {len(df):,} places, {time.time() - t0:.0f} s; " + ", ".join(f"{k} {v:,}" for k, v in df.cls.value_counts().items()), flush=True)
