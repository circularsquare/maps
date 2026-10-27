"""City propers (administrative city limits): metros over 3 million (2025), national capitals of
countries over 1 million, GaWC 2024 Alpha and Beta cities (gawc_2024.csv), and a few added by
hand (city_seeds.csv lists the last three).

    python -I cities.py lookup     # Nominatim: candidate OSM relations per city -> logs/city_candidates.json
    python -I cities.py fetch      # OSM geometry for the relations in cities.csv -> raw/osm/
    python -I cities.py check      # areas and populations of what cities.csv picks
    python -I cities.py units      # logs/city_units.pkl, read by build.py --cities

cities.csv is the record of which boundary each city uses (and why, when it is not obvious).
"""
import csv
import json
import os
import pickle
import sys
import time

import requests
import shapely
from pyproj import Geod, Transformer

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(HERE, "logs")
RAW = os.path.join(HERE, "raw", "osm")
CSV = os.path.join(HERE, "cities.csv")
UA = {"User-Agent": "liveddensity-map-build/1.0 (population density research script)"}
OVERPASS = "https://overpass-api.de/api/interpreter"
MIN_POP = 3_000_000
GEOD = Geod(ellps="WGS84")


def big_fuas():
    pp = json.load(open(os.path.join(LOGS, "sub_pops.json"), encoding="utf-8"))
    c = pickle.load(open(os.path.join(LOGS, "units.pkl"), "rb"))
    rows = [(u, pp.get(u["id"], {}).get("2025", 0)) for u in c["fua"]]
    return sorted([(u, p) for u, p in rows if p > MIN_POP], key=lambda t: -t[1])


def iso2():
    sys.path.insert(0, HERE)
    import build
    feats = json.load(open(build.NE, encoding="utf-8"))["features"]
    out = {}
    for f in feats:
        p = f["properties"]
        k = build.country_key(p)
        a2 = p["ISO_A2_EH"] if p["ISO_A2_EH"] != "-99" else p["ISO_A2"]
        if k and a2 != "-99" and k not in out:
            out[k] = a2.lower()
    return out


def city_name(fua_name):
    return fua_name.split(" [")[0].split(" (")[0]


# one spelling per city across the three lists
ALIAS = {"Washington D.C.": "Washington", "Bogota": "Bogotá", "Yaounde": "Yaoundé",
         "Sana'a": "Sanaa", "Quezon City": "Manila", "New Taipei": "Taipei"}
SKIP_FUA = {"fua:9721"}   # Shenzhen (Bao'an): a second FUA of Shenzhen, which has its city unit


def master():
    """Cities from the three sources, merged: [{name, parent, sources, fua}]."""
    out = {}
    for u, p in big_fuas():
        if u["id"] in SKIP_FUA:
            continue
        name = ALIAS.get(city_name(u["name"]), city_name(u["name"]))
        e = out.setdefault((u["parent"], name), {"name": name, "parent": u["parent"],
                                                 "sources": [], "fua": u["id"]})
        e["sources"].append("metro>3M")
    with open(os.path.join(HERE, "city_seeds.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            name = ALIAS.get(r["name"], r["name"])
            e = out.setdefault((r["parent"], name), {"name": name, "parent": r["parent"],
                                                     "sources": [], "fua": ""})
            if r["source"] not in e["sources"]:
                e["sources"].append(r["source"])
    return list(out.values())


def lookup():
    sys.stdout.reconfigure(encoding="utf-8")
    cc = iso2()
    path = os.path.join(LOGS, "city_candidates.json")
    out = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else []
    done = {(o["parent"], o["query"]) for o in out}
    for m in master():
        if (m["parent"], m["name"]) in done:
            continue
        u = {"id": m["fua"], "name": m["name"], "parent": m["parent"]}
        p = None
        q = m["name"]
        for attempt in range(8):
            try:
                r = requests.get("https://nominatim.openstreetmap.org/search",
                                 params={"q": q, "countrycodes": cc.get(u["parent"], ""),
                                         "format": "jsonv2", "limit": 10, "extratags": 1},
                                 headers=UA, timeout=60)
                r.raise_for_status()
                break
            except requests.RequestException as e:
                print("  retry:", e, flush=True)
                time.sleep(20 * (attempt + 1))
        res = r.json()
        cands = [{"osm_id": x["osm_id"], "cat": x["category"], "type": x["type"],
                  "addresstype": x.get("addresstype"), "name": x["display_name"][:90],
                  "pop": (x.get("extratags") or {}).get("population"),
                  "wikidata": (x.get("extratags") or {}).get("wikidata"),
                  "level": (x.get("extratags") or {}).get("admin_level")}
                 for x in res if x.get("osm_type") == "relation"]
        top_wd = next(((x.get("extratags") or {}).get("wikidata") for x in res), None)
        out.append({"fua": u["id"], "fua_name": u["name"], "parent": u["parent"], "pop2025": p,
                    "query": q, "top_wikidata": top_wd, "cands": cands})
        print(u["id"], q, len(cands), flush=True)
        json.dump(out, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        time.sleep(1.1)


MIRRORS = [OVERPASS, "https://overpass.kumi.systems/api/interpreter",
           "https://overpass.private.coffee/api/interpreter"]


def overpass(query):
    for i in range(9):
        url = MIRRORS[(i + 1) % len(MIRRORS)]
        try:
            r = requests.post(url, data={"data": query}, headers=UA, timeout=600)
            if r.status_code == 200:
                return r.json()
            print("  overpass", url.split("/")[2], r.status_code, "retrying", flush=True)
        except requests.RequestException as e:
            print("  overpass", url.split("/")[2], e.__class__.__name__, flush=True)
        time.sleep(10 * (i + 1))
    raise RuntimeError("overpass failed")


def relation_geom(rid):
    """Polygon of one OSM relation, from polygons.openstreetmap.fr (cached in raw/osm/), or
    assembled from its member ways via Overpass if that service has nothing."""
    path = os.path.join(RAW, f"poly_{rid}.json")
    if not os.path.exists(path):
        os.makedirs(RAW, exist_ok=True)
        for attempt in range(3):
            try:
                r = requests.get("https://polygons.openstreetmap.fr/get_geojson.py",
                                 params={"id": rid, "params": 0}, headers=UA, timeout=300)
                if r.status_code == 200 and r.text.startswith("{"):
                    open(path, "w", encoding="utf-8").write(r.text)
                    break
                # not generated yet: ask for it, then try again
                requests.get("https://polygons.openstreetmap.fr/index.py", params={"id": rid},
                             headers=UA, timeout=300)
            except requests.RequestException:
                pass
            time.sleep(3)
    if os.path.exists(path):
        g = shapely.geometry.shape(json.load(open(path, encoding="utf-8")))
        if g.geom_type == "GeometryCollection":
            g = shapely.union_all([p for p in g.geoms if p.geom_type in ("Polygon", "MultiPolygon")])
        tags = {}
        tp = os.path.join(RAW, f"tags_{rid}.json")
        if os.path.exists(tp):
            tags = json.load(open(tp, encoding="utf-8"))
        return shapely.make_valid(g), tags
    return relation_geom_overpass(rid)


def relation_geom_overpass(rid):
    path = os.path.join(RAW, f"{rid}.json")
    if not os.path.exists(path):
        d = overpass(f"[out:json][timeout:500];rel({rid});out geom;")
        os.makedirs(RAW, exist_ok=True)
        json.dump(d, open(path, "w", encoding="utf-8"))
        time.sleep(2)
    d = json.load(open(path, encoding="utf-8"))
    rel = [e for e in d["elements"] if e["type"] == "relation"][0]
    lines = {"outer": [], "inner": []}
    for m in rel["members"]:
        if m["type"] == "way" and "geometry" in m:
            role = "inner" if m.get("role") == "inner" else "outer"
            lines[role].append(shapely.LineString([(pt["lon"], pt["lat"]) for pt in m["geometry"]]))
    outer = shapely.union_all(list(shapely.polygonize(lines["outer"]).geoms)) if lines["outer"] else None
    if outer is None or outer.is_empty:
        return None, rel.get("tags", {})
    if lines["inner"]:
        inner = shapely.union_all(list(shapely.polygonize(lines["inner"]).geoms))
        if not inner.is_empty:
            outer = outer.difference(inner)
    return shapely.make_valid(outer), rel.get("tags", {})


def fetch_tags(ids):
    """OSM tags for relations, one Overpass call, cached as raw/osm/tags_<id>.json."""
    need = [i for i in ids if not os.path.exists(os.path.join(RAW, f"tags_{i}.json"))]
    for k in range(0, len(need), 200):
        chunk = need[k:k + 200]
        d = overpass(f"[out:json][timeout:300];rel(id:{','.join(map(str, chunk))});out tags;")
        os.makedirs(RAW, exist_ok=True)
        for e in d["elements"]:
            json.dump(e.get("tags", {}), open(os.path.join(RAW, f"tags_{e['id']}.json"), "w",
                                              encoding="utf-8"), ensure_ascii=False)


def wikidata(qids):
    """Label, population (latest), area and coordinates of Wikidata items (cached)."""
    path = os.path.join(LOGS, "city_wikidata.json")
    wd = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    need = sorted({q for q in qids if q and q not in wd})
    for i in range(0, len(need), 50):
        r = requests.get("https://www.wikidata.org/w/api.php",
                         params={"action": "wbgetentities", "ids": "|".join(need[i:i + 50]),
                                 "format": "json", "props": "claims|labels", "languages": "en"},
                         headers=UA, timeout=120)
        for q, e in r.json()["entities"].items():
            cl = e.get("claims", {})
            pops = []
            for s in cl.get("P1082", []):
                try:
                    v = float(s["mainsnak"]["datavalue"]["value"]["amount"])
                    t = (s.get("qualifiers", {}).get("P585", [{}])[0].get("datavalue", {})
                         .get("value", {}).get("time", ""))
                    pops.append((t, v))
                except (KeyError, IndexError, TypeError):
                    pass
            pops.sort()
            try:
                area = float(cl["P2046"][0]["mainsnak"]["datavalue"]["value"]["amount"])
                unit = cl["P2046"][0]["mainsnak"]["datavalue"]["value"].get("unit", "")
                if unit.endswith("Q35852"):      # hectare
                    area /= 100
            except (KeyError, IndexError, TypeError):
                area = None
            wd[q] = {"label": e.get("labels", {}).get("en", {}).get("value"),
                     "pop": pops[-1][1] if pops else None,
                     "pop_year": pops[-1][0][1:5] if pops else None, "area": area}
        time.sleep(1)
    json.dump(wd, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    return wd


def auto():
    """First pick per city: the boundary relation whose wikidata is the city's own item."""
    sys.stdout.reconfigure(encoding="utf-8")
    cand = {(o["parent"], o["query"]): o for o in
            json.load(open(os.path.join(LOGS, "city_candidates.json"), encoding="utf-8"))}
    rows = []
    for m in master():
        o = cand.get((m["parent"], m["name"]), {})
        top = o.get("top_wikidata")
        bnd = [c for c in o.get("cands", []) if c["cat"] == "boundary"]
        same = [c for c in bnd if c["wikidata"] == top]
        pick = max(same, key=lambda c: int(c["level"] or 0)) if same else None
        if pick is None:
            pick = next((c for c in bnd if c["addresstype"] in
                         ("city", "municipality", "town", "county")), None)
        rows.append({**m, "wikidata": top, "rel": pick["osm_id"] if pick else "",
                     "others": " ".join(f"{c['osm_id']}:{c['addresstype']}:L{c['level']}"
                                        for c in bnd if not pick or c["osm_id"] != pick["osm_id"])[:200]})
    wd = wikidata([r["wikidata"] for r in rows])
    fetch_tags([r["rel"] for r in rows if r["rel"]])
    with open(os.path.join(LOGS, "city_auto.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["parent", "name", "sources", "fua", "wikidata", "wd_label", "wd_pop", "wd_year",
                    "wd_km2", "rel", "rel_name", "level", "osm_pop", "osm_km2", "flag", "others"])
        for r in rows:
            km2, tags = None, {}
            if r["rel"]:
                g, tags = relation_geom(r["rel"])
                km2 = round(geodesic_km2(g)) if g is not None else None
            w_ = wd.get(r["wikidata"] or "", {})
            flag = ""
            if not r["rel"]:
                flag = "NO REL"
            elif w_.get("area") and km2 and not (0.6 < km2 / w_["area"] < 1.6):
                flag = "AREA"
            w.writerow([r["parent"], r["name"], "+".join(r["sources"]), r["fua"], r["wikidata"],
                        w_.get("label"), w_.get("pop"), w_.get("pop_year"), w_.get("area"), r["rel"],
                        tags.get("name:en", tags.get("name")), tags.get("admin_level"),
                        tags.get("population"), km2, flag, r["others"]])
            print(r["parent"], r["name"], r["rel"], km2, flag, flush=True)


def children(rid, level, name_re=None):
    """Child admin relations of a relation (e.g. a Chinese city's districts)."""
    path = os.path.join(RAW, f"children_{rid}_{level}.json")
    if not os.path.exists(path):
        flt = f'["name"~"{name_re}"]' if name_re else ""
        q = (f'[out:json][timeout:300];rel({rid});map_to_area->.a;'
             f'rel(area.a)["boundary"="administrative"]["admin_level"="{level}"]{flt};out tags;')
        d = overpass(q)
        os.makedirs(RAW, exist_ok=True)
        json.dump(d, open(path, "w", encoding="utf-8"))
        time.sleep(2)
    d = json.load(open(path, encoding="utf-8"))
    return [(e["id"], e.get("tags", {})) for e in d["elements"] if e["type"] == "relation"]


def geodesic_km2(g):
    return abs(GEOD.geometry_area_perimeter(g)[0]) / 1e6


def read_csv():
    with open(CSV, encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["name"] and not r["parent"].startswith("#")]


def spec_geom(r):
    """Union of the relations in a cities.csv row ('rels' is ;-separated; 'children:<rid>:<level>'
    takes all child relations of that level, minus any listed in 'drop')."""
    geoms, tags = [], []
    drop = {x.strip() for x in r.get("drop", "").split(";") if x.strip()}
    keep = {x.strip() for x in r.get("keep", "").split(";") if x.strip()}
    for item in r["rels"].split(";"):
        item = item.strip()
        if not item:
            continue
        if item.startswith("geob:"):
            # geoBoundaries file in raw/geob/, feature by shapeName: geob:<file stem>:<name>
            _, stem, nm = item.split(":", 2)
            fc = json.load(open(os.path.join(HERE, "raw", "geob", stem + ".geojson"),
                                encoding="utf-8"))
            for f in fc["features"]:
                if f["properties"].get("shapeName") == nm:
                    geoms.append(shapely.make_valid(shapely.geometry.shape(f["geometry"])))
                    tags.append({"name": nm, "source": "geoBoundaries"})
            continue
        if item.startswith("children:"):
            parts = item.split(":")
            rid, level = parts[1], parts[2]
            suffix = parts[3] if len(parts) > 3 else ""
            for cid, t in children(rid, level):
                names = {str(cid), t.get("name", ""), t.get("name:en", "")}
                if suffix and not t.get("name", "").endswith(suffix):
                    continue
                names.discard("")
                if names & drop or (keep and not any(k in n for n in names for k in keep)):
                    continue
                g, _ = relation_geom(cid)
                if g is not None:
                    geoms.append(g)
                    tags.append(t)
        else:
            g, t = relation_geom(item)
            if g is not None:
                geoms.append(g)
                tags.append(t)
    return (shapely.union_all(geoms) if geoms else None), tags


def check():
    """Area of each city's outline next to Wikidata's figure (logs/city_areas.csv)."""
    sys.stdout.reconfigure(encoding="utf-8")
    with open(os.path.join(LOGS, "city_areas.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["parent", "name", "parts", "km2", "ref_km2", "ref_pop", "note"])
        for r in read_csv():
            if not r["rels"]:
                w.writerow([r["parent"], r["name"], 0, "", r["ref_km2"], r["ref_pop"], "SKIPPED " + r["note"]])
                continue
            g, tags = spec_geom(r)
            km2 = round(geodesic_km2(g)) if g is not None else "NO GEOMETRY"
            w.writerow([r["parent"], r["name"], len(tags), km2, r["ref_km2"], r["ref_pop"], r["note"]])
            print(r["parent"], r["name"], km2, flush=True)


def units():
    """Mollweide parts per city, for build.py."""
    tr = Transformer.from_crs("EPSG:4326", "ESRI:54009", always_xy=True)

    def proj(xy):
        import numpy as np
        x, y = tr.transform(xy[:, 0], xy[:, 1])
        return np.column_stack([x, y])
    out = []
    for r in read_csv():
        if not r["rels"]:
            continue
        g, tags = spec_geom(r)
        gm = shapely.make_valid(shapely.transform(shapely.segmentize(g, 0.01), proj))
        parts = [p for p in shapely.get_parts(gm) if p.geom_type in ("Polygon", "MultiPolygon")]
        out.append({"id": r["id"], "name": r["name"], "kind": "city", "parent": r["parent"],
                    "fua": r["fua"], "parts": [shapely.to_wkb(p) for p in parts]})
    pickle.dump(out, open(os.path.join(LOGS, "city_units.pkl"), "wb"))
    print(len(out), "cities")


# ---------------------------------------------------------------- 2025 raster, by tile

EPOCH = 2025
TILE_URL = ("https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_POP_GLOBE_R2023A/"
            "GHS_POP_E{e}_GLOBE_R2023A_54009_100/V1-0/tiles/"
            "GHS_POP_E{e}_GLOBE_R2023A_54009_100_V1_0_R{r}_C{c}.zip")
TILES = os.path.join(HERE, "raw", "ghsl_tiles")
X0, Y0, TILE = -18041000, 9000000, 1_000_000
PAD = 20_000      # metres around each city searched for the nearest country


def city_units():
    return pickle.load(open(os.path.join(LOGS, "city_units.pkl"), "rb"))


def tiles_for(bounds):
    x0, y0, x1, y1 = bounds
    c0, c1 = int((x0 - PAD - X0) // TILE) + 1, int((x1 + PAD - X0) // TILE) + 1
    r0, r1 = int((Y0 - y1 - PAD) // TILE) + 1, int((Y0 - y0 + PAD) // TILE) + 1
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def tiles():
    """Download (and unzip) the 2025 tiles that cover the cities."""
    import subprocess
    need = set()
    for u in city_units():
        g = shapely.union_all(shapely.from_wkb(u["parts"]))
        need.update(tiles_for(g.bounds))
    os.makedirs(TILES, exist_ok=True)
    print(len(need), "tiles", flush=True)
    for r, c in sorted(need):
        name = f"GHS_POP_E{EPOCH}_GLOBE_R2023A_54009_100_V1_0_R{r}_C{c}"
        tif = os.path.join(TILES, name + ".tif")
        if os.path.exists(tif):
            continue
        z = os.path.join(TILES, name + ".zip")
        resp = requests.get(TILE_URL.format(e=EPOCH, r=r, c=c), headers=UA, timeout=600)
        if resp.status_code == 404:
            print(f"R{r}_C{c}: no tile (all water)", flush=True)
            continue
        resp.raise_for_status()
        open(z, "wb").write(resp.content)
        subprocess.run([r"C:\Windows\System32\tar.exe", "-xf", z, "-C", TILES, name + ".tif"],
                       check=True)
        os.remove(z)
        print(f"R{r}_C{c}: {len(resp.content) / 1e6:.1f} MB", flush=True)


def vrt():
    """One VRT over the downloaded tiles, on the global grid."""
    import rasterio
    path = os.path.join(TILES, "mosaic.vrt")
    srcs = []
    for f in sorted(os.listdir(TILES)):
        if f.endswith(".tif"):
            with rasterio.open(os.path.join(TILES, f)) as ds:
                col = int(round((ds.transform.c - X0) / 100))
                row = int(round((Y0 - ds.transform.f) / 100))
                srcs.append(
                    f'<ComplexSource><SourceFilename relativeToVRT="1">{f}</SourceFilename>'
                    f'<SourceBand>1</SourceBand><SrcRect xOff="0" yOff="0" xSize="{ds.width}" '
                    f'ySize="{ds.height}"/><DstRect xOff="{col}" yOff="{row}" xSize="{ds.width}" '
                    f'ySize="{ds.height}"/><NODATA>-200</NODATA></ComplexSource>')
                crs = ds.crs.to_wkt()
    xml = (f'<VRTDataset rasterXSize="360820" rasterYSize="180000"><SRS>{crs}</SRS>'
           f'<GeoTransform>{X0}, 100, 0, {Y0}, 0, -100</GeoTransform>'
           f'<VRTRasterBand dataType="Float64" band="1"><NoDataValue>-200</NoDataValue>'
           + "".join(srcs) + "</VRTRasterBand></VRTDataset>")
    open(path, "w", encoding="utf-8").write(xml)
    return path


def hist():
    """2025 histograms per city, merged into data/sub/<ISO3>.json and data/sub/index.json."""
    sys.stdout.reconfigure(encoding="utf-8")
    import numpy as np
    import rasterio
    import rasterio.windows
    from scipy import ndimage
    sys.path.insert(0, HERE)
    import build
    path = vrt()
    keys, names, parts, part_cid, _ = build.load_countries("ESRI:54009")
    build.init_worker(path, parts, part_cid, len(keys))
    ds = build.G["ds"]
    kix = {k: i + 1 for i, k in enumerate(keys)}
    lo, step, n = build.LO, build.STEP, build.NBINS

    def binof(d):
        return np.clip(np.floor((np.log10(d) - lo) / step).astype(np.int64), 0, n - 1)

    res = {}
    for u in city_units():
        geoms = shapely.from_wkb(u["parts"])
        x0, y0, x1, y1 = shapely.union_all(geoms).bounds
        c0 = int((x0 - PAD - X0) // 1000) * 10
        r0 = int((Y0 - y1 - PAD) // 1000) * 10
        c1 = int(np.ceil((x1 + PAD - X0) / 1000)) * 10
        r1 = int(np.ceil((Y0 - y0 + PAD) / 1000)) * 10
        h, w = r1 - r0, c1 - c0
        pop = ds.read(1, window=rasterio.windows.Window(c0, r0, w, h), boundless=True,
                      fill_value=0).astype(np.float64)
        pop[~(pop > 0)] = 0
        tf = rasterio.windows.transform(rasterio.windows.Window(c0, r0, w, h), ds.transform)
        city = rasterio.features.rasterize([(g, 1) for g in geoms], out_shape=(h, w), transform=tf,
                                           dtype="uint8")
        cty = build.rasterize(r0, c0, h, w).astype(np.int64)
        miss = (cty == 0) & (pop > 0) & (city > 0)
        if miss.any() and (cty > 0).any():
            ind = ndimage.distance_transform_edt(cty == 0, return_distances=False, return_indices=True)
            cty[miss] = cty[ind[0][miss], ind[1][miss]]
        sel = (city > 0) & (pop > 0) & (cty == kix[u["parent"]])
        rr, cc = np.nonzero(sel)
        p = pop[rr, cc]
        blocks = pop.reshape(h // 10, 10, w // 10, 10).sum(axis=(1, 3))
        h100 = np.bincount(binof(p * 100.0), weights=p, minlength=n)
        h1k = np.bincount(binof(blocks[rr // 10, cc // 10]), weights=p, minlength=n)
        other = float(pop[(city > 0) & (pop > 0) & (cty != kix[u["parent"]])].sum())
        res[u["id"]] = (u, h100, h1k)
        print(f"{u['id']:16} {u['parent']} {u['name']:28} {p.sum():12,.0f}"
              + (f"  ({other:,.0f} over the border)" if other >= 1 else ""), flush=True)

    # merge into data/sub
    y = str(EPOCH)
    sub = os.path.join(HERE, "data", "sub")
    ip = os.path.join(sub, "index.json")
    index = json.load(open(ip, encoding="utf-8"))
    index["units"] = [x for x in index["units"] if x["kind"] != "city"]
    by = {}
    for uid, (u, h100, h1k) in res.items():
        index["units"].append({"id": uid, "name": u["name"], "kind": "city", "parent": u["parent"],
                               "pop": {y: int(round(h100.sum()))}})
        by.setdefault(u["parent"], []).append((uid, h100, h1k))
    build.write_atomic(ip, index)
    for k, rows in by.items():
        fp = os.path.join(sub, f"{k}.json")
        d = json.load(open(fp, encoding="utf-8"))
        for g in ("1km", "100m"):
            d["hist"][g][y] = {i: v for i, v in d["hist"][g][y].items() if not i.startswith("city:")}
        for uid, h100, h1k in rows:
            d["hist"]["100m"][y][uid] = build.pack(h100)
            d["hist"]["1km"][y][uid] = build.pack(h1k)
        build.write_atomic(fp, d)
    print(f"wrote {len(res)} cities into data/sub")


if __name__ == "__main__":
    {"lookup": lookup, "auto": auto, "check": check, "units": units, "tiles": tiles,
     "hist": hist}[sys.argv[1]]()
