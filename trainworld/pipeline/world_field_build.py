"""The world field (T-375, design notes/T-362.md): Kontur population at H3 r5 worldwide, each cell
with its country, GDP per head at PPP and landmass, landmasses joined by fixed links from Wikidata.
A pipeline input, never shipped; `world_field.py` reads it.

    python pipeline/world_field_build.py          (downloads cached in data/raw/world_field/)

Writes data/work/world_field.npz (cells), world_field.countries.json, world_field.links.json.
Sources and method: notes/T-375.md."""
import gzip, json, os, shutil, sqlite3, sys, time, urllib.parse, urllib.request, zipfile
os.environ.setdefault("OMP_NUM_THREADS", "3")
from pathlib import Path
import numpy as np
import h3
import h3.api.basic_int as h3i
import pyogrio
import shapely
from shapely.strtree import STRtree

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/world_field"
KONTUR_GZ = ROOT / "data/raw/kontur/kontur_population_20231101_r6.gpkg.gz"
KONTUR_URL = "https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/kontur_population_20231101_r6.gpkg.gz"
WORK = ROOT / "data/work"
OUT = WORK / "world_field.npz"
UA = {"User-Agent": "trainworld-pipeline/0.1 (open-data build script)"}
NE = "https://naciscdn.org/naturalearth/10m/{}/{}.zip"
NE_FILES = [("physical", "ne_10m_land"), ("physical", "ne_10m_minor_islands"), ("cultural", "ne_10m_admin_0_countries")]
WB = "https://api.worldbank.org/v2/country/all/indicator/{}?format=json&per_page=20000&date=2010:2025"
# Maddison Project Database 2023 (CC BY 4.0) via Our World in Data, for what the World Bank lacks
MADDISON = "https://ourworldindata.org/grapher/gdp-per-capita-maddison-project-database.csv?v=1&csvType=full&useColumnShortNames=true"
QLEVER = "https://qlever.dev/api/wikidata"
R_KM = 6371.0088

# Bridges, tunnels, bridge-tunnels and causeways (and their subclasses) with a coordinate; length
# normalised to metres where given. Not proposed (Sakhalin, Malacca Strait), not in a state other
# than in use, not dissolved, closed or retired.
LINKS_SPARQL = """
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX p: <http://www.wikidata.org/prop/>
PREFIX psn: <http://www.wikidata.org/prop/statement/value-normalized/>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?item ?coord (MAX(?len) AS ?m) (SAMPLE(?lab) AS ?label) WHERE {
  { ?item wdt:P31/wdt:P279* wd:Q12280 } UNION { ?item wdt:P31/wdt:P279* wd:Q44377 }
  UNION { ?item wdt:P31/wdt:P279* wd:Q5456378 } UNION { ?item wdt:P31/wdt:P279* wd:Q892168 }
  ?item wdt:P625 ?coord .
  OPTIONAL { ?item p:P2043 ?st . ?st psn:P2043 ?v . ?v wikibase:quantityAmount ?len . }
  OPTIONAL { ?item rdfs:label ?lab . FILTER(LANG(?lab) = "en") }
  MINUS { ?item wdt:P31/wdt:P279* wd:Q64728694 }
  MINUS { ?item wdt:P5817 ?state . FILTER(?state != wd:Q55654238) }
  MINUS { ?item wdt:P576 ?gone }
  MINUS { ?item wdt:P3999 ?closed }
  MINUS { ?item wdt:P730 ?retired }
}
GROUP BY ?item ?coord
"""
# A link joins the land within LINK_TOL_KM of its half length around its coordinate, and counts
# only where its coordinate is within COAST_KM of the sea (inland bridges and mountain tunnels
# cross no strait). Discs of nearby links chain, so a crossing made of several named parts
# (Øresund's bridge and tunnel) joins as a whole.
LINK_TOL_KM = 3.0  # Natural Earth's 1:10M coast sits up to ~2 km off the real one
COAST_KM = 2.0
LINK_MAX_KM = 80.0  # longer "lengths" are lines and systems (a whole motorway), not crossings


def fetch(url, path, data=None, accept=None, timeout=600):
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    hdr = dict(UA)
    if accept:
        hdr["Accept"] = accept
    print("downloading", url[:100], flush=True)
    req = urllib.request.Request(url, data=data, headers=hdr)
    tmp = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    os.replace(tmp, path)
    return path


def fetch_all():
    for kind, name in NE_FILES:
        fetch(NE.format(kind, name), RAW / f"{name}.zip")
    for ind in ("NY.GDP.PCAP.PP.KD", "SP.POP.TOTL"):
        fetch(WB.format(ind), RAW / f"wb_{ind}.json")
    fetch(MADDISON, RAW / "maddison_gdp_per_capita.csv")
    q = urllib.parse.urlencode({"query": LINKS_SPARQL}).encode()
    fetch(QLEVER, RAW / "wikidata_links.json", data=q, accept="application/sparql-results+json")
    if not KONTUR_GZ.exists():
        fetch(KONTUR_URL, KONTUR_GZ, timeout=3600)


def read_ne(name, cols):
    df = pyogrio.read_dataframe(f"zip://{RAW / (name + '.zip')}!{name}.shp", columns=cols) if cols else \
        pyogrio.read_dataframe(f"zip://{RAW / (name + '.zip')}!{name}.shp")
    return df


# ---------------------------------------------------------------- Kontur r6 points

def kontur_r6():
    """r6 cells: h3 int, lon, lat, population. Unpacks the gpkg to data/work and deletes it after."""
    cache = WORK / "world_field.r6.npz"
    if cache.exists():
        z = np.load(cache)
        return z["h"], z["lon"], z["lat"], z["pop"]
    gpkg = WORK / "world_field.k6.gpkg"
    if not gpkg.exists():
        print("unpacking Kontur r6", flush=True)
        with gzip.open(KONTUR_GZ, "rb") as f, open(gpkg, "wb") as g:
            shutil.copyfileobj(f, g, 1 << 24)
    con = sqlite3.connect(gpkg)
    rows = con.execute("select h3, population from population where population > 0").fetchall()
    con.close()
    n = len(rows)
    h = np.empty(n, np.uint64); lon = np.empty(n); lat = np.empty(n); pop = np.empty(n)
    for i, (c, p) in enumerate(rows):
        h[i] = h3.str_to_int(c)
        lat[i], lon[i] = h3.cell_to_latlng(c)
        pop[i] = p
    np.savez(cache, h=h, lon=lon, lat=lat, pop=pop)
    gpkg.unlink()
    print(f"Kontur r6: {n:,} cells, {pop.sum() / 1e9:.3f} bn people", flush=True)
    return h, lon, lat, pop


def assign(points, geoms, max_deg=0.5):
    """Index of the polygon containing each point, else the nearest within max_deg, else -1."""
    tree = STRtree(geoms)
    out = np.full(len(points), -1, np.int64)
    pi, gi = tree.query(points, predicate="within")
    out[pi] = gi
    miss = np.nonzero(out < 0)[0]
    if len(miss):
        pi, gi = tree.query_nearest(points[miss], max_distance=max_deg, all_matches=False)
        out[miss[pi]] = gi
    return out


# ---------------------------------------------------------------- landmasses

def landmasses():
    land = read_ne("ne_10m_land", None)
    minor = read_ne("ne_10m_minor_islands", None)
    parts = []
    for g in list(land.geometry) + list(minor.geometry):
        parts.extend(shapely.get_parts(g))
    polys = np.array([p for p in parts if p.area > 0], dtype=object)
    # one landmass across the antimeridian (Chukotka, Fiji's islands)
    parent = list(range(len(polys)))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    b = shapely.bounds(polys)
    west = np.nonzero(b[:, 0] <= -179.999)[0]; east = np.nonzero(b[:, 2] >= 179.999)[0]
    line_w, line_e = shapely.LineString([(-180, -90), (-180, 90)]), shapely.LineString([(180, -90), (180, 90)])
    for i in west:
        si = polys[i].intersection(line_w)
        if si.is_empty:
            continue
        for j in east:
            sj = polys[j].intersection(line_e)
            if not sj.is_empty:
                y0i, y1i = si.bounds[1], si.bounds[3]; y0j, y1j = sj.bounds[1], sj.bounds[3]
                if min(y1i, y1j) - max(y0i, y0j) > -1e-6:
                    parent[find(i)] = find(j)
    return polys, np.array([find(i) for i in range(len(polys))])


def subdivide(geoms, owner, maxv=400):
    """Cut polygons into pieces of at most ~maxv vertices (spatial queries on Eurasia are slow).
    Returns the pieces and the index of the polygon each came from."""
    out, own = [], []
    stack = [(g, o) for g, o in zip(geoms, owner)]
    while stack:
        g, o = stack.pop()
        if shapely.get_num_coordinates(g) <= maxv:
            out.append(g); own.append(o)
            continue
        x0, y0, x1, y1 = g.bounds
        if x1 - x0 >= y1 - y0:
            xm = (x0 + x1) / 2; halves = [(x0, y0, xm, y1), (xm, y0, x1, y1)]
        else:
            ym = (y0 + y1) / 2; halves = [(x0, y0, x1, ym), (x0, ym, x1, y1)]
        for hb in halves:
            c = shapely.clip_by_rect(g, *hb)
            for p in shapely.get_parts(c):
                if p.geom_type == "Polygon" and p.area > 0:
                    stack.append((p, o))
    return np.array(out, dtype=object), np.array(own)


def links(pieces, piece_island):
    """Fixed links from Wikidata: lists of islands each chain of nearby coastal links joins."""
    rows = json.load(open(RAW / "wikidata_links.json", encoding="utf-8"))["results"]["bindings"]
    recs = []
    for b in rows:
        c = b["coord"]["value"]
        if not c.startswith("POINT("):
            continue  # coordinates on other globes
        x, y = map(float, c[6:-1].split())
        m = float(b["m"]["value"]) if "m" in b else 0.0
        if m / 1000 > LINK_MAX_KM or not (-180 <= x <= 180 and -90 <= y <= 90):
            continue
        recs.append((b["item"]["value"].rsplit("/", 1)[1], b.get("label", {}).get("value", ""), x, y, m / 1000))
    print(f"Wikidata bridges, tunnels and causeways with a coordinate: {len(recs):,}", flush=True)
    X = np.array([r[2] for r in recs]); Y = np.array([r[3] for r in recs]); L = np.array([r[4] for r in recs])
    R = L / 2 + LINK_TOL_KM
    kx = 111.32 * np.maximum(np.cos(np.radians(Y)), 0.05); ky = 110.57
    tree = STRtree(pieces)
    # coastal: some point of a COAST_KM circle round it is not land
    ang = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    cx = (X[:, None] + COAST_KM * np.cos(ang)[None, :] / kx[:, None]).ravel()
    cy = (Y[:, None] + COAST_KM * np.sin(ang)[None, :] / ky).ravel()
    ring = shapely.points(np.column_stack([cx, cy]))
    on_land = np.zeros(len(ring), bool)
    pi, _ = tree.query(ring, predicate="within")
    on_land[pi] = True
    coastal = ~on_land.reshape(len(recs), 16).all(axis=1)
    idx = np.nonzero(coastal)[0]
    print(f"coastal or offshore: {len(idx):,}", flush=True)
    pts = shapely.points(np.column_stack([X[idx], Y[idx]]))
    qi, qg = tree.query(pts, predicate="dwithin", distance=R[idx] / kx[idx] + 1e-9)
    touch = {}
    for a, gi in zip(qi, qg):
        k = idx[a]
        if piece_island[gi] in touch.get(k, ()):
            continue
        g = shapely.transform(pieces[gi], lambda v, x=X[k], y=Y[k], s=kx[k]: (v - [x, y]) * [s, ky])
        if shapely.distance(shapely.Point(0, 0), g) <= R[k]:
            touch.setdefault(k, set()).add(int(piece_island[gi]))
    keep = sorted(touch)
    n = len(keep)
    # chain overlapping discs
    from scipy.spatial import cKDTree
    lo, la = np.radians(X[keep]), np.radians(Y[keep])
    xyz = np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
    Rs = R[keep]
    par = list(range(n))
    def f(a):
        while par[a] != a:
            par[a] = par[par[a]]; a = par[a]
        return a
    kd = cKDTree(xyz)
    for a in range(n):
        for b2 in kd.query_ball_point(xyz[a], 2 * Rs[a] / R_KM):
            if b2 != a and np.linalg.norm(xyz[a] - xyz[b2]) * R_KM <= Rs[a] + Rs[b2]:
                par[f(a)] = f(b2)
    keep = [(k, R[k], touch[k]) for k in keep]
    chains = {}
    for i in range(n):
        chains.setdefault(f(i), []).append(i)
    joins = []
    for members in chains.values():
        isl = sorted({t for i in members for t in keep[i][2]})
        if len(isl) < 2:
            continue
        joins.append({"islands": isl, "links": [{"qid": recs[keep[i][0]][0], "name": recs[keep[i][0]][1],
                      "lon": recs[keep[i][0]][2], "lat": recs[keep[i][0]][3], "km": round(recs[keep[i][0]][4], 2)}
                      for i in members]})
    return joins


# ---------------------------------------------------------------- countries and GDP

def wb_series(ind):
    data = json.load(open(RAW / f"wb_{ind}.json", encoding="utf-8"))[1]
    best = {}
    for r in data:
        iso3, v, yr = r["countryiso3code"], r["value"], int(r["date"])
        if iso3 and v is not None and (iso3 not in best or yr > best[iso3][1]):
            best[iso3] = (float(v), yr)
    return best


def countries_table(adm):
    gdp = wb_series("NY.GDP.PCAP.PP.KD")
    pop = wb_series("SP.POP.TOTL")
    madd = {}
    import csv
    for r in csv.DictReader(open(RAW / "maddison_gdp_per_capita.csv", encoding="utf-8")):
        if r["code"] and r["gdp_per_capita"]:
            madd.setdefault(r["code"], {})[int(r["year"])] = float(r["gdp_per_capita"])
    def madd_latest(iso3):
        s = madd.get(iso3)
        return (s[max(s)], max(s)) if s and max(s) >= 2010 else None
    # Maddison's 2011 dollars onto the World Bank's constant 2021 international dollars
    ratios = []
    wb_all = json.load(open(RAW / "wb_NY.GDP.PCAP.PP.KD.json", encoding="utf-8"))[1]
    for r in wb_all:
        yr, v = int(r["date"]), r["value"]
        if v is not None and yr in madd.get(r["countryiso3code"], {}) and yr == 2022:
            ratios.append(v / madd[r["countryiso3code"]][yr])
    k_madd = float(np.median(ratios))
    # one row per ISO code (Natural Earth gives some islands their own feature: Clipperton, Baikonur),
    # named after its largest feature
    rows, row_of, feat_row = [], {}, np.zeros(len(adm), np.int64)
    for i in np.argsort(-shapely.area(np.array(list(adm.geometry), dtype=object))):
        r = adm.iloc[i]
        iso2, iso3 = r["ISO_A2_EH"], r["ISO_A3_EH"]
        if iso3 == "-99":
            iso3 = r["ADM0_A3"]
        if iso2 == "-99":
            iso2 = ""
        if iso3 in row_of:
            feat_row[i] = row_of[iso3]
            continue
        feat_row[i] = row_of[iso3] = len(rows)
        g, src = None, ""
        if iso3 in gdp:
            g, src = gdp[iso3][0], f"World Bank NY.GDP.PCAP.PP.KD {gdp[iso3][1]}"
        else:
            m = madd_latest(iso3)
            if m:
                g, src = m[0] * k_madd, f"Maddison 2023 {m[1]} x {k_madd:.3f}"
        rows.append({"iso2": iso2, "iso3": iso3, "name": r["NAME"], "subregion": r["SUBREGION"],
                     "gdp": g, "gdp_src": src, "wb_pop": pop.get(iso3, (None,))[0]})
    # what neither has: the median of its subregion
    for row in rows:
        if row["gdp"] is None:
            vals = [o["gdp"] for o in rows if o["subregion"] == row["subregion"] and o["gdp_src"]]
            row["gdp"] = float(np.median(vals)) if vals else float(np.median([o["gdp"] for o in rows if o["gdp_src"]]))
            row["gdp_src"] = f"median of {row['subregion']}"
    print(f"Maddison to World Bank factor {k_madd:.3f} over {len(ratios)} countries", flush=True)
    return rows, feat_row


# ---------------------------------------------------------------- main

def main():
    t0 = time.time()
    fetch_all()
    h6, lon6, lat6, pop6 = kontur_r6()
    pts = shapely.points(np.column_stack([lon6, lat6]))

    adm = read_ne("ne_10m_admin_0_countries", ["ISO_A2_EH", "ISO_A3_EH", "ADM0_A3", "NAME", "SUBREGION"])
    ctab, feat_row = countries_table(adm)
    cpieces, cown = subdivide(list(adm.geometry), feat_row)
    cpiece6 = assign(pts, cpieces)
    cty6 = np.where(cpiece6 >= 0, cown[np.maximum(cpiece6, 0)], -1)
    print(f"countries assigned ({(cty6 < 0).sum()} r6 cells with none) {time.time() - t0:.0f} s", flush=True)

    polys, island_of_poly = landmasses()
    pieces, pown = subdivide(polys, island_of_poly)
    piece6 = assign(pts, pieces)
    isl6 = np.where(piece6 >= 0, pown[np.maximum(piece6, 0)], -1)
    print(f"landmasses assigned ({(isl6 < 0).sum()} r6 cells with none, {len(pieces):,} pieces) {time.time() - t0:.0f} s", flush=True)
    pop_isl = np.bincount(isl6[isl6 >= 0], weights=pop6[isl6 >= 0], minlength=len(polys))

    joins = links(pieces, pown)
    par = np.arange(len(polys))
    def f(a):
        while par[a] != a:
            par[a] = par[par[a]]; a = par[a]
        return a
    for j in joins:
        for t in j["islands"][1:]:
            par[f(t)] = f(j["islands"][0])
    group = np.array([f(i) for i in range(len(polys))])
    print(f"fixed-link joins: {len(joins)} {time.time() - t0:.0f} s", flush=True)

    # r6 -> r5: population sums, weighted centroid, dominant country and island by population
    p5 = np.array([h3i.cell_to_parent(int(c), 5) for c in h6], np.uint64)
    u5, inv = np.unique(p5, return_inverse=True)
    pop5 = np.bincount(inv, weights=pop6)
    lon5 = np.bincount(inv, weights=pop6 * lon6) / pop5
    lat5 = np.bincount(inv, weights=pop6 * lat6) / pop5
    # antimeridian cells: the weighted mean of lon near +-180 is wrong; use the cell centre
    span = np.zeros(len(u5)); np.maximum.at(span, inv, np.abs(lon6))
    wrap = (span > 179) & (np.abs(lon5) < 179)
    for i in np.nonzero(wrap)[0]:
        lat5[i], lon5[i] = h3i.cell_to_latlng(int(u5[i]))
    def dominant(lab):
        key = inv.astype(np.int64) * (lab.max() + 2) + (lab + 1)
        uk, ki = np.unique(key, return_inverse=True)
        w = np.bincount(ki, weights=pop6)
        cell = uk // (lab.max() + 2); val = uk % (lab.max() + 2) - 1
        o = np.lexsort((-w, cell))
        first = np.ones(len(o), bool); first[1:] = cell[o][1:] != cell[o][:-1]
        out = np.full(len(u5), -1, np.int64); out[cell[o][first]] = val[o][first]
        return out
    cty5 = dominant(cty6); isl5 = dominant(isl6)
    land5 = np.where(isl5 >= 0, group[np.maximum(isl5, 0)], -1)
    # compact landmass ids, largest population first
    lp = {}
    for g_, p in zip(land5, pop5):
        lp[g_] = lp.get(g_, 0) + p
    rank = {g_: k for k, (g_, _) in enumerate(sorted(lp.items(), key=lambda t: -t[1]))}
    land5c = np.array([rank[g_] for g_ in land5], np.int32)
    isl_rank = {}
    for g_ in isl5:
        isl_rank.setdefault(g_, len(isl_rank))
    print(f"r5 cells {len(u5):,}, people {pop5.sum() / 1e9:.3f} bn, landmasses {len(rank)}", flush=True)

    # landmass names: the biggest country on each
    lname = {}
    for g_ in rank:
        m = land5 == g_
        cs = np.bincount(np.maximum(cty5[m], 0), weights=pop5[m])
        lname[rank[g_]] = {"pop": float(lp[g_]), "country": ctab[int(cs.argmax())]["iso2"] if lp[g_] > 0 else ""}
    for j in joins:
        j["landmass"] = int(rank.get(int(group[j["islands"][0]]), -1))
        j["islands_pop"] = [round(float(pop_isl[t])) for t in j["islands"]]

    WORK.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, h3=u5, lon=lon5.astype(np.float32), lat=lat5.astype(np.float32),
                        pop=pop5.astype(np.float32), country=cty5.astype(np.int16), landmass=land5c,
                        island=np.array([isl_rank[g_] for g_ in isl5], np.int32))
    json.dump({"countries": ctab}, open(WORK / "world_field.countries.json", "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    json.dump({"landmasses": lname, "joins": joins}, open(WORK / "world_field.links.json", "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
