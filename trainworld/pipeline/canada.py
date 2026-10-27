"""T-277: a city pack from Canadian sources (Toronto first), the same format as the US packs.

    python pipeline/build_city.py tor        (build_city.main hands a "ca" city to main() here)

What each US input becomes (docs/adding-a-city.md, "Outside the US"; docs/demand-worldwide.md):

- Census blocks -> 2021 dissemination areas (DAs, 400-700 people each): population from Statistics
  Canada table 98-10-0015-01, polygons from Esri Canada's copy of the StatCan boundary file (StatCan's
  own download server sits behind a bot check). Each DA is spread over its land by the same 50 m
  raster as a US block (spread.py).
- LODES WAC (jobs where they are) -> the census's employed people by census tract (CT) of work,
  98-10-0498-01, "usual place of work". The 2021 census was taken in May 2021, when a third of
  Toronto's workers worked at home and so count at home, not at their workplace. Their jobs are put
  back at workplaces by income: for each employment income band, the CMA's ratio of people working
  at home to people with a usual place of work is applied to each work tract's workers in that band
  (people working at home earned more, so office tracts get more back). "No fixed workplace"
  workers (98-10-0480-01) are added in proportion to jobs. Within a tract, jobs go to its DAs by
  OpenStreetMap shops, amenities, offices and commercial or industrial land (80%) and population
  (20%). Areas outside every tract take their census subdivision's workplace count from the
  commuting flow table (98-10-0459-01) less its tracts'.
- LODES RAC x ACS work from home (the home end) -> commuters by CT of residence (98-10-0480-01,
  usual place plus no fixed workplace) and people working at home (98-10-0498-01), to DAs by
  population.
- ACS work from home (2019-2023) -> the 2021 census's at-home workers scaled to a current share,
  the Labour Force Survey's May 2024 figure for the Toronto CMA (24.7% mostly at home; config
  "wfh_now"): the same fraction of 2021's at-home workers is taken as commuting again at both ends.
- LODES retail, service, food and leisure sectors (everyday destinations) -> OSM shops and
  services, and OSM restaurants, bars, venues and hotels, counted per DA (no sector counts by small
  area are published).
- Institutional group quarters: none taken off (the census counts collective dwellings only for
  larger areas; docs/demand-worldwide.md).
- State regions and border factor -> one province, one region, factor 1.
- TIGER water -> StatCan's lakes and rivers (Esri Canada's copy) plus everything outside the land:
  Ontario's census subdivisions (cartographic, clipped at the shore) and the US states
  (cb_2020_us_state_500k, also clipped).
- NCES, IPEDS, FAA: the Ontario school information file, Council of Ontario Universities headcounts
  with Ontario's college campus headcounts, and StatCan's airport passenger table; see the
  functions below.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")

import csv
import functools
import gzip
import io
import json
import re
import time
import zipfile

import h3
import numpy as np
import pandas as pd
import shapely
from shapely import STRtree, make_valid, union_all
from shapely.geometry import Point, box, mapping, shape

import arcgis
import build_city as b
import packio
import overpass
import regions
import spread
import water

RAW_CA = os.path.join(b.RAW, "ca")
SUB = "ca/"
STATCAN_CSV = "https://www150.statcan.gc.ca/n1/tbl/csv/{t}-eng.zip"
ESRI = "https://services.arcgis.com/wjcPoefzjpzCgffS/arcgis/rest/services/"
DA_LAYER = ESRI + "DisseminationAreas/FeatureServer/0"            # lda_000b21a_e copy, Esri Canada
CSD_LAYER = ESRI + "CensusSubdivision/FeatureServer/0"            # cartographic, clipped at shores
AGE_LAYER = ESRI + "Canadian_Age_and_Gender_2021/FeatureServer/3"  # DA ages, 2021 census
LAKES_LAYER = ESRI + "Water_Bodies_in_Canada_2/FeatureServer/0"    # StatCan lakes and rivers
CT_LAYER = "https://services1.arcgis.com/ZozYr1WfCwexXgcn/arcgis/rest/services/lct_000b21a_e/FeatureServer/0"
US_STATES_CB = "https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_state_500k.zip"
# blocked by a bot check on 2026-10-13 (the official downloads, for Anita): boundary files
# https://www12.statcan.gc.ca/census-recensement/2021/geo/sip-pis/boundary-limites/index2021-eng.cfm?year=21

WFH_NOW = 0.247       # Toronto CMA, LFS May 2024: employed people mostly working from home
JOB_SIGNAL_SHARE = 0.8  # within a tract, jobs by OSM job places; the rest by population
ERRAND_SHARE, LEISURE_SHARE = 0.15, 0.08  # scale of the OSM counts, in jobs (relative draw only)
LANDUSE_M2_PER_PLACE = 5000.0  # industrial or commercial land counted as one job place per 5,000 m2
OSM_EXTRACT = "north-america/canada/ontario"  # Geofabrik, shared with routes.py (T-275)

ERRAND_AMENITY = {"bank", "pharmacy", "post_office", "fuel", "car_wash", "marketplace", "veterinary",
                  "library", "car_rental", "bureau_de_change", "money_transfer", "post_depot"}
LEISURE_AMENITY = {"restaurant", "cafe", "fast_food", "bar", "pub", "food_court", "ice_cream", "biergarten",
                   "cinema", "theatre", "nightclub", "arts_centre", "casino", "community_centre", "events_venue"}
OTHER_JOB_AMENITY = {"school", "college", "university", "hospital", "clinic", "doctors", "dentist", "townhall",
                     "courthouse", "police", "fire_station", "social_facility", "kindergarten", "childcare",
                     "place_of_worship", "nursing_home", "bus_station", "ferry_terminal"}
LEISURE_TOURISM = {"hotel", "motel", "museum", "attraction", "gallery", "zoo", "theme_park", "aquarium", "hostel"}
LEISURE_LEISURE = {"fitness_centre", "sports_centre", "bowling_alley", "stadium", "water_park", "amusement_arcade",
                   "ice_rink", "dance", "escape_game"}


def cds(cfg):
    return sorted(cfg["counties"])


def statcan(t):
    path = b.fetch(STATCAN_CSV.format(t=t), f"ca/{t}-eng.zip")
    return zipfile.ZipFile(path)


def table(t, usecols=None, chunks=None):
    """A StatCan full-table CSV (the data member of its zip) as a DataFrame."""
    z = statcan(t)
    name = next(m for m in z.namelist() if "MetaData" not in m)
    with z.open(name) as f:
        return pd.read_csv(f, encoding="utf-8-sig", usecols=usecols, dtype=str, low_memory=False)


def num(s):
    return pd.to_numeric(s, errors="coerce").fillna(0.0).astype(np.float64)


# ---------------------------------------------------------------- geography

@functools.lru_cache(maxsize=1)
def csd_features():
    """Ontario's census subdivisions (a 170 MB GeoJSON page: read once a run)."""
    return arcgis.features(f"{SUB}csd_on", CSD_LAYER, where="PRUID='35'", fields="CSDUID,CSDNAME,LANDAREA,OBJECTID")


def build_boundary(city, cfg):
    """The pack's census subdivisions in its census divisions, dissolved (cartographic: the shore
    of Lake Ontario, not the provincial line in the lake)."""
    want = set(cds(cfg))
    geoms = [make_valid(shape(f["geometry"])) for f in csd_features() if f["properties"]["CSDUID"][:4] in want]
    merged = union_all(geoms)

    def rnd(c):
        if isinstance(c, (list, tuple)) and c and isinstance(c[0], (int, float)):
            return [round(c[0], 6), round(c[1], 6)]
        return [rnd(x) for x in c]

    # the file the app draws: simplified to about 30 m (the full shoreline is 3 MB; the US
    # county files are generalised already, 0.1 MB); the pipeline keeps the full geometry
    geom = mapping(merged.simplify(0.0003, preserve_topology=True))
    geom = {"type": geom["type"], "coordinates": rnd(geom["coordinates"])}
    fc = {"type": "FeatureCollection", "features": [{
        "type": "Feature",
        "properties": {"city": city, "counties": sorted(want),
                       "source": "Statistics Canada 2021 census subdivisions (cartographic, Esri Canada's copy) in the census divisions listed, dissolved"},
        "geometry": geom}]}
    out = os.path.join(b.PACKS, f"{city}.boundary.geojson")
    os.makedirs(b.PACKS, exist_ok=True)
    with open(out + ".part", "w") as f:
        json.dump(fc, f, separators=(",", ":"))
    os.replace(out + ".part", out)
    print(f"boundary: {len(geoms)} census subdivisions, {geom['type']}, {os.path.getsize(out) / 1e6:.2f} MB -> {out}")
    return merged


def water_source(bbox_ll):
    """Water polygons over bbox_ll: StatCan lakes and rivers, and everything that is neither
    Ontario land (its cartographic census subdivisions) nor US land (cb 2020 states)."""
    w, s, e, n = bbox_ll
    ext = box(w, s, e, n)
    lakes = arcgis.features(f"{SUB}lakes_{w:.2f}_{s:.2f}", LAKES_LAYER, fields="HYDROUID,NAME,RANK,FID", bbox=bbox_ll)
    geoms = []
    for f in lakes:
        if f.get("geometry"):
            g = make_valid(shape(f["geometry"]))
            if g.intersects(ext):
                geoms.append(g.intersection(ext))
    # bounds first: make_valid on Ontario's far-north subdivisions (Hudson Bay shore) takes minutes
    land = [shape(f["geometry"]) for f in csd_features() if f.get("geometry")]
    land = [make_valid(g).intersection(ext) for g in land if box(*g.bounds).intersects(ext)]
    us, _ = water.read_shp(b.fetch(US_STATES_CB), bbox_ll)
    land += [g.intersection(ext) for g in us if g.intersects(ext)]
    sea = ext.difference(union_all(land))
    if not sea.is_empty:
        geoms.append(sea)
    info = {"source": "StatCan lakes and rivers (Esri Canada's Water Bodies in Canada) + everything outside Ontario's "
                      "cartographic census subdivisions and the Census cb_2020 US states",
            "lake_polygons": len(lakes), "open_water_km2_deg": round(sea.area, 6)}
    return geoms, info


def build_water(cfg, boundary):
    t = time.time()
    lon0, lat0 = cfg["origin"]
    wc = cfg["water"]
    mask, info = water.build_mask(lon0, lat0, boundary.bounds, wc["cell_m"], b.fetch, None, source=water_source)
    raw_runs = len(mask.xs) // 2
    if wc["min_width_px"] > 1:
        mask = mask.without_narrow(wc["min_width_px"])
    info["min_width_m"] = wc["min_width_px"] * wc["cell_m"]
    print(f"water: {mask.W} x {mask.H} pixels of {mask.cell:.0f} m, {info['lake_polygons']:,} lake and river polygons; "
          f"{raw_runs:,} runs, {len(mask.xs) // 2:,} after dropping water under {info['min_width_m']:.0f} m wide; "
          f"water {mask.water_pixels() / (mask.W * mask.H):.1%} of the extent; {time.time() - t:.0f} s")
    return mask, info


def das(cfg):
    """The pack's dissemination areas: DAUID, polygon, representative point, population, its census
    tract (DGUID, by the point) and census subdivision, children and adults."""
    where = " OR ".join(f"DAUID LIKE '{c}%'" for c in cds(cfg))
    feats = arcgis.features(f"{SUB}da_{'_'.join(cds(cfg))}", DA_LAYER, where=where, fields="DAUID,LANDAREA,OBJECTID")
    polys, rows = {}, []
    for f in feats:
        g = make_valid(shape(f["geometry"]))
        u = f["properties"]["DAUID"]
        polys[u] = g
        p = g.representative_point()
        rows.append((u, p.y, p.x, float(f["properties"]["LANDAREA"] or 0.0)))
    df = pd.DataFrame(rows, columns=["geoid", "lat", "lon", "land_km2"])
    pop = table("98100015", usecols=["DGUID", "Population and dwelling counts (5): Population, 2021 [1]"])
    pop = pop[pop.DGUID.str.startswith("2021S0512")]
    pop = dict(zip(pop.DGUID.str[9:], num(pop.iloc[:, 1])))
    df["pop"] = df.geoid.map(pop).fillna(0.0)
    missing = int(df.geoid.map(pop).isna().sum())
    # census tracts (official CT polygons, a copy of lct_000b21a_e), by each DA's point
    w, s, e, n = union_all(list(polys.values())).bounds
    ct = arcgis.features(f"{SUB}ct_on_{'_'.join(cds(cfg))}", CT_LAYER, where="PRUID='35'", fields="CTUID,DGUID,FID", bbox=(w, s, e, n))
    ctg = [make_valid(shape(f["geometry"])) for f in ct]
    ctid = [f["properties"]["DGUID"] for f in ct]
    pts = shapely.points(df.lon.to_numpy(), df.lat.to_numpy())
    pi, gi = STRtree(ctg).query(pts, predicate="within")
    df["ct"] = ""
    df.loc[pi, "ct"] = np.array(ctid, dtype=object)[gi]
    csd = [f for f in csd_features() if f["properties"]["CSDUID"][:4] in set(cds(cfg))]
    pi, gi = STRtree([make_valid(shape(f["geometry"])) for f in csd]).query(pts, predicate="within")
    df["csd"] = ""
    df.loc[pi, "csd"] = np.array([f["properties"]["CSDUID"] for f in csd], dtype=object)[gi]
    # a point the cartographic CSDs miss by a hair (a shoreline DA): its nearest CSD
    lost = np.nonzero(df.csd.to_numpy() == "")[0]
    if len(lost):
        tree = STRtree([make_valid(shape(f["geometry"])) for f in csd])
        near = tree.nearest(pts[lost])
        df.loc[lost, "csd"] = np.array([f["properties"]["CSDUID"] for f in csd], dtype=object)[near]
    # ages: children (5-17, plus 2/5 of under-5s for ages 3-4, as NCES's pre-K) and adults (18+)
    agef = ["DGUID", "Total_Total_Age", "Total_0_to_4_years", "Total_5_to_9_years", "Total_10_to_14_years", "Total_15_to_19_years", "OBJECTID"]
    awhere = " OR ".join(f"DGUID LIKE '2021S0512{c}%'" for c in cds(cfg))
    ages = arcgis.features(f"{SUB}age_{'_'.join(cds(cfg))}", AGE_LAYER, where=awhere, fields=",".join(agef), geometry=False)
    a = pd.DataFrame([f["properties"] for f in ages])
    a["geoid"] = a.DGUID.str[9:]
    a = a.set_index("geoid")
    tot = a.Total_Total_Age.astype(float)
    kids = a.Total_5_to_9_years + a.Total_10_to_14_years + 0.6 * a.Total_15_to_19_years + 0.4 * a.Total_0_to_4_years
    adults = tot - a.Total_0_to_4_years - a.Total_5_to_9_years - a.Total_10_to_14_years - 0.6 * a.Total_15_to_19_years
    ks = (kids / tot.where(tot > 0)).reindex(df.geoid)
    ads = (adults / tot.where(tot > 0)).reindex(df.geoid)
    k_all, a_all = float(kids.sum() / tot.sum()), float(adults.sum() / tot.sum())
    df["child_share"] = ks.fillna(k_all).clip(0, 1).to_numpy()
    df["adult_share"] = ads.fillna(a_all).clip(0, 1).to_numpy()
    print(f"dissemination areas: {len(df):,}, pop {df['pop'].sum():,.0f} ({missing} without a population row), "
          f"{int((df.ct != '').sum()):,} in {df.loc[df.ct != '', 'ct'].nunique():,} census tracts; "
          f"children {k_all:.3f}, adults {a_all:.3f} of residents ({int(ks.isna().sum())} DAs given the pack's shares)")
    return df, polys


# ---------------------------------------------------------------- OSM job places

def place_kind(shop, amenity, tourism, leisure):
    if (shop and shop not in ("vacant", "no")) or amenity in ERRAND_AMENITY:
        return "errand"
    if amenity in LEISURE_AMENITY or tourism in LEISURE_TOURISM or leisure in LEISURE_LEISURE:
        return "leisure"
    return "job"


def _land_rows(geom, w, spread_deg, ref="", name=""):
    """A land area's weight on its representative point, or with spread_deg split over a grid of
    points that far apart inside it (T-459). Every row carries the area's OSM `ref` and `name`."""
    c = geom.representative_point()
    if not spread_deg:
        return [("land", c.y, c.x, w, ref, name)]
    x0, y0, x1, y1 = geom.bounds
    gx, gy = np.meshgrid(np.arange(x0 + spread_deg / 2, x1, spread_deg), np.arange(y0 + spread_deg / 2, y1, spread_deg))
    gx, gy = gx.ravel(), gy.ravel()
    ok = shapely.contains_xy(geom, gx, gy) if len(gx) else np.zeros(0, bool)
    if not ok.any():
        return [("land", c.y, c.x, w, ref, name)]
    return [("land", y, x, w / int(ok.sum()), ref, name) for x, y in zip(gx[ok], gy[ok])]


def _osm_ref(r):
    """An area's OSM id as a short text: r<relation> or w<way>."""
    rel = getattr(r, "osm_id", None)
    return f"r{rel}" if isinstance(rel, str) and rel else f"w{getattr(r, 'osm_way_id', '')}"


def osm_places(city, boundary, extract=OSM_EXTRACT, cache=None, land_cap_m2=None, land_spread_deg=None):
    """Per OSM place: kind (errand, leisure, job; or land, commercial or industrial land in units
    of LANDUSE_M2_PER_PLACE), point and weight. From the extract's cached features (osm_features;
    T-277 first tried Overpass in 0.3 degree tiles, and the public servers took minutes a tile and
    answered 504 on others), cached as a small CSV. Any country's extract works (tier0.py passes
    its own `cache` path). land_cap_m2 caps one land area's weight and land_spread_deg spreads it
    over its area (T-459: Lagos's 23 km2 refinery outweighed every OSM shop and office in the city);
    both off by default, which is every pack built before them."""
    cache = cache or os.path.join(RAW_CA, f"osm_jobs_{city}.csv.gz")
    if os.path.exists(cache):
        out = pd.read_csv(cache)
    else:
        import osm_features
        import routes
        t0 = time.time()
        w, s, e, n = boundary.bounds
        keys = ("shop", "amenity", "office", "craft", "tourism", "leisure")
        pts = osm_features.read(extract, "points", columns=["other_tags"], bbox=(w, s, e, n),
                                where=" OR ".join(f"other_tags LIKE '%\"{k}\"=>%'" for k in keys))
        rows = []
        for o, g in zip(pts.other_tags.fillna("").to_numpy(), pts.geometry.to_numpy()):
            t = {k: routes.tag(o, k) for k in keys}
            am = t["amenity"] if t["amenity"] in ERRAND_AMENITY | LEISURE_AMENITY | OTHER_JOB_AMENITY else ""  # not benches
            tour = t["tourism"] if t["tourism"] in LEISURE_TOURISM else ""
            lei = t["leisure"] if t["leisure"] in LEISURE_LEISURE else ""
            if not (t["shop"] or am or tour or lei or t["office"] or t["craft"]):
                continue
            rows.append((place_kind(t["shop"], am, tour, lei), g.y, g.x, 1.0, "", ""))
        n_pts = len(rows)
        cols = ["shop", "amenity", "office", "craft", "tourism", "leisure", "landuse"]
        mp = osm_features.read(extract, "multipolygons", columns=cols + ["osm_id", "osm_way_id", "name"], bbox=(w, s, e, n),
                               where=" OR ".join(f"{k} IS NOT NULL" for k in cols))
        for r in mp.itertuples():
            c = r.geometry.representative_point()
            if r.landuse in ("industrial", "commercial", "retail") and not (r.shop or r.amenity or r.office):
                g = r.geometry
                try:
                    m2 = overpass.area_m2(g)
                except shapely.errors.GEOSException:
                    # GDAL's OSM driver can return a ring of under four points (Texas, T-1065): its polygons only
                    g = shapely.union_all([p for p in shapely.get_parts(shapely.make_valid(g))
                                           if p.geom_type in ("Polygon", "MultiPolygon")])
                    if g.is_empty:
                        continue
                    m2 = overpass.area_m2(g)
                if land_cap_m2:
                    m2 = min(m2, land_cap_m2)
                rows += _land_rows(g, m2 / LANDUSE_M2_PER_PLACE, land_spread_deg,
                                   _osm_ref(r), r.name if isinstance(r.name, str) else "")
                continue
            shop, am, tour, lei = (v if isinstance(v, str) else "" for v in (r.shop, r.amenity, r.tourism, r.leisure))
            work = isinstance(r.office, str) or isinstance(r.craft, str)
            am = am if am in ERRAND_AMENITY | LEISURE_AMENITY | OTHER_JOB_AMENITY else ""  # not parking lots
            tour = tour if tour in LEISURE_TOURISM else ""
            lei = lei if lei in LEISURE_LEISURE else ""  # parks are their own layer, not job places
            if not (shop or am or tour or lei or work):
                continue
            rows.append((place_kind(shop, am, tour, lei), c.y, c.x, 1.0, "", ""))
        out = pd.DataFrame(rows, columns=["kind", "lat", "lon", "w", "ref", "name"])
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        out.to_csv(cache, index=False)
        print(f"  OSM job places: {n_pts:,} points, {len(mp):,} areas, {time.time() - t0:.0f} s", flush=True)
    print("OSM job places: " + ", ".join(f"{k} {v:,.0f}" for k, v in out.groupby("kind").w.sum().items()))
    return out


def per_da(df, polys, places):
    """Sum each kind of OSM place's weight per DA (by point in polygon)."""
    ids = list(polys)
    tree = STRtree([polys[u] for u in ids])
    pts = shapely.points(places.lon.to_numpy(), places.lat.to_numpy())
    pi, gi = tree.query(pts, predicate="within")
    pi, first = np.unique(pi, return_index=True)  # a point on a shared edge counts once
    gi = gi[first]
    row = {u: i for i, u in enumerate(df.geoid)}
    da_row = np.array([row.get(ids[g], -1) for g in gi])
    out = {}
    for kind in ("errand", "leisure", "job", "land"):
        sel = (places.kind.to_numpy()[pi] == kind) & (da_row >= 0)
        out[kind] = np.bincount(da_row[sel], weights=places.w.to_numpy()[pi][sel], minlength=len(df))
    return out


# ---------------------------------------------------------------- commuting

INCOME = ["Employment income groups (18):Without employment income[2]"] + [None] * 12


def pow_by_ct():
    """98-10-0498-01 by CT of work: usual place of work by income band and worked at home, and the
    CMA rows. Returns (usual by band DataFrame, at home Series, CMA usual by band, CMA at home by band)."""
    t = table("98100498")
    t = t[t["Place of work status (3)"].isin(["Usual place of work", "Worked at home"])]
    cols = [c for c in t.columns if c.startswith("Employment income groups (18):")]
    bands = [cols[1]] + cols[4:16]  # without income, then the twelve income bands
    v = t[bands].apply(num)
    v.index = t.index
    t = t[["GEO", "DGUID", "Place of work status (3)"]].join(v)
    usual = t[t["Place of work status (3)"] == "Usual place of work"].set_index("DGUID")[bands]
    home = t[t["Place of work status (3)"] == "Worked at home"].set_index("DGUID")[bands]
    return usual, home, bands


def commuters_by_ct():
    """98-10-0480-01 by CT of residence, all modes and times: commuters (usual place + no fixed
    workplace) and the no-fixed part."""
    t = table("98100480")
    t = t[(t["Time leaving for work (7)"] == "Total - Time leaving for work") & (t["Commuting duration (7)"] == "Total - Commuting duration")]
    tot = "Place of work status (3A):Total - Place of work status[1]"
    nf = "Place of work status (3A):No fixed workplace address[2]"
    allm = t[t["Main mode of commuting (21)"] == "Total - Main mode of commuting"].set_index("DGUID")
    transit = t[t["Main mode of commuting (21)"] == "Public transit"].set_index("DGUID")
    return pd.DataFrame({"commuters": num(allm[tot]), "no_fixed": num(allm[nf]), "transit": num(transit[tot]).reindex(allm.index).fillna(0.0)})


def csd_flows(cfg):
    """98-10-0459-01, CSD of residence to CSD of work (usual place of work), for every pair with
    either end in the pack's census divisions; cached as a small CSV."""
    cache = os.path.join(RAW_CA, f"csd_flows_{'_'.join(cds(cfg))}.csv")
    if os.path.exists(cache):
        return pd.read_csv(cache, dtype={"h": str, "w": str})
    z = statcan("98100459")
    meta = next(m for m in z.namelist() if "MetaData" in m)
    member = {}
    with z.open(meta) as f:
        for row in csv.reader(io.TextIOWrapper(f, encoding="utf-8-sig")):
            if len(row) >= 4 and row[0] == "1" and row[2].startswith("["):
                member[row[3]] = row[2].strip("[]")
    want = set(cds(cfg))
    data = next(m for m in z.namelist() if "MetaData" not in m)
    out = []
    t0 = time.time()
    with z.open(data) as f:
        for k, ch in enumerate(pd.read_csv(f, encoding="utf-8-sig", usecols=[2, 4, 5], dtype=str, chunksize=2_000_000)):
            ch.columns = ["DGUID", "coord", "n"]
            h = ch.DGUID.str[-7:]
            w = ch.coord.str.split(".").str[1].map(member)
            keep = h.str[:4].isin(want) | w.str[:4].isin(want)
            n = pd.to_numeric(ch.n[keep], errors="coerce").fillna(0)
            out.append(pd.DataFrame({"h": h[keep], "w": w[keep], "n": n})[n > 0])
            print(f"  CSD flows: {(k + 1) * 2}M rows, {time.time() - t0:.0f} s", flush=True)
    d = pd.concat(out, ignore_index=True)
    d.to_csv(cache, index=False)
    return d


def jobs_and_commuters(cfg, df, sig):
    """Per DA: jobs (all jobs at the workplace, like LODES WAC), work-end and home-end commuters.
    Returns (df with jobs, work_c, home_c), stats."""
    usual, home, bands = pow_by_ct()
    com = commuters_by_ct()
    df = df.copy()
    tracted = df.ct != ""
    cts = sorted(set(df.ct[tracted]))
    cma_of = {c: c[:12] for c in cts}  # 2021S0507 + CMA code
    cmas = sorted(set(cma_of.values()))
    s_now = cfg.get("wfh_now", WFH_NOW)
    stats = {"cmas": {}}
    ct_jobs, ct_work, ct_home, ct_u = {}, {}, {}, {}
    for cma in cmas:
        cma_row = "2021S0503" + cma[9:12]
        U_b, H_b = usual.loc[cma_row], home.loc[cma_row]
        r = (H_b / U_b.where(U_b > 0)).fillna(0.0)          # at home per usual-place worker, by band
        C, NF = com.loc[cma_row, "commuters"], com.loc[cma_row, "no_fixed"]
        U, H = float(U_b.sum()), float(H_b.sum())
        s21 = H / (H + C)
        f = max(0.0, 1.0 - s_now / s21)                      # share of 2021's at-home workers commuting now
        nf = NF / max(C - NF, 1.0)                           # no fixed workplace per usual-place worker
        stats["cmas"][cma_row] = dict(usual=U, at_home=H, commuters=C, no_fixed=NF, share_at_home_2021=round(s21, 4),
                                      share_at_home_now=s_now, back_to_commuting=round(f, 4))
        for c in [c for c in cts if cma_of[c] == cma]:
            u_b = usual.loc[c] if c in usual.index else 0.0 * U_b
            x = float((u_b * r).sum())
            u = float(u_b.sum())
            ct_u[c] = u
            ct_jobs[c] = u * (1 + nf) + x
            ct_work[c] = u * (1 + nf) + f * x
            h = float(home.loc[c].sum()) if c in home.index else 0.0
            cm = float(com.loc[c, "commuters"]) if c in com.index else 0.0
            ct_home[c] = cm + f * h
    # within a tract: jobs by OSM job places and population; home commuters by population
    signal = sig["errand"] + sig["leisure"] + sig["job"] + sig["land"]
    pop = df["pop"].to_numpy(np.float64)
    area = df["land_km2"].to_numpy(np.float64) + 1e-6
    g = df.groupby("ct").indices
    jobs = np.zeros(len(df))
    work = np.zeros(len(df))
    homec = np.zeros(len(df))
    for c, idx in g.items():
        if c == "":
            continue
        ps = pop[idx] / pop[idx].sum() if pop[idx].sum() > 0 else area[idx] / area[idx].sum()
        ss = signal[idx] / signal[idx].sum() if signal[idx].sum() > 0 else ps
        js = JOB_SIGNAL_SHARE * ss + (1 - JOB_SIGNAL_SHARE) * ps
        jobs[idx] = ct_jobs.get(c, 0.0) * js
        work[idx] = ct_work.get(c, 0.0) * js
        homec[idx] = ct_home.get(c, 0.0) * ps
    # areas outside every tract: their CSD's workplace count less its tracts', and home commuters
    # at the tracted DAs' rate per resident
    flows = csd_flows(cfg)
    by_w = flows.groupby("w").n.sum()
    rate_home = homec[tracted.to_numpy()].sum() / pop[tracted.to_numpy()].sum()
    rate_jobs = jobs.sum() / max(sum(ct_u.values()), 1.0)
    rate_work = work.sum() / max(sum(ct_u.values()), 1.0)
    un = ~tracted.to_numpy()
    homec[un] = pop[un] * rate_home
    ct_u_csd = pd.Series(ct_u).groupby(df[tracted].drop_duplicates("ct").set_index("ct").csd).sum() if ct_u else pd.Series(dtype=float)
    untr_jobs = 0.0
    for csd_, idx in df[un].groupby("csd").indices.items():
        idx = np.nonzero(un)[0][idx]
        u = max(0.0, float(by_w.get(csd_, 0.0)) - float(ct_u_csd.get(csd_, 0.0)))
        w8 = signal[idx] + pop[idx] / 100.0 + 1e-9
        jobs[idx] = u * rate_jobs * w8 / w8.sum()
        work[idx] = u * rate_work * w8 / w8.sum()
        untr_jobs += u * rate_jobs
    df["jobs"], df["work_c"], df["home_c"] = jobs, work, homec
    # transit share of 2021 commuters by home tract, for the checks
    tr = com.reindex(cts).fillna(0.0)
    stats.update(tracts=len(cts), jobs=float(jobs.sum()), work_c=float(work.sum()), home_c=float(homec.sum()),
                 untracted_pop=float(pop[un].sum()), untracted_jobs=untr_jobs,
                 transit_share_2021=float(tr.transit.sum() / max(tr.commuters.sum(), 1.0)))
    print(f"jobs (census place of work, CT, at-home jobs put back by income): {jobs.sum():,.0f}; commuters work end "
          f"{work.sum():,.0f}, home end {homec.sum():,.0f}; {len(cts):,} tracts; outside tracts {pop[un].sum():,.0f} people, "
          f"{untr_jobs:,.0f} jobs")
    for k, v in stats["cmas"].items():
        print(f"    CMA {k[-3:]}: usual place {v['usual']:,.0f}, at home {v['at_home']:,.0f} ({v['share_at_home_2021']:.1%} in 2021, "
              f"{v['share_at_home_now']:.1%} now: {v['back_to_commuting']:.1%} of them commute again), no fixed workplace {v['no_fixed']:,.0f}")
    return df, stats


# ---------------------------------------------------------------- build

def main(city):
    t0 = time.time()
    cfg = b.CITIES[city]
    for d in (b.RAW, RAW_CA, b.WORK, b.PACKS):
        os.makedirs(d, exist_ok=True)
    boundary = build_boundary(city, cfg)
    mask, mask_info = build_water(cfg, boundary)
    df, polys = das(cfg)
    places = osm_places(city, boundary)
    sig = per_da(df, polys, places)
    df, cstats = jobs_and_commuters(cfg, df, sig)
    J = df["jobs"].sum()
    df["errand_jobs"] = sig["errand"] * (ERRAND_SHARE * J / max(sig["errand"].sum(), 1.0))
    df["leisure_jobs"] = sig["leisure"] * (LEISURE_SHARE * J / max(sig["leisure"].sum(), 1.0))
    df["school_children"] = df["pop"] * df["child_share"]
    df["university_adults"] = df["pop"] * df["adult_share"]
    df = df[(df["pop"] > 0) | (df["jobs"] > 0) | (df["errand_jobs"] > 0) | (df["leisure_jobs"] > 0)].reset_index(drop=True)
    df.drop(columns=[]).to_csv(os.path.join(b.WORK, f"{city}.blocks.csv.gz"), index=False)
    lon0, lat0 = cfg["origin"]
    df, region_codes, region_cols = regions.add_columns(df)
    cols = ("pop", "jobs", "home_c", "work_c", "errand_jobs", "leisure_jobs", "school_children", "university_adults") + tuple(region_cols)
    uniq, vals, st = spread.spread(df, None, b.fetch, mask, lon0, lat0, cols=cols, polygons=polys)
    print(f"spread: {st}")
    pop, job, hc, wc = vals["pop"], vals["jobs"], vals["home_c"], vals["work_c"]
    keep = (pop > 0) | (job > 0)
    lostv = {c: float(vals[c][~keep].sum()) for c in ("errand_jobs", "leisure_jobs")}
    uniq, pop, job, hc, wc = uniq[keep], pop[keep], job[keep], hc[keep], wc[keep]
    region = regions.of_cells({c: vals[c][keep] for c in region_cols}, region_cols)
    print(f"spread: {len(uniq):,} cells, {int((pop > 0).sum()):,} with people, {int((job > 0).sum()):,} with jobs "
          f"(errand and leisure weight on cells with neither, dropped: {lostv})")
    ll = np.array([h3.cell_to_latlng(h3.int_to_str(int(c))) for c in uniq])
    x = b.R_EARTH * np.radians(ll[:, 1] - lon0) * np.cos(np.radians(lat0))
    y = b.R_EARTH * np.radians(ll[:, 0] - lat0)
    np.savez_compressed(os.path.join(b.WORK, f"{city}.origins.npz"), h3=uniq,
                        school_children=vals["school_children"][keep], university_adults=vals["university_adults"][keep])
    totals = {"pop": int(round(pop.sum())), "jobs": int(round(job.sum())),
              "commute_home": int(round(hc.sum())), "commute_work": int(round(wc.sum()))}
    sources = [
        {"what": "population", "name": "Statistics Canada 2021 census, 98-10-0015-01 population by dissemination area (no institutions taken off)",
         "url": STATCAN_CSV.format(t="98100015")},
        {"what": "dissemination areas", "name": "StatCan 2021 DA boundary file lda_000b21a_e, Esri Canada's copy", "url": DA_LAYER},
        {"what": "census tracts", "name": "StatCan 2021 CT boundary file lct_000b21a_e (a public ArcGIS Online copy)", "url": CT_LAYER},
        {"what": "jobs", "name": "StatCan 2021 census 98-10-0498-01, usual place of work by census tract of work and employment income; "
                                 "at-home workers put back at workplaces by income band; no fixed workplace (98-10-0480-01) in proportion",
         "url": STATCAN_CSV.format(t="98100498")},
        {"what": "home end of commutes", "name": "StatCan 2021 census 98-10-0480-01 commuters by census tract of residence, plus at-home workers (98-10-0498-01) commuting again",
         "url": STATCAN_CSV.format(t="98100480")},
        {"what": "work from home", "name": f"Labour Force Survey, May 2024: {cfg.get('wfh_now', WFH_NOW):.1%} of the Toronto CMA's employed mostly at home",
         "url": "https://www150.statcan.gc.ca/n1/daily-quotidien/240826/dq240826a-eng.htm"},
        {"what": "jobs outside census tracts", "name": "StatCan 2021 census 98-10-0459-01 commuting flows between census subdivisions",
         "url": STATCAN_CSV.format(t="98100459")},
        {"what": "everyday destinations", "name": "OpenStreetMap shops and services (errands), restaurants, bars, venues and hotels (leisure), counted per DA",
         "url": "https://www.openstreetmap.org/copyright"},
        {"what": "boundary", "name": "StatCan 2021 census subdivisions in census divisions " + ", ".join(cds(cfg)) + ", " + cfg.get("boundary_notes", ""),
         "url": CSD_LAYER},
        {"what": "water", "name": mask_info["source"], "url": LAKES_LAYER},
    ]
    commute = {"home": "census commuters by home tract plus at-home workers commuting again, to DAs by population",
               "work": "census usual place of work by work tract, no fixed workplace in proportion, plus at-home workers commuting again "
                       "(by income band), to DAs by OSM job places 80% and population 20%",
               "region_share": cfg.get("wfh_now", WFH_NOW), "census": cstats}
    region_block = {"level": "admin-1", "ids": region_codes, "names": ["ON" if c == "35" else c for c in region_codes],
                    "source": "Canadian province: the first two digits of each dissemination area id"}
    cells_json = b.write_cells(city, cfg, {"h3": uniq, "x_m": x, "y_m": y, "pop": pop, "jobs": job,
                                           "errand_jobs": vals["errand_jobs"][keep], "leisure_jobs": vals["leisure_jobs"][keep],
                                           "commute_home": hc, "commute_work": wc, "region": region},
                               totals, sources, mask, mask_info, commute, region_block)
    print(f"cells written in {time.time() - t0:.1f} s")
    finish(city)
    print(f"done in {time.time() - t0:.1f} s")


def finish(city):
    """From data/work/<city>.cells.* on: the gravity, the non-work purposes, the checks.
    `python pipeline/canada.py <city> --finish` reruns only this (the gravity needs the sim crate
    to compile; the cell build before it takes about 10 minutes)."""
    t0 = time.time()
    cfg = b.CITIES[city]
    b.add_gravity(os.path.join(b.WORK, f"{city}.cells.json"), city)
    import tier0
    print("everyday:", tier0.everyday_pack(city))
    schools(city)
    universities(city)
    enrich_osm(city)
    from read_pack import load_pack
    header, a = load_pack(city)
    tt = header["totals"]
    print(f"totals: pop {tt['pop']:,}, jobs {tt['jobs']:,}, commutes {tt['commute_work']:,}")
    checks(cfg, pd.read_csv(os.path.join(b.WORK, f"{city}.blocks.csv.gz"), dtype={"geoid": str}))
    for label in ("pop", "jobs"):
        v = a[label]
        print(f"top cells by {label}:")
        for i in np.argsort(-v)[:8]:
            lat, lon = h3.cell_to_latlng(h3.int_to_str(int(a["h3"][i])))
            print(f"    {h3.int_to_str(int(a['h3'][i]))}  {lat:.4f},{lon:.4f}  pop {a['pop'][i]:>8,.0f}  jobs {a['jobs'][i]:>8,.0f}")
    print(f"cells: {header['cells']:,}; finished in {time.time() - t0:.1f} s")


def checks(cfg, df):
    """Population against the census divisions' published counts; jobs against the commuting flow
    table's workplace counts (2021, usual place of work only, so jobs run above it by design)."""
    t = table("98100015", usecols=["DGUID", "Population and dwelling counts (5): Population, 2021 [1]"])
    cd = t[t.DGUID.str.startswith("2021A0003")]
    pub = {d[-4:]: v for d, v in zip(cd.DGUID, num(cd.iloc[:, 1]))}
    ours = df.groupby(df.geoid.str[:4])["pop"].sum()
    tot_pub = sum(pub.get(c, 0) for c in cds(cfg))
    print(f"check pop: DAs {ours.sum():,.0f} vs census divisions {tot_pub:,.0f} ({ours.sum() / tot_pub - 1:+.3%})")
    for c in cds(cfg):
        print(f"    {cfg['counties'][c]:<10} DAs {ours.get(c, 0):>11,.0f}  CD {pub.get(c, 0):>11,.0f}")
    fl = csd_flows(cfg)
    w = fl[fl.w.str[:4].isin(set(cds(cfg)))].n.sum()
    print(f"check jobs: {df['jobs'].sum():,.0f} vs census 2021 usual place of work in the pack's CSDs {w:,.0f} "
          f"({df['jobs'].sum() / w - 1:+.1%}; the pack adds back at-home workers and no fixed workplace)")


def rewrite(stem, header, buf):
    packio.write(stem, header, buf)


def pack_cells(city):
    stem = os.path.join(b.PACKS, city)
    with open(stem + ".json") as f:
        header = json.load(f)
    with open(stem + ".bin", "rb") as f:
        buf = bytearray(f.read())
    a = next(a for a in header["arrays"] if a["name"] == "h3")
    cells = np.frombuffer(buf, dtype="<u8", count=a["count"], offset=a["offset"]).copy()
    return stem, header, buf, cells


def origin_array(city, cells, name):
    o = np.load(os.path.join(b.WORK, f"{city}.origins.npz"))
    i = np.searchsorted(cells, o["h3"])
    ok = (i < len(cells)) & (cells[np.minimum(i, len(cells) - 1)] == o["h3"])
    assert ok.all(), "origin cells missing from the pack; rebuild"
    out = np.zeros(len(cells), "<f4")
    out[i] = o[name]
    return out


# T-453: 2021 census, usual place of work by census tract of work x NAICS sector (25% sample);
# errands as LODES's CNS07 + CNS19 (44-45, 81), leisure CNS17 + CNS18 (71, 72)
CT_SECTORS = {"errand": ["44-45", "81"], "leisure": ["71", "72"]}


def ct_sector_table():
    """DataFrame CT DGUID x NAICS sector of usual-place-of-work workers (98-10-0492-01), every CT in
    Canada, from the 194 MB zip (3.8 GB inside) streamed once into a small CSV."""
    path = os.path.join(RAW_CA, "98100492_ct_sector_usual.csv")
    if not os.path.exists(path):
        import csv
        import io
        key = "Total - Work activity during the reference year,Total - Age,Total - Gender,"
        with statcan("98100492").open("98100492.csv") as f, open(path + ".part", "w", newline="", encoding="utf-8") as g:
            w = csv.writer(g)
            w.writerow(["dguid", "sector", "usual"])
            for line in io.TextIOWrapper(f, encoding="utf-8-sig"):
                if key in line and ",2021S0507" in line:
                    r = next(csv.reader([line]))
                    w.writerow([r[2], r[6].split(" ")[0], r[12]])
        os.replace(path + ".part", path)
    t = pd.read_csv(path, dtype=str)
    t["usual"] = num(t.usual)
    return t.pivot_table(index="dguid", columns="sector", values="usual", aggfunc="sum").fillna(0.0)


def ct_sectors(city, lat, lon, jobs):
    """tier0's `everyday` hook for a Canadian pack: per cell, its jobs x its census tract of work's
    share of usual-place-of-work workers in the sectors (CT_SECTORS); cells in no tract take the
    pack's tracts' share."""
    cfg = b.CITIES[city]
    return _ct_sectors(f"{SUB}ct_on_{'_'.join(cds(cfg))}", "PRUID='35'", lat, lon, jobs)


def plan_ct_sectors(city, lat, lon, jobs):
    """ct_sectors for any planned Canadian city: the tracts of every province in the pack's box."""
    return _ct_sectors(f"{SUB}ct_{city}", "1=1", lat, lon, jobs)


@functools.lru_cache(maxsize=1)
def da_population():
    """{DAUID: 2021 census population}, every DA in Canada (98-10-0015-01)."""
    pop = table("98100015", usecols=["DGUID", "Population and dwelling counts (5): Population, 2021 [1]"])
    pop = pop[pop.DGUID.str.startswith("2021S0512")]
    return dict(zip(pop.DGUID.str[9:], num(pop.iloc[:, 1])))


def plan_population(city, boundary, sig):
    """T-564: tier0's `population` hook for a planned Canadian row: each 2021 dissemination area whose
    representative point is inside the outline, its census population spread over its cells by GHSL;
    cells in no such DA keep GHSL's people (an edge DA counted outside would pile its total on its few
    inside cells)."""
    import england
    feats = arcgis.features(f"{SUB}da_{city}", DA_LAYER, fields="DAUID,OBJECTID", bbox=boundary.bounds)
    units = []
    for f in feats:
        g = make_valid(shape(f["geometry"]))
        if boundary.contains(g.representative_point()):
            units.append((f["properties"]["DAUID"], g))
    tot = da_population()
    ghsl = np.asarray(sig["pop_ghsl"], float)
    idx, _ = england.cell_units(sig["lat"], sig["lon"], units)
    out = england.spread_totals(sig["lat"], sig["lon"], units, tot, ghsl)
    out[idx < 0] = ghsl[idx < 0]
    print(f"  census population by DA: {len(units):,} of {len(feats):,} DAs in the box, {out.sum():,.0f} people "
          f"(GHSL {ghsl.sum():,.0f}; {int((idx < 0).sum()):,} cells in no DA keep GHSL's)", flush=True)
    return out, ("Statistics Canada 2021 census, 98-10-0015-01 population by dissemination area (Esri Canada's copy "
                 "of the DA boundaries), spread inside each DA by GHSL GHS-POP E2020")


def plan_hooks(row):
    """T-555: Toronto's country keys that fit tier 0, for any planned Canadian row: everyday
    destinations by census tract of work, and (T-564) population by DA. Toronto's jobs by tract of
    work run in canada.main, not as tier 0 hooks, and its schools and campuses are Ontario's files."""
    return {"everyday": plan_ct_sectors, "population": plan_population}


def _ct_sectors(name, where, lat, lon, jobs):
    ct = arcgis.features(name, CT_LAYER, where=where, fields="CTUID,DGUID,FID",
                         bbox=(lon.min(), lat.min(), lon.max(), lat.max()))
    units = [(f["properties"]["DGUID"], make_valid(shape(f["geometry"]))) for f in ct]
    import england
    idx, _ = england.cell_units(lat, lon, units)
    t = ct_sector_table()
    code = pd.Series([units[i][0] if i >= 0 else "" for i in idx])
    here = t.index.isin(code)
    out = {}
    for kind, cols in CT_SECTORS.items():
        s = t[cols].sum(1)
        share = (s / t["Total"].where(t["Total"] > 0)).clip(upper=1.0)
        v = code.map(share).fillna(s[here].sum() / t["Total"][here].sum()).to_numpy()
        out[kind] = (jobs * v, f"StatCan 2021 census 98-10-0492-01, usual place of work by census tract of work x NAICS {', '.join(cols)}")
    return out


# ---------------------------------------------------------------- schools and universities

ON_SCHOOLS = "https://data.ontario.ca/dataset/d85f68c5-fcb0-4b4d-aec5-3047db47dcd5/resource/f381e1de-3aa8-4ae9-826a-20009a6f4a01/download/new_sif_data_table_2023_24prelim_en_january2026.xlsx"
COU_TABLE3 = "https://ontariosuniversities.ca/wp-content/uploads/2024/05/Table-3-Fall-Term-Headcounts-by-Institution-and-Level-of-Study.pdf"
ON_COLLEGES = "https://data.ontario.ca/dataset/e9634682-b9dc-46a6-99b4-e17c86e00190/resource/07fdeefd-fe44-4df8-bd7d-5419a79f90ec/download/college_enrolment_headcount_2024-25.xlsx"
CAMPUSES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ca_campuses.csv")


def place_at_cells(records, cells, limit=500.0):
    """Each record's count at the pack cell under its point, or the nearest within `limit` m."""
    import enrich_airports as air
    index = {int(c): i for i, c in enumerate(cells)}
    out = np.zeros(len(cells), "<f4")
    matched, omitted, snaps = [], [], 0
    for r in records:
        cell, snap = air.cell_of(index, r["lat"], r["lon"], limit)
        if cell is None:
            omitted.append(r)
            continue
        snaps += snap > 0
        out[cell] += r["enrollment"]
        matched.append(r | {"cell": int(cell), "snap_m": snap})
    return out, matched, omitted, snaps


def schools(city):
    """Ontario's school information file: every publicly funded school (public and Catholic,
    English and French) with its enrolment (rounded to 5) and point, inside the boundary."""
    cfg = b.CITIES[city]
    boundary = build_boundary(city, cfg)
    x = pd.read_excel(b.fetch(ON_SCHOOLS, "ca/on_sif_2023_24.xlsx"))
    x = x[np.isfinite(pd.to_numeric(x.Latitude, errors="coerce")) & (pd.to_numeric(x.Enrolment, errors="coerce") > 0)]
    shapely.prepare(boundary)  # T-425
    recs =[dict(name=r["School Name"], board=r["Board Name"], enrollment=float(r.Enrolment), lat=float(r.Latitude), lon=float(r.Longitude))
            for _, r in x.iterrows() if boundary.covers(Point(float(r.Longitude), float(r.Latitude)))]
    stem, header, buf, cells = pack_cells(city)
    enr, matched, omitted, snaps = place_at_cells(recs, cells)
    total = sum(r["enrollment"] for r in recs)
    share = float(enr.sum(dtype=np.float64)) / total
    if share < 0.95:
        raise ValueError(f"school location match {share:.1%} below 95%")
    children = origin_array(city, cells, "school_children")
    buf = packio.replace_cell_arrays(header, buf, {"school_children": children, "school_enrollment": enr})
    header["school"] = {
        "coverage": "Ontario publicly funded schools (public and Catholic, English and French) in the pack's boundary; no private schools",
        "school_year": "2023-2024 (preliminary)", "enrolled": float(enr.sum(dtype=np.float64)), "matched_buildings": len(matched),
        "location_match_share": share, "child_weight": float(children.sum(dtype=np.float64)),
        "child_origin": "2021 census DA population x DA ages 5-17 plus 2/5 of under-5 (ages 15-17 as 3/5 of 15-19)",
        "attendance_factor": 0.9, "sources": [ON_SCHOOLS, AGE_LAYER]}
    rewrite(stem, header, buf)
    with open(os.path.join(b.WORK, city + ".schools.audit.json"), "w") as f:
        json.dump({"metadata": header["school"], "schools": matched, "omitted": omitted}, f, indent=1)
    print(f"Ontario schools: {len(matched)} matched, enrollment {enr.sum():,.0f} ({share:.2%}), {snaps} snapped, {len(omitted)} omitted; "
          f"children {children.sum():,.0f}", flush=True)


def campuses():
    """Universities (Council of Ontario Universities, Table 3, fall 2022-23 headcount, full and
    part time) and colleges (Ontario's college enrolment file, fall full-time headcount by campus,
    latest year), each at a campus point from pipeline/ca_campuses.csv."""
    c = pd.read_csv(CAMPUSES, comment="#")
    x = pd.read_excel(b.fetch(ON_COLLEGES, "ca/on_college_headcount_2024_25.xlsx"), sheet_name="Campus")
    x.columns = ["college", "year", "campus", "n"]
    year = sorted(x.year.unique())[-1]
    x = x[x.year == year]
    head = {(r.college, r.campus): float(np.nan_to_num(pd.to_numeric(r.n, errors="coerce"))) for r in x.itertuples()}
    recs = []
    for r in c.itertuples():
        if r.source == "college":
            n = head.get((r.institution, r.campus))
            if n is None:
                raise ValueError(f"no college row for {r.institution} / {r.campus}")
        else:
            n = float(r.headcount) * float(r.share)
        recs.append(dict(name=f"{r.institution}, {r.campus}", enrollment=n, lat=float(r.lat), lon=float(r.lon), source=r.source))
    return recs, year


def universities(city):
    cfg = b.CITIES[city]
    boundary = build_boundary(city, cfg)
    recs, year = campuses()
    shapely.prepare(boundary)  # T-425
    recs = [r for r in recs if boundary.covers(Point(r["lon"], r["lat"]))]
    stem, header, buf, cells = pack_cells(city)
    enr, matched, omitted, snaps = place_at_cells(recs, cells)
    total = sum(r["enrollment"] for r in recs)
    share = float(enr.sum(dtype=np.float64)) / total
    if share < 0.95:
        raise ValueError(f"campus location match {share:.1%} below 95%")
    adults = origin_array(city, cells, "university_adults")
    buf = packio.replace_cell_arrays(header, buf, {"university_adults": adults, "university_enrollment": enr})
    header["university"] = dict(
        coverage="Ontario universities (Council of Ontario Universities Table 3, fall 2022-23 headcount, full and part time) and colleges "
                 f"(Ontario college enrolment, fall {year} full-time headcount by campus) at campus points in pipeline/ca_campuses.csv",
        enrolled=float(enr.sum(dtype=np.float64)), institutions=len({r["name"].split(",")[0] for r in matched}), sites=len(matched),
        location_match_share=share, adult_weight=float(adults.sum(dtype=np.float64)),
        adult_origin="2021 census DA population x DA ages 18+ (15-19 split 3/5 under 18)",
        attendance_factor=0.6, decay_m=6000,
        limitations="universities split between campuses by their own published campus shares where they have several; "
                    "college partner campuses at private schools left out; part-time college students not counted",
        sources=[COU_TABLE3, ON_COLLEGES, CAMPUSES])
    rewrite(stem, header, buf)
    with open(os.path.join(b.WORK, city + ".universities.audit.json"), "w") as f:
        json.dump(dict(metadata=header["university"], institutions=matched, omitted=omitted), f, indent=1)
    print(f"universities and colleges: {len(matched)} campuses, {enr.sum():,.0f} students ({share:.2%}), {snaps} snapped; adults {adults.sum():,.0f}", flush=True)


# ---------------------------------------------------------------- airports (enrich_airports.py calls these)

AIR_TABLE = "23100253"
# Share of boardings that start the passenger's air journey (not connecting). Canada has no
# national source like DB1B; Pearson is Air Canada's main hub, so a judgement value, the same order
# as the US hubs that DB1B measures (Dulles 0.74, BWI 0.66). Every other airport 1.
OD_SHARE_JUDGEMENT = {"CYYZ": 0.75}


def airport_boardings(year=2024):
    """{normalised airport name: (boardings = enplaned, the year)} from StatCan 23-10-0253-01."""
    t = table(AIR_TABLE, usecols=["REF_DATE", "GEO", "Air passenger traffic", "VALUE"])
    t = t[t["Air passenger traffic"] == "Passengers enplaned"]
    t["v"] = pd.to_numeric(t.VALUE, errors="coerce")
    t = t[t.v.notna()]
    out = {}
    for geo, g in t.groupby("GEO"):
        g = g[g.REF_DATE.astype(int) <= year].sort_values("REF_DATE")
        if len(g):
            out[geo] = (float(g.v.iloc[-1]), int(g.REF_DATE.iloc[-1]))
    return out


def _tokens(s):
    s = re.sub(r"[^a-z ]", " ", s.lower().replace(".", ""))
    drop = {"airport", "international", "municipal", "regional", "ontario", "the", "of", "a", "and", "city"}
    return {w for w in s.split() if w not in drop and len(w) > 1}


def match_airport(ap, table_rows):
    """The StatCan row for an OurAirports airport, by name tokens (StatCan: 'City/Airport, Province')."""
    want = _tokens(ap["name"]) | _tokens(ap.get("municipality", "") or "")
    best, score = None, 0.0
    for geo in table_rows:
        t = _tokens(geo.split(",")[0].replace("/", " "))
        if not t:
            continue
        sc = len(want & t) / len(t)
        if sc > score:
            best, score = geo, sc
    return (best, score) if score >= 0.6 else (None, score)


@functools.lru_cache(maxsize=1)
def population_points():
    """(lat, lon, population, in-pack CD code) for every Ontario CSD (2021) and every US county
    (2020 centres of population), for the airports' pack share."""
    import enrich_airports as air
    t = table("98100015", usecols=["DGUID", "Population and dwelling counts (5): Population, 2021 [1]"])
    t = t[t.DGUID.str.startswith("2021A0005")]
    pop = dict(zip(t.DGUID.str[9:], num(t.iloc[:, 1])))
    rows = []
    for f in csd_features():
        u = f["properties"]["CSDUID"]
        if not f.get("geometry") or u not in pop:
            continue
        p = shape(f["geometry"]).centroid  # no make_valid: slow on the far north's shores
        rows.append((p.y, p.x, pop[u], u[:4]))
    co = pd.read_csv(b.fetch(air.CENPOP, air.SUB + "CenPop2020_Mean_CO.txt"), encoding="utf-8-sig")
    rows += [(r.LATITUDE, r.LONGITUDE, r.POPULATION, "us") for r in co.itertuples()]
    return pd.DataFrame(rows, columns=["lat", "lon", "pop", "cd"])


# ---------------------------------------------------------------- OSM from the extract, not Overpass

def _hstore(s):
    return dict(re.findall(r'"([^"]*)"=>"([^"]*)"', s or ""))


def _ring(coords):
    return [{"lat": round(y, 7), "lon": round(x, 7)} for x, y in coords]


def _elements(df, cols, node=False):
    """GDAL OSM rows as Overpass `out geom tags` elements: nodes; closed ways; relations with
    their outer and inner rings as members (what overpass.polygon reads)."""
    out = []
    for r in df.itertuples():
        tags = _hstore(getattr(r, "other_tags", ""))
        for c in cols:
            v = getattr(r, c, None)
            if isinstance(v, str):
                tags[c] = v
        if isinstance(getattr(r, "name", None), str):
            tags["name"] = r.name
        g = r.geometry
        if node:
            out.append({"type": "node", "id": int(r.osm_id), "lat": g.y, "lon": g.x, "tags": tags})
            continue
        polys = list(g.geoms) if g.geom_type == "MultiPolygon" else [g]
        way_id = getattr(r, "osm_way_id", None)
        if isinstance(way_id, str) and len(polys) == 1 and not polys[0].interiors:
            out.append({"type": "way", "id": int(way_id), "geometry": _ring(polys[0].exterior.coords), "tags": tags})
        else:
            members = []
            for p in polys:
                members.append({"type": "way", "role": "outer", "geometry": _ring(p.exterior.coords)})
                members += [{"type": "way", "role": "inner", "geometry": _ring(i.coords)} for i in p.interiors]
            rid = int(r.osm_id) if isinstance(r.osm_id, str) else -int(way_id)
            out.append({"type": "relation", "id": rid, "members": members, "tags": tags})
    return out


def _write(name, elements, extract):
    path = os.path.join(b.RAW, name + ".json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump({"elements": elements, "source": f"Geofabrik {__import__('osm_features').label(extract)} extract (T-277)"}, f)
    os.replace(path + ".tmp", path)


def osm_caches(city, boundary, extract=OSM_EXTRACT):
    """Write the Overpass caches that enrich_airports, enrich_medical and enrich_parks read, from
    the extract's cached features instead (osm_features; T-277: the public Overpass servers
    answered 504 for the GTHA). Each file is written only if missing, in the same `out geom tags`
    shape. Returns the park elements and their cache name for enrich_parks."""
    import osm_features
    import enrich_airports as air
    import enrich_medical as med
    import enrich_parks as parks

    def read(layer, **kw):
        return osm_features.read(extract, layer, **kw)
    w, s, e, n = boundary.bounds
    bb = (w - 0.05, s - 0.05, e + 0.05, n + 0.05)
    mcols = ["osm_id", "osm_way_id", "name", "amenity", "aeroway", "leisure", "natural", "building", "other_tags"]
    t0 = time.time()
    term = f"{air.SUB}osm_terminals_{city}"
    if not os.path.exists(os.path.join(b.RAW, term + ".json")):
        d = read("multipolygons", columns=mcols, bbox=bb, where="aeroway = 'terminal'")
        _write(term, _elements(d, ["aeroway", "amenity", "building"]), extract)
        print(f"  extract: {len(d)} terminals", flush=True)
    medn = f"{med.SUB}osm_medical_{city}"
    kinds = "('hospital','clinic','doctors','dentist')"
    hc = " OR ".join(f"other_tags LIKE '%\"healthcare\"=>\"{k}\"%'" for k in ("hospital", "clinic", "doctor", "dentist"))
    if not os.path.exists(os.path.join(b.RAW, medn + ".json")):
        p = read("points", columns=["osm_id", "name", "other_tags"], bbox=bb,
                 where=" OR ".join(f"other_tags LIKE '%\"amenity\"=>\"{k}\"%'" for k in ("hospital", "clinic", "doctors", "dentist")) + " OR " + hc)
        m = read("multipolygons", columns=mcols, bbox=bb, where=f"amenity IN {kinds} OR {hc}")
        _write(medn, _elements(p, [], node=True) + _elements(m, ["amenity", "building"]), extract)
        print(f"  extract: {len(p)} medical points, {len(m)} medical areas", flush=True)
    bld = f"{med.SUB}osm_hospital_buildings_{city}"
    if not os.path.exists(os.path.join(b.RAW, bld + ".json")):
        with open(os.path.join(b.RAW, medn + ".json"), encoding="utf-8") as f:
            sites = [overpass.polygon(el) for el in json.load(f)["elements"] if el["type"] != "node" and (el["tags"].get("amenity") == "hospital" or el["tags"].get("healthcare") == "hospital")]
        sites = [g for g in sites if g is not None]
        if sites:
            # building:levels and height come through other_tags, as Overpass gives them
            d = read("multipolygons", columns=mcols, mask=union_all(sites).buffer(0.0005), where="building IS NOT NULL")
            _write(bld, _elements(d, ["building"]), extract)
            print(f"  extract: {len(d)} buildings on {len(sites)} hospital sites", flush=True)
        else:
            _write(bld, [], extract)
    pk = f"{parks.SUB}osm_parks_extract_{city}"
    if not os.path.exists(os.path.join(b.RAW, pk + ".json")):
        lei = "('park','common','recreation_ground','beach_resort','nature_reserve','garden')"
        m = read("multipolygons", columns=mcols, bbox=bb,
                 where=f"leisure IN {lei} OR (\"natural\" = 'beach' AND name IS NOT NULL)")
        p = read("points", columns=["osm_id", "name", "other_tags"], bbox=bb,
                 where="other_tags LIKE '%\"leisure\"=>\"park\"%'")
        _write(pk, _elements(m, ["leisure", "natural"]) + _elements(p, [], node=True), extract)
        print(f"  extract: {len(m)} park areas, {len(p)} park points; {time.time() - t0:.0f} s", flush=True)
    with open(os.path.join(b.RAW, pk + ".json"), encoding="utf-8") as f:
        return json.load(f)["elements"], [pk]


def enrich_osm(city):
    """Airports, medical visits and parks (the shared enrich scripts), their OSM read from the
    extract's cached features (osm_caches) rather than Overpass."""
    import enrich_airports
    import enrich_medical
    import enrich_parks
    els, names = osm_caches(city, build_boundary(city, b.CITIES[city]))
    enrich_airports.main(city)
    enrich_medical.main(city)
    keep = enrich_parks.fetch_all
    enrich_parks.fetch_all = lambda c, boundary: (els, names)
    try:
        enrich_parks.main(city)
    finally:
        enrich_parks.fetch_all = keep


if __name__ == "__main__":
    import sys
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    (finish if "--finish" in sys.argv else main)(args[0] if args else "tor")
