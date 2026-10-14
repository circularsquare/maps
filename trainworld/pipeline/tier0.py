"""T-287: tier 0, a city pack from global sources only (research/world-sampler.md section 2).

    python pipeline/build_city.py <city>      (build_city.main hands a city with "tier0" to main())

Every input comes from a source that covers the world, so any city can be built; a country's own
census or registers then upgrade single inputs by replacing one function. What each input is:

- Population: GHSL GHS-POP (R2023A, 100 m Mollweide tiles, CC BY 4.0), each pixel split over four
  50 m points (points on water dropped, the pixel's people kept on its land points), summed into
  the H3 cells, then scaled to admin totals where the config gives them (any census publishes
  district totals). No institutions taken off.
- Jobs (the jobs proxy, T-012): a total of population x jobs per resident (World Bank: ILO's
  employment-to-population ratio 15+ x the share aged 15+), spread over cells by a blend of three
  normalised signals: GHSL non-residential built-up volume (GHS-BUILT-V NRES) times the cell's
  mean non-residential height (volume over GHS-BUILT-S NRES surface) to the power HEIGHT_POW, OSM
  job places (shops, amenities, offices, crafts, commercial and industrial land;
  canada.osm_places) and population. The weights are JOB_BLEND (notes/T-287.md); a city with much
  informal or home-based work should raise the population share.
- Commuters: jobs x (1 - work from home), the config's one national share (default 0), at both
  ends; the home end in proportion to population (no small-area source of where workers live).
- Everyday destinations: OSM shops and services (errands) and food, venues and hotels (leisure)
  per cell, scaled to 15% and 8% of jobs (as Canada's).
- Schools: OSM amenity=school sites; pupils = population x the national share aged 0-14 (World
  Bank) x SCHOOL_ENROL, spread over the sites by the square root of their area (points take the
  median site). Children are the same share of every cell's people.
- Universities: OSM amenity=university|college, each institution sized by its Wikidata student
  count (P2196) split over its sites by area; sites with no Wikidata count by area at the
  median students per hectare of those with one; points with no count left out. The total is
  then scaled to population x the national tertiary students a resident (UNESCO via the World
  Bank, SE.TER.ENRL / SP.POP.TOTL; T-288), so the counts and areas only spread it.
- Airports: OurAirports scheduled-service airports in the boundary, boardings = Wikidata
  patronage (P3872) / 2, O&D share 1 except the config's judgement values for hubs, pack share
  from the GHSL population around the airport (exp(-d/20 km), as enrich_airports).
- Medical and parks: the shared scripts, OSM read from the Geofabrik extract (canada.osm_caches).
- Water: the config's `water_source` (lon/lat polygons, as water.build_mask takes); for any country
  `osm_water.source([<Geofabrik extracts>])` (T-041: OSM coastline sea plus the extracts' inland
  water).
- Gravity curve: the config's "decay_pow"/"decay_km" if given (borrowed from a fitted city), else
  the length fitted so the model's median commute (straight line between 2 km zones) matches the
  config's published figure ("curve": {"median_km" or "mean_km", "pow"}), fit_curve below.
- Regions: one region, no border factor.

Config keys (a CITIES entry with "tier0": True): origin, water, boundary_file (GeoJSON, any
polygons, dissolved; or a callable returning its path, as england.itl3_boundary), water_source, extract (Geofabrik path), iso2 (World Bank country),
admin_totals (callable boundary -> [(name, shapely geometry, population)], optional), wfh (share
working at home, optional), od_share ({OurAirports ident: share}, optional), curve or
decay_pow/decay_km, region_name (optional), epoch (GHSL epoch, default 2020).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")

import io
import json
import math
import time
import zipfile

import h3
import numpy as np
import pandas as pd
import requests
import shapely
from shapely import make_valid, union_all
from shapely.geometry import Point, mapping, shape

import build_city as b
import education_sources as education

GHSL = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/{group}/{product}/V1-0/tiles/{product}_V1_0_R{r}_C{c}.zip"
PRODUCTS = {
    "pop": ("GHS_POP_GLOBE_R2023A", "GHS_POP_E{epoch}_GLOBE_R2023A_54009_100"),
    "nres_v": ("GHS_BUILT_V_GLOBE_R2023A", "GHS_BUILT_V_NRES_E{epoch}_GLOBE_R2023A_54009_100"),
    "nres_s": ("GHS_BUILT_S_GLOBE_R2023A", "GHS_BUILT_S_NRES_E{epoch}_GLOBE_R2023A_54009_100"),
}
MOLL_X0, MOLL_Y0, TILE_M, PIX_M = -18041000.0, 9000000.0, 1_000_000.0, 100.0
WORLDBANK = "https://api.worldbank.org/v2/country/{iso2}/indicator/SL.EMP.TOTL.SP.ZS;SP.POP.0014.TO.ZS?source=2&format=json&date=2015:2025&per_page=100"
WIKIDATA = "https://query.wikidata.org/sparql"

# The jobs proxy's blend (shares of the job total): GHSL non-residential volume weighted by height,
# OSM job places, population. Chosen by judgement, then checked on Toronto against the census
# (notes/T-287.md); the height power 1.5 was picked on Toronto (in-sample: without it the centre
# held a third of its jobs). Recheck both on London (T-288).
JOB_BLEND = {"nres": 0.5, "osm": 0.3, "pop": 0.2}
HEIGHT_POW = 1.5
ERRAND_SHARE, LEISURE_SHARE = 0.15, 0.08
SCHOOL_ENROL = 0.9                    # pupils per child aged 0-14 (judgement; UNESCO rates would refine)
DEFAULT_CURVE_POW = 1.0               # between the US fit (0.8) and Toronto's (1.2); only the length is fitted
ZONE_M = 2000.0
INTRAZONAL_KM = 1.04                  # the game's same-zone distance


def raw(*p):
    return os.path.join(b.RAW, "tier0", *p)


# ---------------------------------------------------------------- GHSL grids

def _moll():
    from pyproj import Transformer
    return (Transformer.from_crs("EPSG:4326", "ESRI:54009", always_xy=True),
            Transformer.from_crs("ESRI:54009", "EPSG:4326", always_xy=True))


def _tile_path(kind, epoch, r, c):
    group, product = PRODUCTS[kind]
    product = product.format(epoch=epoch)
    url = GHSL.format(group=group, product=product, r=r, c=c)
    path = raw("ghsl", f"{product}_V1_0_R{r}_C{c}.zip")
    if os.path.exists(path):
        return path
    if os.path.exists(path + ".none"):
        return None
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"  downloading {url}", flush=True)
    resp = requests.get(url, headers=b.UA, timeout=600)
    if resp.status_code == 404:  # GHSL has no tile over open ocean
        open(path + ".none", "w").close()
        return None
    resp.raise_for_status()
    with open(path + ".part", "wb") as f:
        f.write(resp.content)
    os.replace(path + ".part", path)
    return path


def _read_tile_window(path, r0, r1, c0, c1, tile_x0=None, tile_y0=None):
    """Rows r0:r1, columns c0:c1 of a GHSL tile (counted from the nominal 1,000 km tile's corner
    tile_x0, tile_y0), decoding only the TIFF tiles they touch. Some tiles are cut short (T-288:
    GHS-BUILT-V NRES R3_C18 is 9,000 pixels wide); the GeoTIFF tie point says where the image
    starts, and rows or columns outside it are zero."""
    import tifffile
    with zipfile.ZipFile(path) as z:
        data = z.read(next(n for n in z.namelist() if n.endswith(".tif")))
    with tifffile.TiffFile(io.BytesIO(data)) as tf:
        pg = tf.pages[0]
        out = np.zeros((r1 - r0, c1 - c0), np.float64)
        dr = dc = 0
        tie = pg.tags.get(33922)
        if tile_x0 is not None and tie is not None:
            dc = int(round((tie.value[3] - tile_x0) / PIX_M))  # image column 0 in tile columns
            dr = int(round((tile_y0 - tie.value[4]) / PIX_M))
        R0, R1 = max(r0 - dr, 0), min(r1 - dr, pg.imagelength)  # in image rows and columns
        C0, C1 = max(c0 - dc, 0), min(c1 - dc, pg.imagewidth)
        if R0 >= R1 or C0 >= C1:
            return out
        oR, oC = R0 + dr - r0, C0 + dc - c0  # where the image part lands in out
        sub = out[oR:oR + (R1 - R0), oC:oC + (C1 - C0)]
        r0, r1, c0, c1 = R0, R1, C0, C1
        th, tw = pg.tilelength, pg.tilewidth
        nx = math.ceil(pg.imagewidth / tw)
        fh = tf.filehandle
        for ty in range(r0 // th, (r1 - 1) // th + 1):
            for tx in range(c0 // tw, (c1 - 1) // tw + 1):
                i = ty * nx + tx
                fh.seek(pg.dataoffsets[i])
                seg = pg.decode(fh.read(pg.databytecounts[i]), i)[0]
                seg = np.asarray(seg)
                seg = seg.reshape(seg.shape[-3], seg.shape[-2]).astype(np.float64)  # (.., rows, cols, samples)
                ya, xa = ty * th, tx * tw
                ra, rb = max(r0, ya), min(r1, ya + seg.shape[0])
                ca, cb = max(c0, xa), min(c1, xa + seg.shape[1])
                sub[ra - r0:rb - r0, ca - c0:cb - c0] = seg[ra - ya:rb - ya, ca - xa:cb - xa]
    out[out < 0] = 0.0  # nodata (-200 in GHS-POP)
    return out


def ghsl_window(kind, bbox_ll, epoch=2020, pad_m=0.0):
    """A GHSL 100 m grid over a lon/lat box (plus pad_m), mosaicked from as many 1,000 km tiles as
    it touches: (array, west edge x, north edge y) in Mollweide metres, row 0 north."""
    fwd, _ = _moll()
    w, s, e, n = bbox_ll
    xs, ys = fwd.transform([w, w, e, e, (w + e) / 2, (w + e) / 2], [s, n, s, n, s, n])
    xmin, xmax = min(xs) - pad_m, max(xs) + pad_m
    ymin, ymax = min(ys) - pad_m, max(ys) + pad_m
    X0 = MOLL_X0 + math.floor((xmin - MOLL_X0) / PIX_M) * PIX_M
    Y0 = MOLL_Y0 - math.floor((MOLL_Y0 - ymax) / PIX_M) * PIX_M
    W, H = math.ceil((xmax - X0) / PIX_M), math.ceil((Y0 - ymin) / PIX_M)
    out = np.zeros((H, W), np.float64)
    r_first, r_last = int((MOLL_Y0 - Y0) // TILE_M) + 1, int((MOLL_Y0 - (Y0 - H * PIX_M) - 1) // TILE_M) + 1
    c_first, c_last = int((X0 - MOLL_X0) // TILE_M) + 1, int((X0 + W * PIX_M - 1 - MOLL_X0) // TILE_M) + 1
    for r in range(r_first, r_last + 1):
        for c in range(c_first, c_last + 1):
            tx0, ty0 = MOLL_X0 + (c - 1) * TILE_M, MOLL_Y0 - (r - 1) * TILE_M
            # this tile's pixels inside the window (tile rows and columns)
            c_a, c_b = max(0, round((X0 - tx0) / PIX_M)), min(10000, round((X0 + W * PIX_M - tx0) / PIX_M))
            r_a, r_b = max(0, round((ty0 - Y0) / PIX_M)), min(10000, round((ty0 - (Y0 - H * PIX_M)) / PIX_M))
            if c_a >= c_b or r_a >= r_b:
                continue
            path = _tile_path(kind, epoch, r, c)
            if path is None:
                continue
            win = _read_tile_window(path, r_a, r_b, c_a, c_b, tx0, ty0)
            wr, wc = round((Y0 - ty0) / PIX_M) + r_a, round((tx0 - X0) / PIX_M) + c_a
            out[wr:wr + win.shape[0], wc:wc + win.shape[1]] = win
    return out, X0, Y0


def grid_to_cells(arr, X0, Y0, origin, land=None, sub=2):
    """Each pixel's value over sub x sub points; points on water (land(x_m, y_m) false) dropped and
    the pixel's value kept on its land points (all points if none). Returns {h3 int: value}."""
    _, inv = _moll()
    rows, cols = np.nonzero(arr > 0)
    v = arr[rows, cols]
    k = sub * sub
    off = (np.arange(sub) + 0.5) / sub * PIX_M
    dx, dy = np.meshgrid(off, off)
    x = (X0 + cols[:, None] * PIX_M + dx.ravel()[None, :]).ravel()
    y = (Y0 - rows[:, None] * PIX_M - dy.ravel()[None, :]).ravel()
    lon, lat = inv.transform(x, y)
    lon, lat = np.asarray(lon), np.asarray(lat)
    val = np.repeat(v / k, k)
    if land is not None:
        lon0, lat0 = origin
        xm = b.R_EARTH * np.radians(lon - lon0) * np.cos(np.radians(lat0))
        ym = b.R_EARTH * np.radians(lat - lat0)
        ok = land(xm, ym).reshape(-1, k)
        n_ok = ok.sum(1)
        w = np.where(n_ok[:, None] > 0, ok / np.maximum(n_ok, 1)[:, None], 1.0 / k)
        val = (v[:, None] * w).ravel()
        keep = val > 0
        lon, lat, val = lon[keep], lat[keep], val[keep]
    ids = np.fromiter((h3.str_to_int(h3.latlng_to_cell(la, lo, 9)) for la, lo in zip(lat, lon)), np.uint64, len(lat))
    u, i = np.unique(ids, return_inverse=True)
    return dict(zip(u.tolist(), np.bincount(i, weights=val).tolist()))


def land_function(mask):
    """A fast vectorised `not water` over the mask's dense raster."""
    d = mask.dense()

    def land(x, y):
        c = np.floor((x - mask.x0) / mask.cell).astype(np.int64)
        r = np.floor((y - mask.y0) / mask.cell).astype(np.int64)
        inside = (c >= 0) & (c < mask.W) & (r >= 0) & (r < mask.H)
        out = np.ones(len(x), bool)
        out[inside] = ~d[r[inside], c[inside]]
        return out
    return land


# ---------------------------------------------------------------- national figures (World Bank)

def worldbank(iso2):
    """(jobs per resident, share aged 0-14) from the World Bank's ILO employment-to-population
    ratio (15+) and population ages 0-14, the latest year with both."""
    path = raw(f"worldbank_{iso2}.json")
    if not os.path.exists(path):
        r = requests.get(WORLDBANK.format(iso2=iso2), headers=b.UA, timeout=60)
        r.raise_for_status()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(r.json(), f)
    with open(path) as f:
        rows = json.load(f)[1]
    by = {}
    for x in rows:
        if x["value"] is not None:
            by.setdefault(x["date"], {})[x["indicator"]["id"]] = x["value"] / 100.0
    year = max(y for y, v in by.items() if len(v) == 2)
    emp, kids = by[year]["SL.EMP.TOTL.SP.ZS"], by[year]["SP.POP.0014.TO.ZS"]
    return emp * (1 - kids), kids, int(year)


WB_TERTIARY = "https://api.worldbank.org/v2/country/{iso2}/indicator/{ind}?format=json&date=2010:2025&per_page=100"


def tertiary_share(iso2):
    """Tertiary students a resident, nationally: UNESCO's enrolment in tertiary education, all
    programmes (World Bank SE.TER.ENRL) over the population (SP.POP.TOTL) of the latest year with
    both; (share, year), or (None, None) where UNESCO has no figure (T-288)."""
    by = {}
    for ind in ("SE.TER.ENRL", "SP.POP.TOTL"):
        path = raw(f"worldbank_{iso2}_{ind}.json")
        if not os.path.exists(path):
            r = requests.get(WB_TERTIARY.format(iso2=iso2, ind=ind), headers=b.UA, timeout=60)
            r.raise_for_status()
            with open(path, "w") as f:
                json.dump(r.json(), f)
        with open(path) as f:
            for x in json.load(f)[1] or []:
                if x["value"] is not None:
                    by.setdefault(x["date"], {})[ind] = x["value"]
    years = [y for y, v in by.items() if len(v) == 2]
    if not years:
        return None, None
    y = max(years)
    return by[y]["SE.TER.ENRL"] / by[y]["SP.POP.TOTL"], int(y)


# ---------------------------------------------------------------- boundary and water

def build_boundary(city, cfg):
    if callable(cfg["boundary_file"]):  # T-288: a country module writes it from admin units
        cfg["boundary_file"] = cfg["boundary_file"]()
    with open(cfg["boundary_file"]) as f:
        fc = json.load(f)
    feats = fc["features"] if "features" in fc else [fc]
    merged = union_all([make_valid(shape(f["geometry"])) for f in feats])

    def rnd(c):
        if isinstance(c, (list, tuple)) and c and isinstance(c[0], (int, float)):
            return [round(c[0], 6), round(c[1], 6)]
        return [rnd(x) for x in c]

    geom = mapping(merged.simplify(0.0003, preserve_topology=True))
    out = {"type": "FeatureCollection", "features": [{
        "type": "Feature", "properties": {"city": city, "source": f"tier 0: {os.path.basename(cfg['boundary_file'])}"},
        "geometry": {"type": geom["type"], "coordinates": rnd(geom["coordinates"])}}]}
    os.makedirs(b.PACKS, exist_ok=True)
    path = os.path.join(b.PACKS, f"{city}.boundary.geojson")
    with open(path + ".part", "w") as f:
        json.dump(out, f, separators=(",", ":"))
    os.replace(path + ".part", path)
    return merged


def build_water(cfg, boundary):
    import water
    if "water_source" not in cfg:
        raise ValueError("tier 0 needs a water_source, such as osm_water.source([<Geofabrik extract>]) (T-041)")
    lon0, lat0 = cfg["origin"]
    wc = cfg["water"]
    mask, info = water.build_mask(lon0, lat0, boundary.bounds, wc["cell_m"], b.fetch, None, source=cfg["water_source"])
    if wc["min_width_px"] > 1:
        mask = mask.without_narrow(wc["min_width_px"])
    info["min_width_m"] = wc["min_width_px"] * wc["cell_m"]
    print(f"water: {mask.W} x {mask.H} pixels, water {mask.water_pixels() / (mask.W * mask.H):.1%} of the extent", flush=True)
    return mask, info


# ---------------------------------------------------------------- the cells

def inside(cells, geom):
    ll = np.array([h3.cell_to_latlng(h3.int_to_str(int(c))) for c in cells])
    shapely.prepare(geom)
    return shapely.contains_xy(geom, ll[:, 1], ll[:, 0]), ll


def signals(city, cfg, boundary, mask):
    """Per cell (inside the boundary): GHSL population (scaled to admin totals), GHSL
    non-residential volume and surface, OSM job places, errand and leisure places. Cached."""
    cache = os.path.join(b.WORK, f"{city}.tier0_signals.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    t0 = time.time()
    epoch = cfg.get("epoch", 2020)
    land = land_function(mask)
    grids = {}
    for kind in ("pop", "nres_v", "nres_s"):
        arr, X0, Y0 = ghsl_window(kind, boundary.bounds, epoch)
        grids[kind] = grid_to_cells(arr, X0, Y0, cfg["origin"], land if kind == "pop" else None)
        print(f"  GHSL {kind}: {arr.shape[1]} x {arr.shape[0]} pixels, {arr.sum():,.0f} in the box, {len(grids[kind]):,} cells; "
              f"{time.time() - t0:.0f} s", flush=True)
    import canada
    places = canada.osm_places(city, boundary, cfg["extract"], cache=raw(f"osm_jobs_{city}.csv.gz"))
    pids = np.fromiter((h3.str_to_int(h3.latlng_to_cell(la, lo, 9)) for la, lo in zip(places.lat, places.lon)), np.uint64, len(places))
    cells = np.unique(np.concatenate([np.fromiter(g.keys(), np.uint64) for g in grids.values()] + [pids]))
    ok, ll = inside(cells, boundary)
    cells, ll = cells[ok], ll[ok]
    index = {int(c): i for i, c in enumerate(cells)}
    out = {"h3": cells, "lat": ll[:, 0], "lon": ll[:, 1]}
    for kind, g in grids.items():
        out[kind] = np.array([g.get(int(c), 0.0) for c in cells])
    pk = places.kind.to_numpy()
    for kind in ("errand", "leisure", "job", "land"):
        out["osm_" + kind] = np.zeros(len(cells))
        sel = pk == kind
        rows = np.array([index.get(int(c), -1) for c in pids[sel]], np.int64)
        w = places.w.to_numpy()[sel]
        np.add.at(out["osm_" + kind], rows[rows >= 0], w[rows >= 0])
    out["pop_ghsl"] = out["pop"].copy()
    if cfg.get("admin_totals"):
        units = cfg["admin_totals"](boundary)
        pts = shapely.points(out["lon"], out["lat"])
        for name, geom, total in units:
            shapely.prepare(geom)
            sel = shapely.contains(geom, pts)
            have = out["pop"][sel].sum()
            if have > 0:
                out["pop"][sel] *= total / have
            print(f"  admin total {name}: GHSL {have:,.0f} -> {total:,.0f} ({total / max(have, 1) - 1:+.1%})", flush=True)
    os.makedirs(b.WORK, exist_ok=True)
    np.savez_compressed(cache, **out)
    print(f"signals: {len(cells):,} cells, {time.time() - t0:.0f} s", flush=True)
    return out


def nres_signal(sig, height_pow=HEIGHT_POW):
    """GHSL non-residential volume weighted by the cell's mean non-residential height (volume over
    surface) to the power height_pow: an office tower holds far more jobs per cubic metre than a
    warehouse of the same volume."""
    v, s = sig["nres_v"], sig["nres_s"]
    h = np.where(s > 0, v / np.maximum(s, 1e-9), 0.0)
    return v * np.power(np.maximum(h, 1e-9) / 10.0, height_pow)


def jobs_from(sig, total, blend=None, height_pow=HEIGHT_POW):
    """The jobs proxy: total jobs over cells by the blend of the three normalised signals."""
    blend = blend or JOB_BLEND
    osm = sig["osm_errand"] + sig["osm_leisure"] + sig["osm_job"] + sig["osm_land"]
    parts = {"nres": nres_signal(sig, height_pow), "osm": osm, "pop": sig["pop"]}
    return total * sum(blend[k] * parts[k] / max(parts[k].sum(), 1e-9) for k in blend)


# ---------------------------------------------------------------- the curve without flows

def zones_of(x_m, y_m):
    gx, gy = np.floor(x_m / ZONE_M).astype(np.int64), np.floor(y_m / ZONE_M).astype(np.int64)
    key = gx * 100000 + gy
    u, zone = np.unique(key, return_inverse=True)
    return zone, (u // 100000).astype(np.float64), (u % 100000).astype(np.float64)


def gravity(home, work, d, pw, km, iters=300, tol=1e-4):
    """Doubly constrained gravity in numpy (Furness), the game's form: f(d) = d^-pw exp(-d/km)."""
    F = (np.exp(-d / np.float32(km)) * d ** np.float32(-pw)).astype(np.float32)
    o = home * work.sum() / home.sum()
    col = work.astype(np.float32)
    has = o > 0
    row = np.ones(len(o), np.float32)
    for it in range(iters):
        s = (F @ col).astype(np.float64)
        if it > 0 and np.max(np.abs(row[has] * s[has] / o[has] - 1)) < tol:
            break
        row = np.where(has, o / np.maximum(s, 1e-30), 0.0).astype(np.float32)
        t = (row @ F).astype(np.float64)
        col = np.where(work > 0, work / np.maximum(t, 1e-30), 0.0).astype(np.float32)
    return F * row[:, None] * col[None, :]


def zone_distances(gx, gy):
    d = ZONE_M / 1000.0 * np.sqrt((gx[:, None] - gx[None, :]) ** 2 + (gy[:, None] - gy[None, :]) ** 2).astype(np.float32)
    d[d == 0] = INTRAZONAL_KM
    return d


def trip_stats(T, d):
    w = T.ravel().astype(np.float64)
    dd = d.ravel()
    order = np.argsort(dd)
    c = np.cumsum(w[order])
    return float((w * dd).sum() / w.sum()), float(dd[order][np.searchsorted(c, c[-1] / 2)])


def fit_curve(x_m, y_m, home, work, target, pw=DEFAULT_CURVE_POW):
    """The curve's length (km) at power pw so the model's median (or mean) commute between 2 km
    zones matches a published figure, by bisection on log km. target: {"median_km": x} or
    {"mean_km": x}; straight-line distances, so a road-distance figure should be divided by ~1.3."""
    zone, gx, gy = zones_of(x_m, y_m)
    H = np.bincount(zone, weights=home)
    J = np.bincount(zone, weights=work)
    d = zone_distances(gx, gy)
    key, want = ("median_km", target["median_km"]) if "median_km" in target else ("mean_km", target["mean_km"])
    lo, hi = math.log(2.0), math.log(300.0)
    got = None
    for _ in range(18):
        km = math.exp((lo + hi) / 2)
        mean, med = trip_stats(gravity(H, J, d, pw, km), d)
        got = med if key == "median_km" else mean
        if got < want:
            lo = math.log(km)
        else:
            hi = math.log(km)
    km = math.exp((lo + hi) / 2)
    mean, med = trip_stats(gravity(H, J, d, pw, km), d)
    print(f"curve fitted to the published {key} {want} km: d^-{pw:g} exp(-d/{km:.1f} km), model median {med:.1f} km, "
          f"mean {mean:.1f} km over {len(H):,} zones", flush=True)
    return pw, round(km, 1)


# ---------------------------------------------------------------- build

def main(city):
    t0 = time.time()
    cfg = b.CITIES[city]
    for d in (b.WORK, b.PACKS, raw()):
        os.makedirs(d, exist_ok=True)
    boundary = build_boundary(city, cfg)
    mask, mask_info = build_water(cfg, boundary)
    sig = signals(city, cfg, boundary, mask)
    jpr, kids, wb_year = worldbank(cfg["iso2"])
    P = float(sig["pop"].sum())
    J = P * jpr
    wfh = cfg.get("wfh", 0.0)
    jobs = jobs_from(sig, J)
    pop = sig["pop"]
    work_c = jobs * (1 - wfh)
    home_c = pop * (work_c.sum() / P)
    errand = sig["osm_errand"] * (ERRAND_SHARE * J / max(sig["osm_errand"].sum(), 1.0))
    leisure = sig["osm_leisure"] * (LEISURE_SHARE * J / max(sig["osm_leisure"].sum(), 1.0))
    keep = (pop > 0) | (jobs > 0)
    cells = sig["h3"][keep]
    lon0, lat0 = cfg["origin"]
    x = b.R_EARTH * np.radians(sig["lon"][keep] - lon0) * np.cos(np.radians(lat0))
    y = b.R_EARTH * np.radians(sig["lat"][keep] - lat0)
    print(f"tier 0: {len(cells):,} cells, pop {P:,.0f}, jobs {J:,.0f} ({jpr:.3f} a resident, World Bank {wb_year}), "
          f"commuters {work_c.sum():,.0f} (work from home {wfh:.1%})", flush=True)
    if "decay_km" not in cfg:
        cfg["decay_pow"], cfg["decay_km"] = fit_curve(x, y, home_c[keep], work_c[keep], cfg["curve"], cfg["curve"].get("pow", DEFAULT_CURVE_POW))
    cfg.setdefault("border_factor", 1.0)
    totals = {"pop": int(round(P)), "jobs": int(round(jobs.sum())),
              "commute_home": int(round(home_c.sum())), "commute_work": int(round(work_c.sum()))}
    blend = ", ".join(f"{k} {v:.0%}" for k, v in JOB_BLEND.items())
    sources = [
        {"what": "population", "name": f"GHSL GHS-POP E{cfg.get('epoch', 2020)} R2023A 100 m" + (", scaled to admin totals" if cfg.get("admin_totals") else ""),
         "url": "https://human-settlement.emergency.copernicus.eu/ghs_pop2023.php"},
        {"what": "jobs", "name": f"tier 0 proxy: population x {jpr:.3f} jobs a resident (World Bank SL.EMP.TOTL.SP.ZS x 15+ share, {wb_year}), "
                                 f"spread by {blend} (GHSL GHS-BUILT-V NRES x (height / 10 m)^{HEIGHT_POW}, OSM job places, population)",
         "url": "https://human-settlement.emergency.copernicus.eu/ghs_buC2023.php"},
        {"what": "everyday destinations", "name": "OpenStreetMap shops and services (errands), restaurants, bars, venues and hotels (leisure), per cell",
         "url": "https://www.openstreetmap.org/copyright"},
        {"what": "boundary", "name": f"tier 0 boundary {os.path.basename(cfg['boundary_file'])}", "url": ""},
        {"what": "water", "name": mask_info.get("source", ""), "url": ""},
    ]
    commute = {"home": "population x commuters per resident (no source of where workers live)",
               "work": f"tier 0 jobs x (1 - {wfh:.3f} working at home)", "region_share": wfh}
    region_block = {"level": "admin-1", "ids": ["00"], "names": [cfg.get("region_name", city.upper())],
                    "source": "tier 0: one region for the whole pack, no border factor"}
    cells_json = b.write_cells(city, cfg, {"h3": cells, "x_m": x, "y_m": y, "pop": pop[keep], "jobs": jobs[keep],
                                           "errand_jobs": errand[keep], "leisure_jobs": leisure[keep],
                                           "commute_home": home_c[keep], "commute_work": work_c[keep],
                                           "region": np.zeros(int(keep.sum()), np.uint8)},
                               totals, sources, mask, mask_info, commute, region_block)
    print(f"cells written in {time.time() - t0:.0f} s", flush=True)
    finish(city, kids)
    print(f"done in {time.time() - t0:.0f} s", flush=True)


def finish(city, kids=None):
    """From data/work/<city>.cells.* on: gravity, everyday header, schools, universities,
    airports, medical, parks."""
    cfg = b.CITIES[city]
    if kids is None:
        kids = worldbank(cfg["iso2"])[1]
    cfg.setdefault("border_factor", 1.0)
    b.add_gravity(os.path.join(b.WORK, f"{city}.cells.json"), city)
    import canada
    stem, header, buf, cells = canada.pack_cells(city)
    arr = {a["name"]: a for a in header["arrays"]}
    tot = {n: float(np.frombuffer(buf, "<f4", arr[n]["count"], arr[n]["offset"]).sum(dtype=np.float64)) for n in ("errand_jobs", "leisure_jobs")}
    header["everyday"] = {"enabled": True, "source": "OpenStreetMap shops and services (errands); restaurants, bars, venues, hotels "
                          f"(leisure); per cell, scaled to {ERRAND_SHARE:.0%} and {LEISURE_SHARE:.0%} of jobs (tier 0)", **tot}
    canada.rewrite(stem, header, bytes(buf))
    boundary = build_boundary(city, cfg)
    education_osm(city, boundary, kids)
    # the Overpass-shaped caches (terminals, medical, parks) from the extract, before the readers
    els, names = canada.osm_caches(city, boundary, cfg["extract"])
    airports(city, boundary)
    medical_parks(city, els, names)


# ---------------------------------------------------------------- schools and universities

def osm_education(city, boundary, extract):
    """OSM schools, universities and colleges: (kind, name, lat, lon, site m2 or 0, wikidata)."""
    cache = raw(f"osm_education_{city}.csv.gz")
    if os.path.exists(cache):
        return pd.read_csv(cache, keep_default_na=False)
    import pyogrio
    import overpass
    import routes
    path = routes.download(extract)
    t0 = time.time()
    bb = boundary.bounds
    kinds = ("school", "university", "college")
    mp = pyogrio.read_dataframe(path, layer="multipolygons", columns=["name", "amenity", "other_tags"], bbox=bb,
                                where="amenity IN ('school','university','college')")
    pts = pyogrio.read_dataframe(path, layer="points", columns=["name", "other_tags"], bbox=bb,
                                 where=" OR ".join(f"other_tags LIKE '%\"amenity\"=>\"{k}\"%'" for k in kinds))
    import routes as rt
    rows = []
    polys = {k: [] for k in kinds}
    for r in mp.itertuples():
        c = r.geometry.representative_point()
        rows.append((r.amenity, r.name or "", c.y, c.x, overpass.area_m2(r.geometry), rt.tag(r.other_tags, "wikidata")))
        polys[r.amenity].append(r.geometry)
    trees = {k: shapely.STRtree(v) for k, v in polys.items() if v}
    n_dup = 0
    for r in pts.itertuples():
        k = rt.tag(r.other_tags, "amenity")
        if k in trees and len(trees[k].query(r.geometry, predicate="within")):
            n_dup += 1  # a point inside a mapped site of the same kind: the site counts
            continue
        rows.append((k, r.name or "", r.geometry.y, r.geometry.x, 0.0, rt.tag(r.other_tags, "wikidata")))
    out = pd.DataFrame(rows, columns=["kind", "name", "lat", "lon", "m2", "wikidata"])
    out.to_csv(cache, index=False)
    print(f"  OSM education: {len(mp):,} sites, {len(pts):,} points ({n_dup} inside a site); {time.time() - t0:.0f} s", flush=True)
    return out


def wikidata_students(qids):
    """{QID: (students, year)} from Wikidata P2196 (latest point in time)."""
    path = raw("wikidata_students.json")
    have = {}
    if os.path.exists(path):
        with open(path) as f:
            have = json.load(f)
    need = sorted(q for q in set(qids) if q and q not in have)
    for i in range(0, len(need), 200):
        part = need[i:i + 200]
        q = ("SELECT ?item ?v ?t WHERE { VALUES ?item { " + " ".join(f"wd:{x}" for x in part) + " } "
             "?item p:P2196 ?s. ?s ps:P2196 ?v. OPTIONAL { ?s pq:P585 ?t } }")
        r = requests.post(WIKIDATA, data={"query": q, "format": "json"}, headers=b.UA, timeout=120)
        r.raise_for_status()
        best = {}
        for row in r.json()["results"]["bindings"]:
            qid = row["item"]["value"].rsplit("/", 1)[1]
            t = row.get("t", {}).get("value", "")
            v = float(row["v"]["value"])
            if qid not in best or t > best[qid][1]:
                best[qid] = (v, t)
        for qid in part:
            have[qid] = best.get(qid, (None, ""))
    with open(path, "w") as f:
        json.dump(have, f)
    return {q: tuple(have[q]) for q in qids if q in have}


def education_osm(city, boundary, kids):
    import canada
    cfg = b.CITIES[city]
    ed = osm_education(city, boundary, cfg["extract"])
    stem, header, buf, cells = canada.pack_cells(city)
    arr = {a["name"]: a for a in header["arrays"]}
    pop = np.frombuffer(buf, "<f4", arr["pop"]["count"], arr["pop"]["offset"]).astype(np.float64)
    index = {int(c): i for i, c in enumerate(cells)}

    def place(rows, values):
        out = np.zeros(len(cells), "<f4")
        import enrich_airports as air
        n = 0
        for r, v in zip(rows.itertuples(), values):
            c, _ = air.cell_of(index, r.lat, r.lon, 500.0)
            if c is not None:
                out[c] += v
                n += 1
        return out, n

    # schools: pupils from the national child share, spread over sites by sqrt(area)
    shapely.prepare(boundary)
    inb = shapely.contains_xy(boundary, ed.lon.to_numpy(float), ed.lat.to_numpy(float))
    sch = ed[(ed.kind == "school").to_numpy() & inb]
    sizes = np.sqrt(sch.m2.to_numpy(float))
    med = np.median(sizes[sizes > 0]) if (sizes > 0).any() else 1.0
    sizes = np.where(sizes > 0, sizes, med)
    children = pop * kids
    pupils = children.sum() * SCHOOL_ENROL
    enr, n_s = place(sch, pupils * sizes / sizes.sum())
    # universities and colleges: Wikidata students by institution, split over its sites by area
    uni = ed[ed.kind.isin(["university", "college"]).to_numpy() & inb].copy()
    wd = wikidata_students(uni.wikidata[uni.wikidata != ""].tolist())
    uni["students"] = [wd.get(q, (None,))[0] if q else None for q in uni.wikidata]
    uni["students"] = pd.to_numeric(uni.students, errors="coerce")
    known = uni[uni.students.notna()]
    site_m2 = known.groupby("wikidata").m2.transform("sum")
    share = np.where(site_m2 > 0, known.m2 / site_m2.where(site_m2 > 0, 1), 1.0 / known.groupby("wikidata").m2.transform("count"))
    known_students = known.students.to_numpy() * share
    dens = known.groupby("wikidata").apply(lambda g: g.students.iloc[0] / max(g.m2.sum(), 1.0))
    per_m2 = float(np.median(dens[dens < 1].values)) if len(dens) else 0.0  # students a square metre of campus
    unknown = uni[uni.students.isna() & (uni.m2 > 0)]
    unk_students = unknown.m2.to_numpy() * per_m2
    rows = pd.concat([known, unknown])
    # T-288: the total from the national tertiary enrolment a resident; Wikidata counts and site
    # areas only spread it. Unscaled, London's counted institutions were dense central buildings
    # (1,925 students a hectare at the median), which put 50-150k students on suburban campuses.
    site_students = np.concatenate([known_students, unk_students])
    ter, ter_year = tertiary_share(cfg["iso2"])
    raw_total = float(site_students.sum())
    if ter and raw_total > 0:
        site_students = site_students * (pop.sum() * ter / raw_total)
    uenr, n_u = place(rows, site_students)
    adults = pop * (1 - kids)
    buf = education.replace_cell_arrays(header, buf, {"school_children": children.astype("<f4"), "school_enrollment": enr,
                                                      "university_adults": adults.astype("<f4"), "university_enrollment": uenr})
    header["school"] = {
        "coverage": "tier 0: OSM amenity=school sites in the boundary (public and private), pupils = population x national share aged 0-14 "
                    f"x {SCHOOL_ENROL}, spread by the square root of site area",
        "enrolled": float(enr.sum(dtype=np.float64)), "matched_buildings": int(n_s), "location_match_share": float(enr.sum() / max(pupils, 1)),
        "child_weight": float(children.sum()), "child_origin": f"population x {kids:.3f} (World Bank, ages 0-14)",
        "attendance_factor": 0.9, "sources": ["https://www.openstreetmap.org/copyright", WORLDBANK.format(iso2=cfg["iso2"])]}
    header["university"] = dict(
        coverage="tier 0: OSM amenity=university|college; Wikidata P2196 students by institution split over its sites by area, other sites "
                 f"at the median {per_m2 * 1e4:.0f} students a hectare"
                 + (f"; scaled to population x {ter:.4f} tertiary students a resident (UNESCO via World Bank SE.TER.ENRL, {ter_year})" if ter else ""),
        enrolled=float(uenr.sum(dtype=np.float64)), institutions=int(known.wikidata.nunique()), sites=int(n_u),
        location_match_share=1.0, adult_weight=float(adults.sum()), adult_origin=f"population x {1 - kids:.3f} (ages 15+, World Bank)",
        attendance_factor=0.6, decay_m=6000, limitations="Wikidata counts are of mixed years and definitions; sites without a count sized by area",
        sources=["https://www.openstreetmap.org/copyright", WIKIDATA, WB_TERTIARY.format(iso2=cfg["iso2"], ind="SE.TER.ENRL")])
    canada.rewrite(stem, header, buf)
    with open(os.path.join(b.WORK, city + ".universities.audit.json"), "w") as f:
        json.dump([dict(name=r.name, wikidata=r.wikidata, m2=r.m2, students=s) for r, s in
                   zip(rows.itertuples(), site_students)], f, indent=1)
    print(f"schools: {len(sch):,} OSM sites, {enr.sum():,.0f} pupils of {pupils:,.0f}; universities: {known.wikidata.nunique()} with Wikidata "
          f"counts ({len(known)} sites), {len(unknown)} sites by area at {per_m2 * 1e4:.0f} a hectare, {raw_total:,.0f} students before "
          f"scaling, {uenr.sum():,.0f} after" + (f" ({ter:.4f} a resident, {ter_year})" if ter else " (no national figure, unscaled)"), flush=True)


# ---------------------------------------------------------------- airports, medical, parks

def airports(city, boundary):
    import canada
    import enrich_airports as air
    cfg = b.CITIES[city]
    aps = air.airports_in(boundary)
    found = air.terminals(city, boundary, aps) if aps else {}
    # pack share: GHSL population around the airport within 300 km at exp(-d/20 km); in the
    # boundary or not. A 60 km pad is enough (exp(-3) beyond it).
    arr, X0, Y0 = ghsl_window("pop", boundary.bounds, cfg.get("epoch", 2020), pad_m=60000.0)
    f = 10
    h, w = arr.shape[0] // f, arr.shape[1] // f
    blk = arr[:h * f, :w * f].reshape(h, f, w, f).sum(axis=(1, 3))
    rr, cc = np.nonzero(blk > 0)
    _, inv = _moll()
    lon, lat = inv.transform(X0 + (cc + 0.5) * f * PIX_M, Y0 - (rr + 0.5) * f * PIX_M)
    lon, lat, pp = np.asarray(lon), np.asarray(lat), blk[rr, cc]
    shapely.prepare(boundary)
    ins = shapely.contains_xy(boundary, lon, lat)
    stem, header, buf, cells = canada.pack_cells(city)
    index = {int(c): i for i, c in enumerate(cells)}
    trips = np.zeros(len(cells), "<f4")
    audit = []
    for ap in aps:
        boardings, when = air.wikidata_boardings(ap["iata"]) if ap["iata"] else (None, None)
        if not boardings:
            audit.append(ap | dict(skipped="no Wikidata patronage"))
            continue
        d = air.metres(ap["lat"], ap["lon"], lat, lon)
        wgt = pp * np.exp(-d / air.DECAY_M)
        share = float(wgt[ins].sum() / wgt.sum())
        od = cfg.get("od_share", {}).get(ap["ident"], 1.0)
        daily = boardings * od * share / air.DAYS
        sites = found.get(ap["ident"]) or [(ap["lat"], ap["lon"], 1.0, "airport point", None)]
        m2 = sum(s[2] for s in sites)
        for la, lo, a, *_ in sites:
            c, _ = air.cell_of(index, la, lo, air.SNAP_M)
            if c is not None:
                trips[c] += daily * a / m2
        audit.append(ap | dict(boardings=boardings, when=when, od_share=od, pack_share=share, daily_outings=daily))
        print(f"  {ap['ident']} {ap['name']}: {boardings:,.0f} boardings (Wikidata {when}), O&D {od:.2f}, pack {share:.2f}, "
              f"{daily:,.0f} outings a day", flush=True)
    buf = education.replace_cell_arrays(header, buf, {"airport_trips": trips})
    header["airport"] = dict(
        coverage="tier 0: OurAirports scheduled-service airports in the boundary", year=None,
        airports=sum(1 for r in audit if "daily_outings" in r), outings=float(trips.sum(dtype=np.float64)), decay_m=air.DECAY_M,
        origin_mix="half residents, half visitors (jobs and leisure/hotel jobs, a quarter each)",
        formula="daily outings = Wikidata patronage / 2 x O&D share (1, or judgement for hubs) x pack share (GHSL population x exp(-d/20 km)) / 366",
        limitations="Wikidata patronage of mixed years; no connecting-share source", sources=[air.OURAIRPORTS, WIKIDATA])
    canada.rewrite(stem, header, buf)
    with open(os.path.join(b.WORK, city + ".airports.audit.json"), "w") as f:
        json.dump(audit, f, indent=1, default=str)


def medical_parks(city, els, names):
    import enrich_medical
    import enrich_parks
    enrich_medical.main(city)
    keep = enrich_parks.fetch_all
    enrich_parks.fetch_all = lambda c, bd: (els, names)
    try:
        enrich_parks.main(city)
    finally:
        enrich_parks.fetch_all = keep


if __name__ == "__main__":
    import sys
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    (finish if "--finish" in sys.argv else main)(args[0])
