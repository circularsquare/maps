"""T-536, pipeline stage 4 (notes/T-535.md): the world plan. Which metros get a pack, in what order,
and which whole admin units each one holds.

    python pipeline/world_plan.py           plan, write pipeline/world_plan.json and the extents
    python pipeline/world_plan.py check     the same, then the eight-pack test and the next-50 picture
    python pipeline/world_plan.py row <pack> <out.json>
                                            one existing pack's metro planned as if the pack did not
                                            exist, as a one-row plan file (T-537's test build)
    python pipeline/world_plan.py zones --countries ES[,PT]
                                            whole-country cover (T-723): the country's zone rows,
                                            every unit no row or pack holds (docs/zones.md 2)

The rules are docs/adding-a-city.md "Which cities, and where their edges go". Downloads land in
data/raw/worldplan/ when missing; extents as GeoJSON in data/work/worldplan/. Everything is computed
in Mollweide metres on GHS-POP's 1 km grid (2025 epoch), a pixel counted where its centre falls.
"""
import collections
import json
import math
import os
import sys
import unicodedata
import zipfile
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pyogrio  # noqa: E402
import requests  # noqa: E402
import shapely  # noqa: E402
from pyproj import Transformer  # noqa: E402
from shapely import STRtree, contains_xy, make_valid, union_all  # noqa: E402
from shapely.geometry import mapping, shape  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/worldplan"
WORK = ROOT / "data/work/worldplan"
PACKS = ROOT / "data/packs"
OUT = ROOT / "pipeline/world_plan.json"
ARTIFACTS = ROOT.parent / "artifacts/trainworld"
UCDB = ROOT.parent / "citybrowser/data/GHS_UCDB_THEME_GENERAL_CHARACTERISTICS_GLOBE_R2024A.gpkg"
ITL3 = ROOT / "data/raw/tier0/itl3_jan2025_uk_bgc.geojson"
NE = ROOT / "data/raw/world_field/ne_10m_admin_0_countries.zip"
REGISTRY = ROOT / "app/src/game/registry.json"
UA = {"User-Agent": "trainworld-pipeline/0.1 (rail game data build; python-requests)"}

GHSL = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL"
POP = ("GHS_POP_E2025_GLOBE_R2023A_54009_1000_V1_0",
       f"{GHSL}/GHS_POP_GLOBE_R2023A/GHS_POP_E2025_GLOBE_R2023A_54009_1000/V1-0/GHS_POP_E2025_GLOBE_R2023A_54009_1000_V1_0.zip")
FUA = ("GHS_FUA_UCDB2015_GLOBE_R2019A_54009_1K_V1_0",
       f"{GHSL}/GHS_FUA_UCDB2015_GLOBE_R2019A/V1-0/GHS_FUA_UCDB2015_GLOBE_R2019A_54009_1K_V1_0.zip")
NUTS = ("NUTS_RG_03M_2024_4326_LEVL_3.geojson",
        "https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/NUTS_RG_03M_2024_4326_LEVL_3.geojson")
GEOFABRIK = ("geofabrik-index-v1.json", "https://download.geofabrik.de/index-v1.json")
GB_META = ("geoboundaries-gbOpen-ALL.json", "https://www.geoboundaries.org/api/current/gbOpen/ALL/ALL/")

MIN_POP = 1_000_000      # a metro's people (decision 1)
CAP = 40_000_000         # people a pack (decision 4)
UNIT_SHARE = 0.5         # a unit is in when this share of its people live inside the area (decision 3)
FUA_SHARE = 0.5          # a FUA joins an urban centre holding this share of its people
SAT_GAP = 3000           # m: a smaller urban centre this close to a metro's centres joins it (T-556)
MEDIAN_UNIT = 20_000     # the finest level where the median person's unit has this many people
COUNTRY_SHARE = 0.02     # a country lends units to a metro holding this share of its area's people
TESTS = ["tor", "lon", "par", "ber", "seo", "mum", "lag", "sao"]
# whole-country zones (T-723, docs/zones.md 2)
ZONE_MIN_POP = 100_000   # a zone under this joins a neighbouring zone of its region
ZONE_MIN_DENSITY = 20.0  # people a km2: an emptier unit stays dead space
ISLAND_KM = 150.0        # an island under ZONE_MIN_POP joins the most populous zone of its region this near
ZONE_FLOOR = 500_000     # zones grow to this within their region (Anita, 2026-10-20, T-788)
ZONE_CAP = 1_500_000     # a zone of ZONE_MIN_POP or more prefers neighbours keeping it within this
# zone rows planned before the floor (T-723 to T-786) used ZONE_MIN_POP as theirs; a replan of those
# countries would regroup their unbuilt rows only
EUROPE_BOX = (-32.0, 27.0, 45.0, 72.0)  # lon/lat: a European country's units outside it (France's overseas départements) are not zoned (T-779)

# What a city in each country can have (docs/data-register.md "Countries and their procedure");
# every other country is archetype D. England's module covers England's ITL3 units only.
ARCHETYPE = {"GBR": ("A", "england"), "FRA": ("A", "france"), "DEU": ("A", "germany"),
             "CAN": ("A", "canada"), "JPN": ("A", "japan"), "BRA": ("C", "brazil"),
             "CHN": ("C", "china"), "KOR": ("C", "korea"), "IND": ("C", "india"),
             "NGA": ("D", "nigeria")}
NE_FIX = {"KOS": "XKX", "SOL": "SOM", "CYN": "CYP", "SAH": "ESH"}

FWD = Transformer.from_crs("EPSG:4326", "ESRI:54009", always_xy=True)
INV = Transformer.from_crs("ESRI:54009", "EPSG:4326", always_xy=True)


def log(*a):
    print(*a, flush=True)


def fetch(name, url, sub=""):
    path = RAW / sub / name
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(4):
        try:
            r = requests.get(url, headers=UA, timeout=600)
            r.raise_for_status()
            break
        except requests.RequestException as e:
            if attempt == 3:
                raise
            log(f"  retry {url}: {e}")
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(r.content)
    os.replace(tmp, path)
    return path


def unzip(stem, url, ext):
    path = RAW / f"{stem}{ext}"
    if not path.exists():
        z = fetch(f"{stem}.zip", url)
        with zipfile.ZipFile(z) as f:
            f.extract(path.name, RAW)
    return path


def to_moll(g):
    return shapely.transform(g, lambda a: np.column_stack(FWD.transform(a[:, 0], a[:, 1])))


def to_ll(g):
    return shapely.transform(g, lambda a: np.column_stack(INV.transform(a[:, 0], a[:, 1])))


def polygonal(g):
    g = make_valid(g)
    parts = [p for p in shapely.get_parts(g) if p.geom_type in ("Polygon", "MultiPolygon")]
    return union_all(parts) if parts else shapely.Polygon()


def ascii_name(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s) if ord(c) < 128)


# ---------------------------------------------------------------- the 1 km population grid

class Pop:
    """GHS-POP 2025 at 1 km, read by TIFF tile (the whole grid is 2.6 GB as floats)."""
    X0, Y0, PIX = -18041000.0, 9000000.0, 1000.0

    def __init__(self):
        import tifffile
        self.tf = tifffile.TiffFile(unzip(POP[0], POP[1], ".tif"))
        self.pg = self.tf.pages[0]
        self.th, self.tw = self.pg.tilelength, self.pg.tilewidth
        self.H, self.W = self.pg.imagelength, self.pg.imagewidth
        self.nx = math.ceil(self.W / self.tw)
        self.cache = collections.OrderedDict()

    def tile(self, ty, tx):
        key = (ty, tx)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        i = ty * self.nx + tx
        fh = self.tf.filehandle
        fh.seek(self.pg.dataoffsets[i])
        seg = np.asarray(self.pg.decode(fh.read(self.pg.databytecounts[i]), i)[0])
        a = seg.reshape(seg.shape[-3], seg.shape[-2]).astype(np.float32)
        a[a < 0] = 0  # nodata -200
        self.cache[key] = a
        if len(self.cache) > 600:
            self.cache.popitem(last=False)
        return a

    def pixels(self, bounds):
        """Centres (x, y) and people of the populated pixels in a Mollweide box."""
        xmin, ymin, xmax, ymax = bounds
        c0 = max(int(math.floor((xmin - self.X0) / self.PIX)), 0)
        c1 = min(int(math.ceil((xmax - self.X0) / self.PIX)), self.W)
        r0 = max(int(math.floor((self.Y0 - ymax) / self.PIX)), 0)
        r1 = min(int(math.ceil((self.Y0 - ymin) / self.PIX)), self.H)
        xs, ys, ps = [], [], []
        for ty in range(r0 // self.th, (r1 - 1) // self.th + 1 if r1 > r0 else 0):
            for tx in range(c0 // self.tw, (c1 - 1) // self.tw + 1 if c1 > c0 else 0):
                a = self.tile(ty, tx)
                ra, rb = max(r0, ty * self.th), min(r1, (ty + 1) * self.th)
                ca, cb = max(c0, tx * self.tw), min(c1, (tx + 1) * self.tw)
                sub = a[ra - ty * self.th:rb - ty * self.th, ca - tx * self.tw:cb - tx * self.tw]
                rr, cc = np.nonzero(sub > 0)
                if len(rr):
                    xs.append(self.X0 + (cc + ca + 0.5) * self.PIX)
                    ys.append(self.Y0 - (rr + ra + 0.5) * self.PIX)
                    ps.append(sub[rr, cc])
        if not xs:
            return np.zeros(0), np.zeros(0), np.zeros(0, np.float32)
        return np.concatenate(xs), np.concatenate(ys), np.concatenate(ps)

    def inside(self, geom):
        x, y, p = self.pixels(geom.bounds)
        shapely.prepare(geom)
        m = contains_xy(geom, x, y)
        return x[m], y[m], p[m]


# ---------------------------------------------------------------- sources

def europe_isos():
    meta = json.loads(fetch(*GB_META).read_text(encoding="utf-8"))
    return {r["boundaryISO"] for r in meta if r["Continent"] == "Europe"}, meta


def countries():
    df = pyogrio.read_dataframe(f"zip://{NE}", columns=["ISO_A3", "ADM0_A3"])
    out = []
    for iso, adm, g in zip(df.ISO_A3, df.ADM0_A3, df.geometry):
        code = iso if iso != "-99" else adm
        out.append((NE_FIX.get(adm, NE_FIX.get(code, code)), polygonal(to_moll(g))))
    return out


def iso_at(ne, tree, pts):
    """The country of each point (nearest country where a point sits just offshore)."""
    pi, ci = tree.query(pts, predicate="within")
    out = [None] * len(pts)
    for a, b in zip(pi, ci):
        out[a] = ne[b][0]
    for a in range(len(pts)):
        if out[a] is None:
            out[a] = ne[tree.nearest(pts[a])][0]
    return out


def packs():
    """The existing packs (every boundary in data/packs), Mollweide, with their names."""
    names = {c["id"]: c["name"] for c in json.loads(REGISTRY.read_text(encoding="utf-8"))["cities"]}
    out = {}
    for f in sorted(PACKS.glob("*.boundary.geojson")):
        pid = f.name.split(".")[0]
        gj = json.loads(f.read_text(encoding="utf-8"))
        g = polygonal(union_all([polygonal(shape(ft["geometry"])) for ft in gj["features"]]))
        out[pid] = {"id": pid, "name": names.get(pid, pid), "geom": polygonal(to_moll(g)), "ll": g}
    return out


# ---------------------------------------------------------------- metros and their areas

def metros(pop, ne, europe):
    """Candidate metros, each with the area its units are judged against: in Europe a functional
    urban area over 1M people; elsewhere an urban centre over 1M with the FUAs it holds half of (and
    the FUA around its population-weighted centre), centres sharing a FUA merged."""
    fpath = unzip(FUA[0], FUA[1], ".gpkg")
    fua = pyogrio.read_dataframe(fpath, columns=["eFUA_ID", "eFUA_name", "Cntry_ISO", "FUA_p_2015"])
    fgeom = [polygonal(g) for g in fua.geometry]
    out = []
    for i in range(len(fua)):
        if fua.Cntry_ISO[i] in europe and fua.FUA_p_2015[i] >= 0.5 * MIN_POP:
            _, _, p = pop.inside(fgeom[i])
            if p.sum() >= MIN_POP:
                out.append({"key": f"fua{int(fua.eFUA_ID[i])}", "name": fua.eFUA_name[i], "iso": fua.Cntry_ISO[i],
                            "region": "uk" if fua.Cntry_ISO[i] == "GBR" else "europe",
                            "source": {"fua": [int(fua.eFUA_ID[i])]}, "alt_names": [], "area": fgeom[i]})
    log(f"Europe: {len(out)} functional urban areas over {MIN_POP:,}")

    lay = "GHS_UCDB_THEME_GENERAL_CHARACTERISTICS_GLOBE_R2024A"
    uc = pyogrio.read_dataframe(UCDB, layer=lay, columns=["ID_UC_G0", "GC_UCN_MAI_2025", "GC_POP_TOT_2025"])
    cen = pyogrio.read_dataframe(UCDB, layer="UC_centroids")
    cen = dict(zip(cen.ID_UC_G0, cen.geometry))
    uc_iso = iso_at(ne, STRtree([g for _, g in ne]), [cen[i] for i in uc.ID_UC_G0])
    big = uc[uc.GC_POP_TOT_2025 >= MIN_POP].reset_index(drop=True)
    pts = [cen[i] for i in big.ID_UC_G0]
    isos = [uc_iso[k] for k in np.nonzero(uc.GC_POP_TOT_2025.to_numpy() >= MIN_POP)[0]]
    ftree = STRtree(fgeom)
    world = []
    for k in range(len(big)):
        if isos[k] in europe or isos[k] == "USA":
            continue
        ug = polygonal(big.geometry[k])
        joined = []
        for i in ftree.query(ug, predicate="intersects"):
            x, y, p = pop.inside(fgeom[i])
            share = p[contains_xy(ug, x, y)].sum() / max(p.sum(), 1)
            if fgeom[i].contains(pts[k]) or share >= FUA_SHARE:
                joined.append(int(i))
        world.append({"uc": int(big.ID_UC_G0[k]), "name": big.GC_UCN_MAI_2025[k], "iso": isos[k],
                      "pop": float(big.GC_POP_TOT_2025[k]), "geom": ug, "fua": joined})
    # merge centres that share a FUA
    parent = list(range(len(world)))

    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    owner = {}
    for k, w in enumerate(world):
        for i in w["fua"]:
            if i in owner:
                parent[root(k)] = root(owner[i])
            else:
                owner[i] = k
    groups = collections.defaultdict(list)
    for k in range(len(world)):
        groups[root(k)].append(world[k])
    for ws in groups.values():
        ws.sort(key=lambda w: -w["pop"])
        fu = sorted({i for w in ws for i in w["fua"]})
        area = polygonal(union_all([w["geom"] for w in ws] + [fgeom[i] for i in fu]))
        out.append({"key": f"uc{ws[0]['uc']}", "name": ws[0]["name"], "iso": ws[0]["iso"], "region": "world",
                    "source": {"ucdb": [w["uc"] for w in ws], "fua": [int(fua.eFUA_ID[i]) for i in fu]},
                    "alt_names": [w["name"] for w in ws[1:]] + [fua.eFUA_name[i] for i in fu], "area": area})
    log(f"elsewhere: {len(big)} urban centres over {MIN_POP:,}, {sum(len(g) for g in groups.values())} outside "
        f"Europe and the US, {len(groups)} metros after merging")
    satellites(out, uc, cen, uc_iso)
    for m in out:
        x, y, p = pop.inside(m["area"])
        m["px"] = (x, y, p)
        m["pop_area"] = float(p.sum())
        i = int(np.argmax(p))
        m["peak"] = shapely.Point(x[i], y[i])
        shapely.prepare(m["area"])
    return out


def satellites(out, uc, cen, uc_iso):
    """A smaller urban centre outside every metro's area, within SAT_GAP of a metro's own centres,
    joins the nearest such metro's area: GHSL gives Ikorodu (0.86M, 2.2 km from Lagos) a FUA of
    its own (T-556)."""
    ids = [int(i) for i in uc.ID_UC_G0]
    geom = [polygonal(g) for g in uc.geometry]
    cpts = [cen[i] for i in uc.ID_UC_G0]
    atree = STRtree([m["area"] for m in out])
    pi, mi = atree.query(cpts, predicate="within")
    inside = set(pi.tolist())
    own = collections.defaultdict(list)
    for a, k in zip(pi, mi):
        if out[k]["region"] != "world" or ids[a] in out[k]["source"]["ucdb"]:
            own[int(k)].append(geom[a])
    centres = [union_all(own[k]) if own[k] else shapely.Polygon() for k in range(len(out))]
    ctree = STRtree(centres)
    add = collections.defaultdict(list)
    for a in range(len(ids)):
        if a in inside or uc.GC_POP_TOT_2025[a] >= MIN_POP or uc_iso[a] == "USA":
            continue
        near = ctree.query(geom[a], predicate="dwithin", distance=SAT_GAP)
        if not len(near):
            continue
        k = int(min(near, key=lambda j: geom[a].distance(centres[j])))
        add[k].append(geom[a])
        out[k]["source"].setdefault("satellites", []).append(ids[a])
        out[k]["alt_names"].append(uc.GC_UCN_MAI_2025[a])
    for k, gs in add.items():
        out[k]["area"] = polygonal(union_all([out[k]["area"]] + gs))
    log(f"satellites: {sum(len(m['source'].get('satellites', [])) for m in out)} smaller centres join {len(add)} metros")


def metro_countries(m, ne, netree):
    x, y, p = m["px"]
    tot = max(p.sum(), 1)
    out = {m["iso"]}
    for i in netree.query(m["area"], predicate="intersects"):
        iso, g = ne[i]
        if iso in out:
            continue
        if p[contains_xy(g, x, y)].sum() / tot >= COUNTRY_SHARE:
            out.add(iso)
    return out


# ---------------------------------------------------------------- admin units

class Unit:
    __slots__ = ("id", "name", "iso", "geom", "parents", "pop", "px", "share", "pshare", "free")

    def __init__(self, uid, name, iso, geom, parents):
        self.id, self.name, self.iso, self.geom, self.parents = uid, name, iso, geom, parents
        self.pop, self.px, self.share, self.pshare = 0.0, None, {}, {}


def gb_level(meta, iso, adm):
    rows = [r for r in meta if r["boundaryISO"] == iso and r["boundaryType"] == adm]
    return max(rows, key=lambda r: int(r["admUnitCount"])) if rows else None


def gb_units(meta, iso, adm):
    r = gb_level(meta, iso, adm)
    path = fetch(f"{iso}_{adm}.geojson", r["simplifiedGeometryGeoJSON"], "geoboundaries")
    gj = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for f in gj["features"]:
        p = f["properties"]
        if f["geometry"] is None:
            continue
        g = polygonal(to_moll(shape(f["geometry"])))
        b = g.bounds
        if g.is_empty or b[2] - b[0] > 2.0e7:  # wrapped across the antimeridian; far from any metro
            continue
        out.append((p.get("shapeID") or f"{iso}-{adm}-{len(out)}", p.get("shapeName") or "", g))
    return out


def assign_pixels(units, x, y, p):
    """Each pixel to the unit holding its centre (-1 for none); x sorted ascending."""
    lab = np.full(len(x), -1, np.int32)
    for k, (_, _, g) in enumerate(units):
        b = g.bounds
        a, z = np.searchsorted(x, b[0]), np.searchsorted(x, b[2], side="right")
        if a == z:
            continue
        sel = np.arange(a, z)
        sel = sel[(y[a:z] >= b[1]) & (y[a:z] <= b[3])]
        if len(sel):
            shapely.prepare(g)
            sel = sel[contains_xy(g, x[sel], y[sel])]
            lab[sel] = k
    return lab


def country_units(iso, meta, nuts, pop, need, whole=None):
    """The units a country lends to metros, at its unit level, with people and (for those touching
    `need`) their pixels. Returns (units, source description)."""
    if iso == "GBR":
        gj = json.loads(ITL3.read_text(encoding="utf-8"))
        raw = [(f["properties"]["ITL325CD"], f["properties"]["ITL325NM"],
                polygonal(to_moll(shape(f["geometry"])))) for f in gj["features"]]
        src = {"kind": "ITL3", "file": "data/raw/tier0/itl3_jan2025_uk_bgc.geojson", "release": "ONS ITL3 January 2025"}
        parents = [[(u[0][:3], ""), (u[0][:4], "")] for u in raw]
        levels = {"ITL3": raw}
        chosen = "ITL3"
    elif iso in nuts:
        raw = nuts[iso]
        src = {"kind": "NUTS3", "file": f"data/raw/worldplan/{NUTS[0]}", "release": "Eurostat GISCO NUTS 2024, 1:3M"}
        parents = [[(u[0][:3], ""), (u[0][:4], "")] for u in raw]
        levels = {"NUTS3": raw}
        chosen = "NUTS3"
    else:
        adms = [a for a in ("ADM1", "ADM2", "ADM3", "ADM4") if gb_level(meta, iso, a)
                and int(gb_level(meta, iso, a)["admUnitCount"]) <= 30000]
        if adms:
            levels = {a: gb_units(meta, iso, a) for a in adms}
            chosen = None
        elif whole is not None and not whole.is_empty:  # no geoBoundaries (Hong Kong, Macau): the territory whole
            raw = [(f"{iso}-ADM0", iso, whole)]
            parents = [[]]
            levels = {"ADM0": raw}
            chosen = "ADM0"
            src = {"kind": "Natural Earth admin 0", "file": "data/raw/world_field/ne_10m_admin_0_countries.zip"}
        else:
            return [], None
    allg = [g for lv in levels.values() for _, _, g in lv]
    bx = np.array([g.bounds for g in allg])
    x, y, p = pop.pixels((bx[:, 0].min(), bx[:, 1].min(), bx[:, 2].max(), bx[:, 3].max()))
    o = np.argsort(x, kind="stable")
    x, y, p = x[o], y[o], p[o]
    labs, sums = {}, {}
    for a, lv in levels.items():
        labs[a] = assign_pixels(lv, x, y, p)
        sums[a] = np.bincount(labs[a][labs[a] >= 0], weights=p[labs[a] >= 0], minlength=len(lv))
    if chosen is None:
        # the finest level whose median person lives in a unit of MEDIAN_UNIT or more, among levels
        # covering at least half the people the best-covering level does
        best = max(s.sum() for s in sums.values())
        chosen = adms[0]
        for a in adms[1:]:
            s = sums[a]
            if s.sum() < 0.5 * best or len(s) < 2:
                continue
            per = s[labs[a][labs[a] >= 0]]  # each pixel's unit people, weighted by the pixel's people
            w = p[labs[a] >= 0]
            o2 = np.argsort(per)
            med = per[o2][np.searchsorted(np.cumsum(w[o2]), 0.5 * w.sum())]
            if med >= MEDIAN_UNIT:
                chosen = a
        coarse = [a for a in adms if a < chosen]

        def reps_in(lv, g):
            r = np.array([u[2].representative_point().coords[0] for u in lv])
            shapely.prepare(g)
            return [lv[k] for k in np.nonzero(contains_xy(g, r[:, 0], r[:, 1]))[0]]
        # where the chosen level misses part of a first-level unit (Moscow has no ADM2 in
        # geoBoundaries), that unit takes the finest level covering 90% of its people
        raw, own, filled = [], [], 0
        if coarse and coarse[0] == "ADM1":
            l1 = labs["ADM1"]
            tot1 = np.bincount(l1[l1 >= 0], weights=p[l1 >= 0], minlength=len(levels["ADM1"]))
            finer = [a for a in adms if a >= chosen][:1] + [a for a in reversed(coarse)]
            for k, top in enumerate(levels["ADM1"]):
                for a in finer:
                    m = (l1 == k) & (labs[a] >= 0)
                    if a == "ADM1" or p[m].sum() >= 0.9 * tot1[k]:
                        got = [top] if a == "ADM1" else reps_in(levels[a], top[2])
                        raw += got
                        own += [a] * len(got)
                        filled += len(got) if a != chosen else 0
                        break
        else:
            raw, own = list(levels[chosen]), [chosen] * len(levels[chosen])
        # parents: the coarser levels' units holding each unit's representative point
        parents = [[] for _ in raw]
        reps = np.array([g.representative_point().coords[0] for _, _, g in raw])
        for a in coarse:
            lv = levels[a]
            got = {}
            for c, (_, _, g) in enumerate(lv):
                shapely.prepare(g)
                for k in np.nonzero(contains_xy(g, reps[:, 0], reps[:, 1]))[0]:
                    got[int(k)] = c
            for k in range(len(raw)):
                c = got.get(k)
                if own[k] <= a:
                    parents[k].append((raw[k][0], raw[k][1]))
                else:
                    parents[k].append((lv[c][0], lv[c][1]) if c is not None else (f"none-{a}", ""))
        labs[chosen] = assign_pixels(raw, x, y, p)
        sums[chosen] = np.bincount(labs[chosen][labs[chosen] >= 0], weights=p[labs[chosen] >= 0], minlength=len(raw))
        r = gb_level(meta, iso, chosen)
        src = {"kind": "geoBoundaries", "level": chosen,
               "urls": {a: gb_level(meta, iso, a)["simplifiedGeometryGeoJSON"] for a in sorted(set(own))},
               "release": f"gbOpen, built {r['buildDate']}", "licence": r["boundaryLicense"]}
        if filled:
            src["coarser_where_missing"] = filled
    lab, s = labs[chosen], sums[chosen]
    units = []
    tree = STRtree([g for _, _, g in raw])
    near = set(int(i) for i in tree.query(need, predicate="intersects")) if not need.is_empty else set()
    for k, (uid, name, g) in enumerate(raw):
        u = Unit(uid, name, iso, g, parents[k])
        u.pop = float(s[k])
        if k in near:
            m = lab == k
            u.px = (x[m], y[m], p[m])
        units.append(u)
    src["level"] = chosen
    src["count"] = len(raw)
    return units, src


def load_nuts(europe):
    gj = json.loads(fetch(*NUTS).read_text(encoding="utf-8"))
    out = collections.defaultdict(list)
    for f in gj["features"]:
        pr = f["properties"]
        iso = {"XK": "XKX"}.get(pr["ISO3_CODE"], pr["ISO3_CODE"])  # GISCO gives Kosovo no ISO3
        if iso in europe and iso != "GBR":
            out[iso].append((pr["NUTS_ID"], pr["NAME_LATN"], polygonal(to_moll(shape(f["geometry"])))))
    return dict(out)


# ---------------------------------------------------------------- shares

def shares(units, ms, pk):
    """Each near unit's share of people inside each metro area and each existing pack."""
    mtree = STRtree([m["area"] for m in ms])
    ptree = STRtree([q["geom"] for q in pk])
    for q in pk:
        shapely.prepare(q["geom"])
    for u in units:
        if u.px is None or u.pop <= 0:
            continue
        x, y, p = u.px
        for i in mtree.query(u.geom, predicate="intersects"):
            f = p[contains_xy(ms[i]["area"], x, y)].sum() / u.pop
            if f > 0:
                u.share[ms[i]["key"]] = float(f)
        for i in ptree.query(u.geom, predicate="intersects"):
            f = p[contains_xy(pk[i]["geom"], x, y)].sum() / u.pop
            if f > 0:
                u.pshare[pk[i]["id"]] = float(f)


# ---------------------------------------------------------------- the plan

def densest(units, taken):
    """The densest populated pixel of these units outside the taken packs: (x, y, unit)."""
    best = (-1.0, None)
    for u in units:
        if u.px is None:
            continue
        x, y, p = u.px
        ok = np.ones(len(x), bool)
        for pid in u.pshare:
            if pid in taken:
                ok &= ~contains_xy(taken[pid]["geom"], x, y)
        if ok.any():
            i = int(np.argmax(np.where(ok, p, -1)))
            if p[i] > best[0]:
                best = (float(p[i]), (float(x[i]), float(y[i]), u))
    return best[1]


def assign(ms, units, taken):
    """Units to metros (free units with half their people in a metro's area, the larger share
    winning), metros with none taking the unit at their area's densest free pixel."""
    owner, out, dropped = {}, collections.defaultdict(list), []
    tk = lambda u: sum(f for pid, f in u.pshare.items() if pid in taken)  # noqa: E731
    for u in units:
        if u.px is None or not u.share or tk(u) >= UNIT_SHARE:
            continue
        key, f = max(u.share.items(), key=lambda kv: kv[1])
        if f >= UNIT_SHARE:
            owner[u.id] = key
            out[key].append(u)
    for m in ms:
        if out.get(m["key"]):
            continue
        x, y, p = m["px"]
        ok = np.ones(len(x), bool)
        for q in taken.values():
            if q["geom"].intersects(m["area"]):
                ok &= ~contains_xy(q["geom"], x, y)
        if p[ok].sum() < 0.1 * p.sum():
            dropped.append((m, "inside existing packs"))
            continue
        i = int(np.argmax(np.where(ok, p, -1)))
        cand = [u for u in m["units_near"] if u.px is not None and u.geom.contains(shapely.Point(x[i], y[i]))]
        if not cand:
            dropped.append((m, "no admin unit at its densest point"))
            continue
        u = cand[0]
        if tk(u) >= UNIT_SHARE:
            dropped.append((m, "its densest unit is in an existing pack"))
        elif u.id in owner:
            dropped.append((m, f"its densest unit is in {owner[u.id]}"))
        else:
            owner[u.id] = m["key"]
            out[m["key"]].append(u)
    return out, dropped


def people(us, taken):
    return sum(u.pop * (1 - sum(f for pid, f in u.pshare.items() if pid in taken)) for u in us)


def split(us, taken, peak):
    """Pieces of at most CAP people along the coarsest parent units that fit, grown by adjacency,
    the first from the group nearest the metro's densest square km (so the core keeps its
    suburbs: T-556's Jakarta), the rest from the most populous left."""
    def leaves(group, depth):
        if people(group, taken) <= CAP or depth >= len(group[0].parents):
            if people(group, taken) <= CAP or len(group) == 1:
                return [group]
            return [[u] for u in group]
        by = collections.defaultdict(list)
        for u in group:
            by[u.parents[depth]].append(u)
        out = []
        for g in by.values():
            out += leaves(g, depth + 1)
        return out
    ls = leaves(us, 0)
    if len(ls) == 1:
        return [us]
    geo = [union_all([u.geom for u in l]).buffer(1500) for l in ls]
    pp = [people(l, taken) for l in ls]
    free = set(range(len(ls)))
    pieces = []
    first = min(range(len(ls)), key=lambda i: min(u.geom.distance(peak) for u in ls[i]))
    while free:
        seed = first if first in free else max(free, key=lambda i: pp[i])
        free.discard(seed)
        piece, tot, g = [seed], pp[seed], geo[seed]
        while True:
            nb = [i for i in free if tot + pp[i] <= CAP and geo[i].intersects(g)]
            if not nb:
                break
            i = max(nb, key=lambda i: pp[i])
            free.discard(i)
            piece.append(i)
            tot += pp[i]
            g = g.union(geo[i])
        pieces.append(piece)
    # a piece under MIN_POP joins the touching piece with the fewest people that still fits
    while True:
        tot = [sum(pp[i] for i in pc) for pc in pieces]
        small = [k for k in range(len(pieces)) if tot[k] < MIN_POP]
        moved = False
        for k in small:
            g = union_all([geo[i] for i in pieces[k]])
            nb = [j for j in range(len(pieces)) if j != k and tot[j] + tot[k] <= CAP
                  and union_all([geo[i] for i in pieces[j]]).intersects(g)]
            if nb:
                j = min(nb, key=lambda j: tot[j])
                pieces[j] += pieces[k]
                del pieces[k]
                moved = True
                break
        if not moved:
            break
    return [[u for i in pc for u in ls[i]] for pc in pieces]


def piece_name(us):
    """A name for a split piece or a pack's remainder: the named second-level parent (a district or
    prefecture, not a province) holding most of its people, else its most populous unit."""
    for depth in range(1, len(us[0].parents)):
        tot = collections.Counter()
        for u in us:
            tot[u.parents[depth]] += u.pop
        (pid, name), _ = tot.most_common(1)[0]
        if name:
            return name
    return max(us, key=lambda u: u.pop).name


def make_id(name, used):
    s = "".join(c for c in ascii_name(name).lower() if c.isalpha()) or "xxx"
    s = s + "xxx"
    cands = [s[:3]]
    cands += [s[0] + s[1] + c for c in s[3:]]
    cands += [s[0] + b + c for i, b in enumerate(s[1:]) for c in s[i + 2:]]
    cands += [s[0] + b + c for b in "abcdefghijklmnopqrstuvwxyz" for c in "abcdefghijklmnopqrstuvwxyz"]
    for c in cands:
        if c not in used and c not in ("con", "prn", "aux", "nul"):  # Windows device names (T-788: Cornwall)
            used.add(c)
            return c
    raise RuntimeError(name)


def std_offset(tz):
    from zoneinfo import ZoneInfo
    z = ZoneInfo(tz)
    offs = [datetime(2025, mth, 15, 12, tzinfo=z).utcoffset().total_seconds() / 3600 for mth in (1, 7)]
    return math.floor(min(offs))


# ---------------------------------------------------------------- Geofabrik

def geofabrik():
    gj = json.loads(fetch(*GEOFABRIK).read_text(encoding="utf-8"))
    ex = {}
    for f in gj["features"]:
        pr = f["properties"]
        url = pr["urls"]["pbf"]
        path = url.split("download.geofabrik.de/")[1].replace("-latest.osm.pbf", "")
        g = polygonal(shape(f["geometry"]))
        ex[pr["id"]] = {"id": pr["id"], "parent": pr.get("parent"), "path": path, "geom": g,
                        "multi": len(pr.get("iso3166-1:alpha2", [])) > 1}
    kids = collections.defaultdict(list)
    for e in ex.values():
        kids[e["parent"]].append(e["id"])
    for e in ex.values():
        shapely.prepare(e["geom"])
    return ex, kids


def extracts(outline, ex, kids, tol=0.005):
    """The smallest Geofabrik extracts covering the outline (lon/lat): descend from the continents
    while a node's children (single-country ones first) cover its part, then drop any extract that
    lies inside another chosen one."""
    ref = outline.area

    def covers(ids, part):
        return part.difference(union_all([ex[c]["geom"] for c in ids])).area <= tol * ref

    def rec(node, part):
        cs = [c for c in kids.get(node, []) if ex[c]["geom"].intersects(part)
              and ex[c]["geom"].intersection(part).area > 1e-3 * ref]
        whole = [rec(c, part) for c in cs if covers([c], part)]
        if whole:
            return min(whole, key=lambda r: sum(ex[c]["geom"].area for c in r))
        for pool in ([c for c in cs if not ex[c]["multi"]], cs):
            if pool and covers(pool, part):
                # greedy cover, largest share of the part first
                pool = sorted(pool, key=lambda c: -ex[c]["geom"].intersection(part).area)
                use = []
                for c in pool:
                    use.append(c)
                    if covers(use, part):
                        break
                out = []
                for c in use:
                    out += rec(c, part.intersection(ex[c]["geom"]))
                return out
        return [node] if node is not None else cs
    got = sorted(set(rec(None, outline)) - {None})
    keep = [a for a in got if not any(b != a and ex[a]["geom"].difference(ex[b]["geom"]).area <= 0.01 * ex[a]["geom"].area
                                      for b in got)]
    return [ex[a]["path"] for a in keep]


# ---------------------------------------------------------------- main

def build(taken_ids=None, quiet=False):
    """Everything up to the units' shares; cached for the test runs."""
    pop = Pop()
    europe, meta = europe_isos()
    ne = countries()
    netree = STRtree([g for _, g in ne])
    pk = packs()
    ms = metros(pop, ne, europe)
    nuts = load_nuts(europe)
    need = collections.defaultdict(list)
    for m in ms:
        m["countries"] = sorted(metro_countries(m, ne, netree) - {"USA"})
        for iso in m["countries"]:
            need[iso].append(m["area"])
    for q in pk.values():
        for iso in {ne[i][0] for i in netree.query(q["geom"], predicate="intersects")}:
            if iso in need:
                need[iso].append(q["geom"])
    units, sources = [], {}
    for iso in sorted(need):
        whole = union_all([g for i, g in ne if i == iso])
        us, src = country_units(iso, meta, nuts, pop, union_all(need[iso]), whole)
        if not us:
            log(f"  {iso}: no units")
            continue
        sources[iso] = src
        units += us
        pop.cache.clear()
        log(f"  {iso}: {src['kind']} {src['level']}, {len(us)} units, {sum(u.px is not None for u in us)} near")
    shares(units, ms, list(pk.values()))
    near = collections.defaultdict(list)
    for u in units:
        for key in u.share:
            near[key].append(u)
    for m in ms:
        m["units_near"] = near[m["key"]]
    return {"ms": ms, "units": units, "sources": sources, "packs": pk, "europe": europe, "pop": pop}


def plan(st, taken):
    """Rows for the metros, given the existing packs that are taken."""
    ms, units = st["ms"], st["units"]
    owned, dropped = assign(ms, units, taken)
    rows = []
    for m in ms:
        us = owned.get(m["key"])
        if not us:
            continue
        n = people(us, taken)
        if n < MIN_POP:
            dropped.append((m, f"{n / 1e6:.2f}M people outside existing packs"))
            continue
        pieces = split(us, taken, m["peak"]) if n > CAP else [us]
        lost_core = any(q["geom"].contains(m["peak"]) for q in taken.values())
        # the piece nearest the area's densest square km keeps the metro's name
        near = min(range(len(pieces)), key=lambda k: min(u.geom.distance(m["peak"]) for u in pieces[k]))
        for k, pc in enumerate(pieces):
            name = m["name"] if k == near and not lost_core else piece_name(pc)
            if not latin(name):
                name = next((s for s in m["alt_names"] + [max(pc, key=lambda u: u.pop).name] if latin(s)), name)
            if people(pc, taken) < MIN_POP:
                dropped.append((m, f"a split piece ({name}) of {people(pc, taken) / 1e6:.2f}M fits no neighbour"))
                continue
            rows.append({"metro": m, "units": pc, "people": people(pc, taken), "core": densest(pc, taken),
                         "name": name, "split": len(pieces) > 1})
    return rows, dropped


def latin(s):
    return bool(s) and all(not c.isalpha() or "LATIN" in unicodedata.name(c, "") for c in s)


def outline_of(us, taken, before):
    g = union_all([u.geom for u in us])
    cut = [q["geom"] for q in taken.values() if q["geom"].intersects(g)] + [b for b in before if b.intersects(g)]
    if cut:
        g = g.difference(union_all(cut))
    return polygonal(g)


def iso2s():
    """ISO 3166 alpha-3 (as countries() codes them) to alpha-2, from Natural Earth."""
    df = pyogrio.read_dataframe(f"zip://{NE}", columns=["ISO_A3", "ADM0_A3", "ISO_A2_EH"], read_geometry=False)
    out = {}
    for iso, adm, a2 in zip(df.ISO_A3, df.ADM0_A3, df.ISO_A2_EH):
        code = iso if iso != "-99" else adm
        key = NE_FIX.get(adm, NE_FIX.get(code, code))
        if a2 != "-99" or key not in out:  # Northern Cyprus (-99) is folded into CYP
            out[key] = a2
    return out


def keep_order(rows, prev):
    """(order, id) for each row. A row keeps the order and id of the previous plan's row it shares
    the most people with (each used once, the largest shares first); the others follow the last."""
    old = {}
    for j, o in enumerate(prev):
        for us in o["units"].values():
            for uid in us:
                old[uid] = j
    pairs = collections.Counter()
    for i, r in enumerate(rows):
        for u in r["units"]:
            if u.id in old:
                pairs[i, old[u.id]] += u.pop
    got, taken_j = {}, set()
    for (i, j), _ in pairs.most_common():
        if i not in got and j not in taken_j:
            got[i] = (prev[j]["order"], prev[j]["id"])
            taken_j.add(j)
    n = max((o["order"] for o in prev), default=0)
    out = []
    for i in range(len(rows)):
        if i not in got:
            n += 1
            got[i] = (n, None)
        out.append(got[i])
    gone = [prev[j]["name"] for j in range(len(prev)) if j not in taken_j]
    if gone:
        log(f"previous rows with no successor (their order and id stay unused): {gone}")
    return out


def finish(st, rows, taken=None, prev=()):
    """Order, ids, names, origins, time zones, extracts, archetypes; the rows as JSON. `prev`, the
    previous plan's rows, keeps their order and ids (keep_order)."""
    from timezonefinder import TimezoneFinder
    tf = TimezoneFinder()
    ex, kids = geofabrik()
    a2 = iso2s()
    taken = st["packs"] if taken is None else taken
    rank = {"uk": 0, "europe": 1, "world": 2}
    rows.sort(key=lambda r: (rank[r["metro"]["region"]], -r["people"]))
    keep = keep_order(rows, prev)
    rows = [r for _, r in sorted(zip(keep, rows), key=lambda kr: kr[0][0])]
    keep.sort()
    used = set(st["packs"]) | {c["id"] for c in json.loads(REGISTRY.read_text(encoding="utf-8"))["cities"]}
    used |= {o["id"] for o in prev}
    out, feats, before = [], [], []
    for (n, cid), r in zip(keep, rows):
        m = r["metro"]
        g = outline_of(r["units"], taken, before)
        before.append(g)
        x, y, _ = r["core"]
        lon, lat = INV.transform(x, y)
        tz = tf.timezone_at(lng=lon, lat=lat) or tf.closest_timezone_at(lng=lon, lat=lat)
        ll = to_ll(g)
        isos = sorted({u.iso for u in r["units"]})
        by = collections.Counter()
        for u in r["units"]:
            by[u.iso] += u.pop
        main_iso = by.most_common(1)[0][0]
        arch, module = ARCHETYPE.get(main_iso, ("D", None))
        if module == "england" and not all(u.id[:3] in ("TLC", "TLD", "TLE", "TLF", "TLG", "TLH", "TLI", "TLJ", "TLK")
                                           for u in r["units"]):
            arch, module = "D", None
        cid = cid or make_id(r["name"], used)
        row = {"order": n, "id": cid, "name": r["name"], "status": "planned", "region": m["region"],
               "countries": isos, "iso2": a2.get(main_iso), "people": round(r["people"]),
               "origin": [round(lon, 4), round(lat, 4)], "tz": tz, "utc_offset": std_offset(tz),
               "extracts": extracts(ll, ex, kids), "archetype": arch, "module": module,
               "units": {iso: [u.id for u in r["units"] if u.iso == iso] for iso in isos},
               "bbox": [round(v, 3) for v in ll.bounds], "source": m["source"]}
        if r["split"]:
            row["split_of"] = m["name"]
        out.append(row)
        feats.append({"type": "Feature", "properties": {"order": n, "id": cid, "name": r["name"], "people": row["people"]},
                      "geometry": mapping(shapely.set_precision(ll, 1e-5))})
    return out, feats


def write(st, out, feats, dropped):
    built = [{"id": q["id"], "name": q["name"], "status": "built"} for q in st["packs"].values()]
    old = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    zone_rows = [r for r in old.get("metros", []) if r.get("kind") == "zone"]  # plan_zones' rows stay as they are
    built = [b for b in built if b["id"] not in {r["id"] for r in zone_rows}]
    doc = {"note": "T-536, pipeline stage 4: written by pipeline/world_plan.py; the rules are in docs/adding-a-city.md",
           "built": str(datetime.now().date()),
           "unit_sources": st["sources"],
           "metros": out + zone_rows + built,
           "dropped": [{"name": m["name"], "source": m["source"], "reason": why} for m, why in dropped]}
    if "zones" in old:
        doc["zones"] = old["zones"]
    OUT.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    WORK.mkdir(parents=True, exist_ok=True)
    ext = WORK / "extents.geojson"
    if zone_rows and ext.exists():
        zids = {r["id"] for r in zone_rows}
        feats = feats + [f for f in json.loads(ext.read_text(encoding="utf-8"))["features"] if f["properties"]["id"] in zids]
    ext.write_text(json.dumps({"type": "FeatureCollection", "features": feats}), encoding="utf-8")
    regions = collections.Counter(r["region"] for r in out)
    log(f"{len(out)} planned metros ({dict(regions)}), {len(dropped)} dropped; {OUT}")


def test(st):
    """Plan each test pack's metro as if the pack did not exist."""
    res = {}
    for pid in TESTS:
        taken = {k: v for k, v in st["packs"].items() if k != pid}
        rows, _ = plan(st, taken)
        q = st["packs"][pid]
        ppl_pack = sum(u.pop * u.pshare.get(pid, 0) for u in st["units"])
        # the pack's people from the grid itself (units can miss pixels at coasts)
        _, _, p = st["pop"].inside(q["geom"])
        tot = float(p.sum())
        best = max(rows, key=lambda r: sum(u.pop * u.pshare.get(pid, 0) for u in r["units"]))
        inside = sum(u.pop * u.pshare.get(pid, 0) for u in best["units"])
        res[pid] = {"pack_people": round(tot), "planned": best["name"], "planned_people": round(best["people"]),
                    "coverage": round(inside / tot, 3), "pack_people_in_units": round(ppl_pack),
                    "units": len(best["units"])}
        log(f"  {pid}: {best['name']} {inside / tot:.1%} of the pack's {tot / 1e6:.2f}M; planned {best['people'] / 1e6:.2f}M "
            f"in {len(best['units'])} units")
    return res


def overlaps(feats, pk):
    """The largest overlap (km2) between any two of these extents, or one and an existing pack."""
    gs = [polygonal(to_moll(shape(f["geometry"]))) for f in feats] + [q["geom"] for q in pk.values()]
    tree = STRtree(gs)
    worst = 0.0
    for i in range(len(feats)):
        for j in tree.query(gs[i], predicate="intersects"):
            if j != i:
                worst = max(worst, gs[i].intersection(gs[j]).area / 1e6)
    return worst


def picture(feats, pk, n=50):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    land = pyogrio.read_dataframe(f"zip://{NE}", columns=[])
    nxt = feats[:n]
    fig, axes = plt.subplots(1, 2, figsize=(22, 10), gridspec_kw={"width_ratios": [1.6, 1]})
    tab = plt.get_cmap("tab20")
    cols = [tab(i) for i in range(20) if i not in (14, 15)]  # tab20's greys read as existing packs
    cmap = lambda i: cols[i % len(cols)]  # noqa: E731
    for ax, box in zip(axes, [(-180, -58, 180, 80), (-11, 35, 45, 66)]):
        land.plot(ax=ax, color="#e8e6e1", edgecolor="#c9c6bf", linewidth=0.3)
        for q in pk.values():
            for g in shapely.get_parts(q["ll"]):
                ax.fill(*g.exterior.xy, color="#77736b", linewidth=0)
        for k, f in enumerate(nxt):
            g = shape(f["geometry"])
            for part in shapely.get_parts(g):
                ax.fill(*part.exterior.xy, color=cmap(k % 20), alpha=0.85, linewidth=0.4, edgecolor="#333")
            c = g.representative_point()
            if box[0] <= c.x <= box[2] and box[1] <= c.y <= box[3]:
                ax.annotate(f"{f['properties']['order']} {f['properties']['name']}" if ax is axes[1] else str(f["properties"]["order"]),
                            (c.x, c.y), fontsize=7 if ax is axes[1] else 6, ha="center", va="center")
        ax.set_xlim(box[0], box[2])
        ax.set_ylim(box[1], box[3])
        ax.set_aspect(1 / math.cos(math.radians((box[1] + box[3]) / 2)) if ax is axes[1] else 1)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].set_title(f"The next {len(nxt)} planned packs (colours, by order) and existing packs (grey)", fontsize=11)
    axes[1].set_title("Europe", fontsize=11)
    fig.tight_layout()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS / "t536-next50.png"
    fig.savefig(path, dpi=110)
    log(f"picture: {path}")


def test_row(st, pid, out):
    """The row an existing pack's metro gets when planned as if the pack did not exist, written as
    a one-row plan file. `test_of` names the pack, so its outline is not cut along it."""
    taken = {k: v for k, v in st["packs"].items() if k != pid}
    rows, _ = plan(st, taken)
    best = max(rows, key=lambda r: sum(u.pop * u.pshare.get(pid, 0) for u in r["units"]))
    done, _ = finish(st, rows, taken)
    row = next(r for r in done if r["name"] == best["name"] and r["people"] == round(best["people"]))
    row["test_of"] = pid
    doc = {"note": f"T-537: pack {pid}'s metro planned as if the pack did not exist (world_plan.py row)",
           "built": str(datetime.now().date()),
           "unit_sources": {iso: st["sources"][iso] for iso in row["countries"]}, "metros": [row]}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    log(f"{pid}: {row['name']} as {row['id']}, {row['people'] / 1e6:.2f}M people, "
        f"{sum(len(v) for v in row['units'].values())} units; {out}")


def zone_groups(units, free, region, floor=ZONE_MIN_POP):
    """Free units (ids) grouped into zones, smallest first, until each holds `floor` people. A zone
    under ZONE_MIN_POP joins the zone of its region it shares the longest border with; a bigger one
    its smallest neighbour there that keeps the two within ZONE_CAP (else its smallest neighbour), so
    one zone does not take a whole region (the longest border put Sarajevo, Zenica and Mostar in one).
    With no neighbour left in its region (a small region, or Osttirol cut off from Tirol) it joins one
    of another region. An island (no land border with any unit) joins the most populous zone of its
    region within ISLAND_KM, or within 10 km once it holds ZONE_MIN_POP (the Isle of Wight stays
    alone). One that can join nothing stays as it is (Ceuta). Returns ({lead id: [units]}, [log lines])."""
    by = {u.id: u for u in units}
    simple = {u.id: u.geom.simplify(500) for u in units}
    zones, done, lines = {i: [i] for i in free}, set(), []
    ppl = lambda z: sum(by[i].free for i in zones[z])  # noqa: E731
    geo = lambda z: union_all([simple[i] for i in zones[z]])  # noqa: E731

    def touching(g, pool):
        b = [(g.boundary.intersection(geo(o).buffer(1000)).length, o) for o in pool]
        return [x for x in b if x[0] > 0]
    while True:
        small = sorted((ppl(z), z) for z in zones if z not in done and ppl(z) < floor)
        if not small:
            break
        n, z = small[0]
        g = geo(z)
        same = [o for o in zones if o != z and region[o] == region[z]]
        border = touching(g, same)
        if not border:
            border = touching(g, [o for o in zones if o != z and region[o] != region[z]])
        island = not any(g.buffer(1000).intersects(simple[u.id]) for u in units if u.id not in zones[z])
        into, how = None, ""
        if border and n < ZONE_MIN_POP:
            length, into = max(border)
            how = f"shared border {length / 1000:.0f} km"
        elif border:
            fit = [b for b in border if n + ppl(b[1]) <= ZONE_CAP]
            into = min(fit or border, key=lambda b: (ppl(b[1]), -b[0]))[1]
            how = f"smallest neighbour, {ppl(into):,.0f}"
        elif island:
            reach = ISLAND_KM if n < ZONE_MIN_POP else 10.0
            near = [(ppl(o), o) for o in same if g.distance(geo(o)) <= reach * 1000]
            if near:
                into = max(near)[1]
                how = f"island, {g.distance(geo(into)) / 1000:.0f} km"
        if into is None:
            done.add(z)
            lines.append(f"  {by[z].name} ({n:,.0f}) stays as it is: {'an island with no zone within reach' if island else 'no zone beside it'}")
            continue
        if region[into] != region[z]:
            how += ", another region"
        lines.append(f"  {by[z].name} ({n:,.0f}) joins {by[into].name} ({how})")
        zones[into] += zones.pop(z)
    return zones, lines


def zone_name(s):
    """A unit's name without its kind ("Rostock, Kreisfreie Stadt", "Arr. Leuven") or second language."""
    import re
    s = s.split("/")[0].strip()
    s = re.sub(r",\s*(Kreisfreie Stadt|Landkreis|Stadtkreis|Kreis)$", "", s)
    s = re.sub(r"^(Landkreis|Kreis|Regionalverband|Arr\.|Arrondissement|Bezirk)\s+", "", s)
    return s.strip()


def plan_zones(isos3):
    """Whole-country cover (T-723): for each country, its units held by no plan row and by no pack
    (half their people), those under ZONE_MIN_DENSITY left as dead space, the rest grouped by
    zone_groups into zone rows (`kind` "zone"), written into the plan beside its metro rows. A
    country's earlier zone rows are replaced; one with the same units keeps its id, order, status
    and run record."""
    import pickle
    import plan_config
    from timezonefinder import TimezoneFinder
    class Here(pickle.Unpickler):  # the cache was written by this file run as __main__
        def find_class(self, mod, name):
            return Unit if name == "Unit" else super().find_class(mod, name)
    with open(WORK / "state.pkl", "rb") as f:
        st = Here(f).load()
    pop = Pop()
    doc = json.loads(OUT.read_text(encoding="utf-8"))
    rows = doc["metros"]
    pk = packs()
    tf = TimezoneFinder()
    ex, kids = geofabrik()
    a2 = iso2s()
    used = set(pk) | {c["id"] for c in json.loads(REGISTRY.read_text(encoding="utf-8"))["cities"]}
    used |= {r["id"] for r in rows}
    order = max(r.get("order", 0) for r in rows)
    record = doc.setdefault("zones", {})
    for iso in isos3:
        units = [u for u in st["units"] if u.iso == iso]
        if not units:  # a country no metro reached, so not in the plan's state (T-788)
            europe, meta = europe_isos()
            whole = union_all([g for i, g in countries() if i == iso])
            units, got = country_units(iso, meta, load_nuts(europe | {iso}), pop, shapely.Polygon(), whole)
            if not units:
                raise SystemExit(f"{iso}: no units at NUTS-3 or in geoBoundaries")
            doc["unit_sources"][iso] = got
        src = doc["unit_sources"][iso]
        # a zone row the runner has moved on (built, gated, shipped, parked) keeps its units; only
        # planned ones are regrouped (T-788: Anita's 0.5M floor came mid-wave)
        old = {r["id"]: r for r in rows if r.get("kind") == "zone" and iso in r.get("units", {})
               and r.get("status", "planned") == "planned"}
        kept = [r for r in rows if r.get("kind") == "zone" and iso in r.get("units", {}) and r["id"] not in old]
        held = {uid for r in rows if r.get("units") and r["id"] not in old for uid in r["units"].get(iso, [])}
        ptree = STRtree([q["geom"] for q in pk.values()])
        pl = list(pk.values())
        free, dead, away = [], [], []
        for u in units:
            if iso in st["europe"]:
                lon, lat = INV.transform(*u.geom.representative_point().coords[0])
                if not (EUROPE_BOX[0] <= lon <= EUROPE_BOX[2] and EUROPE_BOX[1] <= lat <= EUROPE_BOX[3]):
                    away.append(u.name)
                    continue
            x, y, p = pop.inside(u.geom)
            inpk = np.zeros(len(x), bool)
            for i in ptree.query(u.geom, predicate="intersects"):
                inpk |= contains_xy(pl[i]["geom"], x, y)
            u.px = (x[~inpk], y[~inpk], p[~inpk])
            u.free = float(p[~inpk].sum())
            tot = float(p.sum())
            if u.id in held or (tot > 0 and u.free < UNIT_SHARE * tot) or u.free <= 0:
                continue
            dens = u.free / (u.geom.area / 1e6)
            if dens < ZONE_MIN_DENSITY:
                dead.append({"unit": u.id, "name": u.name, "people": round(u.free), "per_km2": round(dens, 1)})
            else:
                free.append(u.id)
        nuts = src["kind"] in ("NUTS3", "ITL3")
        region = {u.id: (u.parents[-1] if nuts else u.parents[0]) if u.parents else iso for u in units}
        groups, lines = zone_groups(units, free, region, ZONE_FLOOR)
        by = {u.id: u for u in units}
        log(f"{iso}: {len(units)} units, {len(held)} in plan rows ({len(kept)} zone rows kept), {len(free)} free, {len(dead)} dead space "
            f"(under {ZONE_MIN_DENSITY:g} a km2: {', '.join(d['name'] for d in dead)}); {len(groups)} zones"
            + (f"; outside Europe, left out: {', '.join(away)}" if away else ""))
        for ln in lines:
            log(ln)
        was = {tuple(sorted(r["units"][iso])): r for r in old.values()}
        out, feats, names = [], [], {r["name"] for r in kept}
        for lead, ids in sorted(groups.items(), key=lambda kv: -sum(by[i].free for i in kv[1])):
            us = sorted((by[i] for i in ids), key=lambda u: -u.free)
            g = polygonal(union_all([u.geom for u in us]))
            cut = [q["geom"] for q in pk.values() if q["geom"].intersects(g)]
            if cut:
                g = polygonal(g.difference(union_all(cut)))
            ll = to_ll(g)
            best = None
            for u in us:
                x, y, p = u.px
                if len(p) and (best is None or p.max() > best[0]):
                    i = int(np.argmax(p))
                    best = (float(p[i]), x[i], y[i])
            lon, lat = INV.transform(best[1], best[2])
            tz = tf.timezone_at(lng=lon, lat=lat) or tf.closest_timezone_at(lng=lon, lat=lat)
            name = zone_name(us[0].name)
            if name in names:  # Leipzig the city and Leipzig the Landkreis: the smaller by its two largest units
                name = ", ".join(zone_name(u.name) for u in us[:2]) if len(us) > 1 else us[0].name
            names.add(name)
            prev = was.get(tuple(sorted(ids)))
            arch, module = "D", None  # zones: global sources and the zone curve, no country module (T-779)
            if prev:
                cid, n = prev["id"], prev["order"]
            else:
                cid, n = make_id(name, used), None
            row = {"order": n, "id": cid, "name": name, "kind": "zone", "status": "planned",
                   "region": "uk" if iso == "GBR" else ("europe" if iso in st["europe"] else "world"),
                   "countries": [iso], "iso2": a2.get(iso), "people": round(sum(u.free for u in us)),
                   "origin": [round(lon, 4), round(lat, 4)], "tz": tz, "utc_offset": std_offset(tz),
                   "extracts": extracts(ll, ex, kids), "archetype": arch, "module": module,
                   "units": {iso: [u.id for u in us]}, "bbox": [round(v, 3) for v in ll.bounds],
                   "source": {"zone": "T-723", "unit_names": [u.name for u in us]}}
            if prev:
                for k in ("status", "run", "gate_relax"):
                    if k in prev:
                        row[k] = prev[k]
            out.append(row)
            feats.append({"type": "Feature", "properties": {"id": cid, "name": name, "people": row["people"], "kind": "zone"},
                          "geometry": mapping(shapely.set_precision(ll, 1e-5))})
        for r in out:
            if r["order"] is None:
                order += 1
                r["order"] = order
            log(f"  {r['order']:>4} {r['id']} {r['name']}: {r['people'] / 1e6:.2f}M, {len(r['units'][iso])} units "
                f"{r['units'][iso]}, extracts {r['extracts']}, origin {r['origin']}{', status ' + r['status'] if r['status'] != 'planned' else ''}")
        gone = [r["id"] for r in old.values() if r["id"] not in {o["id"] for o in out}]
        if gone:
            log(f"  earlier zone rows with no successor, removed: {gone}")
        rows[:] = [r for r in rows if r["id"] not in old] + out
        record[iso] = {"built": str(datetime.now().date()), "task": "T-788",
                       "rule": f"free units at or over {ZONE_MIN_DENSITY:g} people a km2, grouped in their region until each "
                               f"zone holds {ZONE_FLOOR:,}: under {ZONE_MIN_POP:,} by the longest border, then the smallest "
                               f"neighbour keeping both within {ZONE_CAP:,}; another region's when none is left; islands "
                               f"the most populous zone within {ISLAND_KM:g} km (10 km from {ZONE_MIN_POP:,})",
                       "kept": [r["id"] for r in kept], "dead": dead, "merges": [ln.strip() for ln in lines]}
        if away:
            record[iso]["outside_europe"] = away
        ext_path = WORK / "extents.geojson"
        ext = json.loads(ext_path.read_text(encoding="utf-8"))
        drop = set(old) | {f["properties"]["id"] for f in feats}
        ext["features"] = [f for f in ext["features"] if f["properties"]["id"] not in drop] + feats
        ext_path.write_text(json.dumps(ext), encoding="utf-8")
    assert all(plan_config.is_plan_row(r) for r in rows if r.get("kind") == "zone")
    tmp = OUT.with_suffix(".json.part")
    tmp.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, OUT)
    log(f"written {OUT}")


def main(argv):
    import pickle
    if argv and argv[0] == "zones":
        have = json.loads(OUT.read_text(encoding="utf-8"))["unit_sources"]
        # several territories share an alpha-2 in Natural Earth (FR: France and Clipperton, T-779)
        a3 = {v: k for k, v in sorted(iso2s().items(), key=lambda kv: kv[0] in have)}
        cs = argv[argv.index("--countries") + 1].split(",")
        return plan_zones([a3[c] for c in cs])
    cache = WORK / "state.pkl"  # the units and their shares; --fresh rebuilds it
    if cache.exists() and "--fresh" not in argv:
        st = pickle.loads(cache.read_bytes())
        st["pop"] = Pop()
    else:
        st = build()
        WORK.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(pickle.dumps({k: v for k, v in st.items() if k != "pop"}))
    if argv and argv[0] == "row":
        return test_row(st, argv[1], argv[2])
    rows, dropped = plan(st, st["packs"])
    import plan_config
    prev = [r for r in json.loads(OUT.read_text(encoding="utf-8"))["metros"] if plan_config.is_plan_row(r)] if OUT.exists() else []
    out, feats = finish(st, rows, prev=prev)
    # T-540: a row the runner moved on keeps its status and record while its units are the same
    was = {r["id"]: r for r in prev}
    for r in out:
        o = was.get(r["id"])
        if o and "gate_relax" in o and o["units"] == r["units"]:  # T-725: a per-row gate override (pack_gate.py)
            r["gate_relax"] = o["gate_relax"]
        if o and o["status"] != "planned" and o["units"] == r["units"]:
            r["status"] = o["status"]
            if "run" in o:
                r["run"] = o["run"]
    write(st, out, feats, dropped)
    if "check" in argv:
        log(f"largest overlap between planned extents or with a pack: {overlaps(feats, st['packs']):.3f} km2")
        picture(feats, st["packs"])
        res = test(st)
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        (ARTIFACTS / "t536-coverage.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    import shared_lock
    # a replan or zone rows rewrite the plan whole while runners set row statuses (T-786)
    with shared_lock.held("plan", "world_plan.py " + " ".join(sys.argv[1:])[:200]):
        main(sys.argv[1:])
