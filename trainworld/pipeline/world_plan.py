"""T-536, pipeline stage 4 (notes/T-535.md): the world plan. Which metros get a pack, in what order,
and which whole admin units each one holds.

    python pipeline/world_plan.py           plan, write pipeline/world_plan.json and the extents
    python pipeline/world_plan.py check     the same, then the eight-pack test and the next-50 picture
    python pipeline/world_plan.py row <pack> <out.json>
                                            one existing pack's metro planned as if the pack did not
                                            exist, as a one-row plan file (T-537's test build)
    python pipeline/world_plan.py metros --countries US
                                            a FUA country's metro rows planned alone (T-1015), no
                                            other row touched
    python pipeline/world_plan.py zones --countries ES[,PT] [--out <plan.json>]
                                            whole-country cover (T-723): the country's zone rows,
                                            every unit no row or pack holds (docs/zones.md 2);
                                            with --out, a dry run written there

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
DEAD_KEEP = 0.8          # a unit under the density floor holding this share of the zone floor is a zone anyway (T-903: Bergen)
WRAP_HULL = 0.5          # zone A wraps B when this share of B's area lies in A's convex hull (T-1005)
WRAP_BORDER = 0.5        # and A holds this share of B's border with zones, metros and packs
RING_ROUNDS = 3          # ring cuts, each followed by a regroup of the halves (T-1087)
# the merge rule (Anita, 2026-10-24, T-1140): a join passes when the neighbour gets this share of the
# zone's travel with its biggest partner, or else when the joined shape fills COMPACT_HULL of its
# convex hull (or no less than the bigger part's own share less COMPACT_SLACK)
PARTNER_SHARE = 0.5
COMPACT_HULL = 0.75
COMPACT_SLACK = 0.05
# travel without commuting data (T-1140): GHSL people on GRAVITY_KM cells, people x people x the zone
# curve d^-0.8 exp(-d/30 km), pairs within GRAVITY_REACH_KM
GRAVITY_KM = 20.0
GRAVITY_REACH_KM = 200.0
# the floor relaxed for very big regions (Anita, 2026-10-24, T-1140): a zone covering this much holds
# enough at ZONE_MIN_POP (splits Montana in two, Fairbanks from Anchorage; Colorado as without it)
BIG_ZONE_KM2 = 100_000.0
# where commuting data exists, an unbuilt metro row takes a neighbouring free unit sending at least
# this share of its resident workers into it (T-1088; the US's own outlying-county rule)
BELT_SHARE = 0.25
COMMUTE_FLOWS = {"USA": ROOT / "data/raw/acsflows/table1.xlsx"}  # ACS 2016-2020 county to county
# zone rows planned before the floor (T-723 to T-786) used ZONE_MIN_POP as theirs; a replan of those
# countries would regroup their unbuilt rows only
EUROPE_BOX = (-32.0, 27.0, 45.0, 72.0)  # lon/lat: a European country's units outside it (France's overseas départements) are not zoned (T-779)
# per-country zone rules over the defaults above (Anita, 2026-10-22, T-897): China's counties are already the
# size of a European zone, and open land is handled well since T-852, so no unit is dead space by density
# one official clock for a country whose rows' points fall in other tz zones (T-897: Xinjiang's
# Asia/Urumqi is unofficial local time; timetables keep Beijing time)
COUNTRY_TZ = {"CHN": "Asia/Shanghai"}
# de_facto's lines (T-897): for a country, the Natural Earth countries (de facto borders) whose land its
# units lose; only real lines of control and the territories Anita keeps apart (Hong Kong, Macau,
# Taiwan), since elsewhere geoBoundaries and Natural Earth differ only by drawing, and Natural Earth
# gives autonomous parts rows of their own (Åland). Crimea and the Golan go with the side holding
# them, as Natural Earth has them (Anita, 2026-10-22).
# A side with a lon/lat box is that side's land within the box only: India and Pakistan's line of
# control in Kashmir, not the Punjab and Sindh border, where the two sources differ only by drawing.
KASHMIR = (73.0, 32.95, 80.5, 37.5)
CRIMEA = (32.4, 44.3, 36.7, 46.3)
GOLAN = (35.6, 32.6, 36.0, 33.4)
DE_FACTO = {"CHN": {"IND": None, "BTN": None, "TWN": None, "HKG": None, "MAC": None}, "TWN": {"CHN": None},
            "IND": {"CHN": None, "PAK": KASHMIR, "KAS": None}, "PAK": {"IND": KASHMIR, "KAS": None},
            "UKR": {"RUS": CRIMEA}, "SYR": {"ISR": GOLAN}}
# The US and Puerto Rico (T-1004, T-1015): `module` is the country module zone rows take (elsewhere
# zones take none); `zoned_only` limits a first-level unit to these units, the rest dead space
# (Alaska: the road and rail belt, Anchorage to Fairbanks)
ALASKA_BELT = ["02020", "02170", "02122", "02090", "02068", "02240", "02063", "02066"]
ZONE_RULES = {"CHN": {"floor": 1_000_000, "cap": 2_500_000, "min_density": 0.0, "contiguous": True},
              "TWN": {"floor": 1_000_000, "cap": 2_500_000, "min_density": 0.0, "contiguous": True},
              "USA": {"floor": 500_000, "cap": 1_500_000, "min_density": 0.0, "contiguous": True, "module": "usa",
                      "zoned_only": {"02": ALASKA_BELT}, "task": "T-1015"},
              "PRI": {"floor": 500_000, "cap": 1_500_000, "min_density": 0.0, "contiguous": True, "module": "usa",
                      "task": "T-1015"}}
# countries whose metros are GHS-FUAs over 1M as in Europe, planned alone by `metros --countries`
# (T-1004: low-density US cities have urban centres under 1M); every other row stays as it is, and
# a replan of the rest keeps these countries' rows
FUA_COUNTRIES = {"USA"}
# metro rows set by hand in a FUA country: (name, unit ids). Riverside: GHSL's Los Angeles area
# holds San Bernardino but not Riverside, a county over the zone cap (T-1004)
METRO_EDITS = {"USA": [("Riverside", ["06065", "06025"])]}
# a FUA named after the city across the border, or after the smaller of two cores (T-1015)
METRO_NAMES = {"USA": {"Tijuana": "San Diego", "San Jose": "San Francisco"}}
# Census 2020 counties (Puerto Rico's municipios) as the US's units, GEOIDs, the state their parent;
# STATEFP to country, None for the small territories, which wait for their own regions (T-1004)
COUNTIES = ROOT / "data/raw/cb_2020_us_county_500k.zip"
COUNTY_ISO = {"72": "PRI", "60": None, "66": None, "69": None, "78": None}
# a zone's parts touch by land or across at most this much water (a bridge, a narrow strait): the
# `contiguous` rule of zone_groups (Anita, 2026-10-22: no exclaves)
WATER_GAP_KM = 10.0
ARCHIPELAGO_KM = 50.0  # with `contiguous`, island zones this near join as one archipelago (Anita, 2026-10-22)
ZONE_ALONE_MIN = 10_000  # with `contiguous`, a zone that joined nothing and holds fewer people is dead space
# a country's joint commute solve split by first-level unit (T-897: one China group would pass
# intercity.MAX_GROUP_ZONES); first-level names as geoBoundaries gives them, to the solve region's
# name; a first-level unit not listed is its own region, named by its first word
SOLVE_REGIONS = {"CHN": {"Beijing Municipality": "jingjinji", "Tianjin Municipality": "jingjinji",
                         "Hebei Province": "jingjinji", "Shanghai Municipality": "jiangsu",
                         "Guangzhou Province": "guangdong", "Ningxia Ningxia Hui Autonomous Region": "ningxia",
                         "Inner Mongolia Autonomous Region": "innermongolia"},
                 # Census divisions, Delaware, Maryland and DC with the Northeast so the 13 built packs
                 # solve together (T-1004)
                 "USA": {**{s: "northeast" for s in (
                             "Connecticut", "Maine", "Massachusetts", "New Hampshire", "Rhode Island", "Vermont",
                             "New Jersey", "New York", "Pennsylvania", "Delaware", "Maryland", "District of Columbia")},
                         **{s: "south-atlantic" for s in (
                             "Virginia", "West Virginia", "North Carolina", "South Carolina", "Georgia", "Florida")},
                         **{s: "east-south-central" for s in ("Alabama", "Kentucky", "Mississippi", "Tennessee")},
                         **{s: "west-south-central" for s in ("Arkansas", "Louisiana", "Oklahoma", "Texas")},
                         **{s: "east-north-central" for s in ("Illinois", "Indiana", "Michigan", "Ohio", "Wisconsin")},
                         **{s: "west-north-central" for s in (
                             "Iowa", "Kansas", "Minnesota", "Missouri", "Nebraska", "North Dakota", "South Dakota")},
                         **{s: "mountain" for s in (
                             "Arizona", "Colorado", "Idaho", "Montana", "Nevada", "New Mexico", "Utah", "Wyoming")},
                         **{s: "pacific" for s in ("Alaska", "California", "Hawaii", "Oregon", "Washington")}}}

# What a city in each country can have (docs/data-register.md "Countries and their procedure");
# every other country is archetype D. England's module covers England's ITL3 units only.
ARCHETYPE = {"GBR": ("A", "england"), "FRA": ("A", "france"), "DEU": ("A", "germany"),
             "CAN": ("A", "canada"), "JPN": ("A", "japan"), "BRA": ("C", "brazil"),
             "CHN": ("C", "china"), "KOR": ("C", "korea"), "IND": ("C", "india"),
             "NGA": ("D", "nigeria"), "USA": ("A", "usa"), "PRI": ("C", "usa")}
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


_NE = []


def de_facto(iso, g):
    """A unit of country `iso` (Mollweide) less the land Natural Earth gives other countries, whose
    borders are the lines each side holds (T-897: geoBoundaries' Cona County runs over Arunachal
    Pradesh, India's Ladakh over Aksai Chin, China's file holds Taiwan, Hong Kong and Macau). None
    when more of it lies in other countries than in its own and under a tenth in its own: a unit of
    the other side."""
    if not _NE:
        ne = countries()
        _NE[:] = [ne, STRtree([h for _, h in ne])]
    ne, tree = _NE
    sides = DE_FACTO.get(iso, {})
    other = []
    for i in tree.query(g, predicate="intersects"):
        c, h = ne[i]
        if c in sides:
            box = sides[c]
            other.append(h if box is None else h.intersection(to_moll(shapely.box(*box).segmentize(0.1))))
    if not other:
        return g
    cut = union_all(other)
    lost = g.intersection(cut).area
    if lost <= 0:
        return g
    if lost > 0.5 * g.area:
        own = sum(g.intersection(ne[i][1]).area for i in tree.query(g, predicate="intersects") if ne[i][0] == iso)
        if own < lost and own < 0.1 * g.area:
            return None
    return polygonal(g.difference(cut))


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

def load_fua():
    fpath = unzip(FUA[0], FUA[1], ".gpkg")
    fua = pyogrio.read_dataframe(fpath, columns=["eFUA_ID", "eFUA_name", "Cntry_ISO", "FUA_p_2015"])
    return fua, [polygonal(g) for g in fua.geometry]


def load_ucdb(ne):
    """GHSL's urban centres, their centroids by id, and each one's country."""
    lay = "GHS_UCDB_THEME_GENERAL_CHARACTERISTICS_GLOBE_R2024A"
    uc = pyogrio.read_dataframe(UCDB, layer=lay, columns=["ID_UC_G0", "GC_UCN_MAI_2025", "GC_POP_TOT_2025"])
    cen = pyogrio.read_dataframe(UCDB, layer="UC_centroids")
    cen = dict(zip(cen.ID_UC_G0, cen.geometry))
    return uc, cen, iso_at(ne, STRtree([g for _, g in ne]), [cen[i] for i in uc.ID_UC_G0])


def fua_metros(pop, fua, fgeom, isos):
    """The functional urban areas over MIN_POP people of these countries, as metros."""
    out = []
    for i in range(len(fua)):
        iso = fua.Cntry_ISO[i]
        if iso in isos and fua.FUA_p_2015[i] >= 0.5 * MIN_POP:
            _, _, p = pop.inside(fgeom[i])
            if p.sum() >= MIN_POP:
                out.append({"key": f"fua{int(fua.eFUA_ID[i])}", "name": fua.eFUA_name[i], "iso": iso,
                            "region": "uk" if iso == "GBR" else ("world" if iso in FUA_COUNTRIES else "europe"),
                            "source": {"fua": [int(fua.eFUA_ID[i])]}, "alt_names": [], "area": fgeom[i]})
    return out


def measure(pop, out):
    for m in out:
        x, y, p = pop.inside(m["area"])
        m["px"] = (x, y, p)
        m["pop_area"] = float(p.sum())
        i = int(np.argmax(p))
        m["peak"] = shapely.Point(x[i], y[i])
        shapely.prepare(m["area"])


def country_metros(pop, ne, iso):
    """A FUA country's metros (FUA_COUNTRIES, T-1015): its FUAs over 1M, with its smaller centres
    as satellites."""
    fua, fgeom = load_fua()
    out = fua_metros(pop, fua, fgeom, {iso})
    log(f"{iso}: {len(out)} functional urban areas over {MIN_POP:,}")
    uc, cen, uc_iso = load_ucdb(ne)
    satellites(out, uc, cen, uc_iso, only=iso)
    measure(pop, out)
    return out


def metros(pop, ne, europe):
    """Candidate metros, each with the area its units are judged against: in Europe a functional
    urban area over 1M people; elsewhere an urban centre over 1M with the FUAs it holds half of (and
    the FUA around its population-weighted centre), centres sharing a FUA merged. The FUA countries
    outside Europe are planned alone (country_metros)."""
    fua, fgeom = load_fua()
    out = fua_metros(pop, fua, fgeom, europe)
    log(f"Europe: {len(out)} functional urban areas over {MIN_POP:,}")

    uc, cen, uc_iso = load_ucdb(ne)
    big = uc[uc.GC_POP_TOT_2025 >= MIN_POP].reset_index(drop=True)
    pts = [cen[i] for i in big.ID_UC_G0]
    isos = [uc_iso[k] for k in np.nonzero(uc.GC_POP_TOT_2025.to_numpy() >= MIN_POP)[0]]
    ftree = STRtree(fgeom)
    world = []
    for k in range(len(big)):
        if isos[k] in europe or isos[k] in FUA_COUNTRIES:
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
        f"Europe and the FUA countries, {len(groups)} metros after merging")
    satellites(out, uc, cen, uc_iso)
    measure(pop, out)
    return out


def satellites(out, uc, cen, uc_iso, only=None):
    """A smaller urban centre outside every metro's area, within SAT_GAP of a metro's own centres,
    joins the nearest such metro's area: GHSL gives Ikorodu (0.86M, 2.2 km from Lagos) a FUA of
    its own (T-556). With `only` (a FUA country's metros), that country's centres alone; otherwise
    the FUA countries' centres are left out."""
    ids = [int(i) for i in uc.ID_UC_G0]
    geom = [polygonal(g) for g in uc.geometry]
    cpts = [cen[i] for i in uc.ID_UC_G0]
    atree = STRtree([m["area"] for m in out])
    pi, mi = atree.query(cpts, predicate="within")
    inside = set(pi.tolist())
    own = collections.defaultdict(list)
    for a, k in zip(pi, mi):
        if "ucdb" not in out[k]["source"] or ids[a] in out[k]["source"]["ucdb"]:
            own[int(k)].append(geom[a])
    centres = [union_all(own[k]) if own[k] else shapely.Polygon() for k in range(len(out))]
    ctree = STRtree(centres)
    add = collections.defaultdict(list)
    for a in range(len(ids)):
        if a in inside or uc.GC_POP_TOT_2025[a] >= MIN_POP or (uc_iso[a] != only if only else uc_iso[a] in FUA_COUNTRIES):
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


GAP_MIN_KM2 = 5.0  # land of the country no unit covers, from this size, joins a unit (T-897)
GB_CACHE = "t897e"  # bump when gb_units' rules change: the cache beside each file is then rebuilt


def gb_units(meta, iso, adm):
    """[(shapeID, name, Mollweide geometry)] of a geoBoundaries level: each unit on its own side of
    the de facto lines (de_facto), and the country's land the file leaves out (Qingdao's and Nanshan's
    old centres in China's ADM3) added to the unit it shares the longest border with. Cached beside
    the file."""
    import pickle
    r = gb_level(meta, iso, adm)
    path = fetch(f"{iso}_{adm}.geojson", r["simplifiedGeometryGeoJSON"], "geoboundaries")
    cache = path.with_suffix(f".{GB_CACHE}.pkl")
    if cache.exists() and cache.stat().st_mtime >= path.stat().st_mtime:
        return pickle.loads(cache.read_bytes())
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
        g = de_facto(iso, g)
        if g is None or g.is_empty:
            continue
        out.append((p.get("shapeID") or f"{iso}-{adm}-{len(out)}", p.get("shapeName") or "", g))
    out = fill_gaps(iso, out)
    tmp = cache.with_suffix(".part")
    tmp.write_bytes(pickle.dumps(out))
    os.replace(tmp, cache)
    return out


def fill_gaps(iso, units):
    """Units with the pieces of their country's land (Natural Earth) that no unit covers, from
    GAP_MIN_KM2, each added to the unit it shares the longest border with."""
    if not _NE:
        de_facto(iso, shapely.Point(0, 0))
    ne = _NE[0]
    own = [h for c, h in ne if c == iso]
    if not own or not units:
        return units
    gaps = polygonal(union_all(own).difference(union_all([g for _, _, g in units])))
    pieces = [p for p in shapely.get_parts(gaps) if p.area >= GAP_MIN_KM2 * 1e6]
    if not pieces:
        return units
    tree = STRtree([g for _, _, g in units])
    add = collections.defaultdict(list)
    for p in pieces:
        near = tree.query(p, predicate="dwithin", distance=100.0)
        if not len(near):
            continue
        k = max(near, key=lambda k: p.boundary.intersection(units[k][2].buffer(100)).length)
        add[int(k)].append(p)
        if p.area >= 500e6:
            log(f"  {iso}: {p.area / 1e6:,.0f} km2 no unit covers, added to {units[k][1]}")
    log(f"  {iso}: {sum(len(v) for v in add.values())} pieces of land no unit covers ({sum(p.area for v in add.values() for p in v) / 1e6:,.0f} km2) "
        f"added to {len(add)} units")
    out = list(units)
    for k, ps in add.items():
        uid, name, g = out[k]
        out[k] = (uid, name, polygonal(union_all([g] + [p.buffer(1) for p in ps])))
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


def census_units(iso, pop):
    """The US's or Puerto Rico's Census 2020 counties (COUNTIES), each with its people and pixels,
    read part by part (Aleutians West crosses the 180th meridian)."""
    df = pyogrio.read_dataframe(f"zip://{COUNTIES}").sort_values("GEOID")  # by state: Pop's tile cache holds
    units = []
    for r in df.itertuples():
        if COUNTY_ISO.get(r.STATEFP, "USA") != iso:
            continue
        g = polygonal(to_moll(r.geometry))
        u = Unit(r.GEOID, f"{r.NAMELSAD}, {r.STUSPS}", iso, g, [(r.STATEFP, r.STATE_NAME)])
        px = [pop.inside(part) for part in shapely.get_parts(g)]
        u.px = tuple(np.concatenate([q[k] for q in px]) for k in range(3))
        u.pop = float(u.px[2].sum())
        units.append(u)
    src = {"kind": "Census counties", "file": "data/raw/cb_2020_us_county_500k.zip",
           "release": "US Census Bureau cartographic boundary file 2020, 1:500,000", "licence": "public domain",
           "level": "county", "count": len(units)}
    log(f"  {iso}: {len(units)} counties, {sum(u.pop for u in units) / 1e6:.1f}M people (GHSL)")
    return units, src


def country_units(iso, meta, nuts, pop, need, whole=None):
    """The units a country lends to metros, at its unit level, with people and (for those touching
    `need`) their pixels. Returns (units, source description)."""
    if iso in COUNTY_ISO.values() or iso == "USA":
        return census_units(iso, pop)
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


_GROUPS = {}  # Geofabrik id: whether it holds another extract of its parent (extracts)


def extracts(outline, ex, kids, tol=0.005):
    """The smallest Geofabrik extracts covering the outline (lon/lat): descend from the continents
    while a node's children (single-country ones first) cover its part, a set of children that
    group no other child against a single one covering it (states against us-midwest), then drop
    any extract that lies inside another chosen one."""
    ref = outline.area

    def covers(ids, part):
        return part.difference(union_all([ex[c]["geom"] for c in ids])).area <= tol * ref

    def grouping(c):
        if c not in _GROUPS:
            g = ex[c]["geom"]
            _GROUPS[c] = any(d != c and g.contains(ex[d]["geom"].representative_point())
                             and ex[d]["geom"].difference(g).area <= 0.01 * ex[d]["geom"].area
                             for d in kids.get(ex[c]["parent"], []))
        return _GROUPS[c]

    def greedy(pool, part):  # largest share of the part first
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

    def rec(node, part):
        cs = [c for c in kids.get(node, []) if ex[c]["geom"].intersects(part)
              and ex[c]["geom"].intersection(part).area > 1e-3 * ref]
        whole = [rec(c, part) for c in cs if covers([c], part)]
        # siblings that group no other sibling (T-1015: US states, beside the us-midwest group)
        leaves = [c for c in cs if not ex[c]["multi"] and not grouping(c)]
        if whole and len(leaves) > 1 and covers(leaves, part):
            whole.append(greedy(leaves, part))
        if whole:
            return min(whole, key=lambda r: sum(ex[c]["geom"].area for c in r))
        for pool in ([c for c in cs if not ex[c]["multi"]], cs):
            if pool and covers(pool, part):
                return greedy(pool, part)
        return [node] if node is not None else cs
    got = sorted(set(rec(None, outline)) - {None})
    keep = [a for a in got if not any(b != a and ex[a]["geom"].difference(ex[b]["geom"]).area <= 0.01 * ex[a]["geom"].area
                                      for b in got)]
    # one smaller extract elsewhere in the tree covering it all (T-1015: Puerto Rico lies in
    # central-america's polygon, not north-america's, but has an extract of its own there)
    area = sum(ex[a]["geom"].area for a in keep)
    b = outline.bounds
    one = [c for c, e in ex.items() if e["geom"].area < area and not e["multi"]
           and e["geom"].bounds[0] <= b[0] and e["geom"].bounds[1] <= b[1] and e["geom"].bounds[2] >= b[2]
           and e["geom"].bounds[3] >= b[3] and covers([c], outline)]
    if one:
        keep = [min(one, key=lambda c: ex[c]["geom"].area)]
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
        m["countries"] = sorted(metro_countries(m, ne, netree) - FUA_COUNTRIES)
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


def keep_order(rows, prev, start=0):
    """(order, id) for each row. A row keeps the order and id of the previous plan's row it shares
    the most people with (each used once, the largest shares first); the others follow the last,
    or `start` when that is later."""
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
    n = max(max((o["order"] for o in prev), default=0), start)
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


def finish(st, rows, taken=None, prev=(), others=()):
    """Order, ids, names, origins, time zones, extracts, archetypes; the rows as JSON. `prev`, the
    previous plan's rows, keeps their order and ids (keep_order); `others`, rows that stay beside
    them, keep theirs (new rows follow their last order and take no id of theirs)."""
    from timezonefinder import TimezoneFinder
    tf = TimezoneFinder()
    ex, kids = geofabrik()
    a2 = iso2s()
    taken = st["packs"] if taken is None else taken
    rank = {"uk": 0, "europe": 1, "world": 2}
    rows.sort(key=lambda r: (rank[r["metro"]["region"]], -r["people"]))
    keep = keep_order(rows, prev, max((o.get("order") or 0 for o in others), default=0))
    rows = [r for _, r in sorted(zip(keep, rows), key=lambda kr: kr[0][0])]
    keep.sort()
    used = set(st["packs"]) | {c["id"] for c in json.loads(REGISTRY.read_text(encoding="utf-8"))["cities"]}
    used |= {o["id"] for o in prev} | {o["id"] for o in others}
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


def alone_row(r):
    """A metro row of a FUA country, planned alone (plan_metros)."""
    return r.get("kind") != "zone" and bool(r.get("units")) and set(r.get("countries", ())) <= FUA_COUNTRIES


def write(st, out, feats, dropped):
    built = [{"id": q["id"], "name": q["name"], "status": "built"} for q in st["packs"].values()]
    old = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    # plan_zones' rows and the FUA countries' (plan_metros) stay as they are
    zone_rows = [r for r in old.get("metros", []) if r.get("kind") == "zone" or alone_row(r)]
    built = [b for b in built if b["id"] not in {r["id"] for r in zone_rows}]
    doc = {"note": "T-536, pipeline stage 4: written by pipeline/world_plan.py; the rules are in docs/adding-a-city.md",
           "built": str(datetime.now().date()),
           "unit_sources": {**old.get("unit_sources", {}), **st["sources"]},
           "metros": out + zone_rows + built,
           "dropped": [{"name": m["name"], "source": m["source"], "reason": why} for m, why in dropped]
           + [d for d in old.get("dropped", []) if d.get("country") in FUA_COUNTRIES]}
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


def unit_borders(simple):
    """{unit: {neighbour: its border along it, m}} for shapes within 1 km of each other."""
    ids = list(simple)
    tree = STRtree([simple[i] for i in ids])
    adj = {i: {} for i in ids}
    for a, i in enumerate(ids):
        near = simple[i].buffer(1000)
        for b in tree.query(near, predicate="intersects"):
            j = ids[b]
            if b > a:
                ln = simple[i].boundary.intersection(simple[j].buffer(1000)).length
                if ln > 0:
                    adj[i][j] = adj[j][i] = ln
    return adj


def zone_groups(units, free, region, floor=ZONE_MIN_POP, cap=ZONE_CAP, contiguous=False, fixed=(), flows=None,
                fixed_units=None, big_km2=0.0):
    """Free units (ids) grouped into zones, smallest first, until each holds `floor` people, or
    ZONE_MIN_POP over `big_km2` (T-1140, docs/zones.md 2). A zone under ZONE_MIN_POP joins the zone of
    its region it travels with most (`flows`; the longest border breaking ties). A bigger one joins a
    neighbour of its region keeping the two within `cap` that passes `accept`: a travel partner
    first, by nearest people centre, and only when none is one, a compact join; the first whose union
    wraps no other zone and no `fixed` area (`wraps`, T-1087). With none within the cap, the smallest
    passing neighbour, unless it holds 80% of `floor`, when it stays as it is. Under 80% of `floor`
    with no neighbour of its region passing, another region's neighbour that passes. Nothing passing,
    it stays as it is. When every join wraps, one holding 80% of `floor` stays; a smaller one joins
    anyway and the ring cut takes it apart.
    With no neighbour left in its region (a small region, or Osttirol cut off from Tirol) it joins one
    of another region, unless it already holds 80% of `floor` (by land or across water). An island
    (no land border with any unit) joins the most populous zone of its region within ISLAND_KM, or
    within 10 km once it holds ZONE_MIN_POP (the Isle of Wight stays alone). One that can join nothing
    stays as it is (Ceuta). With `contiguous` (T-897), a zone with no land neighbour left joins the
    nearest zone within WATER_GAP_KM across water instead of the island rule, so every zone's parts
    touch, and islands of one region within ARCHIPELAGO_KM join as one.
    Then each zone still wrapping something is cut in two across it (`cut`) and the halves regroup,
    a half whose every join wraps staying as it is, RING_ROUNDS times at most. `fixed`: the metro rows and built packs, [{id, name, geom (Mollweide),
    xy (people centre)}]. `flows`: {key: {key: n}}, travel both ways between units (or a fixed area's
    own key); `fixed_units`: {fixed id: [units or keys]}, the fixed areas' shares of `flows`. Returns
    ({lead id: [units]}, [log lines])."""
    by = {u.id: u for u in units}
    # simplifying can leave invalid rings, which the unions below fail on (T-897, China's counties)
    simple = {u.id: polygonal(u.geom.simplify(500)) for u in units}
    adj = unit_borders(simple)
    cen = {}  # each free unit's people centre
    for i in free:
        x, y, p = by[i].px
        s = float(p.sum())
        c = simple[i].representative_point() if s <= 0 else None
        cen[i] = (c.x, c.y) if c is not None else (float((x * p).sum()) / s, float((y * p).sum()) / s)
    zones = {i: [i] for i in free}
    owner = {i: i for i in free}
    lines, shapes = [], {}

    def ppl(z):
        return sum(by[i].free for i in zones[z])

    def centre(z):
        n = ppl(z)
        if n <= 0:
            return cen[zones[z][0]]
        return (sum(cen[i][0] * by[i].free for i in zones[z]) / n, sum(cen[i][1] * by[i].free for i in zones[z]) / n)

    def geo(z):  # a zone's lead and unit count name its shape (a cut's halves drop theirs)
        k = (z, len(zones[z]))
        if k not in shapes:
            shapes[k] = union_all([simple[i] for i in zones[z]])
        return shapes[k]

    def nbrs(z):
        out = {}
        for i in zones[z]:
            for j, ln in adj[i].items():
                o = owner.get(j)
                if o is not None and o != z:
                    out[o] = out.get(o, 0.0) + ln
        return out

    km2 = {i: simple[i].area / 1e6 for i in free}

    def full(z):
        n = ppl(z)
        return n >= floor or (big_km2 and n >= ZONE_MIN_POP and sum(km2[i] for i in zones[z]) >= big_km2)

    farea = {k: ("f", f) for f, ks in (fixed_units or {}).items() for k in ks}
    flows = flows or {}

    def travel(z):
        """{area: travel} between zone z and every other zone (("z", lead)) and fixed area (("f", id))."""
        to, mine = {}, set(zones[z])
        for i in zones[z]:
            for j, n in flows.get(i, {}).items():
                if j in mine:
                    continue
                a = ("z", owner[j]) if j in owner else farea.get(j)
                if a is not None:
                    to[a] = to.get(a, 0.0) + n
        return to

    def hull(g):
        h = g.convex_hull.area
        return g.area / h if h else 1.0

    def accept(z, o):
        """(passes, how): o gets PARTNER_SHARE of z's travel with its biggest partner, or the union of
        the two is compact (T-1140)."""
        to = travel(z)
        top = max(to.values(), default=0.0)
        r = to.get(("z", o), 0.0) / top if top else 0.0
        if r >= PARTNER_SHARE:
            return True, f"partner {r:.2f}"
        bigger = max((geo(z), geo(o)), key=lambda g: g.area)
        c = hull(union_all([geo(z), geo(o)]))
        if c >= min(COMPACT_HULL, hull(bigger) - COMPACT_SLACK):
            return True, f"compact {c:.2f}"
        return False, f"partner {r:.2f}, compact {c:.2f}"

    # each fixed area's border with the free units and the other fixed areas (its land border), and
    # each free unit's border along it
    fx = [dict(f) for f in fixed]
    ftree = STRtree([simple[i] for i in free]) if free else None
    for f in fx:
        f["near"] = f["geom"].buffer(1000)
        f["units"] = {}
        for b in (ftree.query(f["near"], predicate="intersects") if ftree else ()):
            i = free[b]
            ln = simple[i].boundary.intersection(f["near"]).length
            if ln > 0:
                f["units"][i] = ln
    for f in fx:
        land = 0.0
        if f["units"]:
            land = f["geom"].boundary.intersection(union_all([simple[i] for i in f["units"]]).buffer(1000)).length
        for o in fx:
            if o is not f and o["geom"].intersects(f["near"]):
                land += f["geom"].boundary.intersection(o["near"]).length
        f["land"] = land
    fx = [f for f in fx if f["land"] > 0]

    def wrapped(gq, land, hull, near, b):
        c0, c1, c2, c3 = gq.bounds
        if c0 > b[2] or c2 < b[0] or c1 > b[3] or c3 < b[1] or not land or not gq.intersects(near):
            return False
        return gq.intersection(hull).area >= WRAP_HULL * gq.area and gq.boundary.intersection(near).length >= WRAP_BORDER * land

    def wraps(z, o):
        """T-1005, T-1087: whether the union of z and o wraps another zone or a fixed area: half of its
        area in the union's convex hull, the union holding half of its border with zones and fixed
        areas (coast, dead space and other countries do not count)."""
        m = union_all([geo(z), geo(o)])
        hull, near = m.convex_hull, m.buffer(1000)
        b = near.bounds
        for q in zones:
            if q not in (z, o):
                gq = geo(q)
                c0, c1, c2, c3 = gq.bounds
                if c0 > b[2] or c2 < b[0] or c1 > b[3] or c3 < b[1]:
                    continue
                land = sum(nbrs(q).values()) + sum(f["units"].get(i, 0.0) for f in fx for i in zones[q])
                if wrapped(gq, land, hull, near, b):
                    return True
        return any(wrapped(f["geom"], f["land"], hull, near, b) for f in fx)

    def merge(only=None):
        done = set()
        while True:
            small = sorted((ppl(z), z) for z in zones if z not in done and (only is None or z in only) and not full(z))
            if not small:
                break
            n, z = small[0]
            g = geo(z)
            nb = nbrs(z)
            same = [o for o in zones if o != z and region[o] == region[z]]
            border = {o: ln for o, ln in nb.items() if region[o] == region[z]}
            # T-788: Rijeka and Istria (0.46M) took all of Slavonia across the region line, a zone 460 km
            # long; one near the floor stays as it is rather than join another region's
            if not border and n < 0.8 * floor:
                border = dict(nb)
            island = all(j in zones[z] for i in zones[z] for j in adj[i])
            into, how, why = None, "", "no zone beside it"
            if border and n < ZONE_MIN_POP:
                to = travel(z)
                into = max(border, key=lambda o: (to.get(("z", o), 0.0), border[o]))
                how = f"biggest partner, shared border {border[into] / 1000:.0f} km"
            elif border:
                c = centre(z)
                fit = sorted((o for o in border if n + ppl(o) <= cap), key=lambda o: math.dist(c, centre(o)))
                how = "nearest centre"
                if not fit and n >= 0.8 * floor:
                    why = "no neighbour within the cap"
                elif not fit:  # nothing fits: the smallest, so no zone grows far past the cap
                    fit = sorted(border, key=lambda o: (ppl(o), -border[o]))
                    how = "smallest neighbour, none within the cap"
                ok = {o: accept(z, o) for o in fit}
                good = [o for o in fit if ok[o][0]]
                if fit and not good and n < 0.8 * floor:
                    # T-1140: as when its region has no zone left, another region's neighbour (Kootenai
                    # ID with Spokane)
                    other = sorted((o for o in nb if o not in border and n + ppl(o) <= cap), key=lambda o: math.dist(c, centre(o)))
                    ok.update({o: accept(z, o) for o in other})
                    good = [o for o in other if ok[o][0]]
                    if good:
                        how += ", none of its region passing"
                part = [o for o in good if ok[o][1].startswith("partner")]
                if fit and not good:
                    why = "no neighbour is a partner or makes a compact zone"
                fit = part or good
                if fit:
                    into = next((o for o in fit if not wraps(z, o)), None)
                    if into is not None:
                        how += (" that wraps nothing" if into != fit[0] else "") + f", {ok[into][1]}"
                    elif into is None and n < 0.8 * floor and only is None:
                        into, how = fit[0], "every join wraps; left to the ring cut"
                    elif into is None:  # a cut's half joining anyway would only rebuild the ring
                        why = "every join would wrap"
            elif contiguous:
                # T-897 (Anita, 2026-10-22): no exclaves; across water only as far as a bridge or a narrow
                # strait reaches, to the nearest zone, its own region's first; another region's only under
                # 80% of the floor, as by land (T-1015: South Jersey took Delaware's zone across the bay)
                pools = [same] + ([[o for o in zones if o != z and region[o] != region[z]]] if n < 0.8 * floor else [])
                for pool in pools:
                    near = [(d, o) for d, o in ((g.distance(geo(o)), o) for o in pool) if d <= WATER_GAP_KM * 1000]
                    if near:
                        into = min(near)[1]
                        how = f"across water, {min(near)[0] / 1000:.0f} km"
                        break
                if into is None and island:
                    # an archipelago is one zone (Penghu, Kinmen, Matsu): an island joins the nearest
                    # island zone of its region within ARCHIPELAGO_KM, never the mainland's
                    def alone(o):
                        return all(j in zones[o] for i in zones[o] for j in adj[i])
                    near = sorted((g.distance(geo(o)), o) for o in same if g.distance(geo(o)) <= ARCHIPELAGO_KM * 1000)
                    near = [x for x in near if alone(x[1])]
                    if near:
                        into = near[0][1]
                        how = f"archipelago, {near[0][0] / 1000:.0f} km"
            elif island:
                reach = ISLAND_KM if n < ZONE_MIN_POP else 10.0
                near = [(ppl(o), o) for o in same if g.distance(geo(o)) <= reach * 1000]
                if near:
                    into = max(near)[1]
                    how = f"island, {g.distance(geo(into)) / 1000:.0f} km"
            if into is None:
                done.add(z)
                lines.append(f"  {by[z].name} ({n:,.0f}) stays as it is: {'an island with no zone within reach' if island else why}")
                continue
            if region[into] != region[z]:
                how += ", another region"
            lines.append(f"  {by[z].name} ({n:,.0f}) joins {by[into].name} ({how})")
            for i in zones[z]:
                owner[i] = into
            zones[into] += zones.pop(z)

    def rings():
        """[(strength, zone, (kind, what))]: each zone wrapping a zone ("z", lead) or a fixed area
        ("f", index), the strongest first."""
        shp = [("z", k) for k in zones] + [("f", n) for n in range(len(fx))]
        gs = [geo(k) if t == "z" else fx[k]["geom"] for t, k in shp]
        nears = [g.buffer(1000) for g in gs]
        tree = STRtree(gs)
        nb = [{} for _ in shp]  # a: {b: b's border along a}
        for a in range(len(shp)):
            for b in tree.query(nears[a], predicate="intersects"):
                if b != a:
                    nb[a][int(b)] = gs[b].boundary.intersection(nears[a]).length
        out = []
        for a, (t, k) in enumerate(shp):
            if t != "z":
                continue
            hull = gs[a].convex_hull
            for b, ln in nb[a].items():
                land = sum(nb[x].get(b, 0.0) for x in nb[b])
                if not land or not gs[b].area:
                    continue
                ih = gs[b].intersection(hull).area / gs[b].area
                if ih >= WRAP_HULL and ln / land >= WRAP_BORDER:
                    out.append((ih + ln / land, k, shp[b]))
        return sorted(out, reverse=True)

    def cut(z, c):
        """Zone z's units by direction from point c, the circle opened at the widest gap, split where the
        two halves' people are most even with both in one piece; None when no split is."""
        ang = sorted((math.atan2(cen[i][1] - c[1], cen[i][0] - c[0]), i) for i in zones[z])
        gaps = [(ang[(a + 1) % len(ang)][0] - ang[a][0]) % (2 * math.pi) for a in range(len(ang))]
        start = (max(range(len(gaps)), key=lambda a: gaps[a]) + 1) % len(ang)
        order = [ang[(start + a) % len(ang)][1] for a in range(len(ang))]
        links = {i: set(adj[i]) for i in order}
        if contiguous:  # the zone's own joins across water hold it together too
            t = STRtree([simple[i] for i in order])
            for a, i in enumerate(order):
                for b in t.query(simple[i], predicate="dwithin", distance=WATER_GAP_KM * 1000):
                    links[i].add(order[b])
        tot, run, best = sum(by[i].free for i in order), 0.0, None
        for s in range(1, len(order)):
            run += by[order[s - 1]].free
            if best is None or abs(run - tot / 2) < best[0]:
                a, b = order[:s], order[s:]
                if one_piece(a, links) and one_piece(b, links):
                    best = (abs(run - tot / 2), a, b)
        return best[1:] if best else None

    def one_piece(ids, links):
        ids = set(ids)
        start = next(iter(ids))
        seen, front = {start}, [start]
        while front:
            nxt = []
            for i in front:
                for j in links[i]:
                    if j in ids and j not in seen:
                        seen.add(j)
                        nxt.append(j)
            front = nxt
        return len(seen) == len(ids)

    def name(t, k):
        return by[k].name if t == "z" else fx[k]["name"]

    merge()
    for rnd in range(RING_ROUNDS):
        rs = rings()
        if not rs:
            break
        touched = set()
        for _, a, (t, b) in rs:
            if a in touched or a not in zones or (t == "z" and (b not in zones or b in touched)):
                continue
            got = cut(a, fx[b]["xy"] if t == "f" else centre(b)) if len(zones[a]) > 1 else None
            if not got and t == "z" and ppl(a) + ppl(b) <= cap:
                # one unit round a city (Wrocławski round Wrocław) takes what it wraps
                lines.append(f"  ring cut {rnd + 1}: {by[a].name} wraps {by[b].name}, joined")
                for i in zones[b]:
                    owner[i] = a
                zones[a] += zones.pop(b)
                touched |= {a, b}
                continue
            if not got:
                lines.append(f"  ring cut {rnd + 1}: {by[a].name} wraps {name(t, b)}, no cut keeps both halves in one piece")
                continue
            zones.pop(a)
            for h in got:
                zones[h[0]] = list(h)
                shapes.pop((h[0], len(h)), None)
                for i in h:
                    owner[i] = h[0]
                touched.add(h[0])
            lines.append(f"  ring cut {rnd + 1}: {by[a].name} wraps {name(t, b)}, cut into {by[got[0][0]].name} "
                         f"({len(got[0])} units) and {by[got[1][0]].name} ({len(got[1])})")
        merge(only=touched)
    for _, a, (t, b) in rings():
        lines.append(f"  ring left: {by[a].name} wraps {name(t, b)}")
    return zones, lines


def zone_name(s, iso=None):
    """A unit's name without its kind ("Rostock, Kreisfreie Stadt", "Arr. Leuven") or second language."""
    import re
    s = s.split("/")[0].strip()
    s = re.sub(r",\s*(Kreisfreie Stadt|Landkreis|Stadtkreis|Kreis)$", "", s)
    if iso == "CHN":  # T-897: "Xinhui District", "Weichang Manchu and Mongol Autonomous County"
        s = re.sub(r"\s+((\w+ )*(and \w+ )?Autonomous (County|Banner)|County|District|City|Banner|New Area|Forest Area|Special District)$", "", s)
    s = re.sub(r"^(Landkreis|Kreis|Regionalverband|Arr\.|Arrondissement|Bezirk)\s+", "", s)
    if iso in ("USA", "PRI"):  # census_units' "Kern County, CA"
        s = re.sub(r", [A-Z]{2}$", "", s)
    return s.strip()


def solve_region(iso, u):
    """The joint commute solve region of a unit's row (SOLVE_REGIONS), or None for a country solved whole."""
    if iso not in SOLVE_REGIONS or not u.parents:
        return None
    first = u.parents[0][1]
    return SOLVE_REGIONS[iso].get(first) or ascii_name(first.split()[0]).lower()


def commute_flows(iso):
    """{home unit: {work unit: workers}} from COMMUTE_FLOWS, or None for a country without. The ACS
    table gives Connecticut by planning region, which matches no 2020 county."""
    path = COMMUTE_FLOWS.get(iso)
    if not path:
        return None
    import pandas as pd
    t = pd.read_excel(path, header=None, skiprows=8, dtype=str).iloc[:, [0, 1, 4, 5, 8]]
    t.columns = ["hs", "hc", "ws", "wc", "n"]
    t = t.dropna(subset=["hs", "hc", "ws", "wc"])
    out = collections.defaultdict(dict)
    for h, w, n in zip(t.hs.str.zfill(2) + t.hc.str.zfill(3), t.ws.str[-2:] + t.wc.str.zfill(3),
                       pd.to_numeric(t.n, errors="coerce").fillna(0)):
        out[h][w] = out[h].get(w, 0.0) + float(n)
    return out


def both_ways(flows, ids):
    """Home-to-work `flows` as travel between distinct units of `ids`, both directions summed."""
    ids = set(ids)
    out = collections.defaultdict(dict)
    for h, row in flows.items():
        if h not in ids:
            continue
        for w, n in row.items():
            if w != h and w in ids and n:
                out[h][w] = out[h].get(w, 0.0) + n
                out[w][h] = out[w].get(h, 0.0) + n
    return out


def gravity_flows(points):
    """Travel both ways between distinct keys of {key: (x, y, people)} (Mollweide m): people summed on
    GRAVITY_KM cells, people x people x d^-0.8 exp(-d/30 km) over cell pairs within GRAVITY_REACH_KM,
    d at least 2 km (T-1140)."""
    from scipy.spatial import cKDTree
    keys = [k for k, (_, _, p) in points.items() if len(p) and float(p.sum()) > 0]
    xs, ys, ps, us = [], [], [], []
    m = GRAVITY_KM * 1000.0
    for n, k in enumerate(keys):
        x, y, p = points[k]
        _, inv = np.unique(np.floor(x / m) * 1e7 + np.floor(y / m), return_inverse=True)
        s = np.bincount(inv, weights=p)
        xs.append(np.bincount(inv, weights=x * p) / s)
        ys.append(np.bincount(inv, weights=y * p) / s)
        ps.append(s)
        us.append(np.full(len(s), n, np.int64))
    out = collections.defaultdict(dict)
    if not keys:
        return out
    x, y, p, u = (np.concatenate(a) for a in (xs, ys, ps, us))
    t = cKDTree(np.c_[x, y])
    d = t.sparse_distance_matrix(t, GRAVITY_REACH_KM * 1000.0, output_type="coo_matrix")
    a, b, dist = d.row, d.col, d.data
    keep = u[a] != u[b]
    a, b, dist = a[keep], b[keep], np.maximum(dist[keep] / 1000.0, 2.0)
    w = p[a] * p[b] * dist ** -0.8 * np.exp(-dist / 30.0)
    pair, inv = np.unique(u[a] * len(keys) + u[b], return_inverse=True)
    for k, v in zip(pair, np.bincount(inv, weights=w)):
        out[keys[k // len(keys)]][keys[k % len(keys)]] = float(v)
    return out


def belts_share(flows, h, ids):
    """The share of unit h's resident workers working in units `ids`."""
    f, ids = flows.get(h, {}), set(ids)
    tot = sum(f.values())
    return sum(n for w, n in f.items() if w in ids) / tot if tot else 0.0


def metro_belts(flows, by, free, grow, areas, cap=CAP):
    """T-1088 (Anita, 2026-10-23: "we can let metros take neighboring counties but it doesnt have to
    be a hard rule"): a free unit sending at least BELT_SHARE of its resident workers to the metro it
    sends most to joins that metro, when the metro is one of `grow` (unbuilt rows), the unit touches it
    by land (or touches a unit joining it) and the metro stays within `cap`. A unit that cannot join
    stays free for the zones. `areas`: {unit: metro row or pack id} of every metro in the country.
    Returns {row id: [units added]}."""
    best = {}
    for h in free:
        f = flows.get(h, {})
        tot = sum(f.values())
        to = collections.Counter()
        for w, n in f.items():
            if w in areas and w != h:
                to[areas[w]] += n
        if tot and to:
            a, n = to.most_common(1)[0]
            if a in grow and n >= BELT_SHARE * tot:
                best[h] = a
    simple = {}

    def shp(i):
        if i not in simple:
            simple[i] = polygonal(by[i].geom.simplify(500))
        return simple[i]
    out = {}
    for a, r in grow.items():
        cand = sorted((i for i, b in best.items() if b == a), key=lambda i: -by[i].free)
        if not cand:
            continue
        reach = union_all([shp(i) for us in r["units"].values() for i in us if i in by]).buffer(1000)
        n, added = r["people"], []
        while True:
            nxt = [i for i in cand if i not in added and reach.intersects(shp(i))]
            nxt = [i for i in nxt if n + by[i].free <= cap]
            if not nxt:
                break
            for i in nxt:
                if n + by[i].free <= cap:
                    added.append(i)
                    n += by[i].free
            reach = union_all([reach] + [shp(i).buffer(1000) for i in nxt if i in added])
        if added:
            out[a] = added
    return out


def plan_zones(isos3, out_path=None):
    """Whole-country cover (T-723): for each country, its units held by no plan row and by no pack
    (half their people), those under ZONE_MIN_DENSITY left as dead space; where COMMUTE_FLOWS has
    the country's commuting, its unbuilt metro rows take their belts (metro_belts, T-1088); the rest
    grouped by zone_groups into zone rows (`kind` "zone"), written into the plan beside its metro
    rows. A country's earlier planned zone rows are replaced; one with the same units keeps its id
    and order. With `out_path`, the plan and its extents are written there instead (a dry run)."""
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
    changes = []
    for iso in isos3:
        units = [u for u in st["units"] if u.iso == iso]
        # a country no metro reached, so not in the plan's state (T-788); or one with its own zone
        # rules, read afresh since the state's units predate gb_units' de facto lines and gaps (T-897)
        if not units or iso in ZONE_RULES:
            europe, meta = europe_isos()
            whole = union_all([g for i, g in countries() if i == iso])
            units, got = country_units(iso, meta, load_nuts(europe | {iso}), pop, shapely.Polygon(), whole)
            if not units:
                raise SystemExit(f"{iso}: no units at NUTS-3 or in geoBoundaries")
            doc["unit_sources"][iso] = got
        else:  # the state's units predate de_facto (T-897)
            for u in units:
                u.geom = de_facto(iso, u.geom)
            gone = [u.name for u in units if u.geom is None or u.geom.is_empty]
            units = [u for u in units if u.geom is not None and not u.geom.is_empty]
            if gone:
                log(f"{iso}: {len(gone)} units held by another country on the ground, left out: {', '.join(gone)}")
        src = doc["unit_sources"][iso]
        rule = ZONE_RULES.get(iso, {})
        floor, cap = rule.get("floor", ZONE_FLOOR), rule.get("cap", ZONE_CAP)
        min_density = rule.get("min_density", ZONE_MIN_DENSITY)
        # a zone row the runner has moved on (built, gated, shipped, parked) keeps its units; only
        # planned ones are regrouped (T-788: Anita's 0.5M floor came mid-wave)
        old = {r["id"]: r for r in rows if r.get("kind") == "zone" and iso in r.get("units", {})
               and r.get("status", "planned") == "planned"}
        kept = [r for r in rows if r.get("kind") == "zone" and iso in r.get("units", {}) and r["id"] not in old]
        held = {uid for r in rows if r.get("units") and r["id"] not in old for uid in r["units"].get(iso, [])}
        ptree = STRtree([q["geom"] for q in pk.values()])
        pl = list(pk.values())
        free, dead, away, in_pack = [], [], [], {}
        only = rule.get("zoned_only", {})
        for u in units:
            if u.parents and u.parents[0][0] in only and u.id not in only[u.parents[0][0]]:
                if u.id not in held:
                    dead.append({"unit": u.id, "name": u.name, "people": round(u.pop), "outside_zoned": True})
                continue
            if iso in st["europe"]:
                lon, lat = INV.transform(*u.geom.representative_point().coords[0])
                if not (EUROPE_BOX[0] <= lon <= EUROPE_BOX[2] and EUROPE_BOX[1] <= lat <= EUROPE_BOX[3]):
                    away.append(u.name)
                    continue
            x, y, p = pop.inside(u.geom)
            inpk = np.zeros(len(x), bool)
            most = (0.0, None)
            for i in ptree.query(u.geom, predicate="intersects"):
                here = contains_xy(pl[i]["geom"], x, y)
                inpk |= here
                most = max(most, (float(p[here].sum()), pl[i]["id"]), key=lambda m: m[0])
            u.px = (x[~inpk], y[~inpk], p[~inpk])
            u.free = float(p[~inpk].sum())
            tot = float(p.sum())
            if tot > 0 and u.free < UNIT_SHARE * tot:
                in_pack[u.id] = most[1]
            if u.id in held or (tot > 0 and u.free < UNIT_SHARE * tot) or u.free <= 0:
                continue
            dens = u.free / (u.geom.area / 1e6)
            if dens < min_density and u.free < DEAD_KEEP * floor:
                dead.append({"unit": u.id, "name": u.name, "people": round(u.free), "per_km2": round(dens, 1)})
            else:
                free.append(u.id)
        by = {u.id: u for u in units}
        # T-1088: unbuilt metro rows take the free units commuting into them
        metros = {r["id"]: r for r in rows if r.get("kind") != "zone" and iso in (r.get("units") or {})}
        grow = {k: r for k, r in metros.items() if r.get("status", "planned") == "planned"}
        flows = commute_flows(iso)
        grown, belts = {}, {}
        if flows and grow:
            zone_ids = {r["id"] for r in rows if r.get("kind") == "zone"}
            areas = {i: k for i, k in in_pack.items() if k is not None and k not in zone_ids}
            areas.update({i: k for k, r in metros.items() for i in r["units"][iso]})
            belts = metro_belts(flows, by, free, grow, areas)
            for k, add in belts.items():
                r = json.loads(json.dumps(grow[k]))
                us = [by[i] for i in r["units"][iso]] + [by[i] for i in add]
                g = polygonal(union_all([u.geom for u in us]))
                cut = [q["geom"] for q in pk.values() if q["geom"].intersects(g)]
                ll = to_ll(polygonal(g.difference(union_all(cut))) if cut else g)
                had = r.get("source", {}).get("belt", {}).get("units", [])
                r["units"][iso] = r["units"][iso] + add
                r["people"] = round(r["people"] + sum(by[i].free for i in add))
                r["bbox"] = [round(v, 3) for v in ll.bounds]
                r["extracts"] = extracts(ll, ex, kids)
                r.setdefault("source", {})["belt"] = {"task": "T-1088", "share": BELT_SHARE, "units": had + add}
                grown[k] = (r, {"type": "Feature", "properties": {"order": r["order"], "id": k, "name": r["name"], "people": r["people"]},
                                "geometry": mapping(shapely.set_precision(ll, 1e-5))})
                log(f"  {r['name']} takes {len(add)} units commuting into it, {sum(by[i].free for i in add) / 1e6:.2f}M: "
                    + ", ".join(f"{by[i].name} {belts_share(flows, i, grow[k]['units'][iso]):.0%}" for i in add))
            free = [i for i in free if not any(i in add for add in belts.values())]
        # the metro rows, the zone rows kept and the country's built packs, which no zone may wrap
        fixed, cc = [], {c["id"]: c.get("country") for c in json.loads(REGISTRY.read_text(encoding="utf-8"))["cities"]}
        # the fixed areas' shares of the travel (T-1140): their units, and a pack's own people
        fixed_units, points = collections.defaultdict(list), {}
        for i, k in in_pack.items():
            if k is not None:
                fixed_units[k].append(i)
        for r in rows:
            mine = (grown[r["id"]][0] if r["id"] in grown else r).get("units", {}).get(iso)
            if r["id"] in old or not (mine or (not r.get("units") and r["id"] in pk and cc.get(r["id"]) == a2.get(iso))):
                continue
            fixed_units[r["id"]] += [i for i in mine or () if i in by]
            if r["id"] in pk:
                g = pk[r["id"]]["geom"]
                x, y, p = pop.inside(g)
                points[("pack", r["id"])] = (x, y, p)
                fixed_units[r["id"]].append(("pack", r["id"]))
            else:
                g = union_all([by[i].geom for i in mine if i in by])
                px = [by[i].px for i in mine if i in by and by[i].px is not None] or [(np.zeros(0),) * 3]
                x, y, p = (np.concatenate([q[k] for q in px]) for k in range(3))
            s = float(p.sum())
            xy = (float((x * p).sum()) / s, float((y * p).sum()) / s) if s > 0 else g.representative_point().coords[0]
            fixed.append({"id": r["id"], "name": r["name"], "geom": polygonal(g.simplify(500)), "xy": xy})
        nuts = src["kind"] in ("NUTS3", "ITL3")
        region = {u.id: (u.parents[-1] if nuts else u.parents[0]) if u.parents else iso for u in units}
        if flows:
            travel = both_ways(flows, list(by))
        else:
            points.update({u.id: u.px for u in units if u.px is not None})
            travel = gravity_flows(points)
        big = rule.get("big_km2", BIG_ZONE_KM2)
        groups, lines = zone_groups(units, free, region, floor, cap, rule.get("contiguous", False), fixed, travel,
                                    fixed_units, big)
        if rule.get("contiguous"):  # T-897: no pack for an islet of a few hundred people that can join nothing
            for lead in [k for k, ids in groups.items() if sum(by[i].free for i in ids) < ZONE_ALONE_MIN]:
                for i in groups.pop(lead):
                    dead.append({"unit": i, "name": by[i].name, "people": round(by[i].free), "alone": True})
        log(f"{iso}: {len(units)} units, {len(held)} in plan rows ({len(kept)} zone rows kept), {len(free)} free, {len(dead)} dead space "
            f"(under {min_density:g} a km2: {', '.join(d['name'] for d in dead)}); {len(groups)} zones"
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
            if rule.get("contiguous"):
                parts = sorted(shapely.get_parts(g), key=lambda p: -p.area)
                far = [p for p in parts[1:] if p.distance(parts[0]) > WATER_GAP_KM * 1000 and p.area > 1e6]
                if far:
                    log(f"  {us[0].name}: {len(far)} parts over {WATER_GAP_KM:g} km from its main part "
                        f"({sum(p.area for p in far) / 1e6:,.0f} km2; a unit's own exclave or a pack's cut)")
            ll = to_ll(g)
            best = None
            for u in us:
                x, y, p = u.px
                if len(p) and (best is None or p.max() > best[0]):
                    i = int(np.argmax(p))
                    best = (float(p[i]), x[i], y[i])
            lon, lat = INV.transform(best[1], best[2])
            tz = COUNTRY_TZ.get(iso) or tf.timezone_at(lng=lon, lat=lat) or tf.closest_timezone_at(lng=lon, lat=lat)
            # the most populous unit with a name in Latin script (Taiwan's file has some in Chinese only)
            named = [u for u in us if latin(zone_name(u.name, iso))] or us
            name = zone_name(named[0].name, iso)
            if name in names and iso in ("USA", "PRI"):  # Jefferson County in many states: with its state
                name = named[0].name
            elif name in names:  # Leipzig the city and Leipzig the Landkreis: the smaller by its two largest units
                name = ", ".join(zone_name(u.name, iso) for u in named[:2]) if len(named) > 1 else named[0].name
            names.add(name)
            prev = was.get(tuple(sorted(ids)))
            # zones: global sources and the zone curve, no country module (T-779), but where the rule names one
            arch, module = (ARCHETYPE[iso][0], rule["module"]) if rule.get("module") else ("D", None)
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
            sr = solve_region(iso, us[0])
            if sr:
                row["solve_region"] = sr
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
        rec = {"built": str(datetime.now().date()), "task": rule.get("task", "T-897" if iso in ZONE_RULES else "T-788"),
               "rule": f"free units at or over {min_density:g} people a km2 (or holding {DEAD_KEEP * floor:,.0f}), grouped in their region until each "
                       f"zone holds {floor:,}"
                       + (f" (or {ZONE_MIN_POP:,} over {big:,.0f} km2)" if big else "")
                       + f": under {ZONE_MIN_POP:,} into the zone it travels with most, then a neighbour keeping both within {cap:,} "
                       f"that is a travel partner ({PARTNER_SHARE:.0%} of its biggest partner's travel), else one making a compact zone "
                       f"({COMPACT_HULL:.0%} of its convex hull, or within {COMPACT_SLACK:.2f} of the bigger part's), partners first, "
                       f"each by nearest people centre, whose union wraps no zone, metro or pack (T-1140, T-1087); under "
                       f"{0.8 * floor:,.0f} another region's passing neighbour when its own region has none; nothing passing, it stays; "
                       f"islands the most populous zone within {ISLAND_KM:g} km (10 km from {ZONE_MIN_POP:,}); then each zone still "
                       f"wrapping one cut across it and its halves regrouped; travel "
                       + ("from the commuting table" if flows else f"by gravity on GHSL people ({GRAVITY_KM:g} km cells, within {GRAVITY_REACH_KM:g} km)")
                       + (f"; unbuilt metros first take neighbouring units sending {BELT_SHARE:.0%} of their workers into them (T-1088)" if flows and grow else ""),
               "kept": [r["id"] for r in kept], "dead": dead, "merges": [ln.strip() for ln in lines]}
        if belts:
            rec["belts"] = belts
        if away:
            rec["outside_europe"] = away
        changes.append({"iso": iso, "old": {k: r["units"][iso] for k, r in old.items()},
                        "metros": {k: grow[k]["units"][iso] for k in grown}, "grown": {k: v[0] for k, v in grown.items()},
                        "out": out, "record": rec, "src": src,
                        "feats": feats + [v[1] for v in grown.values()],
                        "drop": set(old) | {f["properties"]["id"] for f in feats} | set(grown)})
        rows[:] = [grown[r["id"]][0] if r["id"] in grown else r for r in rows if r["id"] not in old] + out
    assert all(plan_config.is_plan_row(r) for r in rows if r.get("kind") == "zone")
    if out_path:  # a dry run: the plan and its extents beside each other, the real ones untouched
        doc, ext = json.loads(OUT.read_text(encoding="utf-8")), json.loads((WORK / "extents.geojson").read_text(encoding="utf-8"))
        splice(doc, ext, changes)
        Path(out_path).write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        Path(out_path).with_suffix(".extents.geojson").write_text(json.dumps(ext), encoding="utf-8")
        log(f"written {out_path} (a dry run)")
        return
    import shared_lock
    # the runners set row statuses meanwhile: the plan is read again under the lock and only this
    # country's planned rows replaced (T-1088)
    with shared_lock.held("plan", f"world_plan.py zones {','.join(isos3)}"):
        doc = shared_lock.read_json(OUT)
        ext_path = WORK / "extents.geojson"
        ext = shared_lock.read_json(ext_path)
        splice(doc, ext, changes)
        shared_lock.write_atomic(ext_path, json.dumps(ext))
        shared_lock.write_atomic(OUT, json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    log(f"written {OUT}")


def splice(doc, ext, changes):
    """plan_zones' changes into a plan read afresh: each country's planned zone rows replaced, its grown
    metro rows swapped in where they stand; refused when any of those rows moved on meanwhile."""
    rows = doc["metros"]
    for c in changes:
        iso = c["iso"]
        now = {r["id"]: r for r in rows}
        for k, us in {**c["old"], **c["metros"]}.items():
            r = now.get(k)
            if r is None or r.get("status", "planned") != "planned" or r["units"].get(iso) != us:
                raise SystemExit(f"{iso}: row {k} changed since the plan was read; run again")
        taken = {r["order"] for r in rows if r["id"] not in c["old"] and r.get("order") is not None}
        clash = [r["id"] for r in c["out"] if r["id"] not in c["old"] and (r["id"] in now or r["order"] in taken)]
        if clash:
            raise SystemExit(f"{iso}: new rows clash with rows added since the plan was read: {clash}")
        rows[:] = [c["grown"].get(r["id"], r) for r in rows if r["id"] not in c["old"]] + c["out"]
        doc["unit_sources"][iso] = c["src"]
        doc.setdefault("zones", {})[iso] = c["record"]
        ext["features"] = [f for f in ext["features"] if f["properties"]["id"] not in c["drop"]] + c["feats"]


def metro_edits(iso, units, taken, rows):
    """METRO_EDITS' rows for plan(): each from its listed units, which no planned row may hold."""
    by = {u.id: u for u in units}
    held = {u.id for r in rows for u in r["units"]}
    out = []
    for name, ids in METRO_EDITS.get(iso, []):
        if held & set(ids):
            raise SystemExit(f"{iso}: the edit {name} takes units a metro holds: {sorted(held & set(ids))}")
        us = [by[i] for i in ids]
        m = {"key": f"edit-{name}", "name": name, "iso": iso, "region": "world", "alt_names": [],
             "source": {"plan_edit": "T-1004"}}
        out.append({"metro": m, "units": us, "people": people(us, taken), "core": densest(us, taken),
                    "name": name, "split": False})
    return out


def plan_metros(isos3):
    """A FUA country's metro rows planned alone (FUA_COUNTRIES, T-1015): its FUAs over 1M against
    its own units (no other country's lent, none of its lent to others), with METRO_EDITS, written
    beside every other row, which stay as they are. A rerun keeps each earlier row's order and id
    (keep_order), and a row the runner moved on its status while its units are the same."""
    import plan_config
    pop = Pop()
    ne = countries()
    pk = packs()
    doc = json.loads(OUT.read_text(encoding="utf-8"))
    rows = doc["metros"]
    ext_path = WORK / "extents.geojson"
    ext = json.loads(ext_path.read_text(encoding="utf-8"))
    for iso in isos3:
        if iso not in FUA_COUNTRIES:
            raise SystemExit(f"{iso} is not a FUA country (FUA_COUNTRIES)")
        ms = country_metros(pop, ne, iso)
        units, src = country_units(iso, None, None, pop, shapely.Polygon())
        shares(units, ms, list(pk.values()))
        near = collections.defaultdict(list)
        for u in units:
            for key in u.share:
                near[key].append(u)
        for m in ms:
            m["units_near"] = near[m["key"]]
        st = {"ms": ms, "units": units, "packs": pk}
        planned, dropped = plan(st, pk)
        planned += metro_edits(iso, units, pk, planned)
        for r in planned:
            r["name"] = METRO_NAMES.get(iso, {}).get(r["name"], r["name"])
        mine = [r for r in rows if alone_row(r) and r["countries"] == [iso]]
        others = [r for r in rows if r not in mine]
        out, feats = finish(st, planned, pk, prev=[r for r in mine if plan_config.is_plan_row(r)], others=others)
        was = {r["id"]: r for r in mine}
        by = {u.id: u for u in units}
        for r in out:
            lead = max((by[i] for i in r["units"][iso]), key=lambda u: u.pop)
            sr = solve_region(iso, lead)
            if sr:
                r["solve_region"] = sr
            o = was.get(r["id"])
            if o and o["units"] == r["units"]:
                for k in ("status", "run", "gate_relax"):
                    if k in o:
                        r[k] = o[k]
            log(f"  {r['order']:>4} {r['id']} {r['name']}: {r['people'] / 1e6:.2f}M, {len(r['units'][iso])} units, "
                f"{r.get('solve_region')}, extracts {r['extracts']}")
        gone = [r["id"] for r in mine if r["id"] not in {o["id"] for o in out}]
        if gone:
            log(f"  earlier rows with no successor, removed: {gone}")
        for m, why in dropped:
            log(f"  dropped {m['name']}: {why}")
        rows[:] = others + out
        doc["unit_sources"][iso] = src
        doc["dropped"] = [d for d in doc.get("dropped", []) if d.get("country") != iso] + [
            {"name": m["name"], "source": m["source"], "reason": why, "country": iso} for m, why in dropped]
        drop = {r["id"] for r in mine} | {f["properties"]["id"] for f in feats}
        ext["features"] = [f for f in ext["features"] if f["properties"]["id"] not in drop] + feats
        log(f"{iso}: {len(out)} metro rows, {sum(r['people'] for r in out) / 1e6:.1f}M people; {len(dropped)} dropped")
    tmp = ext_path.with_suffix(".geojson.part")
    tmp.write_text(json.dumps(ext), encoding="utf-8")
    os.replace(tmp, ext_path)
    tmp = OUT.with_suffix(".json.part")
    tmp.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, OUT)
    log(f"written {OUT}")


def main(argv):
    import pickle
    if argv and argv[0] == "metros":
        a3 = {v: k for k, v in iso2s().items() if k in FUA_COUNTRIES}
        return plan_metros([a3.get(c, c) for c in argv[argv.index("--countries") + 1].split(",")])
    if argv and argv[0] == "zones":
        have = json.loads(OUT.read_text(encoding="utf-8"))["unit_sources"]
        # several territories share an alpha-2 in Natural Earth (FR: France and Clipperton, T-779)
        a3 = {v: k for k, v in sorted(iso2s().items(), key=lambda kv: kv[0] in have)}
        cs = argv[argv.index("--countries") + 1].split(",")
        return plan_zones([a3[c] for c in cs], argv[argv.index("--out") + 1] if "--out" in argv else None)
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
    have = json.loads(OUT.read_text(encoding="utf-8"))["metros"] if OUT.exists() else []
    prev = [r for r in have if plan_config.is_plan_row(r) and not alone_row(r)]
    out, feats = finish(st, rows, prev=prev, others=[r for r in have if r.get("kind") == "zone" or alone_row(r)])
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
    if sys.argv[1:2] == ["zones"]:  # takes the lock only to write (T-1088): its minutes would stall the runners
        main(sys.argv[1:])
    else:  # a replan rewrites the plan whole while runners set row statuses (T-786)
        with shared_lock.held("plan", "world_plan.py " + " ".join(sys.argv[1:])[:200]):
            main(sys.argv[1:])
