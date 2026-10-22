"""Singapore: the 1,641,590 non-residents the census does not ask (sources/sg.md, "Coverage gap").

    python sources/sg_nonres.py --build     write data/geo/sg/sg_places.gpkg and
                                            data/normalized/sg_nonres_sites.csv
    python sources/sg_nonres.py             print the groups, mixes and the national language totals

Every row is `modelled`. The stock comes from MOM's foreign workforce table and the dormitory
counts MOM and the press published in 2020; the nationality of each pass type is MOM's to publish
and it does not, so the mixes are the best published fragments and, where there are none, a
stated judgement (sg.md has the list and what is cited).

THE STOCK, June 2020 (SingStat's non-resident population, 1,641,590, the same date as the census):
  MOM, Foreign workforce numbers, Jun 2020 column (web.archive.org snapshot of
  mom.gov.sg/documents-and-publications/foreign-workforce-numbers, 2021-03-01):
    Employment Pass 189,700; S Pass 188,800; Work Permit 940,200 of which foreign domestic
    workers 252,600 and construction/marine/process 351,800; other work passes 33,100.
  Not in that table: dependants, long-term visit pass holders and students. They are what
  remains of 1,641,590 once the 1,351,800 workforce is taken out (289,790).
THE DORMITORIES, spring 2020 (SUTD "Rethinking dorms", 26 May 2020, from MOM): about 323,000
  workers in dormitories: 43 purpose-built dormitories (198,320 residents, Tan et al., JAMA 2021),
  about 1,200 factory-converted dormitories (95,000) and construction temporary quarters (20,000).
  TODAY (May 2020): 85,000 construction work permit and S Pass holders lived outside dormitories.
"""
import os
import sys
import json
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(HERE), str(ROOT / "taxonomy"), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RAW = ROOT / "data" / "raw" / "sg"
GEO = ROOT / "data" / "geo" / "sg"
SITES = ROOT / "data" / "normalized" / "sg_nonres_sites.csv"

NONRES = 1_641_590
EP, SP, WP_FDW, WP_CMP, OTHER_PASS = 189_700, 188_800, 252_600, 351_800, 33_100
WP_TOTAL = 940_200
WORKFORCE = 1_351_800
WP_OTHER = WP_TOTAL - WP_FDW - WP_CMP          # 335,800: manufacturing, services, retail, F&B
DEPENDANTS_STUDENTS = NONRES - WORKFORCE        # 289,790
DORM_ALL, DORM_PBD, DORM_FCD = 323_000, 198_320, 95_000
CMP_OUTSIDE = 85_000
CMP_DORM = WP_CMP - CMP_OUTSIDE                 # 266,800
OTHER_DORM = DORM_ALL - CMP_DORM                # 56,200

PBD_COUNT = 43
# Beds, cited: Aspri-Westlite Papan 7,900 (completed May 2016); Westlite Jalan Tukang 3,420 and
# Kranji Way 1,300 (Centurion Corporation, JTC quick-build dormitory tender announcement, SGX,
# 2 Sep 2020, "approximately" both)
BEDS = {"Aspri-Westlite Papan": 7900, "Westlite Jalan Tukang": 3420, "Kranji Way Dormitory": 1300}

TAMIL = "dravidian.southern.tamil"
ENGLISH = "indoeuropean.germanic.english"
MALAY = "austronesian.malayic.malay"
ST = "sinotibetan.sinitic"

# ---- nationality of each pass type: iso -> share (normalised below) --------------------------
# Dormitories: 1,089 workers at five dormitories and two community spaces, Aug-Nov 2022, PMC11471231
# (Building a healthy migrant workforce in Singapore): Bangladeshi 44%, Indian 40%, "other 16%
# (Chinese, Thai, Burmese, Filipino, Malaysian)". The 16% split below is uncited.
DORM_NAT = {"BD": 44, "IN": 40, "CN": 7, "MM": 4, "TH": 2, "MY": 2, "PH": 1}
# Migrant domestic workers: no breakdown published (MOM declines). Myanmar 30,000 of about 250,000
# (Inquirer, "Myanmar bans nationals from working as maids in Singapore", 2015); Indonesia and the
# Philippines are the two largest sources (TWC2 fact sheet, 2011); their split is uncited.
FDW_NAT = {"ID": 48, "PH": 40, "MM": 12}
# The rest are uncited judgement; MOM publishes no nationality for any pass.
WP_OTHER_NAT = {"MY": 25, "CN": 20, "IN": 15, "BD": 10, "PH": 8, "MM": 8, "TH": 4, "ID": 3,
                "LK": 3, "PK": 2, "VN": 2}
SP_NAT = {"MY": 28, "CN": 20, "IN": 22, "PH": 10, "ID": 3, "MM": 3, "TH": 3, "VN": 2, "KR": 1,
          "JP": 1, "LK": 2, "PK": 1, "NP": 1}
EP_NAT = {"IN": 27, "CN": 15, "MY": 10, "PH": 5, "JP": 5, "US": 5, "GB": 5, "AU": 3, "ID": 3,
          "KR": 3, "FR": 2, "DE": 2, "PK": 2, "CA": 1, "NZ": 1, "VN": 2, "TH": 2, "LK": 2, "TW": 1}
DEP_NAT = {"IN": 25, "CN": 22, "MY": 14, "ID": 7, "PH": 6, "VN": 4, "JP": 2, "KR": 2, "GB": 2,
           "US": 2, "FR": 2, "TH": 2, "PK": 2, "AU": 1, "MM": 1, "LK": 1, "DE": 1, "TW": 1}

# groups: name -> (people, nationality shares, where)
#   where: "dorm" (PBD / FCD sites) or "home" (planning areas by resident population)
GROUPS = {
    "Work permit and S Pass holders in dormitories": (DORM_ALL, DORM_NAT, "dorm"),
    "Construction, marine, process work permits outside dormitories": (CMP_OUTSIDE, DORM_NAT, "home"),
    "Other work permits outside dormitories": (WP_OTHER - OTHER_DORM, WP_OTHER_NAT, "home"),
    "S Pass holders": (SP, SP_NAT, "home"),
    "Employment Pass holders": (EP, EP_NAT, "home"),
    "Other work passes": (OTHER_PASS, EP_NAT, "home"),
    "Foreign domestic workers": (WP_FDW, FDW_NAT, "home"),
    "Dependants, long-term visit pass holders and students": (DEPENDANTS_STUDENTS, DEP_NAT, "home"),
}


def check_stock():
    t = sum(v[0] for v in GROUPS.values())
    assert t == NONRES, (t, NONRES)
    assert WP_OTHER == 335_800 and DEPENDANTS_STUDENTS == 289_790
    # the 323,000 dormitory residents are the 266,800 construction/marine/process workers who live
    # in them plus 56,200 other work permit holders
    assert CMP_DORM + OTHER_DORM == DORM_ALL
    assert DORM_PBD + DORM_FCD <= DORM_ALL


def _norm(d):
    s = float(sum(d.values()))
    return {k: v / s for k, v in d.items()}


_INV = {}


def _written(node):
    """origin_mix reads other countries' drawn (regrouped) ids; fragments and counts() use written
    ones, which countries.py then regroups (taxonomy/regroup.py)."""
    if not _INV:
        from regroup import move
        ids = set()
        files = [ROOT / "taxonomy" / "tree.txt", *sorted((ROOT / "taxonomy" / "tree.d").glob("*.txt"))]
        for f in files:
            for ln in f.read_text(encoding="utf-8").splitlines():
                ln = ln.strip()
                if ln and not ln.startswith("#") and "|" in ln:
                    ids.add(ln.split("|")[0].strip())
        for w in sorted(ids):
            _INV.setdefault(move(w), w)
    return _INV.get(node, node)


def _mix(iso, context):
    """language mix of one nationality in Singapore: {node: share}."""
    import origin_mix as om
    if iso == "IN" and context == "dorm":
        # uncited: Tamil Nadu is by every account the main source of Indian construction labour
        # (Singapore's Tamil workers, Tamil in dormitory signage); 60% Tamil, the rest at India's
        # drawn mix without Tamil
        return om._put(0.60, TAMIL, om._without(om.home("IN"), {TAMIL}))
    if iso == "MY":
        # uncited: Malaysians in Singapore are mostly ethnic Chinese Malaysians and Malays from
        # Johor; the home mix (60% Malay, 17% Chinese) would understate the Chinese
        return _norm({MALAY: 45, f"{ST}.mandarin": 22, f"{ST}.min_nan": 8, f"{ST}.cantonese": 6,
                      f"{ST}.hakka": 5, TAMIL: 8, ENGLISH: 6})
    return om.mix(iso, "sg")


def mix_written(iso, context):
    m = {}
    for n, v in _mix(iso, context).items():
        w = _written(n)
        m[w] = m.get(w, 0.0) + v
    return m


def group_rows():
    """[(group, where, node, people)] for every group and language."""
    check_stock()
    rows = []
    for g, (people, nat, where) in GROUPS.items():
        ctx = "dorm" if (where == "dorm" or "outside dormitories" in g) else "other"
        nat = _norm(nat)
        for iso, sh in nat.items():
            for node, ls in mix_written(iso, ctx).items():
                rows.append((g, where, node, people * sh * ls))
    return rows


# ---- geography -------------------------------------------------------------------------------

def build_layer():
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import Polygon
    from rdlink import RD_GEO
    sub = gpd.read_file(RD_GEO / "sg" / "sg_subzones.gpkg")
    sub["kind"] = "resident"
    res = sub[["unit", "pop", "kind", "geometry"]].copy()

    # purpose-built dormitories: 29 OSM ways named as worker dormitories (ids in OSM_PBD)
    osm = json.loads((RAW / "osm_worker_dorms.json").read_text(encoding="utf-8"))
    recs = []
    for e in osm["elements"]:
        pts = [(p["lon"], p["lat"]) for p in e["geometry"]]
        if len(pts) < 4:
            continue
        recs.append(dict(unit=f"PBD:{e['id']}", name=e["tags"].get("name", ""), geometry=Polygon(pts)))
    pbd = gpd.GeoDataFrame(recs, crs=4326)
    pbd = pbd[pbd.is_valid].copy()
    # residents per site: the published bed counts where there are some, else the mean of the 43
    # purpose-built dormitories (198,320 / 43); OSM footprints do not track beds (Papan has 7,900
    # beds on 1.5 ha, Jalan Tukang 3,420 on 5.3 ha)
    pbd["w"] = [BEDS.get(n, DORM_PBD / PBD_COUNT) for n in pbd["name"]]
    assert len(pbd) == len(osm["elements"]) == 29, (len(pbd), len(osm["elements"]))
    pbd["kind"] = "pbd"
    pbd["pop"] = 1.0

    # factory-converted dormitories: Business 2 (industrial) land of the 2019 Master Plan, cut by
    # subzone, one unit per subzone holding at least 5 ha of it
    b2 = gpd.read_file(RAW / "mp2019_business2.geojson").to_crs(3414)
    s3 = sub[["subzone_code", "subzone", "geometry"]].to_crs(3414)
    inter = gpd.overlay(s3, b2[["geometry"]], how="intersection")
    inter["ha"] = inter.area / 1e4
    by = inter.dissolve(by="subzone_code", aggfunc={"ha": "sum", "subzone": "first"})
    by = by[by["ha"] >= 5].reset_index()
    fcd = by.to_crs(4326)[["subzone_code", "subzone", "ha", "geometry"]].copy()
    fcd["unit"] = "FCD:" + fcd["subzone_code"]
    fcd["w"] = fcd["ha"]
    fcd["name"] = fcd["subzone"]
    fcd["kind"] = "fcd"
    fcd["pop"] = 1.0

    out = pd.concat([res, pbd[["unit", "pop", "kind", "geometry"]], fcd[["unit", "pop", "kind", "geometry"]]],
                    ignore_index=True)
    out = gpd.GeoDataFrame(out, crs=4326)
    GEO.mkdir(parents=True, exist_ok=True)
    tmp = GEO / "sg_places.tmp.gpkg"
    if tmp.exists():
        tmp.unlink()
    out.to_file(tmp, driver="GPKG")
    os.replace(tmp, GEO / "sg_places.gpkg")
    sites = pd.concat([
        pbd[["unit", "name", "kind", "w"]],
        fcd[["unit", "name", "kind", "w"]],
    ], ignore_index=True)
    sites.to_csv(SITES, index=False)
    print(f"sg_places.gpkg: {len(res)} resident subzones, {len(pbd)} dormitory sites "
          f"({pbd['w'].sum():,.0f} residents), {len(fcd)} industrial units ({fcd['w'].sum():,.0f} ha)")
    return out


def unit_pops():
    import geopandas as gpd
    from rdlink import RD_GEO
    sub = gpd.read_file(RD_GEO / "sg" / "sg_subzones.gpkg", columns=["unit", "pop"], ignore_geometry=True)
    return sub.groupby("unit")["pop"].sum()


def counts():
    """DataFrame [unit, node, count, tier] for the non-residents, summing to NONRES."""
    import pandas as pd
    rows = pd.DataFrame(group_rows(), columns=["group", "where", "node", "people"])
    sites = pd.read_csv(SITES)
    pbd = sites[sites["kind"] == "pbd"].copy()
    fcd = sites[sites["kind"] == "fcd"].copy()
    # located purpose-built dormitories take their beds; the rest of the 198,320, in dormitories
    # OSM does not name, goes with the factory-converted ones onto industrial land
    located = float(pbd["w"].sum())
    if located > DORM_PBD:
        raise SystemExit("sg_nonres: located dormitories hold more than 198,320")
    pbd["share"] = pbd["w"] / DORM_ALL
    fcd["share"] = (DORM_PBD - located + DORM_FCD) / DORM_ALL * fcd["w"] / fcd["w"].sum()
    out = []
    dorm = rows[rows["where"] == "dorm"].groupby("node", as_index=False)["people"].sum()
    for s in (pbd, fcd):
        for u, sh in zip(s["unit"], s["share"]):
            d = dorm.assign(unit=u, count=dorm["people"] * sh)
            out.append(d[["unit", "node", "count"]])
    # the rest of the dormitory stock (construction temporary quarters and unclassified) and every
    # other group: planning areas by resident population
    rest = dorm.assign(count=dorm["people"] * (1 - (DORM_PBD + DORM_FCD) / DORM_ALL))
    home = rows[rows["where"] == "home"].groupby("node", as_index=False)["people"].sum()
    home = home.merge(rest[["node", "count"]], on="node", how="outer").fillna(0)
    home["people"] = home["people"] + home["count"]
    pop = unit_pops()
    pop = pop / pop.sum()
    for u, sh in pop.items():
        out.append(home.assign(unit=u, count=home["people"] * sh)[["unit", "node", "count"]])
    df = pd.concat(out, ignore_index=True)
    df = df[df["count"] > 0]
    df["tier"] = "modelled"
    tot = df["count"].sum()
    if abs(tot - NONRES) > 1:
        raise SystemExit(f"sg_nonres: {tot:,.1f} people, expected {NONRES:,}")
    return df


if __name__ == "__main__":
    if "--build" in sys.argv:
        build_layer()
    else:
        check_stock()
        for g, (p, nat, w) in GROUPS.items():
            print(f"{p:>10,}  {g}  [{w}]")
        df = counts()
        t = df.groupby("node")["count"].sum().sort_values(ascending=False)
        print(f"{t.sum():,.0f} people, {len(t)} languages")
        for n, v in t.head(25).items():
            print(f"  {v:>10,.0f}  {n}")
