"""T-291: China's inputs for a tier 0 city (Shanghai first), through tier0's hooks.

- **Outline and districts**: OSM boundary=administrative admin_level=6 relations (the county-level
  districts) read from the Geofabrik extract, kept by their Chinese name, clipped to land by the
  OSM coastline polygons (OSM's district lines run far out to sea), cached as a small GeoJSON (the
  .pbf is deleted after routes.py).
- **Population** (T-1150): the 7th national census (2020-11-01) by township, from helper1m's China
  table (the ASPECT 100 m grid, Ju et al. 2025, built on 40,718 townships' counts, summed over each
  2018 township outline and scaled to the census counties), spread inside each township by tier 0's
  bounded GHSL. Shanghai's result is then held to its bulletin's district totals (SHANGHAI below);
  every planned mainland metro and zone row (T-1185: the zone rule's module) takes the townships
  through plan_hooks.
- **Jobs**: the 5th economic census (end of 2023), persons employed in legal-person units by
  district (JOBS below), spread inside each district by tier 0's proxy.

    python pipeline/china.py sha      (writes the districts file and prints what it holds)
    python pipeline/china.py townships <helper1m/countries/china/adm4>   (writes TOWNSHIPS from it)
"""
import json
import os
import sys

import numpy as np
import shapely
from shapely import make_valid, union_all
from shapely.geometry import mapping, shape

import build_city as b
import england
import tier0

# Shanghai's 16 districts: OSM name -> (pinyin, 2020 census resident population). Source: Shanghai
# Statistics Bureau, 7th census bulletin no. 2 (https://tjj.sh.gov.cn/tjnj/2020rktjnj/fu02.pdf).
SHANGHAI = {
    "黄浦区": ("Huangpu", 662030), "徐汇区": ("Xuhui", 1113078), "长宁区": ("Changning", 693051),
    "静安区": ("Jing'an", 975707), "普陀区": ("Putuo", 1239800), "虹口区": ("Hongkou", 757498),
    "杨浦区": ("Yangpu", 1242548), "闵行区": ("Minhang", 2653489), "宝山区": ("Baoshan", 2235218),
    "嘉定区": ("Jiading", 1834258), "浦东新区": ("Pudong", 5681512), "金山区": ("Jinshan", 822776),
    "松江区": ("Songjiang", 1909713), "青浦区": ("Qingpu", 1271424), "奉贤区": ("Fengxian", 1140872),
    "崇明区": ("Chongming", 637921),
}
DISTRICTS = {"sha": SHANGHAI}
# Jing'an's relation is broken in OSM (2026-10: its box reaches 33.4 N): its 13 subdistricts and
# Pengpu town. Baoshan's does not assemble either and is the remainder.
SUBDISTRICTS = {"sha": {"静安区": ["静安寺街道", "南京西路街道", "江宁路街道", "石门二路街道", "曹家渡街道", "天目西路街道",
                                 "北站街道", "宝山路街道", "芷江西路街道", "共和新路街道", "大宁路街道", "彭浦新村街道",
                                 "临汾路街道", "彭浦镇"]}}
CENSUS_URL = "https://tjj.sh.gov.cn/tjnj/2020rktjnj/fu02.pdf"
# Persons employed in legal-person units by district, end of 2023, in 10,000s: Shanghai's 5th
# economic census bulletin no. 1 (2025-04-24), table 1-5; JOBS_URL.
_JOBS_SH = {"浦东新区": 287.9, "黄浦区": 71.3, "徐汇区": 102.8, "长宁区": 60.2, "静安区": 97.4, "普陀区": 52.5,
            "虹口区": 40.9, "杨浦区": 51.1, "闵行区": 127.8, "宝山区": 76.6, "嘉定区": 87.4, "金山区": 39.4,
            "松江区": 85.2, "青浦区": 55.0, "奉贤区": 58.4, "崇明区": 16.1}
JOBS = {"sha": {k: v * 1e4 for k, v in _JOBS_SH.items()}}
JOBS_URL = "https://tjj.sh.gov.cn/cmsres/b3/b36d6d16841e4ac0a64a2e6664c69833/85fdf28606c6847035976a586d8d028d.pdf"
TOWNSHIPS = os.path.join(b.RAW, "tier0", "china_townships", "townships.fgb")
TOWNSHIPS_NAME = ("7th national census 2020 by township (ASPECT 100 m grid, Ju et al. 2025, figshare 27323106, "
                  "summed over 2018 township outlines and scaled to the census counties)")


def raw(name):
    return os.path.join(b.RAW, "tier0", "china", name)


def write_townships(src):
    """TOWNSHIPS from helper1m's per-province GeoJSONs (code, name, name_cn, populations by year):
    the townships with a 2020 count, as one FlatGeobuf with a spatial index."""
    import glob
    import geopandas as gpd
    rows = []
    for p in sorted(glob.glob(os.path.join(src, "*.geojson"))):
        with open(p, encoding="utf-8") as f:
            for ft in json.load(f)["features"]:
                pr = ft["properties"]
                pop = (pr.get("populations") or {}).get("2020")
                if pop and ft.get("geometry"):
                    rows.append({"code": pr["code"], "name": pr.get("name"), "name_cn": pr.get("name_cn"),
                                 "county": pr.get("parent_code"), "pop": float(pop), "geometry": make_valid(shape(ft["geometry"]))})
    df = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    os.makedirs(os.path.dirname(TOWNSHIPS), exist_ok=True)
    tmp = TOWNSHIPS.replace(".fgb", ".part.fgb")
    df.to_file(tmp, driver="FlatGeobuf", engine="pyogrio")
    os.replace(tmp, TOWNSHIPS)
    print(f"{TOWNSHIPS}: {len(df):,} townships, {df['pop'].sum():,.0f} people", flush=True)


def townships(boundary):
    """[(code, polygon, 2020 people, share inside)] of the townships meeting the boundary. A township
    the outline cuts has the share of its GHSL people inside (of its area where GHSL has none), so a
    coastal township whose outline reaches out to sea stays whole."""
    import pyogrio
    import osm_units
    if not os.path.exists(TOWNSHIPS):
        raise SystemExit(f"no {TOWNSHIPS}: python pipeline/china.py townships <helper1m/countries/china/adm4>")
    df = pyogrio.read_dataframe(TOWNSHIPS, bbox=tuple(boundary.bounds))
    shapely.prepare(boundary)
    g = df.geometry.values
    ok = shapely.intersects(boundary, g)
    g, codes, pops = g[ok], df["code"].values[ok], df["pop"].values[ok]
    share = np.ones(len(g))
    cut = np.flatnonzero(~shapely.contains(boundary, g))
    if len(cut):
        part = [union_all([p for p in shapely.get_parts(shapely.intersection(x, boundary))
                           if p.geom_type in ("Polygon", "MultiPolygon")]) for x in g[cut]]
        some = np.array([not p.is_empty for p in part])
        ppl = osm_units._people_many([g[i] for i in cut[some]] + [p for p, s in zip(part, some) if s])
        whole, inside = np.array(ppl[:some.sum()]), np.array(ppl[some.sum():])
        by_area = shapely.area([part[k] for k in np.flatnonzero(some)]) / np.maximum(shapely.area(g[cut[some]]), 1e-12)
        share[cut] = 0.0
        share[cut[some]] = np.where(whole > 0, inside / np.maximum(whole, 1e-9), by_area)
    return list(zip(codes, g, pops, share))


def township_population(city, boundary, sig, held=None):
    """2020 census population by township spread inside each by GHSL's residential volume
    (sig["res_v"]; a township with none, by the bounded GHSL); a township the outline cuts brings its
    share inside (townships), cells in no township keep the bounded GHSL's people at the townships'
    census-to-GHSL ratio. `held` [(name, polygon, total)]: the result then held to those units'
    totals. Tier 0's `population` hook."""
    if "res_v" not in sig:
        raise SystemExit("tier 0 signal cache from before T-1150 (no res_v): move it aside first "
                         "(notes/t1150/clear_signals.py)")
    ts = [t for t in townships(boundary) if t[3] > 0]
    units = [(t[0], t[1]) for t in ts]
    totals = {t[0]: t[2] * t[3] for t in ts}
    lat, lon, w = sig["lat"], sig["lon"], sig["pop_ghsl"]
    idx, _ = england.cell_units(lat, lon, units)
    lone = idx < 0
    rv = np.bincount(idx[~lone], weights=sig["res_v"][~lone], minlength=len(units))
    novol = lone.copy()
    novol[~lone] = rv[idx[~lone]] <= 0
    out = england.spread_totals(lat, lon, units, totals, np.where(novol, w, sig["res_v"]))
    ratio = sum(totals.values()) / max(float(w[~lone].sum()), 1.0)
    out[lone] = w[lone] * ratio
    print(f"  census 2020 by township: {len(units):,} townships ({sum(t[3] < 1 for t in ts)} cut by the outline, "
          f"{sum(t[2] * (1 - t[3]) for t in ts):,.0f} people of theirs outside), "
          f"{sum(totals.values()):,.0f} people, census / GHSL {ratio:.3f}; {int(lone.sum()):,} cells in no township "
          f"with {out[lone].sum():,.0f}; {int((rv <= 0).sum())} townships without residential volume", flush=True)
    name = (f"{TOWNSHIPS_NAME}, spread inside each by GHSL GHS-BUILT-V E2020 residential volume (total less NRES); "
            f"outside them {tier0.ghsl_pop_name(sig)}")
    if held:
        out = england.spread_totals(lat, lon, [(h[0], h[1]) for h in held], {h[0]: h[2] for h in held}, out)
    return out, name


def districts_file(city):
    """The districts (OSM admin_level 6, by name) clipped to land, as a GeoJSON with name, pinyin,
    population; its path. Read from the extract's .pbf once. A district whose relation GDAL cannot
    assemble (broken in OSM) is the union of its admin_level 8 units in SUBDISTRICTS, or, if it is
    the only one left, the province (admin_level 4) less every other district."""
    extract, wanted = b.CITIES[city]["extract"], DISTRICTS[city]
    out = raw(f"{city}_districts.geojson")
    if os.path.exists(out):
        return out
    import pyogrio
    import routes
    import osm_water
    pbf = routes.download(extract)
    df = pyogrio.read_dataframe(pbf, layer="multipolygons", columns=["osm_id", "name", "admin_level", "boundary"],
                                where="boundary = 'administrative' AND admin_level IN ('4', '6', '8')")
    geoms, how = {}, {}
    for name in wanted:
        rows = df[(df.admin_level == "6") & (df["name"] == name)]
        if len(rows) == 1:
            geoms[name], how[name] = make_valid(rows.geometry.iloc[0]).buffer(0), f"OSM r{rows.osm_id.iloc[0]}"
        elif name in SUBDISTRICTS.get(city, {}):
            subs = SUBDISTRICTS[city][name]
            rows = df[(df.admin_level == "8") & df["name"].isin(subs)]
            if len(rows) != len(subs):
                raise SystemExit(f"{city}: {name}: {len(rows)} of {len(subs)} subdistricts found")
            geoms[name] = union_all([make_valid(g).buffer(0) for g in rows.geometry]).buffer(0)
            how[name] = f"OSM, its {len(subs)} admin_level 8 units"
    left = [n for n in wanted if n not in geoms]
    if len(left) > 1:
        raise SystemExit(f"{city}: no OSM boundary for {left}")
    if left:
        prov = df[df.admin_level == "4"]
        if len(prov) != 1:
            raise SystemExit(f"{city}: {len(prov)} admin_level 4 boundaries in the extract")
        geoms[left[0]] = make_valid(prov.geometry.iloc[0]).buffer(0).difference(union_all(list(geoms.values())))
        how[left[0]] = "OSM, the province less the other districts"
    w, s, e, n = union_all(list(geoms.values())).bounds
    sea = union_all(osm_water.sea((w - 0.01, s - 0.01, e + 0.01, n + 0.01)))
    feats = []
    for name, (pinyin, pop) in wanted.items():
        g = geoms[name].difference(sea).buffer(0)
        if g.geom_type == "MultiPolygon":  # slivers left along the clipped shore
            g = shapely.MultiPolygon([p for p in g.geoms if p.area > 1e-6])
        feats.append({"type": "Feature", "properties": {"name": name, "pinyin": pinyin, "pop": pop, "source": how[name]},
                      "geometry": mapping(g)})
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out + ".part", "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f, ensure_ascii=False)
    os.replace(out + ".part", out)
    return out


def districts(city):
    with open(districts_file(city), encoding="utf-8") as f:
        return [(ft["properties"]["name"], shape(ft["geometry"]), ft["properties"]["pop"]) for ft in json.load(f)["features"]]


def boundary_file(city):
    """The districts dissolved, for tier0's `boundary_file`; its path."""
    out = raw(f"{city}_outline.geojson")
    if not os.path.exists(out):
        g = union_all([d[1] for d in districts(city)]).buffer(0)
        with open(out + ".part", "w", encoding="utf-8") as f:
            json.dump({"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {"city": city},
                                                                   "geometry": mapping(g)}]}, f)
        os.replace(out + ".part", out)
    return out


def census_population(city, boundary, sig):
    """2020 census population by township, held to the city bulletin's district totals; tier0's
    `population` hook."""
    wanted = DISTRICTS[city]
    ds = districts(city)
    out, name = township_population(city, boundary, sig, held=ds)
    idx, _ = england.cell_units(sig["lat"], sig["lon"], [(d[0], d[1]) for d in ds])
    for i, (dn, _, pop) in enumerate(ds):
        have = sig["pop_ghsl"][idx == i].sum()
        print(f"  census 2020 {wanted[dn][0]}: GHSL {have:,.0f} -> {pop:,} ({pop / max(have, 1) - 1:+.1%})", flush=True)
    print(f"  census 2020 by district: {len(ds)} districts, {out.sum():,.0f} people", flush=True)
    return out, f"{name}; held to the resident population by district (Shanghai Statistics Bureau)"


def census_jobs(city, boundary, sig, weight):
    """Persons employed in legal-person units by district (5th economic census, end of 2023), spread
    inside each district by the tier 0 proxy; tier0's `jobs` hook."""
    wanted, jobs = DISTRICTS[city], JOBS[city]
    ds = districts(city)
    units = [(d[0], d[1]) for d in ds]
    out = england.spread_totals(sig["lat"], sig["lon"], units, jobs, weight)
    idx, _ = england.cell_units(sig["lat"], sig["lon"], units)
    w, J = float(weight.sum()), sum(jobs.values())
    for i, (name, _, _) in enumerate(ds):
        print(f"  economic census 2023 {wanted[name][0]}: proxy share {weight[idx == i].sum() / w:.1%}, census {jobs[name] / J:.1%}", flush=True)
    print(f"  economic census 2023 by district: {out.sum():,.0f} jobs", flush=True)
    return out, ("5th national economic census, persons employed in legal-person units at the end of 2023 by district "
                 "(Shanghai Statistics Bureau bulletin no. 1, table 1-5; individual businesses not included)")


def plan_hooks(row):
    """T-555: a planned mainland row's population by township (T-1150; zone rows too, T-1185). Shanghai's district tables
    (economic census jobs), its OSM district names and its survey's mean commute are Shanghai's
    alone; another city needs its own tables entered first."""
    if row.get("iso2") != "CN":
        return {}
    return {"population": lambda city, boundary, sig: township_population(city, boundary, sig)}


if __name__ == "__main__":
    if sys.argv[1] == "townships":
        write_townships(sys.argv[2])
        sys.exit()
    city = sys.argv[1]
    ds = districts(city)
    g = union_all([d[1] for d in ds])
    km2 = 111.32 * 110.57 * 0.855  # degrees squared to km2 near 31 N
    print(f"{city}: {len(ds)} districts, {sum(d[2] for d in ds):,} people, outline {g.geom_type}, "
          f"{len(getattr(g, 'geoms', [g]))} parts, about {g.area * km2:,.0f} km2, bounds {tuple(round(v, 3) for v in g.bounds)}")
    for d in ds:
        parts = list(getattr(d[1], "geoms", [d[1]]))
        print(f"  {DISTRICTS[city][d[0]][0]}: {len(parts)} parts, {sum(len(p.interiors) for p in parts)} holes, "
              f"about {d[1].area * km2:,.1f} km2")
