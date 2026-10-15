"""T-291: China's inputs for a tier 0 city (Shanghai first), through tier0's hooks.

- **Outline and districts**: OSM boundary=administrative admin_level=6 relations (the county-level
  districts) read from the Geofabrik extract, kept by their Chinese name, clipped to land by the
  OSM coastline polygons (OSM's district lines run far out to sea), cached as a small GeoJSON (the
  .pbf is deleted after routes.py).
- **Population**: the 7th national census (2020-11-01) resident population by district, from
  Shanghai's bulletin (SHANGHAI below), spread inside each district by GHSL.
- **Jobs**: the 5th economic census (end of 2023), persons employed in legal-person units by
  district (JOBS below), spread inside each district by tier 0's proxy.

    python pipeline/china.py sha      (writes the districts file and prints what it holds)
"""
import json
import os
import sys

import shapely
from shapely import make_valid, union_all
from shapely.geometry import mapping, shape

import build_city as b
import england

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


def raw(name):
    return os.path.join(b.RAW, "tier0", "china", name)


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
    """2020 census population by district, spread inside each by GHSL; tier0's `population` hook."""
    wanted = DISTRICTS[city]
    ds = districts(city)
    units = [(d[0], d[1]) for d in ds]
    out = england.spread_totals(sig["lat"], sig["lon"], units, {d[0]: d[2] for d in ds}, sig["pop_ghsl"])
    idx, _ = england.cell_units(sig["lat"], sig["lon"], units)
    for i, (name, _, pop) in enumerate(ds):
        have = sig["pop_ghsl"][idx == i].sum()
        print(f"  census 2020 {wanted[name][0]}: GHSL {have:,.0f} -> {pop:,} ({pop / max(have, 1) - 1:+.1%})", flush=True)
    print(f"  census 2020 by district: {len(ds)} districts, {out.sum():,.0f} people", flush=True)
    return out, "7th national census 2020, resident population by district (Shanghai Statistics Bureau), spread inside each by GHSL GHS-POP E2020"


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


if __name__ == "__main__":
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
