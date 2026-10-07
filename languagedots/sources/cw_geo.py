"""Curaçao placement layer: religiondots' Kontur hexes for CW (read only), each given its 2023
geozone and calibrated to that geozone's Census 2023 population -> data/geo/cw/cw_hexes.gpkg
(unit = "CW", the whole island, which is the unit the language table counts in).

    python sources/cw_geo.py [--fetch]

The language table is national only (sources/cw_census.py), so all of this decides only where on
the island a dot falls (AGENT_BRIEF §4.4: a placement inside the counted unit).

SOURCES.
  * CBS Curaçao's neighbourhood polygons, ArcGIS Online feature service `BuurtenCBS` (291
    neighbourhoods, field `geocode`; the first digits, geocode // 100, are the geozone number).
  * The Census 2023 neighbourhood viewer (cbs-curacao.github.io/ndv-static-site), whose data table
    pairs each neighbourhood name with its geozone name: it names the geozone numbers.
  * Census 2023 Table G-3, population by geozone, nationality and sex (60 geozones with 5 or more
    people, plus 1,186 people whose geozone was not reported): each geozone's population, and its
    share holding a nationality other than Dutch.

STEPS.
  1. Geozone number -> name: every neighbourhood the viewer names votes for its number's name;
     a number must get exactly one name (asserted), and the numbers no viewer neighbourhood
     carries are named from the service's own "Undefined (zone X)" rows (UNDEFINED below).
     Every G-3 geozone must get exactly one number (asserted both ways).
  2. Each hex's centroid goes to the neighbourhood it falls in, else the nearest within SNAP_M
     (coastal hexes; playbook "Hex centroids fall just offshore of island units").
  3. Witness: Kontur's people per geozone against G-3, log correlation against 500 shuffles.
  4. Each geozone's hexes are scaled to its G-3 population; a geozone with people and no hex is
     appended as its own polygon; geozones absent from G-3 (under 5 people) get weight 0.
  5. `foreign` = the geozone's share of people with another nationality than Dutch, which
     countries/cw.py uses to place Spanish, English and other dots.
"""
import io
import json
import math
import os
import random
import sys
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import pandas as pd  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
RAW = HERE / "data" / "raw" / "cw"
BUURTEN = RAW / "buurten_cbs.geojson"
PAIRS = RAW / "ndv_buurt_geozone.csv"
G3 = RAW / "table-g-3-population-by-geozone-nationality-and-sex-2023.xlsx"
OUT = HERE / "data" / "geo" / "cw" / "cw_hexes.gpkg"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/124.0 Safari/537.36")
SVC = ("https://services5.arcgis.com/1KFGuk9LOFs0SAY7/arcgis/rest/services/BuurtenCBS/"
       "FeatureServer/0/query?")
NDV = "https://cbs-curacao.github.io/ndv-static-site/"
G3_URL = ("https://cuatro.sim-cdn.nl/sensocbs/uploads/"
          "table-g-3.-population-by-geozone-nationality-and-sex-census-2023.xlsx")
CRS_M = 32619          # UTM 19N
SNAP_M = 1500
FAR_MAX = 50           # Kontur people allowed beyond SNAP_M (Klein Curaçao)
N_BUURTEN = 291
N_G3 = 60
G3_TOTAL = 155_826
G3_NOT_REPORTED = 1_186
# geozone numbers that no neighbourhood in the viewer's table carries, named from the service's
# "Undefined (zone X)" neighbourhoods and the one-neighbourhood zones; checked against G-3 names
UNDEFINED = {2: "LAGUN", 3: "CHRISTOFFEL", 4: "FLIP", 5: "TERA PRETU", 6: "LELIENBERG",
             8: "PANNEKOEK", 9: "WACAO", 17: "HATO", 35: "ASIENTO", 44: "NIEUWE HAVEN"}
NOT_IN_G3 = {"ASIENTO", "NIEUWE HAVEN"}     # under 5 people: G-3 leaves them out


def norm(s):
    s = str(s).upper().replace("Ñ", "NJ").replace("/ ", "/").strip()
    return " ".join(s.split())


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    return urllib.request.urlopen(req, timeout=300).read()


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    q = urllib.parse.urlencode({"where": "1=1", "outFields": "OBJECTID,NAME,geocode,NBHcode",
                                "outSR": 4326, "f": "geojson"})
    data = get(SVC + q)
    BUURTEN.write_bytes(data)
    print(f"  {BUURTEN.name}: {len(data):,} bytes")
    G3.write_bytes(get(G3_URL))
    print(f"  {G3.name}")
    # the viewer is one 30 MB page; keep only the neighbourhood -> geozone pairs of its table
    html = get(NDV).decode("utf-8")
    pairs = set()
    for line in html.split("\n"):
        if '"Buurtnaam' not in line and "Buurtnaam" not in line:
            continue
        if '{"x":' not in line:
            continue
        d = json.loads(line[line.index('{"x":'):line.rindex("</script>")])
        data = d["x"].get("data")
        if not data or "Buurtnaam" not in d["x"].get("container", ""):
            continue
        pairs |= set(zip(data[0], data[1]))
    if not pairs:
        raise SystemExit("cw: no Buurtnaam/GeoZone table in the neighbourhood viewer")
    pd.DataFrame(sorted(pairs), columns=["buurt", "geozone"]).to_csv(PAIRS, index=False)
    print(f"  {PAIRS.name}: {len(pairs)} pairs")


def read_g3():
    import openpyxl
    ws = openpyxl.load_workbook(G3, data_only=True).worksheets[0]
    rows = []
    for r in ws.iter_rows(values_only=True):
        name, dutch, other, grand = r[2], r[5], r[8], r[12]
        if isinstance(grand, (int, float)) and isinstance(dutch, (int, float)):
            rows.append(dict(gz=norm(name), dutch=dutch, other=other, pop=grand))
        elif name == "NOT REPORTED":
            if grand != G3_NOT_REPORTED:
                raise SystemExit(f"cw: G-3 not-reported row {grand}")
    g = pd.DataFrame(rows)
    tot = g[g["gz"] == "TOTAL"]
    g = g[g["gz"] != "TOTAL"]
    if len(g) != N_G3 or g["gz"].duplicated().any():
        raise SystemExit(f"cw: G-3 has {len(g)} geozones, expected {N_G3}")
    if int(tot["pop"].iloc[0]) != G3_TOTAL or int(g["pop"].sum()) + G3_NOT_REPORTED != G3_TOTAL:
        raise SystemExit("cw: G-3 geozones plus not-reported do not sum to 155,826")
    return g.set_index("gz")


def pear(a, b):
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    return num / math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))


def main():
    if "--fetch" in sys.argv or not (BUURTEN.exists() and PAIRS.exists() and G3.exists()):
        fetch()
    import geopandas as gpd
    from rdlink import RD_GEO

    g3 = read_g3()
    bu = gpd.read_file(BUURTEN)
    if len(bu) != N_BUURTEN or bu["geocode"].duplicated().any():
        raise SystemExit(f"cw: {len(bu)} neighbourhoods, expected {N_BUURTEN} with unique codes")
    bu["zc"] = bu["geocode"].astype(int) // 100

    # 1. geozone number -> name
    pairs = pd.read_csv(PAIRS)
    name_of = dict(zip(pairs["buurt"].map(norm), pairs["geozone"].map(norm)))
    votes = defaultdict(Counter)
    for zc, n in zip(bu["zc"], bu["NAME"].map(norm)):
        if n in name_of:
            votes[zc][name_of[n]] += 1
    zname = {}
    for zc, c in votes.items():
        if len(c) != 1:
            raise SystemExit(f"cw: geozone number {zc} carries neighbourhoods of {dict(c)}")
        zname[zc] = next(iter(c))
    for zc, n in UNDEFINED.items():
        if zc in zname:
            raise SystemExit(f"cw: geozone {zc} is named by the viewer too ({zname[zc]})")
        zname[zc] = n
    missing = set(bu["zc"]) - set(zname)
    if missing:
        raise SystemExit(f"cw: geozone numbers without a name: {sorted(missing)}")
    if len(set(zname.values())) != len(zname):
        raise SystemExit("cw: two geozone numbers share a name")
    drawn = set(zname.values()) - NOT_IN_G3
    if drawn != set(g3.index):
        raise SystemExit(f"cw: geozones differ from G-3: {sorted(drawn - set(g3.index))} / "
                         f"{sorted(set(g3.index) - drawn)}")
    bu["gz"] = bu["zc"].map(zname)
    print(f"  {len(bu)} neighbourhoods in {len(zname)} geozones; all {N_G3} of G-3's geozones "
          f"matched one number each ({len(UNDEFINED)} named from the service's own rows)")

    # 2. hexes to neighbourhoods
    k = gpd.read_file(RD_GEO / "cw" / "cw_hexes.gpkg")
    if len(k) == 0:
        raise SystemExit("cw: religiondots' cw_hexes.gpkg has ZERO features")
    km = k.to_crs(CRS_M)
    bum = bu.to_crs(CRS_M)
    pts = gpd.GeoDataFrame({"kpop": km["pop"].astype(float)}, geometry=km.geometry.centroid,
                           crs=CRS_M)
    j = gpd.sjoin(pts, bum[["gz", "geometry"]], how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")].reindex(pts.index)
    out = j["gz"].isna()
    if out.any():
        near = gpd.sjoin_nearest(pts[out], bum[["gz", "geometry"]], distance_col="d")
        near = near[~near.index.duplicated(keep="first")]
        # Klein Curaçao, 10 km off the south-east point, is in no neighbourhood and has no
        # residents; Kontur may model a few people on its buildings. Leave such hexes out.
        far = near[near["d"] > SNAP_M]
        if pts.loc[far.index, "kpop"].sum() > FAR_MAX:
            raise SystemExit(f"cw: {pts.loc[far.index, 'kpop'].sum():.0f} Kontur people more "
                             f"than {SNAP_M} m from every neighbourhood")
        near = near[near["d"] <= SNAP_M]
        j.loc[near.index, "gz"] = near["gz"]
        print(f"  {len(k)} hexes, {pts['kpop'].sum():,.0f} Kontur people; {len(near)} "
              f"({pts.loc[near.index, 'kpop'].sum():,.0f} people) snapped to the nearest "
              f"neighbourhood, at most {near['d'].max():,.0f} m; {len(far)} hexes "
              f"({pts.loc[far.index, 'kpop'].sum():,.0f} people) further than {SNAP_M} m left out "
              f"(" + ", ".join(f"{d / 1000:.1f} km" for d in far['d']) + ")")

    # 3. witness: Kontur per geozone against the census
    per = j.groupby("gz")["kpop"].sum()
    c = g3["pop"].astype(float)
    kz = per.reindex(c.index).fillna(0)
    ratio = kz.sum() / c.sum()
    nrm = (kz / c / ratio).sort_values()
    print(f"  Kontur / G-3 overall {ratio:.3f}; per geozone normalised p10 {nrm.quantile(.1):.2f} "
          f"median {nrm.median():.2f} p90 {nrm.quantile(.9):.2f}; lowest "
          + ", ".join(f"{i} {v:.2f}" for i, v in nrm.head(3).items()) + "; highest "
          + ", ".join(f"{i} {v:.2f}" for i, v in nrm.tail(3).items()))
    lc, lk = [math.log(v) for v in c], [math.log(max(v, 1)) for v in kz]
    r = pear(lc, lk)
    rng = random.Random(0)
    best = max(abs(pear(lc, rng.sample(lk, len(lk)))) for _ in range(500))
    print(f"  log correlation r = {r:.3f} against a best of {best:.3f} over 500 shuffles")
    if r <= best:
        raise SystemExit("cw: the geozone join is not carrying information")
    extra_gz = per[per.index.isin(NOT_IN_G3)]
    if len(extra_gz):
        print("  Kontur people in geozones G-3 leaves out (weight 0): "
              + ", ".join(f"{i} {v:.0f}" for i, v in extra_gz.items()))

    # 4. calibrate each geozone to G-3
    wsum = j.groupby("gz")["kpop"].sum()
    j["pop"] = j["kpop"] * (j["gz"].map(c) / j["gz"].map(wsum)).fillna(0.0)
    hexes = gpd.GeoDataFrame({"cellcode": k["cellcode"], "unit": "CW", "gz": j["gz"],
                              "pop": j["pop"]}, geometry=k.geometry.to_numpy(), crs=k.crs)
    hexes = hexes[hexes["pop"] > 0]
    lost = [z for z in c.index if c[z] > 0 and wsum.get(z, 0) <= 0]
    if lost:
        zp = bu[bu["gz"].isin(lost)].dissolve("gz").reset_index()
        add = gpd.GeoDataFrame({"cellcode": "gz:" + zp["gz"], "unit": "CW", "gz": zp["gz"],
                                "pop": zp["gz"].map(c)}, geometry=zp.geometry.to_numpy(),
                               crs=bu.crs).to_crs(k.crs)
        hexes = pd.concat([hexes, add], ignore_index=True)
        print(f"  {len(lost)} geozones with people and no Kontur hex, drawn on their own polygon: "
              + ", ".join(f"{z} {int(c[z])}" for z in lost))
    got = hexes.groupby("gz")["pop"].sum().reindex(c.index).fillna(0)
    if ((got - c).abs() > 0.01).any():
        raise SystemExit("cw: weights do not reproduce G-3 per geozone")

    # 5. other-nationality share per geozone
    share = (g3["other"] / (g3["dutch"] + g3["other"])).astype(float)
    hexes["foreign"] = hexes["gz"].map(share)
    if hexes["foreign"].isna().any():
        raise SystemExit("cw: hexes without a nationality share")
    nat = g3["other"].sum() / (g3["dutch"].sum() + g3["other"].sum())
    print(f"  other-nationality share per geozone: national {nat:.1%}, from "
          f"{share.min():.1%} ({share.idxmin()}) to {share.max():.1%} ({share.idxmax()})")
    dens = hexes.to_crs(CRS_M)
    dens = (dens["pop"] / (dens.area / 1e6)).max()
    print(f"  calibrated densest feature {dens:,.0f} people/km2; total weight "
          f"{hexes['pop'].sum():,.0f} (G-3 less {G3_NOT_REPORTED:,} with no geozone)")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    hexes[["cellcode", "unit", "gz", "pop", "foreign", "geometry"]].to_file(
        OUT, layer="hexes", driver="GPKG")
    print(f"  wrote {OUT} ({len(hexes)} features, {hexes['gz'].nunique()} geozones)")


if __name__ == "__main__":
    main()
