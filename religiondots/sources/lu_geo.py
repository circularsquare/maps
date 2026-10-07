"""Luxembourg: the 102 communes of the 2021 census, and Kontur's grid calibrated to each.

Writes:
    data/geo/lu/lu_units.gpkg        GISCO LAU 2021's 102 communes, `unit` = LAU code (`0503`)
    data/geo/lu/lu_grid_400m.gpkg    Kontur H3 r8 hexes with `unit` and `pop`   (`place`)

Usage:
    python sources/lu_geo.py --fetch   # Kontur's Luxembourg extract
    python sources/lu_geo.py           # build both layers

THE COMMUNES ONLY PLACE PEOPLE. Every commune is drawn at the same national mix (sources/lu.py),
so they decide where the dots go and nothing about what they are. They are the census's 102 of
8 November 2021 (two pairs merged in 2023: Bous and Waldbredimus, Grosbous and Wahl), which is
GISCO LAU 2021's vintage too.

THE JOIN IS BY CODE, WITNESSED BY NAME. STATEC's DF_B1625 code `LU0000503` ends in the LAU code
`0503`; the build asserts that every code pairs and that the two files carry the same name for it,
which neither file's code decides. GISCO's POP_2021 is a second population source per commune
(geo_checks.ratio_band, the census as the base).

KONTUR IS SCALED SO EACH COMMUNE HOLDS ITS CENSUS COUNT. Iceland's and the Faroes' lesson
(playbooks/geography.md): where the office counts below the drawn unit, share that count over the
unit's hexes. Here the commune is both, so the calibration fixes Kontur's split between communes
and keeps its shape inside each.
"""

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
GEO = os.path.join(ROOT, "data", "geo", "lu")
LAU = os.path.join(ROOT, "data", "geo", "lau2021", "shp4326", "LAU_RG_01M_2021_4326.shp")

KONTUR_DIR = os.path.join(ROOT, "data", "geo", "kontur")
KONTUR_URL = ("https://geodata-eu-central-1-kontur-public.s3.eu-central-1.amazonaws.com/"
              "kontur_datasets/kontur_population_LU_20231101.gpkg.gz")
KONTUR_GZ = os.path.join(KONTUR_DIR, "kontur_population_LU_20231101.gpkg.gz")
KONTUR = os.path.join(KONTUR_DIR, "kontur_population_LU_20231101.gpkg")

UNITS_OUT = os.path.join(GEO, "lu_units.gpkg")
# Named `_grid_<n>m` so kontur_cap.py and the grid-floor check recognise it as a Kontur layer.
GRID_OUT = os.path.join(GEO, "lu_grid_400m.gpkg")

N_COMMUNES = 102
LUREF = 2169                  # Luxembourg's national projected CRS, metres
SNAP_KM_LU = 1.0              # a centre inside Natural Earth's Luxembourg, outside every commune
NE_COUNTRIES = os.path.join(ROOT, "data", "geo", "ne_10m_admin_0_countries.geojson")
# GISCO POP_2021 over the census, per commune, relative to national. POP_2021 is the 1 January 2021
# register, ten months before the census (playbooks/geography.md, "a boundary file's population
# column can be another date"); measured 0.924 (Saeul, 892 people) to 1.028 (Schifflange). Commune
# sizes run 400 to 128,000, so a swapped pair would land far outside this.
GISCO_BAND = (0.90, 1.10)


def fetch():
    import gzip
    import shutil

    import requests

    os.makedirs(KONTUR_DIR, exist_ok=True)
    if os.path.exists(KONTUR) and os.path.getsize(KONTUR) > 100_000:
        print("already have", KONTUR)
        return
    if not os.path.exists(KONTUR_GZ) or os.path.getsize(KONTUR_GZ) < 10_000:
        print("GET", KONTUR_URL)
        r = requests.get(KONTUR_URL, timeout=600, headers={"User-Agent": "religiondots/1.0"})
        r.raise_for_status()
        if r.content[:2] != b"\x1f\x8b":
            raise SystemExit(f"not gzip: first bytes {r.content[:40]!r}")
        with open(KONTUR_GZ + ".tmp", "wb") as fh:
            fh.write(r.content)
        os.replace(KONTUR_GZ + ".tmp", KONTUR_GZ)
        print(f"  {os.path.getsize(KONTUR_GZ):,} bytes")
    with gzip.open(KONTUR_GZ, "rb") as src, open(KONTUR + ".tmp", "wb") as dst:
        shutil.copyfileobj(src, dst)
    os.replace(KONTUR + ".tmp", KONTUR)
    print(f"  decompressed to {KONTUR}  {os.path.getsize(KONTUR):,} bytes")


def build_units():
    import geo_checks
    import lu

    census = lu._census()
    g = geo_checks.read_layer(LAU, "GISCO LAU 2021, Luxembourg", where="CNTR_CODE='LU'").to_crs(4326)
    g["unit"] = g["LAU_ID"].astype(str).str.strip().str.zfill(4)
    if len(g) != N_COMMUNES or g["unit"].duplicated().any():
        raise SystemExit(f"LAU 2021: {len(g)} Luxembourg communes, expected {N_COMMUNES} distinct")
    if set(g["unit"]) != set(census.index):
        raise SystemExit(f"LAU codes against the census: {sorted(set(g['unit']) ^ set(census.index))}")
    names = dict(zip(g["unit"], g["LAU_NAME"]))
    wrong = {u: (names[u], census.loc[u, "name"]) for u in census.index
             if names[u] != census.loc[u, "name"]}
    if wrong:
        raise SystemExit(f"the LAU file and the census name a code differently: {wrong}")
    print(f"  LAU 2021: {len(g)} communes, every code pairs with DF_B1625 and carries its name")

    g["pop"] = g["unit"].map(census["_T"]).astype(float)
    nat = g["POP_2021"].sum() / g["pop"].sum()
    rel = g["POP_2021"] / g["pop"] / nat
    print(f"  GISCO POP_2021 over the census: national {nat:.4f}; per commune {rel.min():.3f} "
          f"({names[g.loc[rel.idxmin(), 'unit']]}) to {rel.max():.3f} ({names[g.loc[rel.idxmax(), 'unit']]})")
    if not (GISCO_BAND[0] <= rel.min() and rel.max() <= GISCO_BAND[1]):
        raise SystemExit(f"a commune's GISCO population is outside {GISCO_BAND} of the census")

    minx, miny, maxx, maxy = g.total_bounds
    print(f"  bbox {minx:.3f},{miny:.3f} .. {maxx:.3f},{maxy:.3f}")
    if not (5.70 < minx < 5.78 and 6.50 < maxx < 6.56 and 49.42 < miny < 49.47 and 50.16 < maxy < 50.20):
        raise SystemExit("bbox is not Luxembourg's; check the CRS and the country filter")
    g["area_km2"] = g.to_crs(LUREF).area / 1e6
    print(f"  area {g['area_km2'].sum():,.1f} km2 (STATEC: 2,586.4); median commune "
          f"{g['area_km2'].median():.1f} km2")
    if abs(g["area_km2"].sum() - 2586.4) > 30:
        raise SystemExit("total area is far from Luxembourg's 2,586 km2")
    os.makedirs(GEO, exist_ok=True)
    out = g[["unit", "LAU_NAME", "pop", "area_km2", "geometry"]].rename(columns={"LAU_NAME": "name"})
    out.to_file(UNITS_OUT, layer="units", driver="GPKG")
    print(f"  wrote {UNITS_OUT}")
    return out


def build_grid(units):
    import geopandas as gpd
    import pandas as pd
    import pyogrio

    import geo_checks

    if not os.path.exists(KONTUR):
        raise SystemExit(f"missing {KONTUR}; run sources/lu_geo.py --fetch first")
    layers = list(pyogrio.list_layers(KONTUR)[:, 0])
    layer = "population" if "population" in layers else layers[0]
    hexes = geo_checks.read_layer(KONTUR, "Kontur LU", layer=layer).to_crs(4326)
    total = float(hexes["population"].sum())
    print(f"\n  Kontur LU: {len(hexes):,} hexes, {total:,.0f} people")

    centres = gpd.GeoDataFrame(geometry=hexes.geometry.representative_point(), crs=4326)
    hit = gpd.sjoin(centres, units[["unit", "geometry"]], how="left", predicate="within")
    hit = hit[~hit.index.duplicated(keep="first")]
    hexes["unit"] = hit["unit"].to_numpy()
    out = hexes["unit"].isna()
    print(f"    {int(out.sum())} hex centres outside every commune "
          f"({hexes.loc[out, 'population'].sum():,.0f} people)")
    # LANDLOCKED, WITH RIVER BORDERS, AND LAU 2021 IS THE 1:1 MILLION EDITION. Measured 2026-10-03:
    # 188 hex centres (21,595 people) fall outside every commune, all within 1 km of the line;
    # Natural Earth puts 78 of them (9,328 people) inside Luxembourg, the Moselle and Sauer towns the
    # generalised line cuts off, and 110 (12,267) in Germany, Belgium or France. Namibia's rule
    # (playbooks/geography.md): a centre inside a neighbour's Natural Earth polygon is the
    # neighbour's town and is dropped (Bhutan's Jaigaon); one inside Luxembourg's is snapped to the
    # nearest commune. The calibration below makes either choice move no commune's count.
    ne = gpd.read_file(NE_COUNTRIES)[["ISO_A2_EH", "geometry"]].to_crs(4326)
    cj = gpd.sjoin(centres[out.to_numpy()], ne, how="left", predicate="within")
    cj = cj[~cj.index.duplicated(keep="first")]
    ours = cj.index[cj["ISO_A2_EH"] == "LU"]
    near = gpd.sjoin_nearest(centres.loc[ours].to_crs(LUREF), units[["unit", "geometry"]].to_crs(LUREF),
                             how="left", max_distance=SNAP_KM_LU * 1000, distance_col="dist")
    near = near[~near.index.duplicated(keep="first")]
    if near["unit"].isna().any():
        raise SystemExit(f"{int(near['unit'].isna().sum())} hexes inside Natural Earth's Luxembourg are "
                         f"over {SNAP_KM_LU} km from every commune")
    hexes.loc[near.index, "unit"] = near["unit"].to_numpy()
    dropped = hexes["unit"].isna()
    lost_people = float(hexes.loc[dropped, "population"].sum())
    print(f"    {len(near)} inside Natural Earth's Luxembourg snapped to the nearest commune "
          f"({hexes.loc[near.index, 'population'].sum():,.0f} people, at most {near['dist'].max():.0f} m); "
          f"{int(dropped.sum())} in a neighbour dropped ({lost_people:,.0f})")
    if lost_people / total > 0.03:
        raise SystemExit("over 3% of Kontur's people dropped as a neighbour's; look before dropping")
    hexes = hexes[~dropped].copy()
    name = units.set_index("unit")["name"]

    # ---- CUT EVERY KEPT HEX BY THE COMMUNES AND SHARE ITS PEOPLE BY LAND AREA (Malta's method,
    # playbooks/geography.md). Clipping a hex to its centre's commune and keeping all its people
    # left slivers holding a whole hex: the first build reached 1.56 million per km2 on a snapped
    # river hex. Each hex's people go to its pieces inside Luxembourg in proportion to area, over
    # the sum of those pieces (the rest of a border hex is a neighbour's land, or nobody's here).
    # A SNAPPED HEX IS KEPT WHOLE, on the commune it was snapped to: its centre is outside the
    # generalised line, so its piece inside can be a sliver (Rambrouch's was under 0.001 km2 and
    # took the hex's people), and Natural Earth already says it is Luxembourg's.
    hexes["hid"] = range(len(hexes))
    snapped = hexes.index.isin(near.index)
    pieces = gpd.overlay(hexes.loc[~snapped, ["hid", "population", "geometry"]].to_crs(LUREF),
                         units[["unit", "geometry"]].to_crs(LUREF), how="intersection",
                         keep_geom_type=True)
    pieces["a"] = pieces.area
    pieces = pieces[pieces["a"] > 1.0]
    pieces["population"] = pieces["population"] * pieces["a"] / pieces.groupby("hid")["a"].transform("sum")
    if set(pieces["hid"]) != set(hexes.loc[~snapped, "hid"]):
        raise SystemExit("a hex whose centre is inside a commune has no piece in any commune")
    whole = hexes.loc[snapped, ["hid", "population", "unit", "geometry"]].to_crs(LUREF)
    whole["a"] = whole.area
    pieces = pd.concat([pieces, whole], ignore_index=True)
    print(f"    {len(whole)} snapped hexes kept whole ({whole['population'].sum():,.0f} people)")
    if abs(pieces["population"].sum() - hexes["population"].sum()) > 1.0:
        raise SystemExit("cutting the hexes lost people")
    print(f"    {len(hexes):,} hexes cut into {len(pieces):,} pieces by commune")

    census = units.set_index("unit")["pop"]
    kon = pieces.groupby("unit")["population"].sum()
    empty = sorted(u for u in census.index if kon.get(u, 0.0) <= 0)
    if empty:
        raise SystemExit(f"communes with people and no Kontur piece: {empty}")
    nat = kon.sum() / census.sum()
    rel = (kon / census / nat).sort_values()
    print(f"    Kontur over the census, per commune, relative to national {nat:.3f}: the five lowest "
          + ", ".join(f"{name[u]} {rel[u]:.2f}" for u in rel.index[:5]) + "; the five highest "
          + ", ".join(f"{name[u]} {rel[u]:.2f}" for u in rel.index[-5:]))

    factor = census / kon
    pieces["pop"] = pieces["population"] * pieces["unit"].map(factor)
    check = pieces.groupby("unit")["pop"].sum()
    if (check - census).abs().max() > 0.5:
        raise SystemExit("calibration did not reproduce the census")
    dens = pieces["pop"] / (pieces["a"] / 1e6)
    top = pieces.loc[dens.idxmax()]
    print(f"    calibrated to the census: {pieces['pop'].sum():,.0f} people; densest piece "
          f"{dens.max():,.0f}/km2 in {name[top['unit']]} ({top['a'] / 1e6:.3f} km2)")
    per = pieces.groupby("unit").size()
    print(f"    pieces per commune: median {per.median():.0f}, fewest {per.min()} ({name[per.idxmin()]})")
    out = pieces[["unit", "pop", "geometry"]].to_crs(4326)
    out.to_file(GRID_OUT, layer="grid", driver="GPKG")
    print(f"  wrote {GRID_OUT}  {len(out):,} pieces")


def main():
    if "--fetch" in sys.argv:
        fetch()
    units = build_units()
    build_grid(units)


if __name__ == "__main__":
    main()
