"""Japan: the placement layer, Kontur 400 m hexagons keyed to the 47 prefectures.

    python sources/jp_grid.py        -> data/geo/jp/jp_hexes.gpkg, data/geo/jp/jp_prefectures.gpkg

Replaces the one-unit layer `sources/jp.py` wrote while Japan was drawn as a single unit.
`countries.py` uses it to decide where inside a prefecture a dot sits, never how many there are;
the counts are `sources/jp_alloc.py`'s.

**BOUNDARIES ARE OCHA COD-AB `jpn_admbnda_adm1_2019`**, 47 features, already on disk in the maps
repo for asia1m (`data/asia1m/japan/`). Its `ADM1_PCODE` is `JP` plus the JIS X 0401 prefecture
code, which is the order every Japanese statistical table prints prefectures in, so the join is
on the code and the Japanese name is asserted against it as a second witness.

**PLACEMENT IS BY HEX CENTROID**, taken in Kontur's own projection and then reprojected (taking
centroids after reprojecting moves them). A hex whose centroid lands in the sea off a coastline
the 2019 polygons simplify is snapped to the nearest prefecture rather than dropped: the loss
from dropping is all on the seaward edge ([[reference_archipelago_grid_snap]]).

**THE CHECK IS KONTUR AGAINST 人口推計 2024 PER PREFECTURE.** Kontur is modelled from GHSL and
building footprints and the estimate is the Statistics Bureau's, so two independent figures for
47 prefectures have to agree, and a join that put Kanagawa's hexes on Tokyo's code would not.
With 47 units the shuffled null has real power, unlike Türkiye's twelve.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "6")          # [[feedback_cap_cpu]]
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")
import random
import sys

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "jp")
GEO = os.path.join(ROOT, "data", "geo", "jp")
KONTUR = os.path.join(RAW, "kontur_population_JP_20231101.gpkg")
POP = os.path.join(RAW, "jinsui_2024_table2.xlsx")
CODAB = os.path.join(os.path.dirname(ROOT), "data", "asia1m", "japan", "jpn_admbnda_adm1_2019.shp")
OUT = os.path.join(GEO, "jp_hexes.gpkg")
UNITS = os.path.join(GEO, "jp_prefectures.gpkg")

# JIS X 0401 order: 人口推計, NHK's 1996 chart and COD-AB's pcodes all use it.
PREFS = ["北海道", "青森", "岩手", "宮城", "秋田", "山形", "福島", "茨城", "栃木", "群馬", "埼玉",
         "千葉", "東京", "神奈川", "新潟", "富山", "石川", "福井", "山梨", "長野", "岐阜", "静岡",
         "愛知", "三重", "滋賀", "京都", "大阪", "兵庫", "奈良", "和歌山", "鳥取", "島根", "岡山",
         "広島", "山口", "徳島", "香川", "愛媛", "高知", "福岡", "佐賀", "長崎", "熊本", "大分",
         "宮崎", "鹿児島", "沖縄"]
POP_2024 = 123_802_000       # 人口推計 2024-10-01 第2表 総人口, the national row

UNIT_BAND = 1.5              # per prefecture, Kontur / estimate after the national ratio
MAX_SNAP_KM = 25             # a snap further than this is an island the layer leaves out...
FAR_SNAP_PEOPLE = 20_000     # ...and more people than this on such islands means a bad layer


def unit_of(pref):
    return f"JP{PREFS.index(pref) + 1:02d}"


def prefecture_population():
    """人口推計 2024-10-01 第2表 -> {prefecture: people}. The workbook prints thousands.

    Parsed exactly as sources/jp_checks.py parses it (the last row naming a prefecture wins),
    and asserted against the national row, which that parse has already been checked on.
    """
    import openpyxl
    pop = {}
    for row in openpyxl.load_workbook(POP, data_only=True).worksheets[0].iter_rows(values_only=True):
        c = [x for x in row if x is not None]
        if len(c) >= 3 and isinstance(c[1], str) and isinstance(c[2], (int, float)):
            raw = c[1].strip()               # some cells carry trailing spaces
            nm = {"東京都": "東京", "京都府": "京都", "大阪府": "大阪", "北海道": "北海道"}.get(
                raw, raw.rstrip("県"))
            if nm in PREFS:
                pop[nm] = int(round(c[2] * 1000))
    if len(pop) != 47:
        raise SystemExit(f"jp_grid: population parsed for {len(pop)} prefectures, expected 47")
    tot = sum(pop.values())
    if abs(tot - POP_2024) > 47_000:      # 47 figures each rounded to the thousand
        raise SystemExit(f"jp_grid: prefectures sum to {tot:,}, national row is {POP_2024:,}")
    return pop


def main():
    import geopandas as gpd
    import pandas as pd

    for path, fix in ((KONTUR, "python sources/jp.py --fetch"), (POP, "python sources/jp_checks.py --fetch"),
                      (CODAB, "the asia1m COD-AB download")):
        if not os.path.exists(path):
            raise SystemExit(f"missing {path}: {fix}")

    adm = gpd.read_file(CODAB)
    if len(adm) != 47:
        raise SystemExit(f"COD-AB ADM1 has {len(adm)} features, expected 47")
    adm["unit"] = adm["ADM1_PCODE"].astype(str)
    expect = {unit_of(p): p for p in PREFS}
    for _, r in adm.iterrows():
        ja = str(r["ADM1_JA"]).strip()
        nm = ja if ja == "北海道" else ja[:-1]
        if expect.get(r["unit"]) != nm:
            raise SystemExit(f"COD-AB {r['unit']} is {ja}; JIS order says {expect.get(r['unit'])}")
    adm = adm.to_crs("EPSG:4326")
    print(f"COD-AB ADM1: 47 prefectures, pcode and Japanese name agree on all 47")

    hexes = gpd.read_file(KONTUR)
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    if popcol is None:
        raise SystemExit(f"no population column in {list(hexes.columns)}")
    hexes = hexes[hexes[popcol] > 0].reset_index(drop=True)
    print(f"Kontur: {len(hexes):,} populated hexes, {hexes[popcol].sum():,.0f} people, crs {hexes.crs}")

    pts = gpd.GeoDataFrame(geometry=hexes.geometry.centroid, crs=hexes.crs).to_crs(adm.crs)
    j = gpd.sjoin(pts, adm[["unit", "geometry"]], how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")]                     # a border point, once only
    unit = j["unit"].reindex(hexes.index)

    miss = unit.isna().to_numpy()
    if miss.any():
        m = pts[miss].to_crs("EPSG:3857")
        a = adm[["unit", "geometry"]].to_crs("EPSG:3857")
        near = gpd.sjoin_nearest(m, a, how="left", distance_col="dist")
        near = near[~near.index.duplicated(keep="first")]
        unit.loc[near.index] = near["unit"]
        # Web Mercator stretches distance by 1/cos(latitude), about 1.2 over Japan.
        km = near["dist"] / 1000 * np.cos(np.radians(35))
        people = hexes.loc[near.index, popcol]
        print(f"  {miss.sum():,} hexes ({people.sum():,.0f} people) outside every polygon, snapped "
              f"to the nearest prefecture: {people[km <= 2].sum():,.0f} of them within 2 km, "
              f"{people[km > MAX_SNAP_KM].sum():,.0f} further than {MAX_SNAP_KM} km")
        far = near[km > MAX_SNAP_KM]
        if len(far):
            # Islands the 2019 layer leaves out entirely. Each is printed with where it went,
            # because the nearest polygon is only right if it belongs to the same prefecture.
            g = far.assign(km=km[far.index], people=people[far.index]).to_crs("EPSG:4326")
            g["lon"], g["lat"] = g.geometry.x.round(0), g.geometry.y.round(0)
            for (u, lon, lat), grp in g.groupby(["unit", "lon", "lat"]):
                print(f"      near {lon:.0f}E {lat:.0f}N -> {u} {PREFS[int(u[2:]) - 1]}, up to "
                      f"{grp['km'].max():.0f} km, {grp['people'].sum():,.0f} people")
            if people[km > MAX_SNAP_KM].sum() > FAR_SNAP_PEOPLE:
                raise SystemExit("too many people snapped a long way; the layer does not cover Japan")
    if unit.isna().any():
        raise SystemExit("hexes left with no prefecture after snapping")
    hexes["unit"] = unit.to_numpy()

    pop = prefecture_population()
    est = pd.Series({unit_of(p): v for p, v in pop.items()})
    k = hexes.groupby("unit")[popcol].sum()
    if set(k.index) != set(est.index):
        raise SystemExit(f"prefectures with no hexes: {sorted(set(est.index) - set(k.index))}")
    nat = k.sum() / est.sum()
    ratio = (k / est / nat).reindex(est.index)
    print(f"\nKontur / 人口推計 2024 nationally {nat:.3f}; per prefecture after that ratio: "
          f"median {ratio.median():.3f}, range {ratio.min():.3f} to {ratio.max():.3f}")
    for u in list(ratio.sort_values().index[:3]) + list(ratio.sort_values().index[-3:]):
        print(f"    {u} {PREFS[int(u[2:]) - 1]:<4} {ratio[u]:.3f}  kontur {k[u]:>12,.0f}  "
              f"estimate {est[u]:>12,}")
    bad = ratio[(ratio < 1 / UNIT_BAND) | (ratio > UNIT_BAND)]
    if len(bad):
        raise SystemExit(f"{len(bad)} prefectures outside {UNIT_BAND}x: {list(bad.index)}")

    rng = random.Random(7)
    shuffled = list(est.index)
    rng.shuffle(shuffled)
    null = k.reindex(est.index).to_numpy() / est.reindex(shuffled).to_numpy() / nat
    n_bad = int(((null < 1 / UNIT_BAND) | (null > UNIT_BAND)).sum())
    print(f"  null (prefecture labels shuffled): {n_bad} of 47 outside the band, against 0 joined")
    if n_bad < 20:
        raise SystemExit("the shuffled null barely fails; the band cannot tell a join from noise")

    w, s, e, n = hexes.to_crs("EPSG:4326").total_bounds
    if not (122 < w < 125 and 140 < e < 155 and 20 < s < 25 and 44 < n < 46):
        raise SystemExit(f"jp: Kontur bbox {w:.2f},{s:.2f},{e:.2f},{n:.2f} is not Japan")

    os.makedirs(GEO, exist_ok=True)
    hexes["cellcode"] = "JP:" + hexes.index.astype(str)
    # **THE COLUMN MUST BE CALLED `pop`**: countries.py's `_kontur_place_weight` tests for that
    # name and falls back to equal shares per prefecture without it.
    out = hexes[["cellcode", "unit", popcol, "geometry"]].rename(columns={popcol: "pop"})
    out = out.to_crs("EPSG:4326")
    tmp = os.path.join(GEO, "jp_hexes.part.gpkg")
    out.to_file(tmp, layer="hexes", driver="GPKG")
    os.replace(tmp, OUT)
    tmp = os.path.join(GEO, "jp_prefectures.part.gpkg")
    adm[["unit", "ADM1_EN", "ADM1_JA", "geometry"]].to_file(tmp, layer="prefectures", driver="GPKG")
    os.replace(tmp, UNITS)
    print(f"\nwrote {OUT}: {len(out):,} hexes over {out['unit'].nunique()} prefectures")


if __name__ == "__main__":
    main()
