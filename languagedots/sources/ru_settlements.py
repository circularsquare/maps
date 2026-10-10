"""Russia: the 2021 census's nationality by settlement, as a PLACEMENT weight inside the 168 units
(AGENT_BRIEF §4.4) -> data/geo/ru/ru_settlement_nat.parquet.

    python sources/ru_settlements.py

THE SOURCE. To Be Precise (tochno.st), "Settlements of Russia: population, ethnic composition and
geographic coordinates", https://tochno.st/datasets/allsettlements, CC BY 4.0, keyless. Every
settlement of the 2021 census with its count for each of the census's 194 nationality columns
(plus "other answers" and the not-stated columns), with coordinates. The file used is
`data_allsettlements_anon_156_v20260925.parquet`, already downloaded for helper1m
(helper1m/data/russia/raw/tochno/, read-only here) and copied to data/raw/ru/tochno/. Download
links sit behind buttons on the dataset page; its HTML carries the storage.yandexcloud.net link.

WHAT IS WRITTEN. One row per settlement (plus Moscow, St Petersburg and Sevastopol's federal-city
rows, which have no settlement rows of their own for the cities themselves): lat, lon, `urban`
(город, пгт, рабочий / курортный / дачный посёлок, and the three federal cities), `subject` (the
ISO 3166-2 code of the placement-layer hex the point falls in, else of the nearest hex), `stated`
(people who stated a nationality) and the 194 nationality columns. Settlements of 10 people or
fewer have their nationality blanked (-9) by the publisher; they are written as 0, so they weigh
nothing. Nothing here is a count the map draws: countries/ru.py only reads shares from it.

CHECKS, asserted:
  1. settlements + federal-city rows hold the federation row's population less the blanked
     villages' and less what is double counted, within 1% (people outside any settlement are few);
  2. every point gets a subject; the subject found by position agrees with the publisher's own
     region name for 99%+ of the people;
  3. stated <= population on every settlement row.
"""
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
NAME = "data_allsettlements_anon_156_v20260925.parquet"
RAW = ROOT / "data" / "raw" / "ru" / "tochno" / NAME
HELPER = ROOT.parent / "helper1m" / "data" / "russia" / "raw" / "tochno" / NAME
PLACE = ROOT / "data" / "geo" / "ru" / "ru_grid_3km.gpkg"
OUT = ROOT / "data" / "geo" / "ru" / "ru_settlement_nat.parquet"

SETTLEMENT = "Населенный пункт"
FEDERAL = "Город федерального значения"
STATED = "Указавшие национальную принадлежность"
NAT_FIRST, NAT_LAST = 24, 218          # the 194 nationality columns, as helper1m reads them
URBAN_PREFIX = ("г.", "г ", "город ", "пгт", "рп ", "рабочий посёлок", "рабочий поселок",
                "кп ", "курортный посёлок", "курортный поселок", "дп ", "дачный посёлок",
                "дачный поселок")


def xyz(lon, lat):
    lo, la = np.radians(lon), np.radians(lat)
    return np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]) * 6371.0


def main():
    if not RAW.exists():
        if not HELPER.exists():
            raise SystemExit(f"{NAME} not found; download it from https://tochno.st/datasets/allsettlements "
                             f"into {RAW.parent}")
        RAW.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(HELPER, RAW)
        print(f"  copied {NAME} from helper1m")
    df = pd.read_parquet(RAW)
    cols = list(df.columns)
    if cols[NAT_FIRST - 1] != STATED or not cols[NAT_LAST].startswith("Указавшие другие ответы"):
        raise SystemExit("tochno column layout changed")
    nat = cols[NAT_FIRST:NAT_LAST]
    country = df[df["object_level"] == "Страна"].iloc[0]
    blanked = df.loc[df["object_level"] == "Регион (анонимизация)", "population"].sum()

    s = df[(df["object_level"] == SETTLEMENT) | (df["object_level"] == FEDERAL)].copy()
    if s["latitude"].isna().any():
        raise SystemExit(f"{s['latitude'].isna().sum()} rows without coordinates")
    name = s["object_name"].str.lower()
    s["urban"] = (s["object_level"] == FEDERAL) | name.str.startswith(URBAN_PREFIX)
    # the federal cities' rows hold their whole territory; their own villages are settlement
    # rows too, so for the composition the federal row stands only for the city
    # (urban), and its villages (rural) count once
    for c in [STATED] + nat:
        s[c] = s[c].fillna(0).clip(lower=0).astype(np.int32)
    s = s.copy()
    if (s[STATED] > s["population"]).any():
        raise SystemExit("stated > population on some row")

    sett_pop = s.loc[s["object_level"] == SETTLEMENT, "population"].sum()
    fed_pop = s.loc[s["object_level"] == FEDERAL, "population"].sum()
    fed_villages = s.loc[(s["object_level"] == SETTLEMENT)
                         & s["region"].isin(["Москва", "Санкт-Петербург", "Севастополь"]), "population"].sum()
    held = sett_pop + fed_pop - fed_villages
    ratio = held / country["population"]
    print(f"  check 1: {len(s):,} points hold {held:,} of {country['population']:,} people ({ratio:.4f}); "
          f"{blanked:,} in villages of 10 or fewer, nationality blanked")
    if abs(1 - ratio) > 0.01:
        raise SystemExit("settlements do not hold the federation's population")

    # subject by position: the placement hex the point falls in, else the nearest hex
    place = gpd.read_file(PLACE)[["unit", "geometry"]]
    place["subject"] = place["unit"].str.split("/").str[0]
    pts = gpd.GeoDataFrame(s[[]], geometry=gpd.points_from_xy(s["longitude"], s["latitude"]), crs=4326)
    hit = gpd.sjoin(pts, place[["subject", "geometry"]], predicate="within", how="left")
    hit = hit[~hit.index.duplicated()]
    subj = hit["subject"].reindex(s.index)
    miss = subj.isna().to_numpy()
    rp = place.geometry.representative_point()
    tree = cKDTree(xyz(rp.x.to_numpy(), rp.y.to_numpy()))
    _, j = tree.query(xyz(s.loc[miss, "longitude"].to_numpy(), s.loc[miss, "latitude"].to_numpy()))
    subj[miss] = place["subject"].to_numpy()[j]
    s["subject"] = subj.to_numpy()
    if s["subject"].isna().any():
        raise SystemExit("points without a subject")
    modal = s.groupby("region")["subject"].agg(lambda x: x.value_counts().index[0])
    agree = s["subject"] == s["region"].map(modal)
    share = s.loc[agree, "population"].sum() / s["population"].sum()
    print(f"  check 2: {miss.sum():,} points outside every hex took the nearest; position agrees "
          f"with the publisher's region for {agree.mean():.2%} of points, {share:.3%} of people "
          f"({len(modal)} regions -> {s['subject'].nunique()} subjects)")
    if share < 0.99 or modal.nunique() != len(modal):
        raise SystemExit("position and region name disagree too often, or two regions share a subject")

    out = s[["latitude", "longitude", "urban", "subject", "population", STATED] + nat].rename(
        columns={"latitude": "lat", "longitude": "lon", STATED: "stated"})
    out.columns = [c.split(" (")[0].strip() if c in nat else c for c in out.columns]
    if out.columns.duplicated().any():
        raise SystemExit("two nationality columns share a short name")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".tmp")
    out.reset_index(drop=True).to_parquet(tmp, index=False)
    os.replace(tmp, OUT)
    print(f"  check 3: stated <= population on every row\n  wrote {OUT.name}: {len(out):,} points, "
          f"{int(out['urban'].sum()):,} urban, {len(nat)} nationalities, {int(out['stated'].sum()):,} "
          "people with a nationality")


if __name__ == "__main__":
    main()
