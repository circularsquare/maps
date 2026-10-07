"""China 2020 census, population by nationality (minzu) per county -> data/normalized/cn.csv.

    python sources/cn_ethnic.py

NOTHING IS FETCHED HERE. The county x 56-nationality table is chinaethnicity's (maps/chinaethnicity,
Anita's 2020 nationality dot map), read-only from its data/work/:

  leaves_2020.csv      16 provinces MEASURED: each province's own 2020 census yearbook, table 1-4,
                       county rows, parsed and checked there (prefectures sum to counties in every
                       column, each province equals the NBS national table)
  leaves_fallback.csv  15 provinces ESTIMATED: the 2000 census county pattern scaled to the 2020
                       prefecture rows (Hebei, Liaoning, Hunan, Sichuan) or the 2020 province total
                       (the other 11); Anita's approved method there (chinaethnicity/NOTES.md)

Each row carries `adcodes`, "code:weight;code:weight", chinaethnicity's own split of a census
row (a county, a development zone, a carved-out district) over its county polygons. Where the
shares are given (development zones) they are applied as given; a bare list of codes (a row
whose polygons were split or merged since) is split by the placement grid's population, where
chinaethnicity's units.py uses ASPECT's, so a few counties differ slightly from its pies. Fractional people (the estimated provinces are scaled) are kept as floats; the
scatter rounds.

In the 15 estimated provinces each county is then put onto its 2020 census total from Dong and
Wang's county panel, keeping its nationality mix (sources/cn_totals.py; 2026-10-06): the 2000
pattern drew Shenzhen at 10.5M against a census 17.6M.

geo_id is the county adcode of chinaethnicity's counties.gpkg (religiondots' DataV boundaries);
`estimated` says whether the province is one of the 15. The checks: every weight list sums to
1, every adcode is a polygon of counties.gpkg, and the country adds to 1,409,778,723, the 31
provinces' census total (the national 1,411,778,724 adds 2 million serving military).
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
CE = HERE.parent / "chinaethnicity" / "data" / "work"
OUT = HERE / "data" / "normalized" / "cn.csv"
GRID = HERE / "data" / "geo" / "cn" / "cn_grid_3km.gpkg"     # sources/cn_geo.py
TOTAL_31 = 1_409_778_723


def main():
    meas = pd.read_csv(CE / "leaves_2020.csv", dtype={"adcodes": str})
    est = pd.read_csv(CE / "leaves_fallback.csv", dtype={"adcodes": str})
    meas["estimated"], est["estimated"] = 0, 1
    if set(meas["prov"]) & set(est["prov"]):
        raise SystemExit(f"a province is both measured and estimated: "
                         f"{sorted(set(meas['prov']) & set(est['prov']))}")
    if len(set(meas["prov"])) != 16 or len(set(est["prov"])) != 15:
        raise SystemExit(f"provinces: {len(set(meas['prov']))} measured, "
                         f"{len(set(est['prov']))} estimated; chinaethnicity has 16 and 15")
    groups = list(meas.columns[meas.columns.get_loc("total") + 1:meas.columns.get_loc("estimated")])
    if len(groups) != 58:   # 56 nationalities, undetermined, naturalised
        raise SystemExit(f"{len(groups)} group columns, expected 58")
    df = pd.concat([meas, est], ignore_index=True)
    off = (df[groups].sum(axis=1) - df["total"]).abs()
    if (off > 0.01 * df["total"].clip(lower=100)).any():
        raise SystemExit(f"{int((off > 1).sum())} rows whose groups do not add to their total")

    import geopandas as gpd
    grid = gpd.read_file(GRID, ignore_geometry=True)
    gpop = grid.groupby("unit")["pop"].sum()
    rows = []
    for r in df.itertuples(index=False):
        parts = [p.split(":") for p in r.adcodes.split(";")]
        if all(len(p) == 2 for p in parts):
            # join.py's explicit shares (development zones), printed to four decimals
            w = [(c, float(x)) for c, x in parts]
            if abs(sum(x for _, x in w) - 1) > 1e-3:
                raise SystemExit(f"{r.name}: weights sum to {sum(x for _, x in w)}")
        else:
            # a census row on several polygons with no shares: chinaethnicity splits it by
            # ASPECT population; here by the placement grid's population, the same idea
            w = [(p[0], float(gpop.get(p[0], 0.0))) for p in parts]
            if sum(x for _, x in w) <= 0:
                raise SystemExit(f"{r.name}: no grid population in {r.adcodes}")
        s = sum(x for _, x in w)
        for code, x in w:
            rows.append((code, int(r.prov), int(r.estimated), x / s, r))
    long = []
    for code, prov, estd, x, r in rows:
        for g in groups:
            v = getattr(r, g) * x
            if v > 0:
                long.append((code, prov, estd, g, v))
    out = pd.DataFrame(long, columns=["geo_id", "prov", "estimated", "source_category", "count"])
    out = out.groupby(["geo_id", "prov", "estimated", "source_category"], as_index=False)["count"].sum()

    poly = gpd.read_file(HERE.parent / "chinaethnicity" / "data" / "geo" / "counties.gpkg",
                         ignore_geometry=True)
    poly["adcode"] = poly["adcode"].astype(str)
    codes = set(poly["adcode"])
    stray = sorted(set(out["geo_id"]) - codes)
    if stray:
        raise SystemExit(f"adcodes not in chinaethnicity's counties.gpkg: {stray[:10]}")
    if out.groupby("geo_id")["prov"].nunique().max() > 1:
        raise SystemExit("a county is filled from two provinces")

    # The 15 estimated provinces: each county onto its 2020 census total (sources/cn_totals.py),
    # its nationality mix kept. Measured provinces are the census's own county rows already.
    import cn_totals
    ctot = out.groupby("geo_id")["count"].sum()
    cprov = out.groupby("geo_id")["prov"].first()
    cest = out.groupby("geo_id")["estimated"].max()
    new = cn_totals.targets(ctot, cprov, cest,
                            pref=dict(zip(poly["adcode"], poly["city_code"].astype(str))),
                            grid=gpop, name=dict(zip(poly["adcode"], poly["name"])))
    fac = (new / ctot.reindex(new.index)).rename("f")
    pref = dict(zip(poly["adcode"], poly["city_code"].astype(str)))
    area_of = {u: cn_totals.area_key(u, cprov[u], pref[u]) for u in new.index}
    est_rows = out["estimated"] == 1
    e = out[est_rows]
    # the census nationality totals chinaethnicity scaled each area to (province, or prefecture)
    col_target = e.groupby([e["geo_id"].map(area_of), "source_category"])["count"].sum().to_dict()
    e = e.assign(count=e["count"] * e["geo_id"].map(fac))
    if e["count"].isna().any():
        raise SystemExit("an estimated county got no census total")
    # Rescaling with the mix fixed moves the nationality totals (Kazakh -15%, Guizhou's Tujia -15%:
    # the counties that grew are not the ones those groups live in); rake back to them
    e = cn_totals.rake(e, area_of, col_target, new)
    out = pd.concat([out[~est_rows], e[out.columns]], ignore_index=True)
    moved = (new - ctot.reindex(new.index)).abs().sum() / 2
    print(f"  estimated provinces rescaled to the 2020 county census: {moved:,.0f} people moved "
          f"between counties; factors {fac.min():.2f}-{fac.max():.2f}")
    tot = out["count"].sum()
    if abs(tot - TOTAL_31) > 5:
        raise SystemExit(f"total {tot:,.0f} != {TOTAL_31:,}")
    out.insert(1, "geo_level", "county")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False, float_format="%.3f")
    print(f"wrote {OUT}: {len(out):,} rows, {out['geo_id'].nunique():,} counties, "
          f"{tot:,.0f} people (31-province census total {TOTAL_31:,}); "
          f"{out.loc[out['estimated'] == 1, 'count'].sum():,.0f} in the 15 estimated provinces")
    nat = out.groupby("source_category")["count"].sum().sort_values(ascending=False)
    print(nat.head(12).round().astype(int).to_string())


if __name__ == "__main__":
    main()
