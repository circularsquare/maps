"""Namibia placement layer: religiondots' Kontur hexes re-keyed to the 2011 census's 107
constituencies, each cut into a town and a countryside part -> data/geo/na/na_hexes.gpkg.

    python sources/na_geo.py          (needs data/normalized/na.csv from sources/na_pums.py)

READ-ONLY INPUTS, from religiondots:
  data/geo/na/na_hexes.gpkg     Kontur 2023-11 400 m hexes keyed to the 14 regions, with the
                                border work done there (hexes across the Kavango and Zambezi
                                dropped, coastal ones snapped; religiondots' sources/na_geo.py)
  data/raw/na/shp/nam_admin2    COD-AB Namibia admin 2: the 107 constituencies of the 2008
                                delimitation, which the 2011 census used (pcodes NA<rr><cc>;
                                sources/na_pums.py has why they are the census's codes)

HOW. Each hex goes to the constituency its centroid falls in; a hex whose centroid is in none
(religiondots snapped it in from just over a coast or river line) goes to the nearest
constituency of its own region. Every hex's constituency must lie in the region religiondots put
it in. Then, as for Zambia (sources/zm_geo.py), each constituency is two units, `<pcode>-U` and
`<pcode>-R`: its hexes ranked by Kontur population (all one size, so by density), the densest
labelled town until they hold the sample's urban share of the constituency's people. A
constituency the 2011 census counts as all rural has no `-U` unit, one all urban no `-R`.

This decides only where inside a constituency each residence's dots go, never how many.

CHECKS (all must pass): the 107 pcodes are na.csv's; every hex's constituency is in its
religiondots region; every constituency x residence with people has hexes; the town share of
Kontur people against the census's urban share; Kontur per constituency against the 2011
counts, as a log correlation against a shuffle of constituencies within each region (what says
the join carries information below the region, which religiondots already checked).
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import geopandas as gpd  # noqa: E402
import pyogrio  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from rdlink import RD, RD_GEO  # noqa: E402

SRC = RD_GEO / "na" / "na_hexes.gpkg"
ADM2 = RD / "data" / "raw" / "na" / "shp" / "nam_admin2.shp"
NORM = HERE / "data" / "normalized" / "na.csv"
OUT = HERE / "data" / "geo" / "na" / "na_hexes.gpkg"
SNAP_MAX_KM = 5          # a centroid outside every constituency must be this close to one


def main():
    ok = True

    def report(good, msg):
        nonlocal ok
        ok &= bool(good)
        print(f"  {'OK ' if good else 'BAD'} {msg}")

    hx = pyogrio.read_dataframe(SRC)
    cod = gpd.read_file(ADM2)[["adm2_pcode", "adm2_name", "adm1_pcode", "geometry"]]
    df = pd.read_csv(NORM, dtype={"geo_id": str})
    report(set(cod["adm2_pcode"]) == set(df["geo_id"]) and len(cod) == 107,
           f"COD-AB's {len(cod)} constituencies are na.csv's {df['geo_id'].nunique()}")

    # centroids in an equal-area CRS for the snap distance; joins in that CRS too
    aea = "+proj=aea +lat_1=-18 +lat_2=-26 +lat_0=-22 +lon_0=17 +datum=WGS84 +units=m"
    cod_m = cod.to_crs(aea)
    pts = gpd.GeoDataFrame({"region": hx["unit"].astype(str).to_numpy()},
                           geometry=hx.geometry.to_crs(aea).centroid.to_numpy(), crs=aea)
    j = gpd.sjoin(pts, cod_m[["adm2_pcode", "adm1_pcode", "geometry"]], how="left",
                  predicate="within")
    j = j[~j.index.duplicated(keep="first")].reindex(pts.index)
    out = j["adm2_pcode"].isna()
    print(f"  {len(hx):,} hexes, {hx['pop'].sum():,.0f} people; {int(out.sum())} hexes "
          f"({hx.loc[out, 'pop'].sum():,.0f} people) with a centroid in no constituency")
    con = j["adm2_pcode"].copy()
    far = 0.0
    for i in np.flatnonzero(out.to_numpy()):
        reg = pts.at[i, "region"]
        cand = cod_m[cod_m["adm1_pcode"] == reg]
        d = cand.distance(pts.geometry.iloc[i])
        con.iat[i] = cand.loc[d.idxmin(), "adm2_pcode"]
        far = max(far, float(d.min()))
    report(far <= SNAP_MAX_KM * 1000,
           f"each of those goes to the nearest constituency of its region, the farthest "
           f"{far / 1000:.1f} km away")
    reg_of = dict(zip(cod["adm2_pcode"], cod["adm1_pcode"]))
    wrong = (con.map(reg_of) != pts["region"]) & ~out
    report(not wrong.any(),
           f"every hex's constituency is in the region religiondots gave it "
           f"({int(wrong.sum())} not, {hx.loc[wrong.to_numpy(), 'pop'].sum():,.0f} people)")
    hx["con"] = con.to_numpy()

    # ---- Kontur per constituency against the 2011 counts ----
    cnt = df.groupby("geo_id")["count"].sum()
    kon = hx.groupby("con")["pop"].sum().reindex(cnt.index).fillna(0)
    report((kon > 0).all(), f"every constituency holds Kontur people "
                            f"({int((kon <= 0).sum())} do not)")
    nat = kon.sum() / cnt.sum()
    ratio = kon / cnt / nat
    lr = np.log(kon.clip(lower=1)) - np.log(cnt)
    regs = pd.Series(cnt.index.map(reg_of), index=cnt.index)
    # within-region correlation of log counts, against shuffles within each region
    def within_r(k):
        a = np.log(k.clip(lower=1)) - np.log(k.clip(lower=1)).groupby(regs).transform("mean")
        b = np.log(cnt) - np.log(cnt).groupby(regs).transform("mean")
        return float(np.corrcoef(a, b)[0, 1])
    r0 = within_r(kon)
    rng = np.random.default_rng(1)
    null = []
    for _ in range(1000):
        sh = kon.copy()
        for reg, idx in regs.groupby(regs).groups.items():
            sh.loc[idx] = rng.permutation(kon.loc[idx].to_numpy())
        null.append(within_r(sh))
    report(r0 > max(null),
           f"Kontur 2023 against the 2011 count, log correlation within regions {r0:.2f}; "
           f"1,000 shuffles within regions at most {max(null):.2f}")
    print(f"     Kontur / 2011 count per constituency over the national {nat:.2f}: "
          f"p10 {ratio.quantile(0.1):.2f}, median {ratio.median():.2f}, p90 {ratio.quantile(0.9):.2f}")
    for pc in list(ratio.sort_values().index[:3]) + list(ratio.sort_values().index[-3:]):
        print(f"       {pc} {cod.set_index('adm2_pcode').at[pc, 'adm2_name']:<26}"
              f"2011 {cnt[pc]:>8,}  Kontur {kon[pc]:>9,.0f}  {ratio[pc]:.2f}")

    # ---- town and countryside ----
    res = df.groupby(["geo_id", "residence"])["count"].sum().unstack(fill_value=0)
    for c in "UR":
        if c not in res:
            res[c] = 0
    res["share_u"] = res["U"] / (res["U"] + res["R"])
    hx = hx.sort_values(["con", "pop"], ascending=[True, False], kind="mergesort")
    lab = np.empty(len(hx), dtype=object)
    rows = []
    pop = hx["pop"].to_numpy()
    for c, idx in hx.groupby("con", sort=False).indices.items():
        p = pop[idx]
        share = res.at[c, "share_u"]
        if share <= 0:
            k = 0
        elif share >= 1:
            k = len(idx)
        else:
            cum = np.cumsum(p) / p.sum()
            k = int(np.searchsorted(cum, share) + 1)
            k = min(max(k, 1), len(idx) - 1)
        lab[idx[:k]] = "U"
        lab[idx[k:]] = "R"
        rows.append((c, share, p[:k].sum() / p.sum() if p.sum() else 0, k, len(idx)))
    hx["unit"] = hx["con"] + "-" + lab
    hx = hx.sort_index()
    chk = pd.DataFrame(rows, columns=["con", "census_u", "kontur_u", "hexes_u", "hexes"])
    need = {f"{c}-{r}" for c in res.index for r in "UR" if res.at[c, r] > 0}
    have = set(hx["unit"])
    report(need <= have, f"every constituency x residence with people has hexes "
                         f"({len(need)} needed, missing {sorted(need - have)[:4]})")
    d = (chk["kontur_u"] - chk["census_u"]).abs()
    report(d.max() < 0.10,
           f"town share of Kontur people against the census's urban share: median gap "
           f"{d.median():.3f}, largest {d.max():.3f} ({chk.loc[d.idxmax(), 'con']})")
    mixed = chk[(chk["census_u"] > 0) & (chk["census_u"] < 1)]
    print(f"     {len(mixed)} constituencies split, {int((chk['census_u'] == 0).sum())} all rural, "
          f"{int((chk['census_u'] >= 1).sum())} all urban; {int((lab == 'U').sum()):,} town hexes "
          f"of {len(hx):,}")
    if not ok:
        raise SystemExit("FAILED; nothing written")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    o = hx[["unit", "pop", "geometry"]]
    if OUT.exists():
        OUT.unlink()
    pyogrio.write_dataframe(o, OUT, layer="hexes")
    print(f"wrote {OUT}: {len(o):,} hexes, {o['unit'].nunique()} units")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
