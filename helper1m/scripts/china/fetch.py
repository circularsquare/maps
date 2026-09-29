"""Build helper1m/data/china/population.csv from the township counts.

2020 starts from zonal_pop.py: the ASPECT grid summed over our townships, and over
each (township, census county) piece, where the census counties are the county
panel's own 2020 polygons. The census county figures are then carried onto our
2014 boundaries through those pieces, by where the grid says people live, with
no matching of names at all.

The rule for how far to trust each side:

  * Each census county's figure is taken as it is, and the grid only says where
    inside that census county its people live. The grid is right to 0.2% at the
    median on the census's own polygons, and where it is not the error is
    usually its own: Hengnan +43% with every neighbour within 2%, Chuzhou's
    urban core +174%, Tongling's 义安区 +190%.
  * The census is used even where its reporting units and the ground part
    company (CENSUS_EVERYWHERE). With it off, the grid decides where people are
    in those prefectures and only the prefecture total follows the census.
    Zhengzhou's first row is
    管城+金水+郑东新区+经开区+航空港区, and much of the Zhengdong and airport
    zones' land is legally 中牟县 and 新郑市; the panel's polygons are sometimes
    out of date or mislabelled (新乡市's "原阳县+平原示范区" row sits on 新乡县).
    Such a prefecture is recognised by a development-zone row the grid is well
    off from, or by two census counties far off in opposite directions. Where
    its grid total is over or under, the difference comes off the census
    counties the grid overfilled (or goes onto the ones it underfilled).
  * Where a province's census counties sum to less than its bulletin, the
    missing people are ones the census books to no county at all (770k in
    Shaanxi, most likely Xixian New Area's), and they are handed back to the counties the grid
    holds them in. See row_factors.

Each township's 2020 figure is the sum of its pieces, each scaled by its census
county's factor. Provinces are then put onto their published census totals,
which moves them by well under a percent.

2010 uses the same pieces: each piece is scaled by its census county's own
2010/2020 ratio. Pieces in a census county with no 2010 figure take what the
province has left over, and each province is then put onto its published 2010
total.

2024 is a forward-cast by province, and it exists because the viewer
extrapolates from the last two years it has. With only 2010 and 2020 it carried
a decade of growth forward and reached 1,456 M for 2026, while China's
population actually peaked around 2021 and was 1,405 M at the end of 2025. No
township data exists after 2020, so every township in a province shares its
province's rate — the recent trend carries no within-province detail, and the
2010-2020 differential is what shows how a place was actually moving.

Levels 1-3 are sums of their townships, so every level reconciles with the one
below exactly.

Writes population.csv (code, level, year, pop), plus two audit tables:
panel_rows.csv (one row per census county: grid, census, factor) and
counties.csv (one row per helper1m county: raw grid sum, published figure, and
the census-implied figure validate_counties.py compares them with).
"""
import io
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data/china"

# County-level census panel: 2010 and 2020 on harmonised boundaries, with official
# codes and a 2020 polygon per row. Dong & Wang, github.com/leiii/census. It was
# assembled from local census bulletins, and Xinjiang's counties published none,
# so all 106 of Xinjiang's rows are empty.
PANEL = DATA / "census_county_2010-2020_v1.csv"
PIECES = DATA / "township_panel_pop2020.csv"

# Transcription slips in the panel. 衡东县 is 562,423 in the panel and 565,423 on
# hongheiku.com; with the latter Hunan's rows sum to its bulletin exactly, and
# with the panel's they fall exactly 3,000 short.
PANEL_FIXES = {"430424": 565423}

# Figures for panel rows the panel leaves empty, keyed by county_code. Xinjiang's
# come from hongheiku.com's transcription of the national county book; see the
# file's header for how they were checked.
SUPPLEMENT = HERE / "xinjiang_counties.csv"

# Year-end provincial resident population, China Statistical Yearbook 2025
# table 2-5. Both of its columns come from that one table so the ratio stays
# inside one series — the yearbook's year-end estimates are not the November
# census counts.
YEARBOOK = "yearbook_provinces.csv"
RECENT_YEAR = 2024

YEARS = [2010, 2020, RECENT_YEAR]

# Xinjiang's XPCC cities are each a prefecture of their own in the panel, but a
# division's regiments are scattered through the counties around its city, so the
# city's census figure and its polygon are not the same people. Each is pooled
# with the prefecture it sits in, as the regional bulletin itself counts six of
# them. 石河子's regiments straddle Changji and Tacheng, so those pool as one.
POOL_WITH = {
    "石河子市": "昌吉回族自治州", "五家渠市": "昌吉回族自治州",
    "胡杨河市": "昌吉回族自治州", "塔城地区": "昌吉回族自治州",
    "阿拉尔市": "阿克苏地区", "图木舒克市": "喀什地区", "北屯市": "阿勒泰地区",
    "铁门关市": "巴音郭楞蒙古自治州", "双河市": "博尔塔拉蒙古自治州",
    "可克达拉市": "伊犁哈萨克自治州", "昆玉市": "和田地区",
}

# Take every census county's figure as it is, everywhere — her call, 2026-09-29:
# the census is used wherever it has a figure. That includes the prefectures
# below where the census's units and the ground part company, so Zhongmou gets
# its census 703k although the grid holds 1.4M on its ground (the Zhengdong and
# airport zones' people, whom the census books to Jinshui). Setting this False
# lets the grid decide inside those prefectures instead, holding only the
# prefecture total to the census.
CENSUS_EVERYWHERE = True

# What marks a prefecture where the census's reporting units and the ground part
# company (used only when CENSUS_EVERYWHERE is False). A row naming a development zone (蜀山区+高新区+经开区) books the zone's
# people to it wherever they live; if the grid is more than ZONE_OFF from such a
# row, the zone's people are on some other county's ground. Two census counties
# more than PAIR_OFF off in opposite directions are a polygon out of date
# (Anyang's 殷都区 -71% beside 安阳县 +102%). In those prefectures the grid decides
# where people are; everywhere else each census county's figure is taken as is.
ZONE_ROW = r"[+＋]|新区|开发区|高新|经开|示范区|管理区|工业园"
ZONE_OFF = 0.10
PAIR_OFF = 0.25

# A census county the grid barely touches cannot carry a factor: a few hundred
# grid people scaled up to a census figure would be a spike, not a correction.
MIN_GRID = 1000

# Bounds on a census county's 2020 factor. Chuzhou's urban core needs 0.37, so
# the floor has to sit below it; anything past these is a join failure.
FACTOR_MIN, FACTOR_MAX = 0.2, 5.0

# A 2010/2020 ratio outside this is a boundary change between the two
# censuses masquerading as growth. Growth itself reaches 0.38 (Lhasa's 堆龙德庆区)
# and 0.44 (Yinchuan's 金凤区); Harbin's 香坊区 0.21 beside 平房区 3.8 is a
# boundary moved between them, and falls outside.
RATIO_MIN, RATIO_MAX = 0.25, 4.0


def load_panel(provinces):
    panel = pd.read_csv(PANEL, dtype={"county_code": str, "city_code": str},
                        usecols=["county", "county_code", "city", "city_code",
                                 "province", "popu_2020", "popu_2010"])
    for code, pop in PANEL_FIXES.items():
        hit = panel["county_code"] == code
        if not hit.any():
            raise SystemExit(f"PANEL_FIXES entry matches nothing: {code}")
        panel.loc[hit, "popu_2020"] = pop
    panel["source"] = np.where(panel["popu_2020"].notna(), "panel", "")
    if SUPPLEMENT.exists():
        sup = pd.read_csv(SUPPLEMENT, dtype={"county_code": str}, comment="#")
        sup = sup.set_index("county_code")
        fill = panel["county_code"].isin(sup.index) & panel["popu_2020"].isna()
        codes = panel.loc[fill, "county_code"]
        panel.loc[fill, "popu_2020"] = codes.map(sup["pop_2020"]).values
        panel.loc[fill, "popu_2010"] = codes.map(sup["pop_2010"]).values
        panel.loc[fill, "source"] = "supplement"
        print(f"  {int(fill.sum())} empty panel rows filled from {SUPPLEMENT.name}")
    # Hong Kong and Macau have rows but none of our townships; a row there would
    # only ever reach the few border cells our Shenzhen and Zhuhai townships share.
    panel["mainland"] = panel["province"].isin(provinces)
    return panel


def row_factors(panel, bulletin):
    """2020 factor per census county.

    First each census county is put onto its own figure, except in the
    prefectures flagged as ones where the census's units and the ground part
    company; those are put onto the sum of their census counties as a whole, the
    difference coming off the counties the grid overfilled or going onto the ones
    it underfilled.

    Then the people the panel does not place at all are handed back. Where a
    province's census counties sum to less than its bulletin total, the missing
    people are ones the census booked to no county row — in Shaanxi 770k, most
    likely Xixian New Area's, whose host counties the grid holds 900k over — and they are still on the ground where the grid has them,
    which is inside the counties the first step cut. So that shortfall is given
    back to the cut counties in proportion to their cuts, never above the grid.
    Hengnan is the other kind: Hunan's rows sum to its bulletin, Hengnan's
    neighbours all match to 2%, and the grid alone is wrong, so it stays cut.
    """
    p = panel
    p["target"] = p["grid"]
    fig = (p["mainland"] & (p["popu_2020"].fillna(0) > 0) & (p["grid"] >= MIN_GRID))
    p["pooled"] = fig
    xj = p["province"] == "新疆维吾尔自治区"
    p["pool"] = p["city"].where(~xj, p["city"].map(lambda c: POOL_WITH.get(c, c)))
    p["pool"] = p["province"] + p["pool"]
    rel = p["grid"] / p["popu_2020"] - 1
    # An XPCC city is a zone in the same sense: its regiments' people are on
    # other counties' ground.
    xpcc = xj & p["city"].isin(set(POOL_WITH) - {"塔城地区"})
    zone_off = fig & (p["county"].astype(str).str.contains(ZONE_ROW) | xpcc) & \
        (rel.abs() > ZONE_OFF)
    far = fig & (rel.abs() > PAIR_OFF)
    p["grid_decides"] = False
    for key, g in p[fig].groupby("pool"):
        # Only where the prefecture shows the census's reporting units and the
        # ground parting company: a zone row the grid disagrees with, or two
        # census counties far off in opposite directions (a stale polygon).
        # Everywhere else each census county's own figure is taken as it is.
        split = not CENSUS_EVERYWHERE and (zone_off[g.index].any() or (
            (rel[g.index][far[g.index]] > 0).any() and (rel[g.index][far[g.index]] < 0).any()))
        if not split:
            p.loc[g.index, "target"] = g["popu_2020"]
            continue
        p.loc[g.index, "grid_decides"] = True
        d = g["grid"] - g["popu_2020"]
        excess = d.sum()
        side = d.clip(lower=0) if excess > 0 else (-d).clip(lower=0)
        if side.sum() <= 0:
            continue
        p.loc[g.index, "target"] = g["grid"] - excess * side / side.sum()

    split = p[p["grid_decides"]].drop_duplicates("pool")
    print(f"  the grid decides inside {len(split)} prefectures: "
          f"{', '.join(split['city'].astype(str).head(40))}")

    p["returned"] = 0.0
    placed = p[p["mainland"]].assign(
        placed=lambda q: q["popu_2020"].where(q["pooled"], q["grid"]).fillna(0))
    placed = placed.groupby("province")["placed"].sum()
    for prov, have in placed.items():
        short = bulletin.get(prov, have) - have
        mine = fig & (p["province"] == prov)
        cut = (p.loc[mine, "grid"] - p.loc[mine, "target"]).clip(lower=0)
        if short <= 0 or cut.sum() <= 0:
            continue
        back = min(short, cut.sum()) * cut / cut.sum()
        p.loc[back.index, "target"] += back
        p.loc[back.index, "returned"] = back
        print(f"    {prov}: census rows {short:,.0f} short of the bulletin; "
              f"{back.sum():,.0f} handed back to the counties the grid holds them in")
    raw = (p["target"] / p["grid"]).where(p["grid"] > 0, 1.0)
    p["factor"] = raw.clip(FACTOR_MIN, FACTOR_MAX)
    clipped = p[(raw - p["factor"]).abs() > 1e-9]
    if len(clipped):
        print(f"  {len(clipped)} census counties' factors clipped to "
              f"{FACTOR_MIN}-{FACTOR_MAX}:")
        for r in clipped.itertuples():
            print(f"    {r.province} {r.city} {r.county}: {raw[r.Index]:.2f}")
    return p


def main():
    units = pd.read_csv(DATA / "units.csv", dtype={"code": str})
    names = {lvl: df.set_index("code")["name_cn"] for lvl, df in units.groupby("level")}
    ref2010 = pd.read_csv(HERE / "census2010_provinces.csv").set_index("name_cn")["pop_2010"]
    ref2020 = pd.read_csv(HERE / "census2020_provinces.csv").set_index("name_cn")["pop_2020"]

    panel = load_panel(set(names[1].values))
    pieces = pd.read_csv(PIECES, dtype={"code": str, "panel_code": str},
                         keep_default_na=False)
    pieces["pop"] = pieces["pop"].astype(float)
    print(f"pieces: {len(pieces)}, {pieces['pop'].sum():,.0f} grid people, "
          f"{pieces.loc[pieces['code'] == '', 'pop'].sum():,.0f} outside our townships")

    panel["grid"] = panel["county_code"].map(
        pieces.groupby("panel_code")["pop"].sum()).fillna(0.0)
    panel = row_factors(panel, ref2020)
    has = panel["pooled"]
    print(f"  {int(has.sum())} of {len(panel)} census counties carry a figure, holding "
          f"{panel.loc[has, 'grid'].sum() / 1e6:,.1f} M of the grid; the rest keep the "
          f"grid as it is")
    moved = panel[has & ((panel["factor"] - 1).abs() > 0.10)]
    print(f"  {len(moved)} census counties moved by more than 10%, the largest:")
    for r in moved.reindex((moved["factor"] - 1).abs()
                           .sort_values(ascending=False).index).head(8).itertuples():
        print(f"    {r.province} {r.city} {r.county[:24]}: grid {r.grid:,.0f} -> "
              f"{r.target:,.0f} (census {r.popu_2020:,.0f})")

    # Our pieces only; the few grid people outside every township are dropped here.
    pc = pieces[pieces["code"] != ""].copy()
    pc["prov"] = pc["code"].str[:2]
    row = panel.set_index("county_code")
    pc["factor"] = pc["panel_code"].map(row["factor"]).fillna(1.0)
    pc["p2020"] = pc["pop"] * pc["factor"]

    # Provinces onto their published census totals.
    target20 = names[1].map(ref2020)
    sums = pc.groupby("prov")["p2020"].sum()
    scale = target20 / sums
    pc["p2020"] *= pc["prov"].map(scale)
    print(f"  province 2020 scaling: median {scale.median():.4f}, "
          f"range {scale.min():.4f}-{scale.max():.4f}")
    for code, s in scale[(scale - 1).abs() > 0.005].sort_values().items():
        print(f"    {names[1][code]} {s:.4f}")

    # 2010: each piece carries its census county's own ratio.
    ratio = (row["popu_2010"] / row["popu_2020"]).where(row["pooled"])
    ratio = ratio.where(ratio.between(RATIO_MIN, RATIO_MAX))
    pc["ratio"] = pc["panel_code"].map(ratio)
    known = pc["ratio"].notna()
    target10 = names[1].map(ref2010)
    done = (pc.loc[known, "p2020"] * pc.loc[known, "ratio"]).groupby(pc["prov"]).sum()
    rest = pc.loc[~known, "p2020"].groupby(pc["prov"]).sum()
    fill = ((target10 - done.reindex(target10.index, fill_value=0)) /
            rest.reindex(target10.index)).clip(RATIO_MIN, RATIO_MAX)
    pc.loc[~known, "ratio"] = pc.loc[~known, "prov"].map(fill)
    share = rest.reindex(target10.index, fill_value=0) / pc.groupby("prov")["p2020"].sum()
    print(f"  2010: {known.mean():.1%} of pieces carry their own census county's ratio")
    for code in share[share > 0.05].index:
        print(f"    {names[1][code]}: {share[code]:.0%} of people on the province "
              f"residual, ratio {fill[code]:.3f}")
    pc["p2010"] = pc["p2020"] * pc["ratio"]
    norm = target10 / pc.groupby("prov")["p2010"].sum()
    pc["p2010"] *= pc["prov"].map(norm)
    print(f"  province 2010 normalisation: median {norm.median():.4f}, "
          f"range {norm.min():.4f}-{norm.max():.4f}")
    for code, s in norm[(norm - 1).abs() > 0.01].sort_values().items():
        print(f"    {names[1][code]} {s:.4f}")

    # Townships, rounded once at the end.
    town = pc.groupby("code")[["p2020", "p2010"]].sum()
    pops = pd.DataFrame({"code": names[4].index})
    pops["pop_2020"] = pops["code"].map(town["p2020"]).fillna(0).round().astype("int64")
    pops["pop_2010"] = pops["code"].map(town["p2010"]).fillna(0).round().astype("int64")
    print(f"  2020 total {pops['pop_2020'].sum():,} (31 provinces {int(ref2020.sum()):,})")
    print(f"  2010 total {pops['pop_2010'].sum():,} (31 provinces {int(ref2010.sum()):,})")

    # Forward-cast by province. The yearbook's own two columns give the rate, so
    # it stays inside one series rather than mixing a census count with a
    # year-end estimate.
    yb = pd.read_csv(HERE / YEARBOOK, comment="#")
    recent = (yb.set_index("name_cn")[f"pop_{RECENT_YEAR}"] /
              yb.set_index("name_cn")["pop_2020"])
    prov_ratio = names[1].map(recent)
    missing = prov_ratio[prov_ratio.isna()]
    if len(missing):
        raise SystemExit(f"no yearbook row for provinces {list(missing.index)}")
    pops[f"pop_{RECENT_YEAR}"] = (
        pops["pop_2020"] * pops["code"].str[:2].map(prov_ratio)).round().astype("int64")
    print(f"  {RECENT_YEAR} total {pops[f'pop_{RECENT_YEAR}'].sum():,} "
          f"(the yearbook's 31 provinces summed to {int(yb[f'pop_{RECENT_YEAR}'].sum()):,})")

    # Roll up: every level is the sum of its townships, so the levels reconcile.
    cols = [f"pop_{y}" for y in YEARS]
    rows = []
    for lvl, width in ((4, 9), (3, 6), (2, 4), (1, 2)):
        grouped = pops.groupby(pops["code"].str[:width])[cols].sum()
        for year in YEARS:
            rows.append(pd.DataFrame({"code": grouped.index, "level": lvl,
                                      "year": year, "pop": grouped[f"pop_{year}"].values}))
    out = pd.concat(rows, ignore_index=True).sort_values(
        ["level", "code", "year"], kind="stable")
    out.to_csv(DATA / "population.csv", index=False)
    print(f"  wrote population.csv: {len(out)} rows")

    # Audit tables.
    panel.drop(columns=["mainland"]).to_csv(DATA / "panel_rows.csv", index=False,
                                            encoding="utf-8")
    pc["county"] = pc["code"].str[:6]
    # What each of our counties would hold if every census county's figure were
    # taken at face value and split by grid share: the comparison validate_counties
    # prints, and the measure of what the prefecture rule chose to leave alone.
    cen = row["popu_2020"].where(row["pooled"])
    pc["implied"] = pc["pop"] / pc["panel_code"].map(row["grid"]) * pc["panel_code"].map(cen)
    pc["covered"] = pc["panel_code"].map(cen).notna()
    cty = pc.groupby("county").agg(raw_2020=("pop", "sum"), pub_2020=("p2020", "sum"),
                                   implied_2020=("implied", "sum"))
    cov = pc[pc["covered"]].groupby("county")["pop"].sum()
    cty["covered"] = (cov / cty["raw_2020"]).reindex(cty.index).fillna(0)
    top = pc.sort_values("pop").drop_duplicates("county", keep="last").set_index("county")
    cty["top_code"] = top["panel_code"]
    cty["top_row"] = top["panel_code"].map(row["county"])
    cty["top_share"] = top["pop"] / cty["raw_2020"]
    cty.insert(0, "cnty", cty.index.map(names[3]))
    cty.insert(0, "pref", cty.index.str[:4].map(names[2]))
    cty.insert(0, "prov", cty.index.str[:2].map(names[1]))
    cty.index.name = "code"
    cty.round(1).to_csv(DATA / "counties.csv", encoding="utf-8")
    print("  wrote panel_rows.csv and counties.csv (the audit trail)")


if __name__ == "__main__":
    main()
