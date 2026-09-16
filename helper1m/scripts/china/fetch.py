"""Build helper1m/data/china/population.csv from the township counts.

2020 comes from zonal_pop.py — the census township counts recovered from the
ASPECT grid. 2010 is a back-cast: each township is scaled by its county's
2010/2020 ratio from the county census panel, the same uniform-within-parent
assumption India's subdistricts use.

2024 is a forward-cast by province, and it exists because the viewer
extrapolates from the last two years it has. With only 2010 and 2020 it carried
a decade of growth forward and reached 1,456 M for 2026, while China's
population actually peaked around 2021 and was 1,405 M at the end of 2025. No
township data exists after 2020, so every township in a province shares its
province's rate — the recent trend carries no within-province detail, and the
2010-2020 differential is what shows how a place was actually moving.

Levels 1-3 are sums of their townships rather than the published figures for
those units, so every level reconciles with the one below exactly. Each county we
can match is first put onto its own published census figure, and the counties
with no figure absorb what is left of their province — so the county level is
right where the census can say so, and the province and national totals stay
exact either way. See validate_counties.py.

Writes population.csv with columns code, level, year, pop.
"""
import io
import re
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data/china"

# County-level census panel: 2010 and 2020 on harmonised boundaries, with
# official codes. Dong & Wang, github.com/leiii/census.
PANEL = DATA / "census_county_2010-2020_v1.csv"

# Year-end provincial resident population, China Statistical Yearbook 2025
# table 2-5. Both of its columns come from that one table so the ratio stays
# inside one series — the yearbook's year-end estimates are not the November
# census counts.
YEARBOOK = "yearbook_provinces.csv"
RECENT_YEAR = 2024

YEARS = [2010, 2020, RECENT_YEAR]

# County-level suffixes that changed between our 2018 boundaries and the 2020
# census — 崇明县 became 崇明区, 腾冲县 became 腾冲市, and so on. Matching on the
# stem recovers those; the population check below throws out stems that collide.
COUNTY_SUFFIXES = ("自治县", "自治旗", "县", "市", "区", "旗")

# A matched county whose panel 2020 population is this far from our own zonal
# sum is the wrong county, not a boundary quibble. Rejecting those is what keeps
# a stem match from silently picking a same-named neighbour.
MATCH_TOLERANCE = 0.25

# Ratios outside this are a boundary change masquerading as growth.
RATIO_MIN, RATIO_MAX = 0.5, 2.0

# The panel merges a district with the development zones carved out of it into
# one row — 蜀山区+高新区+经开区, coded 340104;340171;340172. Those zones are not
# administrative divisions of their own: they are management committees running
# land that legally stays with the district, and the census gives them their own
# row only because they have their own committee. Our single polygon covers the
# whole merged row, so the row's ratio is the right one for it.
MERGED_SPLIT = re.compile(r"[+＋]")
PARENS = re.compile(r"[（(]([^）)]*)[）)]")

# Put every county we can onto its own published 2020 census figure, and let the
# counties with no figure absorb what is left of their province.
#
# This replaced a blanket province scaling, which multiplied every township in a
# province by one factor so the province matched its census total. That is the
# wrong instrument when a province's excess sits in one or two broken counties:
# Anhui's is mostly Chuzhou's urban core and Baohe, and the old scaling paid for
# them by shaving 4.3% off all 88 of Anhui's other counties, which were already
# right. The county is the level these maps get assembled at, so it is the level
# that has to be accurate.
ANCHOR_TO_CENSUS = True


def stem(name):
    for suffix in COUNTY_SUFFIXES:
        if name.endswith(suffix) and len(name) > len(suffix):
            return name[: -len(suffix)]
    return name


def merged_parts(name):
    """The constituent unit names of a multi-code panel row.

    市辖区（昌邑区+龙潭区+…） keeps what is inside the brackets, because the wrapper
    is not a name. 永登县+兰州新区（部分） drops the trailing qualifier and keeps
    both names. Parts that are zones rather than counties — 高新区, 曲江新区 — simply
    match nothing of ours, which is harmless.
    """
    name = str(name)
    inner = PARENS.search(name)
    if inner and MERGED_SPLIT.search(inner.group(1)):
        name = inner.group(1)
    else:
        name = PARENS.sub("", name)
    return [p.strip() for p in MERGED_SPLIT.split(name) if p.strip()]


def load_units():
    units = pd.read_csv(DATA / "units.csv", dtype={"code": str, "parent": str,
                                                   "group": str})
    by_level = {lvl: df.set_index("code") for lvl, df in units.groupby("level")}
    return units, by_level


def county_frame(units, by_level, pops):
    """Our counties with their Chinese name tuple and our own 2020 sum."""
    cnty = by_level[3].reset_index()[["code", "name_cn"]].rename(
        columns={"name_cn": "cnty"})
    cnty["pref"] = cnty["code"].str[:4].map(by_level[2]["name_cn"])
    cnty["prov"] = cnty["code"].str[:2].map(by_level[1]["name_cn"])
    ours = pops.groupby(pops["code"].str[:6])["pop_2020"].sum().rename("our_2020")
    cnty = cnty.merge(ours, left_on="code", right_index=True, how="left")
    return cnty


def match_counties(cnty, panel):
    """Attach each of our counties to a panel row, most specific key first.

    Returns the frame with panel_2010 / panel_2020 / match_key filled where a
    match survived the population check.
    """
    panel = panel.copy()
    panel["stem"] = panel["county"].map(stem)

    keys = [
        ("prov+pref+name", ["province", "city", "county"], ["prov", "pref", "cnty"]),
        ("prov+pref+stem", ["province", "city", "stem"], ["prov", "pref", "cnty_stem"]),
        ("prov+name", ["province", "county"], ["prov", "cnty"]),
        ("prov+stem", ["province", "stem"], ["prov", "cnty_stem"]),
    ]
    cnty = cnty.copy()
    cnty["cnty_stem"] = cnty["cnty"].map(stem)
    cnty["panel_2010"] = pd.NA
    cnty["panel_2020"] = pd.NA
    cnty["match_key"] = pd.NA
    # Which panel row backed the match. A merged row backs several of our
    # counties at once, and the province residual has to count it only once.
    cnty["panel_row"] = pd.NA
    cnty["panel_name"] = pd.NA
    claimed = set()

    for label, right_on, left_on in keys:
        todo = cnty["match_key"].isna()
        if not todo.any():
            break
        # Ambiguous keys are worse than no key: a wrong twin is invisible later.
        unique = panel[panel.groupby(right_on)[right_on[0]].transform("size") == 1].copy()
        unique["row_id"] = unique.index
        # "county" is already a key on some passes; selecting it twice is an error.
        cols = right_on + [c for c in ("popu_2010", "popu_2020", "row_id", "county")
                           if c not in right_on]
        merged = cnty.loc[todo, left_on].merge(
            unique[cols], left_on=left_on, right_on=right_on, how="left")
        merged.index = cnty.index[todo]

        ok = merged["popu_2020"].notna()
        # Reject a match whose population disagrees with our own sum.
        rel = (merged["popu_2020"] - cnty.loc[todo, "our_2020"]).abs() / \
            merged["popu_2020"]
        ok &= rel <= MATCH_TOLERANCE
        # The key has to be unique on our side too, and a row one of our counties
        # already took cannot back another. Jiangsu has a 鼓楼区 in both Nanjing
        # and Xuzhou: Nanjing's matches on the prefecture pass, and the Xuzhou one
        # would then claim that same Nanjing row on the looser prov+name pass.
        # Neither looks wrong against that row on its own, so only these two
        # checks catch it. Both fall through to the province residual instead.
        ok &= ~cnty.loc[todo].duplicated(subset=left_on, keep=False)
        ok &= ~merged["row_id"].isin(claimed)
        cnty.loc[merged.index[ok], "panel_2010"] = merged.loc[ok, "popu_2010"].values
        cnty.loc[merged.index[ok], "panel_2020"] = merged.loc[ok, "popu_2020"].values
        cnty.loc[merged.index[ok], "panel_row"] = merged.loc[ok, "row_id"].values
        cnty.loc[merged.index[ok], "panel_name"] = merged.loc[ok, "county"].values
        cnty.loc[merged.index[ok], "match_key"] = label
        claimed.update(merged.loc[ok, "row_id"].dropna().tolist())
        print(f"  matched on {label:<16} {int(ok.sum()):>5}  "
              f"(running total {int(cnty['match_key'].notna().sum())} of {len(cnty)})")

    return cnty


def match_merged(cnty, panel):
    """Match the counties still left over against the panel's multi-code rows.

    The population check here is made against the group rather than the single
    county. Our one 蜀山区 polygon covers the whole of 蜀山区+高新区+经开区, so
    testing it against the row's total is the honest test; testing Hangzhou's
    上城区 on its own against a four-district row would throw away a good match
    for the wrong reason. Summing every one of our counties that lands on a row
    and testing that sum is what still catches a row we do not actually cover.
    """
    cnty = cnty.copy()
    rows = panel[panel["county"].astype(str).str.contains(MERGED_SPLIT, na=False)]
    todo = cnty["match_key"].isna()
    if rows.empty or not todo.any():
        return cnty

    parts = []
    for idx, r in rows.iterrows():
        for part in merged_parts(r["county"]):
            parts.append({"province": r["province"], "city": r["city"],
                          "part": part, "row_id": idx, "row_name": r["county"],
                          "popu_2010": r["popu_2010"], "popu_2020": r["popu_2020"]})
    parts = pd.DataFrame(parts)
    # A name that points at two different rows in one prefecture is no key.
    parts = parts.drop_duplicates(subset=["province", "city", "part"], keep=False)

    cand = cnty.loc[todo, ["prov", "pref", "cnty", "our_2020"]].merge(
        parts, left_on=["prov", "pref", "cnty"],
        right_on=["province", "city", "part"], how="left")
    cand.index = cnty.index[todo]
    cand = cand[cand["row_id"].notna() & cand["popu_2010"].notna()
                & cand["popu_2020"].notna()]
    if cand.empty:
        return cnty

    ours = cand.groupby("row_id")["our_2020"].sum()
    theirs = cand.groupby("row_id")["popu_2020"].first()
    rel = (theirs - ours).abs() / theirs
    good = rel[rel <= MATCH_TOLERANCE].index
    keep = cand[cand["row_id"].isin(good)]
    if keep.empty:
        return cnty

    cnty.loc[keep.index, "panel_2010"] = keep["popu_2010"].values
    cnty.loc[keep.index, "panel_2020"] = keep["popu_2020"].values
    cnty.loc[keep.index, "panel_row"] = keep["row_id"].values
    cnty.loc[keep.index, "panel_name"] = keep["row_name"].values
    cnty.loc[keep.index, "match_key"] = "merged row"
    print(f"  matched on {'merged row':<16} {len(keep):>5}  "
          f"(running total {int(cnty['match_key'].notna().sum())} of {len(cnty)})")
    print(f"    {len(good)} of {cand['row_id'].nunique()} merged rows passed the "
          f"group population check, covering {keep['our_2020'].sum():,.0f} people")
    return cnty


def anchor_to_census(pops, cnty, ref2020, by_level):
    """Scale each matched county onto its published 2020 census figure.

    A merged panel row covers several of our counties at once, so the group is
    scaled as a unit — scaling one county onto a total it is only part of would
    be nonsense.

    Counties with no census figure then take up whatever their province has left:
    the published province total minus the published figures of the anchored
    counties is, by construction, what the rest of that province holds. So the
    province totals and the national total stay exact, nobody is dropped, and
    nobody is counted twice.
    """
    pops = pops.copy()
    county = pops["code"].str[:6]
    cur = pops.groupby(county)["pop_2020"].sum()

    have = cnty[cnty["panel_2020"].notna() & cnty["panel_row"].notna()].copy()
    now = have.groupby("panel_row")["code"].apply(lambda s: cur.reindex(s).sum())
    target = have.groupby("panel_row")["panel_2020"].first().astype(float)
    row_factor = (target / now).where(now > 0)

    factor = pd.Series(1.0, index=cur.index, dtype=float)
    f = pd.Series(have["panel_row"].map(row_factor).values, index=have["code"]).dropna()
    factor.loc[f.index] = f.values
    anchored = set(f.index)
    print(f"  anchored {len(anchored)} counties onto "
          f"{int(row_factor.notna().sum())} published census figures")

    prov_target = ref2020.set_index("name_cn")["pop_2020"]
    prov_of = pd.Series(cur.index.str[:2], index=cur.index)
    for pcode, pname in by_level[1]["name_cn"].items():
        here = cur.index[prov_of == pcode]
        rest = [c for c in here if c not in anchored]
        if not rest:
            continue
        taken = have[have["code"].isin(anchored) &
                     have["code"].str.startswith(pcode)] \
            .drop_duplicates("panel_row")["panel_2020"].astype(float).sum()
        left = float(prov_target.get(pname, 0)) - taken
        now_rest = float(cur.reindex(rest).sum())
        if left <= 0 or now_rest <= 0:
            print(f"    {pname}: nothing sensible left for its {len(rest)} "
                  f"unmatched counties, leaving them as measured")
            continue
        factor.loc[rest] = left / now_rest

    pops["pop_2020"] = (pops["pop_2020"] * county.map(factor)).round().astype("int64")
    moved = factor[(factor - 1).abs() > 0.10]
    print(f"  factors: median {factor.median():.4f}, "
          f"range {factor.min():.3f}-{factor.max():.3f}; "
          f"{len(moved)} counties moved by more than 10%")
    print(f"  2020 total now {pops['pop_2020'].sum():,}")
    return pops


def county_ratios(cnty, panel, ref2010, ref2020):
    """2010/2020 ratio per county.

    Matched counties take the panel's own ratio. Unmatched ones take the
    province residual — what the panel says is left in that province once the
    matched counties are accounted for. Averaging the matched ratios instead
    biases the result badly: the counties that fail to match are overwhelmingly
    the ones renamed when they became urban districts, which are the
    fastest-growing ones, so the average understates their growth. That error
    put the first national back-cast 46 M above the 2010 census.

    Then each province is normalised so its 2010/2020 ratio equals the published
    one, which fixes the level while keeping the county-to-county variation.
    """
    cnty = cnty.copy()
    raw = pd.to_numeric(cnty["panel_2010"], errors="coerce") / \
        pd.to_numeric(cnty["panel_2020"], errors="coerce")
    cnty["ratio"] = raw.where(raw.between(RATIO_MIN, RATIO_MAX))
    dropped = int((raw.notna() & cnty["ratio"].isna()).sum())
    if dropped:
        print(f"  dropped {dropped} ratios outside {RATIO_MIN}-{RATIO_MAX} "
              f"(boundary change, not growth)")

    cnty["prov_code"] = cnty["code"].str[:2]
    prov_cn = cnty.drop_duplicates("prov_code").set_index("prov_code")["prov"]

    # Only panel rows with both years can contribute to a residual.
    both = panel[panel["popu_2010"].notna() & panel["popu_2020"].notna()]
    panel_2010 = both.groupby("province")["popu_2010"].sum()
    panel_2020 = both.groupby("province")["popu_2020"].sum()

    have = cnty[cnty["ratio"].notna()]
    # One panel row can back several of our counties, because a merged row covers
    # a group of them. Counting its population once per county would inflate what
    # gets subtracted here and leave the province residual far too small — the
    # double count would land on whatever counties are still unmatched.
    uniq = have.drop_duplicates(subset=["panel_row"])
    m2010 = pd.to_numeric(uniq["panel_2010"]).groupby(uniq["prov_code"]).sum()
    m2020 = pd.to_numeric(uniq["panel_2020"]).groupby(uniq["prov_code"]).sum()

    national = float(both["popu_2010"].sum() / both["popu_2020"].sum())
    residual = {}
    for code, name in prov_cn.items():
        num = panel_2010.get(name, 0) - m2010.get(code, 0)
        den = panel_2020.get(name, 0) - m2020.get(code, 0)
        r = num / den if den > 0 else national
        residual[code] = min(max(r, RATIO_MIN), RATIO_MAX)

    need = cnty["ratio"].isna()
    cnty.loc[need, "ratio"] = cnty.loc[need, "prov_code"].map(residual)
    cnty["ratio_source"] = pd.Series("county", index=cnty.index).where(
        ~need, "province residual")
    print("  ratio source:", dict(cnty["ratio_source"].value_counts()))

    # Normalise each province onto its published 2010/2020 growth.
    target = (ref2010.set_index("name_cn")["pop_2010"] /
              ref2020.set_index("name_cn")["pop_2020"])
    ours = ((cnty["ratio"] * cnty["our_2020"]).groupby(cnty["prov_code"]).sum() /
            cnty["our_2020"].groupby(cnty["prov_code"]).sum())
    factor = (prov_cn.map(target) / ours).rename("factor")
    cnty["ratio"] = cnty["ratio"] * cnty["prov_code"].map(factor)
    worst = factor.reindex(factor.abs().sub(1).sort_values().index)
    print(f"  province normalisation factor: median {factor.median():.3f}, "
          f"range {factor.min():.3f}-{factor.max():.3f}")
    off = factor[(factor - 1).abs() > 0.02]
    for code, f in off.sort_values().items():
        print(f"    {prov_cn[code]} {f:.3f}")
    return cnty


def main():
    units, by_level = load_units()
    pops = pd.read_csv(DATA / "township_pop2020.csv", dtype={"code": str})
    print(f"townships: {len(pops)}, 2020 total {pops['pop_2020'].sum():,}")

    panel = pd.read_csv(PANEL, dtype={"county_code": str})
    print(f"county panel: {len(panel)} rows")

    ref2010 = pd.read_csv(HERE / "census2010_provinces.csv")
    ref2020 = pd.read_csv(HERE / "census2020_provinces.csv")

    # Matching runs on the raw zonal sums, so the population check compares a
    # measurement against the census rather than against an already-adjusted
    # number.
    cnty = county_frame(units, by_level, pops)
    cnty = match_counties(cnty, panel)
    cnty = match_merged(cnty, panel)

    if ANCHOR_TO_CENSUS:
        pops = anchor_to_census(pops, cnty, ref2020, by_level)
        # The 2010 ratios are weighted by our own 2020 figures, so refresh those.
        cnty["our_2020"] = cnty["code"].map(
            pops.groupby(pops["code"].str[:6])["pop_2020"].sum())

    cnty = county_ratios(cnty, panel, ref2010, ref2020)

    # Back-cast every township by its county's ratio.
    ratio_by_county = cnty.set_index("code")["ratio"]
    pops["ratio"] = pops["code"].str[:6].map(ratio_by_county)
    pops["pop_2010"] = (pops["pop_2020"] * pops["ratio"]).round().astype("int64")
    print(f"  2010 total {pops['pop_2010'].sum():,} "
          f"(the 31 provinces summed to {ref2010['pop_2010'].sum():,} in 2010)")

    # Forward-cast by province. The yearbook's own two columns give the rate, so
    # it stays inside one series rather than mixing a census count with a
    # year-end estimate.
    yb = pd.read_csv(HERE / YEARBOOK, comment="#")
    recent = (yb.set_index("name_cn")[f"pop_{RECENT_YEAR}"] /
              yb.set_index("name_cn")["pop_2020"])
    prov_ratio = by_level[1]["name_cn"].map(recent)
    missing = prov_ratio[prov_ratio.isna()]
    if len(missing):
        raise SystemExit(f"no yearbook row for provinces {list(missing.index)}")
    pops[f"pop_{RECENT_YEAR}"] = (
        pops["pop_2020"] * pops["code"].str[:2].map(prov_ratio)).round().astype("int64")
    print(f"  {RECENT_YEAR} total {pops[f'pop_{RECENT_YEAR}'].sum():,} "
          f"(the yearbook's 31 provinces summed to {int(yb[f'pop_{RECENT_YEAR}'].sum()):,})")
    grew = int((prov_ratio > 1).sum())
    print(f"    {grew} provinces up, {31 - grew} down since 2020")

    # Roll up: every level is the sum of its townships, so the levels reconcile.
    cols = [f"pop_{y}" for y in YEARS]
    rows = []
    for lvl, width in ((4, 9), (3, 6), (2, 4), (1, 2)):
        grouped = pops.groupby(pops["code"].str[:width])[cols].sum()
        for year in YEARS:
            rows.append(pd.DataFrame({
                "code": grouped.index,
                "level": lvl,
                "year": year,
                "pop": grouped[f"pop_{year}"].values,
            }))
    out = pd.concat(rows, ignore_index=True).sort_values(
        ["level", "code", "year"], kind="stable")
    path = DATA / "population.csv"
    out.to_csv(path, index=False)
    print(f"  wrote {path}: {len(out)} rows")
    for lvl in (1, 2, 3, 4):
        n = out[(out["level"] == lvl) & (out["year"] == RECENT_YEAR)]
        print(f"    adm{lvl}: {len(n)} units, {n['pop'].sum():,} in {RECENT_YEAR}")

    cnty.drop(columns=["cnty_stem"]).to_csv(
        DATA / "county_ratios.csv", index=False, encoding="utf-8")
    print(f"  wrote county_ratios.csv (the 2010 back-cast audit trail)")


if __name__ == "__main__":
    main()
