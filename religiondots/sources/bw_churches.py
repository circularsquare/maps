"""Botswana — Catholics and Seventh-day Adventists split out of the 2011 census's Christians.

Reads data/raw/afrobarometer/*.sav (shared) and data/normalized/bw.csv; writes
data/normalized/bw_split.csv. `sources/bw.md` §10 is the record (2026-10-03, `fafd1067-chea`).

## THE CENSUS NEVER SPLIT ITS CHRISTIANS

The 2001, 2011 and 2022 censuses offer one `Christian` box beside Badimo, no religion and the
small religions (the 2022 *Analytical Report* Vol. 1, Table 3a and the 2001-2022 comparison table,
PDF p.107). Botswana has had no DHS since 1988. So each census locality's Christian count stays as
counted, and a survey splits it at a coarser grain: India's and Iran's `derived` split inside a
counted column. The church rows are `derived` with `parent_column=Christian` and roll back to
`christianity` when inferred dots are hidden.

## WHY ONLY TWO CHURCHES

The Afrobarometer's Botswanan Christians who name no church (`Christian only`) run 7, 33, 37, 69,
66 and 84% of Christians over rounds 4 to 9. That is not church-neutral. As a share of ALL
respondents, which is what a church keeps if its members go on naming it:

    round                R4     R5     R6     R7     R8     R9
    Christian only      5.0   25.6   28.9   56.9   54.6   70.7
    Catholic            3.5    4.3    3.7    2.7    3.4    1.2
    Adventist           2.4    1.6    2.3    1.7    1.6    1.0
    Zion Christian     12.9    9.1    9.1    6.3    7.9    4.1
    Pentecostal         7.8   10.1   13.5    5.7    4.6    1.5
    Lutheran            1.6    1.6    2.4    1.3    0.7    0.6
    Independent        26.2   16.5   12.5    2.6    5.3    2.8

Over rounds 4-8, while `Christian only` rises by 50 points and named Christians keep 0.45 of
their rounds 4-5 share in rounds 7-8, Catholics keep 0.78 of theirs and Adventists 0.81: they
are churches the probing does not move (Cameroon's rule, `sources/cm.md` §4), so their named
share is close to their level and is drawn, as a floor (`HOLD_MIN`). Round 9 is
left out: with 84% of Christians unnamed even these two fall. The others fall with the unnamed
(Lutheran keeps 0.63, Pentecostal 0.57, Independent 0.19) and are not drawn, though Lutheran
and Pentecostal pass the split-half; the card drops UCCSA (the Congregational church of the London Missionary Society)
after round 5, so its members' answers cannot be followed at all. The Zion Christian Church holds
about as well as Catholics after round 4 (a different box, `ZCC`, then), but as a share of each
unit's Christians it fails the split-half (+0.260 against a null 95th of +0.267, p 0.058), so it
stays on the parent too.

## THE UNITS

The survey's REGION is the census district, with the towns apart (`UNIT`). Jwaneng, Ngwaketse
West and Kgalagadi North have 30-50 Christian respondents over five rounds and are read with
the district around or beside them (`MERGE`). Orapa town takes Central Boteti's mix and Sowa
Town Central Tutume's, which is how round 7 labels them; the Okavango Delta's seven villages
take the two Ngamiland districts together; the CKGR takes Ghanzi. Each locality's Christians
are split at its unit's shares: Catholics and Adventists as a share of ALL the unit's Christians,
named or not (the floor reading), weighted, rounds 4-8.

## THE WITNESSES

Neither is a self-identification count, so neither sets anything; both are printed.
  * The Diocese of Gaborone (the south) counts 85,700 Catholics in a population of 1,231,000 in
    2013, 7.0% (catholic-hierarchy.org, from the Annuario Pontificio): baptisms of every age.
  * The Botswana Union Conference of Seventh-day Adventists: 47,590 members, one in 49 people,
    first quarter of 2021 (Southern Africa-Indian Ocean Division statistics page): baptised
    members, so mostly adults, ten years after the census.
  * The Pew Forum's 2008-09 survey (*Tolerance and Tension*, 2010, p.23) puts Catholics at 22% of
    Botswana's Christians and Adventists at 6%; its Catholic figure is four times the
    Afrobarometer's and three times the diocese's share, so it is not used.

Usage:
    python sources/bw_churches.py     rebuild data/normalized/bw_split.csv
"""

import os
import re
import sys

os.environ.setdefault("OMP_NUM_THREADS", "6")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import afrobarometer as ab
import cab
from tz import key, round_within_rows

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CENSUS = os.path.join(ROOT, "data", "normalized", "bw.csv")
OUT = os.path.join(ROOT, "data", "normalized", "bw_split.csv")

COUNTRY = "Botswana"
ROUNDS_READ = [4, 5, 6, 7, 8, 9]
CHURCH_ROUNDS = [4, 5, 6, 7, 8]
SOURCE_ID = "bw_afrobarometer_2008_2019_churches"
YEARS = "2008-2019"

CHURCHES = {"roman catholic": "Roman Catholic", "seventh day adventist": "Seventh Day Adventist"}
DRAWN = ["Roman Catholic", "Seventh Day Adventist"]
UNNAMED = "christian only"
# Every answer Botswanans give that is not Christianity. Anything else is a Christian answer;
# the list is named so an answer new to the file stops the build rather than turning Christian.
NOT_CHRISTIAN = {"none", "atheist", "agnostic", "traditional/ethnic religion", "other",
                 "muslim only", "shia only", "ismaeli", "bahai"}
CHRISTIAN = {"christian only", "independent", "pentecostal", "zionist christian church", "zcc",
             "roman catholic", "seventh day adventist", "lutheran", "anglican", "uccsa",
             "methodist", "evangelical", "baptist", "dutch reformed", "dutch reform",
             "church of christ", "jehovah's witness", "ipcc", "orthodox", "old apostolic",
             "presbyterian", "calvinist", "coptic", "st john apostolic", "mennonite",
             "quaker/friends", "new apostolic church"}
# The level test (share of ALL respondents, rounds 4-8). Drawn churches must stay within
# LEVEL_RANGE_MAX while `Christian only` rises by at least UNNAMED_RISE_MIN; the churches left
# out must still move more than that, so a change in the data re-opens the call.
# A church "holds" when its share of all respondents in rounds 7-8 is at least HOLD_MIN of its
# share in rounds 4-5 (Catholic 0.78, Adventist 0.82, 2026-10-03), while named Christians keep
# about 0.4. The churches left out must keep under HOLD_MIN - 0.05 (Lutheran 0.63, Pentecostal
# 0.58, Independent 0.19).
EARLY, LATE = [4, 5], [7, 8]
HOLD_MIN = 0.75
UNNAMED_RISE_MIN = 0.40
NOT_DRAWN_MOVE = {"lutheran": "falls to 0.7% in round 8", "pentecostal": "13.5% to 4.6%",
                  "independent": "26% to 3%"}

# REGION labels (whitespace collapsed) -> survey unit.
UNIT = {
    "Barolong": "Barolong", "Central Bobonong": "Central Bobonong",
    "Central Boteti": "Central Boteti", "Central Boteti/Orapa": "Central Boteti",
    "Central Mahalapye": "Central Mahalapye", "Central Serowe": "Central Serowe/Palapye",
    "Central Serowe/Palapye": "Central Serowe/Palapye", "Central Tutume": "Central Tutume",
    "Central Tutume/Sowa Town": "Central Tutume", "Sowa": "Central Tutume", "Chobe": "Chobe",
    "Francistown": "Francistown", "Gaborone": "Gaborone", "Ghanzi": "Ghanzi",
    "Jwaneng": "Jwaneng", "Kgalagadi North": "Kgalagadi North",
    "Kgalagadi South": "Kgalagadi South", "Kgatleng": "Kgatleng", "Kweneng East": "Kweneng East",
    "Kweneng West": "Kweneng West", "Lobatse": "Lobatse", "Ngamiland East": "Ngamiland East",
    "Ngamiland West": "Ngamiland West", "Ngwaketse": "Ngwaketse",
    "Ngwaketse West": "Ngwaketse West", "North East": "North East",
    "Seleibe Phikwe": "Selibe Phikwe", "Selibe Phikwe": "Selibe Phikwe",
    "Selibe Pikwe": "Selibe Phikwe", "South East": "South East",
}
MERGE = {"Jwaneng": "Ngwaketse", "Ngwaketse West": "Ngwaketse",
         "Kgalagadi North": "Kgalagadi South"}
# Census district (the `adm2` in bw.csv's note) -> survey unit, after MERGE. BW1403 is the
# Okavango Delta, read as the two Ngamiland units together.
ADM2 = {
    "BW0101": "Gaborone", "BW0201": "Francistown", "BW0301": "Lobatse",
    "BW0401": "Selibe Phikwe", "BW0501": "Central Boteti", "BW0601": "Ngwaketse",
    "BW0701": "Central Tutume", "BW0801": "Ngwaketse", "BW0802": "Barolong",
    "BW0803": "Ngwaketse", "BW0901": "South East", "BW1001": "Kweneng East",
    "BW1002": "Kweneng West", "BW1101": "Kgatleng", "BW1201": "Central Serowe/Palapye",
    "BW1202": "Central Mahalapye", "BW1205": "Central Tutume", "BW1301": "North East",
    "BW1401": "Ngamiland East", "BW1402": "Ngamiland West", "BW1403": "Ngamiland",
    "BW1501": "Chobe", "BW1601": "Ghanzi", "BW1602": "Ghanzi",
    "BW1701": "Kgalagadi South", "BW1702": "Kgalagadi South",
}
NGAMILAND = ("Ngamiland East", "Ngamiland West")

# Witnesses, printed only (docstring).
GABORONE_DIOCESE_2013 = (85_700, 1_231_000)
SDA_2021 = (47_590, 49)


def main():
    raw = ab.load(COUNTRY, expect_rounds=ROUNDS_READ, regroup=True)
    raw["k"] = raw["category"].map(key)
    unknown = sorted(set(raw["k"]) - NOT_CHRISTIAN - CHRISTIAN)
    if unknown:
        raise SystemExit(f"answers in neither list: {unknown}; add them deliberately")
    raw["unit"] = raw["geo_raw"].map(lambda s: UNIT.get(" ".join(str(s).split())))
    if raw["unit"].isna().any():
        raise SystemExit(f"REGION labels with no unit: "
                         f"{sorted(raw.loc[raw['unit'].isna(), 'geo_raw'].astype(str).unique())}")
    raw["unit"] = raw["unit"].replace(MERGE)

    # ---- the level test ----
    tot = raw.groupby("round")["w"].sum()
    lv = raw.groupby(["k", "round"])["w"].sum().unstack(fill_value=0.0).div(tot, axis=1)
    show = [UNNAMED, "roman catholic", "seventh day adventist", "zionist christian church", "zcc",
            "pentecostal", "lutheran", "independent", "uccsa", "anglican", "methodist"]
    print("\n  share of ALL respondents by round (%):")
    print(f"    {'':<26}" + "".join(f"{'R' + str(r):>7}" for r in ROUNDS_READ))
    for k in show:
        row = lv.loc[k] if k in lv.index else pd.Series(0.0, index=tot.index)
        print(f"    {k:<26}" + "".join(f"{100 * row.get(r, 0):7.1f}" for r in ROUNDS_READ))
    cr = lv[CHURCH_ROUNDS]
    rise = float(cr.loc[UNNAMED].max() - cr.loc[UNNAMED].min())
    if rise < UNNAMED_RISE_MIN:
        raise SystemExit(f"`Christian only` now moves only {100 * rise:.1f} points over rounds "
                         f"{CHURCH_ROUNDS}; the level test means less, decide again")
    def kept(row):
        return float(row[LATE].mean() / row[EARLY].mean())

    named = lv.loc[sorted(CHRISTIAN - {UNNAMED} & set(lv.index))].sum()
    print(f"    named Christians keep {kept(named):.2f} of their rounds 4-5 share in rounds 7-8")
    for k in CHURCHES:
        print(f"    {CHURCHES[k]}: keeps {kept(lv.loc[k]):.2f}")
        if kept(lv.loc[k]) < HOLD_MIN:
            raise SystemExit(f"{CHURCHES[k]} now keeps only {kept(lv.loc[k]):.2f} of its level; "
                             "it no longer holds while the unnamed rise")
    for k, why in NOT_DRAWN_MOVE.items():
        print(f"    {k}: keeps {kept(lv.loc[k]):.2f} (not drawn: {why})")
        if kept(lv.loc[k]) >= HOLD_MIN - 0.05:
            raise SystemExit(f"{k} now keeps {kept(lv.loc[k]):.2f} of its level; it was left out "
                             f"because it moved ({why}); decide again")

    # ---- the split-half, on the drawn quantity: share of all the unit's Christians ----
    chr_ = raw[raw["k"].isin(CHRISTIAN) & raw["round"].isin(CHURCH_ROUNDS)].copy()
    chr_["code"] = chr_["k"].map(lambda k: CHURCHES.get(k, {"zionist christian church": "Zion",
                                                               "zcc": "Zion",
                                                               "lutheran": "Lutheran",
                                                               "pentecostal": "Pentecostal"}
                                                        .get(k, "rest")))
    units = sorted(chr_["unit"].unique())
    cats = DRAWN + ["Zion", "Lutheran", "Pentecostal", "rest"]
    dfw = chr_.rename(columns={"round": "wave", "unit": "geo_id"})[["wave", "geo_id", "code", "w"]]
    passed, _t = cab.stability(dfw, cats, units, f"{len(units)} units, Botswanan Christians, "
                               f"rounds {CHURCH_ROUNDS}")
    missing = [c for c in DRAWN if c not in passed]
    if missing:
        raise SystemExit(f"{missing} no longer pass the split-half; decide again")
    if "Zion" in passed:
        raise SystemExit("the Zion Christian Church now passes the split-half; the docstring "
                         "left it out for failing it, decide again")

    by = chr_.groupby(["unit", "code"])["w"].sum().unstack(fill_value=0.0)
    n_u = chr_.groupby("unit").size()
    share = by.div(by.sum(axis=1), axis=0)[DRAWN]
    ng = by.loc[list(NGAMILAND)].sum()
    share.loc["Ngamiland"] = (ng / ng.sum())[DRAWN]
    nat = by.sum() / by.sum().sum()
    print("\n  Catholic and Adventist as a share of each unit's Christians, rounds 4-8 (%):")
    for u in units:
        print(f"    {u:<24}n={int(n_u[u]):4d}  Catholic {100 * share.loc[u, DRAWN[0]]:5.1f}  "
              f"Adventist {100 * share.loc[u, DRAWN[1]]:5.1f}")
    print(f"    {'all Botswana':<30}Catholic {100 * nat[DRAWN[0]]:5.1f}  Adventist "
          f"{100 * nat[DRAWN[1]]:5.1f}")

    # ---- the census's Christians, split ----
    cen = pd.read_csv(CENSUS, dtype={"geo_id": str}, keep_default_na=False, na_values=[""])
    cen = cen[(cen["geo_level"] == "locality") & (cen["source_category"] == "Christian")].copy()
    cen["adm2"] = cen["note"].map(lambda n: re.search(r"adm2=(\w+)", n).group(1))
    cen["unit"] = cen["adm2"].map(ADM2)
    if cen["unit"].isna().any():
        raise SystemExit(f"census districts with no survey unit: "
                         f"{sorted(cen.loc[cen['unit'].isna(), 'adm2'].unique())}")
    cen["count"] = cen["count"].astype(int)
    sh = share.reindex(cen["unit"]).to_numpy()
    want = pd.DataFrame(sh * cen["count"].to_numpy()[:, None], columns=DRAWN, index=cen.index)
    want["rest"] = cen["count"].to_numpy() - want.sum(axis=1)
    got = round_within_rows(want)
    if not (got.sum(axis=1).to_numpy() == cen["count"].to_numpy()).all():
        raise SystemExit("a locality's split does not sum to its Christians")
    def n_lab(u):
        return int(n_u[list(NGAMILAND)].sum()) if u == "Ngamiland" else int(n_u[u])

    rows = []
    for i, r in cen.iterrows():
        for c in DRAWN:
            rows.append(dict(geo_id=r["geo_id"], geo_level="locality", geo_name=r["geo_name"],
                             source_category=c, count=int(got.loc[i, c]), basis="self_id",
                             year=YEARS, source_id=SOURCE_ID,
                             note=(f"census Christians {int(r['count'])} split at the "
                                   f"Afrobarometer's share of {r['unit']}'s Christians (rounds "
                                   f"4-8, {n_lab(r['unit'])} Christian respondents); "
                                   f"adm2={r['adm2']}; parent_column=Christian")))
    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_csv(OUT, index=False, encoding="utf-8")
    christians = int(cen["count"].sum())
    print(f"\nwrote {OUT} ({len(out)} rows); of {christians:,} census Christians aged 12+:")
    for c in DRAWN:
        n = int(out.loc[out["source_category"] == c, "count"].sum())
        print(f"    {c:<24}{n:>9,}  {100 * n / christians:5.2f}% of Christians")

    # ---- witnesses, printed ----
    south = cen["adm2"].isin(["BW0101", "BW0301", "BW0601", "BW0801", "BW0802", "BW0803",
                              "BW0901", "BW1001", "BW1002", "BW1101", "BW1701", "BW1702"])
    cat_south = int(out[(out["source_category"] == DRAWN[0])
                        & out["geo_id"].isin(cen.loc[south, "geo_id"])]["count"].sum())
    print(f"\n  witnesses (not used to draw): Catholics drawn in the southern districts "
          f"{cat_south:,} of {int(cen.loc[south, 'count'].sum()):,} Christians aged 12+; the "
          f"Diocese of Gaborone counts {GABORONE_DIOCESE_2013[0]:,} baptised Catholics of every "
          f"age in {GABORONE_DIOCESE_2013[1]:,} people (2013, "
          f"{100 * GABORONE_DIOCESE_2013[0] / GABORONE_DIOCESE_2013[1]:.1f}%). Adventists: "
          f"{SDA_2021[0]:,} baptised members, 1 in {SDA_2021[1]} people (2021).")


if __name__ == "__main__":
    main()
