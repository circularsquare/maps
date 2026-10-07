"""Give Wales's census Christians a church, where a survey names one — uk_split.py's sibling.

WHAT IT DOES. The census says how many Christians live in each Welsh Output Area (TS030,
1,354,773 in all). The British Election Study says, for each of Wales's twelve ITL3 areas, which
church its Christian respondents name (sources/uk_bes_wales.py). This file divides each Output
Area's census Christians by its area's survey mix and writes the pieces as `derived` rows that
roll back to the census's own `Christian` when a reader turns inferred dots off.

WHY NOT uk_split.py'S METHOD. England's placement comes from a register of churches sized by the
English Church Census 2005. Neither exists for Wales: the church census stopped at the border and
Wales has had none since 1995. So here the survey places as well as sizes, and only where its own
geography survives the split-half test (sources/uk_bes_wales.py has the test and the numbers).

THE ARITHMETIC, per Output Area:

    area        the OA's ITL3 group of authorities (12 in Wales)
    share       PLACE legs: the area's survey share; FLAT legs and the not-drawn legs: Wales's
    rake        a per-leg factor so the Wales-wide result over census Christians reproduces the
                survey's Wales mix (the census puts Christians in different places from where
                the survey's respondents are, so area shares alone do not add up to it)
    classified  the share of census Christians the survey names a church for (BES Christians
                over census Christians, adults); since 2026-10-04 set to 1 (SPREAD_UNNAMED), so
                the unnamed share takes the survey's mix, as England's does
    dots        OA Christians x classified x raked share, per drawn leg
    remainder   OA Christians minus all of those, on the census's own `Christian`

**The remainder is now only the churches too small in the Welsh sample to draw** (Orthodox,
Pentecostal, independent evangelical, Brethren; fewer than 100 respondents each), as England keeps
Pentecostal and New church unplaced. Until 2026-10-04 it also held the census Christians the survey
names no church for (about 30%); Anita then chose to spread them like England's.

TIER. Every drawn row is `derived`: the Christians were counted per Output Area, and what is
inferred is only which church each belongs to, from people who said so themselves.

Run: python uk_split_wales.py --report   print the result, write nothing
     python uk_split_wales.py            -> data/normalized/uk_split_wales.csv
"""
import argparse
import csv
import os
import sys

import pandas as pd

from uk_split import rake

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "raw", "uk")
NORM = os.path.join(HERE, "data", "normalized")
OUT = os.path.join(NORM, "uk_split_wales.csv")
BES = os.path.join(NORM, "uk_bes_wales.csv")
OA_LAD = os.path.join(RAW, "oa_lad.csv")

sys.path.insert(0, os.path.join(HERE, "sources"))
from uk_bes_wales import LA, LEGS  # noqa: E402

# Leg -> category string; taxonomy/uk2021.py resolves these, and England's uk_split.py writes the
# same strings, so the two countries share nodes and legend rows.
CATEGORY = {
    "anglican": "Christian: Anglican",
    "catholic": "Christian: Roman Catholic",
    "methodist": "Christian: Methodist",
    "baptist": "Christian: Baptist",
    "reformed": "Christian: Reformed",
}
REMAINDER = "Christian"
# Anita, 2026-10-04: "ok lets spread wales like england". The census Christians the survey names no
# church for are given the survey's mix, as England's are (uk_split.py, her 2026-09-07 call). BES has
# no "Christian, no denomination" box, so that share is the census/survey gap, not an answer.
SPREAD_UNNAMED = True
SOURCE_ID = "uk_ew_census_2021"
BASIS = "self_id"
OUT_COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
               "basis", "year", "source_id", "tier", "note"]


def inputs():
    bes = pd.read_csv(BES)
    nat = bes[bes["geo_level"] == "country"].set_index("source_category")
    classified = float(nat.loc["classified", "count"])
    decision = {leg: nat.loc[leg, "note"].split("decision=")[1] for leg in LEGS}
    national = {leg: float(nat.loc[leg, "count"]) for leg in LEGS}
    area = (bes[bes["geo_level"] == "itl3"]
            .pivot(index="geo_id", columns="source_category", values="count"))
    return classified, decision, national, area


def christians_by_oa():
    lad = pd.read_csv(OA_LAD)
    wal = lad[lad["LAD23CD"].str.startswith("W")].copy()
    unknown = sorted(set(wal["LAD23CD"]) - set(LA))
    if unknown:
        sys.exit(f"Welsh authorities in oa_lad.csv that uk_bes_wales.LA lacks: {unknown}")
    wal["itl3"] = wal["LAD23CD"].map(lambda c: LA[c][1])
    uk = pd.read_csv(os.path.join(NORM, "uk.csv"),
                     usecols=["geo_id", "geo_level", "source_category", "count", "source_id"],
                     dtype={"geo_id": str}, low_memory=False)
    chr_ = uk[(uk["source_id"] == "uk_ew_census_2021") & (uk["geo_level"] == "output_area")
              & (uk["source_category"] == "Christian") & uk["geo_id"].str.startswith("W")]
    oa = wal[["OA21CD", "itl3"]].merge(chr_[["geo_id", "count"]], left_on="OA21CD",
                                       right_on="geo_id", how="outer", indicator=True)
    lost = oa[oa["_merge"] == "right_only"]
    if len(lost):
        sys.exit(f"{len(lost)} Welsh census Output Areas have no authority in oa_lad.csv")
    oa["christians"] = oa["count"].fillna(0.0)
    return oa[["OA21CD", "itl3", "christians"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    classified, decision, national, area = inputs()
    oa = christians_by_oa()
    total = oa["christians"].sum()
    print(f"Wales's census Christians: {total:,.0f} over {len(oa):,} Output Areas")
    print(f"named a church by the survey: {100 * classified:.1f}% (spread to 100%, as England)")
    if SPREAD_UNNAMED:
        classified = 1.0
    print("decisions:", ", ".join(f"{k} {v}" for k, v in decision.items()))

    drawn = [l for l in LEGS if decision[l] in ("PLACE", "FLAT")]
    legs = drawn + ["unsplit"]
    anchor = {l: national[l] for l in drawn}
    anchor["unsplit"] = sum(national[l] for l in LEGS if l not in drawn)
    for leg in drawn:
        if decision[leg] == "PLACE":
            oa[leg] = oa["itl3"].map(area[leg])
        else:
            oa[leg] = national[leg]
    oa["unsplit"] = anchor["unsplit"]
    if oa[legs].isna().any().any():
        sys.exit("an Output Area's area has no survey share")
    w, k, err = rake(oa, legs, anchor)
    print(f"rake converged to {err:.1e}")

    people = w.mul(oa["christians"], axis=0) * classified
    people["remainder"] = oa["christians"] - people[drawn].sum(axis=1)
    people = people.drop(columns="unsplit")
    people["itl3"] = oa["itl3"].to_numpy()
    by = people.groupby("itl3").sum(numeric_only=True)
    print("\n  share of each area's census Christians, as drawn (%)")
    print((100 * by.div(by.sum(axis=1), axis=0)).round(1).to_string())
    tot = people[drawn + ["remainder"]].sum()
    print("\n  Wales as drawn")
    for c, v in tot.sort_values(ascending=False).items():
        print(f"    {c:12s} {v:12,.0f}  {100 * v / total:5.1f}%")
    if args.report:
        return 0

    note_place = ("level=leaf; derivation=split; structure_geo=itl3; "
                  "structure=uk_wa_bes_w1_31; anchor=uk_wa_bes_w1_31; parent_column=Christian")
    note_flat = ("level=leaf; derivation=split; structure_geo=country; "
                 "structure=uk_wa_bes_w1_31; anchor=uk_wa_bes_w1_31; parent_column=Christian")
    n = 0
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        out = csv.DictWriter(fh, fieldnames=OUT_COLUMNS)
        out.writeheader()
        for leg in drawn + ["remainder"]:
            vals = people[leg].to_numpy().round(4)
            for gid, val in zip(oa["OA21CD"], vals):
                if val <= 0:
                    continue
                if leg == "remainder":
                    row = {"source_category": REMAINDER, "tier": "measured",
                           "note": "level=leaf; cat=Christian; derivation=exact_single_child;"
                                   " parent_column=Christian; denomination not named"}
                else:
                    row = {"source_category": CATEGORY[leg], "tier": "derived",
                           "note": note_place if decision[leg] == "PLACE" else note_flat}
                row.update({"geo_id": gid, "geo_level": "output_area", "geo_name": gid,
                            "count": val, "basis": BASIS, "year": 2021,
                            "source_id": SOURCE_ID})
                out.writerow(row)
                n += 1
    written = pd.read_csv(OUT, usecols=["count"])["count"].sum()
    print(f"\nwrote {n:,} rows -> {OUT}")
    print(f"  people written {written:,.0f} against {total:,.0f} census Christians "
          f"({written - total:+,.0f})")
    if abs(written - total) > 1.0:
        print("  ! the split does not conserve Wales's Christians")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
