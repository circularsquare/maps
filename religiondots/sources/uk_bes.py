"""The British Election Study internet panel -> England's Christian denominations, self-ID.

`sources/uk_bes.py` -> `data/normalized/uk_bes.csv`

WHY THIS EXISTS. This is the survey half of spec.md §3.5a's mechanism applied inside
Christianity: `sources/uk_ecc.py` knows where England's denominations ARE, from a census of
churches, and knows nothing about how many people belong to them, because attendance is not
identity. This file supplies the how-many. `uk_split.py` does the arithmetic.

WHY THE BRITISH ELECTION STUDY AND NOT A BETTER SURVEY. Understanding Society and British
Social Attitudes both carry a deeper denominational list, and both sit behind a UK Data
Service registration. The BES panel needs only a free account, and at n=126,840 over 31
waves it is far larger than either: **25,630 English respondents in wave 21 alone.** Its
religion question is a profile variable asked every wave, which is why the sample is that
big.

    p_religion: "Do you regard yourself as belonging to any particular religion,
                 and if so, to which of these do you belong?"

WHICH WAVE, AND WHY ONE RATHER THAN SEVERAL. **Wave 21, fielded May 2021** -- the wave
closest to census day, 21 March 2021, so the anchor and the totals it is applied to describe
the same England. The waves are not pooled: the same people answer every wave, so pooling
would count individuals repeatedly and quietly narrow the confidence interval on a sample
that had not grown. Waves 22, 23 and 31 are read anyway, by `check()`, purely to show that
the shares are stable.

**THE ELEVEN CHRISTIAN OPTIONS ARE THE REASON THIS WORKS.** The list is deeper than the
census's for England, which has none at all, and deeper than Scotland's, which names two:

    Church of England/Anglican/Episcopal · Roman Catholic · Presbyterian/Church of Scotland
    Methodist · Baptist · United Reformed Church · Free Presbyterian · Brethren
    Orthodox Christian · Pentecostal (e.g. Assemblies of God, Elim, New Testament Church of
    God, Redeemed Christian Church of God) · Evangelical - independent/non-denominational
    (e.g. FIEC, Pioneer, Vineyard, Newfrontiers)

All nineteen options are present in every wave from 17 on, checked, so nothing here depends
on an option appearing mid-panel.

**IT DISAGREES WITH THE CENSUS ABOUT HOW MANY CHRISTIANS THERE ARE, BY THIRTEEN POINTS.**
BES puts Christians at 36.1% of English adults; the census puts them at 49.2% of the same
population (21,994,389 of 44,715,447). Two reasons, both structural: an online panel that
people opt into skews younger and more secular than England, and **there is no "Christian,
no denomination" option on this list**, so the large number of people who would answer a
census with a bare "Christian" have to pick a denomination, pick "Other", or say none.
`uk_split.py` reads only the RATIO BETWEEN the Christian legs and takes every magnitude
from the census. The absolute Christian share emitted here is a survey artefact and must
never be drawn.

**THE LARGEST ASSUMPTION IN THIS CONSTRUCTION LIVES IN THAT GAP, AND IT IS BIGGER THAN THE
CHILD ONE.** BES classifies 36.1/49.2 = **73.4% of the census's adult Christians**, so
applying its ratios to the whole Christian total assumes the other **26.6% -- roughly 5.8
million adults who told the census "Christian" and would not pick a denomination here --
divide the same way as the people who did pick one.** There is no way to test that from
this survey, and two plausible stories run in opposite directions: someone who says only
"Christian" may be a loosely-attached Anglican, which would push the true split further
towards `anglican` than the anchor, or may be deliberately non-denominational, which would
push it towards `newchurch`. The Scotland check below is the only external evidence, and it
says the largest leg comes out right while the small ones do not. Stated, not corrected.

WHAT THE RATIO IS WORTH, MEASURED AGAINST A CENSUS. Scotland is a free test, because there
the census publishes the answer this survey is being asked for. NRS 2022, share within
Christians, against BES Scotland:

                          national church   Catholic   other Christian
      NRS census 2022          52.5%          34.3%         13.2%
      BES W21 / W22 / W23      49.3 51.1 53.2  28.4 25.1 24.1   22.2 23.9 22.7

The national-church leg lands within sampling error in all three waves. **Catholicism comes
out 6 to 10 points short and "other Christian" 9 to 11 points long, consistently.** Most of
that is the child assumption below -- Catholics are much younger than the national church in
both countries, so an adults-only survey finds proportionally fewer of them than an all-ages
census -- and the rest is a long option list attracting people that a two-option census form
would have collected under its residual. `check()` runs this comparison on every build.

THE CHILD ASSUMPTION, AND WHY IT IS CHEAP HERE. BES surveys adults; the census counts
everyone; `uk_split.py` applies these adult shares to all 26,167,899 of England's Christians,
so **15.9% of the drawn people are children whose denomination nobody measured.** Anita's
call, 2026-09-07, over the alternative of leaving 4.17M children on a bare `christianity`
node beside their own denominations: "id rather not have a bare christianity group in
england, i feel like that'd be confusing."

Unlike `us_rebase.py`, where the same assumption scales the survey by 1.28x and is called
load-bearing, here it is small and its size is measurable. English Christians aged 30-49 --
the people whose children these are -- split 55.7 Anglican / 23.8 Catholic against the
all-adult 64.6 / 18.2. Giving every child their parents' generation's split instead would
move the England-wide anchor by:

      anglican -1.42   catholic +0.90   orthodox +0.37   pentecostal +0.20
      newchurch +0.19  methodist -0.10  reformed -0.10   baptist -0.05

At most 1.4 points on the largest leg. Not corrected, stated.

Run: python sources/uk_bes.py            # -> data/normalized/uk_bes.csv, with checks
     python sources/uk_bes.py --report   # print the anchor and the Scotland check
"""
import argparse
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "uk")
SAV = os.path.join(RAW, "BES2024_W31_Panel_v31.05.sav")
OUT = os.path.join(ROOT, "data", "normalized", "uk_bes.csv")

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

SOURCE_ID = "uk_en_bes_w21"
BASIS = "self_id"
YEAR = 2021
WAVE = 21                       # May 2021; census day was 21 March 2021
CHECK_WAVES = (21, 22, 23, 31)

ENGLAND, SCOTLAND = 1, 2

# England's usual residents aged 18 and over, Census 2021. Sum over religion_tb x
# resident_age_7a categories 4-7 at country level, from
#   api.beta.ons.gov.uk/v1/population-types/UR/census-observations
#     ?dimensions=religion_tb,resident_age_7a&area-type=ctry
# The weights below are rescaled to this so `count` is a population estimate rather than a
# respondent tally, matching sources/us_pew.py. Only ratios are consumed downstream.
ENGLAND_ADULTS = 44_715_447
ENGLAND_ADULT_CHRISTIANS = 21_994_389

# p_religion code -> the leg drawn on the map, matching sources/uk_ecc.py's LEG.
#
# `Presbyterian/Church of Scotland` (4) joins `reformed` IN ENGLAND ONLY. In Scotland it is
# the national church and the whole shape of the question is different, which is why
# check()'s Scottish comparison uses its own mapping and why nothing Scottish is emitted
# from this file -- Scotland has a census that answers this directly (NRS UV205, in uk.py).
LEG = {
    2: "anglican",
    3: "catholic",
    5: "methodist",
    6: "baptist",
    17: "orthodox",
    18: "pentecostal",
    19: "newchurch",
    7: "reformed",
    4: "reformed",
    8: "reformed",
    9: "other",          # Brethren -- the only body BES names that lands on the residual
}
CHRISTIAN = set(LEG)

# 15 "Other" and 16 "Prefer not to say" are NOT Christian legs and are never counted as
# such: "Other" pools every non-Christian minority religion with anyone who wanted to write
# something in, and cannot be separated. This is why `other` is the weakest leg in the join
# -- see uk_ecc.py, whose `other` holds the Salvation Army, Quakers, Adventists and
# Lutherans, none of which has a tick box here.
#
# **`other` IS KNOWN TO BE UNDERSTATED AND IS DRAWN ANYWAY, ON `christianity.other`.**
# Sixteen respondents, all of them Brethren, put the leg at 0.19% of England's Christians --
# about 42,000 people -- while uk_ecc.py's matching churches are 7.2% of English
# churchgoing, 2,075 congregations. The Salvation Army alone claims around 30,000 UK
# members and the Adventists about 35,000, so the true figure is some multiple of this one.
# Anita's call, 2026-09-07, over dropping the leg or inferring a size from the gap between
# BES's "Other" and the census's non-Christian "Other religion": use the residual category
# the map already has. `christianity.other` is what Scotland's `Other Christian`, Northern
# Ireland's `Other Christian denominations`, Canada's and Australia's all resolve to, so
# England gains a category its neighbours already display rather than a bespoke one, and
# the number stays traceable to a source instead of being estimated into existence. The
# undercount is real, it is one-directional, and it belongs in note_public.

_LABEL = {
    1: "no religion", 2: "Church of England/Anglican", 3: "Roman Catholic",
    4: "Presbyterian/Church of Scotland", 5: "Methodist", 6: "Baptist",
    7: "United Reformed Church", 8: "Free Presbyterian", 9: "Brethren", 10: "Judaism",
    11: "Hinduism", 12: "Islam", 13: "Sikhism", 14: "Buddhism", 15: "Other",
    16: "prefer not to say", 17: "Orthodox Christian", 18: "Pentecostal",
    19: "Evangelical independent/non-denominational",
}


def _wave(wave, country):
    """(frame of that wave's respondents in that country, with `leg` and `wt`)."""
    import pyreadstat
    cols = [f"p_religionW{wave}", f"countryW{wave}", f"wt_new_W{wave}"]
    if not os.path.exists(SAV):
        sys.exit(f"missing {SAV}\n  see sources/uk_bes.md for the download")
    df, _ = pyreadstat.read_sav(SAV, usecols=cols)
    df = df.dropna(subset=cols[:2])
    df = df[df[cols[1]] == country].copy()
    # An unweighted respondent is better than a dropped one; wave weights are missing for a
    # handful of cases and BES's own documentation treats those as weight 1.
    df["wt"] = df[cols[2]].fillna(1.0)
    df["code"] = df[cols[0]].astype(int)
    return df


def anchor(wave=WAVE):
    """(rows, diagnostics) -- England's adult religion, weighted, scaled to the census."""
    df = _wave(wave, ENGLAND)
    scale = ENGLAND_ADULTS / df["wt"].sum()
    df["people"] = df["wt"] * scale

    chr_ = df[df["code"].isin(CHRISTIAN)].copy()
    chr_["leg"] = chr_["code"].map(LEG)
    by_leg = chr_.groupby("leg").agg(people=("people", "sum"), n=("code", "size"))
    by_leg["share"] = 100.0 * by_leg["people"] / by_leg["people"].sum()

    diag = {
        "wave": wave,
        "respondents_england": len(df),
        "respondents_christian": len(chr_),
        "bes_christian_pct_of_adults": 100.0 * chr_["people"].sum() / df["people"].sum(),
        "census_christian_pct_of_adults":
            100.0 * ENGLAND_ADULT_CHRISTIANS / ENGLAND_ADULTS,
        "all_categories": df.groupby("code")["people"].sum(),
    }
    return by_leg, diag


def _rows(by_leg):
    for leg, r in by_leg.iterrows():
        yield {
            "geo_id": "E92000001",
            "geo_level": "country",
            "geo_name": "England",
            "source_category": leg,
            "count": round(r["people"]),
            "basis": BASIS,
            "year": YEAR,
            "source_id": SOURCE_ID,
            "note": (f"BES wave {WAVE}, {int(r['n'])} respondents; "
                     f"{r['share']:.2f}% of English adult Christians"),
        }


def check(by_leg, diag):
    """Print the anchor, the census disagreement, and the Scottish validation."""
    fails = 0
    print(f"British Election Study wave {diag['wave']} -> England adult religion")
    print(f"  English respondents            {diag['respondents_england']:,}")
    print(f"  of them Christian              {diag['respondents_christian']:,}")
    print(f"  BES Christian % of adults      {diag['bes_christian_pct_of_adults']:.1f}%")
    print(f"  census Christian % of adults   {diag['census_christian_pct_of_adults']:.1f}%"
          "   <- disagreement is expected, see the docstring")

    print("\n  leg              respondents      people    share")
    for leg, r in by_leg.sort_values("people", ascending=False).iterrows():
        thin = "  <- thin" if r["n"] < 100 else ""
        print(f"  {leg:14s} {int(r['n']):11,} {r['people']:11,.0f} "
              f"{r['share']:7.2f}%{thin}")
        if r["n"] < 30:
            print(f"  ! {leg} rests on {int(r['n'])} respondents, too few to anchor a leg")
            fails += 1

    # Scotland: the only place a census answers the question this survey is being asked.
    NRS = {"national church": 1_107_708, "catholic": 723_310, "other": 279_435}
    nrs_total = sum(NRS.values())
    print("\n  Scotland check -- share within Christians (NRS Census 2022 = truth)")
    print("                     national church   catholic      other")
    print("    NRS 2022         %13.1f%% %10.1f%% %9.1f%%"
          % tuple(100.0 * NRS[k] / nrs_total
                  for k in ("national church", "catholic", "other")))
    sc_leg = {4: "national church", 3: "catholic"}
    for w in CHECK_WAVES:
        try:
            df = _wave(w, SCOTLAND)
        except Exception as exc:                                  # noqa: BLE001
            print(f"    W{w}: unavailable ({exc})")
            continue
        c = df[df["code"].isin(CHRISTIAN)].copy()
        c["grp"] = c["code"].map(lambda k: sc_leg.get(k, "other"))
        g = c.groupby("grp")["wt"].sum()
        t = g.sum()
        print("    BES W%-2d n=%4d  %13.1f%% %10.1f%% %9.1f%%"
              % (w, len(c), 100 * g.get("national church", 0) / t,
                 100 * g.get("catholic", 0) / t, 100 * g.get("other", 0) / t))
    print("    Catholic short by 6-10 points and `other` long by 9-11 is the known and"
          "\n    expected shape of this bias; the docstring says why. A wave that matches"
          "\n    NRS exactly, or misses it the other way, is the thing worth investigating.")
    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true",
                    help="print the anchor and the checks, write nothing")
    args = ap.parse_args()

    by_leg, diag = anchor()
    fails = check(by_leg, diag)
    if args.report:
        return 1 if fails else 0

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rows = list(_rows(by_leg))
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows -> {OUT}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
