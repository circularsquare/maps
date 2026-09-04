"""Would a self-identification total survive subtracting ASARB's rolls? - the §3.2 residual test.

The US is the only country on the map whose composition is a `roll` (spec §3.1). Everywhere else
a census asks the person. The consequence is in the arithmetic: ASARB's 161.2M adherents are
48.6% of the country, so **more than half of the United States is currently drawn as nothing at
all** - not as unaffiliated, not as unknown, absent.

Anita's proposal, 2026-09-03: re-base the US on self-identification and demote the rolls to
structure. That is §3.4's "structure from the detailed source, totals from the recent one" with
`basis` in place of `year`, it is the split §3.1 explicitly permits, and the leftover is §3.2's
`...other/unspecified` residual - the row §6.6 already knows how to draw. §3.5 has been asking
for it all along: it lists the United States among the countries that do not ask, and says their
composition should be a survey-derived estimate drawn desaturated.

This script is the feasibility check that has to pass first. `residual = self_id - roll` is only
a residual if it is positive; where a roll exceeds the survey it is evidence that the two are
counting different people, and no clamp makes that go away.

RUN IT with no arguments. The survey figures are hand-entered below from Pew's published pages -
this deliberately does not scrape, so every number has a URL next to it and can be re-checked.

WHAT IT FOUND, 2026-09-03 -- see the bottom of the output for the numbers.
"""
import pandas as pd

from pathlib import Path

HERE = Path(__file__).resolve().parent.parent

# Pew Research Center, 2023-24 Religious Landscape Study, published 2025-02-26. Percentages are
# of ADULTS, which is the first real problem: ASARB counts everyone including children, so these
# have to be applied to the whole population to be comparable at all, and that assumes children
# hold their household's religion.
#   national  https://www.pewresearch.org/religion/2025/02/26/religious-landscape-study-religious-identity/
#   states    https://www.pewresearch.org/religious-landscape-study/state/<state>/
PEW_US = {
    "christianity": 62.0,       # Protestant 40, Catholic 19, LDS 2, Orthodox 1, other Christian 1
    "judaism": 1.7,
    "islam": 1.2,
    "buddhism": 1.1,
    "hinduism": 0.9,
    "unaffiliated": 29.0,
}
PEW_US_SUB = {"christianity.catholic": 19.0, "christianity.latterday": 2.0}
# "Other world religions" - Sikh, Daoist, Baha'i, Zoroastrian TOGETHER - is published as <0.3%,
# and "other religious identifications" (Unitarian, pantheist, Wiccan) as 1.9%. Neither is broken
# out, which is the finding that matters for section 4.4: this survey cannot give Sikhs a number.

# state -> (Christian %, Catholic %, Latter-day Saint % or None), adults, same study
PEW_STATES = {
    "Utah": (62, 4, 50),
    "Rhode Island": (63, 39, None),
    "Massachusetts": (52, 29, None),
    "Louisiana": (74, 23, 1),
    "New Jersey": (57, 33, None),
    "New York": (56, 29, None),
    "New Mexico": (59, 27, 1),
    "Connecticut": (57, 35, None),
}


def asarb():
    """ASARB 2020 by state and by root, plus the two bodies that keep baptismal registers."""
    paths = pd.read_csv(HERE / "taxonomy" / "usrc_groups.csv", dtype={"Group Code": str})
    paths["code"] = paths["Group Code"].str.strip().str.zfill(3)
    grp = pd.read_excel(HERE / "data" / "raw" / "2020_USRC_Group_Detail.xlsx",
                        sheet_name="2020 Group by State", dtype={"Group Code": str})
    grp["code"] = grp["Group Code"].str.strip().str.zfill(3)
    grp = grp.merge(paths[["code", "path"]], on="code", how="left")
    grp = grp[grp.path.notna() & (grp.path != "UNMAPPED")].copy()
    grp["root"] = grp.path.str.split(".").str[0]

    # the sheet ends with a blank row and a "Totals" row, so the states cannot just be summed
    st = pd.read_excel(HERE / "data" / "raw" / "2020_USRC_Summaries.xlsx",
                       sheet_name="2020 State Summary")
    st = st[st["State Name"].notna()]
    us = pd.read_excel(HERE / "data" / "raw" / "2020_USRC_Summaries.xlsx",
                       sheet_name="2020 US Summary")
    return grp, dict(zip(st["State Name"], st["2020 Population"])), \
        float(us["2020 Population"].iloc[0])


def main():
    grp, pop, us_pop = asarb()

    print("=" * 78)
    print("NATIONAL -- self-identification against the roll, applied to the whole population")
    print("=" * 78)
    roots = grp.groupby("root")["Adherents"].sum()
    print(f"{'':<16}{'self-ID':>14}{'ASARB roll':>14}{'residual':>14}   what the residual is")
    for root, pct in PEW_US.items():
        sid = pct / 100 * us_pop
        roll = float(roots.get(root, 0.0))
        note = ("nobody on a roll: the whole thing is new" if root == "unaffiliated" else
                "" if sid > roll else "NEGATIVE -- roll exceeds self-ID")
        print(f"  {root:<14}{sid:>14,.0f}{roll:>14,.0f}{sid - roll:>14,.0f}   {note}")
    for path, pct in PEW_US_SUB.items():
        sid = pct / 100 * us_pop
        roll = float(grp.loc[grp.path.str.startswith(path), "Adherents"].sum())
        flag = "" if sid > roll else "NEGATIVE"
        print(f"  {path:<14}{sid:>14,.0f}{roll:>14,.0f}{sid - roll:>14,.0f}   {flag}")

    drawn = float(roots.sum())
    print(f"\n  drawn today {drawn:,.0f} = {drawn / us_pop:.1%} of {us_pop:,.0f}. "
          f"The rest, {us_pop - drawn:,.0f}, is on the map as nothing.")

    print("\n" + "=" * 78)
    print("BY STATE -- the eight where a roll is most likely to exceed the survey")
    print("=" * 78)
    print(f"{'state':<16}{'level':<22}{'self-ID':>12}{'roll':>12}{'residual':>13}")
    bad = []
    for state, (chr_pct, cath_pct, lds_pct) in PEW_STATES.items():
        p = pop[state]
        rows = grp[grp["State Name"] == state]
        cases = [("christianity", chr_pct, float(rows.loc[rows.root == "christianity",
                                                          "Adherents"].sum())),
                 ("christianity.catholic", cath_pct,
                  float(rows.loc[rows.path.str.startswith("christianity.catholic"),
                                 "Adherents"].sum()))]
        if lds_pct is not None:
            cases.append(("christianity.latterday", lds_pct,
                          float(rows.loc[rows.path.str.startswith("christianity.latterday"),
                                         "Adherents"].sum())))
        for i, (level, pct, roll) in enumerate(cases):
            sid = pct / 100 * p
            r = sid - roll
            if r < 0:
                bad.append((state, level, r))
            print(f"{state if i == 0 else '':<16}{level:<22}{sid:>12,.0f}{roll:>12,.0f}"
                  f"{r:>13,.0f}{'  NEGATIVE' if r < 0 else ''}")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"  {len(bad)} of the {sum(2 + (v[2] is not None) for v in PEW_STATES.values())} "
          f"state-level subtractions tested come out negative, and they are not scattered:")
    for state, level, r in sorted(bad, key=lambda t: t[2]):
        print(f"    {state:<16}{level:<24}{r:>13,.0f}")
    print("""
  Every one of them is a body that keeps a baptismal register rather than a membership
  list. A Catholic diocese reports the baptised living in a parish's territory and an LDS
  ward reports everyone baptised who has not formally resigned; both include people who
  would tell a surveyor they are something else now. That is not noise to be clamped away,
  it is the two instruments disagreeing about the same people, and the disagreement is
  concentrated in exactly two branches out of the whole tree.

  So the proposal survives at the top and fails in the middle. Christianity as a whole
  clears its roll in every state tested but Utah; Catholic does not clear it in Rhode
  Island, Massachusetts, Louisiana, New Mexico or Connecticut, and LDS does not in Utah.

  THREE MORE THINGS THIS TURNED UP.

  The national Christian residual is 53.5M, which is the shape the proposal predicted, and
  the unaffiliated residual is 96.1M of people the map cannot currently draw at all.
  Together they are the 171M now shown as nothing.

  The Catholic reconciliation is carried entirely by the child assumption. Adults are 78%
  of the population, so applying adult shares to everyone scales the survey by 1.28x.
  Catholic then clears its roll by 1.1M nationally - under 2%. Apply the shares to adults
  only and it is 49.1M against a 61.9M roll, negative by 12.8M. Whatever this map ends up
  saying about American Catholics rests on an assumption about children, and it should say
  so out loud.

  Islam is negative nationally, and it is the interesting one: ASARB's figure for it is a
  body literally named "Muslim Estimate", so this is not a roll against a survey, it is two
  estimates of the same population disagreeing by 12%. It is the closest thing available to
  a calibration between the two instruments.

  AND THE ONE IT DOES NOT FIX. Pew publishes Sikh, Daoist, Baha'i and Zoroastrian as a
  single "other world religions" line at <0.3%, and Unitarian, pantheist and Wiccan as
  "other religious identifications" at 1.9%. At n=36,000 nothing smaller can be broken out.
  So re-basing on self-identification does not give the eleven congregations-only religions
  of section 4.4 a number. Those still need their own sources; the two pieces of work are
  complementary, not alternatives.""")


if __name__ == "__main__":
    main()
