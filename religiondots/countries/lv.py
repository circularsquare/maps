# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _lv_counts():
    """Latvia at the 6 statistical regions (NUTS 3), from three parts of one census table.

    The Ministry of Justice publishes the members each religious organisation reports, nationally
    and with no geography. That is `roll` (spec §3.1), so sources/lv.py prints it beside the survey.

      * **Latvian citizens, 1.64M.** ESS `rlgdnlv`, `ctzcntr = Yes`, rounds 4, 9 and 11 (2008-2024)
        at the 6 regions, scaled to rounds 9-11's national level (§3.4). Round 10's region carries
        no geography and it is used nationally only.
      * **Recognised non-citizens, 191k.** Holders of an alien's passport in ESS rounds 4 and 9,
        their national composition inside each region's census count of `RNC` and stateless people.
      * **Foreign citizens, 62k.** `cens_21ctz_r3` at NUTS 3 crossed with Pew, with `RNC` taken
        out of `FOR`, which includes it.
    """
    from lv2024 import resolve

    cit = pd.read_csv(HERE / "data" / "normalized" / "lv.csv", dtype={"geo_id": str},
                      keep_default_na=False, na_values=[""])
    cit = cit[cit["geo_level"] == "nuts3"].copy()
    cit["node"] = cit["source_category"].map(resolve)
    unmapped = sorted(set(cit.loc[cit["node"].isna(), "source_category"]))
    if unmapped:
        raise SystemExit(f"lv.csv has unmapped source categories: {unmapped}")

    ext = pd.read_csv(HERE / "data" / "normalized" / "lv_foreign.csv", dtype={"geo_id": str})
    ext = ext[ext["geo_level"] == "nuts3"]

    df = pd.concat([cit[["geo_id", "node", "count"]], ext[["geo_id", "node", "count"]]],
                   ignore_index=True)
    df["congregations"] = 0
    # A survey and a nationality model: nothing here is `measured`.
    df["tier"] = "modelled"
    return df.rename(columns={"geo_id": "unit"})[
        ["unit", "node", "count", "congregations", "tier"]]


def _lv_place_weight(place):
    """Latvia's 119 LAUs (the municipalities before the July 2021 reform), weighted by GISCO's 2021
    population.

    The Greek weighter again. It places dots by where people live inside a region and not by
    religion, so Daugavpils's Orthodox and Old Believers spread over Latgale by population, and Riga
    is one LAU of 615,000 people (sources/lv.md §8).
    """
    if "pop" not in place.columns:
        print("  !! lv_lau.gpkg has no `pop` column, run sources/lv_geo.py")
        return None
    return _GrLauWeighter(place)


ENTRY = {
    "lv": dict(
        name="Latvia",
        name_in="Latvia",
        source="ESS rounds 4, 9 and 11 (citizens and non-citizens) + Eurostat census 2021 x Pew 2020 "
               "(foreign citizens)",
        basis="self-identification, sample survey (citizens and non-citizens); nationality-derived "
              "(foreign citizens)",
        note_public=(
            "**Latvia's census does not ask about religion, so this map is drawn from a survey.** In "
            "the European Social Survey's Latvian rounds from 2018 to 2024, **62.9%** of citizens said "
            "they belong to no religion, 13.6% named the Catholic Church, 11.8% the Lutheran Church and "
            "6.5% the Russian or Greek Orthodox. The Lutheran Church reported 701,118 members for "
            "2025, 37% of the population. "
            "**Latgale looks like a different country.** Its citizens are **43%** Catholic, 5% Old "
            "Believer and 2% Lutheran by their own answers, and Latgale and Riga have the most "
            "Orthodox. "
            "**What this cannot do.** The survey places people only in Latvia's six statistical "
            "regions, so all of Latgale draws one composition and all of Riga another. The regional "
            "pattern pools interviews from 2008 to 2024 and is scaled to the level of 2018 to 2024. "
            "The 190,544 recognised non-citizens are drawn with the survey's national figures for "
            "them, 45% Orthodox, and the 61,761 foreign citizens by nationality from the 2021 census."),
        how="survey, 3,933 people, 2008 to 2024; foreign citizens by nationality",
        grain="statistical regions, 315,000 people on average",
        gap=("1.11% of Latvia: the 20,175 citizens, 1.07%, and 785 non-citizens, 0.04%, who were "
             "asked about religion and declined"),
        gap_share=0.011071,
        counts=_lv_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "lv" / "lv_lau.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_lv_place_weight,
        note="THE OFFICE COUNTS CONGREGATIONS AND THE JUSTICE MINISTRY COUNTS MEMBERS, NATIONALLY; "
             "printed beside, not drawn (§3.1). ESS rounds 4, 9 and 11 at the 6 NUTS 3 regions, round "
             "4 through `regionlv` (rounds 1-4 carry a country region variable, not `region`), the "
             "level scaled to rounds 9-11 (§3.4). **ROUND 10'S `region` CARRIES NO GEOGRAPHY** (Riga "
             "37% big city against 81-95%); asserted to fail. **EUROSTAT'S `FOR` INCLUDES THE 190,544 "
             "RECOGNISED NON-CITIZENS (`RNC`)**: the foreign half targets FOR - RNC, and the "
             "non-citizens are drawn from ESS alien's-passport holders in rounds 4 and 9. Lutheran and "
             "Russian or Greek Orthodox are drawn by OVERRIDE on two witnesses (round 3, and CSP's "
             "ethnicity table); the residual construction reversed both. sources/lv.md, sources.md "
             "§9dj.",
    ),
}
