# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _dk_counts():
    """Denmark at the 11 landsdele (NUTS 3), from two halves of one census table.

    Danmarks Statistik publishes Church of Denmark membership from the population register (KM6,
    per kommune) and nothing about any other religion. That is `roll` (spec §3.1), and Sweden,
    Norway and Finland are drawn from self-identification, so sources/dk.py prints it beside the
    survey and it is not drawn here.

      * **Danish citizens, 5.30M.** ESS `rlgdnm`, `ctzcntr = Yes`, rounds 5, 6, 7 and 9
        (2010-2019), at the 5 regions the landsdele nest in. Round 9 publishes its region codes in
        Danmarks Statistik's order rather than NUTS order, and sources/dk.py recodes and asserts it.
      * **Foreign residents, 531k.** `cens_21ctz_r3` at NUTS 3 crossed with Pew.
    """
    from dk2024 import resolve

    cit = pd.read_csv(HERE / "data" / "normalized" / "dk.csv", dtype={"geo_id": str},
                      keep_default_na=False, na_values=[""])
    cit = cit[cit["geo_level"] == "nuts3"].copy()
    cit["node"] = cit["source_category"].map(resolve)
    unmapped = sorted(set(cit.loc[cit["node"].isna(), "source_category"]))
    if unmapped:
        raise SystemExit(f"dk.csv has unmapped source categories: {unmapped}")

    ext = pd.read_csv(HERE / "data" / "normalized" / "dk_foreign.csv", dtype={"geo_id": str})
    ext = ext[ext["geo_level"] == "nuts3"]

    df = pd.concat([cit[["geo_id", "node", "count"]], ext[["geo_id", "node", "count"]]],
                   ignore_index=True)
    df["congregations"] = 0
    # A survey and a nationality model: nothing here is `measured`.
    df["tier"] = "modelled"
    return df.rename(columns={"geo_id": "unit"})[
        ["unit", "node", "count", "congregations", "tier"]]


def _dk_place_weight(place):
    """Denmark's 99 LAUs (98 kommuner and Christianso), weighted by GISCO's 2021 population.

    The Greek weighter again. It places dots by where people live inside a landsdel and not by
    religion, and Denmark's landsdele take their composition from the 5 regions, so Bornholm is
    drawn with Hovedstaden's citizen shares although the church roll has it among the most
    Lutheran places in the country (sources/dk.md §7).
    """
    if "pop" not in place.columns:
        print("  !! dk_lau.gpkg has no `pop` column, run sources/dk_geo.py")
        return None
    return _GrLauWeighter(place)


ENTRY = {
    "dk": dict(
        name="Denmark",
        name_in="Denmark",
        source="ESS rounds 5-7 and 9 (citizens) + Eurostat census 2021 x Pew 2020 (residents)",
        basis="self-identification, sample survey (citizens); nationality-derived (residents)",
        note_public=(
            "**Denmark's church keeps a roll, and this map draws what people say instead.** On 1 "
            "January 2014, 78.4% of Denmark were members of the Church of Denmark. In the European "
            "Social Survey's four Danish rounds, from 2010 to 2019, **52.0%** of Danish citizens "
            "named a Protestant church, nearly all of them the Church of Denmark, and 43.8% named "
            "no religion. Sweden, Norway and Finland are drawn the same way. "
            "**The gradient runs from Copenhagen to North Jutland.** The Protestant answer is "
            "**38.8%** of citizens in the capital region and **66.3%** in North Jutland, and the "
            "church roll orders the five regions almost the same way. "
            "**What this cannot do.** The survey places people only in Denmark's five regions, so "
            "Bornholm gets the capital region's figures although 83.5% of it was on the church "
            "roll in 2014. Among Danish citizens, everything except the Protestant answer is "
            "divided within each region in its national proportions, so the map says nothing "
            "about where Muslim or Catholic citizens live. Denmark has not been in the survey "
            "since 2019. The 530,887 foreign citizens are drawn by nationality from the 2021 "
            "census."),
        how="survey, 6,041 people, 2010 to 2019; foreign residents by nationality",
        grain="landsdele, 530,000 people on average",
        gap_share=0.005966,
        gap=("0.60% of Denmark: the 26,232 Danish citizens, 0.45%, who were asked about religion "
             "and declined; and the 8,607 people the 2021 census recorded as stateless or of "
             "unknown citizenship, 0.15%, who are in neither half"),
        counts=_dk_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "dk" / "dk_lau.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_dk_place_weight,
        note="THE OFFICE HAS ONE CHURCH'S ROLL. Danmarks Statistik's KM6 is Church of Denmark "
             "membership per kommune from CPR and it publishes nothing on other religions; it is "
             "`roll` (§3.1) and the Nordic neighbours are self-identification, so sources/dk.py "
             "prints it beside the survey. ESS rounds 5, 6, 7 and 9 only, all NUTS 2. `rlgdndk` "
             "is round 5 only and is `rlgdnm` in Danish, so Protestant is one answer. **ROUND 9 "
             "PUBLISHES `region` IN DST'S REGION-NUMBER ORDER (1081-1085)**, not NUTS order, and "
             "its pspwght was raked to those wrong labels: recoded, asserted on domicile and "
             "party vote, and weighted by dweight. Protestant and No religion pass at 5 regions; "
             "Islam fails the rank test with a chi-square of 4e-05 and is not overridden. No "
             "§3.4 rescale: the survey's Protestant share is flat across the rounds. Orthodox not "
             "split: 94.6% Eastern on the 2009 approved-community list. sources/dk.md, "
             "sources.md §9dg.",
    ),
}
