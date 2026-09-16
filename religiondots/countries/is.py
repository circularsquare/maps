# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _is_counts():
    """Iceland at its two NUTS 3 units, from two halves of one census table.

    Hagstofa publishes the faith-body register nationally for every body and by parish for the
    Church of Iceland alone. That is `roll` (spec §3.1), and Norway, Denmark and Sweden are drawn
    from self-identification, so sources/is.py prints it beside the survey and it is not drawn.

      * **Icelandic citizens, 312k.** ESS `rlgdnis`/`rlgdnais`, `ctzcntr = Yes`, rounds 6, 8, 10
        and 11, at IS001 (the capital area) and IS002 (the rest). Round 9's region is not the NUTS
        3 split and is left out; sources/is.py asserts it.
      * **Foreign residents, 47k.** `cens_21ctz_r3` at NUTS 3 crossed with Pew.
    """
    from is2024 import resolve

    cit = pd.read_csv(HERE / "data" / "normalized" / "is.csv", dtype={"geo_id": str},
                      keep_default_na=False, na_values=[""])
    cit = cit[cit["geo_level"] == "nuts3"].copy()
    cit["node"] = cit["source_category"].map(resolve)
    unmapped = sorted(set(cit.loc[cit["node"].isna(), "source_category"]))
    if unmapped:
        raise SystemExit(f"is.csv has unmapped source categories: {unmapped}")

    ext = pd.read_csv(HERE / "data" / "normalized" / "is_foreign.csv", dtype={"geo_id": str})
    ext = ext[ext["geo_level"] == "nuts3"]

    df = pd.concat([cit[["geo_id", "node", "count"]], ext[["geo_id", "node", "count"]]],
                   ignore_index=True)
    df["congregations"] = 0
    # A survey and a nationality model: nothing here is `measured`.
    df["tier"] = "modelled"
    return df.rename(columns={"geo_id": "unit"})[
        ["unit", "node", "count", "congregations", "tier"]]


def _is_place_weight(place):
    """Kontur's hexes, each scaled so its municipality holds Hagstofa's 1 January 2021 population.

    Uncalibrated, Kontur reads the summer-house municipalities several times over their registered
    population (Skorradalshreppur 9.6x, Grímsnes- og Grafningshreppur 6.8x) and the capital area
    at 0.89 of its census share (sources/is.md §5). A population weight, not a religion one.
    """
    return _kontur_place_weight(place, "is_grid_400m.gpkg", "sources/is_geo.py")


ENTRY = {
    "is": dict(
        name="Iceland",
        name_in="Iceland",
        source="ESS rounds 6, 8, 10 and 11 (citizens) + Eurostat census 2021 x Pew 2020 (residents)",
        basis="self-identification, sample survey (citizens); nationality-derived (residents)",
        note_public=(
            "**Iceland registers every resident in a faith body or in none, and this map draws "
            "what people answer in a survey instead.** In 2021, 63.5% of Iceland's register was in "
            "the Church of Iceland. In four rounds of the European Social Survey from 2012, "
            "**34.8%** of Icelandic citizens named the Church of Iceland and 58.4% named no "
            "religion. Norway, Denmark and Sweden are drawn the same way. "
            "**The survey places people only in the capital area or outside it.** The Church of "
            "Iceland is **31.6%** of citizens' answers in the capital area and **40.0%** outside "
            "it, and the church's parish roll orders the two the same way (56.8% and 69.2% of "
            "adults on it in December 2021). The Free Church, whose congregations are in Reykjavík "
            "and Hafnarfjörður, is 2.2% of the capital area's answers and 0.3% elsewhere. Catholic, "
            "Ásatrú and every other answer among citizens except no religion are divided within "
            "each area in their national proportions, so the map says nothing about where those "
            "citizens live.The 47,073 foreign citizens, 19,267 of them Polish, are drawn by nationality "
            "from the 2021 census."),
        how="survey, 3,222 people in four rounds from 2012; foreign residents by nationality",
        grain="the capital area and the rest of Iceland, 180,000 people on average",
        gap_share=0.004824,
        gap=("0.48% of Iceland: the 1,655 Icelandic citizens, 0.46%, who were asked about religion "
             "and declined; and the 77 people the 2021 census recorded as stateless or of unknown "
             "citizenship, 0.02%, who are in neither half"),
        counts=_is_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "is" / "is_grid_400m.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_is_place_weight,
        note="THE OFFICE HAS A REGISTER OF EVERY FAITH BODY, NATIONALLY. Hagstofa MAN10001 counts "
             "about sixty registered bodies with no geography, and MAN10289-10302 split parishes "
             "only into in and not in the Church of Iceland; it is `roll` (§3.1) and the Nordic "
             "neighbours are self-identification, so sources/is.py prints it beside the survey. "
             "ESS region for Iceland is NUTS 3, two units, in every round. **ROUND 9's `region` "
             "IS NOT THE NUTS 3 SPLIT** (IS001 79% of the sample, IS002 77% village or farm): "
             "left out, asserted on domicile. Cards `rlgdnis` (6, 8) and `rlgdnais` (9-11), two "
             "answers harmonised, nesting in `rlgdnm` asserted. Two units: the rank split-half is "
             "printed and decides nothing; uz.py's two-unit test (respondents shuffled within "
             "round, chi-square) passes the Church of Iceland and the Free Church. No §12 rescale: "
             "1.92 points of drift. Ásatrú on `paganism`; `other.is` new. Kontur calibrated to "
             "MAN02005 municipality totals because it puts summer-house districts at up to 9.6x "
             "their register. sources/is.md, sources.md §is-2026-09-15.",
    ),
}
