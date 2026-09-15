# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _gt_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read.

    Guatemala needs this for §8.2's emptiness reason and needs it badly: 22 departments over
    108,889 km2 is 4,950 km2 a unit, and **Petén alone is 35,903 km2 -- a third of the
    country's land holding 3.6% of its people**, most of it the Maya Biosphere Reserve. An
    equal share per polygon would put the single largest block of colour on the Guatemalan
    map into uninhabited forest, and smear the capital's fifth of the country across a
    department that is also volcanoes (sources/gt_grid.py).
    """
    return _kontur_place_weight(place, "gt_hexes.gpkg", "sources/gt_grid.py")


def _gt_counts():
    """LAPOP AmericasBarometer, waves 2010-2023 pooled, at departamento: 11 categories, 22
    units, and EVERY ROW IS `modelled` IN §7.

    THIS IS THE FIRST COUNTRY ON THIS MAP DRAWN FROM THE AMERICASBAROMETER, and it is here
    because Guatemala's census has never asked. sources.md §11x closed the country on two
    witnesses -- the UNSD oracle reports 1964 as the only Guatemalan religion tabulation ever
    forwarded, and IPUMS's RELIGION variable says the same -- and nothing here reopens that.
    §11ad is the assessment of LAPOP as a source across the nine countries it could serve.

    THE CONSTRUCTION INVENTS NO MAGNITUDE. A department's share comes from the survey; the
    number of people it applies to comes from OCHA COD-PS 2024, which is INE's own
    projection. Every person drawn is a person COD-PS counts in that department, and the
    survey only decides the column -- §14.4 rule 1, the same shape as sources/kz.py.

    `modelled` AND NOT `derived`, ON §7b'S OWN TEST. The tiers are about whether anybody was
    counted, and nobody counted religion in Guatemala at any level: 8,919 respondents cut 22
    ways, pooled over fourteen years, against a projected population. §7b moved the American
    residual for less than this.

    THREE CATEGORIES CARRY THEIR OWN DEPARTMENT GEOGRAPHY AND EIGHT DO NOT, and two of the
    three are measured rather than chosen. sources/gt.py splits the six waves in half and
    ranks the departments in each; Catholic (+0.57) and Evangelical (+0.50) clear the 95% bar
    for 22 units, and `Protestante Tradicional` comes out at **-0.04** despite being 5.4% of
    the country. The ones that fail are drawn at their national rate inside each department's
    own residual, so the people are drawn and only the claim to know where they are is
    withdrawn.

    THE THIRD IS `Ninguna (creyente)` AND IT IS UNDER THE BAR, DRAWN ON ANITA'S CALL
    (2026-09-08, sources/gt.py's `OVERRIDE`, which prints its reason on every run). The
    split-half asks whether the ORDERING replicates and returns +0.21; it never asks whether
    the departments differ at all, and they do at p=3.6e-16. So +0.21 means the ordering is
    not pinned rather than that the variation is fake -- spec §14.16's China at +0.17 exactly,
    which was also drawn. The bar was NOT moved: an override is one named category with a
    written reason, because moving the bar would silently redraw every other category too.

    THE UNIVERSE IS ADULTS AND THE DOTS ARE EVERYBODY. LAPOP interviews people aged 18 and
    over; the shares are applied to the whole population, which assumes Guatemala's children
    are distributed like its adults. Drawing only the adults would leave 45% of a very young
    country blank, and §6.12 is about how badly a blank reads on a dot map.
    """
    from gt2023 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "gt.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])

    lut = pd.read_csv(HERE / "data" / "geo" / "gt" / "gt_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"gt.csv departments with no polygon: {missing} -- re-run "
                         "sources/gt_geo.py, the lookup is stale")
    if df["unit"].nunique() != 22:
        raise SystemExit(f"{df['unit'].nunique()} departments, expected 22")

    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(df.loc[df["node"].isna(), "source_category"].unique())
    if unmapped:
        raise SystemExit(f"gt.csv categories with no node: {unmapped}")
    df = df[df["count"] > 0]
    # EVERY row, without exception -- there is no measured tier in this country (§7).
    df["tier"] = "modelled"
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations", "tier"]]


ENTRY = {
    "gt": dict(
        name="Guatemala",
        source="AmericasBarometer, six rounds 2010 to 2023 (LAPOP Lab, Vanderbilt "
               "University), against OCHA COD-PS 2024 department populations",
        basis="self-identification, adults 18 and over",
        view=[-92.4, 13.5, -88.1, 18.0],
        note_public=(
            "**Guatemala has never counted its religions, so this is a survey standing "
            "where a census would be.** The only religion tabulation a Guatemalan census has "
            "ever published is from 1964. The map is drawn from the LAPOP AmericasBarometer "
            "instead: **8,919 people** interviewed across six rounds between 2010 and 2023, "
            "pooled, and applied to the 2024 population estimates. That is a thinner thing "
            "than any census on this map, and the dots are drawn desaturated to say so. "
            "**A third of the country answers Evangelical or Pentecostal, and it is not "
            "spread evenly.** It is **50.7% of Izabal** on the Caribbean coast, 45.2% of "
            "Retalhuleu and 44.1% of Petén, against **12.5% of Chiquimula** and 17.0% of El "
            "Progreso in the eastern dry corridor. Four to one across a country you can "
            "drive across in a day. The departments that are almost entirely Maya sit in the "
            "middle of that range rather than at either end, so what the map shows is not "
            "the ethnic map with different colours. "
            "**Three of the eleven answers are drawn where the survey found them, and the "
            "rest are spread evenly.** Catholic and Evangelical keep their own department "
            "shares because their ranking survives being asked twice: splitting the six "
            "rounds in half and re-ranking the 22 departments returns **+0.57** and +0.50. "
            "`Protestante Tradicional` returns **-0.04** on 5.4% of the country, so its "
            "955,906 people are drawn at the national rate inside each department rather "
            "than wherever 480 respondents happened to land. "
            "**Believing without a church is the third, and it is the weakest thing on this "
            "map of Guatemala.** Its ranking only half survives the same test, at +0.21, so "
            "trust the two ends and not the middle: **Guatemala department is 8.4%** and it "
            "comes top in both halves of the survey, the Maya highlands sit at 1 to 2% in "
            "both, and the fourteen departments in between move around enough that their "
            "order should not be read. It is drawn rather than spread flat because the "
            "departments do differ, decisively so, and a flat layer would have claimed that "
            "Quiché and the capital are alike. "
            "**0.22% for traditional religion is a floor, in a country that is 43.6% "
            "indigenous by its own 2018 census.** The card offers one worldwide "
            "`Religiones Tradicionales` box and never prints the word *costumbre*. The same "
            "instrument was checked against a census in Suriname, where its "
            "traditional-religion cell comes out at **0.21x** the census figure and the "
            "missing people reappear as Christians. So 39,521 is what the question caught "
            "rather than what is there, and no published source says what the right number "
            "would be. "
            "**Zacapa rests on 40 interviews**, which is a 95% interval of plus or minus 15 "
            "points on its Catholic share; every other department has at least 129 and the "
            "median has 279. And because the rounds are pooled, the level is an average over "
            "fourteen years in which Guatemalan Catholicism fell from 55% to 50%, so this "
            "map is a couple of points more Catholic than the 2023 round alone would draw."),
        how="survey, six rounds 2010 to 2023 pooled",
        grain="departments, 810,000 people on average",
        counts=_gt_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "gt" / "gt_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_gt_place_weight,
        note="THE COUNTRY IS A SURVEY ON A PROJECTION AND EVERY ROW IS `modelled` (§7b). "
             "LAPOP's `q3c` gives a department share; OCHA COD-PS 2024 gives the number of "
             "people it applies to. No magnitude is invented -- every person drawn is a "
             "person COD-PS counts in that department, and the survey only decides the "
             "column (§14.4 rule 1, the construction sources/kz.py uses). "
             "WHICH CATEGORIES CARRY A GEOGRAPHY IS DECIDED BY A SPLIT-HALF, NOT BY SIZE. "
             "The first version of sources/gt.py cut at 4% of the country and would have "
             "drawn `Protestante Tradicional` on its own department shares; its split-half "
             "rank correlation is -0.04. Size is eligibility and stability is evidence, and "
             "the bar is what it takes to be distinguishable from zero at 95% on 22 units, "
             "which is +0.43. `Ninguna (creyente)` returns +0.21 and IS DRAWN ANYWAY, on "
             "Anita's call of 2026-09-08 and against this file's first version, which spread "
             "it flat. The split-half is not the only evidence: the departments differ at "
             "chi-square p=3.6e-16, so +0.21 says the ordering is not pinned rather than that "
             "the variation is fake, which is §14.16's China at +0.17 and that one was drawn. "
             "The bar was not moved -- `sources/gt.py`'s `OVERRIDE` names the one category "
             "and prints the reason on every run, because moving a bar redraws everything "
             "silently and naming a category does not. "
             "THE JOIN IS TWO INDEPENDENT KEYS AND BOTH ARE ASSERTED. LAPOP's `prov` for "
             "Guatemala is 200 plus the official department number, and COD's pcode is `GT` "
             "plus the same number; sources/gt_geo.py refuses to run unless the NAMES on "
             "those codes also agree, one at a time, on all 22. sources/ni_geo.py is why: "
             "Nicaragua's code join matched 145 of 153 and silently sent Waspám's Moravians "
             "inland, and a permutation preserves every total. "
             "AND TWO HELD-OUT CHECKS TEST THE DECODE WITHOUT TOUCHING RELIGION. LAPOP's "
             "weighted department distribution tracks COD-PS's population distribution at "
             "r=+0.965, and **none of 20,000 random pairings of the same 22 departments "
             "reaches that** (best random +0.937). The permutation is the test; a "
             "correlation any random pairing could produce would say nothing. "
             "AN EARLIER VERSION ALSO ASSERTED ON DEPARTMENT MEAN ADULT AGE, AT r=+0.52, AND "
             "THAT WAS NOT A WITNESS. The comparison was measured afterwards and Guatemala's "
             "between-department variance in mean adult age (0.887) is BELOW the sampling "
             "variance on those means (1.013), so the twenty-two departments are "
             "indistinguishable from twenty-two draws on one distribution and the +0.52 was "
             "luck. It is still printed, with that ratio beside it, and it decides nothing. "
             "sources/lapop.py has the numbers for both countries. "
             "THE 2021 ROUND IS ABSENT FROM THE POOL BECAUSE IT CARRIES NEITHER RELIGION NOR "
             "GEOGRAPHY -- it is the COVID telephone round -- and the question does not "
             "appear at all before 2010, so six rounds is the whole of what exists. "
             "sources.md §11ad assesses LAPOP across all nine countries it could serve here; "
             "sources/gt.md is Guatemala's own record.",
    ),
}
