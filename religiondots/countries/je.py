# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


ENTRY = {
    # ---- JERSEY (sources/je.py, sources/je.md) ---------------------------------------------
    # The microstate tier's shape (one unit, Kontur hexes) from survey shares. _terr_counts with
    # tier="modelled", because _micro_counts would mark a survey share `measured`.
    "je": dict(
        name="Jersey",
        source="Opinions and Lifestyle Survey 2023, with church shares from the 2018 survey "
               "(Statistics Jersey), against Statistics Jersey's end-2023 population estimate",
        basis="self-identification, adults 16 and over in private households",
        view=[-2.30, 49.15, -1.98, 49.28],
        how="survey, one round of 1,514 adults in 2023; church shares from the 2018 round",
        grain="the island as one unit, 104,000 people",
        gap_share=0.11,
        gap="the 11% of adults who were not sure whether they had a religion",
        note_public=(
            "Jersey's census does not ask about religion. This map is drawn from Statistics "
            "Jersey's Opinions and Lifestyle Survey, which asked **1,514** adults in June and "
            "July 2023 whether they regarded themselves as having a religion: **39%** said yes, "
            "**50%** no and **11%** were not sure. Of those who said which religion, 93% named "
            "Christianity or a Christian church. The shares are applied to the 104,030 people "
            "Statistics Jersey estimates lived in Jersey at the end of 2023, and the people "
            "who were not sure are not drawn. "
            "The 2023 report does not split Christians by church, so the map takes that split "
            "from the 2018 round of the same survey: of respondents who named a church, 50% "
            "said Catholic, 39% Church of England and 12% another church. In 2015 Catholics "
            "and Anglicans were level, at 43% and 44%, so the split between the two churches "
            "is rough. The other 7% of those who named a religion are drawn as one group; the "
            "survey does not report them separately, and the 2015 report mentions Buddhist, "
            "Hindu, Jewish, Muslim and Sikh answers. "
            "The survey reports St Helier, the three suburban parishes and the eight rural "
            "parishes separately, and between 38% and 40% of adults in each said they had a "
            "religion, so the island is drawn as one unit and the dots follow where people "
            "live."),
        counts=lambda: _terr_counts("je", "je2023", 1, tier="modelled"),
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "je" / "je_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_micro_place_weight("je", "sources/je.py"),
        note="JOLS 2023 LEVEL FROM THE OFFICE'S OPEN RESULTS TABLE, CHURCHES FROM JOLS 2018. "
             "opendata.gov.je's jols_2023_results.csv gives Q17.5 Yes 0.39, No 0.50, Not sure "
             "0.11; the 2023 report gives 93% Christian of those naming a religion and no church "
             "split, so the 2018 report's Catholic 50, Church of England 39, other 12 split "
             "the Christians (a mixed vintage, ask 003's test; the 2015 round and a "
             "place-of-birth witness in sources/je.md §3). Laid on Statistics Jersey's revised "
             "end-2023 estimate, 104,030. Not sure excluded into gap. One unit: the table's "
             "three parish types read Yes 0.38 to 0.40. sources/je.md.",
    ),
}
