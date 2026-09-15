# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


ENTRY = {
    "ms": dict(
        name="Montserrat",
        source="Census 2001 (UN Demographic Yearbook, table 28)",
        basis="self-identification",
        view=[-62.27, 16.65, -62.12, 16.84],
        gap_share=0.108,
        gap="10.8%, whose religion the census did not record",
        note_public=(
            "**This census counts 4,303 people, and the reason is a volcano.** The "
            "Soufriere Hills eruption began in 1995, buried Plymouth, the capital, and drove "
            "roughly two thirds of Montserrat's population off the island. The 2001 census "
            "counted what was left. Everyone drawn here is in the northern third of the "
            "island, because the rest is an exclusion zone. "
            "**The religious composition is an ordinary Leeward Islands one**: Anglicans "
            "21.8%, Methodists 17.0%, Pentecostals 14.2%, Roman Catholics 11.6%, Adventists "
            "10.6%. What is unusual is only how few people it describes. "
            "**One person in nine has no religion recorded**, 465 people, which is the "
            "largest such share in this group of small countries and worth holding in mind "
            "beside the percentages above. "
            "**Montserrat draws no dots either.** Its largest church has 937 members, and "
            "one dot is a thousand people, so every religion here appears as a ring rather "
            "than as a dot. Only Niue is smaller."),
        how="a census question, 2001",
        grain="the country as one unit, 4,300 people",
        counts=lambda: _micro_counts("ms", "ms2001"),
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "ms" / "ms_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_micro_place_weight("ms"),
        note="queue.md: *the 2023 note lists religion among topics collected* — that round "
             "is not in the oracle, so 2001 is what is drawn. Kontur's hexes are all in the "
             "north, which independently reproduces the exclusion zone.",
    ),
}
