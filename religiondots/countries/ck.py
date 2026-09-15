# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


ENTRY = {
    "ck": dict(
        name="Cook Islands",
        source="Census 2011 (UN Demographic Yearbook, table 28)",
        basis="self-identification",
        view=[-166.5, -22.3, -156.8, -8.6],
        gap_share=0.022,
        gap="2.2%, whose religion the census did not record",
        note_public=(
            "**Half the country belongs to one church, and it is the London Missionary "
            "Society's.** The Cook Islands Christian Church descends directly from the "
            "mission planted in 1821, and holds **7,356 people, 49.1%**. Its siblings on "
            "this map are Tuvalu's Ekalesia Kelisiano Tuvalu and Niue's Ekalesia Niue, both "
            "LMS daughters and both drawn in the same colour family. "
            "**The rest is a nineteenth and twentieth century mission list**: Roman "
            "Catholics at 17.0%, Adventists at 7.9%, Latter-day Saints at 4.4%, the "
            "Assembly of God at 3.7% and an Apostolic church at 2.1%. "
            "**The dots are spread over 2,000 km of ocean and almost all the people are "
            "on one island.** Rarotonga holds roughly three quarters of the population, so "
            "the northern atolls carry very few dots, which is correct and easy to "
            "misread as missing data."),
        how="a census question, 2011",
        grain="the country as one unit, 15,000 people",
        counts=lambda: _micro_counts("ck", "ck2011"),
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "ck" / "ck_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_micro_place_weight("ck"),
        note="queue.md had this as *`mfem.gov.ck/statistics` 404s, the office moved*; the "
             "oracle had the whole table and no office was needed. `Apostolic` goes to the "
             "pentecostal PARENT rather than to `.trinitarian`, because the word alone does "
             "not settle trinitarian against Oneness.",
    ),
}
