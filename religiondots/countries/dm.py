# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


ENTRY = {
    "dm": dict(
        name="Dominica",
        source="Census 2001 (UN Demographic Yearbook, table 28)",
        basis="self-identification",
        view=[-61.55, 15.14, -61.20, 15.70],
        gap_share=0.01,
        gap="1.0%, who did not specify a religion",
        note_public=(
            "**Dominica is the odd one out in the eastern Caribbean, and the reason is "
            "French.** It is **61.4% Roman Catholic** where Antigua is 10.4% and Montserrat "
            "11.6%, and **0.6% Anglican** where Antigua is 25.7%. The island changed hands "
            "repeatedly through the eighteenth century and the French missionary period left "
            "a Catholic majority that British rule never displaced. St Lucia, next door, has "
            "the same shape for the same reason. "
            "**The Rastafari are counted separately and are 1.3% of the country**, 879 "
            "people, one of the higher shares anywhere on this map. "
            "**One category is a family rather than a body.** *Other evangelical churches* "
            "holds 4,882 people, 7.1%, and is drawn on an unspecified evangelical colour: "
            "the census knows they are in evangelical congregations and does not say which."),
        how="a census question, 2001",
        grain="the country as one unit, 68,600 people",
        counts=lambda: _micro_counts("dm", "dm2001"),
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "dm" / "dm_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_micro_place_weight("dm"),
        note="queue.md: *religion by sex in the 2011 census tables*; the oracle's latest "
             "Dominican row is 2001 and that is what is drawn.",
    ),
}
