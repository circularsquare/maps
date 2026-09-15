# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


ENTRY = {
    "mh": dict(
        name="Marshall Islands",
        source="Census 1999 (UN Demographic Yearbook, table 28)",
        basis="self-identification",
        view=[160.8, 4.5, 172.3, 15.0],
        gap="none, the four published categories account for every person counted",
        note_public=(
            "**Four categories for 50,848 people: the shallowest religion table drawn "
            "anywhere on this map.** Read the Marshall Islands as a four-way split and "
            "nothing finer. "
            "**The 54.8% marked Protestant is one church, and the census does not say so.** "
            "It is overwhelmingly the United Church of Christ, the Congregational church the "
            "American Board planted in 1857, and the direct sibling of the national churches "
            "this map draws by name for Tuvalu, Niue and the Cook Islands. It is drawn on an "
            "unspecified Protestant colour because naming it would assert something this "
            "source does not. "
            "**The Assembly of God is 25.8%, the highest share of any country here.** That "
            "is a real feature of Micronesia rather than an artefact of the coding. "
            "**And 11.1% is simply *other*.** That cell certainly contains Bukot nan Jesus, "
            "the indigenous Marshallese church, and the country's Baha'i community, and "
            "probably its Latter-day Saints. None of them is named. "
            "**This is also the oldest census on the map.** The Marshall Islands ran "
            "censuses in 2011 and 2021 and neither published a religion table that reached "
            "the Demographic Yearbook."),
        how="a census question, 1999",
        grain="the country as one unit, 50,800 people",
        counts=lambda: _micro_counts("mh", "mh1999"),
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "mh" / "mh_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_micro_place_weight("mh"),
        note="Drawn because §3.9c's variety floor was retired for this tier, not because "
             "four categories is a good table. queue.md's *rmi-data.sprep.org is a 403* "
             "stands and is the route that would replace it. Kontur puts most of the dots "
             "on Majuro and Ebeye, which is where most Marshallese live.",
    ),
}
