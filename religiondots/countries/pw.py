# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


ENTRY = {
    # ---- THE MICROSTATE TIER (sources/micro.py) -----------------------------------------
    # Nine countries, one instrument, one shape: UNSD Demographic Yearbook table 28 at
    # national level, placed on Kontur population hexagons. Anita's call of 2026-09-08 is
    # what makes a national-only table a complete source rather than a coarse one, and
    # `_micro_counts`'s docstring carries the reasoning. Every entry here says its census
    # year in `how=`, because four of the nine are 2001 or older.
    "pw": dict(
        name="Palau",
        source="Census 2005 (UN Demographic Yearbook, table 28)",
        basis="self-identification",
        view=[134.05, 6.85, 134.75, 8.15],
        gap="none, the nine published categories account for every person counted",
        note_public=(
            "**Palau has a religion of its own, and 8.7% of the country belongs to it.** "
            "Modekngei was founded around 1915 by a man named Temedad on Babeldaob, and it "
            "joins Palauan spirit belief to Christian elements and a healing practice. The "
            "Japanese administration suppressed it in the 1930s and 1940s. **1,733 people** "
            "in this census, which is a larger share of its country than any other "
            "indigenous religion drawn on this map. "
            "**Everything else is mission history.** Half the country is Catholic, from the "
            "Spanish and then German and Japanese Catholic missions, and about a quarter is "
            "Protestant, largely the Evangelical Church of Palau descended from the "
            "American Board. The census names no Protestant body, so they are drawn on an "
            "unspecified Protestant colour rather than a named church. "
            "**The whole country is one unit and that is deliberate.** Palau has 19,907 "
            "people, so at one dot per thousand it draws seventeen dots; where inside the "
            "archipelago they sit is decided by population density and asserts nothing "
            "about religion. Read this country as a bar chart that happens to be shaped "
            "like Palau."),
        how="a census question, 2005",
        grain="the country as one unit, 19,900 people",
        counts=lambda: _micro_counts("pw", "pw2005"),
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "pw" / "pw_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_micro_place_weight("pw"),
        note="**THE FIRST COUNTRY BUILT FROM THE ORACLE ALONE**, and the one that justified "
             "the tier: `modekngei` is a new ROOT and nothing else on this map holds it. "
             "The 2005 round publishes nine categories where 1995 published twelve, so the "
             "Baha'i, Assembly of God and Church of Christ counted then are inside `Other` "
             "now. `None or Refused` is drawn as irreligion on the strength of the 1995 "
             "census, which asked them apart and found **None 1,577 against Refused 7**.",
    ),
}
