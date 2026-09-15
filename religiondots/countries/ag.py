# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


ENTRY = {
    "ag": dict(
        name="Antigua and Barbuda",
        source="Census 2001 (UN Demographic Yearbook, table 28)",
        basis="self-identification",
        view=[-62.00, 16.90, -61.62, 17.78],
        gap_share=0.017,
        gap="1.7%, who declined to declare a religion",
        note_public=(
            "**The Moravians are 10.5% of Antigua and Barbuda, the highest Moravian share "
            "of any country on this map.** 8,057 people, and the fourth-largest church in "
            "the country. The mission dates from 1756 and was directed at the enslaved "
            "population, which is why Antigua, Barbados and the Danish Virgin Islands have "
            "Moravian communities where most of the Caribbean does not. "
            "**Above them the shape is the standard British Leeward one**: Anglicans 25.7%, "
            "Adventists 12.3%, Pentecostals 10.6%, Roman Catholics 10.4%, Methodists 7.9%. "
            "Twenty-one categories in all, which is a deep list for a country of 77,000, "
            "and it names the Rastafari, the Hindus, the Baha'is and 228 Muslims separately. "
            "**Barbuda carries about 2% of the dots**, which is about its share of the "
            "population; the census publishes no separate figure for it, so nothing here "
            "distinguishes the two islands beyond where people live."),
        how="a census question, 2001",
        grain="the country as one unit, 76,900 people",
        counts=lambda: _micro_counts("ag", "ag2001"),
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "ag" / "ag_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_micro_place_weight("ag"),
        note="**THE ONE PARTITION IN THIS TIER THAT DOES NOT CLOSE**: the 21 categories sum "
             "to 76,889 against a stated 76,886, three people over, in the Yearbook rather "
             "than in this code. `sources/micro.py` allows exactly 3 for `ag` and 0 for "
             "every other country, so a real break still fails the build. `Spiritualist` "
             "(66) goes to `spiritualism` on the literal reading; in an eastern Caribbean "
             "census it could be the Spiritual Baptist tradition, which gd2021.py files "
             "elsewhere from a category that says so in full.",
    ),
}
