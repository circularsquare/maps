# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _gy_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read."""
    return _kontur_place_weight(place, "gy_grid_400m.gpkg", "sources/gy_geo.py")


def _gy_counts():
    """Guyana 2012 census, at administrative region: 13 nodes on 10 units.

    ONE level, no allocation, nothing modelled, and **100% of the census drawn** — the
    thirteen categories partition the population exactly and there is no non-response
    column to leave out, because the Bureau of Statistics prorated non-response into the
    categories before publishing (see note_public). gy.csv also carries the country row,
    which is the same people a second time.

    TEN UNITS IS THE WHOLE COUNTRY AND IT IS FINER THAN IT SOUNDS. 74,700 people per unit
    is between Lithuania's 40,000 and Kenya's 1,012,000 — Guyana is simply small. The
    reason the map still reads is that Guyana's religious geography is almost entirely a
    coast/interior split, and the regions are cut across exactly that grain.
    """
    from gy2012 import resolve

    # keep_default_na=False: the Bureau's category for no religion is the string "None",
    # so default parsing turns 23,419 people into NaN, they fail to resolve, and the rows
    # are dropped with no error anywhere — §12's Philippines trap, in the second country
    # to hit it. Every check upstream of this line still passes.
    df = pd.read_csv(HERE / "data" / "normalized" / "gy.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    df = df[df["geo_level"] == "region"].copy()
    df["count"] = df["count"].astype(int)

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df = df.rename(columns={"geo_id": "unit"})
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations"]]


ENTRY = {
    "gy": dict(
        name="Guyana",
        source="Population and Housing Census 2012 (Bureau of Statistics)",
        basis="self-identification",
        view=[-61.6, 1.0, -56.4, 8.7],
        note_public=(
            "**Guyana is the indenture map.** A quarter of the country is Hindu and a "
            "fifteenth Muslim — the descendants of labourers brought from India after "
            "emancipation to cut sugar — and they are not spread about: they are on the "
            "coastal strip, in the order the plantations were. **East Berbice-Corentyne is "
            "42.1% Hindu**, Essequibo Islands-West Demerara 37.7%, Mahaica-Berbice 34.1%, "
            "Pomeroon-Supenaam 33.2%. The Muslims sit on the same ground a third as thick — "
            "Essequibo Islands 11.8%, East Berbice 9.5% — because they came on the same "
            "ships to the same estates. "
            "**And the interior is the exact photographic negative of it.** Upper "
            "Takutu-Upper Essequibo, the Rupununi savannah, is **50.1% Roman Catholic and "
            "0.4% Hindu**; Potaro-Siparuni is 39.8% Catholic. That is the Amerindian "
            "interior and the Catholic missions that worked it, and the two halves of "
            "Guyana barely touch: the country's most Hindu region and its most Catholic one "
            "are 200 km apart with almost nothing in between, because almost nobody lives "
            "in between. "
            "**Barima-Waini in the north-west is 39.9% Pentecostal and 33.8% Catholic** — "
            "three quarters of it between two churches — and Pentecostalism is the largest "
            "Christian answer in the country at 22.8%, ahead of the Anglicans and Catholics "
            "put together. "
            "**Linden is the third Guyana.** Upper Demerara-Berbice, the bauxite region, is "
            "36.0% Pentecostal, 14.8% Seventh Day Adventist, and carries both the country's "
            "highest no-religion share (7.2%) and its highest Rastafarian share (1.3%). It "
            "is Afro-Guyanese, industrial, and almost untouched by the Hindu-Muslim coast "
            "20 km away. "
            "**3,496 Rastafarians are counted here by name**, which almost no census "
            "anywhere does — most fold them into a residual — and 421 Bahá'ís. "
            "**What is missing from this map is Amerindian religion**, and its absence is "
            "an artefact of the form rather than a finding: the 2012 census offered no box "
            "for traditional practice, so the nine Amerindian nations of the interior "
            "answered one of the Christian categories instead. Read the Catholic interior "
            "as 'the church people gave as their answer', not as the whole of what is "
            "practised there."),
        how="census, 2012",
        grain="administrative regions, 75,000 people on average",
        counts=_gy_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "gy" / "gy_grid_400m.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_gy_place_weight,
        note="THE WHOLE COUNTRY IS ONE PAGE OF A PDF. The Bureau of Statistics runs no "
             "dissemination platform of any kind; Table 2.19 of the *Final 2012 Census "
             "Compendium 2* is the entire source, and it is a perfect partition — the "
             "thirteen categories sum to each region's total, the ten regions sum to the "
             "separately published national Table 2.17 category by category, and the grand "
             "total is the census population of 746,955. 100% of it is drawn. "
             "**Non-response was prorated into the categories by the office and cannot be "
             "undone.** 363 not stated, 16,331 no-contact and 7,443 institutional — 24,137 "
             "people, 3.2% of Guyana — were distributed across the thirteen categories in "
             "proportion before publication. This is the only source on the map where that "
             "has happened, so unlike everywhere else there is no non-response to leave "
             "undrawn (spec §3.5), and every count here carries its share of those people. "
             "The census is 2012; the 2022 census has published only a preliminary report "
             "with no religion table in it. "
             "Boundaries are geoBoundaries ADM1 and **the join is a published standard**: "
             "the census names no region at all — its columns are `Region 1`..`Region 10` "
             "— and geoBoundaries carries `shapeISO`, which is ISO 3166-2:GY, which is the "
             "ten regions in region-number order. No names were matched, which is as well, "
             "because the boundary file misspells Region 1 as `Barina-Waini`. "
             "Placement is Kontur's 400 m H3 grid, 5,773 hexes, and Guyana is the country "
             "that most needs it: Region 4 holds 41.7% of the population on 1.0% of the "
             "land while Region 8 is 11,077 people over 20,555 km², so an equal share per "
             "polygon would have washed the empty interior in dots of a single colour. "
             "Every region's Kontur/census ratio sits between 0.94x and 1.40x, which is "
             "what confirms the ISO join; the two loosest are Cuyuni-Mazaruni and the "
             "Rupununi, where a building-footprint model over-predicts scattered interior "
             "settlement, so placement WITHIN those two is the weakest on this country.",
    ),
}
