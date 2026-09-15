# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _la_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read.

    8,499 units at 763 people each, and they still need a grid, because they are village
    CATCHMENTS and not villages: the census recorded a GPS point per village and CDE grew
    travel-time polygons around those points, so a polygon is the territory whose nearest
    village is that one and its people are all at the point. Median 14.3 km² but 96 of them
    over 200 km², holding 68,769 people, and those are the upland districts of Phongsaly,
    Houaphan, Xekong and Attapeu where the `No religion` cell is largest.

    KONTUR'S LAOS EXTRACT IS THIN AND 365 VILLAGES GET NOTHING FROM IT (267,465 people,
    4.13%). Each of those carries its own polygon in the layer instead, marked `src=polygon`,
    which is an equal-share wash inside that village. sources/la_grid.py has the evidence.
    """
    return _kontur_place_weight(place, "la_hexes.gpkg", "sources/la_grid.py")


def _la_counts():
    """LSB PHC 2015 religion at VILLAGE level, via K4D: 6 categories on 8,499 villages.

    ONE level, no allocation, nothing modelled — every row is `measured` and may ring. There
    is no lookup step because there is no join: the counts and the polygons are attributes of
    the same LSB layer and `geo_id` is `unit` (sources/la_geo.py).

    THE PARTITION IS EXACT BY CONSTRUCTION AND ITS RESIDUAL IS DERIVED. Five of the six
    categories are published; `Others/not stated` is each village's population less those
    five, is non-negative in all 8,499, and reproduces LSB's own national row to 99.55%
    (sources/la.py). It is `measured` rather than `derived` because the arithmetic happens at
    the drawn unit and spreads nothing.

    THE 0.17% NOT DRAWN is 10,746 people in villages the census enumerated and the K4D layer
    does not carry, concentrated in Savannakhet (7,324), Vientiane Capital (1,474), Khammuane
    (1,388) and Phongsaly (560). Fourteen of the eighteen provinces reproduce the published
    Table 2.3 to the person, and no province is in excess, which is the shape of a coverage
    gap rather than a bad join. Stated in `gap=` per §3.5.
    """
    from la2015 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "la.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    df = df[df["geo_level"] == "village"].copy()
    df["unit"] = df["geo_id"]
    if df["unit"].nunique() != 8499:
        raise SystemExit(f"{df['unit'].nunique()} villages, expected 8,499")

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations"]]


ENTRY = {
    "la": dict(
        name="Laos",
        source="Population and Housing Census 2015, village-level services on K4D "
               "(Lao Statistics Bureau)",
        basis="self-identification, whole census population",
        view=[100.0, 13.8, 108.0, 22.6],
        gap=("10,746 people, 0.2%, in villages the census counted and the village file does "
             "not carry"),
        gap_share=0.0017,
        note_public=(
            "**Nearly a third of Laos is drawn as *Religion unknown*.** The census's answer "
            "for these 2,038,393 people, **31.4%**, is *no religion*. But it defined religion "
            "as a spiritual system with written doctrines, so only Buddhism, Christianity, "
            "Islam and the Baha'i Faith could be recorded, and people who follow the "
            "traditional religions of the uplands had no other box. The 2015 report's own "
            "summary calls the cell *no religion or being animist*, and nothing the census "
            "published says how many are which. "
            "**Its geography is the upland peoples'.** It is **96.5%** of Dakcheung in "
            "Xekong and 93.1% of Samuoi in Salavan, against 5.9% of Vientiane Capital, and "
            "the census's own tables put it at 9.2% among Lao-Tai villagers against 65% "
            "Mon-Khmer and 79% Hmong-Mien. "
            "**Christianity is 1.7% and it is not in the capital.** Vientiane Capital is "
            "0.79%; the top provinces are **Bokeo 4.9%**, Xaisomboun 3.9% and Bolikhamxai "
            "3.5%, which is Hmong and Khmu country. Read the figure as a floor: Laos "
            "regulates religious practice under Decree 315, congregations in the uplands "
            "have been closed and members detained, and a census answer given in that "
            "setting undercounts. "
            "**There are more Baha'is in Laos than Muslims**, 2,121 against 1,603. "
            "**This is drawn at village level**, 8,499 villages at about 760 people each. The "
            "published census report has no subnational religion table; the village figures "
            "are on `k4d.la`, the data platform built for a 2008 socio-economic atlas and "
            "still run by the Lao government with Swiss support."),
        how="census, 2015",
        grain="8,499 villages, 760 people on average",
        counts=_la_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "la" / "la_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_la_place_weight,
        note="THE FINEST COUNTING GEOGRAPHY OF ANY MAINLAND ASIAN COUNTRY HERE, and it was "
             "recorded as blocked until 2026-09-08. sources.md §9an closed Laos on the 2015 "
             "report, whose Table P2.9 really is national only; the office published the "
             "same census at village level through the atlas platform at k4d.la instead. "
             "Anita found the 2008 atlas's printed dot map and asked whether its source was "
             "usable. "
             "THE COUNTS AND THE POLYGONS ARE ATTRIBUTES OF ONE LSB LAYER, so there is no "
             "name join and no p-code join anywhere in this country. What replaces it as a "
             "check is the published report: aggregating the villages by the province "
             "prefix of their code reproduces Table 2.3 TO THE PERSON in 14 of 18 "
             "provinces, with 148 districts against 148 published, and the four differences "
             "are all shortfalls. "
             "FIVE CATEGORIES ARE PUBLISHED AND THE SIXTH IS A RESIDUAL. Two of the five "
             "come as percentages carrying eight significant digits, so the integer counts "
             "are recovered exactly rather than apportioned, and that recovery is itself "
             "the join check: it lands on an integer in 16,997 of 16,998 cells. The one "
             "miss is LSB's own, a Baha'i share computed on a denominator of 1,265 where "
             "the population service says 1,244, and it recovers the same 7 people either "
             "way. "
             "THE 31.4% IS DRAWN AS `unknown`, Anita's ruling of 2026-09-14, the treatment "
             "China uses, together with Mozambique's `Sem religião`. From 2026-09-08 it was "
             "the retired node `indigenous.laos`, on the case kept in sources/la.md §7; "
             "sources/la.md §10 records the national sources on how it splits between "
             "traditional religion and none. One line in taxonomy/la2015.py either way. "
             "THE VILLAGE POLYGONS ARE NOT ADMINISTRATIVE BOUNDARIES. Laos has never had "
             "official digital village boundaries; these are travel-time catchments grown "
             "around the census's own GPS point per village, and the atlas says outright "
             "that it is not intended as a planning tool at the level of single villages. "
             "They cover 97.4% of the country. Kontur's Laos extract is thin enough that "
             "365 villages get no hex at all and fall back to an equal share inside their "
             "own polygon (sources/la_grid.py).",
    ),
}
