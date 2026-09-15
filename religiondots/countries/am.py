# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _am_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read."""
    return _kontur_place_weight(place, "am_grid_400m.gpkg", "sources/am_geo.py")


def _am_counts():
    """Armstat 2022 census at marz: 14 nodes on 11 units.

    ONE level, no allocation, nothing modelled. am.csv also carries the AM country rows and
    a `country_by_ethnicity` block, which are the same people again and are filtered out
    here rather than summed.

    **THE COARSENESS IS THE ARGUMENT AND IT IS SETTLED ON PEOPLE PER UNIT**, which is
    Anita's test and the one Georgia was judged on: 11 marzes for 2.93 million is 266,612
    each, finer per person than Georgia's 334,000 and than Russia's 1.8 million, both drawn.
    Nothing finer exists. Armstat publishes religion in section 5 of the census volumes and
    section 5 stops at the marz, while section 1 of the same volume goes down to the
    settlement for population alone (sources/am.md).
    """
    from am2022 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "am.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    df = df[df["geo_level"] == "marz"].copy()
    df["count"] = df["count"].astype(int)
    if df["geo_id"].nunique() != 11:
        raise SystemExit(f"{df['geo_id'].nunique()} marzes, expected 11 -- re-run "
                         "sources/am.py")

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df = df.rename(columns={"geo_id": "unit"})
    df["congregations"] = 0
    return df.groupby(["unit", "node"], as_index=False)[["count", "congregations"]].sum()


ENTRY = {
    "am": dict(
        name="Armenia",
        source="2022 Population Census (Statistical Committee of Armenia)",
        basis="self-identification",
        note_public=(
            "**One church is 95.2% of Armenia, so this map is about the other five "
            "percent.** The Armenian Apostolic Church has been the country's church since "
            "the early fourth century, and across the eleven marzes it runs from **92.1% "
            "in Lori** to **98.8% in Syunik**. Eleven units for three million people means "
            "this is regional composition and not neighbourhoods. "
            "**The Catholics are two marzes in the north and almost nowhere else.** Lori "
            "is **3.2%** Catholic and Shirak **2.9%**, and between them they hold 77% of "
            "the 17,855 Catholics the marz tables print, against 0.03% in Syunik. These "
            "are the Armenian "
            "Catholic villages around Gyumri, Artik and Tashir, in communion with Rome and "
            "Armenian in rite. The census prints one undivided Catholic box, so the "
            "Latin-rite parish in Yerevan is inside the same colour. "
            "**Armenia's Yazidis are counted under their own name for their religion.** "
            "The box is Sharfadin rather than Yezidi, which is unusual: the three other "
            "censuses on this map that count Yazidis, in Australia, Georgia and the United "
            "Kingdom, all label it with the ethnonym. It is **3.7% of Aragatsotn** and "
            "**2.2% of Armavir**, the slopes of Aragats and the Ararat plain, and the "
            "community is Kurmanji-speaking and descends largely from refugees of the "
            "Ottoman persecutions of the 1910s and 1920s. "
            "**The box does not hold all of them, and the census says so itself.** It also "
            "cuts religion by ethnicity: of the **31,079** people who gave Yezidi as their "
            "ethnicity, 13,256 chose Sharfadin, **9,939 chose Armenian Apostolic**, 3,246 "
            "are in the residual and 1,672 chose Pagan. So the pagan colour on the Ararat "
            "plain is mostly Yazidi rather than the Armenian neopagan revival it looks "
            "like, and the two colours there are largely one community. "
            "**The Molokans are still where the Tsar put them.** 1,578 of the 1,982 "
            "Molokans counted in the marz tables are in Lori: Russian Spiritual Christians "
            "who broke with the Orthodox "
            "Church in the 18th century over icons, priesthood and the sacraments, were "
            "exiled to the Caucasus under Nicholas I, and have farmed the high country at "
            "Fioletovo and Lermontovo since the 1840s. They are drawn as other Christians "
            "because no other source on this map counts them at all. "
            "**Refusing the question is a Yerevan habit.** 38,931 of the 49,359 refusals "
            "recorded across the eleven marzes are in the capital, **3.6% of the city** "
            "against 0.3% in Tavush, and "
            "those people are not drawn. Only 0.6% reported no religion, which is low for a "
            "country that spent seventy years in the Soviet Union."),
        how="census, 2022, direct question; 1.7% refused to answer",
        grain="marzes, 267,000 people on average",
        gap="the 1.7% who refused the religion question",
        gap_share=0.0168,
        counts=_am_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "am" / "am_grid_400m.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_am_place_weight,
        note="**A SWEEP CLOSED THIS COUNTRY ON THE WRONG PAGE AND THE ENGLISH SITE IS WHY.** "
             "sources.md §11o recorded *'armstat.am census pages carry no religion table'*. "
             "The 2022 results page really does look empty: it is a clickable GIF map of "
             "Armenia whose eleven `<area>` polygons **all point at the same national "
             "volume**, so clicking any marz returns the same file. The eleven marz volumes "
             "are one nid each and reachable only from the left nav, and under `/en/` every "
             "one of them is a bare heading over *'Information is not available in "
             "English'*. The same nids under `/am/` carry nine section archives apiece. **A "
             "sweep that reads only the English tree sees a published census as an "
             "unpublished one.** "
             "**The newer census is not the finer one here.** Both 2011 and 2022 stop at "
             "the marz; 2022 is taken for being newer, longer in the category list and "
             "exactly reconciling. "
             "**Each marz prints only the columns it has people in**, so the header is a "
             "different set in every file and the marz tables are FINER than the national "
             "one in two places, breaking out `Protestant` and `TM` that national table 5.5 "
             "folds away. The consequence is that a rare answer in a marz with no column "
             "for it sits in that marz's residual: **483 of Armenia's 515 Muslims are "
             "printed and the other 32 are inside `Other` somewhere among seven marzes.** "
             "So a category's marz sum is a floor, not an equality, and the checks in "
             "sources/am.py assert that shape rather than the wrong one. What does hold "
             "exactly is the arithmetic that matters: every marz's own columns sum to its "
             "own published population, and the eleven totals sum to 2,932,731 to the "
             "person. "
             "**11 units, and the coarseness is settled on people per unit**, which is "
             "Anita's test: 266,612 each is finer per person than Georgia's 334,000 next "
             "door and Russia's 1.8 million, both drawn. The join to geoBoundaries ADM1 is "
             "eleven names both ways with nothing spare, proved on population rather than "
             "on spelling, and Yerevan at 0.79x with Kotayk at 1.34x is §9q's city/ring "
             "pair again, inside the band and asserted rather than excused.",
    ),
}
