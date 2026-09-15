# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _jm_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read."""
    return _kontur_place_weight(place, "jm_grid_400m.gpkg", "sources/jm_geo.py")


def _jm_counts():
    """STATIN 2011 census at parish: 15 nodes on 14 units.

    ONE level, no allocation, nothing modelled — every drawn row is `measured` and may ring.
    jm.csv also carries the JAMAICA row, which is the same people again, and is dropped.

    **THE COUNTRY IS DRAWN FOR ITS CATEGORIES AND THE GEOGRAPHY IS THE PRICE.** 14 parishes
    is the coarsest counting tier on this map after Guyana's 10 and Georgia's 11, and 19
    religion categories is the best denominational detail in the Americas outside ASARB.
    §11j put the trade as *"Kenya was drawn on 47 units for its categories, and Jamaica's are
    better while its units are three times fewer"*; spec §3.9b removed the unit-count floor
    that had been holding it back.

    **4,124 PEOPLE ARE ABSENT FROM THE SOURCE AND CANNOT BE DRAWN.** STATIN excluded Bahá'í,
    Hinduism, Islam and Judaism from the parish tables — 269, 1,836, 1,513 and 506 people —
    and they are absent rather than pooled into `Other religion`. sources/jm.py asserts the
    gap is exactly 4,124 so a re-release cannot change that silently, and `gap` below states
    it on the map, because a blank cannot distinguish "no Muslims here" from "Muslims were
    not tabulated here" (§6.12).
    """
    from jm2011 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "jm.csv",
                     dtype={"geo_id": str}, low_memory=False)
    df = df[df["geo_level"] == "parish"].copy()
    if df["geo_id"].nunique() != 14:
        raise SystemExit(f"{df['geo_id'].nunique()} parishes, expected 14 -- re-run "
                         "sources/jm.py")

    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(set(df.loc[df["node"].isna(), "source_category"])
                      - {"No Data"})
    if unmapped:
        raise SystemExit(f"jm.csv has unmapped source categories: {unmapped}")
    df = df[df["node"].notna() & (df["count"] > 0)]
    df = df.rename(columns={"geo_id": "unit"})
    df["congregations"] = 0
    df["tier"] = "measured"
    return df[["unit", "node", "count", "congregations", "tier"]]


ENTRY = {
    "jm": dict(
        name="Jamaica",
        source="Census 2011 (Statistical Institute of Jamaica)",
        basis="self-identification",
        view=[-78.45, 17.65, -76.15, 18.60],
        gap=("2.3% for whom no religion was recorded; and the Baha'is, Hindus, Muslims and "
             "Jews, who were left out of the parish tables"),
        gap_share=0.02252,
        note_public=(
            "**Jamaica is drawn for its categories, not its geography.** 14 parishes is "
            "coarse — about 190,000 people each — but the census names **19 religions**, "
            "which is the most detailed religion question in the Americas outside the "
            "United States, and three of its answers exist nowhere else on this map. "
            "**Rastafari is counted where it began — 29,026 people.** Read that as a floor "
            "rather than a count: Rastafari is a way of life more than a membership, census "
            "enumeration of it is widely thought to undercount, and 1.1% is far below any "
            "cultural estimate of how many Jamaicans live by it. Nothing here scales it up. "
            "It is highest in Kingston (1.5%). "
            "**Revival Zion and Pukkumina get their own colour**, 36,296 people. They came "
            "out of the Great Revival of 1860-61, when a Christian revival met the surviving "
            "Afro-Jamaican spirit practice, and they are drawn beside Umbanda and Candomblé "
            "rather than as a Christian denomination, because that is what they are. **They "
            "peak in Saint Thomas at 3.6%**, the eastern parish where the Kongo-derived "
            "tradition concentrated. "
            "**The Church of God bodies are the biggest thing in Jamaican religion and "
            "almost nobody counts them apart.** Four of them here — in Jamaica, of Prophecy, "
            "New Testament, and other — **689,868 people between them, 25.7% of the "
            "country**, more than any single denomination. "
            "**And 21.4% report no religion, the highest share in the Americas on this "
            "map**, rising to **34.1% in Kingston** against 11.8% in Manchester. "
            "**What is missing is specific and worth naming.** Jamaica's Bahá'ís, Hindus, "
            "Muslims and Jews — 4,124 people between them — were left out of the parish "
            "tables by the statistical institute, so they are absent from this map "
            "entirely rather than folded into another colour. Jamaica has all four "
            "communities; this source simply does not place them."),
        how="census, 2011",
        grain="parishes, 191,000 people on average",
        counts=_jm_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "jm" / "jm_grid_400m.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_jm_place_weight,
        note="**THE COUNTRY §3.9b UNBLOCKED, AND IT WAS NEVER A TECHNICAL PROBLEM.** §11j "
             "verified this file end to end on 2026-09-06 and still left it unbuilt, as "
             "\"a category source on a geography that fails the floor\" — 14 parishes "
             "against a 12-unit rejection invented in §11d. That floor was already dead "
             "when it was last cited: Guyana (10 regions) and Georgia (11) had both been "
             "drawn. Anita withdrew it and the country took an afternoon. "
             "**THE JOIN IS FREE, WHICH IS THE WHOLE POINT OF THE USCB SERIES.** "
             "`GEO_MATCH` keys the counts to the boundaries by construction — 14 table "
             "keys, 14 geo keys, 14 matched, 0 unmatched — where Bosnia the same week "
             "needed four documented repairs to geoBoundaries before a name join could be "
             "attempted. And the vintage pair is checked rather than assumed: the "
             "geodatabase ships GEOG1 (2011 census) and GEOG2 (2012 survey), and §11j's CAR "
             "finding is that taking the newer layers silently breaks the join. "
             "**THE PARISH TABLES ARE MISSING FOUR RELIGIONS AND ONLY THE METADATA SHEET "
             "SAYS SO.** STATIN's national universe is 2,683,105 and the 19 published "
             "columns sum to 2,678,981; the 4,124-person difference is Bahá'í, Hinduism, "
             "Islam and Judaism, excluded from the parish tables. They are **absent, not "
             "pooled** — every parish's 19 cells sum to its own total exactly and the 14 "
             "parishes sum to the national row exactly on all 19 categories, so a build "
             "from the data sheet alone would assert Jamaica has no Muslims, Hindus, "
             "Bahá'ís or Jews at all **and every reconciliation it ran would pass**. "
             "`sources/jm.py` asserts the gap is exactly 4,124 rather than tolerating it. "
             "This is the country §11h's read-the-metadata-first rule was written for. "
             "**The ADM2 tier exists and the religion table does not reach it.** The gdb "
             "ships STATIN's `Special Areas`, and §9p's lesson is that a level can hide "
             "inside the finest one — checked, and it does not: the religion sheet is 15 "
             "rows, one country and 14 parishes. "
             "**`Other religion` is 6.3% and its geography is sharp** — 2.9% in Kingston "
             "against 14.2% in Westmoreland — which by §9r's rule makes it a missing "
             "category rather than a mixture. What is in it is not published, and it is "
             "left as an open question rather than guessed at. "
             "**Placement is Kontur's 400 m grid**, 13,373 hexes: 14 parishes over 10,991 "
             "km² averages 785 km², and uniform scatter would put dots on the Cockpit "
             "Country and the Blue Mountains. The Kontur/census ratio is the tightest on "
             "this map — **0.99x to 1.07x across all 14** — because the vintages are close "
             "and Jamaica's population barely moved between them.",
    ),
}
