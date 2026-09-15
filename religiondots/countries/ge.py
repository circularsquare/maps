# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _ge_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read."""
    return _kontur_place_weight(place, "ge_grid_400m.gpkg", "sources/ge_geo.py")


def _ge_counts():
    """Geostat 2014 census at region: 10 nodes on 11 units.

    ONE level, no allocation, nothing modelled. ge.csv also carries the GEORGIA row, which
    is the same people again.

    **`None` IS A CATEGORY NAME HERE AND PANDAS READS IT AS NaN.** §9m's trap, third
    country. Georgia's irreligious answer is the literal string `None`, so a default
    `read_csv` silently deletes 19,080 people and the map draws a country with no
    unaffiliated population at all. `keep_default_na=False` is not defensive tidiness in
    this file; it is the difference between drawing a category and not.

    **THE COARSEST GEOGRAPHY DRAWN, AND IT EARNS ITS PLACE ON PEOPLE PER UNIT.** 11 regions
    for 3.7 million is about 334,000 each — finer per person than Russia's 79 federal
    subjects at 1.8 million, which is the right comparison and the one that settles it.
    Nothing finer exists: Geostat's census database publishes municipalities for marital
    status and not for religion (sources/ge.md).
    """
    from ge2014 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "ge.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    df = df[df["geo_level"] == "region"].copy()
    df["count"] = df["count"].astype(int)
    if df["geo_id"].nunique() != 11:
        raise SystemExit(f"{df['geo_id'].nunique()} regions, expected 11 -- re-run "
                         "sources/ge.py")

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df = df.rename(columns={"geo_id": "unit"})
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations"]]


ENTRY = {
    "ge": dict(
        name="Georgia",
        source="2014 General Population Census (Geostat)",
        basis="self-identification",
        view=[40.0, 41.0, 46.8, 43.6],
        note_public=(
            "**Georgia is 83% Orthodox and the interesting quarter of it is the south.** "
            "The Georgian Orthodox Church is one of the oldest autocephalies anywhere — the "
            "country converted in the 320s — and across the western and central regions it "
            "runs to 99%. The map is drawn on eleven regions, so read it as regional "
            "composition rather than as neighbourhoods. "
            "**There are two entirely separate Muslim populations and the census puts them "
            "in one box.** In **Adjara**, on the Black Sea, 39.8% are Muslim and they are "
            "**Georgian-speaking Sunnis** — converted under three centuries of Ottoman rule "
            "and Georgian in language, name and everything else. In **Kvemo Kartli** on the "
            "Azerbaijani border, 43.0% are Muslim and they are **Azerbaijanis, largely "
            "Shia**; lowland Kakheti has the same population at 12.1%. Nothing in the "
            "source separates them, so both draw the same colour, and that is a limit of "
            "the census rather than of the country. "
            "**Samtskhe-Javakheti in the south is a different country religiously.** 39.9% "
            "Armenian Apostolic — the Armenian-majority districts of Akhalkalaki and "
            "Ninotsminda — and **9.4% Catholic, which is 78% of all the Catholics in "
            "Georgia**, the Armenian Catholics of Akhaltsikhe. Orthodoxy is a minority "
            "there at 45%. "
            "**And Tbilisi holds almost all of Georgia's Yazidis** — 8,124 of 8,591, 95% of "
            "them in one city. Kurmanji-speaking, descended largely from refugees of the "
            "Ottoman persecutions of the 1910s and 1920s, and one of the few Yazidi "
            "communities anywhere with a purpose-built temple outside Iraq. "
            "**Only 0.5% report no religion**, which is remarkable for a country that spent "
            "seventy years in the Soviet Union — and lower than anywhere on this map except "
            "Kosovo. Adjara is the exception at 2.8%. "
            "**Two parts of Georgia are not on this map at all.** The census could not "
            "enumerate Abkhazia or the Tskhinvali region (South Ossetia), so both are blank "
            "here — not empty, uncounted. About 1.2% of the people who were counted "
            "declined the question or left it blank and are not drawn either."),
        how="census, 2014",
        grain="regions, 334,000 people on average",
        # `gap` — one line under the country's name in the viewer (§6.12). See index.html's
        # note beside `gapNote`: a blank on a dot map cannot distinguish "nobody here is
        # religious" from "nobody counted here", and this is the only place that difference
        # is stated where the blank is actually on screen. A few words, never a sentence
        # that wants a second line.
        gap="the 1.2% who gave no answer or refused one; and Abkhazia and South Ossetia",
        gap_share=0.01182,
        counts=_ge_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "ge" / "ge_grid_400m.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_ge_place_weight,
        note="**THE TABLE IS ON A HOST NOTHING LINKS TO.** Geostat's census pages publish "
             "religion as a single 33 KB `.xls` at region level and that is all the earlier "
             "scouting found (§11k). The same table is also in a live PxWeb: "
             "`census.geostat.ge` is 404 and was never archived, `api.geostat.ge` answers "
             "200 with an empty body, and `pc-axis.geostat.ge` serves the **default IIS "
             "splash page** at its root — but `pc-axis.geostat.ge/PXWeb/api/v1/en/` is a "
             "working catalogue with a whole `Population Census 2014` database in it. **A "
             "default IIS page is not a dead host; it is a host with nothing mounted at "
             "`/`.** "
             "**Three departures from the modern PxWeb contract, on one server.** Its root "
             "returns `dbid` rather than `id` — the third instance after Kosovo and Moldova "
             "(§11k) — so a generic walker reads it as empty; an empty `{\"query\": []}` "
             "POST 404s where every other PxWeb here accepts it; and `json-stat2` 404s "
             "while `json-stat` v1 works. Each alone looks like 'the table is not there'. "
             "**11 REGIONS, AND THE COARSENESS IS THE POINT OF ARGUMENT.** Anita's test, "
             "and it is the right one: 334,000 people per unit is finer per person than "
             "Russia's 79 federal subjects at 1.8 million, and Russia is drawn. Judge a "
             "geography against the rest of the map, not by unit count. Nothing finer "
             "exists — Geostat's own census database publishes municipalities for marital "
             "status and only regions for religion. "
             "**THE SUPPRESSION IS BOUNDED AND THE CODELIST GIVES THE BOUND AWAY.** A "
             "thirteenth `region` value is not a region: it is the string `… is less or "
             "equal to 10`, the legend for the withheld cells, sitting inside the dimension. "
             "So `sources/ge.py` can assert something stronger than usual — every category's "
             "(national − sum of regions) must be at most ten times its number of withheld "
             "cells. It holds on all twelve: **7 withheld cells and 22 people unaccounted "
             "for in 3.7 million.** "
             "**`None` IS A CATEGORY NAME AND PANDAS READS IT AS NaN** — §9m's trap in its "
             "third country. Georgia's irreligious answer is literally `None`, so a default "
             "`read_csv` deletes 19,080 people without a word. "
             "**Abkhazia and the Tskhinvali region were not enumerated, and they are "
             "handled differently because the boundary files handle them differently.** "
             "Abkhazia is its own geoBoundaries ADM1 with no census row, so it is dropped "
             "from the units layer and simply draws nothing. South Ossetia has no ADM1 of "
             "its own — its municipalities sit inside Shida Kartli (Java) and "
             "Mtskheta-Mtianeti (Akhalgori) — so those two ADM2 polygons are subtracted "
             "from the placement grid, and no dot lands on ground the census did not count. "
             "That correction turns out small (6,093 modelled people, because Kontur's "
             "Georgian extract barely covers South Ossetia) and is worth making anyway. "
             "**The city/ring pair recurs, as §9q said it would.** Kontur/census is 0.85x "
             "for Tbilisi and **1.66x for Mtskheta-Mtianeti, the region wrapped around it** "
             "— geoBoundaries' Tbilisi polygon is 249 km² against the city's ~500, so outer "
             "Tbilisi sits in its ring region here. A fourth post-Soviet country with the "
             "same artefact. It is a placement fact and not a count fact: every region's "
             "dot total still comes from the census. Asserted on the other ten, reported on "
             "the ring. "
             "Placement is Kontur's 400 m H3 grid, 25,189 hexes — eleven regions over "
             "61,000 enumerated km² is 5,500 km² each, and Georgia is two mountain ranges "
             "with the people in the valleys.",
    ),
}
