# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _ls_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read.

    10 districts over 30,500 km2, 1,800 to 4,900 km2 each, about 2,100 Kontur hexes a district.
    Kontur has lost three quarters of Berea, evenly (sources/ls_geo.py); each district's dots are
    its census count, so that moves nobody between districts. No Kontur block reaches the density
    cap (`kontur_cap.py ls`, 2026-10-03).
    """
    return _kontur_place_weight(place, "ls_hexes.gpkg", "sources/ls_geo.py")


def _ls_counts():
    """Six pooled Afrobarometer rounds on the 2016 census's district counts: 11 nodes, 10
    districts, and EVERY ROW IS `modelled` IN §7.

    LESOTHO'S CENSUS DOES NOT ASK RELIGION (2016 dictionary; IPUMS 1996 and 2006), so as in
    Namibia and Madagascar there is no margin to fit and no measured tier. sources/ls.py has the
    construction: each district at its own pooled mix for the five churches that pass the
    split-half, the rest at the national proportion of what they leave, the two DHS reports as
    the level witness.
    """
    from ls2016 import resolve

    # keep_default_na=False: pandas reads the category `None` as missing otherwise (§11aq).
    df = pd.read_csv(HERE / "data" / "normalized" / "ls.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    lut = pd.read_csv(HERE / "data" / "geo" / "ls" / "ls_lookup.csv", dtype=str,
                      keep_default_na=False)
    if set(df["geo_id"]) != set(lut["geo_id"]) or len(lut) != 10:
        raise SystemExit("ls.csv districts do not match ls_lookup.csv's 10 -- re-run "
                         "sources/ls_geo.py and sources/ls.py")
    if "None" not in set(df["source_category"]):
        raise SystemExit("ls.csv has no `None` rows; it was read with the default NA list")
    df["unit"] = df["geo_id"]
    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(df.loc[df["node"].isna(), "source_category"].unique())
    if unmapped:
        raise SystemExit(f"ls.csv categories with no node: {unmapped}")
    df = df[df["count"] > 0]
    # EVERY row, without exception -- there is no measured tier in this country (§7).
    df["tier"] = "modelled"
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations", "tier"]]


ENTRY = {
    "ls": dict(
        name="Lesotho",
        source="six pooled rounds of the Afrobarometer, 2008 to 2022, on the district counts of "
               "the 2016 Population and Housing Census (Bureau of Statistics)",
        basis="self-identification, whole census population",
        note_public=(
            "**Lesotho's census does not ask about religion.** Neither the 2016 census nor the "
            "two before it had a religion question, and the 2026 census has not published yet, "
            "so this map's Lesotho comes from a survey. Six rounds of the Afrobarometer are "
            "pooled, **7,102** adults interviewed between October 2008 and March 2022, and "
            "nobody counted these dots, so they disappear when inferred dots are turned off. Each "
            "district is drawn at its "
            "2016 census population. "
            "**The churches hold steady from round to round.** Catholics are 39 to 44% in every "
            "round and come out at **41.2%**; the Demographic and Health Surveys, which asked "
            "women and men aged 15 to 49, found 39.3% in 2014 and 35.8% in 2023-24. The Lesotho "
            "Evangelical Church is drawn at 19.6% (the same surveys found 17.3% and 15.3%), "
            "Anglicans at 8.8% (7.4% and 6.3%). In the 2022 round the survey's card offered "
            "Calvinist beside Evangelical, and the Evangelical Church's members split between the "
            "two, so both answers are counted as that church. "
            "**Some churches have their own districts and some do not.** Catholics are 58% of "
            "Thaba-Tseka and a quarter of Butha-Buthe. Zionist, apostolic and other independent "
            "churches are 26% of Butha-Buthe and 6% of Maseru; Anglicans are strongest in "
            "Qacha's Nek and Leribe. The Evangelical Church's share does not differ between "
            "districts by more than chance would give, so it is drawn at one proportion "
            "everywhere, as are no religion, other Christians and the small religions. "
            "**One round's answers in the north are left out.** In 2014, 10 to 25% of "
            "respondents in Butha-Buthe, Leribe and Berea gave traditional religion, against 2% "
            "or less there in every other round; those 62 answers are not used."),
        how="a pooled survey, 2008 to 2022, on 2016 census district populations",
        grain="districts, 201,000 people on average",
        counts=_ls_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "ls" / "ls_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_ls_place_weight,
        note="LESOTHO'S CENSUS DOES NOT ASK: the 2016 dictionary and IPUMS 1996 and 2006 have no "
             "religion item (sources.md §11aq); freed by the 2026-10-03 negatives scout. Drawn on "
             "Anita's Nigeria ruling (ask/answered/010-ng): each district at its own pooled survey "
             "mix, every row modelled, the national level computed. Catholic, Anglican, Methodist, "
             "Pentecostal and Zionist/independent placed; the LEC (Evangelical + Calvinist, one "
             "box) fails the split-half and is in the residual with the rest. Round 6's 62 "
             "traditional answers in the three northern districts dropped. DHS 2014 and 2023-24 "
             "witness the church levels. sources/ls.md has the record.",
    ),
}
