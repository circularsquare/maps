# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _mg_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read.

    22 units over 590,000 km2 is about 27,000 km2 a unit. Melaky is 41,000 km2 with 309,000
    people, and most regions hold their people in a few places: the highlands around
    Antananarivo, Antsirabe and Fianarantsoa, the east coast, the north-western valleys
    (sources/mg_grid.py). Seven Kontur blocks reach the density cap; `kontur_cap.csv` has the
    three off-centre ones capped (Mahajanga, Antsiranana, Toliara).
    """
    return _kontur_place_weight(place, "mg_hexes.gpkg", "sources/mg_grid.py")


def _mg_counts():
    """Four pooled Afrobarometer rounds on the 2018 census's region counts: 12 nodes, 22 units,
    and EVERY ROW IS `modelled` IN §7.

    MADAGASCAR'S CENSUS DOES NOT ASK RELIGION (neither the 1993 nor the 2018 form), so as in
    Nigeria and Tanzania there is no margin to fit and no measured tier. Each unit is drawn at the
    mix its own respondents gave and at its census population. sources/mg.py has the construction:
    the three large churches kept apart, None and traditional religion placed as one box and split
    at one national ratio, round 4 left out, and Vatovavy and Fitovinany drawn as the census's one
    region.
    """
    from mg2018 import resolve

    # keep_default_na=False: pandas reads the category `None` as missing otherwise (§11aq).
    df = pd.read_csv(HERE / "data" / "normalized" / "mg.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    lut = pd.read_csv(HERE / "data" / "geo" / "mg" / "mg_lookup.csv", dtype=str,
                      keep_default_na=False)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"mg.csv units with no polygon: {missing} -- re-run "
                         "sources/mg_geo.py, the lookup is stale")
    if df["unit"].nunique() != 22:
        raise SystemExit(f"{df['unit'].nunique()} units, expected 22")
    if "None" not in set(df["source_category"]):
        raise SystemExit("mg.csv has no `None` rows; it was read with the default NA list")

    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(df.loc[df["node"].isna(), "source_category"].unique())
    if unmapped:
        raise SystemExit(f"mg.csv categories with no node: {unmapped}")
    df = df[df["count"] > 0]
    # EVERY row, without exception -- there is no measured tier in this country (§7).
    df["tier"] = "modelled"
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations", "tier"]]


ENTRY = {
    "mg": dict(
        name="Madagascar",
        source="four pooled rounds of the Afrobarometer, 2013 to 2022, on the region counts of "
               "the 2018 census (Troisième Recensement Général de la Population et de "
               "l'Habitation, INSTAT)",
        basis="self-identification, whole census population",
        note_public=(
            "**Madagascar's census does not ask about religion.** Neither the 1993 nor the 2018 "
            "census form has the question, so this map's Madagascar comes from a survey. Four "
            "rounds of the Afrobarometer are pooled, **4,788** adults interviewed between March "
            "2013 and May 2022, and every dot is desaturated because nobody counted it. Each "
            "region is drawn at the mix its own respondents gave and at its 2018 census "
            "population. Vatovavy and Fitovinany, separate regions since 2021, are drawn as the "
            "one region the census counted. "
            "**The churches are drawn separately.** In most African countries this survey's "
            "church answers cannot be used, because the share who say only that they are "
            "Christian changes a lot from round to round. In Madagascar almost everyone names a "
            "church, and the three large ones stay about the same size in every round: Catholics "
            "at **37.8%**, the Church of Jesus Christ in Madagascar (FJKM) at **21.4%** and "
            "Lutherans at 13.9%. The Demographic and Health Surveys of 2008-09 and 2021 also find "
            "about as many Catholics as members of the FJKM, Lutheran and Anglican churches "
            "together. No source outside the Afrobarometer counts the FJKM and the Lutherans "
            "separately. "
            "**Each church has its own part of the island.** Catholics are 60.1% of Haute "
            "Matsiatra in the southern highlands, the FJKM 39.0% of Analamanga and 39.9% of Itasy "
            "around Antananarivo, and Lutherans 37.3% of Menabe and 35.2% of Atsimo Atsinanana in "
            "the south. Muslims are 14.7% of Diana and 7.8% of Melaky; the survey found none in "
            "ten regions, and those regions are drawn with none. "
            "**No religion is drawn at 12.7%**, and at a fifth or more of the people in most "
            "regions outside the central highlands: 39.4% of Sofia and 27.3% of Androy. In Itasy, Bongolava, Vakinankaratra, Amoron'i Mania and Haute "
            "Matsiatra the survey found no one of no religion or of traditional religion. The card "
            "offered traditional religion as a separate answer, but the two answers trade places "
            "between rounds: nationally traditional religion fell from 8.5% of answers in 2013 to "
            "1.3% in 2022 while no religion rose from 8.2% to 12.7%, and the regions that moved "
            "were the same regions. So the two are placed together and split at the ratio of the "
            "two most recent rounds, nine in ten no religion, which is also what both Demographic "
            "and Health Surveys find. Those surveys put no religion higher, at **20 to 25%** of "
            "people aged 15 to 49. "
            "**Anglicans, Pentecostals, Jehovah's Witnesses and other religions are not placed, "
            "because where they live does not repeat between halves of the survey.** They are spread in their national proportions over what the larger groups "
            "leave in each region."),
        how="a pooled survey, 2013 to 2022, on 2018 census region populations",
        grain="regions, 1.2 million people on average",
        counts=_mg_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "mg" / "mg_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_mg_place_weight,
        note="MADAGASCAR'S CENSUS DOES NOT ASK: the 1993 and 2018 forms have no religion item "
             "(sources.md §11aq) and Madagascar is ABSENT from the UNSD oracle. Drawn on Anita's "
             "Nigeria ruling (ask/answered/010-ng): the regional picture from a pooled survey, "
             "every row modelled, the national level computed rather than fitted. The churches "
             "are kept apart because `Christian only` is 0.3-2.3% by round, with both DHS "
             "surveys as the witness to the Catholic/Protestant level. NONE AND TRADITIONAL ARE "
             "PLACED AS ONE BOX (split-half +0.682) and split 90.5/9.5 at rounds 7 and 9's "
             "national ratio, because the two answers trade places between rounds; None is "
             "unaffiliated under step 2 of the draft no-religion procedure. ROUND 4 IS LEFT OUT "
             "(six old provinces, no Betsiboka district sampled). Vatovavy and Fitovinany are one "
             "unit. sources/mg.md has the record.",
    ),
}
