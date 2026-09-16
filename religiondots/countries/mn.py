# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


class _MnHexWeighter(_KeHexWeighter):
    """Split an aimag's dots across Kontur 400 m hexagons by hex POPULATION.

    **Mongolia is the extreme case of the problem Kenya's class was written for.** 2.1 people
    per km², a third of Kazakhstan's density; Ömnögovi is 165,000 km² holding about 70,000
    people, and Ulaanbaatar is 46% of the country. An equal share of dots per polygon would
    spread half of Mongolia's Buddhists evenly across the Gobi and the Altai, where there is
    nobody, and would under-draw the one city almost everyone lives in.

    Same caveat as Kenya's, and it matters more here because the units are so large: it is a
    POPULATION weight and not a religion one. Nothing measures where an aimag's shamanists
    sit inside it, so a shamanist dot and a Buddhist dot are spread identically. Read a
    cluster as "this aimag, drawn where Mongolians live".
    """

    def summary(self):
        return (f"{self.n_pop:,} (unit, node) rows placed on Kontur hex population, "
                f"{self.n_uniform:,} on equal shares where an aimag's hexes sum to zero "
                f"(sources/mn_grid.py)")


def _mn_place_weight(place):
    """countries.py hook. `place` is the hex layer scatter.py has read."""
    if "pop" not in place.columns:
        print("  !! mn_hexes.gpkg has no `pop` column — run sources/mn_grid.py; "
              "placing on equal shares (§8.2)")
        return None
    return _MnHexWeighter(place)


def _mn_counts():
    """NSO 2020 PHC: 6 drawn categories on 19 aimags and Ulaanbaatar's 9 düüregs (20 of 22 units).

    ULAANBAATAR IS DRAWN BY DÜÜREG since 2026-09-15: the capital's own volume charts religion
    for its nine düüregs (figures 3.5 and 3.6), which sources/mn_ub.py measures off the vector
    drawing and sources/mn.py checks against the city's printed tables. mn.csv carries no
    city-wide row, so nothing here can draw the capital twice. sources/mn.md §11.

    ONE level, no allocation, nothing modelled — every row is `measured` and may ring. The
    figures are RECONSTRUCTED from two published percentage tables rather than read as
    counts, because no Mongolian volume prints an absolute religion figure anywhere: the
    share religious, times the share of the religious holding each religion, times the
    aimag's own population aged 15 and over. sources/mn.py does the arithmetic and checks it
    both ways round; the national reconstruction lands within 0.15 points of NSO's published
    national figures on all six categories.

    THE UNIVERSE IS ADULTS AND IT IS NOT SCALED UP TO THE WHOLE POPULATION (Chile's rule,
    §3.5a and sources/cl.md 3). The question went to people aged 15 and over, who are
    2,067,841 of the 20 aimags drawn; Mongolia's religiosity rises steeply with age, from
    35.1% of 15-19 year olds in Ulaanbaatar to 74.5% of the over-70s, so a flat scale-up
    would assert something about children that the census went out of its way not to ask.

    TWO AIMAGS ARE ABSENT and it is a typesetting accident rather than a suppression:
    Darkhan-Uul and Dundgovi published their volumes with the tables as pasted-in IMAGES,
    so there is no text to read. Together they are 148,869 people. See sources/mn.md §7.
    """
    from mn2020 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "mn.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    df = df[df["geo_level"].isin(["aimag", "duureg"])].copy()

    # geo_id IS the polygon id: COD-AB's pcode is the census's own aimag code, so there is
    # no name join anywhere in this country. sources/mn_geo.py asserts that. The capital's
    # nine düüregs are COD's MN11xx, joined inside the city by name with the volume's table
    # order as witness (sources/mn_ub.py::check_join).
    df["unit"] = df["geo_id"]
    geo = HERE / "data" / "geo" / "mn"
    units = (set(gpd_read_units(geo / "mn_aimags.gpkg"))
             | set(gpd_read_units(geo / "mn_soums.gpkg")))
    missing = sorted(set(df["unit"]) - units)
    if missing:
        raise SystemExit(f"mn.csv units with no polygon: {missing} -- re-run "
                         "sources/mn_geo.py")
    if "MN11" in set(df["unit"]):
        raise SystemExit("mn.csv carries Ulaanbaatar city-wide as well as by düüreg")
    if df["unit"].nunique() != 28:
        raise SystemExit(f"{df['unit'].nunique()} units, expected 19 aimags and 9 düüregs")

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations"]]


def gpd_read_units(path, col="unit"):
    import geopandas as gpd
    return gpd.read_file(path)[col].astype(str).tolist()


ENTRY = {
    "mn": dict(
        name="Mongolia",
        source="2020 Population and Housing Census, aimag volumes (National Statistics "
               "Office of Mongolia)",
        basis="self-identification, population aged 15 and over",
        view=[87.5, 41.4, 120.0, 52.3],
        note_public=(
            "**The question went to one adult in ten, and most of what is odd here follows "
            "from that.** Mongolia's 2020 census asked *do you have a religion, and if so "
            "which* of people aged 15 and over who fell into a **10%** sample, so this is a "
            "census long form and not a full count. The office never prints how many people "
            "that was; a tenth of Mongolia's 2,170,573 adults is **about 217,000**, which "
            "is more respondents than any survey on this map has, and it is still a sample: a "
            "cell worth a few tenths of a per cent in a small aimag rests on a handful of "
            "answers, and the smallest unit drawn, Bagakhangai, has only about 280 people "
            "in the sample altogether. "
            "**Bayan-Olgii is the sharpest edge on the map for a thousand miles.** Islam is "
            "**92.5%** of its religious population, against **13.6%** next door in Khovd, "
            "the same **13.6%** in Nalaikh on the edge of Ulaanbaatar, and almost nothing "
            "anywhere else drawn. These are Mongolia's Kazakhs, who settled the far west in "
            "the nineteenth century and are still most of the aimag; the capital's census "
            "volume puts Nalaikh's share down to the many Kazakhs living there. Bayan-Olgii "
            "is also the second most religious unit drawn, **88.7%** against a national "
            "59.6%, behind Ovorkhangai's **89.6%**. "
            "**Ulaanbaatar, nearly half the adults drawn, is split into its nine districts.** "
            "The city's own census volume charts religion for each, from **58.2%** religious "
            "in Chingeltei to **47.1%** in Bagakhangai, against 53.7% for the city. The "
            "charts print percentages only, so each district's figures are measured off the "
            "chart and multiplied by its adults from the same volume's age table. "
            "**Shamanism here is not a survival or a curiosity, and it has a geography.** "
            "It is **10.9%** of the religious in Dornod and **7.1%** in Khovsgol, against "
            "**0.4%** in Ovorkhangai: the north and the east, the Buryat and Darkhad "
            "country, rather than the Buddhist heartland in the Khangai. Nationally it is "
            "2.4%, which is nearly twice the Christian share. "
            "**Forty per cent report no religion, and that number has a history.** The "
            "state suppressed religious practice from the 1920s, destroyed almost every "
            "monastery in the purges of 1937 and 1938, and did not permit open practice "
            "again until 1990. The share is not a settled figure; it was 38.6% at the 2010 "
            "census, and it rose in some aimags while falling in others. "
            "**Two of the twenty-two aimags are missing and it is an accident of "
            "typesetting.** Darkhan-Uul and Dundgovi issued their census volumes with the "
            "tables as scanned pictures rather than text, so their figures cannot be read "
            "out. They are 148,869 people between them, 4.7% of Mongolia, and nothing about "
            "them is suppressed or unpublished."),
        how="census long form, 2020, a 10% sample of adults",
        grain="aimags, and the nine districts of Ulaanbaatar; 74,000 adults on average",
        gap="children under 15; Darkhan-Uul and Dundgovi, whose volumes are scans",
        counts=_mn_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "mn" / "mn_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_mn_place_weight,
        note="NOT ONE ABSOLUTE FIGURE IS PUBLISHED and every count here is reconstructed. "
             "The twenty-two aimag volumes print religion only as percentages, in two "
             "chained tables: the 15+ population split into `Шүтдэггүй` and `Шүтдэг`, and "
             "then the religious population split five ways. A count is therefore the "
             "product of the two shares and the aimag's own population aged 15 and over, "
             "which comes from the national report's appendix table 1.1. Both published "
             "tables carry one decimal place, so a category's count is good to roughly a "
             "tenth of a per cent of the aimag's adults before the sampling error is even "
             "considered. The check that this is right is the national reconstruction: "
             "summed over the 20 aimags drawn it gives 51.9% Buddhist, 40.4% no religion, "
             "3.3% Muslim, 2.4% shamanist, 1.3% Christian and 0.7% other, against NSO's own "
             "published 51.7 / 40.6 / 3.2 / 2.5 / 1.3 / 0.7. "
             "THE UNIVERSE IS ADULTS AND IS NOT SCALED UP to the whole population (Chile's "
             "rule, §3.5a). 2,067,841 people aged 15 and over in the 20 aimags drawn, out "
             "of a resident population of 3,197,020 for all 22; religiosity climbs steeply "
             "with age here, 35.1% at 15-19 against 74.5% over 70 in Ulaanbaatar, so a flat "
             "scale-up would put a claim on children that the census declined to make. "
             "The dots are spread across 121,265 Kontur 400m hexagons weighted by hex "
             "population (sources/mn_grid.py), and Mongolia needs that more than any other "
             "country here: 2.1 people per km², with Ömnögovi at 165,000 km² for about "
             "70,000 people and Ulaanbaatar holding 46% of the country. Read a cluster as "
             "'this aimag, drawn where Mongolians live'. "
             "The join is by CODE and not by name: COD-AB's pcode is the census's own aimag "
             "code, which sources/mn_geo.py asserts against the census DDI on all seventeen "
             "codes the DDI prints. "
             "ULAANBAATAR IS DRAWN BY ITS NINE DÜÜREGS (2026-09-15). Its own volume charts "
             "religion by düüreg in figures 3.5 and 3.6 and nowhere else; both are vector "
             "drawings, so every share is a rectangle width at one scale fitted on the printed "
             "labels (every label within 0.03 points of its own segment), and the düüregs' "
             "adults come from the same volume's appendix table 1.1, which equals the national "
             "report's Ulaanbaatar row exactly. Weighted back up they give the city's printed "
             "53.7% religious and its five type shares within 0.07 points. The volume's prose "
             "calls Nalaikh's 13.6% Christian; the chart draws and labels it Islam, and the "
             "city check fails on the prose's reading. Hexes take their düüreg by centroid "
             "(sources/mn_grid.py::split_capital). sources/mn_ub.py, sources/mn.md §11.",
    ),
}
