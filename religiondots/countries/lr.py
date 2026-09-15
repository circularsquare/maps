# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _lr_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read.

    Fifteen counties over 95,845 km2 of land is 6,390 km2 a unit, coarser than any other
    country drawn from a barometer here, and Liberia's people are nowhere near evenly spread
    inside them. **Montserrado is 36.6% of the country on 1.9% of its land and 92% urban**:
    drawn flat, Greater Monrovia's colour would land on the Bong Range foothills. Gbarpolu,
    Grand Gedeh and Sinoe are the opposite, about 10,000 km2 apiece holding under 220,000
    people each, strung along a road or a river with rainforest between (sources/lr_grid.py).
    """
    return _kontur_place_weight(place, "lr_hexes.gpkg", "sources/lr_grid.py")


def _lr_counts():
    """LISGIS 2022 census on an Afrobarometer county pattern: 5 categories, 15 counties, and
    EVERY ROW IS `modelled` IN §7.

    TWO EXACT CENSUS MARGINS AND A SURVEY BETWEEN THEM. Table A4 gives each county's
    population and Table A13 gives each religion's national total; both sum to 5,250,187, the
    2022 census night count, so the county-by-religion table is fitted to them by iterative
    proportional fitting and the survey supplies only the interaction. No magnitude is
    invented (§14.4 rule 1): every person drawn is a person the census counted, in a county
    the census counted them in, under a religion label the census printed a total for.

    IT IS STILL `modelled` AND NOT `derived`. §7b's rule is whether anybody was counted, and
    nobody counted the cell -- LISGIS has never published religion below the national line in
    any census. What the census pins is the two margins, which is why the national figures on
    this map are a full enumeration and the county figures are not.

    THE DENOMINATIONS ARE VISIBLE IN THE SURVEY AND ARE NOT DRAWN. Afrobarometer's card names
    twenty-odd Christian bodies and Liberia's 7,163 pooled respondents fill them, but the
    share who answer `Christian only` instead of naming one runs from 23.1% to 72.0% between
    rounds. That is the fieldwork's probing depth, not Liberia, so the answers are grouped up
    to the census's own five before anything is placed. sources/lr.py has the argument.

    TWO OF THE FIVE CARRY THEIR OWN GEOGRAPHY. Christian (+0.789) and Muslim (+0.756) clear
    the split-half bar of +0.524 on fifteen counties; No religion is tested and fails at
    +0.114, and Traditional and Other are under the 1% eligibility floor. So irreligion,
    traditional religion and the residual are drawn at the national rate in every county.

    RIVERCESS IS DRAWN WITH NO MUSLIMS. None of its 142 pooled respondents was one, and §3.5
    drops rather than invents; on that sample the true share could be anything up to about 2%.
    """
    from lr2022 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "lr.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])

    lut = pd.read_csv(HERE / "data" / "geo" / "lr" / "lr_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"lr.csv counties with no polygon: {missing} -- re-run "
                         "sources/lr_geo.py, the lookup is stale")
    if df["unit"].nunique() != 15:
        raise SystemExit(f"{df['unit'].nunique()} counties, expected 15")

    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(df.loc[df["node"].isna(), "source_category"].unique())
    if unmapped:
        raise SystemExit(f"lr.csv categories with no node: {unmapped}")
    df = df[df["count"] > 0]
    # EVERY row, without exception -- there is no measured tier in this country (§7).
    df["tier"] = "modelled"
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations", "tier"]]


ENTRY = {
    "lr": dict(
        name="Liberia",
        source="2022 Population and Housing Census (LISGIS), with the county pattern from "
               "six pooled rounds of the Afrobarometer, 2008 to 2022",
        basis="self-identification, whole census population",
        note_public=(
            "**Liberia's census asks about religion and publishes the answer only for the "
            "country as a whole.** The 2022 census counted **5,250,187** people and printed "
            "one religion table: five categories, national, split by sex and nothing else. "
            "The same report gives residence, household size, education, literacy and "
            "employment by county. The 2008 census did the same thing, and so did the "
            "analytical monographs that followed it. So the county pattern on this map comes "
            "from somewhere else, six rounds of the Afrobarometer pooled, **7,163** people "
            "interviewed between December 2008 and September 2022. What the census fixes is "
            "the two margins, each county's population and each religion's national total, "
            "both to the person; the survey decides only how a county's people divide "
            "between the five columns. The dots are drawn desaturated because nobody counted "
            "that. "
            "**Islam is 11.98% of Liberia and nearly all of it is in the north-west.** Grand "
            "Cape Mount is **78.4%** Muslim and Bomi **56.5%**, with Gbarpolu at 24.1% and "
            "Lofa at 23.3% behind them, against **1.5%** of Nimba and 2.3% of Grand Bassa. "
            "That is the Vai and Mandingo country along the Sierra Leone and Guinea borders, "
            "one contiguous block rather than a scatter. The largest number of Muslims is in "
            "Montserrado all the same, **223,400** of them, because Montserrado holds more "
            "than a third of the country. "
            "**There is one published Liberian statement about religion by county, and it "
            "says the same thing.** LISGIS's 2011 Census Atlas mapped the urban population "
            "of the 2008 census and reported that Cape Mount was the only county with a "
            "Muslim majority and the other fourteen were Christian-majority. Pooled over "
            "rounds running a decade later, the survey makes exactly one county "
            "Muslim-majority and it is that one. That is the whole of the outside evidence "
            "for this map's geography. "
            "**River Cess is drawn with no Muslims at all**, because none of the 142 people "
            "interviewed there across six rounds was one. On a sample that size the true "
            "share could be anything up to about 2%, so read it as a county where the survey "
            "found none rather than as a county with none. "
            "**Christianity is a single colour here because no Liberian census has ever "
            "split it.** The 2022 questionnaire defines its Christian box as all Christian "
            "denomination churches and stops there, and so did 2008. The Afrobarometer does "
            "name denominations, and Liberians fill twenty of them, but the share who name "
            "one rather than answering Christian runs from **23.1%** to **72.0%** between "
            "rounds. That is a fact about how hard each round's fieldwork probed and not "
            "about Liberia, so the split is not drawn. "
            "**Traditional religion is 0.48% here and is a floor rather than a count.** The "
            "census offers it as an alternative to Christianity and Islam, so it cannot see "
            "anyone who is both, and the Poro and Sande societies of the interior are not an "
            "alternative to church membership. The 2008 census counted **20,134** "
            "traditionalists in a smaller Liberia, 0.58%, so the two enumerations agree with "
            "each other; what neither of them is measuring is practice. "
            "**Irreligion is drawn at the national rate, near 2.56% everywhere.** The survey "
            "cannot show that it has a geography: ranked across the fifteen counties, the "
            "earlier rounds and the later rounds agree at **+0.11** against a bar of +0.52, "
            "so the map does not give it one. The little spread that remains, **2.0%** in "
            "Grand Cape Mount against 2.7% in Nimba, is a residue of fitting the table to the "
            "census totals rather than anything the survey found."),
        how="census totals, 2022, given a county pattern by a pooled survey",
        grain="counties, 350,000 people on average",
        counts=_lr_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "lr" / "lr_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_lr_place_weight,
        note="LISGIS HAS NEVER PUBLISHED RELIGION BELOW THE NATIONAL LINE. The 2022 census "
             "prints it as Table A13, five categories, national, split by sex; the 2008 "
             "census prints it nationally in the Final Report and nationally plus "
             "urban/rural in the 2012 analytical monographs; the 2011 Census Atlas has one "
             "county map, of the URBAN population only, and no table behind it. UNSD table "
             "28 holds the 2008 national row and nothing finer. The fifteen 2022 thematic "
             "reports, released August 2024, have no religion volume. sources/lr.md lists "
             "what was searched and where each route ended. "
             "THE BUILD IS AN IPF ON TWO EXACT CENSUS MARGINS. Table A4 gives the county "
             "populations and Table A13 the national religion totals; both sum to 5,250,187, "
             "so the county-by-religion table is fitted to them and the Afrobarometer "
             "supplies only the interaction. This is NOT §6's rejected IPF, which forced a "
             "2000 composition onto 2010 totals: both margins here are the same census, the "
             "same night, from the same report. "
             "THE POPULATION IS THE CENSUS AND NOT COD-PS, uniquely because the religion "
             "margin comes from the same table set. Substituting a projection for one margin "
             "would make the two disagree by whatever it had drifted, and the fitted table "
             "would absorb that drift as though it were religion. "
             "THE DENOMINATIONS ARE IN THE SURVEY AND ARE NOT DRAWN. Afrobarometer's card "
             "names twenty-odd Christian bodies and Liberia's pooled respondents give "
             "Pentecostal 9.6%, Methodist 7.0%, Lutheran 5.8%, Baptist 5.3% and Roman "
             "Catholic 4.3%, which no Liberian census has ever published. It is not taken "
             "because the share answering `Christian only` instead of naming a denomination "
             "runs 28.0%, 44.4%, 23.1%, 67.4%, 72.0%, 61.6% across rounds 4 to 9, which is "
             "the fieldwork's probing depth rather than the country. Grouping up to the "
             "census's own five removes the effect entirely, because a Methodist and a "
             "`Christian only` are both Christian in every round. sources/lr.py asserts the "
             "swing still exists on every build, so a future release that fixes the probing "
             "fails the build rather than quietly leaving this decision in place. "
             "THE COUNTY LABELS ARE [[reference_pooled_survey_labels]]: 33 REGION strings "
             "over six rounds for fifteen counties, with Rivercess spelled two ways, three "
             "counties losing their `Grand` in round 8 and the whole set upper-cased in "
             "round 9. Harmonised in sources/lr.py's NORM before any test touches them. "
             "THE DECODE IS PINNED at r = +0.996 between the survey's county shares of "
             "respondents and the census's county populations, which none of 20,000 random "
             "pairings reaches. "
             "AFROBAROMETER READS 0.81x THE CENSUS ON MUSLIMS AND 0.48x ON IRRELIGION, and "
             "the fit corrects both. That offset is why the margins are worth having: drawn "
             "on the survey's own levels Liberia would be 9.7% Muslim against a census that "
             "counted 11.98%.",
    ),
}
