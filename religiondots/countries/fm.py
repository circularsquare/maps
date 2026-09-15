# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _fm_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read.

    A Micronesian unit is mostly ocean. Chuuk is drawn whole and reaches from Houk in the
    Western Islands to Ta in the Lower Mortlocks, about 1,000 km, so an area weight would put
    a third of the country in open water instead of on Weno; Yap's outer municipalities are
    single atolls a kilometre or two across (sources/fm_grid.py).
    """
    return _kontur_place_weight(place, "fm_hexes.gpkg", "sources/fm_grid.py")


def _fm_counts():
    """FSM 2023 census Table B6: 11 categories on 33 units at TWO tiers.

    NOTHING IS ALLOCATED, SPREAD OR MODELLED -- every row is `measured` at the unit it is
    drawn on, which is the municipality in Yap and Pohnpei and the state in Chuuk and Kosrae.
    That mix is spec 12's Ghana answer: draw the fine unit where it reconciles and the coarse
    one where it does not, rather than dropping the whole country to its worst tier.

    THE REASON IT IS MIXED IS THE SOURCE AND NOT A JUDGEMENT. FSM publishes a per-state
    basic-tables workbook beside the national one; Yap's and Pohnpei's carry `Table B6.
    Religion by Municipality`, Chuuk's workbook does not exist and Kosrae's stops at Table B5.
    sources/fm.py asserts Kosrae's missing table on every run, so if it ever appears the build
    fails and says to draw Kosrae finely.

    THE ONE CHECK THAT CROSSES TWO PUBLICATIONS: each per-state workbook's own state column
    reproduces the national table's column for that state on all eleven categories exactly.
    """
    from fm2023 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "fm.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])

    lut = pd.read_csv(HERE / "data" / "geo" / "fm" / "fm_lookup.csv", dtype=str)
    known = set(lut["geo_id"])
    df["unit"] = df["geo_id"].where(df["geo_id"].isin(known))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"fm.csv units with no polygon: {missing} -- re-run "
                         "sources/fm_geo.py, the lookup is stale")
    if df["unit"].nunique() != 33:
        raise SystemExit(f"{df['unit'].nunique()} units, expected 33")

    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(df.loc[df["node"].isna(), "source_category"].unique())
    if unmapped:
        raise SystemExit(f"fm categories with no node: {unmapped}")
    df = df[df["count"] > 0]
    df = df.groupby(["unit", "node"], as_index=False)["count"].sum()
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations"]]


ENTRY = {
    "fm": dict(
        name="Micronesia",
        source="2023 Population and Housing Census, Table B6 (FSM Statistics)",
        basis="self-identification, whole enumerated population",
        note_public=(
            "**The census draws one mission boundary and very little else.** Roman "
            "Catholics are **55.5%** of Micronesia and the Congregational and other "
            "Protestant churches **37.1%**, and the two invert completely across the "
            "2,800 km from Yap to Kosrae. Kosrae is **88.7%** Congregational and 1.7% "
            "Catholic; Yap at the other end is **79.6%** Catholic and 4.6% "
            "Congregational. Twelve of Yap's twenty municipalities record no "
            "Congregational answer at all and six of those are entirely Catholic, while "
            "four of Pohnpei's outer atolls record no Catholic one and are entirely "
            "Congregational. The American Board's Congregational missionaries reached "
            "Kosrae and Pohnpei in 1852 and Chuuk in 1879 and the Catholic missions came "
            "from the west through the Spanish and then German Carolines; where they met "
            "is what the map still shows. "
            "**Just under half the country is drawn municipality by municipality, and "
            "Chuuk is not.** Yap and Pohnpei each publish religion by municipality in "
            "their own 2023 workbooks, so they are drawn on 20 and 11 units of about "
            "1,200 people. Chuuk's workbook does not exist and Kosrae's has no religion "
            "table, so each of those is a single unit, and **44.8%** of everyone on this "
            "map is inside one Chuuk polygon drawn at 54.8% Catholic and 41.8% "
            "Congregational throughout. That is where the census stops rather than a "
            "choice made here, but it does mean the picture is much sharper at the two "
            "ends of the country than in the middle. "
            "**The small churches are island-sized, and only the finer half shows it.** "
            "Fais, an outer island of 463 people in Yap, is **29.4%** Assembly of God "
            "and **28.5%** Baptist against national shares of 0.7% and 1.2%. Sapwuahfik "
            "in Pohnpei is **20.3%** Seventh-day Adventist against 0.6%, and Sokehs "
            "holds 282 of the country's 463 Apostolic. At state level every one of those "
            "rounds away to about one percent. "
            "**Saying you have no religion is a Yapese answer.** The 2023 census merges "
            "no religion with refusal into one cell worth 0.7% of the country, but it is "
            "3.7% of Yap against 0.1% of Chuuk, and inside Yap it reaches **28.9%** of "
            "Kanifay and 33.3% of Rumung, a municipality of 36 people. The 2010 census "
            "asked the two apart, printing 723 with no religion against 72 who refused, "
            "and found the same concentration in Yap, so this is people saying they have "
            "none rather than people declining to answer. "
            "**Micronesia has lost a quarter of its people since the previous census.** "
            "The same table counted 102,843 in 2010 and 75,817 in 2023, and emigration "
            "to the United States under the Compact of Free Association is most of the "
            "difference. The one outside reading of any of this is the UN Statistics "
            "Division's, which carries the 2000 census on eight categories rather than "
            "eleven; fold these figures down to that set and every share sits within "
            "**three points** of it, with Catholics up and Congregationalists down by "
            "about the same amount over the twenty-three years."),
        how="census, 2023, whole enumerated population",
        grain="municipalities in Yap and Pohnpei (1,200 people); whole states in Chuuk "
              "and Kosrae",
        gap="0.3% of the country, 241 people in cells too small for the census to print "
            "at municipality level; it falls on the smallest churches, taking 17.7% of "
            "Yap's and Pohnpei's Adventists and none of their Catholics",
        gap_share=0.00318,
        counts=_fm_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "fm" / "fm_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_fm_place_weight,
        note="THE TIER IS MIXED AND THAT IS THE COUNTRY'S BIGGEST CALL. FSM publishes a "
             "per-state basic-tables workbook beside the national one. Yap's and Pohnpei's "
             "carry `Table B6. Religion by Municipality`; CHUUK'S WORKBOOK DOES NOT EXIST "
             "and KOSRAE'S STOPS AT TABLE B5. So the map is 20 Yap municipalities + 11 "
             "Pohnpei municipalities + Chuuk whole + Kosrae whole = 33 units, 48.4% of the "
             "people at ~1,200 each and 51.6% at ~19,500. This is spec 12's Ghana answer "
             "stated exactly -- draw the fine unit where it reconciles and the coarse one "
             "where it does not, so the drawn tier is two geo_levels and every drawn row is "
             "still `measured`. It is NOT Albania's trade (sources/al.md 3a): the finer half "
             "here is the same census year, the same office and the same eleven categories, "
             "so it costs no vintage and no category. sources/fm.md 4 prices the alternative. "
             "KOSRAE'S MISSING TABLE IS A TRIPWIRE, NOT AN ASSUMPTION. sources/fm.py reads "
             "Kosrae's 2023 workbook on every run and raises if a Table B6 ever appears in "
             "it, naming the four municipalities to draw. Chuuk's absence cannot be checked "
             "from a file that does not exist and rests on the `id=0` library sweep instead. "
             "THE OFFICE MOVED AND THE QUEUE'S HOST IS GONE. fsmstatistics.fm returns HTTP "
             "200 and serves a LiteSpeed directory index holding cgi-bin, dasdas.png and a "
             "stray htaccess; its Wayback captures hold no census tables at all, because the "
             "old site served them through wpDataTables and the archive kept only the shell. "
             "The office is stats.gov.fm, WP File Download, 71 files over nine pages of ten "
             "at task=files.getFiles&id=0 -- the sixth Pacific office here on that plugin. "
             "TWO PUBLICATIONS AGREE EXACTLY, WHICH IS THE STRONGEST CHECK THIS COUNTRY HAS. "
             "Each per-state workbook's own state column reproduces the national table's "
             "column for that state on all eleven categories with zero difference. A whole "
             "column read into the wrong place survives every within-table margin and dies "
             "there. THE OUTSIDE WITNESS EXISTS AND THE RECORD SAID IT DID NOT (review, "
             "2026-09-08). UNSD table 28 DOES carry `Micronesia (Federated States of)` 2000, "
             "8 categories, 107,008, an exact partition. `tools/oracle.py` takes UNSD'S OWN "
             "NAME and not a country code, so `oracle.py fm` returns the ABSENT line for "
             "every cc ever passed to it; the build read that as a negative. Folded to the "
             "2000 category set every share agrees within 3 points, and 2000 prints "
             "`No Religion` 753 against `Refused to answer` 57, a SECOND independent "
             "confirmation of the merged-cell call the 2010 census settled at 90.9%. "
             "THE PUBLISHER PERTURBS ITS OWN MARGINS BY ONE OR TWO. The four printed state "
             "totals sum to 75,818 against a printed national 75,817; Table B1's age rows "
             "give a third set, each within one; Pohnpei's eleven municipalities sum to "
             "26,104 against its own printed 26,102. MARGIN_SLACK absorbs it and every run "
             "prints it. An equality assert would refuse a correct read. "
             "`*` MEANS SUPPRESSED OR ZERO AND IT BITES THE SMALL CHURCHES. 58 of Yap "
             "proper's 132 cells and 73 of Pohnpei's are starred. impute() fills only a cell "
             "that is the single unknown in its line and never apportions, which leaves 241 "
             "people, 0.32%. Per category the loss is 17.7% of Adventists and 12.3% of "
             "Witnesses against 0.0% of Roman Catholics, so the fine tier understates the "
             "minorities and the gap line says so. Per UNIT there is NO lean: the residual "
             "share correlates +0.47 with the irreligion share over 31 units, which sounds "
             "like something and is one municipality -- drop Kanifay and it is -0.01 "
             "(permutation p=0.046 with it, [[reference_check_needs_power]]). "
             "THE JOIN IS BY NAME INSIDE THE STATE AND THE P-CODE TESTS IT AFTER (Benin's "
             "rule). COD-AB's FM101..FM120 and FM301..FM311 run in EXACTLY the census's own "
             "print order, so a name that matched the wrong municipality would move a rank; "
             "31/31 join, two via an alias (COD writes Mwokilloa and Sapwuafik for the "
             "census's Mwoakilloa and Sapwuahfik). And COD's MAIN_OUTER flag independently "
             "reproduces the workbook's own YAP PROPER / OUTER ISLANDS split, 10 and 10. "
             "KONTUR READS 1.50x THE CENSUS AND THAT IS THE COUNTRY, NOT THE GRID. Its "
             "2023-11 extract has 113,340 people, which is the 2010 level; FSM lost nearly a "
             "third of its people in between. The grid is only ever a within-unit weight and "
             "is normalised by that ratio, and the per-unit band is what actually tests it: "
             "all 30 non-synthetic units inside a factor of 5 (0.39 Gilman to 1.95 Elato), "
             "r=0.9760 against a best of 0.5465 over 2,000 random pairings, 0 reaching it. "
             "30% of Kontur's people fall outside every unit and are SNAPPED, not dropped, "
             "on Vanuatu's rule -- an atoll is a strip of land a few hundred metres wide. "
             "Faraulep, Ifalik and Ngulu are smaller than one 400 m hex and get their own "
             "polygon as a single cell.",
    ),
}
