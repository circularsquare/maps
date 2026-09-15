# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _sz_place_weight(place):
    """countries.py hook. `place` is the ~370 m WorldPop cell layer scatter.py has read.

    **THE SECOND COUNTRY HERE THAT IS NOT ON KONTUR, AND THE FIRST WHERE KONTUR IS SIMPLY
    WRONG RATHER THAN MISMATCHED.** Singapore came off it because the grid and the counts do
    not share a universe (§9bp); that grid was accurate and measured the wrong people. This
    one is a modelling failure on the grid's own terms. Kontur's
    Eswatini extract puts 43% of the country in Lubombo, which the census counts at 19%, and
    reads 0.38x on Hhohho against 2.24x on Lubombo. It is built from building footprints, so
    it inherits the northern Lowveld sugar estates that HOT mapped in detail and misses the
    dispersed Highveld homesteads where most Swazis live. WorldPop's Maxar-derived
    constrained raster reads 0.95 to 1.04 on the same four polygons, and a second WorldPop
    release of the census's own year agrees with it to within 0.04. sources/sz_grid.py has
    all three sets of numbers.

    It is needed for the ordinary §8.2 reason on top of that: four regions for a whole
    country, with Manzini and Hhohho at 87-89 people per km2 against Lubombo's 36, and
    Table 3.2.5 says the emptier regions are the ones whose composition is least typical.
    """
    return _kontur_place_weight(place, "sz_cells.gpkg", "sources/sz_grid.py")


def _sz_counts():
    """Eswatini CSO 2017 PHC Volume 3 at region: 20 drawn categories on 4 regions.

    TWO TIERS, AND THE SPLIT IS A PROPERTY OF THE SOURCE RATHER THAN A CHOICE. Table 3.2.4
    publishes the thirteen Christian denominations by region, so those rows are `measured`
    and may ring. Table 3.2.1's nine top-level religions are published nationally only, so
    the eight non-Christian ones are `derived`: each region's non-Christian total is its
    published population (Table 5.2.2) minus its published Christians, which is a counted
    magnitude, and only the split of that total across the eight columns comes from the
    national rates. §14.4 rule 1 -- no magnitude is estimated at a finer level than the
    source publishes it. 89.25% measured, 10.75% derived.

    NO `roll` AND taxonomy/sz2017.py SAYS WHY. spec §7a-i-1 rolls a derived row up to the
    node its own source column names at the drawn unit; the column these eight came out of
    is "everybody who is not a Christian", which is not a religion and has no node. An
    empty COLUMNS is the honest answer and tools/check_rollup.py will report these rows.
    """
    from sz2017 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "sz.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    df = df[df["geo_level"] == "region"].copy()

    lut = pd.read_csv(HERE / "data" / "geo" / "sz" / "sz_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"sz.csv regions with no polygon: {missing} -- re-run "
                         "sources/sz_geo.py, the lookup is stale")
    if df["unit"].nunique() != 4:
        raise SystemExit(f"{df['unit'].nunique()} regions, expected 4")

    df["tier"] = df["note"].str.contains("tier=derived", na=False).map(
        {True: "derived", False: "measured"})
    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df["may_ring"] = df["tier"] == "measured"
    df["congregations"] = 0
    # `Zionists` and `Apostles` both land on christianity.africaninstituted, so the frame is
    # summed rather than returned per source row; tier is part of the key because the weakest
    # tier on a (unit, node) pair must win rather than be averaged away.
    return (df.groupby(["unit", "node", "tier", "may_ring"], as_index=False)
              [["count", "congregations"]].sum()
              [["unit", "node", "count", "congregations", "may_ring", "tier"]])


ENTRY = {
    "sz": dict(
        name="Eswatini",
        source="2017 Population and Housing Census, Volume 3 (Central Statistical Office)",
        basis="self-identification, whole census population",
        note_public=(
            "**A third of Eswatini belongs to a Zionist church.** The Swazi Zionist "
            "churches began when missionaries of John Alexander Dowie's Christian Catholic "
            "Apostolic Church in Zion reached southern Africa in 1904, and they separated "
            "from it almost immediately. What they became is indigenous: prophecy and faith "
            "healing, baptism by immersion in running water, white robes and drums, and "
            "services held outdoors as often as in a building. At **33.6%** they are the "
            "largest single religious answer in the country, half as large again as the "
            "next one, and adding the Apostolic churches takes the African Instituted share "
            "to **38.5%**. Zimbabwe is the only other country here whose largest religion is "
            "one Africans founded, and this census does something Zimbabwe's does not: it "
            "counts these churches beside a named list of Catholics, Anglicans, Lutherans, "
            "Methodists and Nazarenes, rather than as a single cell next to *Protestant*. "
            "**The regional table covers Christians and nobody else.** Volume 3 breaks its "
            "thirteen denominations down by region; the nine top-level religions, including "
            "the **7.4%** with no religion, it publishes for the country as a whole and no "
            "further. So each region's non-Christian total here is a real count, its "
            "published population minus its published Christians, but the split of that "
            "total is the national one applied four times, and those rows are drawn "
            "desaturated. The one to be careful about is no religion: across the border in "
            "Zimbabwe that figure runs from 4.5% to 13.5% depending on the province, and "
            "whatever Eswatini's version of that spread is, this map cannot show it. "
            "**Traditional religion reads 0.45% and the real figure is not close.** The box "
            "is exclusive of the Christian ones, so it counts only the people who put "
            "nothing else first. Swazi ancestral practice is common among people who "
            "answered Christian: the *emadloti*, a visit to an *inyanga* or *sangoma*, and "
            "the *Incwala* and *Umhlanga* rituals the monarchy holds every year. The Zionist "
            "churches grew out of exactly that overlap and are counted at seventy-five times "
            "the size. Read 0.45% as the number who chose it instead of a church. "
            "**Four regions for a million people, so read this as composition and never as "
            "location.** A region is a quarter of the country; a cluster of dots says which "
            "region, drawn where Swazis actually live, and nothing at all about which town. "
            "2017 was the first Eswatini census ever to ask about religion, and its whole "
            "religion output is one chapter of one volume. There is no tinkhundla table, and "
            "the Census Atlas that does reach the tinkhundla has no religion in it."),
        how="census, 2017",
        grain="regions, 273,000 people on average",
        fill="from the same census at national level",
        gap_share=0.022,
        gap="the 2.2% who did not answer the religion question",
        counts=_sz_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "sz" / "sz_cells.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_sz_place_weight,
        note="THE FIRST COUNTRY HERE WHOSE PLURALITY RELIGION IS AN AFRICAN INSTITUTED "
             "CHURCH. `Zionists` are 367,290 people and 33.60%, and with `Apostles` the "
             "node reaches 38.48% — larger than every mission denomination in Eswatini put "
             "together. Zimbabwe's Vapostori are a bigger share of a bigger country, but "
             "ZIMSTAT gives them one cell beside `Protestant`; the CSO counts the Zionists "
             "and Apostles alongside eleven named mission bodies, so this is the first "
             "source that can be read as a comparison rather than as a residual. "
             "THE HOST EVERY EARLIER NOTE NAMES IS DEAD AND THE FILE IS ON THE GOVERNMENT "
             "PORTAL. sources.md §11w recorded Eswatini as `no reachable host` and was "
             "right about the office: eswatinistats.org.sz resolves and times out, "
             "swazistats.org.sz does not resolve, and the Wayback CDX has 61 captures of "
             "the first and not one PDF. The census volumes are Joomla articles on "
             "www.gov.sz under /images/FinanceDocuments/. sources/sz.md §1. "
             "RELIGION BY REGION IS A CHRISTIAN-ONLY TABLE. Table 3.2.4 gives thirteen "
             "denominations on four regions; Table 3.2.1's nine top-level religions are "
             "national only. Each region's non-Christian total is still counted (its "
             "Table 5.2.2 population minus its Table 3.2.4 Christians) and only the split "
             "of it is carried down, so 89.25% is `measured` and 10.75% `derived` with no "
             "magnitude estimated anywhere. "
             "SIX TABLES ARE READ AND ONE IS DRAWN, because every identity inside Table "
             "3.2.4 survives a consistent permutation of its four region columns — "
             "Zimbabwe's warning. What does not: 3.2.4's Total column against 3.2.2's, "
             "3.2.3's urban plus rural against 3.2.2, 3.2.5's printed percentage on all 52 "
             "cells, and above all each region's Christian share against its Table 5.2.2 "
             "population, which lands 88.4-89.8% against a national 89.25% and which "
             "exactly ONE of the 24 orderings passes. "
             "AND THE UNSD DEMOGRAPHIC YEARBOOK IS AN INDEPENDENT TRANSCRIPTION of the same "
             "tables, twenty categories partitioning 1,093,238 exactly. Every figure is "
             "asserted against it — §0.5's third use of the oracle, a second pair of eyes "
             "on the parse rather than a question about what exists. "
             "PLACEMENT IS WORLDPOP AND NOT KONTUR, THE SECOND COUNTRY HERE OFF THAT GRID "
             "AND THE FIRST WHERE IT IS SIMPLY WRONG. Singapore (§9bp) left Kontur because "
             "the grid counts everybody present and its census counts residents, so an "
             "accurate grid measured the wrong people; here the grid is inaccurate on its "
             "own terms. Kontur puts 43% of Eswatini in Lubombo against a census 19%, "
             "reading 0.38x on "
             "Hhohho and 2.24x on Lubombo, because it is built from building footprints and "
             "the northern Lowveld sugar estates are mapped in OSM while the Highveld "
             "homesteads are not. WorldPop's constrained Maxar raster reads 0.95-1.04, and "
             "a second WorldPop release of the census's own year agrees with it to within "
             "0.04. sources/sz_grid.py has all three tables of numbers.",
    ),
}
