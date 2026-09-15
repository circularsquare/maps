# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _al_place_weight(place):
    """countries.py hook. `place` is the Kontur 400 m hex layer scatter.py has read."""
    return _kontur_place_weight(place, "al_hexes.gpkg", "sources/al_grid.py")


def _al_counts():
    """INSTAT 2023 census table 1.13 at qark: 8 nodes on 12 units.

    ONE level, no allocation, nothing modelled — every row is `measured` and may ring.

    **THE TWELVE QARQE ARE THE CEILING AND IT WAS ESTABLISHED RATHER THAN ASSUMED.** INSTAT's
    PxWeb Census 2023 branch has folders for the nation, the 12 qarqe, the 61 bashki and the
    373 njësi administrative; `Besimi fetar` is in the first two and in neither of the other
    two. The published XLS set has the same shape and the 2011 prefecture booklets, which do
    carry a whole second section of tables by bashki/komunë, stop before religion as well.
    200,176 people per unit, which is coarser than Armenia's marzes and finer than Georgia's
    regions, both drawn.

    **WHAT DOES EXIST FINER IS FOUR OF THE TEN CATEGORIES, FOR 2011**, as ArcGIS layers in
    INSTAT's own organisation: Muslim, Bektashi, Catholic and Orthodox on all 373
    administrative units. Not drawn. `sources/al.md` §3 is the argument in full and the short
    version is that those four are 75.6% of the 2011 census, so taking them would put a
    quarter of Albania into one undifferentiated hole, drop the atheists and the believers of
    no denomination entirely, and trade the current census for one twelve years older.
    """
    from al2023 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "al.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    df = df[df["geo_level"] == "qark"].copy()
    df["count"] = df["count"].astype(int)
    if df["geo_id"].nunique() != 12:
        raise SystemExit(f"{df['geo_id'].nunique()} qarqe, expected 12 -- re-run "
                         "sources/al.py")

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df = df.rename(columns={"geo_id": "unit"})
    df["congregations"] = 0
    return df.groupby(["unit", "node"], as_index=False)[["count", "congregations"]].sum()


ENTRY = {
    "al": dict(
        name="Albania",
        source="Population and Housing Census 2023, table 1.13 (INSTAT)",
        basis="self-identification, whole enumerated population",
        note_public=(
            "**No other census on this map counts Bektashis apart from other Muslims, and "
            "the geography that comes out of it is the reason Albania is worth having.** "
            "The Bektashi order is a Sufi tariqa whose "
            "world headquarters moved to Tirana after Turkey closed the tekkes in "
            "1925, and INSTAT gives it a row of its own: **115,644** people, 4.81% of the "
            "country and 9.5% of its Muslims. Those people are not spread evenly. Across "
            "the twelve qarqe the Bektashi share of the population runs from 0.20% in "
            "Shkoder up to **21.04%** in Gjirokaster, where 59.4% of all Muslims are "
            "Bektashi, and the gradient is clean: the south and the Tomorr highlands "
            "against the Catholic and Sunni north. "
            "**The other nine categories are a real form too.** Believers of no "
            "denomination get a box, atheists get a separate one, and the two are not "
            "merged: **332,155** people answered that they believe and belong to no "
            "denomination, which is 13.8% of Albania and its second largest answer of any "
            "kind, while 85,311 called themselves atheists. That second figure is small "
            "for the state that declared itself the world's first officially atheist "
            "country in 1967 and closed every mosque, church and tekke in it, and both "
            "categories have a geography of their own: no denomination runs from 1.87% of "
            "Lezhe to 21.73% of Vlore, atheism from 0.35% of Shkoder to 8.30% of Vlore. "
            "**The map is twelve prefectures because that is where the religion table "
            "stops, and it stops there on purpose.** INSTAT publishes 2023 census tables "
            "at four levels: the nation, the 12 prefectures, the 61 municipalities and the "
            "373 administrative units. Religion appears at the first two and at neither of "
            "the others, where the tables are age, education, disability, households and "
            "dwellings. The office does map four of the ten answers, Muslim, Bektashi, "
            "Catholic and Orthodox, right down to the 373 administrative units, but only "
            "for the 2011 census; that is not drawn here, because those four are three "
            "quarters of a census and would leave the rest of the country blank. "
            "**One in six people is not on this map, and where they are missing from is "
            "not random.** 244,331 preferred not to answer and INSTAT records another "
            "134,451 as not available, together 15.8% of the census. Berat is the extreme: "
            "35.4% of the prefecture gave no usable answer, against 4.5% in Lezhe. So the "
            "eight categories drawn cover 95.5% of Lezhe and 64.6% of Berat, and a "
            "comparison between two prefectures is partly a comparison of how many people "
            "in each of them answered. Which way that leans is measurable: across the "
            "twelve prefectures the missing share tracks the no-denomination share of "
            "those who did answer (r = +0.68, permutation p = 0.009) and runs against the "
            "Catholic one, so every figure drawn here is slightly more Catholic and "
            "slightly less unchurched than Albania is, and the correction is not made."),
        how="census, 2023, full enumeration",
        grain="qarqe (prefectures), 200,000 people on average",
        gap="378,782 people, 15.8% of the census, who preferred not to answer or whose "
            "answer INSTAT records as not available",
        gap_share=0.1577,
        counts=_al_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "al" / "al_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_al_place_weight,
        note="THE HOST WAS NEVER DOWN AND THE DATABASE IS ON PORT 8083. queue.md closed "
             "Albania on three consecutive days on `instat.gov.al did not resolve`; with an "
             "ordinary browser User-Agent it answers 200, and its own census microsite "
             "links databaza.instat.gov.al:8083/pxweb/en/DST/, a live PxWeb 21.1 with a "
             "Census 2023 branch. TWO MORE THINGS A SWEEP MISSES THERE: the ENGLISH "
             "database lists no tables at all under Census 2023 while the Albanian one "
             "lists 105, and the PxWeb search index does not cover the 2023 branch, so "
             "searching it for `religion` or `fetare` returns 2011 and nothing else "
             "(sources/al.md §1). "
             "THE JOIN IS PROVED TO THE PERSON AND TWICE. INSTAT's own ArcGIS Online "
             "organisation carries the 12 prefecture polygons with the 2023 census "
             "population on them, and all twelve match the printed Total of that qark's "
             "worksheet exactly and are distinct from one another; then its four 2023 "
             "religion-share layers reproduce count/total from al.csv to 7e-15 percentage "
             "points on all 48 pairs (sources/al_geo.py). "
             "The dots are spread across 22,289 Kontur 400 m hexagons weighted by hex "
             "population (sources/al_grid.py); Tirana qark is 22 times denser than "
             "Gjirokaster and an equal-area spread would empty the city. "
             "FINER RELIGION EXISTS FOR 2011 ONLY, AND ONLY FOR FOUR CATEGORIES: "
             "administrativeunit_p_{muslim,bekta,cathol,orthod}_2011_view, 373 units each, "
             "in the same ArcGIS organisation. sources/al.md §3 says why it is not drawn "
             "and exactly what it would cost.",
    ),
}
