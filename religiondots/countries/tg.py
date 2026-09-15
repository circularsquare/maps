# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _tg_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read.

    6 units over 56,800 km2. Lomé is 139 km2 and 866,307 people; Plateaux is 17,400 km2 and holds
    its people in Atakpamé, Kpalimé and along the roads north (sources/tg_geo.py).
    """
    return _kontur_place_weight(place, "tg_hexes.gpkg", "sources/tg_geo.py")


def _tg_counts():
    """The 2022 census's national religion rows on an Afrobarometer pattern: 13 nodes, 6 units, and
    EVERY ROW IS `modelled` IN §7.

    TWO EXACT CENSUS MARGINS AND A SURVEY BETWEEN THEM, Liberia's construction. Livret 01 gives the
    unit populations and UNSD table 28 the religion totals; the table is fitted to both and the
    survey supplies only the pattern. Muslim, Animist, No Religion, Pentecostal and the Evangelical
    Presbyterian Church carry their own pattern; the rest are seeded at the national rate.
    sources/tg.py has the construction.
    """
    from tg2022 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "tg.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    lut = pd.read_csv(HERE / "data" / "geo" / "tg" / "tg_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"tg.csv units with no polygon: {missing} -- re-run "
                         "sources/tg_geo.py, the lookup is stale")
    if df["unit"].nunique() != 6:
        raise SystemExit(f"{df['unit'].nunique()} units, expected 6")

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)].copy()
    # EVERY row, without exception -- nobody counted any cell (§7b).
    df["tier"] = "modelled"
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations", "tier"]]


ENTRY = {
    "tg": dict(
        name="Togo",
        source="2022 census (INSEED), religion totals as held by the UN Statistics Division, with "
               "the regional pattern from five pooled rounds of the Afrobarometer, 2012 to 2022",
        basis="self-identification, whole census population",
        note_public=(
            "**Togo's 2022 census asked about religion, and the answer is out only as one national "
            "table.** INSEED sent it to the UN Statistics Division; its own releases from the census "
            "have no religion table at any level. Of the people whose religion was recorded, it "
            "counted **21.6%** Catholic, **19.2%** Muslim, **17.5%** following traditional religion "
            "and 9.7% with no religion, with 8.9% in the Assemblies of God, 6.4% Pentecostal and "
            "3.6% in the Evangelical Presbyterian Church. "
            "**Where those people are comes from a survey.** Five rounds of the Afrobarometer are "
            "pooled, **5,987** adults interviewed between December 2012 and March 2022. The census "
            "fixes each unit's population and each religion's national total, and the survey "
            "decides only how a unit's people divide between them, so the dots are desaturated. "
            "Lomé, the communes Golfe 1 to 5, is drawn apart from the rest of Maritime because the "
            "survey interviews it separately. "
            "**Five religions are placed by the survey, because their order across the units "
            "repeats from one set of rounds to another.** Islam is **52.7%** of Centrale and about a "
            "quarter of Savanes and Kara. Traditional religion is **29.7%** of Savanes, 23.4% of "
            "Maritime outside Lomé and 5.3% of Lomé. The Evangelical Presbyterian Church is **9.0%** "
            "of Plateaux and under 1% in the three northern regions, and Pentecostals are 10.8% of "
            "Plateaux. No religion is highest in Kara, at 15.9%. "
            "**Catholics, the Assemblies of God, Baptists and the smaller churches are not placed.** "
            "The survey's shares for them do not keep their order across the six units, so each is "
            "spread in every unit by what the placed religions leave over. Catholics come out "
            "highest in Lomé and lowest in Centrale for that reason, not because the survey found "
            "them there."),
        how="census totals, 2022, given a regional pattern by a pooled survey",
        grain="regions, with Lomé apart; 1.35 million people on average",
        gap="3.4%: 2.9% whose religion the census gives as not stated or unknown, and 0.5% who "
            "are in the census count and in none of its religion rows",
        gap_share=0.03360,                      # 272,041/8,095,498, exact; tools/gap_share.py "rows only"
        counts=_tg_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "tg" / "tg_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_tg_place_weight,
        note="INSEED HAS PUBLISHED NO RELIGION TABLE FROM THE 2022 CENSUS; UNSD table 28 holds the "
             "national 15 rows by urban and rural (sources.md §11w, §11aq). Built on Liberia's "
             "construction: an IPF of the Afrobarometer R5-R9 pattern to the census's unit "
             "populations (Livret 01 Tableaux 2 and 4) and UNSD's religion totals, every row "
             "modelled. The religion rows sum 37,326 short of the census; that residual is its own "
             "excluded column so the margins meet. 6 UNITS: Lomé is Golfe 1 to 5, drawn on COD-AB's "
             "Lome Commune plus the Aflao Gakli and Amoutive canton pieces (sources/tg_geo.py). "
             "CARRIED on the split-half: Muslim, Traditional, None, Pentecostal, Presbyterian; "
             "Catholic, Assembly of God, Baptist and Other Christians fail and are seeded flat.",
    ),
}
