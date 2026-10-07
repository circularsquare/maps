# Togo. No census table of language or ethnicity is published, so Afrobarometer R5-R7
# (2012-2017) first-language answers by region, times the 2022 census's region populations
# (sources/tg_afro.py); every row `modelled`. On religiondots' Kontur hexes for its six units
# (five regions, Lomé apart). The record is sources/tg.md.
from _shared import *  # noqa: F401,F403

UNITS = 6
CENSUS_2022 = 8_095_498


def _counts():
    import tg2017
    df = pd.read_csv(NORM / "tg.csv", dtype={"geo_id": str})
    if df["geo_id"].nunique() != UNITS:
        raise SystemExit(f"tg.csv: {df['geo_id'].nunique()} units, expected {UNITS} -- "
                         "re-run sources/tg_afro.py")
    if int(df["count"].sum()) != CENSUS_2022:
        raise SystemExit(f"tg.csv sums to {int(df['count'].sum()):,}, not {CENSUS_2022:,}")
    df["node"] = df["source_category"].map(tg2017.resolve)
    missing = sorted(set(df.loc[df["node"].isna(), "source_category"]))
    if missing:
        raise SystemExit(f"tg.csv answers with no node: {missing}")
    lut = pd.read_csv(RD_GEO / "tg" / "tg_lookup.csv", dtype=str)
    if set(df["geo_id"]) != set(lut["unit"]):
        raise SystemExit("tg.csv units do not match religiondots' tg_lookup.csv")
    df["unit"] = df["geo_id"]
    df = df[df["count"] > 0]
    out = by_unit(df)
    out["tier"] = "modelled"
    return out


ENTRY = dict(
    name="Togo",
    source=("Afrobarometer rounds 5, 6 and 7 (2012, 2014, 2017), language of respondent and "
            "mother tongue; region populations from the 2022 census (RGPH-5, INSEED)"),
    how=("survey, 2012-2017, about 3,600 adults' first language (\"language of respondent\", "
         "then \"mother tongue\"), weighted shares per region applied to 2022 populations; "
         "French drawn at its share of the 2017 mother-tongue answers"),
    parts=[
        dict(covers="Everyone",
             source="Afrobarometer 2012-2017, about 3,600 adults' first language, shares per "
                    "region applied to the 2022 census populations",
             rest=True),
    ],
    grain="5 regions and Lomé, 1.3 million people on average",
    gap=("no census count of language or ethnic group is published; the survey's own "
         "\"other\" answers (2.8%) are drawn as other African languages, and children are "
         "drawn on the adult mix"),
    view=[-0.2, 6.0, 1.9, 11.2],
    counts=_counts,
    mappings=["tg2017"],
    place=RD_GEO / "tg" / "tg_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "Togo's census office has not published any count of language or ethnic group. This "
        "map uses the Afrobarometer surveys of 2012, 2014 and 2017, which asked about 3,600 "
        "adults their own language (in 2017, their mother tongue), and applies each region's "
        "shares to its 2022 census population, with Lomé drawn apart from the rest of "
        "Maritime. With about 400 to 1,100 people asked per region, small languages are "
        "uncertain and are spread over their whole region. "
        "Later rounds asked which language is spoken at home, and there French and Ewe are "
        "named more often, as common languages; those rounds are not used. French is drawn at "
        "its share of the 2017 mother-tongue answers, 1.5% nationally, and the other French "
        "answers on the language of the person's ethnic group."),
)
