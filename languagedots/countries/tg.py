# Togo. No census table of language or ethnicity is published, so MICS6 2017 (INSEED and UNICEF),
# mother tongue of the household head read as every member's, weighted shares per region of its
# eleven language groups, each group split by Afrobarometer R5-R7 (2012-2017) first-language
# answers, times the 2022 census's region populations (sources/tg_mics.py; the Afrobarometer-only
# build it replaced is sources/tg_afro.py). Every row `modelled`. On religiondots' Kontur hexes for
# its six units (five regions, Lomé apart). The record is sources/tg.md.
from _shared import *  # noqa: F401,F403

UNITS = 6
CENSUS_2022 = 8_095_498


def _counts():
    import tg2017
    df = pd.read_csv(NORM / "tg.csv", dtype={"geo_id": str})
    if df["geo_id"].nunique() != UNITS:
        raise SystemExit(f"tg.csv: {df['geo_id'].nunique()} units, expected {UNITS} -- "
                         "re-run sources/tg_mics.py")
    if int(df["count"].sum()) != CENSUS_2022:
        raise SystemExit(f"tg.csv sums to {int(df['count'].sum()):,}, not {CENSUS_2022:,}")
    if set(df["source_id"]) != {"mics6_2017_hc1b_afro_split"}:
        raise SystemExit(f"tg.csv sources {sorted(set(df['source_id']))}: re-run "
                         "sources/tg_mics.py")
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
    source=("Togo Multiple Indicator Cluster Survey 2017 (MICS6; INSEED, UNICEF), microdata; "
            "Afrobarometer rounds 5, 6 and 7 (2012, 2014, 2017), language of respondent and "
            "mother tongue, to split MICS's language groups, and rounds 8 and 9 (2021-22, 2024), "
            "language spoken at home, for French; region populations from the 2022 "
            "census (RGPH-5, INSEED)"),
    how=("a household survey, 2017, about 7,900 households, mother tongue of the household "
         "head read as every member's, in eleven groups; weighted shares per region applied to "
         "2022 populations, each group shared among its languages in the proportions "
         "Afrobarometer's 2012-2017 first-language answers give in that region; French drawn "
         "at its share of Afrobarometer's 2021-2024 home-language answers"),
    parts=[
        dict(covers="Everyone",
             source="UNICEF MICS 2017, about 7,900 households, mother tongue of the household "
                    "head in eleven groups, split by Afrobarometer 2012-2017; shares per region "
                    "applied to the 2022 census populations",
             rest=True),
    ],
    grain="5 regions and Lomé, 1.3 million people on average",
    gap=("no census count of language or ethnic group is published; MICS's foreign languages "
         "(4.7%) and the unnamed part of its other national languages (1.4%) are drawn as "
         "other African languages"),
    view=[-0.2, 6.0, 1.9, 11.2],
    counts=_counts,
    mappings=["tg2017"],
    place=RD_GEO / "tg" / "tg_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "Togo's census office has not published any count of language or ethnic group. This "
        "map uses UNICEF's 2017 household survey (MICS), which asked the mother tongue of the "
        "head of each of about 7,900 households, and draws every member of the household on "
        "it. The survey groups languages in pairs such as Ewe and Mina or Bassar and Konkomba, "
        "so each group is shared among its languages as the Afrobarometer surveys of 2012 to "
        "2017 found them in that region; Ouatchi, Aja and Fon are counted with Ewe and Mina. "
        "Each region's shares are applied to its 2022 census population, with Lomé drawn apart "
        "from the rest of Maritime, and dots are spread over the whole region. 4.7% named a "
        "foreign language, mostly people from neighbouring countries, and the survey does not "
        "say which; they are drawn as other African languages, 10% of Lomé. French is drawn "
        "as the language spoken at home by 4.2% nationally (6% of Lomé), the share in the "
        "Afrobarometer surveys of 2021 to 2024; only 0.3% of household heads gave it as their "
        "mother tongue."),
)
