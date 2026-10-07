# Kenya. Afrobarometer R4 and R6-R9 (2008-2022, about 9,900 respondents), home language, shares
# per county applied to KNBS 2019 county totals (sources/ke_afro.py); every row `modelled`. On
# religiondots' Kontur hexes for the 47 counties, placed on population inside each county. The
# record is sources/ke.md.
from _shared import *  # noqa: F401,F403

COUNTIES = 47
POP_2019 = 47_213_282


def _counts():
    import ke2022
    df = pd.read_csv(NORM / "ke.csv", dtype={"geo_id": str})
    if df["geo_id"].nunique() != COUNTIES:
        raise SystemExit(f"ke.csv: {df['geo_id'].nunique()} counties, expected {COUNTIES} -- "
                         "re-run sources/ke_afro.py")
    if int(df["count"].sum()) != POP_2019:
        raise SystemExit(f"ke.csv sums to {int(df['count'].sum()):,}, not {POP_2019:,}")
    df["node"] = df["source_category"].map(ke2022.resolve)
    missing = sorted(set(df.loc[df["node"].isna(), "source_category"]))
    if missing:
        raise SystemExit(f"ke.csv answers with no node: {missing}")
    # religiondots' ke_lookup.csv maps KNBS county codes (001-047) to its hex `unit`
    lut = pd.read_csv(RD_GEO / "ke" / "ke_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    if df["unit"].isna().any():
        raise SystemExit(f"ke.csv counties missing from religiondots' ke_lookup.csv: "
                         f"{sorted(set(df.loc[df['unit'].isna(), 'geo_id']))}")
    df = df[df["count"] > 0]
    out = by_unit(df)
    out["tier"] = "modelled"
    return out


ENTRY = dict(
    name="Kenya",
    source=("Afrobarometer rounds 4 and 6 to 9 (2008-2022), Kenya, home language; county "
            "populations from the 2019 Kenya Population and Housing Census (KNBS)"),
    how=("survey, 2008-2022, about 9,900 adults' home language, shares per county applied to "
         "2019 census populations; Swahili and English at their share of the 2016 mother-tongue "
         "question, other Swahili and English answers drawn on the person's ethnic language"),
    parts=[
        dict(covers="Everyone",
             source="Afrobarometer 2008-2022, home language, shares per county applied to the "
                    "2019 census populations",
             rest=True),
    ],
    grain="47 counties, 1.0 million people on average",
    gap=("no census count of language; people in hotels, hospitals and prisons (0.7%) are "
         "outside the county totals"),
    view=[33.8, -4.8, 42.1, 5.6],
    counts=_counts,
    mappings=["ke2022"],
    place=RD_GEO / "ke" / "ke_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "Kenya's census asks ethnicity, not language, so this map uses the Afrobarometer: about "
        "9,900 adults asked their home language between 2008 and 2022, with each county's "
        "shares applied to its 2019 census population. A county's mix rests on 28 to 1,000 "
        "interviews, so small languages are uncertain. From 2016 many people named Swahili as "
        "the language used at home beside a first language; Swahili and English are drawn at "
        "their 2016 mother-tongue share instead. Luhya, Kalenjin and Mijikenda are each drawn "
        "as one language, as the survey asked them."),
)
