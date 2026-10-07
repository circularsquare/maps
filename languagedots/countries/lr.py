# Liberia. The census publishes ethnicity only nationally, so Afrobarometer R4-R7 (2008-2018)
# first-language answers by county, times the 2022 census's county populations
# (sources/lr_afro.py); every row `modelled`. On religiondots' Kontur hexes for the 15
# counties. The record is sources/lr.md.
from _shared import *  # noqa: F401,F403

COUNTIES = 15
CENSUS_2022 = 5_250_187


def _counts():
    import lr2018
    df = pd.read_csv(NORM / "lr.csv", dtype={"geo_id": str})
    if df["geo_id"].nunique() != COUNTIES:
        raise SystemExit(f"lr.csv: {df['geo_id'].nunique()} counties, expected {COUNTIES} -- "
                         "re-run sources/lr_afro.py")
    if int(df["count"].sum()) != CENSUS_2022:
        raise SystemExit(f"lr.csv sums to {int(df['count'].sum()):,}, not {CENSUS_2022:,}")
    df["node"] = df["source_category"].map(lr2018.resolve)
    missing = sorted(set(df.loc[df["node"].isna(), "source_category"]))
    if missing:
        raise SystemExit(f"lr.csv answers with no node: {missing}")
    lut = pd.read_csv(RD_GEO / "lr" / "lr_lookup.csv", dtype=str)
    if set(df["geo_id"]) != set(lut["unit"]):
        raise SystemExit("lr.csv counties do not match religiondots' lr_lookup.csv")
    df["unit"] = df["geo_id"]
    df = df[df["count"] > 0]
    out = by_unit(df)
    out["tier"] = "modelled"
    return out


ENTRY = dict(
    name="Liberia",
    source=("Afrobarometer rounds 4 to 7 (2008, 2012, 2015, 2018), language of respondent and "
            "mother tongue; county populations from the 2022 Population and Housing Census "
            "(LISGIS); the 2008 census's national ethnic table (Final Results, Table 4.4) as "
            "a check"),
    how="survey, 2008-2018, first language, county shares applied to 2022 census populations",
    parts=[dict(covers="Everyone",
                source="Afrobarometer 2008-18, first language, about 4,800 adults; English at "
                       "its 2018 mother-tongue share",
                rest=True)],
    grain="15 counties, 350,000 people on average",
    gap="Children are drawn on the adults' mix.",
    view=[-11.6, 4.3, -7.3, 8.6],
    counts=_counts,
    mappings=["lr2018"],
    place=RD_GEO / "lr" / "lr_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "Liberia's census asks each person's ethnic group but publishes it only for the "
        "country as a whole, and asks no language question. This map applies each county's "
        "Afrobarometer language shares to its 2022 census population. Nationally the result is "
        "close to the 2008 census's ethnic groups, except Sapo (0.2% here against 1.2%), "
        "because few Sapo were interviewed. Liberian English is the language most Liberians "
        "share. Asked for their mother tongue in 2018, under 1% named English, and it is drawn "
        "at that share. Asked their language at home in later rounds, 39% named English."),
)
