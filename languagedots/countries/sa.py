# Saudi Arabia. No census or survey asks language. 2022 census: Saudis on Saudi Arabic, non-Saudis
# by nationality and sex read as their languages (sources/sa_census.py), 13 regions with Riyadh
# region in two, on religiondots' Kontur 400 m hexes re-keyed (sources/gulf_place.py), citizens
# and foreigners placed apart inside each unit. Every row derived. Record: sources/sa.md,
# sources/gulf_place.md.
from _shared import *  # noqa: F401,F403

CENSUS_2022 = 32_175_224


def _gp():
    """sources/gulf_place.py: the Gulf citizen / foreign placement and unit splits."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("gulf_place", ROOT / "sources" / "gulf_place.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rows():
    import sa2022
    df = pd.read_csv(NORM / "sa.csv", dtype={"geo_id": str})
    if df["geo_id"].nunique() != 13:
        raise SystemExit(f"sa.csv: {df['geo_id'].nunique()} regions, expected 13")
    if int(df["count"].sum()) != CENSUS_2022:
        raise SystemExit(f"sa.csv sums to {df['count'].sum():,}, expected {CENSUS_2022:,}")
    # religiondots' sa_lookup.csv: the census regions' ids are its units (SA01..SA14, no SA13)
    lut = pd.read_csv(RD_GEO / "sa" / "sa_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    if df["unit"].isna().any():
        raise SystemExit(f"sa: regions missing from religiondots' sa_lookup.csv: "
                         f"{sorted(df.loc[df['unit'].isna(), 'geo_id'].unique())}")
    df["node"] = df["source_category"].map(sa2022.resolve)
    # Riyadh region split into Riyadh + Ad Diriyah governorates and the rest (RCRC's census
    # table by citizenship and governorate; sources/gulf_place.py SPLITS)
    return _gp().split_units(df, "sa")


def _counts():
    out = by_unit(_rows())
    out["tier"] = "derived"
    return out


def _weight(place):
    gp = _gp()
    return gp.GulfWeighter("sa", place, *gp.citizen_tables(_rows()))


ENTRY = dict(
    name="Saudi Arabia",
    source=("General Authority for Statistics (GASTAT), Saudi Census 2022: Saudis and non-Saudis "
            "by region, non-Saudis by nationality and sex (as mirrored by the Gulf Labour Markets, "
            "Migration and Population programme); Saudis and non-Saudis in Riyadh and Ad "
            "Diriyah governorates from the Royal Commission for Riyadh City's open data; "
            "OpenStreetMap industrial land; Indians by state from the Kerala Migration "
            "Survey 2023 and India's emigration clearances 2011 to 2017; Pakistanis by province "
            "from the Bureau of Emigration's registrations 2019 to 2021; home languages from the "
            "censuses of India (2011), Pakistan (2023) and twelve other countries on this map"),
    how=("census, 2022, no language question; citizens drawn as Saudi Arabic, foreign residents "
         "by nationality, each on its country's main language or language mix"),
    parts=[
        dict(covers="Saudi citizens", source="2022 census, citizens by region, drawn as Saudi "
                                             "Arabic",
             people=18_792_262),
        dict(covers="Foreign residents",
             source="2022 census, nationality by sex, each drawn on its country's languages; "
                    "Indians by state and Pakistanis by province from emigration records",
             rest=True),
    ],
    grain=("13 regions, Riyadh region split into its capital governorates and the rest, 2.3 "
           "million people on average"),
    gap=("Saudi citizens whose first language is not Arabic, whom no source counts; foreign "
         "residents the census missed"),
    view=[34.5, 16.3, 55.7, 32.2],
    counts=_counts,
    mappings=["sa2022"],
    place=GEO / "sa" / "sa_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=_weight,
    note_public=(
        "Saudi Arabia's 2022 census asked no language question, so nothing here is a count of "
        "a language. Citizens are all drawn as Saudi Arabic: its dialects are not told apart, "
        "and citizens who speak something else at home (Mehri in the far south, for one) are "
        "not shown. Foreign residents, 42% of the people, are drawn by nationality, which the "
        "census gives only for the whole country. Indians and Pakistanis are split by the "
        "states and provinces Gulf migrants come from, each at its own census languages. "
        "Inside each region, foreign residents are placed more heavily in dense city districts "
        "and wholly in industrial areas and labour camps, and citizens in the rest. That rule "
        "was fitted to the counts Kuwait and Oman publish for their districts; it is an "
        "estimate of where people live, not a count."),
)
