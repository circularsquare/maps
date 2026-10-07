# Kyrgyzstan. 2022 census native language by ethnic group, per rayon and city (sources/kg_census.py),
# on Kontur hexes keyed to COD-AB's rayons and cities (sources/kg_geo.py). sources/kg.md is the record.
from _shared import *  # noqa: F401,F403

_LUT = GEO / "kg" / "kg_lookup.csv"


def _counts():
    import kg2022
    df = pd.read_csv(NORM / "kg.csv")
    # rayons and cities of oblast significance, and Bishkek and Osh whole; not the oblast rows,
    # and not Bishkek's four districts, which have no polygon (sources/kg_geo.py)
    df = df[(df["geo_level"] == "unit") | (df["geo_level"] == "city")].copy()
    lut = pd.read_csv(_LUT, dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    if df["unit"].isna().any():
        raise SystemExit(f"kg: {sorted(df.loc[df['unit'].isna(), 'geo_id'].unique())} not in "
                         "kg_lookup.csv; re-run sources/kg_geo.py")
    df["node"] = df["source_category"].map(kg2022.resolve)
    return df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


ENTRY = dict(
    name="Kyrgyzstan",
    source=("Population and Housing Census 2022, Book III (regional volumes), table 3.4, ethnic "
            "groups by native language (National Statistical Committee of the Kyrgyz Republic)"),
    how="census, 2022, native language",
    parts=[
        dict(covers="Everyone", source="2022 census, native language by ethnic group",
             rest=True),
    ],
    grain="51 rayons and cities, and Bishkek and Osh, 131,000 people on average",
    view=[69.2, 39.1, 80.3, 43.3],
    counts=_counts,
    mappings=["kg2022"],
    place=GEO / "kg" / "kg_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "The 2022 census asked each person's native language, which in the former Soviet Union "
        "leans towards identity rather than everyday use: only 9,332 of 5.38 million ethnic "
        "Kyrgyz named Russian. 79.7% named Kyrgyz, 12.7% Uzbek and 4.4% Russian. Smaller groups "
        "often named a neighbour's language, such as 40% of Kazakhs naming Kyrgyz. Bishkek is "
        "drawn as one city, and Kara-Kul with Toktogul rayon around it."),
)
