# Tajikistan. 2010 census native language, printed for the whole country only, split by nationality
# (sources/tj_census.py), placed by region through a nationality model, on religiondots' Kontur
# hexes for the five regions. sources/tj.md is the record.
from _shared import *  # noqa: F401,F403


def _counts():
    import tj2010
    df = pd.read_csv(NORM / "tj_model.csv")
    lut = pd.read_csv(RD_GEO / "tj" / "tj_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    if df["unit"].isna().any() or df["unit"].nunique() != 5:
        raise SystemExit(f"tj: regions not in religiondots' tj_lookup.csv: "
                         f"{sorted(df.loc[df['unit'].isna(), 'geo_id'].unique())}")
    df["node"] = df["source_category"].map(tj2010.resolve)
    return df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


ENTRY = dict(
    name="Tajikistan",
    source=("Population and Housing Census 2010, Volume III, population by nationality and "
            "native language, and nationality by region (Agency on Statistics under the President "
            "of the Republic of Tajikistan)"),
    how=("model: the 2010 census's native language by nationality, printed for the whole "
         "country only, placed by each nationality's count by region"),
    parts=[dict(covers="Everyone",
                source="2010 census, native language by nationality, placed by nationality "
                       "per region",
                rest=True)],
    grain="5 regions, 1.5 million people on average; languages within each modelled",
    gap=("the Pamiri languages (Shughni, Rushani, Wakhi and others), which the census counts as "
         "Tajik; and the 2020 census, whose language volume is unpublished"),
    view=[67.3, 36.6, 75.2, 41.1],
    counts=_counts,
    mappings=["tj2010"],
    place=RD_GEO / "tj" / "tj_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "The 2010 census asked each person's native language, which in the countries of the "
        "former Soviet Union leans towards identity rather than everyday use. The 2020 census "
        "asked again, but its volume on nationality and language has not been published, so "
        "the dots show the 2010 population of 7.6 million rather than the 9.7 million counted "
        "in 2020. The 2010 volume prints native language only for the whole country, split by "
        "nationality, so each nationality is placed where the census counted it, region by "
        "region, and the places are modelled, not counted. The census lists several Uzbek "
        "tribes as nationalities of their own, and their speech is drawn as dialects of Uzbek. "
        "The census counts the Pamiri peoples of Gorno-Badakhshan as Tajiks and their languages "
        "as Tajik, so the region is drawn almost all Tajik, although Shughni, Rushani, Wakhi and "
        "the other Pamiri languages are spoken at home across much of it."),
)
