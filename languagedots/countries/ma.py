# Morocco. RGPH 2024, local languages used, several allowed, per commune and arrondissement
# (sources/ma_rgph.py: HCP's indicator workbook, shares x population municipale, each person shared
# across the languages they use), on religiondots' Kontur hexes re-keyed from its 73 units to
# 1,530 communes by OSM's commune polygons (sources/ma_geo.py). Western Sahara is drawn inside
# Morocco west of the berm, as religiondots draws it (religiondots ask 031, ruled 2026-09-15).
from _shared import *  # noqa: F401,F403


def _counts():
    import ma2024
    df = pd.read_csv(NORM / "ma.csv", dtype={"geo_id": str})
    df["node"] = df["source_category"].map(ma2024.resolve)
    unmapped = sorted(set(df.loc[df["node"].isna(), "source_category"]))
    if unmapped:
        raise SystemExit(f"ma: categories with no node: {unmapped}")
    lut = pd.read_csv(GEO / "ma" / "ma_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    if df["unit"].isna().any():
        raise SystemExit(f"ma: {df.loc[df['unit'].isna(), 'geo_id'].nunique()} communes missing from "
                         "data/geo/ma/ma_lookup.csv; re-run sources/ma_geo.py")
    if df["unit"].nunique() != 1530:
        raise SystemExit(f"ma: {df['unit'].nunique()} units, expected 1,530")
    df = df[df["count"] > 0]
    return df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


ENTRY = dict(
    name="Morocco",
    source=("General Census of Population and Housing (RGPH) 2024, the local languages each "
            "resident uses, per commune, from the Haut-Commissariat au Plan's workbook of "
            "demographic and socioeconomic indicators"),
    how=("census, 2024, local languages used, several allowed; each person shared across the "
         "languages they named"),
    parts=[dict(covers="People in households",
                source="2024 census, local languages used, several allowed",
                rest=True)],
    grain="1,530 communes and city arrondissements, 23,800 people on average",
    gap=("337,739 people counted outside households (barracks, boarding schools, prisons), "
         "41,523 who use none of the five languages tabulated, and 5,371 in four Saharan "
         "communes with no figures"),
    view=[-17.2, 20.7, -0.9, 36.0],
    counts=_counts,
    mappings=["ma2024"],
    place=GEO / "ma" / "ma_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "The 2024 census asked which local languages each person uses, and many Moroccans "
        "named two: the five answers add up to 117% of the country. Each person is drawn "
        "shared across the languages they named, so a commune where everyone uses both Darija "
        "and Tachelhit is drawn half and half. The census asked about use, not mother tongue, "
        "and published only five languages: Darija (Moroccan Arabic), Tachelhit, Tamazight "
        "(the Central Atlas language), Tarifit and Hassania. People who can speak an Amazigh "
        "language but do not use it every day are not counted as using it, and Amazigh "
        "associations have disputed the 2024 Amazigh figures as far too low. The question "
        "appears to have been on the detailed form, which went to a 20% sample in communes of "
        "2,000 households or more, so there the shares are sample estimates. Communes are "
        "placed on OpenStreetMap's boundaries; a few that could not be matched are drawn with "
        "their neighbours. Western Sahara is drawn as far as Morocco administers it, west of "
        "the berm, from the Moroccan census."),
)
