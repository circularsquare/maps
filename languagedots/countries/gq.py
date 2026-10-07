# Equatorial Guinea. No language question anywhere: the 2011 DHS's national ethnic shares read as
# language, on the 2015 census's 7 provinces, each group drawn in its home provinces
# (sources/gq_dhs.py). Placed on religiondots' Kontur hexes, scaled to each province's census
# count (read-only). Record: sources/gq.md.
from _shared import *  # noqa: F401,F403

CENSUS_2015 = 1_225_377


def _counts():
    import gq2011
    df = pd.read_csv(NORM / "gq.csv", dtype={"geo_id": str})
    if df["geo_id"].nunique() != 7 or int(df["count"].sum()) != CENSUS_2015:
        raise SystemExit(f"gq.csv: {df['geo_id'].nunique()} provinces, {df['count'].sum():,} "
                         "people -- run sources/gq_dhs.py")
    df["node"] = df["source_category"].map(gq2011.resolve)
    df["unit"] = df["geo_id"]   # religiondots' gq_lookup.csv: geo_id == unit
    return df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


ENTRY = dict(
    name="Equatorial Guinea",
    source=("Demographic and Health Survey 2011 (EDSGE-I), Cuadro 3.1, ethnicity of women and men "
            "aged 15-49; Population and Housing Census 2015, province counts (INEGE)"),
    how=("survey, 2011, ethnic group read as language, national shares; each group drawn in "
         "its home provinces"),
    parts=[dict(covers="Everyone",
                source="2011 DHS, ethnic group, national shares on the 2015 census population",
                rest=True)],
    grain="the whole country, drawn on 7 provinces",
    gap="survey answers with no ethnic group (0.2%), left out before the shares",
    view=[5.4, -1.6, 11.4, 3.9],
    counts=_counts,
    mappings=["gq2011"],
    place=RD_GEO / "gq" / "gq_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "Equatorial Guinea has no census or survey that asks about language. The 2011 "
        "Demographic and Health Survey asked ethnic group, and each group is drawn as its own "
        "language, though many in Malabo and Bata speak Spanish at home instead. The shares "
        "are national only, so each group is drawn in its home provinces. The 6% who were "
        "foreign gave no nationality and are drawn as other languages."),
)
