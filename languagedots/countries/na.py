# Namibia. Census 2011, main language spoken in the household, from the census's 20% public
# microdata sample (sources/na_pums.py), by the 107 constituencies of 2011 and town or
# countryside. Placed on religiondots' Kontur hexes re-keyed to the constituencies, each cut into
# a town and a countryside part (sources/na_geo.py). The record is sources/na.md.
from _shared import *  # noqa: F401,F403


def _counts():
    import na2011
    df = pd.read_csv(NORM / "na.csv", dtype={"geo_id": str})
    if df["geo_id"].nunique() != 107:
        raise SystemExit(f"na.csv: {df['geo_id'].nunique()} constituencies, expected 107")
    df["node"] = df["source_category"].map(na2011.NAMES)
    unresolved = sorted(set(df.loc[df["node"].isna(), "source_category"]) - set(na2011.EXCLUDED))
    if unresolved:
        raise SystemExit(f"na.csv categories that resolve to nothing: {unresolved}")
    df = df[df["node"].notna()].copy()
    df["unit"] = df["geo_id"] + "-" + df["residence"]
    return by_unit(df)


ENTRY = dict(
    name="Namibia",
    source="Namibia 2011 Population and Housing Census, Public Use Microdata Sample, 20% of "
           "households (Namibia Statistics Agency, 2013)",
    how="census, 2011, main language spoken in the household",
    parts=[dict(covers="Everyone",
                source="2011 census, 20% sample, main language spoken in the household",
                rest=True)],
    grain="107 constituencies, 19,000 people on average, each split into town and countryside",
    gap="50,487 people (2.4%), almost all in hostels, barracks, prisons, hospitals and other "
        "places where the household question was not asked",
    view=[11.7, -29.0, 25.3, -16.9],
    counts=_counts,
    mappings=["na2011"],
    place=GEO / "na" / "na_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "These are 2011 figures: the 2023 census published no language table, so this map uses "
        "the 2011 census's public sample of one household in five. The census asked each "
        "household its main language, so everyone in a household is drawn at its answer. "
        "Three answers name groups of different languages and are drawn as language not "
        "named: the Kavango languages (Rukwangali, Rumanyo, Thimbukushu), the Caprivi "
        "languages of the Zambezi region (Silozi, Subiya, Fwe and others) and the San "
        "languages. Oshiwambo and Otjiherero are each drawn as one language, as they are "
        "taught in Namibia, and the Nama/Damara answer as Khoekhoegowab."),
)
