# Mozambique. IV RGPH 2017, Quadro 22 (mother tongue, aged 5+) of the eleven provincial table
# sets, urban and rural (sources/mz_rgph.py), on religiondots' Kontur hexes cut into urban and
# rural by density within each province (sources/mz_geo.py). The record is sources/mz.md.
from _shared import *  # noqa: F401,F403


def _counts():
    import mz2017
    df = pd.read_csv(NORM / "mz.csv")
    df = df[df["geo_level"] == "province"].copy()
    if df["geo_id"].nunique() != 11:
        raise SystemExit(f"mz.csv: {df['geo_id'].nunique()} provinces, expected 11")
    unknown = sorted(set(df["source_category"]) - set(mz2017.NAMES))
    if unknown:
        raise SystemExit(f"mz.csv categories with no mapping: {unknown}")
    df["node"] = df["source_category"].map(mz2017.resolve)
    df = df[df["node"].notna() & (df["count"] > 0)].copy()
    df["unit"] = df["geo_id"] + "/" + df["area"].str[0]
    return by_unit(df)


ENTRY = dict(
    name="Mozambique",
    source="IV Recenseamento Geral da População e Habitação 2017, Quadro 22 of the provincial "
           "and national tables (INE Moçambique), read from the Wayback Machine's 2019 captures",
    how="census, 2017, mother tongue, aged 5 and over",
    parts=[dict(covers="Everyone aged 5 and over", source="2017 census, mother tongue",
                rest=True)],
    grain="21 units (11 provinces, each split into town and countryside), 1,060,000 people aged "
          "5 and over on average",
    gap="672,976 people with no mother tongue recorded (3.0%, 290,465 of them in Cabo Delgado), "
        "4,173 unable to speak, and 4.66 million children under 5, who were not asked",
    view=[30.2, -26.9, 40.9, -10.4],
    counts=_counts,
    mappings=["mz2017"],
    place=GEO / "mz" / "mz_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "The census asked everyone aged 5 and over their mother tongue. INE publishes the "
        "answers by province only, for towns and countryside separately, so within a "
        "province's towns, and within its countryside, every language is spread the same way. "
        "Each province's table names its five to seven largest languages; the rest, 947,000 "
        "people, are drawn as Bantu, language not named. Portuguese is the mother tongue of "
        "38% of town dwellers and 5% of country dwellers. In Cabo Delgado no mother tongue was "
        "recorded for 15.6% of the province, and they are not drawn."),
)
