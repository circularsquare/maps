# Gibraltar. 2022 census Q12, main language, population aged 14 and over (sources/gi_census.py).
# Spanish and Arabic are tabled; the other eight named languages are read off a bar chart. One
# unit, on religiondots' Kontur hexes (read-only). Record: sources/gi.md.
from _shared import *  # noqa: F401,F403


def _counts():
    import gi2022
    df = pd.read_csv(NORM / "gi.csv")
    if df["geo_id"].nunique() != 1 or int(df["count"].sum()) != 31_958:
        raise SystemExit("gi.csv: expected 31,958 people in one unit -- run sources/gi_census.py")
    df["node"] = df["source_category"].map(gi2022.resolve)
    df["unit"] = "GI"   # religiondots' gi_hexes.gpkg: one unit, GI
    return df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


ENTRY = dict(
    name="Gibraltar",
    source="Census of Gibraltar 2022, Tables 51 and 52 and Figure 11 (HM Government of Gibraltar)",
    how="census, 2022, main language, people aged 14 and over",
    parts=[dict(covers="People aged 14 and over", source="2022 census, main language", rest=True)],
    grain="the territory as one unit, 32,000 people aged 14 and over",
    gap="children aged 13 and under, 5,978 (15.8%), who were not asked",
    view=[-5.38, 36.10, -5.33, 36.165],
    counts=_counts,
    mappings=["gi2022"],
    place=RD_GEO / "gi" / "gi_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "The 2022 census asked everyone aged 14 or over for their main language, English or "
        "another written in. English is 87.6%, Spanish 5.6% and Arabic 2.5%. The report prints "
        "only those three as numbers and shows eight more, Portuguese, Polish, French, German, "
        "Hebrew, Russian, Chinese and Filipino, as bars on a chart, so their counts here are "
        "read off the chart to about 10 people. The Arabic speakers are drawn as Moroccan "
        "Arabic, the language of the island's Moroccan community. Llanito, the local mix of "
        "English and Spanish, was not an option on the form. Children aged 13 and under were "
        "not asked and are not drawn. The territory is drawn as one unit, so the dots follow "
        "where people live."),
)
