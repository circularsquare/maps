# Armenia. 2011 census mother tongue by marz (sources/am_census.py), on religiondots' Kontur hexes
# for the same eleven marzes (read only). The record is sources/am.md.
from _shared import *  # noqa: F401,F403


def _counts():
    import am2011
    df = pd.read_csv(NORM / "am.csv", dtype={"geo_id": str})
    df = df[df["geo_level"] == "marz"].copy()
    if df["geo_id"].nunique() != 11:
        raise SystemExit(f"am.csv: {df['geo_id'].nunique()} marzes, expected 11 (ten and "
                         "Yerevan); re-run sources/am_census.py")
    # geo_id is the ISO 3166-2 code, which religiondots' am_grid_400m.gpkg carries as `unit`.
    df["unit"] = df["geo_id"]
    df["node"] = df["source_category"].map(am2011.resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    return by_unit(df)


ENTRY = dict(
    name="Armenia",
    source="2011 Population Census, table 5.2-1, population by ethnicity, sex and mother tongue, "
           "one table per marz (Statistical Committee of the Republic of Armenia)",
    how="census, 2011, mother tongue",
    parts=[dict(covers="Everyone", source="2011 census, mother tongue", rest=True)],
    grain="11 marzes, 274,000 people on average",
    gap="29 people who refused the question",
    view=[43.4, 38.8, 46.7, 41.35],
    counts=_counts,
    mappings=["am2011"],
    place=RD_GEO / "am" / "am_grid_400m.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "The 2011 census asked each person's mother tongue, which in the countries of the former "
        "Soviet Union leans towards identity rather than everyday use. 97.9% named Armenian. "
        "The census prints Yezidi and Kurdish as two languages, and this map draws them apart, "
        "though linguists count both as Kurmanji: the answer follows which community a person "
        "belongs to. Russian, 0.8%, is half in Yerevan and is also the mother tongue of the "
        "Molokan villages of Lori; about half of those who named it were ethnic Armenians. A "
        "language a marz's table gives no column is drawn there as other. The 2022 census "
        "publishes only whether the mother tongue was the language of the person's own "
        "nationality, so 2011 is the most recent count that names the languages."),
)
