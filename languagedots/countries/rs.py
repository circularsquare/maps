# Serbia. Popis 2022 mother tongue by municipality (sources/rs_census.py), on religiondots' Kontur
# hexes for the same 168 municipalities (read only). The record is sources/rs.md.
from _shared import *  # noqa: F401,F403


def _key(s):
    # religiondots keys Belgrade's Stari grad as RZS wrote it, `Stari  grad` with two spaces;
    # rs.csv collapses whitespace, so the hex layer's keys are collapsed the same way here
    return s.str.split().str.join(" ")


def _counts():
    import rs2022
    df = pd.read_csv(NORM / "rs.csv")
    df = df[df["geo_level"] == "municipality"].copy()
    if df["geo_id"].nunique() != 168:
        raise SystemExit(f"rs.csv: {df['geo_id'].nunique()} municipalities, expected 168")
    df["unit"] = df["geo_id"]
    df["node"] = df["source_category"].map(rs2022.resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    return by_unit(df)


ENTRY = dict(
    name="Serbia",
    source="Popis stanovništva 2022, population by mother tongue, by municipality and city "
           "(Republički zavod za statistiku)",
    how="census, 2022, mother tongue",
    parts=[dict(covers="Everyone", source="2022 census, mother tongue", rest=True)],
    grain="168 municipalities and city municipalities, 40,000 people on average",
    gap="391,301 people (5.9%) whose mother tongue is unknown or undeclared; Kosovo was not "
        "enumerated.",
    view=[18.8, 42.2, 23.1, 46.2],
    counts=_counts,
    mappings=["rs2022"],
    place=RD_GEO / "rs" / "rs_grid_400m.gpkg",
    place_unit=lambda g: _key(g["unit"].astype(str)),
    place_weight=pop_weight,
    note_public=(
        "Mother tongue is what people declared, and in Serbia it follows ethnicity closely. "
        "Serbian, Bosnian, Croatian and Montenegrin are one language by linguists' measure, "
        "and Bunjevac is a dialect of it; the census prints each apart and so does this map. "
        "Vlach is the Romanian spoken in eastern Serbia, printed apart from the Romanian of "
        "the Banat. Unknown or undeclared answers are 5.9% and are not drawn; unknown is "
        "highest in central Belgrade. The census names 18 languages and puts the rest under "
        "one other."),
)
