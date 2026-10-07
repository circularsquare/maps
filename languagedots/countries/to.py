# Tonga. 2021 census, language used at home (Tongan only / Tongan and other / no Tongan), by
# division, placed on the census's village populations (sources/to_census.py). Placed on
# religiondots' Kontur hexes for its 156 villages (read-only). Record: sources/to.md.
from _shared import *  # noqa: F401,F403


def _counts():
    import to2021
    df = pd.read_csv(NORM / "to.csv")
    if df["geo_id"].nunique() != 156:
        raise SystemExit(f"to.csv: {df['geo_id'].nunique()} villages, expected 156 -- "
                         "run sources/to_census.py")
    df["node"] = df["source_category"].map(to2021.resolve)
    df["unit"] = df["geo_id"]   # religiondots' to_lookup.csv: geo_id == unit
    out = by_unit(df)
    out["tier"] = "derived"
    return out


ENTRY = dict(
    name="Tonga",
    source=("2021 Census of Population and Housing, Volume 1, Table G 48 (Tonga Statistics "
            "Department)"),
    how=("census, 2021, language used at home, aged 5 and over; Tongan with another language "
         "drawn as Tongan"),
    parts=[dict(covers="Everyone aged 5 and over",
                source="2021 census, language used at home, division shares on village "
                       "populations",
                rest=True)],
    grain="5 divisions, placed by 156 villages",
    view=[-176.4, -22.3, -173.0, -15.0],
    counts=_counts,
    mappings=["to2021"],
    place=RD_GEO / "to" / "to_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "Tonga's 2021 census asked what language each person speaks at home, with three "
        "answers: Tongan only (85% of people aged 5 and over), Tongan and another language "
        "(14%), or no Tongan (1.1%). The first two are drawn as Tongan speakers. The census "
        "does not record which other language, so the 1,006 people who do not speak Tongan at "
        "home are drawn as other languages. Niuafo'ou has a language of its own, but nearly "
        "everyone in the Niuas answered Tongan only, so it is not drawn. Every village in a "
        "division has the same mix."),
)
