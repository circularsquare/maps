# Kazakhstan. 2021 census native language by rayon and city (218), from the census dashboard's
# engine (sources/kz_census.py), on Kontur hexes keyed to COD-AB's rayons put back to 2021
# (sources/kz_geo.py). sources/kz.md is the record.
from _shared import *  # noqa: F401,F403


def _counts():
    import kz2021
    df = pd.read_csv(NORM / "kz.csv", dtype={"geo_id": str})
    df = df[df["geo_level"] == "rayon"].copy()
    lut = pd.read_csv(GEO / "kz" / "kz_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    if df["unit"].isna().any() or df["unit"].nunique() != 218:
        raise SystemExit(f"kz: {sorted(df.loc[df['unit'].isna(), 'geo_id'].unique())} not in "
                         "kz_lookup.csv, or not 218 rayons; re-run sources/kz_geo.py")
    df["node"] = df["source_category"].map(kz2021.resolve)
    return by_unit(df)


ENTRY = dict(
    name="Kazakhstan",
    source=("National Population Census 2021 (Bureau of National Statistics), native language "
            "by rayon and city from the census dashboard's own data service"),
    how="census, 2021, native language",
    parts=[
        dict(covers="Everyone",
             source="2021 census, native language, rayon figures from the census dashboard",
             rest=True),
    ],
    grain="218 rayons and cities, 88,000 people on average",
    view=[46.0, 40.5, 87.5, 55.5],
    counts=_counts,
    mappings=["kz2021"],
    place=GEO / "kz" / "kz_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "The 2021 census asked each person's native language, which in the countries of the "
        "former Soviet Union leans towards identity rather than everyday use. 13.7 million "
        "people named Kazakh and 3.5 million Russian, and only 58,791 ethnic Kazakhs named "
        "Russian, though many Kazakh families in the cities and the north use Russian every "
        "day. Most Ukrainians, Germans and Poles named Russian. The rayon figures come from "
        "the census dashboard, which matches the printed national table exactly; it names 17 "
        "languages, and the other 1.2% are drawn as other. Rayons are drawn as they were in "
        "2021, before Abai, Jetisu and Ulytau regions were formed."),
)
