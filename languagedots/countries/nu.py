# Niue. 2022 census, languages spoken in the household (Vagahau Niue, English, other; several
# allowed), by village, summed to the island (sources/nu_census.py); mentions shared across the
# 1,564 usual residents. Every row derived. One unit, on religiondots' Kontur hexes (read-only).
# Record: sources/nu.md.
from _shared import *  # noqa: F401,F403


def _counts():
    import nu2022
    df = pd.read_csv(NORM / "nu.csv")
    if df["geo_id"].nunique() != 1 or abs(df["count"].sum() - 1_564) > 1e-6:
        raise SystemExit("nu.csv: expected 1,564 people in one unit -- run sources/nu_census.py")
    df["node"] = df["source_category"].map(nu2022.resolve)
    df["unit"] = "NU"   # religiondots' nu_hexes.gpkg: one unit, NU
    out = by_unit(df)
    out["tier"] = "derived"
    return out


ENTRY = dict(
    name="Niue",
    source="2022 Niue Census of Population and Housing Report, household languages (Niue Statistics Office)",
    how=("census, 2022, languages spoken in the household (Vagahau Niue, English, other languages, "
         "several allowed), each mention shared across the island's usual residents"),
    parts=[dict(covers="Usual residents", source="2022 census, household languages, shared across the "
                                                  "languages named", rest=True)],
    grain="the island as one unit, 1,564 people",
    gap="none: the table covers all 513 private occupied dwellings; the 117 visitors on census night are not counted",
    view=[-170.10, -19.20, -169.72, -18.90],
    counts=_counts,
    mappings=["nu2022"],
    place=RD_GEO / "nu" / "nu_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "The 2022 census asked each household which languages are spoken in it, and several could "
        "be named. Of 513 households, 395 named Vagahau Niue, 328 English and 65 another language. "
        "The mentions are shared across the 1,564 usual residents, so Vagahau Niue is drawn for "
        "half of them, English for 42% and other languages for 8%. These are shares of "
        "mentions, not of speakers: a household naming two languages is counted once under each. "
        "The census does not say which other languages. Niue is small enough that the island "
        "draws one or two dots."),
)
