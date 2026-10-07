# Suriname. Census 7 (2004), the language most spoken in the household, households per ressort
# turned into people (sources/sr_census.py), on religiondots' Kontur hexes for the same 62
# ressorten (COD-AB ADM2 p-codes, read-only).
from _shared import *  # noqa: F401,F403


def _counts():
    """Every row is `derived`: the census counts households, and sources/sr_census.py weights
    each by the national mean size of households of its language (Census 7 Volume 4, Tabel 02)
    and scales each ressort to its census population. Unknown is not drawn."""
    import sr2004
    df = pd.read_csv(NORM / "sr.csv", dtype={"geo_id": str})
    df = df[df["geo_level"] == "ressort"].copy()
    if df["geo_id"].nunique() != 62:
        raise SystemExit(f"sr: {df['geo_id'].nunique()} ressorten, expected 62")
    df["node"] = df["source_category"].map(sr2004.resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df = df.rename(columns={"geo_id": "unit"})
    out = by_unit(df)
    out["tier"] = "derived"
    return out


ENTRY = dict(
    name="Suriname",
    source="Census 7 (2004), Algemeen Bureau voor de Statistiek: census profile at ressort "
           "level, \"Most Spoken Language in the household\", counted in households; household "
           "sizes by language from Census 7 Volume 4, Tabel 02",
    how="census, 2004, language most spoken in the household (one answer per household)",
    parts=[dict(covers="Everyone",
                source="2004 census, language most spoken in the household, weighted by "
                       "household size",
                rest=True)],
    grain="62 ressorten, 7,900 people on average",
    gap="3,075 households (2.5%) whose language is unknown, about 8,500 people",
    view=[-58.15, 1.80, -53.90, 6.10],
    counts=_counts,
    mappings=["sr2004"],
    place=RD_GEO / "sr" / "sr_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "The 2004 census asked each household which language its members usually speak to one "
        "another, and everyone in the household is drawn in that language. The census counts "
        "households, not people, so each household is weighted by the average size of "
        "households that speak its language across the country. Most Surinamese speak more "
        "than one language at home, and only the first is drawn: Sranan Tongo was the second "
        "language of 37% of households. \"Other\" holds the smaller Maroon languages and the "
        "indigenous languages the census does not name, such as Trio in Coeroeni. The 2012 "
        "census published full tables for only two of the ten districts, and the 2024-25 "
        "census has no results yet."),
)
