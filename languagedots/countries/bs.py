# The Bahamas. No census language question: the 2010 census's citizenship and race tables per
# island (sources/bs_census.py) give Bahamian Creole, English, Haitian Creole and the immigrant
# languages as shares, applied to the 2022 census island populations. Every row derived. Placed
# on religiondots' Kontur hexes for the 18 census islands (read-only). Record: sources/bs.md.
from _shared import *  # noqa: F401,F403


def _counts():
    import bs2010
    df = pd.read_csv(NORM / "bs.csv", dtype={"geo_id": str})
    if df["geo_id"].nunique() != 18:
        raise SystemExit(f"bs.csv: {df['geo_id'].nunique()} islands, expected 18 -- "
                         "run sources/bs_census.py")
    df["node"] = df["source_category"].map(bs2010.resolve)
    df["unit"] = df["geo_id"]   # religiondots' bs_lookup.csv: geo_id == unit (BNSI island codes)
    out = by_unit(df)
    out["tier"] = "derived"
    return out


ENTRY = dict(
    name="The Bahamas",
    source=("Census of Population and Housing 2010, island reports, Tables 8.0 (race) and 9.0 "
            "(country of citizenship); Census 2022 island populations (Bahamas National "
            "Statistical Institute)"),
    how=("no language question: each island's 2010 shares by citizenship, foreign nationals on "
         "their country's language (Haitians as Haitian Creole), Bahamian citizens as Bahamian "
         "Creole except white Bahamians as English; applied to the 2022 census population"),
    parts=[
        dict(covers="Foreign nationals",
             source="2010 census shares by citizenship on the 2022 count, drawn on their "
                    "country's language",
             people=67_191),
        dict(covers="White Bahamians", source="2010 census, race, drawn as English",
             people=9_207),
        dict(covers="Other Bahamian citizens", source="drawn as Bahamian Creole", rest=True),
    ],
    grain="18 islands, 22,000 people on average",
    gap="people who did not state a citizenship in 2010 (shares are of stated answers)",
    view=[-79.60, 20.75, -72.40, 27.45],
    counts=_counts,
    mappings=["bs2010"],
    place=RD_GEO / "bs" / "bs_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "The Bahamas census does not ask about language. Most Bahamians grow up speaking "
        "Bahamian Creole and learn standard English at school; white Bahamians, about 3% of "
        "citizens, speak a dialect of English and are drawn as English speakers. Haitian "
        "nationals, 11% of the 2010 count, are drawn as Haitian Creole speakers; they include "
        "children born in the Bahamas to Haitian parents, but not Bahamian citizens of Haitian "
        "descent, who are drawn as Bahamian Creole. Other foreign nationals are drawn on their "
        "country's main language. The shares are from 2010, the last census to publish citizenship by "
        "island, applied to each island's 2022 population; Hurricane Dorian in 2019 destroyed "
        "the largest Haitian settlements on Abaco, so Haitian Creole there is likely "
        "overstated."),
)
