# Maldives. No language question: 2022 census Maldivians on Dhivehi, foreign residents by
# nationality (Male's mix and the atolls' mix, migration report Table 6.1) at their countries'
# drawn mixes, per atoll (sources/mv_census.py). Every row `derived`. Placed on religiondots'
# Kontur hexes cut to islands (read-only). Record: sources/mv.md.
from _shared import *  # noqa: F401,F403

UNITS = 21
TOTAL = 515_132


def _counts():
    import mv2022
    df = pd.read_csv(NORM / "mv.csv", dtype={"geo_id": str})
    if df["geo_id"].nunique() != UNITS or int(df["count"].sum()) != TOTAL:
        raise SystemExit("mv.csv is not 21 units summing to the 2022 census -- run "
                         "sources/mv_census.py")
    df["node"] = df["source_category"].map(mv2022.resolve)
    df["unit"] = df["geo_id"]   # religiondots' mv_lookup.csv units
    out = by_unit(df)
    out["tier"] = "derived"
    return out


ENTRY = dict(
    name="Maldives",
    source=("2022 census (Maldives Bureau of Statistics): usual residents by nationality and "
            "atoll (Table MG11) and foreign residents by nationality for Male and the atolls "
            "(migration report, Table 6.1)"),
    how="no language question; census, 2022, nationality drawn as a language",
    parts=[
        dict(covers="Maldivians", source="2022 census, drawn as Dhivehi", people=382_639),
        dict(covers="Foreign residents",
             source="2022 census, nationality, drawn on that country's languages",
             rest=True),
    ],
    grain="20 atolls and Male, 24,500 people on average",
    gap="5,123 foreign residents of other nationalities, drawn as an unnamed language",
    view=[72.4, -0.8, 73.9, 7.2],
    counts=_counts,
    mappings=["mv2022"],
    place=RD_GEO / "mv" / "mv_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "The Maldives census does not ask about language, so the 382,639 Maldivians are drawn "
        "as Dhivehi speakers and the 132,493 foreign residents by the languages of their "
        "countries. More than half of the foreigners are Bangladeshi and a quarter Indian. "
        "Indians are drawn in India's own language mix, which may not match the workers who "
        "come to the Maldives. Nationality is published for Male and for the atolls as a "
        "whole, so every atoll's foreigners are drawn with the same mix."),
)
