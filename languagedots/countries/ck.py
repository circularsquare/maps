# Cook Islands. 2021 census Q18 (language spoken at home: Cook Islands Maori yes/no, any other
# language yes/no), Table 3.07 by island (sources/ck_census.py). People who answered yes to both are
# shared half and half; "other language" is drawn as English. Every row derived. Placed on
# religiondots' Kontur hexes re-keyed to the 12 islands (sources/ck_geo.py). Record: sources/ck.md.
from _shared import *  # noqa: F401,F403


def _counts():
    import ck2021
    df = pd.read_csv(NORM / "ck.csv")
    if df["geo_id"].nunique() != 12 or abs(df["count"].sum() - 13_874) > 1e-6:
        raise SystemExit("ck.csv: expected 13,874 people on 12 islands -- run sources/ck_census.py")
    df["node"] = df["source_category"].map(ck2021.resolve)
    df["unit"] = df["geo_id"]
    out = by_unit(df)
    out["tier"] = "derived"
    return out


ENTRY = dict(
    name="Cook Islands",
    source="2021 Census of Population and Dwellings, Table 3.07 (Cook Islands Statistics Office)",
    how=("census, 2021, language spoken at home (Cook Islands Maori yes or no, any other language "
         "yes or no); each person shared across the languages they named, and other language "
         "drawn as English"),
    parts=[dict(covers="People aged 5 and over",
                source="2021 census, language spoken at home, shared across the languages named",
                rest=True)],
    grain="12 inhabited islands, 1,200 people aged 5 and over on average",
    gap="children under 5, 1,113 (7.4% of the 14,987 residents)",
    view=[-166.5, -22.3, -156.8, -8.6],
    counts=_counts,
    mappings=["ck2021"],
    place=GEO / "ck" / "ck_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "The census asked two yes or no questions of everyone aged 5 or over: whether they speak "
        "Cook Islands Maori at home, and whether they speak any other language. It did not ask "
        "which. 70% said yes to Maori, 89% yes to another language, and 59% said yes to both. "
        "People who named both are drawn half on each, and the other language is drawn as "
        "English, the second official language, so English reads larger here than a count of "
        "households speaking only English would be. Maori is strongest in the Northern Group "
        "and the smaller Southern islands: on Pukapuka all 386 people answered yes to it, though "
        "Pukapukan is a separate language that the census does not tell apart. On Rarotonga, "
        "where three quarters of people live, 62% said yes to Maori."),
)
